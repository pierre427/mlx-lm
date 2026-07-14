import mlx.core as mx
import pytest
import copy

from mlx_lm.generate import PromptProcessingBatch
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.models.gpt_oss_puzzle import Model, ModelArgs


def _prompt_cache(left_padding, *, backend="bucketed", seed=7):
    mx.random.seed(seed)
    batch = len(left_padding)
    width = 8
    cache = BatchKVCache(left_padding, attention_backend=backend)
    keys = mx.random.normal(shape=(batch, 2, width, 8)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, 2, width, 8)).astype(mx.float16)
    cache.update_and_fetch(keys, values)
    return cache


def _decode_once(cache, *, seed=11, query_length=1):
    mx.random.seed(seed)
    batch = cache.offset.shape[0]
    mask = cache.make_mask(query_length)
    queries = mx.random.normal(shape=(batch, 4, query_length, 8)).astype(mx.float16)
    keys = mx.random.normal(shape=(batch, 2, query_length, 8)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, 2, query_length, 8)).astype(mx.float16)
    dense_keys, dense_values = cache.update_and_fetch(keys, values)
    sinks = mx.zeros((4,), dtype=mx.float16)
    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dense_keys,
        dense_values,
        scale=8**-0.5,
        mask=mask,
        sinks=sinks,
    )
    actual = scaled_dot_product_attention(
        queries,
        dense_keys,
        dense_values,
        cache,
        8**-0.5,
        mask,
        sinks=sinks,
    )
    mx.eval(reference, actual)
    return reference, actual


def test_sdpa_is_the_default(monkeypatch):
    monkeypatch.delenv("MLX_LM_BATCH_ATTENTION_BACKEND", raising=False)
    cache = BatchKVCache([0, 6, 0, 6])
    assert cache.attention_backend == "sdpa"
    reference, actual = _decode_once(cache)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    assert cache.attention_backend_metrics["attempts"] == 0


def test_bucketed_decode_matches_dense_and_persists_groups():
    cache = _prompt_cache([0, 6, 0, 6])
    reference, actual = _decode_once(cache)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    metrics = cache.attention_backend_metrics
    assert metrics["bucketed_calls"] == 1
    assert metrics["group_builds"] == 1
    assert metrics["last_group_count"] == 2
    groups = cache._bucket_groups

    reference, actual = _decode_once(cache, seed=12)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    assert cache._bucket_groups is groups
    assert cache.attention_backend_metrics["group_builds"] == 1


def test_balanced_decode_falls_back_to_dense():
    cache = _prompt_cache([0, 0, 0, 0])
    reference, actual = _decode_once(cache)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    metrics = cache.attention_backend_metrics
    assert metrics["bucketed_calls"] == 0
    assert metrics["fallback_reasons"]["dense_tax"] == 1


def test_prefill_falls_back_to_dense():
    cache = _prompt_cache([0, 6, 0, 6])
    reference, actual = _decode_once(cache, query_length=2)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    assert cache.attention_backend_metrics["fallback_reasons"]["not_decode"] == 1


def test_filter_invalidates_then_rebuilds_exactly():
    cache = _prompt_cache([0, 6, 0, 6])
    _decode_once(cache)
    cache.filter([1, 2])
    assert cache._bucket_groups is None
    assert cache.attention_backend_metrics["group_invalidations"] == 1
    reference, actual = _decode_once(cache, seed=13)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    assert cache.attention_backend_metrics["group_builds"] == 2


def test_trim_invalidates_groups_and_dense_state_stays_authoritative():
    cache = _prompt_cache([0, 6, 0, 6])
    _decode_once(cache)
    before = cache.nbytes
    assert before > cache.keys.nbytes + cache.values.nbytes
    cache.trim(1)
    assert cache._bucket_groups is None
    assert cache.nbytes == cache.keys.nbytes + cache.values.nbytes


def test_extend_invalidates_then_rebuilds_exactly():
    cache = _prompt_cache([0, 6])
    other = _prompt_cache([0, 6], seed=8)
    _decode_once(cache)
    cache.extend(other)
    assert cache._bucket_groups is None
    assert cache.attention_backend_metrics["group_invalidations"] == 1
    reference, actual = _decode_once(cache, seed=14)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()


def test_state_restore_preserves_per_cache_backend_and_invalidates_groups():
    cache = _prompt_cache([0, 6])
    _decode_once(cache)
    state = cache.state
    cache.state = state
    assert cache.attention_backend == "bucketed"
    assert cache._bucket_groups is None
    reference, actual = _decode_once(cache, seed=15)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()


def test_non_positive_length_falls_back_to_dense():
    cache = _prompt_cache([0, 0])
    cache.offset = mx.array([0, 8])
    queries = mx.zeros((2, 4, 1, 8), dtype=mx.float16)
    assert cache.bucketed_attention(queries, 8**-0.5, None) is None
    assert (
        cache.attention_backend_metrics["fallback_reasons"][
            "non_positive_length"
        ]
        == 1
    )


def test_backend_can_be_selected_per_cache_group():
    cache = _prompt_cache([0, 6], backend="sdpa")
    cache.set_attention_backend("bucketed")
    assert cache.attention_backend == "bucketed"
    reference, actual = _decode_once(cache)
    assert mx.allclose(reference, actual, rtol=2e-3, atol=2e-2).item()
    cache.set_attention_backend("sdpa")
    assert cache._bucket_groups is None


def test_invalid_backend_fails_closed():
    with pytest.raises(ValueError, match="sdpa.*bucketed"):
        BatchKVCache([0], attention_backend="unknown")


def test_tiny_puzzle_model_matches_dense_after_heterogeneous_prefill():
    args = ModelArgs(
        num_hidden_layers=2,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        block_configs=[
            {"num_local_experts": 4, "sliding_window": None},
            {"num_local_experts": 4, "sliding_window": 4},
        ],
    )
    model = Model(args)
    prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20],
    ]
    batch = PromptProcessingBatch(
        model=model,
        uids=list(range(4)),
        caches=[model.make_cache() for _ in prompts],
        prefill_step_size=16,
    )
    batch.prompt(prompts)
    dense_cache = copy.deepcopy(batch.prompt_cache)
    bucketed_cache = copy.deepcopy(batch.prompt_cache)
    for cache in dense_cache:
        if isinstance(cache, BatchKVCache):
            cache.set_attention_backend("sdpa")
    for cache in bucketed_cache:
        if isinstance(cache, BatchKVCache):
            cache.set_attention_backend("bucketed")

    tokens = mx.array([[21], [22], [23], [24]])
    dense_logits = model(tokens, cache=dense_cache)
    bucketed_logits = model(tokens, cache=bucketed_cache)
    mx.eval(dense_logits, bucketed_logits)
    assert mx.allclose(dense_logits, bucketed_logits, rtol=2e-3, atol=2e-2).item()
    full_cache = next(c for c in bucketed_cache if isinstance(c, BatchKVCache))
    assert full_cache.attention_backend_metrics["bucketed_calls"] == 1
