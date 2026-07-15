# Copyright © 2026 Apple Inc.

"""Bounded Qwen3-Next benchmark for recurrent checkpoint host offload.

The model is a small, randomly initialized but architecture-faithful hybrid:
linear-attention layers use ``ArraysCache`` and full-attention layers use
``KVCache``. Baseline and capped stores contain identical prefix checkpoints.
The timed operation is equal work: fetch the same prefix and run one-token
continuation. Results are implementation evidence, not a large-model speed
claim.
"""

import argparse
import copy
import gc
import json
import statistics
import time

import mlx.core as mx

from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import LRUPromptCache


def make_model():
    args = qwen3_next.ModelArgs(
        model_type="qwen3_next",
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
        linear_num_value_heads=4,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0],
        moe_intermediate_size=128,
        rms_norm_eps=1e-5,
        head_dim=64,
        rope_theta=1000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=4096,
    )
    model = qwen3_next.Model(args)
    mx.eval(model.parameters())
    return model


def checkpoint(model, tokens):
    state = model.make_cache()
    logits = model(mx.array([tokens]), cache=state)
    mx.eval(logits, [cache.state for cache in state])
    return state


def timed_continuation(model, store, model_key, prefix, next_token):
    start = time.perf_counter()
    state, remaining = store.fetch_nearest_cache(model_key, prefix)
    if remaining:
        raise RuntimeError("benchmark target must be an exact checkpoint")
    logits = model(mx.array([[next_token]]), cache=state)
    mx.eval(logits)
    return 1000 * (time.perf_counter() - start), logits


def build_store(model, model_key, token_stream, checkpoints, stride, cap=None):
    store = LRUPromptCache(max_size=checkpoints + 1, recurrent_state_path_cap=cap)
    for index in range(1, checkpoints + 1):
        prefix = token_stream[: index * stride]
        store.insert_cache(model_key, prefix, checkpoint(model, prefix))
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    return store


def run(cap, checkpoints, stride, repeats):
    if checkpoints <= cap:
        raise ValueError("checkpoints must exceed cap to exercise host restore")
    mx.random.seed(11)
    model = make_model()
    model_key = ("bench-qwen3-next", None, None)
    token_stream = [((index * 17) % 997) + 1 for index in range(checkpoints * stride)]

    gc.collect()
    mx.clear_cache()
    model_active_bytes = mx.get_active_memory()
    memory_baseline = build_store(model, model_key, token_stream, checkpoints, stride)
    baseline_active_delta = mx.get_active_memory() - model_active_bytes
    del memory_baseline
    gc.collect()
    mx.clear_cache()
    memory_capped = build_store(
        model, model_key, token_stream, checkpoints, stride, cap=cap
    )
    capped_active_delta = mx.get_active_memory() - model_active_bytes
    del memory_capped
    gc.collect()
    mx.clear_cache()

    snapshots = []
    for index in range(1, checkpoints + 1):
        prefix = token_stream[: index * stride]
        snapshots.append((prefix, checkpoint(model, prefix)))

    baseline = LRUPromptCache(max_size=checkpoints + 1)
    capped = LRUPromptCache(max_size=checkpoints + 1, recurrent_state_path_cap=cap)
    for prefix, state in snapshots:
        baseline.insert_cache(model_key, prefix, copy.deepcopy(state))
        capped.insert_cache(model_key, prefix, copy.deepcopy(state))

    target = snapshots[0][0]
    next_token = token_stream[len(target)]
    # One untimed pass materializes dispatch and validates equality before the
    # interleaved A/B loop.
    _, reference = timed_continuation(model, baseline, model_key, target, next_token)
    _, restored = timed_continuation(model, capped, model_key, target, next_token)
    if not bool(mx.array_equal(reference, restored)):
        raise RuntimeError("host-restored continuation diverged from baseline")

    baseline_ms = []
    capped_ms = []
    for iteration in range(repeats):
        order = (baseline, capped) if iteration % 2 == 0 else (capped, baseline)
        observed = {}
        for store in order:
            elapsed, logits = timed_continuation(
                model, store, model_key, target, next_token
            )
            observed[id(store)] = (elapsed, logits)
        base_elapsed, base_logits = observed[id(baseline)]
        cap_elapsed, cap_logits = observed[id(capped)]
        if not bool(mx.array_equal(base_logits, cap_logits)):
            raise RuntimeError(f"continuation diverged at repeat {iteration}")
        baseline_ms.append(base_elapsed)
        capped_ms.append(cap_elapsed)

    baseline_stats = baseline.recurrent_state_stats()
    capped_stats = capped.recurrent_state_stats()
    base_bytes = baseline_stats["device_cache_bytes"]
    cap_bytes = capped_stats["device_cache_bytes"]
    base_p50 = statistics.median(baseline_ms)
    cap_p50 = statistics.median(capped_ms)
    base_p95 = statistics.quantiles(baseline_ms, n=20)[18]
    cap_p95 = statistics.quantiles(capped_ms, n=20)[18]
    return {
        "model": "tiny-random-qwen3-next-4-layer",
        "checkpoint_count": checkpoints,
        "checkpoint_stride_tokens": stride,
        "path_cap": cap,
        "target_prefix_tokens": len(target),
        "repeats": repeats,
        "continuation_exact": True,
        "baseline": {
            "device_cache_bytes": base_bytes,
            "mlx_active_cache_delta_bytes": baseline_active_delta,
            "ttft_ms_p50": base_p50,
            "ttft_ms_p95": base_p95,
            "recurrent_stats": baseline_stats,
        },
        "capped": {
            "device_cache_bytes": cap_bytes,
            "mlx_active_cache_delta_bytes": capped_active_delta,
            "host_recurrent_bytes": capped_stats["host_recurrent_bytes"],
            "ttft_ms_p50": cap_p50,
            "ttft_ms_p95": cap_p95,
            "recurrent_stats": capped_stats,
        },
        "device_cache_reduction_percent": round(
            100 * (base_bytes - cap_bytes) / base_bytes, 3
        ),
        "mlx_active_cache_reduction_percent": round(
            100 * (baseline_active_delta - capped_active_delta) / baseline_active_delta,
            3,
        ),
        "ttft_p50_change_percent": round(100 * (cap_p50 - base_p50) / base_p50, 3),
        "scope": "architecture-faithful tiny model; not a checkpoint-scale claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--checkpoints", type=int, default=6)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(run(args.cap, args.checkpoints, args.stride, args.repeats), indent=2)
    )
