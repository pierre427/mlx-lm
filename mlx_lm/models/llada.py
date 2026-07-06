# Copyright © 2025 Apple Inc.

# LLaDA-8B: masked-diffusion language model (MDM).
#
# Structurally OLMo-derived (pre-norm RMSNorm + SwiGLU MLP), but with two
# defining differences from a standard causal LM:
#   1. Attention is BIDIRECTIONAL / non-causal (mask=None everywhere).
#   2. Generation is diffusion-based, re-running a full forward over the whole
#      sequence at every denoising step, so there is NO KV cache.
#
# HF checkpoints prefix everything with ``model.transformer.``; ``sanitize``
# remaps those keys to the standard mlx-lm layout so the weights are
# quantization-friendly.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs

# LLaDA-8B special token ids (from the official config.json).
DEFAULT_MASK_TOKEN_ID = 126336
DEFAULT_EOS_TOKEN_ID = 126081


@dataclass
class ModelArgs(BaseModelArgs):
    # Field names mirror the official config.json keys so ``from_dict`` maps
    # them with no manual translation.
    model_type: str = "llada"
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    embedding_size: Optional[int] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: int = DEFAULT_EOS_TOKEN_ID
    weight_tying: bool = False
    include_bias: bool = False
    include_qkv_bias: bool = False

    def __post_init__(self):
        # ``embedding_size`` is the true vocab / head size in LLaDA configs;
        # keep ``vocab_size`` as an alias when only one is provided.
        if self.embedding_size is None:
            self.embedding_size = self.vocab_size
        else:
            self.vocab_size = self.embedding_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.d_model
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = head_dim = args.d_model // args.n_heads
        self.scale = head_dim**-0.5

        bias = args.include_qkv_bias
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.include_bias)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        # Bidirectional attention: no causal mask.
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.d_model
        hidden_dim = args.mlp_hidden_size
        bias = args.include_bias
        # SwiGLU with three separate matrices (not a fused gate+up).
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm: attn_norm before attention, ff_norm before MLP.
        h = x + self.self_attn(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LLaDAModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h, mask=None)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDAModel(args)
        if not args.weight_tying:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        # Bidirectional, no cache, no mask. Returns logits [B, L, vocab].
        out = self.model(inputs)
        if self.args.weight_tying:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Remap HF ``model.transformer.*`` keys to the mlx-lm layout.

        A leading ``model.`` prefix is optional; we key off the ``transformer.``
        segment. The critical ambiguity is ``blocks.{N}.ff_out`` (the MLP
        down-proj, nested under ``.blocks.``) vs the top-level
        ``transformer.ff_out`` (the LM head), disambiguated by whether
        ``.blocks.`` appears in the path.
        """
        sanitized = {}
        for key, value in weights.items():
            new_key = self._remap_key(key)
            sanitized[new_key] = value
        return sanitized

    @staticmethod
    def _remap_key(key: str) -> str:
        # Strip everything up to and including the ``transformer.`` segment so
        # an optional leading ``model.`` prefix is tolerated.
        marker = "transformer."
        idx = key.find(marker)
        if idx == -1:
            return key
        suffix = key[idx + len(marker) :]

        if suffix.startswith("wte."):
            return "model.embed_tokens." + suffix[len("wte.") :]
        if suffix.startswith("ln_f."):
            return "model.norm." + suffix[len("ln_f.") :]
        if suffix.startswith("blocks."):
            rest = suffix[len("blocks.") :]
            n, _, tail = rest.partition(".")
            attn_map = {
                "q_proj.": "self_attn.q_proj.",
                "k_proj.": "self_attn.k_proj.",
                "v_proj.": "self_attn.v_proj.",
                "attn_out.": "self_attn.o_proj.",
                "ff_proj.": "mlp.gate_proj.",
                "up_proj.": "mlp.up_proj.",
                "ff_out.": "mlp.down_proj.",
                "attn_norm.": "input_layernorm.",
                "ff_norm.": "post_attention_layernorm.",
            }
            for src, dst in attn_map.items():
                if tail.startswith(src):
                    return f"model.layers.{n}.{dst}{tail[len(src):]}"
            return key
        # Top-level ``transformer.ff_out`` is the LM head (no ``.blocks.``).
        if suffix.startswith("ff_out."):
            return "lm_head." + suffix[len("ff_out.") :]
        return key

    @property
    def layers(self):
        return self.model.layers


# ---------------------------------------------------------------------------
# Diffusion sampler (LLaDA official generation reimplemented in pure MLX).
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Gumbel-max sampling helper (matches the official implementation).

    At ``temperature == 0`` this is a no-op (pure argmax downstream).
    """
    if temperature == 0.0:
        return logits
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape).astype(mx.float32)
    gumbel = (-mx.log(noise)) ** temperature
    return mx.exp(logits) / gumbel


def get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """Per-row token-reveal schedule over ``steps`` denoising steps.

    Each row unmasks ``base = mask_count // steps`` tokens per step, plus one
    extra for the first ``remainder = mask_count % steps`` steps. The schedule
    sums exactly to ``mask_count`` per row. Returns an int array [B, steps].
    """
    mask_count = mask_index.sum(axis=1, keepdims=True)  # [B, 1]
    base = mask_count // steps
    remainder = mask_count % steps

    num_transfer = mx.repeat(base, steps, axis=1)  # [B, steps]
    step_idx = mx.arange(steps).reshape(1, steps)
    num_transfer = num_transfer + (step_idx < remainder).astype(num_transfer.dtype)
    return num_transfer


def generate(
    model: Model,
    prompt: mx.array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = DEFAULT_MASK_TOKEN_ID,
    tokenizer=None,
    parallel_threshold: Optional[float] = None,
    return_stats: bool = False,
):
    """Diffusion (MDM) generation for LLaDA.

    Two decoding schedules are supported:

    * **Fixed schedule** (``parallel_threshold=None``, the default): the
      original LLaDA path — each block runs a fixed ``steps_per_block`` number
      of forwards, revealing a pre-computed top-``k`` per step. This path is
      byte-identical to the historical behaviour; benches depend on it.

    * **Confidence-aware parallel** (``parallel_threshold`` set, Fast-dLLM
      style): each block runs a *dynamic* number of forwards, unmasking every
      eligible position whose clean-softmax confidence clears the threshold in
      one shot. Because it can reveal many tokens per forward, it cuts the
      number of full forwards (the ~98% cost) well below ``steps``. It is not
      bitwise-equal to the fixed schedule; see the bench for the drift.

    Args:
        model: a built ``Model``.
        prompt: token ids, shape ``[1, prompt_len]`` (or ``[prompt_len]``).
        steps: total denoising steps (split evenly across blocks). Only used by
            the fixed-schedule path.
        gen_length: number of tokens to generate.
        block_length: semi-autoregressive block size; ``gen_length`` must be
            divisible by it.
        temperature: Gumbel sampling temperature (0 == greedy/argmax).
        cfg_scale: classifier-free guidance scale (0 == disabled).
        remasking: ``"low_confidence"`` or ``"random"``. (Fixed-schedule path;
            the parallel path always uses clean-softmax confidence.)
        mask_id: the mask token id.
        tokenizer: optional; if given, the decoded text is returned alongside
            the generated ids.
        parallel_threshold: if set (e.g. ``0.9``), use the confidence-aware
            parallel path and unmask every eligible position whose confidence
            exceeds this value each forward. ``None`` keeps the fixed schedule.
        return_stats: if True, also return a stats dict
            ``{"forwards": int, "tokens_per_step_mean": float, "steps": int}``.

    Returns:
        Generated ids ``[1, gen_length]``; a trailing decoded ``text`` if a
        tokenizer was supplied; and a trailing stats dict if ``return_stats``.
        The return shape when ``return_stats=False`` is unchanged.
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]
    prompt_len = prompt.shape[1]

    total_len = prompt_len + gen_length
    x = mx.full((1, total_len), mask_id, dtype=prompt.dtype)
    x[:, :prompt_len] = prompt
    prompt_index = x != mask_id

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    col_index = mx.arange(total_len).reshape(1, total_len)

    # Count of full forwards actually run (the win metric — ~98% of step cost).
    forwards = 0

    def _forward_logits(x_cur: mx.array) -> mx.array:
        """One full forward (with optional CFG). Increments the forward count."""
        nonlocal forwards
        forwards += 1
        if cfg_scale > 0.0:
            un_x = mx.where(prompt_index, mask_id, x_cur)
            x_ = mx.concatenate([x_cur, un_x], axis=0)
            logits = model(x_)
            cond_logits, uncond_logits = logits[:1], logits[1:]
            return uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        return model(x_cur)

    if parallel_threshold is None:
        # -------------------------------------------------------------------
        # Fixed-schedule path (unchanged historical behaviour).
        # -------------------------------------------------------------------
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block = steps // num_blocks

        for b in range(num_blocks):
            block_start = prompt_len + b * block_length
            block_end = prompt_len + (b + 1) * block_length

            # Mask positions still masked within the current block.
            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                logits = _forward_logits(x)

                noised = add_gumbel_noise(logits, temperature)
                x0 = mx.argmax(noised, axis=-1)  # [1, total_len]

                if remasking == "low_confidence":
                    p = mx.softmax(logits.astype(mx.float32), axis=-1)
                    x0_p = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
                elif remasking == "random":
                    x0_p = mx.random.uniform(shape=x0.shape)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Never reveal past the current block boundary.
                beyond_block = col_index >= block_end
                x0_p = mx.where(beyond_block, neg_inf, x0_p.astype(mx.float32))

                # Only masked positions are candidates; keep existing tokens.
                x0 = mx.where(mask_index, x0, x)
                confidence = mx.where(mask_index, x0_p, neg_inf)

                # Reveal the top-k highest-confidence positions per row.
                transfer_index = _select_topk(
                    confidence, num_transfer_tokens[:, i : i + 1]
                )
                x = mx.where(transfer_index, x0, x)
                mx.eval(x)
    else:
        # -------------------------------------------------------------------
        # Confidence-aware parallel path (Fast-dLLM style).
        #
        # Per block, loop forwards until the block has no masked positions
        # left. Each forward unmasks EVERY eligible position whose clean-
        # softmax confidence clears ``parallel_threshold`` at once. The loop is
        # capped at ``block_length`` iterations (worst case = reveal one token
        # per step), which guarantees termination.
        # -------------------------------------------------------------------
        for b in range(num_blocks):
            block_start = prompt_len + b * block_length
            block_end = prompt_len + (b + 1) * block_length

            # Eligible = masked AND inside the current block window.
            in_block = (col_index >= block_start) & (col_index < block_end)

            step = 0
            while step < block_length:
                mask_index = x == mask_id
                eligible = mask_index & in_block
                # Stop the block once every position in it is filled.
                if int(eligible.sum()) == 0:
                    break

                logits = _forward_logits(x)

                # Token choice honours temperature via Gumbel noise ...
                noised = add_gumbel_noise(logits, temperature)
                x0 = mx.argmax(noised, axis=-1)  # [1, total_len]

                # ... but confidence comes from the CLEAN softmax, not the
                # gumbel-noised logits.
                p = mx.softmax(logits.astype(mx.float32), axis=-1)
                conf = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
                conf = mx.where(eligible, conf.astype(mx.float32), neg_inf)

                # Unmask every eligible position over the confidence bar.
                reveal = eligible & (conf > parallel_threshold)

                # Progress guarantee: if nothing clears the bar, reveal the
                # single most-confident eligible position so the loop advances.
                if int(reveal.sum()) == 0:
                    top = mx.argmax(conf, axis=-1)  # [1]
                    reveal = col_index == top[:, None]

                x = mx.where(reveal, x0, x)
                mx.eval(x)
                step += 1

    out = x[:, prompt_len:]
    mx.eval(out)

    ret = (out,)
    if tokenizer is not None:
        ret = ret + (tokenizer.decode(out[0].tolist()),)
    if return_stats:
        revealed = gen_length  # every gen position ends unmasked
        stats = {
            "forwards": forwards,
            "tokens_per_step_mean": (revealed / forwards) if forwards else 0.0,
            "steps": forwards,
        }
        ret = ret + (stats,)
    return ret[0] if len(ret) == 1 else ret


def _select_topk(confidence: mx.array, k_per_row: mx.array) -> mx.array:
    """Boolean mask selecting the top-``k`` positions per row of ``confidence``.

    ``confidence`` is [B, L] (with ``-inf`` for ineligible positions); ``k_per_row``
    is [B, 1]. Uses an exact rank via ``argsort`` so ties never cause more than
    ``k`` reveals in a row.
    """
    B, L = confidence.shape
    # Descending order: rank 0 = highest confidence.
    order = mx.argsort(-confidence, axis=1)  # [B, L] column indices, best first
    ranks = mx.zeros((B, L), dtype=mx.int32)
    col_positions = mx.arange(L).reshape(1, L)
    # Scatter ranks: ranks[b, order[b, r]] = r.
    ranks = mx.put_along_axis(
        ranks,
        order,
        mx.broadcast_to(col_positions.astype(mx.int32), (B, L)),
        axis=1,
    )
    return ranks < k_per_row.astype(mx.int32)
