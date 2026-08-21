# Cohere2-MoE support for mlx-lm.
#
# Architecture: Cohere2MoeForCausalLM (model_type "cohere2_moe"), used by
# Cohere's North-Mini-Code-1.0 (30B total / 3B active MoE coding model).
#
# It is the Cohere2 (Command R7B) decoder — parallel attention+MLP block, a
# single LayerNorm(bias=False) per layer, 3:1 sliding:global attention with
# NoPE on the global layers, logit_scale, tied embeddings — with the dense MLP
# replaced by a sigmoid-routed MoE on all but the first `first_k_dense_replace`
# layers.
#
# Deltas from our laguna.py MoE port (same family, different vendor):
#   * parallel block + single LayerNorm (not sequential + RMSNorm)
#   * no QK-norm, no per-head attention output gating (poolside-only)
#   * no shared expert, no router e_score_correction_bias
#   * NoPE (no RoPE) on full_attention (global) layers; RoPE only on sliding
#   * full-attention layers are chosen from the explicit `layer_types` array
#     (North's phase is i % 4 == 0), never a modulus guess.
#
# Checkpoint note: the mlx-community bf16/8bit repacks were converted via
# mlx-vlm and wrap every tensor under a `language_model.` prefix, with experts
# already stacked into `switch_mlp` and the router already at `mlp.gate`. The
# only sanitize needed is the prefix strip (same gotcha as the AtomicChat
# Laguna repack). See NORTH_MINI_CODE_PLAN.md.
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 2048
    head_dim: int = 128
    num_hidden_layers: int = 49
    intermediate_size: int = 768
    prefix_dense_intermediate_size: int = 3072
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 262144
    rope_theta: float = 50000.0
    layer_norm_eps: float = 1e-5
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    max_position_embeddings: int = 500000
    tie_word_embeddings: bool = True
    # MoE
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 0
    norm_topk_prob: bool = False
    first_k_dense_replace: int = 1
    expert_selection_fn: str = "sigmoid"
    layer_types: Optional[List[str]] = None
    # Optional self-speculation (MTP / EAGLE-style depth-1) head. 0 = absent
    # (default; serving unaffected). Trained by train_mtp_north.py.
    mtp_num_hidden_layers: int = 0
    # tolerated-but-unused config keys (kept so BaseModelArgs.from_dict is happy)
    use_parallel_block: bool = True
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            # Fall back to the North 3:1 pattern: full attention every 4th layer
            # starting at layer 0.
            self.layer_types = [
                "full_attention" if (i % 4 == 0) else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers.")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseBlock(nn.Module):
    """Sigmoid-routed MoE. No shared expert, no router correction bias.

    Router weights live at `mlp.gate.weight`; stacked experts at
    `mlp.switch_mlp.{gate,up,down}_proj.*` — matching the checkpoint layout,
    so no key remapping is needed.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.expert_selection_fn == "sigmoid"
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        logits = self.gate(x).astype(mx.float32)
        scores = (
            mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        )

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        weights = weights.astype(dtype)

        y = self.switch_mlp(x, inds)
        return mx.sum(y * weights[..., None], axis=-2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)

        # RoPE only on sliding layers; global (full_attention) layers are NoPE.
        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)
            if self.is_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = getattr(self, "qkv_proj", None)
        if qkv is not None:
            # Fused projection (see fuse_qkv_projections): one GEMV over the
            # row-concatenated [Wq; Wk; Wv], then split. Row-wise quantization
            # makes this numerically identical to the three separate GEMVs.
            fused = qkv(x)
            nq = self.n_heads * self.head_dim
            nk = self.n_kv_heads * self.head_dim
            queries = fused[..., :nq]
            keys = fused[..., nq : nq + nk]
            values = fused[..., nq + nk :]
        else:
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if layer_idx < args.first_k_dense_replace or args.num_experts == 0:
            self.mlp = MLP(args.hidden_size, args.prefix_dense_intermediate_size)
        else:
            self.mlp = Cohere2MoeSparseBlock(args)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )
        self.attention_type = args.layer_types[layer_idx]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Parallel block (Cohere): attention and MLP both read the same
        # normalized hidden state and their outputs are summed into the residual.
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return x + attn_h + ff_h


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)
        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, c)
        return self.norm(h)


class Cohere2MoeMTP(nn.Module):
    """Depth-1 self-speculation head (EAGLE/MTP style) for cohere2_moe.

    Predicts token p+2 from the trunk's post-final-norm hidden at p and the
    embedding of the committed token p+1: fuse the two (each LayerNorm'd, then a
    linear over the concat), run one Cohere2 parallel block (dense MLP + full
    causal RoPE attention), and decode with the backbone's tied lm_head. The
    embedding and lm_head are the backbone's (passed in) — only this module is
    trained. Mirrors the Qwen3NextMTP recipe; see train_mtp_north.py.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        h = args.hidden_size
        eps = args.layer_norm_eps
        self.hnorm = nn.LayerNorm(h, eps=eps, bias=False)
        self.enorm = nn.LayerNorm(h, eps=eps, bias=False)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        # layer_idx=1 is a sliding layer -> RoPE enabled; we pass a plain causal
        # mask (training seq < window 4096, so it is full causal anyway).
        self.self_attn = Attention(args, layer_idx=1)
        self.mlp = MLP(h, args.prefix_dense_intermediate_size)
        self.input_layernorm = nn.LayerNorm(h, eps=eps, bias=False)
        self.norm = nn.LayerNorm(h, eps=eps, bias=False)

    def __call__(self, hidden: mx.array, embeds: mx.array, cache=None) -> mx.array:
        h = self.eh_proj(
            mx.concatenate([self.hnorm(hidden), self.enorm(embeds)], axis=-1)
        )
        mask = create_attention_mask(h, cache)
        hn = self.input_layernorm(h)
        h = h + self.self_attn(hn, mask, cache) + self.mlp(hn)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        self.logit_scale = args.logit_scale
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.mtp_num_hidden_layers > 0:
            self.mtp = Cohere2MoeMTP(args)

    def logits(self, hidden: mx.array) -> mx.array:
        """Apply the (tied) lm_head to a hidden state. Standalone so a self-spec
        engine can verify without recomputing the trunk (self_mtp_generate_step)."""
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)
        return out * self.logit_scale

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        return self.logits(self.model(inputs, cache))

    def make_cache(self):
        # 36 of 49 layers are sliding_attention (window 4096): a bounded
        # RotatingKVCache is both correct and far cheaper at long context. Only
        # the 13 full_attention (global, NoPE) layers need an unbounded KVCache.
        caches = []
        for lt in self.args.layer_types:
            if lt == "full_attention" or not self.args.sliding_window:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def make_mtp_cache(self):
        return [KVCache()]

    def mtp_step(self, hidden: mx.array, tokens: mx.array, cache):
        """One self-spec step: given trunk hidden states and the committed
        tokens, return the head's next-token logits (via the model's lm_head,
        tied or not)."""
        embeds = self.model.embed_tokens(tokens)
        h = self.mtp(hidden, embeds, cache[0])
        return self.logits(h), h

    def sanitize(self, weights):
        # Drop a stray mtp.* head when this model has none (keeps base-checkpoint
        # loads clean); keep it when the head module exists (merged self-spec ckpt).
        if self.args.mtp_num_hidden_layers == 0:
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        # mlx-vlm repacks (mlx-community North-Mini-Code-1.0-*) wrap every
        # tensor under a `language_model.` prefix. Strip it so keys line up with
        # this module tree (model.* / lm_head.*). Experts are already stacked
        # into switch_mlp and the router is already mlp.gate — no remap needed.
        if any(k.startswith("language_model.") for k in weights):
            prefix = "language_model."
            weights = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in weights.items()
            }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k
        }

    @property
    def quant_predicate(self):
        # Keep the router at 8-bit (quant-sensitive component), matching the
        # Laguna finding that the gate is the precision-critical weight.
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers


def fuse_qkv_projections(model) -> int:
    """Runtime graph-level fusion: replace each attention layer's q/k/v
    projections with ONE row-concatenated projection (quantized or dense).

    Motivation (megakernel M-1, 2026-08-21): at M=1 the 2048->512 k/v GEMVs
    run at ~9% of bandwidth because every qmv kernel pays a ~10 us floor; a
    single 2048->5120 GEMV pays it once. Per-row (group) quantization means
    the concatenated rows are computed exactly as before -> token-identical.
    Weights on disk are untouched; call after load(). Returns #layers fused.
    Idempotent. The originals are dropped to free memory, so a fused model
    must not be saved.

    MEASURED NEGATIVE on North q4 (2026-08-21): bit-exact but -2.6% sync /
    -4% async. MLX's Metal encoder already runs the three independent GEMVs
    concurrently (separate+views 12.1 us/layer vs fused 15.2 us), so the
    single-kernel form loses. Kept opt-in for other targets/shapes; do not
    enable by default.
    """
    import copy

    fused_layers = 0
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "qkv_proj", None) is not None:
            continue
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
        kinds = {type(q), type(k), type(v)}
        if len(kinds) != 1:
            continue
        fused = copy.copy(q)
        fused.weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
        if isinstance(q, nn.QuantizedLinear):
            fused.scales = mx.concatenate([q.scales, k.scales, v.scales], axis=0)
            if "biases" in q:
                fused.biases = mx.concatenate([q.biases, k.biases, v.biases], axis=0)
        if "bias" in q:
            fused.bias = mx.concatenate([q.bias, k.bias, v.bias], axis=0)
        attn.qkv_proj = fused
        attn.pop("q_proj")
        attn.pop("k_proj")
        attn.pop("v_proj")
        fused_layers += 1
    if fused_layers:
        mx.eval(model.parameters())
    return fused_layers
