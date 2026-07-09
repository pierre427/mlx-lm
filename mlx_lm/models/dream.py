# Copyright © 2026 Pierre Lamy

# Dream-v0-Instruct: masked diffusion language model.
#
# Architecturally this is a Qwen-like decoder block, but Dream generation is
# bidirectional over a full masked sequence instead of causal KV-cache decoding.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs
from .rope_utils import initialize_rope

DEFAULT_MASK_TOKEN_ID = 151666
DEFAULT_EOS_TOKEN_ID = 151643


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "Dream"
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    bos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: int = DEFAULT_EOS_TOKEN_ID

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        h = x + self.self_attn(self.input_layernorm(x), mask=mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class DreamBaseModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h, mask=attention_mask)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DreamBaseModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, attention_mask, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and "rotary_emb.original_inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers


def _sample_logits(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    if temperature == 0.0:
        tokens = mx.argmax(logits, axis=-1)
        confidence = mx.max(mx.softmax(logits, axis=-1), axis=-1)
        return confidence, tokens

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        kth = mx.sort(logits, axis=-1)[:, -top_k]
        logits = mx.where(logits < kth[:, None], -mx.inf, logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_idx = mx.argsort(logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        keep = mx.cumsum(sorted_probs, axis=-1) >= (1.0 - top_p)
        sorted_logits = mx.where(keep, sorted_logits, -mx.inf)
        logits = mx.put_along_axis(logits, sorted_idx, sorted_logits, axis=-1)

    tokens = mx.random.categorical(logits)
    probs = mx.softmax(logits, axis=-1)
    confidence = mx.take_along_axis(probs, tokens[:, None], axis=-1).squeeze(-1)
    return confidence, tokens


def diffusion_generate(
    model: Model,
    inputs: mx.array,
    *,
    max_length: int,
    steps: int = 512,
    eps: float = 0.001,
    mask_token_id: Optional[int] = None,
    alg: str = "origin",
    alg_temp: Optional[float] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    parallel_threshold: Optional[float] = None,
    return_stats: bool = False,
):
    """Dream denoising loop in MLX.

    The official sampler shifts logits right by one before filling masks. This
    keeps that behavior and supports the checkpoint-default ``origin`` mode.
    """
    if mask_token_id is None:
        mask_token_id = model.args.mask_token_id
    if max_length < inputs.shape[1]:
        raise ValueError("max_length must be >= input length")

    pad = mx.full(
        (inputs.shape[0], max_length - inputs.shape[1]), mask_token_id, dtype=inputs.dtype
    )
    x = mx.concatenate([inputs, pad], axis=1)
    timesteps = mx.linspace(1.0, eps, steps + 1)
    forward_count = 0
    transferred_counts = []

    for i in range(steps):
        mask_index = x == mask_token_id
        logits = model(x)
        forward_count += 1
        logits = mx.concatenate([logits[:, :1], logits[:, :-1]], axis=1)
        if mx.all(~mask_index).item():
            break

        t = timesteps[i].item()
        s = timesteps[i + 1].item()
        if alg == "origin":
            p_transfer = 1.0 - s / t if i < steps - 1 else 1.0
            confidence, sampled = _sample_logits(
                logits.reshape(-1, logits.shape[-1]),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            sampled = sampled.reshape(x.shape)
            if parallel_threshold is None:
                transfer = mx.random.uniform(shape=x.shape) < p_transfer
            else:
                confidence = confidence.reshape(x.shape)
                transfer = confidence >= parallel_threshold
                if i == steps - 1:
                    transfer = mx.ones_like(transfer)
            transfer = mask_index & transfer
            transferred_counts.append(int(mx.sum(transfer).item()))
            x = mx.where(transfer, sampled, x)
        else:
            raise NotImplementedError(
                f"Dream diffusion alg '{alg}' is not implemented in this MLX port yet"
            )

    if return_stats:
        generated = max(1, int(mx.sum(x[:, inputs.shape[1] :] != mask_token_id).item()))
        return x, {
            "forwards": forward_count,
            "transferred_total": int(sum(transferred_counts)),
            "tokens_per_step_mean": float(generated / max(1, forward_count)),
            "parallel_threshold": parallel_threshold,
        }
    return x
