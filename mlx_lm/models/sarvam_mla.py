# Copyright © 2026 mlx-uag lab
#
# Sarvam-105B (`sarvam_mla` / SarvamMLAForCausalLM). Architecturally this is
# DeepSeek-V3: MLA attention (kv_lora_rank 512, q_head_dim 192, no q_lora_rank →
# direct q_proj) + sigmoid `noaux_tc` group-topk MoE with expert-bias correction +
# a shared expert + `first_k_dense_replace`. The reference `use_qk_norm` flag is a
# no-op in Sarvam's own attention forward (standard MLA, no per-head q/k norm), so
# we reuse the DeepSeek-V3 implementation wholesale — it already caches the 512-dim
# latent and does absorbed-decode / materialized-prefill via embed_q / unembed_out,
# and its `sanitize` folds `kv_b_proj` (incl. quantized) into those.
#
# This module only remaps Sarvam's config key names onto the V3 ModelArgs.

from dataclasses import dataclass
from typing import Optional

from .deepseek_v3 import DeepseekV3Model  # noqa: F401  (re-exported for tooling)
from .deepseek_v3 import Model as DeepseekV3ForCausalLM
from .deepseek_v3 import ModelArgs as _V3Args


@dataclass
class ModelArgs(_V3Args):
    model_type: str = "sarvam_mla"
    # Sarvam uses different key names for the MoE cardinalities:
    num_experts: Optional[int] = None          # -> n_routed_experts
    num_shared_experts: Optional[int] = None   # -> n_shared_experts
    # Sarvam has no q_lora_rank -> direct q_proj path (V3 default is 1536).
    q_lora_rank: Optional[int] = None
    # present in Sarvam config; unused by the V3 attention math but kept so
    # BaseModelArgs.from_dict doesn't choke and for documentation.
    q_head_dim: int = 192
    head_dim: int = 576
    use_qk_norm: bool = False
    moe_router_enable_expert_bias: bool = True
    default_theta: float = 10000.0

    def __post_init__(self):
        # Map Sarvam cardinalities onto the DeepSeek-V3 field names.
        if self.num_experts is not None:
            self.n_routed_experts = self.num_experts
        if self.num_shared_experts is not None:
            self.n_shared_experts = self.num_shared_experts
        # Sarvam omits the group-routing dims; its gate defaults to
        # n_group = n_routed_experts // 8, topk_group = 2 (see modeling MoEGate).
        n_routed = self.n_routed_experts or self.num_experts or 128
        if self.n_group in (None, 1):
            self.n_group = n_routed // 8
        if self.topk_group in (None, 1):
            self.topk_group = 2


class Model(DeepseekV3ForCausalLM):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model_type = config.model_type
