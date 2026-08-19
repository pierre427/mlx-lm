# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Recorded fixtures for the absolute-work promotion schema.

Five measured cases where a favorable RATE coincided with equal-or-worse
end-to-end WORK. Numbers are pulled from the lab record:

- E17     wiki/docs/experiments/termination-attractor-steering-results.md
          (frozen 20-prompt Mellum ablation, 2026-07-13)
- Vegas   wiki/docs/experiments/vegas-sparsekv-2026-07-13.md
- KV8     wiki/docs/lessons/quantized-attention-query-routing.md
          + experiments/qsdpa-hybrid-collab-2026-07-09.md (Coder-Next)
- D-Cut   wiki/docs/plans/peer-lever-adoption-2026-07-10.md §W3 result row
- SnapKV-D wiki/docs/experiments/epicache-layer-budget-2026-07-13.md
          + research/conversation-mined-insights-2026-07-13.md §5

Each entry gives a (candidate, baseline) pair and the verdict the record
already reached, so the schema can be regression-tested against known truth.

Where the record gives throughputs rather than seconds, wall_time_s is a
normalized proxy (baseline decode = 1.0 s; candidate = 1.0 / relative-speedup),
which is monotone in real wall time at fixed output length. Step means for E17
are the recorded per-prompt means (kept as floats, not rounded).
"""

from __future__ import annotations

from .absolute_work import AbsoluteWork, RATE_TRAP, NEUTRAL, HORIZON

_GB = 1024 ** 3


def _e17():
    # Frozen Mellum ablation. Baseline = the `time` schedule (duty 100%);
    # candidate = the `sensor` controller (duty 56.38%). Recorded means:
    #   time:   quality 0.7767, think 169.15, steered 169.15, duty 100.0%
    #   sensor: quality 0.7892, think 217.55, steered 153.25, duty 56.38%
    # Duty fell 100%->56.4% but absolute steered steps fell only 9.40%
    # (169.15 -> 153.25) and delayed termination LENGTHENED the trajectory
    # (217.55 vs 169.15 think tokens). wall proxy = think tokens.
    baseline = AbsoluteWork(
        label="E17-time-schedule",
        wall_time_s=169.15,
        target_steps=169.15,   # one target forward per generated think token
        steered_steps=169.15,
        quality=0.7767,
        peak_bytes=0,
        rates={"duty": 1.0},
    )
    candidate = AbsoluteWork(
        label="E17-sensor",
        wall_time_s=217.55,
        target_steps=217.55,
        steered_steps=153.25,
        quality=0.7892,
        peak_bytes=0,
        rates={"duty": 0.5638},
        baseline=baseline,
    )
    return candidate, baseline, RATE_TRAP


def _vegas():
    # Sparse-KV self-spec, lossless in the spec sense, but every setting < 1x
    # throughput on M5 (0.63-0.90x). Use the 16k best-sparse column 0.63x.
    # Acceptance ~41% is far too low to overcome the extra draft forward.
    # Baseline = plain greedy decode (no draft).
    baseline = AbsoluteWork(
        label="Vegas-plain-decode-16k",
        wall_time_s=1.0,             # normalized decode
        target_steps=1000,
        draft_steps=0,
        quality=1.0,                 # lossless -> equal quality
        peak_bytes=8 * _GB,
        rates={},
    )
    candidate = AbsoluteWork(
        label="Vegas-sparsekv-16k",
        wall_time_s=1.0 / 0.63,      # 0.63x throughput -> 1.587s
        target_steps=1000,
        draft_steps=1000,           # drafts every step; ~a full forward on M5
        quality=1.0,                 # lossless
        peak_bytes=8 * _GB,
        rates={"acceptance": 0.41},  # looks like a working spec path
        baseline=baseline,
    )
    return candidate, baseline, RATE_TRAP


def _kv8():
    # Quantized-KV (kv8) on Qwen3-Coder-Next: decode 78.15 t/s vs fp16 89.40
    # t/s (-12.584%), TTFT +0.862%. Saves KV memory but slows decode. On a
    # memory-UNCONSTRAINED target the peak reduction buys nothing you need while
    # decode regresses -> NEUTRAL tradeoff, not a default. (On a memory-bound
    # target the same numbers flip to SHIP: peak reduction becomes the binding
    # win.) 1000 decode tokens: wall = 1000/tps.
    baseline = AbsoluteWork(
        label="KV8-fp16-CoderNext",
        wall_time_s=1000 / 89.40,    # 11.186 s
        target_steps=1000,
        quality=1.0,
        peak_bytes=4 * _GB,
        resident_bytes=4 * _GB,
        rates={},
    )
    candidate = AbsoluteWork(
        label="KV8-quantized-CoderNext",
        wall_time_s=1000 / 78.15,    # 12.796 s (slower)
        target_steps=1000,
        quality=1.0,                 # identity preserved
        peak_bytes=3 * _GB,          # ~half KV -> lower peak
        resident_bytes=3 * _GB,
        rates={"mem_saved": 1 * _GB, "compression": 2.0},
        baseline=baseline,
    )
    return candidate, baseline, NEUTRAL


def _dcut():
    # D-Cut expected-value draft pruning for MTP self-spec: WORKS (1.60x over
    # num_draft=8) but DOMINATED by a smaller static num_draft=2 (2.05x). The
    # pruning trims verify work / raises effective acceptance, but the loop
    # still DRAFTS EVERY TOKEN at width 8, so the static nd=2 baseline does far
    # fewer draft steps. Baseline = static num_draft=2; both measured vs the
    # nd=8 reference (1.0), so wall = 1/relative-speedup. 100 output tokens.
    baseline = AbsoluteWork(
        label="D-Cut-static-nd2",
        wall_time_s=1.0 / 2.05,      # 0.488 s
        target_steps=100,
        draft_steps=200,            # width-2 draft per token
        quality=1.0,
        peak_bytes=2 * _GB,
        rates={"acceptance": 0.70},
    )
    candidate = AbsoluteWork(
        label="D-Cut-prune-nd8",
        wall_time_s=1.0 / 1.60,      # 0.625 s (slower than static nd=2)
        target_steps=100,
        draft_steps=800,            # STILL drafts width-8 every token
        quality=1.0,
        peak_bytes=2 * _GB,
        rates={"acceptance": 0.85},  # pruning raises acceptance length
        baseline=baseline,
    )
    return candidate, baseline, RATE_TRAP


def _snapkv_d():
    # SnapKV-D compact cache: often improves steady-state decode, but the
    # compaction PREPARATION dominates normal-length wall time -> the win is
    # horizon-dependent (pays off with a longer request or a reused prepared
    # cache). Baseline = full KV cache. Normal-length request (say 200 decode
    # tokens): candidate pays a prep tax up front, its whole-request wall is
    # flat-to-slightly-worse, but its decode phase reads fewer KV bytes and
    # runs faster, and its peak is lower. Breaks even after ~3 reuses.
    baseline = AbsoluteWork(
        label="SnapKV-D-full-cache",
        wall_time_s=1.00,
        prefill_prep_s=0.05,
        target_steps=200,
        bytes_read=8 * _GB,          # full KV streamed per step
        quality=1.0,
        peak_bytes=8 * _GB,
        rates={},
    )
    candidate = AbsoluteWork(
        label="SnapKV-D-compact",
        wall_time_s=1.02,            # slightly WORSE on a normal request (prep)
        prefill_prep_s=0.30,        # compaction preparation dominates
        target_steps=200,
        bytes_read=5 * _GB,          # compact cache -> fewer bytes/step
        quality=1.0,
        peak_bytes=5 * _GB,          # lower peak
        amortization_horizon=3,      # break even after ~3 reuses
        rates={"compression": 1.6, "cache_hit": 0.9},
        baseline=baseline,
    )
    return candidate, baseline, HORIZON


FIXTURES = {
    "E17": _e17(),
    "Vegas": _vegas(),
    "KV8": _kv8(),
    "D-Cut": _dcut(),
    "SnapKV-D": _snapkv_d(),
}

__all__ = ["FIXTURES"]
