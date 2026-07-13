# Copyright © 2026 raullenchai and the Rapid-MLX contributors (original work)
# Original PFlash design © 2026 @michaelasper (Rapid-MLX #287)
# Copyright © 2026 Pierre Lamy (mlx-uag port and adaptation)
# SPDX-License-Identifier: Apache-2.0
"""PFlash-style token-statistical prompt compression for prefill.

Ported from ``raullenchai/Rapid-MLX`` (`vllm_mlx/pflash.py`, Apache-2.0), reduced
to the pure compressor (the alias-tier resolver and MLLM guards that depended on
vllm_mlx internals are dropped). See
``wiki/docs/research/rapid-mlx-competitive-2026-07-13.md`` for provenance.

PFlash trades recall on the middle of very long prompts for a cold-prefill TTFT
win. It keeps the leading sink + trailing tail verbatim, then fills a keep-ratio
budget with middle blocks ranked by tail-query overlap and token rarity. Scoring
is deterministic and uses only ``collections.Counter`` — no Metal device, no new
dependency. It is **lossy on prompt content**; whether that costs task accuracy
is exactly what bench_pflash.py measures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Literal

PFlashMode = Literal["off", "auto", "always"]


@dataclass(frozen=True)
class PFlashConfig:
    """Configuration for PFlash prompt compression (defaults from Rapid-MLX #649)."""

    mode: PFlashMode = "off"
    threshold: int = 32_768
    keep_ratio: float = 0.20
    min_keep_tokens: int = 2_048
    sink_tokens: int = 256
    tail_tokens: int = 2_048
    block_size: int = 128
    query_window: int = 512
    stride_blocks: int = 8
    skip_when_tools: bool = True

    def validate(self) -> "PFlashConfig":
        if self.mode not in ("off", "auto", "always"):
            raise ValueError("pflash mode must be one of: off, auto, always")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be > 0.0 and <= 1.0")
        for name in ("threshold", "min_keep_tokens", "sink_tokens", "tail_tokens", "stride_blocks"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("block_size", "query_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        return self


@dataclass(frozen=True)
class PFlashResult:
    tokens: list[int]
    compressed: bool
    reason: str
    original_tokens: int
    kept_tokens: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.kept_tokens / self.original_tokens


@dataclass(frozen=True)
class _BlockScore:
    start: int
    end: int
    score: float


def compress_tokens(
    tokens: list[int],
    config: PFlashConfig,
    *,
    has_tools: bool = False,
    requires_prompt_integrity: bool = False,
) -> PFlashResult:
    """Compress a token list per PFlash settings.

    Always preserves the leading sink and trailing tail; fills the remaining
    budget with middle blocks ranked by tail-query overlap and token rarity.
    Output preserves original order.
    """
    n_tokens = len(tokens)
    if requires_prompt_integrity:
        return _unchanged(tokens, "protected_prompt")
    if has_tools and config.skip_when_tools:
        return _unchanged(tokens, "tools")
    if n_tokens == 0:
        return _unchanged(tokens, "empty")
    if config.mode == "off":
        return _unchanged(tokens, "off")
    if config.mode == "auto" and n_tokens < config.threshold:
        return _unchanged(tokens, "threshold")

    block_size = max(1, config.block_size)
    keep_budget = _keep_budget(n_tokens, config)
    if keep_budget >= n_tokens:
        return _unchanged(tokens, "budget")

    sink_end = min(max(0, config.sink_tokens), n_tokens)
    tail_start = max(sink_end, n_tokens - max(0, config.tail_tokens))

    keep_positions = set(range(sink_end))
    keep_positions.update(range(tail_start, n_tokens))

    remaining_budget = keep_budget - len(keep_positions)
    if remaining_budget > 0:
        scored_blocks = _score_middle_blocks(
            tokens=tokens,
            start=sink_end,
            stop=tail_start,
            block_size=block_size,
            query_window=max(1, config.query_window),
            stride_blocks=max(0, config.stride_blocks),
        )
        selected = 0
        for block in scored_blocks:
            slots = remaining_budget - selected
            if slots <= 0:
                break
            take = min(block.end - block.start, slots)
            keep_positions.update(range(block.start, block.start + take))
            selected += take
            if selected >= remaining_budget:
                break

    kept = [tokens[i] for i in sorted(keep_positions)]
    if len(kept) >= n_tokens:
        return _unchanged(tokens, "budget")
    return _changed(tokens, kept, "compressed")


def _keep_budget(n_tokens: int, config: PFlashConfig) -> int:
    ratio_budget = ceil(n_tokens * _clamp(config.keep_ratio, 0.0, 1.0))
    return max(1, min(n_tokens, max(config.min_keep_tokens, ratio_budget)))


def _score_middle_blocks(
    *,
    tokens: list[int],
    start: int,
    stop: int,
    block_size: int,
    query_window: int,
    stride_blocks: int,
) -> list[_BlockScore]:
    if start >= stop:
        return []
    counts = Counter(tokens)
    query = tokens[max(0, len(tokens) - query_window):]
    query_counts = Counter(query)
    span = max(1, stop - start)

    blocks: list[_BlockScore] = []
    for block_index, block_start in enumerate(range(start, stop, block_size)):
        block_end = min(block_start + block_size, stop)
        block = tokens[block_start:block_end]
        overlap = sum(query_counts.get(t, 0) / counts[t] for t in block)
        rarity = sum(1.0 / counts[t] for t in block) / len(block)
        recency = (block_end - start) / span
        stride_bonus = 0.25 if stride_blocks and block_index % stride_blocks == 0 else 0.0
        score = (4.0 * overlap) + rarity + (0.05 * recency) + stride_bonus
        blocks.append(_BlockScore(block_start, block_end, score))
    return sorted(blocks, key=lambda item: (-item.score, item.start))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unchanged(tokens: list[int], reason: str) -> PFlashResult:
    return PFlashResult(tokens, False, reason, len(tokens), len(tokens))


def _changed(tokens: list[int], kept: list[int], reason: str) -> PFlashResult:
    return PFlashResult(kept, True, reason, len(tokens), len(kept))


__all__ = ["PFlashConfig", "PFlashResult", "compress_tokens"]
