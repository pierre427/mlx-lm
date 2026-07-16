# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Absolute-work promotion schema.

Rate metrics (acceptance, steering duty cycle, compression ratio, cache-hit
rate, tok/s) are *diagnostic rates*, not shipping verdicts. The lab has
repeatedly measured a favorable rate that coincided with equal-or-worse
end-to-end work: a lower steering duty that still pays nearly the same absolute
steered steps (E17), a lossless spec path that still loses net throughput
(Vegas), a quantized cache that saves memory but slows decode (KV8), a pruned
verify chain that still drafts every token (D-Cut).

The fix is a common *absolute-work vector* — wall time, target/draft/steered
steps, bytes read / resident / peak, preparation cost, quality, and an
amortization horizon — plus a verdict function that FAILS CLOSED: when a rate
improves while total steps, wall time, or peak memory regress, the candidate is
a RATE-TRAP, not a ship.

Dependency-free, CPU-only, pure Python. No mlx import, no model load.

See wiki: research/conversation-mined-insights-2026-07-13.md §5 and
experiments/absolute-work-schema-2026-07-15.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

# Verdict strings.
SHIP = "SHIP"
RATE_TRAP = "RATE-TRAP"
NEUTRAL = "NEUTRAL"
HORIZON = "HORIZON-DEPENDENT"


@dataclass
class AbsoluteWork:
    """Absolute end-to-end work for one configuration on one request/battery.

    All fields are *absolute totals* for the measured unit of work, never rates.
    A rate (acceptance, duty cycle, tok/s, compression) may be attached via
    ``rates`` for diagnostics, but it never enters a verdict.

    ``baseline`` optionally holds the counterpart this configuration is measured
    against (e.g. the static/off arm). When present, ``verdict()`` and
    ``is_rate_trap()`` can be called with no argument.
    """

    label: str = ""

    # Wall-clock and preparation.
    wall_time_s: float = 0.0            # total request wall time
    prefill_prep_s: float = 0.0         # one-time prefill/preparation cost

    # Step counts (absolute totals, not per-token rates).
    target_steps: int = 0               # full target-model forward steps
    draft_steps: int = 0                # draft/proposer forward steps
    steered_steps: int = 0              # steering/actuator interventions

    # Memory / bandwidth (bytes).
    bytes_read: int = 0                 # bytes read (KV / weights streamed)
    resident_bytes: int = 0            # steady-state resident footprint
    peak_bytes: int = 0                 # peak allocation

    # Quality — arbitrary but comparable score (higher is better).
    quality: float = 0.0

    # Requests needed to break even on preparation cost. inf = never amortizes,
    # 0 = no preparation to amortize.
    amortization_horizon: float = 0.0

    # Diagnostic rates only (never enter a verdict). e.g. {"duty": 0.564}.
    rates: dict = field(default_factory=dict)

    # Optional counterpart this row is scored against.
    baseline: Optional["AbsoluteWork"] = None

    # ------------------------------------------------------------------ derived
    @property
    def total_steps(self) -> int:
        """All model/actuator steps that cost compute this request."""
        return self.target_steps + self.draft_steps + self.steered_steps

    def _resolve_baseline(
        self, baseline: Optional["AbsoluteWork"]
    ) -> "AbsoluteWork":
        b = baseline if baseline is not None else self.baseline
        if b is None:
            raise ValueError(
                f"{self.label or 'candidate'}: no baseline supplied"
            )
        return b

    def work_delta_vs(self, baseline: Optional["AbsoluteWork"] = None) -> dict:
        """Signed candidate-minus-baseline deltas on every absolute axis.

        Positive means the candidate does MORE work / is worse on that axis
        (except ``quality`` and ``amortization_horizon``, which are reported
        raw so callers can read their own polarity).
        """
        b = self._resolve_baseline(baseline)
        return {
            "wall_time_s": self.wall_time_s - b.wall_time_s,
            "prefill_prep_s": self.prefill_prep_s - b.prefill_prep_s,
            "total_steps": self.total_steps - b.total_steps,
            "target_steps": self.target_steps - b.target_steps,
            "draft_steps": self.draft_steps - b.draft_steps,
            "steered_steps": self.steered_steps - b.steered_steps,
            "bytes_read": self.bytes_read - b.bytes_read,
            "resident_bytes": self.resident_bytes - b.resident_bytes,
            "peak_bytes": self.peak_bytes - b.peak_bytes,
            "quality": self.quality - b.quality,
            "amortization_horizon": self.amortization_horizon,
        }

    # ---------------------------------------------------------- rate detection
    def _improved_rate(self, baseline: "AbsoluteWork") -> Optional[str]:
        """Return the name of a diagnostic rate that looks *better* than the
        baseline, or None. Polarity is per-metric:

        - duty / duty_cycle: lower is "better" (less actuator on-time)
        - acceptance / accept: higher is "better"
        - compression / compression_ratio: higher is "better"
        - cache_hit / cache_hit_rate: higher is "better"
        - tps / tok_s: higher is "better"
        - mem_saved / memory_saved: any positive value is "better"
        """
        lower_better = ("duty", "duty_cycle")
        higher_better = (
            "acceptance", "accept", "compression", "compression_ratio",
            "cache_hit", "cache_hit_rate", "tps", "tok_s", "mem_saved",
            "memory_saved",
        )
        for name, val in self.rates.items():
            bval = baseline.rates.get(name)
            if name in lower_better:
                if bval is None or val < bval:
                    return name
            elif name in higher_better:
                if name in ("mem_saved", "memory_saved"):
                    if val and val > 0:
                        return name
                elif bval is None or val > bval:
                    return name
        return None

    def regresses_work_vs(self, baseline: "AbsoluteWork") -> list:
        """Absolute-work axes on which the candidate is equal-or-worse.

        Fails closed: an axis where the candidate ties the baseline is NOT a
        regression, but ``is_rate_trap`` treats "no absolute improvement" as the
        trap condition — see that method.
        """
        reasons = []
        if self.wall_time_s > baseline.wall_time_s:
            reasons.append(
                f"wall_time {self.wall_time_s:g}s > {baseline.wall_time_s:g}s"
            )
        if self.total_steps > baseline.total_steps:
            reasons.append(
                f"total_steps {self.total_steps} > {baseline.total_steps}"
            )
        if self.peak_bytes > baseline.peak_bytes:
            reasons.append(
                f"peak_bytes {self.peak_bytes} > {baseline.peak_bytes}"
            )
        return reasons

    def is_rate_trap(self, baseline: Optional["AbsoluteWork"] = None) -> bool:
        """True when a favorable RATE coincides with equal-or-worse WORK.

        Fails closed: if a diagnostic rate improved but the candidate does not
        strictly reduce *any* of wall time, total steps, or peak bytes — i.e.
        it regresses or ties on all three headline absolute axes — the rate is
        a trap.
        """
        b = self._resolve_baseline(baseline)
        if self._improved_rate(b) is None:
            return False
        # Did the candidate strictly beat the baseline on any headline axis?
        beats = (
            self.wall_time_s < b.wall_time_s
            or self.total_steps < b.total_steps
            or self.peak_bytes < b.peak_bytes
        )
        return not beats

    def verdict(self, baseline: Optional["AbsoluteWork"] = None) -> dict:
        """Classify the candidate: SHIP / RATE-TRAP / NEUTRAL / HORIZON.

        Returns ``{"verdict", "reason", "rate", "work_regressions"}``.
        """
        b = self._resolve_baseline(baseline)
        improved_rate = self._improved_rate(b)
        regressions = self.regresses_work_vs(b)

        # Horizon-dependent: preparation cost that only pays off after N
        # requests, with a steady-state improvement behind it. "Steady state"
        # is the request minus its one-time preparation — a compact-cache lever
        # (SnapKV-D) reads fewer bytes and decodes faster per step, yet its
        # single-request wall can still be flat-or-worse because prep dominates
        # normal-length requests. Detect the decode-phase / bandwidth win, not
        # the whole-request wall.
        horizon = self.amortization_horizon
        decode_wall_self = self.wall_time_s - self.prefill_prep_s
        decode_wall_base = b.wall_time_s - b.prefill_prep_s
        steady_win = (
            decode_wall_self < decode_wall_base
            or (self.bytes_read and self.bytes_read < b.bytes_read)
        )

        if self.is_rate_trap(b):
            rate_txt = improved_rate or "a rate"
            reason = (
                f"{rate_txt} improved but absolute work did not: "
                + "; ".join(regressions or ["no absolute-work axis improved"])
            )
            return {
                "verdict": RATE_TRAP,
                "reason": reason,
                "rate": improved_rate,
                "work_regressions": regressions,
            }

        if horizon and horizon != float("inf") and horizon > 1 and steady_win:
            return {
                "verdict": HORIZON,
                "reason": (
                    f"preparation dominates short requests; breaks even after "
                    f"~{horizon:g} requests, then steady-state decode wins"
                ),
                "rate": improved_rate,
                "work_regressions": regressions,
            }

        # A genuine ship: strictly reduces a headline axis without regressing
        # wall time, total steps, or peak bytes, and does not lose quality.
        beats_headline = (
            self.wall_time_s < b.wall_time_s
            or self.total_steps < b.total_steps
            or self.peak_bytes < b.peak_bytes
        )
        if beats_headline and not regressions and self.quality >= b.quality:
            gains = []
            if self.wall_time_s < b.wall_time_s:
                gains.append(
                    f"wall_time {self.wall_time_s:g}s < {b.wall_time_s:g}s"
                )
            if self.total_steps < b.total_steps:
                gains.append(
                    f"total_steps {self.total_steps} < {b.total_steps}"
                )
            if self.peak_bytes < b.peak_bytes:
                gains.append(
                    f"peak_bytes {self.peak_bytes} < {b.peak_bytes}"
                )
            return {
                "verdict": SHIP,
                "reason": "absolute work strictly reduced: " + "; ".join(gains),
                "rate": improved_rate,
                "work_regressions": [],
            }

        return {
            "verdict": NEUTRAL,
            "reason": (
                "no rate trap, but no strict absolute-work win either "
                "(equal-or-mixed on headline axes)"
            ),
            "rate": improved_rate,
            "work_regressions": regressions,
        }


def _fmt_bytes(n: int) -> str:
    if not n:
        return "0"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def promotion_table(rows) -> str:
    """Render a comparison table of (candidate, baseline) pairs.

    ``rows`` is an iterable of ``AbsoluteWork`` candidates that each carry a
    ``baseline``, or of ``(candidate, baseline)`` tuples. Returns a GitHub-
    flavoured Markdown table.
    """
    header = (
        "| candidate | verdict | Δwall_s | Δtotal_steps | Δpeak | Δquality "
        "| horizon | reason |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        if isinstance(row, tuple):
            cand, base = row
        else:
            cand, base = row, row.baseline
        v = cand.verdict(base)
        d = cand.work_delta_vs(base)
        horizon = cand.amortization_horizon
        htxt = (
            "—"
            if not horizon
            else ("∞" if horizon == float("inf") else f"{horizon:g}")
        )
        lines.append(
            "| {label} | {verdict} | {dwall:+g} | {dsteps:+g} | {dpeak} "
            "| {dq:+g} | {horizon} | {reason} |".format(
                label=cand.label or "?",
                verdict=v["verdict"],
                dwall=d["wall_time_s"],
                dsteps=d["total_steps"],
                dpeak=(
                    ("+" if d["peak_bytes"] >= 0 else "-")
                    + _fmt_bytes(abs(d["peak_bytes"]))
                ),
                dq=d["quality"],
                horizon=htxt,
                reason=v["reason"],
            )
        )
    return "\n".join(lines)


__all__ = [
    "AbsoluteWork",
    "promotion_table",
    "SHIP",
    "RATE_TRAP",
    "NEUTRAL",
    "HORIZON",
]
