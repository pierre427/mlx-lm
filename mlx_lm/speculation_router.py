# Copyright © 2026 Apple Inc.

"""Pure-Python request-local router for measured speculative decoding.

The router composes the static verify-cost recommendation with marginal
pre-verification truncation, then adds an abstaining hysteretic controller.
It never touches model state: ``num_draft=0`` means take the plain decode floor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .verify_cost_policy import (
    VerifyCostModel,
    recommend_num_draft,
    truncate_draft_by_benefit,
)


@dataclass(frozen=True)
class SpeculationDecision:
    num_draft: int
    reason: str
    accept_prob: float
    latched_plain: bool
    cooldown_remaining: int


class RoutedSpeculationPolicy:
    """Hysteretic measured router with a periodic one-token re-probe."""

    def __init__(
        self,
        *,
        max_draft: int = 16,
        initial_accept_prob: float = 0.5,
        ewma_alpha: float = 0.25,
        min_proposals: int = 8,
        disable_below: float = 0.35,
        bad_cycle_patience: int = 2,
        plain_cooldown_cycles: int = 8,
        token_value_us: Optional[float] = None,
    ):
        if max_draft < 1:
            raise ValueError("max_draft must be positive")
        if not 0.0 <= initial_accept_prob <= 1.0:
            raise ValueError("initial_accept_prob must be in [0, 1]")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if min_proposals < 1 or bad_cycle_patience < 1:
            raise ValueError("sample and patience counts must be positive")
        if plain_cooldown_cycles < 1:
            raise ValueError("plain_cooldown_cycles must be positive")
        self.max_draft = int(max_draft)
        self.accept_prob = float(initial_accept_prob)
        self.ewma_alpha = float(ewma_alpha)
        self.min_proposals = int(min_proposals)
        self.disable_below = float(disable_below)
        self.bad_cycle_patience = int(bad_cycle_patience)
        self.plain_cooldown_cycles = int(plain_cooldown_cycles)
        self.token_value_us = token_value_us
        self.verify_cost_model = VerifyCostModel.from_measured()
        self.total_proposed = 0
        self.total_accepted = 0
        self.bad_cycles = 0
        self.latched_plain = False
        self.cooldown_remaining = 0
        self.decisions = 0
        self.plain_decisions = 0
        self.reengagements = 0
        self.last_decision = SpeculationDecision(
            0, "not_started", self.accept_prob, False, 0
        )

    def _decision(self, num_draft: int, reason: str) -> SpeculationDecision:
        self.decisions += 1
        if num_draft == 0:
            self.plain_decisions += 1
        self.last_decision = SpeculationDecision(
            num_draft=num_draft,
            reason=reason,
            accept_prob=self.accept_prob,
            latched_plain=self.latched_plain,
            cooldown_remaining=self.cooldown_remaining,
        )
        return self.last_decision

    def decide(
        self,
        *,
        max_draft: Optional[int] = None,
        remaining: Optional[int] = None,
    ) -> SpeculationDecision:
        requested_cap = self.max_draft if max_draft is None else int(max_draft)
        cap = min(self.max_draft, requested_cap)
        if remaining is not None:
            cap = min(cap, max(0, int(remaining)))
        if cap <= 0:
            return self._decision(0, "no_remaining_budget")
        if self.latched_plain:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                return self._decision(0, "plain_cooldown")
            self.latched_plain = False
            self.bad_cycles = 0
            self.reengagements += 1
            return self._decision(1, "periodic_reprobe")

        recommended = recommend_num_draft(
            {
                "accept_prob": self.accept_prob,
                "max_draft": cap,
                **(
                    {"token_value_us": self.token_value_us}
                    if self.token_value_us is not None
                    else {}
                ),
            }
        )
        trimmed = truncate_draft_by_benefit(
            [self.accept_prob] * recommended,
            self.verify_cost_model,
            accept_prob_fn=lambda probability: probability,
            token_value_us=self.token_value_us,
        )
        selected = min(cap, int(trimmed))
        if selected <= 0:
            self.latched_plain = True
            self.cooldown_remaining = self.plain_cooldown_cycles
            return self._decision(0, "marginal_verify_cost")
        return self._decision(selected, "measured_verify_cost")

    def observe(self, proposed: int, accepted: int) -> None:
        proposed, accepted = int(proposed), int(accepted)
        if proposed <= 0 or not 0 <= accepted <= proposed:
            raise ValueError("require proposed > 0 and 0 <= accepted <= proposed")
        observed = accepted / proposed
        self.accept_prob = (
            self.ewma_alpha * observed
            + (1.0 - self.ewma_alpha) * self.accept_prob
        )
        self.total_proposed += proposed
        self.total_accepted += accepted
        if (
            self.total_proposed >= self.min_proposals
            and self.accept_prob < self.disable_below
        ):
            self.bad_cycles += 1
        else:
            self.bad_cycles = 0
        if self.bad_cycles >= self.bad_cycle_patience:
            self.latched_plain = True
            self.cooldown_remaining = self.plain_cooldown_cycles

    def snapshot(self) -> Dict[str, Any]:
        return {
            "accept_prob": round(self.accept_prob, 6),
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "decisions": self.decisions,
            "plain_decisions": self.plain_decisions,
            "reengagements": self.reengagements,
            "last_decision": asdict(self.last_decision),
        }
