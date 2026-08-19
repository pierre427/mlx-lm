# Copyright © 2026 Pierre Lamy (mlx-uag)
# SPDX-License-Identifier: Apache-2.0
"""Tests for the absolute-work promotion schema.

CPU-only, dependency-free. No model load. Verifies:
  1. each recorded fixture reproduces its known verdict;
  2. is_rate_trap fails closed when a rate improves while total_steps or
     wall_time regresses;
  3. derived fields (total_steps, work_delta_vs) are correct;
  4. genuine wins are classified SHIP, not trapped.
"""

import unittest

from mlx_lm.absolute_work import (
    AbsoluteWork,
    promotion_table,
    SHIP,
    RATE_TRAP,
    NEUTRAL,
    HORIZON,
)
from mlx_lm.absolute_work_fixtures import FIXTURES


class TestFixtures(unittest.TestCase):
    def test_all_fixtures_reproduce_recorded_verdict(self):
        for name, (candidate, baseline, expected) in FIXTURES.items():
            with self.subTest(fixture=name):
                got = candidate.verdict(baseline)["verdict"]
                self.assertEqual(
                    got,
                    expected,
                    f"{name}: expected {expected}, got {got} "
                    f"({candidate.verdict(baseline)['reason']})",
                )

    def test_e17_is_rate_trap(self):
        cand, base, _ = FIXTURES["E17"]
        # Duty fell 100% -> 56.4% (favorable rate) but steered steps fell only
        # 9.4% and trajectory lengthened -> more total work.
        self.assertTrue(cand.is_rate_trap(base))
        self.assertGreater(cand.total_steps, base.total_steps)
        self.assertLess(cand.rates["duty"], base.rates["duty"])

    def test_vegas_lossless_but_trap(self):
        cand, base, _ = FIXTURES["Vegas"]
        self.assertTrue(cand.is_rate_trap(base))
        # Lossless -> equal quality, but wall regressed and draft steps added.
        self.assertEqual(cand.quality, base.quality)
        self.assertGreater(cand.wall_time_s, base.wall_time_s)
        self.assertGreater(cand.total_steps, base.total_steps)

    def test_kv8_neutral_not_trap(self):
        cand, base, _ = FIXTURES["KV8"]
        # Peak genuinely dropped, so it is not a pure rate trap; but decode
        # regressed, so it is not a clean ship either -> NEUTRAL.
        self.assertFalse(cand.is_rate_trap(base))
        self.assertLess(cand.peak_bytes, base.peak_bytes)
        self.assertGreater(cand.wall_time_s, base.wall_time_s)
        self.assertEqual(cand.verdict(base)["verdict"], NEUTRAL)

    def test_dcut_dominated_by_static(self):
        cand, base, _ = FIXTURES["D-Cut"]
        # Higher acceptance (favorable rate) but drafts every token at width 8,
        # so more draft steps and slower than static nd=2.
        self.assertTrue(cand.is_rate_trap(base))
        self.assertGreater(cand.rates["acceptance"], base.rates["acceptance"])
        self.assertGreater(cand.draft_steps, base.draft_steps)
        self.assertGreater(cand.wall_time_s, base.wall_time_s)

    def test_snapkv_horizon_dependent(self):
        cand, base, _ = FIXTURES["SnapKV-D"]
        self.assertEqual(cand.verdict(base)["verdict"], HORIZON)
        # Prep dominates -> whole-request wall not better...
        self.assertGreaterEqual(cand.wall_time_s, base.wall_time_s)
        # ...but the steady-state decode reads fewer bytes and peak is lower.
        self.assertLess(cand.bytes_read, base.bytes_read)
        self.assertLess(cand.peak_bytes, base.peak_bytes)
        self.assertGreater(cand.amortization_horizon, 1)


class TestFailsClosed(unittest.TestCase):
    def test_rate_up_but_total_steps_regress_is_trap(self):
        base = AbsoluteWork(
            label="base",
            wall_time_s=1.0,
            target_steps=100,
            draft_steps=0,
            peak_bytes=1000,
            rates={"acceptance": 0.5},
        )
        cand = AbsoluteWork(
            label="cand",
            wall_time_s=1.0,           # equal wall
            target_steps=100,
            draft_steps=50,           # MORE total steps
            peak_bytes=1000,           # equal peak
            rates={"acceptance": 0.9},  # rate improved
        )
        self.assertTrue(cand.is_rate_trap(base))
        self.assertEqual(cand.verdict(base)["verdict"], RATE_TRAP)

    def test_rate_up_but_wall_regress_is_trap(self):
        base = AbsoluteWork(
            label="base",
            wall_time_s=1.0,
            target_steps=100,
            peak_bytes=1000,
            rates={"duty": 1.0},
        )
        cand = AbsoluteWork(
            label="cand",
            wall_time_s=1.5,           # wall regressed
            target_steps=100,
            peak_bytes=1000,
            rates={"duty": 0.5},        # duty improved (lower)
        )
        self.assertTrue(cand.is_rate_trap(base))

    def test_rate_up_but_peak_regress_is_trap(self):
        base = AbsoluteWork(
            wall_time_s=1.0, target_steps=100, peak_bytes=1000,
            rates={"compression": 1.0},
        )
        cand = AbsoluteWork(
            wall_time_s=1.0, target_steps=100, peak_bytes=2000,  # worse peak
            rates={"compression": 4.0},
        )
        self.assertTrue(cand.is_rate_trap(base))

    def test_no_rate_improvement_is_not_trap(self):
        base = AbsoluteWork(
            wall_time_s=1.0, target_steps=100, peak_bytes=1000,
            rates={"acceptance": 0.9},
        )
        cand = AbsoluteWork(
            wall_time_s=2.0, target_steps=200, peak_bytes=2000,
            rates={"acceptance": 0.5},  # rate got WORSE -> not a rate trap
        )
        # Worse on everything but not via a seductive rate; is_rate_trap only
        # fires when a rate is favorable.
        self.assertFalse(cand.is_rate_trap(base))

    def test_genuine_win_is_ship(self):
        base = AbsoluteWork(
            wall_time_s=1.0, target_steps=100, draft_steps=0, peak_bytes=1000,
            quality=1.0, rates={"acceptance": 0.5},
        )
        cand = AbsoluteWork(
            wall_time_s=0.6,           # strictly faster
            target_steps=60, draft_steps=30,  # fewer total steps (90 < 100)
            peak_bytes=1000, quality=1.0,
            rates={"acceptance": 0.9},
        )
        self.assertFalse(cand.is_rate_trap(base))
        self.assertEqual(cand.verdict(base)["verdict"], SHIP)


class TestDerived(unittest.TestCase):
    def test_total_steps(self):
        w = AbsoluteWork(target_steps=10, draft_steps=20, steered_steps=5)
        self.assertEqual(w.total_steps, 35)

    def test_work_delta_vs(self):
        base = AbsoluteWork(wall_time_s=1.0, target_steps=100, peak_bytes=1000)
        cand = AbsoluteWork(
            wall_time_s=1.5, target_steps=150, peak_bytes=1200,
            baseline=base,
        )
        d = cand.work_delta_vs()
        self.assertAlmostEqual(d["wall_time_s"], 0.5)
        self.assertEqual(d["total_steps"], 50)
        self.assertEqual(d["peak_bytes"], 200)

    def test_missing_baseline_raises(self):
        w = AbsoluteWork(wall_time_s=1.0)
        with self.assertRaises(ValueError):
            w.verdict()

    def test_promotion_table_renders_all_fixtures(self):
        rows = [(c, b) for (c, b, _) in FIXTURES.values()]
        table = promotion_table(rows)
        for name in ("E17", "Vegas", "KV8", "D-Cut", "SnapKV"):
            self.assertIn(name, table)
        for verdict in (RATE_TRAP, NEUTRAL, HORIZON):
            self.assertIn(verdict, table)
        # header present
        self.assertIn("verdict", table)


if __name__ == "__main__":
    unittest.main()
