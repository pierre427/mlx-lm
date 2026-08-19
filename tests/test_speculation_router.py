"""Pure policy tests; no model or MLX arrays are used."""

import unittest

from mlx_lm.speculation_router import RoutedSpeculationPolicy


class TestRoutedSpeculationPolicy(unittest.TestCase):
    def test_low_acceptance_latches_to_plain_then_reprobes(self):
        policy = RoutedSpeculationPolicy(
            max_draft=16,
            initial_accept_prob=0.2,
            min_proposals=2,
            bad_cycle_patience=1,
            plain_cooldown_cycles=2,
        )
        policy.observe(2, 0)
        self.assertEqual(policy.decide().num_draft, 0)
        self.assertEqual(policy.decide().num_draft, 0)
        self.assertEqual(policy.decide().num_draft, 1)

    def test_high_acceptance_avoids_verify_tax_zone(self):
        policy = RoutedSpeculationPolicy(max_draft=20, initial_accept_prob=0.98)
        decision = policy.decide()
        self.assertTrue(decision.num_draft <= 3 or decision.num_draft >= 15)

    def test_zero_remaining_uses_plain_floor(self):
        policy = RoutedSpeculationPolicy()
        self.assertEqual(policy.decide(remaining=0).num_draft, 0)


if __name__ == "__main__":
    unittest.main()
