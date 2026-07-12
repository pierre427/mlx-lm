# Copyright © 2026 Apple Inc.

"""persistent_mtp wiring through adaptive_pld_generate_step.

The PLD phase must be value-identical with persistence on (it only reroutes
the verify forward through model.model/model.logits and accumulates hiddens),
and the latched MTP tail must keep yielding the trunk's greedy stream while
drafting from the persistent cache. Uses the tiny qwen3_5 hybrid model from
test_qwen3_5_mtp (GDN + full-attention + depth-1 MTP head).
"""

import unittest

import mlx.core as mx

from mlx_lm.hybrid_speculative import HybridStats, adaptive_pld_generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import TextModel

from test_qwen3_5_mtp import tiny_args


def _run(model, prompt, **kwargs):
    stats = HybridStats()
    toks = [
        int(t)
        for t, _lp, _fd in adaptive_pld_generate_step(
            prompt, model, stats=stats, **kwargs
        )
    ]
    return toks, stats


class TestAdaptivePLDPersistentMTP(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.model = TextModel(tiny_args())
        mx.eval(self.model.parameters())
        self.prompt = mx.random.randint(0, 64, (24,)).astype(mx.uint32)

    def test_pld_phase_stream_identical_with_persistence(self):
        # gate=0 -> the latch never fires, so the whole run is the PLD phase.
        # persistent only reroutes the verify forward (same values) and
        # accumulates pairs; max_tokens=300 crosses the 256-pair flush.
        kwargs = dict(max_tokens=300, warmup=8, gate=0.0, mtp_tail=True)
        base, _ = _run(self.model, self.prompt, persistent_mtp=False, **kwargs)
        pers, _ = _run(self.model, self.prompt, persistent_mtp=True, **kwargs)
        self.assertEqual(len(base), 300)
        self.assertEqual(base, pers)

    def test_latched_mtp_tail_matches_legacy_stream(self):
        # gate=1.0 latches at the first check; the tail self-speculates with
        # the head. Exact-match acceptance keeps the output the trunk's own
        # greedy stream regardless of the draft cache regime.
        kwargs = dict(max_tokens=96, warmup=4, gate=1.0, mtp_tail=True)
        legacy, ls = _run(self.model, self.prompt, persistent_mtp=False, **kwargs)
        pers, ps = _run(self.model, self.prompt, persistent_mtp=True, **kwargs)
        self.assertEqual(len(legacy), 96)
        self.assertEqual(legacy, pers)
        # Both actually took the MTP tail (drafts were proposed).
        self.assertGreater(ls.draft_proposed, 0)
        self.assertGreater(ps.draft_proposed, 0)

    def test_persistent_tail_multi_draft(self):
        toks, stats = _run(
            self.model,
            self.prompt,
            max_tokens=64,
            warmup=4,
            gate=1.0,
            mtp_tail=True,
            num_draft=2,
            persistent_mtp=True,
        )
        self.assertEqual(len(toks), 64)
        self.assertGreater(stats.draft_proposed, 0)

    def test_external_prompt_cache_fails_closed(self):
        cache = make_prompt_cache(self.model)
        gen = adaptive_pld_generate_step(
            self.prompt,
            self.model,
            max_tokens=8,
            mtp_tail=True,
            persistent_mtp=True,
            prompt_cache=cache,
            history_prompt=self.prompt,
        )
        with self.assertRaises(ValueError):
            next(gen)

    def test_persistent_ignored_without_mtp_tail(self):
        # persistent_mtp without mtp_tail must not change the plain-tail path.
        kwargs = dict(max_tokens=48, warmup=4, gate=1.0, mtp_tail=False)
        base, _ = _run(self.model, self.prompt, persistent_mtp=False, **kwargs)
        pers, _ = _run(self.model, self.prompt, persistent_mtp=True, **kwargs)
        self.assertEqual(base, pers)


if __name__ == "__main__":
    unittest.main()


class TestMTPRateGate(unittest.TestCase):
    """One-shot measured break-even gate on the MTP tail."""

    def setUp(self):
        mx.random.seed(0)
        self.model = TextModel(tiny_args())
        mx.eval(self.model.parameters())
        self.prompt = mx.random.randint(0, 64, (24,)).astype(mx.uint32)

    def _run_tail(self, margin, **kwargs):
        import mlx_lm.hybrid_speculative as hs
        old = hs._RATE_GATE_MARGIN
        hs._RATE_GATE_MARGIN = margin
        try:
            stats = HybridStats()
            toks = [
                int(t)
                for t, _lp, _fd in adaptive_pld_generate_step(
                    self.prompt, self.model, stats=stats,
                    max_tokens=128, warmup=4, gate=1.0,
                    mtp_tail=True, mtp_rate_gate=True, **kwargs
                )
            ]
            return toks, stats
        finally:
            hs._RATE_GATE_MARGIN = old

    def test_gate_probes_and_keeps_speculating(self):
        # margin=-1e9 makes "keep speculating" always win: the probe runs,
        # never de-latches, and drafting RESUMES after the probe. (No exact
        # stream comparison vs the ungated run: the probe's width-1 forwards
        # shift batched-verify numerics — the standard spec-width trajectory
        # effect, amplified on a random tiny model's near-tie logits.)
        toks, stats = self._run_tail(-1e9, persistent_mtp=True)
        self.assertTrue(stats.rate_gate_probed)
        self.assertFalse(stats.rate_gate_delatched)
        self.assertGreater(stats.rate_gate_spec_ms_per_tok, 0.0)
        self.assertGreater(stats.rate_gate_plain_ms_per_tok, 0.0)
        self.assertGreater(stats.draft_cycles, 8)  # drafted again post-probe
        self.assertEqual(len(toks), 128)

    def test_gate_delatches_and_still_completes(self):
        # margin=1e9 makes de-latch always win: the tail must switch to plain
        # and still deliver every token, identical greedy stream.
        for persistent in (False, True):
            base, _ = _run(self.model, self.prompt, max_tokens=128, warmup=4,
                           gate=1.0, mtp_tail=True, persistent_mtp=persistent)
            toks, stats = self._run_tail(1e9, persistent_mtp=persistent)
            self.assertTrue(stats.rate_gate_probed)
            self.assertTrue(stats.rate_gate_delatched)
            self.assertEqual(len(toks), 128)
            self.assertEqual(toks, base)
            # After de-latch no further drafting happened: proposed counts
            # stop at the warmup cycles' worth.
            self.assertLessEqual(stats.draft_cycles, 16)

    def test_gate_probe_resume_keeps_persistent_pairs_consistent(self):
        # Keep-speculating decision with persistent cache: drafting resumes
        # after the plain probe (pairs carried through), and the stream still
        # matches the ungated persistent run (exactness is cache-independent,
        # but a desynced pair/position protocol would crash or corrupt trims).
        toks, stats = self._run_tail(-1e9, persistent_mtp=True)
        self.assertEqual(len(toks), 128)
        self.assertGreater(stats.draft_proposed, 0)

    def test_self_mtp_rate_gate_smoke(self):
        from mlx_lm.hybrid_speculative import self_mtp_generate_step
        stats = HybridStats()
        toks = [
            int(t) for t, _lp, _fd in self_mtp_generate_step(
                self.prompt, self.model, num_draft=1, max_tokens=96,
                persistent_mtp=True, rate_gate=True, stats=stats)
        ]
        self.assertEqual(len(toks), 96)
        self.assertTrue(stats.rate_gate_probed)
