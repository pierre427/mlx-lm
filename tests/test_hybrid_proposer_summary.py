import importlib.util
import os
import unittest


_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "summarize_hybrid_proposer.py",
)
_SPEC = importlib.util.spec_from_file_location("summarize_hybrid_proposer", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_rows = _MODULE.summarize_rows


def _row(prompt, backend, throughput, *, accepted=0, proposed=0, latched=False, sha=None):
    # Both arms of one prompt share an output hash by default: a summary is
    # only meaningful when baseline and hybrid produced identical output.
    return {
        "kind": "run",
        "prompt_id": prompt,
        "backend": backend,
        "decode_tokens_per_second": throughput,
        "output_sha256": f"{prompt}-output" if sha is None else sha,
        "generation_stats": {
            "retrieval_accepted": accepted,
            "retrieval_proposed": proposed,
            "latched": latched,
        },
    }


class TestHybridProposerSummary(unittest.TestCase):
    def test_aggregates_per_prompt_deltas_not_pooled_arm_medians(self):
        rows = [
            _row("a", "baseline", 100),
            _row("a", "baseline", 110),
            _row("a", "hybrid", 200, accepted=8, proposed=10),
            _row("a", "hybrid", 220, accepted=8, proposed=10),
            _row("b", "baseline", 200),
            _row("b", "baseline", 200),
            _row("b", "hybrid", 100, latched=True),
            _row("b", "hybrid", 100, latched=True),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["win_count"], 1)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["median_delta_percent"], 25.0)
        self.assertAlmostEqual(summary["mean_delta_percent"], 25.0)
        self.assertEqual(
            summary["prompts"][0]["hybrid"]["retrieval_acceptance"], 0.8
        )
        self.assertEqual(summary["prompts"][1]["hybrid"]["latched_runs"], 2)

    def test_ignores_metadata_and_requires_both_arms(self):
        with self.assertRaisesRegex(ValueError, "missing the hybrid arm"):
            summarize_rows(
                [
                    {"kind": "metadata"},
                    _row("a", "baseline", 100),
                ]
            )

    def test_rejects_invalid_throughput(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_rows(
                [
                    _row("a", "baseline", 0),
                    _row("a", "hybrid", 1),
                ]
            )

    def test_cross_arm_hash_mismatch_fails_instead_of_reporting_a_win(self):
        # A faster hybrid arm with DIFFERENT output than baseline must never
        # summarize as a win — unequal hashes mean unequal work.
        rows = [
            _row("a", "baseline", 100, sha="AAA"),
            _row("a", "hybrid", 200, sha="BBB"),
        ]
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            summarize_rows(rows)

    def test_cross_arm_hash_mismatch_fails_even_when_each_arm_is_stable(self):
        # Per-arm hash cardinality of 1 is not enough: the arms must agree
        # with each other, not merely with themselves.
        rows = [
            _row("a", "baseline", 100, sha="AAA"),
            _row("a", "baseline", 101, sha="AAA"),
            _row("a", "hybrid", 200, sha="BBB"),
            _row("a", "hybrid", 201, sha="BBB"),
        ]
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            summarize_rows(rows)

    def test_missing_output_hash_fails(self):
        no_hash = _row("a", "hybrid", 200)
        del no_hash["output_sha256"]
        rows = [_row("a", "baseline", 100), no_hash]
        with self.assertRaisesRegex(ValueError, "missing output_sha256"):
            summarize_rows(rows)

    def test_equal_cross_arm_hashes_still_summarize(self):
        rows = [
            _row("a", "baseline", 100),
            _row("a", "hybrid", 200, accepted=8, proposed=10),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["win_count"], 1)


if __name__ == "__main__":
    unittest.main()
