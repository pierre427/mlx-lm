#!/usr/bin/env python
"""Summarize hybrid-proposer JSONL without pooling incomparable prompts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def summarize_rows(rows: Iterable[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("kind") != "run":
            continue
        prompt = row.get("prompt_id")
        backend = row.get("backend")
        throughput = row.get("decode_tokens_per_second")
        if not isinstance(prompt, str) or backend not in ("baseline", "hybrid"):
            raise ValueError("run rows need prompt_id and baseline/hybrid backend")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            raise ValueError("run rows need positive decode_tokens_per_second")
        grouped[(prompt, backend)].append(row)
    if not grouped:
        raise ValueError("no benchmark run rows found")

    prompt_ids = sorted({prompt for prompt, _ in grouped})
    prompts = []
    deltas = []
    for prompt in prompt_ids:
        arms = {}
        arm_hashes = {}
        for backend in ("baseline", "hybrid"):
            arm_rows = grouped.get((prompt, backend))
            if not arm_rows:
                raise ValueError(f"prompt {prompt!r} is missing the {backend} arm")
            hashes = {row.get("output_sha256") for row in arm_rows}
            if not all(isinstance(h, str) and h for h in hashes):
                raise ValueError(
                    f"prompt {prompt!r} {backend} rows are missing output_sha256; "
                    "cannot verify both arms produced identical output"
                )
            arm_hashes[backend] = hashes
            proposed = sum(
                row.get("generation_stats", {}).get("retrieval_proposed", 0)
                for row in arm_rows
            )
            accepted = sum(
                row.get("generation_stats", {}).get("retrieval_accepted", 0)
                for row in arm_rows
            )
            arms[backend] = {
                "runs": len(arm_rows),
                "median_decode_tokens_per_second": statistics.median(
                    row["decode_tokens_per_second"] for row in arm_rows
                ),
                "latched_runs": sum(
                    bool(row.get("generation_stats", {}).get("latched"))
                    for row in arm_rows
                ),
                "output_hashes": len(hashes),
                "retrieval_proposed": proposed,
                "retrieval_accepted": accepted,
                "retrieval_acceptance": accepted / proposed if proposed else 0.0,
            }
        # Cross-arm equality gate: a throughput "win" is only meaningful when
        # both arms produced the same output. Unequal hashes mean the arms did
        # different work, so the summary must fail, not report a win.
        if arm_hashes["baseline"] != arm_hashes["hybrid"]:
            raise ValueError(
                f"prompt {prompt!r}: baseline and hybrid output hashes differ "
                f"(baseline={sorted(arm_hashes['baseline'])}, "
                f"hybrid={sorted(arm_hashes['hybrid'])}); arms did not do "
                "equal work, refusing to summarize"
            )
        baseline = arms["baseline"]["median_decode_tokens_per_second"]
        hybrid = arms["hybrid"]["median_decode_tokens_per_second"]
        delta = (hybrid / baseline - 1.0) * 100.0
        deltas.append(delta)
        prompts.append(
            {
                "prompt_id": prompt,
                "delta_percent": delta,
                "baseline": arms["baseline"],
                "hybrid": arms["hybrid"],
            }
        )

    return {
        "kind": "summary",
        "prompt_count": len(prompts),
        "win_count": sum(delta > 0 for delta in deltas),
        "win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        # Aggregate comparable per-prompt deltas, never pooled arm throughputs.
        "median_delta_percent": statistics.median(deltas),
        "mean_delta_percent": statistics.fmean(deltas),
        "prompts": prompts,
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    summary = summarize_rows(read_jsonl(args.result))
    print(json.dumps(summary, indent=None if args.compact else 2, sort_keys=True))


if __name__ == "__main__":
    main()
