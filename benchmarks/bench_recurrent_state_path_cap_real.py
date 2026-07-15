# Copyright © 2026 Apple Inc.

"""Checkpoint-scale Qwen3-Next recurrent path-cap promotion battery.

Each worker loads the model in a fresh process so MLX and RSS peaks are not
polluted by a previous cap cell. The suite interleaves default-off baselines
with a mirrored cap sweep. Every worker proves exact topology, cache state,
continuation logits, and generated tokens before reporting performance.
"""

import argparse
import copy
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm import load
from mlx_lm.models.cache import LRUPromptCache

DEFAULT_MODEL = Path(
    "/Users/pierrelamy/.cache/huggingface/hub/"
    "models--lmstudio-community--Qwen3-Next-80B-A3B-Instruct-MLX-4bit/"
    "snapshots/6d14222d4664dba1aa9f3251696c8254c6e994af"
)


def rss_bytes():
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) * 1024


def peak_rss_bytes():
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def checkpoint(model, tokens):
    state = model.make_cache()
    logits = model(mx.array([tokens]), cache=state)
    mx.eval(logits, [cache.state for cache in state])
    return state


def topology(cache):
    return [
        {
            "class": type(layer).__name__,
            "meta_state": layer.meta_state,
        }
        for layer in cache
    ]


def assert_cache_equal(expected, actual, stage):
    if topology(expected) != topology(actual):
        raise RuntimeError(f"cache topology diverged at {stage}")
    for layer_index, (expected_layer, actual_layer) in enumerate(zip(expected, actual)):
        expected_state = tree_flatten(expected_layer.state)
        actual_state = tree_flatten(actual_layer.state)
        if len(expected_state) != len(actual_state):
            raise RuntimeError(
                f"cache state arity diverged at {stage}, layer {layer_index}"
            )
        for state_index, ((_, left), (_, right)) in enumerate(
            zip(expected_state, actual_state)
        ):
            if not bool(mx.array_equal(left, right)):
                raise RuntimeError(
                    f"cache tensor diverged at {stage}, layer {layer_index}, "
                    f"state {state_index}"
                )


def continue_tokens(model, state, first_token, count):
    tokens = []
    logits = None
    token = first_token
    for _ in range(count):
        logits = model(mx.array([[token]]), cache=state)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
    return tokens, logits


def run_worker(model_path, cap, checkpoints, stride, repeats, decode_tokens):
    if cap is not None and checkpoints <= cap:
        raise ValueError("checkpoints must exceed cap")

    mx.random.seed(41)
    load_start = time.perf_counter()
    model, tokenizer = load(str(model_path))
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_start
    gc.collect()
    mx.clear_cache()

    weights_active = mx.get_active_memory()
    weights_rss = rss_bytes()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    text = (
        "A recurrent language model can retain an exact prefix checkpoint while "
        "moving selected state tensors between memory domains. The cache owner "
        "must preserve topology, codec, full attention keys and values, and the "
        "continuation semantics. "
    )
    token_stream = tokenizer.encode(text * 80)
    needed = checkpoints * stride + 1
    if len(token_stream) < needed:
        raise RuntimeError(f"tokenizer produced only {len(token_stream)} tokens")
    token_stream = token_stream[:needed]

    model_key = (str(model_path), None, None)
    store = LRUPromptCache(
        max_size=checkpoints + 1,
        recurrent_state_path_cap=cap,
    )
    build_start = time.perf_counter()
    for index in range(1, checkpoints + 1):
        prefix = token_stream[: index * stride]
        state = checkpoint(model, prefix)
        store.insert_cache(model_key, prefix, state)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    build_seconds = time.perf_counter() - build_start

    store_active = mx.get_active_memory()
    store_rss = rss_bytes()
    stats_before = store.recurrent_state_stats()
    target = token_stream[:stride]
    first_token = token_stream[stride]

    # Construct the correctness reference only after taking store memory
    # measurements. Retaining the first checkpoint here would keep its MLX
    # recurrent arrays alive and mask one host offload in every capped cell.
    reference = checkpoint(model, target)
    restored, remaining = store.fetch_nearest_cache(model_key, target)
    if remaining:
        raise RuntimeError("target prefix was not an exact cache hit")
    assert_cache_equal(reference, restored, "restore")

    expected_state = copy.deepcopy(reference)
    actual_state = restored
    expected_tokens, expected_logits = continue_tokens(
        model, expected_state, first_token, decode_tokens
    )
    actual_tokens, actual_logits = continue_tokens(
        model, actual_state, first_token, decode_tokens
    )
    if expected_tokens != actual_tokens or not bool(
        mx.array_equal(expected_logits, actual_logits)
    ):
        raise RuntimeError("restored continuation logits or tokens diverged")
    assert_cache_equal(expected_state, actual_state, "continuation")
    del expected_state, actual_state, restored
    gc.collect()
    mx.clear_cache()

    ttft_ms = []
    wall_ms = []
    outputs = []
    restore_start = store.recurrent_state_stats()["restore_ms_total"]
    for _ in range(repeats):
        start = time.perf_counter()
        state, remaining = store.fetch_nearest_cache(model_key, target)
        if remaining:
            raise RuntimeError("timed target prefix was not an exact hit")
        first_logits = model(mx.array([[first_token]]), cache=state)
        mx.eval(first_logits)
        first_done = time.perf_counter()
        token = int(mx.argmax(first_logits[0, -1]).item())
        generated = [token]
        for _ in range(decode_tokens - 1):
            logits = model(mx.array([[token]]), cache=state)
            mx.eval(logits)
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
        done = time.perf_counter()
        if generated != expected_tokens:
            raise RuntimeError("timed continuation tokens diverged")
        ttft_ms.append(1000 * (first_done - start))
        wall_ms.append(1000 * (done - start))
        outputs.append(generated)
        del state, first_logits
        gc.collect()
        mx.clear_cache()

    stats_after = store.recurrent_state_stats()
    return {
        "model": str(model_path),
        "model_type": type(model).__module__,
        "layers": len(model.make_cache()),
        "cache_topology": {
            name: sum(item["class"] == name for item in topology(reference))
            for name in sorted({item["class"] for item in topology(reference)})
        },
        "path_cap": cap,
        "checkpoint_count": checkpoints,
        "checkpoint_stride_tokens": stride,
        "target_prefix_tokens": len(target),
        "repeats": repeats,
        "decode_tokens_per_repeat": decode_tokens,
        "load_seconds": load_seconds,
        "checkpoint_build_seconds": build_seconds,
        "continuation_exact": True,
        "topology_exact": True,
        "output_tokens": expected_tokens,
        "weights_active_bytes": weights_active,
        "store_active_bytes": store_active,
        "store_active_delta_bytes": store_active - weights_active,
        "weights_rss_bytes": weights_rss,
        "store_rss_bytes": store_rss,
        "store_rss_delta_bytes": store_rss - weights_rss,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "ttft_ms_p50": statistics.median(ttft_ms),
        "ttft_ms_p95": statistics.quantiles(ttft_ms, n=20)[18],
        "end_to_end_ms_p50": statistics.median(wall_ms),
        "end_to_end_tokens_per_second": 1000
        * decode_tokens
        / statistics.median(wall_ms),
        "exact_cache_hits": repeats + 1,
        "timed_restore_ms_total": stats_after["restore_ms_total"] - restore_start,
        "recurrent_stats_before": stats_before,
        "recurrent_stats_after": stats_after,
    }


def run_suite(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells_dir = output.with_suffix("")
    cells_dir.mkdir(parents=True, exist_ok=True)
    order = [None, 1, None, 2, None, 3, None, 3, None, 2, None, 1, None]
    cells = []
    for index, cap in enumerate(order, 1):
        label = "off" if cap is None else str(cap)
        cell_path = cells_dir / f"cell-{index:02d}-cap-{label}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            str(args.model),
            "--cap",
            label,
            "--checkpoints",
            str(args.checkpoints),
            "--stride",
            str(args.stride),
            "--repeats",
            str(args.repeats),
            "--decode-tokens",
            str(args.decode_tokens),
            "--output",
            str(cell_path),
        ]
        print(f"[{index}/{len(order)}] cap={label}", flush=True)
        subprocess.run(command, check=True)
        with cell_path.open() as handle:
            cells.append(json.load(handle))

    baselines = [cell for cell in cells if cell["path_cap"] is None]
    by_cap = {
        str(cap): [cell for cell in cells if cell["path_cap"] == cap]
        for cap in (1, 2, 3)
    }
    baseline_active = statistics.median(
        cell["store_active_delta_bytes"] for cell in baselines
    )
    baseline_ttft = statistics.median(cell["ttft_ms_p50"] for cell in baselines)
    baseline_tps = statistics.median(
        cell["end_to_end_tokens_per_second"] for cell in baselines
    )
    summary = {}
    for cap, cap_cells in by_cap.items():
        active = statistics.median(
            cell["store_active_delta_bytes"] for cell in cap_cells
        )
        ttft = statistics.median(cell["ttft_ms_p50"] for cell in cap_cells)
        tps = statistics.median(
            cell["end_to_end_tokens_per_second"] for cell in cap_cells
        )
        summary[cap] = {
            "trials": len(cap_cells),
            "continuation_exact": all(
                cell["continuation_exact"] and cell["topology_exact"]
                for cell in cap_cells
            ),
            "store_active_delta_bytes_median": active,
            "device_cache_reduction_percent": 100
            * (baseline_active - active)
            / baseline_active,
            "ttft_ms_p50_median": ttft,
            "ttft_change_percent": 100 * (ttft - baseline_ttft) / baseline_ttft,
            "tokens_per_second_median": tps,
            "tokens_per_second_change_percent": 100
            * (tps - baseline_tps)
            / baseline_tps,
            "host_recurrent_bytes": cap_cells[0]["recurrent_stats_before"][
                "host_recurrent_bytes"
            ],
            "soft_overflow_paths_max": max(
                cell["recurrent_stats_after"]["soft_overflow_paths"]
                for cell in cap_cells
            ),
        }
    result = {
        "scope": "real checkpoint-scale Qwen3-Next hybrid path-cap promotion battery",
        "order": ["off" if cap is None else cap for cap in order],
        "baseline_trials": len(baselines),
        "baseline": {
            "store_active_delta_bytes_median": baseline_active,
            "ttft_ms_p50_median": baseline_ttft,
            "tokens_per_second_median": baseline_tps,
        },
        "caps": summary,
        "cells": cells,
    }
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"baseline": result["baseline"], "caps": summary}, indent=2))


def parse_cap(value):
    return None if value == "off" else int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cap", type=parse_cap, default=None)
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args()
    if parsed.worker:
        worker_result = run_worker(
            parsed.model,
            parsed.cap,
            parsed.checkpoints,
            parsed.stride,
            parsed.repeats,
            parsed.decode_tokens,
        )
        with Path(parsed.output).open("w") as handle:
            json.dump(worker_result, handle, indent=2)
            handle.write("\n")
    else:
        run_suite(parsed)
