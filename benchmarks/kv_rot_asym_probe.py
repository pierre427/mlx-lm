# Copyright © 2026 Apple Inc.
"""Compose PR #1555 (Hadamard-rotated K quantization) with asymmetric K/V bits.

Teacher-forced perplexity over a real-text pack for a grid of KV configs:
fp16, symmetric affine/rotated at 8/4 bits, and the asym cross
(K4/V8, K4rot/V8, K4rot/V4, K8/V4 ...). Rotation applies to keys only, so
the interesting cells are K-low-bit ones.

    python benchmarks/kv_rot_asym_probe.py --model <path> --text-file <f> [--max-tokens N]
"""

import argparse
import math

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache


def perplexity(model, ids, cache):
    logits = model(ids[None], cache=cache)[0, :-1].astype(mx.float32)
    targets = ids[1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    return math.exp(mx.mean(nll).item())


def quant_caches(model, key_bits, value_bits, group_size, rotate):
    n = len(make_prompt_cache(model))
    return [
        QuantizedKVCache(
            group_size=group_size,
            key_bits=key_bits,
            value_bits=value_bits,
            rotate=rotate,
        )
        for _ in range(n)
    ]


CONFIGS = [
    # (label, key_bits, value_bits, rotate)
    ("K8/V8 affine", 8, 8, False),
    ("K8/V8 rotated", 8, 8, True),
    ("K4/V4 affine", 4, 4, False),
    ("K4/V4 rotated", 4, 4, True),
    ("K4/V8 affine (asym #1550)", 4, 8, False),
    ("K4/V8 rotated (compose)", 4, 8, True),
    ("K8/V4 affine (asym #1550)", 8, 4, False),
    ("K8/V4 rotated", 8, 4, True),
    ("K4/V6 rotated", 4, 6, True),
    ("K3/V8 rotated", 3, 8, True),
    # value-floor sweep: keys high, push values down (rotation is K-only,
    # affine here isolates pure V damage)
    ("K8/V3 affine", 8, 3, False),
    ("K8/V2 affine", 8, 2, False),
    ("K6/V4 affine", 6, 4, False),
    ("K6/V3 affine", 6, 3, False),
    ("K5/V4 affine", 5, 4, False),
    ("K6/V4 rotated", 6, 4, True),
    ("K5/V4 rotated", 5, 4, True),
    ("K6/V3 rotated", 6, 3, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--configs",
        default=None,
        help="comma-separated label substrings to select a subset of CONFIGS",
    )
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    text = open(args.text_file).read()
    ids = mx.array(tokenizer.encode(text)[: args.max_tokens])
    print(f"model={args.model}  tokens={len(ids)}  group_size={args.group_size}")

    base = perplexity(model, ids, make_prompt_cache(model))
    print(f"  {'fp16 KV':34s} ppl = {base:10.2f}")

    configs = CONFIGS
    if args.configs:
        wanted = [w.strip() for w in args.configs.split(",")]
        configs = [c for c in CONFIGS if any(w in c[0] for w in wanted)]

    for label, kb, vb, rot in configs:
        cache = quant_caches(model, kb, vb, args.group_size, rot)
        ppl = perplexity(model, ids, cache)
        print(f"  {label:34s} ppl = {ppl:10.2f}")


if __name__ == "__main__":
    main()
