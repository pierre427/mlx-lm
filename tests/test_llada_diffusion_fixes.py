# Copyright © 2026 Pierre Lamy
#
# Adversarial coverage for the July-9 LLaDA diffusion defects:
#   * D2 — never-stale-HIT: cross-model / empty / truncated / wrong-shape /
#     replaced-array snapshots must MISS and reproduce fresh output.
#   * D5 — geometry validation via ValueError (survives ``python -O``).
#
# Tiny 2-layer CPU model with random weights only; no checkpoint, no GPU.

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.llada import Model, ModelArgs, generate


def tiny_args():
    return ModelArgs(
        model_type="llada",
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_hidden_size=128,
        vocab_size=512,
        embedding_size=512,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        weight_tying=False,
    )


MASK_ID = 126336
GEN_KW = dict(gen_length=16, block_length=8, steps=8, temperature=0.0,
              cfg_scale=0.0, mask_id=MASK_ID, kv_cache=True)


def _fresh_model(seed):
    mx.random.seed(seed)
    m = Model(tiny_args())
    # Force distinct random parameters per instance.
    np.random.seed(seed)
    return m


def _toks(out):
    return [int(v) for v in out.reshape(-1).tolist()]


class TestLLaDASnapshotNeverStale(unittest.TestCase):
    def setUp(self):
        self.model_a = _fresh_model(0)
        self.model_b = _fresh_model(1)
        self.prompt = mx.array(np.random.RandomState(7).randint(0, 500, size=(1, 6)))
        # Capture a valid snapshot from model A.
        _, self.stats_a = generate(
            self.model_a, self.prompt, return_stats=True,
            return_prefix_snapshot=True, **GEN_KW,
        )
        self.snap = self.stats_a["prefix_snapshot"]
        self.assertIsNotNone(self.snap, "capture failed")

    def _assert_miss_and_fresh(self, model, snapshot):
        fresh, _ = generate(model, self.prompt, return_stats=True, **GEN_KW)
        injected, stats = generate(
            model, self.prompt, prefix_snapshot=snapshot,
            return_stats=True, **GEN_KW,
        )
        self.assertFalse(stats["prefix_snapshot_used"],
                         "stale/malformed snapshot was accepted as a HIT")
        self.assertEqual(_toks(injected), _toks(fresh),
                         "MISS did not reproduce fresh output")

    def test_same_snapshot_hits_on_capturing_model(self):
        # Control: the legitimate case must still be a HIT (guards over-blocking).
        _, stats = generate(
            self.model_a, self.prompt, prefix_snapshot=self.snap,
            return_stats=True, **GEN_KW,
        )
        self.assertTrue(stats["prefix_snapshot_used"])

    def test_cross_model_same_geometry_misses(self):
        # model_b has identical geometry but different weights -> MISS.
        self._assert_miss_and_fresh(self.model_b, self.snap)

    def test_empty_layers_misses(self):
        self._assert_miss_and_fresh(self.model_a, {**self.snap, "layers": []})

    def test_truncated_layers_misses(self):
        truncated = {**self.snap, "layers": list(self.snap["layers"])[:-1]}
        self._assert_miss_and_fresh(self.model_a, truncated)

    def test_wrong_shape_kv_misses(self):
        layers = list(self.snap["layers"])
        k, v = layers[0]
        layers[0] = (k[:, :, :-1, :], v)  # short prefix length on one layer
        self._assert_miss_and_fresh(self.model_a, {**self.snap, "layers": layers})

    def test_replaced_array_wrong_dtype_misses(self):
        layers = list(self.snap["layers"])
        k, v = layers[0]
        bad_k = mx.zeros(k.shape, dtype=mx.int32)  # right shape, wrong dtype
        layers[0] = (bad_k, v)
        self._assert_miss_and_fresh(self.model_a, {**self.snap, "layers": layers})

    def test_replaced_array_non_array_misses(self):
        layers = list(self.snap["layers"])
        _, v = layers[0]
        layers[0] = (None, v)
        self._assert_miss_and_fresh(self.model_a, {**self.snap, "layers": layers})


class TestLLaDAGeometryValidation(unittest.TestCase):
    """Finding D5: positive/divisible geometry via ValueError, not assert."""

    def setUp(self):
        self.model = _fresh_model(0)
        self.prompt = mx.array(np.random.RandomState(3).randint(0, 500, size=(1, 6)))

    def _gen(self, **over):
        kw = dict(gen_length=16, block_length=8, steps=8, temperature=0.0,
                  cfg_scale=0.0, mask_id=MASK_ID)
        kw.update(over)
        return generate(self.model, self.prompt, **kw)

    def test_zero_steps_raises(self):
        with self.assertRaises(ValueError):
            self._gen(steps=0)

    def test_zero_gen_length_raises(self):
        with self.assertRaises(ValueError):
            self._gen(gen_length=0)

    def test_zero_block_length_raises(self):
        with self.assertRaises(ValueError):
            self._gen(block_length=0)

    def test_indivisible_gen_length_raises(self):
        with self.assertRaises(ValueError):
            self._gen(gen_length=3, block_length=2)

    def test_indivisible_steps_raises(self):
        # gen=16/block=8 -> 2 blocks; steps=3 not divisible by 2.
        with self.assertRaises(ValueError):
            self._gen(steps=3)


if __name__ == "__main__":
    unittest.main()
