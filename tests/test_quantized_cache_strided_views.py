# mlx 0.32.1 regression (ml-explore/mlx#4370): ``mx.dequantize`` on a view
# sliced along a non-last axis reads the wrong groups. ``QuantizedKVCache``
# hands out exactly such views (capacity-padded ring sliced to ``offset``),
# so every runtime ``mx.dequantize`` of cache contents must materialize the
# view first. ``quantized_matmul`` is unaffected. These tests run the real
# cache + attention path against a contiguous reference.
import unittest

import mlx.core as mx

from mlx_lm.models import base
from mlx_lm.models.cache import QuantizedKVCache


class TestQuantizedCacheStridedViews(unittest.TestCase):
    def _warm_cache(self, bits, group_size, n_kv_heads=4, D=128, T=300):
        mx.random.seed(0)
        cache = QuantizedKVCache(group_size=group_size, bits=bits)
        k = mx.random.normal(shape=(1, n_kv_heads, T, D))
        v = mx.random.normal(shape=(1, n_kv_heads, T, D))
        q_keys, q_values = cache.update_and_fetch(k, v)
        # Step is 256, so offset=300 < capacity=512: the views are strided.
        self.assertEqual(q_keys[0].shape[-2], T)
        self.assertLess(T, cache.keys[0].shape[-2])
        return cache, q_keys, q_values

    def test_dequantize_view_matches_contiguous(self):
        for bits, gs in ((4, 32), (4, 64), (8, 32), (8, 64)):
            _, q_keys, q_values = self._warm_cache(bits, gs)
            for t in (q_keys, q_values):
                a = mx.dequantize(*base._contiguous_quant(t), group_size=gs, bits=bits)
                b = mx.dequantize(
                    *(mx.contiguous(x) for x in t), group_size=gs, bits=bits
                )
                self.assertEqual(mx.abs(a - b).max().item(), 0.0, (bits, gs))

    def test_prefill_shaped_attention_on_warm_cache(self):
        # L >= flash threshold takes the dequantize branch of
        # quantized_scaled_dot_product_attention.
        for bits, gs in ((4, 64), (8, 64)):
            _, q_keys, q_values = self._warm_cache(bits, gs)
            L = 256
            q = mx.random.normal(shape=(1, 8, L, 128))
            out = base.quantized_scaled_dot_product_attention(
                q, q_keys, q_values, scale=1.0, mask=None, group_size=gs, bits=bits
            )
            ck = tuple(mx.contiguous(x) for x in q_keys)
            cv = tuple(mx.contiguous(x) for x in q_values)
            ref = mx.fast.scaled_dot_product_attention(
                q,
                mx.dequantize(*ck, group_size=gs, bits=bits),
                mx.dequantize(*cv, group_size=gs, bits=bits),
                scale=1.0,
            )
            self.assertLess(mx.abs(out - ref).max().item(), 1e-3, (bits, gs))


if __name__ == "__main__":
    unittest.main()
