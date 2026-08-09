import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.exact_eager_attention import exact_eager_attention_output


class ExactEagerAttentionTests(unittest.TestCase):
    def _reference(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(2, 3)) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)

    def test_query_tiling_matches_full_eager_with_causal_mask(self):
        torch.manual_seed(0)
        dtype = torch.float32
        q = torch.randn(1, 4, 11, 8, dtype=dtype)
        k = torch.randn(1, 4, 11, 8, dtype=dtype)
        v = torch.randn(1, 4, 11, 8, dtype=dtype)
        minv = torch.finfo(dtype).min
        causal = torch.triu(torch.full((11, 11), minv, dtype=dtype), diagonal=1)
        mask = causal.view(1, 1, 11, 11)
        ref = self._reference(q, k, v, mask)
        got = exact_eager_attention_output(
            q, k, v,
            head_dim=8,
            attention_mask=mask,
            chunk_size=3,
            attention_dropout=0.0,
            training=False,
        )
        self.assertTrue(torch.allclose(ref, got, atol=1e-6, rtol=1e-6))

    def test_query_tiling_matches_full_eager_with_cached_prefix(self):
        torch.manual_seed(1)
        dtype = torch.float32
        q_len, key_len, d = 5, 12, 8
        q = torch.randn(1, 2, q_len, d, dtype=dtype)
        k = torch.randn(1, 2, key_len, d, dtype=dtype)
        v = torch.randn(1, 2, key_len, d, dtype=dtype)
        prefix = key_len - q_len
        minv = torch.finfo(dtype).min
        q_abs = torch.arange(q_len) + prefix
        kk = torch.arange(key_len)
        future = kk.view(1, -1) > q_abs.view(-1, 1)
        mask = torch.zeros((q_len, key_len), dtype=dtype)
        mask[future] = minv
        mask = mask.view(1, 1, q_len, key_len)
        ref = self._reference(q, k, v, mask)
        got = exact_eager_attention_output(
            q, k, v,
            head_dim=d,
            attention_mask=mask,
            chunk_size=2,
            attention_dropout=0.0,
            training=False,
        )
        self.assertTrue(torch.allclose(ref, got, atol=1e-6, rtol=1e-6))

    def test_implicit_causal_mask_matches_explicit_mask(self):
        torch.manual_seed(2)
        dtype = torch.float32
        q = torch.randn(1, 2, 7, 4, dtype=dtype)
        k = torch.randn(1, 2, 7, 4, dtype=dtype)
        v = torch.randn(1, 2, 7, 4, dtype=dtype)
        minv = torch.finfo(dtype).min
        mask = torch.triu(torch.full((7, 7), minv, dtype=dtype), diagonal=1).view(1, 1, 7, 7)
        explicit = exact_eager_attention_output(
            q, k, v, head_dim=4, attention_mask=mask, chunk_size=3
        )
        implicit = exact_eager_attention_output(
            q, k, v, head_dim=4, attention_mask=None, chunk_size=3
        )
        self.assertTrue(torch.equal(explicit, implicit))


if __name__ == "__main__":
    unittest.main()
