from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.centroidkv_common import (  # noqa: E402
    CentroidKVConfig,
    compress_layer,
    initial_or_existing_counts,
)
from models.llama.merge_centroidkv_llama import Repair as LlamaRepair  # noqa: E402
from models.qwen.merge_centroidkv_qwen import Repair as QwenRepair  # noqa: E402


class DummyAttention:
    pass


class DummyLayer:
    def __init__(self):
        self.self_attn = DummyAttention()


class DummyModel:
    def __init__(self, layers=1):
        self.model = SimpleNamespace(layers=[DummyLayer() for _ in range(layers)])
        self._centroidkv_counts = None


class CentroidKVCoreTests(unittest.TestCase):
    def test_protected_sink_local_and_target_length(self):
        # sink=0 and local=7 are far from the middle and must stay bit-identical.
        keys = torch.tensor([0.0, 1.0, 1.1, 3.0, 3.1, 5.0, 5.1, 9.0]).view(1, 1, 8, 1)
        values = (keys * 10).clone()
        counts = torch.ones((1, 1, 8), dtype=torch.int32)
        cfg = CentroidKVConfig(
            sink_budget=1, local_budget=1, chunk_size=6,
            merge_ratio=1.0, decrease_rate=0.0,
        )
        knew, vnew, cnew = compress_layer(keys, values, counts, 5, cfg)
        self.assertEqual(knew.shape[-2], 5)
        torch.testing.assert_close(knew[..., 0, :], keys[..., 0, :])
        torch.testing.assert_close(knew[..., -1, :], keys[..., -1, :])
        torch.testing.assert_close(vnew[..., 0, :], values[..., 0, :])
        torch.testing.assert_close(vnew[..., -1, :], values[..., -1, :])
        self.assertEqual(int(cnew.sum()), 8)

    def test_protected_budgets_are_a_structural_length_floor(self):
        torch.manual_seed(3)
        keys = torch.randn((1, 1, 32, 4))
        values = torch.randn_like(keys)
        counts = torch.ones((1, 1, 32), dtype=torch.int32)
        cfg = CentroidKVConfig(
            sink_budget=8, local_budget=8, chunk_size=16,
            merge_ratio=1.0, decrease_rate=0.0,
        )
        knew, _, _ = compress_layer(keys, values, counts, 4, cfg)
        self.assertGreaterEqual(knew.shape[-2], 16)
        self.assertLess(knew.shape[-2], 32)

    def test_count_weighted_centroid(self):
        # One middle chunk: source nodes 0/2 and destination nodes 1/3.
        # Pair (1, 1.1) is selected; its destination centroid is weighted 1:3.
        keys = torch.tensor([0.0, 1.0, 1.1, 4.0, 4.1, 9.0]).view(1, 1, 6, 1)
        values = torch.tensor([0.0, 10.0, 30.0, 40.0, 50.0, 90.0]).view(1, 1, 6, 1)
        counts = torch.tensor([1, 1, 3, 1, 1, 1], dtype=torch.int32).view(1, 1, 6)
        cfg = CentroidKVConfig(
            sink_budget=1, local_budget=1, chunk_size=4,
            merge_ratio=0.5, decrease_rate=0.0,
        )
        knew, vnew, cnew = compress_layer(keys, values, counts, 5, cfg)
        self.assertEqual(knew.shape[-2], 5)
        self.assertEqual(int(cnew.sum()), 8)
        # Find the merged count=4 slot and verify arithmetic centroids.
        slot = int((cnew[0, 0] == 4).nonzero()[0])
        self.assertAlmostEqual(float(knew[0, 0, slot, 0]), (1.0 + 3 * 1.1) / 4, places=6)
        self.assertAlmostEqual(float(vnew[0, 0, slot, 0]), (10.0 + 3 * 30.0) / 4, places=6)

    def test_cross_round_count_alignment(self):
        model = DummyModel()
        keys = torch.zeros((1, 1, 3, 2))
        first = initial_or_existing_counts(model, 0, keys)
        self.assertEqual(first.tolist(), [[[1, 1, 1]]])
        model._centroidkv_counts = [torch.tensor([[[2, 3]]], dtype=torch.int32)]
        extended = initial_or_existing_counts(model, 0, keys)
        self.assertEqual(extended.tolist(), [[[2, 3, 1]]])


    def test_matches_uploaded_official_press_on_reference_case(self):
        # Load the untouched official file with a minimal kvpress BasePress stub.
        kvpress_mod = types.ModuleType("kvpress")
        presses_mod = types.ModuleType("kvpress.presses")
        base_mod = types.ModuleType("kvpress.presses.base_press")

        class BasePress:
            pass

        base_mod.BasePress = BasePress
        previous = {name: sys.modules.get(name) for name in (
            "kvpress", "kvpress.presses", "kvpress.presses.base_press"
        )}
        sys.modules["kvpress"] = kvpress_mod
        sys.modules["kvpress.presses"] = presses_mod
        sys.modules["kvpress.presses.base_press"] = base_mod
        try:
            official_path = ROOT / "third_party" / "CentroidKV" / "centroidkv_press.py"
            spec = importlib.util.spec_from_file_location(
                "_centroidkv_official_reference", official_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            torch.manual_seed(17)
            keys = torch.randn((1, 2, 128, 8), dtype=torch.float32)
            values = torch.randn_like(keys)
            kwargs = dict(
                compression_ratio=0.4, sink_budget=16, local_budget=64,
                chunk_size=48, merge_ratio=0.8, decrease_rate=0.2,
            )
            official = module.CentroidKVPress(**kwargs)
            k_ref, v_ref = official.compress(None, None, keys, values, None, {})
            cfg = CentroidKVConfig(
                sink_budget=kwargs["sink_budget"],
                local_budget=kwargs["local_budget"],
                chunk_size=kwargs["chunk_size"],
                merge_ratio=kwargs["merge_ratio"],
                decrease_rate=kwargs["decrease_rate"],
            )
            counts = torch.ones(keys.shape[:-1], dtype=torch.int32)
            k_new, v_new, _ = compress_layer(
                keys, values, counts, int(keys.shape[-2] * 0.6), cfg
            )
            torch.testing.assert_close(k_new, k_ref, rtol=0.0, atol=0.0)
            torch.testing.assert_close(v_new, v_ref, rtol=0.0, atol=0.0)
        finally:
            sys.modules.pop("_centroidkv_official_reference", None)
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def test_llama_qwen_adapters_match(self):
        torch.manual_seed(5)
        keys = torch.randn((1, 2, 16, 4))
        values = torch.randn_like(keys)
        cache = ((keys, values),)
        kwargs = dict(
            ratio=0.25, target_length=12,
            sink_budget=2, local_budget=2, chunk_size=12,
            merge_ratio=0.8, decrease_rate=0.2,
        )
        lmodel, qmodel = DummyModel(), DummyModel()
        lout = LlamaRepair(lmodel, cache, None, None, **kwargs)
        qout = QwenRepair(qmodel, cache, None, None, **kwargs)
        torch.testing.assert_close(lout[0][0], qout[0][0])
        torch.testing.assert_close(lout[0][1], qout[0][1])
        torch.testing.assert_close(lmodel._centroidkv_counts[0], qmodel._centroidkv_counts[0])


if __name__ == "__main__":
    unittest.main()
