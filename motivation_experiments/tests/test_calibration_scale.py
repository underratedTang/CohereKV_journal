from pathlib import Path
import json
import math
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from calibrate_layer_head_scales import global_scale_from_key_layers, layer_head_scales_from_key_layers  # noqa: E402
from calibration_scale import load_calibration_scales, scale_for, batched_head_scales  # noqa: E402


def _row_s2(x: torch.Tensor) -> torch.Tensor:
    mean = x.float().mean(dim=1, keepdim=True)
    m2 = (x.float() - mean).square().sum(dim=(1, 2))
    return 2.0 * m2 / float(x.shape[1] - 1)


def test_calibration_preserves_per_layer_head_scales_and_summary_rms():
    layer0 = torch.tensor([[[[0.0], [2.0], [4.0]], [[10.0], [10.0], [16.0]]]])
    layer1 = torch.tensor([[[[-1.0], [1.0], [3.0]], [[100.0], [101.0], [102.0]]]])
    mat, length, _, summary = layer_head_scales_from_key_layers([layer0, layer1])
    expected_s2 = torch.stack([_row_s2(layer0[0]), _row_s2(layer1[0])]).double()
    expected = torch.sqrt(expected_s2)
    assert length == 3
    assert mat.shape == (2, 2)
    assert torch.allclose(mat, expected, atol=1e-6)
    assert abs(summary - math.sqrt(float(expected_s2.mean()))) < 1e-6


def test_backward_summary_helper_matches_matrix_summary():
    layer0 = torch.tensor([[[[0.0], [2.0], [4.0]], [[10.0], [10.0], [16.0]]]])
    layer1 = torch.tensor([[[[-1.0], [1.0], [3.0]], [[100.0], [101.0], [102.0]]]])
    got, rows, length, _ = global_scale_from_key_layers([layer0, layer1])
    expected_s2 = torch.cat([_row_s2(layer0[0]), _row_s2(layer1[0])]).double()
    assert rows == 4 and length == 3
    assert abs(got - math.sqrt(float(expected_s2.mean()))) < 1e-6


def test_loader_and_batch_alignment_keep_layer_head_identity(tmp_path):
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"model": "m", "layer_head_scales": [[1.0, 2.0], [3.0, 4.0]]}))
    scales = load_calibration_scales(str(path), "m")
    assert scale_for(scales, 1, 0) == 3.0
    # keys.reshape(B*H,...) orders [b0h0,b0h1,b1h0,b1h1]
    got = batched_head_scales(scales, 1, 2)
    assert got.tolist() == [3.0, 4.0, 3.0, 4.0]


def test_multi_sequence_calibration_uses_median_of_sequence_scales():
    from calibrate_layer_head_scales import aggregate_sequence_scales
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    b = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    frozen, mean, std, cv = aggregate_sequence_scales([a, b])
    expected = (a + b) / 2.0
    assert torch.allclose(frozen, expected)
    assert torch.allclose(mean, (a + b) / 2.0)
    assert torch.all(std >= 0)
    assert torch.all(cv >= 0)


def test_calibration_windows_are_reproducible_and_disjoint():
    from calibrate_layer_head_scales import select_disjoint_windows
    ids = torch.arange(400).view(1, -1)
    w1, starts1 = select_disjoint_windows(ids, num_sequences=4, seqlen=50, seed=42)
    w2, starts2 = select_disjoint_windows(ids, num_sequences=4, seqlen=50, seed=42)
    assert starts1 == starts2
    assert all(torch.equal(a, b) for a, b in zip(w1, w2))
    spans = [(s, s + 50) for s in starts1]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0]
