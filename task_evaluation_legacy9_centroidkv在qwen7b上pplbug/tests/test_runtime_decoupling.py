import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LB = (ROOT / "evaluate_longbench_kvmerge.py").read_text(encoding="utf-8")
PPL = (ROOT / "evaluate_ppl_kvmerge.py").read_text(encoding="utf-8")


def function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise AssertionError(f"Missing function {name}")


class RuntimeDecouplingTests(unittest.TestCase):
    def test_centroidkv_backend_override_is_explicit(self):
        src = function_source(LB, "effective_attn_implementation")
        self.assertIn('{"None", "CentroidKV"}', src)
        self.assertIn('return "sdpa"', src)

    def test_longbench_centroidkv_branch_has_no_grouplogit_dependency(self):
        src = function_source(LB, "setup_kv_imports")
        start = src.index('if kv_method == "CentroidKV":')
        end = src.index('if kv_method == "GroupLogitKV":', start)
        centroid = src[start:end]
        self.assertIn('merge_centroidkv_', centroid)
        self.assertNotIn('grouplogitkv', centroid.lower())
        self.assertNotIn('score_capture_runtime', centroid)

    def test_longbench_simcal_cam_use_neutral_capture(self):
        src = function_source(LB, "setup_kv_imports")
        sim_start = src.index('if kv_method == "SimCalKV":')
        cam_start = src.index('if kv_method == "CaM":', sim_start)
        keep_start = src.index('if kv_method == "KeepKV":', cam_start)
        sim = src[sim_start:cam_start]
        cam = src[cam_start:keep_start]
        for block in (sim, cam):
            self.assertIn('score_capture_runtime_patch_', block)
            self.assertNotIn('grouplogitkv_runtime_patch', block)

    def test_grouplogitkv_keeps_own_runtime(self):
        src = function_source(LB, "setup_kv_imports")
        start = src.index('if kv_method == "GroupLogitKV":')
        end = src.index('if kv_method == "SimCalKV":', start)
        block = src[start:end]
        self.assertIn('merge_grouplogitkv_', block)
        self.assertIn('grouplogitkv_runtime_patch_', block)

    def test_ppl_dispatch_is_also_method_isolated(self):
        src = function_source(PPL, "apply_runtime_patch_if_needed")
        self.assertIn('if method == "CentroidKV":', src)
        self.assertIn('if method == "GroupLogitKV":', src)
        self.assertIn('if method in {"SimCalKV", "CaM"}:', src)
        self.assertIn('score_capture_runtime_patch_qwen', src)
        self.assertIn('score_capture_runtime_patch_llama', src)


if __name__ == "__main__":
    unittest.main()
