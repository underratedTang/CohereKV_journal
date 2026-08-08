from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_ablation_kvmerge as ablation


class AblationWrapperRegressionTests(unittest.TestCase):
    def test_group_ablation_accepts_mymethod_config_keyword(self):
        calls = []
        fake = types.ModuleType("evaluate_ppl_kvmerge")

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return (("k", "v"),)

        fake.apply_kv_method = original

        def main():
            # Reproduce the evaluator call that previously raised TypeError.
            fake.apply_kv_method(
                "GroupLogitKV", "llama", object(), (("k", "v"),), None, None,
                0.4, method_sim_thrs={}, keepkv_config={}, task_eval_dir=ROOT,
                target_kv_len=8, mymethod_config={"selector": "keepkv"},
            )

        fake.main = main
        old = sys.modules.get("evaluate_ppl_kvmerge")
        sys.modules["evaluate_ppl_kvmerge"] = fake
        try:
            ablation.run_ppl("full", [])
        finally:
            if old is None:
                sys.modules.pop("evaluate_ppl_kvmerge", None)
            else:
                sys.modules["evaluate_ppl_kvmerge"] = old
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "GroupLogitKV")
        self.assertIn("mymethod_config", calls[0][1])


if __name__ == "__main__":
    unittest.main()
