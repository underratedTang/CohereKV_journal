import subprocess
import unittest
from pathlib import Path

from mymethod_selector_tag import active_mymethod_selector_config, build_mymethod_selector_tag


class MyMethodSelectorTagTests(unittest.TestCase):
    def test_attention_tag_ignores_other_selector_parameters(self):
        tag = build_mymethod_selector_tag(
            "attention",
            keepkv_sink_size=99,
            d2o_score_reset_interval=77,
        )
        self.assertEqual(tag, "selector_attention")

    def test_keepkv_tag_contains_only_active_keepkv_parameters(self):
        tag = build_mymethod_selector_tag(
            "keepkv",
            keepkv_sink_size=4,
            keepkv_recent_protect_size=128,
            keepkv_score_window=64,
            keepkv_recency_beta=1.2,
            keepkv_use_log_count=True,
            d2o_sink_size=999,
        )
        self.assertEqual(
            tag,
            "selector_keepkv__sink4__recent128__window64__beta1p2__logc1",
        )
        self.assertNotIn("999", tag)

    def test_d2o_tag_contains_only_active_d2o_parameters(self):
        tag = build_mymethod_selector_tag(
            "d2o",
            d2o_sink_size=8,
            d2o_recent_size=64,
            d2o_use_log_count=False,
            d2o_score_reset_interval=3,
            keepkv_score_window=999,
        )
        self.assertEqual(tag, "selector_d2o__sink8__recent64__logc0__reset3")
        self.assertNotIn("999", tag)

    def test_active_selector_config_is_explicit(self):
        self.assertEqual(
            active_mymethod_selector_config("attention"),
            {"selector": "attention"},
        )

    def test_shell_and_python_tag_formats_match(self):
        helper = Path(__file__).resolve().parents[1] / "mymethod_selector_tag.sh"
        cmd = (
            f'source "{helper}"; '
            'build_mymethod_selector_tag keepkv 4 128 64 1.2 1'
        )
        shell_tag = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
        python_tag = build_mymethod_selector_tag(
            "keepkv",
            keepkv_sink_size=4,
            keepkv_recent_protect_size=128,
            keepkv_score_window=64,
            keepkv_recency_beta=1.2,
            keepkv_use_log_count=True,
        )
        self.assertEqual(shell_tag, python_tag)

        cmd_default = (
            f'source "{helper}"; '
            'build_mymethod_selector_tag keepkv 4 128 128 1.0 0'
        )
        shell_default = subprocess.check_output(["bash", "-lc", cmd_default], text=True).strip()
        python_default = build_mymethod_selector_tag(
            "keepkv",
            keepkv_sink_size=4,
            keepkv_recent_protect_size=128,
            keepkv_score_window=128,
            keepkv_recency_beta=1.0,
            keepkv_use_log_count=False,
        )
        self.assertEqual(shell_default, python_default)


if __name__ == "__main__":
    unittest.main()
