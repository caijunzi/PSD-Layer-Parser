"""输出尺寸决策测试 —— --scale 优先级（回归时暴露的历史缺陷）。

优先级契约：显式 target_w/h > CLI --scale > preset.super_res_scale > 4.0。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_universal_engine import resolve_output_size  # noqa: E402


class TestResolveOutputSize(unittest.TestCase):
    def test_explicit_wh_wins(self):
        self.assertEqual(
            resolve_output_size(4000, 2000, 1.0, 8000, 4000, {"super_res_scale": 4.0}),
            (8000, 4000))

    def test_cli_scale_beats_preset(self):
        """历史缺陷回归：preset.super_res_scale 不得覆盖显式 CLI --scale。"""
        self.assertEqual(
            resolve_output_size(4000, 2000, 1.0, None, None, {"super_res_scale": 4.0}),
            (4000, 2000))

    def test_preset_used_when_cli_absent(self):
        self.assertEqual(
            resolve_output_size(4000, 2000, None, None, None, {"super_res_scale": 4.0}),
            (16000, 8000))

    def test_builtin_default_when_both_absent(self):
        self.assertEqual(
            resolve_output_size(4000, 2000, None, None, None, {}),
            (16000, 8000))

    def test_preset_without_key_uses_cli(self):
        self.assertEqual(
            resolve_output_size(4000, 2000, 0.5, None, None, {}),
            (2000, 1000))


if __name__ == "__main__":
    unittest.main()
