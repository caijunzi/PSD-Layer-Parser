# -*- coding: utf-8 -*-
"""Preset 加载路径的 CWD 独立性回归测试。

背景（2026-09-17 真实缺陷）：engine/schemas/presets.py 原用 CWD 相对路径
`PRESETS_DIR = "presets"`。后端 uvicorn 以 webui/backend 为 CWD 运行时，
人审「采纳」流程在后端进程内调用 load_preset → 解析到 <CWD>/presets/… →
FileNotFoundError。修复为按 __file__ 推导的绝对路径。
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tests/ 的上一级 = 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.schemas.presets import PRESETS_DIR, load_preset  # noqa: E402

ALL_PRESETS = [
    "chinese_ink_landscape_ai",
    "commercial_illustration",
    "japanese_screen_gold",
    "textile_damask",
    "textile_damask_photo",
    "traditional_chinese_ink",
    "western_oil_painting",
]


class TestPresetPathCwdIndependence(unittest.TestCase):
    def test_presets_dir_is_absolute(self):
        self.assertTrue(os.path.isabs(PRESETS_DIR),
                        "PRESETS_DIR 必须是绝对路径，否则依赖进程 CWD")
        self.assertTrue(os.path.isdir(PRESETS_DIR), PRESETS_DIR)

    def test_load_preset_from_foreign_cwd(self):
        """模拟 WebUI 后端：CWD 在 webui/backend 时仍能加载全部 preset。"""
        foreign = ROOT / "webui" / "backend"
        self.assertTrue(foreign.is_dir(), "缺少 webui/backend 目录")
        old = os.getcwd()
        try:
            os.chdir(str(foreign))
            for name in ALL_PRESETS:
                with self.subTest(preset=name):
                    cfg = load_preset(name)
                    self.assertEqual(cfg.get("preset_name"), name)
        finally:
            os.chdir(old)

    def test_load_preset_from_tmp_cwd(self):
        """更强的独立性：CWD 换到系统临时目录（与项目树无关）也不受影响。"""
        import tempfile

        old = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            cfg = load_preset("traditional_chinese_ink")
            self.assertEqual(cfg.get("preset_name"), "traditional_chinese_ink")
        finally:
            os.chdir(old)


if __name__ == "__main__":
    unittest.main()
