"""核心模块纯函数单元测试：file_handler / task_manager。"""
import sys
from pathlib import Path
from unittest import mock

from base import BaseWebUITest

from core.file_handler import (
    is_allowed_file, _recommend_preset, list_presets, append_history, read_history,
)
from core.task_manager import task_manager, _match_stage, PRESET_ESTIMATE


class TestFileHandlerRules(BaseWebUITest):

    def test_is_allowed_file(self):
        self.assertTrue(is_allowed_file("a.png", 100))
        self.assertTrue(is_allowed_file("a.JPG", 100))       # 大写扩展名
        self.assertTrue(is_allowed_file("a.psd", 100))
        self.assertFalse(is_allowed_file("a.txt", 100))       # 格式拒绝
        self.assertFalse(is_allowed_file("a.exe", 100))
        self.assertFalse(is_allowed_file("a.png", 101 * 1024 * 1024))  # 超大

    def test_recommend_preset_ratio(self):
        # 宽:高 ≈ 2:1 → 屏风（横向长卷）
        name, conf = _recommend_preset({"width": 2000, "height": 1000})
        self.assertEqual(name, "japanese_screen_gold")
        self.assertEqual(conf, 0.78)
        # ≈ 1:1 → 壁布纹样
        name, conf = _recommend_preset({"width": 1000, "height": 1000})
        self.assertEqual(name, "textile_damask")
        # 极端比例 → 默认屏风低置信
        name, conf = _recommend_preset({"width": 3000, "height": 1000})
        self.assertEqual((name, conf), ("japanese_screen_gold", 0.55))
        # 读不到尺寸 → 0.5 兜底
        self.assertEqual(_recommend_preset(None), ("japanese_screen_gold", 0.5))

    def test_list_presets_real_dir(self):
        presets = list_presets()
        names = {p["name"] for p in presets}
        self.assertIn("japanese_screen_gold", names)
        self.assertIn("western_oil_painting", names)
        for p in presets:
            self.assertGreaterEqual(p["semantic_classes"], 0)

    def test_history_roundtrip(self):
        append_history({"task_id": "t1", "status": "completed", "created_at": "2026-09-11T10:00:00"})
        append_history({"task_id": "t2", "status": "completed", "created_at": "2026-09-11T11:00:00"})
        items = read_history()
        self.assertEqual(len(items), 2)
        # 倒序：最新的在前
        self.assertEqual(items[0]["task_id"], "t2")


class TestTaskManagerUnit(BaseWebUITest):

    def test_match_stage_keywords(self):
        self.assertEqual(_match_stage("CUDA 探活通过，分割上 GPU"), "加载模型")
        self.assertEqual(_match_stage("[SAM2] 级联分割执行中"), "神经分割")
        self.assertEqual(_match_stage("[RealESRGAN] 渐进超分 2.0x → 4.0x"), "超分重建")
        self.assertEqual(_match_stage("[ICC] FOGRA39 分色驱动"), "制版分色")
        self.assertIsNone(_match_stage("无关普通日志行"))

    def test_create_task_registers(self):
        tid, est, ws_url = task_manager.create_task("f1", {"preset": "textile_damask"})
        self.assertTrue(tid.startswith("task_"))
        self.assertEqual(est, PRESET_ESTIMATE["textile_damask"])
        self.assertIn(f"/ws/progress/{tid}", ws_url)
        task = task_manager.get_task(tid)
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["logs"], [])
        # 清理
        task_manager.tasks.pop(tid, None)

    def test_resolve_input(self):
        fid = "unit_resolve_abc"
        (Path(self._tmp.name) / "uploads" / f"{fid}.png").write_bytes(b"x")
        found = task_manager._resolve_input(fid)
        self.assertIsNotNone(found)
        self.assertEqual(found.stem, fid)
        self.assertIsNone(task_manager._resolve_input("no_such_file"))

    def test_collect_outputs(self):
        out_dir = Path(self._tmp.name) / "outputs" / "t_out"
        (out_dir / "result.masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "result.design.psb").write_bytes(b"d")
        (out_dir / "result.plate.psb").write_bytes(b"p")
        (out_dir / "result.manifest.json").write_text("{}", encoding="utf-8")
        files = task_manager._collect_outputs(out_dir, "t_out", mode="both")
        types = {f["type"] for f in files}
        self.assertEqual(types, {"design", "plate", "manifest"})
        for f in files:
            self.assertTrue(f["download_url"].startswith("/api/download/t_out/"))
            self.assertGreater(f["size"], 0)

    def test_collect_outputs_single_mode(self):
        """单模式（plate）：产物是 result.psb 本身（引擎不加后缀）——回归锁。"""
        out_dir = Path(self._tmp.name) / "outputs" / "t_single"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.psb").write_bytes(b"x" * 10)
        files = task_manager._collect_outputs(out_dir, "t_single", mode="plate")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["type"], "plate")
        self.assertEqual(files[0]["filename"], "result.psb")
