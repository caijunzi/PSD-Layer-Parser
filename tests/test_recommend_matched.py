# -*- coding: utf-8 -*-
"""推荐可信度结构化（matched/reason）与未匹配样本入队回归测试。

背景（2026-09-17，短期 1+2）：推荐链路此前永远硬塞一个 preset 且不披露
「材质其实没认出来」。现要求：
  - matched=True  仅当材质/样块真正命中；
  - matched=False 时 confidence 封顶 0.55（前端「未识别」分级必然触发）；
  - 未匹配样本写入 unknown_samples.jsonl 待标注队列（失败不影响上传）。
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tests/ 的上一级 = 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND = ROOT / "webui" / "backend"   # core 包位于 webui/backend/core
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.file_handler import _recommend_preset  # noqa: E402


class TestRecommendMatched(unittest.TestCase):
    def test_no_dimensions_is_unmatched(self):
        """无尺寸 → 宽高比兜底都不可用 → matched=False，reason=no_dimensions。"""
        r = _recommend_preset(None, None)
        self.assertFalse(r["matched"])
        self.assertEqual(r["reason"], "no_dimensions")
        self.assertLessEqual(r["confidence"], 0.55)
        self.assertEqual(r["preset"], "japanese_screen_gold")

    def test_unmatched_conf_capped(self):
        """matched=False 时 confidence 封顶 0.55（前端「未识别」分级的触发依据）。"""
        r = _recommend_preset({"width": 1000, "height": 1000}, None)  # 方图 → aspect 0.72
        self.assertFalse(r["matched"])
        self.assertEqual(r["reason"], "aspect")
        self.assertLessEqual(r["confidence"], 0.55)
        # 方图比例 1.0 落在 0.85-1.18 的壁布比例带（历史口径，保持不变）
        self.assertEqual(r["preset"], "textile_damask")

    def test_noise_image_is_unmatched_and_reasoned(self):
        """纯噪声图：材质判别应给不出 ≥0.6 的家族 → matched=False 且带原因。"""
        import tempfile

        import cv2
        import numpy as np

        rng = np.random.RandomState(7)
        noise = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tmp = os.path.join(tempfile.gettempdir(), "uls_noise_probe.png")
        cv2.imwrite(tmp, noise)
        try:
            r = _recommend_preset({"width": 256, "height": 256}, tmp)
            self.assertFalse(r["matched"], f"噪声图不应被判定为可信命中: {r}")
            self.assertIn(r["reason"], ("material_low_conf", "material_error"))
            self.assertLessEqual(r["confidence"], 0.55)
            # 结构化披露：家族与置信度必须给出（供待标注队列使用）
            self.assertIn("material_family", r)
            self.assertIn("family_conf", r)
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)

    def test_known_fabric_sample_is_matched(self):
        """已知样本（工艺壁布）：材质应命中 → matched=True 且置信度不被封顶。"""
        sample = ROOT / "inputs" / "壁布实物照_深灰菱块卷草刺绣_斜摄带卷边.jpeg"
        if not sample.is_file():
            self.skipTest("缺少已知样本 inputs/壁布实物照_深灰菱块卷草刺绣_斜摄带卷边.jpeg")
        r = _recommend_preset({"width": 2848, "height": 1600}, str(sample))
        self.assertTrue(r["matched"], f"已知壁布样本应命中: {r}")
        self.assertIn(r["preset"], ("textile_damask", "textile_damask_photo"))
        self.assertGreaterEqual(r["confidence"], 0.6)
        # panel（检出样块）或 material（家族直接命中）均为合法命中路径
        self.assertIn(r["reason"], ("panel", "material"))


if __name__ == "__main__":
    unittest.main()
