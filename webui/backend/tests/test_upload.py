"""上传接口测试：合法 PNG / 非法格式 / 缺文件。"""
import io

from base import BaseWebUITest


def _tiny_png() -> bytes:
    """用 Pillow 生成 4x4 红色 PNG（保证 Pillow 能读出尺寸）。"""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class TestUpload(BaseWebUITest):

    def test_upload_png_ok(self):
        r = self.client.post(
            "/api/upload",
            files={"file": ("tiny.png", _tiny_png(), "image/png")},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        data = body["data"]
        self.assertTrue(data["file_id"])
        self.assertEqual(data["filename"], "tiny.png")
        self.assertEqual(data["dimensions"], {"width": 4, "height": 4})
        self.assertTrue(data["thumbnail_url"].startswith("/thumbnails/"))
        # 推荐值须是**已知 preset 名**（2026-09-16 起推荐改由材质家族优先，
        # 4×4 纯红小图会按材质落到油画布，故不再写死为"金地/壁布"两种）
        self.assertIn(data["recommended_preset"],
                      {"japanese_screen_gold", "textile_damask", "textile_damask_photo",
                       "chinese_ink_landscape_ai", "western_oil_painting"})
        self.assertGreaterEqual(data["confidence"], 0.5)

    def test_upload_txt_rejected(self):
        r = self.client.post(
            "/api/upload",
            files={"file": ("note.txt", b"hello", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("不支持", r.json()["detail"])

    def test_upload_missing_file_422(self):
        r = self.client.post("/api/upload")
        self.assertEqual(r.status_code, 422)

    def test_upload_oversize_rejected(self):
        # 101MB 的假 PNG（不真占内存：is_allowed_file 只看 len）
        big = b"\x89PNG" + b"\x00" * (101 * 1024 * 1024)
        r = self.client.post(
            "/api/upload",
            files={"file": ("big.png", big, "image/png")},
        )
        self.assertEqual(r.status_code, 400)


class TestRecommendPreset(BaseWebUITest):
    """preset 推荐（2026-09-16）：**材质家族优先**，宽高比仅兜底。

    顺序本身是关键。曾踩的坑：把「样块检测」放最前单独决定品类 —— 带绫边外框的
    **金地屏风**四条直边恰好满足"长直边持续性"，被误判为实物样块 → 推荐成壁布 preset。
    现在样块检测只在**纺织类家族**（绢本/织物）内做二次判定。
    """

    @staticmethod
    def _write(arr, name):
        import os
        import tempfile
        import cv2
        d = tempfile.mkdtemp()
        p = os.path.join(d, name)
        cv2.imwrite(p, arr)
        return p

    @staticmethod
    def _inputs(name):
        from pathlib import Path
        p = Path(__file__).resolve().parents[3] / "inputs" / name
        return p if p.is_file() else None

    def _rec(self, rel, w, h):
        from core.file_handler import _recommend_preset
        return _recommend_preset({"width": w, "height": h}, rel)

    def test_real_wallcovering_recommends_photo_preset(self):
        """真实工艺壁布（织物特写，无样块）→ 材质判别「织物壁布」→ 照片 preset。

        不能按宽高比乱荐（2848×1600，ratio=1.78 会落到 japanese_screen_gold）。
        """
        checked = 0
        for i in (1, 2, 3):
            p = self._inputs(f"工艺壁布-{i}.jpeg")
            if p is None:
                continue
            name, conf = self._rec(str(p), 2848, 1600)
            self.assertEqual(name, "textile_damask_photo",
                             f"工艺壁布-{i} 应荐照片 preset，实际 {name}({conf})")
            self.assertGreaterEqual(float(conf), 0.6)
            checked += 1
        if checked == 0:
            self.skipTest("缺少 inputs/工艺壁布-*.jpeg")

    def test_gold_screen_not_misrouted_to_photo_preset(self):
        """★ 回归：金地屏风**不得**被绫边外框带偏成壁布样品照 preset。

        实测 `inputs/source_4000.jpg`（4000×1952，含绫边外框）在旧顺序下会被
        样块检测命中而误荐 textile_damask_photo。
        """
        p = self._inputs("source_4000.jpg")
        if p is None:
            self.skipTest("缺少 inputs/source_4000.jpg")
        name, conf = self._rec(str(p), 4000, 1952)
        self.assertEqual(name, "japanese_screen_gold",
                         f"金地屏风应荐 japanese_screen_gold，实际 {name}({conf})")
        self.assertNotEqual(name, "textile_damask_photo")

    def test_oil_painting_recommends_oil_preset(self):
        p = self._inputs("油画.jpeg")
        if p is None:
            self.skipTest("缺少 inputs/油画.jpeg")
        name, _ = self._rec(str(p), 2880, 1440)
        self.assertEqual(name, "western_oil_painting", f"实际 {name}")

    def test_ink_and_silk_recommend_ink_preset(self):
        """水墨与绢本均归 `chinese_ink_landscape_ai`（README：绢本必须用该 preset）。"""
        for fn in ("宋代写意山水画创作.jpeg", "绢本工笔画-1.jpeg"):
            p = self._inputs(fn)
            if p is None:
                continue
            name, _ = self._rec(str(p), 2880, 1440)
            self.assertEqual(name, "chinese_ink_landscape_ai", f"{fn} 实际 {name}")

    def test_non_fabric_noise_falls_back_to_ratio(self):
        """无结构噪声不得被判为织物/样品照（材质判别置信不足即退回宽高比）。"""
        import numpy as np
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (400, 600, 3)).astype(np.uint8)
        p = self._write(img, "noise.png")
        name, _ = self._rec(p, 600, 400)
        self.assertNotEqual(name, "textile_damask_photo")

    def test_ratio_fallbacks_unchanged(self):
        from core.file_handler import _recommend_preset
        self.assertEqual(_recommend_preset({"width": 4000, "height": 1952})[0], "japanese_screen_gold")
        self.assertEqual(_recommend_preset({"width": 1000, "height": 1000})[0], "textile_damask")
        self.assertEqual(_recommend_preset(None)[0], "japanese_screen_gold")
        # 不存在的路径 → 安全降级
        self.assertEqual(_recommend_preset({"width": 1000, "height": 1000}, "/no/such.png")[0],
                         "textile_damask")
