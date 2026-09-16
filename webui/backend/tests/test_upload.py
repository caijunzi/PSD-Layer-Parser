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
        self.assertIn(data["recommended_preset"],
                      {"japanese_screen_gold", "textile_damask"})
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
    """preset 推荐（2026-09-16 增强）：先判实物样品照，再按宽高比粗判。

    样品照的宽高比任意（实测 1536×1024 = 1.5），纯按比例会误荐 japanese_screen_gold；
    故必须先跑 `engine.core.sample_panel` 的样块检测。
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

    def test_synthetic_sample_panel_recommends_photo_preset(self):
        import numpy as np
        rng = np.random.default_rng(0)
        img = np.clip(100 + rng.normal(0, 2, (400, 600, 3)), 0, 255).astype(np.uint8)
        img[60:340, 90:510] = np.clip(180 + rng.normal(0, 2, (280, 420, 3)), 0, 255).astype(np.uint8)
        p = self._write(img, "syn_panel.png")
        from core.file_handler import _recommend_preset
        name, conf = _recommend_preset({"width": 600, "height": 400}, p)
        self.assertEqual(name, "textile_damask_photo")
        self.assertGreaterEqual(conf, 0.8)

    def test_real_wallcovering_recommends_photo_preset(self):
        """真实工艺壁布（织物特写，无样块）→ 材质判别为「织物壁布」→ 走照片 preset。

        不能按宽高比乱荐（2848×1600 ratio=1.78 会落到 japanese_screen_gold）。
        """
        from pathlib import Path
        p = Path(__file__).resolve().parents[3] / "inputs" / "工艺壁布-1.jpeg"
        if not p.is_file():
            self.skipTest("缺少 inputs/工艺壁布-1.jpeg")
        from core.file_handler import _recommend_preset
        name, conf = _recommend_preset({"width": 2848, "height": 1600}, str(p))
        self.assertEqual(name, "textile_damask_photo",
                         f"织物壁布应荐照片 preset，实际 {name}({conf})")
        self.assertGreaterEqual(float(conf), 0.6)

    def test_fabric_closeup_falls_back_to_ratio(self):
        """无样块的**非织物**图不得误判为样品照（避免把正常画面切成背景带+内容）。"""
        import numpy as np
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (400, 600, 3)).astype(np.uint8)
        p = self._write(img, "noise.png")
        from core.file_handler import _recommend_preset
        name, _ = _recommend_preset({"width": 600, "height": 400}, p)
        self.assertNotEqual(name, "textile_damask_photo")

    def test_ratio_fallbacks_unchanged(self):
        from core.file_handler import _recommend_preset
        self.assertEqual(_recommend_preset({"width": 4000, "height": 1952})[0], "japanese_screen_gold")
        self.assertEqual(_recommend_preset({"width": 1000, "height": 1000})[0], "textile_damask")
        self.assertEqual(_recommend_preset(None)[0], "japanese_screen_gold")
        # 不存在的路径 → 安全降级
        self.assertEqual(_recommend_preset({"width": 1000, "height": 1000}, "/no/such.png")[0],
                         "textile_damask")
