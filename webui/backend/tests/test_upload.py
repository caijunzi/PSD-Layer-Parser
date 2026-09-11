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
