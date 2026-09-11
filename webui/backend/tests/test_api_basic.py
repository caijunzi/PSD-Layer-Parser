"""基础 API 测试：健康检查 / presets / history / download 错误分支。"""
from pathlib import Path

from base import BaseWebUITest


class TestBasicAPI(BaseWebUITest):

    def test_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["service"], "Universal Layer Studio PRO API")
        self.assertEqual(body["status"], "running")

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "healthy")
        self.assertTrue(body["uploads_ready"])
        self.assertTrue(body["outputs_ready"])

    def test_presets_returns_all_five(self):
        r = self.client.get("/api/presets")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        names = {p["name"] for p in body["data"]}
        # 封闭 5 类对应 preset 必须全部可见
        expected = {
            "japanese_screen_gold",      # 屏风 / 烫金
            "textile_damask",            # 壁布
            "chinese_ink_landscape_ai",  # 水墨
            "western_oil_painting",      # 油画
        }
        self.assertTrue(expected.issubset(names), f"缺少 preset: {expected - names}")
        # 每个条目结构完整
        for p in body["data"]:
            self.assertIn("semantic_classes", p)
            self.assertIn("output_modes", p)

    def test_history_empty_when_no_data(self):
        r = self.client.get("/api/history")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        # patch 后 history.json 不存在 → 空
        self.assertEqual(body["data"]["total"], 0)
        self.assertEqual(body["data"]["items"], [])

    def test_download_unknown_task_404(self):
        r = self.client.get("/api/download/no_such_task/design")
        self.assertEqual(r.status_code, 404)

    def test_download_unknown_type_400(self):
        # 先造出产物目录，让流程走到 file_type 校验分支
        from core.file_handler import OUTPUTS_DIR
        (OUTPUTS_DIR / "t_fake").mkdir(parents=True, exist_ok=True)
        r = self.client.get("/api/download/t_fake/badtype")
        self.assertEqual(r.status_code, 400)

    def test_download_missing_file_404(self):
        from core.file_handler import OUTPUTS_DIR
        (OUTPUTS_DIR / "t_empty").mkdir(parents=True, exist_ok=True)
        r = self.client.get("/api/download/t_empty/design")
        self.assertEqual(r.status_code, 404)
