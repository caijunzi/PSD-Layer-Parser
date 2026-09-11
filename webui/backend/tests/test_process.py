"""任务提交接口测试。

核心安全约束：绝不真跑引擎——正常路径 patch 掉 task_manager.execute。
"""
import asyncio
import sys
import unittest
from unittest import mock
from unittest.mock import AsyncMock

from base import BaseWebUITest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class TestProcess(BaseWebUITest):

    def setUp(self):
        super().setUp()
        # 保证测试之间互不污染任务状态
        self.addCleanup(self._reset_task_manager)

    @staticmethod
    def _reset_task_manager():
        from core.task_manager import task_manager
        task_manager.tasks.clear()
        task_manager._current_process = None

    def _post(self, file_id="f_test", preset="textile_damask", mode="plate"):
        return self.client.post("/api/process", json={
            "file_id": file_id,
            "preset": preset,
            "mode": mode,
            "seed": 42,
        })

    def test_process_validation_422(self):
        # 缺 file_id → Pydantic 校验失败
        r = self.client.post("/api/process", json={"preset": "textile_damask"})
        self.assertEqual(r.status_code, 422)

    def test_process_invalid_mode_422(self):
        r = self.client.post("/api/process", json={
            "file_id": "f_test", "mode": "not_a_mode",
        })
        self.assertEqual(r.status_code, 422)

    def test_process_busy_409(self):
        from core.task_manager import task_manager
        # 模拟引擎占用（MVP 串行约束）
        task_manager._current_process = object()
        try:
            r = self._post()
            self.assertEqual(r.status_code, 409)
            self.assertIn("串行", r.json()["detail"])
        finally:
            task_manager._current_process = None

    def test_process_created_ok(self):
        from core.task_manager import task_manager
        # patch 掉 execute：只验证创建链路，绝不真跑引擎
        with mock.patch.object(
            task_manager, "execute", new=AsyncMock(return_value=None)
        ):
            r = self._post()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        data = body["data"]
        self.assertTrue(data["task_id"].startswith("task_"))
        self.assertIn(data["task_id"], data["ws_url"])
        self.assertIn("/ws/progress/", data["ws_url"])
        self.assertGreater(data["estimated_time"], 0)
        # 任务已登记，状态 pending
        task = task_manager.get_task(data["task_id"])
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["config"]["preset"], "textile_damask")

    def test_process_estimated_by_preset(self):
        from core.task_manager import task_manager
        with mock.patch.object(
            task_manager, "execute", new=AsyncMock(return_value=None)
        ):
            r = self._post(preset="textile_damask")   # est=80
            self.assertEqual(r.json()["data"]["estimated_time"], 80)
            r2 = self._post(preset="japanese_screen_gold")  # est=250
            self.assertEqual(r2.json()["data"]["estimated_time"], 250)
