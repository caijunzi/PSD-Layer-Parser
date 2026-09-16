"""WebSocket 进度端点测试。"""
from base import BaseWebUITest


class TestWebSocket(BaseWebUITest):

    def setUp(self):
        super().setUp()
        self.addCleanup(self._clean_tasks)

    @staticmethod
    def _clean_tasks():
        from core.task_manager import task_manager
        task_manager.tasks.pop("t_ws_replay", None)

    def test_ws_connect_no_task(self):
        """连接不存在的 task：应正常 accept，心跳收发后干净断开。"""
        with self.client.websocket_connect("/ws/progress/fake_task") as ws:
            ws.send_text("ping")  # 服务端 receive_text 循环收到即可

    def test_ws_replay_existing_logs(self):
        """断线重连场景：连接后应立即补播存量日志（replay 标记）。"""
        from core.task_manager import task_manager
        task_manager.tasks["t_ws_replay"] = {
            "task_id": "t_ws_replay",
            "status": "processing",
            "progress": 33,
            "stage": "神经分割",
            "logs": [
                {"timestamp": "2026-09-11T10:00:00", "message": "日志行1"},
                {"timestamp": "2026-09-11T10:00:01", "message": "日志行2"},
            ],
            "output_files": [],
            "elapsed_time": 0,
        }
        with self.client.websocket_connect("/ws/progress/t_ws_replay") as ws:
            msg1 = ws.receive_json()
            self.assertEqual(msg1["type"], "progress")
            self.assertTrue(msg1.get("replay"))
            self.assertEqual(msg1["message"], "日志行1")
            msg2 = ws.receive_json()
            self.assertEqual(msg2["message"], "日志行2")
            self.assertEqual(msg2["progress"], 33)
            ws.send_text("ping")

    def test_ws_replay_completed(self):
        """任务已完成时连接：补播日志后立即收到 completed 快照。"""
        from core.task_manager import task_manager
        task_manager.tasks["t_ws_done"] = {
            "task_id": "t_ws_done",
            "status": "completed",
            "progress": 100,
            "stage": "收尾校验",
            "logs": [{"timestamp": "t", "message": "完成"}],
            "output_files": [{"type": "design", "filename": "a.psb",
                              "size": 1, "download_url": "/x"}],
            "elapsed_time": 12.5,
        }
        self.addCleanup(task_manager.tasks.pop, "t_ws_done", None)
        with self.client.websocket_connect("/ws/progress/t_ws_done") as ws:
            replay = ws.receive_json()
            self.assertTrue(replay.get("replay"))
            done = ws.receive_json()
            self.assertEqual(done["type"], "completed")
            self.assertEqual(done["task_id"], "t_ws_done")
            self.assertEqual(done["elapsed_time"], 12.5)
            self.assertEqual(done["output_files"][0]["type"], "design")

    def test_ws_replay_completed_carries_manifest(self):
        """★ 断线重连补播的 completed 必须带**真实 manifest**（2026-09-16 修复）。

        此前补播硬写 `manifest: {}`，而实时路径由 task_manager 从磁盘读真 manifest
        → 刷新/重连后前端 manifest 面板会变空，与实时路径不一致。
        修复：task_manager 完成时把 manifest 落存到 task 记录，补播取同一份。
        """
        from core.task_manager import task_manager
        task_manager.tasks["t_ws_man"] = {
            "task_id": "t_ws_man",
            "status": "completed",
            "progress": 100,
            "stage": "收尾校验",
            "logs": [],
            "output_files": [],
            "elapsed_time": 7.0,
            "manifest": {"totals": {"layer_count": 7}, "source": {"effective_ppi": 150}},
        }
        self.addCleanup(task_manager.tasks.pop, "t_ws_man", None)
        with self.client.websocket_connect("/ws/progress/t_ws_man") as ws:
            done = ws.receive_json()
            self.assertEqual(done["type"], "completed")
            self.assertEqual(done["manifest"], {"totals": {"layer_count": 7},
                                                "source": {"effective_ppi": 150}})
