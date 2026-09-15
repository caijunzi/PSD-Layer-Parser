"""B2：后台学习任务两端点真机验证（TestClient）。

背景：background_learner 的 task_queue 此前**无消费者**、两个文档声明的端点
（POST /adaptive/trigger-learning、GET /adaptive/learning-status）**从未注册**。
本测试走真实 HTTP 路由验证注册与行为。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
import core.background_learner as bl  # noqa: E402


class TestLearningEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self._old = {
            k: os.environ.get(k) for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_INDEX_PATH")
        }
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(t / "ep.jsonl")
        os.environ["ADAPTIVE_INDEX_PATH"] = str(t / "ix.pkl")
        bl._background_learner = None   # 重置单例，避免跨测试污染
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        bl._background_learner = None

    def test_learning_status_returns_tasks(self):
        r = self.client.get("/api/adaptive/learning-status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("tasks", r.json())

    def test_trigger_learning_without_episode_returns_404(self):
        r = self.client.post("/api/adaptive/trigger-learning")
        self.assertEqual(r.status_code, 404)

    def test_trigger_learning_enqueues_and_status_visible(self):
        from engine.adaptive.episode_archiver import archive_episode

        ep = archive_episode(
            material_family="金地屏风",
            category_ids=["water_ripples"],
            fingerprint={"embedding": np.random.randn(128).astype(np.float32),
                         "image_shape": (8, 8)},
            image_path="x.jpg",
        )
        r = self.client.post("/api/adaptive/trigger-learning", data={"episode_id": ep})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("episode_id"), ep)

        tasks = self.client.get("/api/adaptive/learning-status").json()["tasks"]
        self.assertTrue(any(t.get("episode_id") == ep for t in tasks),
                        "入队任务未出现在 learning-status")


if __name__ == "__main__":
    unittest.main()
