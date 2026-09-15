"""B3：审计回填链路隔离验证（task_manager._backfill_episode_audit）。

背景：引擎归档发生在审计之前 → episode 的 audit_passed 恒 False → CBR 检索永远筛空。
WebUI 任务流在审计完成后调用本链路回填。此前**未被任何测试覆盖**。
"""
import json
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

from core.task_manager import TaskManager  # noqa: E402


class TestAuditBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.ep = t / "episodes.jsonl"
        self.idx = t / "index.pkl"
        self._old = {
            k: os.environ.get(k) for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_INDEX_PATH")
        }
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(self.ep)
        os.environ["ADAPTIVE_INDEX_PATH"] = str(self.idx)

        def restore():
            for k, v in self._old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)

    def test_backfill_updates_episode_and_rebuilds_index(self):
        from engine.adaptive.episode_archiver import archive_episode, load_episode_by_id
        from engine.adaptive.episode_indexer import EpisodeIndexer

        emb = np.random.randn(128).astype(np.float32)
        ep_id = archive_episode(
            material_family="金地屏风",
            category_ids=["water_ripples"],
            fingerprint={"embedding": emb, "image_shape": (10, 10)},
            image_path="x.jpg",
            audit_8d=None,           # 归档时审计尚不存在（这正是要修复的场景）
        )

        # 构造审计结果（lost_ratio < 0.05 → 应判为通过）
        out = Path(self.tmp.name) / "out"
        out.mkdir(parents=True, exist_ok=True)
        audit = {
            "dims": {
                "④ 内容承载": {"metrics": {"lost_ratio": 0.01}},
                "⑤ 合成等价性": {"metrics": {"rmse_raw": 20.0, "rmse_lowfreq": 10.0}},
                "⑦ plate 合规": {"metrics": {"plate_purity_ok": True, "tac_max_pct": 300.0}},
                "① 层属性": {"metrics": {"layer_count": 42}},
            }
        }
        (out / "result.audit.json").write_text(
            json.dumps(audit, ensure_ascii=False), encoding="utf-8"
        )

        tm = TaskManager()
        tm.tasks["t1"] = {"logs": [{"message": f"[Adaptive] episode 已归档: {ep_id}"}]}
        tm._backfill_episode_audit("t1", out)

        # episode 已回填 audit_8d
        rec = load_episode_by_id(ep_id, str(self.ep))
        self.assertIsNotNone(rec)
        self.assertIsNotNone(rec.get("audit_8d"), "审计结果未回填到 episode")
        self.assertAlmostEqual(rec["audit_8d"]["lost_ratio"], 0.01, places=6)

        # 索引已重建且 audit_passed=True（修复前恒 False）
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertIn(ep_id, ix.fingerprints)
        self.assertTrue(ix.audit_status[ep_id])

    def test_backfill_without_episode_id_is_noop(self):
        out = Path(self.tmp.name) / "out2"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.audit.json").write_text("{}", encoding="utf-8")
        tm = TaskManager()
        tm.tasks["t2"] = {"logs": [{"message": "no episode here"}]}
        tm._backfill_episode_audit("t2", out)   # 不应抛异常
        self.assertFalse(self.idx.exists())


if __name__ == "__main__":
    unittest.main()
