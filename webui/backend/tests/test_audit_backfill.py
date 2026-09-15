"""B3：审计回填链路隔离验证（逐产物 → 对应 episode，不再用末条日志猜）。

背景：旧 `_backfill_episode_audit` 只审计单个产物（偏好 plate），却用「末条 episode
日志」回填——both 模式下末条是 design episode，导致 plate 审计被错填到 design episode。
现改为 `finalize_episode` 逐产物明确映射：plate 产物 → plate episode、design 产物 →
design episode。本测试覆盖该映射与 CBR 索引准入。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.task_manager import TaskManager  # noqa: E402
from engine.adaptive.episode_archiver import (  # noqa: E402
    archive_episode, load_episode_by_id, finalize_episode, DIM_KEYS, PLATE_DIM,
)


def _make_report(design_exempt=False, fail_dim=None):
    dims = {k: {"passed": True, "evaluated": True, "na": False, "metrics": {}} for k in DIM_KEYS}
    if design_exempt:
        dims[PLATE_DIM]["na"] = True
        dims[PLATE_DIM]["evaluated"] = False
    if fail_dim is not None:
        dims[fail_dim]["passed"] = False
    return {"passed": fail_dim is None, "dims": dims}


class TestPerArtifactFinalize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.ep = t / "episodes.jsonl"
        self.idx = t / "index.pkl"
        self._old = {k: os.environ.get(k) for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_INDEX_PATH")}
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(self.ep)
        os.environ["ADAPTIVE_INDEX_PATH"] = str(self.idx)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _archive_pair(self):
        emb = np.random.randn(128).astype(np.float32)
        ep_plate = archive_episode(
            material_family="金地屏风", category_ids=["water_ripples"],
            fingerprint={"embedding": emb, "image_shape": (10, 10)},
            image_path="x.jpg", output_mode="plate",
        )
        emb2 = np.random.randn(128).astype(np.float32)
        ep_design = archive_episode(
            material_family="金地屏风", category_ids=["water_ripples"],
            fingerprint={"embedding": emb2, "image_shape": (10, 10)},
            image_path="x.jpg", output_mode="design",
        )
        return ep_plate, ep_design

    def test_finalize_per_artifact_maps_correctly(self):
        ep_plate, ep_design = self._archive_pair()
        # 各自 finalize：plate 全过、design 豁免⑦全过
        d_plate = finalize_episode(ep_plate, audit_report=_make_report(), output_mode="plate")
        d_design = finalize_episode(ep_design, audit_report=_make_report(design_exempt=True),
                                    output_mode="design")
        self.assertTrue(d_plate["admitted"])
        self.assertTrue(d_design["admitted"])

        rec_p = load_episode_by_id(ep_plate, str(self.ep))
        rec_d = load_episode_by_id(ep_design, str(self.ep))
        self.assertIsNotNone(rec_p["audit_full"])
        self.assertIsNotNone(rec_d["audit_full"])
        self.assertEqual(rec_p["output_mode"], "plate")
        self.assertEqual(rec_d["output_mode"], "design")

        from engine.adaptive.episode_indexer import EpisodeIndexer
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertIn(ep_plate, ix.fingerprints)
        self.assertIn(ep_design, ix.fingerprints)

    def test_finalize_missing_episode_is_noop(self):
        # 先建日志（含一个真实 episode），再 finalize 一个不存在的 id → episode_not_found
        self._archive_pair()
        d = finalize_episode("ep_nonexistent", audit_report=_make_report(), output_mode="plate")
        self.assertFalse(d["admitted"])
        self.assertEqual(d["reason"], "episode_not_found")
        self.assertFalse(self.idx.exists())  # 未准入，索引不应生成

    def test_task_manager_maps_artifacts_to_episodes(self):
        """TaskManager._finalize_artifacts 把 plate/design 产物明确映射到对应 episode。"""
        ep_plate, ep_design = self._archive_pair()
        tm = TaskManager()
        out = Path(self.tmp.name) / "out"
        out.mkdir(parents=True, exist_ok=True)
        # 模拟引擎日志：先 plate 后 design（与 jobs=[PLATE, DESIGN] 一致）
        tm.tasks["t1"] = {"logs": [
            {"message": f"[Adaptive] episode 已归档: {ep_plate}"},
            {"message": f"[Adaptive] episode 已归档: {ep_design}"},
        ]}
        # 两个产物各自的审计报告（含 color_mode 真实产品线）
        plate_report = dict(_make_report()); plate_report["color_mode"] = "cmyk"
        design_report = dict(_make_report(design_exempt=True)); design_report["color_mode"] = "rgb"
        audit_reports = {
            str(out / "result.plate.psb"): (plate_report, "plate"),
            str(out / "result.design.psb"): (design_report, "design"),
        }
        called = []
        with patch("engine.adaptive.episode_archiver.finalize_episode",
                   side_effect=lambda *a, **k: called.append((a, k)) or {"admitted": True}) as m:
            tm._finalize_artifacts("t1", out, audit_reports, "both")
        self.assertEqual(len(called), 2)
        # 两路调用必须分别带正确的 episode（位置参）与 output_mode（关键字参）
        by_om = {k["output_mode"]: a[0] for a, k in called}
        self.assertEqual(by_om["plate"], ep_plate)
        self.assertEqual(by_om["design"], ep_design)


if __name__ == "__main__":
    unittest.main()
