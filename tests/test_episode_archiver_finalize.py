"""episode_archiver 严格 8 维准入 + finalize_episode API 测试。

覆盖：
- finalize_episode 保留原始完整报告（audit_full）而非压成 4 字段摘要；
- 严格准入：plate 全通过→准入；plate ⑦失败→skip；design 豁免⑦→准入；
  缺必需维度→skip；仅旧式摘要(无 audit_full)→skip（不降低阈值凑通过）；
- sync_index_from_log 权威重建：索引只收严格准入通过的 episode。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.adaptive import episode_archiver as ea  # noqa: E402
from engine.adaptive.episode_indexer import EpisodeIndexer  # noqa: E402


def _make_report(fail_dim=None, not_eval=None, na_dims=None, design_exempt=False):
    """构造一个全通过(full-pass)的 8 维审计报告，可按需注入失败/未核验/不适用。"""
    dims = {}
    for k in ea.DIM_KEYS:
        dims[k] = {"passed": True, "evaluated": True, "na": False, "metrics": {}}
    if design_exempt:
        # design 线：⑦ plate 合规不适用
        dims[ea.PLATE_DIM]["na"] = True
        dims[ea.PLATE_DIM]["evaluated"] = False
    if fail_dim is not None:
        dims[fail_dim]["passed"] = False
        dims[fail_dim]["issues"] = ["injected failure"]
    if not_eval is not None:
        dims[not_eval]["evaluated"] = False
    if na_dims:
        for d in na_dims:
            dims[d]["na"] = True
            dims[d]["evaluated"] = False
    all_pass = (fail_dim is None)
    return {"passed": all_pass, "dims": dims}


class TestFinalizeStoresOriginal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ep = Path(self.tmp.name) / "episodes.jsonl"
        self.idx = Path(self.tmp.name) / "index.pkl"
        self._old = {k: os.environ.get(k) for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_INDEX_PATH")}
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(self.ep)
        os.environ["ADAPTIVE_INDEX_PATH"] = str(self.idx)
        self.addCleanup(self._restore)
        # 清空残余索引
        if self.idx.exists():
            self.idx.unlink()

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _archive(self, output_mode):
        emb = np.random.randn(128).astype(np.float32)
        return ea.archive_episode(
            material_family="金地屏风",
            category_ids=["water_ripples"],
            fingerprint={"embedding": emb, "image_shape": (10, 10)},
            image_path="x.jpg",
            output_mode=output_mode,
        )

    def test_finalize_preserves_full_report(self):
        """finalize 必须保留原始完整 8 维报告(audit_full)，不能丢成摘要。"""
        ep_id = self._archive("plate")
        report = _make_report()
        dec = ea.finalize_episode(ep_id, audit_report=report, output_mode="plate")
        self.assertTrue(dec["admitted"])

        rec = ea.load_episode_by_id(ep_id, str(self.ep))
        self.assertIsNotNone(rec["audit_full"], "原始完整报告未保留")
        self.assertEqual(rec["audit_full"]["dims"], report["dims"])
        # 兼容摘要仍在
        self.assertIsNotNone(rec["audit_8d"])

    def test_plate_full_pass_admitted(self):
        ep_id = self._archive("plate")
        dec = ea.finalize_episode(ep_id, audit_report=_make_report(), output_mode="plate")
        self.assertTrue(dec["admitted"])
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertIn(ep_id, ix.fingerprints)
        self.assertTrue(ix.audit_status[ep_id])

    def test_plate_plate_dim_failed_skip(self):
        """plate 线 ⑦ 失败 → 不准入（skip）。"""
        ep_id = self._archive("plate")
        dec = ea.finalize_episode(ep_id, audit_report=_make_report(fail_dim=ea.PLATE_DIM),
                                  output_mode="plate")
        self.assertFalse(dec["admitted"])
        self.assertIn("failed", dec["reason"])
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertNotIn(ep_id, ix.fingerprints)

    def test_design_exempts_plate_dim(self):
        """design 线 ⑦ plate 合规不适用(na) → 其余全过即准入。"""
        ep_id = self._archive("design")
        dec = ea.finalize_episode(ep_id, audit_report=_make_report(design_exempt=True),
                                  output_mode="design")
        self.assertTrue(dec["admitted"])
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertIn(ep_id, ix.fingerprints)

    def test_missing_required_dim_skip(self):
        """缺必需维度（如 ④）→ 不准入。"""
        ep_id = self._archive("plate")
        report = _make_report()
        del report["dims"]["④ 内容承载"]
        dec = ea.finalize_episode(ep_id, audit_report=report, output_mode="plate")
        self.assertFalse(dec["admitted"])
        self.assertIn("missing", dec["reason"])

    def test_not_evaluated_dim_skip(self):
        """必需维度未真正核验(evaluated=False，如缺源图) → 不准入。"""
        ep_id = self._archive("plate")
        dec = ea.finalize_episode(ep_id, audit_report=_make_report(not_eval="④ 内容承载"),
                                  output_mode="plate")
        self.assertFalse(dec["admitted"])
        self.assertIn("not_evaluated", dec["reason"])

    def test_legacy_summary_only_skip(self):
        """仅旧式 4 字段摘要（无 audit_full）→ 不准入（不降低阈值凑通过）。"""
        ep_id = self._archive("plate")
        legacy = {"lost_ratio": 0.01, "rmse_lowfreq": 10.0, "plate_purity_ok": True,
                  "n_layers": 42}
        dec = ea.finalize_episode(ep_id, audit_report=legacy, output_mode="plate")
        self.assertFalse(dec["admitted"])
        self.assertIn("no_full_report", dec["reason"])
        rec = ea.load_episode_by_id(ep_id, str(self.ep))
        self.assertIsNone(rec.get("audit_full"))


class TestSyncAuthoritative(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ep = Path(self.tmp.name) / "episodes.jsonl"
        self.idx = Path(self.tmp.name) / "index.pkl"
        self._old = {k: os.environ.get(k) for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_INDEX_PATH")}
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(self.ep)
        os.environ["ADAPTIVE_INDEX_PATH"] = str(self.idx)
        self.addCleanup(self._restore)
        if self.idx.exists():
            self.idx.unlink()

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _archive(self, output_mode):
        emb = np.random.randn(128).astype(np.float32)
        return ea.archive_episode(
            material_family="金地屏风", category_ids=["x"],
            fingerprint={"embedding": emb, "image_shape": (4, 4)},
            image_path="x.jpg", output_mode=output_mode,
        )

    def test_sync_only_admits_strict_pass(self):
        # 4 个 episode：plate 全过 / plate ⑦失败 / design 豁免⑦全过 / 仅摘要
        ids = []
        ids.append(self._archive("plate"))
        ea.finalize_episode(ids[0], audit_report=_make_report(), output_mode="plate")
        ids.append(self._archive("plate"))
        ea.finalize_episode(ids[1], audit_report=_make_report(fail_dim=ea.PLATE_DIM), output_mode="plate")
        ids.append(self._archive("design"))
        ea.finalize_episode(ids[2], audit_report=_make_report(design_exempt=True), output_mode="design")
        ids.append(self._archive("plate"))
        ea.finalize_episode(ids[3], audit_report={"lost_ratio": 0.01}, output_mode="plate")

        n = ea.sync_index_from_log()
        self.assertEqual(n, 2, "只应收严格准入通过的 2 条")
        ix = EpisodeIndexer.load_or_create(self.idx)
        self.assertIn(ids[0], ix.fingerprints)
        self.assertIn(ids[2], ix.fingerprints)
        self.assertNotIn(ids[1], ix.fingerprints)
        self.assertNotIn(ids[3], ix.fingerprints)


if __name__ == "__main__":
    unittest.main()
