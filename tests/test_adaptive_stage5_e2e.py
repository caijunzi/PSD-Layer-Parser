"""Stage 5 端到端测试：类目树 → 主动学习 → 人审反馈（accept/rename/delete/merge）

覆盖链路：
    引擎检出 → identify_uncertain_categories(识别低置信)
             → request_human_feedback(写入待审队列)
             → 前端轮询 GET /api/adaptive/pending-feedbacks
             → 用户裁决 POST /api/adaptive/categories/{id}/feedback
             → apply_feedback_to_category(权重/类目真实变更)

⚠️ 隔离原则（重要，避免污染真实数据）：
- 数据库：复制真实 adaptive_semantics.db 到临时目录，通过 ADAPTIVE_DB_PATH 指向副本
- 待审队列：备份真实 pending_feedbacks.jsonl，测试结束恢复原状
（教训：早期冒烟直接打真实库，把 water_ripples 权重改了 ×1.1，事后才回滚）
"""
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.adaptive.active_learner import (
    identify_uncertain_categories,
    request_human_feedback,
    get_pending_feedbacks,
    mark_feedback_reviewed,
    apply_feedback_to_category,
)
from engine.adaptive.category_tree import CategoryTree

REAL_DB = Path("webui/data/adaptive_semantics.db")
REAL_PENDING = Path("webui/data/pending_feedbacks.jsonl")


class Stage5E2EBase(unittest.TestCase):
    """公共夹具：临时 DB 副本 + 待审队列隔离"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stage5e2e_"))

        # 1) 数据库副本（真实库只读，不改动）
        self.db_path = self.tmp / "adaptive_semantics.db"
        if REAL_DB.exists():
            shutil.copy2(REAL_DB, self.db_path)
        self._old_env = os.environ.get("ADAPTIVE_DB_PATH")
        os.environ["ADAPTIVE_DB_PATH"] = str(self.db_path)

        # 2) 待审队列：把真实文件临时挪走（若存在），测试结束还原
        self.pending_path = self.tmp / "pending_feedbacks.jsonl"
        self._real_pending_backup = None
        if REAL_PENDING.exists():
            self._real_pending_backup = self.tmp / "real_pending_backup.jsonl"
            shutil.move(str(REAL_PENDING), str(self._real_pending_backup))

    def tearDown(self):
        # 还原环境变量
        if self._old_env is None:
            os.environ.pop("ADAPTIVE_DB_PATH", None)
        else:
            os.environ["ADAPTIVE_DB_PATH"] = self._old_env

        # 还原真实待审队列
        if self._real_pending_backup and self._real_pending_backup.exists():
            shutil.move(str(self._real_pending_backup), str(REAL_PENDING))

        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 辅助 ----
    def query1(self, sql, args=()):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(sql, args).fetchone()
        finally:
            conn.close()

    def make_detection(self, items):
        return {"detections": items}


@unittest.skipUnless(REAL_DB.exists(), "真实自适应语义库不存在，跳过 e2e")
class TestStage5TreeE2E(Stage5E2EBase):
    """类目树：真实库结构下的祖先链与层级"""

    def test_tree_ancestors_chain(self):
        tree = CategoryTree(db_path=str(self.db_path))
        # distant_mountains（远山）→ mountains（山）→ landscape_painting（山水画）
        ancestors = tree.get_ancestors("distant_mountains")
        self.assertEqual(ancestors, ["mountains", "landscape_painting"])

    def test_tree_root_has_no_ancestors(self):
        tree = CategoryTree(db_path=str(self.db_path))
        self.assertEqual(tree.get_ancestors("landscape_painting"), [])

    def test_tree_built_count(self):
        row = self.query1(
            "SELECT COUNT(*) FROM categories WHERE parent_id IS NOT NULL"
        )
        self.assertGreaterEqual(row[0], 10, "应有 ≥10 个类目已挂到树上")


class TestStage5FeedbackE2E(Stage5E2EBase):
    """人审反馈全链路：识别 → 入队 → 裁决 → 真实变更"""

    def test_e2e_identify_to_pending_queue(self):
        """低置信检出 → 写入待审队列 → 轮询可读"""
        det = self.make_detection(
            [
                {"category_id": "water_ripples", "confidence": 0.62, "bbox": [0.1, 0.2, 0.3, 0.4]},
                {"category_id": "trees_vegetation", "confidence": 0.91, "bbox": [0, 0, 1, 1]},
            ]
        )
        uncertain = identify_uncertain_categories(det, confidence_threshold=0.7)
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(uncertain[0]["category_id"], "water_ripples")

        fid = request_human_feedback(
            uncertain_categories=uncertain,
            task_id="task_e2e_001",
            pending_path=str(self.pending_path),
        )
        self.assertTrue(fid.startswith("fb_"))

        pending = get_pending_feedbacks(pending_path=str(self.pending_path))
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["task_id"], "task_e2e_001")

    def test_e2e_accept_raises_weight(self):
        """accept：prompt 权重 ×1.1，n_accept +1"""
        before = self.query1(
            "SELECT weight, n_accept FROM category_prompts WHERE category_id = 'water_ripples' ORDER BY id LIMIT 1"
        )
        self.assertIsNotNone(before, "副本库应有 water_ripples 的 prompt")

        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="accept",
            db_path=str(self.db_path),
        )
        self.assertTrue(out["ok"], out.get("detail"))

        after = self.query1(
            "SELECT weight, n_accept FROM category_prompts WHERE category_id = 'water_ripples' ORDER BY id LIMIT 1"
        )
        self.assertAlmostEqual(after[0], before[0] * 1.1, places=5)
        self.assertEqual(after[1], before[1] + 1)

    def test_e2e_rename_updates_category(self):
        """rename：更新 name_zh / name_template"""
        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="rename",
            user_data={"new_name": "水面微波"},
            db_path=str(self.db_path),
        )
        self.assertTrue(out["ok"], out.get("detail"))

        row = self.query1(
            "SELECT name_zh, name_template FROM categories WHERE id = 'water_ripples'"
        )
        self.assertEqual(row[0], "水面微波")
        self.assertIn("水面微波", row[1])

    def test_e2e_rename_requires_new_name(self):
        """rename 缺 new_name → 明确失败（不静默）"""
        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="rename",
            user_data={},
            db_path=str(self.db_path),
        )
        self.assertFalse(out["ok"])
        self.assertIn("new_name", out["detail"])

    def test_e2e_delete_soft_deletes_and_hidden(self):
        """delete：软删（deleted_at 非空）且主查询不再返回"""
        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="delete",
            db_path=str(self.db_path),
        )
        self.assertTrue(out["ok"], out.get("detail"))

        row = self.query1(
            "SELECT deleted_at FROM categories WHERE id = 'water_ripples'"
        )
        self.assertIsNotNone(row[0], "deleted_at 应被置位")

        # 查询层过滤：db_manager 主查询带 deleted_at IS NULL
        conn = sqlite3.connect(self.db_path)
        try:
            alive = conn.execute(
                """
                SELECT COUNT(*) FROM categories c
                JOIN category_material_affinity a ON c.id = a.category_id
                WHERE c.id = 'water_ripples' AND c.deleted_at IS NULL
                """
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(alive, 0, "软删后主查询不应再返回该类目")

    def test_e2e_merge_migrates_prompts(self):
        """merge：源类目 prompts 迁到目标 + 源类目软删"""
        # 动态选取一个真实存在的目标类目（避免硬编码不存在的 id）
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id FROM categories WHERE id != 'water_ripples' AND deleted_at IS NULL ORDER BY id LIMIT 1"
            ).fetchall()
        finally:
            conn.close()
        self.assertTrue(rows, "副本库应有除 water_ripples 外的类目")
        target = rows[0][0]

        src_rows = self.query1(
            "SELECT COUNT(*) FROM category_prompts WHERE category_id = 'water_ripples'"
        )[0]
        tgt_before = self.query1(
            "SELECT COUNT(*) FROM category_prompts WHERE category_id = ?", (target,)
        )[0]

        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="merge",
            user_data={"target_category_id": target},
            db_path=str(self.db_path),
        )
        self.assertTrue(out["ok"], out.get("detail"))

        src_after = self.query1(
            "SELECT COUNT(*) FROM category_prompts WHERE category_id = 'water_ripples'"
        )[0]
        tgt_after = self.query1(
            "SELECT COUNT(*) FROM category_prompts WHERE category_id = ?", (target,)
        )[0]

        self.assertEqual(src_after, 0, "源类目不应残留 prompt")
        self.assertGreaterEqual(tgt_after, tgt_before, "目标类目 prompt 数应增加或持平")
        # 源类目软删
        self.assertIsNotNone(
            self.query1("SELECT deleted_at FROM categories WHERE id = 'water_ripples'")[0]
        )

    def test_e2e_merge_rejects_missing_target(self):
        """merge 缺目标 / 目标不存在 → 明确失败"""
        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="merge",
            user_data={},
            db_path=str(self.db_path),
        )
        self.assertFalse(out["ok"])

        out2 = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="merge",
            user_data={"target_category_id": "not_exist_category"},
            db_path=str(self.db_path),
        )
        self.assertFalse(out2["ok"])
        self.assertIn("不存在", out2["detail"])

    def test_e2e_merge_rejects_self(self):
        out = apply_feedback_to_category(
            category_id="water_ripples",
            user_action="merge",
            user_data={"target_category_id": "water_ripples"},
            db_path=str(self.db_path),
        )
        self.assertFalse(out["ok"])
        self.assertIn("相同", out["detail"])


class TestStage5FeedbackHttpE2E(Stage5E2EBase):
    """HTTP 层：轮询端点 + 反馈提交端点（真实 TestClient）"""

    def test_e2e_polling_and_submit(self):
        from fastapi.testclient import TestClient
        from webui.backend.main import app

        # 造一条待审（写入隔离的 pending 文件）
        fid = request_human_feedback(
            uncertain_categories=[
                {"category_id": "water_ripples", "confidence": 0.55, "bbox": [0, 0, 1, 1]}
            ],
            task_id="task_e2e_http",
            pending_path=str(self.pending_path),
        )

        # 轮询端点读的是模块默认路径，这里直接把隔离文件复制到默认位置，结束后删除
        # （避免修改被测端点签名）
        wrote_default = False
        try:
            REAL_PENDING.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.pending_path, REAL_PENDING)
            wrote_default = True

            client = TestClient(app)
            r1 = client.get("/api/adaptive/pending-feedbacks?limit=1")
            self.assertEqual(r1.status_code, 200)
            pf = r1.json().get("pending_feedbacks", [])
            self.assertEqual(len(pf), 1)
            self.assertEqual(pf[0]["feedback_request_id"], fid)

            r2 = client.post(
                "/api/adaptive/categories/water_ripples/feedback",
                data={"feedback_request_id": fid, "user_action": "accept"},
            )
            self.assertEqual(r2.status_code, 200, r2.text)
            body = r2.json()
            self.assertEqual(body["status"], "ok")
            self.assertIn("detail", body)

            # 复查：已 reviewed，队列清空
            r3 = client.get("/api/adaptive/pending-feedbacks?limit=1")
            self.assertEqual(len(r3.json().get("pending_feedbacks", [])), 0)
        finally:
            if wrote_default and REAL_PENDING.exists():
                REAL_PENDING.unlink()

    def test_e2e_submit_rename_via_http(self):
        from fastapi.testclient import TestClient
        from webui.backend.main import app

        fid = request_human_feedback(
            uncertain_categories=[{"category_id": "water_ripples", "confidence": 0.5}],
            task_id="task_e2e_http_rename",
            pending_path=str(self.pending_path),
        )
        wrote_default = False
        try:
            REAL_PENDING.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.pending_path, REAL_PENDING)
            wrote_default = True

            client = TestClient(app)
            r = client.post(
                "/api/adaptive/categories/water_ripples/feedback",
                data={
                    "feedback_request_id": fid,
                    "user_action": "rename",
                    "user_data": json.dumps({"new_name": "水波微澜"}),
                },
            )
            self.assertEqual(r.status_code, 200, r.text)

            row = self.query1(
                "SELECT name_zh FROM categories WHERE id = 'water_ripples'"
            )
            self.assertEqual(row[0], "水波微澜")
        finally:
            if wrote_default and REAL_PENDING.exists():
                REAL_PENDING.unlink()

    def test_e2e_submit_invalid_action_returns_400(self):
        from fastapi.testclient import TestClient
        from webui.backend.main import app

        client = TestClient(app)
        r = client.post(
            "/api/adaptive/categories/water_ripples/feedback",
            data={"feedback_request_id": "fb_x", "user_action": "bomb"},
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
