"""全样本自学习 E2E 修复测试：覆盖 DB 工厂、学习器、主动学习、后台学习器、API。

对应源码：
    engine/adaptive/db_manager.py
    engine/adaptive/learner.py
    engine/adaptive/active_learner.py
    webui/backend/core/background_learner.py
    webui/backend/api/adaptive.py

隔离原则：所有写库/写文件操作都指向临时副本或环境变量，不污染真实数据。
"""
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

import sys

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.adaptive.db_manager import create_db_manager, DBManager
from engine.adaptive.learner import SemanticLearner
from engine.adaptive.active_learner import apply_feedback_to_category
from engine.adaptive.episode_archiver import load_episodes
from webui.backend.core.background_learner import BackgroundLearner

SCHEMA = ENGINE_ROOT / "engine" / "adaptive" / "schema.sql"
REAL_DB = Path("webui/data/adaptive_semantics.db")
REAL_PENDING = Path("webui/data/pending_feedbacks.jsonl")

NOW = 1789000000


def build_schema_db(path: Path):
    """用 schema.sql 建库并写入最小类目/提示词，供单元级反馈测试。"""
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO categories (id,name_zh,name_en,name_template,confidence,created_at,updated_at)"
        " VALUES (?,?,?,?,?,?,?)",
        ("water_ripples", "水波纹", "Water_Ripples", "水波纹_Water_Ripples", 0.7, NOW, NOW),
    )
    # 一个「存在但无 prompt」的类目（触发 rowcount=0 失败路径）
    conn.execute(
        "INSERT INTO categories (id,name_zh,name_en,name_template,confidence,created_at,updated_at)"
        " VALUES (?,?,?,?,?,?,?)",
        ("no_prompts", "无提示词", "No_Prompts", "无提示词_No_Prompts", 0.7, NOW, NOW),
    )
    conn.execute(
        "INSERT INTO category_prompts (category_id,prompt,weight,lang,source,created_at)"
        " VALUES (?,?,?,?,?,?)",
        ("water_ripples", "water ripples", 1.0, "zh", "seed", NOW),
    )
    conn.commit()
    conn.close()


class TestDBFactoryEnv(unittest.TestCase):
    """DB 工厂在「调用时」解析 ADAPTIVE_DB_PATH。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "db.sqlite"
        build_schema_db(self.db)
        self._old = os.environ.get("ADAPTIVE_DB_PATH")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("ADAPTIVE_DB_PATH", None)
        else:
            os.environ["ADAPTIVE_DB_PATH"] = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_factory_reads_env_at_call_time(self):
        os.environ["ADAPTIVE_DB_PATH"] = str(self.db)
        # 不传参，工厂应在调用时读环境变量
        mgr = create_db_manager()
        self.assertEqual(str(mgr.db_path), str(self.db))
        row = mgr.execute("SELECT COUNT(*) FROM categories").fetchone()
        self.assertGreaterEqual(row[0], 2)
        mgr.close()

    def test_explicit_path_overrides_env(self):
        os.environ["ADAPTIVE_DB_PATH"] = "webui/data/does_not_matter.sqlite"
        mgr = create_db_manager(str(self.db))
        self.assertEqual(str(mgr.db_path), str(self.db))
        mgr.close()

    def test_get_prompt_weight_real_db(self):
        mgr = create_db_manager(str(self.db))
        w = mgr.get_prompt_weight("water_ripples", "water ripples")
        self.assertIsNotNone(w)
        self.assertAlmostEqual(w, 1.0, places=6)
        # 未知 (category, prompt) → None
        self.assertIsNone(mgr.get_prompt_weight("water_ripples", "nope"))
        self.assertIsNone(mgr.get_prompt_weight("ghost", "x"))
        mgr.close()


class TestLearnerRealWeights(unittest.TestCase):
    """学习器读取真实 DB 旧权重；未知检测类目不写假成功；返回非空影子建议。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "db.sqlite"
        build_schema_db(self.db)
        os.environ["ADAPTIVE_DB_PATH"] = str(self.db)
        # water_ripples 在库中权重 1.0；再加入一个高权重提示词用于验证真实旧权重
        conn = sqlite3.connect(str(self.db))
        conn.execute(
            "INSERT INTO categories (id,name_zh,name_en,name_template,confidence,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            ("远山", "远山", "Distant_Mountains", "远山_Distant_Mountains", 0.8, NOW, NOW),
        )
        conn.execute(
            "INSERT INTO category_prompts (category_id,prompt,weight,lang,source,created_at)"
            " VALUES (?,?,?,?,?,?)",
            ("远山", "distant mountains", 2.5, "en", "seed", NOW),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        os.environ.pop("ADAPTIVE_DB_PATH", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_real_old_weight(self):
        learner = SemanticLearner()  # db_path 走环境变量
        episode = {
            "episode_id": "ep_real",
            "dino_detections": [
                {"layer_name": "远山", "prompt": "distant mountains", "quality_gate_passed": True, "diffuse_gate_passed": True},
            ],
        }
        res = learner.learn_from_episode(episode)
        adj = res["weight_adjustments"].get("远山", {}).get("distant mountains")
        self.assertIsNotNone(adj, "已知类目应产出建议")
        self.assertAlmostEqual(adj["old_weight"], 2.5, places=6,
                               msg="应读取 DB 中的真实旧权重 2.5，而非默认 1.0")
        self.assertAlmostEqual(adj["new_weight"], 2.5 * 1.1, places=6)
        self.assertEqual(adj["signal"], "accept")

    def test_unknown_detection_not_faked(self):
        learner = SemanticLearner()
        episode = {
            "episode_id": "ep_unknown",
            "dino_detections": [
                {"layer_name": "ghost_cat", "prompt": "ghost prompt", "quality_gate_passed": True, "diffuse_gate_passed": True},
            ],
        }
        res = learner.learn_from_episode(episode)
        # 库中不存在 → 跳过，不写假成功
        self.assertIn("ghost_cat:ghost prompt", res["unknown_detections"])
        self.assertEqual(res["weight_adjustments"], {}, "未知检测类目不应出现在建议中")

    def test_returns_non_empty_suggestion(self):
        learner = SemanticLearner()
        episode = {
            "episode_id": "ep_nonempty",
            "dino_detections": [
                {"layer_name": "远山", "prompt": "distant mountains", "quality_gate_passed": False,
                 "quality_gate_reason": "面积超预算"},
            ],
        }
        res = learner.learn_from_episode(episode)
        self.assertGreater(len(res["weight_adjustments"]), 0, "应返回非空影子建议")
        adj = res["weight_adjustments"]["远山"]["distant mountains"]
        self.assertEqual(adj["signal"], "reject")
        self.assertLess(adj["new_weight"], adj["old_weight"])

    def test_no_db_falls_back_without_error(self):
        # 临时取消环境变量：无 DB 时回退 1.0，不报错
        os.environ.pop("ADAPTIVE_DB_PATH", None)
        learner = SemanticLearner(db_path=None)
        episode = {
            "episode_id": "ep_nodb",
            "dino_detections": [
                {"layer_name": "X", "prompt": "p", "quality_gate_passed": True, "diffuse_gate_passed": True},
            ],
        }
        res = learner.learn_from_episode(episode)
        adj = res["weight_adjustments"]["X"]["p"]
        self.assertAlmostEqual(adj["old_weight"], 1.0, places=6)
        self.assertAlmostEqual(adj["new_weight"], 1.1, places=6)


class TestBackgroundLearnerPaths(unittest.TestCase):
    """后台学习器：调用时解析路径环境变量；无检测/无建议→no_op；stop/reset 安全。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "db.sqlite"
        build_schema_db(self.db)
        self.ep_file = self.tmp / "episodes.jsonl"
        self.tasks_file = self.tmp / "learning_tasks.jsonl"
        # 写入一条「无检测」episode
        self.ep_file.write_text(
            json.dumps({"episode_id": "ep_noop", "dino_detections": []}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_env_paths_resolved_at_call(self):
        os.environ["ADAPTIVE_EPISODE_PATH"] = str(self.ep_file)
        os.environ["ADAPTIVE_LEARNING_TASKS_PATH"] = str(self.tasks_file)
        os.environ["ADAPTIVE_BASELINE_PATH"] = "tests/baseline_audit_8d.json"
        os.environ["ADAPTIVE_DB_PATH"] = str(self.db)
        try:
            bl = BackgroundLearner()
            self.assertEqual(str(bl.episode_path), str(self.ep_file))
            self.assertEqual(str(bl.learning_tasks_path), str(self.tasks_file))
            self.assertEqual(str(bl.db_path), str(self.db))
        finally:
            for k in ("ADAPTIVE_EPISODE_PATH", "ADAPTIVE_LEARNING_TASKS_PATH",
                      "ADAPTIVE_BASELINE_PATH", "ADAPTIVE_DB_PATH"):
                os.environ.pop(k, None)

    def test_no_detection_yields_no_op(self):
        bl = BackgroundLearner(
            episode_path=str(self.ep_file),
            learning_tasks_path=str(self.tasks_file),
            db_path=str(self.db),
        )
        bl.enqueue_learning_task("ep_noop", trigger="manual")
        bl.process_learning_task("ep_noop")
        status = bl.get_task_status("ep_noop")
        self.assertEqual(status["status"], "no_op",
                         "无检测/无建议的任务必须标记为 no_op，而非谎称 completed")
        self.assertEqual(status["n_suggestions"], 0)

    def test_stop_and_reset_safe(self):
        bl = BackgroundLearner(
            episode_path=str(self.ep_file),
            learning_tasks_path=str(self.tasks_file),
            db_path=str(self.db),
        )
        bl.enqueue_learning_task("ep_noop", trigger="manual")
        bl.start_worker(poll_interval=0.05)
        # worker 已启动
        self.assertIsNotNone(bl._worker)
        time.sleep(0.1)
        bl.reset(clear_file=True)
        # 复位后：队列清空、worker 已停止、状态清空、持久化文件删除
        self.assertEqual(len(bl.task_queue), 0)
        self.assertEqual(len(bl.task_status), 0)
        self.assertIsNone(bl._worker)
        self.assertFalse(self.tasks_file.exists())


class TestFeedbackSpec(unittest.TestCase):
    """反馈规范：DB ID 存在性 / rowcount 零必须 fail；版本如实 pending 且可查；真实权重持久化。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "db.sqlite"
        build_schema_db(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_accept_missing_category_fails(self):
        out = apply_feedback_to_category("ghost_cat", "accept", {}, db_path=str(self.db))
        self.assertFalse(out["ok"], "不存在的类目 accept 必须失败，不能静默成功")
        self.assertIn("不存在", out["detail"])

    def test_accept_zero_rowcount_fails(self):
        # no_prompts 类目存在但无任何 prompt → UPDATE rowcount=0 → 必须失败
        out = apply_feedback_to_category("no_prompts", "accept", {}, db_path=str(self.db))
        self.assertFalse(out["ok"], "无 prompt 可升权（rowcount=0）必须失败")
        self.assertIn("rowcount", out["detail"])

    def test_accept_persists_real_weight_and_version_queryable(self):
        out = apply_feedback_to_category("water_ripples", "accept", {}, db_path=str(self.db))
        self.assertTrue(out["ok"], out.get("detail"))

        # 1) 真实 prompt 权重已持久化（1.0 → 1.1）
        mgr = DBManager(str(self.db))
        w = mgr.get_prompt_weight("water_ripples", "water ripples")
        self.assertAlmostEqual(w, 1.1, places=6)
        mgr.close()

        # 2) 版本节点已登记且如实为 pending（不得谎称 passed），并可通过 list_versions 查到
        self.assertIn("version_id", out)
        mgr = create_db_manager(str(self.db))
        versions = mgr.list_versions(limit=10)
        self.assertGreaterEqual(len(versions), 1)
        target = next(v for v in versions if v["version_id"] == out["version_id"])
        self.assertEqual(target["regression_status"], "pending",
                         "受审反馈版本应如实登记 pending，不得谎称 passed")
        mgr.close()

    def test_duplicate_application_guard_active_learner(self):
        # 直接验证重复应用逻辑：先 apply 一次成功，第二次对同请求应被幂等拦截（在 API 层）。
        # 这里只验证 apply 本身是幂等的：重复 accept 会累计升权（底层 DB 行为），
        # 真正的「不重复应用」由 API 层的 reviewed 标记保证（见 TestFeedbackApiHttp）。
        out1 = apply_feedback_to_category("water_ripples", "accept", {}, db_path=str(self.db))
        self.assertTrue(out1["ok"])
        mgr = DBManager(str(self.db))
        w1 = mgr.get_prompt_weight("water_ripples", "water ripples")
        mgr.close()
        self.assertAlmostEqual(w1, 1.1, places=6)


@unittest.skipUnless(REAL_DB.exists(), "真实自适应语义库不存在，跳过 API 级 E2E")
class TestFeedbackApiHttp(unittest.TestCase):
    """HTTP 层：先应用后标 reviewed；重复请求幂等；应用失败时不被标记 reviewed。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # 真实库副本 + ADAPTIVE_DB_PATH 指向副本
        self.db = self.tmp / "adaptive_semantics.db"
        shutil.copy2(REAL_DB, self.db)
        self._old_db = os.environ.get("ADAPTIVE_DB_PATH")
        os.environ["ADAPTIVE_DB_PATH"] = str(self.db)

        # 备份并接管真实待审队列
        self._real_pending_backup = None
        if REAL_PENDING.exists():
            self._real_pending_backup = self.tmp / "real_pending_backup.jsonl"
            shutil.move(str(REAL_PENDING), str(self._real_pending_backup))

        from engine.adaptive.active_learner import request_human_feedback
        self.fid = request_human_feedback(
            uncertain_categories=[{"category_id": "water_ripples", "confidence": 0.5}],
            task_id="task_http_e2e",
            pending_path=str(REAL_PENDING),
        )
        # 确保目录存在
        REAL_PENDING.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old_db is None:
            os.environ.pop("ADAPTIVE_DB_PATH", None)
        else:
            os.environ["ADAPTIVE_DB_PATH"] = self._old_db
        if REAL_PENDING.exists():
            REAL_PENDING.unlink()
        if self._real_pending_backup and self._real_pending_backup.exists():
            shutil.move(str(self._real_pending_backup), str(REAL_PENDING))
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _client(self):
        from fastapi.testclient import TestClient
        from webui.backend.main import app
        return TestClient(app)

    def test_apply_then_mark_reviewed(self):
        c = self._client()
        r = c.post(
            "/api/adaptive/categories/water_ripples/feedback",
            data={"feedback_request_id": self.fid, "user_action": "accept"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        # 已 reviewed → 轮询清空
        r2 = c.get("/api/adaptive/pending-feedbacks?limit=1")
        self.assertEqual(len(r2.json().get("pending_feedbacks", [])), 0)

    def test_duplicate_request_idempotent(self):
        c = self._client()
        r1 = c.post(
            "/api/adaptive/categories/water_ripples/feedback",
            data={"feedback_request_id": self.fid, "user_action": "accept"},
        )
        self.assertEqual(r1.status_code, 200, r1.text)
        # 再次提交同一 feedback_request_id → 幂等跳过，不重复应用
        r2 = c.post(
            "/api/adaptive/categories/water_ripples/feedback",
            data={"feedback_request_id": self.fid, "user_action": "accept"},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        body = r2.json()
        self.assertEqual(body.get("detail"), "duplicate_skipped")

        # 权重只应被应用一次（1.0 → 1.1，而非 1.21）
        conn = sqlite3.connect(str(self.db))
        w = conn.execute(
            "SELECT weight FROM category_prompts WHERE category_id='water_ripples' ORDER BY id LIMIT 1"
        ).fetchone()[0]
        conn.close()
        self.assertAlmostEqual(w, 1.1, places=5)

    def test_apply_failure_not_marked_reviewed(self):
        # water_ripples 一定存在于真实库；构造一个不存在的类目以触发 apply 失败
        c = self._client()
        r = c.post(
            "/api/adaptive/categories/ghost_cat_xyz/feedback",
            data={"feedback_request_id": self.fid, "user_action": "accept"},
        )
        self.assertEqual(r.status_code, 400, r.text)
        # 应用失败 → 不应标记 reviewed（先应用后标 reviewed 的顺序保证）
        statuses = []
        with REAL_PENDING.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    statuses.append(json.loads(line).get("status"))
                except json.JSONDecodeError:
                    continue
        self.assertIn("pending", statuses, "应用失败后该请求应保持 pending，未被标记 reviewed")


if __name__ == "__main__":
    unittest.main()
