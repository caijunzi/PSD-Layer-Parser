"""单元测试：类目树管理（Stage 5.1）

测试 CategoryTree 的核心功能：
- 插入父子关系 + DFS 无环检测
- 查询祖先链
- inherit_priors 骨架（当前标记 TODO）
"""
import unittest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import sys

# 添加 engine 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from engine.adaptive.category_tree import CategoryTree


class TestCategoryTreeInsert(unittest.TestCase):
    """测试插入父子关系与无环检测"""
    
    def setUp(self):
        # 创建临时数据库
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_tree.db"
        
        # 初始化测试数据库（3 个类目）
        conn = sqlite3.connect(self.db_path)
        now = int(datetime.now(timezone.utc).timestamp())
        conn.execute(
            """
            CREATE TABLE categories (
                id TEXT PRIMARY KEY,
                name_zh TEXT NOT NULL,
                name_en TEXT,
                parent_id TEXT,
                path TEXT,
                name_template TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        # 插入 3 个测试类目（扁平）
        for cat_id, name_zh in [("cat_a", "类目A"), ("cat_b", "类目B"), ("cat_c", "类目C")]:
            conn.execute(
                "INSERT INTO categories (id, name_zh, name_template, parent_id, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cat_id, name_zh, f"{name_zh}_Template", None, "/", now, now)
            )
        conn.commit()
        conn.close()
        
        self.tree = CategoryTree(db_path=str(self.db_path))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_insert_simple_parent_child(self):
        """插入简单父子关系"""
        # cat_b → parent=cat_a
        result = self.tree.insert("cat_b", "cat_a")
        self.assertTrue(result)
        
        # 验证 parent_id 与 path
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT parent_id, path FROM categories WHERE id = ?", ("cat_b",)
        ).fetchone()
        conn.close()
        
        self.assertEqual(row[0], "cat_a")
        self.assertEqual(row[1], "/cat_b/")  # path = parent_path + cat_id + "/"
    
    def test_insert_cycle_detection(self):
        """检测循环依赖：parent_id 是 category_id 的后代"""
        # 先建立 cat_b → cat_a（cat_a 是 cat_b 的父）
        self.tree.insert("cat_b", "cat_a")
        
        # 尝试 cat_a → cat_b（会产生循环）
        with self.assertRaises(ValueError) as cm:
            self.tree.insert("cat_a", "cat_b")
        
        self.assertIn("循环依赖", str(cm.exception))
        self.assertIn("cat_b", str(cm.exception))
        self.assertIn("cat_a", str(cm.exception))
    
    def test_insert_three_level_chain(self):
        """插入三层链：cat_c → cat_b → cat_a"""
        # 先建立底层：cat_b → cat_a（cat_a 是根）
        self.tree.insert("cat_b", "cat_a")
        # 再建立上层：cat_c → cat_b
        self.tree.insert("cat_c", "cat_b")
        
        # 验证 path
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, path FROM categories WHERE id IN ('cat_a', 'cat_b', 'cat_c') ORDER BY id"
        ).fetchall()
        conn.close()
        
        # cat_a: 根（path=/）
        # cat_b: parent=cat_a → path=/cat_b/
        # cat_c: parent=cat_b → path=/cat_b/cat_c/
        paths = {row[0]: row[1] for row in rows}
        self.assertEqual(paths["cat_a"], "/")
        self.assertEqual(paths["cat_b"], "/cat_b/")
        self.assertEqual(paths["cat_c"], "/cat_b/cat_c/")


class TestCategoryTreeAncestors(unittest.TestCase):
    """测试查询祖先链"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_tree.db"
        
        # 初始化测试数据库（3 层树：root → mid → leaf）
        conn = sqlite3.connect(self.db_path)
        now = int(datetime.now(timezone.utc).timestamp())
        conn.execute(
            """
            CREATE TABLE categories (
                id TEXT PRIMARY KEY,
                name_zh TEXT NOT NULL,
                name_en TEXT,
                parent_id TEXT,
                path TEXT,
                name_template TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        # 插入三层树
        for cat_id, parent_id, path in [
            ("root", None, "/"),
            ("mid", "root", "/mid/"),
            ("leaf", "mid", "/mid/leaf/"),
        ]:
            conn.execute(
                "INSERT INTO categories (id, name_zh, name_template, parent_id, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cat_id, cat_id, f"{cat_id}_Template", parent_id, path, now, now)
            )
        conn.commit()
        conn.close()
        
        self.tree = CategoryTree(db_path=str(self.db_path))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_ancestors_leaf(self):
        """查询叶节点的祖先链"""
        ancestors = self.tree.get_ancestors("leaf")
        self.assertEqual(ancestors, ["mid", "root"])
    
    def test_get_ancestors_mid(self):
        """查询中间节点的祖先链"""
        ancestors = self.tree.get_ancestors("mid")
        self.assertEqual(ancestors, ["root"])
    
    def test_get_ancestors_root(self):
        """查询根节点的祖先链（应为空）"""
        ancestors = self.tree.get_ancestors("root")
        self.assertEqual(ancestors, [])


class TestCategoryTreeInheritPriors(unittest.TestCase):
    """测试 inherit_priors（priors 表为空时优雅跳过）"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_tree.db"
        
        # 初始化最小数据库
        conn = sqlite3.connect(self.db_path)
        now = int(datetime.now(timezone.utc).timestamp())
        conn.execute(
            """
            CREATE TABLE categories (
                id TEXT PRIMARY KEY,
                name_zh TEXT NOT NULL,
                name_en TEXT,
                parent_id TEXT,
                path TEXT,
                name_template TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO categories (id, name_zh, name_template, parent_id, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("parent", "父类目", "父类目_Template", None, "/", now, now)
        )
        conn.execute(
            "INSERT INTO categories (id, name_zh, name_template, parent_id, path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("child", "子类目", "子类目_Template", "parent", "/child/", now, now)
        )
        conn.commit()
        conn.close()
        
        self.tree = CategoryTree(db_path=str(self.db_path))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_inherit_priors_graceful_when_no_priors(self):
        """priors 表不存在 / 无先验数据时应优雅跳过（返回状态字典），不抛异常"""
        result = self.tree.inherit_priors("child", "parent")

        self.assertIsInstance(result, dict)
        self.assertFalse(result["ok"])
        self.assertFalse(result["inherited"])
        # 说明原因应包含"跳过"/"无先验"等信息
        reason = result["reason"].lower()
        self.assertTrue(
            any(keyword in reason for keyword in ["跳过", "无先验", "skip", "不存在", "category_priors"]),
            f"跳过原因应可读，实际为：{result['reason']}"
        )

    def test_inherit_priors_copies_when_prior_exists(self):
        """父类目存在先验时应复制到子类目（n_samples=0 标记继承来源）"""
        conn = sqlite3.connect(self.db_path)
        now = int(datetime.now(timezone.utc).timestamp())
        conn.execute(
            """
            CREATE TABLE category_priors (
                category_id TEXT PRIMARY KEY,
                area_budget_min REAL, area_budget_max REAL,
                elongation_mean REAL, elongation_std REAL,
                compactness_mean REAL, compactness_std REAL,
                n_components_mode INTEGER, n_samples INTEGER DEFAULT 0,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO category_priors (category_id, area_budget_min, area_budget_max, "
            "elongation_mean, elongation_std, compactness_mean, compactness_std, "
            "n_components_mode, n_samples, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("parent", 0.01, 0.2, 1.5, 0.3, 0.6, 0.1, 2, 10, now),
        )
        conn.commit()
        conn.close()

        result = self.tree.inherit_priors("child", "parent")
        self.assertTrue(result["ok"])
        self.assertTrue(result["inherited"])

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT area_budget_min, area_budget_max, n_samples FROM category_priors WHERE category_id = ?",
            ("child",),
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0.01)
        self.assertEqual(row[1], 0.2)
        self.assertEqual(row[2], 0)  # 继承来源标记

    def test_get_ancestors_cycle_guard(self):
        """祖先链存在自环时应截断而非死循环"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE categories SET parent_id = ? WHERE id = ?", ("child", "child"))
        conn.commit()
        conn.close()

        ancestors = self.tree.get_ancestors("child")  # 不应挂死
        self.assertIsInstance(ancestors, list)


if __name__ == "__main__":
    unittest.main()
