"""
类目选择器与数据库管理 - 单元测试

测试：
1. 数据库管理器基本功能
2. 版本管理（激活/回滚/GC）
3. 类目选择器三种模式（locked/auto/hybrid）
4. 核心类目必出
5. 性能要求（<0.5s）
"""
import unittest
import sqlite3
import tempfile
import shutil
from pathlib import Path
import time

# 导入待测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.adaptive.db_manager import DBManager, create_db_manager
from engine.adaptive.category_selector import select_categories, merge_semantic_classes
from engine.adaptive.fingerprint import extract_fingerprint
from engine.adaptive.material_classifier import classify_material_family
import numpy as np


class TestDBManager(unittest.TestCase):
    """数据库管理器测试"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试数据库（复制真实数据库）"""
        cls.test_db_dir = Path(tempfile.mkdtemp())
        cls.real_db = Path("webui/data/adaptive_semantics.db")
        
        if cls.real_db.exists():
            cls.test_db = cls.test_db_dir / "test_adaptive.db"
            shutil.copy(cls.real_db, cls.test_db)
        else:
            raise FileNotFoundError(f"真实数据库不存在：{cls.real_db}")
    
    @classmethod
    def tearDownClass(cls):
        """清理测试数据库"""
        shutil.rmtree(cls.test_db_dir)
    
    def test_connection(self):
        """测试数据库连接"""
        db = DBManager(str(self.test_db))
        
        # 执行简单查询
        cur = db.execute("SELECT COUNT(*) FROM categories")
        count = cur.fetchone()[0]
        
        self.assertGreater(count, 0, "类目表应有数据")
        db.close()
    
    def test_get_active_version(self):
        """测试获取活跃版本"""
        db = DBManager(str(self.test_db))
        version = db.get_active_version()
        
        self.assertIsNotNone(version, "应有活跃版本")
        self.assertEqual(version, "v0-seed", "初始版本应为 v0-seed")
        db.close()
    
    def test_list_versions(self):
        """测试列出版本历史"""
        db = DBManager(str(self.test_db))
        versions = db.list_versions(limit=10)
        
        self.assertGreater(len(versions), 0, "应有版本历史")
        self.assertIn("version_id", versions[0])
        self.assertIn("changeset", versions[0])
        db.close()
    
    def test_get_categories_by_material(self):
        """测试按材质查询类目"""
        db = DBManager(str(self.test_db))
        
        # 查询宣纸水墨材质的类目
        cats = db.get_categories_by_material("宣纸水墨", limit=10, min_affinity=0.5)
        
        self.assertGreater(len(cats), 0, "宣纸水墨应有匹配类目")
        self.assertIn("category_id", cats[0])
        self.assertIn("affinity", cats[0])
        
        # 验证按亲和度降序
        if len(cats) > 1:
            self.assertGreaterEqual(cats[0]["affinity"], cats[1]["affinity"])
        
        db.close()
    
    def test_get_category_prompts(self):
        """测试获取类目提示词"""
        db = DBManager(str(self.test_db))
        
        # 查询印章类目的提示词
        prompts = db.get_category_prompts("seal", min_weight=0.3)
        
        self.assertGreater(len(prompts), 0, "印章类目应有提示词")
        self.assertIn("prompt", prompts[0])
        self.assertIn("weight", prompts[0])
        
        # 验证按权重降序
        if len(prompts) > 1:
            self.assertGreaterEqual(prompts[0]["weight"], prompts[1]["weight"])
        
        db.close()


class TestCategorySelector(unittest.TestCase):
    """类目选择器测试"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试环境"""
        cls.test_db_dir = Path(tempfile.mkdtemp())
        cls.real_db = Path("webui/data/adaptive_semantics.db")
        
        if cls.real_db.exists():
            cls.test_db = cls.test_db_dir / "test_adaptive.db"
            shutil.copy(cls.real_db, cls.test_db)
        else:
            raise FileNotFoundError(f"真实数据库不存在：{cls.real_db}")
        
        # 准备测试指纹（模拟宣纸水墨图）
        test_img = np.ones((1440, 2880, 3), dtype=np.uint8)
        test_img[:, :] = [245, 245, 245]  # 浅灰（纸白）
        cls.fingerprint = extract_fingerprint(test_img)
        cls.material_family, _ = classify_material_family(cls.fingerprint)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.test_db_dir)
    
    def test_select_locked_mode(self):
        """测试 locked 模式（直接使用 preset）"""
        preset_ids = ["seal", "calligraphy", "mountains_cliffs"]
        
        categories = select_categories(
            self.fingerprint,
            self.material_family,
            str(self.test_db),
            mode="locked",
            preset_categories=preset_ids
        )
        
        # 验证返回类目与 preset 一致
        returned_ids = [cat["category_id"] for cat in categories]
        self.assertEqual(set(returned_ids), set(preset_ids))
    
    def test_select_auto_mode(self):
        """测试 auto 模式（完全由材质亲和度决定）"""
        categories = select_categories(
            self.fingerprint,
            self.material_family,
            str(self.test_db),
            mode="auto"
        )
        
        # 验证返回类目数量合理
        self.assertGreater(len(categories), 0, "auto 模式应返回类目")
        self.assertLessEqual(len(categories), 50, "auto 模式不应超过 50 个类目")
        
        # 验证核心类目必出
        returned_ids = [cat["category_id"] for cat in categories]
        core_cats = {"seal", "calligraphy", "repair_marks"}
        self.assertTrue(core_cats.issubset(set(returned_ids)), "核心类目必须出现")
    
    def test_select_hybrid_mode(self):
        """测试 hybrid 模式（preset + 自动补充）"""
        preset_ids = ["fauna_birds", "architecture"]  # 2 个预设
        
        categories = select_categories(
            self.fingerprint,
            self.material_family,
            str(self.test_db),
            mode="hybrid",
            preset_categories=preset_ids
        )
        
        returned_ids = [cat["category_id"] for cat in categories]
        
        # 验证包含 preset
        self.assertTrue(set(preset_ids).issubset(set(returned_ids)), "hybrid 模式应包含 preset")
        
        # 验证有自动补充
        self.assertGreater(len(returned_ids), len(preset_ids), "hybrid 模式应补充类目")
        
        # 验证核心类目必出
        core_cats = {"seal", "calligraphy", "repair_marks"}
        self.assertTrue(core_cats.issubset(set(returned_ids)), "核心类目必须出现")
    
    def test_category_info_structure(self):
        """测试类目信息结构完整性"""
        categories = select_categories(
            self.fingerprint,
            self.material_family,
            str(self.test_db),
            mode="auto"
        )
        
        # 检查第一个类目的结构
        cat = categories[0]
        required_fields = [
            "category_id", "name_zh", "name_en", "name_template",
            "prompts", "area_budget", "confidence", "affinity"
        ]
        for field in required_fields:
            self.assertIn(field, cat, f"类目信息应包含 {field}")
        
        # 检查 prompts 结构
        if cat["prompts"]:
            prompt = cat["prompts"][0]
            self.assertIn("text", prompt)
            self.assertIn("weight", prompt)
            self.assertIn("lang", prompt)
    
    def test_performance(self):
        """测试查询性能（<0.5s）"""
        t0 = time.time()
        
        categories = select_categories(
            self.fingerprint,
            self.material_family,
            str(self.test_db),
            mode="auto"
        )
        
        elapsed = time.time() - t0
        
        self.assertLess(elapsed, 0.5, f"查询耗时 {elapsed:.3f}s，应 <0.5s")
        self.assertGreater(len(categories), 0, "应返回类目")


class TestIntegrationEndToEnd(unittest.TestCase):
    """端到端集成测试（指纹 → 材质 → 类目选择）"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试环境"""
        cls.test_db_dir = Path(tempfile.mkdtemp())
        cls.real_db = Path("webui/data/adaptive_semantics.db")
        
        if cls.real_db.exists():
            cls.test_db = cls.test_db_dir / "test_adaptive.db"
            shutil.copy(cls.real_db, cls.test_db)
        else:
            raise FileNotFoundError(f"真实数据库不存在：{cls.real_db}")
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        shutil.rmtree(cls.test_db_dir)
    
    def test_full_pipeline(self):
        """测试完整流程（模拟主流程调用）"""
        # 1. 准备测试图像（宣纸水墨）
        test_img = np.ones((1440, 2880, 3), dtype=np.uint8)
        test_img[:, :] = [240, 240, 240]
        
        # 2. 提取指纹
        fingerprint = extract_fingerprint(test_img)
        self.assertIsNotNone(fingerprint)
        self.assertIn("embedding", fingerprint)
        
        # 3. 材质判别
        material_family, confidence = classify_material_family(fingerprint)
        self.assertIsNotNone(material_family)
        self.assertGreater(confidence, 0.0)
        
        # 4. 类目选择（auto 模式）
        categories = select_categories(
            fingerprint,
            material_family,
            str(self.test_db),
            mode="auto"
        )
        
        self.assertGreater(len(categories), 0, "应选出类目")
        
        # 5. 验证核心类目必出
        returned_ids = [cat["category_id"] for cat in categories]
        core_cats = {"seal", "calligraphy", "repair_marks"}
        self.assertTrue(core_cats.issubset(set(returned_ids)), "核心类目必须出现")
        
        # 打印结果（调试用）
        print("\n" + "="*60)
        print("端到端测试结果：")
        print(f"  材质家族：{material_family}（置信度 {confidence:.2f}）")
        print(f"  选出类目数：{len(categories)}")
        print(f"  前 5 个类目：")
        for cat in categories[:5]:
            print(f"    - {cat['name_zh']}（{cat['category_id']}）亲和度 {cat['affinity']:.2f}")
        print("="*60)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
