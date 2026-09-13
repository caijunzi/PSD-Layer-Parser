"""
端到端测试：Stage 4 案例推理库（CBR）
测试指纹索引构建、余弦检索、以及主流程集成的冷/热启动场景
"""
import unittest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加 engine 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from engine.adaptive.episode_indexer import EpisodeIndexer
from engine.adaptive.cbr_retriever import retrieve_similar_episode, get_cbr_hit_info


class TestCBRColdStart(unittest.TestCase):
    """测试冷启动场景：首次运行无索引，降级到从零选择"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "episode_index.pkl"
        self.episodes_dir = Path(self.temp_dir) / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_index_returns_none(self):
        """空索引时检索应返回 None"""
        indexer = EpisodeIndexer.load_or_create(self.index_path)
        self.assertEqual(len(indexer), 0)
        
        # 构造查询指纹
        query_fp = np.random.randn(128).astype(np.float32)
        
        # 检索应返回 None
        result = retrieve_similar_episode(query_fp, indexer, self.episodes_dir)
        self.assertIsNone(result)
    
    def test_no_audit_passed_returns_none(self):
        """索引中无通过审计的样本时应返回 None"""
        indexer = EpisodeIndexer.load_or_create(self.index_path)
        
        # 添加一个未通过审计的样本
        fp = np.random.randn(128).astype(np.float32)
        indexer.add("task_failed_001", fp, audit_passed=False)
        indexer.save()
        
        # 注意：retrieve_similar_episode 内部已硬编码 only_audit_passed=True，
        # 未通过审计的样本不会进入检索结果
        query_fp = np.random.randn(128).astype(np.float32)
        result = retrieve_similar_episode(query_fp, indexer, self.episodes_dir)
        self.assertIsNone(result)


class TestCBRWarmStart(unittest.TestCase):
    """测试热启动场景：第二次运行复用历史参数"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "episode_index.pkl"
        self.episodes_dir = Path(self.temp_dir) / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试用 episode 文件
        self.task_id = "task_20260913_test_001"
        self.episode_path = self.episodes_dir / f"{self.task_id}.episode.json"
        
        # 构造历史指纹
        self.history_fp = np.random.randn(128).astype(np.float32)
        
        # 写入 episode 文件
        episode_data = {
            "task_id": self.task_id,
            "fingerprint": {"embedding": self.history_fp.tolist()},
            "audit": {
                "dims": {"lost_ratio": 0.03},  # 通过审计
                "passed": True
            },
            "material_family": "金地屏风",
            "auto_tune": {
                "regions": [{"x": 0.1, "y": 0.2, "score": 0.8}],
                "density_bands": [{"label": "05A_前景寒林枯木", "min": 0.0, "max": 0.09}],
                "test_param": "value_from_history"
            }
        }
        with open(self.episode_path, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f)
        
        # 构建索引
        self.indexer = EpisodeIndexer.load_or_create(self.index_path)
        self.indexer.add(self.task_id, self.history_fp, audit_passed=True)
        self.indexer.save()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_similar_query_retrieves_history(self):
        """相似查询应检索到历史样本"""
        # 构造与历史指纹相似的查询（添加小噪声）
        query_fp = self.history_fp + np.random.randn(128).astype(np.float32) * 0.1
        query_fp = query_fp / np.linalg.norm(query_fp)  # 归一化
        
        result = retrieve_similar_episode(query_fp, self.indexer, self.episodes_dir, min_similarity=0.7)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], self.task_id)
        self.assertIn("auto_tune", result)
        self.assertEqual(result["auto_tune"]["test_param"], "value_from_history")
        # 验证其他 CBR 返回字段
        self.assertIn("material_family", result)
        self.assertEqual(result["material_family"], "金地屏风")
        self.assertIn("similarity", result)
        self.assertGreaterEqual(result["similarity"], 0.7)
    
    def test_dissimilar_query_returns_none(self):
        """不相似查询应返回 None"""
        # 构造完全不同的查询指纹
        query_fp = np.random.randn(128).astype(np.float32)
        
        result = retrieve_similar_episode(query_fp, self.indexer, self.episodes_dir, min_similarity=0.9)
        
        # 高阈值下应返回 None
        self.assertIsNone(result)
    
    def test_get_cbr_hit_info(self):
        """测试日志信息生成"""
        query_fp = self.history_fp + np.random.randn(128).astype(np.float32) * 0.1
        query_fp = query_fp / np.linalg.norm(query_fp)
        
        result = retrieve_similar_episode(query_fp, self.indexer, self.episodes_dir)
        self.assertIsNotNone(result)
        
        # 实际格式："CBR hit: <task_id> (similarity=0.99, material=金地屏风)"
        info = get_cbr_hit_info(result)
        self.assertIn(self.task_id, info)
        self.assertIn("similarity=", info)
        self.assertIn("material=", info)
        self.assertIn("金地屏风", info)


class TestCBRPerformance(unittest.TestCase):
    """测试 CBR 检索性能：1000 条索引检索 < 0.3s"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "episode_index.pkl"
        
        # 构建 1000 条索引
        self.indexer = EpisodeIndexer.load_or_create(self.index_path)
        for i in range(1000):
            fp = np.random.randn(128).astype(np.float32)
            self.indexer.add(f"task_{i:04d}", fp, audit_passed=(i % 2 == 0))  # 50% 通过审计
        self.indexer.save()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_search_performance(self):
        """1000 条索引检索耗时应 < 0.3s"""
        import time
        
        query_fp = np.random.randn(128).astype(np.float32)
        
        start = time.perf_counter()
        results = self.indexer.search(query_fp, top_k=5, only_audit_passed=True)
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 0.3, f"检索耗时 {elapsed:.4f}s 超过阈值 0.3s")
        self.assertGreater(len(results), 0, "应至少返回一个结果")
        
        # 验证返回格式
        for task_id, sim in results:
            self.assertIsInstance(task_id, str)
            self.assertIsInstance(sim, (float, np.floating))
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)


class TestCBRIndexPersistence(unittest.TestCase):
    """测试索引持久化与加载"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "episode_index.pkl"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load(self):
        """测试索引保存与加载"""
        # 创建索引并添加数据
        indexer1 = EpisodeIndexer.load_or_create(self.index_path)
        fp1 = np.random.randn(128).astype(np.float32)
        fp2 = np.random.randn(128).astype(np.float32)
        indexer1.add("task_001", fp1, audit_passed=True)
        indexer1.add("task_002", fp2, audit_passed=False)
        indexer1.save()
        
        self.assertEqual(len(indexer1), 2)
        
        # 重新加载
        indexer2 = EpisodeIndexer.load_or_create(self.index_path)
        self.assertEqual(len(indexer2), 2)
        
        # 验证检索结果一致
        query_fp = fp1 + np.random.randn(128).astype(np.float32) * 0.01
        results1 = indexer1.search(query_fp, top_k=1)
        results2 = indexer2.search(query_fp, top_k=1)
        
        self.assertEqual(results1[0][0], results2[0][0])  # task_id 一致
        self.assertAlmostEqual(results1[0][1], results2[0][1], places=5)  # 相似度一致


if __name__ == "__main__":
    unittest.main()
