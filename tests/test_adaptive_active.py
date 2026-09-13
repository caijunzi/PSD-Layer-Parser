"""单元测试：主动学习模块（Stage 5.2）

测试 active_learner 的核心功能：
- 识别不确定类目（confidence < threshold）
- 触发人审请求（写入待审队列）
- 读取待审列表（轮询端点）
- 标记已审核 + 应用反馈
"""
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

# 添加 engine 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from engine.adaptive.active_learner import (
    identify_uncertain_categories,
    request_human_feedback,
    get_pending_feedbacks,
    mark_feedback_reviewed,
)


class TestIdentifyUncertainCategories(unittest.TestCase):
    """测试识别不确定类目"""
    
    def test_identify_low_confidence_categories(self):
        """识别低置信度类目（< 0.7）"""
        detection_result = {
            "categories": ["distant_mountains", "water_ripples", "trees_vegetation"],
            "detections": [
                {"category_id": "distant_mountains", "confidence": 0.85, "bbox": [10, 20, 100, 150], "prompt": "远山"},
                {"category_id": "water_ripples", "confidence": 0.62, "bbox": [200, 300, 400, 500], "prompt": "水波纹"},
                {"category_id": "trees_vegetation", "confidence": 0.95, "bbox": [50, 50, 150, 200], "prompt": "树木植被"},
            ]
        }
        
        uncertain = identify_uncertain_categories(detection_result, confidence_threshold=0.7)
        
        # 应只识别出 water_ripples（0.62 < 0.7）
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(uncertain[0]["category_id"], "water_ripples")
        self.assertEqual(uncertain[0]["confidence"], 0.62)
        self.assertEqual(uncertain[0]["bbox"], [200, 300, 400, 500])
    
    def test_no_uncertain_categories(self):
        """全部类目置信度都高（≥ threshold）"""
        detection_result = {
            "detections": [
                {"category_id": "distant_mountains", "confidence": 0.85, "bbox": [10, 20, 100, 150]},
                {"category_id": "trees_vegetation", "confidence": 0.95, "bbox": [50, 50, 150, 200]},
            ]
        }
        
        uncertain = identify_uncertain_categories(detection_result, confidence_threshold=0.7)
        self.assertEqual(len(uncertain), 0)
    
    def test_custom_threshold(self):
        """自定义阈值（0.8）"""
        detection_result = {
            "detections": [
                {"category_id": "cat_a", "confidence": 0.75, "bbox": [0, 0, 10, 10]},
                {"category_id": "cat_b", "confidence": 0.85, "bbox": [0, 0, 10, 10]},
            ]
        }
        
        uncertain = identify_uncertain_categories(detection_result, confidence_threshold=0.8)
        
        # 0.75 < 0.8，应识别出 cat_a
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(uncertain[0]["category_id"], "cat_a")


class TestHumanFeedbackQueue(unittest.TestCase):
    """测试人审队列（写入 + 读取 + 标记）"""
    
    def setUp(self):
        # 创建临时待审队列文件
        self.temp_dir = tempfile.mkdtemp()
        self.pending_path = Path(self.temp_dir) / "pending_feedbacks.jsonl"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_request_and_get_pending_feedbacks(self):
        """写入待审请求 → 读取待审列表"""
        uncertain_categories = [
            {"category_id": "water_ripples", "confidence": 0.62, "bbox": [200, 300, 400, 500], "prompt": "水波纹"}
        ]
        
        # 写入待审请求
        feedback_id = request_human_feedback(
            uncertain_categories=uncertain_categories,
            task_id="task_test_001",
            image_path="/tmp/test_image.png",
            pending_path=str(self.pending_path)
        )
        
        self.assertTrue(feedback_id.startswith("fb_"))
        self.assertTrue(self.pending_path.exists())
        
        # 读取待审列表
        pending = get_pending_feedbacks(pending_path=str(self.pending_path))
        
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["feedback_request_id"], feedback_id)
        self.assertEqual(pending[0]["task_id"], "task_test_001")
        self.assertEqual(pending[0]["status"], "pending")
        self.assertEqual(len(pending[0]["uncertain_categories"]), 1)
        self.assertEqual(pending[0]["uncertain_categories"][0]["category_id"], "water_ripples")
    
    def test_mark_feedback_reviewed(self):
        """标记反馈已审核"""
        uncertain_categories = [
            {"category_id": "water_ripples", "confidence": 0.62, "bbox": [200, 300, 400, 500]}
        ]
        
        # 写入待审请求
        feedback_id = request_human_feedback(
            uncertain_categories=uncertain_categories,
            task_id="task_test_002",
            pending_path=str(self.pending_path)
        )
        
        # 标记已审核
        marked = mark_feedback_reviewed(
            feedback_request_id=feedback_id,
            user_action="accept",
            user_data={"note": "确认正确"},
            pending_path=str(self.pending_path)
        )
        
        self.assertTrue(marked)
        
        # 验证状态已更新
        with self.pending_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
            req = json.loads(line)
            self.assertEqual(req["status"], "reviewed")
            self.assertEqual(req["user_action"], "accept")
            self.assertIsNotNone(req["reviewed_at"])
        
        # 再次读取待审列表（应为空，因为 status="reviewed"）
        pending = get_pending_feedbacks(pending_path=str(self.pending_path))
        self.assertEqual(len(pending), 0)
    
    def test_get_pending_feedbacks_limit(self):
        """测试待审列表 limit 参数"""
        # 写入 3 个待审请求
        for i in range(3):
            request_human_feedback(
                uncertain_categories=[{"category_id": f"cat_{i}", "confidence": 0.5}],
                task_id=f"task_{i}",
                pending_path=str(self.pending_path)
            )
        
        # limit=2 应只返回前 2 个
        pending = get_pending_feedbacks(pending_path=str(self.pending_path), limit=2)
        self.assertEqual(len(pending), 2)
    
    def test_empty_uncertain_categories_no_request(self):
        """不确定类目为空时不写入待审请求"""
        feedback_id = request_human_feedback(
            uncertain_categories=[],
            task_id="task_test_003",
            pending_path=str(self.pending_path)
        )
        
        self.assertEqual(feedback_id, "")
        self.assertFalse(self.pending_path.exists())


if __name__ == "__main__":
    unittest.main()
