"""接线（wiring）守卫测试：防止「模块已实现但无生产调用点」的链路断裂

背景（2026-09-14 审计发现）：
Stage 5.2 的主动学习、Stage 3 的学习闭环都曾出现「代码写完、测试也全绿，
但没有任何生产代码调用」的情况 —— 结果是前端轮询的待审队列永远为空、
学习器从未运行。这类问题单元测试发现不了（测试直接调函数，绕过了接线）。

本测试用两类手段防回退：
1. 静态断言：引擎主流程源码中确实存在关键调用点（被人删掉就会红）
2. 契约断言：provider 记录的字段 与 Stage 3 归因/学习器期望的字段 一致
"""
import re
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENGINE_MAIN = Path("run_universal_engine.py")
PROVIDER = Path("engine/providers/grounded_sam_provider.py")
ATTRIBUTION = Path("engine/adaptive/attribution.py")


class TestActiveLearningWiring(unittest.TestCase):
    """Stage 5.2 主动学习：生产端必须把不确定类目写入待审队列"""

    def setUp(self):
        if not ENGINE_MAIN.exists():
            self.skipTest("引擎主流程文件不存在")
        self.src = ENGINE_MAIN.read_text(encoding="utf-8")

    def test_engine_calls_request_human_feedback(self):
        """引擎必须调用 request_human_feedback（否则待审队列永远为空）"""
        self.assertIn(
            "request_human_feedback",
            self.src,
            "run_universal_engine.py 未调用 request_human_feedback —— "
            "待审队列将永远为空、前端人审弹窗永不触发",
        )

    def test_engine_calls_identify_uncertain_categories(self):
        self.assertIn("identify_uncertain_categories", self.src)

    def test_active_learner_import_inside_try(self):
        """调用必须包在 try 内（失败不影响主流程）"""
        # 找到 request_human_feedback 调用点，向前 800 字符内应有 try:
        idx = self.src.find("request_human_feedback")
        self.assertGreater(idx, 0)
        head = self.src[max(0, idx - 1200) : idx]
        self.assertIn("try:", head, "主动学习调用应包在 try 块内，避免影响主流程")


class TestStage3LearningWiring(unittest.TestCase):
    """Stage 3 学习闭环：引擎必须在分割后调用学习器"""

    def setUp(self):
        if not ENGINE_MAIN.exists():
            self.skipTest("引擎主流程文件不存在")
        self.src = ENGINE_MAIN.read_text(encoding="utf-8")

    def test_engine_calls_learn_from_episode(self):
        self.assertIn(
            "learn_from_episode",
            self.src,
            "run_universal_engine.py 未调用 learn_from_episode —— Stage 3 学习器从未运行",
        )

    def test_learning_after_segmentation(self):
        """学习必须在分割之后（dino_detections 才有值）"""
        i_seg = self.src.find("grounded_sam.segment_objects")
        i_learn = self.src.find("learn_from_episode")
        self.assertGreater(i_seg, 0, "找不到分割调用点")
        self.assertGreater(i_learn, i_seg, "学习调用必须在分割之后，否则 dino_detections 为空")


class TestDetectionFieldContract(unittest.TestCase):
    """provider 记录的字段 必须与 Stage 3 归因期望的字段一致"""

    def test_provider_records_layer_name(self):
        """归因按 layer_name 取类目（不是 category_id）"""
        src = PROVIDER.read_text(encoding="utf-8")
        i = src.find("self.dino_detections.append")
        self.assertGreater(i, 0)
        block = src[i : i + 600]
        self.assertIn('"layer_name"', block)
        self.assertIn('"prompt"', block)

    def test_provider_sets_quality_gate_fields(self):
        """质量门填充 quality_gate_passed / quality_gate_reason"""
        src = PROVIDER.read_text(encoding="utf-8")
        self.assertIn('"quality_gate_passed"', src)
        self.assertIn('"quality_gate_reason"', src)

    def test_attribution_expects_same_fields(self):
        """归因读取的字段与 provider 写入的字段一致（防止改名导致静默 0 归因）"""
        prov = PROVIDER.read_text(encoding="utf-8")
        attr = ATTRIBUTION.read_text(encoding="utf-8")
        for field in ("quality_gate_passed", "quality_gate_reason", "diffuse_gate_passed"):
            self.assertIn(field, prov, f"provider 未写入 {field}")
            self.assertIn(field, attr, f"attribution 未读取 {field}")
        self.assertIn('det.get("layer_name"', attr)


class TestLearningProducesAttribution(unittest.TestCase):
    """用 provider 真实字段结构驱动学习器，应产出归因（非静默 0）"""

    def test_learn_from_episode_with_real_fields(self):
        from engine.adaptive.learner import SemanticLearner

        # 字段结构对齐 grounded_sam_provider.dino_detections
        dino = [
            {
                "layer_name": "人物",
                "prompt": "red stamp",
                "quality_gate_passed": False,
                "quality_gate_reason": "面积超预算",
            },
            {
                "layer_name": "远山",
                "prompt": "distant mountains",
                "quality_gate_passed": True,
                "diffuse_gate_passed": False,
                "diffuse_gate_reason": "弥散",
            },
            {"layer_name": "印章", "prompt": "seal", "quality_gate_passed": True},
        ]
        result = SemanticLearner().learn_from_episode(
            {"episode_id": "wiring_test", "category_ids": ["人物", "远山", "印章"], "dino_detections": dino}
        )
        attrs = result.get("attributions") or []
        self.assertEqual(len(attrs), 3, f"应有 3 条归因，实际 {len(attrs)}")
        signals = {a["signal"] for a in attrs}
        self.assertEqual(
            signals, {"quality_reject", "diffuse_reject", "accept"}, f"信号类型不全：{signals}"
        )

    def test_learn_with_wrong_field_yields_zero(self):
        """若误用 category_id（非 layer_name）则归因为空 —— 记录此契约，便于排错"""
        from engine.adaptive.learner import SemanticLearner

        dino = [{"category_id": "人物", "prompt": "red stamp", "quality_gate_passed": False}]
        result = SemanticLearner().learn_from_episode(
            {"episode_id": "x", "category_ids": ["人物"], "dino_detections": dino}
        )
        self.assertEqual(len(result.get("attributions") or []), 0)


if __name__ == "__main__":
    unittest.main()
