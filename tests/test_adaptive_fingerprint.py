"""
指纹提取与材质判别 - 单元测试

测试：
1. 指纹提取功能正确性
2. 材质家族判别准确性
3. 性能要求（< 3s）
"""
import unittest
import numpy as np
import cv2
from pathlib import Path
import time

# 导入待测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.adaptive.fingerprint import extract_fingerprint, train_pca_from_dataset
from engine.adaptive.material_classifier import classify_material_family, explain_classification


class TestFingerprint(unittest.TestCase):
    """指纹提取测试"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试图像（模拟图）"""
        # 模拟金地屏风（高亮度 + 黄色）
        cls.gold_screen = np.ones((1000, 2000, 3), dtype=np.uint8)
        cls.gold_screen[:, :] = [30, 200, 220]  # BGR: 黄色高亮
        
        # 模拟宣纸水墨（高亮度 + 低饱和）
        cls.ink_paper = np.ones((1440, 2880, 3), dtype=np.uint8)
        cls.ink_paper[:, :] = [245, 245, 245]  # 浅灰（纸白）
        
        # 模拟绢本工笔（中亮度 + 中饱和）
        cls.silk_painting = np.ones((1200, 1800, 3), dtype=np.uint8)
        cls.silk_painting[:, :] = [180, 200, 210]  # 米黄偏红
        
        # 模拟油画布（高饱和 + 高边缘密度）
        cls.oil_canvas = np.ones((1000, 1500, 3), dtype=np.uint8)
        cls.oil_canvas[:, :] = [50, 100, 200]  # 深红高饱和
        # 添加厚重笔触（随机噪声）
        noise = np.random.randint(-50, 50, cls.oil_canvas.shape, dtype=np.int16)
        cls.oil_canvas = np.clip(cls.oil_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def test_fingerprint_structure(self):
        """测试指纹返回结构正确"""
        fp = extract_fingerprint(self.ink_paper)
        
        # 检查返回字段
        self.assertIn("embedding", fp)
        self.assertIn("global_features", fp)
        self.assertIn("texture_features", fp)
        self.assertIn("structure_features", fp)
        self.assertIn("raw_features", fp)
        self.assertIn("image_shape", fp)
        
        # 检查维度
        self.assertEqual(fp["embedding"].shape[0], 128, "PCA 降维后应为 128 维")
        self.assertEqual(len(fp["raw_features"]), 84, "原始特征应为 84 维（55+24+5）")
    
    def test_fingerprint_performance(self):
        """测试指纹提取性能（< 3s）"""
        large_image = np.random.randint(0, 255, (2880, 4320, 3), dtype=np.uint8)
        
        t0 = time.time()
        fp = extract_fingerprint(large_image)
        elapsed = time.time() - t0
        
        self.assertLess(elapsed, 3.0, f"指纹提取耗时 {elapsed:.2f}s，超过 3s 限制")
    
    def test_fingerprint_consistency(self):
        """测试同一图像多次提取结果一致"""
        fp1 = extract_fingerprint(self.ink_paper)
        fp2 = extract_fingerprint(self.ink_paper)
        
        # 检查 embedding 一致性（允许微小浮点误差）
        np.testing.assert_allclose(fp1["embedding"], fp2["embedding"], rtol=1e-5)
    
    def test_global_features(self):
        """测试全局特征提取合理性"""
        fp = extract_fingerprint(self.gold_screen)
        global_feats = fp["global_features"]
        
        # 金地屏风应有高 L*（亮度）
        self.assertGreater(global_feats["bg_median_LAB"][0], 70, "金地屏风 L* 应 > 70")
        
        # 长宽比合理
        self.assertGreater(global_feats["aspect_ratio"], 1.5, "测试图长宽比应 > 1.5")
    
    def test_texture_features_fallback(self):
        """测试 scikit-image 未安装时的降级行为"""
        # 这个测试假设 scikit-image 已安装，验证正常路径
        fp = extract_fingerprint(self.ink_paper)
        texture = fp["texture_features"]
        
        # 检查返回的 vector 不全是零（说明实际提取了特征）
        self.assertGreater(np.abs(texture["vector"]).sum(), 0.0, "纹理特征不应全为零")


class TestMaterialClassifier(unittest.TestCase):
    """材质家族判别测试"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试图像（复用 TestFingerprint 的）"""
        TestFingerprint.setUpClass()
        cls.gold_screen = TestFingerprint.gold_screen
        cls.ink_paper = TestFingerprint.ink_paper
        cls.silk_painting = TestFingerprint.silk_painting
        cls.oil_canvas = TestFingerprint.oil_canvas
    
    def test_classify_gold_screen(self):
        """测试金地屏风分类"""
        fp = extract_fingerprint(self.gold_screen)
        family, conf = classify_material_family(fp)
        
        self.assertEqual(family, "金地屏风", f"应判别为金地屏风，实际：{family}")
        self.assertGreater(conf, 0.5, f"置信度应 > 0.5，实际：{conf:.2f}")
    
    def test_classify_ink_paper(self):
        """测试宣纸水墨分类"""
        fp = extract_fingerprint(self.ink_paper)
        family, conf = classify_material_family(fp)
        
        self.assertEqual(family, "宣纸水墨", f"应判别为宣纸水墨，实际：{family}")
        self.assertGreater(conf, 0.5, f"置信度应 > 0.5，实际：{conf:.2f}")
    
    def test_classify_silk_painting(self):
        """测试绢本工笔分类"""
        fp = extract_fingerprint(self.silk_painting)
        family, conf = classify_material_family(fp)
        
        # 绢本工笔与宣纸水墨可能混淆，允许这两类之一
        self.assertIn(family, ["绢本工笔", "宣纸水墨"], f"应判别为绢本工笔或宣纸水墨，实际：{family}")
        self.assertGreater(conf, 0.3, f"置信度应 > 0.3，实际：{conf:.2f}")
    
    def test_classify_oil_canvas(self):
        """测试油画布分类"""
        fp = extract_fingerprint(self.oil_canvas)
        family, conf = classify_material_family(fp)
        
        # 油画布特征明显，应正确分类
        self.assertEqual(family, "油画布", f"应判别为油画布，实际：{family}")
        self.assertGreater(conf, 0.4, f"置信度应 > 0.4，实际：{conf:.2f}")
    
    def test_explain_classification(self):
        """测试分类解释功能"""
        fp = extract_fingerprint(self.gold_screen)
        family, conf = classify_material_family(fp)
        explanation = explain_classification(fp, family, conf)
        
        # 检查解释包含关键信息
        self.assertIn("分类结果", explanation)
        self.assertIn("关键特征", explanation)
        self.assertIn("判别依据", explanation)
        self.assertIn(family, explanation)
    
    def test_confidence_range(self):
        """测试置信度范围合法（0..1）"""
        test_images = [self.gold_screen, self.ink_paper, self.silk_painting, self.oil_canvas]
        
        for img in test_images:
            fp = extract_fingerprint(img)
            family, conf = classify_material_family(fp)
            
            self.assertGreaterEqual(conf, 0.0, f"{family} 置信度 {conf} 应 >= 0")
            self.assertLessEqual(conf, 1.0, f"{family} 置信度 {conf} 应 <= 1")


class TestIntegration(unittest.TestCase):
    """集成测试：指纹提取 + 材质判别"""
    
    def test_end_to_end_workflow(self):
        """测试完整流程（从图像到分类结果）"""
        # 创建测试图像
        test_img = np.ones((1000, 2000, 3), dtype=np.uint8)
        test_img[:, :] = [240, 240, 240]  # 浅灰
        
        # 完整流程
        fp = extract_fingerprint(test_img)
        family, conf = classify_material_family(fp)
        explanation = explain_classification(fp, family, conf)
        
        # 验证每步都有输出
        self.assertIsNotNone(fp)
        self.assertIsNotNone(family)
        self.assertIsNotNone(conf)
        self.assertIsNotNone(explanation)
        
        # 打印结果（调试用）
        print("\n" + "="*60)
        print("集成测试结果：")
        print(explanation)
        print("="*60)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
