"""指纹标准化/PCA 训练与加载的回归保护（2026-09-16）。

背景：`_pca_model` 长期是 "placeholder"、`train_pca_from_dataset` 生产零调用，
embedding 实为「未标准化的原始特征截断」，CBR 余弦检索被大尺度特征主导。

本测试锁定：
  1. 未训练时回退旧行为（不报错、维度仍为 128）；
  2. 训练后 scaler 是**真实统计**而非恒等，且能持久化/重新加载；
  3. `_apply_pca` 首次调用会**自动加载**已训练模型（生产无需显式调用训练）；
  4. embedding 维度恒为 128（CBR 索引硬要求 len==128）。
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.adaptive.fingerprint as fp  # noqa: E402


def _make_images(d: Path, n: int = 3, size: int = 64) -> list:
    """生成少量随机训练图（避免依赖 inputs/ 真实样本，保证测试隔离）。"""
    import cv2
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        p = d / f"train_{i}.png"
        cv2.imwrite(str(p), img)
        out.append(str(p))
    return out


class _PcaIsolatedTestCase(unittest.TestCase):
    """隔离 PCA 全局状态与模型路径，避免污染真实 `fingerprint_pca.pkl`。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "pca.pkl"
        self._env_old = os.environ.get("ULS_PCA_MODEL")
        os.environ["ULS_PCA_MODEL"] = str(self.model)
        self._st = (fp._pca_model, fp._scaler, fp._pca_loaded)
        self._reset()

    def tearDown(self):
        fp._pca_model, fp._scaler, fp._pca_loaded = self._st
        if self._env_old is None:
            os.environ.pop("ULS_PCA_MODEL", None)
        else:
            os.environ["ULS_PCA_MODEL"] = self._env_old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reset(self):
        fp._pca_model, fp._scaler, fp._pca_loaded = None, None, False


class TestUntrainedFallback(_PcaIsolatedTestCase):
    def test_embedding_always_128(self):
        """无模型时回退截断/填充，不报错且维度仍为 128（索引硬要求）。"""
        emb = fp._apply_pca(np.zeros(84, dtype=np.float32))
        self.assertEqual(len(emb), 128)

    def test_identity_when_untrained(self):
        """未训练：标准化退化为恒等（旧行为），保证向后兼容。"""
        v = np.arange(1, 85, dtype=np.float64)
        emb = fp._apply_pca(v)
        self.assertTrue(np.allclose(emb[:84], v), "未训练时应为恒等映射")


class TestTrainingPersists(_PcaIsolatedTestCase):
    def test_train_produces_real_scaler_and_saves(self):
        imgs = _make_images(self.tmp)
        ok = fp.train_pca_from_dataset(imgs, n_components=128, path=str(self.model))
        self.assertTrue(ok, "训练应成功")
        self.assertTrue(self.model.is_file(), "模型应已落盘")
        # 真实统计：scale 不应全为 1（恒等）
        self.assertTrue(np.any(fp._scaler.scale_ != 1.0),
                        "scaler 应为真实统计，而非恒等占位")

    def test_save_load_roundtrip(self):
        imgs = _make_images(self.tmp)
        fp.train_pca_from_dataset(imgs, n_components=128, path=str(self.model))
        mean_before = fp._scaler.mean_.copy()
        scale_before = fp._scaler.scale_.copy()

        self._reset()
        self.assertTrue(fp.load_pca_model(), "应能加载已保存模型")
        self.assertTrue(np.allclose(fp._scaler.mean_, mean_before))
        self.assertTrue(np.allclose(fp._scaler.scale_, scale_before))

    def test_apply_pca_autoloads_trained_model(self):
        """生产只调 _apply_pca（不显式训练），也应自动加载并生效。"""
        imgs = _make_images(self.tmp)
        fp.train_pca_from_dataset(imgs, n_components=128, path=str(self.model))

        self._reset()
        self.assertFalse(fp._pca_loaded)
        v = np.arange(1, 85, dtype=np.float64)
        emb = fp._apply_pca(v)

        self.assertTrue(fp._pca_loaded, "首次调用应尝试加载")
        self.assertEqual(len(emb), 128)
        # 已加载真实 scaler → 不再是恒等
        self.assertFalse(np.allclose(emb[:84], v),
                         "加载真实模型后不应仍是恒等映射")

    def test_few_samples_means_standardize_only(self):
        """样本远少于特征时只标准化、不降维（PCA 在此规模无统计意义）。"""
        imgs = _make_images(self.tmp, n=3)
        fp.train_pca_from_dataset(imgs, n_components=128, path=str(self.model))
        # 3 < _MIN_SAMPLES_FOR_PCA → 不应产出 PCA 对象
        self.assertIsNone(fp._pca_model,
                          "样本不足时应只做标准化，不强行降维")


if __name__ == "__main__":
    unittest.main()
