"""tools/audit_psb 修复验证：

1. ⑤ 合成等价性改用 psd_tools 真实渲染（psd.composite），尊重 opacity/blend/CMYK，
   不再用手工 alpha-over 忽略混合模式/不透明度；CMYK 产物与转 CMYK 的源图同空间比对。
2. 底板检测/纯净度不写死"金地"——产品线无关关键词即可命中。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_psb import (  # noqa: E402
    _composite_rmse, _find_base_layer, DEFAULT_BASE_KEYWORDS,
)


def _fake_psd(image):
    class FakePSD:
        def __init__(self, img):
            self._img = img
        def composite(self):
            return self._img
    return FakePSD(image)


class _Layer:
    def __init__(self, name):
        self.name = name


class TestCompositeRealAPI(unittest.TestCase):
    def test_composite_uses_real_api_and_rgb(self):
        calls = {"n": 0}
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        real = Image.fromarray(src)

        class FakePSD:
            def composite(self):
                calls["n"] += 1
                return real
        raw, low, _cm = _composite_rmse(FakePSD(), src, target_long=40)
        self.assertEqual(calls["n"], 1, "必须使用 psd.composite() 真实合成")
        self.assertAlmostEqual(raw, 0.0, places=5)
        self.assertAlmostEqual(low, 0.0, places=5)
        self.assertTrue(_cm, "RGB 产物不依赖 ICC，应标注已色彩管理")

    def test_composite_cmyk_uses_same_icc_reference(self):
        """CMYK 产物必须用**同一 ICC** 分色的参考图比对（同口径），而非 PIL 朴素转换。

        构造自洽用例：产物 = 源图经项目 ICC 分色 → 与同 ICC 参考比对，RMSE 应≈0。
        这是 2026-09-16 修复的核心回归——此前用 PIL 朴素 convert（K 恒 0）导致虚高。
        """
        import cv2
        from tools.audit_psb import _src_to_cmyk_same_icc
        rng = np.random.default_rng(0)
        src_rgb = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        prod = _src_to_cmyk_same_icc(Image.fromarray(src_rgb))
        if prod is None:
            self.skipTest("ICC/ImageCms 不可用，无法验证同口径比对")
        # 注意：_composite_rmse 入参是 BGR（内部自行转 RGB），测试须按契约传 BGR，
        # 否则通道被交换，随机彩色图会立刻暴露（全零图看不出来）。
        src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
        raw, low, cm_ok = _composite_rmse(_fake_psd(prod), src_bgr, target_long=40)
        self.assertTrue(cm_ok, "ICC 可用时必须走同口径色彩管理")
        self.assertLess(raw, 1.0, f"同 ICC 分色产物应与参考几乎一致，实际 raw={raw:.3f}")
        self.assertLess(low, 1.0, f"同 ICC 分色产物应与参考几乎一致，实际 low={low:.3f}")

    def test_composite_falls_back_and_flags_when_icc_missing(self):
        """ICC 不可用 → 回退 PIL 朴素转换，并显式标注 color_managed=False（不静默）。"""
        src = np.zeros((40, 40, 3), dtype=np.uint8)
        cmyk = Image.fromarray(src).convert("CMYK")
        raw, low, cm_ok = _composite_rmse(_fake_psd(cmyk), src, target_long=40,
                                           icc_path=str(ROOT / "__no_such_icc__.icc"))
        self.assertFalse(cm_ok, "ICC 缺失时必须标注未做色彩管理")

    def test_composite_downscales_large(self):
        # 模拟大图：用 400x400 但 target_long=100 → 应被缩到 100 宽
        big = Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8))
        raw, low, _cm = _composite_rmse(_fake_psd(big),
                                        np.zeros((400, 400, 3), dtype=np.uint8), target_long=100)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_real_psd_composite_runs(self):
        # 真实小 PSB 端到端跑通（RGB）
        p = ROOT / "intermediate" / "test_patch.psb"
        if not p.exists():
            self.skipTest("无 test_patch.psb")
        from psd_tools import PSDImage
        psd = PSDImage.open(str(p))
        src = np.zeros((psd.size[1], psd.size[0], 3), dtype=np.uint8)
        raw, low, _cm = _composite_rmse(psd, src, target_long=200)
        self.assertIsInstance(raw, float)
        self.assertIsInstance(low, float)


class TestAlphaChannelLayout(unittest.TestCase):
    """回归 2026-09-16 核心缺陷：_alpha 写死 a[:, :, 3]，在 CMYK 产物上取到 K 通道。

    psd_tools 的 layer.numpy() 通道数随色彩模式变化：
      RGB  → (H, W, 4) = R, G, B, Alpha
      CMYK → (H, W, 5) = C, M, Y, K, Alpha   ← Alpha 在 index 4
    取错后掩码被判成"全画布"，②③④⑥ 四维全部基于错误掩码计算。
    """

    def test_rgb_layer_alpha_at_index3(self):
        from tools.audit_psb import _alpha
        a = np.zeros((4, 4, 4), np.float32)
        a[0, 0, 3] = 1.0
        al = _alpha(a)
        self.assertAlmostEqual(float(al.sum()), 1.0, places=6)

    def test_cmyk_layer_alpha_at_index4(self):
        from tools.audit_psb import _alpha
        a = np.zeros((4, 4, 5), np.float32)
        a[0, 0, 4] = 1.0
        al = _alpha(a)
        self.assertAlmostEqual(float(al.sum()), 1.0, places=6,
                               msg="CMYK 层必须取 index 4 作为 Alpha")

    def test_cmyk_k_channel_does_not_leak_into_alpha(self):
        """核心回归：K 通道全非零而 Alpha 稀疏时，旧口径会误判为全画布。"""
        from tools.audit_psb import _alpha
        h, w = 8, 8
        a = np.zeros((h, w, 5), np.float32)
        a[..., :4] = 0.5          # C,M,Y,K 全 0.5 → 旧口径 a[:,:,3] 全 0.5（100% 覆盖）
        a[0:2, :, 4] = 1.0        # 真实 Alpha 只在顶部 2 行 = 25%
        al = _alpha(a)
        self.assertAlmostEqual(float((al > 0.03).mean()), 0.25, places=6,
                               msg="应取真实 Alpha（25%），而非 K 通道（100%）")
        # 对照：旧口径（index 3 = K）确实会得到 100%
        self.assertAlmostEqual(float((a[:, :, 3] > 0.03).mean()), 1.0, places=6)

    def test_layer_without_alpha_returns_ones(self):
        from tools.audit_psb import _alpha
        al = _alpha(np.zeros((4, 4, 3), np.float32))
        self.assertEqual(al.shape, (4, 4))
        self.assertTrue(np.all(al == 1.0), "无 Alpha 通道时应视为完全不透明")


class TestLayerRgbConversion(unittest.TestCase):
    """_layer_rgb：CMYK 层此前被 ba[:, :, :3] 当成 R,G,B 与源图比对，基准完全错位。"""

    class _FakeLayer:
        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

    def test_cmyk_uses_multiplicative_conversion(self):
        from tools.audit_psb import _layer_rgb
        # psd_tools 对 CMYK 层的 numpy() 取值为「呈色」= 1 - 墨量（实测象牙底板
        # C=.961 M=.910 Y=.829 K=1.000，对应墨量 C4%/M9%/Y17%/K0%）。
        # 呈色全 1 → 墨量全 0 → 纯白
        a = np.zeros((2, 2, 5), np.float32)
        a[..., :4] = 1.0
        a[..., 4] = 1.0
        rgb = _layer_rgb(self._FakeLayer(a))
        self.assertAlmostEqual(float(rgb.max()), 255.0, places=3)
        self.assertAlmostEqual(float(rgb.min()), 255.0, places=3)

    def test_cmyk_dark_pixel(self):
        from tools.audit_psb import _layer_rgb
        # C/M/Y 呈色 0 = 满青品黄（叠成黑）；K 呈色 0 亦可。此处 C=M=Y=0 → 全黑
        a = np.zeros((2, 2, 5), np.float32)
        a[..., 4] = 1.0
        rgb = _layer_rgb(self._FakeLayer(a))
        self.assertAlmostEqual(float(rgb.max()), 0.0, places=3)

    def test_rgb_unit_range_is_scaled(self):
        from tools.audit_psb import _layer_rgb
        a = np.full((2, 2, 4), 0.5, np.float32)
        a[..., 3] = 1.0
        rgb = _layer_rgb(self._FakeLayer(a))
        self.assertAlmostEqual(float(rgb.mean()), 127.5, places=3)

    def test_rgb_255_range_untouched(self):
        from tools.audit_psb import _layer_rgb
        a = np.full((2, 2, 4), 200.0, np.float32)
        rgb = _layer_rgb(self._FakeLayer(a))
        self.assertAlmostEqual(float(rgb.mean()), 200.0, places=3)

    def test_channels_are_rgb_not_cmy(self):
        """红色像素在 CMYK 下应还原为红，而不是被当成 C,M,Y 读成青色。"""
        from tools.audit_psb import _layer_rgb
        a = np.zeros((1, 1, 5), np.float32)
        # 红 = 无青、满品、满黄、无黑 → 呈色 C=1, M=0, Y=0, K=1
        a[0, 0, 0] = 1.0
        a[0, 0, 3] = 1.0
        rgb = _layer_rgb(self._FakeLayer(a))
        r, g, b = float(rgb[0, 0, 0]), float(rgb[0, 0, 1]), float(rgb[0, 0, 2])
        self.assertGreater(r, 200.0)
        self.assertLess(g, 60.0)
        self.assertLess(b, 60.0)


class TestProcessLayersExcludedFromCarrier(unittest.TestCase):
    """④ 内容承载的 union 必须排除印前加工层。

    加工层（冲孔/烫金/陷印/专色）整版施加，Alpha 天然覆盖全画布；
    纳入 union 会把 union 撑到 100%，使 lost 恒为 0 → 假通过。
    实证：damask case 的 12_激光冲孔挂点 Alpha 覆盖 100%，
    修复前 ④ lost_ratio=0.0；排除加工层后 0.1129（正确判失败）。
    """

    def test_process_names_cover_real_layer_names(self):
        from tools.audit_psb import TH
        real = [
            "12_激光冲孔挂点_DieCut_Perforations",
            "13B_复古金专色_Foil_Gold",
            "13A_玫瑰铜箔专色_Foil_Copper",
            "11_烫金陷印_Foil_Trap_Spot",
        ]
        for name in real:
            self.assertTrue(any(p in name for p in TH["process_names"]),
                            f"加工层未被识别（会被计入 union 撑满覆盖率）: {name}")

    def test_content_layers_not_mistaken_as_process(self):
        from tools.audit_psb import TH
        content = [
            "11_未分类墨迹残层_Unclassified_Ink_Residue",
            "05A_前景寒林枯木_Foreground_Barren_Trees",
            "04A_前景墨岩峭壁_Foreground_Dark_Cliffs",
        ]
        for name in content:
            self.assertFalse(any(p in name for p in TH["process_names"]),
                             f"内容层被误判为加工层（会导致漏检）: {name}")


class TestBaseLayerNotHardcodedGold(unittest.TestCase):
    def test_finds_non_gold_base(self):
        layers = [_Layer("Fabric_Base_01"), _Layer("some_content")]
        self.assertIsNotNone(_find_base_layer(layers))

    def test_finds_generic_base_names(self):
        for name in ("Base_Ground", "底版", "底板", "Substrate", "Gold_Base"):
            layers = [_Layer(name + "_x"), _Layer("ink")]
            self.assertIsNotNone(_find_base_layer(layers), f"未命中 {name}")

    def test_no_base_returns_none(self):
        layers = [_Layer("mountain"), _Layer("water")]
        self.assertIsNone(_find_base_layer(layers))

    def test_default_keywords_are_product_line_agnostic(self):
        # 关键词不应只包含金地特定命名，须覆盖多产品线
        self.assertIn("Fabric_Base", DEFAULT_BASE_KEYWORDS)
        self.assertIn("Base_Ground", DEFAULT_BASE_KEYWORDS)


class TestBlackPlateMetric(unittest.TestCase):
    """⑦ k_channel_nonzero_pct 口径修复（2026-09-16）。

    旧口径 `sub[:, :, 3] > 0.1 * norm` 判的是「K 呈色大于 10%」——psd_tools 对 CMYK
    返回**呈色**（=1-墨量），故该式在 K 全空（呈色=1）与 K 有墨时**都**接近 100%，
    真黑版判定形同虚设（实测无 ICC / 有 ICC 产物同为 100.0）。
    正确口径：K 版非空 = 存在黑墨 = 呈色 < 1。
    """

    def test_metric_counts_ink_presence(self):
        import inspect
        from tools import audit_psb
        src = inspect.getsource(audit_psb)
        # 只检查「赋值那一行」的口径，避免被注释/文档字符串误伤
        assign = [l for l in src.splitlines()
                  if "k_channel_nonzero_pct" in l and 'd["metrics"]' in l and "=" in l]
        self.assertTrue(assign, "未找到 k_channel_nonzero_pct 赋值行")
        self.assertIn("kk < 0.999", assign[0], "必须基于「存在黑墨（呈色<1）」")
        self.assertNotIn("> 0.1", assign[0], "禁止回退到旧口径（恒 ~100%）")

    def test_metric_semantics_on_synthetic(self):
        # 模拟 psd_tools 呈色：K 全空 = 呈色 1.0 → 非空占比 0%；半幅有墨 → 50%
        empty = np.ones((10, 10), np.float32)
        self.assertAlmostEqual(float((empty < 0.999).mean()) * 100, 0.0, places=3)
        half = np.ones((10, 10), np.float32)
        half[:5] = 0.7
        self.assertAlmostEqual(float((half < 0.999).mean()) * 100, 50.0, places=3)


class TestAuditMetricFailureFailClosed(unittest.TestCase):
    """审计**指标计算失败**必须 fail-closed（2026-09-16 修复静默吞异常）。

    此前 ⑥/⑦ 的指标计算被 `except Exception: pass` 包裹：一旦抛错，指标静默消失而
    维度仍报 passed=True —— 等价于"审计说通过却没测量"，与 ⑦ 口径写反同类事故。
    """

    def test_record_metric_failure_marks_dim_failed(self):
        from tools.audit_psb import _record_metric_failure
        d = {"passed": True, "metrics": {}, "issues": []}
        _record_metric_failure(d, "⑦ plate 合规", ValueError("boom"))
        self.assertFalse(d["passed"], "指标计算失败必须判失败")
        self.assertEqual(d["metrics"]["metric_error"], "ValueError: boom")
        self.assertTrue(d["issues"], "必须留下 issue 说明")
        self.assertIn("不得视为通过", d["issues"][0])

    def test_no_silent_pass_in_audit_dims(self):
        """静态断言：审计维度内不得再出现空的 `except Exception: pass`。"""
        import inspect
        import re
        from tools import audit_psb
        src = inspect.getsource(audit_psb)
        # 允许 docstring 描述，但不得存在「except 后紧跟 pass」的实际语句
        bad = re.findall(r"except[^\n]*:\s*\n\s*pass\s*$", src, re.M)
        self.assertEqual(bad, [], f"审计器内仍有静默吞异常：{bad}")


class TestSharedLayerIoSingleSource(unittest.TestCase):
    """2026-09-16 根因修复：所有 PSD 层读取必须收敛到单一权威入口
    `engine.core.psd_layer_io`，杜绝散落的 `[:, :, 3]` / `[:, :, :3]` 索引假设。

    本测试锁定「无第二份实现」——若有人又在某处手写了 `lyr.numpy()[:, :, 3]`，
    这里的同一对象断言会失败，强制其复用共享入口。
    """

    def test_audit_psb_alpha_is_shared(self):
        from tools.audit_psb import _alpha
        from engine.core.psd_layer_io import layer_alpha
        self.assertIs(_alpha, layer_alpha,
                      "audit_psb._alpha 必须是 psd_layer_io.layer_alpha，不能有第二份实现")

    def test_audit_psb_rgb_is_shared(self):
        from tools.audit_psb import _layer_rgb
        from engine.core.psd_layer_io import layer_rgb
        self.assertIs(_layer_rgb, layer_rgb,
                      "audit_psb._layer_rgb 必须是 psd_layer_io.layer_rgb，不能有第二份实现")

    def test_audit_system_integrity_uses_shared_alpha(self):
        # 防欺骗审计（tests/audit_system_integrity.py）曾用 lyr.numpy()[:, :, 3]，
        # 在 CMYK 产物上误取 K 通道 → 审计整体失效。必须走共享入口。
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_system_integrity_test", ROOT / "tests/audit_system_integrity.py")
        ai = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ai)
        from engine.core.psd_layer_io import layer_alpha
        self.assertIs(ai._layer_alpha, layer_alpha,
                       "audit_system_integrity 必须复用 psd_layer_io.layer_alpha")

    def test_no_stray_numpy_index3_in_audit_modules(self):
        # 静态断言：生产/验收路径不得再出现手写 `numpy()[:, :, 3]` / `[:, :, :3]`。
        import re
        offenders = []
        for path in ("tools/audit_psb.py", "tests/audit_system_integrity.py"):
            src = open(ROOT / path, encoding="utf-8").read()
            for i, line in enumerate(src.splitlines(), 1):
                # 排除 docstring/注释与共享模块里「有意取 K（index 3）」的黑版判定
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "numpy()" in line and (re.search(r"\]\[\s*:\s*,\s*:\s*3\s*\]", line)
                                          or re.search(r"\]\[\s*\.\.\.\s*,\s*3\s*\]", line)
                                          or re.search(r"\]\[\s*:\s*,\s*:\s*:3\s*\]", line)):
                    # ⑦ 黑版判定里「有意取 K」在 psd_layer_io 之外的代码不得出现
                    if "K" not in line and "k_channel" not in line and "黑版" not in line:
                        offenders.append(f"{path}:{i}: {stripped}")
        self.assertEqual(offenders, [],
                         "发现散落的硬编码通道索引（应改用 psd_layer_io 共享入口）：\n"
                         + "\n".join(offenders))


class TestLayerAlphaRealCMYKPSB(unittest.TestCase):
    """端到端回归：真实编译的 CMYK(PLATE) 产物，layer_alpha 取到真实 Alpha 而非 K。

    实测（本测试）：0% 黑墨 → 磁盘 K 通道恒为 255 → 旧 `[:, :, 3]` 口径覆盖 100%，
    而 layer_alpha 取真实 transparency，覆盖应为实际掩码比例。
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        import os
        from engine.codecs_accelerator import install_psb_codec_accelerator
        install_psb_codec_accelerator()
        from engine.core.models import LayerDescriptor, ProcessingContext
        from engine.core.psd_compiler import PsdCompiler
        from psd_tools import PSDImage

        W, H = 64, 48
        ctx = ProcessingContext(ppi=150.0, canvas_px=(W, H))
        ctx.output_mode = "PLATE"
        cmyk = np.zeros((4, H, W), np.uint8)
        cmyk[0], cmyk[1], cmyk[2], cmyk[3] = 30, 60, 120, 0  # K=0% 黑墨
        a1 = np.zeros((H, W), np.uint8)
        a1[10:38, 10:54] = 255  # 部分覆盖：28×44
        ctx.layers.append(LayerDescriptor(name="01_CMYK", layer_type="print_cmyk",
                                           cmyk_channels=cmyk, alpha=a1, bbox=(0, 0, W, H)))
        d = tempfile.mkdtemp()
        cls._psb = os.path.join(d, "one.psb")
        PsdCompiler.compile_psd(ctx, cls._psb, output_mode="PLATE")
        cls._psd_obj = PSDImage.open(cls._psb)
        cls._W, cls._H = W, H

    def test_real_cmyk_layer_has_5_channels(self):
        ly = list(self._psd_obj)[0]
        self.assertEqual(ly.numpy().shape[2], 5, "CMYK 层应为 5 通道（C,M,Y,K,Alpha）")

    def test_layer_alpha_returns_true_mask_not_k(self):
        from engine.core.psd_layer_io import layer_alpha
        ly = list(self._psd_obj)[0]
        al = layer_alpha(ly)
        true_cov = (28 * 44) / (self._H * self._W)
        self.assertAlmostEqual(float((al > 0.03).mean()), true_cov, places=4,
                               msg="应取真实 Alpha（部分覆盖），而非 K 通道（旧口径 100%）")
        # 对照：旧口径读到的是 K（0% 黑墨→磁盘 255→全覆盖）
        a = ly.numpy()
        old_cov = float((a[:, :, 3] > 0.03).mean())
        self.assertAlmostEqual(old_cov, 1.0, places=4,
                                msg="旧口径在 CMYK 上确实误取 K 通道=全画布")


if __name__ == "__main__":
    unittest.main()
