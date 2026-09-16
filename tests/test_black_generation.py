# -*- coding: utf-8 -*-
"""黑版生成曲线（GCR 按品类调优）回归 —— 2026-09-16 新增。

覆盖：
  1. 恒等即**零拷贝**（未配置品类产物字节不变，保护 RK-16）
  2. gain/gamma/shift 对 K 墨量的预期影响
  3. **CMY 等量补偿**（GCR 灰量转移）—— K 增 delta，则 C/M/Y 各减 delta
  4. 总墨量近似守恒（不因曲线额外推高 TAC）
  5. 越界配置被夹取；enabled=false / 缺失 → 恒等
  6. 形状校验
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.black_generation import (  # noqa: E402
    IDENTITY,
    BlackGenerationPolicy,
    apply_black_generation,
    resolve_black_generation,
)


def _mk(c=100, m=100, y=100, k=100, h=4, w=4):
    """构造磁盘反码 CMYK（入参为**墨量%**，函数内部转反码）。"""
    out = np.empty((4, h, w), np.uint8)
    for i, pct in enumerate((c, m, y, k)):
        out[i] = int(round(255 - pct / 100.0 * 255))
    return out


class TestIdentity(unittest.TestCase):
    def test_identity_is_zero_copy(self):
        raw = _mk()
        out, stats = apply_black_generation(raw, IDENTITY)
        self.assertIs(out, raw, "恒等策略必须零拷贝返回入参（保 RK-16）")
        self.assertFalse(stats["black_gen_applied"])

    def test_none_policy_is_identity(self):
        raw = _mk()
        out, stats = apply_black_generation(raw, None)
        self.assertIs(out, raw)
        self.assertFalse(stats["black_gen_applied"])

    def test_resolve_missing_or_disabled_is_identity(self):
        self.assertTrue(resolve_black_generation(None).is_identity)
        self.assertTrue(resolve_black_generation({}).is_identity)
        self.assertTrue(resolve_black_generation({"enabled": False, "k_gain": 1.5}).is_identity)
        self.assertTrue(resolve_black_generation({"k_gain": 1.0, "k_gamma": 1.0, "k_shift_pct": 0.0}).is_identity)


class TestGAffects(unittest.TestCase):
    def test_gain_increases_k_and_reduces_cmy(self):
        raw = _mk(c=50, m=50, y=50, k=50)
        out, stats = apply_black_generation(raw, BlackGenerationPolicy(k_gain=1.2))
        ink_k_before = 255 - raw[3].astype(int)
        ink_k_after = 255 - out[3].astype(int)
        self.assertGreater(ink_k_after.mean(), ink_k_before.mean(), "gain>1 应加深 K")
        # CMY 等量减少（K 增 delta → 各减 delta）
        delta = ink_k_after - ink_k_before
        for ch in range(3):
            before = 255 - raw[ch].astype(int)
            after = 255 - out[ch].astype(int)
            self.assertLessEqual(after.mean(), before.mean() + 1e-6, "K 增时 CMY 应减少")
            self.assertLessEqual(abs((before.mean() - after.mean()) - delta.mean()), 1.5,
                                 "CMY 减少量应≈K 增量（GCR 等量补偿）")
        self.assertTrue(stats["black_gen_applied"])

    def test_gain_lt_one_reduces_k(self):
        raw = _mk(c=50, m=50, y=50, k=50)
        out, _ = apply_black_generation(raw, BlackGenerationPolicy(k_gain=0.8))
        self.assertLess((255 - out[3].astype(int)).mean(),
                        (255 - raw[3].astype(int)).mean(), "gain<1 应减淡 K")

    def test_gamma_attenuates_midtone_but_keeps_endpoints(self):
        # 中间调 50%：gamma>1 → ink^gamma 更小 → K 墨量下降（黑版中间调变淡）
        mid = _mk(c=0, m=0, y=0, k=50)
        out, _ = apply_black_generation(mid, BlackGenerationPolicy(k_gamma=2.0))
        self.assertLess((255 - out[3].astype(int)).mean(), (255 - mid[3].astype(int)).mean())
        # 端点不变
        white = _mk(k=0)
        o_w, _ = apply_black_generation(white, BlackGenerationPolicy(k_gamma=2.0))
        self.assertEqual(int(o_w[3].flat[0]), int(white[3].flat[0]))
        black = _mk(k=100)
        o_b, _ = apply_black_generation(black, BlackGenerationPolicy(k_gamma=2.0))
        self.assertEqual(int(o_b[3].flat[0]), int(black[3].flat[0]))

    def test_shift_moves_k(self):
        raw = _mk(k=50)
        out, _ = apply_black_generation(raw, BlackGenerationPolicy(k_shift_pct=10.0))
        self.assertGreater((255 - out[3].astype(int)).mean(), (255 - raw[3].astype(int)).mean())


class TestConservation(unittest.TestCase):
    def test_tac_approximately_preserved(self):
        """灰量在 K 与 CMY 间等量转移 → 总墨量近似不变（不应额外推高 TAC）。"""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, (4, 32, 32)).astype(np.uint8)
        out, stats = apply_black_generation(raw, BlackGenerationPolicy(k_gain=1.3))
        ink_before = (255 - raw.astype(np.float32)).sum(axis=0).mean()
        ink_after = (255 - out.astype(np.float32)).sum(axis=0).mean()
        self.assertLessEqual(abs(ink_after - ink_before) / 255.0, 8.0,
                             "总墨量变化应很小（等量转移 + 量化）")

    def test_clamping_no_overflow(self):
        raw = np.zeros((4, 4, 4), np.uint8)   # 满墨
        out, _ = apply_black_generation(raw, BlackGenerationPolicy(k_gain=2.0))
        self.assertTrue((out <= 255).all() and (out >= 0).all())
        self.assertTrue((out.dtype == np.uint8))


class TestValidation(unittest.TestCase):
    def test_out_of_range_clamped_and_noted(self):
        p = resolve_black_generation({"k_gain": 9.9, "k_gamma": -1.0, "k_shift_pct": 999.0})
        self.assertLessEqual(p.k_gain, 2.0)
        self.assertGreaterEqual(p.k_gamma, 0.1)
        self.assertLessEqual(p.k_shift_pct, 100.0)
        self.assertIn("夹取", p.source)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            apply_black_generation(np.zeros((3, 4, 4), np.uint8), BlackGenerationPolicy(k_gain=1.2))


class TestBlackGenerationWiring(unittest.TestCase):
    """wiring 断言：模块必须接生产调用点，否则"单测全绿 ≠ 功能可用"。"""

    def test_psb_builder_accepts_and_applies_black_gen(self):
        import inspect
        from engine.psb_builder import UniversalPSBBuilder
        sig = inspect.signature(UniversalPSBBuilder.__init__)
        self.assertIn("black_gen", sig.parameters, "builder 必须接受 black_gen")
        src = inspect.getsource(UniversalPSBBuilder._to_cmyk_limited)
        self.assertIn("apply_black_generation", src,
                      "必须在 ICC 分色后施加黑版曲线")
        # 顺序：黑版曲线须在 TAC 压制之前
        self.assertLess(src.index("apply_black_generation"), src.index("limit_ink"),
                        "黑版曲线应在 TAC 压制之前")

    def test_engine_resolves_and_passes_black_gen(self):
        src = (ROOT / "run_universal_engine.py").read_text(encoding="utf-8")
        self.assertIn("resolve_black_generation", src, "引擎必须解析 preset.black_generation")
        self.assertIn("black_gen=black_gen", src, "引擎必须把策略传给 builder")
        self.assertIn('"black_generation"', src, "manifest 必须披露 black_generation")


if __name__ == "__main__":
    unittest.main()
