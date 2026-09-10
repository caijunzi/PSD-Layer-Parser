"""印前油墨总量（TAC）合规管理测试 —— ADR-007。

覆盖：
  - 超限像素被压制到限值以内，且未超限像素保持不变
  - K 保持中性（TAC 压制时优先削 CMY 而非 K，避免整体偏色）
  - MaxK 单通道限制
  - 策略来源可追溯（preset > 印刷条件 > 默认）
  - audit_tac 只读复核
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core.ink_limiter import (  # noqa: E402
    DEFAULT_CONDITION,
    PRINT_CONDITIONS,
    INK_MAX,
    TacPolicy,
    audit_tac,
    limit_ink,
    resolve_policy,
)


def _mk_cmyk(ink_pct: tuple[float, float, float, float], h: int = 4, w: int = 4) -> np.ndarray:
    """按目标墨量百分比构造磁盘反码 CMYK 数据。"""
    ink = np.array(ink_pct, np.float32) / 100.0 * INK_MAX
    raw = (INK_MAX - ink).astype(np.uint8)
    return np.repeat(raw[:, None, None], h, axis=1).repeat(w, axis=2)


def _tac_pct(cmyk_raw: np.ndarray) -> np.ndarray:
    return (INK_MAX - cmyk_raw.astype(np.float32)).sum(axis=0) / INK_MAX * 100.0


class TestLimitInk(unittest.TestCase):
    def test_over_limit_is_clamped(self):
        """400% 总墨量应被压到限值 300%。"""
        policy = TacPolicy(300.0, 96.0, "test")
        raw = _mk_cmyk((100, 100, 100, 100))          # TAC = 400%
        self.assertAlmostEqual(float(_tac_pct(raw).max()), 400.0, places=3)

        out, stats = limit_ink(raw, policy)
        self.assertLessEqual(float(_tac_pct(out).max()), 300.0 + 1e-3)
        self.assertTrue(stats["compliant"])
        self.assertAlmostEqual(stats["tac_before_max"], 400.0, places=2)
        self.assertLessEqual(stats["tac_after_max"], 300.0)

    def test_k_preserved_when_clamping(self):
        """TAC 压制应优先削 CMY、保留 K（避免中性灰偏移）。"""
        policy = TacPolicy(300.0, 100.0, "test")     # MaxK 放宽，专测 CMY 缩放
        raw = _mk_cmyk((100, 100, 100, 80))          # TAC = 380%
        out, _ = limit_ink(raw, policy)
        ink_out = INK_MAX - out.astype(np.float32)
        # K 应保持 80%
        self.assertAlmostEqual(float(ink_out[3].max()) / INK_MAX * 100, 80.0, places=1)
        # CMY 应被等比压缩：各占 (300-80)/3
        for i in range(3):
            self.assertAlmostEqual(float(ink_out[i].max()) / INK_MAX * 100,
                                   (300.0 - 80.0) / 3.0, places=1)

    def test_max_k_clamped(self):
        """K 单通道不得超过 max_k_pct。"""
        policy = TacPolicy(400.0, 95.0, "test")      # TAC 放宽，专测 MaxK
        raw = _mk_cmyk((0, 0, 0, 100))
        out, stats = limit_ink(raw, policy)
        ink_out = INK_MAX - out.astype(np.float32)
        self.assertLessEqual(float(ink_out[3].max()) / INK_MAX * 100, 95.0 + 1e-3)
        self.assertEqual(stats["k_clipped_pixels"], raw.shape[1] * raw.shape[2])

    def test_compliant_pixels_untouched(self):
        """未超限的像素不得被改动。"""
        policy = TacPolicy(300.0, 96.0, "test")
        raw = _mk_cmyk((20, 20, 20, 10))             # TAC = 70%
        out, stats = limit_ink(raw, policy)
        self.assertTrue(np.array_equal(out, raw))
        self.assertEqual(stats["tac_clipped_pixels"], 0)

    def test_mixed_image_partial_clip(self):
        """混合图：只有超限像素被改，比例统计正确。"""
        policy = TacPolicy(300.0, 96.0, "test")
        raw = np.concatenate([
            _mk_cmyk((100, 100, 100, 100), h=1, w=4),   # 超限 4 px
            _mk_cmyk((10, 10, 10, 10), h=1, w=4),       # 合规 4 px
        ], axis=1)
        out, stats = limit_ink(raw, policy)
        self.assertEqual(stats["tac_clipped_pixels"], 4)
        self.assertAlmostEqual(stats["tac_clipped_ratio"], 0.5, places=4)
        # 合规行的像素保持不变
        self.assertTrue(np.array_equal(out[:, 1, :], raw[:, 1, :]))

    def test_shape_validation(self):
        policy = TacPolicy(300.0, 96.0, "test")
        with self.assertRaises(ValueError):
            limit_ink(np.zeros((3, 4, 4), np.uint8), policy)


class TestPolicyResolution(unittest.TestCase):
    def test_preset_wins(self):
        p = resolve_policy(condition="newsprint", limit_pct=280.0)
        self.assertEqual(p.limit_pct, 280.0)
        self.assertIn("preset", p.source)

    def test_condition_used(self):
        p = resolve_policy(condition="iso_uncoated")
        self.assertEqual(p.limit_pct, PRINT_CONDITIONS["iso_uncoated"].limit_pct)
        self.assertEqual(p.condition_id, "iso_uncoated")

    def test_default_when_unknown(self):
        p = resolve_policy(condition="no_such_condition")
        self.assertEqual(p.condition_id, DEFAULT_CONDITION)

    def test_icc_missing_falls_back(self):
        p = resolve_policy(icc_path="profiles/not_exists.icc")
        self.assertEqual(p.condition_id, DEFAULT_CONDITION)

    def test_all_conditions_have_source(self):
        """每个工艺常量都必须可追溯（禁止无出处的魔法数字）。"""
        for cid, pol in PRINT_CONDITIONS.items():
            self.assertTrue(pol.source, f"{cid} 缺少来源说明")
            self.assertGreater(pol.limit_pct, 0)
            self.assertGreater(pol.max_k_pct, 0)


class TestAudit(unittest.TestCase):
    def test_audit_does_not_modify(self):
        policy = TacPolicy(300.0, 96.0, "test")
        raw = _mk_cmyk((100, 100, 100, 100))
        before = raw.copy()
        res = audit_tac(raw, policy)
        self.assertTrue(np.array_equal(raw, before), "audit_tac 不得修改输入")
        self.assertFalse(res["compliant"])
        self.assertAlmostEqual(res["tac_max"], 400.0, places=2)

    def test_audit_compliant_passes(self):
        policy = TacPolicy(300.0, 96.0, "test")
        raw = _mk_cmyk((50, 50, 50, 50))             # TAC = 200%
        res = audit_tac(raw, policy)
        self.assertTrue(res["compliant"])


if __name__ == "__main__":
    unittest.main()
