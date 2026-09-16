# -*- coding: utf-8 -*-
"""静默降级门禁（2026-09-16 新增）。

由来（真实事故）：`icc_path` 缺失 → PLATE 线**静默**走朴素 RGB→CMYK（K≡0、无黑版、无色管），
而本应抓住它的 ⑦ `k_channel_nonzero_pct` 口径又写反（恒 ~100%）→ "配置缺失→产出退化→审计放行"
整条链无人报警，长期潜伏。本测试把该模式做成**棘轮门禁**：

- **P0 必须为 0**：preset 完整性（plate 必须有有效 icc_path/合法 print_condition）、
  裸 `except:`、生产文件完全不认识 `imread_unicode` 却直接 `cv2.imread`。
- **P1-1 必须全部在已复核白名单内**：新增一处未复核的"静默吞异常"即测试失败，
  强制新增者写下理由（要么修复使其披露，要么登记进白名单并说明）。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_degradation as AD  # noqa: E402


class TestNoP0Degradation(unittest.TestCase):
    def test_preset_integrity(self):
        issues = AD.check_preset_integrity()
        self.assertEqual(
            issues, [],
            "preset 完整性 P0：plate/both 的 preset 必须有有效 icc_path 与合法 print_condition\n"
            + "\n".join(f"  {i['file']}: {i['msg']}" for i in issues))

    def test_no_bare_except(self):
        p0, _ = AD.check_exception_swallow()
        bare = [i for i in p0 if i["code"] == "P0-2"]
        self.assertEqual(bare, [], "禁止裸 except:\n"
                         + "\n".join(f"  {i['file']}:{i['line']}" for i in bare))

    def test_no_unsafe_imread_in_unanware_files(self):
        p0, _ = AD.check_imread_safety()
        self.assertEqual(p0, [],
                         "以下生产文件完全未使用 imread_unicode 却直接 cv2.imread"
                         "（中文路径会静默返回 None）：\n"
                         + "\n".join(f"  {i['file']}:{i['line']} {i['msg']}" for i in p0))


class TestSilentSwallowRatchet(unittest.TestCase):
    def test_all_silent_swallows_are_reviewed(self):
        _, p1 = AD.check_exception_swallow()
        silent = [i for i in p1 if i["code"] == "P1-1"]
        unreviewed = [i for i in silent
                      if (i["file"], i["line"]) not in AD.EXCEPT_PASS_ALLOWLIST]
        self.assertEqual(
            unreviewed, [],
            "发现未复核的『静默吞异常』（新增项必须修复使其披露，或登记进 "
            "tools/audit_degradation.EXCEPT_PASS_ALLOWLIST 并写明理由）：\n"
            + "\n".join(f"  {i['file']}:{i['line']}  {i['msg']}" for i in unreviewed))

    def test_allowlist_entries_still_exist(self):
        """白名单行号漂移检测：条目失效说明代码已变动，需重新复核。"""
        _, p1 = AD.check_exception_swallow(apply_allowlist=False)
        present = {(i["file"], i["line"]) for i in p1 if i["code"] == "P1-1"}
        stale = [k for k in AD.EXCEPT_PASS_ALLOWLIST if k not in present]
        self.assertEqual(stale, [],
                         "以下白名单条目已失效（行号漂移或代码已改），请重新复核：\n"
                         + "\n".join(f"  {f}:{ln}" for f, ln in stale))


if __name__ == "__main__":
    unittest.main()
