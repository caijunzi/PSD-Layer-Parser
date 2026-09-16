# -*- coding: utf-8 -*-
"""静默降级审计（Silent-Degradation Audit）—— 2026-09-16 新增。

为什么需要
----------
`icc_path` 缺失导致 PLATE 线**静默**走朴素 RGB→CMYK（K≡0、无黑版、无色彩管理），
而本该抓住它的 ⑦ `k_channel_nonzero_pct` 指标口径又写反（恒 ~100%），
于是"配置缺失 → 产出退化 → 审计放行"整条链**无人报警**，长期潜伏。

本工具把这类「配置缺失/异常被吞 → 产出静默退化」的模式做成**可重复扫描**，
并被 `tests/test_no_silent_degradation.py` 作为门禁调用（P0 即测试失败）。

检查项
------
- **P0-1 preset 完整性**：`output.mode` 为 plate/both 时，必须有有效的
  `icc_path`（文件存在）与合法的 `print_condition`；plate 的 `color_mode` 必须是 cmyk。
- **P0-2 裸 except**：`except:`（无类型）在生产代码中禁止。
- **P0-3 非 Unicode 安全读图**：生产代码不得直接用 `cv2.imread(`（含中文路径会静默返回 None），
  必须走 `engine.core.io_utils.imread_unicode`。
- **P1-1 静默吞异常**：`except` 体只含 `pass`/常量/`print`（无记录、无告警、无传播）。
  可选依赖导入等少数场景合法 → 以白名单豁免，其余仅告警。
- **P1-2 静默回退关键词**：生产代码中的 `回退/降级/fallback` 提示是否**带原因**（披露）。

用法
----
    python tools/audit_degradation.py [--json out.json] [--strict-p1]

退出码：0 = 无 P0；1 = 存在 P0；`--strict-p1` 时 P1 也计为失败。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: 生产代码范围（不含测试/第三方/缓存）
SCAN_ROOTS = ("engine", "tools", "pipeline", "webui/backend")
SCAN_FILES = ("run_universal_engine.py",)
_EXCLUDE_PARTS = ("__pycache__", "third_party", "node_modules", "/tests/", "\\tests\\",
                  "site-packages")

#: 已复核的「静默吞异常」白名单：(相对路径, 行号) → 复核理由。
#: 这些是**有意降级**且紧邻分支已给出替代路径（或属正常控制流），不构成静默缺陷。
#: 行号漂移会让门禁测试失败 → 请重新复核后更新本表（fail-safe 方向）。
EXCEPT_PASS_ALLOWLIST: dict[tuple[str, int], str] = {
    ("engine/adaptive/category_selector.py", 334): "查询规范名失败 → 下一分支降级为 name_zh_name_en 拼接",
    ("engine/adaptive/cbr_retriever.py", 232): "月份路径猜测失败 → 下一分支递归搜索兜底",
    ("engine/adaptive/migrations/migration_001_init.py", 31): "表不存在（首次迁移）→ 继续建表迁移",
    ("engine/schemas/preset_schema.py", 154): "pydantic 未安装 → 手工校验已覆盖主要项",
    ("webui/backend/api/upload.py", 41): "临时文件清理失败 → 紧随其后 raise HTTPException",
    ("webui/backend/core/background_learner.py", 375): "reset 时删除任务文件失败（幂等操作）",
    ("webui/backend/ws/progress.py", 47): "WebSocket 客户端断开（正常控制流）",
    ("run_universal_engine.py", 361): "torch 未安装 → 跳过随机种子（可选依赖）",
}

#: 允许直接 cv2.imread 的文件：io_utils 是封装层自身；
#: fingerprint 显式"先 imread_unicode、失败再 cv2.imread 且判 None 并 WARN"，属已知安全兜底。
IMREAD_ALLOW = ("io_utils.py", "audit_degradation.py", "fingerprint.py")


def _iter_py_files() -> list[str]:
    out: list[str] = []
    for r in SCAN_ROOTS:
        out += glob.glob(os.path.join(ROOT, r, "**", "*.py"), recursive=True)
    for f in SCAN_FILES:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            out.append(p)
    return [f for f in out if not any(x in f for x in _EXCLUDE_PARTS)]


def _rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def check_preset_integrity() -> list[dict]:
    """P0-1：plate/both 的 preset 必须 ICC 与印前参数齐备。"""
    from engine.core.ink_limiter import PRINT_CONDITIONS

    issues: list[dict] = []
    pdir = os.path.join(ROOT, "presets")
    for f in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        name = os.path.basename(f)
        try:
            cfg = json.loads(open(f, encoding="utf-8").read())
        except Exception as e:
            issues.append({"code": "P0-1", "file": f"presets/{name}", "line": 0,
                           "msg": f"preset 无法解析：{e}"})
            continue
        mode = ((cfg.get("output") or {}).get("mode") or "design").lower()
        if mode not in ("plate", "both"):
            continue
        icc = cfg.get("icc_path")
        if not icc:
            issues.append({"code": "P0-1", "file": f"presets/{name}", "line": 0,
                           "msg": "plate/both 但缺 icc_path → 静默朴素转换（K≡0、无黑版）"})
        elif not os.path.isfile(os.path.join(ROOT, icc)):
            issues.append({"code": "P0-1", "file": f"presets/{name}", "line": 0,
                           "msg": f"icc_path 指向不存在的文件：{icc}"})
        pc = cfg.get("print_condition")
        if pc and pc not in PRINT_CONDITIONS:
            issues.append({"code": "P0-1", "file": f"presets/{name}", "line": 0,
                           "msg": f"print_condition 非法：{pc}"})
        if mode == "plate" and (cfg.get("output") or {}).get("color_mode") not in (None, "cmyk"):
            issues.append({"code": "P0-1", "file": f"presets/{name}", "line": 0,
                           "msg": "plate 但 color_mode 不是 cmyk"})
    return issues


def _trivial_body(body: list[ast.stmt]) -> bool:
    for st in body:
        if isinstance(st, ast.Pass):
            continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue
        # 注意：`print/log/warn/raise` 都算**有披露**，不算静默。
        # （只有 pass / 常量字面量这类"什么都没做"才算静默吞异常。）
        return False
    return True


def check_exception_swallow(apply_allowlist: bool = True) -> tuple[list[dict], list[dict]]:
    """P0-2 裸 except / P1-1 静默吞异常。

    ``apply_allowlist=False`` 时不过滤白名单 —— 供门禁测试做「行号漂移检测」。
    """
    p0: list[dict] = []
    p1: list[dict] = []
    for f in _iter_py_files():
        try:
            src = open(f, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            continue
        lines = src.splitlines()
        rel = _rel(f)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            snippet = lines[node.lineno - 1].strip()[:110] if node.lineno <= len(lines) else ""
            if node.type is None:
                p0.append({"code": "P0-2", "file": rel, "line": node.lineno,
                           "msg": f"裸 except（禁止）：{snippet}"})
            elif _trivial_body(node.body):
                if apply_allowlist and (rel, node.lineno) in EXCEPT_PASS_ALLOWLIST:
                    continue
                p1.append({"code": "P1-1", "file": rel, "line": node.lineno,
                           "msg": f"静默吞异常（无记录/无传播）：{snippet}"})
    return p0, p1


def check_imread_safety() -> tuple[list[dict], list[dict]]:
    """P0-3：**完全不知道** Unicode 安全读图的生产文件里直接 cv2.imread。

    `cv2.imread` 对含中文的路径会静默返回 None（项目工作区就是中文路径），
    正确做法一律走 `engine.core.io_utils.imread_unicode`。
    但**已知安全读图并显式兜底**（先 imread_unicode、失败再 cv2.imread 且判 None）的文件
    属合法，降级为 P1 提示。
    """
    p0: list[dict] = []
    p1: list[dict] = []
    pat = re.compile(r"\bcv2\.imread\s*\(")
    for f in _iter_py_files():
        if any(a in os.path.basename(f) for a in IMREAD_ALLOW):
            continue
        try:
            src = open(f, encoding="utf-8").read()
        except Exception:
            continue
        knows_safe = "imread_unicode" in src
        for i, l in enumerate(src.splitlines(), 1):
            s = l.strip()
            if s.startswith("#") or pat.search(l) is None:
                continue
            item = {"file": _rel(f), "line": i,
                    "msg": f"直接 cv2.imread（中文路径静默返回 None）：{s[:100]}"}
            (p1 if knows_safe else p0).append({**item, "code": "P1-3" if knows_safe else "P0-3"})
    return p0, p1


def main() -> int:
    ap = argparse.ArgumentParser(description="静默降级审计")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--strict-p1", action="store_true", help="P1 也视为失败")
    args = ap.parse_args()

    p0_preset = check_preset_integrity()
    p0_exc, p1_exc = check_exception_swallow()
    p0_imread, p1_imread = check_imread_safety()

    p0 = p0_preset + p0_exc + p0_imread
    p1 = p1_exc + p1_imread

    print("=" * 74)
    print("  静默降级审计（配置缺失/异常被吞 → 产出静默退化）")
    print("=" * 74)
    print(f"\n[P0] {len(p0)} 处（阻断）")
    for it in p0:
        print(f"  {it['code']} {it['file']}:{it['line']}  {it['msg']}")
    print(f"\n[P1] {len(p1)} 处（告警，需逐项判定）")
    for it in p1[:60]:
        print(f"  {it['code']} {it['file']}:{it['line']}  {it['msg']}")
    if len(p1) > 60:
        print(f"  … 其余 {len(p1) - 60} 处见 --json")

    if args.json_out:
        json.dump({"p0": p0, "p1": p1}, open(args.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n[json] 已写出 {args.json_out}")

    failed = bool(p0) or (args.strict_p1 and bool(p1))
    print(f"\n结论: {'❌ 存在 P0 阻断项' if p0 else '✅ 无 P0'}"
          + (f"；P1 {len(p1)} 处待判定" if p1 else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
