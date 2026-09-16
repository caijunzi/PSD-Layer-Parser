# -*- coding: utf-8 -*-
"""提交前自检（preflight）—— 2026-09-16 新增。

为什么需要
----------
本项目踩过的三类坑都不该再靠"人记得":
  1. **静默降级**：`icc_path` 缺失 → PLATE 线静默走朴素转换（K≡0 无黑版）；
     ⑦ 指标口径写反使其被掩盖。→ 门禁：`tools/audit_degradation.py`（P0 必须 0）
  2. **行尾噪音**：曾用"统一转 CRLF"脚本造成 15 个文件整文件等量 +/- diff，
     真实改动被淹没。→ 门禁：逐文件比对**行尾种类**与 HEAD 是否一致
  3. **测试未跑全**：引擎 / WebUI 两套必须都跑。

四道门
------
| # | 检查 | 失败含义 |
|---|---|---|
| 1 | 静默降级审计 P0 | 存在"配置缺失/异常被吞 → 产出静默退化"的风险点 |
| 2 | 行尾一致性（相对 HEAD） | 某文件的 LF/CRLF 被整体翻转（噪音 diff） |
| 3 | 引擎全量 pytest | 引擎回归 |
| 4 | WebUI 全量 unittest | 后端回归 |
| 5 | 前端 `tsc --noEmit`（可选） | 前端类型错误（node 不可用时**披露并跳过**，不静默） |

用法
----
    python scripts/preflight.py            # 全量（含两套测试 + 前端类型检查）
    python scripts/preflight.py --fast     # 只跑静态门禁（1/2），秒级

退出码：0 = 全通过；非 0 = 有失败项（供 pre-commit 钩子直接使用）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
#: 长跑引擎/测试时须停用 WorkBuddy 的 safe-delete 钩子（按 turn 累计删除数会终止进程）
ENV = {**os.environ, "CODEBUDDY_SAFE_DELETE_ENABLED": "0", "PYTHONIOENCODING": "utf-8"}

FAILS: list[str] = []
SKIPS: list[str] = []


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ENV)
    tail = (r.stdout or "") + (r.stderr or "")
    print("\n".join(tail.splitlines()[-14:]) if tail.strip() else "(无输出)")
    return r


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def _ending_kind(data: bytes) -> str:
    if not data:
        return "EMPTY"
    crlf, lf = data.count(b"\r\n"), data.count(b"\n")
    if crlf == lf and lf:
        return "CRLF"
    if crlf == 0:
        return "LF"
    return "MIX"


def gate_degradation() -> None:
    r = _run([PY, "tools/audit_degradation.py"], "门禁 1/5 · 静默降级审计")
    if r.returncode != 0:
        FAILS.append("静默降级审计存在 P0 阻断项")
    else:
        print("  ✅ 无 P0")


def gate_line_endings() -> None:
    print("\n=== 门禁 2/5 · 行尾一致性（相对 HEAD） ===")
    tracked: set[str] = set()
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        tracked.update(x for x in _git(*args).splitlines() if x.strip())
    untracked = [x for x in _git("ls-files", "--others", "--exclude-standard").splitlines()
                 if x.strip()]
    if not tracked and not untracked:
        print("  无任何改动（工作区干净）")
        return
    bad: list[str] = []
    checked = new_files = binary = 0
    for rel in sorted(tracked):
        p = ROOT / rel
        if not p.is_file():
            continue
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(ROOT),
                              capture_output=True).stdout
        if not head:
            new_files += 1
            continue                      # 新文件：无历史可比
        cur_raw = p.read_bytes()
        if b"\x00" in cur_raw or b"\x00" in head:
            binary += 1
            continue
        k_cur, k_head = _ending_kind(cur_raw), _ending_kind(head)
        checked += 1
        if k_cur != k_head:
            bad.append(f"{rel}: {k_head} -> {k_cur}")
    print(f"  已比对 {checked} 个（新文件 {new_files} 跳过、二进制 {binary} 跳过、"
          f"未跟踪新增 {len(untracked)} 个）")
    if bad:
        print("  ❌ 以下文件的行尾被整体翻转（噪音 diff，请恢复其原有行尾）：")
        for b in bad:
            print("     -", b)
        FAILS.append(f"行尾翻转 {len(bad)} 个文件")
    else:
        print(f"  ✅ {checked} 个改动文件行尾与 HEAD 一致")


def gate_engine_tests() -> None:
    r = _run([PY, "-m", "pytest", "tests/", "-q"], "门禁 3/5 · 引擎全量测试")
    if r.returncode != 0:
        FAILS.append("引擎 pytest 未全通过")
    else:
        for l in (r.stdout or "").splitlines():
            if "passed" in l:
                print("  ✅", l.strip())
                break


def gate_webui_tests() -> None:
    r = _run([PY, "-m", "unittest", "discover", "-s", "webui/backend/tests",
              "-p", "test_*.py"], "门禁 4/5 · WebUI 全量测试")
    ok = r.returncode == 0
    joined = (r.stdout or "") + (r.stderr or "")
    if not ok:
        FAILS.append("WebUI unittest 未全通过")
    else:
        for l in joined.splitlines():
            if l.strip().startswith(("OK", "Ran ")):
                print("  ✅", l.strip())
    if "Ran 0 tests" in joined:
        FAILS.append("WebUI 测试疑似未真正执行（Ran 0 tests）")


def gate_frontend_types() -> None:
    fe = ROOT / "webui" / "frontend"
    print("\n=== 门禁 5/5 · 前端类型检查（tsc --noEmit） ===")
    if not (fe / "node_modules").is_dir():
        msg = "webui/frontend/node_modules 不存在 → 跳过前端类型检查（请先 npm install）"
        print("  ⚠️ ", msg)
        SKIPS.append(msg)
        return
    exe = "npx.cmd" if os.name == "nt" else "npx"
    r = subprocess.run([exe, "--no-install", "tsc", "--noEmit"], cwd=str(fe),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        print("\n".join(out.splitlines()[-14:]))
        FAILS.append("前端 tsc --noEmit 失败")
    else:
        print("  ✅ 类型检查通过", (f"（{out.splitlines()[-1][:80]}）" if out else ""))


def check_interpreter() -> bool:
    """先验解释器：必须能 import numpy/cv2/PIL。

    ⚠️ 本机 PATH 上的 `python` 可能是 WorkBuddy managed 3.13（**无 numpy**）→
    直接用它会得到 `ModuleNotFoundError: No module named 'numpy'` 这种**误导性失败**。
    本项目既定测试解释器是系统 Python 3.12.10。
    """
    print("\n=== 门禁 0/5 · 解释器依赖 ===")
    print("  解释器:", PY)
    r = subprocess.run([PY, "-c", "import numpy, cv2, PIL; print('ok')"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    if r.returncode == 0 and "ok" in (r.stdout or ""):
        print("  ✅ numpy / cv2 / PIL 可用")
        return True
    print("  ❌ 当前解释器缺少依赖：", (r.stderr or "").strip().splitlines()[-1:])
    print("     本项目必须使用系统 Python 3.12.10：")
    print("     C:/Users/CK/AppData/Local/Programs/Python/Python312/python.exe")
    print("     （WorkBuddy managed 3.13 无 numpy，会得到误导性的 ModuleNotFoundError）")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="提交前自检")
    ap.add_argument("--fast", action="store_true", help="只跑静态门禁（1/2）")
    args = ap.parse_args()

    print("=" * 74)
    print("  提交前自检 preflight  ·  根目录:", ROOT)
    print("=" * 74)

    if not check_interpreter():
        FAILS.append("解释器缺依赖（应使用系统 Python 3.12.10）")
        print("\n" + "=" * 74)
        print("❌ preflight 中止：解释器不可用，后续门禁无意义")
        print("=" * 74)
        return 1

    gate_degradation()
    gate_line_endings()
    if not args.fast:
        gate_engine_tests()
        gate_webui_tests()
        gate_frontend_types()

    print("\n" + "=" * 74)
    if SKIPS:
        print("跳过项（已披露，非静默）：")
        for s in SKIPS:
            print("  -", s)
    if FAILS:
        print(f"❌ preflight 失败（{len(FAILS)} 项）：")
        for f in FAILS:
            print("  -", f)
        print("=" * 74)
        return 1
    print("✅ preflight 全通过")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
