# -*- coding: utf-8 -*-
"""安全离线备份：git bundle（含全部分支与 tag）+ 完整性校验 + 真实恢复测试。

为什么不用 `git safe-backup` 别名：它依赖 date/basename（本机 Git Bash 精简环境缺失），
且会把 bundle 写进仓库工作区（有被误提交的风险）。本脚本：
  1) bundle 输出到**仓库外**的备份目录（避免误入库）
  2) `git bundle verify` 校验完整性与 SHA-1
  3) **真实恢复测试**：从 bundle clone 出临时仓库，比对 HEAD 提交号后清理
"""
import os
import shutil
import subprocess
import sys
import time

REPO = r"E:\CK\C-desk\CK-WORKS\PSD图层处理 - WorkBuddy"
BACKUP_DIR = r"E:\CK\C-desk\CK-WORKS\PSD图层处理 - WorkBuddy-backups"


def git(*args, cwd=REPO):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    fails = []

    def check(cond, label):
        print(("  ✅ " if cond else "  ❌ ") + label)
        if not cond:
            fails.append(label)

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bundle = os.path.join(BACKUP_DIR, "PSD图层处理-WorkBuddy-%s.bundle" % stamp)

    # 备份前状态（用**完整 SHA**，避免与恢复库比对时短/长不一致造成误报）
    rc, out = git("rev-parse", "HEAD")
    head = out.strip().split()[0] if out.strip() else "?"
    rc, out = git("status", "--porcelain")
    check(not out.strip(), "备份时工作区干净")
    print("  HEAD = %s" % head)

    print("\n[1] 生成 bundle（--all：全部分支与 tag）")
    rc, out = git("bundle", "create", bundle, "--all")
    check(rc == 0, "bundle 创建成功（exit=%d）%s" % (rc, out.strip()[:200]))
    size = os.path.getsize(bundle) / (1024 * 1024) if os.path.isfile(bundle) else 0
    check(size > 10, "bundle 体积合理：%.1f MB" % size)

    print("\n[2] 完整性校验")
    rc, out = git("bundle", "verify", bundle)
    print("  ", out.strip().replace("\n", "\n   ")[:400])
    check(rc == 0, "git bundle verify 通过")

    print("\n[3] 内容清单（分支/tag）")
    rc, out = git("bundle", "list-heads", bundle)
    print("  ", out.strip()[:400])
    check("refs/heads/main" in out or "refs/heads/master" in out, "包含主分支")

    print("\n[4] 真实恢复测试（从 bundle clone）")
    tmp = os.path.join(BACKUP_DIR, "_restore_test_%s" % stamp)
    rc, out = git("clone", "--quiet", bundle, tmp)
    check(rc == 0, "从 bundle clone 成功")
    rc, out = git("rev-parse", "HEAD", cwd=tmp)
    restored = out.strip().split()[0] if out.strip() else "?"
    check(restored == head, "恢复后 HEAD 一致（%s）" % restored[:12])
    rc, out = git("rev-parse", "HEAD", cwd=tmp)
    print("  恢复库最新提交:", out.strip()[:100])
    check(out.strip() == head, "恢复库 HEAD 与源一致（完整 SHA）")

    # 清理临时恢复库（safe-delete 钩子已用环境变量停用）
    shutil.rmtree(tmp, ignore_errors=True)
    check(not os.path.isdir(tmp), "临时恢复库已清理")

    print("\n" + "=" * 70)
    if fails:
        print("❌ 备份未完全通过（%d 项）：" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("✅ 离线备份完成并通过全部验证：")
    print("  ", bundle)
    print("   %.1f MB ｜ HEAD %s ｜ 已含恢复测试" % (size, head))
    return 0


if __name__ == "__main__":
    sys.exit(main())
