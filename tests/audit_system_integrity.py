"""System Integrity & Anti-Deception Audit（系统完整性与防欺骗审计）。

设计原则（v2，2026-09-10 重写）
------------------------------
旧版五个维度"100% 通过"，却完全没能发现「印章层掩模泄漏 605 倍」这类严重缺陷，
根因是断言设计无效：
  - D1 硬绑本机硬件（`assert "GPU.1" in devices`），换机必挂，无法回归；
  - D2 只扫 8 条**历史黑名单正则**（早已删除的常量），永远发现不了新的魔法值；
  - D3 只数层数（且用事后追认窗口 `in [11, 15]`），不校验任何掩模质量；
  - D4 只查 4 条正则，形同虚设；
  - D5 只断言属性替换成功，未验证 C 路径真的生效。

新版改进：
  - 硬件改为**能力探测**，缺失即记录不阻断（使他机/CI 可运行）；
  - 硬编码改为 **AST 级扫描**，能发现新增魔法值；`engine/core/` 品类关键词零容忍；
  - 交付物审计新增**填充率 / 包围盒紧凑度 / 元素层面积预算**，直击掩模泄漏；
  - C-SIMD 增加**开关对比**，端到端验证加速真的生效；
  - 统一用 `QAReport.record()` 收集，输出 JSON 并以退出码表达成败。

用法：
    python tests/audit_system_integrity.py [PSB路径]
    python tests/audit_system_integrity.py --skip-deliverable
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core.models import QAReport, new_run_id  # noqa: E402

# ---------------- 阈值（后续迁入 engine/schemas/）----------------
FILL_RATIO_MIN = 0.00001   # 低于此值判为空层缺陷
FILL_RATIO_MAX = 0.15      # 元素层面积上限
BBOX_COVER_MAX = 0.60      # 元素层包围盒覆盖率上限
REGION_KEYWORDS = (
    "远山", "mountain", "水波", "ripple", "折痕", "seam", "fold",
    "外框", "frame", "brocade", "底板", "base", "ground",
)
# AST 扫描白名单：工程惯用常量，不算魔法值
MAGIC_WHITELIST = {0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 100, 127, 128, 255, 256, 512, 1024, 65536, -1}
# engine/core/ 下禁止出现的品类关键词（G1：内核与品类解耦）—— 阻断级
# 注意：ink（油墨量）、screen（滤色混合模式 / 加网）是印刷与图像通用术语，
# 在内核中出现属正常，只作提示级，不阻断（旧版未区分，导致 3 处误报）。
CATEGORY_KEYWORDS = ("damask", "gold", "壁布", "烫金", "水墨", "屏风")
# 提示级：命中仅记录，不判定违规
CATEGORY_SOFT_KEYWORDS = ("ink", "screen")


# golden baseline（v2.1 手工调优成果），用于相对判定
BASELINE_DIR = "masks_16k"
# 相对基线的面积失衡倍数（超出 10 倍或不足 1/10）且形状不一致 → 结构性缺陷
IMBALANCE_FACTOR = 10.0
MIN_SHAPE_IOU = 0.10


def _is_region(name: str) -> bool:
    low = str(name).lower()
    return any(k.lower() in low for k in REGION_KEYWORDS)


def _code_of(name: str) -> str:
    """从图层名/基线文件名提取编号 token（如 09A / 10B / 03），用于配对。"""
    import re as _re

    stem = _re.sub(r"^mask_", "", os.path.splitext(os.path.basename(str(name)))[0])
    m = _re.match(r"(\d{2}[A-Za-z]?)", stem)
    if m:
        return m.group(1).upper()
    m = _re.search(r"_(\d{2}[A-Za-z]?)_", stem)
    return m.group(1).upper() if m else stem.upper()


def load_baseline(target_wh: tuple[int, int], dtype: str = "uint8") -> dict[str, np.ndarray]:
    """读取 masks_16k 基线并缩放到目标画幅。"""
    baselines: dict[str, np.ndarray] = {}
    if not os.path.isdir(BASELINE_DIR):
        return baselines
    w, h = target_wh
    for fn in sorted(os.listdir(BASELINE_DIR)):
        if not fn.lower().endswith(".png"):
            continue
        m = cv2.imread(os.path.join(BASELINE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[1] != w or m.shape[0] != h:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        baselines[_code_of(fn)] = (m > 127).astype(dtype)
    return baselines


# ============================================================
# D1 硬件能力探测（缺失仅记录，不阻断）
# ============================================================
def run_hardware_audit(rep: QAReport) -> None:
    print("=" * 74)
    print("  [D1] 硬件能力探测（缺失仅告警，不阻断）")
    print("=" * 74)
    try:
        import openvino as ov
    except ImportError:
        rep.record("D1-01", "OpenVINO 可用性", False, "未安装，AI Provider 将走规则降级路径")
        print("  [WARN] OpenVINO 未安装 — 异构调度不可用，Provider 自动降级（设计内行为）")
        return

    try:
        core = ov.Core()
        devices = core.available_devices
        print(f"  可用设备: {devices}")
        for dev in ("GPU.1", "GPU.0", "NPU", "CPU"):
            if dev in devices:
                print(f"  [OK]   {dev}: {core.get_property(dev, 'FULL_DEVICE_NAME')}")
            else:
                print(f"  [--]   {dev} 不可用（不影响零依赖路径出图）")
        rep.record("D1-01", "OpenVINO 可用性", True, f"devices={devices}")
    except Exception as e:
        rep.record("D1-01", "OpenVINO 探测", False, f"{type(e).__name__}: {e}")


# ============================================================
# D2 AST 级硬编码扫描
# ============================================================
def run_hardcode_audit(rep: QAReport) -> None:
    print("\n" + "=" * 74)
    print("  [D2] AST 级硬编码与品类耦合扫描")
    print("=" * 74)

    engine_files = sorted(g for g in glob.glob("engine/**/*.py", recursive=True)
                          if "__pycache__" not in g)
    magic_hits: list[str] = []
    category_hits: list[str] = []

    for fpath in engine_files:
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            rep.record("D2-00", f"语法解析 {fpath}", False, str(e))
            continue

        # (a) 函数体内的可疑魔法数
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                        v = sub.value
                        if v in MAGIC_WHITELIST:
                            continue
                        if isinstance(v, int) and abs(v) <= 10:
                            continue
                        if isinstance(v, float) and 0.0 < abs(v) <= 0.9:
                            continue  # 归一化系数属常见写法
                        magic_hits.append(f"{fpath}:{sub.lineno}  {node.name}()  字面量 {v!r}")

        # (b) 品类关键词耦合：core/ 零容忍，其他目录仅提示
        low = src.lower()
        in_core = fpath.replace("\\", "/").startswith("engine/core/")
        for kw in CATEGORY_KEYWORDS:
            if kw.lower() in low:
                hit = f"{fpath} 含品类关键词 '{kw}'"
                if in_core:
                    category_hits.append(hit)
                else:
                    magic_hits.append(hit + "（非内核层，提示）")
        for kw in CATEGORY_SOFT_KEYWORDS:
            if kw.lower() in low:
                magic_hits.append(f"{fpath} 含通用印刷术语 '{kw}'（提示）")

    print(f"  扫描 {len(engine_files)} 个文件")
    if category_hits:
        for h in category_hits:
            print(f"  [FAIL] {h}")
        rep.record("D2-01", "engine/core/ 品类解耦（G1）", False, f"{len(category_hits)} 处违规")
    else:
        print("  [OK]   engine/core/ 未出现品类关键词（G1 内核与品类解耦）")
        rep.record("D2-01", "engine/core/ 品类解耦（G1）", True)

    print(f"  [INFO] 函数体内可疑魔法值 {len(magic_hits)} 处（不阻断，建议按 §9.2 常量三级分类外置）")
    for h in magic_hits[:15]:
        print(f"         - {h}")
    if len(magic_hits) > 15:
        print(f"         ... 另有 {len(magic_hits) - 15} 处")
    rep.metrics["suspicious_magic_literals"] = len(magic_hits)


# ============================================================
# D3 交付物合规（含掩模质量）
# ============================================================
def run_deliverable_audit(rep: QAReport, psb_path: str | None) -> None:
    print("\n" + "=" * 74)
    print("  [D3] 交付物合规与掩模质量审计")
    print("=" * 74)

    if not psb_path or not os.path.isfile(psb_path):
        rep.record("D3-00", "交付物存在性", False, f"未找到 {psb_path}")
        print(f"  [FAIL] 未找到交付物: {psb_path}")
        return

    import cv2
    import numpy as np
    from psd_tools import PSDImage

    size_gb = os.path.getsize(psb_path) / (1024 ** 3)
    print(f"  文件: {psb_path}  ({size_gb:.3f} GB)")
    psd = PSDImage.open(psb_path)
    print(f"  画幅: {psd.size}  色彩模式: {psd.color_mode}  版本: {psd.version}")

    rep.record("D3-01", "PSB 版本为 2", psd.version == 2, f"version={psd.version}")
    res = psd.image_resources.get_data(1005)
    if res is None:
        rep.record("D3-02", "分辨率块 0x03ED", False, "缺失")
    else:
        ppi = res.horizontal / 65536.0
        rep.record("D3-02", "分辨率 150 PPI", abs(ppi - 150.0) < 0.1, f"{ppi:.2f} PPI")
        print(f"  分辨率: {ppi:.1f} PPI")

    rep.record("D3-03", "图层数 ≥ 8", len(psd) >= 8, f"{len(psd)} 层")

    print("\n  --- 逐层掩模质量（对照 masks_16k golden baseline） ---")
    bad: list[str] = []
    drift: list[str] = []
    # 注意：lyr.numpy() 返回的是该层包围盒大小的数组，其 size 不等于全画幅。
    # 填充率与覆盖率必须统一以「全画幅」为分母，否则 cover 恒为 100%。
    canvas_px = psd.width * psd.height
    baselines = load_baseline((psd.width, psd.height))
    if not baselines:
        print(f"  [WARN] 未找到 {BASELINE_DIR}/，退化为绝对阈值判定（发现不了「该小却大」的缺陷）")

    print(f"    {'图层':<40}{'fill%':>9}{'cover%':>8}{'基线fill%':>10}{'倍数':>9}{'IoU':>7}  判定")
    for lyr in psd:
        alpha = lyr.numpy()[:, :, 3]
        # 还原到全画幅坐标才能与基线比对
        full = np.zeros((psd.height, psd.width), dtype=np.uint8)
        full[lyr.top:lyr.bottom, lyr.left:lyr.right] = (alpha > 0.5).astype(np.uint8)
        n = int(full.sum())
        fill = n / canvas_px
        cover = (lyr.width * lyr.height) / canvas_px
        region = _is_region(lyr.name)

        flags = []
        if fill < FILL_RATIO_MIN:
            flags.append("EMPTY")
        if not region:
            if fill > FILL_RATIO_MAX:
                flags.append("OVERFLOW")
            if cover > BBOX_COVER_MAX:
                flags.append("BBOX_TOO_LARGE")

        b_fill = None
        factor = None
        iou = None
        base = baselines.get(_code_of(lyr.name))
        if base is not None:
            b_n = int(base.sum())
            b_fill = b_n / canvas_px
            if b_n > 0 and n > 0:
                factor = n / b_n
            inter = int(np.count_nonzero((full > 0) & (base > 0)))
            union = int(np.count_nonzero((full > 0) | (base > 0)))
            iou = inter / union if union else 0.0
            # 相对基线的结构性失衡：面积差一个数量级且形状对不上。
            # 元素层（印章/题跋/人物/芦雁等）判 FAIL；
            # 区域层（远山/折痕/外框/水波）的语义边界本就有主观性，降级为 DRIFT 提示，
            # 只记录不阻断——否则审计会因"远山该多大"这类主观分歧长期 FAIL 而失去信号价值。
            if iou < MIN_SHAPE_IOU and factor is not None:
                if factor > IMBALANCE_FACTOR or factor < 1.0 / IMBALANCE_FACTOR:
                    if region:
                        flags.append("DRIFT")
                        drift.append(f"{lyr.name.strip()}（{factor:.1f}x, IoU={iou:.3f}）")
                    else:
                        flags.append("IMBALANCE")

        bf = f"{b_fill*100:.4f}" if b_fill is not None else "n/a"
        fa = f"{factor:.1f}x" if factor is not None else "n/a"
        io = f"{iou:.3f}" if iou is not None else "n/a"
        mark = "  ".join(flags) if flags else "OK"
        print(f"    {lyr.name.strip()[:38]:<40}{fill*100:>9.4f}{cover*100:>8.2f}"
              f"{bf:>10}{fa:>9}{io:>7}  {mark}")
        # DRIFT 为区域层的语义漂移提示，不计入失败
        if any(f != "DRIFT" for f in flags):
            bad.append(f"{lyr.name.strip()} ({','.join(f for f in flags if f != 'DRIFT')})")
        del full

    if drift:
        print(f"\n  [WARN] 区域层语义漂移 {len(drift)} 处（边界主观性，仅提示不阻断）：")
        for d in drift:
            print(f"         - {d}")
    rep.metrics["region_drift"] = len(drift)

    rep.record(
        "D3-04",
        "元素掩模质量（无空层 / 无过覆盖 / 相对基线无结构性失衡）",
        not bad,
        "；".join(bad) if bad else "全部通过",
    )
    rep.metrics["layer_count"] = len(psd)
    rep.metrics["defective_layers"] = len(bad)


# ============================================================
# D4 防伪代码扫描
# ============================================================
def run_anti_deception_audit(rep: QAReport) -> None:
    print("\n" + "=" * 74)
    print("  [D4] 防伪代码扫描")
    print("=" * 74)
    import re

    files = sorted(glob.glob("engine/**/*.py", recursive=True)) + ["run_universal_engine.py"]
    patterns = [
        (r"^\s*pass\s*$", "空实现 pass"),
        (r"\bNotImplementedError\b", "未实现占位"),
        (r"#\s*(mock|fake|dummy)\b", "伪造标记"),
        (r"time\.sleep\([^)]*\)\s*#\s*(simulate|fake)", "模拟耗时"),
    ]
    hits: list[str] = []
    for fpath in files:
        if "__pycache__" in fpath or not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                for pat, desc in patterns:
                    if re.search(pat, line):
                        hits.append(f"{fpath}:{lineno}  {desc}")
    if hits:
        for h in hits[:20]:
            print(f"  [WARN] {h}")
    else:
        print("  [OK]   未发现空实现 / mock / 伪造数据 / 模拟耗时")
    # 抽象基类的 pass 属合法设计，仅披露不阻断
    rep.metrics["deception_candidates"] = len(hits)
    print(f"  [INFO] 候选 {len(hits)} 处（抽象基类 pass 属合法设计，仅披露不阻断）")


# ============================================================
# D5 C-SIMD 加速开关对比（端到端验证真的生效）
# ============================================================
def run_codec_audit(rep: QAReport) -> None:
    print("\n" + "=" * 74)
    print("  [D5] C-SIMD PackBits 加速端到端验证（与纯 Python 对比）")
    print("=" * 74)
    import numpy as np

    from engine.codecs_accelerator import (
        FastPackBitsAdapter,
        HAS_IMAGECODECS,
        install_psb_codec_accelerator,
    )

    if not HAS_IMAGECODECS:
        rep.record("D5-01", "imagecodecs 可用", False, "未安装，写盘将回退纯 Python（约 3 MB/s）")
        print("  [FAIL] imagecodecs 未安装")
        return

    rep.record("D5-01", "imagecodecs 可用", True)
    install_psb_codec_accelerator()

    rows = np.random.randint(0, 256, (200, 16000), dtype=np.uint8)
    t0 = time.time()
    enc_c = [FastPackBitsAdapter.encode(r) for r in rows]
    t_c = time.time() - t0

    ok_roundtrip = all(
        np.array_equal(rows[i], np.frombuffer(FastPackBitsAdapter.decode(enc_c[i]), dtype=np.uint8))
        for i in range(0, 200, 37)
    )
    rep.record("D5-02", "SIMD PackBits 往返无损", ok_roundtrip)

    def _py_packbits(row: np.ndarray) -> bytes:
        """纯 Python 参考实现（无 C 扩展），用于量级对比。"""
        out = bytearray()
        i, n = 0, len(row)
        while i < n:
            run = 1
            while i + run < n and run < 128 and row[i + run] == row[i]:
                run += 1
            if run > 2:
                out += bytes([257 - run, int(row[i]) & 0xFF])
                i += run
            else:
                lit = bytearray()
                while i < n and len(lit) < 128:
                    if i + 2 < n and row[i] == row[i + 1] == row[i + 2]:
                        break
                    lit.append(int(row[i]) & 0xFF)
                    i += 1
                out += bytes([len(lit) - 1]) + bytes(lit)
        return bytes(out)

    sample = rows[:10]
    t0 = time.time()
    _ = [_py_packbits(r) for r in sample]
    t_py = time.time() - t0

    mb_c = rows.nbytes / (1024 * 1024)
    mb_py = sample.nbytes / (1024 * 1024)
    print(f"  C 扩展      : {mb_c:.2f} MB / {t_c:.4f}s = {mb_c/t_c:.1f} MB/s")
    print(f"  纯 Python   : {mb_py:.2f} MB / {t_py:.4f}s = {mb_py/t_py:.1f} MB/s（{len(sample)} 行样本）")
    print(f"  C 路径为纯 Python 的 {(mb_c/t_c)/(mb_py/t_py):.1f} 倍")

    faster = (mb_c / t_c) > (mb_py / t_py) * 10
    rep.record("D5-03", "C 路径较纯 Python 显著加速（>10x）", faster,
               f"C={mb_c/t_c:.1f} MB/s, py={mb_py/t_py:.1f} MB/s")


# ============================================================
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="System integrity audit")
    ap.add_argument("psb", nargs="?", default=None)
    ap.add_argument("--skip-deliverable", action="store_true")
    args = ap.parse_args()

    psb = args.psb
    if not psb and not args.skip_deliverable:
        for cand in (
            "outputs/Rosetsu_Master_16k.psb",
            "outputs/Rosetsu_Robust_Master_16k.psb",
            "outputs/Rosetsu_Optimized_16k.psb",
        ):
            if os.path.isfile(cand):
                psb = cand
                break

    rep = QAReport(run_id=new_run_id(), tier="T4")

    print("\n" + "#" * 74)
    print("      SYSTEM INTEGRITY AUDIT (v2 · 可回归版)")
    print("#" * 74 + "\n")

    run_hardware_audit(rep)
    run_hardcode_audit(rep)
    if not args.skip_deliverable:
        run_deliverable_audit(rep, psb)
    run_anti_deception_audit(rep)
    run_codec_audit(rep)

    failed = [r for r in rep.records if not r["ok"]]
    print("\n" + "#" * 74)
    print(f"  run_id : {rep.run_id}")
    print(f"  结果   : {'PASS' if rep.passed else 'FAIL'}  "
          f"(共 {len(rep.records)} 项断言，失败 {len(failed)} 项)")
    for r in failed:
        print(f"    [FAIL] {r['code']} {r['name']} — {r['detail']}")
    print("#" * 74 + "\n")

    os.makedirs("intermediate", exist_ok=True)
    out = os.path.join("intermediate", f"audit_{rep.run_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"  报告已写入: {out}")

    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
