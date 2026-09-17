"""Element Mask Quality Diagnostic (元素掩模质量诊断).

以 `masks_16k/`（v2.1 手工调优基线）为 golden baseline，跑当前规则分割引擎，
逐层对比填充率 / 包围盒尺寸 / IoU，定位「语义掩模泄漏」的具体图层与量级。

用法：
    python tests/diagnose_mask_quality.py
    python tests/diagnose_mask_quality.py --source inputs/金地屏风_江户芦雁寒林六曲_绫边装裱.jpg --json

判定口径（后续将固化为回归断言）：
    - fill_ratio（有效像素 / 全画幅）应落在 [0.001%, 15%]
    - bbox 覆盖率（bbox 面积 / 全画幅）元素层原则上 < 60%
    - 与基线 IoU 低于 0.10 视为「语义漂移」，需人工确认
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.providers.segmentation_provider import ZeroHardcodeSegmentationProvider  # noqa: E402
from engine.providers.grounded_sam_provider import GroundedSAMProvider  # noqa: E402

BASELINE_DIR = "masks_16k"
DEFAULT_SOURCE = "inputs/金地屏风_江户芦雁寒林六曲_绫边装裱.jpg"

# 判定阈值（后续迁入 presets/ 或 schemas/）
FILL_RATIO_MIN = 0.00001   # 0.001%：低于此值视为空层缺陷
FILL_RATIO_MAX = 0.15      # 15%：元素层面积上限
BBOX_COVER_MAX = 0.60      # 60%：元素层包围盒覆盖率上限
IOU_DRIFT_MIN = 0.10       # 低于此值视为语义漂移

# 区域类图层：几何上本就大面积或线状贯穿（远山、折痕、水波、外框、底板），
# 不适用元素层的紧凑度标准，只校验"不能为空"。
REGION_KEYWORDS = (
    "远山", "mountain", "水波", "ripple", "折痕", "seam", "fold",
    "外框", "frame", "brocade", "底板", "base", "ground",
)


def is_region_layer(name: str) -> bool:
    low = str(name).lower()
    return any(k.lower() in low for k in REGION_KEYWORDS)


def _code_of(name: str) -> str:
    """从图层名/文件名中提取编号 token，如 '09A'、'10B'、'03'。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"^mask_", "", stem)
    m = re.match(r"(\d{2}[A-Za-z]?)", stem)
    if m:
        return m.group(1).upper()
    m = re.search(r"_(\d{2}[A-Za-z]?)_", stem)
    return m.group(1).upper() if m else stem.upper()


def load_baseline(target_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """读取 masks_16k 基线并降采样到与规则引擎输出一致的分辨率。"""
    out: dict[str, np.ndarray] = {}
    if not os.path.isdir(BASELINE_DIR):
        return out
    w, h = target_size
    for fn in sorted(os.listdir(BASELINE_DIR)):
        if not fn.lower().endswith(".png"):
            continue
        m = cv2.imread(os.path.join(BASELINE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[1] != w or m.shape[0] != h:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out[_code_of(fn)] = (m > 127)
    return out


def mask_stats(mask: np.ndarray) -> dict:
    m = (mask > 127) if mask.dtype != np.bool_ else mask
    total = m.size
    n = int(np.count_nonzero(m))
    stats = {
        "pixels": n,
        "fill_ratio": n / total if total else 0.0,
        "bbox": None,
        "bbox_cover": 0.0,
    }
    if n:
        ys, xs = np.where(m)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        stats["bbox"] = (x0, y0, x1, y1)
        stats["bbox_cover"] = ((x1 - x0) * (y1 - y0)) / total if total else 0.0
    return stats


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 127) if a.dtype != np.bool_ else a
    b = (b > 127) if b.dtype != np.bool_ else b
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Element mask quality diagnostic")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    ap.add_argument(
        "--rule-only",
        action="store_true",
        help="只跑规则引擎，跳过 Grounded DINO + SAM 2 神经路径（快，但不等价于生产）",
    )
    args = ap.parse_args()

    src = cv2.imread(args.source)
    if src is None:
        print(f"[FATAL] 无法读取源图: {args.source}")
        return 1
    h, w = src.shape[:2]
    print(f"[INFO] 源图 {w}x{h}")

    if args.rule_only:
        provider = ZeroHardcodeSegmentationProvider()
        masks = provider.segment_by_labels(src)
        mapper = GroundedSAMProvider.__new__(GroundedSAMProvider)
        final = mapper._map_to_bilingual_names(masks)
        print("[INFO] 模式：仅规则引擎")
    else:
        # 与生产一致：完整 GroundedSAMProvider（含神经质量门）
        gsam = GroundedSAMProvider(preferred_device="CPU", preset_name="japanese_screen_gold")
        final = gsam.segment_objects(src, classes=None)
        print(f"[INFO] 模式：生产同款 GroundedSAMProvider（backend={gsam.backend}）")

    baseline = load_baseline((w, h))
    total_px = w * h

    rows = []
    for name in sorted(final):
        st = mask_stats(final[name])
        code = _code_of(name)
        base = baseline.get(code)
        row = {
            "layer": name,
            "code": code,
            "pixels": st["pixels"],
            "fill_ratio": st["fill_ratio"],
            "bbox": st["bbox"],
            "bbox_cover": st["bbox_cover"],
            "baseline_fill_ratio": None,
            "iou": None,
            "verdict": [],
        }
        if base is not None:
            bn = int(np.count_nonzero(base))
            row["baseline_fill_ratio"] = bn / total_px
            row["iou"] = iou(final[name], base)

        region = is_region_layer(name)

        if st["fill_ratio"] < FILL_RATIO_MIN:
            row["verdict"].append("EMPTY(空层)")
        elif not region and st["fill_ratio"] > FILL_RATIO_MAX:
            row["verdict"].append("OVERFLOW(过覆盖)")
        if not region and st["bbox_cover"] > BBOX_COVER_MAX:
            row["verdict"].append("BBOX_TOO_LARGE")
        if row["iou"] is not None and row["iou"] < IOU_DRIFT_MIN:
            row["verdict"].append("SEMANTIC_DRIFT")
        if not row["verdict"]:
            row["verdict"].append("OK")
        rows.append(row)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=float))
    else:
        print()
        print("=" * 108)
        print("  元素掩模质量诊断（当前规则引擎  vs  masks_16k 基线）")
        print("=" * 108)
        hdr = f"{'图层':<40}{'当前填充%':>10}{'基线填充%':>10}{'BBox覆盖%':>10}{'IoU':>8}  判定"
        print(hdr)
        print("-" * 108)
        for r in rows:
            bf = f"{r['baseline_fill_ratio']*100:.4f}" if r["baseline_fill_ratio"] is not None else "   n/a"
            io = f"{r['iou']:.3f}" if r["iou"] is not None else "  n/a"
            print(
                f"{r['layer'][:38]:<40}{r['fill_ratio']*100:>10.4f}{bf:>12}"
                f"{r['bbox_cover']*100:>10.2f}{io:>8}  {','.join(r['verdict'])}"
            )
        print("-" * 108)
        bad = [r for r in rows if r["verdict"] != ["OK"]]
        print(f"  合计 {len(rows)} 层，异常 {len(bad)} 层")
        if bad:
            print()
            print("  【泄漏量级排序】")
            for r in sorted(
                bad,
                key=lambda x: (x["fill_ratio"] / x["baseline_fill_ratio"])
                if (x["baseline_fill_ratio"] and x["baseline_fill_ratio"] > 0)
                else 0.0,
                reverse=True,
            ):
                if r["baseline_fill_ratio"]:
                    ratio = r["fill_ratio"] / r["baseline_fill_ratio"]
                    print(f"    {r['layer'][:38]:<40} 相对基线放大 {ratio:>10.1f}x")
                else:
                    print(f"    {r['layer'][:38]:<40} 无基线可比对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
