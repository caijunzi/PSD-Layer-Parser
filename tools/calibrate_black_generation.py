# -*- coding: utf-8 -*-
"""黑版生成曲线标定工具（按品类调优）—— 2026-09-16 新增。

用途
----
在**不下跑完整流水线**的前提下，快速评估「K 版曲线（gain/gamma/shift）」对三件事的影响，
为每个品类选出合适的值：

| 指标 | 含义 | 对应审计维度 |
|---|---|---|
| `k_mean_pct` / `k_nonzero_pct` | K 墨量均值 / 非空像素占比 | ⑦ plate 合规（真黑版判定） |
| `tac_max_pct` | 限墨后的最大总墨量 | ⑦ TAC 上限 |
| `color_rmse` | 曲线后 CMYK 经**同一 ICC** 反算回 RGB，与源 RGB 的 RMSE | ⑤ 合成等价性（保真代理） |
| `color_rmse_low` | 同上但低频（高斯模糊后），贴近 ⑤ 的口径 | ⑤ |

原理：候选 CMYK 经 `ICC(CMYK→RGB)` 反算回源色彩空间再比对 —— 与 ⑤ 的
「合成图 vs 源图」同源，只是省去了超分/分层的完整链路，故可用作**快速代理**。
最终确认仍须跑一次真实审计（本工具给出候选与推荐，不替代验收）。

用法
----
    python tools/calibrate_black_generation.py --input inputs/damask_sample.png \\
        --preset presets/textile_damask_photo.json
    # 或显式指定
    python tools/calibrate_black_generation.py --input inputs/金地屏风_江户芦雁寒林六曲_绫边装裱.jpg \\
        --icc profiles/CoatedFOGRA39.icc --condition iso_uncoated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.core.io_utils import imread_unicode                       # noqa: E402
from engine.core.color_manager import ColorManager                    # noqa: E402
from engine.core.ink_limiter import resolve_policy, limit_ink         # noqa: E402
from engine.core.black_generation import (                            # noqa: E402
    BlackGenerationPolicy, apply_black_generation,
)

#: 默认候选（增益 × 幂次）；shift 默认 0，可经 --shift 追加
DEFAULT_CANDIDATES = [
    (1.00, 1.00),          # 恒等（基线）
    (1.10, 1.00), (1.20, 1.00), (1.35, 1.00),
    (1.00, 0.90), (1.00, 0.80),          # gamma<1：加深中间调 K
    (1.15, 0.90), (1.30, 0.85),
    (0.90, 1.10), (0.80, 1.20),          # 减少 K
]

ICC_TO_RGB_CACHE: dict = {}


def _icc_cmyk_to_rgb(cmyk_raw: np.ndarray, icc_path: str) -> np.ndarray:
    """磁盘反码 CMYK → RGB（经**同一 ICC** 的 CMYK→sRGB 反算）。"""
    from PIL import Image, ImageCms

    xform = ICC_TO_RGB_CACHE.get(icc_path)
    if xform is None:
        dst = ImageCms.getOpenProfile(icc_path)
        rgb_prof = ImageCms.createProfile("sRGB")
        xform = ImageCms.buildTransform(
            dst, rgb_prof, "CMYK", "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        )
        ICC_TO_RGB_CACHE[icc_path] = xform
    ink = (255 - cmyk_raw).transpose(1, 2, 0).astype(np.uint8)   # (H,W,4) 墨量
    out = ImageCms.applyTransform(Image.fromarray(ink, mode="CMYK"), xform)
    return np.array(out)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def _lowfreq(img_rgb: np.ndarray, long_side: int = 1600) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    s = long_side / float(max(h, w))
    small = cv2.resize(img_rgb, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    k = max(3, (min(small.shape[:2]) // 16) | 1)
    return cv2.GaussianBlur(small, (k, k), 0)


def evaluate(img_bgr: np.ndarray, icc_path: str, condition: Optional[str],
             candidates, shift_pct: float = 0.0, long_side: int = 900) -> list[dict]:
    h, w = img_bgr.shape[:2]
    s = long_side / float(max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    src_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    base_raw = ColorManager.bgr_to_cmyk_raw(small, icc_path=icc_path)
    policy = resolve_policy(condition=condition, icc_path=icc_path)
    base_low = _lowfreq(src_rgb)

    rows = []
    for gain, gamma in candidates:
        pol = BlackGenerationPolicy(k_gain=gain, k_gamma=gamma, k_shift_pct=shift_pct)
        raw, _bg = apply_black_generation(base_raw, pol, inplace=False)
        raw_lim, tac = limit_ink(raw, policy, inplace=False)
        ink_k = 255.0 - raw_lim[3].astype(np.float32)
        rgb = _icc_cmyk_to_rgb(raw_lim, icc_path)
        rows.append({
            "k_gain": gain, "k_gamma": gamma, "k_shift_pct": shift_pct,
            "k_mean_pct": round(float(ink_k.mean()) / 255.0 * 100, 3),
            "k_nonzero_pct": round(float((ink_k > 0.05 * 255).mean()) * 100, 2),
            "tac_max_pct": tac["tac_after_max"],
            "color_rmse": round(_rmse(src_rgb, rgb), 3),
            "color_rmse_low": round(_rmse(base_low, _lowfreq(rgb)), 3),
        })
    return rows


def recommend(rows: list[dict]) -> dict:
    """推荐：在**色彩劣化 ≤15%** 的约束下，取 K 非空占比最高者；否则退回恒等。

    理由：⑤ 合成等价性是硬验收项，不能为"更深的黑版"牺牲太多保真；
    在可接受的保真损失内，K 越实（⑦ 真黑版判定越稳、且省 CMY 墨）。
    """
    ident = next(r for r in rows if abs(r["k_gain"] - 1) < 1e-9 and abs(r["k_gamma"] - 1) < 1e-9)
    tol = ident["color_rmse_low"] * 1.15 + 1e-6
    feasible = [r for r in rows if r["color_rmse_low"] <= tol]
    best = max(feasible, key=lambda r: (r["k_nonzero_pct"], -r["color_rmse_low"]))
    if best is ident:
        return {"action": "keep_identity",
                "reason": f"恒等已是最优：其它候选在保真约束(≤{tol:.3f})内未能提升 K 非空占比",
                "config": None, "baseline": ident}
    return {"action": "tune", "config": {"enabled": True, "k_gain": best["k_gain"],
                                         "k_gamma": best["k_gamma"], "k_shift_pct": best["k_shift_pct"]},
            "reason": f"K 非空 {ident['k_nonzero_pct']}%→{best['k_nonzero_pct']}%，"
                      f"低频 RMSE {ident['color_rmse_low']}→{best['color_rmse_low']}（≤{tol:.3f}）",
            "baseline": ident, "chosen": best}


def main() -> int:
    ap = argparse.ArgumentParser(description="黑版生成曲线标定（按品类调优）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--preset", help="从 preset 取 icc/print_condition")
    ap.add_argument("--icc", default="profiles/CoatedFOGRA39.icc")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    icc = args.icc
    cond = args.condition
    if args.preset:
        p = json.load(open(args.preset, encoding="utf-8"))
        icc = p.get("icc_path") or icc
        cond = cond or p.get("print_condition")

    img = imread_unicode(args.input)
    if img is None:
        print(f"[ERR] 读取失败: {args.input}")
        return 1
    print(f"输入: {args.input}  {img.shape[1]}x{img.shape[0]}")
    print(f"ICC: {icc}（存在={os.path.isfile(icc)}）  印刷条件: {cond or '(由 ICC 推断/默认)'}")
    policy = resolve_policy(condition=cond, icc_path=icc)
    print(f"TAC 上限 {policy.limit_pct:.0f}% / MaxK {policy.max_k_pct:.0f}%  [{policy.source}]")
    print()

    if not os.path.isfile(icc):
        print("[ERR] ICC 不存在 → 无法做 ICC 反算，标定失去意义")
        return 1

    rows = evaluate(img, icc, cond, DEFAULT_CANDIDATES, shift_pct=args.shift)
    print(f"{'k_gain':>7}{'k_gamma':>9}{'k_mean%':>9}{'k_nonzero%':>12}{'TAC_max%':>10}"
          f"{'RMSE':>9}{'RMSE_low':>10}")
    for r in rows:
        print(f"{r['k_gain']:>7.2f}{r['k_gamma']:>9.2f}{r['k_mean_pct']:>9.3f}"
              f"{r['k_nonzero_pct']:>12.2f}{r['tac_max_pct']:>10.2f}"
              f"{r['color_rmse']:>9.3f}{r['color_rmse_low']:>10.3f}")
    print()
    rec = recommend(rows)
    print("推荐:", json.dumps(rec["reason"], ensure_ascii=False))
    print("配置:", json.dumps(rec["config"], ensure_ascii=False))
    print("⚠️ 本工具为**快速代理**（CMYK 域），最终须跑一次真实审计确认 ⑤/⑦。")

    if args.json_out:
        json.dump({"input": args.input, "icc": icc, "condition": cond,
                   "rows": rows, "recommendation": rec},
                  open(args.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[OK] 已写出 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
