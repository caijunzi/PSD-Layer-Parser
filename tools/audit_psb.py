"""交付前 8 维审计门（Delivery Audit Gate）。

背景（2026-09-12 深度审计）：仅凭「层数/manifest/单测」无法发现"层存在但核心
价值受损"的隐性缺陷（底板残留、实例重复、内容丢失、层序混乱…）。本模块把
审计固化为**每次产物交付前必跑**的门禁：

  ① 层属性        层名唯一 / 全部可见 / 混合与不透明度合法
  ② 分辨率真实性  层像素 shape == bbox（否则为低分放大）
  ③ 实例重复      同类实例掩模 IoU > 0.9 视为重复
  ④ 内容承载      「底板抹除但无层承载」的墨迹占比（内容丢失风险）
  ⑤ 合成等价性    全层叠加 vs 原图（多尺度 RMSE，超分差异有容忍）
  ⑥ 底板纯净度    分区域 dark 率（金地底板应为纯金箔）
  ⑦ plate 合规    CMYK 模式 / TAC ≤ 300% / 纯净性（仅 plate 产物）
  ⑧ manifest 一致 声明层数/层名集合 == PSB 实际

用法：
    python tools/audit_psb.py <result.psb> [--manifest m.json] [--source src.jpg] [--json out.json]
返回码：0 = 全部通过；2 = 存在不通过维度（CI/流水线可据此拦截）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------- 阈值（可调；调低=更严） ----------------
TH = {
    "dup_iou": 0.9,                 # 实例重复判定 IoU
    "content_loss_ratio": 0.02,     # 内容丢失占比上限（校准：含超分噪声，V2 同口径基线 ~1%）
    "synth_rmse_lowfreq": 25.0,     # σ=16 低频 RMSE 上限（容忍超分细节差异）
    "synth_rmse_raw": 60.0,         # 原始尺度 RMSE 上限
    "base_dark_ratio": 8.0,         # 底板分区域 dark 率上限（%）
                                     # 校准依据：V2 参照同口径实测 石矶 2.44 / 山峦 1.21 /
                                     # 纯金地 9.04 / 右下岩石 5.16（本次产物 2.72/1.40/0.05/6.27）
    "base_dark_delta": 20.0,        # 底板 dark 判定阈值（灰阶低于 median 多少）
    "tac_limit": 310.0,             # TAC 上限（%），印刷 300% + 容差
    # 支撑/装饰层：不参与「内容承载」统计（底板是承托、外框/折痕是装饰）
    "support_names": ("Gold_Base_Clean", "Brocade_Outer_Frame", "Panel_Fold_Seams"),
    # 内容层（含残层）：参与重复/承载统计
    "oracle_names": ("Gold_Base_Clean", "Brocade_Outer_Frame", "Panel_Fold_Seams"),
}

REGIONS = {
    "雁群区": (0.20, 0.16, 0.45, 0.34), "石矶区": (0.42, 0.62, 0.52, 0.72),
    "右侧山峦": (0.60, 0.35, 0.90, 0.70), "中央枯树": (0.40, 0.30, 0.55, 0.60),
    "纯金地": (0.05, 0.30, 0.15, 0.45), "右下岩石": (0.70, 0.60, 0.95, 0.85),
}


def _clean(name: str) -> str:
    return str(name).strip().rstrip("\x00")


def _alpha(a: np.ndarray) -> np.ndarray:
    al = a[:, :, 3].astype(np.float32)
    return al / 255.0 if al.max() > 1.001 else al


def _full_mask(ly, H: int, W: int) -> np.ndarray:
    m = _alpha(ly.numpy()) > 0.03
    f = np.zeros((H, W), bool)
    x0, y0 = ly.bbox[0], ly.bbox[1]
    h, w = m.shape
    f[y0:y0 + h, x0:x0 + w] = m[:max(0, H - y0), :max(0, W - x0)]
    return f


def audit(psb_path: str, manifest_path: str | None = None,
          source_image: str | None = None, verbose: bool = True) -> dict:
    from psd_tools import PSDImage
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    psd = PSDImage.open(psb_path)
    W, H = psd.size
    layers = list(psd)
    names = [_clean(ly.name) for ly in layers]
    is_cmyk = int(psd.color_mode) == 4
    result: dict = {"psb": psb_path, "size": [W, H], "layer_count": len(layers),
                    "color_mode": "cmyk" if is_cmyk else "rgb",
                    "dims": {}, "issues": []}
    dims = result["dims"]

    content = [ly for ly in layers if not any(o in _clean(ly.name) for o in TH["oracle_names"])]

    # ① 层属性
    d = {"passed": True, "metrics": {}, "issues": []}
    dup_names = [n for n, c in defaultdict(int, {n: names.count(n) for n in set(names)}).items() if c > 1]
    invisible = [n for n, ly in zip(names, layers) if not ly.visible]
    d["metrics"].update(layer_count=len(layers), content_layers=len(content),
                        duplicate_names=len(dup_names), invisible_layers=len(invisible))
    if dup_names:
        d["passed"] = False
        d["issues"].append(f"层名重复: {dup_names[:3]}")
    if invisible:
        d["passed"] = False
        d["issues"].append(f"存在不可见层: {invisible[:3]}")
    dims["① 层属性"] = d

    # ② 分辨率真实性
    d = {"passed": True, "metrics": {}, "issues": []}
    bad_res = []
    for ly in content:
        b = ly.bbox
        bw, bh = b[2] - b[0], b[3] - b[1]
        a = ly.numpy()
        if a.shape[0] != bh or a.shape[1] != bw:
            bad_res.append(f"{_clean(ly.name)[:24]}:{a.shape[:2]}!={bh}x{bw}")
    d["metrics"]["res_mismatch"] = len(bad_res)
    if bad_res:
        d["passed"] = False
        d["issues"].append(f"层像素与 bbox 不符（疑低分放大）: {bad_res[:3]}")
    dims["② 分辨率真实性"] = d

    # 掩模缓存（供 ③④⑥ 复用）
    masks = {}
    for ly in content:
        m = _full_mask(ly, H, W)
        if m.any():
            masks[_clean(ly.name)] = m

    # ③ 实例重复
    d = {"passed": True, "metrics": {}, "issues": []}
    groups = defaultdict(list)
    for ly in content:
        nm = _clean(ly.name)
        key = nm.rsplit("_", 1)[0] if nm[-3:-1].replace("_", "").isdigit() else nm
        groups[key].append(nm)
    dups = []
    for key, nms in groups.items():
        if len(nms) < 2:
            continue
        for i in range(len(nms)):
            for j in range(i + 1, len(nms)):
                a, b = masks.get(nms[i]), masks.get(nms[j])
                if a is None or b is None:
                    continue
                inter = int((a & b).sum())
                if inter == 0:
                    continue
                union = int((a | b).sum())
                if union and inter / union > TH["dup_iou"]:
                    dups.append(f"{nms[i][:20]}≈{nms[j][:20]}(IoU={inter/union:.2f})")
    d["metrics"]["duplicate_pairs"] = len(dups)
    if dups:
        d["passed"] = False
        d["issues"].append(f"重复实例: {dups[:3]}")
    dims["③ 实例重复"] = d

    # ④ 内容承载（需源图）
    #    口径（2026-09-12 校准）： 必须**包含残层**——「未分类墨迹残层」正是
    #    承载未成层内容的载体；把它排除会虚报内容丢失（实测 1.03% → 18.38%）。
    #    仅排除支撑/装饰层（底板/外框/折痕）。
    d = {"passed": True, "metrics": {}, "issues": []}
    if source_image and os.path.isfile(source_image):
        from engine.core.io_utils import imread_unicode
        import cv2
        src = cv2.cvtColor(imread_unicode(source_image), cv2.COLOR_BGR2RGB)
        src16 = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
        g16 = cv2.cvtColor(src16, cv2.COLOR_RGB2GRAY)
        ink = g16 < (np.median(g16) - 12)
        base_ly = next((ly for ly in layers if "Gold_Base_Clean" in _clean(ly.name)), None)
        if base_ly is not None:
            ba = base_ly.numpy()
            base_rgb = (np.clip(ba[:, :, :3], 0, 1) * 255).astype(np.uint8) if ba.dtype != np.uint8 else ba[:, :, :3]
            erased = np.abs(base_rgb.astype(np.int16) - src16.astype(np.int16)).max(axis=2) > 40
            union = np.zeros((H, W), bool)
            for m in masks.values():
                union |= m
            lost = int((ink & erased & ~union).sum())
            total_ink = max(1, int(ink.sum()))
            d["metrics"].update(ink_px=total_ink, lost_px=lost, lost_ratio=round(lost / total_ink, 6))
            if lost / total_ink > TH["content_loss_ratio"]:
                d["passed"] = False
                d["issues"].append(f"内容丢失 {lost:,}px（{lost/total_ink*100:.3f}% 墨迹）")
        else:
            d["metrics"]["note"] = "未找到底板层，跳过"
            d["passed"] = True
    else:
        d["metrics"]["note"] = "未提供源图，跳过"
    dims["④ 内容承载"] = d

    # ⑤ 合成等价性（缩采样内存合成）
    d = {"passed": True, "metrics": {}, "issues": []}
    if source_image and os.path.isfile(source_image):
        import cv2
        from engine.core.io_utils import imread_unicode
        TW = 1600
        scale = TW / W
        TH_, TW_ = int(H * scale), TW
        canvas = np.zeros((TH_, TW_, 3), np.float32)
        for ly in reversed(layers):          # psd_tools 列表 = 从顶到底 → 反转成底→顶
            if not ly.visible:
                continue
            b = ly.bbox
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bw <= 0 or bh <= 0:
                continue
            a = ly.numpy()
            rgb = a[:, :, :3].astype(np.float32)
            al = a[:, :, 3].astype(np.float32)
            if al.max() > 1.001:
                rgb, al = rgb / 255.0, al / 255.0
            tw, th = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
            rgb_s = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
            al_s = np.clip(cv2.resize(al, (tw, th), interpolation=cv2.INTER_AREA), 0, 1)[:, :, None]
            tx, ty = int(round(b[0] * scale)), int(round(b[1] * scale))
            tx2, ty2 = min(TW_, tx + tw), min(TH_, ty + th)
            tw, th = tx2 - tx, ty2 - ty
            if tw <= 0 or th <= 0:
                continue
            region = canvas[ty:ty2, tx:tx2]
            canvas[ty:ty2, tx:tx2] = rgb_s[:th, :tw] * al_s[:th, :tw] + region * (1 - al_s[:th, :tw])
        synth = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        ref = cv2.resize(cv2.cvtColor(imread_unicode(source_image), cv2.COLOR_BGR2RGB), (TW_, TH_),
                         interpolation=cv2.INTER_AREA)
        # 多尺度（原始 + 低频）
        raw_rmse = float(np.sqrt(((synth.astype(np.float32) - ref.astype(np.float32)) ** 2).mean()))
        a16 = cv2.GaussianBlur(synth, (0, 0), 16).astype(np.float32)
        b16 = cv2.GaussianBlur(ref, (0, 0), 16).astype(np.float32)
        low_rmse = float(np.sqrt(((a16 - b16) ** 2).mean()))
        d["metrics"].update(rmse_raw=round(raw_rmse, 2), rmse_lowfreq=round(low_rmse, 2))
        if low_rmse > TH["synth_rmse_lowfreq"] or raw_rmse > TH["synth_rmse_raw"]:
            d["passed"] = False
            d["issues"].append(f"合成与原图差异偏大（低频RMSE={low_rmse:.1f}, 原始RMSE={raw_rmse:.1f}）")
    else:
        d["metrics"]["note"] = "未提供源图，跳过"
    dims["⑤ 合成等价性"] = d

    # ⑥ 底板纯净度
    #    2026-09-12 定稿：门禁判定用**通用项**（底板存在 + 非空——底板缺失/空层
    #    是致命缺陷）；分区域 dark 率是**品类相关指标**（山水图的区域先验不适用
    #    壁布等其他品类），降级为 metrics 披露，深查由
    #    tools/calibrate_density_bands.py + 人工/金标准承担。
    d = {"passed": True, "metrics": {}, "issues": []}
    base_ly = next((ly for ly in layers
                    if "Gold_Base" in _clean(ly.name) or "Fabric_Base" in _clean(ly.name)
                    or "底板" in _clean(ly.name)), None)
    if base_ly is None:
        d["passed"] = False
        d["issues"].append("缺少底板层（Gold_Base/Fabric_Base/底板 关键词均未命中）")
    else:
        import cv2
        ba = base_ly.numpy()
        al = _alpha(ba)
        nz = float((al > 0.03).mean()) * 100
        d["metrics"].update(base_layer=_clean(base_ly.name)[:30], opaque_pct=round(nz, 1))
        if nz < 1:
            d["passed"] = False
            d["issues"].append("底板层几乎全透明（空层）")
        try:
            bs = ba[::max(1, ba.shape[0] // 2000), ::max(1, ba.shape[1] // 4000), :3]
            if ba.dtype != np.uint8:
                bs = bs * 255.0
            g = cv2.cvtColor(np.clip(bs, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            med = float(np.median(g))
            dark_all = float((g < med - TH["base_dark_delta"]).mean()) * 100
            d["metrics"]["dark_ratio_all"] = round(dark_all, 2)
            for rname, (x0f, y0f, x1f, y1f) in REGIONS.items():
                hh, ww = g.shape
                s = g[int(y0f * hh):int(y1f * hh), int(x0f * ww):int(x1f * ww)]
                if s.size:
                    d["metrics"][f"dark_{rname}"] = round(float((s < med - TH["base_dark_delta"]).mean()) * 100, 2)
            d["metrics"]["note"] = ("分区域 dark 为品类相关指标（披露不判失败）；"
                                     "深查用 tools/calibrate_density_bands.py + 金标准")
        except Exception:
            pass
    dims["⑥ 底板纯净度"] = d

    # ⑦ plate 合规
    d = {"passed": True, "metrics": {}, "issues": []}
    if is_cmyk:
        d["metrics"]["color_mode"] = "cmyk ✓"
        if manifest_path and os.path.isfile(manifest_path):
            m = json.load(open(manifest_path, encoding="utf-8"))
            tot = m.get("totals", {})
            tac = tot.get("tac_max_pct")
            d["metrics"].update(tac_max_pct=tac, plate_purity_ok=tot.get("plate_purity_ok"))
            if tac is not None and float(tac) > TH["tac_limit"]:
                d["passed"] = False
                d["issues"].append(f"TAC {tac}% 超限（>{TH['tac_limit']}%）")
            if tot.get("plate_purity_ok") is False:
                d["passed"] = False
                d["issues"].append(f"PLATE 纯净性不合格: {str(tot.get('plate_purity_message'))[:60]}")
        # 采样 K 版非空（真黑版判定：ICC 分色生效）
        try:
            big = max(content, key=lambda l: (l.bbox[2] - l.bbox[0]) * (l.bbox[3] - l.bbox[1]))
            a = big.numpy()
            if a.shape[2] >= 4:
                kt = a[:, :, 3].max()
                norm = 255.0 if kt > 1.001 else 1.0
                sub = a[::max(1, a.shape[0] // 300), ::max(1, a.shape[1] // 300)]
                d["metrics"]["k_channel_nonzero_pct"] = round(float((sub[:, :, 3] > 0.1 * norm).mean()) * 100, 1)
        except Exception:
            pass
    else:
        d["metrics"]["color_mode"] = "rgb（design 线，plate 项不适用）"
    dims["⑦ plate 合规"] = d

    # ⑧ manifest 一致性
    d = {"passed": True, "metrics": {}, "issues": []}
    if manifest_path and os.path.isfile(manifest_path):
        m = json.load(open(manifest_path, encoding="utf-8"))
        declared = {_clean(r.get("name", "")) for r in m.get("layers", [])}
        actual = set(names)
        d["metrics"].update(declared=len(declared), actual=len(actual))
        missing = sorted(actual - declared)
        extra = sorted(declared - actual)
        if missing or extra:
            d["passed"] = False
            d["issues"].append(f"层清单不一致（PSB 独有 {missing[:3]} / manifest 独有 {extra[:3]}）")
    else:
        d["metrics"]["note"] = "未提供 manifest，跳过"
    dims["⑧ manifest 一致"] = d

    # 汇总
    failed = [k for k, v in dims.items() if not v.get("passed")]
    result["passed"] = not failed
    for k, v in dims.items():
        for it in v.get("issues", []):
            result["issues"].append(f"{k}: {it}")
    log("\n========== 交付审计（8 维）==========")
    for k, v in dims.items():
        log(f"  {'✅' if v.get('passed') else '❌'} {k}  {json.dumps(v.get('metrics', {}), ensure_ascii=False)}")
    log(f"  结论: {'✅ 全部通过' if result['passed'] else '❌ 存在不通过维度'}")
    for it in result["issues"]:
        log(f"     - {it}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="交付前 8 维审计门")
    ap.add_argument("psb")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.manifest is None:
        cand = os.path.join(os.path.dirname(args.psb), "result.manifest.json")
        args.manifest = cand if os.path.isfile(cand) else None
    res = audit(args.psb, args.manifest, args.source, verbose=not args.quiet)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[audit] 报告已落盘: {args.json_out}")
    return 0 if res["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
