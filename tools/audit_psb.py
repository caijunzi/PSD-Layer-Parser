"""交付前 8 维审计门（Delivery Audit Gate）。

背景（2026-09-12 深度审计）：仅凭「层数/manifest/单测」无法发现"层存在但核心
价值受损"的隐性缺陷（底板残留、实例重复、内容丢失、层序混乱…）。本模块把
审计固化为**每次产物交付前必跑**的门禁：

  ① 层属性        层名唯一 / 全部可见 / 混合与不透明度合法
  ② 分辨率真实性  层像素 shape == bbox（否则为低分放大）
  ③ 实例重复      同类实例掩模 IoU > 0.9 视为重复
  ④ 内容承载      「底板抹除但无层承载」的墨迹占比（内容丢失风险）
  ⑤ 合成等价性    全层叠加 vs 原图（多尺度 RMSE，超分差异有容忍）
  ⑥ 底板纯净度    分区域 dark 率（底版应为纯净/均匀，产品线无关）
  ⑦ plate 合规    CMYK 模式 / TAC ≤ 300% / 纯净性（仅 plate 产物；design 不适用）
  ⑧ manifest 一致 声明层数/层名集合 == PSB 实际

每个维度在返回里带 evaluated（是否真正用数据核验过）/ na（本产品线不适用，
如 design 线的 ⑦ plate 合规）标志，供 episode 严格准入判断，避免把
"未核验/不适用"误判为通过。

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
    # 2026-09-13：改为关键词口径（兼容 Gold_Base / Fabric_Base / Base_Ground / 中文名），
    # 否则新预设的 01_纯净画布底板_Base_Ground 会被误当内容层统计。
    # 注：support_names 与 oracle_names 目前取值相同（历史遗留，二者曾计划分化），
    #     oracle_names 是实际被 content 过滤使用的那一份；support_names 保留兼容。
    "support_names": ("Gold_Base", "Fabric_Base", "Base_Ground", "底板",
                      "Brocade_Outer_Frame", "Panel_Fold_Seams", "外框", "折痕"),
    # 内容层（含残层）：参与重复/承载统计
    "oracle_names": ("Gold_Base", "Fabric_Base", "Base_Ground", "底板",
                     "Brocade_Outer_Frame", "Panel_Fold_Seams", "外框", "折痕"),
    # 印前加工层：整版施加，掩码天然覆盖全画布（如冲孔挂点、烫金陷印、专色版）。
    # 2026-09-16：④ 内容承载的 union 必须排除它们——否则任一个「全画布加工层」
    # 都会把 union 撑到 100%，使 lost 恒为 0，形成假通过。
    # 实证：damask case 产物含 12_激光冲孔挂点（Alpha 覆盖 100%）与
    # 13B_复古金专色（89.8%），修复前 ④ lost_ratio=0.0（假通过）；
    # 排除加工层后为 0.1129（11.29%，正确判失败）。
    "process_names": ("DieCut", "Perforations", "Foil", "Trap", "Spot",
                      "冲孔", "烫金", "陷印", "专色"),
}

REGIONS = {
    "雁群区": (0.20, 0.16, 0.45, 0.34), "石矶区": (0.42, 0.62, 0.52, 0.72),
    "右侧山峦": (0.60, 0.35, 0.90, 0.70), "中央枯树": (0.40, 0.30, 0.55, 0.60),
    "底色区": (0.05, 0.30, 0.15, 0.45), "右下岩石": (0.70, 0.60, 0.95, 0.85),
}
# 这些区域框是**构图先验**（山水图实测校准），仅作 dark 率披露，不参与任何判定，
# 也不假定底版一定是金地。换品类时这些框不具意义，但纯披露不影响门禁。

# 与 engine/adaptive/episode_archiver.py 严格准入保持一致的 8 维键（顺序即维度序）
DIM_KEYS = ("① 层属性", "② 分辨率真实性", "③ 实例重复", "④ 内容承载",
            "⑤ 合成等价性", "⑥ 底板纯净度", "⑦ plate 合规", "⑧ manifest 一致")
# 仅 PLATE 产品线需要的维度（design 线该维度不适用 na）
PLATE_DIM = "⑦ plate 合规"


def _clean(name: str) -> str:
    return str(name).strip().rstrip("\x00")


# 底板层判定关键词（产品线无关：金地/壁布/水墨/油画等均可，不写死"金地"）。
# 命中任意一个即视为底板层；不要求特定命名。设计稿底色层命名为「底版/底板」也可命中。
DEFAULT_BASE_KEYWORDS = ("Base_Ground", "底版", "底板", "Substrate",
                         "Gold_Base", "Fabric_Base")


def _find_base_layer(layers, base_keywords=None):
    """按统一关键词定位底板层（与 ⑥ 同口径）。

    base_keywords 可被子类/调用方覆盖，默认产品线无关的关键词集合，
    避免把底板命名写死成「金地」。
    """
    keys = base_keywords or DEFAULT_BASE_KEYWORDS
    return next((ly for ly in layers
                 if any(k in _clean(ly.name) for k in keys)), None)


def _alpha(a: np.ndarray) -> np.ndarray:
    """取图层的 Alpha 通道（自动识别通道布局）。

    ⚠️ 关键修复（2026-09-16）：psd_tools 的 `layer.numpy()` 通道数随色彩模式变化：
      - RGB 层  → (H, W, 4) = R, G, B, Alpha    → Alpha 在 index 3
      - CMYK 层 → (H, W, 5) = C, M, Y, K, Alpha → Alpha 在 index **4**
    此前写死 `a[:, :, 3]`，在 CMYK（PLATE 线）产物上取到的是 **K 通道**。
    ICC 真分色后 K 在整幅上大面积非零（去墨重构的底板 K 恒为 1.0）→ 所有掩码
    被判成「全画布」→ ② 分辨率真实性 / ③ 实例重复 / ④ 内容承载 / ⑥ 底板纯净度
    全部基于错误掩码计算，④ 更是恒定为「零丢失」的假通过。

    实测（outputs/adaptive-e2e-20260915-rerun/case-01..03）：
      修复前 CMYK 产物 union 恒为 100.0%；修复后 63.77%（金地）/ 69.95%（商用）。
      RGB（design 线）产物不受影响——4 通道下 index 3 本就是 Alpha，
      这正是此前「只有 PLATE 线样本问题多」的原因。
    """
    if a.ndim != 3 or a.shape[2] < 4:
        return np.ones(a.shape[:2], np.float32)
    idx = 4 if a.shape[2] >= 5 else 3
    al = a[:, :, idx].astype(np.float32)
    return al / 255.0 if al.max() > 1.001 else al


def _layer_rgb(ly) -> np.ndarray:
    """把图层像素归一为 0..255 的 RGB 数组（用于与源图 RGB 比对）。

    psd_tools 的 `numpy()` 对 CMYK 层返回 (H, W, 5) = C, M, Y, K, Alpha，
    但取值语义是**呈色**（= 1 - 墨量），实测：象牙底板 C=.961 M=.910
    Y=.829 K=1.000（K=1.0 表示 0% 黑版）。因此转 RGB 直接用乘法：
        R = (1-墨量C) * (1-墨量K) = v0 * v3
        G = v1 * v3,  B = v2 * v3
    此前 ④ 直接用 `ba[:, :, :3]`，在 CMYK 产物上把 **C,M,Y 当成了 R,G,B** 与源图
    比对，比较基准完全错位。

    ⚠️ 注意：本函数做的是**无 ICC 的近似转换**，只用于「底板 vs 源图」的
    差异定位（阈值 >40 的粗判），不用于 ⑤ 的合成等价性（那里必须走同一 ICC）。
    返回的数组仅覆盖图层 bbox，调用方需自行补边到整画布。
    """
    a = ly.numpy()
    if a.ndim != 3 or a.shape[2] < 3:
        return np.zeros(a.shape[:2] + (3,), np.float32)
    if a.shape[2] >= 5:
        cmyk = a[..., :4].astype(np.float32)
        k_keep = cmyk[..., 3]
        return np.stack([cmyk[..., 0] * k_keep,
                         cmyk[..., 1] * k_keep,
                         cmyk[..., 2] * k_keep], -1) * 255.0
    rgb = a[..., :3].astype(np.float32)
    return rgb * 255.0 if rgb.max() <= 1.001 else rgb


def _full_mask(ly, H: int, W: int) -> np.ndarray:
    m = _alpha(ly.numpy()) > 0.03
    f = np.zeros((H, W), bool)
    x0, y0 = ly.bbox[0], ly.bbox[1]
    h, w = m.shape
    f[y0:y0 + h, x0:x0 + w] = m[:max(0, H - y0), :max(0, W - x0)]
    return f


def _composite_rmse(psd, source_bgr, target_long: int = 1600, icc_path: str | None = None):
    """用 psd_tools 真实渲染（尊重每层的 opacity / blend 模式 / CMYK 分色）合成整图，
    再与源图（统一到同一色彩空间）比对，返回 (原始 RMSE, 低频 RMSE)。

    关键修复（2026-09-15）：此前用手工 alpha-over（`rgb*alpha + bg*(1-alpha)`）逐层叠加，
    **完全忽略图层不透明度与混合模式**，且对 CMYK 产物按 RGB 处理会失真。现改用
    `psd.composite()` —— 这是 psd_tools 的权威渲染路径，会按 PSD 内嵌的 opacity/blend
    正确叠加。

    关键修复（2026-09-16，CMYK 比对口径）：
    PLATE 产物是 **ICC 真分色 CMYK**（FOGRA39，含黑版生成），而此前参考图用
    PIL 朴素 `convert('CMYK')`（简单减色，**K 通道恒为 0**，无黑版、无 TAC/色域映射）。
    产物有黑版、参考无黑版 → K 通道系统性错配，把 RMSE 人为推高（实测：金地
    low=36.49、商用图 32.51，虚高到误判失败）。

    现改为**同口径比对**：参考图经**同一 ICC** 分色为 CMYK 后再与产物比对。
    实测修复后：金地 36.49→0.96、商用图 32.51→2.43，与通过的 design 线样本
    （水墨 low=2.53）同量级，证明此前失败是测量口径问题而非产物缺陷。

    ICC 不可用（文件缺失 / 无 ImageCms）时回退朴素转换，并在 metrics 中如实标注
    `color_managed=False`，绝不静默当作已做色彩管理。

    Args:
        icc_path: 目标 CMYK 的 ICC profile；缺省用 `profiles/CoatedFOGRA39.icc`。
    """
    from PIL import Image
    import cv2

    comp = psd.composite()  # 原生色彩模式（CMYK 或 RGB），由 psd_tools 负责 opacity/blend
    src = Image.fromarray(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB))

    scale = min(1.0, target_long / max(comp.size))
    if scale < 1.0:
        new_size = (max(1, int(comp.size[0] * scale)), max(1, int(comp.size[1] * scale)))
        comp = comp.resize(new_size, Image.BILINEAR)
        src = src.resize(new_size, Image.BILINEAR)

    _cm = True  # 是否真正做到同口径色彩管理
    if comp.mode == "CMYK":
        ref = _src_to_cmyk_same_icc(src, icc_path)
        if ref is None:  # ICC 不可用 → 回退朴素转换（显式标注，避免静默失真）
            ref = src.convert("CMYK")
            _cm = False
    else:
        ref = src.convert("RGB") if src.mode != "RGB" else src

    ca = np.asarray(comp, dtype=np.float32)
    ra = np.asarray(ref, dtype=np.float32)
    raw = float(np.sqrt(((ca - ra) ** 2).mean()))
    ca16 = cv2.GaussianBlur(ca, (0, 0), 16)
    ra16 = cv2.GaussianBlur(ra, (0, 0), 16)
    low = float(np.sqrt(((ca16 - ra16) ** 2).mean()))
    return raw, low, _cm


def _src_to_cmyk_same_icc(src_rgb, icc_path: str | None = None):
    """把源图 RGB 用**与产物相同的 ICC** 分色为 CMYK（同口径比对的参考图）。

    Returns:
        PIL Image（CMYK 墨量图）；ICC 缺失或无 ImageCms 时返回 None（调用方回退）。
    """
    from pathlib import Path

    path = Path(icc_path) if icc_path else Path(__file__).resolve().parents[1] / "profiles" / "CoatedFOGRA39.icc"
    if not path.is_file():
        return None
    try:
        import numpy as np
        from engine.core.color_manager import ColorManager
    except Exception:
        return None
    try:
        arr = np.asarray(src_rgb.convert("RGB"))
        cmyk = ColorManager._rgb_to_cmyk_icc(arr, str(path))  # (H, W, 4) 墨量 0..255
        from PIL import Image
        return Image.fromarray(cmyk.astype("uint8"), mode="CMYK")
    except Exception:
        return None


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
    #    2026-09-16 追加：**加工层（冲孔/烫金/陷印/专色）也必须排除**——它们整版
    #    施加、Alpha 天然全画布，纳入 union 会把 union 撑到 100% 使 lost 恒为 0。
    #    同时修掉 CMYK 产物上「C,M,Y 当 R,G,B」的比对基准错位（改用 _layer_rgb）。
    d = {"passed": True, "metrics": {}, "issues": []}
    if source_image and os.path.isfile(source_image):
        from engine.core.io_utils import imread_unicode
        import cv2
        src = cv2.cvtColor(imread_unicode(source_image), cv2.COLOR_BGR2RGB)
        src16 = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
        g16 = cv2.cvtColor(src16, cv2.COLOR_RGB2GRAY)
        ink = g16 < (np.median(g16) - 12)
        base_ly = _find_base_layer(layers)
        if base_ly is not None:
            brgb = _layer_rgb(base_ly)
            # 图层 numpy 只覆盖 bbox：补白到整画布后再与源图比对
            bh, bw = brgb.shape[:2]
            bx0, by0 = base_ly.bbox[0], base_ly.bbox[1]
            bh, bw = min(bh, H - by0), min(bw, W - bx0)
            canvas = np.full((H, W, 3), 255.0, np.float32)
            canvas[by0:by0 + bh, bx0:bx0 + bw] = brgb[:bh, :bw]
            erased = np.abs(canvas.astype(np.int16) - src16.astype(np.int16)).max(axis=2) > 40
            union = np.zeros((H, W), bool)
            carriers = []
            for name_i, m in masks.items():
                if any(p in name_i for p in TH["process_names"]):
                    continue  # 加工层不是内容承载者
                union |= m
                carriers.append(name_i)
            d["metrics"]["carrier_layers"] = len(carriers)
            d["metrics"]["excluded_process_layers"] = len(masks) - len(carriers)
            lost = int((ink & erased & ~union).sum())
            total_ink = max(1, int(ink.sum()))
            d["metrics"].update(ink_px=total_ink, lost_px=lost, lost_ratio=round(lost / total_ink, 6),
                                erased_pct=round(100.0 * float(erased.mean()), 2),
                                union_pct=round(100.0 * float(union.mean()), 2),
                                ink_erased_pct=round(100.0 * float((ink & erased).sum()) / total_ink, 3))
            if lost / total_ink > TH["content_loss_ratio"]:
                d["passed"] = False
                d["issues"].append(f"内容丢失 {lost:,}px（{lost/total_ink*100:.3f}% 墨迹）")
        else:
            d["metrics"]["note"] = "未找到底板层，跳过"
            d["passed"] = True
    else:
        d["metrics"]["note"] = "未提供源图，跳过"
    dims["④ 内容承载"] = d

    # ⑤ 合成等价性（真实合成：psd_tools 渲染，尊重每层的 opacity / blend / CMYK）
    d = {"passed": True, "metrics": {}, "issues": []}
    if source_image and os.path.isfile(source_image):
        try:
            from engine.core.io_utils import imread_unicode
            src_bgr = imread_unicode(source_image)
            raw_rmse, low_rmse, cm_ok = _composite_rmse(psd, src_bgr, target_long=1600)
            d["metrics"].update(rmse_raw=round(raw_rmse, 2), rmse_lowfreq=round(low_rmse, 2),
                                color_managed=bool(cm_ok))
            if not cm_ok:
                d["metrics"]["note"] = ("ICC 不可用，已回退 PIL 朴素 CMYK 转换——"
                                        "该口径与产物 ICC 分色不同，RMSE 可能虚高")
            if low_rmse > TH["synth_rmse_lowfreq"] or raw_rmse > TH["synth_rmse_raw"]:
                d["passed"] = False
                d["issues"].append(f"合成与原图差异偏大（低频RMSE={low_rmse:.1f}, 原始RMSE={raw_rmse:.1f}）")
        except Exception as e:
            d["metrics"]["note"] = f"合成比对异常（{type(e).__name__}），跳过"
            d["evaluated"] = False
    else:
        d["metrics"]["note"] = "未提供源图，跳过"
    dims["⑤ 合成等价性"] = d

    # ⑥ 底板纯净度
    #    2026-09-12 定稿：门禁判定用**通用项**（底板存在 + 非空——底板缺失/空层
    #    是致命缺陷）；分区域 dark 率是**品类相关指标**（山水图的区域先验不适用
    #    壁布等其他品类），降级为 metrics 披露，深查由
    #    tools/calibrate_density_bands.py + 人工/金标准承担。
    d = {"passed": True, "metrics": {}, "issues": []}
    base_ly = _find_base_layer(layers)
    if base_ly is None:
        d["passed"] = False
        d["issues"].append("缺少底板层（Base_Ground/底版/底板/Substrate/Gold_Base/Fabric_Base 关键词均未命中）")
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
            # 2026-09-16：改用 _layer_rgb 取呈色；此前 ba[:, :, :3] 在 CMYK 产物上
            # 取到的是 C,M,Y（当成了 R,G,B），灰度换算基准错位。
            brgb = _layer_rgb(base_ly)
            bs = brgb[::max(1, brgb.shape[0] // 2000), ::max(1, brgb.shape[1] // 4000)]
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
            # 2026-09-13 新增：制版线（CMYK）**禁止生成内容**——LaMa 等生成式补全必须
            # 已被确定性算法替换（run_universal_engine 的 allow_generative=False 分支）。
            # 这是防御性门禁：一旦 plate 产物混入生成像素，此处直接拦截。
            gen_ratio = tot.get("generated_pixel_ratio")
            gen_deoc = tot.get("deocclusion_generative")
            d["metrics"].update(generated_pixel_ratio=gen_ratio,
                                deocclusion_generative=gen_deoc)
            if gen_deoc is True or (isinstance(gen_ratio, (int, float)) and gen_ratio > 0):
                d["passed"] = False
                d["issues"].append(
                    f"PLATE 含生成内容（deocclusion_generative={gen_deoc}, "
                    f"generated_pixel_ratio={gen_ratio}）——制版线须为确定性输出")
        # 采样 K 版非空（真黑版判定：ICC 分色生效）
        # ⚠️ 此处的 a[:, :, 3] 是**有意**取 K 通道，不是 Alpha（勿按 _alpha 的口径改）：
        #    CMYK 层 numpy() = (H,W,5) = C,M,Y,K,Alpha，index 3 恰为 K。
        #    本项仅在 is_cmyk 分支内执行，故 5 通道布局成立。
        try:
            big = max(content, key=lambda l: (l.bbox[2] - l.bbox[0]) * (l.bbox[3] - l.bbox[1]))
            a = big.numpy()
            if a.shape[2] >= 5:
                kt = a[:, :, 3].max()
                norm = 255.0 if kt > 1.001 else 1.0
                sub = a[::max(1, a.shape[0] // 300), ::max(1, a.shape[1] // 300)]
                d["metrics"]["k_channel_nonzero_pct"] = round(float((sub[:, :, 3] > 0.1 * norm).mean()) * 100, 1)
        except Exception:
            pass
    else:
        d["metrics"]["color_mode"] = "rgb（design 线，plate 项不适用）"
        # design 线允许生成式补全（LaMa）；此处仅**披露**生成内容占比，
        # 供人工确认该产物能否直接用于制版（若要制版应改走 plate 线）。
        if manifest_path and os.path.isfile(manifest_path):
            try:
                _tot = json.load(open(manifest_path, encoding="utf-8")).get("totals", {})
                d["metrics"]["generated_pixel_ratio"] = _tot.get("generated_pixel_ratio")
                d["metrics"]["deocclusion_engine"] = _tot.get("deocclusion_engine")
                d["metrics"]["deocclusion_generative"] = _tot.get("deocclusion_generative")
                if _tot.get("deocclusion_generative") is True:
                    d["metrics"]["note"] = "design 线含生成式补全；如需制版请走 plate 线（确定性）"
            except Exception:
                pass
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

    # ===== 标注每个维度的「是否真正用数据核验(evaluated)」与「是否本产品线不适用(na)」 =====
    # 供 episode 严格准入判断：未核验/不适用绝不能当"通过"。阈值本身（passed 取值）
    # 完全沿用上面各维计算，这里不降低任何重建质量阈值。
    no_src = not (source_image and os.path.isfile(source_image))
    no_manifest = not (manifest_path and os.path.isfile(manifest_path))
    if no_src:
        dims["④ 内容承载"]["evaluated"] = False
        dims["⑤ 合成等价性"]["evaluated"] = False
    if not is_cmyk:
        # design 线（RGB）：⑦ plate 合规本身不适用，明确标 na，且不做核验判定
        dims["⑦ plate 合规"]["na"] = True
        dims["⑦ plate 合规"]["evaluated"] = False
    if no_manifest:
        # 无 manifest 无法核验 TAC/纯净性（⑦）与层清单一致性（⑧）→ 视为未核验
        if is_cmyk:
            dims["⑦ plate 合规"]["evaluated"] = False
        dims["⑧ manifest 一致"]["evaluated"] = False
    for _k, _v in dims.items():
        _v.setdefault("evaluated", True)
        _v.setdefault("na", False)

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
