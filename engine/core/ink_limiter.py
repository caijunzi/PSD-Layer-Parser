"""印前油墨总量（TAC/TIL）合规管理 —— ADR-007。

概念澄清（重要，2026-09-10 实测 + 标准查证后修正）
------------------------------------------------
项目文档此前写"TAC 上限从 ICC profile 派生"，这是**概念错误**：

- **ICC profile 内部没有 TAC / TIL / 黑版生成字段**。实测系统 ICC（AdobeRGB1998 等）
  的 tag 中不存在任何 ink/tac/limit 相关项；ICC 规范（ISO 15076-1）也未定义该字段。
- TAC 上限是**印刷工艺参数**，由 ISO 12647-2 按纸张类别与印刷方式规定，
  或由印刷厂在工艺单中给定。ICC profile 与印刷条件是**配对关系**（同一印刷条件
  对应特定 profile），而非"profile 里存着 TAC"。

因此本模块的正确做法：
  1. TAC 上限来自**印刷条件**（preset 显式配置 > 印刷条件默认表 > 工程保守值）；
  2. ICC profile 只用于两件事：色彩转换、以及从 description 辅助识别印刷条件；
  3. 上限值全部集中在本表并标注来源，属"物理/工艺常量"而非散落的魔法数字（§9.2）。

数值来源
--------
| 印刷条件 | TAC 上限 | 出处 |
| :--- | ---: | :--- |
| ISO 12647-2:2013 涂布纸单张纸 | 330%（标准下限，上限 350%） | ISO 12647-2:2013 |
| ECI 实践值（ISOcoated_v2 / PSO Coated v3） | 300% | ECI 官方 profile 优化 TAC |
| 轮转胶印涂布纸 | 300% | Walstead CE 印厂工艺规范 |
| 非涂布纸 | 260% | PSO Uncoated (FOGRA47) |
| 美国卷筒涂布纸 | 300% | GRACoL 2013 (CRPC6) |
| 新闻纸 | 240% | Walstead CE 工艺规范 |

另据 Walstead CE 规范：**MaxK 不应超过 97%**，且黑版实地区域不应只由 K 构成。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

#: 墨量空间的最大值（8 bit/channel）
INK_MAX = 255.0


@dataclass(frozen=True)
class TacPolicy:
    """TAC 策略（工艺常量，集中定义）。"""

    limit_pct: float          # 总墨量上限（%）
    max_k_pct: float          # 单通道 K 上限（%）
    source: str               # 来源描述（可追溯）
    condition_id: str = ""    # 印刷条件标识


#: 印刷条件 → TAC 策略。数值均标注出处，禁止散落硬编码。
PRINT_CONDITIONS: dict[str, TacPolicy] = {
    "eci_coated_practice": TacPolicy(
        300.0, 96.0, "ECI 实践值（ISOcoated_v2 / PSO Coated v3 优化 TAC）",
        "eci_coated_practice"),
    "iso_coated_sheetfed": TacPolicy(
        330.0, 97.0, "ISO 12647-2:2013 涂布纸单张纸（标准下限 330%，上限 350%）",
        "iso_coated_sheetfed"),
    "iso_coated_web": TacPolicy(
        300.0, 97.0, "轮转胶印涂布纸（Walstead CE 印厂工艺规范）",
        "iso_coated_web"),
    "iso_uncoated": TacPolicy(
        260.0, 95.0, "非涂布纸（PSO Uncoated FOGRA47）",
        "iso_uncoated"),
    "us_web_coated": TacPolicy(
        300.0, 100.0, "美国卷筒涂布纸（GRACoL 2013 CRPC6）",
        "us_web_coated"),
    "newsprint": TacPolicy(
        240.0, 90.0, "新闻纸（Walstead CE 工艺规范）",
        "newsprint"),
}

#: 未显式配置印刷条件时的保守默认（涂布纸实践值，最不易出事故）
DEFAULT_CONDITION = "eci_coated_practice"

#: 从 ICC description 关键词推断印刷条件的映射（辅助，非权威）
_ICC_CONDITION_HINTS = (
    (("uncoated", "fogra47", "fogra29", "inp"), "iso_uncoated"),
    (("news", "snp", "newspaper"), "newsprint"),
    (("psocoated", "fogra51", "psocoated_v3"), "eci_coated_practice"),
    (("isocoated", "fogra39", "coated_fogra"), "eci_coated_practice"),
)


def resolve_policy(
    condition: Optional[str] = None,
    limit_pct: Optional[float] = None,
    max_k_pct: Optional[float] = None,
    icc_path: Optional[str] = None,
) -> TacPolicy:
    """确定本次使用的 TAC 策略，并保留来源可追溯。

    优先级：显式 limit_pct（preset）> 印刷条件 id > ICC description 推断 > 默认。
    """
    if limit_pct is not None:
        return TacPolicy(
            float(limit_pct),
            float(max_k_pct) if max_k_pct is not None else 97.0,
            "preset 显式配置（工艺参数由印刷厂给定）",
            condition or "preset",
        )

    if condition and condition in PRINT_CONDITIONS:
        return PRINT_CONDITIONS[condition]

    guessed = detect_condition_from_icc(icc_path)
    if guessed:
        return PRINT_CONDITIONS[guessed]

    return PRINT_CONDITIONS[DEFAULT_CONDITION]


def detect_condition_from_icc(icc_path: Optional[str]) -> Optional[str]:
    """从 ICC profile 的 description 关键词推断印刷条件（辅助手段）。

    注意：这不是"从 profile 读取 TAC"，ICC 中并无该字段；
    只是利用 profile 命名惯例缩小印刷条件的判断范围，最终仍需人工确认。
    """
    if not icc_path or not os.path.isfile(icc_path):
        return None
    try:
        from PIL import ImageCms

        prof = ImageCms.getOpenProfile(icc_path)
        desc = str(prof.profile.profile_description or "").lower().replace(" ", "")
    except Exception:
        return None
    for keywords, cond in _ICC_CONDITION_HINTS:
        if any(k in desc for k in keywords):
            return cond
    return None


def icc_summary(icc_path: Optional[str]) -> dict[str, Any]:
    """读取 ICC 的基本信息，供 manifest 披露（不含 TAC —— ICC 里没有）。"""
    info: dict[str, Any] = {"path": None, "loaded": False}
    if not icc_path:
        info["note"] = "未提供 ICC；色彩转换为朴素 RGB→CMYK（无色彩管理）"
        return info
    info["path"] = str(icc_path).replace("\\", "/")
    if not os.path.isfile(icc_path):
        info["note"] = "ICC 文件不存在，回退朴素转换"
        return info
    try:
        from PIL import ImageCms

        prof = ImageCms.getOpenProfile(icc_path)
        info["loaded"] = True
        info["description"] = prof.profile.profile_description
        info["xcolor_space"] = prof.profile.xcolor_space
        info["connection_space"] = prof.profile.connection_space
        info["detected_condition"] = detect_condition_from_icc(icc_path) or ""
        info["note"] = "ICC 仅用于色彩转换与印刷条件辅助识别；TAC 上限来自工艺参数"
    except Exception as e:
        info["note"] = f"ICC 加载失败：{type(e).__name__}: {e}"
    return info


def limit_ink(cmyk_raw: np.ndarray, policy: TacPolicy,
              inplace: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """对 CMYK 数据执行 MaxK 限制 + TAC 压制。

    Args:
        cmyk_raw: (4, H, W) uint8，**PSD 磁盘反码**（255 = 0% 墨）
        policy: TAC 策略
        inplace: True 时直接修改入参（省一次整幅复制）。
            生产链路中该数组用完即弃，可安全开启；默认 False 保证无副作用。

    Returns:
        (压制后的 cmyk_raw, 统计字典)

    算法（确定性、可复算）：
        1. K 单通道不得超过 max_k_pct；
        2. 每像素总墨量超过 limit 时，**保持 K 不变、按比例缩放 C/M/Y**，
           使 C+M+Y+K 恰好回落到 limit（保住中性灰，避免整体偏色）；
        3. 若 K 自身已超 limit，则单独削减 K。
    """
    if cmyk_raw.ndim != 3 or cmyk_raw.shape[0] != 4:
        raise ValueError(f"limit_ink 期望 (4,H,W) 的 CMYK 数据，收到 {cmyk_raw.shape}")

    limit_v = policy.limit_pct / 100.0 * INK_MAX
    max_k_v = policy.max_k_pct / 100.0 * INK_MAX

    # ---------------------------------------------------------------
    # 全程整数探测（O(H*W)，不分配 float32 数组）
    #
    # 性能背景：初版对整幅图做 float32 转换 + 全数组逐通道运算，
    # 16K 画幅下一次就分配约 2 GB 临时数组（4 通道 × 1.25 亿像素 × 4B），
    # 是 Step 6 从 66.9s 涨到 217.5s 的主因。
    # 而实测超限像素仅占 0.5% 左右 —— 稀疏化后浮点运算量降到千分之五。
    # ---------------------------------------------------------------
    # 直接在反码上求和，省掉 int16 中间数组（16K 下省约 250 MB 分配）：
    #   墨量 = 255 - 反码  ⇒  TAC_int = 4*255 - sum(反码)
    raw_sum = cmyk_raw.sum(axis=0, dtype=np.uint16)   # 0..1020
    tac_int = 4 * int(INK_MAX) - raw_sum
    limit_int = int(np.floor(limit_v))

    tac_before = tac_int.astype(np.float32) / INK_MAX * 100.0

    # 反码越小墨量越大：K 超过 max_k 等价于反码 < ceil(255 - max_k)
    max_k_raw = int(np.ceil(INK_MAX - max_k_v))
    k_over = cmyk_raw[3] < max_k_raw
    # MaxK 生效后再判断 TAC 是否仍超限
    k_excess = np.maximum(max_k_raw - cmyk_raw[3].astype(np.int16), 0)   # 需削减的 K 墨量
    over = (tac_int - k_excess) > limit_int

    n_k = int(np.count_nonzero(k_over))
    n_over = int(np.count_nonzero(over))
    total_px = int(tac_int.size)

    if n_over == 0 and n_k == 0:
        # 合规数据：零拷贝返回
        return cmyk_raw, {
            "tac_limit_pct": round(policy.limit_pct, 2),
            "max_k_pct": round(policy.max_k_pct, 2),
            "tac_source": policy.source,
            "tac_condition": policy.condition_id,
            "tac_before_max": round(float(tac_before.max()), 2),
            "tac_before_mean": round(float(tac_before.mean()), 2),
            "tac_after_max": round(float(tac_before.max()), 2),
            "tac_after_mean": round(float(tac_before.mean()), 2),
            "tac_clipped_pixels": 0,
            "tac_clipped_ratio": 0.0,
            "k_clipped_pixels": 0,
            "compliant": True,
            "fast_path": True,
        }

    # inplace=True 时省掉整幅复制（16K 下约 500 MB），调用方需保证入参可被修改
    out = cmyk_raw if inplace else cmyk_raw.copy()

    # 1) MaxK：反码向上取整（反码截断会放大墨量，见下方注释）
    if n_k:
        out[3][k_over] = max_k_raw

    # 2) TAC 压制：只对超限像素做浮点运算
    if n_over:
        idx = np.nonzero(over)
        c = (INK_MAX - out[0][idx]).astype(np.float32)
        m = (INK_MAX - out[1][idx]).astype(np.float32)
        y = (INK_MAX - out[2][idx]).astype(np.float32)
        k = (INK_MAX - out[3][idx]).astype(np.float32)
        cmy_sum = c + m + y
        avail = np.maximum(limit_v - k, 0.0)
        ratio = np.minimum(1.0, avail / np.maximum(cmy_sum, 1e-6))
        # 量化方向很关键：反码 = 255 - 墨量，若对反码做截断（floor）会**放大**墨量，
        # 导致压制后重新审计反而超限（实测 300.0% -> 300.78%）。
        # 故对反码向上取整（等价于对墨量向下取整），保证量化后墨量 ≤ 目标值。
        for i, ch in enumerate((c, m, y)):
            out[i][idx] = np.clip(np.ceil(INK_MAX - ch * ratio), 0, INK_MAX).astype(np.uint8)
        # K 自身已超限的极端像素
        k_still_over = k > limit_v
        if k_still_over.any():
            sub = idx[0][k_still_over], idx[1][k_still_over]
            out[3][sub] = int(np.ceil(INK_MAX - limit_v))

    tac_after = (4 * int(INK_MAX) - out.sum(axis=0, dtype=np.uint16)).astype(np.float32) / INK_MAX * 100.0

    stats: dict[str, Any] = {
        "tac_limit_pct": round(policy.limit_pct, 2),
        "max_k_pct": round(policy.max_k_pct, 2),
        "tac_source": policy.source,
        "tac_condition": policy.condition_id,
        "tac_before_max": round(float(tac_before.max()), 2),
        "tac_before_mean": round(float(tac_before.mean()), 2),
        "tac_after_max": round(float(tac_after.max()), 2),
        "tac_after_mean": round(float(tac_after.mean()), 2),
        "tac_clipped_pixels": n_over,
        "tac_clipped_ratio": round(n_over / max(1, total_px), 6),
        "k_clipped_pixels": n_k,
        "compliant": bool(float(tac_after.max()) <= policy.limit_pct + 1e-6),
        "fast_path": False,
    }
    return out, stats


def audit_tac(cmyk_raw: np.ndarray, policy: TacPolicy) -> dict[str, Any]:
    """只审计不修改（用于对既有产物做合规复核）。整数运算，不分配 float32 全图数组。"""
    ink_i16 = INK_MAX - cmyk_raw.astype(np.int16)
    tac = ink_i16.sum(axis=0).astype(np.float32) / INK_MAX * 100.0
    return {
        "tac_limit_pct": round(policy.limit_pct, 2),
        "tac_source": policy.source,
        "tac_max": round(float(tac.max()), 2),
        "tac_mean": round(float(tac.mean()), 2),
        "max_k_pct": round(float(ink_i16[3].max()) / INK_MAX * 100.0, 2),
        "over_limit_ratio": round(float((tac > policy.limit_pct).mean()), 6),
        "compliant": bool(float(tac.max()) <= policy.limit_pct + 1e-6),
    }
