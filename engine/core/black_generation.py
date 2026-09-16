# -*- coding: utf-8 -*-
"""黑版生成曲线（GCR 按品类调优）—— 2026-09-16 新增。

缺口
----
ICC 分色（`ColorManager._rgb_to_cmyk_icc`）按 profile 内建的 A2B/B2A 表把 RGB → CMYK，
**K 的多少完全由 profile 决定**；引擎此前只能事后用 `ink_limiter.limit_ink` 的 MaxK
夹一个上限，**无法按品类调整「灰成分在 K 与 CMY 之间的分配」**（即黑版生成/GCR）。
README 的「ICC 黑版生成曲线按品类调优待做」即指此项。

本模块的位置
------------
ICC 分色 **之后** → TAC 压制 **之前**（`engine/psb_builder.py::_to_cmyk_limited`）。
先调 K 的分配，再压 TAC，保证压制结果仍是合规的最终像素。

算法（在「墨量空间」上进行；0..255，255 = 满墨）
--------------------------------------------
令 `ink = 255 - raw`（`raw` 为 PSD 磁盘反码，255 = 0% 墨）：

1. ``ink_K' = clamp(255 * (ink_K/255)**gamma * gain + shift, 0, 255)``
2. ``delta  = ink_K' - ink_K``
3. ``ink_C/M/Y' = clamp(ink_C/M/Y - delta, 0, 255)``  ← 灰量在 K 与 CMY 间**等量转移**

第 3 步是 GCR 的本质：CMY 等量 ≈ K（灰）。把 K 增减的灰量从 CMY 里等量取走/补回，
可在**基本保持合成色的同时**改变黑版分配，且总墨量近似不变（不额外推高 TAC）。

参数（preset `black_generation`，全部可省，默认恒等）
----------------------------------------------------
| 键 | 默认 | 含义 |
|---|---|---|
| `k_gain`   | 1.0 | K 墨量增益（>1 整体加深黑版，<1 整体减淡） |
| `k_gamma`  | 1.0 | K 墨量幂次（``ink^gamma``）：**>1 减弱中间调 K 墨量（黑版变淡），<1 加深中间调**；两端点不变 |
| `k_shift_pct` | 0.0 | K 墨量平移（百分点，±100 以内） |

⚠️ **恒等即零拷贝**：`k_gain==1 and k_gamma==1 and k_shift_pct==0` 时直接返回入参，
保证未配置该键的品类**产物字节不变**（RK-16 逐像素可复现）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

INK_MAX = 255.0

# 参数合法范围（防误配导致产物失控）
_GAIN_RANGE = (0.0, 2.0)
_GAMMA_RANGE = (0.1, 3.0)
_SHIFT_RANGE = (-100.0, 100.0)


@dataclass(frozen=True)
class BlackGenerationPolicy:
    """黑版生成策略（工艺常量，集中定义、可追溯）。"""

    k_gain: float = 1.0
    k_gamma: float = 1.0
    k_shift_pct: float = 0.0
    source: str = "默认恒等（未配置 black_generation）"
    category: str = ""

    @property
    def is_identity(self) -> bool:
        return (abs(self.k_gain - 1.0) < 1e-9
                and abs(self.k_gamma - 1.0) < 1e-9
                and abs(self.k_shift_pct) < 1e-9)


IDENTITY = BlackGenerationPolicy()


def resolve_black_generation(config: Optional[dict]) -> BlackGenerationPolicy:
    """从 preset `black_generation` 配置解析策略（缺省/关闭 → 恒等）。

    支持 `enabled: false` 显式关闭。越界值会被夹取到合法范围并记入 source。
    """
    if not isinstance(config, dict) or not config.get("enabled", True):
        return IDENTITY
    gain = float(config.get("k_gain", 1.0))
    gamma = float(config.get("k_gamma", 1.0))
    shift = float(config.get("k_shift_pct", 0.0))
    notes = []
    if not (_GAIN_RANGE[0] <= gain <= _GAIN_RANGE[1]):
        notes.append(f"k_gain {gain} 越界 → 夹取")
        gain = min(max(gain, _GAIN_RANGE[0]), _GAIN_RANGE[1])
    if not (_GAMMA_RANGE[0] <= gamma <= _GAMMA_RANGE[1]):
        notes.append(f"k_gamma {gamma} 越界 → 夹取")
        gamma = min(max(gamma, _GAMMA_RANGE[0]), _GAMMA_RANGE[1])
    if not (_SHIFT_RANGE[0] <= shift <= _SHIFT_RANGE[1]):
        notes.append(f"k_shift_pct {shift} 越界 → 夹取")
        shift = min(max(shift, _SHIFT_RANGE[0]), _SHIFT_RANGE[1])
    note = config.get("comment") or config.get("note") or ""
    src = "preset black_generation"
    if note:
        src += f"（{note}）"
    if notes:
        src += "；" + "；".join(notes)
    return BlackGenerationPolicy(gain, gamma, shift, src,
                                 str(config.get("category") or ""))


def apply_black_generation(
    cmyk_raw: np.ndarray,
    policy: Optional[BlackGenerationPolicy],
    inplace: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """对 CMYK 磁盘反码施加黑版生成曲线 + CMY 等量补偿。

    Args:
        cmyk_raw: (4, H, W) uint8，**PSD 磁盘反码**（255 = 0% 墨）
        policy: 策略；None 或恒等 → 零拷贝返回入参
        inplace: 允许原地修改时省一次整幅复制

    Returns:
        (处理后 cmyk_raw, 统计字典)
    """
    if cmyk_raw.ndim != 3 or cmyk_raw.shape[0] != 4:
        raise ValueError(f"apply_black_generation 期望 (4,H,W)，收到 {cmyk_raw.shape}")

    if policy is None or policy.is_identity:
        k_ink = (INK_MAX - cmyk_raw[3].astype(np.float32))
        return cmyk_raw, {
            "black_gen_applied": False,
            "k_gain": 1.0, "k_gamma": 1.0, "k_shift_pct": 0.0,
            "k_ink_mean_before": round(float(k_ink.mean()) / INK_MAX * 100, 3),
            "k_ink_mean_after": round(float(k_ink.mean()) / INK_MAX * 100, 3),
            "tac_delta_mean": 0.0,
            "source": (policy.source if policy else "未提供策略"),
        }

    out = cmyk_raw if inplace else cmyk_raw.copy()

    ink_k = (INK_MAX - cmyk_raw[3].astype(np.float32))          # 0..255
    ink_c = (INK_MAX - cmyk_raw[0].astype(np.float32))
    ink_m = (INK_MAX - cmyk_raw[1].astype(np.float32))
    ink_y = (INK_MAX - cmyk_raw[2].astype(np.float32))

    # 1) K 曲线
    ink_k_new = np.power(np.clip(ink_k / INK_MAX, 0.0, 1.0), policy.k_gamma) \
        * INK_MAX * policy.k_gain + (policy.k_shift_pct / 100.0 * INK_MAX)
    ink_k_new = np.clip(ink_k_new, 0.0, INK_MAX)

    # 2) 灰量等量转移：K 增 → CMY 减；K 减 → CMY 增
    delta = ink_k_new - ink_k

    # 量化方向：对墨量做 floor，等价于反码 ceiling —— 保证不因量化抬高墨量
    out[0] = np.clip(np.floor(INK_MAX - np.clip(ink_c - delta, 0.0, INK_MAX)), 0, INK_MAX).astype(np.uint8)
    out[1] = np.clip(np.floor(INK_MAX - np.clip(ink_m - delta, 0.0, INK_MAX)), 0, INK_MAX).astype(np.uint8)
    out[2] = np.clip(np.floor(INK_MAX - np.clip(ink_y - delta, 0.0, INK_MAX)), 0, INK_MAX).astype(np.uint8)
    out[3] = np.clip(np.floor(INK_MAX - ink_k_new), 0, INK_MAX).astype(np.uint8)

    tac_before = (ink_c + ink_m + ink_y + ink_k).mean()
    tac_after = ((INK_MAX - out[0].astype(np.float32) + INK_MAX - out[1].astype(np.float32)
                  + INK_MAX - out[2].astype(np.float32) + INK_MAX - out[3].astype(np.float32))).mean()

    stats = {
        "black_gen_applied": True,
        "k_gain": policy.k_gain,
        "k_gamma": policy.k_gamma,
        "k_shift_pct": policy.k_shift_pct,
        "k_ink_mean_before": round(float(ink_k.mean()) / INK_MAX * 100, 3),
        "k_ink_mean_after": round(float(ink_k_new.mean()) / INK_MAX * 100, 3),
        "tac_delta_mean": round(float(tac_after - tac_before) / INK_MAX * 100, 3),
        "source": policy.source,
    }
    return out, stats
