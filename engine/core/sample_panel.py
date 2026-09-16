# -*- coding: utf-8 -*-
"""实物样品照的「实体样块」边界检测（零硬编码，2026-09-16 新增）。

用途
----
壁布/面料等**实物样品照** = 中间一块凸起的实体样块 + 四周墙面底衬 + 投影/硬边 + 水印。
分层前需要先判定「样块有效区」，把样块外的墙面底衬、投影、水印统一归入
**「画面外背景带」层**，避免它们混入语义检测与制版内容。

为什么不能复用 `_detect_painting_roi`
------------------------------------
`ZeroHardcodeSegmentationProvider._detect_painting_roi` 基于**灰度对比**
（边缘带参考 vs 中心参考）。但样品照的墙面底衬常与样块**同类同色系纹样**
（实测 damask 样品照：灰度差仅 3.3 < min_contrast 10；HSV 同色系 H≈18），
该方法会直接判定"无外框"并返回整幅 → 背景带层为空。

本模块的方法
------------
改用**边缘持续性（edge persistence）**：
  - 样块边是一条**贯穿画幅的长直边** —— 在几乎每一行/列都有强梯度；
  - 纹样自身的曲线边只在少数位置有梯度。

对每个列 c 统计 `|I[:, c+1] - I[:, c]| > thr` 的行占比，高持续性列即竖直边；
行同理。样块四边 = 高持续性行列的 min/max。

实测（`inputs/damask_sample.png` 1536×1024，真值 rows 48~52/968~970、cols 75~78/1454~1458）：
  样块边持续性 **0.81~0.91**，纹样自身边仅 **0.42~0.47**（≈2×），
  Top4 命中列 76/1457、行 49/968 → 与真值完全吻合。

零硬编码：不含任何绝对像素坐标；无线索/未内缩时回退 `None`（= 无样块，整幅即内容）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# 默认参数（均与画幅无关，由内容统计导出）
DEFAULT_GRAD_THRESHOLD = 12.0     # 单像素梯度超过此值才算"边"
DEFAULT_MIN_PERSISTENCE = 0.45    # 某列/行中"有边"的占比超过此值才算长直边
DEFAULT_MIN_SIDE_RATIO = 0.20     # 检出样块边长至少占画幅的比例（防小噪声）
DEFAULT_MIN_INSET_RATIO = 0.01    # 样块四边相对画幅至少内缩的比例（证明存在"外部"）


def detect_sample_panel_bbox(
    gray: np.ndarray,
    grad_threshold: float = DEFAULT_GRAD_THRESHOLD,
    min_persistence: float = DEFAULT_MIN_PERSISTENCE,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
    min_inset_ratio: float = DEFAULT_MIN_INSET_RATIO,
) -> Optional[Tuple[int, int, int, int]]:
    """检测实体样块边界，返回 ``(top, bottom, left, right)`` 像素坐标；无线索返回 None。

    判定依据：长直边的**持续性**（沿整行/整列强梯度的占比）。
    返回前会校验：① 边长足够大；② 四边均有内缩（否则视为"无外部/整幅即内容"）。
    """
    if gray is None or gray.ndim != 2:
        return None
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 16 or w < 16:
        return None

    dx = np.abs(np.diff(g, axis=1))   # (h, w-1) 竖直边强度（列方向）
    dy = np.abs(np.diff(g, axis=0))   # (h-1, w) 水平边强度（行方向）
    col_persist = (dx > grad_threshold).mean(axis=0)   # 每列
    row_persist = (dy > grad_threshold).mean(axis=1)   # 每行

    cols = np.where(col_persist >= min_persistence)[0]
    rows = np.where(row_persist >= min_persistence)[0]
    if cols.size < 2 or rows.size < 2:
        return None

    left, right = int(cols.min()), int(cols.max())
    top, bottom = int(rows.min()), int(rows.max())

    # ① 边长足够（防把细碎高频纹理误当样块）
    if (right - left) < min_side_ratio * w or (bottom - top) < min_side_ratio * h:
        return None
    # ② 四边均需内缩（否则无"样块外"，整幅即内容）
    inset_x = min_inset_ratio * w
    inset_y = min_inset_ratio * h
    if not (left > inset_x and top > inset_y
            and right < w - 1 - inset_x and bottom < h - 1 - inset_y):
        return None
    return top, bottom, left, right


def sample_panel_roi(
    gray: np.ndarray,
    grad_threshold: float = DEFAULT_GRAD_THRESHOLD,
    min_persistence: float = DEFAULT_MIN_PERSISTENCE,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
) -> Optional[np.ndarray]:
    """返回样块有效区的布尔掩码（True = 样块内）；无线索返回 None。"""
    bbox = detect_sample_panel_bbox(
        gray, grad_threshold=grad_threshold,
        min_persistence=min_persistence, min_side_ratio=min_side_ratio,
    )
    if bbox is None:
        return None
    top, bottom, left, right = bbox
    roi = np.zeros(gray.shape[:2], dtype=bool)
    roi[top:bottom + 1, left:right + 1] = True
    return roi


def resolve_sample_panel_roi(img_bgr: np.ndarray, cfg: Optional[dict]) -> Optional[np.ndarray]:
    """按 preset 的 ``sample_panel`` 配置从 BGR 图解析样块 ROI（返回 bool 掩码或 None）。

    配置项（均带默认值，可不写）：
      - ``enabled``:           是否启用（默认 False）
      - ``grad_threshold``:    梯度阈值（默认 12.0）
      - ``min_persistence``:   长直边持续性阈值（默认 0.45）
      - ``min_side_ratio``:    最小边长占比（默认 0.20）
    """
    cfg = cfg or {}
    if not cfg.get("enabled"):
        return None
    if img_bgr is None or img_bgr.ndim != 3:
        return None
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return sample_panel_roi(
        gray,
        grad_threshold=float(cfg.get("grad_threshold", DEFAULT_GRAD_THRESHOLD)),
        min_persistence=float(cfg.get("min_persistence", DEFAULT_MIN_PERSISTENCE)),
        min_side_ratio=float(cfg.get("min_side_ratio", DEFAULT_MIN_SIDE_RATIO)),
    )
