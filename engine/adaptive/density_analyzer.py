"""语义密度场分析（Stage 2 Auto-Tune 基础算子）。

设计说明（与 03-implementation-plan.md §3 Step 2.1 的偏差，附理由）：
    计划原设想用「多尺度 DINO 前传」构建密度场。但实测 DINO 在本机固定 CPU
    且约 4.5s/prompt，为密度分析额外跑 DINO 代价高、且引入非确定性——
    与 RK-16 逐像素复现冲突。项目已有确定性、零模型、约 2s 的**墨密度场**
    （engine/core/density_field.py，原型实证），且真实痛点（DINO 漏检的
    05A 寒林 / 04B 平渚 / 04D 远山）恰恰是"墨迹密度可辨、语义不可辨"的类别。
    ⇒ 改用墨密度场作 Auto-Tune 基础，检测分支与 R2 合规不变。

R2 合规：密度场仅服务检测分支的参数推荐，绝不写回输出像素。
RK-16 合规：全部为确定性算子（分位数 / 高斯模糊 / 连通域），无随机源。
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from engine.core.density_field import ink_density

# 归一化区域类型：(x0, y0, x1, y1)，坐标 ∈ [0,1]
Region = Tuple[float, float, float, float]


def compute_density_map(
    image: np.ndarray,
    gold_percentile: float = 88,
    sigma: float = 4.0,
) -> np.ndarray:
    """计算平滑墨密度场 D_s（0=纯底色 ~1=浓墨）。薄封装，便于后续替换实现。"""
    return ink_density(image, gold_percentile=gold_percentile, sigma=sigma)


def block_means(density: np.ndarray, block: int = 256) -> Tuple[np.ndarray, int, int]:
    """块均值密度图（用于热力图/粗定位）。

    Returns:
        (grid, gy, gx)：grid 形状 (gy, gx)，值为该块平均密度。
    """
    h, w = density.shape
    block = max(1, int(block))
    gy = max(1, h // block)
    gx = max(1, w // block)
    # 用整块切分（丢弃不足一块的余量），保证 grid 可精确反映块位置
    ys = np.linspace(0, h, gy + 1, dtype=int)
    xs = np.linspace(0, w, gx + 1, dtype=int)
    grid = np.zeros((gy, gx), dtype=np.float64)
    for i in range(gy):
        for j in range(gx):
            cell = density[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            grid[i, j] = float(cell.mean()) if cell.size else 0.0
    return grid, gy, gx


def density_hotspots(
    density: np.ndarray,
    block: int = 256,
    thresh_percentile: float = 85.0,
    min_blocks: int = 1,
    max_regions: int = 5,
) -> List[dict]:
    """从密度场提取"墨迹热点"区域（归一化 bbox），供 region 先验推荐。

    方法：块均值 → 分位数阈值 → 连通域 → 合并为 bbox（确定性）。

    Args:
        density: 密度场（0..1）
        block: 块尺寸（像素）
        thresh_percentile: 块均值的分位阈值（超过即视为热点块）
        min_blocks: 连通域至少包含的块数（滤除孤立噪声块）
        max_regions: 返回的最大区域数（按均密度降序）

    Returns:
        [{region: (x0,y0,x1,y1) 归一化, score: float, blocks: int}, ...]
    """
    grid, gy, gx = block_means(density, block=block)
    thr = float(np.percentile(grid, thresh_percentile))
    if thr <= 0:
        # 全图近乎无墨：返回空（避免把纯底色误判为热点）
        return []
    hot = (grid >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)

    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(1, int(min_blocks)):
            continue
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        # 块索引 → 归一化坐标（用 grid 的真实覆盖范围）
        h, w = density.shape
        x0, x1 = bx / gx, (bx + bw) / gx
        y0, y1 = by / gy, (by + bh) / gy
        mask_i = (labels == i)
        score = float(grid[mask_i].mean()) if mask_i.any() else 0.0
        regions.append({
            "region": (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)),
            "score": round(score, 4),
            "blocks": area,
        })

    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:max_regions]


def region_density_band(
    density: np.ndarray,
    region: Optional[Region] = None,
    dmin_pct: float = 60.0,
    dmax_pct: float = 95.0,
) -> Optional[dict]:
    """计算某区域内的推荐密度带 (dmin, dmax) 与其覆盖率。

    复用 tools/calibrate_density_bands.py 的口径：p60→p95 = 墨迹带。
    """
    h, w = density.shape
    if region:
        x0 = int(region[0] * w); y0 = int(region[1] * h)
        x1 = int(region[2] * w); y1 = int(region[3] * h)
        sub = density[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    else:
        sub = density
    if sub.size == 0:
        return None
    dmin = float(np.percentile(sub, dmin_pct))
    dmax = float(np.percentile(sub, dmax_pct))
    # 区域内无墨迹（全为纯底色）→ 不推荐密度带（否则会给出无意义的 [0,0]）
    if dmax <= 1e-6:
        return None
    # 扁平有墨区域（分位数重合）→ 轻微展宽，保证区间非退化且单调
    if dmin >= dmax:
        dmax = min(1.0, dmin + 0.02)
    coverage = float(((sub >= dmin) & (sub <= dmax)).mean())
    return {
        "region": tuple(region) if region else (0.0, 0.0, 1.0, 1.0),
        "density_min": round(dmin, 4),
        "density_max": round(dmax, 4),
        "coverage": round(coverage, 4),
    }


def grid_regions(nx: int = 3, ny: int = 3) -> List[Tuple[str, Region]]:
    """九宫格（默认 3x3）归一化区域，与 calibrate 工具口径一致。"""
    out = []
    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix / nx, iy / ny
            out.append((f"grid_{ix}{iy}", (x0, y0, x0 + 1 / nx, y0 + 1 / ny)))
    return out
