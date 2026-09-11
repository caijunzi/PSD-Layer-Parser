"""墨密度场与密度精修（检测分支专用）。

原理（2026-09-11 山水图原型实证）：
    G = percentile(gray, 88)              # 全局金地参考（金箔材质全画布统一）
    D = -ln(clip(gray / G, 0.01, 1)) / ln(50)   # 墨密度 0=纯金地 ~1=浓墨
    D_s = GaussianBlur(D, sigma)          # 抑金地纹理噪声，保晕染结构

    refined = mask ∩ (D_s > floor) + close + open

用途：SAM 对无边界水墨元素（峭壁/远山）产出"方向对但弥散"的大掩模，
密度场负责甄别"哪些是真墨、哪些是金地噪声"——两者相乘得到干净层。

R2 合规：密度场仅服务掩模检测分支，绝不写回输出像素。
RK-16 合规：计算为确定性算子，无随机源。
"""
from typing import Optional

import cv2
import numpy as np

# 默认参数（原型实测校准：source_4000.jpg 金地屏风）
DEFAULT_GOLD_PERCENTILE = 88
DEFAULT_SIGMA = 4.0
DEFAULT_FLOOR = 0.12
DEFAULT_CLOSE_KERNEL = 9
DEFAULT_OPEN_KERNEL = 3


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 或灰度 → float64 灰度 [0,1]。"""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64) / 255.0


def ink_density(
    image: np.ndarray,
    gold_percentile: float = DEFAULT_GOLD_PERCENTILE,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """计算平滑后的墨密度场 D_s。

    注意：金地参考取**全局分位数**，不能用局部平场——局部平场会把大面积
    浓墨区内部"漂白"（该区域背景估计被墨拉暗 → T≈1 → D≈0），原型实测踩坑。
    """
    gray = to_gray(image)
    gold = float(np.percentile(gray, gold_percentile))
    transmittance = np.clip(gray / max(gold, 1e-6), 0.01, 1.0)
    density = -np.log(transmittance) / np.log(50.0)
    if sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigma)
    return density


def refine_mask(
    mask: np.ndarray,
    density: np.ndarray,
    floor: float = DEFAULT_FLOOR,
    close_kernel: int = DEFAULT_CLOSE_KERNEL,
    open_kernel: int = DEFAULT_OPEN_KERNEL,
) -> np.ndarray:
    """密度精修：掩模只保留密度高于地板线的（真墨）像素，并做形态学清理。"""
    binary = (mask > 127).astype(np.uint8) * 255
    refined = cv2.bitwise_and(binary, (density > floor).astype(np.uint8) * 255)
    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k)
    if open_kernel > 1:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k)
    return refined


def mask_stats(mask: np.ndarray) -> dict:
    """掩模统计：像素数、覆盖率、外接框与外接框覆盖比。"""
    binary = mask > 127
    pixels = int(np.count_nonzero(binary))
    total = binary.size
    if pixels == 0:
        return {"pixels": 0, "coverage": 0.0, "bbox": None, "bbox_ratio": 0.0}
    ys, xs = np.nonzero(binary)
    h, w = binary.shape
    bw = int(xs.max()) - int(xs.min()) + 1
    bh = int(ys.max()) - int(ys.min()) + 1
    return {
        "pixels": pixels,
        "coverage": pixels / total,
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        "bbox_ratio": (bw * bh) / float(w * h),
    }


def diffuse_reason_bbox_ratio(mask: np.ndarray) -> Optional[float]:
    """返回外接框覆盖比（供门限判断）。"""
    return mask_stats(mask)["bbox_ratio"]
