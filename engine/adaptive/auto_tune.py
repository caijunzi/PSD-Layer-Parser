"""图级 Auto-Tune（Stage 2）：从图像自动推荐 region 先验与密度带参数。

目标：换图免调参 —— 上传一张新图，自动给出「墨迹热点区域」与「各区域墨迹密度带」，
前端展示建议卡片，用户一键采纳或微调。

设计要点：
- 基础算子是确定性墨密度场（engine/adaptive/density_analyzer.py），不依赖 DINO/GPU，
  单图约 2s，RK-16 合规。
- 输出坐标一律归一化 [0,1]，与 preset 的 region / regions 字段同构，可直接落回 preset。
- 复用 tools/calibrate_density_bands.py 的 p60→p95 墨迹带口径。

用法（CLI）：
    python -m engine.adaptive.auto_tune inputs/source_4000.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.adaptive.density_analyzer import (
    compute_density_map,
    density_hotspots,
    grid_regions,
    region_density_band,
)

Region = Tuple[float, float, float, float]


def suggest_regions(
    density: np.ndarray,
    block: int = 256,
    thresh_percentile: float = 85.0,
    max_regions: int = 5,
) -> List[Dict[str, Any]]:
    """从密度场推荐"墨迹热点"区域（供 preset region / regions 使用）。"""
    return density_hotspots(
        density,
        block=block,
        thresh_percentile=thresh_percentile,
        max_regions=max_regions,
    )


def suggest_density_bands(
    density: np.ndarray,
    classes: Optional[Sequence[Dict[str, Any]]] = None,
    regions: Optional[Sequence[Tuple[str, Region]]] = None,
    dmin_pct: float = 60.0,
    dmax_pct: float = 95.0,
) -> List[Dict[str, Any]]:
    """推荐密度带 (density_min, density_max)。

    区域来源优先级：
      1) classes 中带 `region` 的语义类（直接标注为该类名，最有用）；
      2) 显式传入的 regions；
      3) 兜底 3x3 九宫格。

    Returns:
        [{"label", "region", "density_min", "density_max", "coverage"}, ...]
    """
    labeled: List[Tuple[str, Optional[Region]]] = []

    if classes:
        for c in classes:
            r = c.get("region")
            name = (c.get("name") or c.get("layer_name") or c.get("category_id") or "类")
            if r:
                labeled.append((name, tuple(float(v) for v in r)))  # 归一化区域
            # 无 region 的类不推荐（避免全画幅假带）
        if labeled:
            out = []
            for name, reg in labeled:
                band = region_density_band(density, reg, dmin_pct, dmax_pct)
                if band:
                    band["label"] = name
                    out.append(band)
            return out

    if regions:
        out = []
        for name, reg in regions:
            band = region_density_band(density, reg, dmin_pct, dmax_pct)
            if band:
                band["label"] = name
                out.append(band)
        return out

    # 兜底：九宫格
    out = []
    for name, reg in grid_regions():
        band = region_density_band(density, reg, dmin_pct, dmax_pct)
        if band:
            band["label"] = name
            out.append(band)
    return out


def suggest_auto_tune(
    image: np.ndarray,
    classes: Optional[Sequence[Dict[str, Any]]] = None,
    gold_percentile: float = 88,
    sigma: float = 4.0,
    max_regions: int = 5,
) -> Dict[str, Any]:
    """图级 Auto-Tune 主入口：返回 region + 密度带建议。

    Args:
        image: BGR 图像（np.ndarray）
        classes: preset.ai_semantic_classes（可选；用于给带 region 的类推荐密度带）
        gold_percentile / sigma: 密度场参数（与 density_field 一致）
        max_regions: 热点区域上限

    Returns:
        {
          "global_percentiles": {"p50","p75","p90","p95"},
          "regions":  [{"region","score","blocks"}, ...],
          "density_bands": [{"label","region","density_min","density_max","coverage"}, ...],
        }
    """
    density = compute_density_map(image, gold_percentile=gold_percentile, sigma=sigma)
    p = np.percentile(density, [50, 75, 90, 95])
    return {
        "global_percentiles": {
            "p50": round(float(p[0]), 4),
            "p75": round(float(p[1]), 4),
            "p90": round(float(p[2]), 4),
            "p95": round(float(p[3]), 4),
        },
        "regions": suggest_regions(density, max_regions=max_regions),
        "density_bands": suggest_density_bands(density, classes=classes),
    }


def _main() -> int:
    if len(sys.argv) < 2:
        print("用法: python -m engine.adaptive.auto_tune <image>")
        return 2
    from engine.core.io_utils import imread_unicode

    img = imread_unicode(sys.argv[1])
    if img is None:
        print(f"[auto_tune] 读取失败: {sys.argv[1]}")
        return 1
    import time
    t0 = time.perf_counter()
    result = suggest_auto_tune(img)
    dt = time.perf_counter() - t0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[auto_tune] 耗时 {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
