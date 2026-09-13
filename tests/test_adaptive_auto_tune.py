"""Stage 2 Auto-Tune 单元测试（确定性墨密度场路径，无 DINO/GPU 依赖）。"""
import time

import numpy as np
import pytest

from engine.adaptive.density_analyzer import (
    compute_density_map,
    density_hotspots,
    region_density_band,
)
from engine.adaptive.auto_tune import (
    suggest_auto_tune,
    suggest_density_bands,
    suggest_regions,
)


def _gold_with_ink(h=600, w=900):
    """金地底 + 中央浓墨带 + 右下小墨点，用于验证热点提取。"""
    img = np.full((h, w, 3), 0, dtype=np.uint8)
    img[:, :, 0] = 35    # B
    img[:, :, 1] = 165   # G
    img[:, :, 2] = 205   # R  → 金地
    img[250:350, 150:750] = [20, 20, 20]     # 中央横向浓墨带
    img[520:560, 780:830] = [20, 20, 20]     # 右下小墨点
    return img


def test_density_map_range():
    """密度场值域合理（0..~1）。"""
    D = compute_density_map(_gold_with_ink())
    assert D.shape == (600, 900)
    assert np.isfinite(D).all()
    assert D.min() >= 0.0


def test_hotspot_regions_valid():
    """热点区域归一化坐标在界内且 x0<x1, y0<y1。"""
    D = compute_density_map(_gold_with_ink())
    regs = density_hotspots(D, block=128, thresh_percentile=80)
    assert len(regs) >= 1, "应至少检出一个墨迹热点"
    for r in regs:
        x0, y0, x1, y1 = r["region"]
        assert 0.0 <= x0 < x1 <= 1.0, f"区域 x 越界: {r['region']}"
        assert 0.0 <= y0 < y1 <= 1.0, f"区域 y 越界: {r['region']}"
        assert r["score"] > 0
    # 中央浓墨带应比右下小墨点得分高（降序）
    assert regs[0]["score"] >= regs[-1]["score"]


def test_density_band_range():
    """密度带区间合理：0<=min<max<=1，覆盖率 ∈ [0,1]。"""
    D = compute_density_map(_gold_with_ink())
    band = region_density_band(D, (0.1, 0.3, 0.9, 0.7))
    assert band is not None
    assert 0.0 <= band["density_min"] < band["density_max"] <= 1.0
    assert 0.0 <= band["coverage"] <= 1.0


def test_determinism():
    """同图两次结果必须逐字段一致（RK-16 确定性）。"""
    img = _gold_with_ink()
    a = suggest_auto_tune(img)
    b = suggest_auto_tune(img)
    assert a == b, "Auto-Tune 结果不确定（违反 RK-16）"


def test_suggest_density_bands_with_classes():
    """classes 带 region 时按类名标注密度带；无墨/无区域的类被跳过。"""
    D = compute_density_map(_gold_with_ink())
    classes = [
        {"name": "05A_前景寒林枯木", "region": [0.1, 0.35, 0.85, 0.65]},   # 含中央浓墨带
        {"name": "09A_印章", "region": [0.85, 0.85, 0.95, 0.95]},           # 含右下墨点
        {"name": "04D_远山_无墨", "region": [0.2, 0.0, 0.9, 0.15]},          # 纯金地→应跳过
        {"name": "无区域的类"},                                              # 应跳过
    ]
    bands = suggest_density_bands(D, classes=classes)
    labels = [b["label"] for b in bands]
    assert "05A_前景寒林枯木" in labels
    assert "09A_印章" in labels
    assert "无区域的类" not in labels, "无 region 的类不应产生全画幅假带"
    assert "04D_远山_无墨" not in labels, "无墨区域应被跳过（不推荐 [0,0] 假带）"
    for b in bands:
        assert 0.0 <= b["density_min"] < b["density_max"] <= 1.0


def test_suggest_density_bands_fallback_grid():
    """无 classes/regions 时兜底九宫格；纯底色格被跳过，含墨格保留。"""
    D = compute_density_map(_gold_with_ink())
    bands = suggest_density_bands(D)
    labels = [b["label"] for b in bands]
    # 全部格最多 9；纯金地格应被跳过（<9）
    assert 1 <= len(bands) <= 9, f"九宫格带数异常: {len(bands)}"
    assert len(bands) < 9, "纯底色格应被跳过"
    # 中央格（含横向浓墨带）必在
    assert "grid_11" in labels, "含墨的中央格应保留"


def test_suggest_auto_tune_structure():
    """主入口返回结构完整。"""
    out = suggest_auto_tune(_gold_with_ink())
    assert set(out.keys()) == {"global_percentiles", "regions", "density_bands"}
    for k in ("p50", "p75", "p90", "p95"):
        assert k in out["global_percentiles"]
        assert 0.0 <= out["global_percentiles"][k] <= 1.0


def test_performance():
    """单图 Auto-Tune 应在 3s 内（计划验收标准）。"""
    img = _gold_with_ink(h=1440, w=2880)
    t0 = time.perf_counter()
    suggest_auto_tune(img)
    dt = time.perf_counter() - t0
    assert dt < 3.0, f"Auto-Tune 过慢: {dt:.2f}s"


def test_blank_image_no_false_hotspot():
    """纯底色图不应误报热点。"""
    img = np.full((600, 900, 3), 0, dtype=np.uint8)
    img[:, :, 1] = 165
    img[:, :, 2] = 205
    D = compute_density_map(img)
    regs = density_hotspots(D, block=128, thresh_percentile=85)
    assert len(regs) == 0, "纯底色不应产生热点区域"
