"""
v2.2 数据契约（对应开发计划 §4.3 / §6.7）

全局约定
--------
1. **CMYK 量纲**：工作数组一律使用 **逻辑墨量**（0 = 0% 墨，255 = 100% 墨）。
   仅在 `psd_compiler` 写出前统一执行 `255 - x` 转为 PSD 磁盘反码，禁止在算子内反复反转。
2. **坐标命名**：所有 Y 坐标必须带坐标系前缀（src_ / unit_ / canvas_），禁止裸 `y`。
   五套坐标系见计划 §3.1：SRC → UNIT → CUT → CANVAS → PHYS。
3. **dataclass 一律 `eq=False`**：字段含 numpy 数组，默认 `__eq__` 会抛
   `ValueError: truth value ambiguous`。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# 图层命名规范（计划 §6.3，Tier 4 强制断言）
LAYER_NAME_RE = re.compile(r"^\d{2}_[A-Za-z]+(_[A-Za-z0-9]+)*_(CMYK|Mask|Spot|RECON)$")

# 管线生命周期（计划 §6.3）
PIPELINE_STAGES = (
    "load_preset", "prepare", "extract", "compose", "compile", "qa",
)


def new_run_id() -> str:
    """生成 run_id，用于 intermediate/<run_id>/ 归档（ADR-009）。"""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass(eq=False)
class LayerDescriptor:
    """单层描述。制版线用 cmyk + alpha；设计线用 rgba + bbox（计划 §6.7）。"""

    name: str                      # 须匹配 LAYER_NAME_RE
    layer_type: str                # substrate_cmyk | print_cmyk | diecut_mask | spot_channel | element_rgba
    cmyk_channels: Optional[np.ndarray] = None   # (4, H, W) uint8，逻辑墨量
    alpha: Optional[np.ndarray] = None           # (H, W) uint8，255=不透明
    visible: bool = True
    opacity: int = 255
    blend_mode: str = "normal"

    # ---- 设计线专用（§6.7）----
    rgba: Optional[np.ndarray] = None            # (H, W, 4) uint8
    bbox: Optional[tuple[int, int, int, int]] = None   # (left, top, right, bottom) 元素包围盒
    element_type: str = ""                       # 中文语义，如 "亭子"
    is_recon: bool = False                       # 是否为重建区层（§6.8）

    def __post_init__(self) -> None:
        # pytoshop 的 Pascal 串会补 \x00，写入前必须清理（V4-05）
        object.__setattr__(self, "name", str(self.name).rstrip("\x00"))

    # ---- 便利方法 ----
    @property
    def has_rgba(self) -> bool:
        return self.rgba is not None

    def alpha_ratio(self) -> float:
        """alpha > 0 的像素占比（V4-36：拦截空层，当前 04 层为 0%）。"""
        if self.alpha is None:
            return 0.0
        return float((self.alpha > 0).mean())

    def bbox_from_alpha(self) -> Optional[tuple[int, int, int, int]]:
        """由 alpha 推导元素包围盒（§6.7-2：禁止全画布层）。"""
        if self.alpha is None or not self.alpha.any():
            return None
        ys, xs = np.where(self.alpha > 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


@dataclass(eq=False)
class ElementMask:
    """§6.9 分割 Provider 的输出单元。"""
    label: str                    # 语义标签（来自 prompt 或规则）
    mask: np.ndarray              # (H, W) uint8，0/255
    score: float = 1.0
    bbox: Optional[tuple[int, int, int, int]] = None
    is_recon: bool = False


@dataclass(eq=False)
class QAReport:
    """质检单（计划 §8.3）。"""
    run_id: str
    tier: str = "T4"
    passed: bool = True
    records: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def record(self, code: str, name: str, ok: bool, detail: str = "") -> bool:
        self.records.append({"code": code, "name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.passed = False
        return bool(ok)

    def to_dict(self) -> dict[str, Any]:
        failed = [r for r in self.records if not r["ok"]]
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "passed": self.passed,
            "total": len(self.records),
            "failed": len(failed),
            "records": self.records,
            "metrics": self.metrics,
        }


@dataclass(eq=False)
class ProcessingContext:
    """
    全流程共享状态。

    双分支（铁律 R2）：
      - `detect_image`  经照度平场，**只用于检测**（聚类/微孔/轮廓/接缝），永不写入 PSD
      - `output_image`  未经平场，仅色卡标定，用于制版像素
    """

    run_id: str = field(default_factory=new_run_id)

    # ---- 图像分支 ----
    source_bgr: Optional[np.ndarray] = None      # SRC 坐标系原始读入（OpenCV BGR）
    detect_image: Optional[np.ndarray] = None    # 检测分支（经平场）
    output_image: Optional[np.ndarray] = None    # 输出分支（未经平场）

    # ---- 几何与分辨率（§3）----
    ppi: float = 150.0
    physical_mm: tuple[float, float] = (900.0, 1600.0)   # (宽, 高)
    canvas_px: tuple[int, int] = (5315, 9449)            # (宽, 高)
    unit_px: Optional[tuple[int, int]] = None            # UNIT 坐标系尺寸
    unit_cut_top_y: int = 0
    unit_cut_bottom_y: int = 718                         # 兜底默认；运行时由 contour_protection 覆盖
    cut_y_source: str = "preset_default"                 # ADR-008：应为 contour_protection
    transform_chain: dict[str, Any] = field(default_factory=dict)

    # ---- 输出 ----
    output_mode: str = "PLATE"                           # PLATE 制版线 | DESIGN 设计线（ADR-011）
    layers: list[LayerDescriptor] = field(default_factory=list)
    qa_metrics: dict[str, Any] = field(default_factory=dict)

    # ---- 兼容旧字段（旧代码仍在用，保留避免大面积改动）----
    rectified_image: Optional[np.ndarray] = None
    cleaned_mask: Optional[np.ndarray] = None
    color_normalized_image: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ================= 兼容属性（新名 → 旧名）=================
    @property
    def raw_image(self) -> Optional[np.ndarray]:
        return self.source_bgr

    @raw_image.setter
    def raw_image(self, value: Optional[np.ndarray]) -> None:
        self.source_bgr = value

    @property
    def target_dpi(self) -> float:
        return self.ppi

    @property
    def target_pixels(self) -> tuple[int, int]:
        return self.canvas_px

    @property
    def physical_width_mm(self) -> float:
        return self.physical_mm[0]

    @property
    def physical_height_mm(self) -> float:
        return self.physical_mm[1]

    # ================= 分辨率诚实指标（§3.2）=================
    def register_transform(self, stage: str, matrix: Any) -> None:
        """登记各阶段 3x3 单应矩阵，供任意一步反查（§3.1）。"""
        self.transform_chain[stage] = matrix

    def effective_source_ppi(self) -> Optional[float]:
        """有效源分辨率 = 裁切宽像素 ÷ 成品物理宽度（英寸）。"""
        if not self.unit_px:
            return None
        cut_w = self.unit_px[0]
        inches = self.physical_mm[0] / 25.4
        return cut_w / inches if inches else None

    def upscale_factors(self) -> Optional[tuple[float, float]]:
        """(fx, fy) 放大倍率。"""
        if not self.unit_px:
            return None
        cut_h = max(1, self.unit_cut_bottom_y - self.unit_cut_top_y)
        return (self.canvas_px[0] / self.unit_px[0], self.canvas_px[1] / cut_h)

    def resolution_grade(self) -> str:
        """NATIVE / MILD / HEAVY / EXTREME（§3.2 分级告警）。"""
        ppi = self.effective_source_ppi()
        if ppi is None:
            return "UNKNOWN"
        if ppi >= 150:
            return "NATIVE"
        if ppi >= 50:
            return "MILD"
        if ppi >= 15:
            return "HEAVY"
        return "EXTREME"

    def refresh_resolution_metrics(self) -> None:
        """把分辨率诚实指标写入 qa_metrics，质检单强制披露（G3）。"""
        fx_fy = self.upscale_factors()
        self.qa_metrics["effective_source_ppi"] = self.effective_source_ppi()
        self.qa_metrics["resolution_grade"] = self.resolution_grade()
        if fx_fy:
            self.qa_metrics["upscale_factor_x"] = round(fx_fy[0], 3)
            self.qa_metrics["upscale_factor_y"] = round(fx_fy[1], 3)
            self.qa_metrics["max_upscale_factor"] = round(max(fx_fy), 3)
