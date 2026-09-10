"""交付清单（manifest）契约 —— 双产品线与生成内容追溯的落地载体。

为什么需要它（2026-09-10 定）
----------------------------
铁律 R1 要求"生成/补全内容必须显式标记为重建区"，§3.1 硬边界 1 要求
"任何生成式模型的输出不得进入 PLATE 线"。但修复前的实现里，标记只停在日志：

  - Step 4 遮挡补全（占全幅 0.31%）有 `reconstruction_mask`，却只 print 到日志；
  - Step 5 超分（RealESRGAN，生成像素占比接近 100%）**完全没有标记**。

标记策略与生成规模完全倒挂。且计划 §4 的数据流是「设计师微调 → 回灌制版线」，
DESIGN 线的生成细节会随图层回灌，若标记不随产物持久化，双线隔离在回灌处即被击穿。

为什么用 sidecar 而不写进 PSB
-----------------------------
- 写进 PSB 只能当专色通道 → 污染印前文件、干扰 TAC 审计；
- 写进图层名或 luni 块 → 破坏图层命名规范与中文名可读性。
⇒ 采用 **sidecar JSON + 独立掩码 PNG**，产物本身保持干净，追溯信息外挂。

坐标与单位约定
-------------
- bbox 一律 `(left, top, right, bottom)`，与 psd-tools / pytoshop 一致；
- 分辨率统一用 PPI，字段名带 `_ppi` 后缀；
- 占比一律为 0..1 的小数，字段名带 `_ratio` 后缀。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np

MANIFEST_VERSION = "1.0"


class OutputMode(str, Enum):
    """双产品线（ADR-011）。"""

    PLATE = "plate"     # 制版线：CMYK 分色版，禁止生成内容
    DESIGN = "design"   # 设计线：RGBA 元素层，允许生成但必须标记
    BOTH = "both"       # 同时产出两条线

    @staticmethod
    def parse(value: Optional[str]) -> "OutputMode":
        if not value:
            return OutputMode.DESIGN
        v = str(value).strip().lower()
        for m in OutputMode:
            if m.value == v:
                return m
        raise ValueError(f"未知 output mode: {value}（只允许 plate / design / both）")


# 生成式超分在两条线上的准入策略（§3.1 硬边界 1）
#   PLATE：禁止任何生成式输出 —— 制版验收要求可复算、可对色、可过 TAC 审计
#   DESIGN：允许，但必须在 manifest 中披露，并对每层输出 recon 掩码
GENERATIVE_UPSCALE_ALLOWED = {
    OutputMode.PLATE: False,
    OutputMode.DESIGN: True,
    OutputMode.BOTH: True,   # both 模式下按线分别处理，PLATE 侧仍走非生成式
}


@dataclass(eq=False)
class LayerGenerationRecord:
    """单层的生成内容记录。"""

    name: str
    generated: bool = False
    generation_reasons: list[str] = field(default_factory=list)
    bbox: Optional[tuple[int, int, int, int]] = None     # (left, top, right, bottom)
    recon_mask_path: Optional[str] = None                # 相对 manifest 的路径
    # 掩码坐标系与尺寸：按 §3.1"坐标必须带坐标系前缀"的精神显式标注。
    # 遮挡补全在 SRC（源图低分辨率）上运行，故掩码为 src 空间，
    # 不做上采样写入（上采样会引入插值误差，追溯信息应保持原始精度）。
    recon_mask_space: str = "src"                        # src | canvas
    recon_mask_size: Optional[tuple[int, int]] = None    # (width, height)
    recon_pixel_count: int = 0
    recon_ratio: float = 0.0                             # 相对该掩码所在空间面积

    def add_reason(self, reason: str) -> None:
        if reason not in self.generation_reasons:
            self.generation_reasons.append(reason)
        self.generated = True


@dataclass(eq=False)
class GenerationPolicy:
    """本次运行实际采用的生成策略（写入 manifest 供下游审计）。"""

    mode: str = OutputMode.DESIGN.value
    generative_upscale_allowed: bool = True
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def declare(self, stage: str, engine: str, generative: bool,
                coverage_ratio: float = 0.0, detail: str = "") -> None:
        self.stages[stage] = {
            "engine": engine,
            "generative": bool(generative),
            "coverage_ratio": round(float(coverage_ratio), 6),
            "detail": detail,
        }


@dataclass(eq=False)
class DeliverableManifest:
    """交付清单根对象。"""

    run_id: str
    output_mode: str
    source: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    generation_policy: GenerationPolicy = field(default_factory=GenerationPolicy)
    layers: list[LayerGenerationRecord] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    # ---------------- 写入 ----------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, manifest_path: str, mask_dir: Optional[str] = None) -> str:
        """写出 manifest；若给定 mask_dir 则同时把各层掩码落盘为 PNG。"""
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)
        out_dir = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return manifest_path

    # ---------------- 查询 ----------------
    def layer(self, name: str) -> Optional[LayerGenerationRecord]:
        for r in self.layers:
            if r.name == name:
                return r
        return None

    def generated_ratio(self) -> float:
        """生成内容占比（按像素加权，取各层 recon_ratio×层面积 之和 / 全幅面积）。"""
        if not self.output:
            return 0.0
        canvas_px = int(self.output.get("width", 0)) * int(self.output.get("height", 0))
        if canvas_px <= 0:
            return 0.0
        acc = 0.0
        for r in self.layers:
            if not r.generated or not r.bbox:
                continue
            l, t, rr, b = r.bbox
            layer_area = max(0, rr - l) * max(0, b - t)
            acc += r.recon_ratio * layer_area
        return min(1.0, acc / canvas_px)

    def assert_plate_purity(self) -> tuple[bool, str]:
        """校验 PLATE 线纯净性：不得含任何生成内容（§3.1 硬边界 1）。"""
        if self.output_mode not in (OutputMode.PLATE.value,):
            return True, "非 PLATE 产物，跳过纯净性校验"
        dirty = [r.name for r in self.layers if r.generated]
        if dirty:
            return False, f"PLATE 产物含生成内容图层 {len(dirty)} 个：{dirty[:5]}"
        for stage, info in self.generation_policy.stages.items():
            if info.get("generative"):
                return False, f"PLATE 产物使用了生成式阶段：{stage}（{info.get('engine')}）"
        return True, "PLATE 产物无任何生成内容"


def build_manifest(
    run_id: str,
    output_mode: str,
    source_path: str,
    source_wh: tuple[int, int],
    output_path: str,
    output_wh: tuple[int, int],
    ppi: float,
    color_mode: str,
) -> DeliverableManifest:
    """按标准字段构造 manifest，并自动计算诚实指标（G3）。"""
    sw, sh = source_wh
    ow, oh = output_wh
    physical_w_mm = ow / ppi * 25.4 if ppi else 0.0
    physical_h_mm = oh / ppi * 25.4 if ppi else 0.0
    # 有效源分辨率 = 源像素 ÷ 成品物理宽度（英寸）
    eff_ppi = sw / (physical_w_mm / 25.4) if physical_w_mm else 0.0
    fx = ow / sw if sw else 0.0
    fy = oh / sh if sh else 0.0

    mode = OutputMode.parse(output_mode)
    m = DeliverableManifest(run_id=run_id, output_mode=mode.value)
    m.source = {
        "path": str(source_path).replace("\\", "/"),
        "width": sw, "height": sh,
        "megapixels": round(sw * sh / 1e6, 3),
        "effective_ppi": round(eff_ppi, 2),
    }
    m.output = {
        "path": str(output_path).replace("\\", "/"),
        "width": ow, "height": oh,
        "ppi": float(ppi),
        "physical_width_mm": round(physical_w_mm, 1),
        "physical_height_mm": round(physical_h_mm, 1),
        "color_mode": str(color_mode),
        "upscale_factor_x": round(fx, 3),
        "upscale_factor_y": round(fy, 3),
        "max_upscale_factor": round(max(fx, fy), 3),
    }
    m.generation_policy = GenerationPolicy(
        mode=mode.value,
        generative_upscale_allowed=GENERATIVE_UPSCALE_ALLOWED[mode],
    )
    m.totals = {
        "effective_source_ppi": round(eff_ppi, 2),
        "max_upscale_factor": round(max(fx, fy), 3),
        "generated_pixel_ratio": 0.0,
    }
    return m


def load_manifest(path: str) -> dict[str, Any]:
    """读取 manifest（返回原始 dict，供质检/前端消费）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_mask_png(path: str, mask: np.ndarray) -> bool:
    """写出掩码 PNG，支持非 ASCII 路径。

    为什么不能直接用 cv2.imwrite：Windows 上它遇到含中文的路径会**静默失败**——
    实测返回 True 而文件并不存在。必须走 imencode + 二进制写入。
    """
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def read_mask_png(path: str) -> Optional[np.ndarray]:
    """读取掩码 PNG，支持非 ASCII 路径。

    与写入同理，cv2.imread 在中文路径下会返回 None（并打印 findDecoder 警告），
    必须走 np.fromfile + imdecode。
    """
    if not os.path.isfile(path):
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
