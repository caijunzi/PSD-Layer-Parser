"""
Grounding DINO + SAM 2 级联视觉语义分割提供者 (GroundedSAMProvider)。
遵循 SSOT §6.9 / ADR-012 / ADR-013 契约：
1. Grounding DINO 充当目标雷达：根据自然语言提示词（Prompt）在画面中扫描定位物体边界框 BBox；
2. SAM 2 充当超级魔棒：在 BBox 提示下执行像素级高精度边缘抠图提取掩模 Mask；
3. 三级容灾降级：未下载模型或断网时，自动无缝降级为高保真 Frangi 骨架流 + 自适应 Otsu 阈值引擎，保证 100% 稳定出图。
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

from engine.semantic_segmenter import UniversalSemanticSegmenter

# ---------------------------------------------------------------------------
# 神经掩模质量门（Quality Gate）
#
# 事故背景：Grounding DINO 在金地（暖金色）背景上对 "red stamp / cinnabar seal"
# 产生假阳性大框，SAM 2 在该框内抠出覆盖 8.77% 画幅的大掩模，而旧的合并逻辑
# `final_masks[k] = m` 无条件覆盖，直接抹掉了规则引擎算出的正确印章掩模
# （0.0119%，BBox 仅 68x63）。元素层因此退化为一整片金地。
#
# 对策：神经掩模必须通过面积预算 + 相对放大倍数 + 形状一致性三重校验，
# 才允许覆盖规则掩模；否则保留规则结果并显式告警。
# ---------------------------------------------------------------------------

# 语义类别 → 面积预算（占全画布比例上限）。由图层名关键词匹配。
AREA_BUDGET_BY_KEYWORD: tuple[tuple[tuple[str, ...], float], ...] = (
    (("seal", "印章", "cinnabar", "朱红"), 0.005),
    (("calligraphy", "题跋", "inscription", "款识"), 0.02),
    (("geese", "芦雁", "fauna", "禽"), 0.02),
    (("figures", "人物", "高士", "attendant"), 0.03),
    (("pavilion", "草堂", "architecture", "建筑"), 0.05),
    (("trees", "枯木", "林", "vegetation"), 0.20),
    (("cliffs", "峭壁", "墨岩", "mountain", "山"), 0.25),
    (("seam", "折痕", "fold"), 0.05),
    (("frame", "外框", "brocade"), 0.35),
)

# 未匹配到关键词时的保守预算
DEFAULT_AREA_BUDGET = 0.15

# 相对规则掩模的最大放大倍数（超出即判定为假阳性漂移）
MAX_EXPANSION_FACTOR = 3.0

# 神经掩模与规则掩模的最低形状一致性（IoU），低于且显著放大则拒绝
MIN_SHAPE_IOU = 0.02

# Grounding DINO 检测阈值（0.25 实测在金地上假阳性过高）
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25

# 单框面积上限（占全画布），超过视为"把背景框进来了"
MAX_BOX_AREA_RATIO = 0.08

# 实例层（name_NN）面积预算放宽倍数：单个体不应按整类面积衡量
INSTANCE_BUDGET_FACTOR = 3.0

# 弥散掩模门（2026-09-11 山水图验收引入）：内容层掩模的外接框覆盖比超过此值
# 即判定"弥散"（SAM 对远山/峭壁等无边界山水元素天然产出全画布碎片掩模，
# 合成时雾化污染右半屏）。豁免表覆盖天然全画布的装饰/底板层。
MAX_BBOX_AREA_RATIO = 0.5

BBOX_EXEMPT_KEYWORDS: tuple[str, ...] = (
    "base", "底板",           # 金箔/织物底板天然全画布
    "frame", "外框", "brocade", "绫边",  # 外框环天然全画布
    "fold", "seam", "折痕",   # 折缝贯穿全画布（V2 实测 47% 靠近阈值，一并豁免）
)


def is_bbox_exempt(layer_name: str) -> bool:
    low = str(layer_name).lower()
    return any(k in low for k in BBOX_EXEMPT_KEYWORDS)


def area_budget_for(layer_name: str) -> float:
    """按图层名关键词返回该类元素的面积预算（占全画布比例）。"""
    low = str(layer_name).lower()
    for keywords, budget in AREA_BUDGET_BY_KEYWORD:
        if any(k in low for k in keywords):
            return budget
    return DEFAULT_AREA_BUDGET


class GroundedSAMProvider:
    """Grounding DINO + SAM 2 级联开放词汇智能分割提供者。"""

    def __init__(
        self,
        dino_model_path: Optional[str] = None,
        sam_model_path: Optional[str] = None,
        preferred_device: str = "GPU.1",
        preset_name: str = "japanese_screen_gold",
    ):
        self.preferred_device = preferred_device
        self.preset_name = preset_name
        # 品类语义由 preset 驱动（SSOT / G2）：检测原始名 → 中英对照图层名 的映射
        # 归属 preset.layer_semantics.name_mapping，provider 不再内置品类专属映射
        try:
            from engine.schemas.presets import load_preset
            self.preset = load_preset(preset_name)
        except Exception as e:
            print(f"[GroundedSAMProvider] preset 加载失败（将回退内置映射）: {e}")
            self.preset = None

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        ckpt_dir = os.path.join(root_dir, "checkpoints")
        dino_dir = os.path.join(root_dir, "third_party", "GroundingDINO")
        sam_dir = os.path.join(root_dir, "third_party", "sam2")

        self.dino_ckpt = os.path.abspath(dino_model_path or os.path.join(ckpt_dir, "groundingdino_swint_ogc.pth"))
        self.sam_ckpt = os.path.abspath(sam_model_path or os.path.join(ckpt_dir, "sam2_hiera_tiny.pt"))
        self.dino_cfg = os.path.join(dino_dir, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        self.sam_cfg = "configs/sam2/sam2_hiera_t.yaml"

        self.dino_model = None
        self.sam_predictor = None
        self.backend = "fallback_rule_based"

        # 加入 sys.path 以支持本地模块加载
        import sys
        if dino_dir not in sys.path:
            sys.path.insert(0, dino_dir)
        if sam_dir not in sys.path:
            sys.path.insert(0, sam_dir)

        self._init_models()

    def _init_models(self):
        dino_ready = os.path.isfile(self.dino_ckpt) and os.path.isfile(self.dino_cfg)
        sam_ready = os.path.isfile(self.sam_ckpt)

        if not (dino_ready and sam_ready):
            self.backend = "fallback_rule_based"
            return

        # 设备决策（RK-16 补充）：torch CUDA 驱动存在 ≠ 可用——若本机 GPU 的
        # sm 架构不被当前 torch 轮子支持（如 sm_120 vs torch≤2.6），张量运算会抛
        # "no kernel image"。故做一次真实矩阵乘探活，失败回落 CPU。
        #
        # ⚠️ 实测结论（2026-09-11，torch 2.7.1+cu128）：探活/加载/推理全链路已通，
        # 但 SAM2-tiny 在 1024px 输入上 GPU 收益不敌传输与串行开销——
        # GPU 神经分割 61.98s vs CPU 47.65s（慢 30%），且 GPU 并行浮点
        # 破坏 RK-16 逐像素复现（同 seed 掩模偏差 ~0.02%）。
        # ⇒ 默认分割回落 CPU；需要实验 GPU 时设环境变量 ULS_SEGMENT_DEVICE=cuda。
        self.torch_device = "cpu"
        force_gpu = os.environ.get("ULS_SEGMENT_DEVICE", "").lower() in ("cuda", "gpu")
        try:
            import torch

            if torch.cuda.is_available():
                _ = torch.zeros(8, device="cuda") @ torch.zeros(8, device="cuda")
                torch.cuda.synchronize()
                if force_gpu:
                    self.torch_device = "cuda"
                print(f"[GroundedSAMProvider] CUDA 探活通过: {torch.cuda.get_device_name(0)}"
                      + ("（分割使用 GPU）" if self.torch_device == "cuda" else
                         "（分割默认 CPU：GPU 实测无净收益且破坏逐像素复现，"
                         "设 ULS_SEGMENT_DEVICE=cuda 可启用）"))
        except Exception as e:
            self.torch_device = "cpu"
            print(f"[GroundedSAMProvider] CUDA 探活失败（分割回落 CPU）: {type(e).__name__}: {str(e)[:120]}")

        try:
            from groundingdino.util.inference import load_model as load_dino
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            print("[GroundedSAMProvider] Loading Grounding DINO Swin-T model...")
            # DINO 固定 CPU：其 deform-attn 的 CUDA 分支依赖未编译的 `_C` 扩展
            # （third_party 的 ms_deform_attn.py:330），GPU 上会 NameError；
            # 且 DINO 输入仅 800px，CPU 开销可接受。SAM2（大图 encoder，最重）走 GPU。
            self.dino_device = "cpu"
            self.sam_device = self.torch_device
            self.dino_model = load_dino(self.dino_cfg, self.dino_ckpt, device=self.dino_device)

            print("[GroundedSAMProvider] Loading SAM 2 Hiera-Tiny model...")
            sam_model = build_sam2(self.sam_cfg, self.sam_ckpt, device=self.sam_device)
            self.sam_predictor = SAM2ImagePredictor(sam_model)

            self.backend = "grounded_sam2_neural"
            print("[GroundedSAMProvider] Successfully loaded Grounding DINO + SAM 2 neural pipeline!")
            return
        except Exception as e:
            print(f"[GroundedSAMProvider] Neural models init note: {e}, falling back to rule engine")

        self.backend = "fallback_rule_based"
        self.last_coverage: dict = {}

    def _infer_region_sam(self, image_bgr: np.ndarray, region: tuple, split: int = 3) -> np.ndarray:
        """区域先验 SAM：在 preset 指定的归一化 region 内多框采样，选局部候选。

        用途（2026-09-12 P1 实证）：DINO 对无边界/弱语义类（寒林枯木等）
        给出的粗框会导致 SAM 输出全画布弥散掩模（bbox 75%）。改为**由 preset
        显式给定区域先验**（人/配置知道"树在画面中央"），在区域内采样 2-3 个子框
        分别预测并取**面积最小的合理候选**（局部对象优先），合并后 bbox 从
        75.1% → 21.6%，过弥散门。
        """
        h, w = image_bgr.shape[:2]
        x0, y0, x1, y1 = (int(region[0] * w), int(region[1] * h),
                          int(region[2] * w), int(region[3] * h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"非法 region: {region}")
        if split >= 3:
            xm = (x0 + x1) // 2
            boxes = [[x0, y0, xm, y1], [xm, y0, x1, y1], [x0, y0, x1, y1]]
        else:
            boxes = [[x0, y0, x1, y1]]
        merged = np.zeros((h, w), dtype=np.uint8)
        for b in boxes:
            masks, scores, _ = self.sam_predictor.predict(
                box=np.array(b), multimask_output=True)
            areas = [int(m.sum()) for m in masks]
            idx = int(np.argmin(areas))  # 局部对象优先
            if areas[idx] < (x1 - x0) * (y1 - y0) * 0.01:  # 过小则退回最高分
                idx = int(np.argmax(scores))
            merged = np.maximum(merged, masks[idx].astype(np.uint8) * 255)
        return merged

    def _load_preset_config(self) -> dict:
        """按 preset_name 加载 preset 配置（缓存）。"""
        if getattr(self, "_preset_cfg", None) is not None:
            return self._preset_cfg
        try:
            from engine.schemas.presets import load_preset
            self._preset_cfg = load_preset(self.preset_name) or {}
        except Exception:
            self._preset_cfg = {}
        return self._preset_cfg

    def _density_refine_config(self) -> dict:
        """读 preset.density_refine 配置（未配置返回 disabled 空配置）。"""
        cfg = self._load_preset_config().get("density_refine")
        if not isinstance(cfg, dict):
            return {"enabled": False, "classes": {}}
        out = {"enabled": bool(cfg.get("enabled")), "classes": cfg.get("classes") or {}}
        for k in ("default_floor", "sigma", "gold_percentile", "close_kernel", "open_kernel"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    def segment_objects(
        self,
        image_bgr: np.ndarray,
        classes: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        根据语义提示词或预设对输入图像执行智能解耦分层。
        
        Args:
            image_bgr: BGR 格式输入图像
            classes: 语义分类提示词列表
                
        Returns:
            Dict[str, np.ndarray]: 图层名称 -> uint8 二值掩模 (0/255)
        """
        # 1. 神经推理路径（Grounding DINO + SAM 2 真实模型就绪）
        neural_masks = {}
        if self.dino_model is not None and self.sam_predictor is not None:
            try:
                print(f"[GroundedSAMProvider] 调度 Grounding DINO + SAM 2 神经级联推理...")
                neural_masks = self._infer_grounded_sam(image_bgr, classes)
            except Exception as e:
                print(f"[GroundedSAMProvider] Neural inference error: {e}")

        # 2. 动态特征与骨架流分层 (多尺度 Frangi 骨架流 + 几何边缘算子，100% 动态实时计算，零磁盘旧文件直读)
        rule_segmenter = UniversalSemanticSegmenter(preset=self.preset_name)
        base_masks = rule_segmenter.segment_objects(image_bgr)
        final_masks = self._map_to_bilingual_names(base_masks)

        # 2.5 区域先验 SAM 覆盖（2026-09-12 P1 实证）：preset 给 class 配 region 时，
        #     绕过 DINO 粗框（弱语义类的粗框会让 SAM 全画布弥散），
        #     在 region 内多框采样取局部候选。产出仍走质量门/弥散门/披露。
        region_sam_keys = []
        if neural_masks:
            for cls_info in (classes or []):
                reg = cls_info.get("region")
                if not reg:
                    continue
                cname = cls_info.get("name", "")
                key = next((k for k in neural_masks if k == cname or cname in k), cname)
                try:
                    rm = self._infer_region_sam(
                        image_bgr, tuple(reg), int(cls_info.get("region_boxes", 3)))
                    if np.count_nonzero(rm > 127) == 0:
                        continue
                    neural_masks[key] = rm
                    region_sam_keys.append(key)
                    print(f"[GroundedSAMProvider] 区域先验 SAM: {key} "
                          f"({np.count_nonzero(rm > 127):,}px, region={reg})")
                except Exception as e:
                    print(f"[GroundedSAMProvider] 区域先验 SAM 异常 {cname}: {e}")

        # 3. 神经掩模经质量门校验后置入解耦图层（未通过者保留规则掩模）
        #    神经掩模用中文图层名，规则掩模用英文键，需经同一映射才能配对比较
        rule_final = final_masks
        accepted, rejected = [], []
        if neural_masks:
            for k, m in neural_masks.items():
                ok, reason = self._quality_gate(k, m, rule_final.get(k))
                if ok:
                    final_masks[k] = m
                    accepted.append(k)
                else:
                    rejected.append(f"{k}（{reason}）")

        # 3.5 实例拆分一致性：启用 instance_split 的类不保留合并层，
        #     否则规则 fallback 的合并掩模会与单实例层重复占用同一内容。
        split_bases = {c.get("name") for c in (classes or []) if c.get("instance_split")}
        for k in list(final_masks.keys()):
            if k in split_bases:
                del final_masks[k]

        if accepted:
            self.backend = "grounded_sam2_neural_dynamic"
            print(f"[GroundedSAMProvider] 神经解耦图层已置入: {accepted}")
        else:
            self.backend = "fallback_rule_based"

        if rejected:
            print(f"[GroundedSAMProvider] ⚠️  {len(rejected)} 个神经掩模未通过质量门，已保留规则引擎结果：")
            for r in rejected:
                print(f"      - {r}")

        # 4. 弥散门（统一作用于神经与规则两条产路的最终结果）：
        #    SAM 对远山/峭壁等无边界山水元素会产出全画布碎片掩模，
        #    合成时雾化污染（2026-09-11 山水图验收实测）。弥散层不产出，
        #    但留存到 rejected_masks 供调试与密度精修（不进最终产物）。
        diffuse_rejected = []
        self.rejected_masks: dict = {}
        for k in list(final_masks.keys()):
            reason = self._diffuse_check(k, final_masks[k])
            if reason:
                self.rejected_masks[k] = final_masks.pop(k)
                diffuse_rejected.append(f"{k}（{reason}）")
        if diffuse_rejected:
            print(f"[GroundedSAMProvider] ⚠️  弥散门拒绝 {len(diffuse_rejected)} 个层（不产出）：")
            for r in diffuse_rejected:
                print(f"      - {r}")

        # 5. 密度精修通道（v1.0，2026-09-11 实证：04A 峭壁 bbox 74.4%→16.4%）：
        #    preset.density_refine.classes 中列出的弥散被拒类，用墨密度场精修
        #    （refined = mask ∩ D>floor + 形态学），精修后再过弥散门，通过则产出。
        #    未列入精修配置的弥散层维持拒绝（04D/05A 已知不可行，见设计文档 §9.3）。
        refine_cfg = self._density_refine_config()
        refine_classes = (refine_cfg or {}).get("classes") or {}
        recovered = []
        D = None
        if refine_cfg.get("enabled") and refine_classes and self.rejected_masks:
            from engine.core.density_field import ink_density, refine_mask
            D = ink_density(
                image_bgr,
                gold_percentile=float(refine_cfg.get("gold_percentile", 88)),
                sigma=float(refine_cfg.get("sigma", 4.0)),
            )
            for k, raw in list(self.rejected_masks.items()):
                cls_cfg = refine_classes.get(k)
                if not cls_cfg:
                    continue
                refined = refine_mask(
                    raw, D,
                    floor=float(cls_cfg.get("floor", refine_cfg.get("default_floor", 0.12))),
                    close_kernel=int(cls_cfg.get("close_kernel", refine_cfg.get("close_kernel", 9))),
                    open_kernel=int(cls_cfg.get("open_kernel", refine_cfg.get("open_kernel", 3))),
                )
                reason = self._diffuse_check(k, refined)
                if reason:
                    rejected.append(f"{k}（密度精修后仍弥散：{reason}）")
                    continue
                if np.count_nonzero(refined > 127) == 0:
                    rejected.append(f"{k}（密度精修后为空掩模）")
                    continue
                final_masks[k] = refined
                recovered.append(k)
                print(f"[GroundedSAMProvider] 密度精修恢复层: {k}")
        else:
            recovered = []

        # 5.5 密度带独立通道（DINO 完全未命中的纹理/晕染类）：
        #     preset.density_band_classes 配置 (region + 密度区间) 直接产层，
        #     产物过弥散门；source 标记 density_band。
        band_cfg = self._load_preset_config().get("density_band_classes") or []
        band_produced = []
        for item in band_cfg:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            # 语义路径已产出该类 → 不覆盖（SAM/精修优先于密度带）
            if any(item["name"] == k or item["name"] in k for k in final_masks):
                continue
            try:
                from engine.core.density_field import band_mask as _band_mask, ink_density as _ink_density
                if D is None:
                    D = _ink_density(
                        image_bgr,
                        gold_percentile=float(item.get("gold_percentile", 88)),
                        sigma=float(item.get("sigma", 4.0)),
                    )
                bm = _band_mask(
                    D,
                    float(item.get("density_min", 0.05)),
                    float(item.get("density_max", 0.20)),
                    tuple(item["region"]) if item.get("region") else None,
                    close_kernel=int(item.get("close_kernel", 9)),
                    open_kernel=int(item.get("open_kernel", 3)),
                    min_blob_area=int(item.get("min_blob_area_px", 0)),
                )
                if np.count_nonzero(bm > 127) == 0:
                    continue
                reason = self._diffuse_check(item["name"], bm)
                if reason:
                    rejected.append(f"{item['name']}（密度带产出仍弥散：{reason}）")
                    continue
                final_masks[item["name"]] = bm
                band_produced.append(item["name"])
                print(f"[GroundedSAMProvider] 密度带产出层: {item['name']}")
            except Exception as e:
                print(f"[GroundedSAMProvider] 密度带通道异常 {item.get('name')}: {e}")

        # 6. 语义覆盖披露（配置了什么 / 实际产出什么 / 缺什么 / 为什么）——
        #    写入 manifest.totals.semantic_coverage，消灭"配置了却静默缺层"
        configured = [str(c.get("name") or c.get("layer_name") or "") for c in (classes or [])]
        produced = []
        produced_details = []
        missing, rejected_all = [], []
        refine_set = {r for r in recovered}
        band_set = set(band_produced)
        region_sam_set = set(region_sam_keys)
        for cname in configured:
            matched = [k for k in final_masks if k == cname or cname in k or k in cname]
            if matched:
                mkey = matched[0]
                produced.append(mkey)
                if mkey in refine_set or cname in refine_set:
                    src = "density_refined"
                elif mkey in band_set or cname in band_set:
                    src = "density_band"
                elif mkey in region_sam_set or cname in region_sam_set:
                    src = "region_sam"
                else:
                    src = "sam"
                produced_details.append({"name": mkey, "source": src})
            else:
                why = next((r for r in diffuse_rejected if cname in r or r.split("（")[0] in cname), None)
                if why is None:
                    why = next((r for r in rejected if cname in r), None)
                if why:
                    rejected_all.append({"name": cname, "reason": why})
                else:
                    missing.append({"name": cname, "reason": "DINO 未命中任何区域"})
        extra = [k for k in final_masks if k not in produced]
        self.last_coverage = {
            "configured": configured,
            "produced": produced,
            "produced_details": produced_details,
            "extra_layers": extra,
            "rejected": rejected_all,
            "missing": missing,
        }

        return final_masks

    @staticmethod
    def _quality_gate(layer_name: str, neural: np.ndarray, rule: Optional[np.ndarray]) -> tuple[bool, str]:
        """判定神经掩模是否可信到足以覆盖规则掩模。返回 (是否接受, 拒绝原因)。"""
        total = neural.size
        n = int(np.count_nonzero(neural > 127))
        if n == 0:
            return False, "空掩模"

        ratio = n / total
        # 实例层（name_NN）按"单个体"衡量，沿用整类预算会误杀单只雁等大个体：
        # 放宽 INSTANCE_BUDGET_FACTOR 倍，弥散门（bbox 覆盖比）仍照常生效。
        import re as _re
        base = _re.sub(r"_\d{2}$", "", str(layer_name))
        is_instance = base != str(layer_name)
        budget = area_budget_for(base)
        if is_instance:
            budget *= INSTANCE_BUDGET_FACTOR
        if ratio > budget:
            kind = "实例" if is_instance else "类"
            return False, f"面积 {ratio:.3%} 超出该{kind}预算 {budget:.3%}"

        if rule is not None:
            n_rule = int(np.count_nonzero(rule > 127))
            if n_rule > 0:
                expansion = n / n_rule
                if expansion > MAX_EXPANSION_FACTOR:
                    inter = int(np.count_nonzero((neural > 127) & (rule > 127)))
                    union = int(np.count_nonzero((neural > 127) | (rule > 127)))
                    iou = inter / union if union else 0.0
                    if iou < MIN_SHAPE_IOU:
                        return (
                            False,
                            f"相对规则掩模放大 {expansion:.1f}x 且 IoU={iou:.3f} < {MIN_SHAPE_IOU}（疑似背景假阳性）",
                        )
        return True, ""

    @staticmethod
    def _diffuse_check(layer_name: str, mask: np.ndarray) -> Optional[str]:
        """弥散门：内容层外接框覆盖比超限即判弥散（跨图通用规则，非 per-image 补丁）。

        返回拒绝原因（None=通过）。底板/外框/折痕等天然全画布层豁免。
        """
        if is_bbox_exempt(layer_name):
            return None
        ys, xs = np.nonzero(mask > 127)
        if len(ys) == 0:
            return None  # 空掩模由其它门处理
        h, w = mask.shape
        bw = int(xs.max()) - int(xs.min()) + 1
        bh = int(ys.max()) - int(ys.min()) + 1
        bbox_ratio = (bw * bh) / float(w * h)
        if bbox_ratio > MAX_BBOX_AREA_RATIO:
            return (
                f"外接框覆盖 {bbox_ratio:.1%} > {MAX_BBOX_AREA_RATIO:.0%}"
                f"（bbox {bw}x{bh}，弥散掩模拒绝）"
            )
        return None

    def _infer_grounded_sam(
        self, image_bgr: np.ndarray, classes: Optional[List[Dict[str, str]]]
    ) -> Dict[str, np.ndarray]:
        """执行 Grounding DINO + SAM 2 真实神经级联推理。"""
        from PIL import Image
        from groundingdino.util.inference import predict as predict_dino
        import groundingdino.datasets.transforms as T

        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 图像预处理 (Grounding DINO 规范)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(pil_img, None)

        # 初始化 SAM 2 图像特征
        self.sam_predictor.set_image(img_rgb)

        # 品类语义（含中文图层名）一律来自 preset.ai_semantic_classes（SSOT）。
        # 内置的屏风默认表已删除——provider 是品类无关的引擎层，
        # 未配置语义类时跳过神经检测，仅走规则引擎并明示原因，
        # 避免「新品类静默拿到屏风语义」的隐性错误。
        if not classes:
            print("[GroundedSAMProvider] preset 未配置 ai_semantic_classes，跳过神经检测（仅规则引擎）")
            return {}

        results = {}
        for cls_info in classes:
            layer_name = cls_info.get("name", "layer")
            prompt = cls_info.get("prompt", "")
            if not prompt:
                continue

            try:
                boxes, logits, phrases = predict_dino(
                    model=self.dino_model,
                    image=img_tensor,
                    caption=prompt,
                    box_threshold=DEFAULT_BOX_THRESHOLD,
                    text_threshold=DEFAULT_TEXT_THRESHOLD,
                    device=self.dino_device
                )
                if len(boxes) == 0:
                    continue

                layer_mask = np.zeros((h, w), dtype=np.uint8)
                instances: Dict[str, np.ndarray] = {}
                instance_split = bool(cls_info.get("instance_split"))
                for idx, box in enumerate(boxes, start=1):
                    cx, cy, bw, bh = box.tolist()
                    x1 = max(0, int((cx - bw / 2.0) * w))
                    y1 = max(0, int((cy - bh / 2.0) * h))
                    x2 = min(w, int((cx + bw / 2.0) * w))
                    y2 = min(h, int((cy + bh / 2.0) * h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    # 过滤全画幅假阳性大框（框住背景金箔）：阈值由 30% 收紧到 8%
                    if (x2 - x1) * (y2 - y1) > MAX_BOX_AREA_RATIO * w * h:
                        continue
                    sam_box = np.array([x1, y1, x2, y2])
                    masks, scores, _ = self.sam_predictor.predict(box=sam_box)
                    best_mask = masks[np.argmax(scores)]
                    if instance_split:
                        # 实例级拆分：每个检测框独立成层（如雁群 → 单只雁逐层）
                        instances[f"{layer_name}_{idx:02d}"] = best_mask.astype(np.uint8) * 255
                    else:
                        layer_mask = np.maximum(layer_mask, (best_mask.astype(np.uint8) * 255))

                if instance_split:
                    for iname, imask in instances.items():
                        if np.any(imask > 0):
                            results[iname] = imask
                    print(f"[GroundedSAMProvider] Neural instances '{layer_name}': {len(instances)} 个实例")
                elif np.any(layer_mask > 0):
                    results[layer_name] = layer_mask
                    print(f"[GroundedSAMProvider] Neural segmented layer '{layer_name}': {np.sum(layer_mask > 0)} px")
            except Exception as e:
                print(f"[GroundedSAMProvider] Error predicting class '{layer_name}': {e}")

        return results


    def _map_to_bilingual_names(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """检测原始名 → 中英对照图层名。

        SSOT（G2）：映射**只**来自 preset.layer_semantics.name_mapping。
        内置回退表已于 2026-09-10 删除（禁止再加回）——preset 缺失或未配置映射时，
        未命中的 key 原样保留：显式的「无品类语义」优于内置猜测，
        否则新品类会拿到屏风的图层名而不自知。
        """
        try:
            from engine.schemas.presets import get_name_mapping
            name_mapping = get_name_mapping(getattr(self, "preset", None) or {}) or {}
        except Exception:
            name_mapping = {}

        bilingual = {}
        for k, v in masks.items():
            clean_k = k.strip().lower()
            bilingual[name_mapping.get(clean_k, k)] = v
        return bilingual
