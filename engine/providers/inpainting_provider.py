"""Inpainting Providers for Universal Neural Layer Studio.
Supports SOTA Fast Fourier Convolution (FFC) inpainting (LaMa)
with heterogeneous hardware acceleration and deterministic circuit breaker:
  - Primary: NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell sm_120)
  - Shield: Intel Arc 140T GPU (16GB shared memory)
  - Circuit Breaker: Tile-level fault isolation & instant CPU recovery
"""
import os
import gc
import cv2
import numpy as np
from typing import Optional, Any
from engine.providers.base_provider import BaseInpaintingProvider
from engine.schemas.profile_config import resolve_profile, ProfileSettings

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class LaMaInpaintingProvider(BaseInpaintingProvider):
    """
    SOTA Fast Fourier Convolution (FFC) Large Mask Inpainting.
    Implements plugged-in robust high-performance scheduling:
    - Primary Tensor Core throughput on RTX 5070 (GPU.1)
    - 16GB memory shield on Intel Arc 140T (GPU.0)
    - Deterministic tile-level circuit breaker & CPU recovery
    """

    # LaMa 是基于 FFC 的生成式补全模型，其输出属于"生成内容"，
    # 按 §3.1 硬边界 1 不得进入 PLATE 制版线（仅 DESIGN 线可用）。
    is_generative = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        preferred_device: Optional[str] = None,
        profile_name: str = "robust_performance"
    ):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "checkpoints", "lama_fp32.onnx"
        )
        self.model_path = os.path.abspath(self.model_path)
        self.profile: ProfileSettings = resolve_profile(profile_name)
        self.preferred_device = preferred_device or self.profile.primary_device
        self.shield_device = self.profile.shield_device
        self.backend = None # 'openvino_GPU.1', 'openvino_GPU.0', 'openvino_CPU', 'onnxruntime_cpu'
        self.compiled_model = None
        self.shield_model = None
        self.session = None
        self._init_session()

    def _init_session(self):
        if not os.path.isfile(self.model_path):
            return

        # 1. OpenVINO Heterogeneous Dispatch
        if HAS_OPENVINO:
            try:
                core = ov.Core()
                available = core.available_devices
                
                # Determine ordered candidate list
                # LaMa 实测 Intel Arc(GPU.0) 比 RTX 5070(GPU.1) 快 ~25x（OpenVINO 通用后端
                # 对 NVIDIA 无原生 EP；Arc 有原生 Level Zero）：GPU.0 0.225s/片 vs GPU.1 5.76s/片。
                # 故护盾为 Arc 时优先护盾设备；否则尊重首选。见 scratch/bench_lama_gpu.py。
                pref = (self.preferred_device or self.profile.primary_device).upper()
                shield = self.profile.shield_device.upper()
                order = ([shield, pref] if (pref == "GPU.1" and shield == "GPU.0")
                         else [pref, shield])
                order += ["GPU.1", "GPU.0", "CPU"]
                candidates = []
                for cand in order:
                    if cand in available and cand not in candidates:
                        candidates.append(cand)

                model = core.read_model(self.model_path)
                for dev in candidates:
                    try:
                        self.compiled_model = core.compile_model(model, dev)
                        self.backend = f"openvino_{dev}"
                        print(f"[LaMaInpaintingProvider] Active Primary Device: [{dev}] (Requested: {self.preferred_device}, Profile: {self.profile.profile_id})")
                        return
                    except Exception as dev_err:
                        print(f"[LaMaInpaintingProvider] Device [{dev}] compilation unavailable: {dev_err}. Trying fallback device...")
            except Exception as e:
                print(f"[LaMaInpaintingProvider] OpenVINO init warning: {e}, falling back to ORT CPU...")

        # 2. Fallback to ONNX Runtime CPU
        if HAS_ORT:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(8, os.cpu_count() or 4)
                opts.inter_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                self.backend = "onnxruntime_cpu"
                print("[LaMaInpaintingProvider] Active Backend: [ONNXRuntime CPU]")
            except Exception as e:
                print(f"[LaMaInpaintingProvider] Warning: Failed to initialize ORT session: {e}")
                self.session = None

    def inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        Inpaints mask_u8 regions (255 = inpaint target) seamlessly.
        """
        pts = cv2.findNonZero((mask_u8 > 0).astype(np.uint8))
        if pts is None or len(pts) == 0:
            return img_bgr.copy()

        total_hole_area = len(pts)

        # Fast path 1: Micro-holes (< 2500 px) or no neural backend
        # Telea completes in < 5ms without PCIe or VRAM overhead
        if total_hole_area < 2500 or self.backend is None:
            return cv2.inpaint(img_bgr, (mask_u8 > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)

        # Fast path 2: Bounding-box localization
        h, w, _ = img_bgr.shape
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad = 48
        y0 = max(0, by - pad)
        y1 = min(h, by + bh + pad)
        x0 = max(0, bx - pad)
        x1 = min(w, bx + bw + pad)

        roi_img = img_bgr[y0:y1, x0:x1]
        roi_mask = mask_u8[y0:y1, x0:x1]
        roi_h, roi_w, _ = roi_img.shape

        # If ROI fits in a single 512x512 tile, run single inference
        if roi_h <= 512 and roi_w <= 512:
            roi_inpainted = self._inpaint_single(roi_img, roi_mask)
        else:
            roi_inpainted = self._inpaint_tiled(roi_img, roi_mask)

        # Splice back into canvas seamlessly
        result = img_bgr.copy()
        mask_roi = (roi_mask > 0)[:, :, np.newaxis]
        result[y0:y1, x0:x1] = np.where(mask_roi, roi_inpainted, roi_img)

        # Clean memory to maintain safe VRAM headroom
        gc.collect()
        return result

    def _inpaint_tile_512(self, tile_bgr: np.ndarray, tile_mask_u8: np.ndarray) -> np.ndarray:
        """
        Inpaints an exact 512x512 tile via active neural backend.
        Features deterministic circuit breaker: if GPU throws a driver or memory fault,
        immediately catches and recovers via CPU fallback for that tile.
        """
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tile_tensor = np.transpose(tile_rgb, (2, 0, 1))[np.newaxis, :, :, :]
        mask_tensor = (tile_mask_u8.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

        if self.backend and self.backend.startswith("openvino"):
            try:
                out = self.compiled_model({"image": tile_tensor, "mask": mask_tensor})[self.compiled_model.output(0)]
                out_rgb = np.clip(out[0].transpose((1, 2, 0)), 0, 255).astype(np.uint8)
                return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                # Deterministic Circuit Breaker
                print(f"[Circuit Breaker] Warning: Tile execution on {self.backend} failed: {e}. Engaging fail-safe CPU recovery...")
                return cv2.inpaint(tile_bgr, (tile_mask_u8 > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)

        elif self.session is not None:
            try:
                pred = self.session.run(None, {"image": tile_tensor, "mask": mask_tensor})[0]
                out_rgb = np.clip(pred[0].transpose((1, 2, 0)), 0, 255).astype(np.uint8)
                return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[Circuit Breaker] ORT Session error: {e}, falling back to Telea...")
                return cv2.inpaint(tile_bgr, (tile_mask_u8 > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)

        return cv2.inpaint(tile_bgr, (tile_mask_u8 > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)

    def _inpaint_tiles_batched(self, tiles_list: list) -> list:
        """
        Batched tile inference on active hardware accelerator.
        Processes tiles in batches of up to 4 to maximize GPU Tensor Core throughput.
        Features deterministic circuit breaker: falls back to Telea on any batch fault.
        """
        if not tiles_list:
            return []

        results = []
        batch_size = 4

        for b_idx in range(0, len(tiles_list), batch_size):
            chunk = tiles_list[b_idx : b_idx + batch_size]
            b_imgs = [t[0] for t in chunk]
            b_masks = [t[1] for t in chunk]
            B = len(chunk)

            try:
                b_rgb = np.stack([
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    for img in b_imgs
                ], axis=0).transpose((0, 3, 1, 2))
                b_mask = np.stack([
                    (m.astype(np.float32) / 255.0)[np.newaxis, :, :]
                    for m in b_masks
                ], axis=0)

                if self.backend and self.backend.startswith("openvino"):
                    res = self.compiled_model({"image": b_rgb, "mask": b_mask})
                    out = res[self.compiled_model.output(0)]
                    for i in range(B):
                        out_rgb = np.clip(out[i].transpose((1, 2, 0)), 0, 255).astype(np.uint8)
                        results.append(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                elif self.session is not None:
                    preds = self.session.run(None, {"image": b_rgb, "mask": b_mask})[0]
                    for i in range(B):
                        out_rgb = np.clip(preds[i].transpose((1, 2, 0)), 0, 255).astype(np.uint8)
                        results.append(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                else:
                    for i in range(B):
                        results.append(cv2.inpaint(b_imgs[i], (b_masks[i] > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA))
            except Exception as e:
                # Circuit breaker fallback to Telea
                print(f"[Circuit Breaker] Batch inference on {self.backend} failed: {e}. Falling back to Telea...")
                for i in range(B):
                    results.append(cv2.inpaint(b_imgs[i], (b_masks[i] > 0).astype(np.uint8)*255, 5, cv2.INPAINT_TELEA))

        return results

    @staticmethod
    def _tile_starts(length: int, tile: int, step: int) -> list:
        """计算切片起点，保证每个切片都完整落在 [0, length) 内。

        旧实现 `range(0, max(1, length - tile), step)` 在 length < tile 时返回 [0]，
        随后的切片 `[0:tile]` 会被 Python 截断为实际长度，产生非 512 的 tile。
        """
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile, step))
        if starts[-1] + tile < length:
            starts.append(length - tile)
        return starts

    @staticmethod
    def _pad_to_tile(img_bgr: np.ndarray, mask_u8: np.ndarray, size: int = 512):
        """把不足 size 的 tile 补齐到 size x size。

        修复要点（2026-09-10）：OpenVINO LaMa 模型的输入端口固定为 [?,3,512,512]。
        旧实现在 ROI 任一维度 < 512 时会产生 396x512 / 512x396 这类 tile，
        batch stack 后尺寸与模型不符 -> 报错 -> 整批熔断回退 Telea；
        且回退后回填时 `(396,512,3) * hann(512,512,1)` 广播失败抛 ValueError
        （生产中由 deocclusion_operator 外层 try/except 兜底，未暴露）。
        统一 pad 后，所有 tile 均为 512x512，LaMa 才能对所有 ROI 形状生效。
        """
        h, w = img_bgr.shape[:2]
        pad_y, pad_x = max(0, size - h), max(0, size - w)
        if pad_y == 0 and pad_x == 0:
            return img_bgr, mask_u8
        img_pad = cv2.copyMakeBorder(img_bgr, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)
        mask_pad = cv2.copyMakeBorder(mask_u8, 0, pad_y, 0, pad_x,
                                      cv2.BORDER_CONSTANT, value=0)
        return img_pad, mask_pad

    def _inpaint_tiled(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Tiled inpainting with Hann-window cosine feather blending across active tiles."""
        h, w, _ = img_bgr.shape
        tile_size = 512
        step = 448  # 64px smooth cosine overlap (reduces redundant tile count by 30%)

        out_acc = np.zeros((h, w, 3), dtype=np.float32)
        weight_acc = np.zeros((h, w, 1), dtype=np.float32)

        hann_1d_y = np.hanning(tile_size).astype(np.float32)
        hann_1d_x = np.hanning(tile_size).astype(np.float32)
        hann_full = np.clip(np.outer(hann_1d_y, hann_1d_x)[:, :, np.newaxis], 1e-4, 1.0)

        y_starts = self._tile_starts(h, tile_size, step)
        x_starts = self._tile_starts(w, tile_size, step)

        # (y0, x0, th, tw, tile_img, tile_mask) —— th/tw 为实际尺寸，可能 < 512
        tiles_to_inpaint = []

        for y0 in y_starts:
            th = min(tile_size, h - y0)
            for x0 in x_starts:
                tw = min(tile_size, w - x0)
                tile_mask = mask_u8[y0:y0 + th, x0:x0 + tw]
                tile_img = img_bgr[y0:y0 + th, x0:x0 + tw]
                hann = hann_full[:th, :tw]
                nonzero_count = np.count_nonzero(tile_mask)

                if nonzero_count == 0:
                    accumulated = tile_img.astype(np.float32)
                elif nonzero_count < 200 or self.backend is None:
                    # Micro-hole / fringe fast path (< 200 px): Telea takes < 1ms
                    accumulated = cv2.inpaint(
                        tile_img, (tile_mask > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA
                    ).astype(np.float32)
                else:
                    tiles_to_inpaint.append((y0, x0, th, tw, tile_img, tile_mask))
                    continue

                out_acc[y0:y0 + th, x0:x0 + tw] += accumulated * hann
                weight_acc[y0:y0 + th, x0:x0 + tw] += hann

        # Batched neural execution (所有 tile 先 pad 到 512x512，保证 batch 尺寸一致)
        if tiles_to_inpaint:
            padded_inputs = [self._pad_to_tile(t[4], t[5], tile_size) for t in tiles_to_inpaint]
            inpainted_tiles = self._inpaint_tiles_batched(padded_inputs)
            for (y0, x0, th, tw, _, _), inp_tile in zip(tiles_to_inpaint, inpainted_tiles):
                cropped = inp_tile[:th, :tw].astype(np.float32)
                hann = hann_full[:th, :tw]
                out_acc[y0:y0 + th, x0:x0 + tw] += cropped * hann
                weight_acc[y0:y0 + th, x0:x0 + tw] += hann

        weight_acc = np.maximum(weight_acc, 1e-5)
        blended = out_acc / weight_acc
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        mask_bool = (mask_u8 > 0)[:, :, np.newaxis]
        return np.where(mask_bool, blended, img_bgr)

    def _inpaint_single(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Pads an arbitrary small image (< 512) to 512x512, inpaints, and crops back."""
        h, w, _ = img_bgr.shape
        pad_y = max(0, 512 - h)
        pad_x = max(0, 512 - w)
        
        img_pad = cv2.copyMakeBorder(img_bgr, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)
        mask_pad = cv2.copyMakeBorder(mask_u8, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
        
        inpainted_pad = self._inpaint_tile_512(img_pad[:512, :512], mask_pad[:512, :512])
        return inpainted_pad[:h, :w]


class TeleaInpaintingProvider(BaseInpaintingProvider):
    """确定性补全（Telea / Navier-Stokes）—— PLATE 制版线专用。

    与 `LaMaInpaintingProvider` 的关键区别：本实现是**经典图像修复算法**，
    结果确定、可复算、不含模型幻觉，满足 §3.1 硬边界 1 对制版线的要求
    （"任何生成式/扩散模型的输出不得进入 PLATE 线"）。

    类名保留 Telea 是为了向后兼容早期仅实现 Telea 的版本。
    """

    #: 经典插值算法，非生成式模型 → PLATE 线可用
    is_generative = False

    #: telea    —— 快速行进法，细节保留好，适合中小区域
    #: ns       —— Navier-Stokes，边界平滑，适合大面积匀色区
    #: telea_ns —— 两者等权融合（默认），兼顾细节与平滑
    METHODS = ("telea", "ns", "telea_ns")

    def __init__(self, method: str = "telea_ns", radius: int = 5):
        if method not in self.METHODS:
            raise ValueError(f"未知补全方法: {method}（可选 {self.METHODS}）")
        self.method = method
        self.radius = max(1, int(radius))

    def inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask_u8) == 0:
            return img_bgr.copy()
        m = (mask_u8 > 0).astype(np.uint8) * 255
        if self.method == "telea":
            return cv2.inpaint(img_bgr, m, self.radius, cv2.INPAINT_TELEA)
        if self.method == "ns":
            return cv2.inpaint(img_bgr, m, self.radius, cv2.INPAINT_NS)
        # 融合：Telea 保细节、NS 平大面积，等权叠加抑制各自伪影
        telea = cv2.inpaint(img_bgr, m, self.radius, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(img_bgr, m, self.radius, cv2.INPAINT_NS)
        return cv2.addWeighted(telea, 0.5, ns, 0.5, 0)
