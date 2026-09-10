import os
import gc
import cv2
import struct
import numpy as np
from pytoshop import enums, core
from pytoshop.user import nested_layers
from pytoshop.image_resources import GenericImageResourceBlock

from engine.codecs_accelerator import install_psb_codec_accelerator
from engine.core.color_manager import ColorManager
from engine.core.ink_limiter import limit_ink
install_psb_codec_accelerator()

#: CMYK 图层通道标识（必须用 ColorChannel 枚举，不能用整数索引 0-3；
#: pytoshop 的 ColorChannelMapping 会把整数 0 解释为 ColorChannel.bitmap）
CMYK_CHANNELS = (
    enums.ColorChannel.cyan,
    enums.ColorChannel.magenta,
    enums.ColorChannel.yellow,
    enums.ColorChannel.black,
)


class UniversalPSBBuilder:
    def __init__(self, target_w=16000, target_h=7808, dpi=150.0,
                 compression=enums.Compression.rle, color_mode="rgb",
                 icc_path=None, tac_policy=None):
        install_psb_codec_accelerator()
        self.target_w = target_w
        self.target_h = target_h
        self.dpi = float(dpi)
        self.compression = compression
        self.color_mode = str(color_mode).lower()
        if self.color_mode not in ("rgb", "cmyk"):
            raise ValueError(f"未知 color_mode: {color_mode}（只允许 rgb / cmyk）")
        #: 目标印刷条件的 ICC profile（分色与黑版生成由它驱动）
        self.icc_path = icc_path
        #: TAC 工艺策略（上限来自印刷条件，不是 ICC —— ICC 里没有该字段）
        self.tac_policy = tac_policy
        #: 最近一次写盘的 TAC 审计结果 (max_pct, mean_pct)
        self.last_tac = None
        #: 最近一次写盘的详细油墨统计（供 manifest 披露）
        self.last_ink_stats = None

    @property
    def is_cmyk(self) -> bool:
        return self.color_mode == "cmyk"

    def _to_cmyk_limited(self, bgr: np.ndarray) -> np.ndarray:
        """BGR → CMYK 磁盘反码，并执行 TAC 压制。

        为什么每条 CMYK 数据（含 Section 5、背景层、各图层）都要压制：
        TAC 是「最终输出像素」的属性，任何一条超限都会在印厂糊版；
        统一用同一 policy 压制可保证各层墨量一致、可复算。
        """
        raw = ColorManager.bgr_to_cmyk_raw(bgr, icc_path=self.icc_path)
        if self.tac_policy is not None:
            # inplace=True：转换产物用完即弃，可安全原地修改（16K 下省一次 500MB 复制）
            raw, stats = limit_ink(raw, self.tac_policy, inplace=True)
            # 以 Section 5（全画幅）的统计为准对外披露
            if self.last_ink_stats is None or raw.size > 0:
                self.last_ink_stats = stats
        return raw

    def _apply_layer_channels(self, ps_layer, crop_bgr: np.ndarray, crop_m: np.ndarray) -> None:
        """按目标色彩模式写入图层通道。

        CMYK 路径用 `ColorManager.bgr_to_cmyk_raw`（可带 ICC）：
        返回的已是 PSD 磁盘反码（255 = 0% 墨），与 pytoshop 存储约定一致，
        因此**不得**再经 `255 - x` 二次反转。
        """
        if self.is_cmyk:
            raw = self._to_cmyk_limited(crop_bgr)  # (4, h, w) uint8 反码
            for i, cc in enumerate(CMYK_CHANNELS):
                ps_layer.set_channel(cc, np.ascontiguousarray(raw[i]))
        else:
            crop_b, crop_g, crop_r = cv2.split(crop_bgr)
            ps_layer.set_channel(enums.ColorChannel.red, np.ascontiguousarray(crop_r))
            ps_layer.set_channel(enums.ColorChannel.green, np.ascontiguousarray(crop_g))
            ps_layer.set_channel(enums.ColorChannel.blue, np.ascontiguousarray(crop_b))
        ps_layer.set_channel(enums.ColorChannel.transparency, np.ascontiguousarray(crop_m))

    def _create_resolution_block(self):
        """Creates Photoshop 0x03ED (1005) ResolutionInfo block for exact print scaling."""
        h_res = int(self.dpi * 65536)
        v_res = int(self.dpi * 65536)
        # Struct: 4B h_res (16.16), 2B h_unit (1=pixels/in), 2B w_unit (1=in), 4B v_res, 2B v_unit, 2B h_unit
        data = struct.pack('>IHH IHH', h_res, 1, 1, v_res, 1, 1)
        return GenericImageResourceBlock(resource_id=1005, name='', data=data)

    def build_psb(self, output_path, src_hr_bgr, bg_hr_bgr, sorted_layers, hr_masks_dict, bg_layer_name="02_纯净金箔大底板_Gold_Base_Clean"):
        """
        Compiles all layers and writes a production-grade PSB file.
        
        Args:
            output_path: Target .psb file path
            src_hr_bgr: Full-resolution source image (BGR)
            bg_hr_bgr: Full-resolution clean background image (BGR)
            sorted_layers: List of layer metadata dicts from UI Top to UI Bottom
            hr_masks_dict: dict of {name: 16K mask_uint8}
            bg_layer_name: Name for the bottom background layer
        """
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        ps_color_mode = enums.ColorMode.cmyk if self.is_cmyk else enums.ColorMode.rgb

        # 1. Section 5 Pre-rendered Merged Composite Channels
        if self.is_cmyk:
            # (4, H, W) PSD 磁盘反码；转换后立即做 TAC 压制（印前合规要求）
            self.last_ink_stats = None
            comp_channels = self._to_cmyk_limited(src_hr_bgr)
            # TAC 实测（压制后）随 manifest 披露
            self.last_tac = ColorManager.calculate_tac(comp_channels)
        else:
            self.last_tac = None
            self.last_ink_stats = None
            comp_channels = np.stack([
                np.ascontiguousarray(src_hr_bgr[:, :, 2]),
                np.ascontiguousarray(src_hr_bgr[:, :, 1]),
                np.ascontiguousarray(src_hr_bgr[:, :, 0]),
            ], axis=0)

        # 2. Build layers from UI Top to Bottom
        layers_to_build = []

        for layer_info in sorted_layers:
            name = layer_info["name"]
            if name not in hr_masks_dict:
                continue

            m = hr_masks_dict[name]
            pts = cv2.findNonZero(m)
            if pts is None:
                continue

            rx, ry, rw, rh = cv2.boundingRect(pts)
            pad = 8
            x0 = max(0, rx - pad)
            y0 = max(0, ry - pad)
            x1 = min(self.target_w, rx + rw + pad)
            y1 = min(self.target_h, ry + rh + pad)

            crop_m = m[y0:y1, x0:x1]
            
            # 统一成 BGR 裁剪块，再由 _apply_layer_channels 按目标色彩模式落通道
            if layer_info.get("fixed_color") is not None:
                fc = layer_info["fixed_color"]
                crop_bgr = np.empty((*crop_m.shape, 3), dtype=np.uint8)
                crop_bgr[:, :, 0] = fc[0]
                crop_bgr[:, :, 1] = fc[1]
                crop_bgr[:, :, 2] = fc[2]
            else:
                # Use inpainted content if available for deoccluded layers, otherwise source
                color_src = layer_info.get("inpainted_hr_bgr")
                if color_src is None:
                    color_src = src_hr_bgr
                crop_bgr = color_src[y0:y1, x0:x1]

            # Map blend mode
            bmode = enums.BlendMode.multiply if layer_info.get("blend_mode") == "MULTIPLY" else enums.BlendMode.normal
            opacity = layer_info.get("opacity", 255)

            ps_layer = nested_layers.Image(
                name=name,
                color_mode=ps_color_mode,
                blend_mode=bmode,
                opacity=opacity,
                top=y0, left=x0, bottom=y1, right=x1
            )
            self._apply_layer_channels(ps_layer, crop_bgr, crop_m)
            layers_to_build.append(ps_layer)

        # 3. Add base background layer at the very bottom (UI Bottom)
        bg_layer = nested_layers.Image(
            name=bg_layer_name or "02_纯净金箔大底板_Gold_Base_Clean",
            color_mode=ps_color_mode,
            blend_mode=enums.BlendMode.normal,
            opacity=255,
            top=0, left=0, bottom=self.target_h, right=self.target_w
        )
        bg_alpha = np.full((self.target_h, self.target_w), 255, dtype=np.uint8)
        self._apply_layer_channels(bg_layer, bg_hr_bgr, bg_alpha)
        layers_to_build.append(bg_layer)

        # 4. Convert nested layers to PSD document
        psd_doc = nested_layers.nested_layers_to_psd(
            layers=layers_to_build,
            color_mode=ps_color_mode,
            version=enums.Version.version_2, # PSB Format
            compression=self.compression,
            size=(self.target_w, self.target_h)
        )

        # 4.1 注入 Unicode 图层名 (luni Tagged Block)
        # 注意: pytoshop 在 nested_layers_to_psd 内部执行了 reversed(layers) 写入 layer_records
        # 故 recs[0] 为最底层，recs[-1] 为最顶层，与 reversed(layers_to_build) 精确 1:1 对齐
        from pytoshop import tagged_block
        if hasattr(psd_doc, "layer_and_mask_info") and hasattr(psd_doc.layer_and_mask_info, "layer_info"):
            recs = psd_doc.layer_and_mask_info.layer_info.layer_records
            for layer_rec, orig_layer in zip(recs, reversed(layers_to_build)):
                if hasattr(orig_layer, "name") and orig_layer.name:
                    layer_rec.blocks.append(tagged_block.UnicodeLayerName(orig_layer.name))

        # 5. Inject Print Resolution Info (0x03ED)
        res_block = self._create_resolution_block()
        psd_doc.image_resources.blocks.append(res_block)

        # 6. Inject Section 5 Image Data
        psd_doc.image_data = core.ImageData(
            channels=comp_channels,
            compression=self.compression
        )

        # 7. Stream to disk
        with open(output_path, "wb") as fd:
            psd_doc.write(fd)

        del comp_channels
        gc.collect()

        return output_path
