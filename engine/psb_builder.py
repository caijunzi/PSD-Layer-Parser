import os
import gc
import cv2
import struct
import numpy as np
from pytoshop import enums, core
from pytoshop.user import nested_layers
from pytoshop.image_resources import GenericImageResourceBlock

from engine.codecs_accelerator import install_psb_codec_accelerator
install_psb_codec_accelerator()

class UniversalPSBBuilder:
    def __init__(self, target_w=16000, target_h=7808, dpi=150.0, compression=enums.Compression.rle):
        install_psb_codec_accelerator()
        self.target_w = target_w
        self.target_h = target_h
        self.dpi = float(dpi)
        self.compression = compression

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
        
        # 1. Section 5 Pre-rendered Merged Composite Channels (R, G, B)
        comp_r = np.ascontiguousarray(src_hr_bgr[:, :, 2])
        comp_g = np.ascontiguousarray(src_hr_bgr[:, :, 1])
        comp_b = np.ascontiguousarray(src_hr_bgr[:, :, 0])
        comp_channels = np.stack([comp_r, comp_g, comp_b], axis=0)

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
            
            # Check if layer has a fixed color (e.g. fold seams antique tone)
            if layer_info.get("fixed_color") is not None:
                fc = layer_info["fixed_color"]
                crop_b = np.full_like(crop_m, fc[0], dtype=np.uint8)
                crop_g = np.full_like(crop_m, fc[1], dtype=np.uint8)
                crop_r = np.full_like(crop_m, fc[2], dtype=np.uint8)
            else:
                # Use inpainted content if available for deoccluded layers, otherwise source
                color_src = layer_info.get("inpainted_hr_bgr")
                if color_src is None:
                    color_src = src_hr_bgr
                    
                crop_src = color_src[y0:y1, x0:x1]
                crop_b, crop_g, crop_r = cv2.split(crop_src)

            # Map blend mode
            bmode = enums.BlendMode.multiply if layer_info.get("blend_mode") == "MULTIPLY" else enums.BlendMode.normal
            opacity = layer_info.get("opacity", 255)

            ps_layer = nested_layers.Image(
                name=name,
                color_mode=enums.ColorMode.rgb,
                blend_mode=bmode,
                opacity=opacity,
                top=y0, left=x0, bottom=y1, right=x1
            )
            ps_layer.set_channel(enums.ColorChannel.red, crop_r)
            ps_layer.set_channel(enums.ColorChannel.green, crop_g)
            ps_layer.set_channel(enums.ColorChannel.blue, crop_b)
            ps_layer.set_channel(enums.ColorChannel.transparency, crop_m)
            layers_to_build.append(ps_layer)

        # 3. Add base background layer at the very bottom (UI Bottom)
        bg_b, bg_g, bg_r = cv2.split(bg_hr_bgr)
        bg_layer = nested_layers.Image(
            name=bg_layer_name or "02_纯净金箔大底板_Gold_Base_Clean",
            color_mode=enums.ColorMode.rgb,
            blend_mode=enums.BlendMode.normal,
            opacity=255,
            top=0, left=0, bottom=self.target_h, right=self.target_w
        )
        bg_layer.set_channel(enums.ColorChannel.red, bg_r)
        bg_layer.set_channel(enums.ColorChannel.green, bg_g)
        bg_layer.set_channel(enums.ColorChannel.blue, bg_b)
        bg_layer.set_channel(enums.ColorChannel.transparency, np.full((self.target_h, self.target_w), 255, dtype=np.uint8))
        layers_to_build.append(bg_layer)

        # 4. Convert nested layers to PSD document
        psd_doc = nested_layers.nested_layers_to_psd(
            layers=layers_to_build,
            color_mode=enums.ColorMode.rgb,
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

        del comp_r, comp_g, comp_b, comp_channels, bg_b, bg_g, bg_r
        gc.collect()

        return output_path
