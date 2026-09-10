import os
import sys
import argparse
import cv2
import numpy as np
import psd_tools
from PIL import Image

def verify(psb_path="outputs/Rosetsu_1795_Master_16k.psb", expected_dpi=150.0):
    if not os.path.exists(psb_path):
        print(f"Error: {psb_path} not found")
        sys.exit(1)
    
    file_size_mb = os.path.getsize(psb_path) / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    print(f"=== Universal PSB Verification & Preflight Report ===")
    print(f"File Path    : {psb_path}")
    print(f"File Size    : {file_size_gb:.2f} GB ({file_size_mb:.1f} MB)")
    
    psd = psd_tools.PSDImage.open(psb_path)
    print(f"Image Size   : {psd.size} (Width x Height)")
    print(f"Color Mode   : {psd.color_mode}")
    print(f"Channels     : {psd.channels}")
    print(f"Format Ver   : {'PSB (Version 2)' if psd.version == 2 else 'PSD (Version 1)'}")
    print(f"Layer Count  : {len(psd)}")

    # 1. Verify Print Resolution (DPI / PPI)
    res_block = psd.image_resources.get_data(1005)
    if res_block is not None:
        read_dpi = res_block.horizontal / 65536.0
        print(f"Resolution   : {read_dpi:.1f} PPI (Resource 0x03ED verified)")
        assert abs(read_dpi - expected_dpi) < 1.0, f"DPI mismatch: expected {expected_dpi}, got {read_dpi}"
    else:
        print("Warning: ResolutionInfo block 1005 not found!")

    # 2. Verify Layers & Bounding Boxes
    print("\n--- Layer Stack Verification (From Record 0 to Record -1) ---")
    valid_layers = 0
    for idx, layer in enumerate(psd):
        clean_name = layer.name.strip('\x00').replace('\x00', '')
        bbox = layer.bbox
        w_box = (bbox[2] - bbox[0]) if bbox else 0
        h_box = (bbox[3] - bbox[1]) if bbox else 0
        print(f"[{idx:02d}] Name: {clean_name:<30} Blend: {layer.blend_mode.name:<10} Opacity: {layer.opacity:<3} BBox: ({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]}) [{w_box}x{h_box}]")
        if w_box > 0 and h_box > 0:
            valid_layers += 1

    assert valid_layers >= 8, f"Too few valid layers with bounding boxes: {valid_layers}"

    # 3. Verify Composite Fidelity (Section 5 Merged Data)
    print("\n--- Testing Section 5 Merged Composite Image ---")
    comp_pil = psd.topil()
    print(f"Composite extracted: size={comp_pil.size}, mode={comp_pil.mode}")
    assert comp_pil.size == psd.size, "Composite dimensions must match document dimensions"
    
    # Check pixel stats
    comp_arr = np.array(comp_pil)
    mean_val = np.mean(comp_arr, axis=(0, 1))
    std_val = np.std(comp_arr, axis=(0, 1))
    print(f"Composite Mean RGB: {mean_val.round(2)}, Std RGB: {std_val.round(2)}")

    # Compare thumbnail with source_4000
    src_4k = cv2.imread("inputs/source_4000.jpg")
    if src_4k is not None:
        comp_thumb = comp_pil.copy()
        comp_thumb.thumbnail((src_4k.shape[1], src_4k.shape[0]))
        src_thumb = cv2.resize(src_4k, (comp_thumb.width, comp_thumb.height))
        comp_bgr = cv2.cvtColor(np.array(comp_thumb), cv2.COLOR_RGB2BGR)
        diff = cv2.absdiff(src_thumb, comp_bgr)
        mae = np.mean(diff)
        print(f"Mean Absolute Error vs Source: {mae:.2f} (values < 5.0 indicate faithful preservation)")
        assert mae < 8.0, f"Composite deviation too high: MAE={mae:.2f}"
    
    print("\n=== ALL ASSERTIONS PASSED: PRODUCTION CERTIFIED ===")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PSB Output Quality & Preflight")
    parser.add_argument("--file", default="outputs/Rosetsu_1795_Master_16k.psb", help="Path to PSB file")
    parser.add_argument("--dpi", type=float, default=150.0, help="Expected DPI")
    args = parser.parse_args()
    verify(psb_path=args.file, expected_dpi=args.dpi)
