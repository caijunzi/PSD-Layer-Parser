import time
import os
import struct
import cv2
import numpy as np
import pytoshop
from pytoshop import enums, image_resources
from pytoshop.user import nested_layers
from psd_tools import PSDImage
from psd_tools.constants import Resource

def build_print_psd():
    t0 = time.time()
    print("=== Starting Textile Swatch 1:1 Restoration & CMYK PSD Assembly ===")

    # 1. Load rectified clean master (450 x 800)
    master_path = 'intermediate/swatch/master_restored_clean.png'
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Missing master file: {master_path}")
    
    img_low = cv2.imread(master_path)
    h_low, w_low, _ = img_low.shape
    print(f"Loaded master prototype: {w_low}x{h_low}")

    # 2. Target dimensions: 900 mm x 1600 mm at 150 DPI
    target_w = 5315
    target_h = 9449
    scale_x = target_w / float(w_low)
    scale_y = target_h / float(h_low)
    print(f"Target print resolution: {target_w}x{target_h} (150 DPI, 900x1600 mm), scale factor: {scale_x:.4f}")

    # 3. Detect micro-holes at prototype scale
    print("Detecting micro-holes via Difference-of-Gaussians (DoG)...")
    gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray_low, (0, 0), sigmaX=0.9)
    g2 = cv2.GaussianBlur(gray_low, (0, 0), sigmaX=2.0)
    dog = g2 - g1

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(dog, kernel3)
    is_print_area = (gray_low < 225) & (dog > 4.5)
    peaks = (dog == dilated) & is_print_area
    y_peaks, x_peaks = np.where(peaks)
    print(f"Detected {len(x_peaks)} distinct micro-hole coordinates.")

    # 4. Render High-Resolution DieCut Perforation Mask (Layer 02)
    print("Rendering high-resolution vector-grade DieCut Mask (5315x9449)...")
    # Standard die-cut stencil: White = solid fabric (keep), Black = die-cut hole (punch)
    diecut_mask_hi = np.full((target_h, target_w), 255, dtype=np.uint8)
    
    # Scale hole positions and render smooth anti-aliased ellipses
    hole_rx = int(round(1.9 * scale_x))
    hole_ry = int(round(2.3 * scale_y))
    for yp, xp in zip(y_peaks, x_peaks):
        X = int(round(xp * scale_x))
        Y = int(round(yp * scale_y))
        cv2.ellipse(diecut_mask_hi, (X, Y), (hole_rx, hole_ry), 0, 0, 360, 0, -1, lineType=cv2.LINE_AA)
    
    # 5. Extract and Scale Print Masks
    print("Upscaling master image and separating Gold vs Copper Foil...")
    img_hi = cv2.resize(img_low, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    gray_hi = cv2.cvtColor(img_hi, cv2.COLOR_BGR2GRAY)

    # In prototype:
    lab_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2LAB)
    A_low = lab_low[:, :, 1]
    R_low = img_low[:, :, 2].astype(float)
    B_low = img_low[:, :, 0].astype(float)

    is_print_low = gray_low < 225
    is_copper_low = is_print_low & (R_low - B_low > 85) & (A_low >= 139)
    is_copper_low[:190, :] = False
    is_copper_low[440:, :] = False
    copper_mask_low = cv2.morphologyEx(is_copper_low.astype(np.uint8)*255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    gold_mask_low = (is_print_low.astype(np.uint8)*255) & (~copper_mask_low)
    gold_mask_low = cv2.morphologyEx(gold_mask_low, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    # Scale alpha masks to high resolution
    copper_mask_hi = cv2.resize(copper_mask_low, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    gold_mask_hi = cv2.resize(gold_mask_low, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Threshold and knockout diecut holes
    copper_alpha = (copper_mask_hi > 120).astype(np.uint8) * 255
    copper_alpha[diecut_mask_hi == 0] = 0 # knockout holes!

    gold_alpha = (gold_mask_hi > 120).astype(np.uint8) * 255
    gold_alpha[diecut_mask_hi == 0] = 0 # knockout holes!

    print(f"High-res masks ready. Copper active px: {np.sum(copper_alpha > 0)}, Gold active px: {np.sum(gold_alpha > 0)}")

    # 6. CMYK Color Channel Mapping (Japan Color 2001 Coated / FOGRA39 standard)
    # Reminder: In pytoshop raw CMYK, raw = 255 - ink_density_8bit
    # 0% ink = 255 (blank paper), 100% ink = 0.
    
    print("Synthesizing CMYK Channels for Layer 01 (Base Substrate)...")
    # Base Substrate: C: 2%, M: 3%, Y: 8%, K: 0%
    # ink: C=5, M=8, Y=20, K=0 -> raw: C=250, M=247, Y=235, K=255
    l1_c = np.full((target_h, target_w), 250, dtype=np.uint8)
    l1_m = np.full((target_h, target_w), 247, dtype=np.uint8)
    l1_y = np.full((target_h, target_w), 235, dtype=np.uint8)
    l1_k = np.full((target_h, target_w), 255, dtype=np.uint8)
    l1_alpha = np.full((target_h, target_w), 255, dtype=np.uint8)

    layer01 = nested_layers.Image(
        name='01_Base_Substrate_CMYK',
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=0, left=0, bottom=target_h, right=target_w,
        channels={-1: l1_alpha, 0: l1_c, 1: l1_m, 2: l1_y, 3: l1_k}
    )

    print("Synthesizing CMYK Channels for Layer 02 (DieCut Perforations Mask)...")
    # Mask Layer: White fabric = 0% ink (raw 255), Black punched holes = 100% K (raw 0)
    l2_c = np.full((target_h, target_w), 255, dtype=np.uint8)
    l2_m = np.full((target_h, target_w), 255, dtype=np.uint8)
    l2_y = np.full((target_h, target_w), 255, dtype=np.uint8)
    l2_k = diecut_mask_hi.copy() # 255 for fabric (no ink), 0 for holes (100% black ink)
    l2_alpha = np.full((target_h, target_w), 255, dtype=np.uint8)

    layer02 = nested_layers.Image(
        name='02_DieCut_Perforations_Mask',
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=0, left=0, bottom=target_h, right=target_w,
        channels={-1: l2_alpha, 0: l2_c, 1: l2_m, 2: l2_y, 3: l2_k}
    )

    print("Synthesizing CMYK Channels for Layer 03 (Antique Gold Print)...")
    darkness = np.clip((235.0 - gray_hi.astype(np.float32)) / 140.0, 0.0, 1.0)
    
    # Calculate active ink values
    gold_ink_c = np.clip(225.0 - 46.0 * darkness, 0, 255).astype(np.uint8)
    gold_ink_m = np.clip(184.0 - 71.0 * darkness, 0, 255).astype(np.uint8)
    gold_ink_y = np.clip(82.0 - 51.0 * darkness, 0, 255).astype(np.uint8)
    gold_ink_k = np.clip(245.0 - 66.0 * darkness, 0, 255).astype(np.uint8)

    # Initialize layer channels with pure blank (255 = 0% ink)
    l3_c = np.full((target_h, target_w), 255, dtype=np.uint8)
    l3_m = np.full((target_h, target_w), 255, dtype=np.uint8)
    l3_y = np.full((target_h, target_w), 255, dtype=np.uint8)
    l3_k = np.full((target_h, target_w), 255, dtype=np.uint8)

    # Apply gold ink where gold_alpha > 0
    g_active = gold_alpha > 0
    l3_c[g_active] = gold_ink_c[g_active]
    l3_m[g_active] = gold_ink_m[g_active]
    l3_y[g_active] = gold_ink_y[g_active]
    l3_k[g_active] = gold_ink_k[g_active]

    layer03 = nested_layers.Image(
        name='03_Print_Antique_Gold_CMYK',
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=0, left=0, bottom=target_h, right=target_w,
        channels={-1: gold_alpha, 0: l3_c, 1: l3_m, 2: l3_y, 3: l3_k}
    )

    print("Synthesizing CMYK Channels for Layer 04 (Rose Copper Foil)...")
    copper_ink_c = np.clip(230.0 - 25.0 * darkness, 0, 255).astype(np.uint8)
    copper_ink_m = np.clip(123.0 - 51.0 * darkness, 0, 255).astype(np.uint8)
    copper_ink_y = np.clip(102.0 - 46.0 * darkness, 0, 255).astype(np.uint8)
    copper_ink_k = np.clip(242.0 - 46.0 * darkness, 0, 255).astype(np.uint8)

    l4_c = np.full((target_h, target_w), 255, dtype=np.uint8)
    l4_m = np.full((target_h, target_w), 255, dtype=np.uint8)
    l4_y = np.full((target_h, target_w), 255, dtype=np.uint8)
    l4_k = np.full((target_h, target_w), 255, dtype=np.uint8)

    c_active = copper_alpha > 0
    l4_c[c_active] = copper_ink_c[c_active]
    l4_m[c_active] = copper_ink_m[c_active]
    l4_y[c_active] = copper_ink_y[c_active]
    l4_k[c_active] = copper_ink_k[c_active]

    layer04 = nested_layers.Image(
        name='04_Print_Rose_Copper_CMYK',
        visible=True,
        opacity=255,
        blend_mode=enums.BlendMode.normal,
        top=0, left=0, bottom=target_h, right=target_w,
        channels={-1: copper_alpha, 0: l4_c, 1: l4_m, 2: l4_y, 3: l4_k}
    )

    # 7. Synthesize Section 5 Master Composite
    print("Compositing Section 5 CMYK preview...")
    comp_c = l1_c.copy()
    comp_m = l1_m.copy()
    comp_y = l1_y.copy()
    comp_k = l1_k.copy()

    # Overlay gold
    comp_c[g_active] = gold_ink_c[g_active]
    comp_m[g_active] = gold_ink_m[g_active]
    comp_y[g_active] = gold_ink_y[g_active]
    comp_k[g_active] = gold_ink_k[g_active]

    # Overlay copper
    comp_c[c_active] = copper_ink_c[c_active]
    comp_m[c_active] = copper_ink_m[c_active]
    comp_y[c_active] = copper_ink_y[c_active]
    comp_k[c_active] = copper_ink_k[c_active]

    # Where diecut holes punch through, show base substrate
    is_hole = diecut_mask_hi == 0
    comp_c[is_hole] = l1_c[is_hole]
    comp_m[is_hole] = l1_m[is_hole]
    comp_y[is_hole] = l1_y[is_hole]
    comp_k[is_hole] = l1_k[is_hole]

    comp_channels = np.stack([comp_c, comp_m, comp_y, comp_k], axis=0)

    # 8. Assemble PSD Document
    print("Assembling PSD document with pytoshop...")
    layers_stack = [layer01, layer02, layer03, layer04]

    psd = nested_layers.nested_layers_to_psd(
        layers_stack,
        color_mode=enums.ColorMode.cmyk,
        compression=enums.Compression.raw
    )

    # Set Section 5 composite
    psd.image_data._channels = comp_channels

    # Embed 150 DPI ResolutionInfo Block (ID 1005 / 0x03ED)
    hRes = 150 << 16
    vRes = 150 << 16
    res_data = struct.pack('>IHH IHH', hRes, 1, 2, vRes, 1, 2)
    res_block = image_resources.GenericImageResourceBlock(
        resource_id=1005,
        name='',
        data=res_data
    )
    psd.image_resources.blocks.append(res_block)

    # Write output
    os.makedirs('outputs', exist_ok=True)
    out_psd_path = 'outputs/Swatch_Damask_Print_900x1600_CMYK.psd'
    print(f"Writing PSD to {out_psd_path}...")
    with open(out_psd_path, 'wb') as f:
        psd.write(f)
    
    file_size_mb = os.path.getsize(out_psd_path) / (1024 * 1024)
    print(f"SUCCESS: PSD generated in {time.time()-t0:.2f}s, file size: {file_size_mb:.2f} MB")

    # 9. Verification with psd-tools
    print("Verifying generated PSD with psd-tools...")
    psd_read = PSDImage.open(out_psd_path)
    print(f"Verified dimensions: {psd_read.width}x{psd_read.height} px")
    print(f"Verified color mode: {psd_read.color_mode} (CMYK=4)")
    print(f"Verified layer count: {len(psd_read)}")
    for i, lyr in enumerate(psd_read):
        clean_name = lyr.name.replace('\x00', '').strip()
        print(f"  Layer [{i+1}]: {clean_name}, size: {lyr.width}x{lyr.height}, visible: {lyr.is_visible()}")
    
    res_info = psd_read.image_resources.get_data(Resource.RESOLUTION_INFO)
    if res_info:
        dpi_h = res_info.horizontal / 65536.0
        dpi_v = res_info.vertical / 65536.0
        phys_w_mm = psd_read.width / dpi_h * 25.4
        phys_h_mm = psd_read.height / dpi_v * 25.4
        print(f"Verified Resolution: {dpi_h:.1f} DPI x {dpi_v:.1f} DPI")
        print(f"Verified Physical Size: {phys_w_mm:.1f} mm x {phys_h_mm:.1f} mm")

    # Export downscaled composite verification image
    print("Exporting composite and layer previews...")
    comp_pil = psd_read.composite()
    comp_rgb = comp_pil.convert('RGB')
    comp_rgb.thumbnail((1200, 2133))
    comp_rgb.save('intermediate/swatch/psd_composite_preview.jpg', quality=95)

    # 100% zoom crop
    crop_pil = comp_pil.crop((2000, 4000, 3200, 5200)).convert('RGB')
    crop_pil.save('intermediate/swatch/psd_100pct_zoom.jpg', quality=95)

    # Layer thumbnails
    for i in [3, 2, 1, 0]:
        lyr = psd_read[i]
        clean_name = lyr.name.replace('\x00', '').strip()
        pil_l = lyr.topil()
        pil_l.thumbnail((600, 1067))
        pil_l.convert('RGB').save(f'intermediate/swatch/layer_vis_{i}_{clean_name}.jpg', quality=92)

    print("PSD build and verification complete.")

if __name__ == '__main__':
    build_print_psd()
