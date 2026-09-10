import os
import gc
import time
import cv2
import numpy as np
from pytoshop import enums, core
from pytoshop.user import nested_layers

TARGET_W = 16000
TARGET_H = 7808

def assemble_master_psb(output_path="outputs/Rosetsu_1795_Master_16k.psb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t_start = time.time()

    print("[Phase 5] 1. 加载 16K 超分原图与纯净金底...")
    src_16k = cv2.imread("intermediate/source_16k.jpg")
    gold_16k = cv2.imread("intermediate/gold_base_clean_16k.jpg")
    
    if src_16k is None or gold_16k is None:
        raise FileNotFoundError("16K 源文件缺失")

    # 准备 Section 5 (Image Data Section) 预渲染合成图像通道
    # PSD/PSB 规范要求 channels 为 (num_channels, H, W) 的 uint8 numpy array, 顺序为 R, G, B
    print("[Phase 5] 2. 准备 Section 5 Merged Composite Image 通道...")
    comp_r = np.ascontiguousarray(src_16k[:, :, 2])
    comp_g = np.ascontiguousarray(src_16k[:, :, 1])
    comp_b = np.ascontiguousarray(src_16k[:, :, 0])
    comp_channels = np.stack([comp_r, comp_g, comp_b], axis=0)

    # 图层裁剪与构建辅助函数
    def create_layer_from_mask(name, mask_path, blend_mode=enums.BlendMode.normal, opacity=255, fixed_color=None):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"警告: 找不到掩模 {mask_path}，跳过")
            return None
        
        pts = cv2.findNonZero(m)
        if pts is None:
            print(f"掩模 {mask_path} 为空，跳过")
            return None
        
        rx, ry, rw, rh = cv2.boundingRect(pts)
        pad = 8
        x0 = max(0, rx - pad)
        y0 = max(0, ry - pad)
        x1 = min(TARGET_W, rx + rw + pad)
        y1 = min(TARGET_H, ry + rh + pad)

        crop_m = m[y0:y1, x0:x1]

        if fixed_color is not None:
            crop_r = np.full_like(crop_m, fixed_color[2], dtype=np.uint8)
            crop_g = np.full_like(crop_m, fixed_color[1], dtype=np.uint8)
            crop_b = np.full_like(crop_m, fixed_color[0], dtype=np.uint8)
        else:
            crop_src = src_16k[y0:y1, x0:x1]
            crop_b, crop_g, crop_r = cv2.split(crop_src)

        crop_a = crop_m

        layer = nested_layers.Image(
            name=name,
            color_mode=enums.ColorMode.rgb,
            blend_mode=blend_mode,
            opacity=opacity,
            top=y0,
            left=x0,
            bottom=y1,
            right=x1
        )
        layer.set_channel(enums.ColorChannel.red, crop_r)
        layer.set_channel(enums.ColorChannel.green, crop_g)
        layer.set_channel(enums.ColorChannel.blue, crop_b)
        layer.set_channel(enums.ColorChannel.transparency, crop_a)
        
        print(f"[Phase 5] 图层 [{name}] 构建完成: BBox=({x0}, {y0}) -> ({x1}, {y1}), 尺寸={x1-x0}x{y1-y0}")
        return layer

    # 构建各个图层
    # 注意：在 pytoshop.user.nested_layers.nested_layers_to_psd 中：
    # 接收顺序为 UI 从顶至底 (Top -> Bottom)，函数内部 [::-1] 翻转写入 PSD Record 0 至 Record -1
    layers_to_build = []

    # UI 从顶向下排序
    top_layers_config = [
        ("10A_Brocade_Outer_Frame", "masks_16k/mask_10A_frame.png", enums.BlendMode.normal, 255, None),
        ("10B_Panel_Fold_Seams", "masks_16k/mask_10B_seams.png", enums.BlendMode.multiply, 190, [25, 20, 15]),
        ("09A_Seal_Nagasawa_Gyo", "masks_16k/mask_09A_seal.png", enums.BlendMode.normal, 255, None),
        ("09B_Calligraphy_Inscription", "masks_16k/mask_09B_calligraphy.png", enums.BlendMode.normal, 255, None),
        ("08_Geese_Flock", "masks_16k/mask_08_geese.png", enums.BlendMode.normal, 255, None),
        ("07_Figures_Scholar_Attendant", "masks_16k/mask_07_figures.png", enums.BlendMode.normal, 255, None),
        ("06_Architecture_Pavilion", "masks_16k/mask_06_pavilion.png", enums.BlendMode.normal, 255, None),
        ("05A_Foreground_Barren_Trees", "masks_16k/mask_05A_barren_trees.png", enums.BlendMode.normal, 255, None),
        ("05B_Midground_Water_Trees", "masks_16k/mask_05B_water_trees.png", enums.BlendMode.normal, 255, None),
        ("04A_Foreground_Dark_Cliffs", "masks_16k/mask_04A_foreground_cliffs.png", enums.BlendMode.normal, 255, None),
        ("04D_Distant_Soft_Mountain", "masks_16k/mask_04D_distant_mountain.png", enums.BlendMode.normal, 255, None),
        ("04B_Midground_Shorelines", "masks_16k/mask_04B_shorelines.png", enums.BlendMode.normal, 255, None),
        ("04C_Solitary_Water_Rock", "masks_16k/mask_04C_solitary_rock.png", enums.BlendMode.normal, 255, None),
        ("03_Water_Ripples", "masks_16k/mask_03_water_ripples.png", enums.BlendMode.normal, 255, None)
    ]

    print("[Phase 5] 3. 按照 UI 自顶向下顺序依次构建各上层对象图层...")
    for name, m_path, b_mode, op, fix_col in top_layers_config:
        lyr = create_layer_from_mask(name, m_path, blend_mode=b_mode, opacity=op, fixed_color=fix_col)
        if lyr is not None:
            layers_to_build.append(lyr)

    # 底板图层 02_Gold_Base_Clean（最底层）
    print("[Phase 5] 4. 构建 Layer 02: 纯净金箔大底板 (Gold_Base_Clean) 作为底板...")
    b_g, g_g, r_g = cv2.split(gold_16k)
    layer_02_gold = nested_layers.Image(
        name="02_Gold_Base_Clean",
        color_mode=enums.ColorMode.rgb,
        blend_mode=enums.BlendMode.normal,
        opacity=255,
        top=0, left=0, bottom=TARGET_H, right=TARGET_W
    )
    layer_02_gold.set_channel(enums.ColorChannel.red, r_g)
    layer_02_gold.set_channel(enums.ColorChannel.green, g_g)
    layer_02_gold.set_channel(enums.ColorChannel.blue, b_g)
    layer_02_gold.set_channel(enums.ColorChannel.transparency, np.full((TARGET_H, TARGET_W), 255, dtype=np.uint8))
    layers_to_build.append(layer_02_gold)

    # 内存释放
    del gold_16k, b_g, g_g, r_g, src_16k
    gc.collect()

    print(f"[Phase 5] 5. 成功装配 {len(layers_to_build)} 个图层，正在编译大型 PSB 格式结构...")
    
    psd_doc = nested_layers.nested_layers_to_psd(
        layers=layers_to_build,
        color_mode=enums.ColorMode.rgb,
        version=enums.Version.version_2,
        compression=enums.Compression.raw,
        size=(TARGET_W, TARGET_H)
    )

    print("[Phase 5] 6. 注入 Section 5 (Image Data Section) 16K 真实预渲染合并画幅...")
    psd_doc.image_data = core.ImageData(
        channels=comp_channels,
        compression=enums.Compression.raw
    )

    print(f"[Phase 5] 7. 正在流式写入磁盘 -> {output_path} ...")
    with open(output_path, "wb") as fd:
        psd_doc.write(fd)

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    elapsed = time.time() - t_start
    print(f"[Phase 5 Complete] PSB 文件输出成功: {output_path} (体积: {file_size_gb:.2f} GB, 耗时: {elapsed:.1f} 秒)")
    return output_path

if __name__ == "__main__":
    assemble_master_psb()
