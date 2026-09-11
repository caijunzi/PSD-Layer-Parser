import cv2  # noqa: F401 (其余 cv2 能力仍在用)
from engine.core.io_utils import imread_unicode, imwrite_unicode
import numpy as np
import os

def run_deocclusion(source_path="inputs/source_4000.jpg", mask_dir="intermediate", output_dir="intermediate"):
    os.makedirs(output_dir, exist_ok=True)
    src = imread_unicode(source_path)
    if src is None:
        raise FileNotFoundError(f"Cannot read {source_path}")
    
    h, w, c = src.shape
    
    m_ink = imread_unicode(os.path.join(mask_dir, "mask_ink_total.png"), 0)
    m_figures = imread_unicode(os.path.join(mask_dir, "mask_07_figures.png"), 0)
    m_trees_a = imread_unicode(os.path.join(mask_dir, "mask_05A_barren_trees.png"), 0)
    m_trees_b = imread_unicode(os.path.join(mask_dir, "mask_05B_water_trees.png"), 0)
    m_trees = cv2.bitwise_or(m_trees_a, m_trees_b)
    m_pavilion = imread_unicode(os.path.join(mask_dir, "mask_06_pavilion.png"), 0)
    m_rocks_a = imread_unicode(os.path.join(mask_dir, "mask_04A_foreground_cliffs.png"), 0)
    m_rocks_b = imread_unicode(os.path.join(mask_dir, "mask_04B_shorelines.png"), 0)
    m_rocks_c = imread_unicode(os.path.join(mask_dir, "mask_04C_solitary_rock.png"), 0)
    m_rocks = cv2.bitwise_or(m_rocks_a, cv2.bitwise_or(m_rocks_b, m_rocks_c))

    # 1. 2.5D 解闭环：人物遮挡的草堂坐榻与立柱
    print("[Phase 3] 1. 2.5D 解闭环：补全被人物遮挡的草堂坐榻...")
    fig_dil = cv2.dilate(m_figures, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    pav_covered = cv2.bitwise_and(fig_dil, m_pavilion)
    inpainted_pavilion = cv2.inpaint(src, pav_covered, 7, cv2.INPAINT_NS)
    imwrite_unicode(os.path.join(output_dir, "inpainted_pavilion_4k.png"), inpainted_pavilion)

    # 2. 2.5D 解闭环：树干遮挡的山石皴纹
    print("[Phase 3] 2. 2.5D 解闭环：补全被枯树遮挡的山石纹理...")
    tree_dil = cv2.dilate(m_trees, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    rock_covered = cv2.bitwise_and(tree_dil, m_rocks)
    inpainted_rocks = cv2.inpaint(src, rock_covered, 7, cv2.INPAINT_NS)
    imwrite_unicode(os.path.join(output_dir, "inpainted_rocks_4k.png"), inpainted_rocks)

    # 3. 重构博物馆级纯净金箔大底板 (Gold_Base_Clean)
    # 基于六曲屏风物理结构：第 1 曲 (Panel 1) 为未作画之纯金箔地，含完整金箔方格肌理与自然风化包浆
    # 按屏风 6 扇结构依次进行光照场校准与金箔方格平铺，彻底杜绝任何油墨残斑或模糊水渍
    print("[Phase 3] 3. 正在重构全画幅纯净金箔底板 (Gold_Base_Clean)...")
    inner_t, inner_b = 238, 1745
    inner_l, inner_r = 235, 3745
    w_panel = (inner_r - inner_l) / 6.0 # 585.0

    p1_x0 = int(inner_l)
    p1_x1 = int(inner_l + w_panel)
    panel_1_gold = src[inner_t:inner_b, p1_x0:p1_x1].copy()

    gold_base = src.copy()

    for i in range(6):
        px0 = int(inner_l + i * w_panel)
        px1 = int(inner_l + (i + 1) * w_panel)
        p_w = px1 - px0
        
        # 尺寸对齐
        p_gold_slice = cv2.resize(panel_1_gold, (p_w, inner_b - inner_t))
        
        if i == 0:
            # 第 1 扇本身为 100% 原始金箔地
            continue
            
        # 计算第 i 扇与第 1 扇在顶部纯金地 (Y: 260..380) 的光照差异
        top_curr = src[inner_t+20:inner_t+140, px0+20:px1-20]
        top_p1 = src[inner_t+20:inner_t+140, p1_x0+20:p1_x1-20]
        
        diff = np.mean(top_curr, axis=(0, 1)) - np.mean(top_p1, axis=(0, 1))
        p_gold_adj = np.clip(p_gold_slice.astype(np.float32) + diff, 0, 255).astype(np.uint8)
        
        # 获取该扇区域的墨迹掩模
        panel_ink = m_ink[inner_t:inner_b, px0:px1]
        panel_ink_dil = cv2.dilate(panel_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        alpha_p = cv2.GaussianBlur(panel_ink_dil.astype(np.float32) / 255.0, (25, 25), 8)[:, :, None]
        
        curr_slice = src[inner_t:inner_b, px0:px1]
        blended_slice = curr_slice.astype(np.float32) * (1.0 - alpha_p) + p_gold_adj.astype(np.float32) * alpha_p
        gold_base[inner_t:inner_b, px0:px1] = np.clip(blended_slice, 0, 255).astype(np.uint8)

    imwrite_unicode(os.path.join(output_dir, "gold_base_clean_4k.png"), gold_base)

    # 4. 提取金箔方格网格肌理
    gray_gold = cv2.cvtColor(gold_base, cv2.COLOR_BGR2GRAY)
    foil_blur = cv2.GaussianBlur(gray_gold, (31, 31), 10)
    foil_texture = cv2.subtract(gray_gold, foil_blur)
    foil_norm = cv2.normalize(foil_texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imwrite_unicode(os.path.join(output_dir, "gold_foil_grid_4k.png"), foil_norm)

    print("[Phase 3 Complete] 纯净金地底板与 2.5D 解闭环修复完成。")

if __name__ == "__main__":
    run_deocclusion()

