import os
import cv2
import numpy as np

def build_semantic_masks(source_path="inputs/source_4000.jpg", output_dir="intermediate"):
    os.makedirs(output_dir, exist_ok=True)
    src = cv2.imread(source_path)
    if src is None:
        raise FileNotFoundError(f"Cannot read {source_path}")
    
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # 翻拍有效画心边界: inner_t=238, inner_b=1745, inner_l=235, inner_r=3745
    inner_t, inner_b, inner_l, inner_r = 238, 1745, 235, 3745
    
    active_ink = np.zeros((h, w), dtype=bool)
    active_ink[inner_t:inner_b, inner_l:inner_r] = (gray[inner_t:inner_b, inner_l:inner_r] < 172)
    
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    y_grid, x_grid = np.indices((h, w))

    # ==========================================
    # 1. Level 09A: Cinnabar Seal (朱砂印章 "長澤魚")
    # ==========================================
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    m_red = ((hsv[:,:,0] < 16) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] > 40) & (lab[:,:,1] > 138)
    seal_roi = np.zeros((h, w), dtype=bool)
    seal_roi[415:495, 3440:3520] = True
    m_09a = m_red & seal_roi
    m_09a = cv2.morphologyEx(m_09a.astype(np.uint8)*255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    cv2.imwrite(os.path.join(output_dir, "mask_09A_seal.png"), (m_09a * 255).astype(np.uint8))
    print(f"[Phase 2] 09A 朱砂印章: {np.count_nonzero(m_09a)} 像素")

    # ==========================================
    # 2. Level 09B: Calligraphy Inscription ("平安芦雪寫", 3 列完整款识)
    # ==========================================
    # 顶部 Y>=268 彻底避开外框织锦边缘投影阴影，X: 3420..3630, Y: 268..510
    callig_roi = np.zeros((h, w), dtype=bool)
    callig_roi[268:510, 3420:3630] = True
    m_09b_raw = active_ink & callig_roi & (~m_09a)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_09b_raw.astype(np.uint8))
    m_09b = np.zeros_like(m_09b_raw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 6: # 滤除背景微小金箔噪点
            m_09b |= (labels == i)
    cv2.imwrite(os.path.join(output_dir, "mask_09B_calligraphy.png"), (m_09b * 255).astype(np.uint8))
    print(f"[Phase 2] 09B 题跋款识: {np.count_nonzero(m_09b)} 像素")

    # ==========================================
    # 3. Level 08: Geese Flock (17 只平沙落雁完整群落)
    # ==========================================
    geese_roi = np.zeros((h, w), dtype=bool)
    geese_roi[440:780, 820:1750] = True
    geese_roi[:, 1382:1394] = False # 避开第2道折缝
    m_08_raw = active_ink & geese_roi
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_08_raw.astype(np.uint8))
    m_08 = np.zeros_like(m_08_raw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            m_08 |= (labels == i)
    cv2.imwrite(os.path.join(output_dir, "mask_08_geese.png"), (m_08 * 255).astype(np.uint8))
    print(f"[Phase 2] 08 飞禽微物: {np.count_nonzero(m_08)} 像素")

    # ==========================================
    # 4. Level 07: Figures (高士与三位侍童、茶案)
    # ==========================================
    fig_poly = np.array([
        [2700, 920], [2740, 915], [2780, 930], [2830, 960],
        [2880, 970], [2920, 975], [2955, 990], [2960, 1050],
        [2955, 1120], [2890, 1125], [2850, 1145], [2770, 1150],
        [2710, 1140], [2685, 1080], [2685, 980]
    ], dtype=np.int32)
    fig_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fig_roi, [fig_poly], 255)
    m_07 = active_ink & (fig_roi > 0)
    cv2.imwrite(os.path.join(output_dir, "mask_07_figures.png"), (m_07 * 255).astype(np.uint8))
    print(f"[Phase 2] 07 点景人物: {np.count_nonzero(m_07)} 像素")

    # ==========================================
    # 5. Level 06: Architecture Pavilion (水榭草堂纯净建筑结构)
    # ==========================================
    pav_poly = np.array([
        [2680, 815], [2900, 680], [3050, 780], [3060, 950],
        [2985, 1050], [2985, 1220], [2720, 1220], [2720, 1140],
        [2680, 1120], [2680, 850]
    ], dtype=np.int32)
    pav_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(pav_roi, [pav_poly], 255)
    m_06 = active_ink & (pav_roi > 0)
    cv2.imwrite(os.path.join(output_dir, "mask_06_pavilion.png"), (m_06 * 255).astype(np.uint8))
    print(f"[Phase 2] 06 建筑陈设: {np.count_nonzero(m_06)} 像素")

    # ==========================================
    # 6. Level 04C: Solitary Water Rock (左侧水中独立孤石)
    # ==========================================
    rock_c_poly = np.array([
        [1000, 1280], [1420, 1280], [1460, 1550], [1000, 1550]
    ], dtype=np.int32)
    rock_c_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(rock_c_roi, [rock_c_poly], 255)
    m_04c = active_ink & (rock_c_roi > 0)
    cv2.imwrite(os.path.join(output_dir, "mask_04C_solitary_rock.png"), (m_04c * 255).astype(np.uint8))
    print(f"[Phase 2] 04C 水中孤石: {np.count_nonzero(m_04c)} 像素")

    # ==========================================
    # 7. Level 04B & 05B: Midground Sandspit & Water Trees (中景渚岸与渚上水木)
    # ==========================================
    spit_poly = np.array([
        [1450, 600], [2260, 600], [2260, 1500], [1450, 1500]
    ], dtype=np.int32)
    spit_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(spit_roi, [spit_poly], 255)
    spit_ink = active_ink & (spit_roi > 0) & (~m_04c) & (~m_08)

    ground_base_y = np.clip(1160 + (x_coords - 1450) * (1250 - 1160) / (2260 - 1450), 1160, 1280)
    m_04b = spit_ink & (y_grid >= ground_base_y[None, :])
    m_05b = spit_ink & (y_grid < ground_base_y[None, :])
    cv2.imwrite(os.path.join(output_dir, "mask_04B_shorelines.png"), (m_04b * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "mask_05B_water_trees.png"), (m_05b * 255).astype(np.uint8))
    print(f"[Phase 2] 04B 中景渚岸: {np.count_nonzero(m_04b)} 像素")
    print(f"[Phase 2] 05B 渚上水木: {np.count_nonzero(m_05b)} 像素")

    # ==========================================
    # 8. Level 04D: Distant Soft Mountain (右上角淡墨远山峰峦)
    # ==========================================
    dist_poly = np.array([
        [3460, 520], [3660, 460], [3745, 490], [3745, 780], [3520, 780]
    ], dtype=np.int32)
    dist_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(dist_mask, [dist_poly], 255)
    m_04d = (dist_mask > 0) & active_ink & (~m_09a) & (~m_09b)
    kernel_dist = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    m_04d = cv2.morphologyEx(m_04d.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel_dist) > 0
    m_04d = m_04d & (~m_09a) & (~m_09b) & (dist_mask > 0)
    cv2.imwrite(os.path.join(output_dir, "mask_04D_distant_mountain.png"), (m_04d * 255).astype(np.uint8))
    print(f"[Phase 2] 04D 淡墨远山: {np.count_nonzero(m_04d)} 像素")

    # ==========================================
    # 9. Level 05A: Foreground Barren Trees (枯木枝干)
    # & Level 04A: Foreground Dark Cliffs (连续完整山石叠嶂)
    # ==========================================
    # 真实有机分界线：
    # A. 主山圆丘 (X: 2950..3450):
    #    主山圆丘顶部天然弧线 ys = 770.0 + ((xs - 3210.0) / 220.0)**2 * 100.0
    #    弧线以上为刺入金天之枝桠 (05A)；弧线以下为连续丰润之山体 (04A)！
    # B. 左侧石壁 (X: 2280..2650):
    #    石壁顶面台地在 Y=1100，以上冲天枯木为 05A，以下斧劈石柱为 04A！
    # C. 草堂后方淡墨杂树 (X: 2650..2950, Y: 400..750) 归入 05A。
    # D. 右崖三角坡前杂树 (X: 3450..3745, Y: 600..1100) 归入 05A。

    dome_crest_y = 770.0 + ((x_coords - 3210.0) / 220.0)**2 * 100.0
    dome_crest_y = np.clip(dome_crest_y, 770, 950)

    trees_sky_roi = np.zeros((h, w), dtype=bool)
    trees_sky_roi |= (x_grid >= 2950) & (x_grid <= 3450) & (y_grid >= 350) & (y_grid < dome_crest_y[None, :])
    trees_sky_roi |= (x_grid >= 2280) & (x_grid <= 2650) & (y_grid >= 400) & (y_grid < 1100)
    trees_sky_roi |= (x_grid > 2650) & (x_grid < 2950) & (y_grid >= 400) & (y_grid < 750)
    trees_sky_roi |= (x_grid > 3450) & (x_grid <= 3745) & (y_grid >= 600) & (y_grid < 1100)

    m_05a = active_ink & trees_sky_roi & (~m_09a) & (~m_09b) & (~m_08) & (~m_06) & (~m_04d)
    cv2.imwrite(os.path.join(output_dir, "mask_05A_barren_trees.png"), (m_05a * 255).astype(np.uint8))
    print(f"[Phase 2] 05A 枯木树群: {np.count_nonzero(m_05a)} 像素")

    # 10. Level 04A: Foreground Dark Cliffs & Mountain Masses (连续完整山石叠嶂)
    cliffs_roi = np.zeros((h, w), dtype=bool)
    cliffs_roi[1250:1745, 2250:3745] = True
    cliffs_roi |= (x_grid >= 2280) & (x_grid <= 2650) & (y_grid >= 1100) & (y_grid <= 1745)
    cliffs_roi |= (x_grid > 2650) & (x_grid < 2950) & (y_grid >= 750) & (y_grid <= 1745)
    cliffs_roi |= (x_grid >= 2950) & (x_grid <= 3450) & (y_grid >= dome_crest_y[None, :]) & (y_grid <= 1745)
    cliffs_roi |= (x_grid > 3450) & (x_grid <= 3745) & (y_grid >= 750) & (y_grid <= 1745)

    m_04a = active_ink & cliffs_roi & (~m_05a) & (~m_09a) & (~m_09b) & (~m_08) & (~m_04d)
    cv2.imwrite(os.path.join(output_dir, "mask_04A_foreground_cliffs.png"), (m_04a * 255).astype(np.uint8))
    print(f"[Phase 2] 04A 山石叠嶂: {np.count_nonzero(m_04a)} 像素")

    # ==========================================
    # 11. Level 03: Water Ripples (所有水面波澜)
    # ==========================================
    m_03 = active_ink & (x_coords[None, :] < 2260) & (~m_09a) & (~m_09b) & (~m_08) & (~m_04c) & (~m_04b) & (~m_05b)
    cv2.imwrite(os.path.join(output_dir, "mask_03_water_ripples.png"), (m_03 * 255).astype(np.uint8))
    print(f"[Phase 2] 03 水流微澜: {np.count_nonzero(m_03)} 像素")

    # 保存总墨迹掩模
    m_ink_total = m_09a | m_09b | m_08 | m_07 | m_06 | m_05a | m_05b | m_04a | m_04b | m_04c | m_04d | m_03
    cv2.imwrite(os.path.join(output_dir, "mask_ink_total.png"), (m_ink_total * 255).astype(np.uint8))
    print(f"[Phase 2 Complete] 所有 12 项对象级掩模全部生成完毕，总墨迹像素: {np.count_nonzero(m_ink_total)}。")

if __name__ == "__main__":
    build_semantic_masks()

