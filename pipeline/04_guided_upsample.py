import cv2  # noqa: F401 (其余 cv2 能力仍在用)
from engine.core.io_utils import imread_unicode, imwrite_unicode
import numpy as np
import os
import glob
import time

TARGET_W = 16000
TARGET_H = 7808

def guided_filter_banded(guide_16k, mask_16k, radius=8, eps=1e-3, num_bands=4):
    """内存受控的分幅导向滤波：在 16000x7808 画幅下峰值内存 < 2GB"""
    h, w = guide_16k.shape
    band_h = h // num_bands
    out = np.empty((h, w), dtype=np.uint8)
    
    pad = radius * 2
    for band_idx in range(num_bands):
        y0 = max(0, band_idx * band_h - pad)
        y1 = min(h, (band_idx + 1) * band_h + pad if band_idx < num_bands - 1 else h)
        
        g_band = guide_16k[y0:y1].astype(np.float32) / 255.0
        p_band = mask_16k[y0:y1].astype(np.float32) / 255.0
        
        mean_I = cv2.boxFilter(g_band, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(p_band, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(g_band * p_band, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(g_band * g_band, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        coeff_a = cov_Ip / (var_I + eps)
        coeff_b = mean_p - coeff_a * mean_I
        
        mean_a = cv2.boxFilter(coeff_a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(coeff_b, cv2.CV_32F, (radius, radius))
        
        q = mean_a * g_band + mean_b
        q_clip = np.clip(q * 255.0, 0, 255).astype(np.uint8)
        
        by0 = band_idx * band_h
        by1 = min(h, (band_idx + 1) * band_h if band_idx < num_bands - 1 else h)
        out[by0:by1] = q_clip[by0 - y0 : by0 - y0 + (by1 - by0)]
    return out

def run_guided_upsample():
    os.makedirs("masks_16k", exist_ok=True)
    os.makedirs("intermediate", exist_ok=True)

    # 1. 超分辨率生成 16K 主图
    source_16k_path = "intermediate/source_16k.jpg"
    if not os.path.exists(source_16k_path):
        print(f"[Phase 4] 正在将 source_4000 超分放大至 16000x7808 (Lanczos-4)...")
        src_4k = imread_unicode("inputs/source_4000.jpg")
        src_16k = cv2.resize(src_4k, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        imwrite_unicode(source_16k_path, src_16k, [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(f"[Phase 4] source_16k.jpg 生成完毕。")

    # 2. 重新超分生成 16K 纯净金地
    gold_16k_path = "intermediate/gold_base_clean_16k.jpg"
    print(f"[Phase 4] 正在将全新 gold_base_clean_4k 超分放大至 16000x7808...")
    gold_4k = imread_unicode("intermediate/gold_base_clean_4k.png")
    gold_16k = cv2.resize(gold_4k, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    imwrite_unicode(gold_16k_path, gold_16k, [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"[Phase 4] gold_base_clean_16k.jpg 生成完毕。")

    # 3. 加载 16K 灰度引导通道
    print("[Phase 4] 加载 16K 灰度引导通道...")
    src_16k = imread_unicode(source_16k_path)
    guide_16k = cv2.cvtColor(src_16k, cv2.COLOR_BGR2GRAY)
    del src_16k

    # 4. 批量导向滤波超采样所有 4K 蒙版
    mask_files = sorted(glob.glob("intermediate/mask_*.png"))
    print(f"[Phase 4] 开始批量执行 16K 导向滤波，共 {len(mask_files)} 个蒙版...")

    for mf in mask_files:
        fname = os.path.basename(mf)
        out_path = os.path.join("masks_16k", fname)
        
        if fname in ["mask_ink_total.png", "mask_painting_roi.png"]:
            continue
            
        t0 = time.time()
        m_4k = imread_unicode(mf, cv2.IMREAD_GRAYSCALE)
        m_16k_raw = cv2.resize(m_4k, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        
        if "seal" in fname or "geese" in fname or "calligraphy" in fname:
            m_16k_refined = guided_filter_banded(guide_16k, m_16k_raw, radius=4, eps=1e-3)
        elif "trees" in fname or "water" in fname:
            m_16k_refined = guided_filter_banded(guide_16k, m_16k_raw, radius=6, eps=1e-3)
        else:
            m_16k_refined = guided_filter_banded(guide_16k, m_16k_raw, radius=8, eps=1e-2)
            
        imwrite_unicode(out_path, m_16k_refined, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"[Phase 4] {fname} -> 16000x7808 导向滤波完成 ({time.time()-t0:.2f}s)")

    print("[Phase 4 Complete] 所有 16K 高精度导向蒙版全部重新生成完毕。")

if __name__ == "__main__":
    run_guided_upsample()
