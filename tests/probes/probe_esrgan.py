# -*- coding: utf-8 -*-
"""探针：RealESRGAN 神经超分相对 Lanczos 是否有实质增益。

问题：模型加载并跑完 45 个切片，不等于"起作用"。若输出与 Lanczos 插值几乎相同，
则模型只是白跑（耗时换不来质量）。用客观指标判定：
  - 高频能量（Laplacian 方差）：越大越锐利，神经超分应显著高于双三次/Lanczos
  - 与源图的往返保真：超分后再降采样回原尺寸，与源图比较 MAE，越大表示偏离越多
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.providers.realesrgan_provider import RealESRGANProvider

CROP = 768  # 取源图中央 768x768 做超分对比（4x -> 3072x3072）


def high_freq_energy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    src_full = cv2.imread("inputs/source_4000.jpg")
    h, w = src_full.shape[:2]
    y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    src = src_full[y0:y0 + CROP, x0:x0 + CROP].copy()
    print(f"输入裁剪: {src.shape[1]}x{src.shape[0]} -> 目标 4x = {CROP*4}x{CROP*4}")

    tgt_w, tgt_h = CROP * 4, CROP * 4

    # 参考基线：Lanczos4
    t0 = time.time()
    lanczos = cv2.resize(src, (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
    t_lanczos = time.time() - t0

    # 神经超分
    provider = RealESRGANProvider(preferred_device="GPU.1", profile_name="robust_performance",
                                  tile_size=512, tile_pad=32)
    print(f"[probe] RealESRGAN backend = {provider.backend}")
    t0 = time.time()
    neural = provider.upscale(src, target_w=tgt_w, target_h=tgt_h)
    t_neural = time.time() - t0
    print(f"[probe] 神经输出 shape = {neural.shape}")

    if neural.shape[:2] != (tgt_h, tgt_w):
        neural = cv2.resize(neural, (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
        print("[probe] 尺寸不符，已重采样对齐后再比较")

    print()
    print(f"{'指标':<34}{'Lanczos4':>14}{'RealESRGAN':>14}{'神经/插值':>12}")
    print("-" * 76)

    hf_l, hf_n = high_freq_energy(lanczos), high_freq_energy(neural)
    print(f"{'高频能量 (Laplacian var)':<34}{hf_l:>14.1f}{hf_n:>14.1f}{hf_n/max(hf_l,1e-9):>11.2f}x")

    # 往返保真：超分 -> 降回原尺寸 vs 源图
    rt_l = cv2.resize(lanczos, (CROP, CROP), interpolation=cv2.INTER_AREA)
    rt_n = cv2.resize(neural, (CROP, CROP), interpolation=cv2.INTER_AREA)
    mae_l = float(np.mean(cv2.absdiff(rt_l, src)))
    mae_n = float(np.mean(cv2.absdiff(rt_n, src)))
    print(f"{'往返 MAE (超分->降回 vs 源)':<34}{mae_l:>14.3f}{mae_n:>14.3f}{mae_n/max(mae_l,1e-9):>11.2f}x")

    # 与 Lanczos 的差异幅度：若接近 0 说明神经路径没带来变化
    diff = float(np.mean(cv2.absdiff(neural, lanczos)))
    print(f"{'神经输出 vs Lanczos 平均绝对差':<34}{'-':>14}{diff:>14.3f}{'-':>12}")
    print(f"{'耗时 (秒)':<34}{t_lanczos:>14.3f}{t_neural:>14.3f}{t_neural/max(t_lanczos,1e-9):>11.1f}x")

    print()
    verdict = []
    if hf_n > hf_l * 1.5:
        verdict.append("神经超分显著更锐利（高频能量高 50%+）→ 模型确实起了作用")
    elif hf_n > hf_l * 1.1:
        verdict.append("神经超分略锐利")
    else:
        verdict.append("⚠️ 神经超分与 Lanczos 锐度相当 → 模型增益可疑")
    if diff < 1.0:
        verdict.append("⚠️ 与 Lanczos 差异极小 → 近乎等价，模型可能是白跑")
    for v in verdict:
        print(f"  判定：{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
