# -*- coding: utf-8 -*-
"""RK-16 复现性验证：同 seed 跑两次 Step1（GroundingDINO + SAM2），
对比输出掩模的逐像素 hash 与耗时。

若 dict 完全相等 → 分割阶段可复现（PLATE 线「可复算」验收的前提达成）；
若不等 → 打印差异层名与首个差异像素，定位非确定性来源。
"""
import hashlib
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "run_universal_engine.py")

from run_universal_engine import _seed_everything, load_preset


def run_step1(seed: int):
    _seed_everything(seed)
    from engine.providers.grounded_sam_provider import GroundedSAMProvider

    preset = load_preset("japanese_screen_gold")
    src = cv2.imread("inputs/source_4000.jpg")
    gs = GroundedSAMProvider(preferred_device="GPU.1", preset_name="japanese_screen_gold")
    t0 = time.time()
    masks = gs.segment_objects(src, classes=preset.get("ai_semantic_classes"))
    dt = time.time() - t0
    hashed = {
        name: {
            "md5": hashlib.md5(m.tobytes()).hexdigest(),
            "px": int(np.count_nonzero(m)),
        }
        for name, m in masks.items()
    }
    return hashed, dt


def main() -> int:
    seed = 42
    print(f"[RK-16] seed={seed}，跑两次 Step1（GroundingDINO + SAM2, CPU）...\n")
    r1, t1 = run_step1(seed)
    r2, t2 = run_step1(seed)

    print(f"第一次: {t1:.2f}s  第二次: {t2:.2f}s  耗时差 {abs(t2-t1):.2f}s\n")

    keys = sorted(set(r1) | set(r2))
    same_keys = (sorted(r1) == sorted(r2))
    diffs = []
    for k in keys:
        h1 = r1.get(k, {}).get("md5")
        h2 = r2.get(k, {}).get("md5")
        if h1 != h2:
            diffs.append((k, r1.get(k, {}).get("px"), r2.get(k, {}).get("px")))

    print(f"图层键集一致: {same_keys}  层数: {len(keys)}")
    if not diffs and same_keys:
        print("✅ 两次运行掩模逐像素一致（RK-16 可复现性达成）")
        for k in keys:
            print(f"   {k[:40]:<42} px={r1[k]['px']}")
    else:
        print("❌ 存在差异层：")
        for k, p1, p2 in diffs:
            print(f"   {k[:40]:<42} px {p1} vs {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
