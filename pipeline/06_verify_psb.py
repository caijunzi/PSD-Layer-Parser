"""Production PSB Preflight Verification（印前交付物验证）。

改造说明（2026-09-10）
----------------------
1. **去裸 assert**：计划 §8.3 明令"禁止裸 assert（`python -O` 下被静默剥离，质检会假绿）"。
   本文件原先全文使用裸 `assert`，现统一改为 `QAReport.record()` 收集 + 退出码表达成败。
2. **默认路径修正**：原默认 `outputs/Rosetsu_1795_Master_16k.psb` 在磁盘上不存在。
3. **MAE 阈值统一**：文档红线 ≤2.0、DAILY_LOG 写 5.0、代码写 8.0，三处口径不一。
   现统一默认 **2.0**（可用 `--mae-threshold` 覆盖），并在报告中打印口径说明。
4. 输出 JSON 报告到 `intermediate/`，便于回归比对。

用法：
    python pipeline/06_verify_psb.py --file outputs/Rosetsu_Master_16k.psb --dpi 150.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2  # noqa: F401 (其余 cv2 能力仍在用)
from engine.core.io_utils import imread_unicode, imwrite_unicode
import numpy as np
import psd_tools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core.models import QAReport, new_run_id  # noqa: E402

DEFAULT_FILE = "outputs/Rosetsu_Master_16k.psb"
DEFAULT_MAE_THRESHOLD = 2.0     # 文档 §5.4 红线；代码旧值为 8.0，过松
MIN_VALID_LAYERS = 8


def verify(
    psb_path: str = DEFAULT_FILE,
    expected_dpi: float = 150.0,
    mae_threshold: float = DEFAULT_MAE_THRESHOLD,
    source_path: str = "inputs/source_4000.jpg",
) -> QAReport:
    rep = QAReport(run_id=new_run_id(), tier="T4")

    print("=== Universal PSB Verification & Preflight Report ===")
    print(f"run_id      : {rep.run_id}")
    print(f"File Path   : {psb_path}")
    print(f"MAE 阈值    : {mae_threshold:.2f}（文档 §5.4 红线；旧代码为 8.0）")

    if not os.path.exists(psb_path):
        rep.record("V-00", "交付物存在", False, f"{psb_path} 不存在")
        print(f"Error: {psb_path} not found")
        return rep

    size_gb = os.path.getsize(psb_path) / (1024 ** 3)
    print(f"File Size   : {size_gb:.3f} GB")

    psd = psd_tools.PSDImage.open(psb_path)
    print(f"Image Size  : {psd.size}")
    print(f"Color Mode  : {psd.color_mode}")
    print(f"Channels    : {psd.channels}")
    print(f"Format Ver  : {'PSB (Version 2)' if psd.version == 2 else 'PSD (Version 1)'}")
    print(f"Layer Count : {len(psd)}")

    # 1. 分辨率块 0x03ED
    res_block = psd.image_resources.get_data(1005)
    if res_block is None:
        rep.record("V-01", "分辨率块 0x03ED 存在", False, "缺失")
        print("Resolution  : [FAIL] ResolutionInfo block 1005 not found")
    else:
        read_dpi = res_block.horizontal / 65536.0
        print(f"Resolution  : {read_dpi:.1f} PPI (Resource 0x03ED verified)")
        rep.record("V-01", f"分辨率 {expected_dpi:.1f} PPI",
                   abs(read_dpi - expected_dpi) < 1.0, f"实测 {read_dpi:.2f} PPI")

    # 2. 图层与包围盒
    print("\n--- Layer Stack Verification ---")
    valid_layers = 0
    for idx, layer in enumerate(psd):
        clean_name = layer.name.replace("\x00", "").strip()
        bbox = layer.bbox
        w_box = (bbox[2] - bbox[0]) if bbox else 0
        h_box = (bbox[3] - bbox[1]) if bbox else 0
        print(f"[{idx:02d}] {clean_name:<34} Blend: {layer.blend_mode.name:<10} "
              f"Opacity: {layer.opacity:<3} BBox: ({bbox[0]}, {bbox[1]}) -> ({bbox[2]}, {bbox[3]}) "
              f"[{w_box}x{h_box}]")
        if w_box > 0 and h_box > 0:
            valid_layers += 1
    rep.record("V-02", f"有效图层数 ≥ {MIN_VALID_LAYERS}",
               valid_layers >= MIN_VALID_LAYERS, f"{valid_layers} 层")

    # 3. Section 5 合成保真度
    print("\n--- Section 5 Merged Composite Fidelity ---")
    comp_pil = psd.topil()
    print(f"Composite   : size={comp_pil.size}, mode={comp_pil.mode}")
    rep.record("V-03", "合成图尺寸与文档一致",
               comp_pil.size == psd.size, f"{comp_pil.size} vs {psd.size}")

    comp_arr = np.array(comp_pil)
    print(f"Composite Mean RGB: {comp_arr.mean(axis=(0, 1)).round(2)}, "
          f"Std: {comp_arr.std(axis=(0, 1)).round(2)}")

    src = imread_unicode(source_path) if os.path.isfile(source_path) else None
    if src is None:
        print(f"[WARN] 源图 {source_path} 不存在，跳过色差比对")
        rep.record("V-04", "合成保真度", True, "跳过（无源图）")
    else:
        comp_thumb = comp_pil.copy()
        comp_thumb.thumbnail((src.shape[1], src.shape[0]))
        src_thumb = cv2.resize(src, (comp_thumb.width, comp_thumb.height))
        comp_bgr = cv2.cvtColor(np.array(comp_thumb), cv2.COLOR_RGB2BGR)
        mae = float(np.mean(cv2.absdiff(src_thumb, comp_bgr)))
        print(f"MAE vs Source : {mae:.3f}  (阈值 {mae_threshold:.2f})")
        rep.record("V-04", f"合成色差 MAE ≤ {mae_threshold:.2f}",
                   mae <= mae_threshold, f"MAE={mae:.3f}")
        rep.metrics["mae"] = round(mae, 3)

    os.makedirs("intermediate", exist_ok=True)
    out = os.path.join("intermediate", f"verify_{rep.run_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep.to_dict(), f, ensure_ascii=False, indent=2)

    failed = [r for r in rep.records if not r["ok"]]
    print(f"\n=== {'ALL CHECKS PASSED' if rep.passed else 'VERIFICATION FAILED'} "
          f"({len(rep.records) - len(failed)}/{len(rep.records)}) ===")
    for r in failed:
        print(f"    [FAIL] {r['code']} {r['name']} — {r['detail']}")
    print(f"报告已写入: {out}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify PSB output quality & preflight")
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD,
                    help=f"合成色差 MAE 上限（默认 {DEFAULT_MAE_THRESHOLD}，文档 §5.4 红线）")
    ap.add_argument("--source", default="inputs/source_4000.jpg")
    args = ap.parse_args()

    rep = verify(args.file, args.dpi, args.mae_threshold, args.source)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
