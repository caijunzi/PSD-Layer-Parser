"""密度参数校准工具：对金标准图分析密度分布，推荐 density_band / floor 参数。

用法：
    python tools/calibrate_density_bands.py inputs/source_4000.jpg

输出：
- 全局密度分位数与直方图（saved to scratch/calib_density_hist.png）
- 若干候选区域的密度分布（默认按画幅九宫格，可用 --region x0,y0,x1,y1 追加）
- 每区域的推荐 (density_min, density_max)：取该区 p60~p95 作为"墨迹带"

定位：把"手调阈值"降级为"工具推荐 + 人工确认"（设计文档 §5）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.io_utils import imread_unicode  # noqa: E402
from engine.core.density_field import ink_density  # noqa: E402


def _grid_regions(nx: int = 3, ny: int = 3) -> list[tuple[str, tuple]]:
    out = []
    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix / nx, iy / ny
            out.append((f"grid_{ix}{iy}", (x0, y0, x0 + 1 / nx, y0 + 1 / ny)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--region", action="append", default=[],
                    help="追加归一化区域 name:x0,y0,x1,y1")
    ap.add_argument("--gold-percentile", type=float, default=88)
    ap.add_argument("--sigma", type=float, default=4.0)
    args = ap.parse_args()

    img = imread_unicode(args.image)
    if img is None:
        print(f"[calib] 读取失败: {args.image}")
        return 1
    h, w = img.shape[:2]
    D = ink_density(img, gold_percentile=args.gold_percentile, sigma=args.sigma)

    q = np.percentile(D, [50, 75, 90, 95, 99])
    print(f"[calib] {w}x{h}  密度分位 p50/75/90/95/99 = "
          + " ".join(f"{v:.3f}" for v in q))

    regions = _grid_regions()
    for spec in args.region:
        try:
            name, box = spec.split(":")
            x0, y0, x1, y1 = [float(v) for v in box.split(",")]
            regions.append((name, (x0, y0, x1, y1)))
        except Exception as e:
            print(f"[calib] 忽略非法 region '{spec}': {e}")

    print("\n[calib] 区域密度分布与推荐带（p60→p95 = 墨迹带）")
    print(f"{'region':>10} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} | {'min':>6} {'max':>6} {'cover%':>7}")
    for name, (x0, y0, x1, y1) in regions:
        sub = D[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if sub.size == 0:
            continue
        p50, p75, p90, p95 = np.percentile(sub, [50, 75, 90, 95])
        dmin, dmax = np.percentile(sub, 60), p95
        cov = float(((sub >= dmin) & (sub <= dmax)).mean()) * 100
        print(f"{name:>10} {p50:6.3f} {p75:6.3f} {p90:6.3f} {p95:6.3f} | "
              f"{dmin:6.3f} {dmax:6.3f} {cov:7.1f}")

    # 直方图图
    try:
        from PIL import Image, ImageDraw
        hist, edges = np.histogram(D, bins=64, range=(0, 1))
        hn = hist / max(hist.max(), 1)
        im = Image.new("RGB", (1024, 420), (245, 245, 245))
        dr = ImageDraw.Draw(im)
        for i in range(64):
            x0, x1 = int(i * 16), int((i + 1) * 16)
            dr.rectangle([x0, 409 - int(hn[i] * 380), x1, 409], fill=(90, 90, 220))
        out = ROOT / "scratch" / "calib_density_hist.png"
        im.save(out)
        print(f"\n[calib] 直方图: {out}")
    except Exception as e:
        print(f"[calib] 直方图跳过: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
