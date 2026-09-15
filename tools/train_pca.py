"""训练图级指纹的标准化/PCA 模型并持久化。

背景（2026-09-16）：`engine/adaptive/fingerprint.py` 的 PCA 长期是占位实现
（`_pca_model = "placeholder"`），`train_pca_from_dataset` 生产零调用，导致
embedding 实为「未标准化的原始特征截断/填充」——各特征量纲差异巨大
（edge_density≈0.1 vs 直方图计数≈1e3），CBR 余弦检索被大尺度特征主导。

用法：
    python tools/train_pca.py                      # 用 inputs/ 下全部图片训练
    python tools/train_pca.py --dir inputs --dir other
    python tools/train_pca.py --out path/to.pkl    # 指定模型输出位置

训练后模型默认写入 `engine/checkpoints/fingerprint_pca.pkl`；
生产运行时由 `_apply_pca` 首次调用自动加载（也可用 ULS_PCA_MODEL 覆盖路径）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.adaptive.fingerprint import (  # noqa: E402
    DEFAULT_PCA_PATH, _model_path, train_pca_from_dataset,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect(dirpath: Path) -> list:
    if not dirpath.is_dir():
        print(f"[WARN] 目录不存在，跳过: {dirpath}")
        return []
    return sorted(p for p in dirpath.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="训练指纹标准化/PCA 模型")
    ap.add_argument("--dir", action="append", default=None,
                    help="训练图片目录（可多次指定）；默认 inputs/")
    ap.add_argument("--out", default=None, help=f"模型输出路径（默认 {DEFAULT_PCA_PATH}）")
    ap.add_argument("--no-save", action="store_true", help="只训练不落盘（用于验证）")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir] if args.dir else [ROOT / "inputs"]
    paths = []
    for d in dirs:
        found = collect(d)
        print(f"[INFO] {d}: {len(found)} 张图片")
        paths.extend(found)

    if not paths:
        print("[ERROR] 未找到任何训练图片")
        return 1

    print(f"[INFO] 共 {len(paths)} 张训练图 -> {_model_path(args.out)}")
    ok = train_pca_from_dataset(paths, n_components=128,
                                save=not args.no_save, path=args.out)
    if not ok:
        print("[ERROR] 训练失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
