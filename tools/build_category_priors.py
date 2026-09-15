"""从真实 episode 检出统计生成 category_priors（形状/面积先验）。

背景（2026-09-16）：`category_priors` 表长期为空 → `inherit_priors()` 只能优雅跳过，
类目继承不到任何形状/面积先验。本脚本从**真实跑批产生的 episode 检出**统计先验，
不臆造、不写未经验证的 seed。

统计口径（检测框为归一化 [cx, cy, w, h]）：
  - area_budget_min / max : 框面积 (w*h) 的 p10 / p90（留余量，避免过拟合极端值）
  - elongation_mean / std : 长宽比 max(w,h)/min(w,h) 的均值 / 标准差
  - compactness_mean / std: π·w·h/(w+h)²（矩形紧凑度，正方形≈0.785，细长条→0）
  - n_components_mode     : 单次运行该类目实例数的众数
  - n_samples             : 参与统计的检测框总数（真实统计，非继承）

安全说明：当前引擎的面积预算走 `grounded_sam_provider.area_budget_for`（硬编码关键词表），
**不读本表**，因此写入 priors 不会改变产物/破坏字节可复现；本表供类目继承与后续消费使用。

用法：
    python tools/build_category_priors.py --dry-run          # 只看统计，不写库
    python tools/build_category_priors.py                    # 统计并写入（自动备份 DB）
    python tools/build_category_priors.py --min-samples 5
"""
import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EPISODES = sorted((ROOT / "outputs" / "adaptive-e2e-20260915-rerun").glob("case-*/episodes.jsonl"))
DEFAULT_DB = ROOT / "webui" / "data" / "adaptive_semantics.db"


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n if n > 1 else 0.0
    return m, var ** 0.5


def collect(episode_files):
    """聚合 {category_id: {"areas":[], "elong":[], "compact":[], "counts":[]}}"""
    import math
    agg = defaultdict(lambda: {"areas": [], "elong": [], "compact": [], "counts": []})
    n_eps = 0
    for f in episode_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_eps += 1
            dets = rec.get("dino_detections") or rec.get("detections") or []
            per_cat = defaultdict(int)
            for d in dets:
                cid = d.get("category_id") or d.get("layer_name")
                if not cid:
                    continue
                boxes = d.get("boxes") or []
                if not boxes:
                    continue
                per_cat[cid] += len(boxes)
                for b in boxes:
                    if not (isinstance(b, (list, tuple)) and len(b) >= 4):
                        continue
                    try:
                        w, h = float(b[2]), float(b[3])
                    except (TypeError, ValueError):
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    a = w * h
                    agg[cid]["areas"].append(a)
                    agg[cid]["elong"].append(max(w, h) / min(w, h))
                    agg[cid]["compact"].append(math.pi * w * h / ((w + h) ** 2))
            for cid, c in per_cat.items():
                agg[cid]["counts"].append(c)
    return agg, n_eps


def rollup_to_ancestors(agg, db_path):
    """把每个类目的检出统计**上卷到其所有祖先**（二级/根）。

    为何需要：episode 里的 category_id 都是三级具体类目，若只写三级，
    二级父类仍无先验 → `inherit_priors()` 依旧只能跳过。
    上卷后二级类目拥有「其所有子类检出」的真实统计，继承才真正可用。
    上卷数据仍是真实检出，不是臆造（n_samples 为累计真实框数）。
    """
    try:
        conn = sqlite3.connect(str(db_path))
        parent = {r[0]: r[1] for r in conn.execute(
            "SELECT id, parent_id FROM categories").fetchall()}
        conn.close()
    except Exception as e:
        print(f"[WARN] 读取类目树失败，跳过上卷: {e}")
        return agg

    out = defaultdict(lambda: {"areas": [], "elong": [], "compact": [], "counts": []})
    for cid, st in agg.items():
        targets = [cid]
        p, seen = parent.get(cid), {cid}
        while p and p not in seen:
            seen.add(p)
            targets.append(p)
            p = parent.get(p)
        for t in targets:
            for key in ("areas", "elong", "compact", "counts"):
                out[t][key].extend(st[key])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从真实 episode 统计生成 category_priors")
    ap.add_argument("--episode", action="append", default=None,
                    help="episode JSONL（可多次指定）；默认取最新跑批目录的 case-*/episodes.jsonl")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="目标 SQLite 库")
    ap.add_argument("--min-samples", type=int, default=3,
                    help="少于该检测框数的类目不写入（默认 3，避免小样本噪声）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写库")
    ap.add_argument("--no-rollup", action="store_true",
                    help="不做父子关系上卷（默认上卷，否则二级类目仍无先验、继承不可用）")
    args = ap.parse_args()

    files = [Path(p) for p in args.episode] if args.episode else DEFAULT_EPISODES
    files = [f for f in files if f.is_file()]
    if not files:
        print("[ERROR] 未找到 episode 文件")
        return 1
    print(f"[INFO] episode 文件 {len(files)} 个")

    agg, n_eps = collect(files)
    print(f"[INFO] 解析 episode {n_eps} 条，涉及类目 {len(agg)} 个")
    if not args.no_rollup:
        agg = rollup_to_ancestors(agg, args.db)
        print(f"[INFO] 已按父子关系上卷到祖先，类目数 -> {len(agg)}")

    print()
    rows = []
    print(f"{'类目':<34} {'n':>4} {'area_min':>9} {'area_max':>9} {'elong':>7} {'compact':>8}")
    print("-" * 78)
    for cid, st in sorted(agg.items(), key=lambda kv: -len(kv[1]["areas"])):
        n = len(st["areas"])
        if n < args.min_samples:
            continue
        amin, amax = _pct(st["areas"], 10), _pct(st["areas"], 90)
        emean, estd = _mean_std(st["elong"])
        cmean, cstd = _mean_std(st["compact"])
        # 实例数众数
        counts = st["counts"]
        mode = max(set(counts), key=counts.count) if counts else None
        rows.append((cid, amin, amax, emean, estd, cmean, cstd, mode, n))
        print(f"{str(cid)[:33]:<34} {n:>4} {amin:>9.5f} {amax:>9.5f} "
              f"{(emean or 0):>7.2f} {(cmean or 0):>8.3f}")

    print(f"\n[INFO] 满足 min_samples>={args.min_samples} 的类目 {len(rows)} 个")
    if args.dry_run:
        print("[DRY-RUN] 未写库")
        return 0
    if not rows:
        print("[WARN] 无满足条件的类目，不写库")
        return 0

    db = Path(args.db)
    if not db.is_file():
        print(f"[ERROR] 数据库不存在: {db}")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = db.with_suffix(db.suffix + f".bak-priors-{stamp}")
    shutil.copy(db, bak)
    print(f"[OK] 已备份数据库 -> {bak.name}")

    conn = sqlite3.connect(str(db))
    try:
        has = conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name='category_priors'").fetchone()
        if not has:
            print("[ERROR] category_priors 表不存在")
            return 1
        # 仅写入库中**已存在**的类目（外键约束，不臆造 ID）
        existing = {r[0] for r in conn.execute("SELECT id FROM categories").fetchall()}
        now = int(datetime.now().timestamp())
        written = skipped = 0
        for cid, amin, amax, emean, estd, cmean, cstd, mode, n in rows:
            if cid not in existing:
                skipped += 1
                continue
            conn.execute(
                "INSERT OR REPLACE INTO category_priors (category_id, area_budget_min, "
                "area_budget_max, elongation_mean, elongation_std, compactness_mean, "
                "compactness_std, n_components_mode, n_samples, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, amin, amax, emean, estd, cmean, cstd, mode, n, now))
            written += 1
        conn.commit()
        print(f"[OK] 写入 {written} 条先验；跳过 {skipped} 条（库中无此类目，不臆造）")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
