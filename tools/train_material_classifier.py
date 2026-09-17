# -*- coding: utf-8 -*-
"""T1 材质家族分类器训练/评测管线（中期学习能力的核心工具）。

用法：
  # 1) 只做语料校验（每族数量、规则判别器不一致清单）——投喂阶段反复跑
  python tools/train_material_classifier.py --validate-only

  # 2) 训练 + 留出集评测 + 与规则判别器同集对比
  python tools/train_material_classifier.py --train

  # 3) 训练并落盘新模型（默认只评测不落盘）
  python tools/train_material_classifier.py --train --save

语料：inputs_training/<家族>/<图片>（见 engine/adaptive/material_taxonomy.json）。
依赖：numpy / scikit-learn（GradientBoosting；本项目 Python 3.12 环境已含 sklearn）。
产出（--save 时）：engine/adaptive/material_clf_v2.pkl
  {model, feature_names, classes_, taxonomy_version, train_hash, report}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "inputs_training"
TAXONOMY = ROOT / "engine" / "adaptive" / "material_taxonomy.json"
MODEL_OUT = ROOT / "engine" / "adaptive" / "material_clf_v2.pkl"
TARGET_PER_FAMILY = 15          # 与 taxonomy.target_per_family 一致的门槛提示
HOLDOUT_RATIO = 0.2
RANDOM_STATE = 42


def _load_taxonomy() -> dict:
    return json.loads(TAXONOMY.read_text(encoding="utf-8"))


def scan_corpus() -> dict[str, list[Path]]:
    """扫描 inputs_training/<家族>/*.jpg|png，返回 家族→文件列表。"""
    if not TRAIN_DIR.is_dir():
        raise SystemExit(f"训练目录不存在: {TRAIN_DIR}（请按 家族/图片 结构投喂）")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out: dict[str, list[Path]] = {}
    for fam_dir in sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir()):
        files = [f for f in sorted(fam_dir.iterdir()) if f.suffix.lower() in exts]
        if files:
            out[fam_dir.name] = files
    return out


def _rule_family(img_bgr: np.ndarray) -> tuple[str, float]:
    """现有规则判别器（作为对比基线）。"""
    from engine.adaptive.fingerprint import extract_fingerprint
    from engine.adaptive.material_classifier import classify_material_family

    fam, conf = classify_material_family(extract_fingerprint(img_bgr))
    return fam, conf


def validate(corpus: dict[str, list[Path]]) -> bool:
    tax = _load_taxonomy()
    target = tax.get("target_per_family", TARGET_PER_FAMILY)
    ok = True
    print("=== 语料校验 ===")
    known = {f["id"] for f in tax["families"]}
    for fam in sorted(corpus):
        n = len(corpus[fam])
        flag = "✅" if n >= target else ("⚠️ " if n >= target // 2 else "❌")
        if n < target:
            ok = False
        print(f"  {flag} {fam}: {n} 张（目标 {target}）")
    for fam in sorted(known - set(corpus)):
        print(f"  ❌ {fam}: 0 张（taxonomy 已定义但语料缺失）")
        ok = False
    return ok


def build_dataset(corpus: dict[str, list[Path]]):
    """提取指纹特征（84 维原始向量），返回 X/y/文件清单 + 规则判别器对照。"""
    from engine.adaptive.fingerprint import extract_fingerprint
    from engine.core.io_utils import imread_unicode

    X, y, files, rule = [], [], [], []
    for fam, files_i in corpus.items():
        for p in files_i:
            img = imread_unicode(str(p))  # 语料文件名含中文：cv2.imread 会静默返回 None
            if img is None:
                raise RuntimeError(f"语料读取失败: {p}")  # 显式失败，绝不静默跳过
            fp = extract_fingerprint(img)
            # 直接使用指纹内的完整 84 维原始特征向量（与 PCA/CBR 同源，维度最全）
            vec = np.asarray(fp["raw_features"], dtype=np.float64)
            X.append(vec)
            y.append(fam)
            files.append(p.name)
            rule.append(_rule_family(img)[0])
    return np.asarray(X, dtype=np.float64), y, files, rule


def fingerprint_vec(fp: dict) -> np.ndarray:
    """把指纹 dict 拍平成稳定顺序的一维向量（与 trainer 版本绑定，落盘记录键序）。"""
    keys = sorted(fp.keys())
    vals: list[float] = []
    names: list[str] = []
    for k in keys:
        v = fp[k]
        if isinstance(v, dict):
            for sk in sorted(v.keys()):
                sv = v[sk]
                if isinstance(sv, (int, float)):
                    vals.append(float(sv))
                    names.append(f"{k}.{sk}")
                elif isinstance(sv, (list, tuple)) and sv and isinstance(sv[0], (int, float)):
                    for i, x in enumerate(sv):
                        vals.append(float(x))
                        names.append(f"{k}.{sk}[{i}]")
        elif isinstance(v, (int, float)):
            vals.append(float(v))
            names.append(k)
    fingerprint_vec.last_names = names  # 供落盘记录特征键序
    return np.asarray(vals, dtype=np.float64)


def train(X, y) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from collections import Counter as _C
    strat = y if min(_C(y).values()) >= 2 else None  # 单样本族无法分层 → 退化为随机切分
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=HOLDOUT_RATIO,
                                          random_state=RANDOM_STATE, stratify=strat)
    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    print("=== 留出集评测（GBoost，holdout %.0f%%） ===" % (HOLDOUT_RATIO * 100))
    print(classification_report(yte, yp, digits=3, zero_division=0))
    print("混淆矩阵 classes:", sorted(set(y)))
    print(confusion_matrix(yte, yp, labels=sorted(set(y))))
    acc = float((yp == np.asarray(yte)).mean())
    return {"model": clf, "holdout_acc": acc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    corpus = scan_corpus()
    ok = validate(corpus)
    if args.validate_only or not args.train:
        return 0 if ok else 1

    X, y, files, rule = build_dataset(corpus)
    print(f"\n样本总数: {len(y)}（特征 {X.shape[1]} 维）")
    counts = Counter(y)
    if len(set(y)) < 2 or min(counts.values()) < 5:
        print("⚠️ 每族样本不足 5 张，训练结果仅供参考；建议按校验清单继续补样本。")

    import pickle

    from sklearn.metrics import accuracy_score

    res = train(X, y)
    clf = res["model"]

    # 与规则判别器同集对比（新模型必须在**全部数据**上不劣于规则版才建议上线）
    rule_pred = np.asarray(rule)
    clf_pred = clf.predict(X)
    rule_acc = accuracy_score(y, rule_pred)
    clf_acc = accuracy_score(y, clf_pred)
    print("\n=== 与规则判别器同集对比 ===")
    print(f"  规则 v1 acc={rule_acc:.3f} | GBoost acc={clf_acc:.3f} "
          f"（{'✅ 更优' if clf_acc > rule_acc else '⚠️ 未超过，不建议上线'}）")

    if args.save:
        payload = {
            "model": clf,
            "classes_": sorted(set(y)),
            "feature_names": getattr(fingerprint_vec, "last_names", []),
            "taxonomy_version": _load_taxonomy().get("version"),
            "train_hash": hashlib.sha1(
                json.dumps(sorted(map(str, files))).encode()).hexdigest()[:12],
            "report": {"holdout_acc": res["holdout_acc"], "full_acc": clf_acc,
                       "rule_acc": rule_acc, "n_samples": len(y)},
        }
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n模型已落盘: {MODEL_OUT}")
    print("\n注意：模型只影响**推荐**，产物链路不变；上线前需跑金标准回归守卫。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
