"""
Episode 归档模块（Stage 1.5）

一个 episode = 一次「图像 → 材质判别 → 类目选择 → 结果/反馈」的完整交互记录。
后续阶段（Stage 2+ 反馈学习 / Stage 3 自动进化）会消费这些 episode 来：
  - 统计材质 × 类目共现，校准亲和度；
  - 识别假阳性（被用户拒绝的类目）进入负样本库；
  - 触发词库版本迭代（category_versions）。

Stage 1 仅做「归档」：以 JSONL 追加写（带锁），不改动 SQLite 词库。
这样无需迁移 schema，且天然可追加、可回放。
"""
import json
import threading
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# 线程安全写（主流程与 Web 可能并发归档）
_lock = threading.Lock()

DEFAULT_EPISODE_PATH = "webui/data/adaptive_episodes.jsonl"


def _episode_id(material_family: str, image_path: Optional[str], ts: str) -> str:
    """生成稳定且可读的 episode id。"""
    base = f"{material_family}|{image_path or 'none'}|{ts}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"ep_{ts.replace(':', '').replace('-', '').replace('.', '')}_{digest}"


def archive_episode(
    material_family: str,
    category_ids: List[str],
    fingerprint: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
    outcome: str = "selected",
    confidence: float = 0.0,
    notes: str = "",
    episode_path: str = DEFAULT_EPISODE_PATH,
    update_index: bool = True,
    audit_8d: Optional[Dict[str, Any]] = None,
) -> str:
    """
    归档一次自适应语义交互。

    Args:
        material_family: 判别出的材质家族（如 "金地屏风"）
        category_ids:   本次选中的类目 id 列表
        fingerprint:    extract_fingerprint 返回的字典（可选，仅存摘要避免体积爆炸）
        image_path:     源图路径（用于去重/回放，可选）
        outcome:        "selected" / "accepted" / "rejected" / "corrected"
        confidence:     材质判别置信度
        notes:          自由文本备注
        episode_path:   episode 日志路径（默认 webui/data/adaptive_episodes.jsonl）
        update_index:   是否更新 CBR 指纹索引（Stage 4，默认 True）
        audit_8d:       审计 8 维数据（用于判断 audit_passed，可选）

    Returns:
        episode id（形如 ep_20260913T..._a1b2c3d4e5f6）
    """
    ts = datetime.now(timezone.utc).isoformat()

    # 指纹只保留摘要（embedding 128 维不入库，避免每条数 KB）
    fp_summary = None
    if fingerprint:
        fp_summary = {
            "image_shape": fingerprint.get("image_shape"),
            "affinity_top": None,  # 由调用方按需填充
        }

    record = {
        "episode_id": _episode_id(material_family, image_path, ts),
        "timestamp": ts,
        "material_family": material_family,
        "category_ids": list(category_ids or []),
        "n_categories": len(category_ids or []),
        "outcome": outcome,
        "confidence": float(confidence),
        "image_path": image_path,
        "fingerprint_summary": fp_summary,
        "notes": notes,
    }

    path = Path(episode_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Stage 4：归档后触发索引更新（CBR 案例推理库）
    if update_index and fingerprint and "embedding" in fingerprint:
        try:
            from .episode_indexer import get_default_indexer
            import numpy as np
            
            # 提取指纹向量（PCA128）
            embedding = fingerprint.get("embedding")
            if embedding is not None and len(embedding) == 128:
                # 判断是否通过审计（lost_ratio < 0.05 为通过）
                audit_passed = False
                if audit_8d:
                    lost_ratio = audit_8d.get("lost_ratio", 1.0)
                    audit_passed = (lost_ratio < 0.05)
                
                # 更新索引
                indexer = get_default_indexer()
                indexer.add(
                    task_id=record["episode_id"],
                    fingerprint=np.array(embedding, dtype=np.float32),
                    audit_passed=audit_passed
                )
                indexer.save()
        except Exception as e:
            # 索引更新失败不影响归档主流程
            print(f"⚠️ 索引更新失败（{e}），归档已完成")

    return record["episode_id"]


def load_episodes(
    episode_path: str = DEFAULT_EPISODE_PATH,
    limit: int = 1000,
    material_family: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    读取 episode 日志（供 Stage 2+ 学习消费）。

    Args:
        episode_path: 日志路径
        limit:        返回条数上限（默认 1000，最近的在前）
        material_family: 可选材质过滤

    Returns:
        episode 记录列表（按时间倒序）
    """
    path = Path(episode_path)
    if not path.exists():
        return []

    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if material_family and rec.get("material_family") != material_family:
                continue
            out.append(rec)

    out.reverse()  # 最近在前
    return out[:limit]
