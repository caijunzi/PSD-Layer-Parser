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
    auto_tune: Optional[Dict[str, Any]] = None,
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
        auto_tune:      本次图级 Auto-Tune 参数（供 CBR 复用，可选）

    Returns:
        episode id（形如 ep_20260913T..._a1b2c3d4e5f6）
    """
    ts = datetime.now(timezone.utc).isoformat()

    # 指纹摘要 + 128 维 embedding（embedding 用于 CBR 索引延后重建；128 float ≈ 1KB/条）
    fp_summary = None
    embedding = None
    if fingerprint:
        fp_summary = {
            "image_shape": fingerprint.get("image_shape"),
            "affinity_top": None,  # 由调用方按需填充
        }
        _emb = fingerprint.get("embedding")
        if _emb is not None and len(_emb) == 128:
            embedding = [round(float(x), 6) for x in _emb]

    # 审计摘要：只保留可用于回归判定与 CBR 门槛的关键字段
    audit_summary = None
    if audit_8d:
        audit_summary = {
            "lost_ratio": audit_8d.get("lost_ratio"),
            "rmse_lowfreq": audit_8d.get("rmse_lowfreq"),
            "plate_purity_ok": audit_8d.get("plate_purity_ok"),
            "n_layers": audit_8d.get("n_layers"),
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
        "embedding": embedding,
        "audit_8d": audit_summary,
        "auto_tune": auto_tune or {},
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


def load_episode_by_id(
    episode_id: str,
    episode_path: str = DEFAULT_EPISODE_PATH,
) -> Optional[Dict[str, Any]]:
    """按 episode_id 从 JSONL 日志中读取单条记录（供 CBR 检索复用参数）。

    这是与 archive_episode 对称的读取入口：archive 写 JSONL，CBR 按 episode_id 回读，
    修复此前"归档写 JSONL、检索找 episodes/**/*.episode.json"的格式断裂。

    Args:
        episode_id: archive_episode 返回的 episode id
        episode_path: episode 日志路径

    Returns:
        episode 记录字典，未找到返回 None
    """
    path = Path(episode_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("episode_id") == episode_id:
                return rec
    return None


def update_episode_audit(
    episode_id: str,
    audit_8d: Dict[str, Any],
    episode_path: str = DEFAULT_EPISODE_PATH,
) -> bool:
    """回填某 episode 的审计 8 维结果（审计在出图之后才产生，故需事后回填）。

    修复（2026-09-15）：引擎归档发生在分割/超分之前，此时审计尚不存在，
    导致 archive 时 audit_passed 恒为 False、CBR 索引筛空。本函数让审计完成后
    由 WebUI 侧回填，并配合 sync_index_from_log 重建索引。

    Args:
        episode_id: episode id
        audit_8d: 审计 8 维数据（lost_ratio / rmse_lowfreq / plate_purity_ok / n_layers）
        episode_path: JSONL 日志路径

    Returns:
        是否找到并更新
    """
    path = Path(episode_path)
    if not path.exists():
        return False

    found = False
    records: List[Dict[str, Any]] = []
    with _lock:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("episode_id") == episode_id:
                    rec["audit_8d"] = {
                        "lost_ratio": audit_8d.get("lost_ratio"),
                        "rmse_lowfreq": audit_8d.get("rmse_lowfreq"),
                        "plate_purity_ok": audit_8d.get("plate_purity_ok"),
                        "n_layers": audit_8d.get("n_layers"),
                    }
                    found = True
                records.append(rec)

        if found:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return found


def sync_index_from_log(
    episode_path: str = DEFAULT_EPISODE_PATH,
) -> int:
    """从 JSONL 日志重建 CBR 指纹索引（用记录中的 embedding + audit_8d 判定 audit_passed）。

    这是"审计回填后再刷新索引"的收口函数：archive 时审计未知 → 审计完成后
    update_episode_audit 回填 → 本函数重建索引，使 audit_passed 反映真实审计结果。

    Returns:
        本次写入索引的 episode 数量
    """
    import numpy as np
    from .episode_indexer import get_default_indexer

    path = Path(episode_path)
    if not path.exists():
        return 0

    indexer = get_default_indexer()
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            emb = rec.get("embedding")
            if not emb or len(emb) != 128:
                continue
            audit = rec.get("audit_8d") or {}
            lost = audit.get("lost_ratio")
            audit_passed = bool(lost is not None and float(lost) < 0.05)
            indexer.add(rec.get("episode_id"), np.array(emb, dtype=np.float32), audit_passed)
            n += 1

    if n:
        indexer.save()
    return n
