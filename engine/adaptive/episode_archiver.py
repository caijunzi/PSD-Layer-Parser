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
import os
import threading
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# 线程安全写（主流程与 Web 可能并发归档）
_lock = threading.Lock()

# episode 日志默认路径（保留常量供外部引用）
DEFAULT_EPISODE_PATH = "webui/data/adaptive_episodes.jsonl"


def default_episode_path() -> str:
    """解析 episode 日志路径（**调用时**读环境变量，便于测试隔离/多实例部署）。"""
    return os.environ.get("ADAPTIVE_EPISODE_PATH", DEFAULT_EPISODE_PATH)


# =============================================================================
# 严格准入相关常量（与 tools/audit_psb.audit() 产出的 dims 键严格一致）
# =============================================================================
# 8 维审计维度键（顺序即维度序）
DIM_KEYS = ("① 层属性", "② 分辨率真实性", "③ 实例重复", "④ 内容承载",
            "⑤ 合成等价性", "⑥ 底板纯净度", "⑦ plate 合规", "⑧ manifest 一致")
# 仅 PLATE 产品线必需的维度；DESIGN 线该维度不适用（na）
PLATE_DIM = "⑦ plate 合规"
# 产品线取值
PRODUCT_LINES = ("plate", "design", "both")


def _resolve_index_path() -> Path:
    """解析 CBR 索引文件路径（与 episode_indexer.get_default_indexer 同口径，但不加载已有索引）。"""
    override = os.environ.get("ADAPTIVE_INDEX_PATH")
    if override:
        return Path(override)
    engine_root = Path(__file__).resolve().parent.parent.parent
    return engine_root / "webui" / "data" / "episode_index.pkl"


def _summary_from_full(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从完整 8 维审计报告派生 4 字段摘要（向后兼容，旧代码读 audit_8d.lost_ratio 等）。

    原始完整报告始终保留在 audit_full，摘要仅作便捷视图，绝不替代原始。
    """
    dims = (report or {}).get("dims", {})

    def _m(dim: str, key: str, default=None):
        return dims.get(dim, {}).get("metrics", {}).get(key, default)

    return {
        "lost_ratio": _m("④ 内容承载", "lost_ratio"),
        "rmse_lowfreq": _m("⑤ 合成等价性", "rmse_lowfreq"),
        "rmse_raw": _m("⑤ 合成等价性", "rmse_raw"),
        "plate_purity_ok": _m("⑦ plate 合规", "plate_purity_ok"),
        "tac_max_pct": _m("⑦ plate 合规", "tac_max_pct"),
        "n_layers": _m("① 层属性", "layer_count"),
        "passed": (report or {}).get("passed"),
    }


def _store_audit(rec: Dict[str, Any], report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把审计报告写入 episode 记录：保留原始完整报告(audit_full) + 兼容摘要(audit_8d)。

    - report 含 'dims' 视为完整 8 维报告 → 同时存 audit_full 与派生 audit_8d
    - 否则视为旧式 4 字段摘要 → 仅存 audit_8d，audit_full 保持 None（严格准入会据此 skip）
    """
    if isinstance(report, dict) and "dims" in report:
        rec["audit_full"] = report
        rec["audit_8d"] = _summary_from_full(report)
    elif isinstance(report, dict):
        rec["audit_8d"] = {
            k: report.get(k)
            for k in ("lost_ratio", "rmse_lowfreq", "rmse_raw", "plate_purity_ok", "tac_max_pct", "n_layers", "passed")
        }
        rec.setdefault("audit_full", None)
    return rec


def _admit(rec: Dict[str, Any]) -> Dict[str, Any]:
    """严格 8 维准入判定。

    规则（2026-09-15 修正）：
    - 必须持有**原始完整 8 维报告**（audit_full）；缺字段/缺报告 → 不准入（skip）。
    - 每个必需维度必须：存在、真正核验过(evaluated)、且 passed=True。
      否则（失败 / 未核验 / 缺字段）→ 不准入。
    - 产品线适用性：DESIGN 线豁免 ⑦ plate 合规（该维度 na）；PLATE 线 ⑦ 为必需。
    - **绝不降低任何重建质量阈值**来凑通过——阈值由 audit_psb 计算，这里只消费 passed。

    Returns: {"admitted": bool, "reason": str}
    """
    report = rec.get("audit_full")
    if not isinstance(report, dict) or "dims" not in report:
        return {"admitted": False, "reason": "no_full_report"}
    output_mode = str(rec.get("output_mode") or "").lower()
    required = list(DIM_KEYS)
    if output_mode == "design":
        required = [d for d in required if d != PLATE_DIM]  # design 线 ⑦ 不适用
    dims = report.get("dims", {})
    for dim in required:
        d = dims.get(dim)
        if not isinstance(d, dict):
            return {"admitted": False, "reason": f"missing_dim:{dim}"}
        if d.get("na"):
            return {"admitted": False, "reason": f"not_evaluated:{dim}"}
        if not d.get("evaluated", False):
            return {"admitted": False, "reason": f"not_evaluated:{dim}"}
        if not d.get("passed"):
            return {"admitted": False, "reason": f"failed:{dim}"}
    return {"admitted": True, "reason": "ok"}


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
    episode_path: Optional[str] = None,
    update_index: bool = True,
    audit_8d: Optional[Dict[str, Any]] = None,
    audit_report: Optional[Dict[str, Any]] = None,
    output_mode: Optional[str] = None,
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

    # 审计数据：优先完整报告(audit_report)，回退旧式 4 字段摘要(audit_8d)。
    # 原始完整报告保留在 audit_full，摘要 audit_8d 仅作便捷视图（详见 _store_audit）。
    audit_summary = None
    _full_report = audit_report if isinstance(audit_report, dict) else (
        audit_8d if isinstance(audit_8d, dict) and "dims" in audit_8d else None)
    if _full_report is not None:
        _tmp = {"audit_full": None, "audit_8d": None}
        _store_audit(_tmp, _full_report)
        audit_summary = _tmp.get("audit_8d")
    elif isinstance(audit_8d, dict):
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
        "audit_full": _tmp.get("audit_full") if _full_report is not None else None,
        "audit_8d": audit_summary,
        "output_mode": output_mode,
        "auto_tune": auto_tune or {},
        "notes": notes,
    }

    path = Path(episode_path or default_episode_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 注意：索引更新**不**在此处做。归档发生在分割/超分之前，此时审计尚不存在，
    # 即便有 embedding 也无法判定 audit_passed（强填 False 只会留下陈旧的未通过条目）。
    # 正确收口由「审计完成后」的 finalize_episode（单条）或 sync_index_from_log
    # （全量权威重建）完成——它们基于完整 8 维报告做严格准入。

    return record["episode_id"]


def load_episodes(
    episode_path: Optional[str] = None,
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
    path = Path(episode_path or default_episode_path())
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
    episode_path: Optional[str] = None,
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
    path = Path(episode_path or default_episode_path())
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
    audit_report: Optional[Dict[str, Any]] = None,
    episode_path: Optional[str] = None,
    output_mode: Optional[str] = None,
) -> bool:
    """回填某 episode 的审计结果（审计在出图之后才产生，故需事后回填）。

    修复（2026-09-15）：引擎归档发生在分割/超分之前，此时审计尚不存在，
    导致 archive 时 audit_passed 恒为 False、CBR 索引筛空。本函数让审计完成后
    由 WebUI 侧回填，**保留原始完整 8 维报告(audit_full)**，并供 sync_index_from_log 重建索引。

    参数兼容旧式 4 字段摘要（audit_8d 形式）：若传入 dict 不含 'dims' 视为旧摘要，
    仅写 audit_8d，audit_full 保持 None（严格准入据此 skip）。

    Args:
        episode_id:   episode id
        audit_report: 完整 8 维审计报告（tools/audit_psb.audit() 返回值）；
                      也兼容旧式 4 字段摘要 dict
        episode_path: JSONL 日志路径
        output_mode: 产品线（plate/design/both），用于严格准入所需维度判定

    Returns:
        是否找到并更新
    """
    path = Path(episode_path or default_episode_path())
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
                    _store_audit(rec, audit_report)
                    if output_mode is not None:
                        rec["output_mode"] = output_mode
                    found = True
                records.append(rec)

        if found:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return found


def finalize_episode(
    epid: str,
    audit_report: Optional[Dict[str, Any]] = None,
    detections: Optional[Dict[str, Any]] = None,
    output_mode: Optional[str] = None,
    artifact: Optional[str] = None,
    episode_path: Optional[str] = None,
    update_index: bool = True,
) -> Dict[str, Any]:
    """收口 API：审计完成后把**原始完整 8 维报告**写入 episode，并按严格准入刷新 CBR 索引。

    设计给**主代理 / WebUI** 接线：一次产品线出图 → 跑审计 → 调用本函数完成 episode 终态。
    与 `archive_episode`（出图前归档）配对：archive 写语义/指纹，finalize 写审计终态。

    Args:
        epid:          archive_episode 返回的 episode id
        audit_report / detections: 完整 8 维审计报告（tools/audit_psb.audit() 返回值，
                                    含 dims + passed）。detections 为同义别名。
        output_mode:   "plate" | "design" | "both" —— 决定严格准入所需维度
                       （design 线豁免 ⑦ plate 合规；plate 线 ⑦ 必需）
        artifact:      被审计产物路径（如 result.plate.psb），仅作记录，便于回放/排查
        episode_path:  JSONL 日志路径
        update_index:  是否刷新 CBR 索引（默认 True）

    Returns:
        {"episode_id": str, "admitted": bool, "audit_passed": bool, "reason": str}
        - admitted=True  → 已写入索引（audit_passed=True）
        - admitted=False → 未准入（失败/缺字段/必需维度未核验），不入索引，reason 指明原因
    """
    report = audit_report if isinstance(audit_report, dict) else detections
    path = Path(episode_path or default_episode_path())
    if not path.exists():
        return {"episode_id": epid, "admitted": False, "audit_passed": False,
                "reason": "no_episode_log"}

    found = False
    matched_rec = None
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
                    records.append(line)
                    continue
                if rec.get("episode_id") == epid:
                    # 保留原始完整报告（audit_full）+ 兼容摘要（audit_8d）
                    _store_audit(rec, report)
                    if output_mode is not None:
                        rec["output_mode"] = output_mode
                    if artifact is not None:
                        rec["audited_artifact"] = artifact
                    found = True
                    matched_rec = rec
                records.append(rec)
        if found:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    if isinstance(rec, str):
                        f.write(rec + "\n")
                    else:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not found:
        return {"episode_id": epid, "admitted": False, "audit_passed": False,
                "reason": "episode_not_found"}

    # 注意：必须用 matched_rec（命中的那条），不能用循环后的 rec（那是最后一条）
    decision = _admit({"audit_full": matched_rec.get("audit_full"),
                       "output_mode": matched_rec.get("output_mode")})
    admitted = decision["admitted"]
    if update_index:
        sync_index_from_log(str(path))

    decision["episode_id"] = epid
    decision["audit_passed"] = admitted
    return decision


def sync_index_from_log(
    episode_path: Optional[str] = None,
) -> int:
    """从 JSONL 日志重建 CBR 指纹索引（用记录中的 embedding + audit_8d 判定 audit_passed）。

    这是"审计回填后再刷新索引"的收口函数：archive 时审计未知 → 审计完成后
    update_episode_audit 回填 → 本函数重建索引，使 audit_passed 反映真实审计结果。

    Returns:
        本次写入索引（严格准入通过）的 episode 数量
    """
    import numpy as np
    from .episode_indexer import EpisodeIndexer

    path = Path(episode_path or default_episode_path())
    if not path.exists():
        return 0

    # 权威重建：全新空索引，只收「严格 8 维准入通过」的 episode。
    # 不再像旧逻辑那样用 4 维宽松指标(lost_ratio<0.05)且缺值默认良好——
    # 失败/缺字段/必需维度未核验一律不准入（skip）。
    indexer = EpisodeIndexer(_resolve_index_path())
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
            if _admit(rec)["admitted"]:
                indexer.add(rec.get("episode_id"), np.array(emb, dtype=np.float32), True)
                n += 1

    indexer.save()
    return n
