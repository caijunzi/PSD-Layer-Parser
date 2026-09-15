"""Episode 日志写回辅助模块（独立于 episode_archiver，安全更新 JSONL）。

为何独立存在（2026-09-15）：
- ``episode_archiver.finalize_episode`` 的 ``detections`` 参数实为其审计报告的同义别名，
  **不是**检测记录；直接在 finalize 里塞检测会污染审计语义。
- 分割后的「最终检测」需要在出图之后写回对应 episode，并保留
  ``artifact`` / ``output_mode`` / ``preset`` / ``effective_auto_tune`` 等元数据。
- 本模块复用 ``episode_archiver`` 的同一把文件锁（``_lock``）与默认路径解析，
  以**追加/就地更新同 episode 最新记录**的方式安全改写 JSONL，避免与归档/索引并发写坏。

本模块只读取与追加/更新，不改动 episode_archiver 的任何逻辑。
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# 复用 archiver 的文件锁，确保与 archive_episode / finalize_episode 互斥写同一 JSONL
from .episode_archiver import _lock, default_episode_path  # noqa: E402


def update_episode_record(
    episode_id: str,
    episode_path: Optional[str] = None,
    **fields: Any,
) -> bool:
    """就地更新某 episode 记录的指定字段（仅覆盖显式传入的非 None 字段）。

    读取整份 JSONL，找到 ``episode_id`` 对应记录后合并 ``fields``，写回。
    未传入或为 None 的字段保持不变（保留既有值）。线程安全（复用 archiver._lock）。

    Args:
        episode_id:   archive_episode 返回的 episode id
        episode_path: JSONL 日志路径（默认走 default_episode_path，即 ADAPTIVE_EPISODE_PATH）
        **fields:     要写回的字段（如 detections / artifact / output_mode /
                      preset / effective_auto_tune / auto_tune）

    Returns:
        是否找到并更新了该 episode
    """
    path = Path(episode_path or default_episode_path())
    if not path.exists():
        return False

    found = False
    records: List[Any] = []
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
                if rec.get("episode_id") == episode_id:
                    for k, v in fields.items():
                        if v is not None:
                            rec[k] = v
                    found = True
                records.append(rec)
        if found:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    if isinstance(rec, str):
                        f.write(rec + "\n")
                    else:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return found


def update_episode_detections(
    episode_id: str,
    detections: List[Dict[str, Any]],
    episode_path: Optional[str] = None,
    artifact: Optional[str] = None,
    output_mode: Optional[str] = None,
    preset: Optional[str] = None,
    effective_auto_tune: Optional[Dict[str, Any]] = None,
    auto_tune: Optional[Dict[str, Any]] = None,
) -> bool:
    """分割后把最终检测写回对应 episode，并保留 artifact/output_mode/preset/effective_auto_tune。

    这是「出图后」归档补完：archive_episode 发生在分割/超分之前（尚无真实检测），
    此处用分割后真实 dino_detections 回填，并固化产物路径与本次生效的调优参数。

    字段语义：
      - detections:            分割后真实检测列表（已附规范 category_id，含 layer_name 契约）
      - artifact:             被审计产物路径（如 result.plate.psb）
      - output_mode:          产品线（plate/design/both）
      - preset:               实际使用的 preset 名（保留以备 CBR 同 preset 白名单复用）
      - effective_auto_tune:  本次生效的图级调优参数（CBR 复用来源）
      - auto_tune:            （兼容）若传入则同 effective_auto_tune 一并落盘

    Returns:
        是否找到并更新
    """
    fields: Dict[str, Any] = {"detections": detections, "dino_detections": detections}
    if artifact is not None:
        fields["artifact"] = artifact
    if output_mode is not None:
        fields["output_mode"] = output_mode
    if preset is not None:
        fields["preset"] = preset
    if effective_auto_tune is not None:
        fields["effective_auto_tune"] = effective_auto_tune
    if auto_tune is not None and effective_auto_tune is None:
        fields["auto_tune"] = auto_tune

    ok = update_episode_record(episode_id, episode_path=episode_path, **fields)
    return ok
