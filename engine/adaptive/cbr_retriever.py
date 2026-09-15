"""案例推理（CBR）检索器 - Stage 4 核心模块。

基于指纹索引检索最相似的历史 episode，复用其已验证的 auto_tune 参数。
用于冷启动加速：新图直接继承相似图参数，无需从零 Auto-Tune。

典型用法：
    indexer = EpisodeIndexer.load_or_create(index_path)
    similar_ep = retrieve_similar_episode(fingerprint, indexer, episodes_dir)
    if similar_ep:
        # 复用参数作起点
        auto_tune = similar_ep["auto_tune"]
    else:
        # 无相似图，从零 Auto-Tune
        auto_tune = suggest_auto_tune(fingerprint, density_map)
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .episode_indexer import EpisodeIndexer
from .episode_archiver import load_episode_by_id, DEFAULT_EPISODE_PATH as DEFAULT_EPISODE_LOG


def retrieve_similar_episode(
    fingerprint: np.ndarray,
    indexer: EpisodeIndexer,
    episodes_dir: Path,
    min_similarity: float = 0.7,
    episode_log_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    product_line: Optional[str] = None,
    param_schema: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """检索最相似的历史 episode，复用其 auto_tune 参数。

    安全白名单（2026-09-15 修复「CBR 命中打印复用却没应用 / 跨类型配置污染」）：
    仅当命中历史 episode 与**当前图**满足全部显式约束时才复用——
      - ``preset_name``：同 preset（episode 记录的 ``preset``，缺省时回退 ``material_family``）；
      - ``product_line``：同产品线（episode 记录的 ``output_mode``）；
      - ``param_schema``：auto_tune 的顶层参数键集合完全一致（防止跨类型配置污染）。
    任一约束不匹配 → 视为无可用命中（返回 None），绝不跨类型复用参数。
    三个约束均留空（None）时退化为旧行为（向后兼容，不影响既有单测）。

    「调用时读取隔离 episode env」：索引与 JSONL 日志均经环境变量
    ``ADAPTIVE_INDEX_PATH`` / ``ADAPTIVE_EPISODE_PATH`` 解析（get_default_indexer /
    default_episode_path 已在内部处理），测试可安全隔离。

    Args:
        fingerprint: 查询指纹（PCA128）
        indexer: Episode 指纹索引器
        episodes_dir: episode 归档目录（如 webui/data/episodes/，文件式通道）
        min_similarity: 最小相似度阈值（余弦相似度，[0, 1]）
        episode_log_path: episode JSONL 日志路径（生产归档通道，默认 webui/data/adaptive_episodes.jsonl）
        preset_name: 安全白名单——仅复用同 preset 的历史参数
        product_line: 安全白名单——仅复用同产品线（plate/design/both）的历史参数
        param_schema: 安全白名单——仅复用 auto_tune 顶层参数键集合一致的历史参数

    Returns:
        最相似的 episode 数据（含 auto_tune 参数），如果无相似图 / 白名单不匹配则返回 None

        返回结构：
        {
            "task_id": "task_20260913_120555_b2b831",
            "similarity": 0.92,
            "auto_tune": {
                "regions": [...],
                "density_bands": [...],
                "thresholds": {...}
            },
            "fingerprint_meta": {...},
            "material_family": "金地屏风",
            "audit_8d": {...}
        }
    """
    # 检索 Top-5 相似任务（只返回通过审计的）
    similar_tasks = indexer.search(
        fingerprint,
        top_k=5,
        only_audit_passed=True
    )

    if not similar_tasks:
        return None

    # 取最相似的任务
    best_task_id, best_similarity = similar_tasks[0]

    # 相似度低于阈值，视为无相似图
    if best_similarity < min_similarity:
        return None

    # 读取 episode 数据：先文件式通道，再 JSONL 归档通道（二者格式统一后不再断裂）
    episode_data = _load_episode_data(best_task_id, episodes_dir, episode_log_path)
    if episode_data is None:
        print(f"⚠️ 未找到 episode 记录：{best_task_id}（已查文件与 JSONL 日志）")
        return None

    # ---- 安全白名单：防止跨 preset / 产品线 / 参数 schema 复用 ----
    whitelist_fail = _check_whitelist(
        episode_data,
        preset_name=preset_name,
        product_line=product_line,
        param_schema=param_schema,
    )
    if whitelist_fail:
        print(f"[CBR] 拒绝复用 {best_task_id}：{whitelist_fail}（安全白名单保护，不跨类型污染）")
        return None

    try:
        # 提取关键信息
        return {
            "task_id": best_task_id,
            "similarity": best_similarity,
            "auto_tune": episode_data.get("auto_tune", {}),
            "fingerprint_meta": (episode_data.get("fingerprint_meta")
                                 or episode_data.get("fingerprint_summary") or {}),
            "material_family": episode_data.get("material_family", "unknown"),
            "audit_8d": episode_data.get("audit_8d", {}),
        }
    except Exception as e:
        print(f"⚠️ 解析 episode 数据失败（{best_task_id}）：{e}")
        return None


def _check_whitelist(
    episode_data: Dict[str, Any],
    preset_name: Optional[str] = None,
    product_line: Optional[str] = None,
    param_schema: Optional[Any] = None,
) -> Optional[str]:
    """校验候选 episode 是否满足安全白名单约束。

    Returns:
        不满足时的原因字符串（调用方据此拒绝复用）；全部满足返回 None。
    """
    # 1) preset 一致：episode 记录的 preset 优先，缺省回退 material_family
    if preset_name is not None:
        cand_preset = episode_data.get("preset") or episode_data.get("material_family")
        if cand_preset != preset_name:
            return f"preset 不匹配(候选={cand_preset} ≠ 当前={preset_name})"

    # 2) 产品线一致：episode 记录的 output_mode
    if product_line is not None:
        cand_line = str(episode_data.get("output_mode") or "").lower()
        if cand_line and cand_line != str(product_line).lower():
            return f"产品线不匹配(候选={cand_line} ≠ 当前={product_line})"

    # 3) 参数 schema 一致：auto_tune 顶层键集合完全一致（防止跨类型配置污染）
    if param_schema is not None:
        expected = set(param_schema) if hasattr(param_schema, "__iter__") else set()
        got = set((episode_data.get("auto_tune") or {}).keys())
        if not got:
            return "候选 auto_tune 为空（无可复用参数）"
        if got != expected:
            return f"参数 schema 不匹配(候选={sorted(got)} ≠ 期望={sorted(expected)})"

    return None


def _load_episode_data(
    task_id: str,
    episodes_dir: Path,
    episode_log_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """双通道加载 episode：文件式（历史/测试）→ JSONL 归档（生产）。

    Args:
        task_id: episode id / task id
        episodes_dir: 文件式归档目录
        episode_log_path: JSONL 日志路径（None 时用默认）

    Returns:
        episode 字典，均未命中返回 None
    """
    # 通道 1：文件式 `<task_id>.episode.json`
    episode_file = _find_episode_file(task_id, episodes_dir)
    if episode_file:
        try:
            with episode_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取 episode 文件失败（{task_id}）：{e}")

    # 通道 2：JSONL 归档日志（生产 archive_episode 写入此处）
    from .episode_archiver import default_episode_path
    log_path = episode_log_path or default_episode_path()
    try:
        return load_episode_by_id(task_id, log_path)
    except Exception as e:
        print(f"⚠️ 读取 episode 日志失败（{log_path}）：{e}")
        return None


def _find_episode_file(task_id: str, episodes_dir: Path) -> Optional[Path]:
    """查找 episode 文件（支持按月归档的目录结构）。
    
    Args:
        task_id: 任务 ID（如 task_20260913_120555_b2b831）
        episodes_dir: episode 归档目录（如 webui/data/episodes/）
    
    Returns:
        episode 文件路径，如果未找到则返回 None
        
    目录结构示例：
        webui/data/episodes/
        ├── 2026-09/
        │   ├── task_20260913_120555_b2b831.episode.json
        │   └── task_20260913_103835_3cfc3b.episode.json
        └── 2026-08/
            └── task_20260812_132516_371e55.episode.json
    """
    episodes_dir = Path(episodes_dir)
    
    # 方案 1：直接查找（无按月归档）
    direct_path = episodes_dir / f"{task_id}.episode.json"
    if direct_path.exists():
        return direct_path
    
    # 方案 2：按月归档目录查找（2026-09/task_xxx.episode.json）
    # 从 task_id 中提取日期：task_20260913_120555_b2b831 → 20260913 → 2026-09
    try:
        # task_id 格式：task_<YYYYMMDD>_<HHMMSS>_<hash>
        date_str = task_id.split("_")[1]  # 20260913
        year = date_str[:4]  # 2026
        month = date_str[4:6]  # 09
        month_dir = episodes_dir / f"{year}-{month}"
        
        month_path = month_dir / f"{task_id}.episode.json"
        if month_path.exists():
            return month_path
    except (IndexError, ValueError):
        pass
    
    # 方案 3：递归搜索（兜底，性能较差）
    for ep_file in episodes_dir.rglob(f"{task_id}.episode.json"):
        return ep_file
    
    return None


def get_cbr_hit_info(similar_episode: Optional[Dict[str, Any]]) -> str:
    """生成 CBR 命中信息（用于日志）。
    
    Args:
        similar_episode: retrieve_similar_episode 返回的结果
    
    Returns:
        CBR 命中信息字符串，如果无命中则返回空字符串
        
    示例：
        "CBR hit: task_20260913_120555_b2b831 (similarity=0.92, material=金地屏风)"
    """
    if not similar_episode:
        return ""
    
    task_id = similar_episode["task_id"]
    similarity = similar_episode["similarity"]
    material = similar_episode.get("material_family", "unknown")
    
    return f"CBR hit: {task_id} (similarity={similarity:.2f}, material={material})"
