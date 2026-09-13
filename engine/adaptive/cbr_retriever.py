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


def retrieve_similar_episode(
    fingerprint: np.ndarray,
    indexer: EpisodeIndexer,
    episodes_dir: Path,
    min_similarity: float = 0.7
) -> Optional[Dict[str, Any]]:
    """检索最相似的历史 episode，复用其 auto_tune 参数。
    
    Args:
        fingerprint: 查询指纹（PCA128）
        indexer: Episode 指纹索引器
        episodes_dir: episode 归档目录（如 webui/data/episodes/）
        min_similarity: 最小相似度阈值（余弦相似度，[0, 1]）
    
    Returns:
        最相似的 episode 数据（含 auto_tune 参数），如果无相似图则返回 None
        
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
    
    # 读取 episode JSON，提取 auto_tune 参数
    episode_file = _find_episode_file(best_task_id, episodes_dir)
    if not episode_file:
        print(f"⚠️ 未找到 episode 文件：{best_task_id}")
        return None
    
    try:
        with episode_file.open("r", encoding="utf-8") as f:
            episode_data = json.load(f)
        
        # 提取关键信息
        return {
            "task_id": best_task_id,
            "similarity": best_similarity,
            "auto_tune": episode_data.get("auto_tune", {}),
            "fingerprint_meta": episode_data.get("fingerprint_meta", {}),
            "material_family": episode_data.get("material_family", "unknown"),
            "audit_8d": episode_data.get("audit_8d", {})
        }
    except Exception as e:
        print(f"⚠️ 读取 episode 文件失败（{best_task_id}）：{e}")
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
