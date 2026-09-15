"""Episode 指纹索引器 - Stage 4 案例推理库（CBR）核心模块。

提供基于指纹（PCA128）的余弦相似度检索，支持持久化索引到 pickle 文件。
用于从历史 episode 中快速找到最相似的已验证任务，复用其 auto_tune 参数。

索引数据结构：
- task_id → fingerprint_128d（PCA 降维后的指纹向量）
- task_id → audit_passed（bool，该任务是否通过审计 8 维）

检索算法：余弦相似度（cosine similarity）
持久化：pickle 文件（webui/data/episode_index.pkl）

典型用法：
    indexer = EpisodeIndexer.load_or_create()
    indexer.add(task_id, fingerprint_128d, audit_passed=True)
    indexer.save()
    
    similar_tasks = indexer.search(query_fingerprint, top_k=5)
    # [(task_id, cosine_sim), ...]
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class EpisodeIndexer:
    """Episode 指纹索引器（基于余弦相似度）。
    
    Attributes:
        index_path: 索引文件路径（pickle）
        fingerprints: {task_id: fingerprint_128d}
        audit_status: {task_id: audit_passed}
    """
    
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.fingerprints: dict[str, np.ndarray] = {}
        self.audit_status: dict[str, bool] = {}
    
    def add(self, task_id: str, fingerprint: np.ndarray, audit_passed: bool):
        """添加一个 episode 到索引。
        
        Args:
            task_id: 任务 ID（如 task_20260913_120555_b2b831）
            fingerprint: PCA128 指纹向量（128 维）
            audit_passed: 该任务是否通过审计 8 维（lost_ratio < 0.05）
        """
        if fingerprint.shape != (128,):
            raise ValueError(f"指纹维度必须是 (128,)，实际: {fingerprint.shape}")
        
        self.fingerprints[task_id] = fingerprint.astype(np.float32)
        self.audit_status[task_id] = audit_passed
    
    def search(
        self, 
        query_fingerprint: np.ndarray, 
        top_k: int = 5,
        only_audit_passed: bool = True
    ) -> List[Tuple[str, float]]:
        """检索最相似的 episode（基于余弦相似度）。
        
        Args:
            query_fingerprint: 查询指纹（128 维）
            top_k: 返回 Top-K 相似任务
            only_audit_passed: 是否只返回通过审计的任务
        
        Returns:
            [(task_id, cosine_similarity), ...]，按相似度降序排列
        """
        if query_fingerprint.shape != (128,):
            raise ValueError(f"查询指纹维度必须是 (128,)，实际: {query_fingerprint.shape}")
        
        if not self.fingerprints:
            return []
        
        # 过滤：只返回通过审计的任务
        candidates = {
            task_id: fp
            for task_id, fp in self.fingerprints.items()
            if not only_audit_passed or self.audit_status.get(task_id, False)
        }
        
        if not candidates:
            return []
        
        # 计算余弦相似度
        query_norm = np.linalg.norm(query_fingerprint)
        if query_norm == 0:
            return []
        
        similarities = []
        for task_id, fp in candidates.items():
            fp_norm = np.linalg.norm(fp)
            if fp_norm == 0:
                continue
            
            # cosine_sim = dot(q, fp) / (||q|| * ||fp||)
            cosine_sim = np.dot(query_fingerprint, fp) / (query_norm * fp_norm)
            similarities.append((task_id, float(cosine_sim)))
        
        # 按相似度降序排列，返回 Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self):
        """持久化索引到 pickle 文件。"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.index_path.open("wb") as f:
            pickle.dump({
                "fingerprints": self.fingerprints,
                "audit_status": self.audit_status
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load_or_create(cls, index_path: Path) -> "EpisodeIndexer":
        """加载已有索引或创建新索引。
        
        Args:
            index_path: 索引文件路径
        
        Returns:
            EpisodeIndexer 实例
        """
        indexer = cls(index_path)
        
        if index_path.exists():
            try:
                with index_path.open("rb") as f:
                    data = pickle.load(f)
                indexer.fingerprints = data.get("fingerprints", {})
                indexer.audit_status = data.get("audit_status", {})
            except Exception as e:
                print(f"⚠️ 加载索引失败（{e}），将创建新索引")
        
        return indexer
    
    def __len__(self) -> int:
        """返回索引中的 episode 数量。"""
        return len(self.fingerprints)
    
    def __repr__(self) -> str:
        n_total = len(self.fingerprints)
        n_passed = sum(self.audit_status.values())
        return f"<EpisodeIndexer: {n_total} episodes, {n_passed} audit_passed>"


def get_default_indexer() -> EpisodeIndexer:
    """获取默认索引器（全局单例）。
    
    Returns:
        EpisodeIndexer 实例，索引文件位于 webui/data/episode_index.pkl
        （可用环境变量 ADAPTIVE_INDEX_PATH 覆盖，便于测试隔离/多实例部署）
    """
    import os
    from pathlib import Path

    override = os.environ.get("ADAPTIVE_INDEX_PATH")
    if override:
        return EpisodeIndexer.load_or_create(Path(override))

    # 推断项目根目录（engine/adaptive/ 往上两级）
    engine_root = Path(__file__).resolve().parent.parent.parent
    index_path = engine_root / "webui" / "data" / "episode_index.pkl"
    
    return EpisodeIndexer.load_or_create(index_path)
