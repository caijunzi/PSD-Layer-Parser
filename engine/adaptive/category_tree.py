"""类目树管理（Stage 5.1）

层级化类目（如"山水画 → 山 → 远山"），支持：
- 插入父子关系 + DFS 无环检测
- 查询祖先链（用于泛化/继承）
- 预留：继承父类目先验（area_budget / shape_prior，TODO 待真实统计数据）
"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime, timezone


class CategoryTree:
    """类目树管理器，基于 categories 表的 parent_id / path 字段"""
    
    def __init__(self, db_path: str = "webui/data/adaptive_semantics.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def insert(self, category_id: str, parent_id: Optional[str] = None) -> bool:
        """插入父子关系，带无环检测
        
        Args:
            category_id: 类目 ID
            parent_id: 父类目 ID（None 表示根类目）
        
        Returns:
            是否插入成功
        
        Raises:
            ValueError: 循环依赖（parent_id 是 category_id 的后代）
        """
        conn = self._get_conn()
        try:
            # 无环检测：parent_id 不能是 category_id 的后代
            if parent_id and self._would_create_cycle(category_id, parent_id, conn):
                raise ValueError(
                    f"循环依赖：{parent_id} 是 {category_id} 的后代，"
                    f"不能将 {category_id} 挂到 {parent_id} 下"
                )
            
            # 更新 parent_id 与 path
            if parent_id:
                # 查询父类目的 path
                parent_path = conn.execute(
                    "SELECT path FROM categories WHERE id = ?", (parent_id,)
                ).fetchone()
                if not parent_path:
                    raise ValueError(f"父类目不存在：{parent_id}")
                
                # path = parent_path + category_id + "/"
                new_path = parent_path[0] + category_id + "/"
            else:
                new_path = "/"
            
            # 更新数据库
            conn.execute(
                "UPDATE categories SET parent_id = ?, path = ? WHERE id = ?",
                (parent_id, new_path, category_id)
            )
            conn.commit()
            return True
        
        finally:
            conn.close()
    
    def _would_create_cycle(
        self, category_id: str, parent_id: str, conn: sqlite3.Connection
    ) -> bool:
        """检测：parent_id 是否是 category_id 的后代（DFS）
        
        Args:
            category_id: 当前类目 ID
            parent_id: 候选父类目 ID
            conn: 数据库连接
        
        Returns:
            True 表示会产生循环依赖
        """
        # DFS 遍历 category_id 的所有后代
        visited: Set[str] = set()
        stack = [category_id]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # 如果 parent_id 在后代中，则会产生循环
            if current == parent_id:
                return True
            
            # 查询当前节点的所有子节点
            children = conn.execute(
                "SELECT id FROM categories WHERE parent_id = ?", (current,)
            ).fetchall()
            stack.extend([child[0] for child in children])
        
        return False
    
    def get_ancestors(self, category_id: str) -> List[str]:
        """查询类目的祖先链（从直接父类到根）
        
        Args:
            category_id: 类目 ID
        
        Returns:
            祖先 ID 列表（从近到远）[parent_id, grandparent_id, ...]
            
        示例：
            - "distant_mountains" → ["山", "山水画"]
            - "山" → ["山水画"]
            - "山水画" → []
        """
        conn = self._get_conn()
        try:
            ancestors = []
            current_id = category_id
            # 防环：数据异常（如 parent_id 自环）时避免死循环
            visited: Set[str] = {category_id}
            
            # 向上遍历 parent_id 链
            while True:
                parent = conn.execute(
                    "SELECT parent_id FROM categories WHERE id = ?", (current_id,)
                ).fetchone()
                
                if not parent or parent[0] is None:
                    break
                
                pid = parent[0]
                if pid in visited:
                    # 检测到祖先链存在环（数据损坏），停止遍历而非挂死
                    print(f"⚠️ 类目树祖先链检测到环：{category_id} → ... → {pid}，已截断")
                    break

                ancestors.append(pid)
                visited.add(pid)
                current_id = pid
            
            return ancestors
        
        finally:
            conn.close()
    
    def inherit_priors(self, category_id: str, parent_id: str) -> dict:
        """从父类目继承形状先验（area_budget / elongation / compactness）

        设计逻辑：
        1. 查询 parent_id 的 category_priors（area_budget_min/max, elongation_mean/std, ...）
        2. 若父类目有先验数据，复制到 category_id 的 priors 行
        3. 标记来源：n_samples=0（表示继承而非统计）

        priors 表为空 / 父类目无先验时**优雅跳过**（返回状态字典，不抛异常），
        避免在默认数据下让调用方崩溃。待 Stage 3 统计填充后再自然生效。

        Args:
            category_id: 子类目 ID
            parent_id: 父类目 ID

        Returns:
            {"ok": bool, "inherited": bool, "reason": str}
        """
        conn = self._get_conn()
        try:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='category_priors'"
            ).fetchone() is not None
            if not has_table:
                return {"ok": False, "inherited": False,
                        "reason": "category_priors 表不存在（跳过继承）"}

            row = conn.execute(
                """
                SELECT area_budget_min, area_budget_max, elongation_mean, elongation_std,
                       compactness_mean, compactness_std, n_components_mode
                FROM category_priors WHERE category_id = ?
                """,
                (parent_id,),
            ).fetchone()

            if row is None:
                return {"ok": False, "inherited": False,
                        "reason": f"父类目 {parent_id} 无先验数据（跳过，待 Stage3 统计填充）"}

            now = int(datetime.now(timezone.utc).timestamp())
            conn.execute(
                """
                INSERT OR REPLACE INTO category_priors (
                    category_id, area_budget_min, area_budget_max, elongation_mean,
                    elongation_std, compactness_mean, compactness_std,
                    n_components_mode, n_samples, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (category_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6], now),
            )
            conn.commit()
            return {"ok": True, "inherited": True,
                    "reason": f"从 {parent_id} 继承先验（n_samples=0 标记为继承来源）"}
        finally:
            conn.close()


def get_default_tree() -> CategoryTree:
    """获取默认类目树实例（单例模式）"""
    return CategoryTree()
