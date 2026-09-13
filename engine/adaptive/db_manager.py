"""
数据库管理模块

提供：
- SQLite 连接池（线程安全）
- 版本管理（激活/回滚/GC）
- 事务封装
"""
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta


class DBManager:
    """
    SQLite 数据库管理器（线程安全）
    """
    def __init__(self, db_path: str):
        """
        Args:
            db_path: 数据库文件路径（如 webui/data/adaptive_semantics.db）
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        获取线程本地连接（懒初始化）
        """
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._local.conn.row_factory = sqlite3.Row  # 允许按列名访问
        return self._local.conn
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        执行 SQL 查询
        
        Args:
            sql: SQL 语句
            params: 参数元组
        
        Returns:
            游标对象
        """
        conn = self._get_connection()
        return conn.execute(sql, params)
    
    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """
        批量执行 SQL
        """
        conn = self._get_connection()
        return conn.executemany(sql, params_list)
    
    def commit(self):
        """
        提交事务
        """
        conn = self._get_connection()
        conn.commit()
    
    def rollback(self):
        """
        回滚事务
        """
        conn = self._get_connection()
        conn.rollback()
    
    def close(self):
        """
        关闭连接（线程退出时调用）
        """
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn
    
    # ========== 版本管理 API ==========
    
    def get_active_version(self) -> Optional[str]:
        """
        获取当前激活的版本 ID
        
        Returns:
            version_id 或 None（数据库未初始化）
        """
        cur = self.execute("SELECT version_id FROM active_version WHERE id = 1")
        row = cur.fetchone()
        return row["version_id"] if row else None
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出版本历史（按激活时间倒序）
        
        Args:
            limit: 返回数量
        
        Returns:
            版本列表，每项包含 version_id / changeset / evidence / regression_status / activated_at
        """
        sql = """
        SELECT version_id, changeset, evidence, regression_status, 
               datetime(created_at, 'unixepoch', 'localtime') AS created_at,
               datetime(activated_at, 'unixepoch', 'localtime') AS activated_at
        FROM category_versions
        ORDER BY created_at DESC
        LIMIT ?
        """
        cur = self.execute(sql, (limit,))
        return [dict(row) for row in cur.fetchall()]
    
    def activate_version(self, version_id: str) -> bool:
        """
        激活指定版本（更新 active_version 表）
        
        Args:
            version_id: 版本 ID
        
        Returns:
            是否成功
        """
        try:
            # 检查版本是否存在
            cur = self.execute("SELECT 1 FROM category_versions WHERE version_id = ?", (version_id,))
            if cur.fetchone() is None:
                return False
            
            # 更新激活时间
            now = int(datetime.now().timestamp())
            self.execute(
                "UPDATE category_versions SET activated_at = ? WHERE version_id = ?",
                (now, version_id)
            )
            
            # 更新活跃版本
            self.execute(
                "INSERT OR REPLACE INTO active_version (id, version_id, activated_at) VALUES (1, ?, ?)",
                (version_id, now)
            )
            
            self.commit()
            return True
        except Exception as e:
            self.rollback()
            print(f"[ERROR] 激活版本失败：{e}")
            return False
    
    def rollback_to_parent(self, version_id: str) -> Optional[str]:
        """
        回滚到指定版本的父版本
        
        Args:
            version_id: 当前版本 ID
        
        Returns:
            父版本 ID 或 None（无父版本）
        """
        cur = self.execute(
            "SELECT parent_version_id FROM category_versions WHERE version_id = ?",
            (version_id,)
        )
        row = cur.fetchone()
        if row is None or row["parent_version_id"] is None:
            return None
        
        parent_id = row["parent_version_id"]
        if self.activate_version(parent_id):
            return parent_id
        return None
    
    def gc_old_versions(self, days: int = 30) -> int:
        """
        垃圾回收：删除 N 天前创建且未激活的版本
        
        Args:
            days: 天数阈值
        
        Returns:
            删除的版本数量
        """
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # 删除未激活的旧版本（保留快照 JSON）
        cur = self.execute(
            """
            DELETE FROM category_versions
            WHERE activated_at IS NULL
              AND created_at < ?
            """,
            (cutoff,)
        )
        deleted = cur.rowcount
        self.commit()
        
        return deleted
    
    # ========== 类目查询 API ==========
    
    def get_categories_by_material(
        self,
        material_family: str,
        limit: int = 50,
        min_affinity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        按材质亲和度查询类目（Top-K）
        
        Args:
            material_family: 材质家族（如"宣纸水墨"）
            limit: 返回数量
            min_affinity: 最低亲和度阈值
        
        Returns:
            类目列表，每项包含 category_id / name_zh / name_en / confidence / affinity
        """
        sql = """
        SELECT c.id AS category_id, c.name_zh, c.name_en, 
               c.name_template, c.confidence, a.affinity
        FROM categories c
        JOIN category_material_affinity a ON c.id = a.category_id
        WHERE a.material_family = ? AND a.affinity >= ?
        ORDER BY a.affinity DESC, c.confidence DESC
        LIMIT ?
        """
        cur = self.execute(sql, (material_family, min_affinity, limit))
        return [dict(row) for row in cur.fetchall()]
    
    def get_category_prompts(
        self,
        category_id: str,
        min_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        获取类目的提示词列表（按权重排序）
        
        Args:
            category_id: 类目 ID
            min_weight: 最低权重阈值
        
        Returns:
            提示词列表，每项包含 prompt / weight / lang / source
        """
        sql = """
        SELECT prompt, weight, lang, source, n_accept, n_reject
        FROM category_prompts
        WHERE category_id = ?
          AND source != 'shadow'
          AND weight >= ?
        ORDER BY weight DESC
        """
        cur = self.execute(sql, (category_id, min_weight))
        return [dict(row) for row in cur.fetchall()]
    
    def get_category_priors(self, category_id: str) -> Optional[Dict[str, Any]]:
        """
        获取类目的形状/面积先验
        
        Returns:
            先验字典或 None（未记录）
        """
        sql = """
        SELECT area_budget_min, area_budget_max, elongation_mean, elongation_std,
               compactness_mean, compactness_std, n_components_mode, n_samples
        FROM category_priors
        WHERE category_id = ?
        """
        cur = self.execute(sql, (category_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ========== 辅助函数 ==========

def create_db_manager(db_path: str = "webui/data/adaptive_semantics.db") -> DBManager:
    """
    创建数据库管理器（单例模式，可选）
    
    Args:
        db_path: 数据库路径
    
    Returns:
        DBManager 实例
    """
    return DBManager(db_path)
