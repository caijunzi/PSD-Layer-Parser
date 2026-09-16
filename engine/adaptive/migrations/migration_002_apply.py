"""
Migration 002: preset 命名空间桥接
执行 migration_002_preset_aliases.sql + seed_data_002_aliases.sql
"""
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# 定位项目根目录
ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "webui" / "data" / "adaptive_semantics.db"
MIGRATION_SQL = Path(__file__).parent / "migration_002_preset_aliases.sql"
SEED_SQL = ROOT / "engine" / "adaptive" / "seed_data_002_aliases.sql"


def apply_migration(db_path: Optional[str] = None):
    """执行 migration 002

    Args:
        db_path: 目标数据库路径；缺省时用生产默认路径（向后兼容）。
                 2026-09-16 新增参数：全新 ADAPTIVE_DB_PATH 首连自动迁移需要指定目标。
    """
    target = Path(db_path) if db_path else DB_PATH
    if not target.exists():
        print(f"❌ 数据库不存在: {target}")
        print("请先运行 migration_001_init.py")
        return False

    conn = sqlite3.connect(str(target))
    conn.execute("PRAGMA foreign_keys = ON")
    
    try:
        # 1. 执行 DDL（创建映射表 + 视图）
        print(f"[1/3] 执行 DDL: {MIGRATION_SQL.name}")
        ddl = MIGRATION_SQL.read_text(encoding="utf-8")
        conn.executescript(ddl)
        
        # 2. 插入种子数据（preset 别名）
        print(f"[2/3] 插入别名种子数据: {SEED_SQL.name}")
        seed = SEED_SQL.read_text(encoding="utf-8")
        conn.executescript(seed)
        
        conn.commit()
        
        # 3. 验证
        print("[3/3] 验证...")
        cur = conn.cursor()
        
        # 验证表创建
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='category_preset_aliases'")
        if not cur.fetchone():
            raise RuntimeError("category_preset_aliases 表未创建")
        
        # 验证别名数量
        cur.execute("SELECT COUNT(*) FROM category_preset_aliases")
        alias_count = cur.fetchone()[0]
        print(f"   ✅ category_preset_aliases: {alias_count} 条别名")
        
        if alias_count < 20:
            raise RuntimeError(f"别名数量不足 ({alias_count} < 20)")
        
        # 验证视图
        cur.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='category_with_aliases'")
        if not cur.fetchone():
            raise RuntimeError("category_with_aliases 视图未创建")
        
        # 抽样查询视图
        cur.execute("""
            SELECT id, name_zh, primary_preset_name 
            FROM category_with_aliases 
            WHERE primary_preset_name IS NOT NULL 
            LIMIT 5
        """)
        samples = cur.fetchall()
        print("   ✅ category_with_aliases 视图可用")
        print("   示例映射:")
        for cat_id, name_zh, preset_name in samples:
            print(f"      {cat_id:20} → {preset_name}")
        
        print("\n✅ Migration 002 完成!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration 002 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = apply_migration()
    sys.exit(0 if success else 1)
