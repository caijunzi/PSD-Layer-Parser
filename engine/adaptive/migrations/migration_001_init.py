"""
迁移脚本 001: 初始化自进化语义匹配机制数据库

执行时机: 阶段 1 启动前
依赖: schema.sql / seed_data.sql
"""
import sqlite3
from pathlib import Path
import sys

def run_migration(db_path: str, sql_dir: str):
    """
    执行初始化迁移
    """
    db_path = Path(db_path)
    sql_dir = Path(sql_dir)
    
    # 1. 检查数据库是否已存在
    if db_path.exists():
        print(f"[INFO] 数据库已存在: {db_path}")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # 检查是否已迁移（查 categories 表）
        try:
            cur.execute("SELECT COUNT(*) FROM categories")
            count = cur.fetchone()[0]
            if count > 0:
                print(f"[INFO] 数据库已初始化（{count} 个类目），跳过迁移")
                conn.close()
                return
        except sqlite3.OperationalError:
            # 表不存在，继续迁移
            pass
        conn.close()
    
    # 2. 创建数据库目录
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. 连接数据库
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        # 4. 执行 schema.sql
        schema_sql = (sql_dir / "schema.sql").read_text(encoding="utf-8")
        cur.executescript(schema_sql)
        print("[OK] 数据库模式创建完成")
        
        # 5. 执行 seed_data.sql
        seed_sql = (sql_dir / "seed_data.sql").read_text(encoding="utf-8")
        cur.executescript(seed_sql)
        print("[OK] 种子数据导入完成")
        
        # 6. 验证
        cur.execute("SELECT COUNT(*) FROM categories")
        n_cat = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM category_prompts")
        n_prompt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM category_material_affinity")
        n_affinity = cur.fetchone()[0]
        
        print(f"[验证] 类目: {n_cat}, 提示词: {n_prompt}, 材质亲和度: {n_affinity}")
        
        if n_cat < 20 or n_prompt < 40:
            raise ValueError("种子数据不完整")
        
        conn.commit()
        print("[SUCCESS] 迁移 001 完成")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 迁移失败: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    # 默认路径（从项目根目录执行）
    db_path = "webui/data/adaptive_semantics.db"
    sql_dir = "engine/adaptive"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        sql_dir = sys.argv[2]
    
    run_migration(db_path, sql_dir)
