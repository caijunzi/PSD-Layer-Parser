"""迁移脚本 004：人审反馈操作支持（Stage 5.3 补完）

为 categories 表新增 deleted_at 字段，支撑 rename / delete / merge 三种人审操作：
- rename: 更新 name_zh / name_en / name_template
- delete: 软删除（置 deleted_at 时间戳），保留历史引用不悬空
- merge:  源类目 prompts 迁移到目标类目 + 软删源类目

采用**软删除**而非硬删除的原因：
1. categories 被 category_prompts / category_material_affinity / category_priors 外键引用，
   硬删会级联清空学习到的权重（ON DELETE CASCADE），人审误操作无法挽回。
2. 历史 episode JSONL 里记录了 task → category_id，硬删会让历史回放松散引用。
3. 软删后由查询层过滤（db_manager 主查询加 deleted_at IS NULL），等价于"不可见"。

执行：python -m engine.adaptive.migrations.migration_004_feedback_ops
"""
import sqlite3
from pathlib import Path


def apply_migration(db_path: str = "webui/data/adaptive_semantics.db"):
    """应用迁移：新增 deleted_at 字段（幂等）"""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    conn = sqlite3.connect(db_path)
    try:
        print("=== 迁移 004：人审反馈操作支持 ===\n")

        # 1. 检查字段是否已存在（幂等）
        cols = [r[1] for r in conn.execute("PRAGMA table_info(categories)").fetchall()]
        if "deleted_at" in cols:
            print("   跳过（已存在）：categories.deleted_at")
        else:
            conn.execute("ALTER TABLE categories ADD COLUMN deleted_at INTEGER")
            print("   ✅ 新增字段：categories.deleted_at INTEGER（NULL = 未删除）")

        conn.commit()

        # 2. 验证
        cols_after = [r[1] for r in conn.execute("PRAGMA table_info(categories)").fetchall()]
        total = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        alive = conn.execute(
            "SELECT COUNT(*) FROM categories WHERE deleted_at IS NULL"
        ).fetchone()[0]

        print("\n=== 迁移完成 ===")
        print(f"- categories 字段：{cols_after}")
        print(f"- 总类目数：{total}；未删除：{alive}")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ 迁移失败：{e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    apply_migration()
