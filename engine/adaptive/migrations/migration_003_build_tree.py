"""迁移脚本 003：建立类目树（Stage 5.1）

将 20 个扁平类目挂成 3 层树：
  根"山水画"
  ├─ 山（distant_mountains / mountains_cliffs / peaks_ridges）
  ├─ 水（water_ripples / rivers_streams / waterfalls）
  ├─ 植被（trees_vegetation / grass_ground / bamboo_plants）
  ├─ 建筑（architecture / bridges / boats_vessels）
  ├─ 天象与云雾（clouds_mist / sky_celestial）
  ├─ 人物与动物（figures_people / animals_birds）
  └─ 底板与边框（decorative_borders / base_ground）

执行：python -m engine.adaptive.migrations.migration_003_build_tree
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# 3 层树结构定义
TREE_STRUCTURE = {
    # 根类目（新增）
    "landscape_painting": {
        "name_zh": "山水画",
        "name_en": "Landscape_Painting",
        "parent_id": None,
        "path": "/",
        "name_template": "山水画_Landscape_Painting",
        "confidence": 1.0,
    },
    # 二级类目（7 大类）
    "mountains": {
        "name_zh": "山",
        "name_en": "Mountains",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "山_Mountains",
        "confidence": 0.95,
    },
    "water": {
        "name_zh": "水",
        "name_en": "Water",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "水_Water",
        "confidence": 0.95,
    },
    "vegetation": {
        "name_zh": "植被",
        "name_en": "Vegetation",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "植被_Vegetation",
        "confidence": 0.95,
    },
    "architecture": {
        "name_zh": "建筑",
        "name_en": "Architecture",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "建筑_Architecture",
        "confidence": 0.95,
    },
    "celestial": {
        "name_zh": "天象与云雾",
        "name_en": "Celestial_And_Mist",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "天象与云雾_Celestial_And_Mist",
        "confidence": 0.95,
    },
    "figures_animals": {
        "name_zh": "人物与动物",
        "name_en": "Figures_And_Animals",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "人物与动物_Figures_And_Animals",
        "confidence": 0.95,
    },
    "base_borders": {
        "name_zh": "底板与边框",
        "name_en": "Base_And_Borders",
        "parent_id": "landscape_painting",
        "path": "/landscape_painting/",
        "name_template": "底板与边框_Base_And_Borders",
        "confidence": 0.95,
    },
}

# 三级类目：现有 20 个类目的父子关系映射
LEAF_CATEGORIES = {
    # 山
    "distant_mountains": "mountains",
    "mountains_cliffs": "mountains",
    "peaks_ridges": "mountains",
    # 水
    "water_ripples": "water",
    "rivers_streams": "water",
    "waterfalls": "water",
    # 植被
    "trees_vegetation": "vegetation",
    "grass_ground": "vegetation",
    "bamboo_plants": "vegetation",
    # 建筑
    "architecture": "architecture",  # 注意：二级与三级同名，需特殊处理
    "bridges": "architecture",
    "boats_vessels": "architecture",
    # 天象与云雾
    "clouds_mist": "celestial",
    "sky_celestial": "celestial",
    # 人物与动物
    "figures_people": "figures_animals",
    "animals_birds": "figures_animals",
    # 底板与边框
    "decorative_borders": "base_borders",
    "base_ground": "base_borders",
}


def apply_migration(db_path: str = "webui/data/adaptive_semantics.db"):
    """应用迁移：建立类目树"""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")
    
    conn = sqlite3.connect(db_path)
    now = int(datetime.now(timezone.utc).timestamp())
    
    try:
        print("=== 迁移 003：建立类目树 ===\n")
        
        # 1. 插入根类目与二级类目（8 个新类目）
        print("1. 插入根类目与二级类目（8 个）...")
        for cat_id, cat_data in TREE_STRUCTURE.items():
            # 检查是否已存在
            existing = conn.execute(
                "SELECT id FROM categories WHERE id = ?", (cat_id,)
            ).fetchone()
            
            if existing:
                print(f"   跳过（已存在）：{cat_id}")
                continue
            
            # 插入新类目
            conn.execute(
                """
                INSERT INTO categories (
                    id, name_zh, name_en, parent_id, path, 
                    name_template, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cat_id,
                    cat_data["name_zh"],
                    cat_data["name_en"],
                    cat_data["parent_id"],
                    cat_data["path"],
                    cat_data["name_template"],
                    cat_data["confidence"],
                    now,
                    now,
                ),
            )
            print(f"   ✅ 插入：{cat_id} ({cat_data['name_zh']})")
        
        # 2. 更新三级类目的 parent_id 与 path
        print("\n2. 更新三级类目（20 个）的父子关系...")
        for leaf_id, parent_id in LEAF_CATEGORIES.items():
            # 查询父类目的 path
            parent_path = conn.execute(
                "SELECT path FROM categories WHERE id = ?", (parent_id,)
            ).fetchone()
            
            if not parent_path:
                print(f"   ⚠️ 父类目不存在：{parent_id}（跳过 {leaf_id}）")
                continue
            
            # 新 path = parent_path + leaf_id + "/"
            new_path = parent_path[0] + leaf_id + "/"
            
            # 更新三级类目
            conn.execute(
                "UPDATE categories SET parent_id = ?, path = ?, updated_at = ? WHERE id = ?",
                (parent_id, new_path, now, leaf_id),
            )
            print(f"   ✅ 更新：{leaf_id} → parent={parent_id}, path={new_path}")
        
        # 3. 提交事务
        conn.commit()
        print("\n=== 迁移完成 ===")
        
        # 4. 验证结果
        print("\n验证结果：")
        total = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        with_parent = conn.execute(
            "SELECT COUNT(*) FROM categories WHERE parent_id IS NOT NULL"
        ).fetchone()[0]
        print(f"- 总类目数：{total}")
        print(f"- 有父类目的数量：{with_parent}（≥10 为预期）")
        
        if with_parent < 10:
            print("\n⚠️ 警告：parent_id 非空数量不足 10，迁移可能未完全生效")
        
        # 5. 展示树结构样本
        print("\n树结构样本（前 10 条）：")
        rows = conn.execute(
            "SELECT id, name_zh, parent_id, path FROM categories LIMIT 10"
        ).fetchall()
        for row in rows:
            print(f"  {row[0]:30s} | {row[1]:12s} | parent={row[2] or 'None':20s} | path={row[3]}")
    
    except Exception as e:
        conn.rollback()
        print(f"\n❌ 迁移失败：{e}")
        raise
    
    finally:
        conn.close()


if __name__ == "__main__":
    apply_migration()
