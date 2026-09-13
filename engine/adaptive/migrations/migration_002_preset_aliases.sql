-- ============================================================
-- Migration 002: preset 命名空间桥接
-- 创建时间: 2026-09-13
-- 依赖: migration_001_init.sql
-- ============================================================

-- ------------------------------------------------------------
-- 表: 类目 preset 别名映射
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS category_preset_aliases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id TEXT NOT NULL,
  preset_name TEXT NOT NULL,              -- preset ai_semantic_classes[].name 格式
  preset_id TEXT,                         -- 预设标识符（如 "japanese_screen_gold"）
  priority INTEGER DEFAULT 100,           -- 优先级（数字越小越优先，用于同类目多别名）
  source TEXT DEFAULT 'manual' CHECK(source IN ('manual', 'mined', 'inferred')),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE,
  UNIQUE(category_id, preset_name)        -- 同类目不重复别名
);

CREATE INDEX IF NOT EXISTS idx_alias_cat ON category_preset_aliases(category_id);
CREATE INDEX IF NOT EXISTS idx_alias_preset ON category_preset_aliases(preset_id);
CREATE INDEX IF NOT EXISTS idx_alias_priority ON category_preset_aliases(category_id, priority ASC);

-- ------------------------------------------------------------
-- 视图: 类目完整信息（含首选 preset 别名）
-- ------------------------------------------------------------
CREATE VIEW IF NOT EXISTS category_with_aliases AS
SELECT 
  c.id,
  c.name_zh,
  c.name_en,
  c.name_template,
  c.confidence,
  GROUP_CONCAT(a.preset_name, '|') as preset_aliases,
  (
    SELECT a2.preset_name 
    FROM category_preset_aliases a2 
    WHERE a2.category_id = c.id 
    ORDER BY a2.priority ASC 
    LIMIT 1
  ) as primary_preset_name
FROM categories c
LEFT JOIN category_preset_aliases a ON c.id = a.category_id
GROUP BY c.id;
