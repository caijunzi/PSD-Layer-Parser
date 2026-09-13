# 自进化语义匹配机制 - 数据结构与种子数据

## 文档信息
- **版本**: v1.0-draft
- **日期**: 2026-09-13
- **依赖**: 01-architecture.md
- **交付物**: DDL 脚本 + 种子数据 SQL + 迁移脚本

---

## 一、数据库 DDL（SQLite）

### 完整建表脚本

**路径**: `engine/adaptive/schema.sql`

```sql
-- ============================================================
-- 自进化语义匹配机制 - 数据库模式 v1.0
-- 创建时间: 2026-09-13
-- 数据库文件: webui/data/adaptive_semantics.db
-- ============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;  -- 写前日志，提升并发读性能

-- ------------------------------------------------------------
-- 表 1: 类目原型（唯一持久实体）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS categories (
  id TEXT PRIMARY KEY,                    -- "mountains_cliffs"
  name_zh TEXT NOT NULL,                  -- "山石崖壁"
  name_en TEXT,                           -- "Mountains_Cliffs"
  parent_id TEXT,                         -- 类目树父节点（扁平树初期为 NULL）
  path TEXT,                              -- "/山水画/山石" 路径枚举
  name_template TEXT NOT NULL,            -- "NN_中文_English"
  confidence REAL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(parent_id) REFERENCES categories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_cat_parent ON categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_cat_confidence ON categories(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_cat_path ON categories(path);

-- ------------------------------------------------------------
-- 表 2: 类目提示词（逐条 prompt 学权重）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS category_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  weight REAL DEFAULT 1.0 CHECK(weight >= 0),
  lang TEXT DEFAULT 'zh' CHECK(lang IN ('zh', 'en')),
  source TEXT DEFAULT 'seed' CHECK(source IN ('seed', 'mined', 'human', 'shadow')),
  n_hits INTEGER DEFAULT 0 CHECK(n_hits >= 0),
  n_accept INTEGER DEFAULT 0 CHECK(n_accept >= 0),
  n_reject INTEGER DEFAULT 0 CHECK(n_reject >= 0),
  n_images INTEGER DEFAULT 0 CHECK(n_images >= 0),  -- 见过多少张不同图（去重）
  last_hit_at INTEGER,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE,
  UNIQUE(category_id, prompt)  -- 同类目不重复 prompt
);

CREATE INDEX IF NOT EXISTS idx_prompt_cat ON category_prompts(category_id);
CREATE INDEX IF NOT EXISTS idx_prompt_weight ON category_prompts(category_id, weight DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_source ON category_prompts(source);

-- ------------------------------------------------------------
-- 表 3: 材质亲和度（材质家族 × 类目）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS category_material_affinity (
  category_id TEXT NOT NULL,
  material_family TEXT NOT NULL,          -- "金地屏风" / "宣纸水墨" / ...
  affinity REAL DEFAULT 0.5 CHECK(affinity >= 0 AND affinity <= 1),
  PRIMARY KEY(category_id, material_family),
  FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_affinity_mat ON category_material_affinity(material_family, affinity DESC);

-- ------------------------------------------------------------
-- 表 4: 形状/面积先验（防假阳性）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS category_priors (
  category_id TEXT PRIMARY KEY,
  area_budget_min REAL CHECK(area_budget_min >= 0 AND area_budget_min <= 1),
  area_budget_max REAL CHECK(area_budget_max >= 0 AND area_budget_max <= 1),
  elongation_mean REAL,                   -- 长宽比均值
  elongation_std REAL,
  compactness_mean REAL,                  -- 紧密度均值
  compactness_std REAL,
  n_components_mode INTEGER,              -- 连通域数量众数
  n_samples INTEGER DEFAULT 0,            -- 统计样本数（护栏：≥5 才可信）
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 表 5: 负样本库（假阳性模式）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS negative_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  context_fingerprint BLOB,               -- 128维指纹（何种图会假阳性）
  reject_reason TEXT NOT NULL,            -- 来自质量门/弥散门
  created_at INTEGER NOT NULL,
  FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_neg_cat ON negative_samples(category_id);
CREATE INDEX IF NOT EXISTS idx_neg_created ON negative_samples(created_at DESC);

-- ------------------------------------------------------------
-- 表 6: 词库版本链（Git 式版本管理）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS category_versions (
  version_id TEXT PRIMARY KEY,            -- "v2026-09-13T12:34:56Z"
  parent_version_id TEXT,                 -- 父版本（回滚用）
  snapshot_json TEXT NOT NULL,            -- 完整快照（压缩 JSON）
  changeset TEXT,                         -- 变更摘要（人类可读）
  evidence TEXT,                          -- 证据摘要（多少图/一致性）
  regression_status TEXT DEFAULT 'pending' CHECK(regression_status IN ('pending', 'passed', 'failed')),
  created_at INTEGER NOT NULL,
  activated_at INTEGER,                   -- 发布时间（NULL = 未激活）
  FOREIGN KEY(parent_version_id) REFERENCES category_versions(version_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_ver_activated ON category_versions(activated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ver_status ON category_versions(regression_status);

-- ------------------------------------------------------------
-- 表 7: 活跃版本标记（单行表，只存当前版本）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS active_version (
  id INTEGER PRIMARY KEY CHECK(id = 1),   -- 强制单行
  version_id TEXT NOT NULL,
  activated_at INTEGER NOT NULL,
  FOREIGN KEY(version_id) REFERENCES category_versions(version_id)
);

-- ------------------------------------------------------------
-- 视图：带权重排序的类目提示词（供引擎查询）
-- ------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_category_prompts_ranked AS
SELECT
  cp.category_id,
  cp.prompt,
  cp.weight,
  cp.lang,
  cp.n_accept,
  cp.n_reject,
  cp.n_images,
  c.name_zh,
  c.name_en,
  c.confidence AS category_confidence
FROM category_prompts cp
JOIN categories c ON cp.category_id = c.id
WHERE cp.source != 'shadow'  -- 影子 prompt 不参与正式检出
  AND cp.weight >= 0.3       -- 低权重 prompt 不启用
ORDER BY cp.category_id, cp.weight DESC;

-- ------------------------------------------------------------
-- 触发器：更新 categories.updated_at
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_cat_update_timestamp
AFTER UPDATE ON categories
FOR EACH ROW
BEGIN
  UPDATE categories SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

-- ------------------------------------------------------------
-- 触发器：更新 category_priors.updated_at
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_prior_update_timestamp
AFTER UPDATE ON category_priors
FOR EACH ROW
BEGIN
  UPDATE category_priors SET updated_at = strftime('%s', 'now') WHERE category_id = NEW.category_id;
END;
```

---

## 二、种子数据设计

### 2.1 五类材质家族（人工定义）

| material_family | 描述 | 典型背景色（L*a*b*） |
|---|---|---|
| `金地屏风` | 金箔/金粉底，华丽装饰性 | L*>80, b*>20（黄） |
| `宣纸水墨` | 白纸/米黄底，墨色为主 | L*>90, C*<10（低饱和）|
| `绢本工笔` | 细腻绢本，设色工整 | L*70–85, 中饱和 |
| `油画布` | 西式油画，厚重笔触 | 多样，边缘密度高 |
| `其他` | 兜底类（织锦/壁画等） | — |

### 2.2 通用类目原型（20 个 × 5 材质 = 100 条亲和度）

**种子数据 SQL**: `engine/adaptive/seed_data.sql`

```sql
-- ============================================================
-- 种子数据：20 个通用类目 + 100 条材质亲和度
-- 来源：从 4 个现有预设（chinese_ink_landscape_ai / japanese_screen_gold 等）
--       统计共现频率 + 人工校准
-- ============================================================

-- ------------------------------------------------------------
-- 插入 20 个通用类目（扁平树，parent_id=NULL）
-- ------------------------------------------------------------
INSERT INTO categories (id, name_zh, name_en, parent_id, path, name_template, confidence, created_at, updated_at) VALUES
('distant_mountains',    '远山',         'Distant_Mountains',         NULL, '/', 'NN_远山_Distant_Mountains',         0.85, strftime('%s','now'), strftime('%s','now')),
('mountains_cliffs',     '山石崖壁',     'Mountains_Cliffs',          NULL, '/', 'NN_山石崖壁_Mountains_Cliffs',      0.88, strftime('%s','now'), strftime('%s','now')),
('trees_vegetation',     '树木植被',     'Trees_Vegetation',          NULL, '/', 'NN_树木植被_Trees_Vegetation',      0.90, strftime('%s','now'), strftime('%s','now')),
('water_ripples',        '水波纹',       'Water_Ripples',             NULL, '/', 'NN_水波_Water_Ripples',             0.80, strftime('%s','now'), strftime('%s','now')),
('architecture',         '建筑亭台',     'Architecture_Pavilion',     NULL, '/', 'NN_建筑亭台_Architecture_Pavilion', 0.82, strftime('%s','now'), strftime('%s','now')),
('figures_scholar',      '文人高士',     'Figures_Scholar',           NULL, '/', 'NN_文人高士_Figures_Scholar',       0.78, strftime('%s','now'), strftime('%s','now')),
('fauna_birds',          '鸟禽',         'Fauna_Birds',               NULL, '/', 'NN_鸟禽_Fauna_Birds',               0.83, strftime('%s','now'), strftime('%s','now')),
('fauna_geese',          '雁',           'Fauna_Geese',               NULL, '/', 'NN_雁_Fauna_Geese',                 0.75, strftime('%s','now'), strftime('%s','now')),
('calligraphy',          '题跋书法',     'Calligraphy_Inscription',   NULL, '/', 'NN_题跋书法_Calligraphy',           0.92, strftime('%s','now'), strftime('%s','now')),
('seal',                 '印章',         'Cinnabar_Seal',             NULL, '/', 'NN_印章_Cinnabar_Seal',             0.95, strftime('%s','now'), strftime('%s','now')),
('brocade_frame',        '织锦边框',     'Brocade_Outer_Frame',       NULL, '/', 'NN_织锦边框_Brocade_Outer_Frame',   0.70, strftime('%s','now'), strftime('%s','now')),
('panel_folds',          '屏风折痕',     'Panel_Fold_Seams',          NULL, '/', 'NN_屏风折痕_Panel_Fold_Seams',      0.65, strftime('%s','now'), strftime('%s','now')),
('clouds_mist',          '云雾',         'Clouds_Mist',               NULL, '/', 'NN_云雾_Clouds_Mist',               0.72, strftime('%s','now'), strftime('%s','now')),
('boats_ships',          '舟船',         'Boats_Ships',               NULL, '/', 'NN_舟船_Boats_Ships',               0.68, strftime('%s','now'), strftime('%s','now')),
('figures_fisherman',    '渔人',         'Figures_Fisherman',         NULL, '/', 'NN_渔人_Figures_Fisherman',         0.70, strftime('%s','now'), strftime('%s','now')),
('stone_steps',          '石阶',         'Stone_Steps',               NULL, '/', 'NN_石阶_Stone_Steps',               0.65, strftime('%s','now'), strftime('%s','now')),
('waterfalls',           '瀑布',         'Waterfalls',                NULL, '/', 'NN_瀑布_Waterfalls',                0.74, strftime('%s','now'), strftime('%s','now')),
('bamboo',               '竹',           'Bamboo',                    NULL, '/', 'NN_竹_Bamboo',                      0.76, strftime('%s','now'), strftime('%s','now')),
('plum_blossom',         '梅花',         'Plum_Blossom',              NULL, '/', 'NN_梅花_Plum_Blossom',              0.73, strftime('%s','now'), strftime('%s','now')),
('repair_marks',         '修复痕迹',     'Repair_Marks',              NULL, '/', 'NN_修复痕迹_Repair_Marks',          0.88, strftime('%s','now'), strftime('%s','now'));

-- ------------------------------------------------------------
-- 插入材质亲和度（100 条：20 类目 × 5 材质）
-- 规则：
--   - 金地屏风：装饰性强（鸟/花/建筑高亲和），自然元素中等
--   - 宣纸水墨：自然元素高（山水树云），装饰低
--   - 绢本工笔：人物/细节高，粗犷元素低
--   - 油画布：西式题材高（建筑/人物），东方符号低
--   - 其他：均衡
-- ------------------------------------------------------------
INSERT INTO category_material_affinity (category_id, material_family, affinity) VALUES
-- 远山（宣纸水墨最高）
('distant_mountains', '金地屏风', 0.6), ('distant_mountains', '宣纸水墨', 0.95), ('distant_mountains', '绢本工笔', 0.75), ('distant_mountains', '油画布', 0.50), ('distant_mountains', '其他', 0.60),
-- 山石崖壁（宣纸/绢本高）
('mountains_cliffs', '金地屏风', 0.65), ('mountains_cliffs', '宣纸水墨', 0.92), ('mountains_cliffs', '绢本工笔', 0.88), ('mountains_cliffs', '油画布', 0.70), ('mountains_cliffs', '其他', 0.65),
-- 树木植被（通用高）
('trees_vegetation', '金地屏风', 0.80), ('trees_vegetation', '宣纸水墨', 0.90), ('trees_vegetation', '绢本工笔', 0.85), ('trees_vegetation', '油画布', 0.75), ('trees_vegetation', '其他', 0.75),
-- 水波（宣纸高）
('water_ripples', '金地屏风', 0.55), ('water_ripples', '宣纸水墨', 0.88), ('water_ripples', '绢本工笔', 0.72), ('water_ripples', '油画布', 0.60), ('water_ripples', '其他', 0.60),
-- 建筑（金地/工笔高）
('architecture', '金地屏风', 0.92), ('architecture', '宣纸水墨', 0.70), ('architecture', '绢本工笔', 0.88), ('architecture', '油画布', 0.85), ('architecture', '其他', 0.75),
-- 文人高士（绢本/宣纸高）
('figures_scholar', '金地屏风', 0.75), ('figures_scholar', '宣纸水墨', 0.82), ('figures_scholar', '绢本工笔', 0.95), ('figures_scholar', '油画布', 0.65), ('figures_scholar', '其他', 0.70),
-- 鸟禽（金地高）
('fauna_birds', '金地屏风', 0.95), ('fauna_birds', '宣纸水墨', 0.75), ('fauna_birds', '绢本工笔', 0.88), ('fauna_birds', '油画布', 0.60), ('fauna_birds', '其他', 0.70),
-- 雁（金地高）
('fauna_geese', '金地屏风', 0.92), ('fauna_geese', '宣纸水墨', 0.78), ('fauna_geese', '绢本工笔', 0.85), ('fauna_geese', '油画布', 0.55), ('fauna_geese', '其他', 0.68),
-- 题跋（通用必检）
('calligraphy', '金地屏风', 0.88), ('calligraphy', '宣纸水墨', 0.95), ('calligraphy', '绢本工笔', 0.92), ('calligraphy', '油画布', 0.70), ('calligraphy', '其他', 0.80),
-- 印章（通用必检）
('seal', '金地屏风', 0.90), ('seal', '宣纸水墨', 0.98), ('seal', '绢本工笔', 0.95), ('seal', '油画布', 0.65), ('seal', '其他', 0.82),
-- 织锦边框（金地独高）
('brocade_frame', '金地屏风', 0.98), ('brocade_frame', '宣纸水墨', 0.40), ('brocade_frame', '绢本工笔', 0.55), ('brocade_frame', '油画布', 0.45), ('brocade_frame', '其他', 0.85),
-- 屏风折痕（金地独高）
('panel_folds', '金地屏风', 0.95), ('panel_folds', '宣纸水墨', 0.30), ('panel_folds', '绢本工笔', 0.40), ('panel_folds', '油画布', 0.25), ('panel_folds', '其他', 0.50),
-- 云雾（宣纸高）
('clouds_mist', '金地屏风', 0.60), ('clouds_mist', '宣纸水墨', 0.90), ('clouds_mist', '绢本工笔', 0.78), ('clouds_mist', '油画布', 0.70), ('clouds_mist', '其他', 0.65),
-- 舟船（宣纸/绢本中）
('boats_ships', '金地屏风', 0.70), ('boats_ships', '宣纸水墨', 0.82), ('boats_ships', '绢本工笔', 0.80), ('boats_ships', '油画布', 0.65), ('boats_ships', '其他', 0.68),
-- 渔人（宣纸高）
('figures_fisherman', '金地屏风', 0.65), ('figures_fisherman', '宣纸水墨', 0.88), ('figures_fisherman', '绢本工笔', 0.82), ('figures_fisherman', '油画布', 0.60), ('figures_fisherman', '其他', 0.68),
-- 石阶（通用中）
('stone_steps', '金地屏风', 0.68), ('stone_steps', '宣纸水墨', 0.75), ('stone_steps', '绢本工笔', 0.78), ('stone_steps', '油画布', 0.72), ('stone_steps', '其他', 0.70),
-- 瀑布（宣纸高）
('waterfalls', '金地屏风', 0.60), ('waterfalls', '宣纸水墨', 0.92), ('waterfalls', '绢本工笔', 0.80), ('waterfalls', '油画布', 0.68), ('waterfalls', '其他', 0.70),
-- 竹（通用高）
('bamboo', '金地屏风', 0.75), ('bamboo', '宣纸水墨', 0.95), ('bamboo', '绢本工笔', 0.88), ('bamboo', '油画布', 0.60), ('bamboo', '其他', 0.72),
-- 梅花（通用高）
('plum_blossom', '金地屏风', 0.85), ('plum_blossom', '宣纸水墨', 0.90), ('plum_blossom', '绢本工笔', 0.92), ('plum_blossom', '油画布', 0.58), ('plum_blossom', '其他', 0.70),
-- 修复痕迹（通用必检）
('repair_marks', '金地屏风', 0.85), ('repair_marks', '宣纸水墨', 0.88), ('repair_marks', '绢本工笔', 0.90), ('repair_marks', '油画布', 0.82), ('repair_marks', '其他', 0.85);

-- ------------------------------------------------------------
-- 插入类目提示词（平均每类目 2–3 条，共 ~50 条）
-- 来源：从 4 个预设的 ai_semantic_classes[].prompt 统计 + 人工补充
-- ------------------------------------------------------------
INSERT INTO category_prompts (category_id, prompt, weight, lang, source, created_at) VALUES
-- 远山
('distant_mountains', '远山', 1.0, 'zh', 'seed', strftime('%s','now')),
('distant_mountains', 'distant mountains', 0.9, 'en', 'seed', strftime('%s','now')),
-- 山石崖壁
('mountains_cliffs', '山石', 1.2, 'zh', 'seed', strftime('%s','now')),
('mountains_cliffs', '崖壁', 1.0, 'zh', 'seed', strftime('%s','now')),
('mountains_cliffs', 'cliffs', 0.95, 'en', 'seed', strftime('%s','now')),
-- 树木植被
('trees_vegetation', '树木', 1.1, 'zh', 'seed', strftime('%s','now')),
('trees_vegetation', '植被', 0.95, 'zh', 'seed', strftime('%s','now')),
('trees_vegetation', 'trees', 1.0, 'en', 'seed', strftime('%s','now')),
-- 水波
('water_ripples', '水波', 1.0, 'zh', 'seed', strftime('%s','now')),
('water_ripples', '涟漪', 0.85, 'zh', 'seed', strftime('%s','now')),
-- 建筑
('architecture', '建筑', 1.1, 'zh', 'seed', strftime('%s','now')),
('architecture', '亭台', 1.0, 'zh', 'seed', strftime('%s','now')),
('architecture', 'pavilion', 0.95, 'en', 'seed', strftime('%s','now')),
-- 文人高士
('figures_scholar', '文人', 1.0, 'zh', 'seed', strftime('%s','now')),
('figures_scholar', '高士', 0.95, 'zh', 'seed', strftime('%s','now')),
('figures_scholar', 'scholar', 0.9, 'en', 'seed', strftime('%s','now')),
-- 鸟禽
('fauna_birds', '鸟', 1.2, 'zh', 'seed', strftime('%s','now')),
('fauna_birds', '禽', 0.9, 'zh', 'seed', strftime('%s','now')),
-- 雁
('fauna_geese', '雁', 1.1, 'zh', 'seed', strftime('%s','now')),
('fauna_geese', 'geese', 1.0, 'en', 'seed', strftime('%s','now')),
-- 题跋
('calligraphy', '题跋', 1.3, 'zh', 'seed', strftime('%s','now')),
('calligraphy', '书法', 1.1, 'zh', 'seed', strftime('%s','now')),
('calligraphy', 'calligraphy', 1.0, 'en', 'seed', strftime('%s','now')),
-- 印章（必检，高权重）
('seal', '印章', 1.5, 'zh', 'seed', strftime('%s','now')),
('seal', '朱砂', 1.2, 'zh', 'seed', strftime('%s','now')),
('seal', 'seal', 1.3, 'en', 'seed', strftime('%s','now')),
-- 织锦边框
('brocade_frame', '织锦边框', 1.0, 'zh', 'seed', strftime('%s','now')),
('brocade_frame', 'brocade', 0.95, 'en', 'seed', strftime('%s','now')),
-- 屏风折痕
('panel_folds', '折痕', 1.0, 'zh', 'seed', strftime('%s','now')),
('panel_folds', '屏风缝', 0.85, 'zh', 'seed', strftime('%s','now')),
-- 云雾
('clouds_mist', '云', 1.0, 'zh', 'seed', strftime('%s','now')),
('clouds_mist', '雾', 0.95, 'zh', 'seed', strftime('%s','now')),
-- 舟船
('boats_ships', '舟', 1.0, 'zh', 'seed', strftime('%s','now')),
('boats_ships', '船', 1.0, 'zh', 'seed', strftime('%s','now')),
-- 渔人
('figures_fisherman', '渔人', 1.1, 'zh', 'seed', strftime('%s','now')),
('figures_fisherman', 'fisherman', 0.95, 'en', 'seed', strftime('%s','now')),
-- 石阶
('stone_steps', '石阶', 1.0, 'zh', 'seed', strftime('%s','now')),
('stone_steps', 'stone steps', 0.9, 'en', 'seed', strftime('%s','now')),
-- 瀑布
('waterfalls', '瀑布', 1.2, 'zh', 'seed', strftime('%s','now')),
('waterfalls', 'waterfall', 1.0, 'en', 'seed', strftime('%s','now')),
-- 竹
('bamboo', '竹', 1.1, 'zh', 'seed', strftime('%s','now')),
('bamboo', 'bamboo', 1.0, 'en', 'seed', strftime('%s','now')),
-- 梅花
('plum_blossom', '梅花', 1.1, 'zh', 'seed', strftime('%s','now')),
('plum_blossom', 'plum blossom', 1.0, 'en', 'seed', strftime('%s','now')),
-- 修复痕迹（必检）
('repair_marks', '修复痕迹', 1.4, 'zh', 'seed', strftime('%s','now')),
('repair_marks', '破损', 1.2, 'zh', 'seed', strftime('%s','now')),
('repair_marks', 'repair', 1.1, 'en', 'seed', strftime('%s','now'));

-- ------------------------------------------------------------
-- 插入初始版本（v0-seed）
-- ------------------------------------------------------------
INSERT INTO category_versions (version_id, parent_version_id, snapshot_json, changeset, evidence, regression_status, created_at, activated_at)
VALUES (
  'v0-seed',
  NULL,
  '{"categories": 20, "prompts": 50, "affinity": 100, "source": "human_curated"}',
  '初始化：20 类目 + 50 提示词 + 100 材质亲和度',
  '人工校准种子数据',
  'passed',
  strftime('%s', 'now'),
  strftime('%s', 'now')
);

-- 激活初始版本
INSERT INTO active_version (id, version_id, activated_at)
VALUES (1, 'v0-seed', strftime('%s', 'now'));
```

---

## 三、数据迁移脚本

**路径**: `engine/adaptive/migrations/migration_001_init.py`

```python
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
```

**执行方式**:
```bash
cd "E:/CK/C-desk/CK-WORKS/PSD图层处理 - WorkBuddy"
python engine/adaptive/migrations/migration_001_init.py
```

---

## 四、从现有预设统计生成补充种子数据

**脚本**: `tools/generate_affinity_from_presets.py`（可选，阶段 1 后执行）

```python
"""
从 4 个现有预设统计材质×类目共现频率，补充/校准 material_affinity

用法:
  python tools/generate_affinity_from_presets.py --update-db
"""
import json
from pathlib import Path
import sqlite3
from collections import defaultdict

def analyze_presets():
    presets_dir = Path("presets")
    preset_files = [
        "chinese_ink_landscape_ai.json",
        "japanese_screen_gold.json",
        # 其他预设...
    ]
    
    # 材质家族判别规则（简化）
    material_map = {
        "chinese_ink_landscape_ai": "宣纸水墨",
        "japanese_screen_gold": "金地屏风",
        # ...
    }
    
    # 统计: {material_family: {category_key: count}}
    stats = defaultdict(lambda: defaultdict(int))
    
    for pf in preset_files:
        preset = json.loads((presets_dir / pf).read_text(encoding="utf-8"))
        material = material_map.get(pf.replace(".json", ""), "其他")
        
        for cls in preset.get("ai_semantic_classes", []):
            # 从 layer_name 提取类目键（如 "03_远山_Distant_Mountains" → "distant_mountains"）
            layer_name = cls.get("layer_name", "")
            # 简化：取英文部分转小写下划线
            parts = layer_name.split("_")
            if len(parts) >= 3:
                cat_key = "_".join(parts[2:]).lower()
                stats[material][cat_key] += 1
    
    return stats

def update_affinity_from_stats(db_path: str, stats: dict):
    """
    用统计结果更新 category_material_affinity（仅补充，不覆盖种子数据）
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for material, cat_counts in stats.items():
        total = sum(cat_counts.values())
        for cat_key, count in cat_counts.items():
            # 归一化为 0..1
            affinity = count / total
            
            # 检查是否已有记录
            cur.execute(
                "SELECT affinity FROM category_material_affinity WHERE category_id=? AND material_family=?",
                (cat_key, material)
            )
            row = cur.fetchone()
            
            if row is None:
                # 插入新记录
                cur.execute(
                    "INSERT INTO category_material_affinity (category_id, material_family, affinity) VALUES (?, ?, ?)",
                    (cat_key, material, affinity)
                )
                print(f"[新增] {material} × {cat_key} = {affinity:.2f}")
            else:
                # 已有记录，可选：加权平均（种子 0.7 + 统计 0.3）
                old = row[0]
                new = old * 0.7 + affinity * 0.3
                cur.execute(
                    "UPDATE category_material_affinity SET affinity=? WHERE category_id=? AND material_family=?",
                    (new, cat_key, material)
                )
                print(f"[更新] {material} × {cat_key}: {old:.2f} → {new:.2f}")
    
    conn.commit()
    conn.close()
    print("[OK] 材质亲和度更新完成")

if __name__ == "__main__":
    import sys
    stats = analyze_presets()
    
    if "--update-db" in sys.argv:
        db_path = "webui/data/adaptive_semantics.db"
        update_affinity_from_stats(db_path, stats)
    else:
        # 仅打印统计
        import pprint
        pprint.pprint(dict(stats))
```

---

## 五、数据完整性校验

**脚本**: `engine/adaptive/validate_db.py`

```python
"""
校验数据库完整性（CI/测试前执行）
"""
import sqlite3
import sys

def validate_db(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    checks = []
    
    # 检查 1: 类目数量 ≥ 20
    cur.execute("SELECT COUNT(*) FROM categories")
    n_cat = cur.fetchone()[0]
    checks.append(("类目数量 ≥ 20", n_cat >= 20, n_cat))
    
    # 检查 2: 提示词数量 ≥ 40
    cur.execute("SELECT COUNT(*) FROM category_prompts WHERE source='seed'")
    n_prompt = cur.fetchone()[0]
    checks.append(("种子提示词 ≥ 40", n_prompt >= 40, n_prompt))
    
    # 检查 3: 材质亲和度 ≥ 100
    cur.execute("SELECT COUNT(*) FROM category_material_affinity")
    n_affinity = cur.fetchone()[0]
    checks.append(("材质亲和度 ≥ 100", n_affinity >= 100, n_affinity))
    
    # 检查 4: 激活版本存在
    cur.execute("SELECT version_id FROM active_version WHERE id=1")
    row = cur.fetchone()
    checks.append(("激活版本存在", row is not None, row[0] if row else None))
    
    # 检查 5: 所有类目至少有 1 个提示词
    cur.execute("""
        SELECT c.id FROM categories c
        LEFT JOIN category_prompts cp ON c.id = cp.category_id
        GROUP BY c.id HAVING COUNT(cp.id) = 0
    """)
    orphan_cats = cur.fetchall()
    checks.append(("无孤儿类目（无提示词）", len(orphan_cats) == 0, orphan_cats))
    
    # 检查 6: 核心类目（印章/题跋/修复痕迹）confidence ≥ 0.85
    cur.execute("""
        SELECT id, confidence FROM categories
        WHERE id IN ('seal', 'calligraphy', 'repair_marks')
          AND confidence < 0.85
    """)
    low_conf = cur.fetchall()
    checks.append(("核心类目置信度 ≥ 0.85", len(low_conf) == 0, low_conf))
    
    conn.close()
    
    # 打印结果
    all_pass = True
    for desc, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {desc}: {detail}")
        if not passed:
            all_pass = False
    
    return all_pass

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "webui/data/adaptive_semantics.db"
    passed = validate_db(db_path)
    sys.exit(0 if passed else 1)
```

---

**下一步**: 阅读 `03-implementation-plan.md`（分阶段文件清单 + 测试用例 + 开发路线图）。
