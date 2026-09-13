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
  deleted_at INTEGER,                    -- Stage 5.3：人审软删除时间戳（NULL = 未删除）
  FOREIGN KEY(parent_id) REFERENCES categories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_cat_parent ON categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_cat_confidence ON categories(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_cat_path ON categories(path);
CREATE INDEX IF NOT EXISTS idx_cat_deleted ON categories(deleted_at);

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
