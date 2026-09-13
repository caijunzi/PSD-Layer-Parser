-- ============================================================
-- 种子数据：20 个通用类目 + 100 条材质亲和度 + 50 prompts
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
-- 规则：金地屏风装饰性强、宣纸水墨自然高、绢本工笔细节高、油画布西式高
-- ------------------------------------------------------------
INSERT INTO category_material_affinity (category_id, material_family, affinity) VALUES
-- 远山
('distant_mountains', '金地屏风', 0.6), ('distant_mountains', '宣纸水墨', 0.95), ('distant_mountains', '绢本工笔', 0.75), ('distant_mountains', '油画布', 0.50), ('distant_mountains', '其他', 0.60),
-- 山石崖壁
('mountains_cliffs', '金地屏风', 0.65), ('mountains_cliffs', '宣纸水墨', 0.92), ('mountains_cliffs', '绢本工笔', 0.88), ('mountains_cliffs', '油画布', 0.70), ('mountains_cliffs', '其他', 0.65),
-- 树木植被
('trees_vegetation', '金地屏风', 0.80), ('trees_vegetation', '宣纸水墨', 0.90), ('trees_vegetation', '绢本工笔', 0.85), ('trees_vegetation', '油画布', 0.75), ('trees_vegetation', '其他', 0.75),
-- 水波
('water_ripples', '金地屏风', 0.55), ('water_ripples', '宣纸水墨', 0.88), ('water_ripples', '绢本工笔', 0.72), ('water_ripples', '油画布', 0.60), ('water_ripples', '其他', 0.60),
-- 建筑
('architecture', '金地屏风', 0.92), ('architecture', '宣纸水墨', 0.70), ('architecture', '绢本工笔', 0.88), ('architecture', '油画布', 0.85), ('architecture', '其他', 0.75),
-- 文人高士
('figures_scholar', '金地屏风', 0.75), ('figures_scholar', '宣纸水墨', 0.82), ('figures_scholar', '绢本工笔', 0.95), ('figures_scholar', '油画布', 0.65), ('figures_scholar', '其他', 0.70),
-- 鸟禽
('fauna_birds', '金地屏风', 0.95), ('fauna_birds', '宣纸水墨', 0.75), ('fauna_birds', '绢本工笔', 0.88), ('fauna_birds', '油画布', 0.60), ('fauna_birds', '其他', 0.70),
-- 雁
('fauna_geese', '金地屏风', 0.92), ('fauna_geese', '宣纸水墨', 0.78), ('fauna_geese', '绢本工笔', 0.85), ('fauna_geese', '油画布', 0.55), ('fauna_geese', '其他', 0.68),
-- 题跋（通用必检）
('calligraphy', '金地屏风', 0.88), ('calligraphy', '宣纸水墨', 0.95), ('calligraphy', '绢本工笔', 0.92), ('calligraphy', '油画布', 0.70), ('calligraphy', '其他', 0.80),
-- 印章（通用必检）
('seal', '金地屏风', 0.90), ('seal', '宣纸水墨', 0.98), ('seal', '绢本工笔', 0.95), ('seal', '油画布', 0.65), ('seal', '其他', 0.82),
-- 织锦边框（金地独高）
('brocade_frame', '金地屏风', 0.98), ('brocade_frame', '宣纸水墨', 0.40), ('brocade_frame', '绢本工笔', 0.55), ('brocade_frame', '油画布', 0.45), ('brocade_frame', '其他', 0.85),
-- 屏风折痕（金地独高）
('panel_folds', '金地屏风', 0.95), ('panel_folds', '宣纸水墨', 0.30), ('panel_folds', '绢本工笔', 0.40), ('panel_folds', '油画布', 0.25), ('panel_folds', '其他', 0.50),
-- 云雾
('clouds_mist', '金地屏风', 0.60), ('clouds_mist', '宣纸水墨', 0.90), ('clouds_mist', '绢本工笔', 0.78), ('clouds_mist', '油画布', 0.70), ('clouds_mist', '其他', 0.65),
-- 舟船
('boats_ships', '金地屏风', 0.70), ('boats_ships', '宣纸水墨', 0.82), ('boats_ships', '绢本工笔', 0.80), ('boats_ships', '油画布', 0.65), ('boats_ships', '其他', 0.68),
-- 渔人
('figures_fisherman', '金地屏风', 0.65), ('figures_fisherman', '宣纸水墨', 0.88), ('figures_fisherman', '绢本工笔', 0.82), ('figures_fisherman', '油画布', 0.60), ('figures_fisherman', '其他', 0.68),
-- 石阶
('stone_steps', '金地屏风', 0.68), ('stone_steps', '宣纸水墨', 0.75), ('stone_steps', '绢本工笔', 0.78), ('stone_steps', '油画布', 0.72), ('stone_steps', '其他', 0.70),
-- 瀑布
('waterfalls', '金地屏风', 0.60), ('waterfalls', '宣纸水墨', 0.92), ('waterfalls', '绢本工笔', 0.80), ('waterfalls', '油画布', 0.68), ('waterfalls', '其他', 0.70),
-- 竹
('bamboo', '金地屏风', 0.75), ('bamboo', '宣纸水墨', 0.95), ('bamboo', '绢本工笔', 0.88), ('bamboo', '油画布', 0.60), ('bamboo', '其他', 0.72),
-- 梅花
('plum_blossom', '金地屏风', 0.85), ('plum_blossom', '宣纸水墨', 0.90), ('plum_blossom', '绢本工笔', 0.92), ('plum_blossom', '油画布', 0.58), ('plum_blossom', '其他', 0.70),
-- 修复痕迹（通用必检）
('repair_marks', '金地屏风', 0.85), ('repair_marks', '宣纸水墨', 0.88), ('repair_marks', '绢本工笔', 0.90), ('repair_marks', '油画布', 0.82), ('repair_marks', '其他', 0.85);

-- ------------------------------------------------------------
-- 插入类目提示词（平均每类目 2–3 条，共 ~50 条）
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
