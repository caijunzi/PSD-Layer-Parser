-- ============================================================
-- 种子数据 002: preset 别名映射
-- 创建时间: 2026-09-13
-- 依赖: seed_data.sql（20 个类目）
-- ============================================================

-- ------------------------------------------------------------
-- 插入 preset 别名（20 个类目）
-- 规则：对齐 japanese_screen_gold / chinese_ink_landscape_ai 预设
--       name 格式：NN_中文_English（NN 为序号占位符，下游填充）
-- ------------------------------------------------------------
INSERT INTO category_preset_aliases (category_id, preset_name, preset_id, priority, source, created_at, updated_at) VALUES
-- 远山
('distant_mountains', '远山_Distant_Mountains', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 山石崖壁
('mountains_cliffs', '山石崖壁_Mountains_Cliffs', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 树木植被
('trees_vegetation', '树木植被_Trees_Vegetation', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 水波纹
('water_ripples', '水波纹_Water_Ripples', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 建筑亭台
('architecture', '建筑亭台_Architecture_Pavilion', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 文人高士
('figures_scholar', '文人高士侍童_Figures_Attendants', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
('figures_scholar', '文人高士_Figures_Scholar', NULL, 110, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 鸟禽
('fauna_birds', '鸟禽_Fauna_Birds', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 雁
('fauna_geese', '雁_Fauna_Geese', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
('fauna_geese', '芦雁群禽_Geese_Flock', NULL, 110, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 题跋书法
('calligraphy', '题跋书法_Calligraphy_Inscription', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
('calligraphy', '题跋书法_Calligraphy', NULL, 110, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 印章
('seal', '印章_Cinnabar_Seal', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 织锦边框
('brocade_frame', '织锦边框_Brocade_Outer_Frame', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 屏风折痕
('panel_folds', '屏风折痕折缝_Panel_Fold_Seams', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
('panel_folds', '屏风折痕_Panel_Fold_Seams', NULL, 110, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 云雾
('clouds_mist', '云雾_Clouds_Mist', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 舟船
('boats_ships', '舟船_Boats_Ships', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 渔人
('figures_fisherman', '渔人_Figures_Fisherman', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 石阶
('stone_steps', '石阶_Stone_Steps', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 瀑布
('waterfalls', '瀑布_Waterfalls', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 竹
('bamboo', '竹_Bamboo', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 梅花
('plum_blossom', '梅花_Plum_Blossom', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now')),
-- 修复痕迹
('repair_marks', '修复痕迹_Repair_Marks', NULL, 100, 'manual', strftime('%s','now'), strftime('%s','now'));
