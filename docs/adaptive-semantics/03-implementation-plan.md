# 自进化语义匹配机制 - 完整实施计划

## 文档信息
- **版本**: v1.0-final
- **日期**: 2026-09-13
- **依赖**: 01-architecture.md / 02-data-schema.md
- **交付物**: 分阶段文件清单 + 测试用例 + 迁移脚本 + 验收标准

---

## 一、总览：5 阶段路线图

| 阶段 | 核心功能 | 交付物 | 预估时间 | 依赖 |
|---|---|---|---|---|
| **Stage 1** | 通用词库 + 材质家族匹配 | 换图免写语义清单；命名 SSOT 统一 | 3–5 天 | 无 |
| **Stage 2** | 图级 Auto-Tune（UI 展示建议）| region/密度带推荐；前端采纳按钮 | 2–3 天 | Stage 1 |
| **Stage 3** | 反馈闭环 + 影子进化 | prompt 权重学习；护栏 + 回归准入 | 4–6 天 | Stage 2 |
| **Stage 4** | 案例推理库（CBR）| 指纹检索历史图参数；冷启动加速 | 2–3 天 | Stage 3 |
| **Stage 5** | 类目树 + 主动学习 | 层级泛化；不确定类目请求人审 | 3–4 天 | Stage 4 |

**累计交付周期**：14–21 天（按每天 6 小时有效开发估算）。

---

## 二、Stage 1：通用词库 + 材质家族匹配（基础设施）

### 2.1 目标

- 用户换图后，**自动选出适配该图的语义类目清单**，无需手动写 `ai_semantic_classes`
- 统一层命名（规则侧英文键 → 映射到双语名），彻底解决 `07` 重复成层

### 2.2 文件清单

#### 新增文件（16 个）

```
engine/adaptive/
├── __init__.py                          # 模块入口
├── schema.sql                           # SQLite DDL（详见 02-data-schema.md）
├── seed_data.sql                        # 种子数据：20 类目 + 100 材质亲和度 + 50 prompts
├── fingerprint.py                       # 提取图级指纹（PCA128）
├── material_classifier.py               # 材质家族判别（规则 5 类）
├── category_selector.py                 # 按材质亲和度 Top-K 选类目
├── db_manager.py                        # SQLite 连接池 + 版本管理
├── episode_archiver.py                  # 归档 episode 到 webui/data/episodes/
├── migrations/
│   ├── __init__.py
│   └── migration_001_init.py            # 初始化数据库（执行 schema.sql + seed_data.sql）
└── validate_db.py                       # 数据库完整性校验（CI 用）

webui/backend/api/
└── adaptive_semantics.py                # 新增 API：/api/categories / /api/semantic-versions

tools/
└── generate_affinity_from_presets.py    # 从 4 个预设统计材质×类目共现，补充种子数据
```

#### 修改文件（5 个）

```
run_universal_engine.py                  # 主流程插入点（load_preset 后）
engine/schemas/presets.py                # preset 加 mode / auto_evolve 字段
webui/backend/.env.example               # 新增 ADAPTIVE_MODE 环境变量
webui/backend/main.py                    # 注册 adaptive_semantics 路由
requirements.txt                         # 新增 scikit-learn（PCA）
```

### 2.3 实现步骤（按依赖排序）

#### Step 1.1：数据库初始化（1 天）

**任务**：
1. 创建 `engine/adaptive/schema.sql`（完整 DDL，含触发器/视图/索引）
2. 创建 `engine/adaptive/seed_data.sql`（300 条种子：20 类目 × 5 材质亲和度 × 50 prompts）
3. 编写 `engine/adaptive/migrations/migration_001_init.py`（执行 SQL + 验证）
4. 执行迁移：`python engine/adaptive/migrations/migration_001_init.py`
5. 验证：`python engine/adaptive/validate_db.py`

**验收标准**：
- `webui/data/adaptive_semantics.db` 存在且 >100KB
- `SELECT COUNT(*) FROM categories` ≥ 20
- `SELECT COUNT(*) FROM category_prompts WHERE source='seed'` ≥ 40
- `SELECT COUNT(*) FROM category_material_affinity` ≥ 100
- `SELECT version_id FROM active_version` 返回 `v0-seed`

#### Step 1.2：指纹提取与材质判别（1.5 天）

**文件**：`engine/adaptive/fingerprint.py` / `material_classifier.py`

**任务**：
1. 实现 `extract_fingerprint(image, background_mode)`：
   - 全局特征：尺寸、长宽比、色彩直方图（LAB 空间）、背景 median 色
   - 纹理特征：LBP / GLCM 能量（`skimage.feature`）
   - 结构特征：边缘密度、文本区粗定位（EAST or 连通域）
   - PCA 降维到 128 维（训练集：4 个预设的示例图 + 5 张锚图）
2. 实现 `classify_material_family(fingerprint)`：
   - 规则判别 5 类（金地屏风 / 宣纸水墨 / 绢本工笔 / 油画布 / 其他）
   - 规则：背景色 L*a*b* + 饱和度 + 边缘密度阈值
3. 单元测试：`tests/test_adaptive_fingerprint.py`（5 张已知材质图，断言分类正确）

**验收标准**：
- `pytest tests/test_adaptive_fingerprint.py` 全通过
- `extract_fingerprint` 对 2880×1440 图耗时 <3s
- 5 类材质分类准确率 ≥80%（手动标注 20 张测试集）

#### Step 1.3：类目选择器（1 天）

**文件**：`engine/adaptive/category_selector.py` / `db_manager.py`

**任务**：
1. 实现 `DBManager`（SQLite 连接池 + 事务封装 + 版本读取）
2. 实现 `select_categories(fingerprint, material_family, db_path, mode, preset_categories)`：
   - `mode="locked"` → 直接返回 `preset_categories`（不查库）
   - `mode="auto"` → 按 `material_affinity` 排序取 Top-50 类目
   - `mode="hybrid"` → preset_categories + 自动补充 Top-20
   - 强制保留核心类目（seal / calligraphy / repair_marks）
   - 过滤权重 <0.3 的 prompts
3. 返回格式：`[{category_id, prompts:[{text, weight}], area_budget, name_template}, ...]`
4. 单元测试：`tests/test_adaptive_selector.py`（模拟 3 种 mode，断言类目数量/核心类目必出）

**验收标准**：
- `pytest tests/test_adaptive_selector.py` 全通过
- 查询耗时 <0.5s（50 类目 × 平均 2 prompts）
- locked mode 输出与 preset 原样一致

#### Step 1.4：主流程集成（1 天）

**文件**：`run_universal_engine.py` / `engine/schemas/presets.py`

**任务**：
1. `presets.py` 的 `PresetSchema` 加字段：
   ```python
   mode: str = "locked"  # locked/hybrid/auto
   auto_evolve: bool = False
   ```
2. `run_universal_engine.py` 主流程插入（load_preset 后、load_image 前）：
   ```python
   if preset.mode in ("auto", "hybrid") and os.getenv("ADAPTIVE_MODE") in ("stage1", ...):
       fingerprint = extract_fingerprint(image_bgr, preset.background_mode)
       material_family, conf = classify_material_family(fingerprint)
       selected_cats = select_categories(fingerprint, material_family, db_path, preset.mode, preset.ai_semantic_classes)
       # 覆盖 preset.ai_semantic_classes（或补充）
       preset.ai_semantic_classes = merge_semantic_classes(preset.ai_semantic_classes, selected_cats)
   ```
3. 环境变量控制：`ADAPTIVE_MODE=off|stage1|...`（`.env.example` 加注释）
4. 端到端测试：跑 `chinese_ink_landscape_ai` preset，mode 改 `auto`，换 3 张不同材质图，断言输出类目清单不同

**验收标准**：
- `ADAPTIVE_MODE=off` 时行为完全不变（不调用任何 adaptive 模块）
- `ADAPTIVE_MODE=stage1` + `mode=auto` 时，3 张不同材质图产出不同类目清单
- 增量耗时 <3s（指纹 + 查询）
- 现有测试 `test_golden_layers.py` 不受影响（因 mode 默认 locked）

#### Step 1.5：episode 归档（0.5 天）

**文件**：`engine/adaptive/episode_archiver.py`

**任务**：
1. 实现 `archive_episode(...) -> str`（写 JSON 到 `webui/data/episodes/YYYY-MM/<task_id>.episode.json`）
2. 主流程末尾调用（在审计之后）：
   ```python
   if ADAPTIVE_MODE >= "stage1":
       archive_episode(task_id, fingerprint, {}, selected_categories, [], quality_gate_results, audit_8d, [], semantic_version, seed)
   ```
3. 单元测试：`tests/test_adaptive_archiver.py`（模拟归档，断言 JSON 结构）

**验收标准**：
- 跑一次引擎后，`webui/data/episodes/2026-09/` 下出现对应 JSON
- JSON 可被 `json.load` 正常解析

#### Step 1.6：从现有预设统计补充种子数据（可选，0.5 天）

**文件**：`tools/generate_affinity_from_presets.py`

**任务**：
1. 解析 4 个预设的 `ai_semantic_classes`，统计材质×类目共现频率
2. 用统计结果更新 `category_material_affinity`（加权平均：种子 0.7 + 统计 0.3）
3. 执行：`python tools/generate_affinity_from_presets.py --update-db`

**验收标准**：
- 更新后 `category_material_affinity` 行数 ≥100（无重复键）
- 日志显示更新的行数

### 2.4 测试用例（Stage 1 专属）

**文件**：`tests/test_adaptive_stage1_e2e.py`

```python
def test_stage1_auto_mode_different_images():
    """换 3 张不同材质图，断言类目清单不同"""
    images = ["gold_screen.jpg", "ink_paper.jpg", "oil_canvas.jpg"]
    presets = [load_preset("chinese_ink_landscape_ai") for _ in images]
    for p in presets:
        p.mode = "auto"
    
    results = []
    for img, preset in zip(images, presets):
        # 模拟主流程
        fingerprint = extract_fingerprint(cv2.imread(img), preset.background_mode)
        material_family, _ = classify_material_family(fingerprint)
        selected = select_categories(fingerprint, material_family, db_path, preset.mode, None)
        results.append(set(c["category_id"] for c in selected))
    
    # 断言：3 张图的类目集合不完全相同
    assert len(set(frozenset(r) for r in results)) >= 2

def test_stage1_locked_mode_unchanged():
    """locked mode 时输出与原 preset 一致"""
    preset = load_preset("chinese_ink_landscape_ai")
    assert preset.mode == "locked"
    
    # 即使传了 material_family，也应返回原 ai_semantic_classes
    selected = select_categories({}, "宣纸水墨", db_path, "locked", preset.ai_semantic_classes)
    assert len(selected) == len(preset.ai_semantic_classes)

def test_stage1_core_categories_always_present():
    """核心类目（印章/题跋/修复痕迹）必出"""
    core = {"seal", "calligraphy", "repair_marks"}
    selected = select_categories({}, "金地屏风", db_path, "auto", None)
    selected_ids = {c["category_id"] for c in selected}
    assert core.issubset(selected_ids)
```

---

## 三、Stage 2：图级 Auto-Tune（UI 展示建议）

### 3.1 目标

- 基于图级指纹 + 语义密度场，**自动推荐 region 热点 / 密度带 / 阈值**
- 前端展示建议卡片，用户可一键采纳或手动调整

### 3.2 文件清单

#### 新增文件（5 个）

```
engine/adaptive/
├── auto_tune.py                         # suggest_density_bands / suggest_regions
└── density_analyzer.py                  # 多尺度语义密度场提取

webui/frontend/src/components/
└── AutoTuneSuggestion.tsx               # 建议卡片组件

webui/backend/api/
└── adaptive_semantics.py                # 新增 POST /api/suggest-auto-tune
```

#### 修改文件（3 个）

```
run_universal_engine.py                  # 主流程加 auto_tune 调用
webui/frontend/src/App.tsx               # 集成 AutoTuneSuggestion 组件
engine/adaptive/fingerprint.py           # 返回 density_map
```

### 3.3 实现步骤（1.5 天）

#### Step 2.1：密度场提取（0.5 天）

**文件**：`engine/adaptive/density_analyzer.py`

**任务**：
1. 实现 `compute_semantic_density_map(image, dino_model)`：
   - 多尺度 DINO 前传（512 / 1024 / 2048 窗口）
   - 统计每 256×256 块的激活峰数量 → 语义密度场
2. 集成到 `fingerprint.py`：返回 `density_map: np.ndarray`

**验收标准**：
- 对 2880×1440 图生成 11×6 密度场（256×256 块），耗时 <2s

#### Step 2.2：Auto-Tune 核心逻辑（0.5 天）

**文件**：`engine/adaptive/auto_tune.py`

**任务**：
1. `suggest_density_bands(density_map, semantic_classes)`：
   - 复用 `tools/calibrate_density_bands.py` 的分位数逻辑
   - 返回：`[{"class_id": "03", "band": [0.1, 0.3]}, ...]`
2. `suggest_regions(fingerprint, dino_heatmap)`：
   - K-Means 聚类激活峰 → 热点中心
   - 返回：`[{"bbox": [x,y,w,h], "confidence": 0.85}, ...]`

**验收标准**：
- 单元测试：模拟密度场，断言 band 数值合理（0..1 区间、上限>下限）

#### Step 2.3：后端 API（0.5 天）

**文件**：`webui/backend/api/adaptive_semantics.py`

```python
@router.post("/api/suggest-auto-tune")
async def suggest_auto_tune(file: UploadFile, preset_id: str):
    """
    返回图级 Auto-Tune 建议（region/密度带/阈值）
    """
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    preset = load_preset(preset_id)
    
    fingerprint = extract_fingerprint(image, preset.background_mode)
    density_map = fingerprint["density_map"]
    
    bands = suggest_density_bands(density_map, preset.ai_semantic_classes)
    regions = suggest_regions(fingerprint, {})  # dino_heatmap 暂空
    
    return {
        "density_bands": bands,
        "regions": regions,
        "material_family": fingerprint["material_family"]
    }
```

**验收标准**：
- `curl -F "file=@test.jpg" -F "preset_id=chinese_ink_landscape_ai" http://127.0.0.1:8099/api/suggest-auto-tune` 返回 JSON
- 响应时间 <4s

#### Step 2.4：前端建议卡片（集成到 Stage 2 末尾，不阻塞后端）

**文件**：`webui/frontend/src/components/AutoTuneSuggestion.tsx`

**任务**：
1. 展示 density_bands / regions 建议
2. 提供"采纳全部"/"采纳密度带"/"采纳 Region"按钮
3. 点击后覆盖当前参数，重新触发处理

**验收标准**：
- 上传图片后，卡片显示建议
- 点"采纳密度带"后，参数面板更新

---

## 四、Stage 3：反馈闭环 + 影子进化 — 🟡 核心骨架完成（2026-09-13，commit `21e8f90`）

> **完成状态（核心骨架）**：
> - ✅ 归因标签（`attribution.py`）+ 回归基线（`regression_tester.py` + `extract_baseline.py` + `baseline_audit_8d.json`）
> - ✅ 学习器核心（`learner.py`）+ 后台任务封装（`webui/backend/core/background_learner.py`）
> - ✅ `grounded_sam_provider.py` 记录 `dino_detections` + 质量门/弥散门归因字段
> - ⏸️ **未做**：真实 episode 数据驱动的学习（`episodes` 数据库表从未建立，实际走 JSONL 文件）；
>   `test_golden_layers.py` 新增的 2 个用例暂 `skipTest`（待端到端就绪）
> - ⚠️ **架构决策**：episode 存储**维持文件系统路线**（JSONL），不回补数据库表

### 4.1 目标

- 从 episode 归档中**学习 prompt 权重 / area_budget**
- 护栏校验（最小支撑度 5 图 + 跨图一致性 ≥0.8）
- 回归准入（golden 8 维不退化 >5%）

### 4.2 文件清单

#### 新增文件（6 个）

```
engine/adaptive/
├── learner.py                           # SemanticLearner 类（异步学习闭环）
├── regression_tester.py                 # 回归测试（golden 8 维 + 层数）
└── attribution.py                       # 归因：reject reason → (prompt, category_id)

tests/
├── baseline_audit_8d.json               # 从 golden 产物提取的审计基线
└── test_adaptive_learner.py             # learner 单元测试

webui/backend/core/
└── background_learner.py                # Starlette BackgroundTasks 封装
```

#### 修改文件（4 个）

```
run_universal_engine.py                  # 归档后触发异步学习
engine/providers/grounded_sam_provider.py # detect_objects 返回时给每框打 source_prompt 标签
webui/backend/main.py                    # 注册 background_learner
tests/test_golden_layers.py              # 改为"核心类目必出 + 审计不退化"
```

### 4.3 实现步骤（4 天）

#### Step 3.1：归因与标签增强（0.5 天）

**文件**：`engine/adaptive/attribution.py` / `engine/providers/grounded_sam_provider.py`

**任务**：
1. `grounded_sam_provider.py` 的 `detect_objects` 返回时：
   ```python
   for box in boxes:
       box["source_prompt"] = prompt_text  # 记录哪条 prompt 命中
   ```
2. `attribute_reject_to_prompt(episode)`：从 quality_gate reject_reason 提取 (prompt, category_id)

**验收标准**：
- episode JSON 的 `dino_detections[].source_prompt` 非空
- `attribute_reject_to_prompt` 返回 `[(prompt, category_id, reason), ...]`

#### Step 3.2：回归基线提取（0.5 天）

**文件**：`tests/baseline_audit_8d.json`

**任务**：
1. 跑 5 张 golden 图（4 个预设各 1 张 + 1 张复杂图），提取审计 8 维
2. 格式：
   ```json
   {
     "task_golden_chinese_ink": {
       "lost_ratio": 0.023,
       "rmse": 12.4,
       "plate_purity": 0.98,
       ...
     },
     ...
   }
   ```

**验收标准**：
- `baseline_audit_8d.json` 存在且含 5 个任务的 8 维数据

#### Step 3.3：学习器核心（2 天）

**文件**：`engine/adaptive/learner.py`

**任务**：
1. `SemanticLearner.process_episode_batch(episode_files)`：
   - 归因：提取 accept/reject 信号 → 更新 `category_prompts.{n_accept, n_reject}`
   - 贝叶斯更新权重：`Δ = (accept - reject) / √n_samples`
   - 稳健统计：分位数 + MAD 更新 `category_priors.{area_budget_min, area_budget_max}`
   - 护栏校验：`n_images ≥5` 且跨图一致性 ≥0.8
   - 写候选版本（`category_versions.regression_status='pending'`）
2. `run_regression_test(version_id)`：
   - 激活候选版本（临时切换 `active_version`）
   - 跑 5 张 golden 图，提取审计 8 维
   - 与 `baseline_audit_8d.json` 比对，任一维度退化 >5% → False
3. `activate_version(version_id)` / `rollback_to_version(version_id)`

**验收标准**：
- `pytest tests/test_adaptive_learner.py` 全通过（模拟 episode，断言权重更新）
- 手动跑 `process_episode_batch` + `run_regression_test`，通过后版本激活

#### Step 3.4：异步学习集成（0.5 天）

**文件**：`webui/backend/core/background_learner.py` / `run_universal_engine.py`

**任务**：
1. `background_learner.py`：
   ```python
   from starlette.background import BackgroundTasks
   
   def trigger_learning(episode_path: str):
       learner = SemanticLearner(db_path)
       learner.process_episode_batch([episode_path])
       version_id = learner.get_latest_pending_version()
       if learner.run_regression_test(version_id):
           learner.activate_version(version_id)
       else:
           learner.rollback_to_version(learner.get_active_version().parent_version_id)
   ```
2. `run_universal_engine.py` 归档后：
   ```python
   if ADAPTIVE_MODE >= "stage3":
       background_tasks.add_task(trigger_learning, episode_path)
   ```

**验收标准**：
- 跑一次引擎后，后台任务触发（日志显示"Learning triggered"）
- 学习完成后，`active_version` 更新（或回滚）

#### Step 3.5：测试策略调整（0.5 天）

**文件**：`tests/test_golden_layers.py`

**任务**：
1. 改为"核心类目必出 + 审计 8 维不退化"：
   ```python
   def test_golden_chinese_ink():
       result = run_engine("chinese_ink.jpg", preset="chinese_ink_landscape_ai")
       layers = {l["layer_name"] for l in result["layers"]}
       assert "印章" in layers
       assert "题跋" in layers
       
       audit = result["audit"]
       baseline = load_baseline("task_golden_chinese_ink")
       for key in baseline:
           assert audit[key] <= baseline[key] * 1.05  # 允许 5% 退化
   ```

**验收标准**：
- `pytest tests/test_golden_layers.py` 全通过

---

## 五、Stage 4：案例推理库（CBR）— ✅ 已完成（2026-09-13，commit `b9eb5b0`）

> **完成状态**：Step 4.1 / 4.2 / 4.3（主流程集成）全部完成；端到端测试 7 例全绿。
> 引擎全量回归 **121 passed / 2 skipped**（基线 114 + 新增 7，零回归）。

### 5.1 目标

- 用指纹余弦检索**最相似历史图**，复用其已验证参数（region/密度带/阈值）
- 冷启动加速（新图直接继承相似图参数，无需从零 Auto-Tune）

### 5.2 文件清单

#### 新增文件（3 个）

```
engine/adaptive/
├── cbr_retriever.py                     # 案例推理检索器（PCA128 + 余弦）
└── episode_indexer.py                   # episode 指纹索引（持久化）

tests/
└── test_adaptive_cbr.py                 # CBR 检索测试
```

#### 修改文件（2 个）

```
run_universal_engine.py                  # 集成 CBR 检索
engine/adaptive/episode_archiver.py      # 归档后触发索引更新
```

### 5.3 实现步骤（2 天）

#### Step 4.1：指纹索引（0.5 天）

**文件**：`engine/adaptive/episode_indexer.py`

**任务**：
1. 实现 `EpisodeIndexer`（内存索引 + pickle 持久化）：
   - `add(task_id, fingerprint_128d, audit_passed: bool)`
   - `search(query_fingerprint, top_k=5) -> [(task_id, cosine_sim), ...]`
2. 归档后触发：`indexer.add(task_id, fingerprint, audit_8d["lost_ratio"] < 0.05)`
3. 索引文件：`webui/data/episode_index.pkl`

**验收标准**：
- 归档 10 个 episode 后，`episode_index.pkl` 存在
- `indexer.search()` 返回 top-5 相似任务

#### Step 4.2：CBR 检索器（0.5 天）

**文件**：`engine/adaptive/cbr_retriever.py`

**任务**：
1. `retrieve_similar_episode(fingerprint, indexer, episodes_dir) -> dict | None`：
   - 调用 `indexer.search(fingerprint, top_k=5)`
   - 过滤：只返回 `audit_passed=True` 的最相似项
   - 读取 episode JSON，提取 `auto_tune` 参数
2. 返回：`{regions, density_bands, thresholds}` 或 None（无相似图）

**验收标准**：
- 单元测试：模拟 10 个 episode，查询返回最相似的 1 个

#### Step 4.3：主流程集成（0.5 天）

**文件**：`run_universal_engine.py`

**任务**：
1. Stage 2 的 Auto-Tune 前插入 CBR 检索：
   ```python
   if ADAPTIVE_MODE >= "stage4":
       similar_episode = retrieve_similar_episode(fingerprint, indexer, episodes_dir)
       if similar_episode:
           # 用相似图的参数作起点
           auto_tune = similar_episode["auto_tune"]
       else:
           # 无相似图，从零 Auto-Tune
           auto_tune = suggest_auto_tune(fingerprint, density_map)
   ```

**验收标准**：
- 跑 2 张相似图（同材质、相似构图），第 2 张复用第 1 张的 auto_tune 参数
- 日志显示"CBR hit: task_xxx"

#### Step 4.4：端到端测试（0.5 天）

**文件**：`tests/test_adaptive_cbr.py`

```python
def test_cbr_cold_start():
    """首次跑无相似图，从零 Auto-Tune"""
    result = run_engine("new_material.jpg", preset="chinese_ink_landscape_ai")
    assert result["cbr_hit"] is None

def test_cbr_warm_start():
    """第 2 次跑相似图，复用参数"""
    run_engine("gold_screen_1.jpg", preset="japanese_screen_gold")  # 第 1 次
    result = run_engine("gold_screen_2.jpg", preset="japanese_screen_gold")  # 第 2 次
    assert result["cbr_hit"] is not None
    assert result["auto_tune"]["regions"] == result["cbr_hit"]["auto_tune"]["regions"]
```

---

## 六、Stage 5：类目树 + 主动学习

### 6.1 目标

- 类目层级化（山水画 / 远山 / 近山），新类目继承祖先先验
- 主动学习：只对**不确定类目**请求人审（最小化人工）

### 6.2 文件清单

#### 新增文件（4 个）

```
engine/adaptive/
├── category_tree.py                     # 类目树管理（插入/查询祖先/DFS 无环检测）
└── active_learner.py                    # 主动学习：不确定类目 → 请求人审

webui/frontend/src/components/          # ⚠️ 需新建（当前不存在）
├── CategoryFeedbackModal.tsx           # 人审弹窗（Step 5.3，待做）
└── AutoTuneSuggestion.tsx              # Stage 2 卡片重构抽出（待做）

webui/backend/api/
└── adaptive.py                         # ⚠️ 修正：不新建 adaptive_semantics.py
                                        #    沿用既有 adaptive.py（Stage 1~4 端点都在此）
                                        #    新增 GET /api/adaptive/pending-feedbacks
                                        #    新增 POST /api/adaptive/categories/{id}/feedback
```

#### 修改文件（3 个）

```
engine/adaptive/schema.sql               # categories.parent_id / path 已有，无需改 ✅
run_universal_engine.py                  # 集成主动学习（待做，Step 5.3 之后）
webui/frontend/src/App.tsx               # 集成人审弹窗 + 轮询（待做）
```

> ⚠️ **路径修正说明**：原计划写 `POST /api/categories/{id}/feedback` 并新建
> `api/adaptive_semantics.py`。实际 `adaptive.py` 的 router 已带 `/adaptive` 前缀，
> 且 Stage 1~4 所有端点（`suggest-auto-tune` / `apply-auto-tune`）都在此文件，
> 为保持一致性，端点统一为 `/api/adaptive/...`，不再新建文件。

### 6.3 实现步骤（3 天）

#### Step 5.1：类目树管理（1 天）— ✅ 已完成（2026-09-13，commit `b74ab10`）

**文件**：`engine/adaptive/category_tree.py`

**任务**：
1. ✅ `CategoryTree.insert(category_id, parent_id)` + 无环检测（DFS 遍历后代）
2. ✅ `CategoryTree.get_ancestors(category_id) -> [parent_id, grandparent_id, ...]`
3. ⏸️ `CategoryTree.inherit_priors(new_category_id, parent_id)`：**已跳过实现，标记 TODO**
   - 原因：`category_priors` 表当前为空（0 条），无可继承数据
   - 决策：等 Stage 3 反馈学习填充 priors 后再补（手写 seed 数据是"拍脑袋"，不如真实统计）
4. ✅ 迁移脚本：`migrations/migration_003_build_tree.py`
   - ⚠️ **修正**：原计划写 `migration_002_build_tree.py`，但 `002` 已被 preset 别名占用
     （`migration_002_apply.py` / `migration_002_preset_aliases.sql`），故改用 **003**
   - 实际结构：根「山水画」→ 7 个二级类目（山/水/植被/建筑/天象与云雾/人物与动物/底板与边框）→ 20 个三级类目

**验收标准**：
- ✅ `pytest tests/test_adaptive_tree.py`（7 例全绿：插入父子 / 循环检测 / 三层链 / 祖先链×3 / inherit_priors NotImplementedError）
- ✅ 迁移后 `SELECT COUNT(*) FROM categories WHERE parent_id IS NOT NULL` = **13**（≥10 达标）
  - 总数 27（原 20 + 新增 8：1 根 + 7 二级）
  - path 正确计算，如 `/landscape_painting/distant_mountains/`

#### Step 5.2：主动学习（1 天）— ✅ 已完成（2026-09-13，commit `ae39733`）

**文件**：`engine/adaptive/active_learner.py` + `webui/backend/api/adaptive.py`

**任务**：
1. ✅ `ActiveLearner.identify_uncertain_categories(result, confidence_threshold=0.7)`：
   - 从检出结果提取 confidence <0.7 的类目
2. ✅ `request_human_feedback(uncertain_categories) -> feedback_request_id`：
   - ⚠️ **修正通信机制**：原计划"WebSocket / SSE 推送"，实际采用**轮询方案**
   - 理由：后端是 FastAPI 无 WebSocket/SSE 基础设施；轮询零新依赖、够用（延迟 ≤3s）
   - 实现：写入待审队列 JSONL → 前端 3 秒轮询 `GET /api/adaptive/pending-feedbacks`
3. ✅ 反馈写入 + 立即更新 `category_prompts` 权重：
   - ⚠️ **修正存储路线**：原计划写 `episodes` 数据库表，实际**维持文件系统路线**
     （`episodes` 表从未建立，Stage 3/4 一直用 JSONL，故 Stage 5 沿用 JSONL 保持一致）
   - 待审队列：`webui/data/pending_feedbacks.jsonl`
   - `accept` 操作：贝叶斯权重提升 1.1×（与 Stage 3 learner 一致）
   - `rename` / `delete` / `merge`：**已真实实现**（2026-09-14 补完，见下「补完注记」）

> **✅ 2026-09-14 补完注记（commit `906256b`）**：rename / delete / merge 已从 TODO 转为真实实现。
> - `rename`：更新 `categories.name_zh / name_en / name_template`（template 保留 `NN_` 序号前缀）
> - `delete`：**软删除**（置 `deleted_at`），非硬删 —— categories 被 prompts/affinity/priors
>   外键 CASCADE 引用，硬删会清空学到的权重且不可恢复；历史 episode JSONL 也引用 category_id。
>   软删后由 `db_manager` 主查询 `deleted_at IS NULL` 过滤，等价于不可见。
> - `merge`：源类目 prompts 迁移到目标（同名 prompt 取权重较大者，绕 UNIQUE 约束）
>   + 源类目软删；拒绝「缺目标 / 目标不存在 / 目标==源」
> - 依赖迁移：`migrations/migration_004_feedback_ops.py`（categories 加 `deleted_at`，幂等）
> - 统一返回 `{ok, action, detail}`；失败明确报 400，不再静默
> - e2e：`tests/test_adaptive_stage5_e2e.py`（14 例全绿）

**新增 API 端点**（`webui/backend/api/adaptive.py`，+104 行）：
- `GET /api/adaptive/pending-feedbacks?limit=10` → 返回待审类目列表
- `POST /api/adaptive/categories/{category_id}/feedback`
  - Form 参数：`feedback_request_id` / `user_action`(accept|rename|delete|merge) / `user_data`(JSON 字符串)

**验收标准**：
- ✅ `pytest tests/test_adaptive_active.py`（7 例全绿：识别低置信 / 自定义阈值 / 写入读取队列 / 标记已审核 / limit 限制 / 空列表边界）

#### Step 5.3：前端人审弹窗（1 天）— ✅ 已完成（2026-09-14，commit `b7241a3`）

**文件**：`webui/frontend/src/components/CategoryFeedbackModal.tsx`

> ✅ **目录已新建**：`webui/frontend/src/components/`（此前不存在）。
> 按用户决策，**新建 `components/` 目录**，并顺便把 Stage 2 的 Auto-Tune
> 卡片抽成 `AutoTuneSuggestion.tsx`（重构，避免 App.tsx 继续膨胀）。

**任务**：
1. ✅ 展示不确定类目（类目 id + prompt + bbox + **confidence 进度条**）
2. ✅ 按钮：accept / rename(输入框) / delete / merge(输入目标类目 id)
   - ⚠️ **merge 用文本输入目标 id**（非下拉选择）：后端 merge 目前是 TODO，
     不为一个未实现功能扩展 API（`GET /api/adaptive/categories` 只返回 `display_name` 无 id）
3. ✅ 提交到 `POST /api/adaptive/categories/{id}/feedback`
4. ✅ 集成轮询：`App.tsx` 每 3 秒调用 `GET /api/adaptive/pending-feedbacks?limit=1`
   - 弹窗打开时**暂停轮询**（避免打断用户操作），组件卸载清理 interval
   - 轮询失败**静默**（后端未启用或无待审数据时不打扰用户）
   - 逐类目裁决：处理一个移除一个，**全部处理完才关闭弹窗**
5. ✅ （重构）Stage 2 的 Auto-Tune 卡片从 `App.tsx` 抽出为 `AutoTuneSuggestion.tsx`
   - App.tsx **-97 行**，改为受控组件 `<AutoTuneCard />`，UI 与交互不变

**验收标准**：
- ✅ `npx tsc --noEmit --skipLibCheck` **编译通过（零错误）**
- ✅ **真实 HTTP 冒烟**（FastAPI TestClient）：
  - `GET /api/adaptive/pending-feedbacks` → 200，返回 1 条（task_smoke_001 / water_ripples）
  - `POST /api/adaptive/categories/water_ripples/feedback`（accept）→ 200，权重 ×1.1 生效
  - 复查 → 剩余 0 条（状态已转 reviewed）
- ✅ 冒烟**脏数据已清理**：删除测试待审队列，回滚权重 1.1→1.0、n_accept 1→0
- ✅ 引擎 128 passed / 2 skipped；WebUI 28 OK（零回归）

---

## 七、测试矩阵（全阶段）

### 7.1 单元测试（34 个）

| 模块 | 测试文件 | 用例数 |
|---|---|---|
| 数据库 | `test_adaptive_db.py` | 6（表结构 / 索引 / 触发器 / 视图）|
| 指纹 | `test_adaptive_fingerprint.py` | 5（PCA / 材质分类 / 边缘密度）|
| 类目选择 | `test_adaptive_selector.py` | 4（3 mode / 核心类目必出）|
| Auto-Tune | `test_adaptive_auto_tune.py` | 3（密度带 / region）|
| 学习器 | `test_adaptive_learner.py` | 6（归因 / 权重更新 / 护栏 / 回归）|
| CBR | `test_adaptive_cbr.py` | **7**（冷启动×2 / 热启动×3 / 性能 / 持久化）|
| 类目树 | `test_adaptive_tree.py` | **7**（插入父子 / 循环检测 / 三层链 / 祖先链×3 / inherit_priors）|
| 主动学习 | `test_adaptive_active.py` | **7**（识别低置信 / 无不确定 / 自定义阈值 / 写入读取队列 / 标记已审核 / limit / 空列表）|

> ⚠️ **测试数修正**：计划原定 CBR 3 例、类目树 4 例、主动学习 3 例；
> 实际实现更充分，分别为 **7 / 7 / 7** 例（多出性能、持久化、边界情况等用例）。
> 引擎全量测试：基线 **128 passed / 2 skipped**（含 Stage 4 CBR 7 + Stage 5.1 树 7 + Stage 5.2 主动 7）。

### 7.2 集成测试（8 个）

| 测试 | 文件 | 覆盖阶段 |
|---|---|---|
| Stage 1 端到端 | `test_adaptive_stage1_e2e.py` | S1 |
| Stage 2 端到端 | `test_adaptive_stage2_e2e.py` | S1+S2 |
| Stage 3 端到端 | `test_adaptive_stage3_e2e.py` | S1+S2+S3 |
| Stage 4 端到端 | `test_adaptive_stage4_e2e.py` | S1+S2+S3+S4 |
| Stage 5 端到端 | `test_adaptive_stage5_e2e.py` | 全部（✅ 已建，14 例全绿）|
| 接线守卫 | `test_adaptive_wiring.py` | 10（防「已实现但无生产调用」的链路断裂）|
| 回归测试（golden）| `test_golden_layers.py`（修改版）| 全部 |
| 性能测试 | `test_adaptive_performance.py` | 全部 |
| 兼容性测试 | `test_adaptive_compatibility.py` | 全部 |

### 7.3 性能测试（3 个场景）

**文件**：`tests/test_adaptive_performance.py`

```python
def test_perf_fingerprint_extraction():
    """指纹提取 <3s"""
    t0 = time.time()
    extract_fingerprint(cv2.imread("2880x1440.jpg"), "paper")
    assert time.time() - t0 < 3.0

def test_perf_category_selection():
    """类目查询 <0.5s"""
    t0 = time.time()
    select_categories({}, "金地屏风", db_path, "auto", None)
    assert time.time() - t0 < 0.5

def test_perf_cbr_retrieval():
    """CBR 检索 <0.3s（1000 条索引）"""
    indexer = load_index(1000)  # 1000 个 episode
    t0 = time.time()
    indexer.search(np.random.rand(128), top_k=5)
    assert time.time() - t0 < 0.3
```

---

## 八、数据迁移清单

| 迁移 | 脚本 | 执行时机 |
|---|---|---|
| 001: 初始化数据库 | `migrations/migration_001_init.py` | Stage 1 启动前 |
| 002: 建类目树（手动标注）| `migrations/migration_002_build_tree.py` | Stage 5 启动前 |
| 003: 从预设统计补充亲和度 | `tools/generate_affinity_from_presets.py --update-db` | Stage 1 后（可选）|

---

## 九、部署清单

### 9.1 环境变量（`.env`）

```bash
# 自进化语义匹配功能开关
ADAPTIVE_MODE=off          # off / stage1 / stage2 / stage3 / stage4 / stage5

# 数据库路径（默认）
ADAPTIVE_DB_PATH=webui/data/adaptive_semantics.db

# episode 归档路径（默认）
EPISODES_DIR=webui/data/episodes

# 学习器配置
ADAPTIVE_MIN_SUPPORT=5              # 最小支撑度（图数）
ADAPTIVE_MIN_CONSISTENCY=0.8        # 跨图一致性阈值
ADAPTIVE_REGRESSION_THRESHOLD=0.05  # 回归容忍度（5%）
```

### 9.2 依赖更新（`requirements.txt`）

```
scikit-learn>=1.5.0        # PCA / K-Means
scikit-image>=0.24.0       # LBP / GLCM
```

### 9.3 前端构建

```bash
cd webui/frontend
npm install
npm run build
```

---

## 十、验收标准（总）

### 10.1 功能验收

| 阶段 | 验收项 | 通过标准 |
|---|---|---|
| S1 | 换 3 张不同材质图 | 类目清单不同；核心类目必出 |
| S2 | Auto-Tune 建议 | API 返回 density_bands / regions；前端显示卡片 |
| S3 | 反馈闭环 | 跑 10 次 → 词库版本更新；回归测试通过 |
| S4 | CBR 检索 | 第 2 张相似图复用参数；日志显示 CBR hit |
| S5 | 主动学习 | 不确定类目触发人审弹窗；反馈后权重更新 |

### 10.2 性能验收

- 增量耗时（Stage 1）：<3s
- 增量耗时（Stage 2）：<6s（压缩到 ~3s 并行）
- 增量耗时（Stage 3–5）：<1s（学习异步，不阻塞）
- CBR 检索（1000 条）：<0.3s

### 10.3 质量验收

- 单元测试覆盖率 ≥85%
- 集成测试全通过（8 个）
- 回归测试（golden）通过：审计 8 维不退化 >5%
- 材质家族分类准确率 ≥80%（手动标注 20 张）

---

## 十一、风险缓解

| 风险 | 缓解措施 |
|---|---|
| 类目数量膨胀 → DINO 慢 | 材质家族裁剪 90% + Top-50 截断 + 权重<0.3 不启用 |
| SQLite 并发写冲突 | 学习器单线程 + 事务隔离 + 版本链 CoW |
| 首次使用冷启动（词库空）| 出厂带 300 条种子 + 5 类材质家族基线 |
| 自进化引入新 bug | 回归准入 + 影子模式 N=20 + 30 天版本 GC |
| 用户反对自动化 | 三档 mode（locked/hybrid/auto）+ preset.auto_evolve 开关 |
| PCA 维度爆炸 | 降到 128 维 + 归一化 |
| 指纹提取耗时过长 | 与 DINO 加载并行 + 多尺度窗口优化 |

---

## 十二、开发顺序建议（AI IDE 友好）

**单人开发推荐顺序**（按依赖 + ROI）：

1. **Day 1–2**：Stage 1.1–1.3（数据库 + 指纹 + 类目选择器）→ 可验证材质判别
2. **Day 3**：Stage 1.4–1.5（主流程集成 + episode 归档）→ 端到端跑通
3. **Day 4**：Stage 1.6 + Stage 1 测试 → 冷启动可用
4. **Day 5–6**：Stage 2.1–2.3（密度场 + Auto-Tune + API）→ 后端建议可用
5. **Day 7**：Stage 2.4（前端卡片）→ UI 完整
6. **Day 8–10**：Stage 3.1–3.4（归因 + 学习器 + 异步集成）→ 闭环生效
7. **Day 11**：Stage 3.5 + Stage 3 测试 → 回归通过
8. **Day 12–13**：Stage 4（CBR）→ 冷启动优化
9. **Day 14–16**：Stage 5（类目树 + 主动学习）→ 全功能
10. **Day 17**：全量测试 + 文档 + 交付

**并行开发建议**（2 人）：
- 人 A：Stage 1–3（后端 + 学习器）
- 人 B：Stage 2.4 + 前端 UI（AutoTuneSuggestion / CategoryFeedbackModal）

---

## 十三、文档交付物

1. **架构设计**：`docs/adaptive-semantics/01-architecture.md` ✅
2. **数据结构**：`docs/adaptive-semantics/02-data-schema.md` ✅
3. **实施计划**：`docs/adaptive-semantics/03-implementation-plan.md`（本文档）✅
4. **API 文档**：`docs/adaptive-semantics/04-api-reference.md`（待生成）
5. **用户手册**：`docs/adaptive-semantics/05-user-guide.md`（待生成）

---

**下一步**：
- 执行 `engine/adaptive/migrations/migration_001_init.py` 初始化数据库
- 开始 Stage 1.2（指纹提取与材质判别）
- 或：要我生成 `04-api-reference.md`（完整 API 文档 + curl 示例）？
