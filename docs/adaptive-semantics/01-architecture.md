# 自进化语义匹配机制 - 架构设计

## 文档信息
- **版本**: v1.0-draft
- **日期**: 2026-09-13
- **状态**: 架构定稿，待评审
- **依赖决策**: 基于 20 个澄清问题的 ✅ 推荐方案

> ### ⚠️ 2026-09-14 同步注记（实现偏差，以本注记与 `03-implementation-plan.md` 为准）
>
> 本文为**架构定稿草案**，Stage 1~5 落地后有三处与实现不一致：
>
> | 本文描述 | 实际情况 | 依据 |
> |---|---|---|
> | 表 4 `category_priors`（形状/面积先验，防假阳性） | 表结构**已建但为空表（0 条）**，故 `inherit_priors()` **跳过实现、标 TODO** | ADR-023 |
> | 人审反馈（§交互层"人审反馈按钮"） | 通信用**轮询** `GET /api/adaptive/pending-feedbacks`，**非** WebSocket/SSE | ADR-022 |
> | — | episode 存储确为文件 `webui/data/episodes/YYYY-MM/*.json`，**未建数据库表**（与本文 §表 3 一致 ✅） | ADR-021 |
>
> **当前进度**：Stage 1/2/4/5.1/5.2 ✅ 已完成；Stage 3 🟡 核心骨架；Stage 5.3 ⏳ 待做。
> 完整状态矩阵与 commit 见 `ARCHITECTURE.md §8.2` 与 `03-implementation-plan.md`。

---

## 一、总体架构

### 1.1 五层分工

```
┌─────────────────────────────────────────────────────────────────┐
│  交互层（用户/前端）                                              │
│  - WebUI 参数调整、人审反馈按钮                                   │
│  - CLI --semantic-mode / --semantic-version 控制                 │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│  图级自适应层（每次运行时现算，永不入库）                          │
│  - 指纹提取（PCA128 维）                                          │
│  - 材质家族判别（规则 5 类 → LightGBM）                          │
│  - Auto-Tune: region热点 / 密度带 / 阈值                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│  品类模板层（低频进化，人工可干预）                                │
│  - preset.mode = locked/hybrid/auto                              │
│  - preset.auto_evolve: bool（冻结开关）                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│  通用语义词库（持续进化，SQLite 持久化）                           │
│  - Category 类目原型（prompts权重/area_budget/material_affinity） │
│  - 版本链管理（version_id / parent_id / 30天GC）                 │
│  - 负样本库（假阳性模式）                                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│  机制内核（DINO/SAM/超分/补全/审计，不可变）                      │
│  - 开放词表（DINO prompt）、8维审计、质量门禁                     │
└─────────────────────────────────────────────────────────────────┘
```

**右侧异步学习流程（Starlette BackgroundTasks）**:
```
episode归档 → 归因（reject reason → 类目/prompt）
           → 稳健统计（分位数/MAD/Beta先验）
           → 护栏校验（最小支撑度5图 + 跨图一致性≥0.8）
           → 影子模式（只记录不改行为）
           → 回归准入（test_golden_layers + 审计8维不退化）
           → 灰度发布（version_id++）
```

### 1.2 关键约束（边界）

| 约束 | 实现方式 |
|---|---|
| **逐像素可复现**（RK-16） | 词库版本化（version_id） + seed 固定 = 同结果 |
| **制版线零生成内容** | plate/design 隔离保持；自进化不引入生成式决策 |
| **8 维审计不退化** | 回归准入：新版本必须在 golden 产物上不退化 >5% |
| **信息论不可提取边界** | 目标是"不退化 + 少人工"，保留交互补齐通道 |

---

## 二、数据结构设计

### 2.1 语义命中库（SQLite）

**数据库路径**: `webui/data/adaptive_semantics.db`

#### 表 1: `categories`（类目原型，唯一持久实体）

```sql
CREATE TABLE categories (
  id TEXT PRIMARY KEY,                    -- 如 "mountains_cliffs"
  name_zh TEXT NOT NULL,                  -- "山石崖壁"
  name_en TEXT,                           -- "Mountains_Cliffs"
  parent_id TEXT,                         -- 类目树父节点（扁平树初期为 NULL）
  path TEXT,                              -- "/山水画/山石" 路径枚举（查祖先 O(1)）
  name_template TEXT,                     -- "NN_中文_English"
  confidence REAL DEFAULT 0.5,            -- 0..1，决定能否自动启用
  created_at INTEGER,                     -- Unix 时间戳
  updated_at INTEGER,
  FOREIGN KEY(parent_id) REFERENCES categories(id)
);

CREATE INDEX idx_cat_parent ON categories(parent_id);
CREATE INDEX idx_cat_confidence ON categories(confidence);
```

#### 表 2: `category_prompts`（逐条 prompt 学权重）

```sql
CREATE TABLE category_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id TEXT NOT NULL,
  prompt TEXT NOT NULL,                   -- DINO 提示词
  weight REAL DEFAULT 1.0,                -- 贝叶斯更新（Δ ∝ 1/√n_samples）
  lang TEXT DEFAULT 'zh',                 -- zh/en
  source TEXT DEFAULT 'seed',             -- seed|mined|human
  n_hits INTEGER DEFAULT 0,               -- 被 DINO 检出次数
  n_accept INTEGER DEFAULT 0,             -- 通过质量门次数
  n_reject INTEGER DEFAULT 0,             -- 被拒绝次数
  n_images INTEGER DEFAULT 0,             -- 见过多少张不同图（去重）
  last_hit_at INTEGER,                    -- 最后命中时间（衰减遗忘）
  created_at INTEGER,
  FOREIGN KEY(category_id) REFERENCES categories(id)
);

CREATE INDEX idx_prompt_cat ON category_prompts(category_id);
CREATE INDEX idx_prompt_weight ON category_prompts(weight DESC);
```

#### 表 3: `category_material_affinity`（材质亲和度）

```sql
CREATE TABLE category_material_affinity (
  category_id TEXT NOT NULL,
  material_family TEXT NOT NULL,          -- "金地屏风" / "宣纸水墨" / ...
  affinity REAL DEFAULT 0.5,              -- 0..1，决定"该材质用这个词"
  PRIMARY KEY(category_id, material_family),
  FOREIGN KEY(category_id) REFERENCES categories(id)
);

CREATE INDEX idx_affinity_mat ON category_material_affinity(material_family, affinity DESC);
```

#### 表 4: `category_priors`（形状/面积先验，防假阳性）

```sql
CREATE TABLE category_priors (
  category_id TEXT PRIMARY KEY,
  area_budget_min REAL,                   -- 面积占比下限（分位数 + MAD）
  area_budget_max REAL,
  elongation_mean REAL,                   -- 形状先验：长宽比
  elongation_std REAL,
  compactness_mean REAL,                  -- 紧密度
  compactness_std REAL,
  n_components_mode INTEGER,              -- 连通域数量众数
  updated_at INTEGER,
  FOREIGN KEY(category_id) REFERENCES categories(id)
);
```

#### 表 5: `negative_samples`（假阳性模式，负样本）

```sql
CREATE TABLE negative_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  context_fingerprint BLOB,               -- 128维指纹（何种图会假阳性）
  reject_reason TEXT,                     -- 来自质量门/弥散门
  created_at INTEGER,
  FOREIGN KEY(category_id) REFERENCES categories(id)
);

CREATE INDEX idx_neg_cat ON negative_samples(category_id);
```

#### 表 6: `category_versions`（词库版本链）

```sql
CREATE TABLE category_versions (
  version_id TEXT PRIMARY KEY,            -- "v2026-09-13T12:34:56Z"
  parent_version_id TEXT,                 -- Git 式版本链
  snapshot_json TEXT,                     -- 完整快照（JSON 导出）
  changeset TEXT,                         -- 变更摘要（哪些类目/prompt 改了）
  evidence TEXT,                          -- 证据摘要（多少图/一致性）
  regression_status TEXT DEFAULT 'pending', -- pending|passed|failed
  created_at INTEGER,
  activated_at INTEGER                    -- 发布时间（NULL = 未发布）
);

CREATE INDEX idx_ver_activated ON category_versions(activated_at DESC);
```

### 2.2 Episode 归档（本地文件）

**路径**: `webui/data/episodes/YYYY-MM/<task_id>.episode.json`

```json
{
  "task_id": "task_20260913_120555_b2b831",
  "fingerprint": [0.123, -0.456, ...],  // 128维 PCA
  "material_family": "宣纸水墨",
  "material_family_confidence": 0.92,
  "auto_tune": {
    "regions": [...],
    "density_bands": [...],
    "thresholds": {...}
  },
  "selected_categories": ["03_远山", "04_山石崖壁", ...],
  "dino_detections": [
    {"category_id": "mountains_cliffs", "prompt": "悬崖", "bbox": [...], "score": 0.85}
  ],
  "quality_gate_results": {
    "03_远山": {"accept": true, "reason": ""},
    "10_印章": {"accept": false, "reason": "弥散门：面积超预算 3.2x"}
  },
  "audit_8d": {
    "lost_ratio": 0.023,
    "rmse": 12.4,
    ...
  },
  "user_feedback": [
    {"layer_id": "07", "action": "rename", "from": "文人高士", "to": "文人高士侍童"}
  ],
  "semantic_version": "v2026-09-12T08:00:00Z",
  "seed": 42,
  "created_at": 1726201555
}
```

---

## 三、接口设计

### 3.1 引擎侧新增模块

#### `engine/adaptive/fingerprint.py`

```python
def extract_fingerprint(image: np.ndarray, background_mode: str) -> dict:
    """
    提取图级指纹（128维 PCA）
    返回: {
        "embedding": np.ndarray(128,),
        "material_family": str,  # "金地屏风" / "宣纸水墨" / ...
        "material_confidence": float,
        "texture_stats": {...},
        "density_map": np.ndarray  # 多尺度语义密度（供 Auto-Tune 用）
    }
    """
```

#### `engine/adaptive/material_classifier.py`

```python
def classify_material_family(fingerprint: dict) -> tuple[str, float]:
    """
    材质家族判别（阶段1规则5类 → 阶段3 LightGBM）
    返回: (family_name, confidence)
    5类: "金地屏风" / "宣纸水墨" / "绢本工笔" / "油画布" / "其他"
    """
```

#### `engine/adaptive/category_selector.py`

```python
def select_categories(
    fingerprint: dict,
    material_family: str,
    db_path: str,
    mode: str = "auto",  # locked/hybrid/auto
    preset_categories: list[str] | None = None
) -> list[dict]:
    """
    从词库按材质亲和度 Top-K 选类目
    返回: [
        {
            "category_id": "mountains_cliffs",
            "prompts": [{"text": "悬崖", "weight": 1.2}, ...],
            "area_budget": (0.01, 0.15),
            "name_template": "NN_山石崖壁_Mountains_Cliffs"
        },
        ...
    ]
    mode=locked 时直接用 preset_categories，不查库
    """
```

#### `engine/adaptive/auto_tune.py`

```python
def suggest_density_bands(density_map: np.ndarray, semantic_classes: list) -> list:
    """
    基于语义密度场推荐密度带（复用 tools/calibrate_density_bands.py 逻辑）
    """

def suggest_regions(fingerprint: dict, dino_heatmap: np.ndarray) -> list:
    """
    推荐 region 热点（基于 DINO 激活峰聚类）
    """
```

#### `engine/adaptive/episode_archiver.py`

```python
def archive_episode(
    task_id: str,
    fingerprint: dict,
    auto_tune: dict,
    selected_categories: list,
    dino_detections: list,
    quality_gate_results: dict,
    audit_8d: dict,
    user_feedback: list,
    semantic_version: str,
    seed: int
) -> str:
    """
    归档 episode 到 webui/data/episodes/YYYY-MM/<task_id>.episode.json
    返回: 文件路径
    """
```

### 3.2 学习器模块（异步）

#### `engine/adaptive/learner.py`

```python
class SemanticLearner:
    """
    异步学习闭环（Starlette BackgroundTasks 调用）
    """
    def __init__(self, db_path: str, min_support: int = 5, min_consistency: float = 0.8):
        self.db = sqlite3.connect(db_path)
        self.min_support = min_support
        self.min_consistency = min_consistency
    
    def process_episode_batch(self, episode_files: list[str]):
        """
        批量处理 episode，更新 prompt 权重 / area_budget / negative_samples
        """
        # 1. 归因：从 quality_gate reject_reason 提取 (prompt, category_id)
        # 2. 稳健统计：分位数 + MAD 更新 area_budget；贝叶斯更新 prompt weight
        # 3. 护栏校验：跨图一致性 ≥0.8 且 n_images ≥5 才允许更新
        # 4. 写入候选版本（regression_status=pending）
    
    def run_regression_test(self, version_id: str) -> bool:
        """
        在 test_golden_layers + 审计8维基线上跑回归
        返回: 是否通过（任一退化 >5% → False）
        """
    
    def activate_version(self, version_id: str):
        """
        通过回归 → 更新 activated_at，成为当前版本
        """
    
    def rollback_to_version(self, version_id: str):
        """
        失败回滚到 parent_version_id
        """
```

### 3.3 WebUI 后端新增 API

#### `webui/backend/api/adaptive_semantics.py`

```python
@router.post("/api/suggest-auto-tune")
async def suggest_auto_tune(file: UploadFile, preset_id: str):
    """
    返回图级 Auto-Tune 建议（region/密度带/阈值）
    前端用户可一键采纳 or 手动调整
    """

@router.get("/api/semantic-versions")
async def list_semantic_versions():
    """
    列出词库版本历史（含回归状态）
    """

@router.post("/api/semantic-versions/{version_id}/activate")
async def activate_semantic_version(version_id: str):
    """
    手动激活某版本（需管理员权限）
    """

@router.post("/api/semantic-versions/{version_id}/rollback")
async def rollback_semantic_version(version_id: str):
    """
    回滚到指定版本
    """

@router.get("/api/categories")
async def list_categories(material_family: str | None = None):
    """
    列出类目（可按材质筛选），供前端展示词库
    """

@router.post("/api/categories/{category_id}/feedback")
async def submit_category_feedback(
    category_id: str,
    layer_id: str,
    action: str,  # "accept" / "rename" / "delete" / "merge"
    new_name: str | None = None
):
    """
    人审反馈（S3 级最强信号）→ 立即归档到 episode
    """
```

### 3.4 修改现有接口

#### `run_universal_engine.py` 主流程插入点

```python
# 原流程：
# load_preset → load_image → detect_objects(DINO) → segment(SAM) → super_res → audit

# 新流程（阶段1开始插入）：
load_preset
↓
if preset.mode in ("auto", "hybrid"):
    fingerprint = extract_fingerprint(image)  # ← 新增
    material_family = classify_material_family(fingerprint)
    selected_categories = select_categories(fingerprint, material_family, db_path, preset.mode, preset.ai_semantic_classes)
    # 用 selected_categories 覆盖或补充 preset.ai_semantic_classes
    auto_tune = suggest_auto_tune(fingerprint, density_map)  # ← 阶段2
    # 用 auto_tune 建议覆盖 CLI 参数（如果用户未显式指定）
↓
load_image → detect_objects → segment → super_res → audit
↓
archive_episode(...)  # ← 归档
↓
if ADAPTIVE_MODE >= "stage3":
    background_tasks.add_task(learner.process_episode_batch, [episode_path])  # ← 异步学习
```

---

## 四、护栏与防护

### 4.1 过拟合防护（六层护栏）

| 护栏 | 实现位置 | 参数 |
|---|---|---|
| **图级参数绝不入库** | `archive_episode`：auto_tune 只存 episode，不写 DB | — |
| **最小支撑度** | `SemanticLearner.process_episode_batch` | `min_support=5`, `min_consistency=0.8` |
| **稳健统计** | 同上：分位数 + MAD + Beta(2,2) 先验 | α=2, β=2 |
| **负样本对等** | `negative_samples` 表 + 弥散门 reject reason | — |
| **回归准入** | `run_regression_test`：golden 8维 ≤ 基线×1.05 | 阈值 5% |
| **影子模式** | `category_prompts.source='shadow'`：只记录不改掩模 | N=20 次后评估 |

### 4.2 失败隔离

- 词库更新失败 → 自动 rollback 到 parent_version_id
- 单图处理失败 → 不影响词库学习（episode 标 `error: true`，跳过该条）
- 回归测试失败 → 版本 `regression_status='failed'`，不激活，告警

### 4.3 版本管理

- 每次更新生成新 `version_id`（ISO 8601 时间戳）
- 保留 parent_version_id 链
- 30 天 GC 未激活的旧版本（保留快照 JSON）

---

## 五、计算成本估算

| 模块 | 耗时（单图） | 是否可并行 |
|---|---|---|
| 指纹提取（PCA128） | ~2s | 与 DINO 加载并行 |
| 材质家族判别 | <0.1s（规则）| — |
| 类目检索（Top-50） | <0.5s（SQLite LIMIT 50） | — |
| Auto-Tune（密度带） | ~3s | 可与 DINO 检出后并行 |
| **总增量（最坏）** | **~6s** | 压缩到 ~3s |

阶段 1 只加指纹+检索（~2.5s），对现有流程影响可控。

---

## 六、功能开关（环境变量）

```bash
# 环境变量控制（webui/backend/.env）
ADAPTIVE_MODE=off          # 完全关闭自适应（回退固定预设模式）
ADAPTIVE_MODE=stage1       # 通用词库 + 材质家族（只影响类目选择）
ADAPTIVE_MODE=stage2       # + 图级 Auto-Tune（UI 展示建议）
ADAPTIVE_MODE=stage3       # + 反馈闭环 + 影子进化（异步学习）
ADAPTIVE_MODE=stage4       # + 案例推理库（CBR 检索历史图参数）
ADAPTIVE_MODE=stage5       # + 类目树 + 主动学习（全功能）

# CLI 控制
--semantic-mode=locked|hybrid|auto  # 单次覆盖 preset.mode
--semantic-version=v2026-09-12      # 指定词库版本（调试/验证）
--semantic-freeze                   # 本次运行不更新词库（只读）
```

---

## 七、兼容性保证

### 7.1 向后兼容

- 现有 4 个预设（`chinese_ink_landscape_ai` 等）默认 `mode="locked"` + `auto_evolve=false` → 行为不变
- 词库为空时，回退到硬编码 `AREA_BUDGET_BY_KEYWORD`（双轨制）
- `ADAPTIVE_MODE=off` 时，完全跳过新模块

### 7.2 测试兼容

- `test_golden_layers.py` 改为"核心类目必出 + 审计 8 维不退化"（不固定层数）
- 保留 `test_locked_mode.py`：frozen 预设 → 断言固定层数（专供回归）

---

## 八、待补充（下一份文档）

- **数据迁移脚本**（`migration_001_init_categories.py`）
- **种子数据**（300 条：5 材质 × 20 类目 × 3 prompts）
- **前端 UI Mock**（Auto-Tune 建议卡片、词库版本管理面板）
- **回归基线提取**（从 golden 产物跑一遍审计 8 维，落 `tests/baseline_audit_8d.json`）

---

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 类目数量膨胀 → DINO 前传变慢 | 材质家族裁剪 90% + Top-50 截断 + 权重<0.3 不启用 |
| SQLite 并发写冲突 | 学习器单线程 + 事务隔离 + 版本链 CoW |
| 首次使用冷启动（词库空） | 出厂带 300 条种子 + 5 类材质家族基线 |
| 自进化引入新 bug | 回归准入 + 影子模式 N=20 + 30 天版本 GC |
| 用户反对自动化 | 三档 mode（locked/hybrid/auto）+ preset.auto_evolve 开关 |

---

**下一步**: 阅读 `02-data-schema.md`（数据库 DDL + 种子数据）、`03-implementation-plan.md`（分阶段文件清单 + 测试用例）。
