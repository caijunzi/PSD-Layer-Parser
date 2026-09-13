# Universal Layer Studio PRO

工业级 2D 智能图像分层与印前制版工作站。**单一真相来源（SSOT）**：`docs/UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md`（v2.4）。

| 文档 | 用途 |
| :--- | :--- |
| `docs/UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md` | 顶层开发计划（目标/架构/契约/流水线/质检/里程碑） |
| `docs/BENCHMARK_REPORT.md` | **全量流水线效率与性能实测基准报告（秒级精确实测）** |
| `docs/MEMORY.md` | ADR 决策、六条铁律、实测基线、避坑 |
| `docs/ARCHITECTURE.md` | 架构与运行时视图 |
| `docs/DAILY_LOG.md` | 每日演化日志 |
| `docs/技术尽调与代码审查报告_20260910.md` | **第三方视角全量尽调：实测复核、完成度评估、P0/P1/P2 改进建议（§十二为内核合流后全量回归与状态对齐）** |
| `docs/技术栈适配性专业评估与建议_20260910.md` | **硬件实效实测 / 技术栈匹配度评估 / 三条路线建议** |
| `docs/文档对齐清单_20260910.md` | 文档对齐记录 |
| `docs/adaptive-semantics/` | **自适应语义匹配机制**（Stage 1~5）：`01-architecture.md` / `02-data-schema.md` / `03-implementation-plan.md` |
| `GEMINI.md` | AI 入口摘要与停止条件 |

> **当前状态速览（2026-09-11 凌晨，内核合流后）**：双产品线均已端到端跑通——
> PLATE 线输出真 CMYK（ICC FOGRA39 分色、有黑版、TAC≤300%、零生成内容、纯净性自动校验）；
> DESIGN 线输出 RGB 元素层（RealESRGAN + LaMa，生成内容逐层落掩码）；
> `--mode both` 一次产出两份。运行时裁切线由 contour_protection 计算（Y=718 硬编码退役）；
> 随机种子固定（--seed，同 seed 产物逐像素可复现）。详见尽调报告 §十二。

---

## 一、这个系统做什么

面向**软装壁布、金属烫印面料、传统书画复刻、商业专色网印**的分层与制版。业务图范围为**封闭 5 类**：

| 品类 | Preset 文件 | 状态 |
|---|---|---|
| 屏风（日本金地） | `japanese_screen_gold.json` | ✅ 端到端验证通过 |
| 壁布（大马士革） | `textile_damask.json` | ✅ 端到端验证通过 |
| 水墨山水 | `chinese_ink_landscape_ai.json` | ✅ Preset 完整 |
| 烫金（含金地屏风变体） | `japanese_screen_gold.json` | ✅ Preset 完整 |
| 油画（西洋古典） | `western_oil_painting.json` | ✅ Preset 完整（2026-09-11 新增）|

超出范围转人工，不承诺任意图全自动（ADR-012）。

最终业务流是三段：

```
① 拆解（设计线，RGB 元素层） → ② 设计师微调/重组/排版 → ③ 制版（制版线，CMYK 分色）
```

### 双产品线

| | 制版线 `PLATE` | 设计线 `DESIGN` |
| :--- | :--- | :--- |
| 给谁 | 印刷厂 | 设计师 |
| 色彩模式 | CMYK 分色版 | RGBA 可编辑 |
| 层语义 | 工艺版层（固定，由工艺决定） | 元素层（不定，由画面内容决定） |
| 层几何 | 全画布 | 元素紧凑最小外接包围盒 (Non-zero BBox) |
| 命名 | `03_Print_Antique_Gold_CMYK` | `06_Architecture_Pavilion` |
| 状态 | 🔴 **未打通**：微孔/烫金/陷印/轮廓算子已实现，但**均未接入生产链路**，实测产物为 RGB | 🟡 能产出 16K PSB；元素掩模泄漏已修复，**待重跑产物复核** |

---

## 一之二、自适应语义匹配机制（Stage 1~5，2026-09-13 ~ 09-14 新增）

让引擎「越用越准」的自进化层：换图免调参、冷启动复用历史、不确定类目请求人审。
独立文档见 `docs/adaptive-semantics/`。

| Stage | 内容 | 状态 |
| :--- | :--- | :--- |
| 1 | 通用词库 + 材质匹配 + 类目选择（SQLite `webui/data/adaptive_semantics.db`） | ✅ |
| 2 | 图级 Auto-Tune（墨密度场 → region / 密度带建议 → 一键采纳写回 preset） | ✅ |
| 3 | 反馈闭环 + 影子进化（归因 / 回归基线 / 学习器，核心骨架） | 🟡 |
| 4 | CBR 案例推理库（PCA128 指纹余弦检索，复用相似历史图参数，冷启动加速） | ✅ |
| 5.1 | 类目树（3 层：山水画 → 7 大类 → 20 类目，含无环检测） | ✅ |
| 5.2 | 主动学习（识别 confidence<0.7 类目 → 人审 → 权重更新，轮询方案） | ✅ |
| 5.3 | 前端人审弹窗 + Auto-Tune 卡片重构（新建 `components/`） | ⏳ 待做 |

**三条关键架构决策**（ADR-021/022/023，详见 `docs/MEMORY.md`）：
1. episode 存储走**文件系统 JSONL**，不建 `episodes` 数据库表
2. 人审通信用**轮询**（`GET /api/adaptive/pending-feedbacks`），不引入 WebSocket/SSE
3. `inherit_priors()` **跳过实现标 TODO**（`category_priors` 表为空，等真实统计）

> ⚠️ 引擎全量测试基线 **128 passed / 2 skipped**。

---

## 二、六条铁律（改动前先对照）

| 编号 | 内容 |
| :--- | :--- |
| R1 | 禁止人工伪色；输出像素走比色链路，**生成/补全内容必须标记为重建区** |
| R2 | 照度平场归一化只服务检测分支，不得写回输出像素 |
| R3 | 禁止裸 `vstack` 硬拼，接缝走流场对齐或几何平直裁切 |
| R4 | 核心轮廓几何保全，裁切线不得低于算子算出的安全下限 |
| R5 | 微孔检测仅在有效基材掩模内运行 |
| R6 | 层合成等价性：可见层叠加必须还原原图 |

---

## 三、安装与运行

### 运行环境要求
- Python 3.12+ (64-bit)
- 硬件推荐：Intel Core Ultra 9 / NVIDIA RTX 50 系列独显 (Blackwell sm_120) / Intel Arc 独立或核心显卡
- 依赖安装：
```bash
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

### 使用方式

#### 方式 1：WebUI（推荐，开发中）

```bash
# 启动 Web 界面
cd webui/backend
python main.py

# 浏览器访问 http://localhost:8099
```

**特性**：
- 🎨 拖拽上传、可视化配置
- 📊 实时进度与日志推送
- 🔍 智能 preset 推荐
- 📦 批量结果管理

详见 [webui/README.md](webui/README.md) 与 [docs/WEBUI_TECHNICAL_SPEC.md](docs/WEBUI_TECHNICAL_SPEC.md)（12000 字完整技术设计）

---

#### 方式 2：命令行（稳定）

### 1. 全流程端到端 16K 母版生产
```bash
python -u run_universal_engine.py --input inputs/source_4000.jpg --output outputs/Rosetsu_Master_16k.psb --preset japanese_screen_gold --scale 4.0 --dpi 150.0 --profile robust_performance
```

### 2. 生产交付物印前规范解析验证
```bash
python pipeline/06_verify_psb.py --file outputs/Rosetsu_Master_16k.psb --dpi 150.0
```

### 3. 五维系统完整性与防欺骗代码审核
```bash
python tests/audit_system_integrity.py outputs/Rosetsu_Master_16k.psb
```

---

## 四、目录速览

```text
engine/core/        内核（geometry / lighting / color_manager / psd_compiler / models）
engine/operators/   算子（micro_holes / metallic_foil / contour_protection / seam_harmonizer）
engine/providers/   AI/算法可插拔 Provider（LaMa 频域补全 / OpenVINO 硬件分发）
engine/codecs_accelerator.py  C-Extension SIMD PackBits RLE 高性能编解码器
engine/device_config.py       异构硬件与防护模式配置管理器
presets/            品类预设（japanese_screen_gold 等 5 类）
pipeline/           设计交付线核心流水线与质检脚本（01-06）
tests/              五维系统完整性与防欺骗自动化审计测试套件
outputs/            生产交付物存储目录（Rosetsu_Master_16k.psb）
docs/               单一真相来源技术文档中心与基准测试报告
```

---

## 五、系统状态与实测基线（⚠️ 历史快照：掩模泄漏修复前的产物，最新验收见 §六）

> ⚠️ **2026-09-10 复核更正**：下表数值为**实际打开产物文件复测**所得，取代此前文档中的
> 2.34 GB / MAE 0.73 / 15 图层 等互相矛盾的表述。完整核查见
> `docs/技术尽调与代码审查报告_20260910.md`。
>
> 当前 `outputs/Rosetsu_Master_16k.psb` 是**掩模泄漏修复前**生成的产物，
> 元素层存在严重缺陷（见下）；修复已并入代码，**需重跑生产后复核**。
>
> ⚠️ **下表为 2026-09-10 白天的历史快照**（泄漏修复前产物）。同日深夜已完成：
> 掩模质量门落地（泄漏消除 + 审计重写）、双产品线（PLATE 真 CMYK / DESIGN RGB）、
> ICC 分色 + TAC 合规、内核合流。**最新验收状态见本表末尾「内核合流后验收」小节**
> 与尽调报告 §十二，下列缺陷行仅作历史对照保留。

| 维度 | 状态 / 实测指标 | 验证依据 |
| :--- | :--- | :--- |
| **产出文件** | `outputs/Rosetsu_Master_16k.psb`（**1.847 GB**，1,983,219,201 字节） | 磁盘实读（旧文档称 2.34 GB / 1.08 GB，均不符） |
| **画幅与分辨率** | **$16000 \times 7808$** @ **150.0 PPI**（物理尺寸 $2709.3 \times 1322.2\text{ mm}$） | Photoshop 资源块 `0x03ED` 严格通过 |
| **色彩模式** | ~~RGB（mode=3）—— PLATE/CMYK 未落地~~ ❗**已过时**：2026-09-10 深夜起 PLATE 线产出真 CMYK（mode=4），见 §六 | `psd_tools` 实测 |
| **图层完整性** | **11 个独立图层**，但**6 层掩模存在泄漏或空层** | `tests/audit_system_integrity.py` 判定 FAIL |
| **元素掩模质量** | 🔴 印章层覆盖 **8.775%** 画幅（基线 0.0145%，**放大 605 倍**，IoU 0.002）；芦雁 100×、题跋 42×、人物 29×；外框层近乎空层 | 对照 `masks_16k` golden baseline |
| **图像保真度** | Section 5 合成与源图 **MAE = 3.498** —— **超出 ≤2.0 红线**（旧文档称 0.73） | `pipeline/06_verify_psb.py` |
| **全流程总耗时** | **168.00 秒（2.80 分钟）**（初版 1468.7s，提速 **8.74 倍**） | `docs/BENCHMARK_REPORT.md` |
| **硬件调度** | GPU.1 (RTX 5070) + GPU.0 (Arc 140T 护盾) + CPU 熔断回路 | OpenVINO 2026.3.1 实测 |
| **C 扩展写盘** | imagecodecs SIMD PackBits 实测 **207~358 MB/s**，为纯 Python 的 **43~100 倍** | 新版 D5 开关对比（旧版仅断言属性替换） |

### 修复进度（2026-09-10 · 已端到端复核）

重跑产物：`outputs/Rosetsu_Master_16k_fixed.psb`（**1.45 GB**，16000×7808，150 PPI，11 图层）

| 图层 | 修复前 fill% | 复核产物 fill% | 基线 fill% | IoU |
| :--- | ---: | ---: | ---: | ---: |
| 09A 印章 | 8.7752 | **0.0115** | 0.0145 | **0.776** |
| 08 芦雁 | 6.6328 | **0.1498** | 0.0662 | 0.147 |
| 07 人物 | 6.7688 | **0.2683** | 0.2361 | 0.314 |
| 09B 题跋 | 2.7453 | **0.2474** | 0.0652 | 0.259 |
| 10A 外框 | 0.0143（空层） | **24.9424** | 11.1555 | 0.119 |
| 10B 折痕 | IoU 0.094 | 0.5127 | 0.4758 | **0.901** |

图层包围盒（设计师可拖动性）：印章 `10585×5464 → 300×280`、题跋 `11804×5260 → 756×840`、
人物 `10585×3185 → 556×929`。

验证命令：

```bash
python tests/diagnose_mask_quality.py                 # 元素掩模质量诊断（对照 masks_16k 基线）
python tests/audit_system_integrity.py <psb>          # 五维审计（可回归版）→ 新产物 PASS
python tests/verify_r6_composition.py <psb>           # 铁律 R6 层合成等价性 → MAE 4.880 通过
python pipeline/06_verify_psb.py --file <psb>         # 印前结构校验
```

> ⚠️ 已知偏差：`06_verify_psb.py` 的 MAE（3.498）比对的是「Section 5 vs 源图」，
> 实测在修复前后两版产物上数值**完全相同**，说明它只反映超分链路差异、
> **对掩模正确性不敏感**。掩模正确性请以 `audit_system_integrity.py` 与
> `verify_r6_composition.py` 为准。

---

## 六、内核合流后验收（2026-09-11 凌晨 · 全量回归通过）

双产品线端到端实测（`--seed 42`，`--icc CoatedFOGRA39`，scale=4.0 → 16000×7808）：

| 维度 | PLATE 制版线（`--mode plate`） | DESIGN 设计线（`--mode design`） |
| :--- | :--- | :--- |
| **产物** | `*.plate.psb`（1.73 GB） | `*.design.psb`（1.45 GB） |
| **色彩模式** | **CMYK（mode=4）**，每层 5 通道（CMYK+α） | RGB（mode=3），每层 4 通道（RGBA） |
| **黑版** | ✅ 真黑版（ICC FOGRA39 分色，K 最大 96%） | —（RGB 无黑版概念） |
| **墨量合规** | ✅ TAC 330.2% → **300.0%**（ECI 实践值，MaxK 96%） | — |
| **生成内容** | **零生成**（Telea/NS 确定性补全 + Lanczos 插值超分），`plate_purity_ok=True` | 允许生成（RealESRGAN + LaMa），逐层落重建区掩码 |
| **端到端** | 276.9s（此前 294.6s） | 228.6s |
| **运行时裁切线** | `contour_protection` 计算 `safe_bottom_y=1950`（Y=718 硬编码退役，ADR-008） | 同左（规则共用品类预设） |
| **可复现性** | ✅ 同 seed 双跑掩模逐像素一致（RK-16） | ✅ 补全像素与 PLATE 完全一致（434310） |

`--mode both` 一次产出双份：`<name>.plate.psb`（给印刷厂，审计凭据 `plate_purity_ok=True`）+
`<name>.design.psb`（给设计师，生成内容逐层落掩码）——即 ADR-011 双产品线的完整实现。

交付物新增 sidecar：`<name>.manifest.json`（有效 PPI / 放大倍率 / TAC / 生成占比 / seed / ICC
全量披露）+ `<name>.masks/`（逐层重建区掩码 PNG）。**回灌制版线时必须读取掩码剔除生成内容**
（§3.1 硬边界 1 的落地要求）。

已知限制：封闭 5 类 preset 已全部完成（屏风/壁布端到端验证通过，水墨/烫金/油画 preset 完整待测试）；
WebUI（G4）未实现；ICC 黑版生成曲线按品类调优待做。
