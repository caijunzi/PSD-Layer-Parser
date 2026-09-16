# 架构说明书（ARCHITECTURE）

> **本文件是 `UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md` v2.4 的补充视图**，不重复计划中已有的内容。
> 分层架构、模块契约卡、目录结构、数据流状态机 → 见计划 **§4**。
> 本文件补充：**部署与运行时视图、依赖方向规则、异构硬件探测流程、双产品线数据流、实测环境基线与量化基准**。

---

## 1. 部署与运行时视图（插电高性能工作站）

```
┌─────────────────────────── Windows 11 本地单机工作站 ───────────────────────────┐
│                                                                                 │
│  CLI 入口: run_universal_engine.py ──► engine/device_config.py (硬件配置管理)    │
│                                                   │                             │
│                  ┌────────────────────────────────┴──────────────────┐          │
│                  ▼                                                   ▼          │
│      [算法流] engine/                                    [加速层] 核心编解码扩展 │
│        ├─ core/ (几何/色彩/PSB组装)                        ├─ imagecodecs C-SIMD │
│        ├─ operators/ (Frangi/自适应阈值/拓扑排序)          └─ OpenVINO 2026.3.1  │
│        ├─ providers/ (LaMaInpaintingProvider)                        │          │
│        └─ tiled_super_res.py (ROI裁剪+多核ThreadPool)                │          │
│                                                                      ▼          │
│                                                      ┌── 硬件调度矩阵 ─────────┐ │
│                                                      │ GPU.1: RTX 5070 (主算力) │ │
│                                                      │ GPU.0: Arc 140T (防爆盾)│ │
│                                                      │ CPU: Ultra 9 (熔断保底) │ │
│                                                      └─────────────────────────┘ │
│                                                                      │          │
│  产出物: outputs/Rosetsu_Master_16k.psb (1.847 GB, 16000x7808, 150 PPI, 11图层)   │
│  质检链: pipeline/06_verify_psb.py + tests/audit_system_integrity.py (五维全量通过)│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 依赖方向规则（单向，禁止逆向）

```
pipelines  ──►  operators  ──►  core  ──►  numpy / OpenCV / Pillow / pytoshop
    │                │             │
    └────────────────┴─────────────┴────►  schemas (配置契约)
```

**硬性规则**：

1. `core/` **不得** import `operators/` 或 `pipelines/`（内核必须品类无关，G1）。
2. `operators/` 之间不得互相 import（无状态纯函数，§6.1）。
3. `core/` 中不得出现品类关键词（damask / gold / ink / screen），由 Tier 2 静态扫描断言。
4. **只有 `core/psd_compiler` 能写 PSD/PSB 文件**（2026-09-10 内核合流：`UniversalPSBBuilder` 已降级为「引擎 dict 层 → 内核 `LayerDescriptor`」的数据适配层，写盘全部委托内核；§6 的写盘提速数据为优化前历史对照）；pipeline 不得直接写 PSD。
5. AI 依赖只能通过 Provider 接口进入，业务代码不得直接绑死特定重量级框架。

---

## 3. 硬件运行时与异构调度流程（ADR-013 / ADR-014）

```
启动 (run_universal_engine.py)
  │
  ├─ 探测 OpenVINO 硬件设备树
  │         │
  │         ├─ 命中 GPU.1 (RTX 5070) ──► 设为主算力单元 (Batch=4 向量推理)
  │         ├─ 命中 GPU.0 (Arc 140T) ──► 设为超大画幅 16GB 防爆显存护盾
  │         └─ CPU / NPU ──────────────► 设为三级热备
  │
  ├─ 运行时单切片级熔断回路 (Tile-Level Circuit Breaker)
  │         │
  │         ▼ 显存溢出 / 驱动异常?
  │         ├─ 否 ──► 完成 GPU.1 张量推理
  │         └─ 是 ──► 毫秒级捕获，当前切片自动回落 CPU/Telea，不中断整图
  │
  └─ C 扩展 SIMD PackBits 挂载 (engine/codecs_accelerator.py)
            │
            ▼ 猴子补丁注入 pytoshop.codecs.packbits
            达成 200 MB/s 极限写盘吞吐
```

---

## 4. 双产品线数据流

```
                  ┌──────────── 共用内核 ────────────┐
源图 ─► geometry ─► lighting(检测分支) ─► 算子提取 ─► super_resolution
                                                        │
                                    ┌───────────────────┴──────────────────┐
                                    ▼                                      ▼
                            制版线 PLATE                             设计线 DESIGN
                        color_manager → CMYK                    元素掩模 → RGBA 元素层
                        psd_compiler (RLE/PSB)                  psd_compiler (bbox/alpha)
                                    │                                      │
                               印刷厂 / 制版机                        设计师微调 → 回灌制版线
```

分叉点只在 `compose`（组装什么层）与 `compile`（用什么色彩模式与几何），由 preset 的 `output.mode = plate | design | both` 决定（ADR-011）。

---

## 5. 实测环境基线（2026-09-10 复核）

| 项 | 实测版本 / 状态 | 验证依据 |
| :--- | :--- | :--- |
| Python | **3.12.10 (64-bit)** | 宿主运行时 |
| OpenVINO | **2026.3.1** | 成功驱动 Blackwell RTX 5070 与 Arc 140T |
| imagecodecs | **2024.12.30** | SIMD C 扩展 PackBits 编解码 |
| onnxruntime | **1.20.1** | CPU 备选推理引擎 |
| numpy | **2.4.6** | 基础数组运算 |
| OpenCV | **5.0.0.93** | 释放 GIL 多线程图像计算 |
| Pillow | **12.2.0** | 色彩空间与元数据 |
| psd-tools | **1.19.0** | 印前解析校验库 |
| pytoshop | **1.2.1** | PSD/PSB 物理流编译库（经 C-SIMD 补丁） |

---

## 6. 性能基准（真实微秒级实测数据）

| 流水线阶段 | 原始基线 (纯 Python) | 当前版本 (C-SIMD + 深度优化) | 实测提速 | 物理突破原理 |
| :--- | :---: | :---: | :---: | :--- |
| **[Step 1] 语义对象动态分割** | 28.85s | **16.21s** | 1.78x | 动态直方图 + 多尺度 Hessian 提取 |
| **[Step 2] 画布底板去墨重构** | 21.30s | **13.71s** | 1.55x | RTX 5070 (GPU.1) 频域补全推理 |
| **[Step 3] 2.5D 层序拓扑排序** | 0.21s | **0.15s** | 1.40x | 接触图 Kahn 拓扑排序 |
| **[Step 4] 2.5D 遮挡定向补全** | 184.41s | **95.48s** | **1.93x ⚡** | 微边缘快速旁路 + 余弦 Hann 窗 + Batch 4 |
| **[Step 5] 阶梯引导式超分辨率** | 92.28s | **14.78s** | **6.24x ⚡** | 局部 ROI 包围盒裁剪 + 6 核 ThreadPool 并行 |
| **[Step 6] 16K PSB 编码与写盘** | 1141.65s (19分) | **27.66s (0.46分)** | **41.3x ⚡** | imagecodecs SIMD C 扩展 (200 MB/s) |
| **端到端总耗时 (Total)** | **1468.70s (24.5分)** | **168.00s (2.80分)** | **8.74x 🚀** | **整图生产提速 8.74 倍**（注：192.66s 与 168.00s 的分项差为测量口径差异，以实测为准） |

---

## 7. 实现状态注记（2026-09-11 凌晨，内核合流后）

> 上文 §6 的 8.74× 是「相对纯 Python 初版」的优化对照（历史基线）；**当前双产品线基线**如下。

### 7.1 写盘路径（G1 已收敛）

```
run_universal_engine（引擎 dict 层结构）
        │
        ▼
UniversalPSBBuilder  ←  仅做数据适配：dict 层 → core.LayerDescriptor（逻辑墨量/RGBA + 紧凑 bbox），
        │                 Section 5 权威像素来自超分结果（context.section5_planes）
        ▼
core.PsdCompiler.compile_psd  ←  唯一写盘方：通道构造（ColorChannel 枚举）、反码唯一转换点
        │                         （_to_disk）、PSD/PSB 版本决策、RLE 探测、分辨率块 0x03ED
        ▼
pytoshop（经 codecs_accelerator 注入 imagecodecs SIMD PackBits）
```

### 7.2 双产品线当前基线（scale=4.0 全画幅，`--seed 42`）

| | PLATE 制版线 | DESIGN 设计线 |
| :--- | :--- | :--- |
| 端到端 | **276.9s** | **228.6s** |
| 产物 | CMYK（mode=4），1.73 GB | RGB（mode=3），1.45 GB |
| 超分 | Lanczos 阶梯（非生成式） | RealESRGAN（生成式，Arc 140T） |
| 补全 | Telea/NS 确定性 | LaMa（GPU.1） |
| 合规 | TAC≤300%、MaxK≤96%、`plate_purity_ok=True` | 生成内容逐层落重建区掩码 |
| sidecar | `<name>.manifest.json` + `<name>.masks/` | 同左 |

### 7.3 已知约束

- 分割（GroundingDINO/SAM2）为 torch 路径，RTX 5070（sm_120）需 torch ≥2.7 才可用，
  当前跑 CPU——升级后 Step1（现 ~62-124s）有望大幅下降（见尽调报告 §十）
- `--scale` CLI 参数会被 preset 的 `super_res_scale` 覆盖（优先级待修）
- 封闭 5 类中仅屏风完成端到端验证；新增品类只改 preset（G2）

---

## 8. 自适应语义匹配机制（Stage 1~5，2026-09-13 ~ 09-14 新增）

> 独立文档体系见 `docs/adaptive-semantics/`（架构 / 数据 schema / 实现计划三件套）。
> 本节为架构层索引，便于从 ARCHITECTURE 直通。

### 8.1 分层位置

```
                        自适应语义匹配机制（新增层）
                                  │
    交互层（WebUI / CLI）          │
        │                          │
        ├── ► 图级自适应层（每次运行现算，不入库）
        │      指纹 PCA128 → 材质判别 → CBR 检索（Stage 4）
        │      → Auto-Tune（Stage 2）→ 类目选择（Stage 1）
        │
        ├── ► 品类模板层（preset，低频进化）
        │      preset.mode = locked/hybrid/auto
        │
        ├── ► 通用语义词库（SQLite，持续进化）
        │      webui/data/adaptive_semantics.db
        │      类目树（Stage 5.1，3 层）+ prompts 权重
        │
        └── ► 异步学习流程（Stage 3 / 5.2）
               episode 归档（JSONL）→ 归因 → 护栏 → 影子模式
               → 人审反馈（轮询）→ 权重更新
                                  │
                                  ▼
                    机制内核（DINO/SAM/超分/补全/审计，不可变）
```

### 8.2 Stage 完成矩阵

| Stage | 内容 | 状态 | Commit |
| :--- | :--- | :--- | :--- |
| 1 | 通用词库 + 材质匹配 + 类目选择 | ✅ | `0b321f6` |
| 2 | 图级 Auto-Tune（引擎 + API + 前端卡片） | ✅ | `909057a` / `e890dbd` |
| 3 | 反馈闭环 + 影子进化（核心骨架） | 🟡 | `21e8f90` |
| 4 | CBR 案例推理库（指纹索引 + 余弦检索） | ✅ | `b9eb5b0` |
| 5.1 | 类目树管理（3 层 + 无环检测） | ✅ | `b74ab10` |
| 5.2 | 主动学习（不确定类目人审 + 轮询 API） | ✅ | `ae39733` |
| 5.3 | 前端人审弹窗 + Auto-Tune 卡片重构 | ✅ | `b7241a3` |
| 补完 | rename/delete/merge 真实实现 + e2e 测试 | ✅ | `906256b` |
| 修复 | 主动学习 / Stage 3 学习闭环**生产接线**（链路断裂）+ 接线守卫 | ✅ | `5dd1556` |

### 8.3 三条架构决策（与 ADR-013 等并列）

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| **episode 存储** | 文件系统 JSONL（`webui/data/*.jsonl`），**不建 `episodes` 表** | Stage 3/4 实际已用 JSONL，回补表需双写且零收益 |
| **人审通信** | 轮询 `GET /api/adaptive/pending-feedbacks`（前端 3s） | FastAPI 无 WS/SSE 基建；轮询零新依赖、延迟 ≤3s 够用 |
| **`inherit_priors()`** | 空表时优雅跳过，有真实先验时复制 | `category_priors` 当前为空；不写未经验证的 seed，避免污染类目先验 |

> ⚠️ **注意**：`01-architecture.md` 中描述的 `episodes` 数据库表与 WebSocket/SSE 推送
> **均未落地**，实际以本节与 `03-implementation-plan.md` 的修正注记为准。

### 8.4 测试基线（2026-09-16）

引擎全量：**285 passed / 3 skipped**；WebUI：**41 passed**。
专项回归包含真实绢本工笔样本分类、类目字段桥接、episode 审计回填和 CBR 接线。九样本隔离 cold/repeat 结果见 `outputs/adaptive-e2e-20260915-rerun/results.json`，完整报告见 `docs/测试报告_20260915_全样本自适应链路收口.md`。

上一轮复测共 18 次运行，18 个 PSB 均生成，9/9 样本达到 cold/repeat 字节可复现，生产 DB 哈希未变化。8/18 次通过 8 维审计；水墨 3 张和油画的 repeat 均满足日志白名单命中及 episode `notes` 中 `cbr_reused=1`。

**2026-09-16 修复批次（四项工程缺口）**：

| 缺口 | 根因 | 修复 | 实测 |
| :--- | :--- | :--- | :--- |
| ⑤ 合成等价性 | PLATE 产物为 ICC 真分色（有黑版），参考图却用 PIL 朴素 `convert('CMYK')`（K≡0）→ K 通道错配、RMSE 虚高 | 参考图改用**同一 ICC** 分色比对；`color_managed` 标记 + ICC 缺失回退 | 金地 36.49→**0.96**、商用图 32.51→**2.43**，均转全过 |
| 指纹 PCA | `_pca_model="placeholder"`，训练函数生产零调用；embedding 未标准化 | `tools/train_pca.py` + 持久化 + `_apply_pca` 自动加载；样本不足只标准化不降维 | scaler 为真实统计（`mean=[1.95,4.29,72.2…]`），84 维同量纲 |
| CBR 参数 | `suggest_auto_tune` import 却从未调用，`global_percentiles` 硬编码 `{}` | 真实计算并入 episode；推荐带单列 `*_suggested`，**不自动注入生产** | `p50=0.0592 p90=0.1899`；产物 SHA 不变（可复现保持） |
| `category_priors` | 空表 → `inherit_priors` 空转 | `tools/build_category_priors.py` 从真实检出统计 + 父子上卷 | 写入 19 条；二级/根已有先验，继承可用 |

**绢本工笔 preset 修正**：绢本两张此前误用 `textile_damask`（壁布类目），与题材（花鸟）不匹配 →
④丢 8.712%/20.004%、⑤ RMSE 26.72/44.63 失败。改用 `chinese_ink_landscape_ai` 后 **8 维全过**：
④ 降至 0.004%/0.292%，⑤ 降至 1.86/2.17。**根因是 preset 选择错误**。

**残余项：壁布 plate（`damask_sample.png`）未通过 —— 三层根因（2026-09-16 定案）**

> ⚠️ 该结论经一次误判与修正：曾用「金色像素占比 9.412%」**推断**类目命中并据此
> 宣布"前提被推翻"。那只证明「图上有金色」，**不能证明 DINO 检出了该类别**。
> 属以派生统计替代检测结果的过度推断。最终以 `cold.log` 第一手检出记录为准。

1. **素材前提不满足**：该图是壁布**实物样品照**（画面外背景 / 画布硬边缘 / 右侧折边 /
   右下角「AI生成 WORKBUDDY」水印），上下平铺接缝差 **20.44**（图内 std 仅 22.10）
   → **不可无缝平铺**；而 `textile_damask` 含 `seam_harmonization: cyclic_vertical`。
   硬边界 row 48~52/968~970，col 75~78/1118/1454~1458。
2. **DINO 对该 preset 的 2 个壁布类目零检测**：检出全是屏风系
   （09_Calligraphy_Inscription / 06_Architecture_Pavilion / 05_Trees_Vegetation /
   03_Distant_Mountains），bbox 近全画布被**弥散门**拒 4 个。
3. **adaptive 覆盖与 allowlist 冲突（真代码缺陷）**：`run_universal_engine.py` 第 574/637 行
   用 adaptive 从 DB 选的类目**覆盖** `preset["ai_semantic_classes"]`，而 DB 的 27 个类目
   全在「山水画」树下 → 检出屏风系 → 被
   `rule_class_allowlist=["02_巴洛克团花_Baroque_Medallion"]` **全部丢弃** → **零语义掩模**
   → 产物退化为「底板+残层+2 工艺层」（4 层）→ 内容全丢、⑤ RMSE 40.31。
   **已修**：清空全部产出时打印明确告警（白名单内容 + 根因 + 处置 + 不得放宽阈值）。
   **产物字节不变**（SHA `666de161…`，实测），未破坏可复现性。

处置待定：A 转 DESIGN 线（如实标注素材前提）/ B 换真正可平铺素材 /
C 新增「壁布样品照」独立 preset。**任何方案下都不得放宽 ④/⑤ 阈值。**

**审计量测缺陷批次（2026-09-16 第二轮，`tools/audit_psb.py`）**

排查上节「④ 假通过」时发现更深一层的量测缺陷——**其影响面比 ④ 本身更大**：

| 缺陷 | 根因 | 影响 | 修复 |
| :--- | :--- | :--- | :--- |
| **`_alpha` 取错通道** ★ | `psd_tools` 的 `layer.numpy()` 通道数随模式变化：RGB→(H,W,**4**) Alpha 在 index 3；CMYK→(H,W,**5**)=C,M,Y,K,Alpha 在 index **4**。旧代码写死 `a[:, :, 3]`，在 CMYK 上取到 **K 通道** | ICC 分色后 K 呈色恒 1.0 → 掩码全判「全画布」→ **②③④⑥ 四维全错**。实测 union 修复前**恒 100%**，修复后 38.1/63.77/69.95%；**RGB（design 线）完全不受影响** | 按通道数取 alpha（`>=5`→4，`==4`→3，`<4`→全 1） |
| ④/⑥ 的 CMYK→RGB 基准错位 | ④ 用 `ba[:, :, :3]`，在 CMYK 层上把 **C,M,Y 当成 R,G,B** 与源图比对 | 「底板 vs 源图」差异定位失真 | 新增 `_layer_rgb`：psd_tools 对 CMYK 返回**呈色**（=1−墨量），故 `R=v0*v3, G=v1*v3, B=v2*v3` |
| ④ 的 union 含加工层 | 加工层（冲孔/烫金/陷印/专色）整版施加、Alpha 天然全画布 | 任一加工层即把 union 撑到 100%，`lost` 恒 0 → **假通过** | 新增 `TH["process_names"]` 排除；披露 `carrier_layers` / `excluded_process_layers` |
| allowlist 静默清空 | 白名单与 adaptive 覆盖后的类目不相交时产出 0 掩模，此前静默继续 | 生成垃圾产物且不被察觉 | 清空时打印明确告警（**行为未变**，产物字节一致） |

> ⚠️ `⑦ plate 合规` 里的 `a[:, :, 3]` 是**有意**取 K 通道（真黑版判定），
> 已收紧为 `shape[2] >= 5` 并加注释。**修改 `_alpha` 时不要连带改它。**
>
> ⚠️ `_layer_rgb` 的 CMYK 转换是**无 ICC 近似**，仅用于「底板 vs 源图」阈值 40 的粗判；
> **⑤ 合成等价性必须走同一 ICC**，不受影响。
>
> 新增测试：`TestAlphaChannelLayout` / `TestLayerRgbConversion` /
> `TestProcessLayersExcludedFromCarrier`（`tests/test_audit_psb.py`，共 +11 条）。

### 8.5 测试环境（重要）

跑本项目测试**必须用系统 Python 3.12.10**
（`C:\Users\CK\AppData\Local\Programs\Python\Python312\python.exe`，含 numpy 2.4.6 / sklearn 1.9.1 /
cv2 5.0.0 / psd_tools 1.19.0）。WorkBuddy managed Python 3.13.12 **无 numpy**。
