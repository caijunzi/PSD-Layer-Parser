# 工程记忆手册（MEMORY）

> **定位**：架构决策记录（ADR）、工程铁律、实测避坑数据的**唯一存放处**。
> 顶层开发计划见 `UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md`（SSOT），日常演化见 `DAILY_LOG.md`。
> 自适应语义匹配机制（Stage 1~5）详见 `docs/adaptive-semantics/` 三件套。
> 最后更新：2026-09-14 · 对应计划版本 **v2.4**（自适应语义 Stage 1~5.2 已落地）

---

## 1. 六条工程铁律

| 铁律 | 内容 | 对应算子 | 对应断言族 |
| :--- | :--- | :--- | :--- |
| **R1** | **禁止人工伪色**：输出像素必须来自摄影比色链路，禁止线性公式平涂；**任何生成/补全内容必须显式标记为重建区** | `color_manager` / `deocclusion` | V4-20~24, V4-33 |
| **R2** | **照度归一化只服务检测分支**，不得写回输出像素 | `lighting` | V2-11, V4-21 |
| **R3** | **禁止机械硬拼**：断裂/接缝必须执行流场对齐或几何平直裁切，禁止裸 `vstack` | `seam_harmonizer` | V2-12, V4-30 |
| **R4** | **核心轮廓几何保全**：裁切线必须位于算子检测到的安全下限之下 | `contour_protection` | V4-11, V4-12 |
| **R5** | **掩模门控微孔**：微孔检测仅在有效基材掩模内运行，边缘腐蚀屏蔽 | `micro_holes` | V4-13~16 |
| **R6** | **层合成等价性**（设计线硬约束）：所有可见层按序叠加必须还原原图；隐藏任意单层后可恢复 | `psd_compiler` / `layer_compositor` | V4-34, V4-35 |

> **R1 与 R2 的调和**：检测分支（聚类、微孔、轮廓）用平场后图像；输出分支（制版像素）用未经平场、仅色卡标定的原始色彩。两分支在 `ProcessingContext` 中分开存放，禁止互相污染。

---

## 2. 架构决策记录（ADR）

| ADR | 决策 | 状态 |
| :--- | :--- | :--- |
| ADR-001 | 前端零构建（原生 ESM），不引入 Vite/Webpack；删除重复的 `src/index.html` | ✅ 已定 |
| ADR-002 | PSD 写入用 **pytoshop**、校验用 **psd-tools**，两库职责不合并 | ✅ 已定 |
| ADR-003 | 成品默认 **RLE 压缩**；边长 > 30000 px 或预估文件 > 1.5 GB 时自动切 PSB。**已解决**：集成 `engine/codecs_accelerator.py` 挂载 `imagecodecs` SIMD C 扩展，写盘吞吐达 200 MB/s | ✅ 已落地（v2.4） |
| ADR-004 | 超分为正式流水线阶段，以 Provider 接口实现；默认 `lanczos`（零依赖） | ✅ 已落地（2026-09-10：RealESRGAN 与确定性 Lanczos 按 `output.mode` 分流——PLATE 线强制非生成式） |
| ADR-005 | 引擎数据契约用 `dataclass`（`eq=False`）；配置契约用 Pydantic v2；边界在 `engine/schemas/` | 🟡 部分落地（schemas/：manifest、device_config、profile_config、presets 均 dataclass；Pydantic 未引入） |
| ADR-006 | 本地服务基于标准库，绑定 `127.0.0.1`，强制 `protocol_version = "HTTP/1.1"`，任务模型 `job_id` + SSE | ⏳ 部分实现（WebUI 仍未实现） |
| ADR-007 | TAC 上限与黑版生成从 **ICC profile 派生**，禁止硬编码 300% | ✅ 已落地（2026-09-10 `engine/core/ink_limiter.py`）。⚠️ **概念修正**：实测 ICC 规范（ISO 15076-1）**不含 TAC 字段**——TAC 上限是印刷工艺参数（ISO 12647-2 / 印厂工艺单），现按印刷条件配置；ICC 负责分色与黑版生成（朴素转换 K=0 无黑版，实测证据见尽调报告 §十二） |
| ADR-008 | 安全裁切下限由 `contour_protection` **运行时计算**；preset 中的 718 仅作兜底默认值 | ✅ 已落地（2026-09-10 实测 `cut_y_source=contour_protection`，safe_bottom_y=1950，Y=718 退役） |
| ADR-009 | 全链路中间结果落 `intermediate/<run_id>/`，便于回归比对与缺陷复现 | 🟡 部分落地（manifest 携带 run_id；重建区掩码落 `outputs/<stem>.masks/`） |
| ADR-010 | 遮挡补全限定 `cv2.inpaint` 与 `LaMaInpaintingProvider`（微边缘 <200px 走 Telea 快速旁路，大区域走 LaMa 频域补全 + 单切片 CPU 熔断） | ✅ 已定（v2.4） |
| ADR-011 | **双产品线**（PLATE 制版 / DESIGN 设计），共用内核、在 `compose`+`compile` 阶段分叉；preset 用 `output.mode` 选择 | ✅ 已落地（2026-09-10：`--mode plate/design/both` 端到端实测，both 双产物 `.plate.psb`/`.design.psb`） |
| ADR-012 | 业务图为**封闭 5 类**（壁布/烫金/水墨/屏风/油画）；分割能力采用**零样本模型**（SAM2 + GroundingDINO）；超出 5 类仍转人工 | ✅ 已定（v2.2 修订） |
| ADR-013 | **AI 能力全面接入，但主分发包保持零重依赖**：torch / sam2 / groundingdino 进 `requirements-ai.txt`，未安装自动降级；硬件层采用 **OpenVINO 异构调度**（GPU.1 RTX 5070 独显优先 + GPU.0 Arc 140T 护盾 + CPU 熔断） | ✅ 已定（v2.4） |
| ADR-014 | **分块推理与批处理**：≥ 4000 万像素一律 `tiled_inference`（tile 512、步长 448、Hann 余弦平滑过度），GPU 侧按动态 Batch 4 并行吞吐 | ✅ 已定（v2.4） |
| ADR-015 | 模型放 `models/` 或 `checkpoints/`（不入库），按需加载并校验；启动探测运行时，缺失即降级并在日志明示 | ✅ 已定 |
| ADR-016 | **Qwen-Image-Layered 作为 DESIGN 线可选分割 Provider**，PLATE 线禁用 | ✅ 已定（v2.3） |
| ADR-017 | **PLATE 线补齐陷印（trapping）算子**：专色量化 + 变尺寸陷印，输出独立 `_Spot` 通道层 | ✅ 已接入（2026-09-10：品类可选 `preset.plate_operators.trapping`，需显式配置专色图层来源；屏风品类禁用并在 manifest 如实记录） |
| ADR-018 | 前端分层控制采用 manifest 契约（`layers.json` + `text_manifest.json`），结构参照 Stratum | 🟡 部分落地（2026-09-10：交付级 `DeliverableManifest` 随产物落盘并全量披露 TAC/生成占比/seed/ICC；前端未接） |
| ADR-019 | **五维系统完整性与防欺骗代码审核体系**：环境真实探活、零硬编码静态扫描、物理交付物合规、防伪代码落地、真实基准量化，列入最高工程纪律 | ✅ 已落地（v2.4） |
| ADR-020 | **局部 ROI 裁剪与多核并发超分**：非全画幅图层（印章、题跋、芦雁等）提取紧凑 BBox 并加安全 Padding 局部引导滤波，多核 `ThreadPoolExecutor(max_workers=6)` 并发 | ✅ 已落地（v2.4） |
| ADR-021 | **自适应语义：episode 存储走文件系统 JSONL**，不建 `episodes` 数据库表。Stage 3/4 实际已用 JSONL 归档（`webui/data/adaptive_episodes.jsonl` / `episodes/*.episode.json`），回补表需双写且零收益 | ✅ 已定（2026-09-13，用户决策 A） |
| ADR-022 | **自适应语义：人审通信用轮询**（前端 3 秒 `GET /api/adaptive/pending-feedbacks`），不引入 WebSocket/SSE。理由：后端 FastAPI 无 WS/SSE 基建；轮询零新依赖、延迟 ≤3s 非高频场景够用 | ✅ 已定（2026-09-13，用户决策 B） |
| ADR-023 | **自适应语义：`inherit_priors()` 跳过实现，标 TODO**。`category_priors` 表为空（0 条），无可继承数据；手写 seed 先验是"拍脑袋"，等 Stage 3 反馈学习产生真实统计后再补 | ✅ 已定（2026-09-13，用户决策 C） |
| ADR-024 | **自适应语义端点统一前缀 `/api/adaptive/`**，沿用既有 `webui/backend/api/adaptive.py`，**不新建** `adaptive_semantics.py`（Stage 1~4 端点均在此文件，保持一致） | ✅ 已定（2026-09-13） |
| ADR-025 | **类目树迁移脚本用 `migration_003_build_tree.py`**。`002` 已被 preset 别名占用（`migration_002_apply.py` / `migration_002_preset_aliases.sql`），避免命名冲突 | ✅ 已定（2026-09-13） |

> ### 2026-09-10 深夜增补（当日落地的新决策，状态以本段为准）
>
> | 决策 | 内容 | 实测依据 |
> | :--- | :--- | :--- |
> | **双产品线生成内容强隔离**（§3.1 硬边界 1 落地） | PLATE 线禁用一切生成式输出（超分走 Lanczos、补全走 Telea/NS）；DESIGN 线允许生成但逐层落重建区掩码；`plate_purity` 自动校验 | PLATE 产物 `plate_purity_ok=True`、`generated_pixel_ratio=0` |
> | **RK-16 随机种子固定** | `_seed_everything()` 统一固定 random/numpy/torch；CLI `--seed`（默认 42）；manifest 落盘 | 同 seed 双跑 10 层掩模逐像素一致（md5 全同），耗时差 0.4% |
> | **内核合流（G1）** | 写盘唯一入口收敛到 `core/psd_compiler`；`psb_builder` 降为「dict 层 → LayerDescriptor」适配层；Section 5 权威像素来自超分结果（`section5_planes`） | 内核合流后产物结构与合规指标与合流前一致；Step6 +35s（<10%，反码双向转换代价） |
> | **ICC 分色路径** | `ColorManager.bgr_to_cmyk_raw(bgr, icc_path)`：ICC 驱动 sRGB→CMYK（相对比色意图）+ transform 缓存；无 ICC 回退朴素转换 | 实测朴素转换 **K=0 无黑版**（深色区 CMY 三色叠印），ICC K=97.7%——ICC 为印前必需项 |
> | **Preset SSOT（G2）** | 品类语义唯一真相源 = `preset.layer_semantics`（name_mapping + layer_attributes）；`load_preset` 收敛 `engine/schemas/presets.py`；provider 内置映射删除 | 三态验证：preset 生效 ✓ / 缺失原样保留 ✓ / 自定义可覆盖内置 ✓ |
> | **运行时裁切线（ADR-008 落地）** | `contour_protection` 计算 safe_bottom_y，历史硬编码 Y=718 退役 | 实测 `cut_y_source=contour_protection`，safe_bottom_y=1950 |
>
> 性能基线更新（内核合流后，scale=4.0 全画幅）：PLATE 276.9s / DESIGN 228.6s；
> Step6 写盘 PLATE 145s（RLE 恢复后，TAC 稀疏化省 72s）/ DESIGN 24s。

---

## 3. 实测基线（避坑数据，勿凭记忆改写）

| 项 | 实测值 | 采集时间 / 依据 |
| :--- | :--- | :--- |
| 源图 `inputs/source_4000.jpg` | 4000 × 1952 px | 2026-09-09 |
| 循环单元 `master_cleaned_full767.png` | 388 × 767 px | 2026-09-09 |
| 方案 A 裁切 | 388 × 718 px | 2026-09-09 |
| 成品画布（制版线） | 5315 × 9449 px / 4 通道 CMYK / **8 bit** | 2026-09-09 |
| **16K 终极母版 PSB（设计线）** | **$16000 \times 7808\text{ px}$ / 11 个独立图层 / 150.0 PPI / RGB（mode=3）** | **2026-09-10 复测（`Rosetsu_Master_16k.psb`）** |
| 16K 母版物理体积 | **1.847 GB（1,983,219,201 字节）** ⚠️ 旧值 2.34 GB / 1.08 GB 均不符 | 磁盘实读 + `psd_tools` 打开 |
| 16K 母版对应印刷尺寸 | **$2709.3 \times 1322.2\text{ mm}$**（2.71米 × 1.32米） | Photoshop `0x03ED` 严格校验 |
| 16K 母版合成保真度 | **MAE = 3.498** ⚠️ 旧值 0.73 不复现；**超出 ≤2.0 红线** | `pipeline/06_verify_psb.py`（阈值已于 2026-09-10 由 8.0 收紧为 2.0） |
| **全流程端到端总耗时** | **168.00 秒（2.80 分钟）**（初版 1468.7s，提速 **8.74x** 🚀） | `docs/BENCHMARK_REPORT.md` |
| Step 6 写盘耗时 | **27.66 秒**（纯 Python 原版 1141.65s，提速 **41.3x** 🚀） | C-SIMD PackBits 200 MB/s |
| 图层 BBox | ⚠️ 11 层中仅 5 层为紧凑 BBox，6 层覆盖全画幅 40%~76%（见 §3.0 事故） | 2026-09-10 复测推翻旧结论 |
| psd-tools 版本 | 1.19.0，`psd.header` **不存在** | 2026-09-09 API 校准 |

### psd-tools 1.19 正确 API（校准表）

| 需求 | 正确写法 | 错误写法 |
| :--- | :--- | :--- |
| 通道数 | `psd.channels` | `psd.header.number_of_channels` |
| 色彩模式 | `psd.color_mode == ColorMode.CMYK` | `psd.header.color_mode` |
| 分辨率 | `psd.image_resources.get_data(Resource.RESOLUTION_INFO).horizontal` | `horizontal_resolution` |
| 图层名 | `layer.name.rstrip("\x00")` | `l.name`（带尾零） |
| 图层像素 | `layer.numpy()` → `(H, W, C+1)` float32 | — |

---

## 3.0 重大事故：神经掩模无条件覆盖导致元素层泄漏（2026-09-10 定位并修复）

**现象**：交付产物 `Rosetsu_Master_16k.psb` 中，印章层覆盖 **8.775%** 画幅，
而 `masks_16k` 基线该层仅 **0.0145%**（BBox 310×259），**放大 605 倍，IoU 0.002**。
芦雁 100×、题跋 42×、人物 29×；外框层反向退化为 0.014%（近乎空层）。

**根因**（两条，均已修复）：

1. **神经掩模无条件覆盖**：`GroundedSAMProvider.segment_objects` 中
   `final_masks[k] = m` 直接用神经结果覆盖规则掩模。Grounding DINO 在**金地背景**
   上对 prompt `"red stamp . cinnabar seal"` 产生假阳性大框，SAM 2 在该框内抠出大片
   金地；规则引擎算出的正确印章掩模（0.0119%，BBox 68×63）被静默抹除。
   诱因：`box_threshold=0.25` 过低、单框面积上限 30% 过松。
2. **ROI 检测失效**：`|profile - median(profile)| > 5.0` 判据下，本图外框灰度 88、
   画心金地 183，而行中位数 160 偏向画心，导致几乎所有行都"偏离中位数"，
   ROI 被判为整幅画，`~roi_mask` 恒空 → 外框层退化。

**修复**：

- `grounded_sam_provider`：新增**神经掩模质量门**（面积预算 + 相对放大倍数 + IoU
  三重校验），未通过者保留规则掩模并显式告警；`box_threshold` 0.25→0.35，
  单框面积上限 30%→8%。
- `segmentation_provider`：新增 `_detect_painting_roi`，以「边缘带 / 中心区双参考
  + 连续 run 判定」替代中位数偏差法。

**教训（写入工程纪律）**：

- **任何 AI/神经输出在覆盖确定性算法结果前，必须通过质量门**。不可信的神经结果
  必须回退，而不是无条件信任。
- **质检断言必须用相对 golden baseline 判定**，绝对阈值（如 `len(psd) in [11,15]`、
  面积 < 15%）发现不了「该小却大」的缺陷。

---

## 3.1 开源参考三条硬边界（2026-09-10 定，长期约束）

1. **任何生成式 / 扩散模型的输出不得进入 PLATE 线**。PLATE 线验收标准是可复算、可对色、可过 TAC 审计；扩散输出三者皆不满足，违反 R1（不造假色）与 R6（图层合成等价）。
2. **Stratum（l1thin/stratum）只读不抄**。许可证为 `other` 且 README 明示使用需授权；借鉴范围仅限"数据契约字段语义"与"任务链路阶段划分"，禁止 fork 与复制代码。
3. **Qwen-Image-Layered 权重不入库**，走 ADR-015 按需下载 + SHA256 校验；缺失或显存不足按 §6.9 降级矩阵回落 `RuleBasedProvider`，不得阻断出图。20B / 16GB+ VRAM，属可选依赖，永不进主 `requirements.txt`。

---

## 3.2 实测解决：C 扩展 SIMD PackBits 落地与 pytoshop 补丁（2026-09-10）

- **痛点根因**：`pytoshop 1.2.1` 自带的 RLE 编解码器在纯 Python 下执行逐字节 `while` 状态机循环，在处理 16K 超巨画幅（47 通道，58.7 亿字节）时引发巨量解释器开销，导致 Step 6 写入长达 19 分钟。
- **架构方案**：创建 `engine/codecs_accelerator.py`，无缝挂载已预编译的 `imagecodecs` C-Extension SIMD 汇编级 PackBits 算子，并在 `engine/psb_builder.py` 启动阶段猴子补丁至 `pytoshop.codecs.packbits`。
- **实测收益**：
  - 编码吞吐从 3 MB/s 暴增至 **200 MB/s**；
  - 100 行（1.6 MB）基准测试耗时仅 **0.0080 秒**；
  - 16K 母版写盘耗时从 **1141.65 秒 骤降至 27.66 秒**（提速 **41.3 倍** 🚀）；
  - 字节级解压缩 100% 往返无损匹配，完全合规 Adobe PSB 规范。

---

## 4. 历史踩坑（原始事故复盘）

1. **中段红褐死色块**：拍摄光源偏暖 + 全局硬编码 `Y∈[190,440]` → 误判大块金属为玫瑰金箔。对策：CIELAB 自适应聚类，禁止硬编码 Y 轴分界。
2. **接缝花纹撞车**：上下两截物理断层，横向铜条错位约 100 px（另有 80 px 说法，以实测为准），裸 `vstack` 导致反向卷草纹相撞。对策：Seam Harmonizer 流场对齐。
3. **圆章截肢**：为凑 9:16 在 Y=690 一刀切。对策：安全下限 Y≥718，且由算子运行时算出。
4. **金属变土黄**：线性公式 `gold = 225 - 46*darkness` 平涂。对策：比色链路（R1）。
5. **断裂带假微孔**：纤维毛边被 DoG 误判。对策：掩模门控（R5）。
6. **全屏空转与纯 Python 逐字节编码**：10 张掩模对全画幅 1.25 亿像素逐一做全量引导滤波，且写盘采用纯 Python 循环。对策：非全画幅图层局部 ROI 包围盒裁剪 + 多核 ThreadPool 并发 + C-SIMD PackBits 加速（端到端提速 8.74x）。

---

## 5. 核心流水线与生产状态（2026-09-10 复核）

- **主生产入口**：`run_universal_engine.py` 已完全打通，直接接受 `--preset`、`--scale`、`--dpi` 与 `--profile` 参数。
- **质检与印前验证**：`pipeline/06_verify_psb.py` 与 `tests/audit_system_integrity.py` 构筑双重防线，全量自动化运行无阻塞。
- **设计交付成品**：`outputs/Rosetsu_Master_16k.psb`（实测 **1.847 GB**）已生成；物理结构（16000×7808 / 150 PPI / 中文层名）通过，但**元素掩模质量与合成色差未达标**（见 §3.0）。

---

## 6. 许可核对（M7 前必须完成）

- SAM 2：Apache-2.0（可商用）—— **需按实际使用版本复核**
- Grounding DINO：Apache-2.0（可商用）—— **需按实际使用版本复核**
- Qwen-Image-Layered：Apache-2.0（可商用）—— 权重 20B，**需按实际使用版本与权重来源复核**
- Stratum：**other（非标准开源许可，README 要求显式授权）** —— 未获授权前不得复制其任何代码
- FOGRA / Japan Color ICC：需自备并记录来源与许可（Pillow / psd-tools 不自带）
