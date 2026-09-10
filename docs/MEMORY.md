# 工程记忆手册（MEMORY）

> **定位**：架构决策记录（ADR）、工程铁律、实测避坑数据的**唯一存放处**。
> 顶层开发计划见 `UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md`（SSOT），日常演化见 `DAILY_LOG.md`。
> 最后更新：2026-09-10 · 对应计划版本 **v2.4**

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
| ADR-004 | 超分为正式流水线阶段，以 Provider 接口实现；默认 `lanczos`（零依赖） | ⬜ 待实现 |
| ADR-005 | 引擎数据契约用 `dataclass`（`eq=False`）；配置契约用 Pydantic v2；边界在 `engine/schemas/` | ⬜ 待实现 |
| ADR-006 | 本地服务基于标准库，绑定 `127.0.0.1`，强制 `protocol_version = "HTTP/1.1"`，任务模型 `job_id` + SSE | ⏳ 部分实现 |
| ADR-007 | TAC 上限与黑版生成从 **ICC profile 派生**，禁止硬编码 300% | ⬜ 待实现 |
| ADR-008 | 安全裁切下限由 `contour_protection` **运行时计算**；preset 中的 718 仅作兜底默认值 | ⬜ 待实现 |
| ADR-009 | 全链路中间结果落 `intermediate/<run_id>/`，便于回归比对与缺陷复现 | ⬜ 待实现 |
| ADR-010 | 遮挡补全限定 `cv2.inpaint` 与 `LaMaInpaintingProvider`（微边缘 <200px 走 Telea 快速旁路，大区域走 LaMa 频域补全 + 单切片 CPU 熔断） | ✅ 已定（v2.4） |
| ADR-011 | **双产品线**（PLATE 制版 / DESIGN 设计），共用内核、在 `compose`+`compile` 阶段分叉；preset 用 `output.mode` 选择 | ✅ 已定 |
| ADR-012 | 业务图为**封闭 5 类**（壁布/烫金/水墨/屏风/油画）；分割能力采用**零样本模型**（SAM2 + GroundingDINO）；超出 5 类仍转人工 | ✅ 已定（v2.2 修订） |
| ADR-013 | **AI 能力全面接入，但主分发包保持零重依赖**：torch / sam2 / groundingdino 进 `requirements-ai.txt`，未安装自动降级；硬件层采用 **OpenVINO 异构调度**（GPU.1 RTX 5070 独显优先 + GPU.0 Arc 140T 护盾 + CPU 熔断） | ✅ 已定（v2.4） |
| ADR-014 | **分块推理与批处理**：≥ 4000 万像素一律 `tiled_inference`（tile 512、步长 448、Hann 余弦平滑过度），GPU 侧按动态 Batch 4 并行吞吐 | ✅ 已定（v2.4） |
| ADR-015 | 模型放 `models/` 或 `checkpoints/`（不入库），按需加载并校验；启动探测运行时，缺失即降级并在日志明示 | ✅ 已定 |
| ADR-016 | **Qwen-Image-Layered 作为 DESIGN 线可选分割 Provider**，PLATE 线禁用 | ✅ 已定（v2.3） |
| ADR-017 | **PLATE 线补齐陷印（trapping）算子**：专色量化 + 变尺寸陷印，输出独立 `_Spot` 通道层 | ⬜ 待实现（v2.3） |
| ADR-018 | 前端分层控制采用 manifest 契约（`layers.json` + `text_manifest.json`），结构参照 Stratum | ⬜ 待实现（v2.3） |
| ADR-019 | **五维系统完整性与防欺骗代码审核体系**：环境真实探活、零硬编码静态扫描、物理交付物合规、防伪代码落地、真实基准量化，列入最高工程纪律 | ✅ 已落地（v2.4） |
| ADR-020 | **局部 ROI 裁剪与多核并发超分**：非全画幅图层（印章、题跋、芦雁等）提取紧凑 BBox 并加安全 Padding 局部引导滤波，多核 `ThreadPoolExecutor(max_workers=6)` 并发 | ✅ 已落地（v2.4） |

---

## 3. 实测基线（避坑数据，勿凭记忆改写）

| 项 | 实测值 | 采集时间 / 依据 |
| :--- | :--- | :--- |
| 源图 `inputs/source_4000.jpg` | 4000 × 1952 px | 2026-09-09 |
| 循环单元 `master_cleaned_full767.png` | 388 × 767 px | 2026-09-09 |
| 方案 A 裁切 | 388 × 718 px | 2026-09-09 |
| 成品画布（制版线） | 5315 × 9449 px / 4 通道 CMYK / **8 bit** | 2026-09-09 |
| **16K 终极母版 PSB（设计线）** | **$16000 \times 7808\text{ px}$ / 11 个独立图层 / 150.0 PPI** | **2026-09-10 实测（`Rosetsu_Master_16k.psb`）** |
| 16K 母版物理体积 | **2.34 GB（2,517,517,320 字节 / 2400.9 MB）** | `pipeline/06_verify_psb.py` |
| 16K 母版对应印刷尺寸 | **$2709.3 \times 1322.2\text{ mm}$**（2.71米 × 1.32米） | Photoshop `0x03ED` 严格校验 |
| 16K 母版合成保真度 | **MAE = 0.73**（远优于行业 ≤ 2.0 质检红线） | 像素级色差均方误差 |
| **全流程端到端总耗时** | **168.00 秒（2.80 分钟）**（初版 1468.7s，提速 **8.74x** 🚀） | `docs/BENCHMARK_REPORT.md` |
| Step 6 写盘耗时 | **27.66 秒**（纯 Python 原版 1141.65s，提速 **41.3x** 🚀） | C-SIMD PackBits 200 MB/s |
| 图层 BBox | 11 层全部具备动态计算的最小外接包围盒，无全屏黑底 | 零硬编码自动追踪 |
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
- **设计交付成品**：`outputs/Rosetsu_Master_16k.psb`（2.34 GB）已成功生成并通过全部物理与色彩断言。

---

## 6. 许可核对（M7 前必须完成）

- SAM 2：Apache-2.0（可商用）—— **需按实际使用版本复核**
- Grounding DINO：Apache-2.0（可商用）—— **需按实际使用版本复核**
- Qwen-Image-Layered：Apache-2.0（可商用）—— 权重 20B，**需按实际使用版本与权重来源复核**
- Stratum：**other（非标准开源许可，README 要求显式授权）** —— 未获授权前不得复制其任何代码
- FOGRA / Japan Color ICC：需自备并记录来源与许可（Pillow / psd-tools 不自带）
