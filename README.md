# Universal Layer Studio PRO

工业级 2D 智能图像分层与印前制版工作站。**单一真相来源（SSOT）**：`docs/UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md`（v2.4）。

| 文档 | 用途 |
| :--- | :--- |
| `docs/UNIVERSAL_LAYER_ENGINE_DEVELOPMENT_PLAN.md` | 顶层开发计划（目标/架构/契约/流水线/质检/里程碑） |
| `docs/BENCHMARK_REPORT.md` | **全量流水线效率与性能实测基准报告（秒级精确实测）** |
| `docs/MEMORY.md` | ADR 决策、六条铁律、实测基线、避坑 |
| `docs/ARCHITECTURE.md` | 架构与运行时视图 |
| `docs/DAILY_LOG.md` | 每日演化日志 |
| `docs/技术尽调与代码审查报告_20260910.md` | **第三方视角全量尽调：实测复核、完成度评估、P0/P1/P2 改进建议** |
| `docs/文档对齐清单_20260910.md` | 文档对齐记录 |
| `GEMINI.md` | AI 入口摘要与停止条件 |

---

## 一、这个系统做什么

面向**软装壁布、金属烫印面料、传统书画复刻、商业专色网印**的分层与制版。业务图范围为**封闭 5 类**：壁布 / 烫金 / 水墨 / 屏风 / 油画；超出范围转人工，不承诺任意图全自动（ADR-012）。

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

## 五、当前系统状态与实测基线

> ⚠️ **2026-09-10 复核更正**：下表数值为**实际打开产物文件复测**所得，取代此前文档中的
> 2.34 GB / MAE 0.73 / 15 图层 等互相矛盾的表述。完整核查见
> `docs/技术尽调与代码审查报告_20260910.md`。
>
> 当前 `outputs/Rosetsu_Master_16k.psb` 是**掩模泄漏修复前**生成的产物，
> 元素层存在严重缺陷（见下）；修复已并入代码，**需重跑生产后复核**。

| 维度 | 状态 / 实测指标 | 验证依据 |
| :--- | :--- | :--- |
| **产出文件** | `outputs/Rosetsu_Master_16k.psb`（**1.847 GB**，1,983,219,201 字节） | 磁盘实读（旧文档称 2.34 GB / 1.08 GB，均不符） |
| **画幅与分辨率** | **$16000 \times 7808$** @ **150.0 PPI**（物理尺寸 $2709.3 \times 1322.2\text{ mm}$） | Photoshop 资源块 `0x03ED` 严格通过 |
| **色彩模式** | **RGB（mode=3）** —— 双产品线中仅 DESIGN 线跑通，PLATE/CMYK 未落地 | `psd_tools` 实测 |
| **图层完整性** | **11 个独立图层**，但**6 层掩模存在泄漏或空层** | `tests/audit_system_integrity.py` 判定 FAIL |
| **元素掩模质量** | 🔴 印章层覆盖 **8.775%** 画幅（基线 0.0145%，**放大 605 倍**，IoU 0.002）；芦雁 100×、题跋 42×、人物 29×；外框层近乎空层 | 对照 `masks_16k` golden baseline |
| **图像保真度** | Section 5 合成与源图 **MAE = 3.498** —— **超出 ≤2.0 红线**（旧文档称 0.73） | `pipeline/06_verify_psb.py` |
| **全流程总耗时** | **168.00 秒（2.80 分钟）**（初版 1468.7s，提速 **8.74 倍**） | `docs/BENCHMARK_REPORT.md` |
| **硬件调度** | GPU.1 (RTX 5070) + GPU.0 (Arc 140T 护盾) + CPU 熔断回路 | OpenVINO 2026.3.1 实测 |
| **C 扩展写盘** | imagecodecs SIMD PackBits 实测 **207~358 MB/s**，为纯 Python 的 **43~100 倍** | 新版 D5 开关对比（旧版仅断言属性替换） |

### 修复进度（2026-09-10）

| 项 | 修复前 | 修复后（LR 掩模，待重跑产物复核） |
| :--- | ---: | ---: |
| 印章层 | 8.7745% | **0.0119%**（BBox 68×63，IoU 0.700） |
| 芦雁层 | 6.6332% | **0.1744%** |
| 题跋层 | 2.7458% | **0.2475%** |
| 人物层 | 6.6537% | **0.2683%**（基线 0.2346%） |
| 外框层 | 0.0762%（空层） | **24.94%** |
| 折痕层 IoU | 0.079 | **0.706** |

验证命令：

```bash
python tests/diagnose_mask_quality.py                 # 元素掩模质量诊断（对照 masks_16k 基线）
python tests/audit_system_integrity.py                # 五维审计（可回归版）
python pipeline/06_verify_psb.py --file outputs/Rosetsu_Master_16k.psb
```
