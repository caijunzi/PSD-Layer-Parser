# HANDOFF — Ultra-Layer Studio（PSD 图层处理）

> 换账号/换机器时的**唯一入口文档**。新会话第一句：先读 `HANDOFF.md`。
> 深度细节在 `docs/`；项目日志在 `.workbuddy/memory/`。
> 最后更新：2026-09-13 18:30 by WorkBuddy AI

---

## 1. 项目是什么

把高精度古画扫描件（当前主力：长泽芦雪《金地山水六曲》约 1795，MET 公共领域，
原图 4000×1952）转成**可大幅面印刷的分层 PSD/PSB**，供设计师局部修改。

用户原始需求（2026-09-11 明确澄清）：
1. 可大幅面印刷的分层 PSD —— **已达成**
2. **尽可能多的、足够细致的对象级图层**（远山/水流/松树/建筑/人物/飞鸟/题款印章等
   一切可识别类别各自一层、边界干净），方便设计师局部修改 —— **约 70%，见 §4**
3. 尽可能大的印刷篇幅 —— **已达成**

## 2. 架构（一页看懂）

```
WebUI (React+Vite :5173)  ──proxy 127.0.0.1 必写死──▶  后端 (FastAPI :8099)
                                                          │
                                                          ├─ 上传 → data/uploads/
                                                          ├─ 提交任务 → task_manager（asyncio.create_subprocess_exec）
                                                          │     └─ 引擎子进程 run_universal_engine.py
                                                          ├─ WS 进度（逐行 stdout + 10s 心跳）
                                                          └─ 产物 → data/outputs/{task_id}/
引擎 6 步：语义分割 → 底板提取/深度排序 → 印前算子 → 遮挡补全 → 阶梯超分 → PSB 组装
双产品线：DESIGN(RGB/可编辑/允许生成) ｜ PLATE(CMYK/制版/禁生成内容)
```

- **SSOT**：preset（`presets/*.json`）是品类语义唯一真相源，provider 不内置任何品类语义
- **铁律**：R1 指标诚实 / R2 平场只服务检测分支 / RK-16 逐像素可复现 / PLATE 禁生成内容
- **单编译器内核**：`engine/core/psd_compiler.py`；`pytoshop` 为唯一依赖（不并存 psd-tools 写路径）

## 3. 关键决策（含踩坑，勿回退）

| 决策 | 原因 |
|---|---|
| vite proxy target 写 `127.0.0.1` 不用 localhost | Node 把 localhost 解析成 ::1，后端只听 IPv4 → 间歇 ECONNREFUSED |
| 后端/前端服务一律 `run_in_background` | Popen 游离进程几分钟被守卫清理；前台命令 ~2-3 分钟 SIGTERM |
| agent-browser 调用必须文件重定向 | subprocess 管道 + daemon 继承句柄 = EOF 死锁 |
| 金地参考取**全局 p88 分位数** | 局部平场会把大面积浓墨区内部"漂白"成 D≈0 |
| `imagecodecs` 必须装 | SIMD RLE 加速；缺失回退纯 Python packbits（3MB/s vs 356MB/s，差 120×） |
| 分割（DINO+SAM2）固定 CPU | 2026-09-13 实测：SAM2 单 `predict` GPU 反慢 6.88×、`set_image` 仅 1.30×，DINO（14 prompt）GPU 仅 1.18×，且破坏 RK-16 逐像素复现 → 维持 CPU。DINO 缺 `_C` 已加守卫（ms_deform_attn.py，CUDA 走纯 torch 回退，不再 NameError） |
| ICC 用 `profiles/CoatedFOGRA39.icc` | 系统库无 Fogra39L；**禁止 FOGRA27 冒名 39L**（README 明令勿混用），生产需换官方 ISOcoated_v2_300_eci |

## 4. 分层能力现状（核心指标）

| 类别 | 状态 | 来源通道 |
|---|---|---|
| 印章 / 题跋 / 人物 / 建筑 / 孤石 | ✅ 边界干净 | SAM |
| 雁群 | ✅ **2 个单实例层**（instance_split） | SAM 实例 |
| 峭壁（04A） | ✅ **密度精修救回**（bbox 74.4%→16.4%） | density_refined |
| 水波（03） | ✅ 密度带（区域+密度区间） | density_band |
| 渚上水木（05B） | ✅ prompt 迭代后命中 | SAM |
| 外框 / 折痕 / 金地底板 | ✅ | 装饰检测 + 背景提取 |
| **远山（04D）** | ❌ 源图对比度极低（肉眼勉强可辨），不可自动提取 | — |
| **寒林枯木（05A）** | ❌ 细线结构与金地纹理信噪比重叠 | 待 Frangi 骨架流 |
| 平渚（04B） | ❌ DINO 未命中（滩涂无边界） | — |

**被拒/未命中层的内容保留在底板，合成完整性零损失，仅缺独立可编辑图层**，
且全部写入 `manifest.totals.semantic_coverage`（produced 带 source / rejected 带原因 / missing）。

## 5. 可靠性机制（勿拆）

1. **弥散门**：内容层掩模外接框覆盖比 >50% 即拒绝（底板/外框/折痕豁免）——曾因缺失导致右半屏雾化污染
2. **密度精修通道**：弥散被拒类 → `mask ∩ D>floor + 形态学` → 再过弥散门 → 产出（04A 实证）
3. **密度带通道**：DINO 未命中类 → `region × 密度区间` 直接产层（03 实证）
4. **语义覆盖披露**：消灭"配置了却静默缺层"
5. **实例预算放宽**（×3）：单只雁不该按雁群总面积衡量
6. **金标准回归** `tests/test_golden_layers.py`：preset 改动必须过

## 6. 未决风险 / 下一步

1. **05A 寒林**：Frangi 骨架流（规则引擎有组件）产线状层，或 WebUI 半自动笔刷
2. **04D 远山**：需更高对比度扫描件或人工定位；本图不可自动
3. **04B 平渚**：DINO 未命中（滩涂无边界）→ 可试密度带通道
4. **跨图配置**：region 框仍是构图先验 → 已提供 `tools/calibrate_density_bands.py`
   （直方图推荐参数），配金标准回归防漂移
5. 前端 vite/后端 8099 为手动启停（用户会自行 kill，勿自动重启）
6. **自适应语义机制**（`docs/adaptive-semantics/`）：Stage 1（DB/指纹/材质判别/类目选择/episode 归档）
   + 命名空间桥接（DB 类目 → preset 别名）**已完成**；计划中 **Stage 2（图级 Auto-Tune：region/密度带
   建议 + 前端采纳卡片）/ Stage 3（反馈闭环+影子进化）/ Stage 4（CBR）/ Stage 5（类目树+主动学习）未做**。
7. **品类扩展**：封闭 5 类仅 1 类（金地屏风）端到端验证；壁布/烫印/水墨/油画 4 类改 preset 即可接入但**未验证**。
8. **未提交改动**：Stage 2 命名空间桥接（category_selector / migration_002 / seed_data_002 / tests / run_universal_engine）。
9. GPU 分割优化**已实测否决**（见 §3），勿重复投入；DINO `_C` 编译**用户决定放弃**。

## 7. 常用命令

```bash
# 服务
python -m uvicorn main:app --host 127.0.0.1 --port 8099     # webui/backend
npm run dev                                                  # webui/frontend → :5173
# 测试
python -m unittest discover -s tests -p 'test_*.py'          # 引擎 64
python -m unittest discover -s webui/backend/tests -p 'test_*.py'  # WebUI 28
# 实验
python scratch/quick_seg.py [preset]                         # 分割+掩模统计（~100s）
python tools/calibrate_density_bands.py inputs/source_4000.jpg
```

## 8. 出口约定

- 输出产物：`webui/data/outputs/{task_id}/`（PSB + manifest + masks）
- 不要提交：PSB/大文件（`webui/data/` 已 gitignore）、`scratch/` 中间产物
- 提交前必跑：两组全量测试 + 前端 `tsc --noEmit`
