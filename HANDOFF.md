# HANDOFF — Ultra-Layer Studio（PSD 图层处理）

> 换账号/换机器时的**唯一入口文档**。新会话第一句：先读 `HANDOFF.md`。
> 深度细节在 `docs/`；项目日志在 `.workbuddy/memory/`。
> 最后更新：2026-09-15 11:30 by WorkBuddy AI（P0–P2 修复批次；**请先读 §9**）

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

1. **05A 寒林**：✅ **已解决**——`区域先验 SAM` 通道直接出 2 层（无需 Frangi）
2. **04D 远山**：❌ **确认不可自动**（源图对比度不足，弥散门正确拒绝，外接框覆盖 74.5%）；维持"人工/换源"
3. **04B 平渚**：✅ **已解决**——`区域先验 SAM` 通道出层
   详见 `docs/未决提取难题结论_20260915.md`
4. **跨图配置**：region 框仍是构图先验 → 已提供 `tools/calibrate_density_bands.py`
   （直方图推荐参数），配金标准回归防漂移
5. 前端 vite/后端 8099 为手动启停（用户会自行 kill，勿自动重启）
6. **自适应语义机制**（`docs/adaptive-semantics/`）：**Stage 1–5.3 代码均已落地**，且自 2026-09-15 起
   **真正启用**（`japanese_screen_gold.json` 顶层 `mode=hybrid` + `auto_evolve=true`；此前因 preset 缺
   顶层 `mode` 恒 `locked`，Stage 1/2/4 引擎侧从不执行）。启用后 hybrid 会**限量补充**高亲和度 DB 类目
   （护栏：affinity≥0.6 且 ≤10 条），须重跑金标准与 RK-16 复核。
7. **品类扩展**：封闭 5 类仅 1 类（金地屏风）端到端验证；壁布/烫印/水墨/油画 4 类改 preset 即可接入但**未验证**。
8. **提交状态**：2026-09-15 批次改动见 §9；提交前必跑两组全量测试
   （当前 169 passed / 2 skipped + WebUI 33 OK）。`webui/data/adaptive_semantics.db.bak-20260915`
   为迁移修复前备份，确认无误后可删。
9. GPU 分割优化**已实测否决**（见 §3），勿重复投入；DINO `_C` 编译**用户决定放弃**。

## 9. 2026-09-15 P0–P2 修复批次（**最新状态，优先阅读**）

对全仓库逐文件审计后落地的修复，分 A/B/C 三组：

**A 组｜文档对齐**：本文件 §6 与 `README.md` §一/§速度览 已按代码事实更正。

**B 组｜缺陷修复（行为向设计意图收敛，测试全绿）**
| 项 | 文件 | 内容 |
|---|---|---|
| ④ CBR 格式统一 | `engine/adaptive/{cbr_retriever,episode_archiver}.py` | 归档写 JSONL 而检索找 `episodes/**/*.episode.json` → 现双通道；归档存 embedding + `auto_tune`；新增 `load_episode_by_id` |
| ⑤ 迁移自环 | `migrations/migration_003_build_tree.py` | `architecture` 二级/三级同名致 `parent_id=self` → 跳过自环 + 数据修复；改用 `rowcount` 杜绝"假成功"日志 |
| ⑥ 主动学习字段错配 | `engine/adaptive/active_learner.py` | 兼容 `dino_detections`（`layer_name`/`logits`/`boxes`），修复"全部检出被判不确定" |
| ⑦ 掩码落盘空壳 | `engine/schemas/manifest.py` | `save(mask_dir)` 真正写掩码 PNG + 回填 `recon_mask_path`（relpath）；新增 `attach_recon_mask` |
| ⑩ 先验继承 | `engine/adaptive/category_tree.py` | `inherit_priors` 由 `NotImplementedError` 改为优雅跳过 + 有数据时复制；`get_ancestors` 加防环 |
| ⑭ 学习权重 | `engine/adaptive/learner.py` | 权重 clamp 到 `[0.05, 5.0]`，跨 episode 累乘亦 clamp |
| ⑮ 回归维度 | `engine/adaptive/regression_tester.py` | 名义 8 维实判 3 维 → 实判 4 恶化型 + 层数 + 纯净性；消除除零 |
| ⑯ DB 版本 | `engine/adaptive/db_manager.py` | 新增 `create_version()`（此前版本链无创建入口） |
| ⑪ `--scale` 优先级 | `run_universal_engine.resolve_output_size` | 核实：**代码已修**（CLI > preset），仅文档滞后 |
| ⑫ 解释器路径 | `webui/backend/core/task_manager.py` | `PYTHON_BIN` 改为 env `ULS_PYTHON_BIN` > 当前解释器 > 兜底 |
| ⑱ IO 扩展名 | `engine/core/io_utils.py` | 未知扩展名不再回落 jpg，改默认 PNG（无损） |
| ③ 写盘内存 | `engine/codecs_accelerator.py` | 核实：**早已分块流式**（CHUNK_ROWS=256），无需再改 |

**C 组｜架构/行为变更（改变 PSB 输出，须复核）**
| 项 | 内容 |
|---|---|
| ① 自适应启用 | `japanese_screen_gold.json` 加顶层 `mode=hybrid`/`auto_evolve`；`merge_into_preset_format` 加"限量补充"护栏 |
| ⑧ R2 平场接线 | 新增 `_build_operator_ctx()`：`detect_image` 走 `LightingManager.normalize_illumination`，算子的检测分支取平场图 |
| ⑨ R3/金属接线 | `seam_harmonizer` 作可选前置步（preset `seam_harmonization.enabled`）；`metallic_foil` 作可选印前算子（preset `plate_operators.metallic_foil.enabled`）|
| ⑬ 后台学习接线 | `background_learner` 增守护线程消费队列；注册 `POST /api/adaptive/trigger-learning` 与 `GET /api/adaptive/learning-status` |
| ⑰ preset 校验 | 新增 `engine/schemas/preset_schema.py`，`load_preset` 非破坏校验（默认告警，`ULS_PRESET_STRICT=1` 升级为错误）|
| 审计回填 | `task_manager._backfill_episode_audit()`：审计完成后回填 episode 的 `audit_8d` 并重建 CBR 索引 |

**端到端实测（2026-09-15，本机 --mode both）**：PLATE 474.5s/1.80GB/43 层、DESIGN 399.9s/1.53GB/43 层。
①⑧⑨ 全部验证生效（`plate_purity_ok=true`、TAC 300%、有效源 37.5 PPI）。发现 hybrid 曾灌入 10 个泛类
（含与 preset 重复者），已加固为"解析后名称 + 英文词元 + 全 preset 语料"三重去重 + `max_supplement=3`。

**A 方案：材质分类器金地判据修复**（`fingerprint.py` / `material_classifier.py`）
- 根因：指纹的"背景"取自外框 10%，而扫描件外框是博物馆灰底（实测 L37/b0），画心才是金地（L79/b38）。
- 修复：指纹新增 `center_median_LAB` / `center_saturation`（去外框 15%，不改 embedding 维度）；
  **仅金地规则**改用中心区（以 b\* 黄度为主），其余三族保持原口径。
- 验证：source_4000.jpg 由 `油画布 0.55` → **`金地屏风 1.00`**；其它图判定不变；
  金地家族下 hybrid 仅补充 **3 个**（修复痕迹/梅花/竹）。
- 新增回归：`tests/test_material_classifier.py`（7 例）。引擎 **166 passed / 2 skipped**。

**追加项：宣纸水墨 / 绢本工笔 亦切中心区并单独标定**
- 区分点 = 笔触密度/纹理能量：宣纸水墨 cL>60、cb<20、edge<0.16、glcm<0.28（稀疏写意）；
  绢本工笔 cb 10–30、65<cL<88、edge≥0.16 或 glcm≥0.28（细腻密集）。
- 实测：写意水墨山水 → **宣纸水墨 0.65**（原误判绢本 0.70）；织物 → 绢本工笔 1.00；金地图仍 1.00。
- 油画布仍保留原外框口径（未标定）。⚠️ 无绢本真值样本，绢本阈值属保守估计。

**测试**：引擎 169 passed / 2 skipped；WebUI 33 OK；迁移 003 在真实库执行成功（残留自环 0）。

### 9.2 B/C 组收尾（2026-09-15 下午）

| 项 | 内容 | 验证 |
|---|---|---|
| **B1** | `textile_damask` 启用 **R3 接缝对齐**（`seam_harmonization`）+ **金属分色**（`plate_operators.metallic_foil`） | 实跑 damask：`[R3] pre_rmse=21.461 → post_rmse=15.588`（↓27%）；`metallic_foil: ok` |
| **B2** | `background_learner` 守护线程消费队列 + 注册 `POST /adaptive/trigger-learning`、`GET /adaptive/learning-status` | 新测试 `webui/backend/tests/test_adaptive_learning_endpoints.py`（3 例，真机 TestClient） |
| **B3** | 审计回填链路单测隔离 | 新测试 `webui/backend/tests/test_audit_backfill.py`（2 例；日志实证"索引重建 1 条"） |
| **B4** | `create_version` **接线到人审反馈**（每次改动落版本节点并激活） | 新测试 `tests/test_feedback_version.py`（3 例） |
| **B5** | 引擎新增 `ULS_AUDIT_AFTER_RUN=1`：出图后自动审计 → 回填 episode → 重建 CBR 索引；路径支持 `ADAPTIVE_EPISODE_PATH`/`ADAPTIVE_PENDING_PATH`/`ADAPTIVE_INDEX_PATH` 覆写 | 实跑 japanese：`[PostRun] 8 维审计门 exit=0` + `CBR 索引已按真实审计重建：9 条` |
| **C1** | 05A/04B/04A/03 均由既有通道解决；仅 04D 确认不可自动（无新算法） | `docs/未决提取难题结论_20260915.md` |
| **C2** | 品类端到端：金地 ✅（多次）、**壁布 ✅**（damask exit=0）、**水墨 ✅**（ink 172.8s/9 层） | 烫金=金地 preset 变体（已覆盖）；**油画缺样本**（待补图） |
| **C3** | 油画布族统一到**中心区**口径（阈值保守沿用，标注"未标定"） | 四族口径一致；4 张实测图判定无回归 |
| **C4** | 文档—代码矛盾勘误（M1–M19），**不改历史文档**、以追加新文档形式 | `docs/文档与代码对齐勘误_20260915.md` |

**已知未决**：油画（无样本）、绢本（无真值样本，阈值保守）、04D（数据条件）。

## 7. 常用命令

```bash
# 服务
python -m uvicorn main:app --host 127.0.0.1 --port 8099     # webui/backend
npm run dev                                                  # webui/frontend → :5173
# 测试
python -m unittest discover -s tests -p 'test_*.py'          # 引擎 169 passed / 2 skipped（py -m pytest tests/ -q 亦可）
python -m unittest discover -s webui/backend/tests -p 'test_*.py'  # WebUI 33 OK
# 实验
python scratch/quick_seg.py [preset]                         # 分割+掩模统计（~100s）
python tools/calibrate_density_bands.py inputs/source_4000.jpg
```

## 8. 出口约定

- 输出产物：`webui/data/outputs/{task_id}/`（PSB + manifest + masks）
- 不要提交：PSB/大文件（`webui/data/` 已 gitignore）、`scratch/` 中间产物
- 提交前必跑：两组全量测试 + 前端 `tsc --noEmit`
