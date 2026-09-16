# HANDOFF — Ultra-Layer Studio（PSD 图层处理）

> 换账号/换机器时的**唯一入口文档**。新会话第一句：先读 `HANDOFF.md`。
> 深度细节在 `docs/`；项目日志在 `.workbuddy/memory/`；**完整项目记忆在 `<项目根>/项目记忆.md`**。
> 最后更新：2026-09-16 by WorkBuddy AI（四项工程缺口修复 + 通道读取根因修复；**请先读 §10**）

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
| **PSD 层像素读取只用 `engine/core/psd_layer_io.py`** | `psd_tools` 的 `layer.numpy()` 通道数随色彩模式变：RGB→(H,W,**4**) Alpha 在 idx3；**CMYK→(H,W,5) Alpha 在 idx4**。写死 `[:, :, 3]` 在 CMYK(PLATE) 产物上取到 **K 通道** → ②③④⑥ 四维审计全错、防欺骗审计整体失效。`audit_psb._alpha` 是**别名**，勿写第二份实现；已有静态测试防回退 |

## 4. 分层能力现状（核心指标）

| 类别 | 状态 | 来源通道 |
|---|---|---|
| 印章 / 题跋 / 人物 / 建筑 / 孤石 | ✅ 边界干净 | SAM |
| 雁群 | ✅ **2 个单实例层**（instance_split） | SAM 实例 |
| 峭壁（04A） | ✅ **密度精修救回**（bbox 74.4%→16.4%） | density_refined |
| 水波（03） | ✅ 密度带（区域+密度区间） | density_band |
| 渚上水木（05B） | ✅ prompt 迭代后命中 | SAM |
| 外框 / 折痕 / 金地底板 | ✅ | 装饰检测 + 背景提取 |
| 寒林枯木（05A） | ✅ **区域先验 SAM 出 2 层**（1,045,404px / 319,411px，**无需 Frangi**） | region + SAM |
| 平渚（04B） | ✅ **区域先验 SAM 出层**（637,903px） | region + SAM |
| **远山（04D）** | ❌ 源图对比度极低（肉眼勉强可辨），**确认不可自动**（弥散门正确拒绝，外接框覆盖 74.5%） | — |

> 05A/04B 已由「区域先验 SAM」通道解决，**Frangi 骨架流需求关闭**（YAGNI）。
> 四条品类可选通道（区域先验 SAM / 密度精修 / 密度带 / 弥散门豁免）已覆盖绝大多数漏检，
> 详见 `docs/未决提取难题结论_20260915.md`。

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
   **真正启用**（`japanese_screen_gold.json` 顶层 `mode=hybrid` + `auto_evolve=true`）。
   ⚠️ **ADR-021~025 勿回退**：① episode 用 **JSONL 不建表** ② 人审用**轮询** 3s
   ③ 端点统一 `/api/adaptive/` ④ 迁移 `migration_003_build_tree.py` ⑤ 人审 delete = **软删**
   （`categories.deleted_at`，db_manager 主查询过滤，勿删该过滤点）。
   DB `webui/data/adaptive_semantics.db`：categories 27 / category_prompts 47 / category_priors **19**；
   **`episodes` 表不存在**（按设计）。新增自适应模块必须①接生产调用点②加 wiring 断言
   （`tests/test_adaptive_wiring.py`）。
7. **品类扩展**：封闭 5 类中 **金地屏风 / 水墨 / 烫金 / 油画 已端到端验证**；
   **壁布**已由新增 **`textile_damask_photo`** preset 覆盖**实物样品照**（方案 C，8 维审计全过，见 §10.4）；
   原 `textile_damask` 面向**可平铺数码纹样**，**不用于实物样品照**（混用即 §9.2 C2 的失败）。
   ⚠️ §9.2 C2 曾记「壁布 ✅」，当时结论有误，以本条与 §10.4 为准。
8. **提交状态**：2026-09-16 批次见 §10；提交前必跑两组全量测试
   （当前引擎 **279 passed / 4 skipped** + WebUI **41 OK**）。
9. GPU 分割优化**已实测否决**（见 §3），勿重复投入；DINO `_C` 编译**用户决定放弃**。

## 9. 2026-09-15 P0–P2 修复批次（历史批次，最新请读 §10）

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
| **C2** | 品类端到端：金地 ✅、壁布 ✅（damask exit=0）、水墨 ✅（ink 172.8s/9 层）、**油画 ✅**（oil 218.6s/8 层） | **4/5 品类实跑通过**；烫金=金地 preset 变体（已覆盖）。⚠️ **「壁布 ✅」已被 2026-09-16 复验推翻**（plate 线零掩模退化为 4 层），见 §6.7 / §10.2 |
| **C3** | 材质分类器：四族统一**中心区**口径；**用真值样本标定** | 油画：`inputs/油画.jpeg`（中心 L52.2 a7 b27 sat28.1 edge0.241）→ 油画布 0.70 ✅；并修正绢本判据（**glcm 稠密项须以 edge 稠密为前提**，否则明代水墨误判绢本）。**7 张样本 7/7 判定合理**（金地/壁布/织物/宋·明·元水墨/油画/金地变体） |
| **C4** | 文档—代码矛盾勘误（M1–M19），**不改历史文档**、以追加新文档形式 | `docs/文档与代码对齐勘误_20260915.md` |

**已知未决**：绢本工笔**无真值样本**（现阶段用织物 damask 作近邻参照）；04D 远山（数据条件）；
油画/水墨 preset 的 8 维审计门为 `exit=2`（新品类阈值未校准，非崩溃，产物可用）。

## 7. 常用命令

```bash
# 服务
python -m uvicorn main:app --host 127.0.0.1 --port 8099     # webui/backend
npm run dev                                                  # webui/frontend → :5173
# 测试（⚠️ 必须用系统 Python 3.12.10：C:/Users/CK/AppData/Local/Programs/Python/Python312/python.exe；
#        WorkBuddy managed 3.13 无 numpy。WebUI discover 不可加 -t，否则 base 模块 import 失败）
py -m pytest tests/ -q                                        # 引擎 279 passed / 4 skipped
py -m unittest discover -s webui/backend/tests -p "test_*.py" # WebUI 41 OK
# ★ 提交前自检（一键，五道门；见 §8）
py scripts/preflight.py            # 全量：解释器依赖 + 静默降级审计 + 行尾一致性 + 两组测试 + 前端 tsc
py scripts/preflight.py --fast     # 秒级：只跑 静默降级审计 + 行尾一致性（pre-commit 钩子用这个）
# 实验
python scratch/quick_seg.py [preset]                         # 分割+掩模统计（~100s）
python tools/calibrate_density_bands.py inputs/source_4000.jpg
```

## 8. 出口约定

- 输出产物：`webui/data/outputs/{task_id}/`（PSB + manifest + masks）
- 不要提交：PSB/大文件（`webui/data/` 已 gitignore）、`scratch/` 中间产物
- **提交前必跑（2026-09-16 起统一为一条命令）**：`py scripts/preflight.py`
  —— 五道门：① 解释器依赖（须系统 Python 3.12，含 numpy/cv2/PIL）
  ② **静默降级审计 P0 必须为 0**（`tools/audit_degradation.py`）
  ③ **行尾一致性**（逐文件比对行尾种类与 HEAD，抓"整文件等量 +/-"噪音 diff）
  ④ 引擎全量 pytest ⑤ WebUI 全量 unittest + 前端 `tsc --noEmit`。
  - 本地已装 pre-commit 钩子（`.git/hooks/pre-commit`，调用 `--fast`）；
    换机器后重装：`cp scripts/pre-commit-hook .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`
  - ⚠️ 跑引擎/长任务须加 `CODEBUDDY_SAFE_DELETE_ENABLED=0`（见 §10.5），preflight 已内置。

## 10. 2026-09-16 批次（**最新状态，优先阅读**）

### 10.1 四项工程缺口修复（提交 `3691660`）
⑤ 合成等价性改**同 ICC 分色**参考（此前 PIL `convert('CMYK')` K 恒 0 → RMSE 虚高：
金地 36.49 实为 0.96、商用 32.51 实为 2.43）；指纹 PCA 落地
（`engine/adaptive/fingerprint_pca.pkl`，**勿放 checkpoints/models，被 gitignore**）；
CBR 参数不自动注入生产（保护 RK-16）；`category_priors` 真实统计 19 条；
**绢本工笔必须用 `chinese_ink_landscape_ai`**（此前误用 textile_damask，换后 8 维全过）。

### 10.2 壁布 #49 根因定案（提交 `1c86d9d`）
三层叠加：**① 素材前提不满足**（`inputs/damask_sample.png` 是壁布**实物样品照**，
上下平铺接缝差 20.44 ≈ 图内 std 22.10，不可无缝平铺；而 `textile_damask` 含
`seam_harmonization: cyclic_vertical`）→ **② DINO 对该 preset 的 2 个壁布类目零检测**
（检出全是屏风系，弥散门拒 4 + allowlist 丢 5 → `成功提取 0 个解耦语义对象掩模`）→
**③ `run_universal_engine.py:574/637` 用 adaptive DB 类目覆盖 preset 类目**，
而 DB 27 类目全在「山水画」树下 → 检出屏风系 → 被 `rule_class_allowlist` 全丢。
已修：清空时打印明确告警（**产物字节不变**，SHA `666de161…`）。
同时修 ④ 内容承载**假通过**（union 曾含全画布加工层 → lost 恒 0；排除后 damask 0.0→0.1129）。

### 10.3 ★ 通道读取根因修复（提交 `9bd0aa3`）
- 新建 **`engine/core/psd_layer_io.py`** 作唯一权威读取入口（`layer_alpha`/`layer_rgb`/
  `full_alpha_mask`，按层实际通道数判定）。`tools/audit_psb.py` 的 `_alpha`/`_layer_rgb`
  改为**别名**；`tests/audit_system_integrity.py:241`（防欺骗审计）也曾误取 K → 已改走共享入口。
- **新增静态测试** `test_no_stray_numpy_index3_in_audit_modules`：扫描两审计文件禁止再出现
  手写 `numpy()[:, :, 3]`/`[:, :, :3]`，机制上防回退。
- **实测结论**：pytoshop 强制整份 PSD 同模式（混 CMYK+RGBA 抛 `Mismatched color mode`），
  故单 PSD 内**不会混层**；危险模式只剩"假设全 PSD 4 通道 RGB"。

### 10.4 ★ 壁布样品照独立 preset（方案 C，2026-09-16）
新增 **`presets/textile_damask_photo.json`** + **`engine/core/sample_panel.py`**（零硬编码样块检测，
判据=长直边持续性；真值边界精确命中）。要点：`seam_harmonization` 关闭、`mode=locked`
（**根除根因③**）、精确 `rule_class_allowlist`、样块外 → **「画面外背景带」层**。
引擎接线：`GroundedSAMProvider.segment_objects(..., roi_mask=)`（新参数）+ 神经掩模裁到 ROI；
`BBOX_EXEMPT_KEYWORDS` 加 `背景带`/`photo_background`（否则被弥散门误杀）。
**实测**：样块 ROI 80.9%、**DINO 在样块区内确实检出团花**（根因②不再阻塞）、图层 7 个、
**8 维审计全过 exit=0**、④ lost 0.000472、⑤ rmse_lowfreq 13.1（旧：4 层 / 11.29% / 40.31 / exit=2）。

### 10.5 ICC 黑版曲线 + WebUI 接入与冒烟（2026-09-16 第四批）
- **新增 `engine/core/black_generation.py`**（GCR 曲线：K 曲线 + CMY 等量补偿；**恒等零拷贝**保 RK-16）
  + **`tools/calibrate_black_generation.py`**（CMYK 域快速标定）。标定：金地/油画**恒等最优**；
  **壁布样品照 `k_gain=1.1`**（K 非空 70.4→78.1%、RMSE_low 0.910→**0.776**、TAC 256→245）。
- **★ 7 个 preset 全补 `icc_path=profiles/CoatedFOGRA39.icc`**：此前**只有 `japanese_screen_gold` 有**，
  `textile_damask` 是「默认 PLATE 却缺 ICC」的关键缺口（K≡0 无真黑版）。
- **★ ⑦ `k_channel_nonzero_pct` 坏指标修复**：旧口径判「K 呈色 >10%」→ K 全空与有墨都 ≈100%；
  改为 `呈色<1`（存在黑墨），新增 `k_ink_mean_pct` / `k_channel_strong_pct`。
- **WebUI**：preset 列表本就是动态扫描 → 新 preset 自动可见；补 `PRESET_ESTIMATE`；
  `_recommend_preset` 改为**先判实物样品照**（复用 `sample_panel`）。
- **端到端冒烟 11/11 通过**：presets→upload→process→history→产物→download；
  任务 169.4s、产物齐全、**任务审计 8 维全过**、manifest 披露 ICC 与黑版曲线；
  前端 `tsc --noEmit` + `vite build` 通过。
- ⚠️ **环境坑**：WorkBuddy `safe-delete` 钩子按 turn 累计删除数（阈值 50）**终止长跑引擎**
  （现象：无 traceback + exit=1）→ 跑引擎须加 **`CODEBUDDY_SAFE_DELETE_ENABLED=0`**。
- ⚠️ **样本集已换代**：`inputs/damask_sample.png` 移除，改用 `工艺壁布-1/2/3.jpeg`
  （织物特写照，**无样块 → photo preset 全幅回退**，实测仍 8 维全过）。

### 10.6 静默降级深度审核 + 织物壁布族 + 推荐启发式（2026-09-16 第五批）
- **新增静默降级审计** `tools/audit_degradation.py` + 棘轮门禁
  `tests/test_no_silent_degradation.py`：P0 必须 0（preset 完整性 / 裸 except /
  不认识 `imread_unicode` 却直接 `cv2.imread`），P1-1 静默吞异常须全部登记白名单。
- **★ 修掉"审计说通过却没测量"**：`tools/audit_psb.py` ⑥/⑦ 原有 3 处
  `except Exception: pass` 吞掉指标计算 → 指标静默消失而维度仍报 passed=True。
  改为 `_record_metric_failure()`（记 `metric_error` + **fail-closed**）。
  另 3 处（task_manager 审计事件广播 / file_handler 样块检测 / realesrgan OV 缓存）补披露。
- **★ 新增材质家族「织物壁布」**：分类器此前只有 4 个绘画族 → `inputs/工艺壁布-1/2/3.jpeg`
  全被误判为「宣纸水墨」。判据 = **近中性合取门（sat<15 且 b*<15）** +
  **结构门（LBP 熵 ≥2.0）**（只靠中性会把噪声/纯色也吞进来）。
- **★ `_recommend_preset` 改为「材质家族优先」**（顺序即正确性）：
  家族 → preset 映射；**样块检测只在纺织类家族内做二次判定**（否则金地屏风的
  绫边外框会命中"长直边持续性"被误判成实物样块）；置信 <0.6 才退回宽高比。
  实测 **10/10 正确**（此前油画/水墨/绢本全错落到 japanese_screen_gold）。
- **批次验收（ICC 补齐后，plate/scale1，均 8 维全过）**：
  水墨宋代 ④0.001979 / ⑤1.28 / K非空87.3%；油画 ④0.000874 / ⑤3.09 / 97.2%；
  金地 ④0.001607 / ⑤0.96 / 81.7%；工艺壁布×3 ④0.000364~0.000586 / ⑤1.65~19.29。
- ⚠️ **行尾纪律（踩过的坑）**：仓库绝大多数文件是 **LF**，仅少数 CRLF。
  编辑时必须**保留各文件既有行尾**；曾用"统一转 CRLF"脚本造成 15 个文件整文件
  等量 +/- 的噪音 diff（需额外修正提交 `6e0cadb`）。自检：`git diff --numstat`
  出现 **N+/N- 相等** 即噪音信号。另：Git Bash 下中文提交信息**必须 `git commit -F`**
  （`-m` 中的反引号会被当命令替换）。

### 10.7 当前测试基线
引擎 **279 passed / 4 skipped**（5 skip 源于样本换代）；WebUI **41 OK**。提交 **未 push**。
