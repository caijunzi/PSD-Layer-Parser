"""Universal Multi-layer PSB Engine - CLI Entry Point.
Takes ANY input image and automatically produces a layered, print-ready PSB file.

Usage:
    python run_universal_engine.py --input inputs/source_4000.jpg --output outputs/universal_output.psb --preset traditional_chinese_ink
"""
import os
import re
import sys
import json
import time
import argparse
from typing import Any
import cv2
import numpy as np

# Preset 加载唯一入口（SSOT）：provider 与入口脚本共用，避免引擎反向依赖入口脚本
from engine.core.io_utils import imread_unicode
from engine.schemas.presets import (
    load_preset as _load_preset_file,
    match_layer_attributes,
)


def load_preset(preset_arg):
    """兼容保留：实现已收敛到 engine/schemas/presets.py（SSOT）。"""
    return _load_preset_file(preset_arg)

from engine.background_extractor import UniversalBackgroundExtractor
from engine.semantic_segmenter import UniversalSemanticSegmenter
from engine.depth_layer_sorter import UniversalLayerSorter
from engine.deocclusion import UniversalDeoccluder
from engine.tiled_super_res import UniversalTiledSuperRes
from engine.psb_builder import UniversalPSBBuilder
from concurrent.futures import ThreadPoolExecutor

# 本次运行归档的 episode id（B5：供 --mode 循环后做审计回填 + CBR 索引重建）
LAST_EPISODE_ID: Any = None

def _build_operator_ctx(src_lr, preset=None):
    """构造算子用 ProcessingContext，落实铁律 R2（检测/输出双分支）。

    - detect_image：经照度平场（LightingManager），**只服务检测**（轮廓/微孔/接缝）；
    - output_image：原始未平场像素（供输出/比色）。
    此前 lighting.py 零调用 → R2 的"平场检测分支"在代码中是空承诺，现接线。
    """
    from engine.core.models import ProcessingContext

    ctx = ProcessingContext(source_bgr=src_lr)
    ctx.output_image = src_lr
    try:
        from engine.core.lighting import LightingManager

        flat = LightingManager.normalize_illumination(
            src_lr, target_mean=float((preset or {}).get("flatfield_target_mean", 230.0))
        )
        ctx.detect_image = flat if flat is not None else src_lr
    except Exception as e:
        print(f"  -> [R2] 照度平场不可用，detect 分支回退原图: {e}")
        ctx.detect_image = src_lr
    return ctx


def _run_plate_operators(src_lr, sorted_layers, preset, ppi):
    """PLATE 线印前算子链：轮廓保全 / 微孔刀模 / 专色陷印 / 金属分色。

    对应 ADR-008（裁切线运行时化）/ ADR-017（陷印）/ 铁律 R4、R5。
    算子均为**品类可选**：由 preset 的 plate_operators 开关控制，
    未启用的算子如实记录 skipped，不伪造结果。

    返回 (operators_summary, extra_layers)：
      operators_summary: 写入 manifest 的逐算子结果
      extra_layers: 需并入分层流程的新图层（刀模层 / 专色层 / 金属分色层）
    """
    from engine.operators.contour_protection import ContourProtectionOperator
    from engine.operators.micro_holes import MicroHolesOperator
    from engine.operators.trapping import TrappingOperator
    from engine.operators.metallic_foil import MetallicFoilOperator

    # R2：算子的检测分支取 detect_image（经照度平场），不被写回输出像素
    op_ctx = _build_operator_ctx(src_lr, preset)

    cfg = preset.get("plate_operators", {}) or {}
    summary: dict[str, Any] = {}
    extra_layers: list[dict] = []

    # 1) 轮廓保全：运行时计算安全裁切下限（ADR-008，替代历史硬编码 Y=718）
    c_cfg = cfg.get("contour_protection", {})
    if c_cfg.get("enabled", True):
        res = ContourProtectionOperator().run(op_ctx, params=c_cfg.get("params"))
        if res.success:
            summary["contour_protection"] = {
                "status": "ok",
                "safe_bottom_y_src": int(res.data.get("safe_bottom_y", -1)),
                **{k: v for k, v in res.metrics.items()},
            }
        else:
            summary["contour_protection"] = {"status": "skipped", "reason": res.message}
    else:
        summary["contour_protection"] = {"status": "disabled"}

    # 2) 微孔刀模（铁律 R5：仅在有效基材掩模内检测）
    m_cfg = cfg.get("micro_holes", {})
    if m_cfg.get("enabled", False):
        res = MicroHolesOperator().run(op_ctx, params={
            **(m_cfg.get("params") or {}),
            "target_size": (src_lr.shape[1], src_lr.shape[0]),
        })
        if res.success:
            # ⚠️ numpy 数组禁用 `or`（truth value ambiguous），必须显式判 None
            diecut = res.data.get("diecut_mask")
            if diecut is None:
                diecut = res.data.get("mask")
            n_holes = int(res.metrics.get("hole_count", 0) or 0)
            if diecut is not None and n_holes > 0:
                extra_layers.append({
                    "name": m_cfg.get("layer_name", "12_DieCut_Mask"),
                    "mask": diecut,
                    "blend_mode": "NORMAL",
                    "opacity": 255,
                    "z_index": 105.0,
                })
                summary["micro_holes"] = {"status": "ok", "layer_added": True,
                                          **{k: v for k, v in res.metrics.items()}}
            else:
                summary["micro_holes"] = {"status": "ok", "layer_added": False,
                                          "reason": "未检出微孔", **res.metrics}
        else:
            summary["micro_holes"] = {"status": "skipped", "reason": res.message}
    else:
        summary["micro_holes"] = {"status": "disabled"}

    # 3) 专色陷印（ADR-017）：需显式指定专色图层来源，避免金底板被误判为专色
    t_cfg = cfg.get("trapping", {})
    spot_layer_name = t_cfg.get("spot_layer_name")
    if t_cfg.get("enabled", False) and spot_layer_name:
        spot = next((l["mask"] for l in sorted_layers
                     if l["name"] == spot_layer_name and l.get("mask") is not None), None)
        if spot is not None:
            res = TrappingOperator().run(None, params={
                "spot_mask": spot,
                "ppi": ppi,
                **(t_cfg.get("params") or {}),
            })
            if res.success:
                extra_layers.append({
                    "name": t_cfg.get("layer_name", "11_FoilTrap_Spot"),
                    "mask": res.data["spot_channel"],
                    "blend_mode": "NORMAL",
                    "opacity": 255,
                    "z_index": 104.0,
                })
                summary["trapping"] = {"status": "ok", "layer_added": True,
                                       **{k: v for k, v in res.metrics.items()}}
            else:
                summary["trapping"] = {"status": "skipped", "reason": res.message}
        else:
            summary["trapping"] = {"status": "skipped",
                                   "reason": f"配置的专色图层不存在: {spot_layer_name}"}
    else:
        summary["trapping"] = {
            "status": "disabled" if not t_cfg.get("enabled", False) else "skipped",
            **({"reason": "未配置 spot_layer_name"} if t_cfg.get("enabled", False) else {}),
        }

    # 4) 金属分色（金属箔/烫金工艺）：把铜箔与金地分成两张专色掩模层
    #    品类可选：需显式提供 valid_print_mask 语义（否则在有效印刷区外误判）
    foil_cfg = cfg.get("metallic_foil", {})
    if foil_cfg.get("enabled", False):
        valid_mask = None
        vm_name = foil_cfg.get("valid_mask_layer")
        if vm_name:
            valid_mask = next((l["mask"] for l in sorted_layers
                               if l["name"] == vm_name and l.get("mask") is not None), None)
        if valid_mask is None:
            # 回退：用内容层掩模并集作为有效印刷区
            valid_mask = np.zeros(src_lr.shape[:2], dtype=np.uint8)
            for l in sorted_layers:
                if l.get("mask") is not None:
                    valid_mask = cv2.bitwise_or(valid_mask, l["mask"])
        res = MetallicFoilOperator().run(op_ctx, params={
            "valid_print_mask": valid_mask,
            **(foil_cfg.get("params") or {}),
        })
        if res.success:
            for key, ly_name, z in (
                ("copper_mask", foil_cfg.get("copper_layer_name", "13A_玫瑰铜箔专色_Foil_Copper"), 106.0),
                ("gold_mask", foil_cfg.get("gold_layer_name", "13B_复古金专色_Foil_Gold"), 106.5),
            ):
                m = res.data.get(key)
                if m is not None and np.count_nonzero(m) > 0:
                    extra_layers.append({
                        "name": ly_name, "mask": m,
                        "blend_mode": "NORMAL", "opacity": 255, "z_index": z,
                    })
            summary["metallic_foil"] = {"status": "ok",
                                        **{k: v for k, v in res.metrics.items()}}
        else:
            summary["metallic_foil"] = {"status": "skipped", "reason": res.message}
    else:
        summary["metallic_foil"] = {"status": "disabled"}

    return summary, extra_layers


def _audit_strict() -> bool:
    """严格审计环境：ULS_AUDIT_STRICT=1/true/yes 时，审计失败需传播为非零退出。"""
    return str(os.environ.get("ULS_AUDIT_STRICT", "")).lower() in ("1", "true", "yes")


def _describe_admit_reason(reason: str) -> str:
    """把 finalize_episode 的拒绝原因码翻译成可读中文说明。"""
    if reason == "no_full_report":
        return "审计报告缺少完整 8 维 'dims'（旧式/不完整报告，已拒绝收口）"
    if reason == "no_episode_log":
        return "episode 日志不存在"
    if reason == "episode_not_found":
        return "未找到对应 episode 记录"
    if reason and reason.startswith("missing_dim:"):
        return f"缺必需审计维度：{reason.split(':',1)[1]}"
    if reason and reason.startswith("not_evaluated:"):
        return f"必需维度未真正核验：{reason.split(':',1)[1]}"
    if reason and reason.startswith("failed:"):
        return f"必需维度未通过：{reason.split(':',1)[1]}"
    return reason or "unknown"


def _post_run_audit_and_sync(output_path: str, source_path: str, output_mode: str = "design") -> int:
    """出图后自动跑 8 维审计 + finalize_episode 收口 + 重建 CBR 索引（B5：CLI 冷启动闭环）。

    由环境变量 `ULS_AUDIT_AFTER_RUN=1` 触发。此前只有 WebUI 任务流会审计并回填，
    CLI 直跑时 episode 的 audit_passed 恒为 False → CBR 永远检索不中。

    **完整审计报告收口（2026-09-15 修复「旧报告被误当通过」）**：
    - 必须用**完整** 8 维审计报告（含 ``dims`` + ``passed``）调用 ``finalize_episode``；
      旧式 4 字段摘要（无 ``dims``）被视为不完整、**拒绝收口**并打印详细原因。
    - ``finalize_episode`` 的 ``detections`` 参数实为其审计报告的同义别名，本函数只传审计报告，
      不把检测记录误塞进审计语义。
    - 详细失败原因始终打印（准入/未准入/不完整报告）。
    - 严格测试环境 ``ULS_AUDIT_STRICT=1``：审计失败 / 报告不完整 / 未准入 → 返回非零，
      由 main 传播为进程退出码；正常（非严格）保留可配置行为（仅日志、不阻断产物）。

    Returns:
        0 = 收口成功（或无可收口）；非 0 = 审计失败（供严格环境传播非零退出）
    """
    stem, _ = os.path.splitext(output_path)
    manifest = f"{stem}.manifest.json"
    audit_json = f"{stem}.audit.json"
    try:
        import subprocess
        import sys as _sys

        cmd = [
            _sys.executable, os.path.join("tools", "audit_psb.py"), output_path,
            "--manifest", manifest, "--source", source_path, "--json", audit_json,
        ]
        if os.path.isfile(audit_json):
            os.replace(audit_json, audit_json + ".previous")
        r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode not in (0, 2):
            print(r.stderr[-4000:])
            return r.returncode
        print(f"[PostRun] 8 维审计门 exit={r.returncode}（0=全过 / 2=有不通过维度）")

        if not os.path.isfile(audit_json):
            print("[PostRun] 未生成 audit.json，审计失败：" + r.stderr[-2000:])
            return 2
        if not LAST_EPISODE_ID:
            print("[PostRun] 本次无 episode（自适应未启用），跳过 CBR 回填")
            return r.returncode

        # 读取完整 8 维审计报告（原始 JSON，含 dims + passed）
        import json as _json
        with open(audit_json, "r", encoding="utf-8") as _f:
            full_report = _json.load(_f)

        # 拒旧报告：缺少完整 8 维 dims → 视为不完整，拒绝收口并详细打印
        if not isinstance(full_report, dict) or "dims" not in full_report:
            _reason = "审计报告缺少完整 8 维 'dims' 字段（旧式/不完整报告，已拒绝收口）"
            print(f"[PostRun] ❌ {_reason}")
            print(f"[PostRun]    audit.json 顶层键：{list(full_report.keys()) if isinstance(full_report, dict) else type(full_report)}")
            return 2 if _audit_strict() else 0

        from engine.adaptive.episode_archiver import finalize_episode, sync_index_from_log

        # 完整报告收口：保留原始 8 维（audit_full），并按严格准入刷新 CBR 索引
        decision = finalize_episode(
            LAST_EPISODE_ID,
            audit_report=full_report,
            output_mode=output_mode,
            artifact=output_path,
        )
        if decision["admitted"]:
            print(f"[PostRun] ✅ episode 终态：审计准入通过（audit_passed），已写入 CBR 索引。")
            return 0

        # 未准入：打印详细原因；严格环境传播非零
        _detailed = _describe_admit_reason(decision.get("reason"))
        print(f"[PostRun] ⚠️ episode 终态：未准入（reason={decision.get('reason')}）。{_detailed}")
        print(f"[PostRun]   完整审计报告 passed={full_report.get('passed')}；"
              f"该 episode 不进入 CBR 索引（避免污染冷启动复用）。")
        return 2 if _audit_strict() else 0
    except Exception as e:  # 旁路异常：默认不阻断产物
        print(f"[PostRun] 审计/回填失败（不影响产物）: {e}")
        return 2 if _audit_strict() else 0



def _apply_cbr_auto_tune(preset: dict, cbr_auto_tune: dict) -> bool:
    """把 CBR 命中的已验证调优参数真正应用到当前 preset（Stage 4 收口）。

    仅注入**结构性**参数（regions / density_bands），与 preset 既有的
    region / density_band_classes 字段同构；仅当候选确有这些参数时才应用并返回 True。
    不臆造、不覆盖无关字段，避免跨类型配置污染。

    Args:
        preset: 当前使用的 preset dict（原地修改）
        cbr_auto_tune: 命中历史 episode 的 auto_tune（含 regions / density_bands 等）

    Returns:
        是否成功应用了至少一类结构性参数
    """
    if not isinstance(cbr_auto_tune, dict) or not isinstance(preset, dict):
        return False
    applied = False
    regions = cbr_auto_tune.get("regions")
    density_bands = cbr_auto_tune.get("density_bands")
    if isinstance(regions, dict):
        for cls in preset.get("ai_semantic_classes", []):
            saved = regions.get(cls.get("name") or cls.get("layer_name"))
            if isinstance(saved, dict):
                for key in ("region", "regions"):
                    if key in saved:
                        cls[key] = saved[key]
                        applied = True
    if density_bands:  # 密度带（非空）
        preset["density_band_classes"] = density_bands
        applied = True
        print(f"[CBR] 已应用 density_bands（{len(density_bands)} 条）")
    return applied


def _seed_everything(seed: int) -> None:
    """固定随机源（RK-16：产物可复现是 PLATE 线「可复算」验收的前提）。

    覆盖 random / numpy / torch（可选依赖，缺失时跳过）。
    注意：不开 torch.use_deterministic_algorithms —— 部分算子无确定性实现会抛错，
    且会显著拖慢 CPU 推理；先以种子固定为主，算子级确定性按需另行开启。
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_output_size(w_lr: int, h_lr: int, target_scale, target_w, target_h, preset: dict):
    """输出尺寸决策。优先级：显式 target_w/h > CLI --scale > preset.super_res_scale > 4.0。

    历史缺陷（全量回归时暴露）：原实现 `preset.get("super_res_scale", target_scale)`
    在 preset 提供该键时无条件覆盖 CLI —— CLI 形同虚设。
    """
    if target_w is not None and target_h is not None:
        return int(target_w), int(target_h)
    scale = target_scale
    if scale is None:
        scale = preset.get("super_res_scale", 4.0)
    return int(w_lr * scale), int(h_lr * scale)


def run_pipeline(input_path, output_path, preset_name="japanese_screen_gold", target_scale=None, target_w=None, target_h=None, dpi=None, device=None, profile="robust_performance", output_mode="design", icc_override=None, seed=42, adaptive_mode_override=None):
    global LAST_EPISODE_ID
    # 每次 run_pipeline 重置 episode 句柄：避免多产品线（both 模式）或多次调用间串味，
    # 保证「本次 run」对应唯一 episode（2026-09-15 修复：此前 LAST_EPISODE_ID 跨 run 残留）。
    LAST_EPISODE_ID = None
    t_start = time.time()
    from engine.schemas.profile_config import resolve_profile
    from engine.schemas.manifest import (
        GENERATIVE_UPSCALE_ALLOWED,
        LayerGenerationRecord,
        OutputMode,
        build_manifest,
        write_mask_png,
    )
    prof_settings = resolve_profile(profile)
    chosen_hw = device if device is not None else prof_settings.primary_device
    # RK-16：每次 run_pipeline 重置随机源（both 模式两条线各自从头复现）
    _seed_everything(seed)

    mode = OutputMode.parse(output_mode)
    if mode == OutputMode.BOTH:
        raise ValueError("run_pipeline 只处理单一产品线；both 模式请在 main 中分别调用 plate 与 design")
    # 该产品线是否允许生成内容 —— 同时控制「超分」与「遮挡补全」两条路径
    # （PLATE 线两条都必须走确定性算法，见 §3.1 硬边界 1）
    allow_generative = GENERATIVE_UPSCALE_ALLOWED[mode]
    is_plate = (mode == OutputMode.PLATE)

    print("=" * 65)
    print("   UNIVERSAL MULTI-LAYER PSB PRODUCTION ENGINE v2.1   ")
    print("=" * 65)
    print(f"[Engine] Input Image : {input_path}")
    print(f"[Engine] Output PSB  : {output_path}")
    print(f"[Engine] Preset      : {preset_name}")
    print(f"[Engine] Random Seed : {seed} (RK-16 reproducibility)")
    print(f"[Engine] Output Mode : {mode.value.upper()} "
          f"({'CMYK 制版线 · 禁用生成内容' if is_plate else 'RGBA 设计线 · 生成内容须标记'})")
    print(f"[Engine] Profile     : {prof_settings.display_name}")
    print(f"[Engine] Primary HW  : {chosen_hw.upper()} (Shield: {prof_settings.shield_device})")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    preset = load_preset(preset_name)

    # -------------------------------------------------------------
    # Adaptive Semantics: Material Classification & Category Selection
    # 根据 preset 顶层 mode 或 CLI --adaptive-mode 覆盖决定是否启用自适应语义（Stage 1）
    # -------------------------------------------------------------
    from engine.schemas.presets import get_adaptive_mode
    # 显式 CLI 覆盖优先；缺省沿 preset（不改生产 preset 默认行为，仅用于跨类实验）
    if adaptive_mode_override in ("locked", "auto", "hybrid"):
        adaptive_mode = adaptive_mode_override
    else:
        adaptive_mode = get_adaptive_mode(preset)  # "locked" | "auto" | "hybrid"

    # 仅当显式启用 auto/hybrid 时才做自适应类目选择；
    # 默认 "locked" 保持 preset.ai_semantic_classes 为唯一真相（SSOT），不触碰。
    # effective_auto_tune / cbr_hit 在函数作用域内声明，供出图后写回 episode 使用。
    effective_auto_tune = None
    cbr_hit = None
    if adaptive_mode in ("auto", "hybrid"):
        print(f"[Adaptive] 自适应语义已启用: {adaptive_mode.upper()}"
              + (f"（CLI 覆盖）" if adaptive_mode_override in ("locked", "auto", "hybrid") else ""))
        try:
            from engine.adaptive.fingerprint import extract_fingerprint
            from engine.adaptive.material_classifier import classify_material_family
            from engine.adaptive.category_selector import (
                select_categories,
                merge_into_preset_format,
            )
            from engine.adaptive.db_manager import create_db_manager
            from engine.adaptive.episode_archiver import archive_episode

            # 读取源图像（提前读取用于指纹提取）
            src_lr_temp = imread_unicode(input_path)
            if src_lr_temp is not None:
                # 1. 提取图像指纹（84 维原始特征 → 128D PCA embedding）
                fp = extract_fingerprint(
                    src_lr_temp,
                    background_mode=preset.get("background_mode", "paper"),
                )

                # 2. 材质家族判别（规则方式，Stage 1）
                material_family, mat_conf = classify_material_family(fp)
                print(f"[Adaptive] 材质判别: {material_family}（置信度 {mat_conf:.2f}）")

                # 2.5. Stage 4：CBR 案例推理检索（复用相似历史图的已验证参数）
                #     **安全白名单**：仅当命中历史 episode 与当前图同 preset / 同产品线 /
                #     同 auto_tune 参数 schema 时才复用，杜绝跨类型配置污染。
                try:
                    from engine.adaptive.cbr_retriever import retrieve_similar_episode, get_cbr_hit_info
                    from engine.adaptive.episode_indexer import get_default_indexer
                    from pathlib import Path

                    # 获取 episode 归档目录（默认 webui/data/episodes/）
                    engine_root = Path(__file__).resolve().parent
                    episodes_dir = engine_root / "webui" / "data" / "episodes"

                    # 检索最相似的历史 episode（隔离 env：索引/日志经 ADAPTIVE_INDEX/ EPISODE_PATH）
                    indexer = get_default_indexer()
                    if len(indexer) > 0:  # 有历史索引才检索
                        cbr_hit = retrieve_similar_episode(
                            fingerprint=fp.get("embedding"),
                            indexer=indexer,
                            episodes_dir=episodes_dir,
                            min_similarity=0.7,
                            preset_name=preset_name,
                            product_line=output_mode,
                            param_schema={"global_percentiles", "regions", "density_bands"},
                        )

                        if cbr_hit:
                            cbr_info = get_cbr_hit_info(cbr_hit)
                            print(f"[Adaptive] {cbr_info}")
                            print(f"[Adaptive] CBR 命中（白名单通过）：将复用已验证调优参数")
                        else:
                            print(f"[Adaptive] 无相似历史图 / 白名单不匹配（相似度阈值 0.7），从零选择类目")
                    else:
                        print(f"[Adaptive] 索引为空（首次运行），从零选择类目")
                except Exception as cbr_err:
                    print(f"[Adaptive] CBR 检索失败（降级到从零选择）: {cbr_err}")

                # 3. 类目选择：DB 亲和度 Top-K（+ 强制核心类目 seal/calligraphy/repair_marks）
                #    注意：DB 的 category_id 命名空间 ≠ preset ai_semantic_classes[].name，
                #    桥接（preset 名 ↔ DB id）是后续阶段工作；Stage 1 以 auto 取候选，
                #    再由 merge_into_preset_format 转成 preset 格式并保留 preset 元数据。
                db_mgr = create_db_manager()
                selected_db = select_categories(
                    fp,
                    material_family,
                    str(db_mgr.db_path),
                    mode="auto",
                )

                # 4. 合并进 preset 格式（auto 替换 / hybrid 补充）
                #    Stage 2：传递 db_mgr 让转换函数查询 preset 别名表；
                #    为 merged 类补规范 category_id（DB alias / name_zh / name_en 映射）。
                merged = merge_into_preset_format(
                    preset.get("ai_semantic_classes", []),
                    selected_db,
                    adaptive_mode,
                    db=db_mgr,
                    preset_full=preset,
                )

                # 4.5. 计算本次图级有效调优参数（供 CBR 归档复用）
                #     suggest_auto_tune 基于确定性墨密度场，输出 regions / density_bands /
                #     global_percentiles，与 preset region 同构，可直接落回。
                try:
                    from engine.adaptive.auto_tune import suggest_auto_tune
                    import copy

                    # 真实计算图级调优参数（2026-09-16 修复）：
                    # 此前 import 了 suggest_auto_tune 却**从未调用**，global_percentiles
                    # 被硬编码为 {}，导致归档的 auto_tune 恒为空、CBR 命中也无参数可复用。
                    # 密度场对分辨率不敏感，先降采样到长边 1024，避免大图（16K）上耗时失控。
                    _suggested = None
                    if src_lr_temp is not None:
                        _at_img = src_lr_temp
                        try:
                            _mx = max(_at_img.shape[:2])
                            if _mx > 1024:
                                _s = 1024.0 / _mx
                                _at_img = cv2.resize(
                                    _at_img,
                                    (max(1, int(_at_img.shape[1] * _s)),
                                     max(1, int(_at_img.shape[0] * _s))),
                                    interpolation=cv2.INTER_AREA)
                        except Exception:
                            _at_img = src_lr_temp
                        _suggested = suggest_auto_tune(_at_img, classes=merged)

                    effective_auto_tune = {
                        # 真实统计（纯描述，不影响产物）
                        "global_percentiles": (_suggested or {}).get("global_percentiles", {}) or {},
                        # 生产生效的结构参数：仅取类目已有 region 与 preset 已配置的密度带
                        "regions": {(c.get("name") or c.get("layer_name")): {k: copy.deepcopy(c[k]) for k in ("region", "regions") if k in c}
                                    for c in merged if (c.get("name") or c.get("layer_name")) and ("region" in c or "regions" in c)},
                        "density_bands": copy.deepcopy(preset.get("density_band_classes") or []),
                        # 本次推荐（含九宫格兜底带）单独归档，供人工采纳/分析：
                        # **不自动注入生产** —— density_band_classes 会直接产层
                        # （grounded_sam_provider），自动注入会改变分层结果并破坏
                        # cold/repeat 字节可复现。CBR 复用只取上面「已生效」的参数。
                        "density_bands_suggested": copy.deepcopy((_suggested or {}).get("density_bands") or []),
                        "regions_suggested": copy.deepcopy((_suggested or {}).get("regions") or []),
                    }
                    if _suggested is not None:
                        _gp = effective_auto_tune["global_percentiles"]
                        print(f"[Adaptive] auto_tune 已计算：p50={_gp.get('p50')} p90={_gp.get('p90')} "
                              f"p95={_gp.get('p95')}；推荐密度带 "
                              f"{len(effective_auto_tune['density_bands_suggested'])} 条（不自动注入）")
                except Exception as at_err:
                    print(f"[Adaptive] auto_tune 计算失败（不影响主流程）: {at_err}")
                    effective_auto_tune = None

                preset["ai_semantic_classes"] = merged
                # 4.6. 仅复用真实执行过的配置，不把未经验证的密度建议注入生产。
                if cbr_hit and isinstance(effective_auto_tune, dict):
                    applied = _apply_cbr_auto_tune(preset, cbr_hit.get("auto_tune") or {})
                    if applied:
                        # 以已验证参数覆盖本次生效参数（CBR 复用收口）
                        effective_auto_tune = cbr_hit.get("auto_tune")
                        print(f"[Adaptive] CBR 已应用：复用 preset={preset_name} "
                              f"产品线={output_mode} 的已验证调优参数（冷启动加速）")
                    else:
                        print(f"[Adaptive] CBR 命中但无可应用结构参数（仅 global_percentiles），"
                              f"本次仍用本地 auto_tune")

                # 6. Stage 5.2 主动学习：把不确定类目（confidence < 0.7）写入待审队列
                #    前端每 3 秒轮询 GET /api/adaptive/pending-feedbacks，读到即弹窗请求人审。
                #    这是「生产端」——没有这段，待审队列永远为空、弹窗永不触发。
                try:
                    import os as _os, time as _time
                    from pathlib import Path as _Path
                    from engine.adaptive.active_learner import (
                        identify_uncertain_categories,
                        request_human_feedback,
                    )

                    _detections = [
                        {
                            "category_id": c.get("category_id"),
                            "confidence": float(c.get("confidence") or 0.0),
                            "prompt": c.get("name_zh") or "",
                        }
                        for c in (selected_db or [])
                        if c.get("category_id")
                    ]
                    _uncertain = identify_uncertain_categories(
                        {"detections": _detections}, confidence_threshold=0.7
                    )

                    if _uncertain:
                        # task_id：WebUI 调用时可经 TASK_ID 传入；否则由输入名+时间戳派生
                        _task_id = _os.environ.get("TASK_ID") or (
                            f"engine_{_Path(input_path).stem}_{int(_time.time())}"
                        )
                        _fb_id = request_human_feedback(
                            uncertain_categories=_uncertain,
                            task_id=_task_id,
                            image_path=input_path,
                        )
                        print(
                            f"[Adaptive] 已提交 {len(_uncertain)} 个不确定类目待人审"
                            f"（request={_fb_id}）"
                        )
                    else:
                        print("[Adaptive] 无不确定类目（选中类目 confidence 均 ≥0.7）")
                except Exception as al_err:
                    print(f"[Adaptive] 主动学习入队失败（不影响主流程）: {al_err}")

                # 5. 归档 episode（Stage 1.5）：记录本次「图像 → 材质判别 → 类目选择」交互，
                #    供 Stage 2+ 反馈学习 / 自动进化消费。默认仅 JSONL 追加，不改动 SQLite 词库，
                #    因此失败也不影响主流程（单独 try 包裹）。
                _cat_ids = [(c.get("category_id") or c.get("name")) for c in (merged or [])]

                if merged:
                    # 更新 preset 的 ai_semantic_classes（下游 segment_objects 直接消费）
                    preset["ai_semantic_classes"] = merged
                    print(f"[Adaptive] 类目已更新: {len(merged)} 个（{adaptive_mode}）")
                    for cat in merged[:5]:  # 只显示前 5 个
                        print(f"     * {cat.get('name') or cat.get('layer_name') or cat.get('label_cn')}")
                    if len(merged) > 5:
                        print(f"     ... 以及其他 {len(merged) - 5} 个类目")
                    _outcome = "selected"
                else:
                    print("[Adaptive] 未选出类目，保持原 preset 配置")
                    _outcome = "fallback_to_preset"

                try:
                    epid = archive_episode(
                        material_family=material_family,
                        category_ids=_cat_ids,
                        fingerprint=fp,
                        image_path=input_path,
                        outcome=_outcome,
                        confidence=float(mat_conf),
                        notes=f"mode={adaptive_mode}; n_selected={len(_cat_ids)}"
                              + ("; cbr_reused=1" if cbr_hit else ""),
                        output_mode=output_mode,
                        auto_tune=effective_auto_tune,
                    )
                    print(f"[Adaptive] episode 已归档: {epid}")
                    LAST_EPISODE_ID = epid
                except Exception as ae:
                    print(f"[Adaptive] episode 归档失败（不影响主流程）: {ae}")
        except Exception as e:
            print(f"[Adaptive] 自适应语义失败（降级到原 preset）: {e}")
    
    # ⚠️ cv2.imread 对含非 ASCII 字符的路径（如中文工作区）会静默返回 None——
    # 用户从 IDE / 一键 bat 传绝对路径是常态，必须走 Unicode 安全读取
    src_lr = imread_unicode(input_path)
    if src_lr is None:
        raise ValueError(f"Could not load image: {input_path}")

    h_lr, w_lr, _ = src_lr.shape
    print(f"[Engine] Source Resolution: {w_lr} x {h_lr}")

    # -------------------------------------------------------------
    # Step 0（可选）：接缝流场对齐（铁律 R3：禁止裸 vstack / hstack 硬拼）
    #   仅当 preset 显式开启 seam_harmonization 时执行（循环花纹/拼接品类，如壁布）。
    #   注意：此步会改写用于下游分层的源图（改变输出），未开启则完全不影响。
    # -------------------------------------------------------------
    _seam_cfg = preset.get("seam_harmonization") or {}
    if _seam_cfg.get("enabled", False):
        try:
            from engine.operators.seam_harmonizer import SeamHarmonizerOperator

            _seam_res = SeamHarmonizerOperator().run(
                _build_operator_ctx(src_lr, preset), params=_seam_cfg.get("params")
            )
            if _seam_res.success and _seam_res.data.get("harmonized_image") is not None:
                src_lr = _seam_res.data["harmonized_image"]
                print(f"  -> [R3] 接缝流场对齐完成: "
                      f"pre_rmse={_seam_res.metrics.get('pre_seam_rmse')} → "
                      f"post_rmse={_seam_res.metrics.get('post_seam_rmse')}")
            else:
                print(f"  -> [R3] 接缝对齐跳过: {_seam_res.message}")
        except Exception as _e:
            print(f"  -> [R3] 接缝对齐失败（不影响主流程）: {_e}")

    # Determine target resolution
    out_w, out_h = resolve_output_size(w_lr, h_lr, target_scale, target_w, target_h, preset)

    target_dpi = float(dpi) if dpi is not None else float(preset.get("dpi", 150.0))
    print(f"[Engine] Target Output Resolution: {out_w} x {out_h} (Scale: {out_w/w_lr:.2f}x)")
    print(f"[Engine] Print Resolution Target: {target_dpi:.1f} PPI (Physical: {out_w/target_dpi*25.4:.1f} x {out_h/target_dpi*25.4:.1f} mm)")

    # 2026-09-13：显式披露 preset 声明的目标画幅与实际输出的关系。
    # preset.target_w/h 是**设计参照**（如 16000x7808 对应 4000x1952 输入 x4），
    # 引擎的输出尺寸恒为 输入 x 倍率，不强制该画幅——此处打印以免被误读为强制 16K。
    _decl_w, _decl_h = preset.get("target_w"), preset.get("target_h")
    if _decl_w and _decl_h:
        if (int(_decl_w), int(_decl_h)) != (out_w, out_h):
            print(f"[Engine] 注意：preset 声明目标画幅 {int(_decl_w)}x{int(_decl_h)} 与实际输出 "
                  f"{out_w}x{out_h} 不一致——输出尺寸 = 输入 x 倍率（target_w/h 仅设计参照，不强制）")
        else:
            print(f"[Engine] preset 声明目标画幅 {int(_decl_w)}x{int(_decl_h)} 与实际输出一致")

    # -------------------------------------------------------------
    # Step 1: Grounded SAM / Universal Semantic Object Segmentation
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 1 步/共 6 步] 智能语义分割与目标解耦 (Grounded SAM / Semantic Object Segmentation)...")
    from engine.providers.grounded_sam_provider import GroundedSAMProvider
    grounded_sam = GroundedSAMProvider(preferred_device=chosen_hw, preset_name=preset_name)
    grounded_sam._preset_cfg = preset
    grounded_sam.preset = preset
    grounded_sam._load_preset_config = lambda: preset
    masks_dict = grounded_sam.segment_objects(src_lr, classes=preset.get("ai_semantic_classes"))
    # 语义覆盖披露（G5）：配置了什么/产出什么/缺什么/为什么 —— 供 manifest.totals 披露
    semantic_coverage = getattr(grounded_sam, "last_coverage", None)
    # 品类语义白名单（G2）：规则引擎的类集合是屏风系内置的，非屏风品类若不过滤，
    # 会把假阳性层（如壁布图上的「人物/建筑」）带进产物。preset 配置了
    # rule_class_allowlist（含空数组）即启用白名单；未配置则保持旧行为不过滤。
    rule_allow = preset.get("rule_class_allowlist")
    if isinstance(rule_allow, list):
        dropped = [k for k in masks_dict if k not in set(rule_allow)]
        for k in dropped:
            print(f"     - [allowlist] 丢弃非品类层: {k} ({np.count_nonzero(masks_dict[k])} px)")
        masks_dict = {k: v for k, v in masks_dict.items() if k in set(rule_allow)}
    print(f"  -> [{grounded_sam.backend}] 成功提取 {len(masks_dict)} 个解耦语义对象掩模:")
    for mname, mdata in sorted(masks_dict.items()):
        print(f"     * {mname:<38}: {np.count_nonzero(mdata):>8} 像素")
    t_step1 = time.time() - t0

    # Stage 3 学习闭环（影子模式）+ Stage 5.2 检出级不确定类目入队。
    # 必须放在分割之后：grounded_sam.dino_detections（检出 prompt/boxes/logits）此时才有值，
    # 归因与「检出置信度」判定都依赖它。全程 try 包裹，失败不影响主流程。
    try:
        from pathlib import Path as _P2

        _dino_dets = getattr(grounded_sam, "dino_detections", None) or []

        # 0) 给最终 dino_detections 附规范 category_id（DB alias / name_zh / name_en 映射）。
        #    经 preset 已 enriched 的 ai_semantic_classes（name → category_id）桥接；
        #    保留 layer_name 契约（不覆盖、不丢弃）。无映射者保持缺省
        #    （attribution 会显式判为 unknown，不伪造 ID）。
        _name_to_cid = {}
        for c in preset.get("ai_semantic_classes", []):
            _cid = c.get("category_id")
            if not _cid:
                continue
            for _key in (c.get("name"), c.get("layer_name"), c.get("label_cn")):
                if _key:
                    _name_to_cid[_key] = _cid
        for _det in _dino_dets:
            _ln = _det.get("layer_name")
            if _ln and "category_id" not in _det:
                _cid = _name_to_cid.get(_ln)
                if _cid:
                    _det["category_id"] = _cid

        # 1) Stage 3：从本次检出生成权重调整建议（只生成、不落库 = 影子模式）
        from engine.adaptive.learner import SemanticLearner

        _learn = SemanticLearner().learn_from_episode(
            {
                "episode_id": f"engine_{_P2(input_path).stem}",
                "category_ids": [c.get("category_id") or c.get("name")
                                 for c in preset.get("ai_semantic_classes", [])],
                "material_family": preset_name,
                "dino_detections": _dino_dets,
            }
        )
        _n_attr = len(_learn.get("attributions") or [])
        _n_adj = sum(len(v) for v in (_learn.get("weight_adjustments") or {}).values())
        print(f"[Adaptive] Stage 3 学习（影子模式）：{_n_attr} 条归因、{_n_adj} 条权重建议"
              + ("（已带规范 category_id，可命中 DB 提示词权重）" if _name_to_cid else "")
              + "（未落库）")

        # 2) Stage 5.2：检出置信度 <0.7 的类目 → 写入待审队列（前端弹窗人审）
        from engine.adaptive.active_learner import (
            identify_uncertain_categories,
            request_human_feedback,
        )

        _uncertain = identify_uncertain_categories(
            {"detections": _dino_dets}, confidence_threshold=0.7
        )
        if _uncertain:
            _task_id2 = os.environ.get("TASK_ID") or (
                f"engine_{_P2(input_path).stem}_{int(time.time())}"
            )
            _fb2 = request_human_feedback(
                uncertain_categories=_uncertain,
                task_id=_task_id2,
                image_path=input_path,
            )
            print(f"[Adaptive] 检出级不确定类目 {len(_uncertain)} 个已入队待人审（request={_fb2}）")

        # 3) 写回最终检测到对应 episode（出图后补完）：
        #    分割前 archive 只存语义/指纹，真实检测此时才产生。
        #    保留 artifact / output_mode / preset / effective_auto_tune。
        if LAST_EPISODE_ID:
            try:
                from engine.adaptive.episode_log_updater import update_episode_detections
                _wb_ok = update_episode_detections(
                    LAST_EPISODE_ID,
                    detections=_dino_dets,
                    artifact=output_path,
                    output_mode=output_mode,
                    preset=preset_name,
                    effective_auto_tune=effective_auto_tune,
                )
                print(f"[Adaptive] 最终检测已写回 episode {LAST_EPISODE_ID}"
                      + ("（含规范 category_id，保留 layer_name 契约）" if _name_to_cid else ""))
            except Exception as wb_err:
                print(f"[Adaptive] 最终检测写回失败（不影响主流程）: {wb_err}")
    except Exception as ln_err:
        print(f"[Adaptive] Stage 3/5.2 后置处理失败（不影响主流程）: {ln_err}")

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Step 2: Universal Background / Support Extraction
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 2 步/共 6 步] 纯净画布底板提取与去墨重构 (Canvas Support Extraction)...")
    # 补全策略同样按产品线隔离（§3.1 硬边界 1）：
    #   PLATE 线 —— LaMa 也算生成式模型输出，必须换确定性算法，
    #              否则 plate_purity 校验会判不合格（本项由 manifest 自动校验）
    inpaint_provider = None
    if allow_generative:
        try:
            from engine.providers.inpainting_provider import LaMaInpaintingProvider
            from engine.schemas.device_config import DEVICE_HARDWARE_MAP, DeviceOption

            target_hw = DEVICE_HARDWARE_MAP.get(chosen_hw.lower(), chosen_hw)
            lama = LaMaInpaintingProvider(preferred_device=target_hw, profile_name=profile)
            if lama.backend is not None:
                inpaint_provider = lama
                print(f"  -> [Neural Inpainting] LaMa FFC 频域补全已激活 [{lama.backend}].")
        except Exception as e:
            print(f"  -> [Neural Inpainting] Provider fallback: {e}")
    else:
        from engine.providers.inpainting_provider import TeleaInpaintingProvider

        inpaint_provider = TeleaInpaintingProvider(
            method=preset.get("plate_inpaint_method", "telea_ns"),
            radius=preset.get("deocclusion_radius", 7),
        )
        print(f"  -> [Deterministic Inpainting] PLATE 线使用 {inpaint_provider.method} "
              f"确定性补全（无生成模型输出，结果可复算）")

    # 供 manifest 记录的实际补全引擎名（形如 LaMaInpaintingProvider:openvino_GPU.1）
    if inpaint_provider is None:
        inpaint_engine = "none"
    else:
        _bk = getattr(inpaint_provider, "backend", None)
        _mth = getattr(inpaint_provider, "method", None)
        inpaint_engine = type(inpaint_provider).__name__
        if _mth:
            inpaint_engine += f"[{_mth}]"
        if _bk:
            inpaint_engine += f":{_bk}"

    bg_extractor = UniversalBackgroundExtractor(
        mode=preset.get("background_mode", "paper_or_gold_screen"),
        inpainting_provider=inpaint_provider
    )
    
    total_fg = np.zeros((h_lr, w_lr), dtype=np.uint8)
    for mname, m in masks_dict.items():
        mname_lower = mname.lower()
        if "frame" not in mname_lower and "seam" not in mname_lower and "fold" not in mname_lower:
            total_fg = cv2.bitwise_or(total_fg, m)

    # 底板去墨：**双保险掩模**（2026-09-12 深度审计两轮修正后定稿）
    #   ① 全画幅墨迹：低阈值覆盖淡墨（雁/淡影岩石等灰阶接近金地的对象；
    #      首版用 median-12 偏高 → 淡墨雁未被覆盖，视觉残留）
    #   ② 语义层掩模并集：被识别为对象的区域**必定**从底板抹除，不依赖灰阶阈值
    #   两者并集后再做二遍清理（旧实现只抹语义层 → 山峦残留；只抹低阈值墨迹
    #   → 淡墨对象残留；均被用户对照 V2 底板发现）
    # 折痕（屏风物理特征）不从底板抹除；未成层墨迹由"未分类墨迹残层"承载。
    gray_lr = cv2.cvtColor(src_lr, cv2.COLOR_BGR2GRAY)
    bg_median = float(np.median(gray_lr))
    ink_gray = (gray_lr < bg_median - 20.0).astype(np.uint8) * 255
    ink_all = cv2.bitwise_or(ink_gray, total_fg)
    seam_key = next((k for k in masks_dict if "seam" in k.lower() or "fold" in k.lower()), None)
    if seam_key is not None:
        ink_all = cv2.bitwise_and(ink_all, cv2.bitwise_not(masks_dict[seam_key]))
    print(f"  -> [底板去墨] 双保险掩模覆盖 {np.count_nonzero(ink_all)/ink_all.size*100:.1f}% "
          f"（灰阶墨迹 {np.count_nonzero(ink_gray)/ink_gray.size*100:.1f}% ∪ 语义层 "
          f"{np.count_nonzero(total_fg)/total_fg.size*100:.1f}%），折痕保留")

    clean_bg_lr, _ = bg_extractor.extract_clean_background(
        src_lr, foreground_mask=ink_all, panel_count=preset.get("panel_count")
    )
    print("  -> 纯净画布金箔底板已高质量重构完成。")

    # 未分类墨迹残层：全墨迹减去所有已产出语义层 —— 承载未被语义类覆盖的内容，
    # 避免"底板抹墨 + 无层承载"导致内容丢失（如远山等源图固有难度类）。
    unassigned = cv2.bitwise_and(ink_all, cv2.bitwise_not(total_fg))
    n_unassigned = int(np.count_nonzero(unassigned))
    if n_unassigned > 0.001 * ink_all.size:
        masks_dict["11_未分类墨迹残层_Unclassified_Ink_Residue"] = unassigned
        print(f"  -> [未分类墨迹残层] {n_unassigned:,}px "
              f"({n_unassigned/unassigned.size*100:.2f}%) 已作为独立层承载（避免内容丢失）")
    else:
        print("  -> [未分类墨迹残层] 无残留（语义层已覆盖全部墨迹）")
    t_step2 = time.time() - t0

    # -------------------------------------------------------------
    # Step 3: Universal 2.5D Layer Depth & Topology Sorting
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 3 步/共 6 步] 2.5D 层序深度拓扑排序 (2.5D Layer Depth & Topology Sorting)...")
    sorter = UniversalLayerSorter()
    sorted_layers = sorter.sort_layers(masks_dict, src_lr)
    
    # 层序规范化（2026-09-12 深度审计修复）：
    # 原实现直接沿用深度排序器输出，实例层（04C_03/_05/_04…、08_雁_09/_01…）
    # 与语义层交错，设计师在 PS 中找层困难（实测 39/43 层与规范顺序不符）。
    # 现按「名称前缀数字 + 字母」稳定排序，使堆叠呈现
    # 底板(02) → 内容(03…09) → 装饰(10A/10B) → 残层(11) 的规范层次。
    def _layer_sort_key(nm: str):
        import re as _re
        m = _re.match(r"^(\d+)([A-Za-z]?)", str(nm))
        return (int(m.group(1)), m.group(2).upper(), str(nm)) if m else (99, "", str(nm))

    sorted_layers.sort(key=lambda _l: _layer_sort_key(_l.get("name", "")))

    # 图层属性由 preset.layer_semantics.layer_attributes 驱动（SSOT，G2）：
    # 品类语义（如折痕层的正片叠底与古雅基色）只改 preset，不改内核
    for lyr in sorted_layers:
        applied = match_layer_attributes(preset, lyr["name"])
        if applied:
            lyr.update(applied)

    print("  -> Photoshop UI 图层从顶至底排列顺序:")
    for idx, lyr in enumerate(sorted_layers):
        print(f"     [{idx:02d}] {lyr['name']:<38} (混合: {lyr['blend_mode']:<8} 不透明度: {lyr['opacity']:<3} 深度Z: {lyr['z_index']:.1f})")
    t_step3 = time.time() - t0

    # -------------------------------------------------------------
    # Step 3.5: PLATE 线印前算子链（轮廓保全 / 微孔刀模 / 专色陷印）
    #   品类可选：由 preset.plate_operators 配置启停（ADR-008 / ADR-017）
    # -------------------------------------------------------------
    plate_ops_summary: dict = {}
    if is_plate:
        t_ops = time.time()
        print("\n[第 3.5 步] PLATE 印前算子链 (轮廓保全 / 微孔刀模 / 专色陷印)...")
        plate_ops_summary, extra_layers = _run_plate_operators(
            src_lr, sorted_layers, preset, target_dpi)
        for lyr in extra_layers:
            sorted_layers.append(lyr)
            print(f"     + 新增图层: {lyr['name']}")
        for op_name, info in plate_ops_summary.items():
            print(f"     - {op_name}: {info.get('status')}"
                  + (f" ({info['reason']})" if info.get("reason") else ""))
        plate_ops_time = time.time() - t_ops

    # -------------------------------------------------------------
    # Step 4: Universal 2.5D De-occlusion Completion & Reconstruction Marking
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 4 步/共 6 步] 2.5D 遮挡定向补全与重建区标定 (De-occlusion & Reconstruction Marking)...")
    
    # 纯动态 2.5D 定向遮挡补全（由 DeocclusionOperator 与 LaMa/Telea 算子实时执行）

    from engine.operators.deocclusion_operator import DeocclusionOperator
    deoc_op = DeocclusionOperator()
    deoc_res = deoc_op.run(
        ctx=None,
        params={
            "image": src_lr,
            "layers": sorted_layers,
            "inpainting_provider": inpaint_provider,
            "extension_pixels": preset.get("deocclusion_extension_px", 20),
            "inpaint_radius": preset.get("deocclusion_radius", 7),
            "max_area_ratio": preset.get("max_reconstruction_ratio", 0.08),
        }
    )
    if deoc_res.success:
        recon_px = deoc_res.metrics.get("reconstruction_pixel_count", 0)
        recon_ratio = deoc_res.metrics.get("reconstruction_area_ratio", 0.0)
        print(f"  -> 【铁律 R1 合规】重建区独立标定: 共计 {recon_px} 颗生成像素 (全幅占比 {recon_ratio:.2%})，受控追溯。")
    t_step4 = time.time() - t0

    # -------------------------------------------------------------
    # Step 5: Progressive Multi-Scale Super-Resolution & Guided Upsampling
    # -------------------------------------------------------------
    t0 = time.time()
    stages = preset.get("progressive_stages", [1.5, 2.0, 4.0])
    print(f"\n[第 5 步/共 6 步] 阶梯式超分辨率与引导滤波 (Progressive Super-Res Scaling to {out_w}x{out_h})...")

    super_res = UniversalTiledSuperRes(target_w=out_w, target_h=out_h)

    # -------------------------------------------------------------
    # 超分策略按产品线隔离（ADR-011 / §3.1 硬边界 1）
    #   PLATE 线：禁止生成式输出 —— 制版验收要求可复算、可对色、可过 TAC 审计，
    #            故只走阶梯式 Lanczos + 引导滤波（确定性插值，不生成新结构）；
    #   DESIGN 线：允许生成式超分，但 manifest 必须披露并输出生成区掩模。
    # -------------------------------------------------------------
    if allow_generative:
        from engine.providers.realesrgan_provider import RealESRGANProvider
        esrgan = RealESRGANProvider(
            preferred_device=chosen_hw,
            profile_name=profile,
            tile_size=preset.get("tile_size", 512),
            tile_pad=preset.get("tile_pad", 32),
        )
        if esrgan.backend and "openvino" in esrgan.backend:
            print(f"  -> [Neural Super-Res] Real-ESRGAN 已激活 [{esrgan.backend}] (Tiled {esrgan.tile_size}px)")
        else:
            print(f"  -> [Neural Super-Res] Provider 已降级 [{esrgan.backend}]")
        print(f"  -> [Real-ESRGAN] 启动全图神经切片超分重建 ({w_lr}x{h_lr} -> {out_w}x{out_h})...")
        src_hr = esrgan.upscale(src_lr, target_w=out_w, target_h=out_h)
        sr_engine = f"realesrgan:{esrgan.backend}"
        sr_generative = True
        print(f"  -> [Real-ESRGAN] 神经推理完成: 实际输出={src_hr.shape}  "
              f"[DESIGN 线·生成式，已记入 manifest]")
    else:
        # PLATE 线：确定性插值，不引入生成结构
        print(f"  -> [Deterministic Super-Res] PLATE 线禁用生成式超分，改用阶梯式 Lanczos + 引导滤波")
        src_hr = super_res.upscale_image_progressive(src_lr, stages=stages)
        sr_engine = "lanczos_guided_non_generative"
        sr_generative = False
        print(f"  -> [Deterministic Super-Res] 完成: 实际输出={src_hr.shape}  [PLATE 线·无生成内容]")

    # 纯净金箔底板超分辨率生成
    print(f"  -> [Super-Res] 阶梯式引导超分生成 16K 纯净金箔底板 ({out_w}x{out_h})...")
    bg_hr = super_res.upscale_image_progressive(clean_bg_lr, stages)

    guide_hr_gray = cv2.cvtColor(src_hr, cv2.COLOR_BGR2GRAY)

    def process_single_layer(lyr):
        name = lyr["name"]
        m_lr = lyr["mask"]
        
        # 【方案 A 纯动态计算】废除任何 masks_16k/*.png 直读，100% 走真实引导滤波超分！
        hr_m = super_res.guided_upsample_mask(m_lr, guide_hr_gray, radius=6, eps=1e-3)
            
        if lyr.get("inpainted_bgr") is not None:
            hr_bgr = super_res.upscale_image_progressive(lyr["inpainted_bgr"], stages=stages)
        else:
            hr_bgr = None
        return name, hr_m, hr_bgr

    hr_masks_dict = {}
    workers = min(6, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_single_layer, sorted_layers))

    for name, hr_m, hr_bgr in results:
        hr_masks_dict[name] = hr_m
        for lyr in sorted_layers:
            if lyr["name"] == name:
                lyr["inpainted_hr_bgr"] = hr_bgr
                break
        print(f"     * 超分边缘细化完成: {name}")
    t_step5 = time.time() - t0

    # -------------------------------------------------------------
    # Step 6: Universal PSB Stream Assembly
    # -------------------------------------------------------------
    t0 = time.time()
    from pytoshop import enums
    print(f"\n[第 6 步/共 6 步] 多图层 PSB 流式组装与 SIMD 极速编码 (DPI={target_dpi}, C-Accelerated RLE)...")
    # 底板图层名从 preset 读取，消除原先 `if preset_name == "japanese_screen_gold"` 的品类硬编码分支（G1）
    bg_name = preset.get("bg_layer_name", "01_纯净画布底板_Base_Ground")
    # PLATE 线产出真正的 CMYK 分色版；DESIGN 线产出 RGB 元素层
    ps_color_mode = "cmyk" if is_plate else "rgb"

    # 印前合规：ICC 驱动分色（含黑版生成）+ TAC 工艺压制（ADR-007）
    icc_path = icc_override or preset.get("icc_path")
    tac_policy = None
    icc_info = None
    if is_plate:
        from engine.core.ink_limiter import icc_summary, resolve_policy
        icc_info = icc_summary(icc_path)
        tac_policy = resolve_policy(
            condition=preset.get("print_condition"),
            limit_pct=preset.get("tac_limit_pct"),
            max_k_pct=preset.get("max_k_pct"),
            icc_path=icc_path,
        )
        print(f"  -> [Prepress] ICC: {icc_info.get('description') or '未提供（朴素 RGB→CMYK 转换，无色彩管理）'}")
        print(f"  -> [Prepress] TAC 上限 {tac_policy.limit_pct:.0f}% / MaxK {tac_policy.max_k_pct:.0f}%"
              f"  来源: {tac_policy.source}")

    builder = UniversalPSBBuilder(
        target_w=out_w, target_h=out_h, dpi=target_dpi,
        compression=enums.Compression.rle, color_mode=ps_color_mode,
        icc_path=icc_path, tac_policy=tac_policy,
    )
    builder.build_psb(output_path, src_hr, bg_hr, sorted_layers, hr_masks_dict, bg_layer_name=bg_name)
    t_step6 = time.time() - t0

    # -------------------------------------------------------------
    # 交付清单（manifest）：把生成内容从「只打日志」改为随产物持久化（铁律 R1 / §3.1）
    # -------------------------------------------------------------
    from engine.core.models import new_run_id
    run_id = new_run_id()
    stem = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    mask_dir = os.path.join(out_dir, f"{stem}.masks")
    manifest_path = os.path.join(out_dir, f"{stem}.manifest.json")
    os.makedirs(mask_dir, exist_ok=True)

    man = build_manifest(
        run_id=run_id,
        output_mode=mode.value,
        source_path=input_path,
        source_wh=(w_lr, h_lr),
        output_path=output_path,
        output_wh=(out_w, out_h),
        ppi=target_dpi,
        color_mode=ps_color_mode,
    )
    man.generation_policy.declare(
        "super_resolution", sr_engine, sr_generative, 1.0,
        f"{w_lr}x{h_lr} -> {out_w}x{out_h}"
    )
    if deoc_res.success:
        man.generation_policy.declare(
            "deocclusion",
            inpaint_engine,
            bool(getattr(inpaint_provider, "is_generative", False)),
            deoc_res.metrics.get("reconstruction_area_ratio", 0.0),
            "2.5D 遮挡定向补全",
        )

    total_recon_px = 0
    for lyr in sorted_layers:
        name = lyr["name"]
        rec = LayerGenerationRecord(name=name)

        # (a) 生成式超分：作用于整层
        if sr_generative:
            rec.add_reason(f"super_resolution({sr_engine})")

        # (b) 遮挡补全：精确掩码，落盘为 PNG 便于回灌时剔除
        rmask = lyr.get("reconstruction_mask")
        hr_mask = hr_masks_dict.get(name)
        if hr_mask is not None:
            pts = cv2.findNonZero(hr_mask)
            if pts is not None:
                rec.bbox = tuple(int(v) for v in cv2.boundingRect(pts))  # (x, y, w, h)
                rx, ry, rw, rh = rec.bbox
                rec.bbox = (rx, ry, rx + rw, ry + rh)                   # -> (l, t, r, b)
        if rmask is not None and np.count_nonzero(rmask) > 0:
            # 生成式补全（LaMa）计为生成内容；确定性补全（Telea/NS）仅留追溯记录，
            # 这样 PLATE 产物既保住了补全可用性，又不会判为含生成内容。
            if getattr(inpaint_provider, "is_generative", False):
                rec.add_reason(f"deocclusion({inpaint_engine})")
            else:
                rec.add_deterministic_fill(f"deocclusion({inpaint_engine})")
            rec.recon_pixel_count = int(np.count_nonzero(rmask))
            rec.recon_ratio = rec.recon_pixel_count / max(1, rmask.size)
            rec.recon_mask_size = (int(rmask.shape[1]), int(rmask.shape[0]))
            # 掩码文件名用 ASCII slug（跨平台/工具链安全）；中英文完整图层名
            # 保留在 manifest 的 name 字段与 PSB 图层名（交付显示文本）
            slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "layer"
            mask_name = f"{slug}.recon.png"
            rec.recon_mask_path = f"{stem}.masks/{mask_name}"
            if not write_mask_png(os.path.join(mask_dir, mask_name), rmask):
                print(f"  ⚠️  [Manifest] 掩码写入失败: {mask_name}")
            total_recon_px += rec.recon_pixel_count

        man.layers.append(rec)

    # 支撑层补录（2026-09-12 审计修复）：底板由 psb_builder 单独添加，
    # 不在 sorted_layers 中，导致 manifest 层清单与实际 PSB 差 1 层
    # （实测 42 vs 43）。此处补录，保证「manifest 层清单 == PSB 实际层」。
    # 补全语义与 deocclusion 一致（manifest.py:184）：生成引擎（LaMa）计为生成
    # 内容；确定性引擎（Telea/NS）走 deterministic_fill（追溯不判生成），
    # 否则 PLATE 线纯净性校验会误判底板为生成内容（实测回归，已修正）。
    _declared = {getattr(r, "name", "") for r in man.layers}
    for _sup_name in (bg_name,):
        if _sup_name and _sup_name not in _declared:
            _sup_rec = LayerGenerationRecord(name=_sup_name)
            if getattr(inpaint_provider, "is_generative", False):
                _sup_rec.add_reason(f"support_background({inpaint_engine})")
            else:
                _sup_rec.add_deterministic_fill(f"support_background({inpaint_engine})")
            _sup_rec.recon_pixel_count = int(total_recon_px)
            man.layers.append(_sup_rec)

    man.totals["generated_pixel_ratio"] = round(man.generated_ratio(), 6)
    man.totals["deterministic_fill_ratio"] = round(man.deterministic_fill_ratio(), 6)
    man.totals["super_resolution_engine"] = sr_engine
    man.totals["super_resolution_generative"] = sr_generative
    man.totals["deocclusion_engine"] = inpaint_engine
    man.totals["deocclusion_generative"] = bool(getattr(inpaint_provider, "is_generative", False))
    man.totals["deocclusion_pixel_count"] = total_recon_px
    man.totals["random_seed"] = int(seed)
    if getattr(builder, "last_tac", None):
        man.totals["tac_max_pct"] = round(float(builder.last_tac[0]), 2)
        man.totals["tac_mean_pct"] = round(float(builder.last_tac[1]), 2)
    ink_stats = getattr(builder, "last_ink_stats", None)
    if ink_stats:
        man.totals["ink_compliance"] = ink_stats
    if icc_info is not None:
        man.totals["color_management"] = icc_info
    if semantic_coverage:
        man.totals["semantic_coverage"] = semantic_coverage

    # PLATE 纯净性校验结果写入 manifest，供下游质检与回灌环节读取
    ok_plate, plate_msg = man.assert_plate_purity()
    man.totals["plate_purity_ok"] = bool(ok_plate)
    man.totals["plate_purity_message"] = plate_msg
    if plate_ops_summary:
        man.totals["plate_operators"] = plate_ops_summary
    man.save(manifest_path, mask_dir=None)

    if is_plate and not ok_plate:
        print(f"  ⚠️  [PLATE 纯净性告警] {plate_msg}")
        print("      （依据 §3.1 硬边界 1：生成式模型输出不得进入 PLATE 线）")
    print(f"  -> [Manifest] 交付清单已落盘: {manifest_path}")
    print(f"  -> [Manifest] 生成内容披露: 超分引擎={sr_engine}(生成式={sr_generative})、"
          f"补全像素={total_recon_px}、有效源={man.source['effective_ppi']} PPI")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    elapsed = time.time() - t_start

    print("\n" + "=" * 65)
    print("           UNIVERSAL PSB GENERATION COMPLETE!            ")
    print("=" * 65)
    print(f"[Success] 输出成品文件 : {output_path}")
    print(f"[Success] 几何输出画幅 : {out_w} x {out_h} @ {target_dpi:.1f} PPI")
    print(f"[Success] 物理文件体积 : {file_size_gb:.2f} GB ({file_size_mb:.1f} MB)")
    print(f"[Success] 端到端总耗时 : {elapsed:.1f} 秒 ({elapsed/60:.2f} 分钟)")
    print(f"[Success] 独立图层总数 : {len(sorted_layers) + 1} 个图层 (中英对照/含最小外接矩形)")
    print("-" * 65)
    print("  全流程步骤耗时明细 (REAL BENCHMARK):")
    print(f"  - 第 1 步 智能语义对象解耦分割  : {t_step1:.2f}s")
    print(f"  - 第 2 步 画布底板无缝去墨重构  : {t_step2:.2f}s")
    print(f"  - 第 3 步 2.5D 图层深度拓扑排序 : {t_step3:.2f}s")
    print(f"  - 第 4 步 2.5D 遮挡补全与重建区 : {t_step4:.2f}s (铁律 R1 达成)")
    print(f"  - 第 5 步 阶梯式超分辨率与滤波  : {t_step5:.2f}s")
    print(f"  - 第 6 步 SIMD C-PackBits 编码 : {t_step6:.2f}s (C-Accelerated)")
    print("=" * 65)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Multi-layer PSB Production Engine")
    parser.add_argument("--input", required=True, help="Path to input source image")
    parser.add_argument("--output", required=True, help="Path to output .psb file")
    parser.add_argument("--preset", default="japanese_screen_gold", help="Style preset name or JSON path")
    parser.add_argument("--scale", type=float, default=None,
                        help="Upscale scaling factor（CLI 最高优先；未提供时用 preset.super_res_scale，再否则 4.0）")
    parser.add_argument("--dpi", type=float, default=150.0, help="Target print resolution in PPI (default: 150.0)")
    parser.add_argument("--width", type=int, default=None, help="Explicit target width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Explicit target height in pixels")
    parser.add_argument(
        "--profile",
        choices=["robust_performance", "5070", "arc", "cpu"],
        default="robust_performance",
        help="Work intent profile: robust_performance (Default: RTX 5070 + Arc 16GB shield + CPU circuit breaker), 5070 (direct dGPU), arc (direct iGPU 16GB), cpu (pure CPU)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "5070", "arc", "npu", "cpu"],
        default=None,
        help="Low-level hardware override (optional, defaults to profile configuration)"
    )
    parser.add_argument(
        "--icc",
        default=None,
        help="目标印刷条件的 ICC profile 路径（覆盖 preset.icc_path）。"
             "用于 CMYK 分色与黑版生成；缺失时回退朴素转换并如实标注。"
             "注意：TAC 上限来自印刷工艺参数，不在 ICC 中",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="全局随机种子（RK-16：固定后两次运行的分割掩模应逐像素一致）",
    )
    parser.add_argument(
        "--mode",
        choices=["plate", "design", "both"],
        default=None,
        help="产品线（ADR-011）：plate=CMYK 制版线（禁用生成内容）；"
             "design=RGBA 设计线（允许生成但须标记）；both=两条线各产一份。"
             "留空则读 preset 的 output.mode，再兜底为 design"
    )
    parser.add_argument(
        "--adaptive-mode",
        choices=["locked", "auto", "hybrid"],
        default=None,
        help="自适应语义模式显式覆盖（跨类实验用）：locked/auto/hybrid。"
             "缺省沿 preset 顶层 mode（不改生产 preset 默认行为）。"
             "仅当 preset 顶层 mode 非 locked 时才生效（auto/hybrid 需 preset 含自适应配置）。"
    )

    args = parser.parse_args()

    # 产品线优先级：命令行 > preset.output.mode > design
    mode = args.mode
    if mode is None:
        try:
            mode = (load_preset(args.preset).get("output") or {}).get("mode") or "design"
        except Exception:
            mode = "design"

    from engine.schemas.manifest import OutputMode
    mode_enum = OutputMode.parse(mode)

    if mode_enum == OutputMode.BOTH:
        # 两条线各产一份，路径加后缀以避免互相覆盖
        root, ext = os.path.splitext(args.output)
        jobs = [(OutputMode.PLATE, f"{root}.plate{ext}"), (OutputMode.DESIGN, f"{root}.design{ext}")]
    else:
        jobs = [(mode_enum, args.output)]

    for job_mode, job_out in jobs:
        run_pipeline(
            input_path=args.input,
            output_path=job_out,
            preset_name=args.preset,
            target_scale=args.scale,
            target_w=args.width,
            target_h=args.height,
            dpi=args.dpi,
            device=args.device,
            profile=args.profile,
            output_mode=job_mode.value,
            icc_override=args.icc,
            seed=args.seed,
            adaptive_mode_override=args.adaptive_mode,
        )
        # B5：出图后自动审计 + finalize_episode 收口 + 重建 CBR 索引（CLI 冷启动闭环）
        if str(os.environ.get("ULS_AUDIT_AFTER_RUN", "")).lower() in ("1", "true", "yes"):
            _rc = _post_run_audit_and_sync(job_out, args.input, output_mode=job_mode.value)
            # 严格测试环境：审计失败 / 报告不完整 / 未准入 → 传播为非零退出
            if _audit_strict() and _rc != 0:
                print(f"[Main] ULS_AUDIT_STRICT=1：审计收口失败（rc={_rc}），进程非零退出。")
                sys.exit(_rc)
