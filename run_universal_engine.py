"""Universal Multi-layer PSB Engine - CLI Entry Point.
Takes ANY input image and automatically produces a layered, print-ready PSB file.

Usage:
    python run_universal_engine.py --input inputs/source_4000.jpg --output outputs/universal_output.psb --preset traditional_chinese_ink
"""
import os
import sys
import json
import time
import argparse
import cv2
import numpy as np

from engine.background_extractor import UniversalBackgroundExtractor
from engine.semantic_segmenter import UniversalSemanticSegmenter
from engine.depth_layer_sorter import UniversalLayerSorter
from engine.deocclusion import UniversalDeoccluder
from engine.tiled_super_res import UniversalTiledSuperRes
from engine.psb_builder import UniversalPSBBuilder
from concurrent.futures import ThreadPoolExecutor

def load_preset(preset_arg):
    # Check if preset_arg is a preset name or file path
    if os.path.isfile(preset_arg):
        with open(preset_arg, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    preset_path = os.path.join("presets", f"{preset_arg}.json")
    if os.path.isfile(preset_path):
        with open(preset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    # Default fallback
    print(f"Warning: Preset '{preset_arg}' not found, falling back to 'traditional_chinese_ink'")
    default_path = os.path.join("presets", "traditional_chinese_ink.json")
    with open(default_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_pipeline(input_path, output_path, preset_name="japanese_screen_gold", target_scale=4.0, target_w=None, target_h=None, dpi=None, device=None, profile="robust_performance", output_mode="design"):
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

    mode = OutputMode.parse(output_mode)
    if mode == OutputMode.BOTH:
        raise ValueError("run_pipeline 只处理单一产品线；both 模式请在 main 中分别调用 plate 与 design")
    use_generative_sr = GENERATIVE_UPSCALE_ALLOWED[mode]
    is_plate = (mode == OutputMode.PLATE)

    print("=" * 65)
    print("   UNIVERSAL MULTI-LAYER PSB PRODUCTION ENGINE v2.1   ")
    print("=" * 65)
    print(f"[Engine] Input Image : {input_path}")
    print(f"[Engine] Output PSB  : {output_path}")
    print(f"[Engine] Preset      : {preset_name}")
    print(f"[Engine] Output Mode : {mode.value.upper()} "
          f"({'CMYK 制版线 · 禁用生成内容' if is_plate else 'RGBA 设计线 · 生成内容须标记'})")
    print(f"[Engine] Profile     : {prof_settings.display_name}")
    print(f"[Engine] Primary HW  : {chosen_hw.upper()} (Shield: {prof_settings.shield_device})")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    preset = load_preset(preset_name)
    src_lr = cv2.imread(input_path)
    if src_lr is None:
        raise ValueError(f"Could not load image: {input_path}")

    h_lr, w_lr, _ = src_lr.shape
    print(f"[Engine] Source Resolution: {w_lr} x {h_lr}")

    # Determine target resolution
    if target_w is not None and target_h is not None:
        out_w, out_h = int(target_w), int(target_h)
    else:
        scale = preset.get("super_res_scale", target_scale)
        out_w, out_h = int(w_lr * scale), int(h_lr * scale)

    target_dpi = float(dpi) if dpi is not None else float(preset.get("dpi", 150.0))
    print(f"[Engine] Target Output Resolution: {out_w} x {out_h} (Scale: {out_w/w_lr:.2f}x)")
    print(f"[Engine] Print Resolution Target: {target_dpi:.1f} PPI (Physical: {out_w/target_dpi*25.4:.1f} x {out_h/target_dpi*25.4:.1f} mm)")

    # -------------------------------------------------------------
    # Step 1: Grounded SAM / Universal Semantic Object Segmentation
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 1 步/共 6 步] 智能语义分割与目标解耦 (Grounded SAM / Semantic Object Segmentation)...")
    from engine.providers.grounded_sam_provider import GroundedSAMProvider
    grounded_sam = GroundedSAMProvider(preferred_device=chosen_hw, preset_name=preset_name)
    masks_dict = grounded_sam.segment_objects(src_lr, classes=preset.get("ai_semantic_classes"))
    print(f"  -> [{grounded_sam.backend}] 成功提取 {len(masks_dict)} 个解耦语义对象掩模:")
    for mname, mdata in sorted(masks_dict.items()):
        print(f"     * {mname:<38}: {np.count_nonzero(mdata):>8} 像素")
    t_step1 = time.time() - t0

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Step 2: Universal Background / Support Extraction
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 2 步/共 6 步] 纯净画布底板提取与去墨重构 (Canvas Support Extraction)...")
    lama_provider = None
    try:
        from engine.providers.inpainting_provider import LaMaInpaintingProvider
        from engine.schemas.device_config import DEVICE_HARDWARE_MAP, DeviceOption

        target_hw = DEVICE_HARDWARE_MAP.get(chosen_hw.lower(), chosen_hw)
        lama = LaMaInpaintingProvider(preferred_device=target_hw, profile_name=profile)
        if lama.backend is not None:
            lama_provider = lama
            print(f"  -> [Neural Inpainting] LaMa FFC 频域补全已激活 [{lama.backend}].")
    except Exception as e:
        print(f"  -> [Neural Inpainting] Provider fallback: {e}")

    bg_extractor = UniversalBackgroundExtractor(
        mode=preset.get("background_mode", "paper_or_gold_screen"),
        inpainting_provider=lama_provider
    )
    
    total_fg = np.zeros((h_lr, w_lr), dtype=np.uint8)
    for mname, m in masks_dict.items():
        mname_lower = mname.lower()
        if "frame" not in mname_lower and "seam" not in mname_lower and "fold" not in mname_lower:
            total_fg = cv2.bitwise_or(total_fg, m)

    clean_bg_lr, _ = bg_extractor.extract_clean_background(
        src_lr, foreground_mask=total_fg, panel_count=preset.get("panel_count")
    )
    print("  -> 纯净画布金箔底板已高质量重构完成。")
    t_step2 = time.time() - t0

    # -------------------------------------------------------------
    # Step 3: Universal 2.5D Layer Depth & Topology Sorting
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n[第 3 步/共 6 步] 2.5D 层序深度拓扑排序 (2.5D Layer Depth & Topology Sorting)...")
    sorter = UniversalLayerSorter()
    sorted_layers = sorter.sort_layers(masks_dict, src_lr)
    
    # 针对屏风物理折痕，强制赋予 MULTIPLY 正片叠底模式与古雅折痕基色
    for lyr in sorted_layers:
        lname = lyr["name"].lower()
        if "seam" in lname or "fold" in lname or "折痕" in lname or "折缝" in lname:
            lyr["fixed_color"] = [25, 20, 15]
            lyr["blend_mode"] = "MULTIPLY"
            lyr["opacity"] = 190

    print("  -> Photoshop UI 图层从顶至底排列顺序:")
    for idx, lyr in enumerate(sorted_layers):
        print(f"     [{idx:02d}] {lyr['name']:<38} (混合: {lyr['blend_mode']:<8} 不透明度: {lyr['opacity']:<3} 深度Z: {lyr['z_index']:.1f})")
    t_step3 = time.time() - t0

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
            "inpainting_provider": lama_provider,
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
    if use_generative_sr:
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
    builder = UniversalPSBBuilder(target_w=out_w, target_h=out_h, dpi=target_dpi, compression=enums.Compression.rle)
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
        color_mode="rgb",   # PLATE 的 CMYK 输出尚未接入（见尽调报告 P1-5），如实标注
    )
    man.generation_policy.declare(
        "super_resolution", sr_engine, sr_generative, 1.0,
        f"{w_lr}x{h_lr} -> {out_w}x{out_h}"
    )
    if deoc_res.success:
        man.generation_policy.declare(
            "deocclusion",
            ("lama:" + str(lama_provider.backend)) if lama_provider is not None else "cv2.telea",
            lama_provider is not None,
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
            rec.add_reason("deocclusion")
            rec.recon_pixel_count = int(np.count_nonzero(rmask))
            rec.recon_ratio = rec.recon_pixel_count / max(1, rmask.size)
            rec.recon_mask_size = (int(rmask.shape[1]), int(rmask.shape[0]))
            mask_name = f"{name}.recon.png"
            rec.recon_mask_path = f"{stem}.masks/{mask_name}"
            # 图层名含中文，cv2.imwrite 在 Windows 上会静默失败（返回 True 但不写文件）
            if not write_mask_png(os.path.join(mask_dir, mask_name), rmask):
                print(f"  ⚠️  [Manifest] 掩码写入失败: {mask_name}")
            total_recon_px += rec.recon_pixel_count

        man.layers.append(rec)

    man.totals["generated_pixel_ratio"] = round(man.generated_ratio(), 6)
    man.totals["super_resolution_engine"] = sr_engine
    man.totals["super_resolution_generative"] = sr_generative
    man.totals["deocclusion_pixel_count"] = total_recon_px

    # PLATE 纯净性校验结果写入 manifest，供下游质检与回灌环节读取
    ok_plate, plate_msg = man.assert_plate_purity()
    man.totals["plate_purity_ok"] = bool(ok_plate)
    man.totals["plate_purity_message"] = plate_msg
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
    parser.add_argument("--scale", type=float, default=4.0, help="Upscale scaling factor (default: 4.0)")
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
        "--mode",
        choices=["plate", "design", "both"],
        default=None,
        help="产品线（ADR-011）：plate=CMYK 制版线（禁用生成内容）；"
             "design=RGBA 设计线（允许生成但须标记）；both=两条线各产一份。"
             "留空则读 preset 的 output.mode，再兜底为 design"
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
        )
