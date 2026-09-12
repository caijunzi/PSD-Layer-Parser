"""Universal Background Extractor.
Extracts and reconstructs clean canvas/support plates (宣纸/绢本/金箔/画布底板)
from arbitrary art images without hardcoded coordinates.
"""
import cv2
import numpy as np
import os

class UniversalBackgroundExtractor:
    def __init__(self, mode="paper_or_gold_screen", inpainting_provider=None):
        self.mode = mode
        self.inpainting_provider = inpainting_provider

    def extract_clean_background(self, img_bgr, foreground_mask=None, panel_count=None, painting_roi=None, cached_bg_path=None):
        """
        Reconstructs a clean, seamless background support plate.
        
        Args:
            img_bgr: Source image in BGR format.
            foreground_mask: Binary mask of detected foreground objects (uint8, 255=foreground).
                             If None, automatically detected via color/luminance analysis.
            panel_count: Optional number of folding panels (e.g., 6 for six-panel screens).
                         If None, auto-detected or treated as a continuous surface.
            painting_roi: Optional active canvas boundary mask.
            cached_bg_path: Optional path to pre-extracted clean background file.
        Returns:
            clean_bg: Clean reconstructed background image (BGR).
            bg_mask: Mask of pure background regions in original image.
        """
        h, w, c = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 0. If pre-extracted / cached clean background is provided and valid, prioritize it
        if cached_bg_path and os.path.isfile(cached_bg_path):
            # cv2.imread 对含非 ASCII 字符的路径会静默返回 None，必须走 imdecode
            from engine.core.io_utils import imread_unicode

            cached = imread_unicode(cached_bg_path)
            if cached is not None:
                if cached.shape[:2] != (h, w):
                    cached = cv2.resize(cached, (w, h), interpolation=cv2.INTER_LANCZOS4)
                pure_bg_mask = np.ones((h, w), dtype=bool)
                return cached, pure_bg_mask

        # 1. Automatic foreground ink/paint detection if not provided
        if foreground_mask is None:
            # Otsu + adaptive thresholding for ink/pigment detection
            blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
            # Support material is typically the dominant bright/warm background
            # Compute Otsu threshold on the canvas
            thresh_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Cap threshold relative to median luminance to avoid over-segmenting
            median_val = np.median(gray)
            dynamic_thresh = min(thresh_val, median_val - 15)
            foreground_mask = (gray < dynamic_thresh).astype(np.uint8) * 255

        # 2. Pure background mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil_fg = cv2.dilate(foreground_mask, kernel)
        pure_bg_mask = (dil_fg == 0)

        # If pure background is too small (<5% of image), fall back to border sampling
        if np.count_nonzero(pure_bg_mask) < 0.05 * h * w:
            pure_bg_mask[:int(h*0.1), :] = True
            pure_bg_mask[int(h*0.9):, :] = True

        # 3. Reconstruction strategy
        #    2026-09-12 深度审计（两轮实证后定稿）：
        #    第一轮：把折叠屏风从 _reconstruct_paneled_screen 改为 LaMa 逐像素补全
        #    —— 实测**更差**（大墨块区补不净：右侧山峦残留 1.1%→21.8%、
        #    右下岩石 17.8%）。LaMa 擅长小缝/细节补全，不擅长整片墨区。
        #    第二轮（当前）：保留 panel 参考扇复制（大区域去墨彻底）+ **强化墨迹
        #    核心区 alpha**（原实现只用高斯模糊 alpha，墨迹核心仅 ~0.6 → 残留；
        #    现取 max(模糊, 二值×0.95) 使核心区近乎完全替换，边缘仍渐变防硬边）。
        if panel_count is not None and panel_count > 1:
            clean_bg = self._reconstruct_paneled_screen(img_bgr, dil_fg, panel_count, painting_roi=painting_roi)
        elif self.inpainting_provider is not None:
            clean_bg = self.inpainting_provider.inpaint(img_bgr, dil_fg)
        else:
            clean_bg = self._reconstruct_continuous_canvas(img_bgr, dil_fg, pure_bg_mask)

        return clean_bg, pure_bg_mask

    def _reconstruct_continuous_canvas(self, img_bgr, dil_fg, pure_bg_mask):
        """Reconstructs continuous canvas (xuan paper, silk, or oil canvas)."""
        h, w, _ = img_bgr.shape
        
        # Multi-scale smooth illumination field via downsampled normalized inpainting
        scale = min(0.25, 1024.0 / max(h, w))
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        
        s_src = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        s_mask = cv2.resize(dil_fg, (sw, sh), interpolation=cv2.INTER_NEAREST)
        
        s_inpaint = cv2.inpaint(s_src, s_mask, 7, cv2.INPAINT_TELEA)
        s_inpaint = cv2.inpaint(s_inpaint, s_mask, 7, cv2.INPAINT_NS)
        
        smooth_field = cv2.resize(s_inpaint, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Extract support texture (grain/weave) from the most pristine background patch
        pts = cv2.findNonZero(pure_bg_mask.astype(np.uint8))
        if pts is not None:
            # Find largest connected component of background
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pure_bg_mask.astype(np.uint8))
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
            largest_bbox = stats[largest_label]
            bx, by, bw, bh = largest_bbox[cv2.CC_STAT_LEFT], largest_bbox[cv2.CC_STAT_TOP], largest_bbox[cv2.CC_STAT_WIDTH], largest_bbox[cv2.CC_STAT_HEIGHT]
            
            # Sample texture tile
            tw = min(bw, 400)
            th = min(bh, 400)
            if tw > 30 and th > 30:
                sample_patch = img_bgr[by:by+th, bx:bx+tw]
                patch_blur = cv2.GaussianBlur(sample_patch, (21, 21), 5.0)
                tex_highfreq = cv2.subtract(sample_patch, patch_blur).astype(np.float32)
                
                # Tile texture seamlessly
                tiled_tex = np.zeros_like(img_bgr, dtype=np.float32)
                for y in range(0, h, th):
                    for x in range(0, w, tw):
                        ch = min(th, h - y)
                        cw = min(tw, w - x)
                        tiled_tex[y:y+ch, x:x+cw] = tex_highfreq[:ch, :cw]
            else:
                tiled_tex = np.zeros_like(img_bgr, dtype=np.float32)
        else:
            tiled_tex = np.zeros_like(img_bgr, dtype=np.float32)

        alpha = cv2.GaussianBlur(dil_fg.astype(np.float32) / 255.0, (31, 31), 10.0)[:, :, None]
        clean_bg = img_bgr.astype(np.float32) * (1.0 - alpha) + (smooth_field.astype(np.float32) + tiled_tex * 0.6) * alpha
        return np.clip(clean_bg, 0, 255).astype(np.uint8)

    def _reconstruct_paneled_screen(self, img_bgr, dil_fg, panel_count, painting_roi=None):
        """Reconstructs Japanese/Chinese paneled folding screen with authentic foil/silk grid."""
        h, w, _ = img_bgr.shape
        
        # 1. Determine active inner canvas ROI (excluding outer mounting brocade and photography backdrop)
        if painting_roi is not None:
            pts = cv2.findNonZero(painting_roi.astype(np.uint8))
            if pts is not None:
                rx, ry, rw, rh = cv2.boundingRect(pts)
                inner_l, inner_t, inner_r, inner_b = rx, ry, rx + rw, ry + rh
            else:
                inner_l, inner_t, inner_r, inner_b = 0, 0, w, h
        else:
            # Dynamic detection of painting canvas boundaries
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            row_m = gray.mean(axis=1)
            col_m = gray.mean(axis=0)
            is_inner_y = np.abs(row_m - np.median(row_m[:50])) > 8.0
            is_inner_x = np.abs(col_m - np.median(col_m[:50])) > 8.0
            y_idx = np.where(is_inner_y)[0]
            x_idx = np.where(is_inner_x)[0]
            if len(y_idx) > 50 and len(x_idx) > 50:
                inner_t = max(0, int(y_idx[0]))
                inner_b = min(h, int(y_idx[-1]))
                inner_l = max(0, int(x_idx[0]))
                inner_r = min(w, int(x_idx[-1]))
            else:
                inner_l, inner_t, inner_r, inner_b = 0, 0, w, h
        
        w_active = inner_r - inner_l
        w_panel = w_active / float(panel_count)
        
        # Find the panel with the least foreground ink within active painting area
        best_panel_idx = 0
        min_ink = float('inf')
        for i in range(panel_count):
            x0 = int(inner_l + i * w_panel)
            x1 = int(inner_l + (i + 1) * w_panel)
            ink_count = np.count_nonzero(dil_fg[inner_t:inner_b, x0:x1])
            if ink_count < min_ink:
                min_ink = ink_count
                best_panel_idx = i
                
        ref_x0 = int(inner_l + best_panel_idx * w_panel)
        ref_x1 = int(inner_l + (best_panel_idx + 1) * w_panel)
        ref_panel = img_bgr[inner_t:inner_b, ref_x0:ref_x1].copy()

        # 2026-09-12 审计修复：参考扇是「墨最少」而非「零墨」——它自带的墨迹会被
        # 复制到其余各扇（石矶区残留的主因之一）。先清理参考扇自身墨迹：
        # 按列用非墨像素的均值填充（保留金箔纵向纹理），整列皆墨时用全扇净色。
        ref_ink = dil_fg[inner_t:inner_b, ref_x0:ref_x1] > 0
        if ref_ink.any():
            global_clean = np.median(ref_panel[~ref_ink], axis=0) if (~ref_ink).any() else np.array([200, 200, 200], np.uint8)
            for x in range(ref_panel.shape[1]):
                col_ink = ref_ink[:, x]
                if not col_ink.any():
                    continue
                if (~col_ink).any():
                    ref_panel[col_ink, x] = ref_panel[~col_ink, x].mean(axis=0)
                else:
                    ref_panel[col_ink, x] = global_clean

        clean_bg = img_bgr.copy()

        # 2026-09-12 审计修复：参考扇（墨最少，但非零墨）自身的墨迹必须**写回**
        # clean_bg —— 原实现只在复制源 ref_panel 上清理，而 clean_bg 中参考扇那一片
        # 仍是原图（含墨）→ 雁群/岩石在该扇上残留（视觉实测确认）。
        clean_bg[inner_t:inner_b, ref_x0:ref_x1] = ref_panel
        
        # Define sky sampling vertical range within painting ROI
        sky_t = inner_t + int((inner_b - inner_t) * 0.05)
        sky_b = inner_t + int((inner_b - inner_t) * 0.20)
        
        pad_x = max(5, int(w_panel * 0.05))
        top_ref = img_bgr[sky_t:sky_b, ref_x0+pad_x:ref_x1-pad_x]
        mean_ref = np.mean(top_ref, axis=(0, 1)) if top_ref.size > 0 else np.array([0, 0, 0])

        for i in range(panel_count):
            if i == best_panel_idx:
                continue
            px0 = int(inner_l + i * w_panel)
            px1 = int(inner_l + (i + 1) * w_panel)
            pw = px1 - px0
            
            p_slice = cv2.resize(ref_panel, (pw, inner_b - inner_t))
            
            top_curr = img_bgr[sky_t:sky_b, px0+pad_x:px1-pad_x]
            if top_curr.size > 0:
                diff = np.mean(top_curr, axis=(0, 1)) - mean_ref
                diff = np.clip(diff, -25.0, 25.0)
            else:
                diff = np.array([0, 0, 0])
                
            p_adj = np.clip(p_slice.astype(np.float32) + diff, 0, 255).astype(np.uint8)
            
            p_mask = dil_fg[inner_t:inner_b, px0:px1]
            if np.count_nonzero(p_mask) == 0:
                continue

            # 2026-09-12 审计修复：原实现只用高斯模糊 alpha（墨迹核心 alpha 仅约
            # 0.6）→ 墨迹中心残留（石矶区实测 9.7% 暗像素）。现叠加二值核心：
            # max(模糊渐变, 二值×0.95) —— 核心近乎完全替换，边缘仍渐变防硬边。
            alpha_blur = cv2.GaussianBlur(p_mask.astype(np.float32) / 255.0, (25, 25), 8)
            alpha_core = (p_mask.astype(np.float32) / 255.0) * 0.95
            alpha_p = np.maximum(alpha_blur, alpha_core)[:, :, None]

            curr_slice = img_bgr[inner_t:inner_b, px0:px1]
            blended = curr_slice.astype(np.float32) * (1.0 - alpha_p) + p_adj.astype(np.float32) * alpha_p
            clean_bg[inner_t:inner_b, px0:px1] = np.clip(blended, 0, 255).astype(np.uint8)

        return clean_bg
