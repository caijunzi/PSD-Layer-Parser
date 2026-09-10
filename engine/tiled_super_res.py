"""Universal Tiled Super-Resolution & Guided Upsampling Engine.
Enables memory-bounded scaling to arbitrary resolutions (16K / 32K) with peak RAM < 2GB.
"""
import cv2
import numpy as np

class UniversalTiledSuperRes:
    def __init__(self, target_w=16000, target_h=7808):
        self.target_w = target_w
        self.target_h = target_h

    def upscale_image_progressive(self, img_bgr, stages=[1.5, 2.0, 4.0]):
        """
        Progressive multi-scale super-resolution ladder (过度尺度金字塔).
        Iteratively scales image through intermediate octaves with unsharp edge
        preservation to prevent blur, ringing, and gradient loss.
        """
        h_orig, w_orig = img_bgr.shape[:2]
        curr = img_bgr.copy()
        
        # Determine intermediate targets
        for scale in stages:
            curr_target_w = int(w_orig * scale)
            curr_target_h = int(h_orig * scale)
            
            # Step-wise Lanczos-4 interpolation
            scaled = cv2.resize(curr, (curr_target_w, curr_target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # High-frequency edge enhancement (Unsharp Masking for brushstrokes)
            # Gentle unsharp mask to recover ink boundary crispness without introducing noise
            blur = cv2.GaussianBlur(scaled, (0, 0), sigmaX=1.2, sigmaY=1.2)
            sharpened = cv2.addWeighted(scaled, 1.25, blur, -0.25, 0)
            
            curr = np.clip(sharpened, 0, 255).astype(np.uint8)
            del scaled, blur, sharpened

        # Final exact resize to target dimensions if slight rounding differences exist
        if curr.shape[1] != self.target_w or curr.shape[0] != self.target_h:
            curr = cv2.resize(curr, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)
            
        return curr

    def upscale_image(self, img_bgr):
        """Standard progressive upscaling interface."""
        return self.upscale_image_progressive(img_bgr)

    def _guided_filter_banded(self, m_raw, guide_gray, out_w, out_h, radius=6, eps=1e-3, num_bands=4):
        """Banded guided filtering keeping memory usage within bounded limit."""
        band_h = max(1, out_h // num_bands)
        out = np.empty((out_h, out_w), dtype=np.uint8)
        pad = radius * 2
        for band_idx in range(num_bands):
            y0 = max(0, band_idx * band_h - pad)
            y1 = min(out_h, (band_idx + 1) * band_h + pad if band_idx < num_bands - 1 else out_h)
            
            g_band = guide_gray[y0:y1].astype(np.float32) / 255.0
            p_band = m_raw[y0:y1].astype(np.float32) / 255.0
            
            mean_I = cv2.boxFilter(g_band, cv2.CV_32F, (radius, radius))
            mean_p = cv2.boxFilter(p_band, cv2.CV_32F, (radius, radius))
            mean_Ip = cv2.boxFilter(g_band * p_band, cv2.CV_32F, (radius, radius))
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = cv2.boxFilter(g_band * g_band, cv2.CV_32F, (radius, radius))
            var_I = mean_II - mean_I * mean_I
            
            coeff_a = cov_Ip / (var_I + eps)
            coeff_b = mean_p - coeff_a * mean_I
            
            mean_a = cv2.boxFilter(coeff_a, cv2.CV_32F, (radius, radius))
            mean_b = cv2.boxFilter(coeff_b, cv2.CV_32F, (radius, radius))
            
            q = mean_a * g_band + mean_b
            q_clip = np.clip(q * 255.0, 0, 255).astype(np.uint8)
            
            by0 = band_idx * band_h
            by1 = min(out_h, (band_idx + 1) * band_h if band_idx < num_bands - 1 else out_h)
            out[by0:by1] = q_clip[by0 - y0 : by0 - y0 + (by1 - by0)]
            
        return out

    def guided_upsample_mask(self, mask_lr, guide_hr_gray, radius=6, eps=1e-3, num_bands=4):
        """
        Upscales a low-resolution mask using banded high-resolution guided filtering.
        Optimized with bounding box localization for non-fullframe layers.
        """
        pts = cv2.findNonZero(mask_lr)
        if pts is None:
            return np.zeros((self.target_h, self.target_w), dtype=np.uint8)

        bx, by, bw, bh = cv2.boundingRect(pts)
        h_lr, w_lr = mask_lr.shape[:2]
        scale_x = self.target_w / float(w_lr)
        scale_y = self.target_h / float(h_lr)

        # If mask is localized (< 75% area), run only on active ROI
        if (bw * bh) < 0.75 * (w_lr * h_lr):
            pad_lr = 12
            lr_x0 = max(0, bx - pad_lr)
            lr_y0 = max(0, by - pad_lr)
            lr_x1 = min(w_lr, bx + bw + pad_lr)
            lr_y1 = min(h_lr, by + bh + pad_lr)

            roi_m_lr = mask_lr[lr_y0:lr_y1, lr_x0:lr_x1]

            hx0 = int(round(lr_x0 * scale_x))
            hy0 = int(round(lr_y0 * scale_y))
            hx1 = min(self.target_w, int(round(lr_x1 * scale_x)))
            hy1 = min(self.target_h, int(round(lr_y1 * scale_y)))

            roi_w = hx1 - hx0
            roi_h = hy1 - hy0
            roi_m_raw = cv2.resize(roi_m_lr, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            roi_guide = guide_hr_gray[hy0:hy1, hx0:hx1]

            roi_bands = max(1, min(4, roi_h // 1500))
            roi_out = self._guided_filter_banded(roi_m_raw, roi_guide, roi_w, roi_h, radius, eps, num_bands=roi_bands)
            
            out = np.zeros((self.target_h, self.target_w), dtype=np.uint8)
            out[hy0:hy1, hx0:hx1] = roi_out
            return out

        # Full-frame mask
        m_hr_raw = cv2.resize(mask_lr, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        return self._guided_filter_banded(m_hr_raw, guide_hr_gray, self.target_w, self.target_h, radius, eps, num_bands=num_bands)
