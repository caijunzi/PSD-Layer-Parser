import numpy as np
import cv2
from PIL import Image
from typing import Tuple

class ColorManager:
    """Handles professional color conversions, CMYK mapping, TAC control, and pytoshop raw inversion."""

    @staticmethod
    def bgr_to_cmyk_raw(bgr_image: np.ndarray) -> np.ndarray:
        """
        Converts BGR image directly to pytoshop raw CMYK channels (4, H, W) with zero synthetic formula.
        In pytoshop raw CMYK: 255 = 0% ink (blank paper), 0 = 100% ink.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        cmyk_pil = pil_img.convert('CMYK')
        cmyk_np = np.array(cmyk_pil)  # shape (H, W, 4), 0..255 where 255 is 100% ink

        raw_c = 255 - cmyk_np[:, :, 0]
        raw_m = 255 - cmyk_np[:, :, 1]
        raw_y = 255 - cmyk_np[:, :, 2]
        raw_k = 255 - cmyk_np[:, :, 3]

        return np.stack([raw_c, raw_m, raw_y, raw_k], axis=0)

    @staticmethod
    def get_ivory_substrate_cmyk_raw(height: int, width: int,
                                     c_pct: float = 2.0, m_pct: float = 3.0,
                                     y_pct: float = 8.0, k_pct: float = 0.0) -> np.ndarray:
        """
        Generates calibrated luxury ivory base substrate channels in pytoshop raw format.
        """
        raw_c = int(round(255 - (c_pct / 100.0) * 255))
        raw_m = int(round(255 - (m_pct / 100.0) * 255))
        raw_y = int(round(255 - (y_pct / 100.0) * 255))
        raw_k = int(round(255 - (k_pct / 100.0) * 255))

        c = np.full((height, width), raw_c, dtype=np.uint8)
        m = np.full((height, width), raw_m, dtype=np.uint8)
        y = np.full((height, width), raw_y, dtype=np.uint8)
        k = np.full((height, width), raw_k, dtype=np.uint8)

        return np.stack([c, m, y, k], axis=0)

    @staticmethod
    def cmyk_raw_to_bgr_preview(cmyk_raw_channels: np.ndarray, max_dim: int = 1600) -> np.ndarray:
        """
        Converts pytoshop raw CMYK channels (4, H, W) to a high-quality BGR preview image.
        """
        c, m, y, k = cmyk_raw_channels[0], cmyk_raw_channels[1], cmyk_raw_channels[2], cmyk_raw_channels[3]
        h, w = c.shape

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)
            k = cv2.resize(k, (nw, nh), interpolation=cv2.INTER_AREA)

        ink_c = 255 - c
        ink_m = 255 - m
        ink_y = 255 - y
        ink_k = 255 - k

        cmyk_stack = np.stack([ink_c, ink_m, ink_y, ink_k], axis=2)
        rgb = np.array(Image.fromarray(cmyk_stack, mode='CMYK').convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def calculate_tac(cmyk_raw_channels: np.ndarray) -> Tuple[float, float]:
        """
        Calculates Total Area Coverage (TAC) percentage (0% ~ 400%).
        Returns: (max_tac_pct, mean_tac_pct)
        """
        c, m, y, k = cmyk_raw_channels[0], cmyk_raw_channels[1], cmyk_raw_channels[2], cmyk_raw_channels[3]
        tac = ((255 - c).astype(float) + (255 - m).astype(float) + (255 - y).astype(float) + (255 - k).astype(float)) / 255.0 * 100.0
        return float(np.max(tac)), float(np.mean(tac))
