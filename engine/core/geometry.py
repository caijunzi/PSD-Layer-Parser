import cv2
import numpy as np
from typing import Tuple, Optional

class GeometryManager:
    """Provides orthogonal rectification, perspective transform, and safe rectangular boundary trimming."""

    @staticmethod
    def warp_quadrilateral(image: np.ndarray, src_corners: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Warps a 4-point quadrilateral into an exact orthogonal rectangle of (target_w, target_h).
        src_corners: float32 array of shape (4, 2) in order [top-left, top-right, bottom-right, bottom-left].
        """
        tw, th = target_size
        dst_corners = np.float32([
            [0, 0],
            [tw, 0],
            [tw, th],
            [0, th]
        ])
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        return cv2.warpPerspective(image, M, (tw, th), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def find_safe_horizontal_cut(image: np.ndarray, min_y: int, max_y: int, max_tear_ratio: float = 0.0) -> int:
        """
        Finds the lowest horizontal cut line Y within [min_y, max_y] where the fabric is 100% solid,
        ensuring zero torn fringe, zero desk background, and full preservation of the main pattern loops.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_smooth = cv2.GaussianBlur(lap, (7, 7), 0)

        h, w = image.shape[:2]
        # Desk background in photographic swatches has low texture energy, high brightness, and cool color
        r = image[:, :, 2].astype(int)
        b = image[:, :, 0].astype(int)
        is_desk = (lap_smooth < 4.5) & (gray > 212) & (r - b < 18)

        safe_y = min_y
        for y in range(min_y, min(max_y, h)):
            tear_count = np.sum(is_desk[y, :])
            if tear_count / float(w) <= max_tear_ratio:
                safe_y = y
            else:
                break
        return safe_y
