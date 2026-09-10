import cv2
import numpy as np

class LightingManager:
    """Provides illumination normalization, flat-fielding, and drop shadow suppression."""

    @staticmethod
    def estimate_illumination_gradient(image: np.ndarray, sigma: float = 60.0) -> np.ndarray:
        """
        Estimates non-uniform lighting gradient across the photograph using large-kernel Gaussian filter.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return illum

    @staticmethod
    def normalize_illumination(image: np.ndarray, target_mean: float = 230.0) -> np.ndarray:
        """
        Normalizes warm/cold lighting drift across the image to prevent color clustering threshold overflow.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        illum = cv2.GaussianBlur(L, (0, 0), sigmaX=50.0)
        norm_L = np.clip(L - illum + target_mean, 0, 255)
        lab[:, :, 0] = norm_L
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
