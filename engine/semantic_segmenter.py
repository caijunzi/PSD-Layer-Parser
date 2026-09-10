"""Universal Semantic Segmenter.
Provides object-level semantic mask decomposition for classical Chinese/Japanese artworks.
Adheres strictly to traditional painterly layer logic ("一个东西一个图层"):
  - Trees are organic connected wholes (roots, trunks, branches, foliage) without severed limbs.
  - Mountains and cliffs form a continuous geological mass without arbitrary coordinate cuts.
  - Architecture, figures, calligraphy, seals, fauna, and water ripples are independent subject layers.

Strictly delegates to zero-hardcode feature analysis and pluggable neural segmentation providers.
Contains ZERO hardcoded polygon coordinates or image-specific pixel slices.
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional, Any
from engine.providers.segmentation_provider import ZeroHardcodeSegmentationProvider


class UniversalSemanticSegmenter:
    """
    Universal semantic segmenter.
    Delegates to feature-driven or neural segmentation providers.
    Maintains zero hardcoded coordinates across all image types.
    """
    def __init__(self, preset: Optional[str] = "japanese_screen_gold", provider: Optional[Any] = None):
        self.preset = preset
        self.provider = provider or ZeroHardcodeSegmentationProvider()

    def segment_objects(self, img_bgr: np.ndarray, painting_roi: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Decomposes an artwork into distinct object-level semantic masks.
        
        Args:
            img_bgr: Source image in BGR format.
            painting_roi: Optional active canvas boundary mask (uint8 or bool).
            
        Returns:
            dict of {layer_name: mask_uint8 (255=foreground)}
        """
        roi_mask = None
        if painting_roi is not None and isinstance(painting_roi, np.ndarray):
            roi_mask = painting_roi.astype(bool)
            
        return self.provider.segment_by_labels(img_bgr, roi_mask=roi_mask)
