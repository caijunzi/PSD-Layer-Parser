"""Base Provider Contracts for Universal Neural Layer Studio.
Defines clean abstract interfaces for pluggable AI model backends:
- Segmentation & Matting (Zero-shot / Text-guided / High-res)
- Inpainting (Fast Fourier Convolutions / Telea fallback)
- Super-Resolution (Tiled ESRGAN / Guided pyramid fallback)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class BaseSegmentationProvider(ABC):
    """Abstract interface for object-level semantic segmentation & matting."""
    
    @abstractmethod
    def segment_by_labels(self, img_bgr: np.ndarray, labels: List[str], roi_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Segments distinct artistic subjects given a list of target semantic labels.
        
        Args:
            img_bgr: Source image in BGR format.
            labels: List of semantic target labels (e.g. ['trees', 'mountains', 'pavilion', 'figures', 'seal'])
            roi_mask: Optional active canvas boundary mask.
            
        Returns:
            dict of {label: binary_mask_uint8 (255=foreground)}
        """
        pass

class BaseInpaintingProvider(ABC):
    """Abstract interface for occluded background reconstruction & inpainting."""
    
    @abstractmethod
    def inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        Inpaints the regions marked by mask_u8 (255 = inpaint region).
        
        Args:
            img_bgr: Source image in BGR format.
            mask_u8: Binary inpaint target mask (255 = area to inpaint).
            
        Returns:
            inpainted_bgr: Seamless reconstructed image.
        """
        pass

class BaseSuperResProvider(ABC):
    """Abstract interface for micro-scale texture super-resolution."""
    
    @abstractmethod
    def upscale(self, img_bgr: np.ndarray, scale: float = 4.0) -> np.ndarray:
        """
        Upscales an image while synthesizing authentic micro-textures.
        
        Args:
            img_bgr: Source image in BGR format.
            scale: Scaling multiplier (e.g. 2.0, 4.0).
            
        Returns:
            upscaled_bgr: High-resolution image.
        """
        pass
