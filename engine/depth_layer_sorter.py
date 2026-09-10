"""Universal Depth & Layer Sorter.
Computes 2.5D visual depth and establishes the correct physical layer hierarchy
for arbitrary artworks without manual configuration.
"""
import cv2
import numpy as np

# Layer stacking priority categories (Higher = Top of UI Stack)
SEMANTIC_Z_ORDER = {
    "Frame": 100,         # Mounting, brocade borders, black lacquer frames
    "Seams": 95,          # Panel fold seams (Multiply)
    "Seal": 90,           # Artist cinnabar vermilion seals
    "Calligraphy": 85,    # Inscriptions, signatures, poems
    "Fauna": 75,          # Birds, animals, flying geese
    "Figures": 70,        # Human figures, scholars, attendants
    "Architecture": 60,   # Buildings, pavilions, houses, bridges
    "Trees": 50,          # Tree branches, trunks, foliage
    "Cliffs": 40,         # Mountain cliffs, boulders, hills
    "Distant_Mountain": 35,# Soft wash peaks, distant horizon
    "Shorelines": 30,     # Sandspits, banks, ground paths
    "Water": 20,          # River, lake, ripples, waves
    "Background": 0       # Clean paper, gold ground, sky, canvas
}

class UniversalLayerSorter:
    def __init__(self, mode="semantic_and_depth"):
        self.mode = mode

    def sort_layers(self, masks_dict, img_bgr=None):
        """
        Sorts layer masks from Top (UI Top / Foreground) to Bottom (UI Bottom / Background).
        
        Args:
            masks_dict: dict of {name: mask_uint8}
            img_bgr: optional source image to compute relative vertical perspective
            
        Returns:
            list of dict: [
                {"name": str, "mask": np.ndarray, "blend_mode": str, "opacity": int, "z_index": float}
            ] (Ordered from UI Top to UI Bottom)
        """
        layer_items = []

        for name, mask in masks_dict.items():
            pts = cv2.findNonZero(mask)
            if pts is None:
                continue

            # 1. Determine semantic category base Z
            base_z = 50.0
            blend_mode = "NORMAL"
            opacity = 255

            # Keyword matching
            name_lower = name.lower()
            if "frame" in name_lower or "brocade" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Frame"]
            elif "seam" in name_lower or "fold" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Seams"]
                blend_mode = "MULTIPLY"
                opacity = 190
            elif "seal" in name_lower or "stamp" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Seal"]
            elif "callig" in name_lower or "text" in name_lower or "inscript" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Calligraphy"]
            elif "fauna" in name_lower or "geese" in name_lower or "bird" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Fauna"]
            elif "fig" in name_lower or "person" in name_lower or "scholar" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Figures"]
            elif "pavilion" in name_lower or "arch" in name_lower or "house" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Architecture"]
            elif "tree" in name_lower or "branch" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Trees"]
            elif "distant" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Distant_Mountain"]
            elif "cliff" in name_lower or "rock" in name_lower or "mountain" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Cliffs"]
            elif "shore" in name_lower or "spit" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Shorelines"]
            elif "water" in name_lower or "ripple" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Water"]
            elif "gold" in name_lower or "base" in name_lower or "bg" in name_lower or "canvas" in name_lower:
                base_z = SEMANTIC_Z_ORDER["Background"]

            # 2. Add subtle perspective bias based on vertical centroid (Near-bottom is closer)
            # In classical landscape, lower Y is usually foreground
            m = cv2.moments(mask)
            if m["m00"] > 0:
                cy = m["m01"] / m["m00"]
                h = mask.shape[0]
                # Modulate +/- 3 units by vertical screen position
                perspective_bias = (cy / float(h)) * 3.0
            else:
                perspective_bias = 0.0

            final_z = base_z + perspective_bias

            layer_items.append({
                "name": name,
                "mask": mask,
                "blend_mode": blend_mode,
                "opacity": opacity,
                "z_index": final_z
            })

        # Sort descending by z_index: Highest Z first (UI Top), Lowest Z last (UI Bottom)
        layer_items.sort(key=lambda x: x["z_index"], reverse=True)
        return layer_items
