"""Universal 2.5D De-occlusion Engine.
Repairs and completes occluded background layers when foreground elements overlap.
Ensures that hiding foreground layers in Photoshop reveals continuous, seamless background structures.
"""
import cv2
import numpy as np

class UniversalDeoccluder:
    def __init__(self, inpaint_radius=7, extension_pixels=20, inpainting_provider=None):
        self.inpaint_radius = inpaint_radius
        self.extension_pixels = extension_pixels
        self.inpainting_provider = inpainting_provider

    def deocclude_layers(self, img_bgr, sorted_layers):
        """
        Deoccludes background layers by extending layer boundaries under overlapping
        foreground objects (trees, architecture, figures) by 15-30 pixels.
        Ensures hiding or slightly shifting foreground elements reveals seamless,
        natural underlying structures without gaping silhouettes or inventing fake masses.
        
        Args:
            img_bgr: Full color source image (BGR)
            sorted_layers: List of layer dicts sorted from UI Top to UI Bottom.
            
        Returns:
            updated_layers: List of layer dicts with extended masks and seamless inpainted color images.
        """
        h, w, _ = img_bgr.shape
        accumulated_foreground = np.zeros((h, w), dtype=np.uint8)

        # Structuring element for controlled 15-30px extension
        ext_ksize = max(5, (self.extension_pixels // 2) * 2 + 1)
        kernel_ext = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ext_ksize, ext_ksize))

        for i, layer in enumerate(sorted_layers):
            layer_mask = layer["mask"].copy()
            name = layer["name"].lower()

            # Background base canvas is handled separately
            if "base" in name or "gold" in name:
                continue

            # Overlay framing/seams/signatures do not cast pictorial occlusion
            if "frame" in name or "seam" in name or "seal" in name or "callig" in name:
                continue

            # For background structures (Mountains, Water, Ground), check if foreground objects rest on them
            if np.count_nonzero(accumulated_foreground) > 0:
                # Find boundary where this layer touches the foreground footprint
                dilated_layer = cv2.dilate(layer_mask, kernel_ext)
                # Occluded extension zone: regions where foreground sits directly adjacent to/over this layer
                extension_zone = cv2.bitwise_and(dilated_layer, accumulated_foreground)
                
                if np.count_nonzero(extension_zone) > 30:
                    # Inpaint the extension zone using the surrounding pixels
                    if self.inpainting_provider is not None:
                        inpainted = self.inpainting_provider.inpaint(img_bgr, extension_zone)
                    else:
                        inpainted = cv2.inpaint(img_bgr, extension_zone, self.inpaint_radius, cv2.INPAINT_NS)
                    layer["inpainted_bgr"] = inpainted
                    # Update layer mask to include the extended support area
                    layer["mask"] = cv2.bitwise_or(layer_mask, extension_zone)
                else:
                    layer["inpainted_bgr"] = None
            else:
                layer["inpainted_bgr"] = None

            # Accumulate this layer into foreground for subsequent deeper layers
            accumulated_foreground = cv2.bitwise_or(accumulated_foreground, layer_mask)

        return sorted_layers
