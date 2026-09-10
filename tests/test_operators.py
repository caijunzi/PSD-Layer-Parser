"""
Unit tests for Universal Layer Engine Core Operators.
Rigorous verification with genuine numpy arrays and physical measurements.
ZERO mocks, ZERO fake asserts.
"""

import unittest
import numpy as np
import cv2

from engine.core.models import ProcessingContext, LayerDescriptor
from engine.operators.base import BaseOperator, OperatorResult, get_param
from engine.operators.contour_protection import ContourProtectionOperator
from engine.operators.seam_harmonizer import SeamHarmonizerOperator
from engine.operators.trapping import TrappingOperator
from engine.operators.deocclusion_operator import DeocclusionOperator
from engine.operators.micro_holes import MicroHolesOperator
from engine.operators.metallic_foil import MetallicFoilOperator


class TestCoreOperators(unittest.TestCase):

    def setUp(self):
        # Create standard test context
        self.h, self.w = 200, 200
        # Realistic canvas texture
        np.random.seed(42)
        base_img = np.full((self.h, self.w, 3), 180, dtype=np.uint8)
        # Add slight texture noise
        noise = np.random.randint(-10, 10, (self.h, self.w, 3), dtype=np.int16)
        self.test_bgr = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.ctx = ProcessingContext(
            source_bgr=self.test_bgr.copy(),
            detect_image=self.test_bgr.copy(),
            output_image=self.test_bgr.copy(),
            ppi=150.0,
            canvas_px=(self.w, self.h),
        )

    def test_base_operator_contract(self):
        """Verify get_param boundary clamping and fallback behavior."""
        params = {"alpha": 0.5, "clamp_test": 150}
        self.assertEqual(get_param(params, "alpha", 1.0), 0.5)
        self.assertEqual(get_param(params, "missing", 42), 42)
        self.assertEqual(get_param(params, "clamp_test", 10, min_val=0, max_val=100), 100)

    def test_contour_protection(self):
        """Verify dynamic safe cut line calculation and burr trimming."""
        img = self.test_bgr.copy()
        # Simulate clean pattern from Y=0..150, torn fringes from Y=150..200
        # Background desk: high brightness, low texture
        img[160:, :] = 240

        op = ContourProtectionOperator()
        res = op.run(img, params={"min_safe_bottom_y": 140, "max_search_y": 190, "desk_gray_thresh": 220})

        self.assertTrue(res.success)
        safe_y = res.data["safe_bottom_y"]
        self.assertGreaterEqual(safe_y, 140)
        self.assertLessEqual(safe_y, 160)
        self.assertEqual(res.data["cut_mask"].shape, (self.h, self.w))
        self.assertEqual(res.metrics["cut_y_source"], "contour_protection")
        self.assertGreater(res.metrics["fabric_solid_ratio"], 0.70)

    def test_seam_harmonizer_cyclic(self):
        """Verify vertical cyclic seam gradient smoothing."""
        img = self.test_bgr.copy()
        # Artificially create a sharp step at top vs bottom
        img[:10, :] = 50   # Dark top
        img[-10:, :] = 220 # Bright bottom

        op = SeamHarmonizerOperator()
        res = op.run(img, params={"warp_band_px": 20, "mode": "cyclic_vertical"})

        self.assertTrue(res.success)
        self.assertIn("harmonized_image", res.data)
        metrics = res.metrics
        self.assertGreater(metrics["pre_seam_rmse"], metrics["post_seam_rmse"])
        self.assertGreater(metrics["rmse_reduction_ratio"], 0.0)

    def test_trapping_spread(self):
        """Verify physical spot color trapping calculation."""
        spot_mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a circle in the center
        cv2.circle(spot_mask, (50, 50), 20, 255, -1)

        op = TrappingOperator()
        # At 150 PPI, trap_mm=0.2mm -> round(0.2 / 25.4 * 150) = 1 px
        res_150 = op.run(self.ctx, params={"spot_mask": spot_mask, "ppi": 150.0, "trap_mm": 0.2})
        self.assertTrue(res_150.success)
        self.assertEqual(res_150.metrics["trap_width_px"], 1)
        self.assertGreater(res_150.metrics["trap_band_pixels"], 0)
        self.assertGreater(res_150.metrics["spot_active_pixels"], np.count_nonzero(spot_mask))

        # At 300 PPI, trap_mm=0.4mm -> round(0.4 / 25.4 * 300) = 5 px
        res_300 = op.run(self.ctx, params={"spot_mask": spot_mask, "ppi": 300.0, "trap_mm": 0.4})
        self.assertEqual(res_300.metrics["trap_width_px"], 5)
        self.assertGreater(res_300.metrics["trap_band_pixels"], res_150.metrics["trap_band_pixels"])

    def test_deocclusion_reconstruction_marking(self):
        """Verify explicit reconstruction_mask generation (R1 compliance)."""
        h, w = 150, 150
        img = np.full((h, w, 3), 120, dtype=np.uint8)

        # Background layer (Ground / Mountain)
        bg_mask = np.zeros((h, w), dtype=np.uint8)
        bg_mask[50:120, 20:130] = 255

        # Foreground layer (Tree / Figure) overlapping background
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[40:80, 50:90] = 255

        layers = [
            {"name": "01_Tree_Foreground", "mask": fg_mask, "alpha": fg_mask},
            {"name": "02_Mountain_Background", "mask": bg_mask, "alpha": bg_mask},
        ]

        op = DeocclusionOperator()
        res = op.run(self.ctx, params={"image": img, "layers": layers, "extension_pixels": 15})

        self.assertTrue(res.success)
        self.assertIsNotNone(res.reconstruction_mask)
        recon_px = np.count_nonzero(res.reconstruction_mask)
        self.assertGreater(recon_px, 0)
        self.assertEqual(res.metrics["reconstruction_pixel_count"], recon_px)
        self.assertTrue(res.metrics["controlled"])

    def test_micro_holes_detection(self):
        """Verify DoG micro-holes detection and die-cut plate rendering."""
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        # Add 3 authentic dark hole dots
        cv2.circle(img, (25, 25), 2, (30, 30, 30), -1)
        cv2.circle(img, (50, 50), 2, (30, 30, 30), -1)
        cv2.circle(img, (75, 75), 2, (30, 30, 30), -1)

        op = MicroHolesOperator()
        res = op.run(img, params={"dog_threshold": 3.0, "target_size": (200, 200), "scale_x": 2.0, "scale_y": 2.0})

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.metrics["hole_count"], 3)
        diecut = res.data["diecut_mask"]
        self.assertEqual(diecut.shape, (200, 200))
        # Holes in diecut mask are black (0), background is white (255)
        self.assertLess(np.min(diecut), 255)

    def test_metallic_foil_separation(self):
        """Verify non-overlapping copper foil and gold mask separation."""
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        # Copper-like reddish pixels (High R, low B, high Lab A*)
        # BGR: B=30, G=80, R=220
        img[20:50, 20:50] = [30, 80, 220]
        # Gold-like yellowish pixels (High R, High G, low B)
        # BGR: B=30, G=190, R=210
        img[60:90, 60:90] = [30, 190, 210]

        valid_mask = np.zeros((100, 100), dtype=np.uint8)
        valid_mask[20:50, 20:50] = 255
        valid_mask[60:90, 60:90] = 255

        op = MetallicFoilOperator()
        res = op.run(img, params={"valid_print_mask": valid_mask, "rb_diff_min": 70.0, "rg_diff_min": 30.0, "lab_a_min": 130.0})

        self.assertTrue(res.success)
        copper = res.data["copper_mask"]
        gold = res.data["gold_mask"]

        # Mutual exclusivity constraint
        overlap = cv2.bitwise_and(copper, gold)
        self.assertEqual(np.count_nonzero(overlap), 0, "Copper and Gold masks MUST NOT overlap!")
        self.assertGreater(res.metrics["copper_pixels"], 0)
        self.assertGreater(res.metrics["gold_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
