"""新增回归测试（2026-09-15 P0–P2 修复配套）。

覆盖：
- manifest.save(mask_dir) 真实落盘重建掩码（修复空壳）
- 重建掩码字段（size / pixel_count / ratio）与 relpath 一致性
- preset Pydantic 校验层（非破坏 + strict 升级）
"""
import json
import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.schemas.manifest import (
    DeliverableManifest,
    LayerGenerationRecord,
    build_manifest,
)
from engine.schemas.preset_schema import validate_preset


class TestManifestMaskWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_writes_masks_and_paths(self):
        """save(mask_dir) 必须真正写出掩码 PNG，并回填 recon_mask_path。"""
        man = build_manifest(
            run_id="r1", output_mode="design",
            source_path="s.png", source_wh=(100, 50),
            output_path="o.psb", output_wh=(400, 200),
            ppi=150.0, color_mode="rgb",
        )
        rec = LayerGenerationRecord(name="04A_峭壁")
        rec.bbox = (0, 0, 10, 10)
        rec.add_reason("deocclusion(lama)")
        man.layers.append(rec)

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255  # 9 px
        man.attach_recon_mask("04A_峭壁", mask)

        out_dir = os.path.join(self.tmp, "out")
        mask_dir = os.path.join(out_dir, "res.masks")
        mpath = os.path.join(out_dir, "res.manifest.json")
        man.save(mpath, mask_dir=mask_dir)

        # 掩码文件存在
        written = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
        self.assertEqual(len(written), 1)
        # 字段回填
        r = man.layer("04A_峭壁")
        self.assertEqual(r.recon_pixel_count, 9)
        self.assertAlmostEqual(r.recon_ratio, 0.09, places=6)
        self.assertEqual(r.recon_mask_size, (10, 10))
        self.assertTrue(r.recon_mask_path.endswith(".png"))
        # relpath 应指向实际存在的文件
        full = os.path.join(out_dir, r.recon_mask_path)
        self.assertTrue(os.path.isfile(full), f"recon_mask_path 不可达: {full}")
        # manifest JSON 中不含 _recon_masks（不应序列化 ndarray）
        data = json.loads(Path(mpath).read_text(encoding="utf-8"))
        self.assertNotIn("_recon_masks", data)

    def test_save_without_mask_dir_ok(self):
        man = build_manifest(
            run_id="r2", output_mode="plate",
            source_path="s.png", source_wh=(100, 50),
            output_path="o.psb", output_wh=(400, 200),
            ppi=150.0, color_mode="cmyk",
        )
        mpath = os.path.join(self.tmp, "m.json")
        man.save(mpath, mask_dir=None)
        self.assertTrue(os.path.isfile(mpath))


class TestPresetSchema(unittest.TestCase):
    def test_valid_preset_passes(self):
        preset = {
            "preset_name": "x", "mode": "hybrid", "super_res_scale": 4.0,
            "ai_semantic_classes": [{"name": "04A_x", "prompt": "cliff", "region": [0, 0, 1, 1]}],
            "density_refine": {"enabled": True, "classes": {"04A_x": {"floor": 0.12}}},
            "plate_operators": {"trapping": {"enabled": True, "spot_layer_name": "spot"}},
        }
        self.assertEqual(validate_preset(preset), [])

    def test_bad_mode_and_regions_reported(self):
        preset = {
            "mode": "weird",
            "super_res_scale": -1,
            "ai_semantic_classes": [{"name": "a"}],  # 缺 prompt
            "density_refine": {"classes": {"a": {}}},  # 缺 floor
        }
        issues = validate_preset(preset)
        self.assertTrue(any("mode" in i for i in issues))
        self.assertTrue(any("super_res_scale" in i for i in issues))
        self.assertTrue(any("prompt" in i for i in issues))
        self.assertTrue(any("floor" in i for i in issues))

    def test_real_presets_non_blocking(self):
        """现有 preset 只告警不抛错（load_preset 非破坏）。

        ⚠️ 必须断言真的扫到了 preset —— 否则目录为空/路径错时本测试会**真空通过**。
        """
        from engine.schemas.presets import load_preset

        presets_dir = ENGINE_ROOT / "presets"
        files = sorted(presets_dir.glob("*.json"))
        self.assertGreaterEqual(len(files), 5,
                                f"presets 目录疑似为空/路径错误（仅找到 {len(files)} 个）——本测试将变成真空通过")
        for p in files:
            data = json.loads(p.read_text(encoding="utf-8"))
            # 不应抛异常
            validate_preset(data)
            # load_preset 也不应抛（非 strict）
            os.environ.pop("ULS_PRESET_STRICT", None)
            load_preset(str(p))


if __name__ == "__main__":
    unittest.main()
