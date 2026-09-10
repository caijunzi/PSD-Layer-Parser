"""PsdCompiler 内核编译测试（RGB / CMYK 双线）。

背景：`engine/core/psd_compiler.py` 此前**从未被真正执行过**——生产链路一直用
`engine/psb_builder.py`。首次真跑即暴露两个错误：

    ValueError: Color '0' is not valid for color mode '4', expected '0'

根因：图层通道用整数索引 0/1/2(/3) 标识，而 pytoshop 的 `ColorChannelMapping`
把整数 0 解释为 `ColorChannel.bitmap`，与 RGB/CMYK 模式均不匹配。
正确做法是传 `enums.ColorChannel.red/green/blue` 与
`cyan/magenta/yellow/black` 枚举。

本测试的作用是**守住"内核真的能产文件"这一底线**，避免同类失效再次发生。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.codecs_accelerator import install_psb_codec_accelerator  # noqa: E402

install_psb_codec_accelerator()  # pytoshop 纯 wheel 不含 packbits 扩展

from engine.core.models import LayerDescriptor, ProcessingContext  # noqa: E402
from engine.core.psd_compiler import PsdCompiler  # noqa: E402
from psd_tools import PSDImage  # noqa: E402

W, H = 96, 64


def _ctx(mode: str) -> ProcessingContext:
    ctx = ProcessingContext(ppi=150.0, canvas_px=(W, H))
    ctx.output_mode = mode
    return ctx


class TestDesignRgb(unittest.TestCase):
    def test_rgb_roundtrip(self):
        ctx = _ctx("DESIGN")
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 180, 120, 60, 255
        ctx.layers.append(
            LayerDescriptor(name="01_Test_CMYK", layer_type="element_rgba",
                            rgba=rgba, bbox=(0, 0, W, H))
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "design.psb")
            PsdCompiler.compile_psd(ctx, p, output_mode="DESIGN")
            self.assertTrue(os.path.isfile(p))
            psd = PSDImage.open(p)
            self.assertEqual(psd.size, (W, H))
            self.assertEqual(int(psd.color_mode), 3, "DESIGN 产物应为 RGB(3)")
            self.assertEqual(len(psd), 1)
            # R 通道应接近 180
            arr = psd[0].numpy()
            self.assertAlmostEqual(float(arr[:, :, 0].mean()) * 255, 180, delta=4)

    def test_empty_layers_rejected(self):
        ctx = _ctx("DESIGN")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                PsdCompiler.compile_psd(ctx, os.path.join(d, "x.psb"), output_mode="DESIGN")


class TestPlateCmyk(unittest.TestCase):
    def _cmyk_layer(self, name="01_Test_CMYK"):
        # 逻辑墨量：0 = 0% 墨，255 = 100% 墨
        ch = np.zeros((4, H, W), np.uint8)
        ch[0] = 30    # C
        ch[1] = 60    # M
        ch[2] = 120   # Y
        ch[3] = 0     # K
        return LayerDescriptor(name=name, layer_type="print_cmyk",
                               cmyk_channels=ch, alpha=np.full((H, W), 255, np.uint8),
                               bbox=(0, 0, W, H))

    def test_cmyk_roundtrip(self):
        """PLATE 线必须产出真正的 4 通道 CMYK 文件。"""
        ctx = _ctx("PLATE")
        ctx.layers.append(self._cmyk_layer())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "plate.psb")
            PsdCompiler.compile_psd(ctx, p, output_mode="PLATE")
            self.assertTrue(os.path.isfile(p))
            psd = PSDImage.open(p)
            self.assertEqual(psd.size, (W, H))
            self.assertEqual(int(psd.color_mode), 4, "PLATE 产物应为 CMYK(4)")
            self.assertEqual(psd.channels, 4, "PSB 头应声明 4 个颜色通道")
            self.assertEqual(len(psd), 1)
            self.assertEqual(ctx.qa_metrics.get("output_mode"), "PLATE")

    def test_plate_requires_cmyk_channels(self):
        """PLATE 线图层缺少 cmyk_channels 时必须拒绝写出。"""
        ctx = _ctx("PLATE")
        ctx.layers.append(
            LayerDescriptor(name="01_Bad_CMYK", layer_type="print_cmyk",
                            rgba=np.zeros((H, W, 4), np.uint8), bbox=(0, 0, W, H))
        )
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                PsdCompiler.compile_psd(ctx, os.path.join(d, "bad.psb"), output_mode="PLATE")

    def test_unknown_mode_rejected(self):
        ctx = _ctx("RGB")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                PsdCompiler.compile_psd(ctx, os.path.join(d, "x.psb"), output_mode="RGB")


if __name__ == "__main__":
    unittest.main()
