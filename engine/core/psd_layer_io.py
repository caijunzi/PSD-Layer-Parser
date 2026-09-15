# -*- coding: utf-8 -*-
"""PSD/PSB 图层像素的**权威读取入口**（2026-09-16 根因修复）。

⚠️ 为什么需要这个模块
---------------------
psd_tools 的 `layer.numpy()` 通道数随色彩模式变化：

  - RGB 层  → (H, W, 4) = R, G, B, **Alpha**   → Alpha 在 **index 3**
  - CMYK 层 → (H, W, 5) = C, M, Y, K, **Alpha** → Alpha 在 **index 4**

此前散落在 `tools/audit_psb.py`、`tests/audit_system_integrity.py` 等处的
`lyr.numpy()[:, :, 3]` / `lyr.numpy()[:, :, :3]`，在 CMYK（PLATE 线）产物上
分别取到 **K 通道** / 把 **C,M,Y 当成 R,G,B**，造成掩码与比对基准系统性错乱
（②③④⑥ 四维审计全错、防欺骗审计整体失效）。

本模块集中所有读取逻辑，统一按「该层**实际通道数**」判断，杜绝散落各处的索引假设。
pytoshop 在写出时强制整份 PSD 同一色彩模式，故真实产物要么全 RGB(4 通道)、
要么全 CMYK(5 通道)；读取端只要逐层按 `shape[2]` 判定即可正确，无需假设全局模式。

两个函数都同时接受「层对象（含 `.numpy()`）」与「ndarray」两种入参，
方便既有调用点（如 `_alpha(ly.numpy())`）无缝迁移。
"""
from __future__ import annotations

import numpy as np


def _as_array(lyr) -> np.ndarray:
    """接受层对象或 ndarray，统一返回 ndarray。"""
    return lyr.numpy() if hasattr(lyr, "numpy") else lyr


def layer_alpha(lyr) -> np.ndarray:
    """取图层 Alpha（透明）通道，返回 0..1 浮点、bbox 尺寸数组。

    - RGB 层（4 通道）：取 index 3
    - CMYK 层（5 通道）：取 index **4**
    - 通道不足 4：视为完全不透明（返回全 1）
    - psd_tools `numpy()` 默认返回 0..1 浮点；若返回 0..255 则归一化到 0..1
    """
    a = _as_array(lyr)
    if a.ndim != 3 or a.shape[2] < 4:
        return np.ones(a.shape[:2], np.float32)
    idx = 4 if a.shape[2] >= 5 else 3
    al = a[:, :, idx].astype(np.float32)
    return al / 255.0 if al.max() > 1.001 else al


def layer_rgb(lyr) -> np.ndarray:
    """取图层像素归一为 0..255 的 RGB 数组（bbox 尺寸）。

    psd_tools 对 CMYK 层返回的是**呈色**（= 1 - 墨量），实测象牙底板
    C=.961 M=.910 Y=.829 K=1.000（K=1.0 表示 0% 黑版）。转 RGB 用乘法还原：
        R = (1-墨量C) * (1-墨量K) = v0 * v3
        G = v1 * v3,  B = v2 * v3
    对 RGB 层直接取前 3 通道。

    ⚠️ 这是**无 ICC 的近似转换**，仅用于「底板 vs 源图」的粗判（阈值 >40 的差异定位），
    不用于 ⑤ 的合成等价性（那里必须走同一 ICC 分色）。返回的数组仅覆盖图层 bbox，
    调用方如需整画布请自行补边。
    """
    a = _as_array(lyr)
    if a.ndim != 3 or a.shape[2] < 3:
        return np.zeros(a.shape[:2] + (3,), np.float32)
    if a.shape[2] >= 5:
        cmyk = a[..., :4].astype(np.float32)
        k_keep = cmyk[..., 3]
        return np.stack([cmyk[..., 0] * k_keep,
                         cmyk[..., 1] * k_keep,
                         cmyk[..., 2] * k_keep], -1) * 255.0
    rgb = a[..., :3].astype(np.float32)
    return rgb * 255.0 if rgb.max() <= 1.001 else rgb


def full_alpha_mask(lyr, H: int, W: int, threshold: float = 0.03) -> np.ndarray:
    """把图层 Alpha 二值掩码还原到整画幅 (H, W) 的布尔数组（按 bbox 放置）。"""
    m = layer_alpha(lyr) > threshold
    f = np.zeros((H, W), bool)
    x0, y0 = lyr.bbox[0], lyr.bbox[1]
    h, w = m.shape
    f[y0:y0 + h, x0:x0 + w] = m[:max(0, H - y0), :max(0, W - x0)]
    return f
