"""Preset 加载与语义查询的唯一入口（SSOT，2026-09-10）。

问题背景（G2 / 尽调报告 P1-7）
------------------------------
此前 `load_preset()` 定义在入口脚本 `run_universal_engine.py` 里，
导致 `engine/providers` 无法引用（引擎层反向依赖入口脚本，架构倒挂），
provider 只能把自己的品类语义硬编码在代码里：
  - `_map_to_bilingual_names()` 内置 27 条 英文检测名 → 中英对照图层名 映射；
  - `run_universal_engine` Step3 里硬编码折痕图层的 fixed_color / blend_mode / opacity；
  - preset 里的 `layer_hierarchy` 因此沦为死配置（零引用）。

本模块收敛后：preset 加载唯一化，品类语义由 preset 驱动（G2：新增品类不改内核）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

PRESETS_DIR = "presets"


def load_preset(preset_arg: str) -> dict[str, Any]:
    """按名称或路径加载 preset JSON；找不到时回退传统水墨并告警。"""
    if os.path.isfile(preset_arg):
        with open(preset_arg, "r", encoding="utf-8") as f:
            return json.load(f)

    path = os.path.join(PRESETS_DIR, f"{preset_arg}.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    fallback = "traditional_chinese_ink"
    print(f"Warning: Preset '{preset_arg}' not found, falling back to '{fallback}'")
    with open(os.path.join(PRESETS_DIR, f"{fallback}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_name_mapping(preset: dict[str, Any]) -> Optional[dict[str, str]]:
    """取检测原始名 → 中英对照图层名 的映射（无则 None，调用方回退内置默认）。"""
    sem = preset.get("layer_semantics") or {}
    mapping = sem.get("name_mapping")
    return mapping if isinstance(mapping, dict) and mapping else None


def get_layer_attributes(preset: dict[str, Any]) -> list[dict[str, Any]]:
    """取图层属性规则列表（关键词匹配 → fixed_color / blend_mode / opacity）。"""
    sem = preset.get("layer_semantics") or {}
    rules = sem.get("layer_attributes")
    return rules if isinstance(rules, list) else []


def match_layer_attributes(preset: dict[str, Any], layer_name: str) -> dict[str, Any]:
    """按图层名匹配属性规则，返回应套用的属性子集（空 dict 表示不套用）。

    匹配为大小写不敏感的子串匹配（兼容中英文关键词）。
    """
    low = str(layer_name).lower()
    applied: dict[str, Any] = {}
    for rule in get_layer_attributes(preset):
        kws = rule.get("match") or []
        if any(str(k).lower() in low for k in kws):
            for key in ("fixed_color", "blend_mode", "opacity"):
                if key in rule:
                    applied[key] = rule[key]
    return applied
