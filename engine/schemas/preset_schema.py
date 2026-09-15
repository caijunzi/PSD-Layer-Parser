"""Preset 校验层（ADR-005，2026-09-15 落地）。

背景
----
此前 `engine/schemas/presets.py` 只是 `json.load` + `.get()`，preset 是裸 JSON，
字段无类型/范围校验（尽调 P1-7 / 计划 V1-06 均指出该缺口）。非法
`density_refine.classes`、缺失 `layer_semantics` 等错误只会在运行深处才暴露。

设计原则（非破坏）
------------------
- **校验默认只告警，不拒绝**：现有 6 个 preset 存在 schema 差异（如
  `chinese_ink_landscape_ai` 用 `layer_name/label_cn`），强行拒绝会直接打断流水线。
- 若安装了 pydantic，用其做类型/范围校验；否则退回等价的轻量手工校验（零硬依赖）。
- 环境变量 `ULS_PRESET_STRICT=1` 时把"告警"升级为"异常"，供 CI/交付前门禁用。

返回
----
`validate_preset(preset) -> list[str]`，元素为人类可读的问题描述（空列表 = 通过）。
"""

from __future__ import annotations

from typing import Any

_ALLOWED_MODES = {"locked", "auto", "hybrid"}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_ai_semantic_classes(preset: dict, issues: list[str]) -> None:
    classes = preset.get("ai_semantic_classes")
    if classes is None:
        return
    if not isinstance(classes, list):
        issues.append("ai_semantic_classes 必须是列表")
        return
    for i, c in enumerate(classes):
        if not isinstance(c, dict):
            issues.append(f"ai_semantic_classes[{i}] 必须是对象")
            continue
        # 名称：name 或 layer_name 至少其一
        if not (c.get("name") or c.get("layer_name")):
            issues.append(f"ai_semantic_classes[{i}] 缺少 name/layer_name")
        # prompt 建议存在（缺失则无法检测）
        if not c.get("prompt"):
            issues.append(f"ai_semantic_classes[{i}]（{c.get('name') or c.get('layer_name')}）缺少 prompt")
        # region 归一化范围校验
        for rk in ("region", "regions"):
            reg = c.get(rk)
            if reg is None:
                continue
            regions = reg if isinstance(reg, list) and reg and isinstance(reg[0], (list, tuple)) else [reg]
            for r in regions:
                if not (isinstance(r, (list, tuple)) and len(r) == 4 and all(_is_num(x) for x in r)):
                    issues.append(f"ai_semantic_classes[{i}].{rk} 必须是 4 元归一化数值")
                elif not all(0.0 <= float(x) <= 1.0 for x in r):
                    issues.append(f"ai_semantic_classes[{i}].{rk} 取值应在 [0,1]")


def _check_density_refine(preset: dict, issues: list[str]) -> None:
    dr = preset.get("density_refine")
    if dr is None:
        return
    if not isinstance(dr, dict):
        issues.append("density_refine 必须是对象")
        return
    classes = dr.get("classes")
    if classes is not None and not isinstance(classes, dict):
        issues.append("density_refine.classes 必须是 对象(图层名→{floor})")
    elif isinstance(classes, dict):
        for name, cfg in classes.items():
            if not isinstance(cfg, dict) or not _is_num(cfg.get("floor")):
                issues.append(f"density_refine.classes['{name}'] 缺少数值 floor")


def _check_plate_operators(preset: dict, issues: list[str]) -> None:
    po = preset.get("plate_operators")
    if po is None:
        return
    if not isinstance(po, dict):
        issues.append("plate_operators 必须是对象")
        return
    for op in ("contour_protection", "micro_holes", "trapping"):
        cfg = po.get(op)
        if cfg is None:
            continue
        if not isinstance(cfg, dict):
            issues.append(f"plate_operators.{op} 必须是对象")
            continue
        if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
            issues.append(f"plate_operators.{op}.enabled 必须是布尔")
        if op == "trapping" and cfg.get("enabled") and not cfg.get("spot_layer_name"):
            issues.append("plate_operators.trapping 启用时须提供 spot_layer_name")


def validate_preset(preset: dict[str, Any]) -> list[str]:
    """校验 preset，返回问题列表（空 = 通过）。不抛异常（除非 caller 选择）。"""
    issues: list[str] = []

    if not isinstance(preset, dict):
        return ["preset 根必须是对象"]

    # 顶层类型/枚举
    for key in ("super_res_scale", "dpi"):
        if key in preset and not _is_num(preset[key]):
            issues.append(f"{key} 必须是数值")
    if preset.get("super_res_scale") is not None and _is_num(preset.get("super_res_scale")):
        if float(preset["super_res_scale"]) <= 0:
            issues.append("super_res_scale 必须 > 0")
    if "mode" in preset and preset["mode"] not in _ALLOWED_MODES:
        issues.append(f"mode 必须是 {sorted(_ALLOWED_MODES)} 之一，实际 {preset['mode']!r}")
    if "auto_evolve" in preset and not isinstance(preset["auto_evolve"], bool):
        issues.append("auto_evolve 必须是布尔")

    # layer_semantics
    sem = preset.get("layer_semantics")
    if sem is not None and not isinstance(sem, dict):
        issues.append("layer_semantics 必须是对象")
    elif isinstance(sem, dict):
        nm = sem.get("name_mapping")
        if nm is not None and not isinstance(nm, dict):
            issues.append("layer_semantics.name_mapping 必须是对象")
        la = sem.get("layer_attributes")
        if la is not None and not isinstance(la, list):
            issues.append("layer_semantics.layer_attributes 必须是列表")

    _check_ai_semantic_classes(preset, issues)
    _check_density_refine(preset, issues)
    _check_plate_operators(preset, issues)

    # 可选 pydantic 强化校验（安装时生效）
    try:
        from pydantic import BaseModel, ValidationError, field_validator  # type: ignore

        class _PresetModel(BaseModel):
            model_config = {"extra": "allow", "arbitrary_types_allowed": True}

            mode: str = "locked"
            super_res_scale: float | None = None

            @field_validator("mode")
            @classmethod
            def _v_mode(cls, v: str) -> str:
                if v not in _ALLOWED_MODES:
                    raise ValueError(f"mode 非法: {v}")
                return v

        try:
            _PresetModel(**preset)
        except ValidationError as ve:
            issues.extend([f"[pydantic] {e.get('msg')}" for e in ve.errors()])
    except Exception:
        # pydantic 未安装：手工校验已覆盖主要项
        pass

    return issues
