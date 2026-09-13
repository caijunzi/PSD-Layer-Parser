"""
回归测试器（Regression Tester）—— Stage 3（反馈闭环 + 影子进化）

职责：
- 从 golden 图的 result.audit.json 中提取审计 8 维基线数据
- 写入 tests/baseline_audit_8d.json 供回归测试使用
- 每次学习器更新 prompt 后，跑回归测试验证 8 维指标不退化

设计原则：
- 审计 8 维来自 tools/audit_psb.py 的输出（result.audit.json）
- 8 维指标：lost_ratio, rmse_raw, rmse_lowfreq, plate_purity_ok, tac_max_pct, 
            n_layers, total_size_mb, backend（神经 vs 规则）
- 回归测试策略：核心指标（lost_ratio, rmse_lowfreq）退化 > 5% 视为失败
- golden 图覆盖 4 个预设 + 1 张复杂图（山水图 both 线）
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def extract_audit_8d(audit_json_path: str) -> Dict[str, Any]:
    """
    从 result.audit.json 中提取审计 8 维数据。
    
    Args:
        audit_json_path: result.audit.json 文件路径
        
    Returns:
        8 维指标字典：
        {
            "lost_ratio": float,
            "rmse_raw": float,
            "rmse_lowfreq": float,
            "plate_purity_ok": bool,
            "tac_max_pct": float,
            "n_layers": int,
            "total_size_mb": float,
            "backend": str  # "grounded_sam2_neural_dynamic" | "fallback_rule_based"
        }
    """
    audit_path = Path(audit_json_path)
    if not audit_path.exists():
        raise FileNotFoundError(f"审计文件不存在: {audit_json_path}")
    
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    
    # 从审计维度中提取指标（修正：使用 "dims" 而非 "dimensions"，并处理缺失字段）
    dims = audit.get("dims", {})
    
    # ④ 内容承载（lost_ratio）
    content_metrics = dims.get("④ 内容承载", {}).get("metrics", {})
    lost_ratio = content_metrics.get("lost_ratio", 0.0)
    
    # ⑤ 合成等价性（rmse_raw, rmse_lowfreq）
    synth_metrics = dims.get("⑤ 合成等价性", {}).get("metrics", {})
    rmse_raw = synth_metrics.get("rmse_raw", 0.0)
    rmse_lowfreq = synth_metrics.get("rmse_lowfreq", 0.0)
    
    # ⑦ plate 合规（plate_purity_ok, tac_max_pct）
    plate_metrics = dims.get("⑦ plate 合规", {}).get("metrics", {})
    plate_purity_ok = plate_metrics.get("plate_purity_ok", True)
    tac_max_pct = plate_metrics.get("tac_max_pct", 0.0)
    
    # ① 层属性（layer_count）
    layer_metrics = dims.get("① 层属性", {}).get("metrics", {})
    n_layers = layer_metrics.get("layer_count", 0)
    
    # total_size_mb 从顶层提取（如果审计文件有 size 字段）
    total_size_mb = 0.0  # 简化：审计文件中没有直接的文件大小字段
    
    # backend（从 manifest 或 audit 元数据中提取，此处简化为从文件名判断）
    # 实际实现中可从 manifest.json 的 totals.backend 字段读取
    backend = "unknown"
    
    return {
        "lost_ratio": lost_ratio,
        "rmse_raw": rmse_raw,
        "rmse_lowfreq": rmse_lowfreq,
        "plate_purity_ok": plate_purity_ok,
        "tac_max_pct": tac_max_pct,
        "n_layers": n_layers,
        "total_size_mb": total_size_mb,
        "backend": backend,
    }


def extract_golden_baseline(
    golden_tasks: List[Dict[str, str]],
    output_path: str = "tests/baseline_audit_8d.json"
) -> Dict[str, Dict[str, Any]]:
    """
    批量提取 golden 图的审计 8 维基线数据，写入 baseline_audit_8d.json。
    
    Args:
        golden_tasks: golden 任务列表，每项包含 task_id 和 audit_json_path
        output_path: 基线数据输出路径
        
    Returns:
        {task_id: audit_8d, ...}
    """
    baseline = {}
    
    for task in golden_tasks:
        task_id = task["task_id"]
        audit_path = task["audit_json_path"]
        
        try:
            audit_8d = extract_audit_8d(audit_path)
            baseline[task_id] = audit_8d
            print(f"[Regression] 提取基线: {task_id} → {audit_8d}")
        except Exception as e:
            print(f"[Regression] ⚠️  提取失败: {task_id} ({e})")
            continue
    
    # 写入基线文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    print(f"[Regression] ✅ 基线数据已写入: {output_path}（{len(baseline)} 个任务）")
    return baseline


def run_regression_test(
    current_audit_path: str,
    task_id: str,
    baseline_path: str = "tests/baseline_audit_8d.json",
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    跑回归测试：对比当前任务的审计 8 维与基线，检查是否退化。
    
    Args:
        current_audit_path: 当前任务的 result.audit.json 路径
        task_id: 任务 ID（用于查找基线）
        baseline_path: 基线数据文件路径
        threshold: 退化阈值（5%）
        
    Returns:
        {
            "passed": bool,
            "task_id": str,
            "issues": [str, ...],  # 退化项
            "current": dict,
            "baseline": dict
        }
    """
    # 加载基线
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        raise FileNotFoundError(f"基线数据不存在: {baseline_path}，请先运行 extract_golden_baseline")
    
    with open(baseline_file, "r", encoding="utf-8") as f:
        baseline_all = json.load(f)
    
    if task_id not in baseline_all:
        raise KeyError(f"基线中不存在任务 {task_id}")
    
    baseline = baseline_all[task_id]
    current = extract_audit_8d(current_audit_path)
    
    # 回归检查：核心指标（lost_ratio, rmse_lowfreq）退化 > threshold 视为失败
    issues = []
    
    # lost_ratio 增大 > threshold
    if current["lost_ratio"] > baseline["lost_ratio"] * (1 + threshold):
        delta = (current["lost_ratio"] - baseline["lost_ratio"]) / baseline["lost_ratio"] * 100
        issues.append(f"内容丢失比例退化: {baseline['lost_ratio']:.4f} → {current['lost_ratio']:.4f} (+{delta:.1f}%)")
    
    # rmse_lowfreq 增大 > threshold
    if current["rmse_lowfreq"] > baseline["rmse_lowfreq"] * (1 + threshold):
        delta = (current["rmse_lowfreq"] - baseline["rmse_lowfreq"]) / baseline["rmse_lowfreq"] * 100
        issues.append(f"低频 RMSE 退化: {baseline['rmse_lowfreq']:.2f} → {current['rmse_lowfreq']:.2f} (+{delta:.1f}%)")
    
    # plate_purity_ok 从 True 变 False
    if baseline["plate_purity_ok"] and not current["plate_purity_ok"]:
        issues.append("底板纯度退化: True → False")
    
    passed = len(issues) == 0
    
    return {
        "passed": passed,
        "task_id": task_id,
        "issues": issues,
        "current": current,
        "baseline": baseline,
    }
