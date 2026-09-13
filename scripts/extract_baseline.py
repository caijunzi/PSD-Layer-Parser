#!/usr/bin/env python3
"""
提取回归基线脚本（Stage 3.2）

从现有的 golden 图审计数据中提取回归基线，写入 tests/baseline_audit_8d.json。
"""

import sys
from pathlib import Path

# 添加 engine 到 sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from engine.adaptive.regression_tester import extract_golden_baseline

# Golden 任务列表（从现有输出目录中选取）
golden_tasks = [
    {
        "task_id": "task_20260912_132516_371e55",
        "audit_json_path": "webui/data/outputs/task_20260912_132516_371e55/result.audit.json",
        "description": "山水图 both 线（复杂图，含 DINO + 密度精修）"
    },
    {
        "task_id": "task_20260913_122144_d023f4",
        "audit_json_path": "webui/data/outputs/task_20260913_122144_d023f4/result.audit.json",
        "description": "锦纹壁布（damask）"
    },
    {
        "task_id": "task_20260913_120555_b2b831",
        "audit_json_path": "webui/data/outputs/task_20260913_120555_b2b831/result.audit.json",
        "description": "金地屏风（japanese_screen_gold）"
    },
    {
        "task_id": "task_20260913_103835_3cfc3b",
        "audit_json_path": "webui/data/outputs/task_20260913_103835_3cfc3b/result.audit.json",
        "description": "山水图 plate 线（chinese_ink_landscape_ai）"
    },
    {
        "task_id": "task_20260912_122614_8ac9e5",
        "audit_json_path": "webui/data/outputs/task_20260912_122614_8ac9e5/result.audit.json",
        "description": "象牙织物（ivory_fabric）"
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3.2：提取回归基线")
    print("=" * 60)
    
    # 转换为绝对路径
    for task in golden_tasks:
        task["audit_json_path"] = str(root_dir / task["audit_json_path"])
    
    # 提取基线
    baseline = extract_golden_baseline(
        golden_tasks=golden_tasks,
        output_path=str(root_dir / "tests/baseline_audit_8d.json")
    )
    
    print("\n" + "=" * 60)
    print(f"✅ 成功提取 {len(baseline)} 个任务的回归基线")
    print("=" * 60)
    
    # 打印摘要
    print("\n基线摘要：")
    print("-" * 60)
    for task_id, audit_8d in baseline.items():
        desc = next((t["description"] for t in golden_tasks if t["task_id"] == task_id), "")
        print(f"\n{task_id} ({desc})")
        print(f"  - lost_ratio: {audit_8d['lost_ratio']:.4f}")
        print(f"  - rmse_lowfreq: {audit_8d['rmse_lowfreq']:.2f}")
        print(f"  - plate_purity_ok: {audit_8d['plate_purity_ok']}")
        print(f"  - n_layers: {audit_8d['n_layers']}")
