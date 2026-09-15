"""
归因模块（Attribution）—— Stage 3（反馈闭环 + 影子进化）

职责：
- 从 episode 归档的 quality_gate / diffuse_gate reject_reason 中提取归因信息
- 将拒绝原因映射到具体的 (prompt, category_id) 对，供学习器使用
- 支持 accept/reject 信号的提取，用于贝叶斯权重更新

设计原则：
- 归因信息来自 grounded_sam_provider 的 dino_detections 记录
- 每个检测记录包含：layer_name, prompt, boxes, logits, quality_gate_passed, 
  quality_gate_reason, diffuse_gate_passed, diffuse_gate_reason
- 归因结果：[(prompt, category_id, signal, reason), ...]
  其中 signal ∈ {"accept", "quality_reject", "diffuse_reject"}
"""

from typing import List, Tuple, Optional, Dict, Any


def attribute_reject_to_prompt(
    episode: Dict[str, Any]
) -> List[Tuple[str, str, str, str]]:
    """
    从 episode 归档中提取归因信息：哪些 prompt 产生了哪些检测结果，以及质量门/弥散门的判定。
    
    Args:
        episode: episode 归档 JSON（包含 dino_detections 字段）
        
    Returns:
        List of (prompt, category_id, signal, reason) tuples:
        - prompt: 检测时使用的 prompt 文本
        - category_id: 规范类目 ID（优先 det.category_id，否则回退 layer_name；
          二者皆缺则为 "unknown"）
        - signal: "accept" | "quality_reject" | "diffuse_reject"
          （质量门/弥散门字段缺失视为未通过，不得默认通过）
        - reason: 拒绝原因（accept 时为空字符串）
        
    Example:
        >>> episode = {
        ...     "dino_detections": [
        ...         {
        ...             "layer_name": "印章",
        ...             "prompt": "red stamp, cinnabar seal",
        ...             "quality_gate_passed": False,
        ...             "quality_gate_reason": "面积超预算 8.77% > 0.5%"
        ...         }
        ...     ]
        ... }
        >>> attribute_reject_to_prompt(episode)
        [('red stamp, cinnabar seal', '印章', 'quality_reject', '面积超预算 8.77% > 0.5%')]
    """
    dino_detections = episode.get("dino_detections", [])
    if not dino_detections:
        return []
    
    attributions = []
    for det in dino_detections:
        prompt = det.get("prompt", "")
        # 规范 category_id 优先；缺省回退 layer_name（保持旧契约）；二者皆缺 → 显式 unknown
        category_id = det.get("category_id") or det.get("layer_name") or "unknown"
        layer_name = det.get("layer_name", "")

        if not category_id or category_id == "unknown" or not prompt:
            # 仍记录（便于调试），但 category_id 显式为 unknown
            if not prompt:
                continue

        # 判断信号类型：质量门拒绝 > 弥散门拒绝 > 接受
        # 质量门/弥散门缺字段（未评估）一律视为未通过，不得默认通过
        quality_passed = det.get("quality_gate_passed")
        diffuse_passed = det.get("diffuse_gate_passed")
        quality_reason = det.get("quality_gate_reason")
        diffuse_reason = det.get("diffuse_gate_reason")

        if quality_passed is False:
            signal = "quality_reject"
            reason = quality_reason or "质量门未通过"
        elif quality_passed is None:
            # 质量门字段缺失 = 未评估 → 不能默认通过
            signal = "quality_reject"
            reason = quality_reason or "质量门缺字段(未评估)"
        elif diffuse_passed is False:
            signal = "diffuse_reject"
            reason = diffuse_reason or "弥散门未通过"
        elif diffuse_passed is None:
            signal = "diffuse_reject"
            reason = diffuse_reason or "弥散门缺字段(未评估)"
        else:
            signal = "accept"
            reason = ""

        attributions.append((prompt, category_id, signal, reason))

    return attributions


def extract_accept_reject_signals(
    episode: Dict[str, Any]
) -> Dict[str, Dict[str, int]]:
    """
    从 episode 归档中提取 accept/reject 信号统计，供贝叶斯权重更新使用。
    
    Args:
        episode: episode 归档 JSON（包含 dino_detections 字段）
        
    Returns:
        {
            "category_id": {
                "n_accept": int,
                "n_quality_reject": int,
                "n_diffuse_reject": int,
                "prompts": {
                    "prompt_text": {
                        "n_accept": int,
                        "n_quality_reject": int,
                        "n_diffuse_reject": int
                    }
                }
            }
        }
    """
    dino_detections = episode.get("dino_detections", [])
    if not dino_detections:
        return {}
    
    signals = {}
    for det in dino_detections:
        prompt = det.get("prompt", "")
        # 规范 category_id 优先；缺省回退 layer_name；二者皆缺 → 显式 unknown
        category_id = det.get("category_id") or det.get("layer_name") or "unknown"

        if not category_id or not prompt:
            continue

        # 初始化类目统计
        if category_id not in signals:
            signals[category_id] = {
                "n_accept": 0,
                "n_quality_reject": 0,
                "n_diffuse_reject": 0,
                "prompts": {}
            }

        # 初始化 prompt 统计
        if prompt not in signals[category_id]["prompts"]:
            signals[category_id]["prompts"][prompt] = {
                "n_accept": 0,
                "n_quality_reject": 0,
                "n_diffuse_reject": 0
            }

        # 统计信号；质量门/弥散门缺字段视为未通过（不得默认通过）
        quality_passed = det.get("quality_gate_passed")
        diffuse_passed = det.get("diffuse_gate_passed")

        if quality_passed is False or quality_passed is None:
            signals[category_id]["n_quality_reject"] += 1
            signals[category_id]["prompts"][prompt]["n_quality_reject"] += 1
        elif diffuse_passed is False or diffuse_passed is None:
            signals[category_id]["n_diffuse_reject"] += 1
            signals[category_id]["prompts"][prompt]["n_diffuse_reject"] += 1
        else:
            signals[category_id]["n_accept"] += 1
            signals[category_id]["prompts"][prompt]["n_accept"] += 1

    return signals
