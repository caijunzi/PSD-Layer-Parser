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
        - category_id: 类目 ID（layer_name）
        - signal: "accept" | "quality_reject" | "diffuse_reject"
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
        layer_name = det.get("layer_name", "")
        prompt = det.get("prompt", "")
        
        if not layer_name or not prompt:
            continue
        
        # 判断信号类型：质量门拒绝 > 弥散门拒绝 > 接受
        quality_passed = det.get("quality_gate_passed", True)
        diffuse_passed = det.get("diffuse_gate_passed", True)
        
        if not quality_passed:
            signal = "quality_reject"
            reason = det.get("quality_gate_reason", "未知原因")
        elif not diffuse_passed:
            signal = "diffuse_reject"
            reason = det.get("diffuse_gate_reason", "未知原因")
        else:
            signal = "accept"
            reason = ""
        
        attributions.append((prompt, layer_name, signal, reason))
    
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
        layer_name = det.get("layer_name", "")
        prompt = det.get("prompt", "")
        
        if not layer_name or not prompt:
            continue
        
        # 初始化类目统计
        if layer_name not in signals:
            signals[layer_name] = {
                "n_accept": 0,
                "n_quality_reject": 0,
                "n_diffuse_reject": 0,
                "prompts": {}
            }
        
        # 初始化 prompt 统计
        if prompt not in signals[layer_name]["prompts"]:
            signals[layer_name]["prompts"][prompt] = {
                "n_accept": 0,
                "n_quality_reject": 0,
                "n_diffuse_reject": 0
            }
        
        # 统计信号
        quality_passed = det.get("quality_gate_passed", True)
        diffuse_passed = det.get("diffuse_gate_passed", True)
        
        if not quality_passed:
            signals[layer_name]["n_quality_reject"] += 1
            signals[layer_name]["prompts"][prompt]["n_quality_reject"] += 1
        elif not diffuse_passed:
            signals[layer_name]["n_diffuse_reject"] += 1
            signals[layer_name]["prompts"][prompt]["n_diffuse_reject"] += 1
        else:
            signals[layer_name]["n_accept"] += 1
            signals[layer_name]["prompts"][prompt]["n_accept"] += 1
    
    return signals
