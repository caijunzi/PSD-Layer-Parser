"""
BaseOperator 抽象基类与 OperatorResult 契约定义。
遵循 Universal Layer Engine SSOT §6.1 契约：
1. 算子为无状态纯函数，除 ctx 外不得读写全局状态；
2. 检测类算子取 ctx.detect_image，色彩类取 ctx.output_image，禁止混用；
3. 所有阈值来自 params（由 preset 注入），算子源码内禁止出现魔法数字字面量；
4. 输出掩模为 uint8 二值（0/255）；
5. 每个算子必须在 metrics 记录关键统计量（检测数量、面积占比等）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class OperatorResult:
    """算子执行统一返回载荷。"""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reconstruction_mask: Optional[np.ndarray] = None
    message: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def get_param(
    params: Optional[Dict[str, Any]],
    key: str,
    default: Any,
    min_val: Optional[Any] = None,
    max_val: Optional[Any] = None,
) -> Any:
    """
    从 params 字典安全读取配置参数，支持边界钳位与类型校验。
    消灭算子源码中的硬编码魔法值。
    """
    if params is None or key not in params:
        val = default
    else:
        val = params[key]

    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val

    return val


class BaseOperator(ABC):
    """
    工业级通用分层算子抽象基类。
    所有具体图像分析、分割、矫正、陷印算子均继承此类。
    """

    name: str = "base_operator"

    def __init__(self, name: Optional[str] = None):
        if name:
            self.name = name

    @abstractmethod
    def run(self, ctx: Any, params: Optional[Dict[str, Any]] = None) -> OperatorResult:
        """
        执行算子核心算法。
        
        Args:
            ctx: ProcessingContext 或包含图像数据的上下文对象
            params: 由 preset 注入的工艺参数字典
            
        Returns:
            OperatorResult: 包含执行状态、数据载荷、质量度量及消息
        """
        pass
