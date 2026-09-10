"""
Universal Layer Engine 算子包 (engine.operators)。
包含符合 SSOT §6.1 / §6.2 标准契约的核心图像分析、分割与印前制版算子。
"""

from engine.operators.base import BaseOperator, OperatorResult, get_param
from engine.operators.contour_protection import ContourProtectionOperator
from engine.operators.seam_harmonizer import SeamHarmonizerOperator
from engine.operators.trapping import TrappingOperator
from engine.operators.deocclusion_operator import DeocclusionOperator
from engine.operators.micro_holes import MicroHolesOperator, MicroHolesExtractor
from engine.operators.metallic_foil import MetallicFoilOperator, MetallicFoilSeparator

__all__ = [
    "BaseOperator",
    "OperatorResult",
    "get_param",
    "ContourProtectionOperator",
    "SeamHarmonizerOperator",
    "TrappingOperator",
    "DeocclusionOperator",
    "MicroHolesOperator",
    "MicroHolesExtractor",
    "MetallicFoilOperator",
    "MetallicFoilSeparator",
]
