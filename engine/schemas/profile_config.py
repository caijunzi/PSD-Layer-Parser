"""Execution Profile Configurations for Universal Neural Layer Studio.
Defines high-performance, robust work intent profiles designed specifically
for plugged-in high-end mobile workstations (RTX 5070 dGPU + Intel Arc 140T iGPU + 16-core CPU).
"""
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


class ProfileType(str, Enum):
    ROBUST_PERFORMANCE = "robust_performance" # Default: RTX 5070 primary + Arc 16GB shield + CPU circuit breaker
    DIRECT_5070 = "5070"                      # Pure RTX 5070 dGPU dedicated
    DIRECT_ARC = "arc"                        # Pure Intel Arc 140T 16GB shared memory shield
    DIRECT_CPU = "cpu"                        # Pure Intel Ultra 9 16-core CPU mode


@dataclass
class ProfileSettings:
    profile_id: str
    display_name: str
    description: str
    primary_device: str
    shield_device: str
    enable_circuit_breaker: bool
    max_tile_size: int = 512
    vram_safety_limit_gb: float = 3.0
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PROFILES: Dict[str, ProfileSettings] = {
    ProfileType.ROBUST_PERFORMANCE: ProfileSettings(
        profile_id="robust_performance",
        display_name="插电极致鲁棒高性能模式 (Robust High-Performance)",
        description="【生产推荐】RTX 5070 满血冲锋处理密集张量算子，Intel Arc 140T 16GB 共享内存防爆显存护盾，单瓦片驱动超时自愈熔断",
        primary_device="GPU.1",
        shield_device="GPU.0",
        enable_circuit_breaker=True,
        max_tile_size=512,
        vram_safety_limit_gb=3.0,
        is_default=True
    ),
    ProfileType.DIRECT_5070: ProfileSettings(
        profile_id="5070",
        display_name="RTX 5070 独显直通模式 (Direct RTX 5070)",
        description="强制所有神经网络算子直通 NVIDIA RTX 5070 独显，全力释放 Tensor Cores 浮点算力",
        primary_device="GPU.1",
        shield_device="CPU",
        enable_circuit_breaker=True,
        max_tile_size=512,
        vram_safety_limit_gb=4.0,
        is_default=False
    ),
    ProfileType.DIRECT_ARC: ProfileSettings(
        profile_id="arc",
        display_name="Intel Arc 140T 大显存模式 (16GB Safe VRAM Shield)",
        description="利用核显 16GB 共享系统内存池，彻底免疫任何超大画幅爆显存闪退风险",
        primary_device="GPU.0",
        shield_device="CPU",
        enable_circuit_breaker=True,
        max_tile_size=512,
        vram_safety_limit_gb=12.0,
        is_default=False
    ),
    ProfileType.DIRECT_CPU: ProfileSettings(
        profile_id="cpu",
        display_name="Intel Ultra 9 纯 CPU 模式 (Deterministic CPU)",
        description="纯 CPU 16 核心多线程数学运算与内存直连，零硬件驱动依赖",
        primary_device="CPU",
        shield_device="CPU",
        enable_circuit_breaker=False,
        max_tile_size=512,
        vram_safety_limit_gb=16.0,
        is_default=False
    )
}


def resolve_profile(name: str) -> ProfileSettings:
    """Resolves profile string (CLI or UI) to concrete ProfileSettings."""
    if not name:
        return PROFILES[ProfileType.ROBUST_PERFORMANCE]
    
    key = name.lower().strip()
    if key in PROFILES:
        return PROFILES[key]
    
    # Aliases
    if key in ["auto", "default", "performance", "robust"]:
        return PROFILES[ProfileType.ROBUST_PERFORMANCE]
    elif key in ["gpu", "dgpu", "nvidia", "5070"]:
        return PROFILES[ProfileType.DIRECT_5070]
    elif key in ["igpu", "intel", "arc", "safe"]:
        return PROFILES[ProfileType.DIRECT_ARC]
    elif key in ["cpu", "pure_cpu"]:
        return PROFILES[ProfileType.DIRECT_CPU]

    # Fallback to robust_performance
    print(f"[ProfileResolver] Warning: Unknown profile '{name}', defaulting to 'robust_performance'")
    return PROFILES[ProfileType.ROBUST_PERFORMANCE]


def list_profiles_for_ui() -> List[Dict[str, Any]]:
    """Returns available profiles formatted for UI dropdowns/cards."""
    return [p.to_dict() for p in PROFILES.values()]
