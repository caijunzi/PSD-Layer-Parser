"""Hardware Device Schema & Inspection for Universal Neural Layer Studio.
Provides unified hardware device introspection and configuration schema
for both CLI and future UI (Web / Electron / Desktop) frontends.
"""
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


class DeviceOption(str, Enum):
    AUTO = "auto"
    RTX_5070 = "5070"
    INTEL_ARC = "arc"
    INTEL_NPU = "npu"
    CPU = "cpu"


DEVICE_HARDWARE_MAP = {
    DeviceOption.AUTO: "GPU.1",     # Priority 1: RTX 5070 with auto fallback
    DeviceOption.RTX_5070: "GPU.1", # NVIDIA RTX 5070 Laptop GPU (dGPU)
    DeviceOption.INTEL_ARC: "GPU.0",# Intel Arc 140T GPU 16GB (iGPU)
    DeviceOption.INTEL_NPU: "NPU",  # Intel AI Boost NPU
    DeviceOption.CPU: "CPU",        # Multi-core CPU
}


@dataclass
class HardwareDeviceInfo:
    id: str
    display_name: str
    hardware_type: str # 'dGPU' | 'iGPU' | 'NPU' | 'CPU'
    memory_description: str
    is_available: bool
    is_recommended: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_available_hardware_devices() -> List[HardwareDeviceInfo]:
    """
    Scans host hardware and returns structured device options
    tailored for UI dropdowns, selector cards, and CLI choices.
    """
    detected_devices = []
    if HAS_OPENVINO:
        try:
            core = ov.Core()
            detected_devices = core.available_devices
        except Exception:
            detected_devices = []

    options = [
        HardwareDeviceInfo(
            id="auto",
            display_name="自动选择 (Auto Smart Dispatch)",
            hardware_type="AUTO",
            memory_description="动态分配",
            is_available=True,
            is_recommended=True,
            description="根据算子自动分流：重型神经网络优先调度 RTX 5070，防爆显存超大图调度 Arc 140T，轻量检测调度 NPU"
        ),
        HardwareDeviceInfo(
            id="5070",
            display_name="NVIDIA GeForce RTX 5070 Laptop GPU",
            hardware_type="dGPU",
            memory_description="8 GB GDDR7 独立显存",
            is_available="GPU.1" in detected_devices,
            is_recommended=False,
            description="最强 FP32 浮点张量算力与 Tensor Cores，适合极速完成 LaMa 频域补全与高精 Alpha 抠图"
        ),
        HardwareDeviceInfo(
            id="arc",
            display_name="Intel(R) Arc(TM) 140T GPU",
            hardware_type="iGPU",
            memory_description="16 GB 共享显存",
            is_available="GPU.0" in detected_devices,
            is_recommended=False,
            description="超大 16GB 共享显存，彻底杜绝爆显存崩溃，适合超大画幅金箔底板与连续材质重构"
        ),
        HardwareDeviceInfo(
            id="npu",
            display_name="Intel(R) AI Boost NPU",
            hardware_type="NPU",
            memory_description="专用神经网络加速芯片",
            is_available="NPU" in detected_devices,
            is_recommended=False,
            description="3~5W 极致超低功耗，适合后台静默目标检测、轮廓提取与微孔羽化分类"
        ),
        HardwareDeviceInfo(
            id="cpu",
            display_name="Intel(R) Core(TM) Ultra 9 285H",
            hardware_type="CPU",
            memory_description="16 核心多线程系统内存",
            is_available=True,
            is_recommended=False,
            description="通用逻辑运算、Frangi 血管骨架流追踪、2.5D深度层序拓扑排序与 PSB 文件汇编"
        ),
    ]
    return options
