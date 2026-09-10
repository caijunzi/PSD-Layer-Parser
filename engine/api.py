"""Unified API Interface for Future UI Frontends (Web / Electron / Desktop).
Provides clean, structured entrypoints for device enumeration, status query, and job execution.
"""
from typing import List, Dict, Any, Optional
from engine.schemas.device_config import get_available_hardware_devices, HardwareDeviceInfo
from engine.schemas.profile_config import list_profiles_for_ui, ProfileSettings
from run_universal_engine import run_pipeline


def list_devices_for_ui() -> List[Dict[str, Any]]:
    """Returns available compute devices formatted for UI dropdowns/cards."""
    return [d.to_dict() for d in get_available_hardware_devices()]


def list_intent_profiles_for_ui() -> List[Dict[str, Any]]:
    """Returns available work intent profiles formatted for UI selection cards."""
    return list_profiles_for_ui()


def execute_pipeline_from_ui(
    input_image_path: str,
    output_psb_path: str,
    preset_name: str = "japanese_screen_gold",
    profile: str = "robust_performance",
    device: Optional[str] = None,
    scale: float = 4.0,
    dpi: float = 150.0
) -> Dict[str, Any]:
    """
    Direct Python programmatic interface for UI backends (FastAPI / Electron IPC / PyQt).
    """
    output_path = run_pipeline(
        input_path=input_image_path,
        output_path=output_psb_path,
        preset_name=preset_name,
        target_scale=scale,
        dpi=dpi,
        device=device,
        profile=profile
    )
    return {
        "status": "success",
        "output_file": output_path,
        "profile_used": profile,
        "device_used": device or "auto"
    }
