"""Preset 列表接口。"""
from datetime import datetime, timezone

from fastapi import APIRouter

from core.file_handler import list_presets

router = APIRouter()


@router.get("/presets")
def get_presets():
    """返回所有可用的品类 preset 清单。"""
    presets = list_presets()
    return {
        "success": True,
        "data": presets,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
