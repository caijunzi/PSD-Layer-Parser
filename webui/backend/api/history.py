"""历史记录接口。"""
from fastapi import APIRouter

from core.file_handler import read_history

router = APIRouter()


@router.get("/history")
def get_history(limit: int = 20, offset: int = 0):
    """分页返回历史任务记录（按创建时间倒序）。"""
    items = read_history()
    total = len(items)
    page = items[offset: offset + limit]
    return {
        "success": True,
        "data": {"total": total, "items": page},
    }
