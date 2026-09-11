"""提交处理任务接口。

创建任务 → 后台异步执行引擎 → 客户端订阅 WebSocket 拿进度。
"""
import asyncio

from fastapi import APIRouter, HTTPException

from core.task_manager import task_manager
from models.schemas import ProcessConfig, TaskCreated

router = APIRouter()


@router.post("/process", response_model=None)
async def process(cfg: ProcessConfig):
    """提交一个处理任务，返回 task_id 与 WebSocket 地址。"""
    # MVP 串行约束：同一时刻只允许一个引擎进程
    if task_manager.is_busy:
        raise HTTPException(
            status_code=409,
            detail="已有任务在处理中，请等待完成或取消后再提交（MVP 单任务串行）",
        )

    task_id, estimated, ws_url = task_manager.create_task(
        cfg.file_id, cfg.model_dump()
    )
    # 后台启动执行，不阻塞当前请求返回
    asyncio.create_task(task_manager.execute(task_id))

    return {
        "success": True,
        "data": TaskCreated(
            task_id=task_id, estimated_time=estimated, ws_url=ws_url
        ).model_dump(),
    }
