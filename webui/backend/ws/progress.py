"""WebSocket 进度推送端点。"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.connection_manager import manager
from core.task_manager import task_manager

router = APIRouter()


@router.websocket("/ws/progress/{task_id}")
async def progress_ws(websocket: WebSocket, task_id: str):
    """订阅某任务的实时进度。

    推送消息类型：progress | completed | error
    """
    await manager.connect(task_id, websocket)
    task = task_manager.get_task(task_id)

    # 已有任务的存量日志补播（断线重连场景）
    if task:
        for log in task.get("logs", [])[-20:]:
            await websocket.send_json({
                "type": "progress", "task_id": task_id,
                "message": log.get("message"),
                "progress": task.get("progress", 0),
                "stage": task.get("stage"),
                "timestamp": log.get("timestamp"),
                "replay": True,
            })
        if task.get("status") == "completed":
            await websocket.send_json({
                "type": "completed", "task_id": task_id,
                "elapsed_time": task.get("elapsed_time", 0),
                "output_files": task.get("output_files", []),
                "manifest": {},
            })

    try:
        # 保持连接，等待广播；同时接收客户端心跳/关闭
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(task_id, websocket)
