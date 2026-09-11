"""WebSocket 连接管理器（模块级单例）。

task_manager 与 ws/progress 共享此实例，避免循环 import。
"""
from fastapi import WebSocket
from typing import Dict, List


class ConnectionManager:
    """按 task_id 分组管理 WebSocket 连接，支持广播进度。"""

    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, task_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self.active.setdefault(task_id, []).append(ws)

    def disconnect(self, task_id: str, ws: WebSocket) -> None:
        conns = self.active.get(task_id)
        if conns and ws in conns:
            conns.remove(ws)

    async def broadcast(self, task_id: str, message: dict) -> None:
        """向某任务的所有订阅者推送消息。"""
        conns = self.active.get(task_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(task_id, ws)


# 模块级单例
manager = ConnectionManager()
