"""Universal Layer Studio PRO - Backend API Server (FastAPI)

MVP 后端：上传 / Preset 列表 / 任务提交 / WebSocket 进度 / 下载 / 历史。

启动：
    cd webui/backend && python main.py
    或：python -m uvicorn main:app --host 127.0.0.1 --port 8099 --reload
"""
import sys
from pathlib import Path

# 把 backend 目录加到 sys.path，让 `from api.xxx import` 能工作
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.file_handler import DATA_DIR, ensure_dirs
from api.upload import router as upload_router
from api.process import router as process_router
from api.download import router as download_router
from api.presets import router as presets_router
from api.history import router as history_router
from ws.progress import router as ws_router

# 启动前确保数据目录存在
ensure_dirs()

app = FastAPI(
    title="Universal Layer Studio PRO API",
    version="1.0.0",
    description="工业级 2D 智能图像分层与印前制版工作站 Web API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS（开发环境，前端 vite :5173）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件：缩略图
app.mount(
    "/thumbnails",
    StaticFiles(directory=str(DATA_DIR / "thumbnails")),
    name="thumbnails",
)

# REST 路由
app.include_router(upload_router, prefix="/api")
app.include_router(process_router, prefix="/api")
app.include_router(download_router, prefix="/api")
app.include_router(presets_router, prefix="/api")
app.include_router(history_router, prefix="/api")

# WebSocket 路由
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "service": "Universal Layer Studio PRO API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "data_dir": str(DATA_DIR),
        "uploads_ready": (DATA_DIR / "uploads").exists(),
        "outputs_ready": (DATA_DIR / "outputs").exists(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8099,
        reload=True,
        log_level="info",
    )
