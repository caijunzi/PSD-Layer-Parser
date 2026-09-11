"""
Universal Layer Studio PRO - Backend API Server
FastAPI 服务主入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# 确保数据目录存在
DATA_DIR = Path(__file__).parent.parent / "data"
for subdir in ["uploads", "outputs", "thumbnails"]:
    (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Universal Layer Studio PRO API",
    version="1.0.0",
    description="工业级 2D 智能图像分层与印前制版工作站 Web API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 开发服务器
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount(
    "/thumbnails",
    StaticFiles(directory=str(DATA_DIR / "thumbnails")),
    name="thumbnails"
)

# TODO: 注册 API 路由
# app.include_router(upload_router, prefix="/api")
# app.include_router(process_router, prefix="/api")
# app.include_router(download_router, prefix="/api")
# app.include_router(presets_router, prefix="/api")

# TODO: WebSocket 路由
# @app.websocket("/ws/progress/{task_id}")
# async def progress_ws(websocket: WebSocket, task_id: str):
#     await websocket_endpoint(websocket, task_id)

@app.get("/")
async def root():
    """服务健康检查"""
    return {
        "service": "Universal Layer Studio PRO API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "data_dir": str(DATA_DIR),
        "uploads_ready": (DATA_DIR / "uploads").exists(),
        "outputs_ready": (DATA_DIR / "outputs").exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8099,
        reload=True,
        log_level="info"
    )
