# WebUI 快速开始指南

本目录包含 Universal Layer Studio PRO 的 Web 用户界面。

## 📁 目录结构

```
webui/
├── frontend/          # React 前端（开发中）
├── backend/           # FastAPI 后端（开发中）
├── data/              # 运行时数据
│   ├── uploads/       # 用户上传
│   ├── outputs/       # 处理结果
│   ├── thumbnails/    # 缩略图缓存
│   └── history.json   # 历史记录
└── README.md          # 本文件
```

## 🚀 快速启动

### 环境要求

- Python 3.12+
- Node.js 20+
- 已安装项目依赖（`requirements.txt` + `requirements-ai.txt`）

### 1. 启动后端服务

```bash
cd webui/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install fastapi uvicorn[standard] python-multipart pillow

# 从项目根目录启动
cd ../..
python -m uvicorn webui.backend.main:app --reload --port 8099
```

后端服务运行在 `http://localhost:8099`，访问 `/docs` 查看 API 文档。

### 2. 启动前端开发服务器

```bash
cd webui/frontend
npm install
npm run dev
```

前端运行在 `http://localhost:5173`，自动连接后端 API。

### 3. 访问应用

在浏览器打开 `http://localhost:5173`，开始使用 WebUI。

## 🔌 API 端点一览（前端 ↔ 后端 契约）

> 前端调用点核对于 2026-09-16（`webui/frontend/src/App.tsx`）。
> ✅ 已接入 = 前端真实调用；🧩 集成专用 = 后端可用但 UI 暂不接入
> （供 CLI / 运维脚本 / 第三方集成直接调用，**不算悬空**）。

| 方法 | 路径 | 用途 | 前端 |
|---|---|---|---|
| GET  | `/api/health` | 健康检查（页面顶部连接状态） | ✅ |
| POST | `/api/upload` | 上传图片，返回 file_id + 智能推荐 preset | ✅ |
| GET  | `/api/presets` | 品类 preset 列表 | ✅ |
| POST | `/api/process` | 提交处理任务（返回 task_id，WebSocket 推进度） | ✅ |
| GET  | `/api/download/{task_id}/{file_type}` | 下载产物（psb / manifest / masks / audit） | ✅ |
| GET  | `/api/history?limit=&offset=` | 历史任务列表（页面底部「历史记录」面板） | ✅ |
| POST | `/api/adaptive/suggest-auto-tune` | 请求自适应 Auto-Tune 建议 | ✅ |
| POST | `/api/adaptive/apply-auto-tune` | 采纳建议并写入 preset | ✅ |
| GET  | `/api/adaptive/pending-feedbacks` | 待人审类目（Stage 5.3 弹窗） | ✅ |
| POST | `/api/adaptive/categories/{id}/feedback` | 提交人审反馈 | ✅ |
| GET  | `/api/adaptive/stats` | 类目/先验/episode 统计 | 🧩 集成专用 |
| GET  | `/api/adaptive/categories` | 类目树查询 | 🧩 集成专用 |
| POST | `/api/adaptive/trigger-learning` | 手动触发后台学习 | 🧩 集成专用 |
| GET  | `/api/adaptive/learning-status` | 后台学习状态轮询 | 🧩 集成专用 |
| WS   | `/ws/progress/{task_id}` | 实时进度（progress / completed / audit / error） | ✅ |

🧩 端点示例用途：学习流水线的定时触发与状态轮询、CI 中检查类目树健康度等。
若未来要在 UI 中接入，请同步更新本表（新增端点也须登记，防止出现无人调用的悬空接口）。

## 📋 开发状态

**当前版本**: v1.x（核心链路已闭环，2026-09-16 实测）

**已完成**:
- ✅ 后端 API 全部实现（FastAPI，41 项后端单测通过）
- ✅ 前端全流程：上传 → 智能推荐 → 参数配置 → 处理 → 8 维审计 → 下载
- ✅ WebSocket 实时进度 + 断线重连补播
- ✅ 历史记录面板（`/api/history`）
- ✅ 实体样块识别 + 材质家族判别的 preset 智能推荐
- ✅ 真实浏览器端到端点击流验证（agent-browser）

## 📖 技术文档

详细的技术实现规范请查看：
- [WEBUI_TECHNICAL_SPEC.md](../docs/WEBUI_TECHNICAL_SPEC.md) — 完整技术设计文档（12000 字）

## 🐛 常见问题

**Q: 后端启动报错 "ModuleNotFoundError: No module named 'fastapi'"**  
A: 确保在 `webui/backend/venv` 虚拟环境中安装了依赖。

**Q: 前端无法连接后端 API**  
A: 检查 `vite.config.ts` 中的 proxy 配置，确保指向 `http://localhost:8099`。

**Q: WebSocket 连接失败**  
A: 确认后端服务运行正常，查看浏览器控制台 Network 标签的 WS 连接状态。

## 🤝 贡献指南

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 许可证

遵循项目主许可证。
