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

## 📋 开发状态

**当前版本**: v0.1.0 (开发中)

**已完成**:
- ✅ 技术架构设计
- ✅ API 接口定义
- ✅ UI 设计规范

**开发中**:
- 🔄 后端 API 实现（Week 1）
- 🔄 前端核心组件（Week 1-2）
- 🔄 WebSocket 实时进度（Week 1）

**待开发**:
- ⏳ 历史记录功能（Week 3）
- ⏳ 智能推荐优化（Week 3）
- ⏳ E2E 测试覆盖（Week 2-3）

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
