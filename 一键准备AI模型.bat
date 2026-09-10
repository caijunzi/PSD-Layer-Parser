@echo off
chcp 65001 >nul
title 视觉 AI 模型自动下载与健康校验中心 - Universal Layer Studio PRO
echo =====================================================================
echo       UNIVERSAL LAYER STUDIO PRO - 视觉 AI 模型一键下载中心
echo =====================================================================
echo.
echo 【提示】本程序将自动连接国内开源高速镜像（阿里魔搭/清华源），
echo       一键下载 SAM 2、Grounding DINO、Real-ESRGAN 等开源视觉模型。
echo       承诺：100%% 免费、0 商业云端 API、纯本地运行、不花一分钱！
echo.
echo 正在检查并启动下载器，请稍候...
echo.

python scripts/fetch_models.py

echo.
echo =====================================================================
echo 执行完毕！您可以按任意键关闭本窗口。
pause >nul
