@echo off
chcp 65001 >nul
title 山水图 16K 超高清智能分层与超分流水线 - Universal Layer Studio PRO
echo =====================================================================
echo    UNIVERSAL LAYER STUDIO PRO - 16K 终极母版智能分层一键流水线
echo =====================================================================
echo.
echo 【输入图纸】 inputs/source_4000.jpg (4000 x 1952 像素)
echo 【目标输出】 outputs/Rosetsu_Master_16k.psb (16000 x 7808 像素，150 PPI)
echo 【硬件加速】 NVIDIA GeForce RTX 5070 (8GB) + Intel Arc 140T (16GB) 护盾
echo.
echo 正在启动全量自动化生产管线，请耐心等待（时长视硬件与放大倍率而定，
echo       16K 全量通常在十几分钟量级；进度与阶段请看控制台实时日志）。
echo.

python run_universal_engine.py --input inputs/source_4000.jpg --output outputs/Rosetsu_Master_16k.psb --preset japanese_screen_gold --scale 4.0 --dpi 150.0 --profile robust_performance

echo.
echo =====================================================================
echo 【处理完成！】
echo 您可以直接在 outputs/ 目录下找到 Rosetsu_Master_16k.psb 文件，
echo 使用 Adobe Photoshop 打开即可自由调整和编辑分层结果。
echo （实际图层数与各层名称以同目录 result.manifest.json 为准，勿以本脚本文案为准）
echo.
pause
