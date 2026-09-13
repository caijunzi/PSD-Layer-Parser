# 项目长期记忆（PSD图层处理）


## 设备路由事实（2026-09-13 实测确认，重要）
本机三算力：Intel Arc 140T iGPU(GPU.0, 16GB共享) / RTX 5070 Laptop dGPU(GPU.1, 8GB) / Intel NPU。
4 个本地模型的**实际落点**（robust_performance 档）：
- GroundingDINO(661MB) → **CPU**（写死；deform-attn CUDA 分支缺 _C 扩展）
- SAM2(149MB) → **CPU**（写死；ULS_SEGMENT_DEVICE=cuda 可 opt-in，未在本机复验是否更快）
- RealESRGAN(64MB) → **Intel Arc GPU.0**（OpenVINO cached IR）
- LaMa(198MB onnx) → **Intel Arc GPU.0**（OpenVINO）
=> RTX 5070 在神经计算上**≈0 贡献**（仅 torch.cuda 探活碰一下）。

根因：**5070 走 OpenVINO 通用后端（无原生 NVIDIA EP/CUDA），实测远慢于 Arc 原生 Level Zero**：
- RealESRGAN 512x512：Arc 1.299s/片 vs 5070 92.71s/片 → **Arc 快 71.4x**
- LaMa 512x512：Arc 0.225s/片 vs 5070 5.763s/片 → **Arc 快 25.6x**
基准脚本：scratch/bench_esrgan_gpu.py、scratch/bench_lama_gpu.py（结果落同名 .json）

已修复（未提交）：realesrgan_provider / inpainting_provider 均改为
profile primary_device=GPU.1 且 shield=GPU.0(Arc) 时**优先 Arc**；首个编译成功者胜出，失败回退。
5070 直通档(shield=CPU)仍尊重首选。教训：探针须测速不止测"能跑"；改独显≠更快。
若要真正启用 5070：需接原生 CUDA/ORT-DML 路径（当前未接）并实测 Blackwell 支持度。

## 算子×设备实测矩阵（2026-09-13，覆盖前述设备路由结论）
| 算子 | OV-CPU | OV-Arc | OV-5070 | torch-CPU | torch-CUDA(5070) | ORT-CPU | ORT-DML | OV-NPU |
|---|---|---|---|---|---|---|---|---|
| RealESRGAN 512² | 15.41s | **1.223s** | 80.21s | 23.64s | 5.233s | — | —无ONNX | — |
| LaMa 512² | 1.428s | **0.220s** | 4.884s | — | — | 1.705s | ✗FFC MatMul失败 | ✗不支持 |
| SAM2 1024²(set_image) | — | — | — | 0.92s | **0.41s** | — | — | — |
| DINO 800px | — | — | — | 3.04s | ✗缺_C扩展 | — | — | — |

**修正后的正确分工（以此为长期结论）**：
- **Intel Arc(GPU.0)** = RealESRGAN 超分 + LaMa 补全（实测双料最快）。
- **CPU = 分割（DINO + SAM2）** —— ⚠️ 2026-09-13 复验**推翻**旧结论"SAM2 走 torch-CUDA 快 2.2x"：
  - SAM2 `set_image`（编码器）：CPU 1.378s vs CUDA 1.061s = **仅 1.30×**（非 2.2×）
  - SAM2 `predict`（单 prompt）：CPU 0.096s vs CUDA 0.661s = **GPU 反慢 6.88×**（CUDA kernel 启动开销）
  - SAM2 `predict`（batch 10）：CPU 0.055s vs CUDA 0.020s = 2.75×（但引擎是**逐框串行**，吃不到批量收益）
  - DINO（14 prompt，纯 torch 回退）：CPU 63.1s vs CUDA 53.4s = **仅 1.18×**，占 1.8GB 显存
  - ⇒ **分割维持 CPU**（收益不足 + 破坏 RK-16 逐像素复现）。基准脚本：scratch/bench_sam2_gpu.py、bench_dino_gpu.py
- **绝不可用 OpenVINO 喂 5070**（连 CPU 都不如：ESRGAN 80.21s vs CPU 15.41s）。
- **NPU** = 不支持神经算子，仅轻量任务。
- **DINO `_C` 扩展**：2026-09-13 已给 third_party/ms_deform_attn.py 加 `_C_AVAILABLE` 守卫——
  缺 `_C` 时 **CUDA 张量自动走纯-PyTorch 回退**（`F.grid_sample`），不再 NameError。
  编译 `_C` 内核**可行**（nvcc 有 win_amd64 wheel，可 pip 拼装 CUDA_HOME，免 3GB 安装器），
  但实测纯 torch 路径仅 1.18×、收益不足，**用户决定放弃编译、维持 CPU**。
- DML 因 LaMa 的 FFC 算子不支持而不可用。
- torch 2.7.1+cu128 的 arch_list **原生含 sm_120**（Blackwell 可用）；ORT 是 onnxruntime-directml 1.24.4。
- 本机 nvcc **不存在**（无任何 CUDA Toolkit）；MSVC 存在（VS2022 BuildTools cl.exe 14.44）。
## 测试环境（重要，避免重复踩坑）
**跑项目测试必须用系统 Python 3.12.10**：`C://Users//CK//AppData//Local//Programs//Python//Python312//python.exe`
- 已装齐：numpy 2.4.6 / sklearn 1.9.1 / cv2 5.0.0 / PIL 12.2.0 / psd_tools 1.19.0 / pytoshop 1.2.1 / skimage 0.26.0
- **WorkBuddy managed Python 3.13.12 无 numpy**，用它跑 pytest 会 ModuleNotFoundError
- 全量测试命令（均用该解释器）：
  - 引擎：`py -m pytest tests/ -q`（基线 114 passed / 2 skipped）
  - WebUI：`py -m unittest discover -s webui/backend/tests -p "test_*.py"`（**不可加 -t**，否则 base 模块 import 失败；28 OK）
- Git Bash 缺 tail/head（已知），勿用管道过滤
