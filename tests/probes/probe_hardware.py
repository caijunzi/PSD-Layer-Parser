# -*- coding: utf-8 -*-
"""探针：本机硬件在调用这些模型时到底承担了什么。

实测三件事：
  1. OpenVINO 可见设备清单
  2. LaMa ONNX 在各设备上的编译 + 推理耗时（同一模型、同一输入）
  3. PyTorch 对 RTX 5070 (sm_120) 的可用性 —— 这决定 GroundingDINO/SAM2 能否上 GPU
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LAMA_ONNX = "checkpoints/lama_fp32.onnx"
REPEAT = 3


def bench_openvino() -> None:
    import openvino as ov

    core = ov.Core()
    devices = core.available_devices
    print("=" * 78)
    print("  [1] OpenVINO 可见设备")
    print("=" * 78)
    for d in devices:
        try:
            name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            name = "?"
        print(f"  {d:<8} {name}")

    if not os.path.isfile(LAMA_ONNX):
        print(f"\n[SKIP] 未找到 {LAMA_ONNX}")
        return

    print()
    print("=" * 78)
    print("  [2] LaMa ONNX 在各设备上的实测（512x512 输入，取最快一次）")
    print("=" * 78)
    print(f"  {'设备':<10}{'编译s':>10}{'推理s':>10}{'状态'}")
    print("  " + "-" * 72)

    image = np.random.rand(1, 3, 512, 512).astype(np.float32)
    mask = np.ones((1, 1, 512, 512), dtype=np.float32)

    model = core.read_model(LAMA_ONNX)
    for dev in devices + ["NPU"]:
        if devices and dev not in devices and dev != "NPU":
            continue
        try:
            t0 = time.time()
            cm = core.compile_model(model, dev)
            t_compile = time.time() - t0
            cm({"image": image, "mask": mask})  # warmup
            ts = []
            for _ in range(REPEAT):
                t0 = time.time()
                cm({"image": image, "mask": mask})
                ts.append(time.time() - t0)
            print(f"  {dev:<10}{t_compile:>10.2f}{min(ts):>10.3f}  ✅ 可用")
        except Exception as e:
            msg = str(e).split("\n")[0][:60]
            print(f"  {dev:<10}{'-':>10}{'-':>10}  ❌ {msg}")


def bench_torch() -> None:
    print()
    print("=" * 78)
    print("  [3] PyTorch / CUDA 可用性（决定 GroundingDINO + SAM2 能否上 GPU）")
    print("=" * 78)
    try:
        import torch
    except ImportError:
        print("  torch 未安装")
        return
    print(f"  torch 版本            : {torch.__version__}")
    print(f"  torch.version.cuda    : {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"  cuda.is_available()   : {avail}")
    if not avail:
        print("  ⇒ 无可用 CUDA 设备，GroundingDINO/SAM2 只能跑 CPU")
        return
    print(f"  torch 编译支持的架构  : {torch.cuda.get_arch_list()}")
    try:
        n = torch.cuda.device_count()
        print(f"  可见 CUDA 设备数      : {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  算力 sm_{p.major}{p.minor}  "
                  f"显存 {p.total_memory / 1024 ** 3:.1f} GB")
    except Exception as e:
        print(f"  ⚠️ 获取设备属性失败    : {type(e).__name__}: {str(e)[:120]}")

    # 决定性测试：真在 GPU 上跑一次张量运算
    try:
        t0 = time.time()
        a = torch.randn(2048, 2048, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        print(f"  2048x2048 矩阵乘 GPU  : ✅ 成功 ({dt:.1f} ms)")
        print("  ⇒ PyTorch 可直接使用该 GPU")
    except Exception as e:
        print(f"  GPU 张量运算           : ❌ {type(e).__name__}: {str(e)[:200]}")
        print("  ⇒ PyTorch 无法使用该 GPU（sm_120 不在编译架构内），")
        print("    GroundingDINO / SAM2 这类 torch 模型只能退到 CPU")


if __name__ == "__main__":
    bench_openvino()
    bench_torch()
