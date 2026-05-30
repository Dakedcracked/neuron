#!/usr/bin/env python3
"""
benchmark_sota.py — Benchmark Script for SOTA Medical AI Models
This script evaluates the clinical deployment readiness (latency, throughput, memory)
of the top 3 open-source deep learning models on local infrastructure.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Try to import required medical deep learning libraries
try:
    import monai
    from monai.networks.nets import SwinUNETR, DynUNet
except ImportError:
    print("Error: MONAI not found. Please run within the project's virtual environment.")
    sys.exit(1)

try:
    import torchxrayvision as xrv
except ImportError:
    print("Error: torchxrayvision not found. Please run within the project's virtual environment.")
    sys.exit(1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_model(name, model, input_shape, runs=10, warmups=3, device=torch.device("cpu")):
    print(f"\n--- Benchmarking: {name} ---")
    print(f"Device: {device}")
    print(f"Input Shape: {input_shape}")
    print(f"Warmup runs: {warmups} | Active benchmark runs: {runs}")

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Generate dummy input tensor
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy_input)
    
    # Synchronize if using GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Active Benchmark
    latencies = []
    with torch.no_grad():
        for i in range(runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # to milliseconds

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    throughput = 1000.0 / avg_latency  # inferences per second (for batch size 1)

    print(f"Latency: {avg_latency:.2f} ms ± {std_latency:.2f} ms (Min: {min_latency:.2f} ms, Max: {max_latency:.2f} ms)")
    print(f"Throughput: {throughput:.2f} inferences/sec")

    # Estimate Peak VRAM or RAM
    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU Memory Used: {vram_mb:.2f} MB")
        torch.cuda.reset_peak_memory_stats(device)
    
    return {
        "model_name": name,
        "avg_latency_ms": float(avg_latency),
        "std_latency_ms": float(std_latency),
        "min_latency_ms": float(min_latency),
        "max_latency_ms": float(max_latency),
        "throughput_ips": float(throughput),
        "peak_vram_mb": float(vram_mb),
    }


def main():
    print("====================================================")
    print("    NEURON AI — SOTA MODEL BENCHMARK UTILITY        ")
    print("====================================================")
    
    device = get_device()
    print(f"Active Hardware Accelerator: {device.type.upper()}")
    if device.type == "cpu":
        print("WARNING: Running benchmarks on CPU. Latencies will be significantly higher than in production clinical environments.")

    results = []

    # 1. Chest X-Ray: DenseNet121-XRV (TorchXRayVision)
    print("\n[1/3] Loading DenseNet121 (torchxrayvision) weights...")
    t0 = time.perf_counter()
    try:
        # Load DenseNet121 from torchxrayvision without loading pretrained weights to avoid network hangs,
        # but if we can, we instantiate it. In this benchmark, we test model structure responsiveness.
        xray_model = xrv.models.DenseNet(weights=None)
        load_time = (time.perf_counter() - t0) * 1000.0
        print(f"Loaded in {load_time:.2f} ms")
        
        # Shape: (Batch, Channel, Height, Width) -> torchxrayvision expects 1 channel, 224x224
        xray_results = benchmark_model(
            name="DenseNet121-XRV (Chest X-Ray Pathology)",
            model=xray_model,
            input_shape=(1, 1, 224, 224),
            runs=15,
            warmups=5,
            device=device
        )
        results.append(xray_results)
    except Exception as e:
        print(f"Failed to benchmark DenseNet121: {e}")

    # 2. 3D MRI Segmentation: Swin UNETR (MONAI)
    print("\n[2/3] Loading Swin UNETR (MONAI) network structure...")
    t0 = time.perf_counter()
    try:
        # Initialize Swin UNETR. We use feature_size=12 for CPU benchmark speed,
        # but in production, feature_size=24 or 48 is typical.
        swin_model = SwinUNETR(
            in_channels=1,
            out_channels=3,  # e.g., enhancing tumor, tumor core, whole tumor
            feature_size=12,
            use_checkpoint=False
        )
        load_time = (time.perf_counter() - t0) * 1000.0
        print(f"Loaded in {load_time:.2f} ms")
        
        # Shape: (Batch, Channel, Depth, Height, Width) -> Standard resolution 96x96x96
        swin_results = benchmark_model(
            name="Swin UNETR (3D MRI Brain Tumor Segmentation)",
            model=swin_model,
            input_shape=(1, 1, 96, 96, 96),
            runs=5,
            warmups=2,
            device=device
        )
        results.append(swin_results)
    except Exception as e:
        print(f"Failed to benchmark Swin UNETR: {e}")

    # 3. 3D CT Segmentation: DynUNet / nnU-Net Style (MONAI)
    print("\n[3/3] Loading DynUNet (MONAI / nnU-Net Style) network structure...")
    t0 = time.perf_counter()
    try:
        # Initialize DynUNet with standard parameters
        spatial_dims = 3
        in_channels = 1
        out_channels = 2
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]
        upsample = [[2, 2, 2], [2, 2, 2]]
        
        dyn_model = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=upsample,
        )
        load_time = (time.perf_counter() - t0) * 1000.0
        print(f"Loaded in {load_time:.2f} ms")
        
        # Shape: (Batch, Channel, Depth, Height, Width) -> 96x96x96
        dyn_results = benchmark_model(
            name="DynUNet (3D CT Multi-Organ Segmentation)",
            model=dyn_model,
            input_shape=(1, 1, 96, 96, 96),
            runs=5,
            warmups=2,
            device=device
        )
        results.append(dyn_results)
    except Exception as e:
        print(f"Failed to benchmark DynUNet: {e}")

    # Output summaries
    print("\n====================================================")
    print("             BENCHMARK SUMMARY REPORT               ")
    print("====================================================")
    for res in results:
        print(f"\nModel: {res['model_name']}")
        print(f"  Avg Latency: {res['avg_latency_ms']:.2f} ms")
        print(f"  Throughput:  {res['throughput_ips']:.2f} inferences/sec")
        if device.type == "cuda":
            print(f"  Peak VRAM:   {res['peak_vram_mb']:.2f} MB")
            
    # Save results to JSON file
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": device.type,
            "torch_version": torch.__version__,
            "monai_version": monai.__version__,
            "torchxrayvision_version": xrv.__version__,
            "results": results
        }, f, indent=4)
    print(f"\n✓ Saved detailed benchmark JSON report to {out_path}")
    print("====================================================")


if __name__ == "__main__":
    main()
