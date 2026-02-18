import cupy as cp
import numpy as np
import neunet
import time

import os, sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.nn.experimental import CUDALinearSwish
from neunet.nn.layers import Linear
from neunet.nn.activations import Swish


def benchmark_forward(ref_linear, ref_swish, cutlass_layer, x_data, device, warmup=50, iters=200):
    """Benchmark forward pass: Linear+Swish vs fused CUTLASS"""

    # --- Warmup Reference ---
    for _ in range(warmup):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = ref_swish(ref_linear(X))
    cp.cuda.Device().synchronize()

    # --- Benchmark Reference ---
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = ref_swish(ref_linear(X))
    end.record()
    end.synchronize()
    ref_time_ms = cp.cuda.get_elapsed_time(start, end) / iters

    # --- Warmup CUTLASS ---
    for _ in range(warmup):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = cutlass_layer(X)
    cp.cuda.Device().synchronize()

    # --- Benchmark CUTLASS ---
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = cutlass_layer(X)
    end.record()
    end.synchronize()
    cutlass_time_ms = cp.cuda.get_elapsed_time(start, end) / iters

    return ref_time_ms, cutlass_time_ms


def benchmark_backward(ref_linear, ref_swish, cutlass_layer, x_data, device, warmup=50, iters=200):
    """Benchmark backward pass: Linear+Swish vs fused CUTLASS"""

    # Pre-generate grad_output shape
    X_tmp = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    out_tmp = ref_swish(ref_linear(X_tmp))
    grad_shape = out_tmp.shape
    del X_tmp, out_tmp

    # --- Warmup Reference ---
    for _ in range(warmup):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = ref_swish(ref_linear(X))
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)
        out.backward(grad)
        # Reset grads
        ref_linear.weight.grad = None
        ref_linear.bias.grad = None
    cp.cuda.Device().synchronize()

    # --- Benchmark Reference ---
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = ref_swish(ref_linear(X))
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)
        out.backward(grad)
        ref_linear.weight.grad = None
        ref_linear.bias.grad = None
    end.record()
    end.synchronize()
    ref_time_ms = cp.cuda.get_elapsed_time(start, end) / iters

    # --- Warmup CUTLASS ---
    for _ in range(warmup):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = cutlass_layer(X)
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)
        out.backward(grad)
        cutlass_layer.weight.grad = None
        cutlass_layer.bias.grad = None
    cp.cuda.Device().synchronize()

    # --- Benchmark CUTLASS ---
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        X = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = cutlass_layer(X)
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)
        out.backward(grad)
        cutlass_layer.weight.grad = None
        cutlass_layer.bias.grad = None
    end.record()
    end.synchronize()
    cutlass_time_ms = cp.cuda.get_elapsed_time(start, end) / iters

    return ref_time_ms, cutlass_time_ms


def run_benchmark():
    device = "cuda"
    swish_beta = 1.0

    configs = [
        # (batch_size, in_features, out_features)
        (32,   256,  512),
        (64,   256,  512),
        (128,  256,  512),
        (256,  512,  1024),
        (512,  512,  1024),
        (1024, 512,  1024),
        (1024, 1024, 2048),
        (2048, 1024, 2048),
        (4096, 1024, 4096),
    ]

    print("=" * 90)
    print("Benchmark: Linear+Swish (reference) vs CUTLASS Fused Linear+Swish (EVT backward)")
    print("=" * 90)

    # ========================= FORWARD =========================
    print(f"\n{'='*90}")
    print(f"{'FORWARD PASS':^90}")
    print(f"{'='*90}")
    print(f"{'Config (B x In x Out)':<25} {'Ref (ms)':>10} {'CUTLASS (ms)':>14} {'Speedup':>10}")
    print("-" * 60)

    for batch_size, in_features, out_features in configs:
        ref_linear = Linear(in_features, out_features, bias=True, device=device)
        ref_swish = Swish(beta=swish_beta)
        cutlass_layer = CUDALinearSwish(in_features, out_features, bias=True, swish_beta=swish_beta, device=device)

        # Sync weights
        cutlass_layer.weight.data = cp.copy(ref_linear.weight.data)
        cutlass_layer.bias.data = cp.copy(ref_linear.bias.data)

        x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)

        ref_ms, cutlass_ms = benchmark_forward(
            ref_linear, ref_swish, cutlass_layer, x_data, device
        )

        speedup = ref_ms / cutlass_ms if cutlass_ms > 0 else float('inf')
        config_str = f"{batch_size} x {in_features} x {out_features}"
        print(f"{config_str:<25} {ref_ms:>10.4f} {cutlass_ms:>14.4f} {speedup:>9.2f}x")

    # ========================= BACKWARD =========================
    print(f"\n{'='*90}")
    print(f"{'BACKWARD PASS (forward + backward)':^90}")
    print(f"{'='*90}")
    print(f"{'Config (B x In x Out)':<25} {'Ref (ms)':>10} {'CUTLASS (ms)':>14} {'Speedup':>10}")
    print("-" * 60)

    for batch_size, in_features, out_features in configs:
        ref_linear = Linear(in_features, out_features, bias=True, device=device)
        ref_swish = Swish(beta=swish_beta)
        cutlass_layer = CUDALinearSwish(in_features, out_features, bias=True, swish_beta=swish_beta, device=device)

        # Sync weights
        cutlass_layer.weight.data = cp.copy(ref_linear.weight.data)
        cutlass_layer.bias.data = cp.copy(ref_linear.bias.data)

        x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)

        ref_ms, cutlass_ms = benchmark_backward(
            ref_linear, ref_swish, cutlass_layer, x_data, device
        )

        speedup = ref_ms / cutlass_ms if cutlass_ms > 0 else float('inf')
        config_str = f"{batch_size} x {in_features} x {out_features}"
        print(f"{config_str:<25} {ref_ms:>10.4f} {cutlass_ms:>14.4f} {speedup:>9.2f}x")

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    run_benchmark()
