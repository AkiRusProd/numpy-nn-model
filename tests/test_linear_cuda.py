import numpy as np
import cupy as cp
import pytest

import os, sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet
from neunet.nn.experimental import CUDALinear
from neunet.nn.layers import Linear


@pytest.mark.parametrize(
    "backend,batch_size,in_features,out_features,rtol,atol",
    [
        ("cutlass", 128, 256, 512, 1e-4, 1e-4),
        ("cublaslt", 128, 256, 512, 1e-4, 1e-4),
    ],
)
def test_cutlass_linear(backend, batch_size, in_features, out_features, rtol, atol):
    # Settings
    device = "cuda"

    print(
        f"Testing Linear Layer ({backend}): Batch={batch_size}, In={in_features}, Out={out_features}"
    )

    # 1. Layer initialization
    ref_layer = Linear(in_features, out_features, bias=True, device=device)
    cutlass_layer = CUDALinear(
        in_features, out_features, bias=True, device=device, backend=backend
    )

    # 2. Sync weights and bias
    cutlass_layer.weight.data = cp.copy(ref_layer.weight.data)
    cutlass_layer.bias.data = cp.copy(ref_layer.bias.data)

    # 3. Input data
    x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)
    X_ref = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    X_cutlass = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)

    # --- FORWARD PASS ---
    print("Running Forward pass...")
    out_ref = ref_layer(X_ref)
    out_cutlass = cutlass_layer(X_cutlass)

    forward_passed = cp.allclose(out_ref.data, out_cutlass.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("Forward pass: SUCCESS")
    else:
        diff = cp.abs(out_ref.data - out_cutlass.data).max()
        print(f"Forward pass: FAILED (Max diff: {diff})")
    assert forward_passed

    # --- BACKWARD PASS ---
    print("Running Backward pass...")
    grad_output = cp.random.uniform(-1, 1, out_ref.shape).astype(cp.float32)

    out_ref.backward(grad_output)
    out_cutlass.backward(grad_output)

    grad_X_passed = cp.allclose(X_ref.grad, X_cutlass.grad, rtol=rtol, atol=atol)
    if grad_X_passed:
        print("Backward X grad: SUCCESS")
    else:
        diff = cp.abs(X_ref.grad - X_cutlass.grad).max()
        print(f"Backward X grad: FAILED (Max diff: {diff})")
    assert grad_X_passed

    grad_W_passed = cp.allclose(ref_layer.weight.grad, cutlass_layer.weight.grad, rtol=rtol, atol=atol)
    if grad_W_passed:
        print("Backward Weight grad: SUCCESS")
    else:
        diff = cp.abs(ref_layer.weight.grad - cutlass_layer.weight.grad).max()
        print(f"Backward Weight grad: FAILED (Max diff: {diff})")
    assert grad_W_passed

    grad_b_passed = cp.allclose(ref_layer.bias.grad, cutlass_layer.bias.grad, rtol=rtol, atol=atol)
    if grad_b_passed:
        print("Backward Bias grad: SUCCESS")
    else:
        diff = cp.abs(ref_layer.bias.grad - cutlass_layer.bias.grad).max()
        print(f"Backward Bias grad: FAILED (Max diff: {diff})")
    assert grad_b_passed

    print(f"\nALL TESTS PASSED! {backend} implementation is correct.")
