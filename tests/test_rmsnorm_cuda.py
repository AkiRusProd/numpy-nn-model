import numpy as np
import cupy as cp
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.layers import RMSNorm
from neunet.nn.experimental import CUDARMSNorm


@pytest.mark.parametrize(
    "batch_size,features,eps,rtol,atol",
    [
        (32, 128, 1e-6, 1e-4, 1e-4),
        (16, 256, 1e-6, 1e-4, 1e-4),
    ],
)
def test_rmsnorm_comparison(batch_size, features, eps, rtol, atol):
    print(
        f"Testing CUDA RMSNorm vs Native RMSNorm: Batch={batch_size}, Features={features}, eps={eps}"
    )

    np.random.seed(42)
    data = np.random.randn(batch_size, features).astype(np.float32)

    # Native RMSNorm on GPU
    t_native = Tensor(data.copy(), device="cuda", requires_grad=True)
    model_native = RMSNorm(features, eps=eps, device="cuda")

    # CUDA RMSNorm
    t_cuda = Tensor(data.copy(), device="cuda", requires_grad=True)
    model_cuda = CUDARMSNorm(features, eps=eps, device="cuda")

    # Sync weights and bias if present
    model_cuda.weight.data = cp.copy(model_native.weight.data)
    if getattr(model_native, "bias", None) is not None:
        model_cuda.bias.data = cp.copy(model_native.bias.data)

    # Forward Pass
    out_native = model_native(t_native)
    out_cuda = model_cuda(t_cuda)

    print("\n--- Forward Pass Comparison ---")
    forward_diff = cp.abs(out_native.data - out_cuda.data).max()
    print(f"Max absolute difference (Forward): {forward_diff}")

    forward_passed = cp.allclose(out_native.data, out_cuda.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("Forward pass: SUCCESS")
    else:
        print("Forward pass: FAILED")
    assert forward_passed

    # Backward Pass
    grad_np = np.random.randn(*out_native.shape).astype(np.float32)
    grad_cuda = cp.array(grad_np)

    out_native.backward(grad_cuda.copy())
    out_cuda.backward(grad_cuda.copy())

    print("\n--- Backward Pass Comparison ---")
    grad_diff = cp.abs(t_native.grad - t_cuda.grad).max()
    print(f"Max absolute difference (Gradients): {grad_diff}")

    grad_passed = cp.allclose(t_native.grad, t_cuda.grad, rtol=rtol, atol=atol)
    if grad_passed:
        print("Backward pass: SUCCESS")
    else:
        print("Backward pass: FAILED")
    assert grad_passed
