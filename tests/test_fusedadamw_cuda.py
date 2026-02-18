import numpy as np
import cupy as cp
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.optim import AdamW
from neunet.nn.experimental import CUDAFusedAdamW, CUDAFusedMultiTensorAdamW


@pytest.mark.parametrize(
    "shape,lr,betas,eps,weight_decay",
    [
        ((128, 256), 1e-3, (0.9, 0.999), 1e-8, 1e-2),
        ((64, 128), 5e-4, (0.9, 0.999), 1e-8, 0.0),
    ],
)
def test_fused_adamw_step(shape, lr, betas, eps, weight_decay):
    print(f"Testing CUDA FusedAdamW vs AdamW: shape={shape}")

    np.random.seed(42)
    data = np.random.randn(*shape).astype(np.float32)
    grad = np.random.randn(*shape).astype(np.float32)

    # Native AdamW
    t_native = Tensor(cp.array(data), device="cuda", requires_grad=True)
    t_native.grad = cp.array(grad)
    opt_native = AdamW([t_native], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # CUDA FusedAdamW
    t_cuda = Tensor(cp.array(data), device="cuda", requires_grad=True)
    t_cuda.grad = cp.array(grad)
    opt_cuda = CUDAFusedAdamW([t_cuda], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # Single step
    opt_native.step()
    opt_cuda.step()

    # Compare params
    diff = cp.abs(t_native.data - t_cuda.data).max()
    print(f"Max abs diff: {diff}")

    assert cp.allclose(t_native.data, t_cuda.data, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "num_tensors,shape,lr,betas,eps,weight_decay",
    [
        (3, (64, 128), 1e-3, (0.9, 0.999), 1e-8, 1e-2),
        (5, (32, 64), 5e-4, (0.9, 0.999), 1e-8, 0.0),
    ],
)
def test_fused_multitensor_adamw_step(num_tensors, shape, lr, betas, eps, weight_decay):
    print(f"Testing CUDA FusedMultiTensorAdamW vs AdamW: num_tensors={num_tensors}, shape={shape}")

    np.random.seed(123)

    # Native AdamW params
    native_params = []
    fused_params = []

    for _ in range(num_tensors):
        data = np.random.randn(*shape).astype(np.float32)
        grad = np.random.randn(*shape).astype(np.float32)

        t_native = Tensor(cp.array(data), device="cuda", requires_grad=True)
        t_native.grad = cp.array(grad)
        native_params.append(t_native)

        t_fused = Tensor(cp.array(data), device="cuda", requires_grad=True)
        t_fused.grad = cp.array(grad)
        fused_params.append(t_fused)

    opt_native = AdamW(native_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    opt_fused = CUDAFusedMultiTensorAdamW(fused_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    opt_native.step()
    opt_fused.step()

    for p_native, p_fused in zip(native_params, fused_params):
        assert cp.allclose(p_native.data, p_fused.data, rtol=1e-4, atol=1e-4)
