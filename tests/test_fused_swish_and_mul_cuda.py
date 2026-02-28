import numpy as np
import cupy as cp
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.experimental.activations.fused_swish_and_mul.fused_swish_and_mul import (
    CUDAFusedSwishAndMul,
    cuda_fused_swish_and_mul,
    cuda_fused_swish_and_mul_backward,
)


def _ref_forward(x: cp.ndarray, hidden_size: int, beta: float):
    gate = x[..., :hidden_size]
    up = x[..., hidden_size:]
    sig = 1.0 / (1.0 + cp.exp(-beta * gate))
    return gate * sig * up


def _ref_backward(x: cp.ndarray, grad_out: cp.ndarray, hidden_size: int, beta: float):
    gate = x[..., :hidden_size]
    up = x[..., hidden_size:]
    sig = 1.0 / (1.0 + cp.exp(-beta * gate))
    swish_gate = gate * sig
    d_swish = sig + beta * swish_gate * (1.0 - sig)

    grad_gate = grad_out * up * d_swish
    grad_up = grad_out * swish_gate
    return cp.concatenate([grad_gate, grad_up], axis=-1)


@pytest.mark.parametrize(
    "shape,hidden_size,beta,rtol,atol",
    [
        ((32, 256), 128, 1.0, 1e-5, 1e-5),
        ((8, 16, 128), 64, 1.5, 1e-5, 1e-5),
    ],
)
def test_fused_swish_and_mul_module_forward_backward(shape, hidden_size, beta, rtol, atol):
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    grad_np = np.random.randn(*shape[:-1], hidden_size).astype(np.float32)

    x = Tensor(x_np.copy(), device="cuda", requires_grad=True)
    module = CUDAFusedSwishAndMul(beta=beta)

    out = module(x)
    out_ref = _ref_forward(cp.asarray(x_np), hidden_size=hidden_size, beta=beta)

    cp.testing.assert_allclose(out.data, out_ref, rtol=rtol, atol=atol)

    grad = cp.asarray(grad_np)
    out.backward(grad.copy())
    grad_ref = _ref_backward(cp.asarray(x_np), grad, hidden_size=hidden_size, beta=beta)

    cp.testing.assert_allclose(x.grad, grad_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape,hidden_size,beta,rtol,atol",
    [
        ((16, 128), 64, 1.0, 1e-5, 1e-5),
    ],
)
def test_fused_swish_and_mul_raw_kernels_forward_backward(shape, hidden_size, beta, rtol, atol):
    np.random.seed(123)
    x = cp.asarray(np.random.randn(*shape).astype(np.float32))
    grad_out = cp.asarray(np.random.randn(*shape[:-1], hidden_size).astype(np.float32))

    out = cp.empty(shape[:-1] + (hidden_size,), dtype=cp.float32)
    cuda_fused_swish_and_mul(x, out, hidden_size=hidden_size, beta=beta)
    out_ref = _ref_forward(x, hidden_size=hidden_size, beta=beta)
    cp.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol)

    grad_input = cp.empty_like(x)
    cuda_fused_swish_and_mul_backward(
        grad_input,
        grad_out,
        x,
        hidden_size=hidden_size,
        beta=beta,
    )
    grad_ref = _ref_backward(x, grad_out, hidden_size=hidden_size, beta=beta)
    cp.testing.assert_allclose(grad_input, grad_ref, rtol=rtol, atol=atol)
