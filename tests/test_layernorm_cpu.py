import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as torch_nn

import neunet
import neunet.nn as nnn
from neunet.autograd import Tensor


@pytest.mark.parametrize(
    "shape,normalized_shape,eps,affine,rtol,atol",
    [
        ((2, 3, 4), 4, 1e-5, True, 1e-5, 1e-5),
        ((2, 3, 4), 4, 1e-5, False, 1e-5, 1e-5),
    ],
)
def test_layernorm_cpu(shape, normalized_shape, eps, affine, rtol, atol):
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)

    # Neunet
    x_nn = Tensor(x_np.copy(), device="cpu", requires_grad=True)
    ln_nn = nnn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)

    if affine:
        ln_nn.weight.data = np.random.randn(normalized_shape).astype(np.float32)
        ln_nn.bias.data = np.random.randn(normalized_shape).astype(np.float32)

    y_nn = ln_nn(x_nn)
    y_nn.backward(np.ones_like(y_nn.data))

    # Torch
    x_t = torch.tensor(x_np.copy(), dtype=torch.float32, requires_grad=True)
    ln_t = torch_nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)

    if affine:
        ln_t.weight.data = torch.tensor(ln_nn.weight.data, dtype=torch.float32)
        ln_t.bias.data = torch.tensor(ln_nn.bias.data, dtype=torch.float32)

    y_t = ln_t(x_t)
    y_t.backward(torch.ones_like(y_t))

    assert np.allclose(y_nn.data, y_t.detach().numpy(), rtol=rtol, atol=atol)
    assert np.allclose(x_nn.grad, x_t.grad.detach().numpy(), rtol=rtol, atol=atol)

    if affine:
        assert np.allclose(
            ln_nn.weight.grad, ln_t.weight.grad.detach().numpy(), rtol=rtol, atol=atol
        )
        assert np.allclose(
            ln_nn.bias.grad, ln_t.bias.grad.detach().numpy(), rtol=rtol, atol=atol
        )
