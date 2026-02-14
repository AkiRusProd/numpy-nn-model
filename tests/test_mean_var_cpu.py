import numpy as np
import pytest

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor


def _get_torch():
    try:
        import torch
    except Exception:
        pytest.skip("PyTorch is not available")
    return torch


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 2, 5)])
def test_mean_var_broadcast_shapes(shape):
    x_arr = np.random.randn(*shape).astype(np.float32)
    x = Tensor(x_arr, device="cpu", requires_grad=True)

    mean_last = x.mean(axis=-1, keepdims=True)
    var_last = x.var(axis=-1, keepdims=True)

    assert mean_last.data.shape == x_arr.mean(axis=-1, keepdims=True).shape
    assert var_last.data.shape == x_arr.var(axis=-1, keepdims=True).shape

    centered = x - mean_last
    assert centered.data.shape == x_arr.shape

    mean_all = x.mean()
    centered_all = x - mean_all
    assert centered_all.data.shape == x_arr.shape


@pytest.mark.parametrize("shape", [(2, 3), (3, 4)])
def test_grad_centered_mean(shape):
    torch = _get_torch()

    np.random.seed(42)
    x_arr = np.random.randn(*shape).astype(np.float32)

    x = Tensor(x_arr.copy(), device="cpu", requires_grad=True)
    mean = x.mean(axis=1, keepdims=True)
    x_c = x - mean
    y = (x_c * x_c).sum()
    y.backward()

    t = torch.tensor(x_arr.copy(), requires_grad=True)
    t_mean = t.mean(dim=1, keepdim=True)
    t_c = t - t_mean
    t_y = (t_c * t_c).sum()
    t_y.backward()

    assert np.allclose(x.grad, t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, 3), (3, 4)])
def test_grad_var(shape):
    torch = _get_torch()

    np.random.seed(123)
    x_arr = np.random.randn(*shape).astype(np.float32)

    x = Tensor(x_arr.copy(), device="cpu", requires_grad=True)
    v = x.var(axis=1, keepdims=True)
    y = v.sum()
    y.backward()

    t = torch.tensor(x_arr.copy(), requires_grad=True)
    t_v = t.var(dim=1, unbiased=False, keepdim=True)
    t_y = t_v.sum()
    t_y.backward()

    assert np.allclose(x.grad, t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)
