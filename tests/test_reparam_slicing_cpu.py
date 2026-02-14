import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import neunet
import neunet.nn as nn
from neunet.autograd import Tensor


@pytest.mark.parametrize("batch,latent,input_size,rtol,atol", [(2, 3, 6, 1e-5, 1e-5)])
def test_reparam_slicing_cpu(batch, latent, input_size, rtol, atol):
    np.random.seed(123)

    # Neunet
    layer_nn = nn.Linear(input_size, latent * 2, bias=False)
    weights = layer_nn.weight.data.copy()

    x_np = np.random.randn(batch, input_size).astype(np.float32)
    x_nn = Tensor(x_np.copy(), device="cpu", requires_grad=True)

    y_nn = layer_nn(x_nn)
    mu_nn, logvar_nn = y_nn[:, :latent], y_nn[:, latent:]
    std_nn = logvar_nn.mul(0.5).exp()
    eps_nn = neunet.tensor(np.random.normal(0, 1, size=std_nn.shape))
    z_nn = mu_nn + eps_nn * std_nn
    z_nn.backward(np.ones_like(z_nn.data))

    # Torch
    x_t = torch.tensor(x_np.copy(), dtype=torch.float32, requires_grad=True)
    layer_t = torch.nn.Linear(input_size, latent * 2, bias=False)
    layer_t.weight.data = torch.tensor(weights, dtype=torch.float32)

    y_t = layer_t(x_t)
    mu_t, logvar_t = y_t[:, :latent], y_t[:, latent:]
    std_t = torch.exp(0.5 * logvar_t)
    eps_t = torch.tensor(eps_nn.data, dtype=torch.float32)
    z_t = mu_t + eps_t * std_t
    z_t.backward(torch.ones_like(z_t))

    assert np.allclose(z_nn.data, z_t.detach().numpy(), rtol=rtol, atol=atol)
    assert np.allclose(x_nn.grad, x_t.grad.detach().numpy(), rtol=rtol, atol=atol)
