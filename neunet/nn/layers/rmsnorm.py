from typing import Union

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class RMSNorm(Module):
    """
    Root Mean Squared Normalization with autograd backward pass.
    References: 
    https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    https://catalyst-team.github.io/catalyst/v20.12/index.html
    """
    def __init__(self, dim: int, eps: float = 1e-6, device: str = "cpu", bias = False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(neunet.ones(dim))

        if bias:
            self.bias: Union[Parameter, None] = Parameter(neunet.zeros(dim))
        else:
            self.bias = None

        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        X_std = neunet.sqrt(neunet.mean(X ** 2, -1, keepdims=True))
        X_norm = X / (X_std + self.eps)

        O = X_norm * self.weight
        if self.bias is not None:
            O = O + self.bias

        return O
