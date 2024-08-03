from typing import Literal, Union

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter

# class RMSNorm(Module): #layer with dynamic backpropagation
#     """
#     Root Mean Squared Normalization with autograd backward pass.
#     References: 
#     https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
#     https://github.com/meta-llama/llama/blob/main/llama/model.py
#     https://catalyst-team.github.io/catalyst/v20.12/_modules/catalyst/contrib/nn/modules/rms_norm.html
#     """
#     def __init__(self, dim: int, eps: float = 1e-6, device: str = "cpu", bias = False):
#         super().__init__()
#         self.eps = eps
#         self.weight = Parameter(neunet.ones(dim))

#         if bias:
#             self.bias: Union[Parameter, None] = Parameter(neunet.zeros(dim))
#         else:
#             self.bias = None

#         self.to(device)

#     def forward(self, X: Tensor) -> Tensor:
#         X_std = neunet.sqrt(neunet.mean(X ** 2, -1, keepdims=True))
#         X_norm = X / (X_std + self.eps)

#         O = X_norm * self.weight
#         if self.bias is not None:
#             O = O + self.bias

#         return O


class _RMSNormTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, weight: Tensor, bias: Tensor, X_norm, X_std, grad):

            N = X.shape[-1]

            dX_hat = weight.data * grad

            # (U/V)' = (U' * V - U * V') / V^2

            grad_X = (dX_hat * X_std - X.data * X.xp.sum(dX_hat * X.data / X_std, axis = -1, keepdims = True) / N) / X_std ** 2

            X.apply_grad(grad_X)

            grad_weight = X.xp.sum(grad * X_norm, axis=0)
            weight.apply_grad(grad_weight)
            if bias is not None:
                grad_bias = X.xp.sum(grad, axis=0)
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class RMSNorm(Module): #layer with static backpropagation
    """
    Root Mean Squared Normalization with autograd backward pass.
    References: 
    https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    https://catalyst-team.github.io/catalyst/v20.12/_modules/catalyst/contrib/nn/modules/rms_norm.html
    """
    def __init__(self, dim: int, eps: float = 1e-6, device: Literal["cpu", "cuda"] = "cpu", bias = False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(neunet.ones(dim))

        if bias:
            self.bias: Union[Parameter, None] = Parameter(neunet.zeros(dim))
        else:
            self.bias = None

        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        xp = X.xp

        X_std = xp.sqrt(xp.mean(X.data ** 2, -1, keepdims=True))
        X_norm = X.data / (X_std + self.eps)

        O = X_norm * self.weight.data
        if self.bias is not None:
            O = O + self.bias.data

        return _RMSNormTensor(O, (X, self.weight, self.bias, X_norm, X_std), "rmsnorm", device = self.device)

