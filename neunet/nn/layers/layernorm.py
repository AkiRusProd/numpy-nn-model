from typing import Literal, Union

import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter

# class LayerNorm(Module): #layer with dynamic backpropagation
#     def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
#         self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
#         self.eps = eps
#         self.elementwise_affine = elementwise_affine

#         if elementwise_affine:
#             self.weight: Union[Tensor, None] = Parameter(neunet.tensor(np.ones((normalized_shape)), dtype=np.float32))
#             self.bias: Union[Tensor, None] = Parameter(neunet.tensor(np.zeros((normalized_shape)), dtype=np.float32))
#         else:
#             self.weight = None
#             self.bias = None

#     def forward(self, X: Tensor):
#         axis = tuple(range(-len(self.normalized_shape), 0))

#         mean = X.mean(axis = axis, keepdims=True)
#         var = X.var(axis = axis, keepdims=True)

#         X_centered = X - mean

#         stddev_inv = 1 / Tensor.sqrt(var + self.eps)

#         O = X_centered * stddev_inv

#         if self.elementwise_affine:
#             O = self.weight * O + self.bias

#         return O

#     def __call__(self, X):
#         return self.forward(X)


class _LayerNormTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, weight: Tensor, bias: Tensor, X_centered, stddev_inv, axis, elementwise_affine, grad):
            # The method of calculating the derivative is similar to BatchNorm.
            _axis = list(axis) if isinstance(axis, tuple) else axis
            X_hat = X_centered * stddev_inv

            weight_data = weight.data if elementwise_affine else 1
            # N = X.xp.prod(X.xp.array(X.shape)[_axis]) # Takes up a lot of GPU memory
            N = np.prod(np.array(X.shape)[_axis])


            dX_hat = weight_data * grad
            dstddev_inv = (
                -0.5
                * X.xp.power(stddev_inv, 3)
                * X.xp.sum(dX_hat * X_centered, axis=axis, keepdims=True)
            )
            dvar = (
                X.xp.ones_like(X.data) * dstddev_inv * 2 * X_centered / N
            )  # X.xp.prod(X.xp.array(X.shape)[_axis])
            dmean = (
                X.xp.ones_like(X.data)
                * X.xp.sum(dX_hat * stddev_inv, axis=axis, keepdims=True)
                * (-1)
                / N
            )  # X.xp.prod(X.xp.array(X.shape)[_axis])
            grad_X = dX_hat * stddev_inv + dvar + dmean

            # grad_X = (1 / N) * weight_data * stddev_inv * (
            #     N * grad
            #     - X.xp.sum(grad, axis = axis, keepdims = True)
            #     - X_centered * X.xp.power(stddev_inv, 2) * X.xp.sum(grad * X_centered, axis = axis, keepdims = True)
            #     )

            # dX_hat = weight_data * grad
            # dvar = X.xp.sum(dX_hat * X_centered, axis = axis, keepdims = True) * (-0.5) * X.xp.power(stddev_inv, 3) 
            # dmean = (X.xp.sum(dX_hat * (-stddev_inv), axis = axis, keepdims = True) + dvar * X.xp.mean(-2.0 * X_centered, axis = axis, keepdims = True)) * X.xp.ones_like(X.data) / N
            # grad_X = dX_hat * stddev_inv + dvar * 2 * X_centered / N + dmean / N

            if elementwise_affine:
                grad_weight = X.xp.sum(grad * X_hat, axis=0)
                grad_bias = X.xp.sum(grad, axis=0)

            X.apply_grad(grad_X)
            if elementwise_affine:
                weight.apply_grad(grad_weight)
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn


class LayerNorm(Module):  # layer with static backpropagation
    def __init__(self, normalized_shape: Union[int, tuple[int]], eps: float=1e-05, elementwise_affine: bool=True, device: Literal["cpu", "cuda"] = "cpu"):
        self.normalized_shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight: Union[Tensor, None] = Parameter(neunet.tensor(np.ones((normalized_shape)), dtype=np.float32))
            self.bias: Union[Tensor, None] = Parameter(neunet.tensor(np.zeros((normalized_shape)), dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = self.xp.mean(X.data, axis=axis, keepdims=True)
        var = self.xp.var(X.data, axis=axis, keepdims=True)

        X_centered = X.data - mean
        stddev_inv = 1 / self.xp.sqrt(var + self.eps)

        O = X_centered * stddev_inv

        if self.elementwise_affine:
            O = self.weight.data * O + self.bias.data # type: ignore

        return _LayerNormTensor(
            O,
            (
                X,
                self.weight,
                self.bias,
                X_centered,
                stddev_inv,
                axis,
                self.elementwise_affine,
            ),
            "layernorm",
            device=self.device,
        )

    def __call__(self, X):
        return self.forward(X)
