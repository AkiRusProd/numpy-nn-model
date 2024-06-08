from typing import Union

import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class _BatchNorm2dTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

    def backward(self, grad=1):
        X, weight, bias, X_centered, stddev_inv, affine = self.args

        batch_size = X.data.shape[0] * X.data.shape[2] * X.data.shape[3]

        axis = (0, 2, 3)
        # _axis = list(axis) if isinstance(axis, tuple) else axis
        X_hat = X_centered * stddev_inv[..., None, None]

        weight_data = weight.data[..., None, None] if affine else 1

        dX_hat = weight_data * grad
        dstddev_inv = (
            -0.5
            * self.xp.power(stddev_inv[..., None, None], 3)
            * self.xp.sum(dX_hat * X_centered, axis=axis, keepdims=True)
        )
        dvar = (
            self.xp.ones_like(X.data) * dstddev_inv * 2 * X_centered / batch_size
        )  # self.xp.prod(self.xp.array(X.shape)[_axis])
        dmean = (
            self.xp.ones_like(X.data)
            * self.xp.sum(dX_hat * stddev_inv[..., None, None], axis=axis, keepdims=True)
            * (-1)
            / batch_size
        )  # self.xp.prod(self.xp.array(X.shape)[_axis])
        grad_X = dX_hat * stddev_inv[..., None, None] + dvar + dmean

        if affine:
            grad_weight = self.xp.sum(grad * X_hat, axis=(0, 2, 3), keepdims=True).reshape(
                weight.data.shape
            )
            grad_bias = self.xp.sum(grad, axis=(0, 2, 3), keepdims=True).reshape(bias.data.shape)

        X.backward(grad_X)
        if affine:
            weight.backward(grad_weight)
            bias.backward(grad_bias)


class BatchNorm2d(Module):  # layer with static backpropagation
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, device: str="cpu"):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = Tensor(np.zeros((1, num_features)), dtype=np.float32)
        self.running_var = Tensor(np.ones((1, num_features)), dtype=np.float32)

        if affine:
            self.weight: Union[Tensor, None] = Parameter(neunet.tensor(np.ones((1, num_features)), dtype=np.float32))
            self.bias: Union[Tensor, None] = Parameter(neunet.tensor(np.zeros((1, num_features)), dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

        self.training = True
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        if self.training:
            mean = self.xp.mean(X.data, axis=(0, 2, 3))
            var = self.xp.var(X.data, axis=(0, 2, 3))

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean.data
            var = self.running_var.data

        X_centered = X.data - mean[..., None, None]
        stddev_inv = 1 / self.xp.sqrt(var + self.eps)

        O = X_centered * stddev_inv[..., None, None]

        if self.affine:
            O = self.weight.data[..., None, None] * O + self.bias.data[..., None, None] # type: ignore

        return _BatchNorm2dTensor(
            O,
            [X, self.weight, self.bias, X_centered, stddev_inv, self.affine],
            "batchnorm2d",
            device=self.device,
        )

    def __call__(self, X):
        return self.forward(X)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False


# class BatchNorm2d(): #layer with dynamic backpropagation
#     def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine

#         self.running_mean = Tensor(self.xp.zeros((1, num_features)), dtype=self.xp.float32)
#         self.running_var = Tensor(self.xp.ones((1, num_features)), dtype=self.xp.float32)

#         if affine:
#             self.weight = Tensor(self.xp.ones((1, num_features)), dtype=self.xp.float32)
#             self.bias = Tensor(self.xp.zeros((1, num_features)), dtype=self.xp.float32)
#         else:
#             self.weight = None
#             self.bias = None

#         self.training = True

#     def forward(self, X):

#         if self.training:
#             mean = X.mean(axis = (0, 2, 3))
#             var = X.var(axis = (0, 2, 3))


#             self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.data
#             self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.data
#         else:
#             mean = self.running_mean
#             var = self.running_var

#         X_centered = X - mean[..., None, None]

#         stddev_inv = 1 / Tensor.sqrt(var + self.eps)

#         O = X_centered * stddev_inv[..., None, None]

#         if self.affine:
#             O = self.weight[..., None, None] * O + self.bias[..., None, None]


#         return O

#     def __call__(self, X):
#         return self.forward(X)

# def train(self, mode = True):
#     self.training = mode

# def eval(self):
#     self.training = False

# x_rand = self.xp.random.randn(2, 3, 2, 2)
# x_rand = self.xp.arange(0, 24).reshape(2, 3, 2, 2)
# x = Tensor(x_rand)
# bn = BatchNorm2d(3)

# bn.train = True
# y = bn(x)

# print(y.data)

# y.backward(self.xp.ones_like(y.data))

# print(x.grad)

# import torch
# import torch.nn as nn


# x = torch.tensor(x_rand, requires_grad=True, dtype=torch.float32)
# bn = nn.BatchNorm2d(3)

# bn.train()
# y = bn(x)
# # print(y)

# y.backward(torch.ones_like(y))

# print(x.grad)
