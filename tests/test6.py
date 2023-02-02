from autograd import Tensor
import numpy as np






class LayerNorm():
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = Tensor(np.ones((normalized_shape)))
        self.bias = Tensor(np.zeros((normalized_shape)))

    def forward(self, X):
        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = X.mean(axis = axis, keepdims=True)
        var = X.var(axis = axis, keepdims=True)

        X_centered = X + mean
        varaddeps = var + self.eps
        powvaraddeps = varaddeps.power(0.5)
        stddev_inv = Tensor(1).div(powvaraddeps) #1 / np.sqrt(var + self.eps) BUG

        O = X_centered * stddev_inv

        if self.elementwise_affine:
            O = self.weight * O + self.bias
        
        return O

    def __call__(self, X):
        return self.forward(X)

x_arr = np.random.randn(2, 3, 3)
x = Tensor(x_arr)
ln = LayerNorm(3)
y = ln(x)
print(y)

y.backward(np.ones_like(y.data))
print("-----------------")
x_grad = x.grad
print(x.grad)
print(ln.weight.grad)
print(ln.bias.grad)

import torch
import torch.nn as nn
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = nn.Parameter(torch.ones((normalized_shape)))
        self.bias = nn.Parameter(torch.zeros((normalized_shape)))

    def forward(self, X):
        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = X.mean(axis, keepdims=True)
        var = X.var(axis, keepdims=True, unbiased=False)
        sum = X.sum(axis, keepdims=True)

        # axis = 0
        # mean = X.mean(axis)
        # var = X.var(axis, unbiased=False)
        # sum = X.sum(axis)

        X_centered = X + mean
        varaddeps = var + self.eps
        powvaraddeps = varaddeps.pow(0.5)
        stddev_inv = 1 / powvaraddeps

        O = X_centered * stddev_inv

        # if self.elementwise_affine:
        #     O = self.weight * O + self.bias
        
        return O

    def __call__(self, X):
        return self.forward(X)


ln = LayerNorm(3)
x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
y = ln(x)
print(y)

y.backward(torch.ones_like(y.data))
print("-----------------")
xt_grad = x.grad
print(x.grad)
print(ln.weight.grad)
print(ln.bias.grad)

print(x_grad / xt_grad)


# test torch mean and Tensor mean
# x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.random.randn(2, 3, 4, 5)
# x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)


# x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
# print("X shape: ", x.shape)
# z = 1 - x.mean(axis = (1, 2))
# print(z.shape)
# z.backward(torch.ones_like(z.data))
# print("X grad: ", x)


# x = Tensor(x_arr)


# z = 1 - x.mean(axis = (1, 2))
# print(z.shape, np.ones_like(z.data).shape)
# z.backward(np.ones_like(z.data))
# print("X grad: ", x)
# x = Tensor(x_arr)
# eps = 1e-5

# mean = x.mean(axis = 0)
# var = x.var(axis = 0)


# x_c = x + mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# varaddeps = var + eps
# powvaraddeps = varaddeps.power(0.5)
# stddev_inv = Tensor(1).div(powvaraddeps)
# # stddev_inv = var + eps

# x_hat = x_c * stddev_inv

# x_hat.backward(np.ones_like(x_hat.data))
# print("X_HAT", x_hat.data)
# x_grad = x.grad

# print("x.grad", x_grad)
# print("--------------------")
# eps = 1e-5
# x = torch.tensor(x_arr, requires_grad = True)

# mean = x.mean(axis = 0)
# var = x.var(axis = 0, unbiased=False)

# x_c = x + mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# varaddeps = var + eps
# powvaraddeps = varaddeps.pow(0.5)
# stddev_inv = 1 / powvaraddeps
# print(mean.shape, x_c.shape)
# x_hat = x_c * stddev_inv

# x_hat.backward(torch.ones_like(x_hat.data))
# print("X_HAT", x_hat.data)
# print("x.grad", x.grad)