from autograd import Tensor
import numpy as np


class _BatchNorm2dTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.X, self.weight, self.bias, self.X_centered, self.stddev_inv = self.args
        
        X_hat = self.X_centered * self.stddev_inv[..., None, None]
        batch_size = self.X.data.shape[0] * self.X.data.shape[2] * self.X.data.shape[3]

        dX_hat =  grad * self.weight.data[..., None, None]
        dvar = (-0.5 * dX_hat * self.X_centered).sum((0, 2, 3), keepdims=True)  * (self.stddev_inv[..., None, None] ** 3.0)
        dmu = (- self.stddev_inv[..., None, None] * dX_hat).sum((0, 2, 3), keepdims = True) + (dvar * (-2.0 * self.X_centered).sum((0, 2, 3), keepdims = True) / batch_size)

        grad_O = self.stddev_inv[..., None, None] * dX_hat + dvar * (2.0 * self.X_centered / batch_size) + dmu / batch_size

        grad_weight = np.sum(grad * X_hat, axis = (0, 2, 3), keepdims = True)
        grad_bias = np.sum(grad, axis = (0, 2, 3), keepdims = True)

        self.X.backward(grad_O)
        self.weight.backward(grad_weight)
        self.bias.backward(grad_bias)



class BatchNorm2d():
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = Tensor(np.zeros((1, num_features)))
        self.running_var = Tensor(np.ones((1, num_features)))

        self.weight = Tensor(np.ones((1, num_features)))
        self.bias = Tensor(np.zeros((1, num_features)))

        self.train = True

    def forward(self, X):
        self.X = X

        if self.train:
            self.mean = np.mean(self.X.data, axis = (0, 2, 3))
            self.var = np.var(self.X.data, axis = (0, 2, 3))
 

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.X_centered = self.X.data - self.mean[..., None, None]
        self.stddev_inv = 1 / np.sqrt(self.var + self.eps)

        self.O = self.X_centered * self.stddev_inv[..., None, None]

        if self.affine:
            self.O = self.weight.data[..., None, None] * self.O + self.bias.data[..., None, None]

        
        return _BatchNorm2dTensor(self.O, [self.X, self.weight, self.bias, self.X_centered, self.stddev_inv], "batchnorm2d")

    def __call__(self, X):
        return self.forward(X)


# x_rand = np.random.randn(2, 3, 2, 2)
x_rand = np.arange(0, 24).reshape(2, 3, 2, 2)
x = Tensor(x_rand)
bn = BatchNorm2d(3)

bn.train = True
y = bn(x)

print(y.data)

y.backward(np.ones_like(y.data))

# print(x.grad)

import torch
import torch.nn as nn


x = torch.tensor(x_rand, requires_grad=True, dtype=torch.float32)
bn = nn.BatchNorm2d(3)

bn.train()
y = bn(x)
# print(y)

y.backward(torch.ones_like(y))

# print(x.grad)







