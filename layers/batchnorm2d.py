from autograd import Tensor
import numpy as np


class _BatchNorm2dTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        X, weight, bias, X_centered, stddev_inv = self.args
        
        X_hat = X_centered * stddev_inv[..., None, None]
        batch_size = X.data.shape[0] * X.data.shape[2] * X.data.shape[3]

        dX_hat =  grad * weight.data[..., None, None]
        dvar = (-0.5 * dX_hat * X_centered).sum((0, 2, 3), keepdims=True)  * (stddev_inv[..., None, None] ** 3.0)
        dmu = (- stddev_inv[..., None, None] * dX_hat).sum((0, 2, 3), keepdims = True) + (dvar * (-2.0 * X_centered).sum((0, 2, 3), keepdims = True) / batch_size)

        grad_X = stddev_inv[..., None, None] * dX_hat + dvar * (2.0 * X_centered / batch_size) + dmu / batch_size

        grad_weight = np.sum(grad * X_hat, axis = (0, 2, 3), keepdims = True).reshape(weight.data.shape)
        grad_bias = np.sum(grad, axis = (0, 2, 3), keepdims = True).reshape(bias.data.shape)

        X.backward(grad_X)
        weight.backward(grad_weight)
        bias.backward(grad_bias)



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

        if self.train:
            mean = np.mean(X.data, axis = (0, 2, 3))
            var = np.var(X.data, axis = (0, 2, 3))
 

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        X_centered = X.data - mean[..., None, None]
        stddev_inv = 1 / np.sqrt(var + self.eps)

        O = X_centered * stddev_inv[..., None, None]

        if self.affine:
            O = self.weight.data[..., None, None] * O + self.bias.data[..., None, None]

        
        return _BatchNorm2dTensor(O, [X, self.weight, self.bias, X_centered, stddev_inv], "batchnorm2d")

    def __call__(self, X):
        return self.forward(X)


# x_rand = np.random.randn(2, 3, 2, 2)
# x_rand = np.arange(0, 24).reshape(2, 3, 2, 2)
# x = Tensor(x_rand)
# bn = BatchNorm2d(3)

# bn.train = True
# y = bn(x)

# print(y.data)

# y.backward(np.ones_like(y.data))

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







