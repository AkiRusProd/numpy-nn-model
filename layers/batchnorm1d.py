from autograd import Tensor
import numpy as np


class _BatchNorm1dTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.X, self.weight, self.bias, self.X_centered, self.stddev_inv = self.args
        
        X_hat = self.X_centered * self.stddev_inv
        batch_size = self.X.data.shape[0]

        grad_O = (1 / batch_size) * self.weight.data * self.stddev_inv * (
            batch_size * grad
            - np.sum(grad, axis = 0)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(grad * self.X_centered, axis = 0)
            )

        grad_weight = np.sum(grad * X_hat, axis = 0, keepdims = True)
        grad_bias = np.sum(grad, axis = 0, keepdims = True)

        self.X.backward(grad_O)
        self.weight.backward(grad_weight)
        self.bias.backward(grad_bias)



class BatchNorm1d():
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
            self.mean = np.mean(self.X.data, axis = 0, keepdims = True)
            self.var = np.var(self.X.data, axis = 0, keepdims = True)

            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * self.mean
            self.running_var.data = self.momentum * self.running_var.data + (1 - self.momentum) * self.var
        else:
            self.mean = self.running_mean.data
            self.var = self.running_var.data

        self.X_centered = self.X.data - self.mean
        self.stddev_inv = 1 / np.sqrt(self.var + self.eps)

        self.O = self.X_centered * self.stddev_inv

        if self.affine:
            self.O = self.weight.data * self.O + self.bias.data

        return _BatchNorm1dTensor(self.O, [self.X, self.weight, self.bias, self.X_centered, self.stddev_inv], "batchnorm")

    def __call__(self, X):
        return self.forward(X)