from autograd import Tensor
import numpy as np


class _BatchNorm1dTensor(Tensor): #Static BatchNorm1d tensor for backpropagation
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



class BatchNorm1d(): #Static layer
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



# x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])

# x = Tensor(x)
# bn = BatchNorm1d(3)

# bn.train = True
# y = bn(x)

# print(f"y: {y.data}")

# y.backward(np.ones_like(y.data))

# print(x.grad)
# print(bn.weight.grad)
# print(bn.bias.grad)

# class BatchNorm1d(): #Dynamic backward layer
#     def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine

#         self.running_mean = Tensor(np.zeros((1, num_features)), requires_grad = False)
#         self.running_var = Tensor(np.ones((1, num_features)), requires_grad = False)

#         self.weight = Tensor(np.ones((1, num_features)))
#         self.bias = Tensor(np.zeros((1, num_features)))

#         self.train = True

#     def forward(self, X):
#         if self.train:
#             mean = X.mean(axis = 0)#.reshape(1, -1)
#             var = X.var(axis = 0)#.reshape(1, -1)

#             self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean.data #BUG overflow memory when use * between non grad tensor and grad tensor, that's why I use data
#             self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var.data
#         else:
#             mean = self.running_mean
#             var = self.running_var
        
#         X_centered = (X - mean) #errro
        
#         varaddeps = var + self.eps
#         powvaraddeps = varaddeps.power(0.5)
#         stddev_inv = Tensor(1).div(powvaraddeps) #1 / np.sqrt(var + self.eps) BUG

#         X_hat = X_centered * stddev_inv

#         if self.affine:
#             output = self.weight * X_hat + self.bias
#         else:
#             output = X_hat

#         return output

#     def __call__(self, X):
#         return self.forward(X)



# x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])

# x = Tensor(x)
# bn = BatchNorm1d(3)

# bn.train = True
# y = bn(x)

# print(f"y: {y.data}")

# y.backward(np.ones_like(y.data))

# print(x.grad)
# print(bn.weight.grad)
# print(bn.bias.grad)

    
