from neunet.autograd import Tensor
import numpy as np


class _BatchNorm1dTensor(Tensor): # tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        X, weight, bias, X_centered, stddev_inv, affine = self.args
        
        X_hat = X_centered * stddev_inv
        batch_size = X.data.shape[0]

        weight_data = weight.data if affine else 1

        grad_X = (1 / batch_size) * weight_data * stddev_inv * (
            batch_size * grad
            - np.sum(grad, axis = 0)
            - X_centered * np.power(stddev_inv, 2) * np.sum(grad * X_centered, axis = 0)
            )
        
        if affine:
            grad_weight = np.sum(grad * X_hat, axis = 0, keepdims = True)
            grad_bias = np.sum(grad, axis = 0, keepdims = True)

        X.backward(grad_X)
        if affine:
            weight.backward(grad_weight)
            bias.backward(grad_bias)



class BatchNorm1d(): # layer with static backpropagation
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = Tensor(np.zeros((1, num_features)), dtype=np.float32)
        self.running_var = Tensor(np.ones((1, num_features)), dtype=np.float32)

        if affine:
            self.weight = Tensor(np.ones((1, num_features)), dtype=np.float32)
            self.bias = Tensor(np.zeros((1, num_features)), dtype=np.float32)
        else:
            self.weight = None
            self.bias = None

        self.training = True

    def forward(self, X):

        if self.training:
            mean = np.mean(X.data, axis = 0, keepdims = True)
            var = np.var(X.data, axis = 0, keepdims = True)

            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * mean
            self.running_var.data = self.momentum * self.running_var.data + (1 - self.momentum) * var
        else:
            mean = self.running_mean.data
            var = self.running_var.data

        X_centered = X.data - mean
        stddev_inv = 1 / np.sqrt(var + self.eps)
        
        O = X_centered * stddev_inv
       
        if self.affine:
            O = self.weight.data * O + self.bias.data
        
        return _BatchNorm1dTensor(O, [X, self.weight, self.bias, X_centered, stddev_inv, self.affine], "batchnorm")

    def __call__(self, X):
        return self.forward(X)

    def train(self, mode = True):
        self.training = mode

    def eval(self):
        self.training = False



# x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.random.rand(5, 3)

# x = Tensor(x_arr)
# bn = BatchNorm1d(3)

# bn.train = True
# y = bn(x)

# print(f"y: {y.data}")

# y.backward(np.ones_like(y.data))
# x_grad = x.grad
# print(x.grad)
# print(bn.weight.grad)
# print(bn.bias.grad)

# class BatchNorm1d(): #layer with dynamic backpropagation
#     def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True):
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine

#         self.running_mean = Tensor(np.zeros((1, num_features)), requires_grad = False, dtype=np.float32)
#         self.running_var = Tensor(np.ones((1, num_features)), requires_grad = False, dtype=np.float32)

#         if affine:
#             self.weight = Tensor(np.ones((1, num_features)), dtype=np.float32)
#             self.bias = Tensor(np.zeros((1, num_features)), dtype=np.float32)
#         else:
#             self.weight = None
#             self.bias = None

#         self.training = True

#     def forward(self, X):
#         if self.training:
#             mean = X.mean(axis = 0)#.reshape(1, -1)
#             var = X.var(axis = 0)#.reshape(1, -1)

#             self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean.data #BUG overflow memory when use * between non grad tensor and grad tensor, that's why I use data
#             self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var.data
#         else:
#             mean = self.running_mean
#             var = self.running_var
        
#         X_centered = (X - mean) #errro
        
#         stddev_inv = 1 / Tensor.sqrt(var + self.eps)

#         O = X_centered * stddev_inv

#         if self.affine:
#             O = self.weight * O + self.bias

#         return O

#     def __call__(self, X):
#         return self.forward(X)

    # def train(self, mode = True):
    #     self.training = mode

    # def eval(self):
    #     self.training = False


# # x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])

# x = Tensor(x_arr)
# bn = BatchNorm1d(3)

# bn.train = True
# y = bn(x)

# print(f"y: {y.data}")

# y.backward(np.ones_like(y.data))

# print(x.grad)
# print(bn.weight.grad)
# print(bn.bias.grad)

# print(np.allclose(x.grad, x_grad))
