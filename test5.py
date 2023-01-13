import numpy as np
from autograd import Tensor

import torch
from torch import nn

eps = 1e-5
num_features = 3

x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(x_arr.shape)

weight =  Tensor(np.ones((1, num_features)))
bias = Tensor(np.zeros((1, num_features)))
X = Tensor(x_arr)

mean = X.mean(axis = 0)
var = X.var(axis = 0)

X_centered = (X - mean) #errro
        
varaddeps = var + eps
powvaraddeps = varaddeps.power(0.5)
stddev_inv = Tensor(1).div(powvaraddeps) #1 / np.sqrt(var + self.eps) BUG

X_hat = X_centered * stddev_inv


X_hat.backward(np.ones_like(X_hat.data))
print(X.grad)

print("--------------------")
weight = torch.ones((1, num_features))
bias = torch.zeros((1, num_features))
X = torch.tensor(x_arr, requires_grad = True)

mean = X.mean(axis = 0)
var = X.var(axis = 0, unbiased = False)


X_centered = (X - mean) #errro
        
varaddeps = var + eps
powvaraddeps = varaddeps.pow(0.5)
stddev_inv = 1 / powvaraddeps

X_hat = X_centered * stddev_inv



X_hat.backward(torch.ones_like(X_hat.data))
print(X.grad)




