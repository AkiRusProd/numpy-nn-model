import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)



import numpy as np
from neunet.autograd import Tensor

import torch
from torch import nn

"""
eps = 1e-5
num_features = 3

x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(x_arr.shape)

weight =  Tensor(np.ones((1, num_features)))
bias = Tensor(np.zeros((1, num_features)))
X = Tensor(x_arr)

mean = X.mean(axis = 0)
var = X.mean(axis = 0)

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
var = X.mean(axis = 0)


X_centered = (X - mean) #errro
        
varaddeps = var + eps
powvaraddeps = varaddeps.pow(0.5)
stddev_inv = 1 / powvaraddeps

X_hat = X_centered * stddev_inv



X_hat.backward(torch.ones_like(X_hat.data))
print(X.grad)





print("--------------------")
"""
x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x_arr = np.random.randn(2, 3, 3)
x = Tensor(x_arr)
eps = 1e-5

mean = x.mean(axis = -1, keepdims=True)
var = x.var(axis = -1,  keepdims=True)


x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
varaddeps = var + eps
powvaraddeps = varaddeps.power(0.5)
stddev_inv = Tensor(1).div(powvaraddeps)
# stddev_inv = var + eps

x_hat = x_c * stddev_inv

x_hat.backward(np.ones_like(x_hat.data))
print("X_HAT", x_hat.data)
x_grad = x.grad

print("x.grad", x_grad)

print("--------------------")

x = torch.tensor(x_arr, requires_grad = True)

mean = x.mean(axis = -1, keepdim=True)
var = x.var(axis = -1, unbiased=False, keepdim=True)

x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
varaddeps = var + eps
powvaraddeps = varaddeps.pow(0.5)
stddev_inv = 1 / powvaraddeps
print(mean.shape, x_c.shape)
x_hat = x_c * stddev_inv

x_hat.backward(torch.ones_like(x_hat.data))
print("X_HAT", x_hat.data)
print("x.grad", x.grad)
# print(x_grad / x.grad)


def test_mean():
    print("test_mean")
    # x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
    # x_arr = np.array([[1.0, 2.0, 3.0]])
    x_arr = 1.0
    x = Tensor(x_arr)
    val = 5#x.sum()
    x_c = x.sub(val) # Tensor fix bug частично
    x_hat = x_c.add(x_c) # Tensor fix bug частично
    x_hat.backward(np.ones_like(x_hat.data))
    print("X_HAT", x_hat.data)
    x_grad = x.grad

    print("x.grad", x_grad)

    print("--------------------")

    x = torch.tensor(x_arr, requires_grad = True)

    val = 5#x.sum()
    x_c = x - val

    x_hat = x_c + x_c

    x_hat.backward(torch.ones_like(x_hat.data))
    print("X_HAT", x_hat.data)
    print("x.grad", x.grad)


    # print("--------------------")
    # print(x_grad / x.grad)


# test_mean()


# def test_mean2():
#     x_arr = 1.0
#     x = Tensor(x_arr)

#     x_hat = x.add(x)

#     x_hat.backward(np.ones_like(x_hat.data))
#     print("X_HAT", x_hat.data)
#     x_grad = x.grad

#     print("x.grad", x_grad)

#     print("--------------------")

#     x = torch.tensor(x_arr, requires_grad = True)

#     x_hat = x.add(x)

#     x_hat.backward(torch.ones_like(x_hat.data))
#     print("X_HAT", x_hat.data)
#     print("x.grad", x.grad)


#     print("--------------------")
#     print(x_grad / x.grad)

# test_mean2()


def test_mean3():
    x = Tensor([1])
    # x = Tensor(1)
    print(x.data.shape)




    val = 5

    x_c = x.add(val)
    x_h = x_c.add(x_c)
    x_h.backward() #if grad np array - bug
    print(x_h.data)
    print(x.grad)

# test_mean3()


# def test_mean4():
#     x = Tensor([1])
#     # x = Tensor(1)
#     print(x.data.shape)

#     val = 5

#     x_c = x - val
#     x_h = x_c * x_c
#     x_h.backward() #if grad np array - bug
#     print(x_h.data)
#     print(x.grad)

#     x = torch.tensor([1], requires_grad = True, dtype = torch.float32)
#     val = 5

#     x_c = x - val
#     x_h = x_c * x_c
#     x_h.backward() #if grad np array - bug
#     print(x_h.data)
#     print(x.grad)


# test_mean4()

# x_arr = np.array([[0.1, 0.2, 0.3]])
# x_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x_arr = np.random.randn(2, 3, 3)
x = Tensor(x_arr)
eps = 1e-5

mean = x.mean()
var = x.var()


x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# varaddeps = var + eps
# powvaraddeps = varaddeps.power(0.5)
# stddev_inv = Tensor(1).div(powvaraddeps)
# stddev_inv = var + eps

# x_hat = x_c * mean# * mean
# x_hat = (x - mean) * mean# * mean
x_hat = x * mean - mean * mean
# print(mean, x)

x_hat.backward(np.ones_like(x_hat.data))
print("X_HAT", x_hat.data)
x_grad = x.grad

print("x.grad", x_grad)

print("--------------------")

x = torch.tensor(x_arr, requires_grad = True)

mean = x.mean()
var = x.var(unbiased=False)

x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# varaddeps = var + eps
# powvaraddeps = varaddeps.pow(0.5)
# stddev_inv = 1 / powvaraddeps
# print(mean.shape, x_c.shape)
x_hat = x_c * mean# * mean

x_hat.backward(torch.ones_like(x_hat.data))
# grad check



print("X_HAT", x_hat.data)
print("x.grad", x.grad)
print(x_grad / x.grad)