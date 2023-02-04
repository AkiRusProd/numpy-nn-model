import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from autograd import Tensor
from nn import BatchNorm2d, MaxPool2d

import torch
from torch import nn

# # x_rand = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# # xx = Tensor(x_rand)
# # xx.op = "input"

# # weights = Tensor(np.array([[0.1, 0.2, 0.3]]))

# # # y = xx.mean(axis = 0)

# # y = weights * xx

# # print(y.op, np.ones_like(y), y.shape)

# # y.backward(grad = np.ones_like(y.data))

# # print(xx.grad)

# # x = np.array([[1, 2, 3], [4, 5, 6]])
# # var = np.var(x, axis = 0, ddof=1)

# # print(var)


# x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# var = x.var(axis = 0)
# print(var)
# print(np.ones_like(var.data).shape)
# var.backward(np.ones_like(var.data))


# print(x.grad)



# x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# var = x.var(axis = 0, unbiased=False)
# print(torch.ones_like(var).shape)
# var.backward(torch.ones_like(var))

# print(var)
# print(x.grad)


# # X  = np.array([[1, 2, 3], [4, 5, 6]])
# # X = Tensor(X)

# # Y = 1 / X #BUG
# # # Y = Tensor(1).div(X)
# # # Y = Tensor(1) / X

# # # Y = 2 * X
# # # Y = Tensor(2) * X
# # # Y = X * 2
# # # Y = X / 2

# # print(f"Y:\n {Y.data}")
# # Y.backward()

# # print(f"X.grad:\n {X.grad}")

# # a = np.random.rand(5, 3)
# # b = np.random.rand(1, 3)

# # print(a + b)

# x = Tensor(2)
# l = 2 * x
# nll = l * 3

# z = nll + l 

# z.backward()
# print(x.grad)

# class Dense(Tensor):
#     def __init__(self, in_features, out_features):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         # self.weight = Tensor(np.random.uniform(-stdv, stdv, (in_features, out_features)))
#         self.weight = Tensor(np.ones((in_features, out_features)))
      
#     def forward(self, x):
#         return x.mm(self.weight)

#     def __call__(self, x):
#         return self.forward(x)


# x = Tensor(np.array([[1]]))
# dense1 = Dense(1, 1)
# dense2 = Dense(1, 1)
# dense2.weight.data *= 2

# y = dense1(x)
# y2 = dense2(y)

# z = y + y2
# print(f"z = {z.data}, {y.data}, {y2.data}")

# z.backward(grad = np.ones_like(z.data) * 3)
# print(x.grad)
# print(dense1.weight.grad)
# print(dense2.weight.grad)



# x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# y = x.mean()
# print(y)
# y.backward(np.ones_like(y.data))
# print(x.grad)

# x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# y = x.mean()
# print(y)
# y.backward(torch.ones_like(y))
# print(x.grad)



# x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# x_mean = x.mean(axis=0)
# x_var = x.var(axis = 0)

# z = x_mean + x_var
# z.backward(np.ones_like(z.data))
# print(x.grad)

# x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# x_mean = x.mean(axis=0)
# x_var = x.var(axis = 0, unbiased=False)
# z = x_mean + x_var
# z.backward(torch.ones_like(z))
# print(x.grad)

# x = np.random.rand(5, 3, 4)
# axis = (1, 2)
# axis = list(axis) if isinstance(axis, tuple) else axis
# reps = np.prod(np.array(x.shape)[axis])
# print(reps)
# reps2 = np.prod([x.shape[i] for i in axis])
# print(reps2)

# from nn import LayerNorm
# # x_arr = np.random.randn(2, 3, 3)
# x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# x = Tensor(x_arr)
# ln = LayerNorm(3)
# y = ln(x)
# print(y)

# y.backward(np.ones_like(y.data))
# print("-----------------")
# print(x.grad)
# x_grad = x.grad
# ln_weight_grad = ln.weight.grad
# # print(ln.weight.grad.shape, ln.weight.shape)
# print(ln.weight.grad)
# print(ln.bias.grad)

# x = Tensor(1)
# y = Tensor(2)
# z = Tensor(3)

# l = x + y + z
# l.backward()
# print(x.grad, y.grad, z.grad)


# x = 1
# print(np.ones_like(x))


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



x = np.random.randn(1, 1, 9, 9)
x = np.ones((1, 1, 9, 9))
layer = MaxPool2d(4, 2, 0, dilation=2)
x_t = Tensor(x)
out = layer(x_t)
# print(out)
out.backward(np.ones_like(out.data))
print(x_t.grad)


x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
layer = nn.MaxPool2d(4, 2, 0, dilation=2)

y = layer(x)
print(y)
y.backward(torch.ones_like(y))
print(x.grad)
print(y.data.shape, out.shape)
print(np.allclose(y.data, out.data))
print(np.allclose(x.grad, x_t.grad))
# print(np.isclose(y.data, out.data))


