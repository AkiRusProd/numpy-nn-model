import numpy as np
from neunet.autograd import Tensor
from neunet.nn import Linear, Sequential, Module, MSELoss, Sigmoid, ReLU, BCELoss

import torch


# latent_size = 2
# multiplier = 2
# x = Tensor(np.array([[1, 0], [0, 1]]))
# y = Tensor(np.array([[1], [0]]))

# # Linear_weight = np.arange(latent_size * 2 * multiplier).reshape((latent_size * multiplier, 2))
# Linear_weight = np.ones((latent_size * multiplier, 2))

# Linear1 = Linear(2, latent_size * multiplier)
# Linear1.weight.data = Linear_weight

# relu = ReLU()
# sigmoid = Sigmoid()
# loss_fn= MSELoss()

# encoder = Sequential(Linear1)
# loss_fn= MSELoss()

# x1 = encoder(x)
# print("x1: ", x1)

# # x1_split = x1.split(latent_size, axis=1)
# x1_split, x2_split = x1.split(2, axis=1)

# x3_split = x1_split.add(x2_split)

# loss = loss_fn(x3_split, y)
# print("loss: ", loss)

# loss.backward()


# x = Tensor(np.array([[1, 0], [0, 1]]))


# y = x * 2
# print(f'y: {y}')
# z = x * 3

# print(f'z: {z}')


# f = y + z

# print(f'f: {f}')

# f.backward()

# print(f'x.grad: {x.grad}')


x = Tensor(np.array([[1, 0], [0, 1]]))
x2 = Tensor(np.array([[1, 0], [0, 1]]))
x3 = Tensor(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]).T)
print(f"x.shape: {x.shape}")
print(f"x2.shape: {x2.shape}")
print(f"x3.shape: {x3.shape}")

y = x.concatenate(x2, x2, x3, axis=1)
print(y.shape)

print(f"y: {y}")

y.backward(np.ones_like(y.data))

print(f"x.grad: {x.grad}")
print(f"x2.grad: {x2.grad}")
print(f"x3.grad: {x3.grad}")

x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32, requires_grad=True)
x2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32, requires_grad=True)
x3 = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32).T
print(f"x.shape: {x.shape}")
print(f"x2.shape: {x2.shape}")
print(f"x3.shape: {x3.shape}")

y = torch.cat((x, x2, x2, x3), dim=1)
print(y)

y.backward(torch.ones_like(y))

print(x.grad)
print(x2.grad)


def reverse_concatenate(shapes, axis=0):
    """
    Reverse the operation of concatenate
    :param shapes: a list of shapes
    :param axis: the axis to concatenate on
    :return: a list of slices
    """
    slices = []
    start = 0
    for shape in shapes:
        end = start + shape[axis]
        slices.append((start, end))
        start = end
    return slices


print(reverse_concatenate([x.shape, x2.shape, x2.shape, x3.shape], axis=1))

# x = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])

# x0, x1 = x[:, 0], x[:, 1]
# print(x0, x1)
# y = x0 * x1
# y = y.reshape(4, 1)


# print(x.shape, y.shape)

# y.backward()

# print(f"x.grad: {x.grad}")

# x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
# x0, x1 = x[:, 0], x[:, 1]
# print(x0, x1)
# y = x0 * x1
# y = y.reshape(4, 1)


# x = Tensor(1)
# y = Tensor(2)

# z = x + y

# z.backward()

# print(f'x.grad: {x.grad}')

x = Tensor(np.random.randn(2, 3))
y = Tensor(np.random.randn(1, 3))
z = x / y
print(z.shape)
print(x.data.shape)
z.backward(np.ones_like(z.data))
print(x.grad)
print(y.grad)
# print(np.sum(x.grad, axis=0))

# ####################

import neunet as nnet

import torch
import torch.nn as nn


# x = np.random.randn(1, 2, 3 ,4)
# print(x.shape)


print("####################")
x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)
# x = torch.tensor([1.0], requires_grad=True)
# y = torch.tensor([2.0], requires_grad=True)
z = torch.matmul(x, y)
print(z.shape)

z.backward(torch.ones_like(z))

print(x.grad)
print(y.grad)

# mat = torch.randn(2, 3)
# vec = torch.randn(3)
# OUT = torch.mv(mat, vec)
# OUT2 = torch.matmul(mat, vec)
# print(OUT)
# print(OUT2)
x = nnet.tensor(x.detach().numpy(), requires_grad=True)
y = nnet.tensor(y.detach().numpy(), requires_grad=True)
z = nnet.matmul(x, y)

print(z.ndim)

z.backward()

print(x.grad)
print(y.grad)
print(y.grad.shape, "y.grad.shape")


linear = nn.Linear(2, 3)
x = torch.randn(2, 2, 2)
y = linear(x)
print(y.shape)
y.backward(torch.ones_like(y))
print(linear.weight.grad.shape)

linear = Linear(2, 3)
x = nnet.tensor(np.random.randn(2, 2, 2), requires_grad=True)
y = linear(x)
print(y.shape, "y.shape")
y.backward(np.ones_like(y.data))
print(linear.weight.grad.shape)


x = nnet.rand((1, 2, 3))
print(x.shape, "x.shape")
