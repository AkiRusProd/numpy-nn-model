import numpy as np
from autograd import Tensor
from nn import Linear, Sequential, Module, MSELoss, Sigmoid, ReLU, BCELoss

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

print(f'y: {y}')

y.backward()

print(f'x.grad: {x.grad}')
print(f'x2.grad: {x2.grad}')
print(f'x3.grad: {x3.grad}')

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