import numpy as np
from autograd import Tensor
from nn import Linear, MSELoss, Sigmoid, LeakyReLU, Tanh, ReLU, BCELoss
from optim import SGD, Adam

# x = Tensor(5)
# y = Tensor(1)
# z = Tensor(3)

# a = x.add(y)
# b = a.mul(z)

# b.backward()

# print(x.grad)
# print(y.grad)
# print(z.grad)




# dense = Dense(3, 2)
# x = Tensor(np.random.randn(3, 3))
# y = dense(x)

# loss = MSELoss()
# y_true = Tensor(np.random.randn(3, 2))
# l = loss(y, y_true)

# l.backward()

# print(dense.weight.grad)
# print(dense.bias.grad)

model = Linear(2, 1)
loss_fn = MSELoss()
sigmod = Sigmoid()

x = Tensor(np.array([[1, 0], [0, 1]]))
y = Tensor(np.array([[1], [0]]))

y_pred = model(x)
y_pred = sigmod(y_pred)
loss = loss_fn(y_pred, y)
print(x.data.shape, y.data.shape, y_pred.data.shape, loss.data.shape)

# loss.backward()

# print(model.weight.grad)
# print(model.bias.grad)

# epochs = 5000
# learning_rate = 0.01

# for i in range(epochs):
#     y_pred = model(x)
#     y_pred = sigmod(y_pred)
#     loss = loss_fn(y_pred, y)
#     loss.backward()

#     # print(model.bias.data.shape, model.bias.grad.shape)
#     model.weight.data -= learning_rate * model.weight.grad
#     model.bias.data -= learning_rate * model.bias.grad.sum()
#     # print(model.bias.data.shape, model.bias.grad.sum().shape, model.bias.grad.shape)

#     model.weight.grad = 0
#     model.bias.grad = 0

#     if i % 100 == 0:
#         print(loss.data)


from torch import nn
import torch
import torch.nn.functional as F



# model_l1 = nn.Linear(2, 3)
# model_l2 = nn.Linear(3, 1)
# loss_fn = nn.MSELoss()
# sigmod = nn.Sigmoid()

# x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
# y = torch.tensor([[1], [1], [0], [0]], dtype=torch.float32)

# epochs = 1000
# learning_rate = 0.1

# for i in range(epochs):
#     y_pred = model_l1(x)
#     y_pred = model_l2(y_pred)
#     y_pred = sigmod(y_pred)
#     loss = loss_fn(y_pred, y)
#     loss.backward()

#     with torch.no_grad():
#         model_l1.weight -= learning_rate * model_l1.weight.grad
#         model_l1.bias -= learning_rate * model_l1.bias.grad
#         model_l2.weight -= learning_rate * model_l2.weight.grad
#         model_l2.bias -= learning_rate * model_l2.bias.grad

#         # zero the gradients
#         model_l1.weight.grad.zero_()
#         model_l1.bias.grad.zero_()
#         model_l2.weight.grad.zero_()
#         model_l2.bias.grad.zero_()


#     if i % 100 == 0:
#         print(loss.data)

# sigmod = Sigmod()
# x = Tensor(1)
# y = sigmod(x)
# print(y.data)


loss_fn = MSELoss()
x = Tensor([2])
y = Tensor([1])
loss = loss_fn(x, y)
loss.backward()

print(f'x = {x}, y = {y}')
print(x.grad)
print(loss.data)


loss_fn = nn.MSELoss()
x = torch.tensor(2, dtype=torch.float32, requires_grad=True)
y = torch.tensor(1, dtype=torch.float32, requires_grad=True)
loss = loss_fn(x, y)
loss.backward()
print(x.grad)
print(loss.data)


x = Tensor(np.array([1, 2, 3, 4]))

# split the data into 2 parts
x1, x2 = x.split(2)
# print(f'x1 = {x1.data}, x2 = {x2}')
# x1.backward()
# z = x1.add(x2)
# print(z.data)

# z.backward()

# print(x1.grad)

y0, y1 = x.split(2)
z = y0.add(y1)
print(z.data, y0.data, y1.data)

z.backward()

print(y0.grad)
print(y1.grad)
print(x.grad)


y = x.split(2)
z = y[0].add(y[1])
print(z.data, y[0].data, y[1].data)

z.backward()
y[0]
print(y[0].grad)
print(x.grad)

print(x.grad.shape)



# x = Tensor(np.array([1, 2, 3, 4]))
# y = x.split(2)
# y[0].backward()
# print(x.grad)


# x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
# y0, y1 = x.split(2)
# y0.backward()
# print(x.grad)

# x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
# # y = x + 2
# # z = y * y * 3
# # out = z.mean()
# print(x[:, 1], x[:, 0])
# out, out2 = x[:, 1], x[:, 0]
# out.backward()
# print(x.grad)
# print(out.grad)

# x = torch.ones(2, 2, requires_grad=True, retains_grad=True)
# print(x)
# y = x + 2
# print(y)
# z = y * y * 3
# out = z.mean()
# print(z, out)

# out.backward()
# print(x[0].grad)

x = np.random.randn(2, 2, 3, 3)
x2d = Tensor(x)
y = Tensor(np.ones((2, 2, 3, 3)))

loss = MSELoss()
out = loss(x2d, y)
out.backward()
grad_x = x2d.grad
print(grad_x)

x2d = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y = torch.tensor(np.ones((2, 3, 3)), dtype=torch.float32, requires_grad=True)
loss = torch.nn.MSELoss()
out = loss(x2d, y)
out.backward()
grad_x_torch = x2d.grad
print(grad_x_torch)

print(grad_x / grad_x_torch)

layer1 = nn.Linear(2, 3)
layer2 = nn.Linear(3, 2)
sigmod = nn.Sigmoid()

x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
y = torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=torch.float32)


l1_weight = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
l1_bias = np.array([0.1, 0.2, 0.3])
l2_weight = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
l2_bias = np.array([0.1, 0.2])
print(l1_weight.shape, l1_bias.shape, l2_weight.shape, l2_bias.shape)
# flip on 180 degree
# l1_weight = np.rot90(l1_weight, 2).copy()
# l2_weight = np.rot90(l2_weight, 2).copy()
# l1_bias = l1_bias[::-1].copy()
# l2_bias = l2_bias[::-1].copy()

print(l1_weight.shape, l1_bias.shape, l2_weight.shape, l2_bias.shape)



layer1.weight = torch.nn.Parameter(torch.tensor(l1_weight, dtype=torch.float32))
layer1.bias = torch.nn.Parameter(torch.tensor(l1_bias, dtype=torch.float32))
layer2.weight = torch.nn.Parameter(torch.tensor(l2_weight, dtype=torch.float32))
layer2.bias = torch.nn.Parameter(torch.tensor(l2_bias, dtype=torch.float32))
print(layer1.weight.shape, layer1.bias.shape, layer2.weight.shape, layer2.bias.shape)

loss_fn = nn.BCELoss()

x = layer1(x)
x = layer2(x)
x = sigmod(x)
loss = loss_fn(x, y)
print("-----------------")
print(x.data.shape)

loss.backward()

print("-----------------")
print(layer1.weight.grad)
print(layer1.bias.grad)
print(layer2.weight.grad)
print(layer2.bias.grad)


layer1 = Linear(2, 3)
layer2 = Linear(3, 2)
sigmod = Sigmoid()

x = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
y = Tensor([[1, 0], [0, 1], [0, 0], [1, 1]])


l1_weight = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
l1_bias = np.array([0.1, 0.2, 0.3])
l2_weight = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
l2_bias = np.array([0.1, 0.2])
print(l1_weight.shape, l1_bias.shape, l2_weight.shape, l2_bias.shape)


layer1.weight = Tensor(l1_weight)
layer1.bias = Tensor(l1_bias)
layer2.weight = Tensor(l2_weight)
layer2.bias = Tensor(l2_bias)
print(layer1.weight.shape, layer1.bias.shape, layer2.weight.shape, layer2.bias.shape)

loss_fn = BCELoss()

x = layer1(x)
x = layer2(x)
x = sigmod(x)
loss = loss_fn(x, y)

loss.backward()
print("-----------------")
print(x.data.shape)

print("-----------------")
print(layer1.weight.grad, '\n')

print(layer1.bias.grad, '\n')

print(layer2.weight.grad, '\n')

print(layer2.bias.grad)



# x = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
# y = Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]])

# linear = Linear(2, 3)
# sigmod = Sigmoid()
# loss_fn = MSELoss()

# x = linear(x)
# x = sigmod(x)


# MSE = loss_fn(x, y)

# MSE.backward()
# print(linear.weight.grad)


# x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
# y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)

# linear = nn.Linear(2, 3)
# sigmod = nn.Sigmoid()
# loss_fn = nn.MSELoss()

# x1 = linear(x)
# x2 = sigmod(x1)

# MSE = loss_fn(x2, y)

# MSE.backward()
# print(linear.weight.grad)



# x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
# x = torch.tensor(x)


# x = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
# print(x)
# print(x.T)
# print(x.shape)
# x = Tensor(x)
# print(x.shape)

print("####################")
x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
# y = x + 2
# z = y * y * 3
# out = z.mean()
print(x[:, 1], x[:, 0])
out, out2 = x[:, 1], x[:, 0]
out.backward()
# print(x.grad)
# print(out.grad)
from torch.autograd import Variable
x = Variable(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32), requires_grad=True)

print(x[:, 1], x[:, 0])
out, out2 = x[:, 1], x[:, 0]
out.backward()
# print(x.grad)
# print(out.grad)

# BUG GETITEM; ITER
# BUG STATIC TENSOR
# => BUG VAE
# BUG SPLIT
# BUG LINEAR at th end from encoder to decoder