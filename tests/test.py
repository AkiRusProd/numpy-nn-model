# import sys, os
# from pathlib import Path
# sys.path[0] = str(Path(sys.path[0]).parent)

# import numpy as np
# from autograd import Tensor
# from nn import BatchNorm2d, MaxPool2d, AvgPool2d, Embedding, GELU, ELU, TanhExp, Softmax

# import torch
# from torch import nn

# # # x_rand = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# # # xx = Tensor(x_rand)
# # # xx.op = "input"

# # # weights = Tensor(np.array([[0.1, 0.2, 0.3]]))

# # # # y = xx.mean(axis = 0)

# # # y = weights * xx

# # # print(y.op, np.ones_like(y), y.shape)

# # # y.backward(grad = np.ones_like(y.data))

# # # print(xx.grad)

# # # x = np.array([[1, 2, 3], [4, 5, 6]])
# # # var = np.var(x, axis = 0, ddof=1)

# # # print(var)


# # x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# # var = x.var(axis = 0)
# # print(var)
# # print(np.ones_like(var.data).shape)
# # var.backward(np.ones_like(var.data))


# # print(x.grad)



# # x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# # var = x.var(axis = 0, unbiased=False)
# # print(torch.ones_like(var).shape)
# # var.backward(torch.ones_like(var))

# # print(var)
# # print(x.grad)


# # # X  = np.array([[1, 2, 3], [4, 5, 6]])
# # # X = Tensor(X)

# # # Y = 1 / X #BUG
# # # # Y = Tensor(1).div(X)
# # # # Y = Tensor(1) / X

# # # # Y = 2 * X
# # # # Y = Tensor(2) * X
# # # # Y = X * 2
# # # # Y = X / 2

# # # print(f"Y:\n {Y.data}")
# # # Y.backward()

# # # print(f"X.grad:\n {X.grad}")

# # # a = np.random.rand(5, 3)
# # # b = np.random.rand(1, 3)

# # # print(a + b)

# # x = Tensor(2)
# # l = 2 * x
# # nll = l * 3

# # z = nll + l 

# # z.backward()
# # print(x.grad)

# # class Dense(Tensor):
# #     def __init__(self, in_features, out_features):
# #         self.in_features = in_features
# #         self.out_features = out_features

# #         stdv = 1. / np.sqrt(in_features)
# #         # self.weight = Tensor(np.random.uniform(-stdv, stdv, (in_features, out_features)))
# #         self.weight = Tensor(np.ones((in_features, out_features)))
      
# #     def forward(self, x):
# #         return x.mm(self.weight)

# #     def __call__(self, x):
# #         return self.forward(x)


# # x = Tensor(np.array([[1]]))
# # dense1 = Dense(1, 1)
# # dense2 = Dense(1, 1)
# # dense2.weight.data *= 2

# # y = dense1(x)
# # y2 = dense2(y)

# # z = y + y2
# # print(f"z = {z.data}, {y.data}, {y2.data}")

# # z.backward(grad = np.ones_like(z.data) * 3)
# # print(x.grad)
# # print(dense1.weight.grad)
# # print(dense2.weight.grad)



# # x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# # y = x.mean()
# # print(y)
# # y.backward(np.ones_like(y.data))
# # print(x.grad)

# # x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# # y = x.mean()
# # print(y)
# # y.backward(torch.ones_like(y))
# # print(x.grad)



# # x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# # x_mean = x.mean(axis=0)
# # x_var = x.var(axis = 0)

# # z = x_mean + x_var
# # z.backward(np.ones_like(z.data))
# # print(x.grad)

# # x = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32, requires_grad=True)
# # x_mean = x.mean(axis=0)
# # x_var = x.var(axis = 0, unbiased=False)
# # z = x_mean + x_var
# # z.backward(torch.ones_like(z))
# # print(x.grad)

# # x = np.random.rand(5, 3, 4)
# # axis = (1, 2)
# # axis = list(axis) if isinstance(axis, tuple) else axis
# # reps = np.prod(np.array(x.shape)[axis])
# # print(reps)
# # reps2 = np.prod([x.shape[i] for i in axis])
# # print(reps2)

# # from nn import LayerNorm
# # # x_arr = np.random.randn(2, 3, 3)
# # x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# # x = Tensor(x_arr)
# # ln = LayerNorm(3)
# # y = ln(x)
# # print(y)

# # y.backward(np.ones_like(y.data))
# # print("-----------------")
# # print(x.grad)
# # x_grad = x.grad
# # ln_weight_grad = ln.weight.grad
# # # print(ln.weight.grad.shape, ln.weight.shape)
# # print(ln.weight.grad)
# # print(ln.bias.grad)

# # x = Tensor(1)
# # y = Tensor(2)
# # z = Tensor(3)

# # l = x + y + z
# # l.backward()
# # print(x.grad, y.grad, z.grad)


# # x = 1
# # print(np.ones_like(x))


# # x_rand = np.random.randn(2, 3, 2, 2)
# # x_rand = np.arange(0, 24).reshape(2, 3, 2, 2)
# # x = Tensor(x_rand)
# # bn = BatchNorm2d(3)

# # bn.train = True
# # y = bn(x)

# # print(y.data)

# # y.backward(np.ones_like(y.data))

# # print(x.grad)

# # import torch
# # import torch.nn as nn


# # x = torch.tensor(x_rand, requires_grad=True, dtype=torch.float32)
# # bn = nn.BatchNorm2d(3)

# # bn.train()
# # y = bn(x)
# # # print(y)

# # y.backward(torch.ones_like(y))

# # print(x.grad)



# # x = np.random.randn(2, 2, 5, 5)
# # x = np.ones((1, 1, 9, 9))
# # layer = AvgPool2d(4, 2, 1)
# # x_t = Tensor(x)
# # out = layer(x_t)
# # print(out)


# # grad = np.random.randn(*out.shape)
# # out.backward(grad)
# # print(x_t.grad)


# # x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
# # layer = nn.AvgPool2d(4, 2, 1)

# # y = layer(x)
# # print(y)
# # y.backward(torch.tensor(grad))
# # print(x.grad)
# # print(y.data.shape, out.shape)
# # print(np.allclose(y.data, out.data))
# # print(np.allclose(x.grad, x_t.grad))
# # # print(np.isclose(y.data, out.data))


# # rnn = nn.RNN(3, 2, 2)
# # x = torch.randn(5, 3, 3)
# # # h0 = torch.randn(2, 3, 2)
# # h0 = torch.zeros(2, 3, 2)
# # out, hn = rnn(x, h0)
# # print(out.shape, hn.shape)
# # print(out)



# # rnn = nn.RNN(10, 20, 1, batch_first=True)
# # input = torch.randn(5, 10) # 5 seq, 3 batch, 10 input if batch_first=False else batch seq input
# # # h0 = torch.randn(1, 3, 20) # 1 layer, 3 batch, 20 hidden
# # output1, hn = rnn(input)
# # print(output1.shape, hn.shape)

# # rnn.train(False)

# # output2, hn = rnn(input, h0)
# # print(output2.shape, hn.shape)
# # print(np.allclose(output1.data, output2.data))
# # for name, param in rnn.named_parameters():
# #     print(name, param.shape)

# # a = 1
# # b = 0

# # b = b + a if a is not None else + 0
# # print(b)


# # emb = Embedding(10, 5)
# # w = emb.weight.data

# # x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]])
# # x = np.array([1, 2, 3])
# # print(x.shape)

# # x_t = Tensor(x)
# # out = emb(x_t)
# # print(out)
# # print(f"out shape: {out.shape}")
# # out.backward(np.ones_like(out.data))


# # print(emb.weight.grad)
# # print(x_t.grad)

# # emb = nn.Embedding(10, 5)
# # emb.weight.data = torch.tensor(w)
# # x = torch.tensor(x, dtype=torch.long)
# # out = emb(x)
# # print(out.shape)
# # out.backward(torch.ones_like(out))

# # print(out)
# # print(emb.weight.grad)
# # print(x.grad)


# # rnn = nn.RNN(10, 20, 10, batch_first=True)
# # # print rnn weight
# # for name, param in rnn.named_parameters():
# #     print(name, param.shape)
# # # input = torch.randn(3, 5, 10) # 5 seq, 3 batch, 10 input if batch_first=False else batch seq input
# # # h0 = torch.randn(1, 3, 20) # 1 layer, 3 batch, 20 hidden
# # # output1, hn = rnn(input, h0)
# # # print(output1.shape, hn.shape)




# # arr = np.random.randn(2, 3, 4, 5)
# # my_x = Tensor(arr)
# # # gelu = TanhExp()
# # # y = gelu(x)
# # my_y = my_x.max(0, keepdims=True)
# # print(my_y)
# # my_y.backward()
# # print(my_x.grad)

# # x = torch.tensor(arr, requires_grad=True, dtype=torch.float32)
# # y = x.amax(0, keepdim=False) #contains max and argmax
# # print(y)
# # y.backward(torch.ones_like(y))
# # print(x.grad)

# # print(np.allclose(y.data, my_y.data))
# # print(np.allclose(x.grad, my_x.grad))

# # print(np.allclose(y[0].data, my_y.data))
# # print(np.allclose(y[1].data, my_x.grad))


# # x = torch.tensor(-2.324, dtype=torch.float32, requires_grad=True)
# # gelu = nn.TanhExp()
# # y = gelu(x)
# # print(y)
# # y.backward()
# # print(x.grad)



# # arr = np.random.randn(3, 4, 5)
# # arr = np.random.randn(3, 5)

# # softmax = Softmax()
# # my_x = Tensor(arr)
# # # my_y = softmax(my_x)
# # e_x = my_x.sub(my_x.max(axis=-1, keepdims=True)).exp()
# # e_x_sum = e_x.sum(axis=-1, keepdims=True)
# # # e_x = my_x.max(axis=-1, keepdims=True)
# # # e_x_sum = e_x.sum()
# # my_y =  e_x.mul(e_x_sum)
# # # print(e_x.shape, e_x_sum.shape)

# # print(my_y)
# # my_y.backward()
# # print(my_x.grad)

# # x = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
# # # y = nn.Softmax(dim=-1)(x)
# # e_x = x.sub(x.amax(dim = -1, keepdim=True)).exp()
# # e_x_sum = e_x.sum(dim = -1, keepdim=True)
# # # e_x = x.amax(axis=-1, keepdims=True)
# # # e_x_sum = e_x.sum()
# # y =  e_x.mul(e_x_sum)

# # print(y)
# # y.backward(torch.ones_like(y))
# # print(x.grad)

# # print(np.allclose(y.data, my_y.data))
# # print(np.allclose(x.grad, my_x.grad))
# # print(e_x.shape, e_x_sum.shape)


# # arr2 = np.random.randn(3, 4, 1)

# # my_x = Tensor(arr)
# # my_x2 = Tensor(arr2)

# # my_y = my_x.div(my_x2)
# # my_y.backward()
# # print(my_x.grad)

# # x = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
# # x2 = torch.tensor(arr2, dtype=torch.float32, requires_grad=True)
# # y = x.div(x2)
# # y.backward(torch.ones_like(y))
# # print(x.grad)

# # print(np.allclose(y.data, my_y.data))
# # print(np.allclose(x.grad, my_x.grad))




# x = Tensor(4)
# # y = x**2
# # y = x*2
# # y = 2 * x
# # y = Tensor(2)/x
# y = 2/x/x
# # y = x + 2
# # y = x/2

# # print(y)
# # y.backward()
# # print(x.grad)


# x = Tensor([[1.1, 2.0, 3.6], [4.7, 3.14, 2.718]], requires_grad=True)
# y = Tensor([[2., 3.99], [8.4, 1.5], [2.5, 7.8]], requires_grad=True)
# output = Tensor.tanh(1/(Tensor.concatenate(Tensor.sin((Tensor.exp(x ** 1.4) / 3.1 ** Tensor.log(x)).mm(y)), y).mean()))

# print(x.shape, y.shape)

# # z = x.mm(y)
# # print(z.shape)
# # z.backward(np.ones_like(z.data))
# # print(x.grad)

# # step1 = Tensor.sin((Tensor.exp(x ** 1.4) / 3.1 ** Tensor.log(x)).mm(y))

# pow_x = x ** 1.4
# exp_x = pow_x.exp()
# log_x = x.log()
# div_x = 3.1 ** log_x
# outsin = exp_x / div_x
# step = outsin.mm(y)


# step1 = step.sin()
# # step1 = div_x

# # step2 = Tensor.concatenate(step1, y)
# # step3 = step2.mean()

# print(output)
# output.backward()


# # tensor(0.3489, grad_fn=<tanhBackward>)
# print(x.grad)
# # [[ -0.0276   0.6477 -11.827 ]
# #  [ 27.2554  -5.3062   1.0978]]
# print(y.grad)
# # [[-10.8005   8.2755]
# #  [ -0.3336   0.2187]
# #  [  0.8972  -0.9684]]


# x = torch.tensor([[1.1, 2.0, 3.6], [4.7, 3.14, 2.718]], requires_grad=True)
# y = torch.tensor([[2., 3.99], [8.4, 1.5], [2.5, 7.8]], requires_grad=True)
# output = torch.tanh(1/(torch.cat([torch.sin((torch.exp(x ** 1.4) / 3.1 ** torch.log(x)) @ y), y]).mean()))

# # step1 = torch.sin((torch.exp(x ** 1.4) / 3.1 ** torch.log(x)) @ y)
# # step2 = torch.cat([step1, y])
# # step3 = step2.mean()

# pow_x = x ** 1.4
# exp_x = pow_x.exp()
# log_x = x.log()
# div_x = 3.1 ** log_x
# outsin = exp_x / div_x
# step = outsin.mm(y)


# step1 = step.sin()
# # step1 = div_x

# output.backward(torch.ones_like(output))

# print(output)
# # tensor(0.3489, grad_fn=<tanhBackward>)
# print(x.grad)
# # [[ -0.0276   0.6477 -11.827 ]
# #  [ 27.2554  -5.3062   1.0978]]
# print(y.grad)
# # [[-10.8005   8.2755]
# #  [ -0.3336   0.2187]
# #  [  0.8972  -0.9684]]


# x = Tensor(2)
# y = 2 - x
# y.backward()
# print(x.grad)

# x = torch.tensor(2, requires_grad=True, dtype=torch.float32)
# y = 2 - x
# y.backward()
# print(x.grad)


# x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# # x_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# x_arr = np.random.randn(2, 3, 3)

# x = Tensor(x_arr)
# eps = 1e-5

# mean = x.mean(axis = -1, keepdims=True)
# var = x.var(axis = -1,  keepdims=True)


# x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# varaddeps = var + eps
# x_hat = x_c * x

# x_hat.backward(np.ones_like(x_hat.data))
# print("X_HAT", x_hat.data)
# x_grad = x.grad

# print("x.grad", x_grad)

# print("--------------------")

# x = torch.tensor(x_arr, requires_grad = True)

# mean = x.mean(axis = -1, keepdim=True)
# var = x.var(axis = -1, unbiased=False, keepdim=True)

# x_c = x - mean # WHEN - MEAN NOT COMPATIBLE WITH PYTORCH
# x_hat = x_c * x

# x_hat.backward(torch.ones_like(x_hat.data))
# print("X_HAT", x_hat.data)
# print("x.grad", x.grad)

import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import neunet as nnet
from neunet import nn
from neunet.autograd import Tensor

x = Tensor([[1.1, 2.0, 3.6], [4.7, 3.14, 2.718]], requires_grad=True)
y = Tensor([[2., 3.99], [8.4, 1.5], [2.5, 7.8]], requires_grad=True)

output = Tensor.tanh(1/(Tensor.concatenate(Tensor.sin((Tensor.exp(x ** 1.4) / 3.1 ** Tensor.log(x)).mm(y)), y).mean()))

print(output)
output.backward()

print(x.grad)
print(y.grad)

x = nnet.tensor([[1.1, 2.0, 3.6], [4.7, 3.14, 2.718]], requires_grad=True)
y = nnet.tensor([[2., 3.99], [8.4, 1.5], [2.5, 7.8]], requires_grad=True)

output = nnet.tanh(1/(nnet.concatenate([nnet.sin((nnet.exp(x ** 1.4) / 3.1 ** nnet.log(x)).mm(y)), y]).mean()))

print(output)
output.backward()

print(x.grad)
print(y.grad)



# x = nnet.tensor([1])
# y = nnet.tensor([2])

# z = nnet.add(x, y)
# # z = x - y

# print(z)
# z.backward()
# print(x.grad)
# print(y.grad)
import numpy as np
import torch


# x = torch.randn(2, 3, 3)
# x = x.reshape((((2,3,3))))
# print(x.shape)

# x= nnet.tensor(x.data)
# y= nnet.tensor(x.data)

# z = x.max(axis = -1, keepdims=True)
# print(z.shape)

# z2 = nnet.max(x, axis = -1, keepdims=True)
# print(z2.shape)

# z3 = nnet.maximum(x, 0)
# print(x)
# print(z3)

# x = nnet.tensor(np.random.randn(2, 3, 3))
# x = nnet.reshape(x, *(2, 3, 3))
# print(x.shape)

x = Tensor(np.random.randn(2, 3, 3))
# y = nnet.reshape(x, (3, 3, 2))
# y = x.reshape(*(3, 3, 2))
# y = Tensor.reshape(x, *(3, 3, 2))
print(y.shape, "y")
y.backward(np.ones_like(y.data))
print(x.grad.shape)

# x = torch.tensor(np.random.randn(2, 3, 3), requires_grad=True)
# x = x.reshape((3, 2, 3)).retain_grad()
# # print(y.shape)
# x.backward(torch.ones_like(x.data))
# print(x.grad.shape)

x = [1, 2, 3]
print(*x)

x = torch.tensor(np.random.randn(2, 3, 3))  # , requires_grad=True)
y = torch.tensor(np.random.randn(2, 3, 3))  # , requires_grad=True)
z = torch.cat((x, y), axis=0)
print(z.shape)
# # z = x.cat(y)

x = nnet.tensor(np.random.randn(2, 3, 3))
y = nnet.tensor(np.random.randn(2, 3, 3))
w = nnet.tensor(np.random.randn(2, 3, 3))
z = nnet.concatenate(x, y, w, axis=1)
# z = x.concatenate(y, w, axis=1)
# z = Tensor.concatenate(x, y, w, axis=1)
z.backward(np.ones_like(z.data))
print(z.shape)
print(x.grad.shape)
print(y.grad.shape)
print(w.grad.shape)