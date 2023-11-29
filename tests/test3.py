# import torch
# import numpy as np
# import neunet as nnet
# import neunet.nn as nnn
# import torch.nn as nn

# loss_fn = nn.CrossEntropyLoss(reduction =  'none', weight=torch.tensor([1, 2, 3], dtype=torch.double), ignore_index=-100)

# x = np.random.randn(2, 3, 4, 5)
# y = np.random.randint(0, 3, size=(2, 4, 5))
# # x = np.random.randn(2, 3, 4)
# # y = np.random.randint(0, 3, size=(2, 4))
# # batch_size, n_classes = 5, 3
# # x = torch.randn(batch_size, n_classes)
# # y = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)

# x = torch.tensor(x, requires_grad=True)
# y = torch.tensor(y, dtype=torch.int64)

# loss = loss_fn(x, y)

# print(loss.shape)
# print(loss)

# loss.backward(torch.ones_like(loss))

# print(x.grad)




# loss_fn = nnn.CrossEntropyLoss(reduction =  'none', weight=nnet.tensor([1, 2, 3]), ignore_index=-100)
# xnnet = nnet.tensor(x.detach().numpy(), requires_grad=True)
# ynnet = nnet.tensor(y.detach().numpy(), dtype=np.int32)


# loss = loss_fn(xnnet, ynnet)

# print(loss.shape)
# print(loss)

# loss.backward()
# print(xnnet.grad)


# x = nnet.tensor([1., 2, 3 ,4], dtype=np.float32, requires_grad=True)
# y = nnet.tensor([2., 2, 1, 1], dtype=np.float16)

# # y.data = x - y

# z = nnet.sqrt(x + y)


# # z = torch.cat([y, x])
# print(z.dtype, y.dtype) # dtype = max precision dtype
# z.backward(nnet.ones_like(z, dtype=np.float64))
# print(x.grad.dtype) # grad.dtype = x.dtype
# print(y.grad.dtype)

# a = np.zeros((1, 2, 3))
# print(a.shape)
# b = np.array(a)
# print(b.shape)


# x = nnet.tensor([1, 2 ,3])
# y = np.array(x)
# print(type(y))



import torch
import torch.nn as nn

# Create a tensor and set requires_grad=True to track computation with it
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# Apply softmax
y = nn.Softmax(dim=1)(x)

target = torch.tensor([0])

# Compute NLL loss
loss_fn = nn.NLLLoss()

loss = loss_fn(y, target)

loss.backward()

# Print gradients
print(x.grad)


import neunet
import neunet.nn as nnn

x = neunet.tensor([[1, 2, 3]]) 

y = nnn.Softmax(axis=1)(x)

loss_fn = nnn.NLLLoss()

loss = loss_fn(y, neunet.tensor([0], dtype=neunet.int64))
print(loss)

loss.backward()

print(x.grad)