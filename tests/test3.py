import torch
import numpy as np
import neunet as nnet
import neunet.nn as nnn
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss(reduction =  'none', weight=torch.tensor([1, 2, 3], dtype=torch.double), ignore_index=-100)

x = np.random.randn(2, 3, 4, 5)
y = np.random.randint(0, 3, size=(2, 4, 5))
# x = np.random.randn(2, 3, 4)
# y = np.random.randint(0, 3, size=(2, 4))
# batch_size, n_classes = 5, 3
# x = torch.randn(batch_size, n_classes)
# y = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)

x = torch.tensor(x, requires_grad=True)
y = torch.tensor(y, dtype=torch.int64)

loss = loss_fn(x, y)

print(loss.shape)
print(loss)

loss.backward(torch.ones_like(loss))

print(x.grad)




loss_fn = nnn.CrossEntropyLoss(reduction =  'none', weight=nnet.tensor([1, 2, 3]), ignore_index=-100)
xnnet = nnet.tensor(x.detach().numpy(), requires_grad=True)
ynnet = nnet.tensor(y.detach().numpy())


loss = loss_fn(xnnet, ynnet)

print(loss.shape)
print(loss)

loss.backward()
print(xnnet.grad)
