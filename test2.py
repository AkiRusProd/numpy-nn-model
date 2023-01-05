import numpy as np
from autograd import Tensor, MSELoss, Sigmoid, LeakyReLU, Tanh, Sequential, SGD, Adam, ReLU
import torch
import torch.nn as nn




class LinearTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)


    def backward(self, grad=1):
       
        self.args[0].backward(np.dot(grad, self.args[1].data))
        self.args[1].backward(np.dot(self.args[0].data.T, grad).T)
        self.args[2].backward(np.sum(grad, axis = 0, keepdims = True))


class Linear():
    def __init__(self, in_features, out_features, bias = True):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / np.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)))
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad = bias)

    def forward(self, X, training = True): 
        self.X = X
        
        self.output_data = np.dot(self.X.data, self.weight.data.T) + self.bias.data
        
        return LinearTensor(self.output_data, [self.X, self.weight, self.bias], "linear")

    def __call__(self, X, training = True):
       
        return self.forward(X, training)





layer_weight = np.array([[1, 1, 1], [1, 1, 1]]).T
layer_bias = np.array([[1], [0], [1]]).T


layer = Linear(2, 3)
layer.weight.data = layer_weight
layer.bias.data = layer_bias
print(layer.weight.data.shape)
print(layer.bias.data.shape)


x = Tensor(np.array([[1, 0], [0, 1], [1, 1], [0, 0]]))
y = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]))
print(x.data.shape, y.data.shape)

y_pred = layer(x)
y_pred = Sigmoid()(y_pred)
loss = MSELoss()(y_pred, y)
print(f'y_pred: {y_pred}')
print(x.data.shape, y.data.shape, y_pred.data.shape, loss.data.shape)

loss.backward()

print(layer.weight.grad.shape)
print(layer.weight.grad)
print(layer.bias.grad.shape)
print(layer.bias.grad)



layer = nn.Linear(2, 3)
layer.weight.data = torch.nn.Parameter(torch.tensor(layer_weight, dtype=torch.float32))
layer.bias.data = torch.nn.Parameter(torch.tensor(layer_bias, dtype=torch.float32))
print(layer.weight.data.shape)
print(layer.bias.data.shape)


x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float32)

y_pred = layer(x)
y_pred = torch.sigmoid(y_pred)
print(f"y_pred: {y_pred}")
loss = nn.MSELoss()(y_pred, y)
print(x.shape, y.shape, y_pred.shape, loss.shape)

loss.backward()

print(layer.weight.grad.shape)
print(layer.weight.grad)
print(layer.bias.grad.shape)
print(layer.bias.grad)




# dense1 = Linear(2, 3)
# relu = ReLU()
# dense2 = Linear(3, 1)
# loss_op = MSELoss()
# sigmod = Sigmoid()




# x = Tensor(np.array([[1, 0], [0, 1]]))
# y = Tensor(np.array([[1], [0]]))
# print(x.data.shape, y.data.shape)

# y_pred = dense1(x)
# y_pred = relu(y_pred)
# y_pred = dense2(y_pred)

# y_pred = sigmod(y_pred)
# loss = loss_op(y_pred, y)