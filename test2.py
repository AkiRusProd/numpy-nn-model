import numpy as np
from autograd import Tensor
from nn import Linear, Sequential, Module, MSELoss, Sigmoid, ReLU, BCELoss, Tanh
import torch
import torch.nn as nn








# layer_weight = np.array([[1, 1, 1], [1, 1, 1]]).T
# layer_bias = np.array([[1], [0], [1]]).T


# layer = Linear(2, 3)
# layer.weight.data = layer_weight
# layer.bias.data = layer_bias
# print(layer.weight.data.shape)
# print(layer.bias.data.shape)


# x = Tensor(np.array([[1, 0], [0, 1], [1, 1], [0, 0]]))
# y = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]))
# print(x.data.shape, y.data.shape)

# y_pred = layer(x)
# y_pred = Sigmoid()(y_pred)
# loss = MSELoss()(y_pred, y)
# print(f'y_pred: {y_pred}')
# print(x.data.shape, y.data.shape, y_pred.data.shape, loss.data.shape)

# loss.backward()

# print(layer.weight.grad.shape)
# print(layer.weight.grad)
# print(layer.bias.grad.shape)
# print(layer.bias.grad)



# layer = nn.Linear(2, 3)
# layer.weight.data = torch.nn.Parameter(torch.tensor(layer_weight, dtype=torch.float32))
# layer.bias.data = torch.nn.Parameter(torch.tensor(layer_bias, dtype=torch.float32))
# print(layer.weight.data.shape)
# print(layer.bias.data.shape)


# x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
# y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float32)

# y_pred = layer(x)
# y_pred = torch.sigmoid(y_pred)
# print(f"y_pred: {y_pred}")
# loss = nn.MSELoss()(y_pred, y)
# print(x.shape, y.shape, y_pred.shape, loss.shape)

# loss.backward()

# print(layer.weight.grad.shape)
# print(layer.weight.grad)
# print(layer.bias.grad.shape)
# print(layer.bias.grad)




# # dense1 = Linear(2, 3)
# # relu = ReLU()
# # dense2 = Linear(3, 1)
# # loss_op = MSELoss()
# # sigmod = Sigmoid()




# # x = Tensor(np.array([[1, 0], [0, 1]]))
# # y = Tensor(np.array([[1], [0]]))
# # print(x.data.shape, y.data.shape)

# # y_pred = dense1(x)
# # y_pred = relu(y_pred)
# # y_pred = dense2(y_pred)

# # y_pred = sigmod(y_pred)
# # loss = loss_op(y_pred, y)




# x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
# y = torch.tensor([[1], [0]], dtype=torch.float32)

# Linear1 = nn.Linear(2, 3)
# Linear2 = nn.Linear(3, 1)
# relu = nn.ReLU()
# sigmoid = nn.Sigmoid()
# loss= nn.MSELoss()

# y_pred = Linear1(x)
# y_pred = relu(y_pred)
# y_pred = Linear2(y_pred)

# y_pred = sigmoid(y_pred)
# loss = loss(y_pred, y)

# loss.backward()

# print(Linear1.weight.grad.shape)




class Dense(Tensor):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / np.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)))
        #self.bias = Tensor(np.zeros((1, out_features)))

    def forward(self, x):
        return x.mm(self.weight.transpose())#.add(self.bias)

    def __call__(self, x):
        return self.forward(x)




latent_size = 2
multiplier = 2
x = Tensor(np.array([[1, 0], [0, 1]]))
y = Tensor(np.array([[1], [0]]))

Linear_weight = np.arange(latent_size * 2 * multiplier).reshape((latent_size * multiplier, 2))

Linear1 = Linear(2, latent_size * multiplier)
Linear1.weight.data = Linear_weight

relu = ReLU()
sigmoid = Tanh()
loss_fn= MSELoss()

# encoder = Sequential(Linear1, relu)
dense1 = Dense(2, latent_size * multiplier)
encoder = Sequential(dense1)


Linear2_weight = np.arange(latent_size * 2).reshape((2, latent_size))
Linear2 = Linear(latent_size, 2)
Linear2.weight.data = Linear2_weight
# decoder = Sequential(Linear2, sigmoid)
dense2 = Dense(latent_size, 2)
decoder = Sequential(dense2)

print("Linear2.weight.data: ", Linear2.weight.data)

mu_encoder = Linear(latent_size, latent_size)
mu_encoder.weight.data = np.ones((latent_size, latent_size))
logvar_encoder = Linear(latent_size, latent_size)
logvar_encoder.weight.data = np.ones((latent_size, latent_size))




def forward(x):
    x_enc = encoder(x)
 
    # mu = mu_encoder(x_enc)
    # logvar = logvar_encoder(x_enc)
    mu, logvar = x_enc[:, :latent_size], x_enc[:, latent_size:]

    
    std = logvar.mul(0.5).exp()
    eps = Tensor(np.ones_like(std.data))
    z = mu + eps * std

    x_recon = decoder(z)

    return x_recon, mu, logvar

def loss_function(x, x_recon, mu = None, logvar=None):
    MSE = loss_fn(x_recon, x)
    KLD = Tensor(-0.5) * Tensor.sum(Tensor(1) + logvar - mu.power(2) - logvar.exp())
    return MSE + KLD
    


def train(x):
    mu = None
    logvar = None
    x_recon, mu, logvar = forward(x)
    # x_recon = forward(x)
    print("x_recon: ", x_recon)

    # loss = loss_function(x_recon, x)
    loss = loss_function(x, x_recon, mu, logvar)
    loss.backward()
    print("grad1")
    # print(Linear1.weight.grad)
    print(dense1.weight.grad)
    print("grad2")
    print(dense2.weight.grad)


    return loss

train(x)



