import numpy as np
from autograd import Tensor
from nn import Linear, Sequential, Module, MSELoss, Sigmoid, ReLU, BCELoss, Tanh
import torch
import torch.nn as nn





# class Dense(Tensor):
#     def __init__(self, in_features, out_features):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)))
#         #self.bias = Tensor(np.zeros((1, out_features)))

#     def forward(self, x):
#         return x.mm(self.weight.transpose())#.add(self.bias)

#     def __call__(self, x):
#         return self.forward(x)




# latent_size = 2
# multiplier = 2
# x = Tensor(np.array([[1, 0], [0, 1]]))
# y = Tensor(np.array([[1], [0]]))

# Linear_weight = np.arange(latent_size * 2 * multiplier).reshape((latent_size * multiplier, 2))

# Linear1 = Linear(2, latent_size * multiplier)
# Linear1.weight.data = Linear_weight

# relu = ReLU()
# sigmoid = Tanh()
# loss_fn= MSELoss()

# # encoder = Sequential(Linear1, relu)
# dense1 = Dense(2, latent_size * multiplier)
# encoder = Sequential(dense1)


# Linear2_weight = np.arange(latent_size * 2).reshape((2, latent_size))
# Linear2 = Linear(latent_size, 2)
# Linear2.weight.data = Linear2_weight
# # decoder = Sequential(Linear2, sigmoid)
# dense2 = Dense(latent_size, 2)
# decoder = Sequential(dense2)

# print("Linear2.weight.data: ", Linear2.weight.data)

# mu_encoder = Linear(latent_size, latent_size)
# mu_encoder.weight.data = np.ones((latent_size, latent_size))
# logvar_encoder = Linear(latent_size, latent_size)
# logvar_encoder.weight.data = np.ones((latent_size, latent_size))




# def forward(x):
#     x_enc = encoder(x)
 
#     # mu = mu_encoder(x_enc)
#     # logvar = logvar_encoder(x_enc)
#     mu, logvar = x_enc[:, :latent_size], x_enc[:, latent_size:]

    
#     std = logvar.mul(0.5).exp()
#     eps = Tensor(np.ones_like(std.data))
#     z = mu + eps * std

#     x_recon = decoder(z)

#     return x_recon, mu, logvar

# def loss_function(x, x_recon, mu = None, logvar=None):
#     MSE = loss_fn(x_recon, x)
#     KLD = Tensor(-0.5) * Tensor.sum(Tensor(1) + logvar - mu.power(2) - logvar.exp())
#     return MSE + KLD
    


# def train(x):
#     mu = None
#     logvar = None
#     x_recon, mu, logvar = forward(x)
#     # x_recon = forward(x)
#     print("x_recon: ", x_recon)

#     # loss = loss_function(x_recon, x)
#     loss = loss_function(x, x_recon, mu, logvar)
#     loss.backward()
#     print("grad1")
#     # print(Linear1.weight.grad)
#     print(dense1.weight.grad)
#     print("grad2")
#     print(dense2.weight.grad)


#     return loss

# train(x)



# x = np.random.randn(2, 2)
# y = np.random.randn(2, 1, 1, 1)
# z = x + y

# x = np.random.randn(2, 3, 3)
# y = np.random.randn(5, 1, 3, 3)
# z = x + y

# x = np.random.randn(2, 1)
# y = np.random.randn(3, 3, 1, 1)
# z = x + y

# x = np.random.randn(1, 2)
# y = np.random.randn(3, 2)
# z = x + y

# x = np.random.randn(2, 1)
# y = np.random.randn(2, 3)
# z = x + y

# x = np.random.randn(2, 1, 2)
# y = np.random.randn(2, 1, 1, 1, 2)
# z = x + y

# x = np.random.randn(2, 1)
# y = np.random.randn(2, 1, 1, 1, 2)
# z = x + y

# x = np.random.randn(2)
# y = np.random.randn(2, 1, 1, 1)
# z = x + y

x = np.random.randn(2, 3, 3)
y = np.random.randn(3)
z = x + y

x = np.random.randn(1, 3)
y = np.random.randn(3)
z = x + y


def add_grad(x, grad):
    print("grad init: ", grad.shape)
    print("x.size: ", x.size)
    print("grad.size: ", grad.size)
    print("x.ndim: ", x.ndim)
    print("grad.ndim: ", grad.ndim)
    if grad.size != x.size or grad.ndim != x.ndim or grad.shape != x.shape:
        if x.ndim == grad.ndim:
            print("x.ndim == grad.ndim")
            print("x.shape: ", x.shape)
            print("grad.shape: ", grad.shape)
            # axis = np.where(np.array(x.shape) != np.array(grad.shape))[0][0]
            axis = tuple(np.where(np.array(x.shape) != np.array(grad.shape))[0])
            print("axis: ", axis)
            grad = grad.sum(axis=axis, keepdims=True)

        elif x.ndim > grad.ndim: #this case never happens
            print("x.ndim > grad.ndim")
            grad = grad.sum(axis=tuple(range(grad.ndim - x.ndim)), keepdims=True)
        # else:
        #     print("x.ndim < grad.ndim")
        #     print("GRAD SHAPE: ", grad.shape)
        #     grad = grad.sum(axis=tuple(range(grad.ndim - x.ndim)))
        #     print("GRAD SHAPE: ", grad.shape, tuple(range(grad.ndim - x.ndim)))
        else:
            # add dimension to x to match grad
            # xtmp = x.reshape((1,) * (grad.ndim - x.ndim) + x.shape)
            xtmp_shape = (1,) * (grad.ndim - x.ndim) + x.shape
            print("xtmp.shape: ", xtmp_shape)
            axis = tuple(np.where(np.array(xtmp_shape) != np.array(grad.shape))[0])
            print("x.ndim < grad.ndim")
            print("x.shape: ", x.shape)
            print("grad.shape: ", grad.shape)
            print("axis: ", axis)
            grad = grad.sum(axis=axis).reshape(x.shape)
            print("grad.shape: ", grad.shape)
       

    print("x: ", x.shape)
    print("grad: ", grad.shape)
    print(x.shape == grad.shape)





# def add_grad(x, grad):

#     if x.size != grad.size:
#         if x.ndim == grad.ndim:
#             grad = grad.sum(axis=tuple(np.where(np.array(x.shape) != np.array(grad.shape))[0]), keepdims=True)
#         elif x.ndim < grad.ndim:
#             grad = grad.sum(axis=tuple(range(grad.ndim - x.ndim)))

#     print(x.shape == grad.shape)

add_grad(y, z)



# x = Tensor(x)
# y = Tensor(y)
# z = x + y

# z.backward(np.ones_like(z.data))

# print("x.grad: ", x.grad.shape)
# print("y.grad: ", y.grad.shape)
