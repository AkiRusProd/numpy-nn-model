import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid, Sequential, Module, MSELoss, BCELoss
from torch import Tensor
import numpy as np



latent_size = 2
multiplier = 1
x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
y = torch.tensor([[1], [0]], dtype=torch.float32)

Linear_weight =  np.arange(latent_size * 2 * multiplier).reshape((latent_size * multiplier, 2))

Linear1 = Linear(2, latent_size * multiplier, bias=False)
Linear1.weight = torch.nn.Parameter(torch.tensor(Linear_weight, dtype=torch.float32))
Linear1.bias = torch.nn.Parameter(torch.tensor(np.zeros(latent_size * multiplier), dtype=torch.float32))

relu = ReLU()
sigmoid = nn.Tanh()
loss_fn= MSELoss()

encoder = Sequential(Linear1, relu)

Linear2_weight = np.arange(latent_size * 2).reshape((2, latent_size))
Linear2 = Linear(latent_size, 2, bias=False)
Linear2.weight = torch.nn.Parameter(torch.tensor(Linear2_weight, dtype=torch.float32))
Linear2.bias = torch.nn.Parameter(torch.tensor(np.zeros(2), dtype=torch.float32))
decoder = Sequential(Linear2, sigmoid)
# decoder = Sequential(Linear2)


mu_encoder = Linear(latent_size, latent_size, bias=False)
mu_encoder.weight = torch.nn.Parameter(torch.tensor(np.ones((latent_size, latent_size)), dtype=torch.float32))

logvar_encoder = Linear(latent_size, latent_size, bias=False)
logvar_encoder.weight = torch.nn.Parameter(torch.tensor(np.ones((latent_size, latent_size)), dtype=torch.float32))

def forward(x):
    x_enc = encoder(x)

    mu = mu_encoder(x_enc)
    logvar = logvar_encoder(x_enc)
    
    std = logvar.mul(0.5).exp()
    eps = torch.tensor(np.ones_like(std.data))
    z = mu + eps * std
    
    x_recon = decoder(z)

    
    return x_recon, mu, logvar
   

def loss_function(x, x_recon, mu, logvar):
    MSE = loss_fn(x_recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
   


def train(x):
    mu = None
    logvar = None
    x_recon, mu, logvar = forward(x)


    # loss = loss_function(x_recon, x)
    loss = loss_function(x, x_recon, mu, logvar)
    loss.backward()
    print("grads1")
    print(Linear1.weight.grad)
    print("grads2")
    print(Linear2.weight.grad)


    return loss

train(x)

