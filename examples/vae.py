import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import os

from PIL import Image
from data_loader import load_mnist



dataset, _, _, _ = load_mnist()
dataset = dataset / 255 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]

latent_size = 64


class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
        self.mu_encoder = nn.Linear(latent_size, latent_size)
        self.logvar_encoder = nn.Linear(latent_size, latent_size)

        self.loss_fn = nn.BCELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = nnet.tensor(np.random.normal(0, 1, size=std.shape))
        z = mu + eps * std
        return z

    def forward(self, x):
        x = self.encoder(x)

        # mu, logvar = x_enc[:, :self.latent_size], x_enc[:, self.latent_size:] #
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        MSE = self.loss_fn(x_recon, x)
        KLD = -0.5 * nnet.sum(1 + logvar - mu.power(2) - logvar.exp())
        return MSE + KLD
    
    def train(self, x, optimizer):
        x_recon, mu, logvar = self.forward(x)

        loss = self.loss_function(x, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
        

    def generate(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)[0]

vae = VAE(28 * 28, latent_size)
optimizer = Adam(vae.parameters(), lr=0.001)


batch_size = 100
epochs = 100

for epoch in range(epochs):
    
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = 'epoch %d' % epoch)
    for i in tqdm_range:
        batch = dataset[i:i+batch_size]
        batch = nnet.tensor(batch, requires_grad=False).reshape(-1, 28 * 28)
        loss = vae.train(batch, optimizer)
        
        tqdm_range.set_description(f'epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}')


    generated = vae.generate(nnet.tensor(np.random.normal(0, 1, size=(100, latent_size)), requires_grad=False))
    # generated = vae.reconstruct(Tensor(dataset[:100], requires_grad=False).reshape(-1, 28 * 28))
    generated = generated.data

    for i in range(25):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated images/{i}.png')

