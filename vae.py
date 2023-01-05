
from torch.nn import Linear, MSELoss, Sigmoid, Sequential, ReLU, LeakyReLU, Tanh, BCELoss
from torch import nn
from torch.optim import Adam
from torch import Tensor
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import torch




training_data = open('datasets/mnist/mnist_train.csv','r').readlines()
test_data = open('datasets/mnist/mnist_test.csv','r').readlines()

image_size = (1, 28, 28)


def prepare_data(data, number_to_take = None):
    inputs = []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_take != None:
            if str(line[0]) == number_to_take:
                inputs.append(np.asfarray(line[1:]) / 255)
        else:
            inputs.append(np.asfarray(line[1:]) / 255)
        
    return inputs



dataset = np.asfarray(prepare_data(test_data, '0'))


# mnist vae

class VAE(nn.Module):
    def __init__(self, image_size, latent_size, hidden_size = 256):
        super().__init__()

        self.image_size = image_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.encoder = Sequential(
            Linear(np.prod(image_size), hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, 2 * latent_size)
        )

        self.decoder = Sequential(
            Linear(latent_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, np.prod(image_size)),
            Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
       
        mu, logvar = z[:, :self.latent_size], z[:, self.latent_size:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, z):
        return self.decoder(z)

    def loss(self, x, x_hat, mu, logvar):
        reconstruction_loss = MSELoss()(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def generate_and_save_image(self, epoch, path = 'vae_images'):
        if not os.path.exists(path):
            os.makedirs(path)

        z = torch.randn(64, self.latent_size)
        samples = self.generate(z).detach().numpy()
        
        figure = np.zeros((28 * 8, 28 * 8))
        for i, sample in enumerate(samples):
            row = i // 8
            col = i % 8
            figure[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = sample.reshape(28, 28)

        Image.fromarray(figure * 255).convert('RGB').save(f'{path}/image_at_epoch_{epoch:04d}.png')

vae = VAE(image_size, 128)
optimizer = Adam(vae.parameters(), lr = 1e-3)

batch_size = 128

for epoch in range(100):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch = torch.from_numpy(batch).float()
        x_hat, mu, logvar = vae(batch)
        loss = vae.loss(batch, x_hat, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch: {epoch}, loss: {loss.item()}')
    vae.generate_and_save_image(epoch)



