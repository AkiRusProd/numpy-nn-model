import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from neunet.autograd import Tensor
from neunet.nn import Linear, Sequential, Module, MSELoss, Sigmoid, ReLU, BCELoss
from neunet.optim import SGD, Adam
from tqdm import tqdm
import numpy as np
import os
from PIL import Image



training_data = open('datasets/mnist/mnist_train.csv','r').readlines()
test_data = open('datasets/mnist/mnist_test.csv','r').readlines()

image_size = (1, 28, 28)

def prepare_data(data, number_to_take = None):
    inputs = []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_take != None:
            if str(line[0]) == number_to_take:
                inputs.append(np.asfarray(line[1:]))
        else:
            inputs.append(np.asfarray(line[1:]))
        
    return inputs



mnist_data_path = "datasets/mnist/"

if not os.path.exists(mnist_data_path + "mnist_train.npy") or not os.path.exists(mnist_data_path + "mnist_test.npy"):
    train_inputs = np.asfarray(prepare_data(training_data))
    test_inputs = np.asfarray(prepare_data(test_data))

    np.save(mnist_data_path + "mnist_train.npy", train_inputs)
    np.save(mnist_data_path + "mnist_test.npy", test_inputs)
else:
    train_inputs = np.load(mnist_data_path + "mnist_train.npy")
    test_inputs = np.load(mnist_data_path + "mnist_test.npy")


dataset = train_inputs / 255 #/ 255 => [0; 1]  #/ 127.5-1 => [-1; 1]


class VAE(Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = Sequential(
            Linear(input_size, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, latent_size),
            ReLU(),
        )

        self.decoder = Sequential(
            Linear(latent_size, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Linear(256, input_size),
            Sigmoid()
        )
        self.mu_encoder = Linear(latent_size, latent_size)
        self.logvar_encoder = Linear(latent_size, latent_size)

        self.loss_fn = BCELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = Tensor(np.random.normal(0, 1, size=std.shape))
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
        KLD = -0.5 * Tensor.sum(1 + logvar - mu.power(2) - logvar.exp())
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

vae = VAE(28 * 28, 64)
optimizer = Adam(vae.parameters(), lr=0.001)


batch_size = 100
epochs = 100

for epoch in range(epochs):
    
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = 'epoch %d' % epoch)
    for i in tqdm_range:
        batch = dataset[i:i+batch_size]
        batch = Tensor(batch, requires_grad=False).reshape(-1, 28 * 28)
        loss = vae.train(batch, optimizer)
        
        tqdm_range.set_description('epoch %d, loss: %.4f' % (epoch, loss.data))



    generated = vae.generate(Tensor(np.random.normal(0, 1, size=(100, 64)), requires_grad=False))
    # generated = vae.reconstruct(Tensor(dataset[:100], requires_grad=False).reshape(-1, 28 * 28))
    generated = generated.data

    for i in range(100):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated_images/{i}.png')

