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




image_size = (1, 28, 28)

dataset, _, _, _ = load_mnist()
dataset = dataset / 127.5-1 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]



generator = nn.Sequential(
    nn.Linear(100, 256),
    nn.LeakyReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 512),
    nn.Dropout(0.2),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(),
    nn.Linear(512, 784),
    nn.Tanh()
)

discriminator = nn.Sequential(
    nn.Linear(784, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

loss_fn = nn.MSELoss()

g_optimizer = Adam(generator.parameters(), lr = 0.001, betas = (0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr = 0.001, betas = (0.5, 0.999))

batch_size = 64
epochs = 100


for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = f'epoch {epoch}')
    for i in tqdm_range:
        batch = dataset[i:i+batch_size]
        batch = nnet.tensor(batch, requires_grad = False)

        d_optimizer.zero_grad()

        # train discriminator on real data
        real_data = batch
        real_data = real_data.reshape(real_data.shape[0], -1)

        real_data_prediction = discriminator(real_data) 
        real_data_loss = loss_fn(real_data_prediction, nnet.tensor(np.ones((real_data_prediction.shape[0], 1)), requires_grad = False))
        real_data_loss.backward()
        d_optimizer.step()

        # train discriminator on fake data
        noise = nnet.tensor(np.random.normal(0, 1, (batch_size, 100)), requires_grad = False)
        fake_data = generator(noise)
        fake_data_prediction = discriminator(fake_data)
        fake_data_loss = loss_fn(fake_data_prediction, nnet.tensor(np.zeros((fake_data_prediction.shape[0], 1)), requires_grad = False))
        fake_data_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        noise = nnet.tensor(np.random.normal(0, 1, (batch_size, 100)), requires_grad = False)
        fake_data = generator(noise)
        fake_data_prediction = discriminator(fake_data)
        fake_data_loss = loss_fn(fake_data_prediction, nnet.tensor(np.ones((fake_data_prediction.shape[0], 1)), requires_grad = False))
        fake_data_loss.backward()
        g_optimizer.step()

        g_loss = -np.log(fake_data_prediction.data).mean()
        d_loss = -np.log(real_data_prediction.data).mean() - np.log(1 - fake_data_prediction.data).mean()
        tqdm_range.set_description(
            f'epoch: {epoch + 1}/{epochs}, G loss: {g_loss:.7f}, D loss: {d_loss:.7f}'
        )

    noise = nnet.tensor(np.random.normal(0, 1, (100, 100)), requires_grad = False)
    generated_images = generator(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], 1, 28, 28)
    generated_images = generated_images.data

    for i in range(25):
        image = generated_images[i] * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated images/{i}.png')







