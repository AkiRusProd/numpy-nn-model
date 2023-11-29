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
x_num, y_num = 5, 5
samples_num = x_num * y_num
margin = 15

dataset, _, _, _ = load_mnist()
dataset = dataset / 127.5-1 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]

noise_size = 100


generator = nn.Sequential(
    nn.Linear(noise_size, 256),
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

batch_size = 100
epochs = 30

each_epoch_generated_samples = []
const_noise = nnet.tensor(np.random.normal(0, 1, (samples_num, noise_size)), requires_grad = False)

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = f'epoch {epoch}')
    generator.train()
    discriminator.train()
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
        noise = nnet.tensor(np.random.normal(0, 1, (batch_size, noise_size)), requires_grad = False)
        fake_data = generator(noise)
        fake_data_prediction = discriminator(fake_data)
        fake_data_loss = loss_fn(fake_data_prediction, nnet.tensor(np.zeros((fake_data_prediction.shape[0], 1)), requires_grad = False))
        fake_data_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        noise = nnet.tensor(np.random.normal(0, 1, (batch_size, noise_size)), requires_grad = False)
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

    if const_noise == None:
        noise = nnet.tensor(np.random.normal(0, 1, (batch_size, noise_size)), requires_grad = False)
    else:
        noise = const_noise

    each_epoch_generated_samples.append(generator(noise).data.reshape(-1, 28, 28))

    generator.eval()
    discriminator.eval()
    for i in range(samples_num):
        image = each_epoch_generated_samples[-1][i] * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated images/{i}.png')


generator.eval()
discriminator.eval()



def get_images_set(images):
    images_array = np.full((y_num * (margin + image_size[1]), x_num * (margin + image_size[2])), 255, dtype=np.uint8)
    num = 0
    for i in range(y_num):
        for j in range(x_num):
            y = i*(margin + image_size[1])
            x = j*(margin + image_size[2])

            images_array[y:y+image_size[1],x:x+image_size[2]] = images[num]
            num+=1

    images_array = images_array[: (y_num - 1) * (image_size[1] + margin) + image_size[1], : (x_num - 1) * (image_size[2] + margin) + image_size[2]]

    return Image.fromarray(images_array).convert("L")


def create_vectors_interpolation():
    '''Create vectors create interpolation  in the latent space between two sets of noise vectors'''
    steps = 10
    interval = 15
    images=[]

    noise_1 = nnet.tensor(np.random.normal(0, 1, (samples_num, noise_size)), requires_grad = False)

    for step in range(steps):
        noise_2 = nnet.tensor(np.random.normal(0, 1, (samples_num, noise_size)), requires_grad = False)

        noise_interp = (np.linspace(noise_1.data, noise_2.data, interval))

        noise_1 = noise_2

        for vectors in noise_interp:  
            generated_images = generator(vectors).reshape(-1, 28, 28).data * 127.5 + 127.5

            images.append(get_images_set(generated_images).convert('L').convert('P'))

    images[0].save('generated images/gan_vectors_interpolation.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

create_vectors_interpolation()


each_epoch_generated_images = []
for epoch_samples in each_epoch_generated_samples:
    each_epoch_generated_images.append(get_images_set(epoch_samples * 127.5 + 127.5).convert('L').convert('P'))

each_epoch_generated_images[0].save('generated images/gan_training_process.gif', save_all=True, append_images=each_epoch_generated_images[1:], optimize=False, duration=150, loop=0)






