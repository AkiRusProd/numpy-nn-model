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
input_dataset = dataset / 255  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]
target_dataset = dataset / 255  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]

noisy_inputs = True


def add_noise(data):
    noise_factor = 0.5

    noisy_data = data + noise_factor * np.random.normal(0, 1, (data.shape))

    return np.clip(noisy_data, 0, 1)


if noisy_inputs:
    input_dataset = add_noise(input_dataset)

latent_size = 64


model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, latent_size),
    nn.ReLU(),
    nn.Linear(latent_size, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid(),
)

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

batch_size = 100
epochs = 100

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(input_dataset), batch_size), desc="epoch %d" % epoch)
    model.train()
    for i in tqdm_range:
        input_batch = input_dataset[i : i + batch_size]
        input_batch = nnet.tensor(input_batch, requires_grad=False).reshape(-1, 28 * 28)

        target_batch = target_dataset[i : i + batch_size]
        target_batch = nnet.tensor(target_batch, requires_grad=False).reshape(
            -1, 28 * 28
        )

        optimizer.zero_grad()
        output = model(input_batch)
        loss = loss_fn(output, target_batch)
        loss.backward()
        optimizer.step()

        tqdm_range.set_description(
            f"epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}"
        )

    generated = model(
        nnet.tensor(input_dataset[:25], requires_grad=False).reshape(-1, 28 * 28)
    ).data
    model.eval()
    for i in range(25):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f"generated images/{i}.png")
