import sys, os
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from data_loader import load_mnist

noisy_inputs = False

image_size = (1, 28, 28)
x_num, y_num = 5, 5
samples_num = x_num * y_num
margin = 15


def add_noise(data):
    noise_factor = 0.5

    noisy_data = data + noise_factor * np.random.normal(0, 1, (data.shape))

    return np.clip(noisy_data, 0, 1)


training_data, test_data, training_labels, test_labels = load_mnist()
training_data = (
    training_data / 255
)  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]
test_data = test_data / 255  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]

num_embeddings = 100
latent_size = 2


class VQVAE(nn.Module):
    def __init__(self, input_size, latent_size, num_embeddings):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size),
        )

        self.codebook = nn.Embedding(self.num_embeddings, self.latent_size)
        self.codebook.weight = nnet.tensor(
            np.random.uniform(
                -1 / self.num_embeddings,
                1 / self.num_embeddings,
                (self.num_embeddings, self.latent_size),
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.quantize(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q

    def quantize(self, z):
        # distances = ((z[:, None, :] - self.codebook.weight) ** 2).sum(-1)
        # z = z.reshape(-1, self.latent_size)
        similarity = nnet.matmul(z, self.codebook.weight.T)
        distances = (
            nnet.sum(z**2, axis=1, keepdims=True)
            + nnet.sum(self.codebook.weight**2, axis=1)
            - 2 * similarity
        )

        min_indices = nnet.argmin(distances, axis=1)
        z_q = self.codebook(min_indices)

        return z_q, min_indices

    def loss_function(self, x, x_recon, z_e, z_q, beta=0.25):
        recon_loss = self.loss_fn(x_recon, x)
        vq_loss = self.loss_fn(z_q, z_e.detach())
        commit_loss = self.loss_fn(z_q.detach(), z_e)
        return recon_loss + vq_loss + beta * commit_loss

    def train_step(self, in_x, out_x, optimizer):
        x_recon, z_e, z_q = self.forward(in_x)

        loss = self.loss_function(out_x, x_recon, z_e, z_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.quantize(z_e)
        return z_q

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)[0]


vqvae = VQVAE(28 * 28, latent_size, num_embeddings)
optimizer = Adam(vqvae.parameters(), lr=0.0005)


batch_size = 100
epochs = 30

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(training_data), batch_size), desc="epoch %d" % epoch)
    vqvae.train()
    for i in tqdm_range:
        batch = training_data[i : i + batch_size]

        in_batch = nnet.tensor(batch, requires_grad=False).reshape(-1, 28 * 28)
        if noisy_inputs:
            in_batch = nnet.tensor(add_noise(in_batch.data), requires_grad=False)

        out_batch = nnet.tensor(batch, requires_grad=False).reshape(-1, 28 * 28)

        loss = vqvae.train_step(in_batch, out_batch, optimizer)

        tqdm_range.set_description(
            f"epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}"
        )

    generated = vqvae.decode(
        nnet.tensor(
            np.random.normal(0, 1, size=(samples_num, latent_size)), requires_grad=False
        )
    ).data

    # samples = training_data[np.random.randint(0, len(training_data), samples_num)]
    # if noisy_inputs:
    #     samples = add_noise(samples)
    # generated = vqvae.reconstruct(nnet.tensor(samples, requires_grad=False).reshape(-1, 28 * 28)).data
    vqvae.eval()
    for i in range(25):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f"generated images/{i}.png")


vqvae.eval()


def get_images_set(images):
    images_array = np.full(
        (y_num * (margin + image_size[1]), x_num * (margin + image_size[2])),
        255,
        dtype=np.uint8,
    )
    num = 0
    for i in range(y_num):
        for j in range(x_num):
            y = i * (margin + image_size[1])
            x = j * (margin + image_size[2])

            images_array[y : y + image_size[1], x : x + image_size[2]] = images[num]
            num += 1

    images_array = images_array[
        : (y_num - 1) * (image_size[1] + margin) + image_size[1],
        : (x_num - 1) * (image_size[2] + margin) + image_size[2],
    ]

    return Image.fromarray(images_array).convert("L")


samples = test_data[np.random.randint(0, len(test_data), samples_num)]
if noisy_inputs:
    samples = add_noise(samples)
generated = vqvae.reconstruct(
    nnet.tensor(samples, requires_grad=False).reshape(-1, 28 * 28)
).data

get_images_set(samples.reshape(-1, 28, 28) * 255).save(
    "generated images/vqvae_in_samples.jpeg"
)
get_images_set(generated.reshape(-1, 28, 28) * 255).save(
    "generated images/vqvae_out_samples.jpeg"
)


"""Visualize latent space only with latent_dim = 2"""


def plot_latent_space_digits(n=30, figsize=15):
    if latent_size != 2:
        print("Can`t plot 2d latent space for non-2d latent space")
        return
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vqvae.decode(nnet.tensor(z_sample, requires_grad=False)).data
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")

    plt.savefig(f"generated images/vqvae 2d latent space.jpeg")
    plt.show()


plot_latent_space_digits()

"""Visualize latent space of labels only with latent_dim = 2"""


def plot_label_clusters(data, labels):
    if latent_size != 2:
        print("Can`t plot 2d latent space for non-2d latent space")
        return
    # display a 2D plot of the digit classes in the latent space
    z_mean = vqvae.encode(
        nnet.tensor(data, requires_grad=False).reshape(-1, 28 * 28)
    ).data
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    plt.savefig(f"generated images/vqvae 2d latent space labels.jpeg")
    plt.show()


plot_label_clusters(
    add_noise(training_data) if noisy_inputs else training_data, training_labels
)
