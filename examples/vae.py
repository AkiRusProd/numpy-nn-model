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

latent_size = 2

device = "cpu"


class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

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
        self.mu_encoder = nn.Linear(latent_size, latent_size)
        self.logvar_encoder = nn.Linear(latent_size, latent_size)

        self.loss_fn = nn.BCELoss(reduction="sum")

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = nnet.tensor(np.random.normal(0, 1, size=std.shape), device=device)
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
        BCE = self.loss_fn(x_recon, x)
        KLD = -0.5 * nnet.sum(1 + logvar - mu.power(2) - logvar.exp())
        return BCE + KLD

    def train_step(self, in_x, out_x, optimizer):
        x_recon, mu, logvar = self.forward(in_x)

        loss = self.loss_function(out_x, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def encode(self, x):
        x = self.encoder(x)

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)

        z = self.reparameterize(mu, logvar)

        return z

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)[0]


vae = VAE(28 * 28, latent_size).to(device)
optimizer = Adam(vae.parameters(), lr=0.0005)


batch_size = 100
epochs = 20

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(training_data), batch_size), desc="epoch %d" % epoch)
    vae.train()
    for i in tqdm_range:
        batch = training_data[i : i + batch_size]

        in_batch = nnet.tensor(batch, device=device).reshape(-1, 28 * 28)
        if noisy_inputs:
            in_batch = nnet.tensor(add_noise(in_batch.cpu().numpy()), device=device)

        out_batch = nnet.tensor(batch, device=device).reshape(-1, 28 * 28)

        loss = vae.train_step(in_batch, out_batch, optimizer)

        tqdm_range.set_description(
            f"epoch: {epoch + 1}/{epochs}, loss: {loss.item():.7f}"
        )

    generated = (
        vae.decode(
            nnet.tensor(
                np.random.normal(0, 1, size=(samples_num, latent_size)),
                device=device,
            )
        )
        .to("cpu")
        .detach()
        .numpy()
    )

    # samples = training_data[np.random.randint(0, len(training_data), samples_num)]
    # if noisy_inputs:
    #     samples = add_noise(samples)
    # generated = vae.reconstruct(nnet.tensor(samples, requires_grad=False).reshape(-1, 28 * 28)).data
    vae.eval()
    for i in range(25):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f"generated images/{i}.png")


vae.eval()


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
generated = (
    vae.reconstruct(nnet.tensor(samples, device=device).reshape(-1, 28 * 28))
    .to("cpu")
    .detach()
    .numpy()
)

get_images_set(samples.reshape(-1, 28, 28) * 255).save(
    "generated images/vae_in_samples.jpeg"
)
get_images_set(generated.reshape(-1, 28, 28) * 255).save(
    "generated images/vae_out_samples.jpeg"
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
            x_decoded = (
                vae.decode(nnet.tensor(z_sample, device=device)).to("cpu").detach().numpy()
            )
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

    plt.savefig(f"generated images/vae 2d latent space.jpeg")
    plt.show()


plot_latent_space_digits()

"""Visualize latent space of labels only with latent_dim = 2"""


def plot_label_clusters(data, labels):
    if latent_size != 2:
        print("Can`t plot 2d latent space for non-2d latent space")
        return
    # display a 2D plot of the digit classes in the latent space
    z_mean = (
        vae.encode(nnet.tensor(data, device=device).reshape(-1, 28 * 28))
        .to("cpu")
        .detach()
        .numpy()
    )
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    plt.savefig(f"generated images/vae 2d latent space labels.jpeg")
    plt.show()


plot_label_clusters(
    add_noise(training_data) if noisy_inputs else training_data, training_labels
)
