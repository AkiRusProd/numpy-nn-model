import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from PIL import Image

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, RepeatVector, \
TimeDistributed, RNN, LSTM, GRU, Bidirectional
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam, Nadam
from nnmodel.modules import VAE

training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


def prepare_data(data, number_to_take = None):
    inputs = []
    labels = []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_take != None:
            if str(line[0]) == number_to_take:
                inputs.append(np.asfarray(line[1:]) / 255)
                labels.append([np.asfarray(line[0])])
        else:
            inputs.append(np.asfarray(line[1:]) / 255)
            labels.append([np.asfarray(int(line[0]))])
        
    return np.asfarray(inputs), np.asfarray(labels)


inputs, labels = prepare_data(training_data, number_to_take = None)


latent_dim = 2

encoder = Model()
encoder.add(Dense(256, input_shape = (784), use_bias=True))
encoder.add(Activation('relu'))
encoder.add(Dense(128, use_bias=True))
encoder.add(Activation('relu'))
encoder.add(Dense(latent_dim * 2, use_bias=True))

decoder = Model()
decoder.add(Dense(128, input_shape = (latent_dim), use_bias=True))
decoder.add(Activation('leaky_relu'))
decoder.add(Dense(256, use_bias=True))
decoder.add(Activation('leaky_relu'))
decoder.add(Dense(784, activation='sigmoid', use_bias=True))

# encoder = Model()
# encoder.add(Reshape((1, 28, 28)))
# encoder.add(Conv2D(8, 3, activation="relu", stride=2, padding="same"))
# encoder.add(Flatten())
# encoder.add(Dense(64, activation="relu"))
# encoder.add(Dense(latent_dim * 2, use_bias=True))

# decoder = Model()
# decoder.add(Dense(14 * 14 * 8, input_shape = latent_dim, activation="relu"))
# decoder.add(Reshape((8, 14, 14)))
# decoder.add(Conv2DTranspose(8, 3, activation="relu", stride=2, padding="same"))
# decoder.add(Conv2DTranspose(1, 3, activation="sigmoid", padding="same"))
# decoder.add(Flatten())

vae = VAE(encoder, decoder)
vae.compile(optimizer = Adam(), loss = 'binary_crossentropy')
loss, decoder_loss, kl_loss = vae.fit(inputs, inputs, epochs = 30, batch_size = 100)

vae.save("saved models/2d_latent_dim_VAE")
vae.load("saved models/2d_latent_dim_VAE")


x_num = 5
y_num = 5
margin = 15
image_size = 28


def get_images_set(images):
    '''Create set of images'''
    images_array = np.full((x_num * (margin + image_size), y_num * (margin + image_size)), 255, dtype=np.uint8)
    num = 0
    for i in range(y_num):
        for j in range(x_num):
            y = i*(margin + image_size)
            x = j*(margin + image_size)

            images_array[y:y+image_size,x:x+image_size] = images[num]
            num+=1

    images_array = images_array[: (y_num - 1) * (image_size + margin) + image_size, : (x_num - 1) * (image_size + margin) + image_size]

    return Image.fromarray(images_array).convert("L")


denoised_images = []

for i in range(x_num * y_num):
    rand_i = np.random.randint(0, len(inputs))

    outputs = vae.predict(inputs[rand_i].reshape(1, 784)) #from encoder; input: image
    # outputs = vae.predict_decoder(np.random.normal(0, 1, (1, latent_dim)))
    denoised_images.append(np.reshape(outputs, (image_size, image_size)) * 255)


get_images_set(denoised_images).save(f'examples/autoencoder images/2d dim vae set of images.jpeg')



'''Build losses'''
plt.plot([i for i in range(len(loss))], loss)
plt.plot([i for i in range(len(decoder_loss))], decoder_loss)
plt.plot([i for i in range(len(kl_loss))], kl_loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(['Loss (Decoder Loss + KL Loss)', 'Decoder Loss', 'KL Loss'])
plt.show()


"""Visualize latent space only with latent_dim = 2"""
def plot_latent_space_digits(vae, n=30, figsize=15):
    if latent_dim != 2:
        print('Can`t plot 2d latent space for non-2d latent space')
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
            x_decoded = vae.predict_decoder(z_sample)
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

    plt.savefig(f'examples/autoencoder images/vae 2d latent space.jpeg')
    plt.show()

plot_latent_space_digits(vae)

"""Visualize latent space of labels only with latent_dim = 2"""
def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = vae.predict_encoder(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    plt.savefig(f'examples/autoencoder images/vae 2d latent labels clusters space')
    plt.show()
    

plot_label_clusters(vae, inputs, labels)



