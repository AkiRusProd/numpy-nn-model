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

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')

        if number_to_take != None:
            if str(line[0]) == number_to_take:
                inputs.append(np.asfarray(line[1:]) / 255)
        else:
            inputs.append(np.asfarray(line[1:]) / 255)
        
    return inputs


inputs = np.asfarray(prepare_data(test_data, number_to_take = '3'))

def add_noise(data):
    noise_factor = 0.5

    noisy_data = data + noise_factor * np.random.normal(0, 1, (data.shape))

    return np.clip(noisy_data, 0, 1)


noisy_inputs = add_noise(inputs)

latent_dim = 16

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
loss, decoder_loss, kl_loss = vae.fit(noisy_inputs[0:10000], inputs[0:10000], epochs = 100, batch_size = 100)

vae.save("saved models/VAE")
vae.load("saved models/VAE")


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


noised_images, denoised_images = [], []

for i in range(x_num * y_num):
    rand_i = np.random.randint(0, len(inputs))

    outputs = vae.predict(noisy_inputs[rand_i].reshape(1, 784))

    noised_images.append(np.reshape(noisy_inputs[rand_i], (image_size, image_size)) * 255)
    denoised_images.append(np.reshape(outputs, (image_size, image_size)) * 255)

get_images_set(noised_images).save(f'examples/autoencoder images/vae noised set of images.jpeg')
get_images_set(denoised_images).save(f'examples/autoencoder images/vae denoised set of images.jpeg')

'''Build D and G losses'''
plt.plot([i for i in range(len(loss))], loss)
plt.plot([i for i in range(len(decoder_loss))], decoder_loss)
plt.plot([i for i in range(len(kl_loss))], kl_loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(['Loss (Decoder Loss + KL Loss)', 'Decoder Loss', 'KL Loss'])
plt.show()
