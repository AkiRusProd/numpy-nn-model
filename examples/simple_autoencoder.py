import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from PIL import Image

from nnmodel.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Activation
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam, Nadam


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

model = Model()
model.add(Dense(256, input_shape = (784), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(784, activation='sigmoid'))

"""Convolutional AutoEncoder model topology example (works much slower):"""
# model.add(Reshape((1, 28, 28)))
# model.add(Conv2D(kernels_num = 16, kernel_shape=(3,3), stride=(2, 2), padding='same', input_shape=(1, 28, 28)))
# model.add(Activation(LeakyReLU(alpha=0.2)))
# model.add(Conv2D(16, (3,3), stride=(2, 2), padding='same'))
# model.add(Activation(LeakyReLU(alpha=0.2)))
# model.add(Flatten())
# model.add(Dense(32 * 7 * 7, activation='relu'))
# model.add(Reshape((32, 7, 7)))
# model.add(Conv2DTranspose(16, (4,4), stride=(2,2), padding='same'))
# model.add(Activation(LeakyReLU(alpha=0.2)))
# model.add(Conv2DTranspose(16, (4,4), stride=(2,2), padding='same'))
# model.add(Activation(LeakyReLU(alpha=0.2)))
# model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
# model.add(Flatten())

model.compile(optimizer = Adam(), loss = 'binary_crossentropy')
loss = model.fit(noisy_inputs, inputs, epochs = 100, batch_size = 100)

model.save("saved models/AE")
model.load("saved models/AE")


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

    _, outputs = model.predict(noisy_inputs[rand_i].reshape(1, 784))

    noised_images.append(np.reshape(noisy_inputs[rand_i], (image_size, image_size)) * 255)
    denoised_images.append(np.reshape(outputs, (image_size, image_size)) * 255)

get_images_set(noised_images).save(f'examples/autoencoder images/ae noised set of images.jpeg')
get_images_set(denoised_images).save(f'examples/autoencoder images/ae denoised set of images.jpeg')

'''Build D and G losses'''
plt.plot([i for i in range(len(loss))], loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend(['Loss'])
plt.show()
