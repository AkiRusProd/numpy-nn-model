# from autograd import Tensor, Linear, MSELoss, Sigmoid, Sequential, SGD, Adam, ReLU, LeakyReLU, Tanh
from torch.nn import Linear, MSELoss, Sigmoid, Sequential, ReLU, LeakyReLU, Tanh
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


dataset = train_inputs / 127.5-1 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]


generator = Sequential(
    Linear(100, 256),
    LeakyReLU(),
    Linear(256, 512),
    LeakyReLU(),
    Linear(512, 784),
    Tanh()
).to('cuda:0')

discriminator = Sequential(
    Linear(784, 128),
    LeakyReLU(),
    Linear(128, 64),
    LeakyReLU(),
    Linear(64, 1),
    Sigmoid()
).to('cuda:0')

loss = MSELoss().to('cuda:0')

g_optimizer = Adam(generator.parameters(), lr = 0.001, betas = (0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr = 0.001, betas = (0.5, 0.999))

batch_size = 64



for epoch in range(100):
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = f'epoch {epoch}')
    for i in tqdm_range:
        batch = dataset[i:i+batch_size]
        batch = Tensor(batch).to('cuda:0')

        # train discriminator
        d_optimizer.zero_grad()

        # train discriminator on real data
        real_data = batch
        real_data = real_data.reshape(real_data.shape[0], -1)
        # real_data = Tensor(real_data, requires_grad = False) # Double Tensor MemoryAllocationError BUG

        real_data_prediction = discriminator(real_data) 
        real_data_loss = loss(real_data_prediction, torch.ones((real_data_prediction.shape[0], 1)).to('cuda:0'))
        real_data_loss.backward()
        d_optimizer.step()

        # train discriminator on fake data
        noise = torch.randn((batch_size, 100)).to('cuda:0')
        fake_data = generator(noise)
        fake_data_prediction = discriminator(fake_data)
        fake_data_loss = loss(fake_data_prediction, torch.zeros((fake_data_prediction.shape[0], 1)).to('cuda:0'))
        fake_data_loss.backward()
        d_optimizer.step()

        # train generator
        g_optimizer.zero_grad()

        noise = torch.randn((batch_size, 100)).to('cuda:0')
        fake_data = generator(noise)
        fake_data_prediction = discriminator(fake_data)
        fake_data_loss = loss(fake_data_prediction, torch.ones((fake_data_prediction.shape[0], 1)).to('cuda:0'))
        fake_data_loss.backward()
        g_optimizer.step()

        # tqdm epoch Generator Loss and Discriminator Loss
        g_loss = -np.log(fake_data_prediction.data.cpu().numpy()).mean()
        d_loss = -np.log(real_data_prediction.data.cpu().numpy()).mean() - np.log(1 - fake_data_prediction.data.cpu().numpy()).mean()
        tqdm_range.set_description(
            f'epoch {epoch} GLoss: {g_loss:.4f} D Loss: {d_loss:.4f}'
        )

    # # save model
    # if not os.path.exists('models'):
    #     os.mkdir('models')

    # np.save('models/generator.npy', generator.state_dict())
    # np.save('models/discriminator.npy', discriminator.state_dict())

    # save generated images
    if not os.path.exists('generated_images'):
        os.mkdir('generated_images')

    noise = torch.randn((100, 100)).to('cuda:0')
    generated_images = generator(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], 1, 28, 28)
    generated_images = generated_images.data

    for i in range(100):
        image = generated_images[i] * 127.5 + 127.5
        image = image.cpu().numpy().astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated_images/{i}.png')

    print('epoch %d finished' % epoch)






