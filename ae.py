from autograd import Tensor, MSELoss, Sigmoid, ReLU, BCELoss
from nn import Linear, Sequential, Module
from optim import SGD, Adam
from tqdm import tqdm
import numpy as np
import os
from PIL import Image



training_data = open('datasets/mnist/mnist_train.csv','r').readlines()
test_data = open('datasets/mnist/mnist_test.csv','r').readlines()

image_size = (1, 28, 28)

# def prepare_data(data):
#     inputs, targets = [], []

#     for raw_line in tqdm(data, desc = 'preparing data'):

#         line = raw_line.split(',')
    
#         inputs.append(np.asfarray(line[1:]).reshape(1, 28, 28) / 255)#/ 255 [0; 1]  #/ 127.5-1 [-1; 1]
#         targets.append(int(line[0]))


#     # one hot encoding
#     targets = np.eye(10)[targets]

#     return inputs, targets

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





# if not os.path.exists("datasets/mnist/mnist_train.npy"):
#     training_inputs, training_targets = prepare_data(training_data)
#     np.save("datasets/mnist/mnist_train.npy", training_inputs)
#     np.save("datasets/mnist/mnist_train_targets.npy", training_targets)
# else:
#     training_inputs = np.load("datasets/mnist/mnist_train.npy")
#     training_targets = np.load("datasets/mnist/mnist_train_targets.npy")

# test_inputs = np.asfarray(prepare_data(test_data, number_to_take = '3'))
# dataset = test_inputs


dataset = np.asfarray(prepare_data(training_data))
# dataset = np.asfarray(prepare_data(training_data, '7'))


        

        
    


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
            Linear(128, latent_size * 2),
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
        self.loss_fn = BCELoss()

    def forward(self, x):
        x_enc = self.encoder(x)

        mu, logvar = x_enc[:, :self.latent_size], x_enc[:, self.latent_size:] #
        std = logvar.mul(0.5).exp()
        eps = Tensor(np.random.normal(0, 1, size=std.shape))
        z = mu + eps * std
        x_recon = self.decoder(z)

        # x_recon = self.decoder(x_enc)

        return x_recon, mu, logvar

    # def loss_function(self, x, x_recon):
    #     MSE = self.loss_fn(x_recon, x)
    #     return MSE

    def loss_function(self, x, x_recon, mu, logvar):
        MSE = self.loss_fn(x_recon, x)
        KLD = Tensor(-0.5) * Tensor.sum(Tensor(1) + logvar - mu.power(2) - logvar.exp())
        return MSE + KLD
    
    # train without KL divergence
    def train(self, x, optimizer):
        x_recon, mu, logvar = self.forward(x)

        # loss = self.loss_function(x_recon, x)
        loss = self.loss_function(x, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
        

    def generate(self, n):
        z = Tensor(np.random.normal(0, 1, size=(n, self.latent_size)))
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)[0]

vae = VAE(28 * 28, 2)
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



    # generated = vae.generate(100)
    generated = vae.reconstruct(Tensor(dataset[:100], requires_grad=False).reshape(-1, 28 * 28))
    generated = generated.data

    for i in range(100):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated_images/{i}.png')

