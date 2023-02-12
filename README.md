# numpy-nn-model
Сustom CPU torch style machine learning framework implemented on numpy with automatic differentiation.

## Some information and features:

### Implemented Activation Functions:
1) Sigmoid
2) Tanh
3) Softmax
4) Softplus
5) Softsign
6) Swish
7) Mish
8) TanhExp
9) ReLU
10) LeakyReLU
11) ELU
12) SELU
13) GELU


*[See Activation Functions...](neunet/nn/activations.py)*

### Implemented Optimizers:
1) SGD
2) Momentum
3) RMSProp
4) Adam
5) NAdam
6) AdaMax
7) AdaGrad
8) AdaDelta

*[See Optimizers...](neunet/optim.py)*

### Implemented Loss Functions:
1) MSE
2) BinaryCrossEntropy

*[See Loss Functions...](neunet/nn/losses.py)*

### Implemented Layers:
1) Linear
2) Dropout
3) BatchNorm1d
4) BatchNorm2d
5) LayerNorm
6) Conv2d
7) ConvTranspose2d
8) MaxPool2d
9) AvgPool2d
10) ZeroPad2d
11) Flatten
12) Embedding
13) Bidirectional
14) RNN
15) LSTM
16) GRU

*[See Layers...](neunet/nn/layers)*


### Implemented Tensor Operations:

1) add, sub, mul, div, dot, abs
2) sum, mean, var, max, min, maximum, minimum
3) transpose, reshape, concatenate, flip
4) power, exp, log, sqrt, sin, cos, tanh

### Tensor Operations with autograd Example:
```python
import neunet as nnet
import numpy as np


x = nnet.tensor([[7.0, 6.0, 5.0], [4.0, 5.0, 6.0]])
y = nnet.tensor([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
z = nnet.tensor([[2.3, 3.4], [4.5, 5.6]])

out = nnet.tanh(1 / nnet.log(nnet.concatenate([(x @ y) @ z, nnet.exp(x) / nnet.sqrt(x)], axis = 1)))
out.backward(np.ones_like(out.data))

print(out, '\n')
# Tensor([[0.16149469 0.15483168 0.16441284 0.19345127 0.23394899]
#  [0.16278233 0.15598084 0.29350953 0.23394899 0.19345127]], requires_grad=True)

print(x.grad, '\n')
# [[-0.02619586 -0.03680556 -0.05288506]
#  [-0.07453468 -0.05146679 -0.03871813]]
print(y.grad, '\n')
# [[-0.00294922 -0.00530528]
#  [-0.00296649 -0.0053364 ]
#  [-0.00298376 -0.00536752]]
print(z.grad, '\n')
# [[-0.00628155 -0.00441077]
#  [-0.00836989 -0.00587731]]
```



### Model Examples:
All examples was trained on [MNIST](https://pjreddie.com/projects/mnist-in-csv/) Dataset   

Code:   
*[Base training module](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/nn_model.py)*

#### Convolutional Classifier
```python
from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np

from data_loader import load_mnist

image_size = (1, 28, 28)

training_dataset, test_dataset, training_targets, test_targets = load_mnist()
training_dataset = training_dataset / 127.5-1
test_dataset = test_dataset / 127.5-1

class Conv2dClassifier(nn.Module):
    def __init__(self):
        super(Conv2dClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.bnorm = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(784, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.maxpool2(x)

        x = self.bnorm(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        
        return x

classifier = Conv2dClassifier()

def one_hot_encode(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))

    for i in range(labels.shape[0]):
        one_hot_labels[i, int(labels[i])] = 1

    return one_hot_labels


loss_fn = nn.MSELoss()
optimizer = Adam(classifier.parameters(), lr = 0.001)

batch_size = 100
epochs = 3

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(training_dataset), batch_size), desc = 'epoch ' + str(epoch))
    for i in tqdm_range:
        batch = training_dataset[i:i+batch_size]
        batch = batch.reshape(batch.shape[0], image_size[0], image_size[1], image_size[2])
        batch = nnet.tensor(batch)

        labels = one_hot_encode(training_targets[i:i+batch_size])

        optimizer.zero_grad()
        outputs = classifier(batch)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        tqdm_range.set_description(f'epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}')
```
###### (prediction on test MNIST data with this model is 97 %)

Code:   
*[Model Example](examples/convolutional_digits_classifier.py)*

#### LSTM Classifier
```python
from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import os

from data_loader import load_mnist

image_size = (1, 28, 28)

training_dataset, test_dataset, training_targets, test_targets = load_mnist()
training_dataset = training_dataset / 127.5-1
test_dataset = test_dataset / 127.5-1

class RecurrentClassifier(nn.Module):
    def __init__(self):
        super(RecurrentClassifier, self).__init__()

        self.lstm1 = nn.LSTM(28, 128, return_sequences=True)
        self.lstm2 = nn.LSTM(128, 128, return_sequences=False)
        self.fc1 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


classifier = RecurrentClassifier()

def one_hot_encode(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, int(labels[i])] = 1

    return one_hot_labels


loss_fn = nn.MSELoss()
optimizer = Adam(classifier.parameters(), lr = 0.0001)

batch_size = 100
epochs = 5

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(training_dataset), batch_size), desc = 'epoch ' + str(epoch))
    for i in tqdm_range:
        batch = training_dataset[i:i+batch_size]
        batch = batch.reshape(batch.shape[0], image_size[1], image_size[2])
        batch = nnet.tensor(batch)

        labels = one_hot_encode(training_targets[i:i+batch_size])

        optimizer.zero_grad()

        outputs = classifier(batch)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        tqdm_range.set_description(f'epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}')
```
###### (prediction on test MNIST data with this model is 93 %)

Code:   
*[Model Example](examples/recurrent_digits_classifier.py)*

#### Variational Autoencoder (VAE)
```python
from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from data_loader import load_mnist



noisy_inputs = False
samples_num = 25

def add_noise(data):
    noise_factor = 0.5

    noisy_data = data + noise_factor * np.random.normal(0, 1, (data.shape))

    return np.clip(noisy_data, 0, 1)

training_data, test_data, training_labels, test_labels = load_mnist()
training_data = training_data / 255
test_data = test_data / 255

latent_size = 2


class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        self.mu_encoder = nn.Linear(latent_size, latent_size)
        self.logvar_encoder = nn.Linear(latent_size, latent_size)

        self.loss_fn = nn.BCELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = nnet.tensor(np.random.normal(0, 1, size=std.shape))
        z = mu + eps * std
        return z

    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        BCE = self.loss_fn(x_recon, x)
        KLD = -0.5 * nnet.sum(1 + logvar - mu.power(2) - logvar.exp())
        return BCE + KLD
    
    def train(self, in_x, out_x, optimizer):
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

vae = VAE(28 * 28, latent_size)
optimizer = Adam(vae.parameters(), lr=0.001)


batch_size = 100
epochs = 30

for epoch in range(epochs):
    
    tqdm_range = tqdm(range(0, len(training_data), batch_size), desc = 'epoch %d' % epoch)
    for i in tqdm_range:
        batch = training_data[i:i+batch_size]

        in_batch = nnet.tensor(batch, requires_grad=False).reshape(-1, 28 * 28)
        if noisy_inputs:
            in_batch = nnet.tensor(add_noise(in_batch.data), requires_grad=False)

        out_batch = nnet.tensor(batch, requires_grad=False).reshape(-1, 28 * 28)

        loss = vae.train(in_batch, out_batch, optimizer)
        
        tqdm_range.set_description(f'epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}')


    generated = vae.decode(nnet.tensor(np.random.normal(0, 1, size=(samples_num, latent_size)), requires_grad=False)).data
    
    # samples = training_data[np.random.randint(0, len(training_data), samples_num)]
    # if noisy_inputs:
    #     samples = add_noise(samples)
    # generated = vae.reconstruct(nnet.tensor(samples, requires_grad=False).reshape(-1, 28 * 28)).data

    for i in range(25):
        image = generated[i] * 255
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated images/{i}.png')
```
Code:   
*[Model example](examples/vae.py)*   

##### VAE Results:
Noisy Data Example | Noise Removed Data Example
:-------------------------:|:-------------------------:
<img src="generated images/vae_input_samples.jpg"> |  <img src="generated images/vae_output_samples.jpg">

##### VAE 2D latent dim Plots:
Digits location in 2D latent space:   
<img src="generated images/vae_2d_latent_space.jpg" width=75% height=75%>

Digits labels in 2D latent space:   
<img src="generated images/vae_2d_latent_space_labels.jpg" width=75% height=75%>

#### Generative Adversarial Network (GAN)
```python
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

batch_size = 64
epochs = 3


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

    noise = nnet.tensor(np.random.normal(0, 1, (samples_num, noise_size)), requires_grad = False)
    generated_images = generator(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], 1, 28, 28)
    generated_images = generated_images.data

    for i in range(samples_num):
        image = generated_images[i] * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = image.reshape(28, 28)
        image = Image.fromarray(image)
        image.save(f'generated images/{i}.png')
```


Code:   
*[Model example](examples/gan.py)*   


##### GAN Results:
Training process Example | Interpolation between images Example
:-------------------------:|:-------------------------:
<img src="generated images/gan_training_process.gif"> |  <img src="generated images/gan_vectors_interpolation.gif">


### TODO:
- [ ] Fix bugs (if found), clean up and refactor the code
- [ ] Add more Tensor operations
- [ ] Add more layers, activations, optimizers, loss functions, etc
- [ ] Add more examples
- [ ] Add CuPy support
