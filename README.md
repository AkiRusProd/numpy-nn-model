# numpy-nn-model
Сustom CPU/GPU torch style machine learning framework with automatic differentiation for creating neural networks, implemented on numpy with cupy.

## Some information and features:

<details>
<summary>Activation Functions</summary>

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

</details>


<details>
<summary>Optimizers</summary>

1) SGD
2) Momentum
3) RMSProp
4) Adam
5) NAdam
6) AdaMax
7) AdaGrad
8) AdaDelta

*[See Optimizers...](neunet/optim.py)*

</details>


<details>
<summary>Loss Functions</summary>

1) MSELoss
2) BCELoss
3) CrossEntropyLoss
4) NLLLoss
5) L1Loss
6) KLDivLoss

*[See Loss Functions...](neunet/nn/losses.py)*

</details>


<details>
<summary>Modules</summary>

1) Linear
2) Dropout
3) BatchNorm1d
4) BatchNorm2d
5) LayerNorm
6) RMSNorm
7) Conv2d
8) ConvTranspose2d
9) MaxPool2d
10) AvgPool2d
11) ZeroPad2d
12) Flatten
13) Embedding
14) Bidirectional
15) RNN
16) LSTM
17) GRU

*[See Modules...](neunet/nn/layers)*

</details>


<details>
<summary>Tensor Operations</summary>

1) add, sub, mul, div, matmul, abs
2) sum, mean, var, max, min, maximum, minimum, argmax, argmin   
3) transpose, swapaxes, reshape, concatenate, flip, slicing
4) power, exp, log, sqrt, sin, cos, tanh
5) ones, zeros, ones_like, zeros_like, arange, rand, randn

</details>


### Tensor Operations with autograd Example:
```python
import neunet as nnet
import numpy as np


x = nnet.tensor([[7.0, 6.0, 5.0], [4.0, 5.0, 6.0]], requires_grad=True)
y = nnet.tensor([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], requires_grad=True)
z = nnet.tensor([[2.3, 3.4], [4.5, 5.6]], requires_grad=True)

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
Some [examples](examples/) were trained on the [MNIST](https://pjreddie.com/projects/mnist-in-csv/) Dataset   

#### All of them:
1. *[Autoencoder](examples/ae.py)*        
2. *[Convolutional Digits Classifier](examples/convolutional_digits_classifier.py)*    
3. *[Conway`s Game of Life](examples/conway.py)*  
4. *[Denoising Diffusion Probabilistic Model](examples/ddpm.py)*
5. *[Generative Adversarial Network](examples/gan.py)*     
6. *[Recurrent Digits Classifier](examples/recurrent_digits_classifier.py)*    
7. *[Recurrent Sequences Classifier](examples/recurrent_sequences_classifier.py)*    
8. *[Variational Autoencoder](examples/vae.py)*    
9. *[Vector Quantized Variational Autoencoder](examples/vqvae.py)* 
10. *[Word2Vec](examples/word2vec.py)*



#### More details about some of them:

<details>
<summary>Convolutional Classifier</summary>

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
</details>

<details>
<summary>LSTM Classifier</summary>

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
</details>

<details>
<summary>Variational Autoencoder (VAE)</summary>

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
</details>

<details>
<summary>Generative Adversarial Network (GAN)</summary>

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
</details>

<details>
<summary>Conway`s Game of Life Neural Network Simulation</summary>

```python
import itertools
import numpy as np
import neunet
import neunet.nn as nn
import neunet.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from tqdm import tqdm

'''
Conway's Game of Life

This example illustrates how to implement a neural network that can be trained to simulate Conway's Game of Life.
'''

N = 128
# Randomly create a grid
# grid = np.random.binomial(1, p = 0.2, size = (N, N))

# or define for example the Glider Gun configuration as shown in 
# https://conwaylife.com/wiki/Gosper_glider_gun 
# Other examples can be found in
# https://conwaylife.com/patterns/

grid = np.zeros((N, N))

gun_pattern_src = """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
"""

# Split the pattern into lines
lines = gun_pattern_src.strip().split('\n')

# Convert each line into an array of 1s and 0s
gun_pattern_grid = np.array([[1 if char == 'O' else 0 for char in line] for line in lines])

grid[0:gun_pattern_grid.shape[0], 0:gun_pattern_grid.shape[1]] = gun_pattern_grid

def update(grid):
    '''
    Native implementation of Conway's Game of Life
    '''
    updated_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Use the modulo operator % to ensure that the indices wrap around the grid.
            # Using the modulus operator % to index an array creates the effect of a "toroidal" mesh, which can be thought of as the surface of a donut
            n_alived_neighbors = int(grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, j] + grid[(i-1)%N, (j+1)%N] + grid[i, (j-1)%N] + grid[i, (j+1)%N] + grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, j] + grid[(i+1)%N, (j+1)%N])

            if grid[i, j] == 1:
                if n_alived_neighbors < 2 or n_alived_neighbors > 3:
                    updated_grid[i, j] = 0
            else:
                if n_alived_neighbors == 3:
                    updated_grid[i, j] = 1

    return updated_grid

class GameOfLife(nn.Module):
    def __init__(self, ):
        super(GameOfLife, self).__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=0, bias=False)
        kernel = neunet.tensor([[[[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]]]])
        self.conv.weight.data = kernel

    def forward(self, grid: np.ndarray):
        '''
        Implementation of Conway's Game of Life using a convolution (works much faster)
        '''
        # Pad the grid to create a "toroidal" mesh effect
        grid_tensor = neunet.tensor(np.pad(grid, pad_width=1, mode='wrap'))[None, None, :, :]
        n_alive_neighbors = self.conv(grid_tensor).data
        updated_grid = ((n_alive_neighbors.astype(int) == 3) | ((grid.astype(int) == 1) & (n_alive_neighbors.astype(int) == 2)))
        updated_grid = updated_grid[0, 0, :, :]

        return updated_grid

game = GameOfLife()


class Dataset:
    def get_data(self):
        '''
        Generate data from all probable situations (2^9), 
        where (1 point - current point, 8 points - surrounding neighbors points)
        '''
        X = list(itertools.product([0, 1], repeat = 9))

        X = [np.array(x).reshape(3, 3) for x in X]
        Y = [game(x).astype(int) for x in X]

        return np.array(X), np.array(Y)
    
# architecture was borrowed from https://gist.github.com/failure-to-thrive/61048f3407836cc91ab1430eb8e342d9
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0) # 2
        self.conv2 = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x = neunet.tanh(self.conv1(x))
        x = self.conv2(x)
        return x

    def predict(self, x):
        # Pad the grid to create a "toroidal" mesh effect
        x = neunet.tensor(np.pad(x, pad_width = 1, mode='wrap'))[None, None, :, :]
        # Squeeze
        return self.forward(x).data[0, 0, :, :]

model = Net()

dataset = Dataset()
X, Y = dataset.get_data()

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 500

for epoch in range(epochs):
    tqdm_range = tqdm(zip(X, Y), total=len(X))
    perm = np.random.permutation(len(X))

    X = X[perm]
    Y = Y[perm]
    losses = []
    for x, y in tqdm_range:
        optimizer.zero_grad()

        x = neunet.tensor(np.pad(x, pad_width=1, mode='wrap'))[None, None, :, :]
        y = neunet.tensor(y)[None, None, :, :]
        y_pred = model(x)
  
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        tqdm_range.set_description(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.data:.7f}, Mean Loss: {np.mean(losses):.7f}")
        
model.eval()

def animate(i):
    global grid
    ax.clear()
    # grid = update(grid) # Native implementation
    # grid = game(grid) # Implementation using convolution
    grid = model.predict(grid) # Neural network
    ax.imshow(grid, cmap=ListedColormap(['black', 'lime']))#, interpolation='lanczos'

fig, ax = plt.subplots(figsize = (10, 10))
ani = animation.FuncAnimation(fig, animate, frames=30, interval=5)
plt.show()
```

Code:   
*[Model example](examples/conway.py)*   


##### Conway`s Game of Life Simulation Results:
Native implementation Example | Neural network Example
:-------------------------:|:-------------------------:
<img src="generated images/native_conway.gif"> |  <img src="generated images/neunet_conway.gif">

</details>



### TODO:
- [ ] Fix bugs (if found), clean up and refactor the code
- [ ] Add more Tensor operations
- [ ] Add more layers, activations, optimizers, loss functions, etc
- [ ] Add more examples
- [x] Add CuPy support
