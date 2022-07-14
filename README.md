# numpy-nn-model (in the pipeline)
Сustom numpy neural network model implementation in which you can add fully connected  and convolutional layers by one line (2 version)

## Some information and features:

### Implemented activation functions:
1) Sigmoid
2) TanH
3) Softmax
4) Softplus
5) Softsign
6) ReLU
7) LeakyReLU
8) ELU
9) SELU
10) GELU
11) Identity (default; returns the same argument)

### Implemented Optimizers:
1) SGD
2) Momentum
3) RMSProp
4) Adam
5) Nadam

### Implemented Layers (still needs to be tested and improved in some places):
1) Dense
2) Activation
3) Flatten
4) ZeroPadding2D
5) Reshape
6) RepeatVector
7) Dropout
8) BatchNormalization
9) RNN
10) LSTM
11) GRU
12) TimeDistributed
13) Bidirectional
14) Conv2D
15) Conv2DTranspose
16) MaxPooling2D
17) AveragePooling2D
18) UpSamling2D


### Model Examples:

#### Convolutional Classifier

```python
from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, Activation
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD
from nnmodel import Model

model = Model()

model.add(Reshape(shape = (1, 28, 28)))
model.add(Conv2D(kernels_num = 8, kernel_shape = (5, 5), activation = "relu"))
model.add(MaxPooling2D())
model.add(Conv2D(kernels_num = 32, kernel_shape = (3, 3), padding = "same", activation = LeakyReLU()))
model.add(MaxPooling2D(pool_size = (4 ,4)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout())
model.add(Dense(units_num = 10, activation = "sigmoid"))
model.add(Activation(activation = "softmax"))

model.compile(optimizer = "adam", loss = "mse")
model.fit(training_inputs,  training_targets, epochs = 3, batch_size = 100)
model.predict(test_inputs, test_targets)
```

#### Bidirectional GRU Classifier
```python
model = Model()
model.add(Reshape(shape = (28, 28)))
model.add(Bidirectional(GRU(256, input_shape=(28, 28), return_sequences=False, cycled_states = True)))
model.add(RepeatVector(28))
model.add(TimeDistributed(Dense(50, activation = LeakyReLU(0.2), use_bias=False)))
model.add(TimeDistributed(BatchNormalization()))
model.add(Bidirectional(GRU(128, input_shape=(28, 28), cycled_states = True)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = "adam", loss = "mse")
model.fit(training_inputs,  training_targets, epochs = 5, batch_size = 200)
model.predict(test_inputs, test_targets)
```

#### Simple Denoising AutoEncoder
```python
model = Model()
model.add(Dense(256, input_shape = (784), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
```

#### Denoising Variational Autoencoder (VAE)
```python
from nnmodel.modules import VAE

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

vae = VAE(encoder, decoder)
vae.compile(optimizer = Adam(), loss = 'binary_crossentropy')
loss, decoder_loss, kl_loss = vae.fit(noisy_inputs[0:10000], inputs[0:10000], epochs = 100, batch_size = 100)
```
##### VAE Results:
Noisy Data Example | Noise Removed Data Example
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/autoencoder%20images/vae%20noised%20set%20of%20images.jpeg)  |  ![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/autoencoder%20images/vae%20denoised%20set%20of%20images.jpeg)


#### Generative Adversarial Network (GAN)
```python
from nnmodel.modules import GAN

generator = Model()
generator.add(Dense(128, input_shape = (noise_vector_size), use_bias=False))
generator.add(Activation('leaky_relu'))
generator.add(Dense(512, use_bias=False))
generator.add(Dropout(0.2))
generator.add(Activation('leaky_relu'))
generator.add(Dense(784, use_bias=False))
generator.add(Activation('tanh'))

discriminator = Model()
discriminator.add(Dense(128, input_shape = (784), use_bias=False))
discriminator.add(Activation('leaky_relu'))
discriminator.add(Dense(64, use_bias=False))
discriminator.add(Activation('leaky_relu'))
discriminator.add(Dense(2, use_bias=False))
discriminator.add(Activation('sigmoid'))

gan = GAN(generator, discriminator)
gan.compile(optimizer = Nadam(alpha = 0.001, beta = 0.5), loss = 'minimax_crossentropy', each_epoch_predict={"mode": True, "num" : x_num * y_num})
G_loss, D_loss = gan.fit(data, epochs = 30, batch_size = 64, noise_vector_size = noise_vector_size)
```
##### GAN Results:
Training process Example | Interpolation between images Example
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/training%20process.gif)  |  ![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/images%20latent%20dim.gif)


