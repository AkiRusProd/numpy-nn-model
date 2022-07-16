# numpy-nn-model (in the pipeline)
Сustom CPU numpy neural network model implementation in which you can add fully connected  and convolutional layers by one line

## Some information and features:

### Implemented Activation Functions:
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

*[See Activation Functions...](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/activations.py)*

### Implemented Optimizers:
1) SGD
2) Momentum
3) RMSProp
4) Adam
5) Nadam

*[See Optimizers...](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/optimizers.py)*

### Implemented Loss Functions:
1) MSE
2) BinaryCrossEntropy
3) CategoricalCrossEntropy
4) MiniMaxCrossEntropy (used only for GANs)

*[See Loss Functions...](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/loss_functions.py)*

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

*[See Layers...](https://github.com/AkiRusProd/numpy-nn-model/tree/master/nnmodel/layers)*


### Some Model Examples:
All examples was trained on [MNIST](https://pjreddie.com/projects/mnist-in-csv/) Dataset   

Code:   
*[Base training module](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/nn_model.py)*

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

model.save("saved models/convolutional_digits_classifier")
```
###### (prediction on test MNIST data with this model is 94.19 %)

Code:   
*[Model Example](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/convolutional_digits_classifier.py)*

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

model.save("saved models/bidirectional_recurrent_digits_classifier")
```
###### (prediction on test MNIST data with this model is 98.38 %)

Code:   
*[Model Example](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/bidirectional_recurrent_digits_classifier.py)*

#### Simple Denoising AutoEncoder
```python
model = Model()
model.add(Dense(256, input_shape = (784), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(784, activation='sigmoid'))

model.compile(optimizer = Adam(), loss = 'binary_crossentropy')
loss = model.fit(noisy_inputs, inputs, epochs = 100, batch_size = 100)

model.save("saved models/AE")
```

```python
"""Convolutional AutoEncoder model topology example (works much slower):"""
model.add(Reshape((1, 28, 28)))
model.add(Conv2D(kernels_num = 16, kernel_shape=(3,3), stride=(2, 2), padding='same', input_shape=(1, 28, 28)))
model.add(Activation(LeakyReLU(alpha=0.2)))
model.add(Conv2D(16, (3,3), stride=(2, 2), padding='same'))
model.add(Activation(LeakyReLU(alpha=0.2)))
model.add(Flatten())
model.add(Dense(32 * 7 * 7, activation='relu'))
model.add(Reshape((32, 7, 7)))
model.add(Conv2DTranspose(16, (4,4), stride=(2,2), padding='same'))
model.add(Activation(LeakyReLU(alpha=0.2)))
model.add(Conv2DTranspose(16, (4,4), stride=(2,2), padding='same'))
model.add(Activation(LeakyReLU(alpha=0.2)))
model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
model.add(Flatten())
```
Code:   
*[Model Example](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/simple_autoencoder.py)*

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

vae.save("saved models/VAE")
```
Code:   
*[Model example](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/variational_autoencoder.py)*   
*[VAE training module](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/modules/vae.py)*   

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

gan.save(f'saved models/GAN')
```
```python
'''Convolutional GAN model topology example (works much slower):'''
generator.add(Dense(128, input_shape = (noise_vector_size), use_bias=False))
generator.add(Activation('leaky_relu'))
generator.add(Dense(8 * 7 * 7, use_bias=False))
generator.add(Reshape((8, 7, 7)))
generator.add(Conv2DTranspose(8, (4,4), stride=(2,2), padding='same'))
generator.add(Activation(LeakyReLU(alpha=0.2)))
generator.add(Conv2DTranspose(8, (4,4), stride=(2,2), padding='same'))
generator.add(Activation(LeakyReLU(alpha=0.2)))
generator.add(Conv2D(1, (7,7), activation='tanh', padding='same'))

discriminator = Model()
discriminator.add(Reshape((1, 28, 28)))
discriminator.add(Conv2D(kernels_num = 64, kernel_shape=(3,3), stride=(2, 2), input_shape=(1, 28, 28)))
discriminator.add(Activation(LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(0.4))
discriminator.add(Conv2D(16, (3,3), stride=(2, 2)))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
```


Code:   
*[Model example](https://github.com/AkiRusProd/numpy-nn-model/blob/master/examples/generative_adversarial_network.py)*   
*[GAN training module](https://github.com/AkiRusProd/numpy-nn-model/blob/master/nnmodel/modules/gan.py)*   

##### GAN Results:
Training process Example | Interpolation between images Example
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/training%20process.gif)  |  ![](https://raw.githubusercontent.com/AkiRusProd/numpy-nn-model/master/examples/generated%20images/images%20latent%20dim.gif)


