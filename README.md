# numpy-nn-model (in the pipeline)
Сustom numpy neural network model implementation in which you can add fully connected  and convolutional layers by one line (2 version)

## Some information and features:

### Implemented activation functions:
1) Sigmoid
2) TanH
3) Softmax
4) Softplus
5) ReLU
6) LeakyReLU
7) ELU
8) SELU
9) GELU
10) Identity (default; returns the same argument)

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
13) Bidirectional (Not implemented Yet)
14) Conv2D
15) Conv2DTranspose
16) MaxPooling2D
17) AveragePooling2D
18) UpSamling2D


### Model Initialization (Example):

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

