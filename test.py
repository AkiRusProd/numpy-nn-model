import numpy as np
from tqdm import tqdm



training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:])/255)
        targets.append(int(line[0]))

    return inputs, targets



training_inputs, training_targets = prepare_data(training_data)
test_inputs, test_targets = prepare_data(test_data)





from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, RepeatVector, TimeDistributed, RNN, LSTM, GRU
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam
model = Model()

# from keras import Sequential
# from keras.layers import Dense, TimeDistributed, RepeatVector, Flatten, Reshape, Conv2D, BatchNormalization
# from keras.layers import SimpleRNN as RNN
# model = Sequential()

"""SOME KINDA TEST LAYERS"""
# model.add(Dense(units_num = 256, input_shape = (1, 784), activation = LeakyReLU()))
# model.add(BatchNormalization())
# model.add(Dropout())
# model.add(Flatten())
# model.add(Dense(units_num = 128, activation = "sigmoid"))
# model.add(BatchNormalization())
# model.add(Dropout())
# model.add(Dense(units_num = 10, activation = "sigmoid"))
# model.add(BatchNormalization(input_shape = (1, 784)))

"""CONV2D CLASSIFIER TEST EXAMPLE"""
# model.add(Reshape(shape = (1, 28, 28)))
# model.add(Conv2D(kernels_num = 8, kernel_shape = (5, 5), activation = "relu"))

# model.add(MaxPooling2D())
# model.add(Conv2D(kernels_num = 32, kernel_shape = (3, 3), padding = "same", activation = LeakyReLU()))
# # model.add(Conv2DTranspose(kernels_num = 16, kernel_shape = (3, 3), activation = "relu"))
# model.add(MaxPooling2D())

# # model.add(UpSampling2D())
# model.add(Flatten())
# model.add(BatchNormalization())
# # model.add(Dense(units_num = 50,  activation = "relu"))
# # model.add(Dropout())
# model.add(Dense(units_num = 10, activation = None))
# model.add(Activation(activation = "softmax"))

# model.compile(optimizer = "adam", loss = "mse")
# model.fit(training_inputs,  training_targets, epochs = 3, batch_size = 100)
# model.predict(test_inputs, test_targets)

# timesteps = 3
# inputs_num = 16
# training_inputs = np.arange(0, timesteps * inputs_num).reshape(inputs_num, timesteps, 1)
# test_outputs = training_inputs + 15
# test_outputs = np.sum(training_inputs, axis=1)
# # print(test_outputs)

"""SHIT RNN TEST  PERFORMANCE CHECK EXAMPLE"""
# model.add(RNN(100, activation='relu', input_shape=(timesteps, 1), return_sequences=True, use_bias = False))
# # model.add(RepeatVector(3))
# # model.add(BatchNormalization())
# # model.add(RNN(70, activation='relu', return_sequences=True, use_bias = False))
# model.add(RNN(50, activation='relu', return_sequences=True, use_bias = False))
# # model.add(Reshape(target_shape = (2, 5, 5)))
# # model.add(RepeatVector(3))

# # model.add(TimeDistributed(Conv2D(kernels_num = 5, kernel_shape = (2, 2), activation = "relu")))

# # model.add(TimeDistributed(Flatten()))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(Dense(1, use_bias=False)))
# # model.add(TimeDistributed(Flatten()))

# # model.add(RNN(50, activation='relu', return_sequences=False, use_bias = False, cycled_states = False))
# # model.add(RNN(50, activation='relu', input_shape=(3, 50), use_bias = False))
# # model.add(Dense(1))

# model.compile(optimizer="adam", loss='mse')
# model.fit(training_inputs,  test_outputs, epochs = 10000, batch_size = 4)
# # model.predict(training_inputs,  test_outputs)


"""RNN CLASSIFIER TEST EXAMPLE"""

model.add(Reshape(shape = (28, 28)))
model.add(GRU(256, input_shape=(28, 28), return_sequences=False, cycled_states = True))
model.add(RepeatVector(28))
model.add(TimeDistributed(Dense(50, use_bias=False)))
model.add(TimeDistributed(BatchNormalization()))
# model.add(BatchNormalization())
model.add(GRU(128, input_shape=(28, 28), cycled_states = True))
# model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = "adam", loss = "mse")
model.fit(training_inputs,  training_targets, epochs = 5, batch_size = 200)
model.predict(test_inputs, test_targets)


"""SOME SHIT KINDA TEST LAYERS AND THEIR FUNCTIONS (I DON'T UNDERSTAND THIS ANYMORE)"""
# X = np.random.normal(0, 1, (5, 1, 28, 28))

# layer = Conv2D(kernels_num = 5, kernel_shape = (7, 7), input_shape = (1, 28, 28), activation = "sigmoid")
# layer.conv_height = 28 - 7 + 1
# layer.conv_width= 28 - 7 + 1
# layer.channels_num = 1
# layer.w = np.random.normal(0, 1, (5, 1, 7, 7))
# print(layer.forward_prop(X, training = True).shape)
# print(layer.backward_prop(X).shape)


# from NNModel.activations import activations
# from NNModel.activations import Sigmoid


# activation = Sigmoid()
# activation2 = activations["sigmoid"]


# print(activation2.function(3))

# from reshape import Reshape
# X = np.random.normal(0, 1, (5, 1, 48))

# layer = Reshape(shape = (3, 1, 16))
# print(layer.forward_prop(X, training = True).shape)
# print(layer.backward_prop(X).shape)

# print(X.shape)

# layer = Flatten()
# print(layer.forward_prop(X, training = True).shape)
# print(layer.backward_prop(X).shape)

# # print(np.array([1, 2 ,3 ,4, 5, 6, 7, 8, 9]).reshape(1, *(3, 3)))

# from numba import njit
# @njit
# def create_arr(shape):
#     arr = np.zeros((shape[0], shape[1]))


# create_arr(shape = (1, 2, 3 ,4))


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# kernels_num = 1
# kernel_size = 3
# inputs_num = 1
# input_size = 4
# stride_f = 2

# # conv_size = input_size - kernel_size + 1

# conv = nn.Conv2d(inputs_num, kernels_num, kernel_size, stride = stride_f, dilation = (1, 1), bias=False)

# X = np.arange(0, np.float(1 * inputs_num * input_size * input_size)).reshape((1, inputs_num, input_size, input_size))
# W = np.random.normal(0, 1, (kernels_num, inputs_num, kernel_size, kernel_size))
# BIAS = np.random.normal(0, 1, (kernels_num))



# conv.weight = nn.Parameter(torch.from_numpy(W).float())

# x = conv(torch.from_numpy(X).float())
# print("Pytorch conv")
# print(x)


# conv = Conv2D(kernels_num, kernel_shape = (kernel_size, kernel_size), input_shape = (inputs_num, input_size, input_size),stride = (stride_f, stride_f), dilation = (1, 1))
# conv.w = W

# conv.channels_num, conv.input_height, conv.input_width = inputs_num, input_size, input_size
# # conv.conv_height, conv.conv_width = conv_size, conv_size
# conv.conv_height = (conv.input_height + 2 * conv.padding[0]  - conv.dilation[0] * (conv.kernel_height - 1) - 1) // conv.stride[0]   + 1
# conv.conv_width =  (conv.input_width + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_width - 1) - 1) // conv.stride[1] + 1

# conv.dilated_kernel_height = conv.dilation[0] * (conv.kernel_height - 1) + 1
# conv.dilated_kernel_width = conv.dilation[1] * (conv.kernel_width - 1) + 1

# x = conv.forward_prop(X, training = True)

# print(conv.conv_height, conv.conv_width, conv.dilated_kernel_height, conv.dilated_kernel_width)
# print("My conv")
# print(x)
# LOSS = np.random.normal(0, 1, (1, kernels_num, conv.conv_height, conv.conv_width ))

# x = conv.backward_prop(LOSS)

# print("My conv backward")
# print(x)

# conv = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, dilation = 1, bias=False)
# x = conv(torch.from_numpy(X).float())
# print(x.shape)

# With square kernels and equal stride
# m = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, stride=stride_f,  output_padding = 1, bias = False)
# m.weight = nn.Parameter(torch.from_numpy(W).float())
# m.bias = nn.Parameter(torch.from_numpy(BIAS).float())

# input = torch.from_numpy(X).float()
# output = m(input)
# print(output.shape)
# print(output)


# def make_transposing(layer, stride):
#         transposed_layer = np.zeros(
#             (   
#                 layer.shape[0],
#                 layer.shape[1],
#                 stride * layer.shape[2] - (stride - 1),
#                 stride * layer.shape[3] - (stride - 1),
#             ),
#             dtype=layer.dtype,
#         )
#         transposed_layer[:, :, ::stride, ::stride] = layer

#         return transposed_layer

# X_tr = make_transposing(X, 2)

# tmp = np.zeros((1, inputs_num, input_size  * stride_f, input_size  * stride_f))
# tmp[:, :, : X_tr.shape[2], : X_tr.shape[3]] = X_tr
# X = tmp
# print(X)

# n = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, stride= 1, bias = False)
# n.weight = nn.Parameter(torch.from_numpy(W).float())
# n.bias = nn.Parameter(torch.from_numpy(BIAS).float())
# input = torch.from_numpy(X).float()
# output = n(input)
# print(output.shape)
# print(output)



# import tensorflow as tf
# from tensorflow.nn import
# from keras.layers import Conv2DTranspose, Conv2D
# import keras.backend as K 
# # from keras.backend import conv2d_ranspose

# conv = Conv2DTranspose(kernels_num, kernel_size, kernel_size)(X)
# shape = K.int_shape(conv)
# print(shape)
# # x = Conv2D()(x) 
# tf.nn.conv2d_transpose(
#     X, W, strides = 1, padding='SAME',
#     data_format='NCHW', dilations=None, name=None
# )

# print(convt.shape)

# input = tf.Variable(X.reshape(1, input_size, input_size, inputs_num))#tf.random.normal([1,3,3,5])
# filter = tf.Variable(W.reshape(kernel_size, kernel_size, inputs_num, kernels_num))#tf.random.normal([3,3,5,1])

# stride = 1
# cv2d_size = H = (input_size-1) * stride + kernel_size
# op = tf.nn.conv2d_transpose(input, filter, output_shape = cv2d_size, strides=[1, 1, 1, 1], padding='VALID')
# print(op)

# from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
# from keras.models import Sequential, Model, load_model

# model = Sequential()

# model.add(Conv2D(1, kernel_size = 3, activation="relu", strides = 2, input_shape=(4, 4, 1)))
# model.summary()

