
import numpy as np
from tqdm import tqdm
from numba import njit



from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, ZeroPadding2D, RepeatVector
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam






import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
kernels_num = 2
kernel_size = 2
inputs_num = 3
input_size = 5
stride_f = 2



# conv = nn.Conv2d(inputs_num, kernels_num, kernel_size, stride = stride_f, bias=False)

X = np.arange(0.0, (batch_size * inputs_num * input_size * input_size)).reshape((batch_size, inputs_num, input_size, input_size))
W = np.arange(0.0, (inputs_num * kernels_num * kernel_size * kernel_size)).reshape((kernels_num, inputs_num, kernel_size, kernel_size))#np.ones((kernels_num, inputs_num, kernel_size, kernel_size)) #np.random.normal(0, 1, (kernels_num, inputs_num, kernel_size, kernel_size))
# BIAS = np.random.normal(0, 1, (kernels_num))

# ZP = ZeroPadding2D(padding = 2, input_shape = (inputs_num, input_size, input_size))
# ZP.build()
rp = RepeatVector(3)
XRP = np.arange(0.0, (batch_size * inputs_num * input_size * input_size)).reshape((batch_size, inputs_num, input_size, input_size))
print(XRP.shape)
print(XRP)
YRP = rp.forward_prop(XRP, training = True)
print(YRP.shape)
print(YRP)
# print(rp.backward_prop(YRP.shape))
OUT = rp.backward_prop(YRP)
print(OUT.shape)
print(OUT)
# Y = ZP.forward_prop(X, training = True)
# print(f"{Y=}\n")

# Y = ZP.backward_prop(X)
# print(f"{Y=}\n")

# conv = Conv2D(kernels_num, kernel_shape = (kernel_size, kernel_size), input_shape = (inputs_num, input_size, input_size), padding = 0, stride = stride_f, dilation = (1, 1))

# conv.build(None)
# conv.w = W

# conv.channels_num, conv.input_height, conv.input_width = inputs_num, input_size, input_size


# # conv.conv_height = (conv.input_height - 1) * conv.stride[0] - 2 * conv.padding[0]  +  conv.dilation[0] * (conv.kernel_height - 1) + conv.output_padding[0] + 1
# # conv.conv_width =  (conv.input_width - 1) * conv.stride[1] - 2 * conv.padding[1] + conv.dilation[1] * (conv.kernel_width - 1) + conv.output_padding[1] + 1
# conv.conv_height = (conv.input_height + conv.padding[0] + conv.padding[1] - conv.dilation[0] * (conv.kernel_height - 1) - 1) // conv.stride[0]   + 1
# conv.conv_width =  (conv.input_width + conv.padding[2] + conv.padding[3] - conv.dilation[1] * (conv.kernel_width - 1) - 1) // conv.stride[1] + 1


# conv.dilated_kernel_height = conv.dilation[0] * (conv.kernel_height - 1) + 1
# conv.dilated_kernel_width = conv.dilation[1] * (conv.kernel_width - 1) + 1

# conv.input_height = (conv.conv_height - 1) * conv.stride[0] - conv.padding[0] + conv.padding[1] +  conv.dilated_kernel_height
# conv.input_width = (conv.conv_width - 1) * conv.stride[1] - conv.padding[2] + conv.padding[3] +  conv.dilated_kernel_width

# # conv.prepared_input_height = (conv.input_height - 1) * conv.stride[0] + 1 - 2 * conv.padding[0] + conv.output_padding[0] + 2 * conv.dilated_kernel_height - 2
# # conv.prepared_input_width = (conv.input_width - 1) * conv.stride[1] + 1 - 2 * conv.padding[1] + conv.output_padding[1] + 2 * conv.dilated_kernel_width - 2
# conv.prepared_input_height = (conv.input_height + conv.padding[0] + conv.padding[1])
# conv.prepared_input_width = (conv.input_width + conv.padding[2] + conv.padding[3])
# # print(conv.prepared_input_height)



# LOSS = np.arange(0.0, batch_size * kernels_num * conv.conv_height * conv.conv_width).reshape((batch_size, kernels_num, conv.conv_height, conv.conv_width))
# print(f"x shape is {X.shape}")

# print(X)
# print(W)
# print(X)

# print("FORWARD")
# x = conv.forward_prop(X, training = True)
# print(x.shape)
# print(x)


# x = conv.backward_prop(LOSS)

# print("BACKWARD")
# print(x.shape)
# print(x)


# # x = conv.forward_prop(X, training = True)
# # print(x)
# m = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, stride= 2, padding = (1, 2),  output_padding = 0, dilation = (1, 1), bias = False)

# m.weight = nn.Parameter(torch.from_numpy(W).float())
# #m.bias = nn.Parameter(torch.from_numpy(BIAS).float())

# input = torch.from_numpy(X).float()
# output = m(input)
# print(output.shape)
# print(output)

# GRAD = output.backward(torch.from_numpy(LOSS).float(), retain_graph=True)
# print(GRAD)



# import tensorflow as tf
# from keras.layers import MaxPooling2D as KerasMaxPooling2D#Conv2DTranspose as C2DT

# input_shape = (1, kernel_size, kernel_size, kernels_num)
# X_keras = tf.convert_to_tensor(X.reshape(1, input_size, input_size, inputs_num))
# maxpooling = KerasMaxPooling2D(pool_size = (2, 2), strides = (1, 1))
# y = tf.keras.layers.Conv2DTranspose(1, 3, activation='relu', input_shape = input_shape[1:])(x)
# conv_keras = C2DT(1, 2, activation='relu', input_shape = input_shape[1:], data_format="channels_last", use_bias = False)
# print(W)

# conv.set_weights([W.reshape(kernel_size, kernel_size, kernels_num, 1)])   
# y = maxpooling(X_keras)
# print(y)

# print("X=\n", X)
# mymaxpooling = MaxPooling2D(pool_size = (2, 2), stride = (2, 2), input_shape = (inputs_num,  input_size, input_size))
# mymaxpooling.build()

# y = mymaxpooling.forward_prop(X, training = True)
# print("FORWARD=\n", y)

# print("POOL IND\n", mymaxpooling.pooling_layer_ind)

# LOSS = y.copy()
# y = mymaxpooling.backward_prop(LOSS)
# print("BACKWARD=\n", y)

# y = conv_keras(X_keras)
# print(y)
# # print(y)
# print(conv_keras.get_weights()[0].shape)
# W = conv_keras.get_weights()[0].reshape(1, kernels_num, kernel_size, kernel_size)
# conv.w = W
# x = conv.forward_prop(X, training = True)
# print(x)

# m.weight = nn.Parameter(torch.from_numpy(W).float())
# # m.bias = nn.Parameter(torch.from_numpy(BIAS).float())

# input = torch.from_numpy(X).float()
# output = m(input)
# print(output.shape)
# print(output)

# class MyModel(tf.keras.Model):

#   def __init__(conv):
#     super().__init__()
#     conv.convT = Conv2DTranspose(kernels_num, kernel_size, padding = (0, 0), use_bias = False)
#     conv.convT.kernel = W
#     # conv.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

#   def call(conv, inputs):
#     x = conv.convT(tf.convert_to_tensor(inputs.reshape(1, input_size, input_size, 1)))
#     return conv.convT(x)

# model = MyModel()
# x = model.call(X)
# print(x)

# def set_padding(layer, padding):
#     # padded_layer = np.pad(layer, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), constant_values = 0)
#     padded_layer = np.zeros(
#         (   
#             layer.shape[0],
#             layer.shape[1],
#             layer.shape[2] + 2 * padding[0],
#             layer.shape[3] + 2 * padding[1],
#         )
#     )


#     padded_layer[
#                 :,
#                 :,
#                 padding[0] : layer.shape[2] + padding[0],
#                 padding[1] : layer.shape[3] + padding[1],
#             ] = layer

#     return padded_layer


# def remove_padding(layer, padding):
#     # losses[k] = losses[k][...,conv.topology[k+1]['padding']:-conv.topology[k+1]['padding'],conv.topology[k+1]['padding']:-conv.topology[k+1]['padding']]
#     unpadded_layer = np.zeros(
#         (
#             layer.shape[0],
#             layer.shape[1],
#             layer.shape[2] - 2 * padding[0],
#             layer.shape[3] - 2 * padding[1],
#         )
#     )

#     unpadded_layer = layer[
#                 :,
#                 :,
#                 padding[0] : layer.shape[2] - padding[0],
#                 padding[1] : layer.shape[3] - padding[1],
#             ]

#     return unpadded_layer



# arr = np.random.normal(0, 1, (1000, 10, 10, 10))




# start_time = time.time()
# arr2 = set_padding(arr, (5, 5))
# print(arr2.shape)
# print(time.time() - start_time)


# start_time = time.time()
# arr2 = np.pad(arr, ((0, 0), (0, 0), (5, 5), (5, 5)), constant_values = 0)
# print(arr2.shape)
# print(time.time() - start_time)



