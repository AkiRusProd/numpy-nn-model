import numpy as np
from tqdm import tqdm
from numba import njit



from NNModel.Layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D
from NNModel import Model
from NNModel.activations import LeakyReLU
from NNModel.optimizers import SGD


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# kernels_num = 1
# kernel_size = 2
# inputs_num = 1
# input_size = 3
# stride_f = 2



# conv = nn.Conv2d(inputs_num, kernels_num, kernel_size, bias=False)

# X = np.arange(0, np.float(1 * inputs_num * input_size * input_size)).reshape((1, inputs_num, input_size, input_size))
# W = np.arange(0, np.float(1 * kernels_num * kernel_size * kernel_size)).reshape((1, kernels_num, kernel_size, kernel_size))#np.ones((kernels_num, inputs_num, kernel_size, kernel_size)) #np.random.normal(0, 1, (kernels_num, inputs_num, kernel_size, kernel_size))
# BIAS = np.random.normal(0, 1, (kernels_num))


# conv = Conv2DTranspose(kernels_num, kernel_shape = (kernel_size, kernel_size), input_shape = (inputs_num, input_size, input_size), padding = (0, 0), stride = (1, 1), output_padding = (0, 0), dilation = (1, 1))
# conv.w = W

# conv.channels_num, conv.input_height, conv.input_width = inputs_num, input_size, input_size


# conv.conv_height = (conv.input_height - 1) * conv.stride[0] - 2 * conv.padding[0]  +  conv.dilation[0] * (conv.kernel_height - 1) + conv.output_padding[0] + 1
# conv.conv_width =  (conv.input_width - 1) * conv.stride[1] - 2 * conv.padding[1] + conv.dilation[1] * (conv.kernel_width - 1) + conv.output_padding[1] + 1


# conv.dilated_kernel_height = conv.dilation[0] * (conv.kernel_height - 1) + 1
# conv.dilated_kernel_width = conv.dilation[1] * (conv.kernel_width - 1) + 1

# conv.prepared_input_height = (conv.input_height - 1) * conv.stride[0] + 1 - 2 * conv.padding[0] + conv.output_padding[0] + 2 * conv.dilated_kernel_height - 2
# conv.prepared_input_width = (conv.input_width - 1) * conv.stride[1] + 1 - 2 * conv.padding[1] + conv.output_padding[1] + 2 * conv.dilated_kernel_width - 2
# print(conv.prepared_input_height)



# LOSS = np.arange(0, np.float(1 * kernels_num * conv.conv_height * conv.conv_width)).reshape((1, kernels_num, conv.conv_height, conv.conv_width))
# print(X)
# x = conv.forward_prop(X, training = True)


# print("FORWARD")
# print(x.shape)
# print(x)

# x = conv.backward_prop(LOSS)

# print("BACKWARD")
# print(x.shape)
# print(x)

# m = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, stride= 1, padding = (0, 0),  output_padding = 0, dilation = (1, 1), bias = False)

# m.weight = nn.Parameter(torch.from_numpy(W).float())
# # m.bias = nn.Parameter(torch.from_numpy(BIAS).float())

# input = torch.from_numpy(X).float()
# output = m(input)
# print(output.shape)
# print(output)

# GRAD = output.backward(torch.from_numpy(LOSS).float(), retain_graph=True)
# print(GRAD)

import time



def set_padding(layer, padding):
    # padded_layer = np.pad(layer, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), constant_values = 0)
    padded_layer = np.zeros(
        (   
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] + 2 * padding[0],
            layer.shape[3] + 2 * padding[1],
        )
    )


    padded_layer[
                :,
                :,
                padding[0] : layer.shape[2] + padding[0],
                padding[1] : layer.shape[3] + padding[1],
            ] = layer

    return padded_layer


def remove_padding(layer, padding):
    # losses[k] = losses[k][...,self.topology[k+1]['padding']:-self.topology[k+1]['padding'],self.topology[k+1]['padding']:-self.topology[k+1]['padding']]
    unpadded_layer = np.zeros(
        (
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] - 2 * padding[0],
            layer.shape[3] - 2 * padding[1],
        )
    )

    unpadded_layer = layer[
                :,
                :,
                padding[0] : layer.shape[2] - padding[0],
                padding[1] : layer.shape[3] - padding[1],
            ]

    return unpadded_layer



arr = np.random.normal(0, 1, (1000, 10, 10, 10))




start_time = time.time()
arr2 = set_padding(arr, (5, 5))
print(arr2.shape)
print(time.time() - start_time)


start_time = time.time()
arr2 = np.pad(arr, ((0, 0), (0, 0), (5, 5), (5, 5)), constant_values = 0)
print(arr2.shape)
print(time.time() - start_time)



