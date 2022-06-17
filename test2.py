import numpy as np
from tqdm import tqdm


# training_data = open('dataset/mnist_train.csv','r').readlines()
# test_data = open('dataset/mnist_test.csv','r').readlines()


# def prepare_data(data):
#     inputs, targets = [], []

#     for raw_line in tqdm(data, desc = 'preparing data'):

#         line = raw_line.split(',')
    
#         inputs.append(np.asfarray(line[1:])/255)
#         targets.append(int(line[0]))

#     return inputs, targets



# training_inputs, training_targets = prepare_data(training_data)
# test_inputs, test_targets = prepare_data(test_data)




from NNModel.Layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D
from NNModel import Model
from NNModel.activations import LeakyReLU
from NNModel.optimizers import SGD

model = Model()

# model.add(Dense(units_num = 256, input_shape = (1, 784), activation = LeakyReLU()))
# model.add(BatchNormalization())
# model.add(Dropout())
# model.add(Flatten())
# model.add(Dense(units_num = 128, activation = "sigmoid"))
# model.add(BatchNormalization())
# model.add(Dropout())
# model.add(Dense(units_num = 10, activation = "sigmoid"))

# model.add(Reshape(shape = (1, 28, 28)))
# model.add(Conv2D(kernels_num = 8, kernel_shape = (5, 5), activation = "relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(kernels_num = 32, kernel_shape = (3, 3), activation = "relu"))
# model.add(MaxPooling2D())
# # model.add(UpSampling2D())
# model.add(Flatten())
# model.add(BatchNormalization())
# # model.add(Dense(units_num = 50,  activation = "relu"))
# # model.add(Dropout())
# model.add(Dense(units_num = 10, activation = "softmax"))

# model.compile(optimizer = "adam", loss_function = "mse")
# model.fit(training_inputs,  training_targets, epochs = 3, batch_size = 100)
# model.predict(test_inputs, test_targets)



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


import torch
import torch.nn as nn
import torch.nn.functional as F

kernels_num = 1
kernel_size = 2
inputs_num = 1
input_size = 3
stride_f = 2



conv = nn.Conv2d(inputs_num, kernels_num, kernel_size, bias=False)

X = np.arange(0, np.float(1 * inputs_num * input_size * input_size)).reshape((1, inputs_num, input_size, input_size))
W = np.arange(0, np.float(1 * kernels_num * kernel_size * kernel_size)).reshape((1, kernels_num, kernel_size, kernel_size))#np.ones((kernels_num, inputs_num, kernel_size, kernel_size)) #np.random.normal(0, 1, (kernels_num, inputs_num, kernel_size, kernel_size))
BIAS = np.random.normal(0, 1, (kernels_num))



# conv.weight = nn.Parameter(torch.from_numpy(W).float())

# x = conv(torch.from_numpy(X).float())
# print("Pytorch conv")
# print(x)


conv = Conv2DTranspose(kernels_num, kernel_shape = (kernel_size, kernel_size), input_shape = (inputs_num, input_size, input_size), padding = (0, 0), stride = (1, 1), output_padding = (0, 0), dilation = (1, 1))
conv.w = W

conv.channels_num, conv.input_height, conv.input_width = inputs_num, input_size, input_size


conv.conv_height = (conv.input_height - 1) * conv.stride[0] - 2 * conv.padding[0]  +  conv.dilation[0] * (conv.kernel_height - 1) + conv.output_padding[0] + 1
conv.conv_width =  (conv.input_width - 1) * conv.stride[1] - 2 * conv.padding[1] + conv.dilation[1] * (conv.kernel_width - 1) + conv.output_padding[1] + 1

#conv shape =

# +
conv.dilated_kernel_height = conv.dilation[0] * (conv.kernel_height - 1) + 1
conv.dilated_kernel_width = conv.dilation[1] * (conv.kernel_width - 1) + 1

conv.prepared_input_height = (conv.input_height - 1) * conv.stride[0] + 1 - 2 * conv.padding[0] + conv.output_padding[0] + 2 * conv.dilated_kernel_height - 2
conv.prepared_input_width = (conv.input_width - 1) * conv.stride[1] + 1 - 2 * conv.padding[1] + conv.output_padding[1] + 2 * conv.dilated_kernel_width - 2
print(conv.prepared_input_height)


# x = conv.prepare_inputs(X)

# print("My conv")
# print(x)


LOSS = np.arange(0, np.float(1 * kernels_num * conv.conv_height * conv.conv_width)).reshape((1, kernels_num, conv.conv_height, conv.conv_width))
print(X)
x = conv.forward_prop(X, training = True)


print("FORWARD")
print(x.shape)
print(x)

x = conv.backward_prop(LOSS)

print("BACKWARD")
print(x.shape)
print(x)

# def rot180(w_rot_180):
#     for s in range(kernels_num):
#         w_rot_180[s] = np.fliplr(w_rot_180[s])6
#         w_rot_180[s] = np.flipud(w_rot_180[s])
#     return w_rot_180

# print("ROT180\n", rot180(W))

# W = np.array([[2, 0], [3, 1]]).reshape(1, 1, kernel_size, kernel_size)

m = nn.ConvTranspose2d(inputs_num, kernels_num, kernel_size, stride= 1, padding = (0, 0),  output_padding = 0, dilation = (1, 1), bias = False)

m.weight = nn.Parameter(torch.from_numpy(W).float())
# m.bias = nn.Parameter(torch.from_numpy(BIAS).float())

input = torch.from_numpy(X).float()
output = m(input)
print(output.shape)
print(output)

GRAD = output.backward(torch.from_numpy(LOSS).float(), retain_graph=True)
print(GRAD)




# W = np.arange(0, np.float(1 * kernels_num * 2 * 4)).reshape((1, kernels_num, 2, 4))
# print(W)
# print(rot180(W))
# print(rot180(rot180(rot180(W))))

