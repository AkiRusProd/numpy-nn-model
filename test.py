import numpy as np


# n1, n2 = 5, 5
# X = np.arange(1, n1 * n2 + 1).reshape(n1, n2)

# print(f'Input X: \n {X}')

# m1, m2 = 3, 3
# w = np.arange(1, m1 * m2 + 1).reshape(m1, m2)

# print(f'Input W: \n {w}')

# conv_h, conv_w = n1 - m1 +1, n2 - m2 +1

# zeropadded_w = np.pad(w, [(n1 - m1, 0), (0, n2 - m2)])
# print(f'Zeropadded W: \n {zeropadded_w}')

# rows_num = zeropadded_w.shape[0]
# cols_num = zeropadded_w.shape[1]


# toeplitz_filters = []#np.zeros((cols_num, cols_num - 1))

# for i in reversed(range(rows_num)):
#     y = np.vstack([np.roll(zeropadded_w[i], j) for j in range(cols_num - 1)])
    
#     #MORE FASTER METHOD
#     # y = np.zeros((cols_num - 1, cols_num))
#     # for j, v in enumerate(zeropadded_w[i]):
#     #     np.fill_diagonal(y[:,j:], v)

#     toeplitz_filters.append(y.T)

#     print(f'Toeplitz filter: \n {y.T}')


# toeplitz_matrix = np.zeros((cols_num * rows_num, (cols_num - 1) * (conv_h * conv_w - 7)))#2 len(toeplitz_filters) - 1)

# print(toeplitz_filters[i + 1].shape, rows_num, cols_num)
# toeplitz_matrix_ind = np.vstack([np.roll(np.arange(0, len(toeplitz_filters)), j) for j in range(conv_h * conv_w - 7)]).T
# print(f' Toeplitz ind \n {toeplitz_matrix_ind}')


# for i in range(rows_num):
#     for j in range(conv_h * conv_w  - 7):
#         # print((i) * rows_num , (i+1) * rows_num,  (j) * (cols_num - 1) , (j+1) * (cols_num - 1))
#         toeplitz_matrix[(i) * rows_num : (i+1) * rows_num,  (j) * (cols_num - 1) : (j+1) * (cols_num - 1)] = toeplitz_filters[toeplitz_matrix_ind[i][j]]

    
    
    


# print(f'Toeplitz matrix: \n {np.flip(toeplitz_matrix)}')



# print(f'Convolution: \n {np.dot(np.array(X.flatten(), ndmin = 2), np.flip(toeplitz_matrix)).reshape(conv_h, conv_w)}')





#CONV COMPARION WITH torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# conv = nn.Conv2d(6, 3, m1)
# # conv2 = nn.Conv2d(3, 1, 2)
# t_w = np.arange(1, m1 * m2 * 3 + 1).reshape(3, 1, m1, m2)
# # t_w2 = np.arange(1, m1 * m2 * 2 + 1).reshape(2, 1, m1, m2)

# # conv.weight.data = torch.from_numpy(w).float()
# conv.weight = nn.Parameter(torch.from_numpy(t_w).float())
# # conv2.weight = nn.Parameter(torch.from_numpy(t_w2).float())

# t_X = np.arange(1, n1 * n2 * 5 + 1).reshape(5, 1, n1, n2)

# x = conv(torch.from_numpy(t_X).float())
# # x = conv2(torch.from_numpy(x.reshape(5, 3, 3, 3).cpu().detach().numpy()).float())

# print(x)



import time


start_time = time.time()

# def convolve_1d(array, kernel):
#     ks = kernel.shape[0] # shape gives the dimensions of an array, as a tuple
#     final_length = array.shape[0] - ks + 1
#     return np.array([(array[i:i+ks]*kernel).sum() for i in range(final_length)])

# def convolve_2d(array,kernel):
#   ks = kernel.shape[1] # shape gives the dimensions of an array, as a tuple
#   final_height = array.shape[1] - ks + 1
#   return np.array([convolve_1d(array[:,i:i+ks],kernel) for i in range(final_height)]).T


# # import scipy.signal
# # print(np.array_equal(c, scipy.signal.convolve(a,b, mode="valid")))
# print(convolve_2d(X,w))

# # print(scipy.signal.convolve(a,b, mode="valid"))

# print(f'T1 {time.time()-start_time}')

start_time = time.time()


# X_size = 4
# batch_size = 1
# channels = 1
# kernels_num = 6
# X = np.arange(1, batch_size * channels * X_size * X_size + 1).reshape(batch_size, channels, X_size, X_size)

# print(f'Input X: \n {X}')

# w_size = 2
# w0 = np.arange(1,kernels_num * w_size * w_size + 1).reshape(kernels_num, w_size, w_size)

# print(f'Input W: \n {w0}')

# stride = 1

# conv_size = (X_size - w_size) // (stride) + 1


# ###МОЯ СВЕРТКА БЫСТРЕЙ
# def convolution(
#         input_layer = X,
#         weights = w0,
#         kernels_size = w_size,
#       ):

#         conv_layer = np.zeros((batch_size, channels, conv_size, conv_size))

#         for b in range(batch_size):
#             for c in range(channels):
#                 for h in range(0, conv_size):
#                     for w in range(0, conv_size):
#                         conv_layer[b, c, h, w] = (
#                             np.sum(input_layer[b, c, h * stride : h * stride + kernels_size, w * stride : w * stride + kernels_size] * weights)
#                         )

#         return conv_layer

# conv_layer = convolution()
# # print(conv_layer)
# # print(f'T2 {time.time()-start_time}')

# temp_error = np.zeros(
#             (
#                 stride * conv_size - (stride - 1),
#                 stride * conv_size - (stride - 1),
#             )
#         )


# gradient = np.zeros((w0.shape))
# error = np.ones((conv_layer.shape))

# for b in range(batch_size):
#     for c in range(channels):
#         for h in range(w_size):
#             for w in range(w_size):
#                 temp_error[::stride, ::stride] = error[b, c]

#                 gradient[h][w] = np.sum(
#                     temp_error * X[b, c, h : h + stride * conv_size - (stride - 1), w  : w + stride * conv_size - (stride - 1)]
#                 )

# # print(f'{gradient = }')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# conv = nn.Conv2d(1, kernels_num, w_size, stride = stride, bias=False)
# t_w = w0.reshape(1, kernels_num, w_size, w_size)


# conv.weight = nn.Parameter(torch.from_numpy(t_w).float())


# t_X = X.reshape(1, channels, X_size, X_size)

# x = conv(torch.from_numpy(t_X).float())


# print(x)

# conv = nn.Conv2d(4, 7, 3, stride=2)
# # non-square kernels and unequal stride and with padding
# conv2 = nn.Conv2d(7, 10, (3, 3), stride=(1, 1), padding=(2, 2))
# # non-square kernels and unequal stride and with padding and dilation
# conv3 = nn.Conv2d(10, 10, (3, 3), stride=(1, 1), padding=(2, 2))#, dilation=(3, 1)
# input = torch.randn(1, 4, 5, 5)
# x1 = conv(input)
# x2 = conv2(x1)
# x = conv3(x2)

# print(x1.shape)
# print(x2.shape)
# print(x.shape)


# conv = nn.ConvTranspose2d(1, 1, w_size, stride = stride)
# t_w = w0.reshape(1, 1, w_size, w_size)


# conv.weight = nn.Parameter(torch.from_numpy(t_w).float())


# t_X = X.reshape(1, 1, X_size, X_size)

# x = conv(torch.from_numpy(t_X).float())
# print(x)

# from numba import njit

# start_time = time.time()
# # @njit
# def make_padding(layer, padding):
#     padded_layer = np.pad(layer, ((padding[0], padding[1]), (padding[1], padding[0])), constant_values = 0)
#     # padded_layer = np.zeros(
#     #     (
#     #         layer.shape[0] + 2 * padding[0],
#     #         layer.shape[1] + 2 * padding[1],
#     #     )
#     # )


#     # padded_layer[
#     #         padding[0] : layer.shape[0] + padding[0],
#     #         padding[1] : layer.shape[1] + padding[1],
#     #     ] = layer

#     return padded_layer


# print(make_padding(X, (2,2)))
# print(f'T2 {time.time()-start_time}')


# z = np.tile([1, 2, 3], (2, 1))
# print(z)

# print(np.sum(z, axis=0))


# def forward_pass(X, p = 0.1, is_training = True):
#     assert p > 0
#     if is_training:
#         _mask = np.random.uniform(size=X.shape) > p
#         y = X * _mask
#     else:
#         y = X * (1.0 - p)

#     # print(_mask)
#     return y

# print(forward_pass(X))



# def forward_prop(X, rate = 0.1):
#     mask = np.random.binomial(
#                     n = 1,
#                     p = 1 - rate,
#                     size = X.shape,
#                 )

#     return X * mask


# print(forward_prop(X))

# error = [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11]]
# temp = np.zeros_like((error))
# print(temp[2, :])

# temp[2, :] = error[-1]

# error = np.zeros_like((error))[2, :]
# print(temp)


# from reshape import Reshape

# layer = Reshape(shape = (3, 1, 16))
# print(layer.forward_prop(X).shape)
# print(layer.backward_prop(X).shape)

# from flatten import Flatten
# layer = Flatten()
# print(layer.forward_prop(X).shape)
# print(layer.backward_prop(X).shape)

# print(np.array([1, 2 ,3 ,4, 5, 6, 7, 8, 9]).reshape(1, *(3, 3)))




# input_height, input_width = 4, 5
# batch_size = 1
# channels = 10
# kernels_num = 6
# kernels_height, kernel_width = 3, 4
# X = np.arange(1, batch_size * channels * input_height * input_width + 1).reshape(batch_size, channels, input_height, input_width)

# print(f'Input X: \n {X.shape}')

# w0 = np.arange(1,kernels_num * channels * kernels_height * kernel_width + 1).reshape(kernels_num, channels, kernels_height, kernel_width)

# print(f'Input W: \n {w0.shape}')

# stride = 1

# # conv_height= (input_height - kernels_size) // (stride) + 1


# from conv2d import Conv2D

# conv = Conv2D(kernels_num, (kernels_height, kernel_width), (channels, input_height, input_width))
# conv.init_weights()

# x = conv.forward_prop(X)
# print(x.shape)
# print(x)

# x = conv.backward_prop(x)
# print(x.shape)
# print(x)

from NNModel.Layers import Dense
from NNModel import Model

model = Model()

model.fit()