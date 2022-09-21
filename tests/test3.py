import numpy as np

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, RepeatVector, \
TimeDistributed, RNN, LSTM, GRU, Bidirectional, Embedding
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam
from nnmodel.losses import MSE

# import torch
# import torch.nn.functional as F
model = Model()



# kernels_num = 2
# X_size = 7

# conv2d = Conv2D(kernels_num = kernels_num, kernel_shape = (3, 3), input_shape = (1, X_size, X_size), padding = 1, stride = 2, use_bias = False)
# conv2d.build()

# X = np.arange(X_size*X_size*1).reshape(1, 1, X_size, X_size)
# W = np.arange(3*3*1*kernels_num).reshape(kernels_num, 1, 3, 3)

# conv2d.w = W

# Y = conv2d.forward_prop(X, training = True)
# print(Y)

# loss = MSE()
# loss = loss.derivative(Y, np.ones_like(Y)) * 2  / np.prod(Y.shape[1:])
# loss = conv2d.backward_prop(loss)
# print(loss)


# torch_conv2d = torch.nn.Conv2d(in_channels = 1, out_channels = kernels_num, kernel_size = 3, padding = 1, stride = 2,bias = False)
# torch_conv2d.weight = torch.nn.Parameter(torch.from_numpy(W.reshape(kernels_num, 1, 3, 3)).float())
# TX = torch.tensor(torch.from_numpy(X).float(), requires_grad = True)
# Y = torch_conv2d.forward(TX)
# print(Y)

# torch_loss = torch.nn.MSELoss()(Y, torch.ones_like(Y))
# torch_loss.backward()
# print("Torch Grad\n", TX.grad)

# #I wanted to make sure the Conv2D derivatives match. Yes it is!



# kernels_num = 2
# X_size = 3
# kernel_size=2

# conv2d = Conv2DTranspose(kernels_num = kernels_num, kernel_shape = (kernel_size, kernel_size), input_shape = (1, X_size, X_size), padding = 1, stride = 2, use_bias = False)
# conv2d.build()

# X = np.arange(X_size*X_size*1).reshape(1, 1, X_size, X_size)
# W = np.arange(kernel_size*kernel_size*1*kernels_num).reshape(kernels_num, 1, kernel_size, kernel_size)

# print("X\n", X)
# print("W\n", W)

# conv2d.w = W

# Y = conv2d.forward_prop(X, training = True)
# print(Y)

# loss = MSE()
# loss = loss.derivative(Y, np.ones_like(Y)) * 2  / np.prod(Y.shape[1:])
# loss = conv2d.backward_prop(loss)
# print(loss)


# torch_conv2d = torch.nn.ConvTranspose2d(in_channels = 1, out_channels = kernels_num, kernel_size = kernel_size,padding = 1, stride = 2, bias = False)

# W = np.rot90(np.rot90(W, axes = (2 , 3)), axes = (2 , 3)).copy()
# torch_conv2d.weight = torch.nn.Parameter(torch.from_numpy(W.reshape(1, kernels_num, kernel_size, kernel_size)).float())#.transpose(0, 1, 3, 2)
# TX = torch.tensor(torch.from_numpy(X).float(), requires_grad = True)
# Y = torch_conv2d.forward(TX)
# print(Y)

# torch_loss = torch.nn.MSELoss()(Y, torch.ones_like(Y))
# torch_loss.backward()
# print("Torch Grad\n", TX.grad)

#I wanted to make sure the Conv2DTranspose derivatives match. Yes it is!




in_channels = 1
out_channels = 1
kernel_size = (3, 2)
x_size = 7
stride = (2, 3)
padding = 0


conv2d = Conv2D(kernels_num = out_channels, kernel_shape = kernel_size, input_shape = (in_channels, x_size, x_size), padding = padding, stride = stride, dilation = (2, 2), use_bias = False)
conv2d.build()

X = np.arange(x_size*x_size*in_channels).reshape(1, in_channels, x_size, x_size)
W = np.arange(kernel_size[1]*kernel_size[0]*in_channels*out_channels).reshape(out_channels, in_channels, kernel_size[0], kernel_size[1])


conv2d.w = W

naive_Y = conv2d.naive_forward_prop(X, training = True)
print("default conv:\n", naive_Y)

Y = conv2d.naive_backward_prop(naive_Y)
print("default conv grad:\n", Y)
print("default weights grad:\n", conv2d.grad_w)
print("default bias grad:\n", conv2d.grad_b)

vec_Y = conv2d.forward_prop(X, training = True)
print("vec conv:\n", vec_Y)

Y = conv2d.backward_prop(vec_Y)
print("vec conv grad:\n", Y)
print("vec weights grad:\n", conv2d.grad_w)
print("vec bias grad:\n", conv2d.grad_b)
# Y = conv2d.vect_forward_prop(X, training = True)
# print("conv:\n", Y)

# loss = MSE()
# loss = loss.derivative(Y, np.ones_like(Y)) * 2  / np.prod(Y.shape[1:])
# loss = conv2d.backward_prop(loss)
# print(loss)

# https://blog.ca.meron.dev/Vectorized-CNN/
#https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
# def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
#     print(output_size, kernel_size, stride, dilate, padding)
#     working_input = input
#     working_pad = padding
#     # dilate the input if necessary
#     if dilate != 0:
#         working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
#         working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

#     # pad the input if necessary
#     if working_pad != 0:
#         working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

#     in_b, in_c, out_h, out_w = output_size
#     out_b, out_c, _, _ = input.shape
#     batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

#     return np.lib.stride_tricks.as_strided(
#         working_input,
#         (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
#         (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
#     )



# class Conv2DNew:
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         self.cache = None

#         self._init_weights()

#     def _init_weights(self):
#         self.weight = None #1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
#         self.bias = np.zeros(self.out_channels)


#     def forward(self, x):
#         n, c, h, w = x.shape
#         out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
#         out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

#         windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

#         out = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

#         # add bias to kernels
#         # out += self.bias[None, :, None, None]

#         self.cache = x, windows
#         return out

#     def backward(self, dout):
#         x, windows = self.cache

#         padding = self.kernel_size - 1 if self.padding == 0 else self.padding
#         # print(dout.shape, x.shape, self.kernel_size, self.stride)
#         dout_windows = getWindows(dout, x.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
#         # print("dout_windows\n", dout_windows)
#         rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

#         db = np.sum(dout, axis=(0, 2, 3))
#         dw = np.einsum('bihwkl,bohw->oikl', windows, dout)
#         dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

#         return db, dw, dx


# in_channels = 3
# out_channels = 128
# kernel_size = 3
# stride = 2
# padding = 1
# batch_size = (4, in_channels, 12, 10)  # expected input size
# dout_size = (4, out_channels, 6, 5)


# x = np.random.random(batch_size)  # create data for forward pass
# dout = np.random.random(dout_size)  # create random data for backward
# print('x: ', x.shape)
# print('d_out: ', dout.shape)






# conv = Conv2DNew(in_channels, out_channels, kernel_size, stride, padding)

# conv.weight = W
# conv_out = conv.forward(X)
# db, dw, dx = conv.backward(conv_out)

# print('conv_out\n: ', conv_out)
# # print('db: ', db.shape)
# # print('dw: ', dw.shape)
# print('dx\n: ', dx)



    # def vect_forward_prop(self, X, training):
    #     self.input_data = self.set_padding(X, self.padding)
    #     self.w = self.set_stride(self.w, self.dilation)
        
    #     self.batch_size = len(self.input_data)

    #     self.output_data = self._vect_forward_prop(self.input_data, self.w, self.b, self.batch_size, self.channels_num, self.kernels_num, self.conv_height, self.conv_width, self.dilated_kernel_height, self.dilated_kernel_width, self.stride)

    #     return self.activation.function(self.output_data)


    # @staticmethod
    # def _vect_forward_prop(input_data, weights, bias, batch_size, channels_num, kernels_num, conv_height, conv_width, kernel_height, kernel_width, stride):
    #     # conv_layer = np.zeros((batch_size, kernels_num, conv_height, conv_width))





    #     #Now: Inp: 1; Ker: 1
    #     input_height, input_width = input_data.shape[2], input_data.shape[3]

    #     # W = np.zeros((conv_height * conv_width, input_height * input_width))
    #     # weights = weights.reshape(kernel_width, kernel_height)
    #     # for i in range(conv_height):
    #     #     for j in range(conv_width):
    #     #         for ii in range(kernel_height):
    #     #             for jj in range(kernel_width):
    #     #                 W[i * conv_width + j, i * input_width + j + ii * input_width + jj] = weights[ii, jj]
    #     # print("W\n", W)



    #     weights_matrix = np.zeros((conv_height * conv_width, input_height * input_width))


    #     input_data = input_data.reshape(-1, 1)

    #     weights_line = np.pad(weights.reshape(kernel_height, kernel_width),[(0, 0), (0, input_width - kernel_width)], constant_values = 0).reshape(1, -1)
    #     weights_line = np.tile(weights_line, (conv_height * conv_width, 1)) 
    #     weights_matrix[0 : weights_line.shape[0], 0 : weights_line.shape[1]] = weights_line
    #     # print("weights_matrix\n", weights_matrix)

    #     #https://stackoverflow.com/questions/71308321/shift-a-numpy-array-by-an-increasing-value-with-each-row
    #     # # add enough "0" columns for the shift
    #     # arr2 = np.c_[weights_matrix, np.zeros((weights_matrix.shape[0], weights_matrix.shape[0]-1), dtype=weights_matrix.dtype)]
    #     # # get the indices as ogrid
    #     # r, c = np.ogrid[:arr2.shape[0], :arr2.shape[1]]
    #     # # roll the values
    #     # arr2 = arr2[r, c-r]

    #     # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently

    #     rows, column_indices = np.ogrid[:weights_matrix.shape[0], :weights_matrix.shape[1]]

    #     # Use always a negative shift, so that column_indices are valid.
    #     # (could also use module operation)
    #     r = np.array([0, 1, 3, 4]) #add stride[0]
    #     r2 = np.arange(0, stride[1] * conv_height * conv_width, stride[1])
    #     # r3 = np.arange(0, stride[0] * conv_height, stride[0]).repeat(conv_height, axis=0)  #(stride[0] -1) #Problem is here
    #     r3 = np.arange(0, conv_height, 1).repeat(conv_width, axis=0) * (kernel_height * stride[0]- 1) #* (stride[0] + 1)#(stride[0] -1) #Problem is here

    #     print(r2, r3)
    #     r = r2 + r3
    #     print(r)
        
    #     r[r < 0] += weights_matrix.shape[1]
    #     column_indices = column_indices - r[:, np.newaxis]

    #     weights_matrix = weights_matrix[rows, column_indices]
        
    #     # print("W_mat\n", weights_matrix)
    #     # print("inp\n", input_data)

    #     conv = np.matmul(weights_matrix, input_data)


    #     return conv.reshape(batch_size, kernels_num, conv_height, conv_width)