from neunet.autograd import Tensor
import numpy as np



class _MaxPool2dTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad):
        (
            X,
            kernel_size,
            stride,
            padding,
            dilation,
            input_size,
            output_size,
            dilated_kernel_size,
            kernel,
            windows,
            O_einsum
        ) = self.args

        # grad_X = np.where(O_args == 1, grad[..., None, None], 0)
        batch_size, in_channels, in_height, in_width = input_size
        # X_data = set_padding(X.data, padding, value=-np.inf)

        #TODO: vectorize this
        # grad_X = np.zeros_like(X_data)
        # #https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec
        # for n in range(batch_size):
        #     for c in range(in_channels):
        #         for i in range(output_size[0]):
        #             for j in range(output_size[1]):
        #                 # get the index in the region i,j where the value is the maximum
        #                 i_t, j_t = np.where(np.nanmax(X_data[n, c, i * stride[0] : i * stride[0] + dilated_kernel_size[0], j * stride[1] : j * stride[1] + dilated_kernel_size[1]] * kernel[None, None, ...]) == X_data[n, c, i * stride[0] : i * stride[0] + dilated_kernel_size[0], j * stride[1] : j * stride[1] + dilated_kernel_size[1]])
        #                 i_t, j_t = i_t[0], j_t[0] # ignore the other maximum values indices
                        
        #                 # only the position of the maximum element in the region i,j gets the incoming gradient, the other gradients are zero
        #                 grad_X[n, c, i * stride[0] : i * stride[0] + dilated_kernel_size[0], j * stride[1] : j * stride[1] + dilated_kernel_size[1]][i_t, j_t] += grad[n, c, i, j]


        grad = grad.reshape(-1, 1)
        windows = O_einsum.reshape(-1, dilated_kernel_size[0] * dilated_kernel_size[1])

        grad_col = np.zeros_like(windows)
        grad_col[np.arange(grad_col.shape[0]), np.nanargmax(windows, axis=1)] = grad.reshape(-1)
        grad_col = grad_col.reshape(batch_size, in_channels, output_size[0], output_size[1], dilated_kernel_size[0], dilated_kernel_size[0])

        grad_X = np.zeros((batch_size, in_channels, in_height + 2 * padding[0], in_width + 2 * padding[1]))
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                grad_X[:, :, i*stride[0]:i*stride[0]+dilated_kernel_size[0], j*stride[1]:j*stride[1]+dilated_kernel_size[1]] += grad_col[:, :, i, j, :, :]

        grad_X = remove_padding(grad_X, padding)

        X.backward(grad_X)



class MaxPool2d:
    
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride) if stride else self.kernel_size
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

            self.input_size = None

        def build(self):
            self.kernel_height, self.kernel_width = self.kernel_size
            self.input_height, self.input_width = self.input_size[2:]

            if self.padding == "valid":
                self.padding == (0, 0, 0, 0)
            elif self.padding == "same" or self.padding == "real same":
                if self.padding == "same":
                    padding_up_down = self.dilation[0] * (self.kernel_height - 1) - self.stride[0] + 1 
                    padding_left_right = self.dilation[1] * (self.kernel_width  - 1) - self.stride[1] + 1
                elif self.padding == "real same":
                    padding_up_down = (self.stride[0] - 1) * (self.input_height - 1) + self.dilation[0] * (self.kernel_height - 1)
                    padding_left_right = (self.stride[1] - 1) * (self.input_width- 1) + self.dilation[1] * (self.kernel_width  - 1)

                if padding_up_down % 2 == 0:
                    padding_up, padding_down = padding_up_down // 2, padding_up_down // 2
                else:
                    padding_up, padding_down = padding_up_down // 2, padding_up_down - padding_up_down // 2

                if padding_left_right % 2 == 0:
                    padding_left, padding_right = padding_left_right // 2, padding_left_right // 2
                else:
                    padding_left, padding_right = padding_left_right // 2, padding_left_right - padding_left_right // 2
        

                self.padding = (abs(padding_up), abs(padding_down), abs(padding_left), abs(padding_right))

            elif len(self.padding) == 2:
                self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1]) #(up, down, left, right) padding ≃ (2 * vertical, 2 *horizontal) padding


            self.output_height = (self.input_height + self.padding[0] + self.padding[1] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            self.output_width  = (self.input_width  + self.padding[2] + self.padding[3] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
            self.output_size = (self.output_height, self.output_width)

            self.dilated_kernel_height =  self.dilation[0] * (self.kernel_height - 1) + 1
            self.dilated_kernel_width = self.dilation[1] * (self.kernel_width - 1) + 1
            self.dilated_kernel_size = (self.dilated_kernel_height, self.dilated_kernel_width)

            # self.stride_compared_input_height = (self.output_height - 1) * self.stride[0] - self.padding[0] - self.padding[1] +  self.dilated_kernel_height
            # self.stride_compared_input_width = (self.output_width - 1) * self.stride[1] - self.padding[2] - self.padding[3] +  self.dilated_kernel_width
        
            # self.prepared_input_height = (self.stride_compared_input_height + self.padding[0] + self.padding[1])
            # self.prepared_input_width = (self.stride_compared_input_width + self.padding[2] + self.padding[3])

            self.kernel = set_dilation_stride(np.ones((self.kernel_height, self.kernel_width)), self.dilation, value = np.nan)


            
        
        def forward(self, X):
            if self.input_size == None:
                self.input_size = X.shape
                self.build()
            
            X_data = set_padding(X.data, self.padding, value = -np.inf)

            batch_size, in_channels, in_height, in_width = X_data.shape
            
            batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
            windows = np.lib.stride_tricks.as_strided(
                X_data,
                (batch_size, in_channels, self.output_size[0], self.output_size[1], self.dilated_kernel_size[0], self.dilated_kernel_size[1]),
                (batch_str, channel_str, self.stride[0] * kern_h_str, self.stride[1] * kern_w_str, kern_h_str, kern_w_str)
            )

            O_einsum = np.einsum('bihwkl,oikl->bihwkl', windows, self.kernel[None, None, ...])
            # O_args = np.where(O_einsum == np.nanmax(O_einsum, axis=(4, 5))[..., None, None], 1, 0)
            # O_argw = np.argwhere(O_einsum == np.nanmax(O_einsum, axis=(4, 5))[..., None, None])
            O = np.nanmax(O_einsum, axis=(4, 5)) #np.amax(windows, axis=(4, 5))

            #remove lines where first 4 elements are the same
            # new_O_argw = []
            # for i in range(len(O_argw)):
            #     if i == 0:
            #         new_O_argw.append(O_argw[i])
            #     elif np.all(O_argw[i][:4] == O_argw[i-1][:4]):
            #         pass
            #     else:
            #         new_O_argw.append(O_argw[i])
            # new_O_argw = np.array(new_O_argw)

           

            return _MaxPool2dTensor(O, [X, self.kernel_size, self.stride, self.padding, self.dilation, self.input_size, self.output_size, 
            self.dilated_kernel_size, self.kernel, windows, O_einsum], "maxpool2d")
           

        def backward(self, grad):
            pass

        def __call__(self, X):
            return self.forward(X)




   
def set_dilation_stride(layer, stride, value = 0):
    transposed_layer = np.full(
        (
            stride[0] * layer.shape[0] - (stride[0] - 1),
            stride[1] * layer.shape[1] - (stride[1] - 1),
        ),
        value,
    )
    
    transposed_layer[::stride[0], ::stride[1]] = layer

    return transposed_layer


def set_padding(layer, padding, value = 0):
    padded_layer = np.full(
        (   
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] + padding[0] + padding[1],
            layer.shape[3] + padding[2] + padding[3],
        ),
        value
    )

    padded_layer[
                :,
                :,
                padding[0] :padded_layer.shape[2] - padding[1],
                padding[2] :padded_layer.shape[3] - padding[3],
            ] = layer

    return padded_layer


def remove_padding(layer, padding):
    unpadded_layer = np.zeros(
        (
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] - padding[0] - padding[1],
            layer.shape[3] - padding[2] - padding[3],
        )
    )
    
    unpadded_layer = layer[
                :,
                :,
                padding[0] :layer.shape[2] - padding[1],
                padding[2] :layer.shape[3] - padding[3],
            ]

    return unpadded_layer





# class Maxpool():

#     def __init__(self, kernel_size, stride, padding, dilation=1): # dilation is not supported yet
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
        

#     def forward(self, x):
#         self.x = x
#         self.n, self.c, self.h, self.w = x.shape
#         self.h_out = (self.h + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)//self.stride + 1
#         self.w_out = (self.w + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)//self.stride + 1
#         self.x_col = self.im2col(x)
#         self.x_col = self.x_col.reshape(-1, self.kernel_size*self.kernel_size)
#         self.out = np.max(self.x_col, axis=1)
#         self.out = self.out.reshape(self.n, self.c, self.h_out, self.w_out)
#         return self.out

#     def backward(self, dout):
#         dout = dout.reshape(-1, 1)
#         self.dx_col = np.zeros_like(self.x_col)
#         self.dx_col[np.arange(self.dx_col.shape[0]), np.argmax(self.x_col, axis=1)] = dout.reshape(-1)
#         self.dx_col = self.dx_col.reshape(self.n, self.c, self.h_out, self.w_out, self.kernel_size, self.kernel_size)
#         self.dx = self.col2im(self.dx_col)
#         return self.dx

#     def im2col(self, x):
#         x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
#         x_col = np.zeros((self.n, self.c, self.h_out, self.w_out, self.kernel_size, self.kernel_size))
#         for i in range(self.h_out):
#             for j in range(self.w_out):
#                 x_col[:, :, i, j, :, :] = x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
#         x_col = x_col.reshape(-1, self.kernel_size*self.kernel_size)
#         return x_col

#     def col2im(self, x_col):
#         x_col = x_col.reshape(self.n, self.c, self.h_out, self.w_out, self.kernel_size, self.kernel_size)
#         x = np.zeros((self.n, self.c, self.h + 2*self.padding, self.w + 2*self.padding))
#         for i in range(self.h_out):
#             for j in range(self.w_out):
#                 x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += x_col[:, :, i, j, :, :]
#         return x[:, :, self.padding:self.padding+self.h, self.padding:self.padding+self.w]

# x = np.random.randn(10, 4, 28, 28)
# layer = Maxpool(2, 3, 0)
# myy = layer.forward(x)
# print(myy.shape)
# print(myy)

# mydx = layer.backward(np.ones_like(myy))
# print(layer.dx.shape)
# # print(layer.dx)

# from torch.nn import MaxPool2d
# import torch

# x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
# layer = MaxPool2d(2, 3, 0)

# y = layer(x)
# y.backward(torch.ones_like(y))
# print(x.grad)
# print(x.grad.shape)

# print(np.allclose(y.data, myy.data))
# print(np.allclose(x.grad, mydx))