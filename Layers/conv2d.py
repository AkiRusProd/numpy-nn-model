import numpy as np
from numba import njit


class Conv2D():
    #TODO
    #add accumulate gradients for batch backprop
    #maybe remove batch_size; only propagate one sample, not batch |yep
    #add bias
    #verify speed of padding/unpadding; maybe native numpy is faster than numba
    #check backprop and gradients
    #add numba to forward/ backward/ gradient

    def __init__(self, kernels_num, kernel_shape, input_shape, padding = (0, 0), stride = (1, 1)):
        self.kernels_num = kernels_num
        self.kernel_height, self.kernel_width = kernel_shape
        self.padding = padding
        self.stride = stride
        self.input_channels, self.input_height, self.input_width = input_shape

        self.kernel_per_input = self.kernels_num // self.input_channels

        self.activation = NotImplemented
        self.activation_der = NotImplemented

        self.conv_height = (self.input_height + 2 * self.padding[0]  -  self.kernel_height) // self.stride[0]   + 1
        self.conv_width =  (self.input_width + 2 * self.padding[1] - self.kernel_width) // self.stride[1] + 1

        self.w = None

       

    def init_weights(self):
        self.w = np.random.normal(0, 1, (self.kernels_num, self.kernel_height, self.kernel_width))
     

    def forward_prop(self, X):
        self.input_data = self.make_padding(X, self.padding)
        
        self.batch_size, self.input_channels, self.input_height, self.input_width = self.input_data.shape

        self.conv_layer = np.zeros((self.batch_size, self.kernels_num, self.conv_height, self.conv_width))

        l, r = 0, self.kernel_per_input
        if self.input_channels <= self.kernels_num:
            for b in range(self.batch_size):
                for c in range(self.input_channels):
                    for i in range(l, r):
                        for h in range(self.conv_height):
                            for w in range(self.conv_width):

                                self.conv_layer[b, i, h, w] = (
                                    np.sum(self.input_data[b, c, h * self.stride[0] : h * self.stride[0] + self.kernel_height, w * self.stride[1] : w * self.stride[1] + self.kernel_width] *  w[i]
                                    )
                                    # + bias
                                )
                    l = r
                    r += self.kernel_per_input

        else:
            for b in range(self.batch_size):
                for k in range(self.kernels_num):
                    for i in range(l, r):
                        for h in range(self.conv_height):
                            for w in range(self.conv_width):

                                self.conv_layer[b, k, h, w] += (
                                    np.sum(self.input_data[b, i, h * self.stride[0] : h * self.stride[0] + self.kernel_height, w * self.stride[1] : w * self.stride[1] + self.kernel_width] * w[k]
                                    )
                                    # + bias
                                )
                    l = r
                    r += self.kernel_per_input

        return self.activation(self.conv_layer)

    def backward_prop(self, error):
        error *= self.activation_der(self.conv_layer)
        
        w_rot_180 = self.w
        
        error_pattern = np.zeros((
                        self.batch_size,
                        self.kernels_num, 
                        self.input_height + np.max([self.conv_height, self.kernel_height]) - 1, 
                        self.input_width + np.max([self.conv_width, self.kernel_width]) - 1
                        ))

        conv_backprop_error = np.zeros((self.batch_size, self.kernels_num, self.input_height, self.input_width))

        temp_error = np.zeros(
            (
                self.stride[0] * self.conv_height - (self.stride[0] - 1),
                self.stride[1] * self.conv_width - (self.stride[1] - 1),
            )
        )

        for i in range(self.kernels_num):
            temp_error[::self.stride[0], ::self.stride[1]] = error[i]

            error_pattern[
                i,
                self.kernel_height - 1 : self.conv_height + self.kernel_height - 1,
                self.kernel_width - 1 : self.conv_width + self.kernel_width - 1,
            ] = temp_error # Матрица ошибок нужного размера для прогона по ней весов

        for s in range(self.kernels_num):
            w_rot_180[s] = np.fliplr(w_rot_180[s])
            w_rot_180[s] = np.flipud(w_rot_180[s])

        for b in range(self.batch_size):
            for s in range(self.kernels_num):
                for h in range(self.input_height):
                    for w in range(self.input_width):

                        conv_backprop_error[b, s, h, w] = np.sum(
                            error_pattern[b, s, h : h + self.kernel_height, w : w + self.kernel_width] * w_rot_180[s]
                        )

        self.grad_w = self.compute_gradients(error)

        conv_backprop_error = self.make_unpadding(conv_backprop_error, self.padding)
        conv_backprop_error = self.get_actual_errors_num(conv_backprop_error, self.input_channels, self.kernels_num, self.kernel_per_input)

        return conv_backprop_error


    def compute_gradients(self, error):

        l = 0
        r = self.kernel_per_input

        gradient = np.zeros((w.shape))

        temp_error = np.zeros(
            (
                self.stride[0] * self.conv_height - (self.stride[0] - 1),
                self.stride[1] * self.conv_width - (self.stride[1] - 1),
            )
        )

        if self.input_channels <= self.kernels_num:
            for b in range(self.batch_size):
                for ker in range(self.input_channels):
                    for i in range(l, r):
                        for h in range(self.kernel_height):
                            for w in range(self.kernel_width):
                                temp_error[::self.stride[0], ::self.stride[1]] = error[i, b]

                                gradient[i][h][w] = np.sum(
                                    temp_error
                                    * self.input_data[b, ker, h : h + self.stride[0] * self.conv_height - (self.stride[0] - 1), w : w + self.stride[1] * self.conv_width - (self.stride[1] - 1)]
                                )

                    l = r
                    r += self.kernel_per_input
        else:
            for b in range(self.batch_size):
                for ker in range(self.kernels_num):
                    for h in range(self.kernel_height):
                        for w in range(self.kernel_width):
                            for i in range(l, r):
                                temp_error[::self.stride[0], ::self.stride[1]] = error[ker, b]

                                gradient[ker][h][w] += np.sum(
                                    temp_error[ker]
                                    * self.input_data[b, i, h : h + self.stride[0] * self.conv_height - (self.stride[0] - 1), w  : w + self.stride[1] * self.conv_width - (self.stride[1] - 1)]
                                )

                    l = r
                    r += self.kernel_per_input

        return gradient


    @staticmethod
    @njit
    def get_actual_errors_num(
        error,
        prev_kernels_num,
        kernels_num,
        kernel_per_input,
    ):  #
        pooling_layer_error = np.zeros((prev_kernels_num, error[1], error[2]))

        l = 0
        r = kernel_per_input

        if prev_kernels_num <= kernels_num:
            for k in range(
                prev_kernels_num,
            ):
                for s in range(l, r):

                    pooling_layer_error[k] += error[s]


                l = r
                r += kernel_per_input
        else:
            for k in range(kernels_num):
                for s in range(l, r):

                    pooling_layer_error[s] = error[k]

                l = r
                r += kernel_per_input

        return pooling_layer_error



    @staticmethod
    @njit
    def make_padding(layer, padding):
        # padded_layer = np.pad(layer, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), constant_values = 0)
        padded_layer = np.zeros(
            (   
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] + 2 * padding[0],
                layer.shape[3] + 2 * padding[1],
            )
        )

        for b in range(layer.shape[0]):
            for i in range(layer.shape[1]):
                padded_layer[
                    b,
                    i,
                    padding[0] : layer.shape[1] + padding[0],
                    padding[1] : layer.shape[2] + padding[1],
                ] = layer[b, i]

        return padded_layer

    @staticmethod
    @njit
    def make_unpadding(layer, padding):
        # losses[k] = losses[k][...,self.topology[k+1]['padding']:-self.topology[k+1]['padding'],self.topology[k+1]['padding']:-self.topology[k+1]['padding']]
        unpadded_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] - 2 * padding[0],
                layer.shape[3] - 2 * padding[1],
            )
        )
        for b in range(layer.shape[0]):
            for i in range(layer.shape[1]):
                unpadded_layer[b, i] = layer[
                    b,
                    i,
                    padding[0] : layer.shape[1] - padding[0],
                    padding[1] : layer.shape[2] - padding[1],
                ]

        return unpadded_layer
 