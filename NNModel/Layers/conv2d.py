import numpy as np
from numba import njit
from NNModel.activations import activations


class Conv2D():
    #TODO
    #add accumulate gradients for batch backprop
    #maybe remove batch_size; only propagate one sample, not batch |yep
    #add bias
    #verify speed of padding/unpadding; maybe native numpy is faster than numba
    #check backprop and gradients
    #add numba to forward/ backward/ gradient

    def __init__(self, kernels_num, kernel_shape, input_shape, activation = None, padding = (0, 0), stride = (1, 1)):
        self.kernels_num = kernels_num
        self.kernel_height, self.kernel_width = kernel_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape

        if type(activation) is str:
            self.activation_function = activations[activation]    
        else:
            self.activation_function = activation
          

        self.w = None

       

    def build(self, optimizer):
        self.optimizer = optimizer
        self.channels_num, self.input_height, self.input_width = self.input_shape
        
        self.conv_height = (self.input_height + 2 * self.padding[0]  -  self.kernel_height) // self.stride[0]   + 1
        self.conv_width =  (self.input_width + 2 * self.padding[1] - self.kernel_width) // self.stride[1] + 1

        self.w = np.random.normal(0, pow(self.kernel_height * self.kernel_width, -0.5), (self.kernels_num, self.channels_num, self.kernel_height, self.kernel_width))

        
        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.output_shape = (self.kernels_num, self.conv_width, self.conv_height)
     

    def forward_prop(self, X, training):
        self.input_data = self.make_padding(X, self.padding)
        
        self.batch_size = len(self.input_data)

        self.conv_layer = np.zeros((self.batch_size, self.kernels_num, self.conv_height, self.conv_width))

        
        for b in range(self.batch_size):
            for k in range(self.kernels_num):
                for c in range(self.channels_num):
                    for h in range(self.conv_height):
                        for w in range(self.conv_width):

                            self.conv_layer[b, k, h, w] += (
                                np.sum(self.input_data[b, c, h * self.stride[0] : h * self.stride[0] + self.kernel_height, w * self.stride[1] : w * self.stride[1] + self.kernel_width] *  self.w[k, c]
                                )
                                # + bias
                            )


        return self.activation.function(self.conv_layer)

    def backward_prop(self, error):
        error *= self.activation.derivative(self.conv_layer)
        
        w_rot_180 = self.w
        
        error_pattern = np.zeros((
                        self.batch_size,
                        self.kernels_num, 
                        self.input_height + np.max([self.conv_height, self.kernel_height]) - 1, 
                        self.input_width + np.max([self.conv_width, self.kernel_width]) - 1
                        ))

        conv_backprop_error = np.zeros((self.batch_size, self.channels_num, self.input_height, self.input_width))

        temp_error = np.zeros(
            (
                self.stride[0] * self.conv_height - (self.stride[0] - 1),
                self.stride[1] * self.conv_width - (self.stride[1] - 1),
            )
        )

        for b in range(self.batch_size):
            for i in range(self.kernels_num):
                temp_error[::self.stride[0], ::self.stride[1]]  = error[b, i]

                error_pattern[
                    b,
                    i,
                    self.kernel_height - 1 : self.conv_height + self.kernel_height - 1,
                    self.kernel_width - 1 : self.conv_width + self.kernel_width - 1,
                ] = temp_error # Матрица ошибок нужного размера для прогона по ней весов

        for s in range(self.kernels_num):
            w_rot_180[s] = np.fliplr(w_rot_180[s])
            w_rot_180[s] = np.flipud(w_rot_180[s])

        for b in range(self.batch_size):
            for c in range(self.channels_num):
                for k in range(self.kernels_num):
                    for h in range(self.input_height):
                        for w in range(self.input_width):

                            conv_backprop_error[b, c, h, w] += np.sum(
                                error_pattern[b, k, h : h + self.kernel_height, w : w + self.kernel_width] * w_rot_180[k, c]
                            )

        self.grad_w = self.compute_gradients(error)

        conv_backprop_error = self.make_unpadding(conv_backprop_error, self.padding)

        return conv_backprop_error


    def compute_gradients(self, error):

        gradient = np.zeros((self.w.shape))

        temp_error = np.zeros(
            (
                self.stride[0] * self.conv_height - (self.stride[0] - 1),
                self.stride[1] * self.conv_width - (self.stride[1] - 1),
            )
        )

        for b in range(self.batch_size):
            for k in range(self.kernels_num):
                for c in range(self.channels_num):
                    for h in range(self.kernel_height):
                        for w in range(self.kernel_width):
                            temp_error[::self.stride[0], ::self.stride[1]] = error[b, k]

                            gradient[k, c, h, w] += np.sum(
                                temp_error
                                * self.input_data[b, c, h : h + self.stride[0] * self.conv_height - (self.stride[0] - 1), w : w + self.stride[1] * self.conv_width - (self.stride[1] - 1)]
                            )


        return gradient




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
                    padding[0] : layer.shape[2] + padding[0],
                    padding[1] : layer.shape[3] + padding[1],
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
                    padding[0] : layer.shape[2] - padding[0],
                    padding[1] : layer.shape[3] - padding[1],
                ]

        return unpadded_layer
 

    def update_weights(self, layer_num):
        self.w = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)