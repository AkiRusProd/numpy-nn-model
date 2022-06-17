import numpy as np
from numba import njit
from NNModel.activations import activations


class Conv2D():
    #TODO
    #add bias
    #verify speed of padding/unpadding; maybe native numpy is faster than numba
    #maybe implement padding as separate layer
    #add padding when stride is out of scope
    

    def __init__(self, kernels_num, kernel_shape, input_shape = None, activation = None, padding = (0, 0), stride = (1, 1), dilation = (1, 1)):
        self.kernels_num = kernels_num
        self.kernel_height, self.kernel_width = kernel_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.dilation = dilation
        
        if type(activation) is str or activation is None:
            self.activation = activations[activation]
        else:
            self.activation = activation

          

        self.w = None

       

    def build(self, optimizer):
        self.optimizer = optimizer
        self.channels_num, self.input_height, self.input_width = self.input_shape
        
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv_height = (self.input_height + 2 * self.padding[0]  - self.dilation[0] * (self.kernel_height - 1) - 1) // self.stride[0]   + 1
        self.conv_width =  (self.input_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_width - 1) - 1) // self.stride[1] + 1

        self.dilated_kernel_height = self.dilation[0] * (self.kernel_height - 1) + 1
        self.dilated_kernel_width = self.dilation[1] * (self.kernel_width - 1) + 1

        self.w = np.random.normal(0, pow(self.kernel_height * self.kernel_width, -0.5), (self.kernels_num, self.channels_num, self.kernel_height, self.kernel_width))

        
        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.output_shape = (self.kernels_num, self.conv_width, self.conv_height)
     

    def forward_prop(self, X, training):
        self.input_data = self.set_padding(X, self.padding)
        self.w = self.set_stride(self.w, self.dilation)
        
        self.batch_size = len(self.input_data)

        self.conv_layer = self._forward_prop(self.input_data, self.w, self.batch_size, self.channels_num, self.kernels_num, self.conv_height, self.conv_width, self.dilated_kernel_height, self.dilated_kernel_width, self.stride)

        return self.activation.function(self.conv_layer)


    @staticmethod
    @njit
    def _forward_prop(input_data, weights, batch_size, channels_num, kernels_num, conv_height, conv_width, kernel_height, kernel_width, stride):
        conv_layer = np.zeros((batch_size, kernels_num, conv_height, conv_width))

        for b in range(batch_size):
            for k in range(kernels_num):
                for c in range(channels_num):
                    for h in range(conv_height):
                        for w in range(conv_width):
                            
                            conv_layer[b, k, h, w] += (
                                np.sum(input_data[b, c, h * stride[0] : h * stride[0] + kernel_height, w * stride[1] : w * stride[1] + kernel_width] *  weights[k, c]
                                )
                                # + bias
                            )

        return conv_layer

    def backward_prop(self, error):
        error *= self.activation.derivative(self.conv_layer)

        self.grad_w = self.compute_gradients(error, self.input_data, self.w, self.batch_size, self.channels_num, self.kernels_num,  self.conv_height, self.conv_width, self.dilated_kernel_height, self.dilated_kernel_width, self.stride)
        conv_backprop_error = self._backward_prop(error, self.w, self.batch_size, self.channels_num, self.kernels_num, self.input_height, self.input_width, self.conv_height, self.conv_width, self.dilated_kernel_height, self.dilated_kernel_width, self.stride)

        conv_backprop_error = self.remove_padding(conv_backprop_error, self.padding)
        self.w = self.remove_stride(self.w, self.dilation)

        return conv_backprop_error


    @staticmethod
    @njit
    def _backward_prop(error, weights, batch_size, channels_num, kernels_num, input_height, input_width, conv_height, conv_width, kernel_height, kernel_width, stride):

        w_rot_180 = weights
        
        error_pattern = np.zeros((
                        batch_size,
                        kernels_num, 
                        stride[0] * conv_height - (stride[0] - 1) +  2 * (kernel_height - 1),     #input_height + np.max(np.array([conv_height, kernel_height])) - 1, 
                        stride[1] * conv_width - (stride[1] - 1) +  2 * (kernel_width - 1),       #input_width + np.max(np.array([conv_width, kernel_width])) - 1
                        ))

        conv_backprop_error = np.zeros((batch_size, channels_num, input_height, input_width))

        temp_error = np.zeros(
            (   batch_size,
                kernels_num, 
                stride[0] * conv_height - (stride[0] - 1),
                stride[1] * conv_width - (stride[1] - 1),
            )
        )

        temp_error[:, :, ::stride[0], ::stride[1]]  = error

        error_pattern[
                    :,
                    :,
                    kernel_height - 1 : stride[0] * conv_height - (stride[0] - 1) + kernel_height - 1, #kernel_height - 1 : conv_height + kernel_height - 1,
                    kernel_width - 1 :  stride[0] * conv_width - (stride[0] - 1) + kernel_width - 1,   #kernel_width - 1 :  conv_width  + kernel_width - 1,
                ] = temp_error # Матрица ошибок нужного размера для прогона по ней весов

        for k in range(kernels_num):
            for c in range(channels_num):
                w_rot_180[k, c] = np.fliplr(w_rot_180[k, c])
                w_rot_180[k, c] = np.flipud(w_rot_180[k, c])

        for b in range(batch_size):
            for c in range(channels_num):
                for k in range(kernels_num):
                    for h in range(input_height):
                        for w in range(input_width):

                            conv_backprop_error[b, c, h, w] += np.sum(
                                error_pattern[b, k, h : h + kernel_height, w : w + kernel_width] * w_rot_180[k, c]
                            )
    
        return conv_backprop_error
        

    @staticmethod
    @njit
    def compute_gradients(error, input_data, weights, batch_size, channels_num, kernels_num, conv_height, conv_width, kernel_height, kernel_width, stride):

        gradient = np.zeros((weights.shape))

        temp_error = np.zeros(
            (
                stride[0] * conv_height - (stride[0] - 1),
                stride[1] * conv_width - (stride[1] - 1),
            )
        )

        for b in range(batch_size):
            for k in range(kernels_num):
                for c in range(channels_num):
                    for h in range(kernel_height):
                        for w in range(kernel_width):
                            temp_error[::stride[0], ::stride[1]] = error[b, k]

                            gradient[k, c, h, w] += np.sum(
                                temp_error
                                * input_data[b, c, h : h + stride[0] * conv_height - (stride[0] - 1), w : w + stride[1] * conv_width - (stride[1] - 1)]
                            )


        return gradient

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)


    @staticmethod
    @njit
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

    @staticmethod
    @njit
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

    @staticmethod
    @njit
    def set_stride(layer, stride):
        
        transposed_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                stride[0] * layer.shape[2] - (stride[0] - 1),
                stride[1] * layer.shape[3] - (stride[1] - 1),
            ),
            dtype=layer.dtype,
        )
        
        transposed_layer[:, :, ::stride[0], ::stride[1]] = layer

        return transposed_layer

    @staticmethod
    @njit
    def remove_stride(layer, stride):
        # losses[k] = losses[k][:,::self.topology[k+1]['stride'], ::self.topology[k+1]['stride']]
        untransposed_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                (layer.shape[2] + (stride[0] - 1)) // stride[0],
                (layer.shape[3] + (stride[1] - 1)) // stride[1],
            ),
            dtype=layer.dtype,
        )
        untransposed_layer = layer[:, :, ::stride[0], ::stride[1]]

        return untransposed_layer
 

