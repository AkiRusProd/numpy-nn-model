from autograd import Tensor
import numpy as np


# from numba import njit

class _ConvTranspose2dTensor(Tensor): #tensor for static backpropagation
    def __init__(self, data, args, op):
            super().__init__(data, args, op)
    
    def backward(self, grad = 1):
        (
            X, 
            weight, 
            bias, 
            in_channels, 
            out_channels, 
            kernel_size, 
            padding, 
            stride, 
            dilation,
            output_padding, 
            prepared_input_size, 
            conv_size, 
            dilated_kernel_size, 
            windows 
        ) = self.args

        batch_size, in_channels, in_height, in_width = X.shape
        input_size = (in_height, in_width)


        grad_pattern = np.zeros((
                batch_size,
                out_channels, 
                prepared_input_size[0] + np.max(np.array([conv_size[0], dilated_kernel_size[0]])) - 1, 
                prepared_input_size[1] + np.max(np.array([conv_size[1], dilated_kernel_size[1]])) - 1
                ))

        grad_pattern[
            :,
            :,
            dilated_kernel_size[0] - 1 : conv_size[0] + dilated_kernel_size[0] - 1,
            dilated_kernel_size[1] - 1 : conv_size[1] + dilated_kernel_size[1] - 1,
        ] = grad

        batch_str, channel_str, kern_h_str, kern_w_str = grad_pattern.strides
        grad_windows = np.lib.stride_tricks.as_strided(grad_pattern,
            (batch_size, out_channels, prepared_input_size[0], prepared_input_size[1], dilated_kernel_size[0], dilated_kernel_size[1]),
            (batch_str, channel_str, kern_h_str, kern_w_str, kern_h_str, kern_w_str)
        )

        weight_rot_180 = np.rot90(weight.data, 2, axes=(2, 3))

        grad_weight = np.einsum('bihwkl,bohw->oikl', windows, grad)
        grad_bias = np.sum(grad, axis=(0, 2, 3))

        grad_X = np.einsum('bohwkl,oikl->bihw', grad_windows, weight_rot_180)
        grad_X = self.prepare_grad(grad_X, padding, stride, dilated_kernel_size, output_padding)

        weight.data = remove_stride(weight.data, dilation)
        grad_weight = remove_stride(grad_weight, dilation)

        X.backward(grad_X)
        weight.backward(grad_weight)

        if bias is not None:
            bias.backward(grad_bias)

    def prepare_grad(self, grad, padding, stride, dilated_kernel_size, output_padding):
       
        padded_grad = set_padding(grad, padding)#ADD set padding that we removed in forward #in conv2dTranspose set padding equals remove padding

        grad = padded_grad[:, :, dilated_kernel_size[0] - 1 : padded_grad.shape[2] - (dilated_kernel_size[0] - 1) - output_padding[0], #remove kernel padding that we added
                                 dilated_kernel_size[1] - 1 : padded_grad.shape[3] - (dilated_kernel_size[1] - 1) - output_padding[1]].copy()
        
        unstrided_grad = remove_stride(grad, stride)
        
        return unstrided_grad



class ConvTranspose2d():  # layer with static backpropagation
    """
    Add 2d transposed convolutional layer
    -------------------------------------
        Args:
            `kernels_num`: number of kernels
            `kernel_shape` (tuple), (list) of size 2 or (int): height and width of kernel
            `padding` (tuple), (list) of size 2 or (int)  or `"same"`, `"real same"`, `"valid"` string value: the "inverted" padding of the input window (removing padding)
            
            {
                `"valid"`: padding is 0
                `"same"`: keras "same" implementation, that returns the output of size "input_size + stride_size"
                `"real same"`: my "same" implementation, that returns the output of size "input_size"
            }
            `stride` (tuple), (list) of size 2 or (int): the transposed stride (operation similar to dilation) of the 2d input window
            `dilation` (tuple), (list) of size 2 or (int): the dilation of the sliding kernel
            `use_bias` (bool):  `True` if used. `False` if not used

        Returns:
            output: output_layer (numpy.ndarray): the output layer with shape: (batch_size, channels_num, conv_height, conv_width)
        References:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride = (1, 1), padding = (0, 0), dilation = (1, 1), output_padding = (0, 0), bias = True):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding      = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride       = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation     = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        

        stdv = 1. / np.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])

        self.weight = Tensor(np.random.uniform(-stdv, stdv, (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])))
        if bias == True:
            self.bias = Tensor(np.zeros(self.out_channels))
        else:
            self.bias = None

        self.input_size = None

    def build(self):
        
        self.kernel_height, self.kernel_width = self.kernel_size
        self.input_height, self.input_width = self.input_size[2:]

        if self.padding == "valid":
            self.padding == (0, 0, 0, 0)
        elif self.padding == "same" or self.padding == "real same":

            if self.padding == "same":
                padding_up_down = (1 - self.stride[0]) + self.dilation[0] * (self.kernel_height - 1) + self.output_padding[0] 
                padding_left_right = (1 - self.stride[1]) + self.dilation[1] * (self.kernel_width  - 1) + self.output_padding[1]
            elif self.padding == "real same":
                padding_up_down = (self.stride[0] - 1) * (self.input_height - 1) + self.dilation[0] * (self.kernel_height - 1) + self.output_padding[0]
                padding_left_right = (self.stride[1] - 1) * (self.input_width- 1) + self.dilation[1] * (self.kernel_width  - 1) + self.output_padding[1]

            if padding_up_down % 2 == 0:
                padding_up, padding_down = padding_up_down // 2, padding_up_down // 2
            else:
                padding_up, padding_down = padding_up_down // 2, padding_up_down - padding_up_down // 2

            if padding_left_right % 2 == 0:
                padding_left, padding_right = padding_left_right // 2, padding_left_right // 2
            else:
                padding_left, padding_right = padding_left_right // 2, padding_left_right - padding_left_right // 2
    

            self.padding = (padding_up, padding_down, padding_left, padding_right)

        elif len(self.padding) == 2:
            self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1]) #(top, bottom, left, right) padding ≃ (2 * vertical, 2 *horizontal) padding
    
        
        self.conv_height = (self.input_height - 1) * self.stride[0] - (self.padding[0] + self.padding[1])  +  self.dilation[0] * (self.kernel_height - 1) + self.output_padding[0] + 1
        self.conv_width =  (self.input_width - 1) * self.stride[1] - (self.padding[2] + self.padding[3]) + self.dilation[1] * (self.kernel_width - 1) + self.output_padding[1] + 1
        self.conv_size = (self.conv_height, self.conv_width)
        
        self.dilated_kernel_height = self.dilation[0] * (self.kernel_height - 1) + 1
        self.dilated_kernel_width = self.dilation[1] * (self.kernel_width - 1) + 1
        self.dilated_kernel_size = (self.dilated_kernel_height, self.dilated_kernel_width)

        self.prepared_input_height = (self.input_height - 1) * self.stride[0] + 1 - (self.padding[0] + self.padding[1]) + self.output_padding[0] + 2 * self.dilated_kernel_height - 2
        self.prepared_input_width = (self.input_width - 1) * self.stride[1] + 1 - (self.padding[2] + self.padding[3])+ self.output_padding[1] + 2 * self.dilated_kernel_width - 2
        self.prepared_input_size = (self.prepared_input_height, self.prepared_input_width)
            
        


    def forward(self, X):
        if self.input_size == None:
            self.input_size = X.shape
            self.build()

        X_data = self.prepare_inputs(X.data)
        self.weight.data = set_stride(self.weight.data, self.dilation)

        batch_size = len(X_data)
        

        batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
        windows = np.lib.stride_tricks.as_strided(
            X_data,
            (batch_size, self.in_channels, self.conv_size[0], self.conv_size[1], self.dilated_kernel_size[0], self.dilated_kernel_size[1]),
            (batch_str, channel_str, kern_h_str, kern_w_str, kern_h_str, kern_w_str)
        )

        O = np.einsum('bihwkl,oikl->bohw', windows, self.weight.data)

        if self.bias is not None:
            O += self.bias.data[:, None, None]

        return _ConvTranspose2dTensor(O, 
            [X, self.weight, self.bias, self.in_channels, self.out_channels, self.kernel_size, self.padding, self.stride, self.dilation, self.output_padding,
            self.prepared_input_size, self.conv_size, self.dilated_kernel_size, windows], "convtranspose2d")



    def prepare_inputs(self, input_data):

        temp_strided = set_stride(input_data, self.stride) #ADD STRIDING

        #add output_padding here #WARNING output padding must be smaller than either stride or dilation,
        temp_out = np.zeros((temp_strided.shape[0], 
                                       temp_strided.shape[1], 
                                       temp_strided.shape[2] + self.output_padding[0],
                                       temp_strided.shape[3] + self.output_padding[1]))
        temp_out[:, :, : temp_strided.shape[2], : temp_strided.shape[3]] = temp_strided #ADD output_padding

        input_data = np.zeros((#add kernel padding
                        input_data.shape[0],
                        input_data.shape[1], 
                        temp_out.shape[2] + 2 * (self.dilated_kernel_size[0] - 1), 
                        temp_out.shape[3] + 2 * (self.dilated_kernel_size[1] - 1)
                        ))

        input_data[:, :, self.dilated_kernel_size[0] - 1 : temp_out.shape[2] + self.dilated_kernel_size[0] - 1, 
                         self.dilated_kernel_size[1] - 1 : temp_out.shape[3] + self.dilated_kernel_size[1] - 1] = temp_out

        input_data = remove_padding(input_data, self.padding)#ADD remove padding #in conv2dTranspose set padding equals remove padding
        return input_data

    def __call__(self, X):
        return self.forward(X)



def set_padding(layer, padding):
    # padded_layer = np.pad(layer, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[1], padding[0])), constant_values = 0)
    padded_layer = np.zeros(
        (   
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] + padding[0] + padding[1],
            layer.shape[3] + padding[2] + padding[3],
        )
    )

    padded_layer[
                :,
                :,
                padding[0] :padded_layer.shape[2] - padding[1],
                padding[2] :padded_layer.shape[3] - padding[3],
            ] = layer

    return padded_layer


def remove_padding(layer, padding):
    # unpadded_layer = unpadded_layer[:, :, padding[0]:-padding[1], padding[1]:-padding[0]]
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
