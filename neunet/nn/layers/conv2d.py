from neunet.autograd import Tensor
import numpy as np
import cupy as cp

# from numba import njit

class _Conv2dTensor(Tensor): #tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device = device)
 
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
            prepared_input_size, 
            stride_compared_input_size, 
            conv_size, 
            dilated_kernel_size, 
            windows 
        ) = self.args

        batch_size, in_channels, in_height, in_width = X.shape
        input_size = (in_height, in_width)

        grad_pattern = self.xp.zeros((
                        batch_size,
                        out_channels, 
                        stride[0] * conv_size[0] - (stride[0] - 1) +  2 * (dilated_kernel_size[0] - 1),     
                        stride[1] * conv_size[1] - (stride[1] - 1) +  2 * (dilated_kernel_size[1] - 1),      
                        ))

        temp_grad = self.xp.zeros(
            (   batch_size,
                out_channels, 
                stride[0] * conv_size[0] - (stride[0] - 1),
                stride[1] * conv_size[1] - (stride[1] - 1),
            )
        )

        temp_grad[:, :, ::stride[0], ::stride[1]]  = grad

        grad_pattern[
                    :,
                    :,
                    dilated_kernel_size[0] - 1 : stride[0] * conv_size[0] - (stride[0] - 1) + dilated_kernel_size[0] - 1,
                    dilated_kernel_size[1] - 1 : stride[1] * conv_size[1] - (stride[1] - 1) + dilated_kernel_size[1] - 1,
                ] = temp_grad

        
        batch_str, channel_str, kern_h_str, kern_w_str = grad_pattern.strides
        grad_windows = self.xp.lib.stride_tricks.as_strided(grad_pattern,
            (batch_size, out_channels, prepared_input_size[0], prepared_input_size[1], dilated_kernel_size[0], dilated_kernel_size[1]),
            (batch_str, channel_str, 1 * kern_h_str, 1 * kern_w_str, kern_h_str, kern_w_str)
        )

        weight_rot_180 = self.xp.rot90(weight.data, 2, axes=(2, 3))

        grad_weight = self.xp.einsum('bihwkl,bohw->oikl', windows, grad)
        grad_bias = self.xp.sum(grad, axis=(0, 2, 3))

        grad_X = self.xp.einsum('bohwkl,oikl->bihw', grad_windows, weight_rot_180)
        grad_X = set_padding(grad_X, (0, input_size[0] - stride_compared_input_size[0], 0, input_size[1] - stride_compared_input_size[1]))
        grad_X = remove_padding(grad_X, padding)

        weight.data = remove_stride(weight.data, dilation)
        grad_weight = remove_stride(grad_weight, dilation)

        X.backward(grad_X)
        weight.backward(grad_weight)

        if bias is not None:
            bias.backward(grad_bias)





class Conv2d(): # layer with static backpropagation
    """
    Add 2d convolutional layer
    --------------------------
        Args:
            `in_channels`: number of input channels
            `out_channels`: number of kernels
            `kernel_size` (tuple), (list) of size 2 or (int): height and width of kernel 
            `padding` (tuple), (list) of size 2 or (int)  or `"same"`, `"real same"`, `"valid"` string value: the padding of the input window
            
            {
                `"valid"`: padding is 0
                `"same"`: keras "same" implementation, that returns the output of size "input_size + stride_size"
                `"real same"`: my "same" implementation, that returns the output of size "input_size"
            }
            `stride` (tuple), (list) of size 2 or (int): the stride of the sliding kernel
            `dilation` (tuple), (list) of size 2 or (int): the dilation of the sliding kernel
            `bias` (bool):  `True` if used. `False` if not used

        Returns:
            output: output_layer (numpy.ndarray): the output layer with shape: (batch_size, channels_num, conv_height, conv_width)
        References:
            https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

            https://blog.ca.meron.dev/Vectorized-CNN/
            
            https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    """


    def __init__(self, in_channels, out_channels, kernel_size, stride = (1, 1),  padding = (0, 0), dilation = (1, 1), bias = True, device = "cpu"):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding      = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride       = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation     = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        stdv = 1. / np.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])

        self.weight = Tensor(np.random.uniform(-stdv, stdv, (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])), dtype=np.float32)
        if bias == True:
            self.bias = Tensor(np.zeros(self.out_channels), dtype=np.float32)
        else:
            self.bias = None


        self.input_size = None
        self.to(device)


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


        self.conv_height = (self.input_height + self.padding[0] + self.padding[1] - self.dilation[0] * (self.kernel_height - 1) - 1) // self.stride[0]   + 1
        self.conv_width =  (self.input_width + self.padding[2] + self.padding[3] - self.dilation[1] * (self.kernel_width - 1) - 1) // self.stride[1] + 1
        self.conv_size = (self.conv_height, self.conv_width)

        self.dilated_kernel_height = self.dilation[0] * (self.kernel_height - 1) + 1
        self.dilated_kernel_width  = self.dilation[1] * (self.kernel_width - 1) + 1
        self.dilated_kernel_size = (self.dilated_kernel_height, self.dilated_kernel_width)

        #input height and width for comparing with stride
        self.stride_compared_input_height = (self.conv_height - 1) * self.stride[0] - self.padding[0] - self.padding[1] +  self.dilated_kernel_height
        self.stride_compared_input_width = (self.conv_width - 1) * self.stride[1] - self.padding[2] - self.padding[3] +  self.dilated_kernel_width
        self.stride_compared_input_size = (self.stride_compared_input_height, self.stride_compared_input_width)
    
        self.prepared_input_height = (self.stride_compared_input_height + self.padding[0] + self.padding[1])
        self.prepared_input_width = (self.stride_compared_input_width + self.padding[2] + self.padding[3])
        self.prepared_input_size = (self.prepared_input_height, self.prepared_input_width)
            

    def forward(self, X):
        # if self.input_size == None:
        self.input_size = X.shape
        self.build()

        X_data = set_padding(X.data, self.padding)
        self.weight.data = set_stride(self.weight.data, self.dilation)

        batch_size = len(X_data)


        batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
        windows = self.xp.lib.stride_tricks.as_strided(
            X_data,
            (batch_size, self.in_channels, self.conv_size[0], self.conv_size[1], self.dilated_kernel_size[0], self.dilated_kernel_size[1]),
            (batch_str, channel_str, self.stride[0] * kern_h_str, self.stride[1] * kern_w_str, kern_h_str, kern_w_str)
        )

        O = self.xp.einsum('bihwkl,oikl->bohw', windows, self.weight.data)

        if self.bias is not None:
            O += self.bias.data[None, :, None, None]

        return _Conv2dTensor(O, 
            [X, self.weight, self.bias, self.in_channels, self.out_channels, self.kernel_size, self.padding, self.stride, self.dilation, 
            self.prepared_input_size, self.stride_compared_input_size, self.conv_size, self.dilated_kernel_size, windows], "conv2d", self.device)

    def __call__(self, X):
        return self.forward(X)

    def to (self, device):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.device = device
        self.weight = self.weight.to(device)
        if self.bias:
            self.bias = self.bias.to(device)

        return self





# @njit
def set_padding(layer, padding):
    # padded_layer = np.pad(layer, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[1], padding[0])), constant_values = 0)
    xp = np if isinstance(layer, np.ndarray) else cp 
    padded_layer = xp.zeros(
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


# @njit
def remove_padding(layer, padding):
    # unpadded_layer = unpadded_layer[:, :, padding[0]:-padding[1], padding[1]:-padding[0]]
    xp = np if isinstance(layer, np.ndarray) else cp 
    unpadded_layer = xp.zeros(
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


# @njit
def set_stride(layer, stride):
    xp = np if isinstance(layer, np.ndarray) else cp 
    transposed_layer = xp.zeros(
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


# @njit
def remove_stride(layer, stride):
    # losses[k] = losses[k][:,::self.topology[k+1]['stride'], ::self.topology[k+1]['stride']]
    xp = np if isinstance(layer, np.ndarray) else cp 
    untransposed_layer = xp.zeros(
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