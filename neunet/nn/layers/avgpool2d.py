from neunet.autograd import Tensor
import numpy as np
import cupy as cp



class _AvgPool2dTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device = device)

    def backward(self, grad):
        (
            X,
            kernel_size,
            stride,
            padding,
            input_size,
            output_size,
            windows,
        ) = self.args
        
        batch_size, in_channels, in_height, in_width = input_size

        grad_X = self.xp.zeros((batch_size, in_channels, in_height + 2 * padding[0], in_width + 2 * padding[1]))
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                grad_X[:, :, i*stride[0]:i*stride[0]+kernel_size[0], j*stride[1]:j*stride[1]+kernel_size[1]] += grad[:, :, i, j, None, None]/(kernel_size[0] * kernel_size[1])

        grad_X = remove_padding(grad_X, padding)

        X.backward(grad_X)



class AvgPool2d:
    
        def __init__(self, kernel_size, stride=None, padding=0):
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride) if stride else self.kernel_size
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)

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


            self.output_height = (self.input_height + self.padding[0] + self.padding[1] - self.kernel_size[0]) // self.stride[0] + 1
            self.output_width  = (self.input_width  + self.padding[2] + self.padding[3] - self.kernel_size[1]) // self.stride[1] + 1
            self.output_size = (self.output_height, self.output_width)
        
        def forward(self, X: Tensor):
            if self.input_size == None:
                self.input_size = X.shape
                self.build()
            
            X_data = set_padding(X.data, self.padding)
     
            batch_size, in_channels, in_height, in_width = X_data.shape
            
            batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
            windows = X.xp.lib.stride_tricks.as_strided(
                X_data,
                (batch_size, in_channels, self.output_size[0], self.output_size[1], self.kernel_size[0], self.kernel_size[1]),
                (batch_str, channel_str, self.stride[0] * kern_h_str, self.stride[1] * kern_w_str, kern_h_str, kern_w_str)
            )

            O = X.xp.mean(windows, axis=(4, 5))

            return _AvgPool2dTensor(O, [X, self.kernel_size, self.stride, self.padding, self.input_size, self.output_size, 
            windows], "maxpool2d", device = X.device)


        def __call__(self, X):
            return self.forward(X)




   



def set_padding(layer, padding):
    xp = np if isinstance(layer, np.ndarray) else cp 
    padded_layer = xp.zeros(
        (   
            layer.shape[0],
            layer.shape[1],
            layer.shape[2] + padding[0] + padding[1],
            layer.shape[3] + padding[2] + padding[3],
        ),
    )

    padded_layer[
                :,
                :,
                padding[0] :padded_layer.shape[2] - padding[1],
                padding[2] :padded_layer.shape[3] - padding[3],
            ] = layer

    return padded_layer


def remove_padding(layer, padding):
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