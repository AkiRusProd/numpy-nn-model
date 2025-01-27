from typing import Any, Literal, Union

import cupy as cp
import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class _Conv2dTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(
                X: Tensor,
                weight: Tensor,
                bias: Tensor,
                out_channels,
                padding,
                stride,
                dilation,
                prepared_input_size,
                stride_compared_input_size,
                conv_size,
                dilated_kernel_size,
                windows,
                grad
            ):

            batch_size, _, in_height, in_width = X.shape
            input_size = (in_height, in_width)

            grad_pattern = X.xp.zeros(
                (
                    batch_size,
                    out_channels,
                    stride[0] * conv_size[0] - (stride[0] - 1) + 2 * (dilated_kernel_size[0] - 1),
                    stride[1] * conv_size[1] - (stride[1] - 1) + 2 * (dilated_kernel_size[1] - 1),
                ),
                dtype=grad.dtype,
            )

            temp_grad = X.xp.zeros(
                (
                    batch_size,
                    out_channels,
                    stride[0] * conv_size[0] - (stride[0] - 1),
                    stride[1] * conv_size[1] - (stride[1] - 1),
                ),
                dtype=grad.dtype,
            )

            temp_grad[:, :, :: stride[0], :: stride[1]] = grad

            grad_pattern[
                :,
                :,
                dilated_kernel_size[0] - 1 : stride[0] * conv_size[0]
                - (stride[0] - 1)
                + dilated_kernel_size[0]
                - 1,
                dilated_kernel_size[1] - 1 : stride[1] * conv_size[1]
                - (stride[1] - 1)
                + dilated_kernel_size[1]
                - 1,
            ] = temp_grad

            batch_str, channel_str, kern_h_str, kern_w_str = grad_pattern.strides
            grad_windows = X.xp.lib.stride_tricks.as_strided(
                grad_pattern,
                (
                    batch_size,
                    out_channels,
                    prepared_input_size[0],
                    prepared_input_size[1],
                    dilated_kernel_size[0],
                    dilated_kernel_size[1],
                ),
                (
                    batch_str,
                    channel_str,
                    1 * kern_h_str,
                    1 * kern_w_str,
                    kern_h_str,
                    kern_w_str,
                ),
            )

            weight_rot_180 = X.xp.rot90(weight.data, 2, axes=(2, 3))

            grad_weight = X.xp.einsum("bihwkl,bohw->oikl", windows, grad)
            grad_bias = X.xp.sum(grad, axis=(0, 2, 3))

            grad_X = X.xp.einsum("bohwkl,oikl->bihw", grad_windows, weight_rot_180)
            grad_X = set_padding(
                grad_X,
                (
                    0,
                    input_size[0] - stride_compared_input_size[0],
                    0,
                    input_size[1] - stride_compared_input_size[1],
                ),
            )
            grad_X = remove_padding(grad_X, padding)

            weight.data = remove_stride(weight.data, dilation)
            grad_weight = remove_stride(grad_weight, dilation)

            X.apply_grad(grad_X)
            weight.apply_grad(grad_weight)

            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class Conv2d(Module):  # layer with static backpropagation
    """
    Add 2d convolutional layer
    --------------------------
        Args:
            `in_channels`: number of input channels
            `out_channels`: number of kernels
            `kernel_size` (tuple) of size 2 or (int): height and width of kernel
            `padding` (tuple) of size 2 or (int)  or `"same"`, `"real same"`, `"valid"` string value: the padding of the input window

            {
                `"valid"`: padding is 0
                `"same"`: keras "same" implementation, that returns the output of size "input_size + stride_size"
                `"real same"`: my "same" implementation, that returns the output of size "input_size"
            }
            `stride` (tuple) of size 2 or (int): the stride of the sliding kernel
            `dilation` (tuple) of size 2 or (int): the dilation of the sliding kernel
            `bias` (bool):  `True` if used. `False` if not used

        Returns:
            output: output_array (numpy.ndarray): the output array with shape: (batch_size, channels_num, conv_height, conv_width)
        References:
            https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

            https://blog.ca.meron.dev/Vectorized-CNN/

            https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]]=(1, 1),
        padding: Union[int, tuple[int, int]]=(0, 0),
        dilation: Union[int, tuple[int, int]]=(1, 1),
        bias: bool = True, 
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        stdv = 1.0 / np.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])

        self.weight = Parameter(
            neunet.tensor(
                np.random.uniform(
                    -stdv,
                    stdv,
                    (
                        self.out_channels,
                        self.in_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    ),
                ),
                dtype=np.float32,
            )
        )
        if bias == True:
            self.bias: Union[Tensor, None] =  Parameter(neunet.tensor(np.zeros(self.out_channels), dtype=np.float32))
        else:
            self.bias = None

        self.input_size: Any = None
        self.to(device)

    def build(self):
        self.kernel_height, self.kernel_width = self.kernel_size
        self.input_height, self.input_width = self.input_size[2:]

        if self.padding == "valid":
            self.padding = (0, 0, 0, 0)
        elif self.padding == "same" or self.padding == "real same":
            if self.padding == "same":
                padding_up_down = self.dilation[0] * (self.kernel_height - 1) - self.stride[0] + 1
                padding_left_right = self.dilation[1] * (self.kernel_width - 1) - self.stride[1] + 1
            elif self.padding == "real same":
                padding_up_down = (self.stride[0] - 1) * (self.input_height - 1) + self.dilation[
                    0
                ] * (self.kernel_height - 1)
                padding_left_right = (self.stride[1] - 1) * (self.input_width - 1) + self.dilation[
                    1
                ] * (self.kernel_width - 1)

            if padding_up_down % 2 == 0:
                padding_up, padding_down = padding_up_down // 2, padding_up_down // 2
            else:
                padding_up, padding_down = (
                    padding_up_down // 2,
                    padding_up_down - padding_up_down // 2,
                )

            if padding_left_right % 2 == 0:
                padding_left, padding_right = (
                    padding_left_right // 2,
                    padding_left_right // 2,
                )
            else:
                padding_left, padding_right = (
                    padding_left_right // 2,
                    padding_left_right - padding_left_right // 2,
                )

            self.padding = (
                abs(padding_up),
                abs(padding_down),
                abs(padding_left),
                abs(padding_right),
            )

        elif len(self.padding) == 2:
            self.padding = (
                self.padding[0],
                self.padding[0],
                self.padding[1],
                self.padding[1],
            )  # (up, down, left, right) padding ≃ (2 * vertical, 2 *horizontal) padding

        self.conv_height = (
            self.input_height
            + self.padding[0]
            + self.padding[1]
            - self.dilation[0] * (self.kernel_height - 1)
            - 1
        ) // self.stride[0] + 1
        self.conv_width = (
            self.input_width
            + self.padding[2]
            + self.padding[3]
            - self.dilation[1] * (self.kernel_width - 1)
            - 1
        ) // self.stride[1] + 1
        self.conv_size = (self.conv_height, self.conv_width)

        self.dilated_kernel_height = self.dilation[0] * (self.kernel_height - 1) + 1
        self.dilated_kernel_width = self.dilation[1] * (self.kernel_width - 1) + 1
        self.dilated_kernel_size = (
            self.dilated_kernel_height,
            self.dilated_kernel_width,
        )

        # input height and width for comparing with stride
        self.stride_compared_input_height = (
            (self.conv_height - 1) * self.stride[0]
            - self.padding[0]
            - self.padding[1]
            + self.dilated_kernel_height
        )
        self.stride_compared_input_width = (
            (self.conv_width - 1) * self.stride[1]
            - self.padding[2]
            - self.padding[3]
            + self.dilated_kernel_width
        )
        self.stride_compared_input_size = (
            self.stride_compared_input_height,
            self.stride_compared_input_width,
        )

        self.prepared_input_height = (
            self.stride_compared_input_height + self.padding[0] + self.padding[1]
        )
        self.prepared_input_width = (
            self.stride_compared_input_width + self.padding[2] + self.padding[3]
        )
        self.prepared_input_size = (
            self.prepared_input_height,
            self.prepared_input_width,
        )

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        self.input_size = X.shape
        self.build()

        X_data = set_padding(X.data, self.padding)
        self.weight.data = set_stride(self.weight.data, self.dilation)

        batch_size = len(X_data)

        batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
        windows = self.xp.lib.stride_tricks.as_strided(
            X_data,
            (
                batch_size,
                self.in_channels,
                self.conv_size[0],
                self.conv_size[1],
                self.dilated_kernel_size[0],
                self.dilated_kernel_size[1],
            ),
            (
                batch_str,
                channel_str,
                self.stride[0] * kern_h_str,
                self.stride[1] * kern_w_str,
                kern_h_str,
                kern_w_str,
            ),
        )

        O = self.xp.einsum("bihwkl,oikl->bohw", windows, self.weight.data)

        if self.bias is not None:
            O += self.bias.data[None, :, None, None]

        return _Conv2dTensor(
            O,
            (
                X,
                self.weight,
                self.bias,
                self.out_channels,
                self.padding,
                self.stride,
                self.dilation,
                self.prepared_input_size,
                self.stride_compared_input_size,
                self.conv_size,
                self.dilated_kernel_size,
                windows,
            ),
            "conv2d",
            self.device,
        )

    def __call__(self, X):
        return self.forward(X)


def set_padding(array, padding):
    # New shape: (_, _, H + P[0] + P[1], W + P[2] + P[3])
    xp = np if isinstance(array, np.ndarray) else cp
    return xp.pad(
        array,
        ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])),
        constant_values=0,
    )


def remove_padding(array, padding):
    # New shape: (_, _, H - P[0] - P[1], W - P[2] - P[3])
    return array[
        :,
        :,
        padding[0] : array.shape[2] - padding[1],
        padding[2] : array.shape[3] - padding[3],
    ]


def set_stride(array, stride):
    # New shape: (_, _, S[0] * H - (S[0] - 1), S[1] * W - (S[1] - 1)
    xp = np if isinstance(array, np.ndarray) else cp
    strided_array = xp.zeros(
        (
            array.shape[0],
            array.shape[1],
            stride[0] * array.shape[2] - (stride[0] - 1),
            stride[1] * array.shape[3] - (stride[1] - 1),
        ),
        dtype=array.dtype,
    )

    strided_array[:, :, :: stride[0], :: stride[1]] = array

    return strided_array


def remove_stride(array, stride):
    # New shape: (_, _, (H + S[0] - 1) // S[0], (W + S[1] - 1) // S[1])
    return array[:, :, :: stride[0], :: stride[1]]
