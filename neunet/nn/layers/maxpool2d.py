from typing import Any, Optional, Union

import cupy as cp
import numpy as np

from neunet.autograd import Tensor
from neunet.nn.modules import Module


class _MaxPool2dTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(
                X: Tensor,
                stride,
                padding,
                input_size,
                output_size,
                dilated_kernel_size,
                windows,
                O_einsum,
                grad
            ):

            # grad_X = np.where(O_args == 1, grad[..., None, None], 0)
            batch_size, in_channels, in_height, in_width = input_size
            # X_data = set_padding(X.data, padding, value=-np.inf)

            # TODO: vectorize this
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

            grad_col = X.xp.zeros_like(windows, dtype=grad.dtype)
            grad_col[X.xp.arange(grad_col.shape[0]), np.nanargmax(windows, axis=1)] = grad.reshape(
                -1
            )
            grad_col = grad_col.reshape(
                batch_size,
                in_channels,
                output_size[0],
                output_size[1],
                dilated_kernel_size[0],
                dilated_kernel_size[0],
            )

            grad_X = X.xp.zeros(
                (
                    batch_size,
                    in_channels,
                    in_height + 2 * padding[0],
                    in_width + 2 * padding[1],
                ),
                dtype=grad.dtype,
            )
            for i in range(output_size[0]):
                for j in range(output_size[1]):
                    grad_X[
                        :,
                        :,
                        i * stride[0] : i * stride[0] + dilated_kernel_size[0],
                        j * stride[1] : j * stride[1] + dilated_kernel_size[1],
                    ] += grad_col[:, :, i, j, :, :]

            grad_X = remove_padding(grad_X, padding)

            X.apply_grad(grad_X)

        self.grad_fn = grad_fn


class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple[int, int]], stride: Optional[Union[int, tuple[int, int]]]=None, padding: Union[int, tuple[int, int]]=0, dilation: Union[int, tuple[int, int]]=1):
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride)
            if stride
            else self.kernel_size
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.input_size: Any = None

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

        self.output_height = (
            self.input_height
            + self.padding[0]
            + self.padding[1]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        self.output_width = (
            self.input_width
            + self.padding[2]
            + self.padding[3]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        self.output_size = (self.output_height, self.output_width)

        self.dilated_kernel_height = self.dilation[0] * (self.kernel_height - 1) + 1
        self.dilated_kernel_width = self.dilation[1] * (self.kernel_width - 1) + 1
        self.dilated_kernel_size = (
            self.dilated_kernel_height,
            self.dilated_kernel_width,
        )

        self.kernel = set_dilation_stride(
            np.ones((self.kernel_height, self.kernel_width), dtype=self.input_dtype),
            self.dilation,
            value=np.nan,
        )

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")

        self.input_size = X.shape
        self.input_dtype = X.dtype
        self.build()

        X_data = set_padding(X.data, self.padding, value=-np.inf)

        batch_size, in_channels, _, _ = X_data.shape

        batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
        windows = X.xp.lib.stride_tricks.as_strided(
            X_data,
            (
                batch_size,
                in_channels,
                self.output_size[0],
                self.output_size[1],
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

        O_einsum = X.xp.einsum("bihwkl,oikl->bihwkl", windows, self.kernel[None, None, ...])
        # O_args = np.where(O_einsum == np.nanmax(O_einsum, axis=(4, 5))[..., None, None], 1, 0)
        # O_argw = np.argwhere(O_einsum == np.nanmax(O_einsum, axis=(4, 5))[..., None, None])
        O = X.xp.nanmax(O_einsum, axis=(4, 5))  # np.amax(windows, axis=(4, 5))

        # remove lines where first 4 elements are the same
        # new_O_argw = []
        # for i in range(len(O_argw)):
        #     if i == 0:
        #         new_O_argw.append(O_argw[i])
        #     elif np.all(O_argw[i][:4] == O_argw[i-1][:4]):
        #         pass
        #     else:
        #         new_O_argw.append(O_argw[i])
        # new_O_argw = np.array(new_O_argw)

        return _MaxPool2dTensor(
            O,
            [
                X,
                self.stride,
                self.padding,
                self.input_size,
                self.output_size,
                self.dilated_kernel_size,
                windows,
                O_einsum,
            ],
            "maxpool2d",
            device=X.device,
        )

    def __call__(self, X):
        return self.forward(X)


def set_dilation_stride(array, stride, value=0):
    # New shape: (_, _, S[0] * H - (S[0] - 1), S[1] * W - (S[1] - 1)
    xp = np if isinstance(array, np.ndarray) else cp
    strided_layer = xp.full(
        (
            stride[0] * array.shape[0] - (stride[0] - 1),
            stride[1] * array.shape[1] - (stride[1] - 1),
        ),
        value,
        dtype=array.dtype,
    )

    strided_layer[:: stride[0], :: stride[1]] = array

    return strided_layer


def set_padding(array, padding, value=0):
    # New shape: (_, _, H + P[0] + P[1], W + P[2] + P[3])
    xp = np if isinstance(array, np.ndarray) else cp
    return xp.pad(
        array,
        ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])),
        constant_values=value,
    )


def remove_padding(array, padding):
    # New shape: (_, _, H - P[0] - P[1], W - P[2] - P[3])
    return array[
        :,
        :,
        padding[0] : array.shape[2] - padding[1],
        padding[2] : array.shape[3] - padding[3],
    ]
