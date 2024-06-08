import cupy as cp
import numpy as np

from neunet.autograd import Tensor
from neunet.nn.modules import Module


class _AvgPool2dTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

    def backward(self, grad):
        (
            X,
            kernel_size,
            stride,
            padding,
            input_size,
            output_size,
        ) = self.args

        batch_size, in_channels, in_height, in_width = input_size

        grad_X = self.xp.zeros(
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
                    i * stride[0] : i * stride[0] + kernel_size[0],
                    j * stride[1] : j * stride[1] + kernel_size[1],
                ] += grad[:, :, i, j, None, None] / (kernel_size[0] * kernel_size[1])

        grad_X = remove_padding(grad_X, padding)

        X.backward(grad_X)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
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

        self.input_size = None

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
            self.input_height + self.padding[0] + self.padding[1] - self.kernel_size[0]
        ) // self.stride[0] + 1
        self.output_width = (
            self.input_width + self.padding[2] + self.padding[3] - self.kernel_size[1]
        ) // self.stride[1] + 1
        self.output_size = (self.output_height, self.output_width)

    def forward(self, X: Tensor):
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")

        self.input_size = X.shape
        self.build()

        X_data = set_padding(X.data, self.padding)

        batch_size, in_channels, _, _ = X_data.shape

        batch_str, channel_str, kern_h_str, kern_w_str = X_data.strides
        windows = X.xp.lib.stride_tricks.as_strided(
            X_data,
            (
                batch_size,
                in_channels,
                self.output_size[0],
                self.output_size[1],
                self.kernel_size[0],
                self.kernel_size[1],
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

        O = X.xp.mean(windows, axis=(4, 5))

        return _AvgPool2dTensor(
            O,
            [
                X,
                self.kernel_size,
                self.stride,
                self.padding,
                self.input_size,
                self.output_size,
            ],
            "maxpool2d",
            device=X.device,
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
