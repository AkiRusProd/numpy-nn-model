from typing import Union

import cupy as cp
import numpy as np

from neunet.autograd import Tensor
from neunet.nn.modules import Module


class ZeroPad2dTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def _backward(X: Tensor, padding, grad):

            if X.data.ndim == 3:
                unpadded_grad = remove_padding(grad.reshape(1, *grad.shape), padding)[0]
            else:
                unpadded_grad = remove_padding(grad, padding)

            X.apply_grad(unpadded_grad)

        self._backward = _backward


class ZeroPad2d(Module):
    def __init__(self, padding: Union[int, tuple[int]]):
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding = (
            (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
            if len(self.padding) == 2
            else self.padding
        )

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.ndim != 3 and X.ndim != 4:
            raise ValueError("X must be 3D or 4D tensor")

        X.data = X.data

        if X.data.ndim == 3:
            padded_data = set_padding(X.data.reshape(1, *X.data.shape), self.padding)[0]
        else:
            padded_data = set_padding(X.data, self.padding)

        return ZeroPad2dTensor(padded_data, [X, self.padding], "zeropad2d", device=X.device)

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