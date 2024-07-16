import copy as copy_object
from typing import Any, Union

from neunet.autograd import Tensor
from neunet.nn.modules import Module

# In feature can be moved to rnn layer as argument


class _BidirectionalTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def _backward(D_O: Tensor, R_O: Tensor, merge_mode, grad):

            if merge_mode == "concat":
                direct_grad, reverse_grad = D_O.xp.split(grad, 2, axis=-1)
            elif merge_mode == "sum":
                direct_grad, reverse_grad = grad, grad
            elif merge_mode == "mul":
                direct_grad, reverse_grad = grad * R_O.data, grad * D_O.data
            elif merge_mode == "avg":
                direct_grad, reverse_grad = grad / 2, grad / 2

            D_O._apply_grad(direct_grad)
            R_O._apply_grad(reverse_grad)

        self._backward = _backward


class Bidirectional(Module):
    def __init__(self, layer: Any, merge_mode: str="sum", device: str="cpu"):
        if layer.__class__.__name__ not in ["LSTM", "GRU", "RNN"]:
            raise ValueError("Bidirectional layer can only be used with LSTM, GRU or RNN layers")

        self.direct_layer = layer
        self.reverse_layer = copy_object.copy(layer)

        self.merge_mode = merge_mode
        self.merge = merge_modes[self.merge_mode]

        self.return_sequences = layer.return_sequences

        self.to(device)

    def forward(self, X: Tensor)-> Union[Tensor, tuple[Tensor, Tensor]]:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        if len(X.shape) == 2:
            X = X.reshape(1, *X.shape)

        D_O = self.direct_layer(X)
        R_O = self.reverse_layer(X.flip(1))

        if self.return_sequences == "both":
            O = (self.merge(D_O[0], R_O[0]), self.merge(D_O[1], R_O[1]))

            return (
                _BidirectionalTensor(
                    O[0],
                    [X, D_O[0], R_O[0], self.merge_mode],
                    f"bidirectional{self.direct_layer.__class__.__name__}",
                    self.device,
                ),
                _BidirectionalTensor(
                    O[1],
                    [X, D_O[1], R_O[1], self.merge_mode],
                    f"bidirectional{self.direct_layer.__class__.__name__}",
                    self.device,
                ),
            )
        else:
            O = self.merge(D_O, R_O)

            return _BidirectionalTensor(
                O,
                [D_O, R_O, self.merge_mode],
                f"bidirectional{self.direct_layer.__class__.__name__}",
                self.device,
            )

    def __call__(self, X):
        return self.forward(X)


def concat(D_O, R_O):
    xp = D_O.xp
    return xp.concatenate((D_O.data, R_O.data), axis=-1)


def sum(D_O, R_O):
    return D_O.data + R_O.data


def mul(D_O, R_O):
    return D_O.data * R_O.data


def avg(D_O, R_O):
    return (D_O.data + R_O.data) / 2


merge_modes = {"concat": concat, "sum": sum, "mul": mul, "avg": avg}
