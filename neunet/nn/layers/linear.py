from typing import Literal, Union

import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter

# Y = X matmul W.T + b


class _LinearTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, weight: Tensor, bias: Union[Tensor, None], grad):
            # X, weight, bias = self.args
            X.apply_grad(X.xp.matmul(grad, weight.data))
            weight.apply_grad(
                X.xp.matmul(X.data.swapaxes(-1, -2), grad).swapaxes(-1, -2)
            )
            if bias is not None:
                bias.apply_grad(X.xp.sum(grad, axis=0, keepdims=True))

        self.grad_fn = grad_fn


class Linear(Module):  # layer with static backpropagation
    def __init__(self, in_features: int, out_features: int, bias: bool=True, device: Literal["cpu", "cuda"] = "cpu"):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (out_features, in_features)),
                dtype=np.float32,
            )
        )

        if bias == True:
            self.bias: Union[Tensor, None] = Parameter(neunet.tensor(np.random.uniform(-stdv, stdv, (1, out_features)), dtype=np.float32))
        else:
            self.bias = None
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        O = self.xp.matmul(X.data, self.weight.data.T)
        if self.bias is not None:
            O = O + self.bias.data

        return _LinearTensor(O, (X, self.weight, self.bias), "linear", device=self.device)

    def __call__(self, X):
        return self.forward(X)


# class Linear(): # layer with dynamic backpropagation
#     def __init__(self, in_features, out_features, bias = True):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)), dtype=np.float32)

#         if bias == True:
#             self.bias = Tensor(np.zeros((1, out_features)), dtype=np.float32)
#         else:
#             self.bias = None

#     def forward(self, X):
#         O = X.matmul(self.weight.T)

#         if self.bias is not None:
#             O = O.add(self.bias)

#         return O

#     def __call__(self, X):
#         return self.forward(X)


# class LinearTensor(Tensor):
#     def __init__(self, data, args, op):
#         super().__init__(data, args, op)


#     def backward(self, grad=1):
#         # return super().backward(grad)

#         self.args[0].backward(np.matmul(grad, self.args[1].data.swapaxes(-1, -2)))
#         self.args[1].backward(np.matmul(self.args[0].data.swapaxes(-1, -2), grad))
#         self.args[2].backward(np.sum(grad, axis = 0, keepdims = True))


# class Linear():
#     def __init__(self, in_features, out_features, bias = True):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         self.weight = Tensor(np.random.uniform(-stdv, stdv, (in_features, out_features)), dtype=np.float32)
#         self.bias = Tensor(np.zeros((1, out_features)), requires_grad = bias, dtype=np.float32)
#         # self.weight = Tensor(np.random.normal(0, pow(out_features, -0.5), (in_features, out_features)), dtype=np.float32)
#         # self.bias = Tensor(np.zeros((1, out_features)), dtype=np.float32)

#     def forward(self, X):
#         self.X = X

#         self.O = np.matmul(self.X.data, self.weight.data) + self.bias.data

#         return LinearTensor(self.O, [self.X, self.weight, self.bias], "linear")

#     def __call__(self, X):

#         return self.forward(X)
