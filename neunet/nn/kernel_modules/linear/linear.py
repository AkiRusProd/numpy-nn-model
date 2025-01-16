import numpy as np
import cupy as cp
from ctypes import *
import ctypes
import os
from typing import Literal
import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter

# for cublas methods (my os doesn`t see this path in environment variables)
DLL_PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin'

os.add_dll_directory(DLL_PATH)

def __get_cuda_linear_layer():
    dll = ctypes.CDLL('neunet/nn/kernel_modules/linear/linear.dll', mode=ctypes.RTLD_GLOBAL)
    func = dll.cudaLinearModuleForward
    func.argtypes = [
        POINTER(c_float),  # input
        POINTER(c_float),  # weights
        POINTER(c_float),  # bias
        POINTER(c_float),  # output
        c_size_t,          # input rows
        c_size_t,          # input cols
        c_size_t           # output cols
    ]
    return func


def __get_cuda_linear_layer_backward():
    dll = ctypes.CDLL('neunet/nn/kernel_modules/linear/linear.dll', mode=ctypes.RTLD_GLOBAL)
    func = dll.cudaLinearModuleBackward
    func.argtypes = [
        POINTER(c_float),  # input
        POINTER(c_float),  # weights
        POINTER(c_float),  # d_output
        POINTER(c_float),  # d_input
        POINTER(c_float),  # d_weights
        POINTER(c_float),  # d_bias
        c_size_t,          # input rows
        c_size_t,          # input cols
        c_size_t           # output cols
    ]
    return func


__cuda_linear = __get_cuda_linear_layer()
__cuda_linear_backward = __get_cuda_linear_layer_backward()


def cuda_linear_layer(X, weights, bias, output_matrix, input_rows, input_cols, output_cols):

    input_p = ctypes.cast(X.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(X, cp.ndarray) else X.ctypes.data_as(POINTER(c_float))
    weights_p = ctypes.cast(weights.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(weights, cp.ndarray) else weights.ctypes.data_as(POINTER(c_float))
    bias_p = ctypes.cast(bias.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(bias, cp.ndarray) else bias.ctypes.data_as(POINTER(c_float))
    output_p = ctypes.cast(output_matrix.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(output_matrix, cp.ndarray) else output_matrix.ctypes.data_as(POINTER(c_float))

    __cuda_linear(input_p, weights_p, bias_p, output_p, input_rows, input_cols, output_cols)


def cuda_linear_layer_backward(X, weights, d_output, d_input, d_weights, d_bias,
                               input_rows, input_cols, output_cols):
    input_p = ctypes.cast(X.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(X, cp.ndarray) else X.ctypes.data_as(POINTER(c_float))
    weights_p = ctypes.cast(weights.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(weights, cp.ndarray) else weights.ctypes.data_as(POINTER(c_float))
    d_output_p = ctypes.cast(d_output.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(d_output, cp.ndarray) else d_output.ctypes.data_as(POINTER(c_float))
    d_input_p = ctypes.cast(d_input.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(d_input, cp.ndarray) else d_input.ctypes.data_as(POINTER(c_float))
    d_weights_p = ctypes.cast(d_weights.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(d_weights, cp.ndarray) else d_weights.ctypes.data_as(POINTER(c_float))
    d_bias_p = ctypes.cast(d_bias.data.ptr, ctypes.POINTER(ctypes.c_float)) if isinstance(d_bias, cp.ndarray) else d_bias.ctypes.data_as(POINTER(c_float))
    __cuda_linear_backward(input_p, weights_p, d_output_p, d_input_p, d_weights_p, d_bias_p,
                           input_rows, input_cols, output_cols)


class _CUDALinearTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, X_data, weight: Tensor, bias: Tensor, batch_size, in_features, out_features, grad):

            grad = grad.reshape(-1, grad.shape[-1])

            grad_X = X.xp.zeros_like(X_data, dtype=np.float32)
            grad_weight = cp.zeros_like(weight.data, dtype=np.float32)
            grad_bias = cp.zeros_like(bias.data, dtype=np.float32)

            cuda_linear_layer_backward(X_data, weight.data,
                                        grad, grad_X,
                                        grad_weight, grad_bias,
                                        X_data.shape[0], in_features, out_features)
            
            grad_X = grad_X.reshape(X.shape)
            X.apply_grad(grad_X)
            weight.apply_grad(
                grad_weight
            )
            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class CUDALinear(Module):
    def __init__(self, in_features, out_features, device: Literal["cpu", "cuda"] = "cpu"):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (out_features, in_features)),
                dtype=np.float32,
            )
        )

        bias = True # TODO: Make it optional
        if bias == True:
            self.bias: Union[Tensor, None] = Parameter(neunet.tensor(np.random.uniform(-stdv, stdv, (1, out_features)), dtype=np.float32))
        else:
            self.bias = None
        self.to(device)

        # Gradients
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, X: Tensor) -> Tensor:
        self.X = X
        batch_size = X.shape[0]

        # hack
        X_data = X.data.reshape((-1, self.in_features))

        # O = X.xp.zeros((batch_size, self.out_features), dtype=cp.float32)
        O = X.xp.zeros((np.prod(X.shape[:-1]), self.out_features), dtype=np.float32)

        cuda_linear_layer(X_data, self.weight.data, self.bias.data, O,
                            O.shape[0], self.in_features, self.out_features)
        
        O = O.reshape(X.shape[:-1] + (self.out_features,))
        return _CUDALinearTensor(O, (X, X_data, self.weight, self.bias, O.shape[0], self.in_features, self.out_features), "linear", device=self.device)

    def __call__(self, X):
        return self.forward(X)
