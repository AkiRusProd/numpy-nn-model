import ctypes
from ctypes import POINTER, c_float, c_size_t
from typing import Literal, Union

import cupy as cp
import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter
from neunet.nn.experimental.utils import (
    load_dlls,
    get_module_path,
    CUDA_LINEAR_MODULE,
    CUDA_CUTLASS_LINEAR_MODULE,
    load_cuda_function,
    call_cuda_function
)

load_dlls()

ArrayLike = Union[np.ndarray, cp.ndarray]

_BACKEND_MODULES = {
    "cublaslt": CUDA_LINEAR_MODULE,
    "cutlass": CUDA_CUTLASS_LINEAR_MODULE,
}

_BACKEND_CACHE: dict[str, tuple[ctypes._CFuncPtr, ctypes._CFuncPtr]] = {}

def _get_backend_functions(backend: str):
    if backend not in _BACKEND_CACHE:
        module = _BACKEND_MODULES.get(backend)
        if module is None:
            raise ValueError(f"Unknown backend: {backend}")
        dll_path = get_module_path(module)
        forward = load_cuda_function(
            dll_path,
            "cudaLinearModuleForward",
            [
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                c_size_t,
                c_size_t,
                c_size_t,
            ],
        )
        backward = load_cuda_function(
            dll_path,
            "cudaLinearModuleBackward",
            [
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                POINTER(c_float),
                c_size_t,
                c_size_t,
                c_size_t,
            ],
        )
        _BACKEND_CACHE[backend] = (forward, backward)
    return _BACKEND_CACHE[backend]



def cuda_linear_module_forward(
    X: ArrayLike,
    weights: ArrayLike,
    bias: ArrayLike,
    O: ArrayLike,
    input_rows: int,
    input_cols: int,
    output_cols: int,
    forward_fn=None,
):
    if forward_fn is None:
        forward_fn, _ = _get_backend_functions("cublaslt")
    return call_cuda_function(
        forward_fn, 
        X, 
        weights, 
        bias, 
        O, 
        input_rows, 
        input_cols, 
        output_cols
    )


def cuda_linear_module_backward(
    X: ArrayLike,
    weights: ArrayLike,
    grad_O: ArrayLike,
    grad_X: ArrayLike,
    grad_weight: ArrayLike,
    grad_bias: ArrayLike,
    input_rows: int,
    input_cols: int,
    output_cols: int,
    backward_fn=None,
):
    if backward_fn is None:
        _, backward_fn = _get_backend_functions("cublaslt")
    return call_cuda_function(
        backward_fn,
        X,
        weights,
        grad_O,
        grad_X,
        grad_weight,
        grad_bias,
        input_rows,
        input_cols,
        output_cols,
    )

class _CUDALinearTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(
            X: Tensor,
            weight: Tensor,
            bias: Tensor,
            in_rows_num,
            in_features,
            out_features,
            backward_fn,
            grad,
        ):
            grad_X = X.xp.empty_like(X.data, dtype=X.xp.float32)
            grad_weight = X.xp.empty_like(weight.data, dtype=X.xp.float32)
            grad_bias = X.xp.empty_like(bias.data, dtype=X.xp.float32) if bias is not None else None

            cuda_linear_module_backward(
                X.data, weight.data, grad, grad_X,
                grad_weight, grad_bias,
                in_rows_num, in_features, out_features,
                backward_fn=backward_fn,
            )

            X.apply_grad(grad_X)
            weight.apply_grad(grad_weight)
            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class CUDALinear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        device: Literal["cuda"] = "cuda",
        backend: Literal["cublaslt", "cutlass"] = "cublaslt",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self._forward_fn, self._backward_fn = _get_backend_functions(backend)

        # Initialize weights and bias
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
        # Allocate output tensor
        if X.device != self.device:
            raise ValueError(f"Input tensor must be on {self.device}")
        
        output_shape = X.shape[:-1] + (self.out_features,)
        output = X.xp.empty(output_shape, dtype=X.xp.float32)

        # Compute forward pass
        input_rows = int(np.prod(X.shape[:-1]))
        cuda_linear_module_forward(
            X.data, self.weight.data, self.bias.data if self.bias is not None else None, output,
            input_rows, self.in_features, self.out_features,
            forward_fn=self._forward_fn,
        )

        return _CUDALinearTensor(
            output,
            (
                X,
                self.weight,
                self.bias,
                input_rows,
                self.in_features,
                self.out_features,
                self._backward_fn,
            ),
            "linear", device=self.device
        )

    def __call__(self, X):
        return self.forward(X)
