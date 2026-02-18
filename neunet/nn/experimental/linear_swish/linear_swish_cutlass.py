import ctypes
from ctypes import POINTER, c_float, c_int, c_size_t, c_void_p
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
    CUDA_LINEAR_SWISH_CUTLASS_MODULE,
    load_cuda_function,
    call_cuda_function,
    get_current_stream_ptr,
    to_pointer,
)

load_dlls()

ArrayLike = Union[np.ndarray, cp.ndarray]

CUDA_LINEAR_SWISH_CUTLASS_DLL = get_module_path(CUDA_LINEAR_SWISH_CUTLASS_MODULE)

CUDA_LINEAR_SWISH_FORWARD = load_cuda_function(
    CUDA_LINEAR_SWISH_CUTLASS_DLL,
    "cudaLinearSwishForward",
    [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        c_size_t,
        c_float,
        c_int,
        c_void_p,  # cudaStream_t
    ],
)

CUDA_LINEAR_SWISH_BACKWARD = load_cuda_function(
    CUDA_LINEAR_SWISH_CUTLASS_DLL,
    "cudaLinearSwishBackward",
    [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        c_size_t,
        c_float,
        c_int,
        c_void_p,  # cudaStream_t
    ],
)


def cuda_linear_swish_forward(
    X: ArrayLike,
    weights: ArrayLike,
    bias: Union[ArrayLike, None],
    O: ArrayLike,
    preactivation: Union[ArrayLike, None],
    input_rows: int,
    input_cols: int,
    output_cols: int,
    swish_beta: float = 1.0,
    save_preactivation: bool = False,
    stream=None,
):
    if stream is None:
        stream_ptr = c_void_p(get_current_stream_ptr())
    else:
        stream_ptr = c_void_p(stream)
    return call_cuda_function(
        CUDA_LINEAR_SWISH_FORWARD,
        X,
        weights,
        bias,
        O,
        preactivation,
        input_rows,
        input_cols,
        output_cols,
        swish_beta,
        int(save_preactivation),
        stream_ptr,
    )


def cuda_linear_swish_backward(
    X: ArrayLike,
    weights: ArrayLike,
    bias: Union[ArrayLike, None],
    grad_O: ArrayLike,
    d_linear_tmp: ArrayLike,
    grad_X: ArrayLike,
    grad_weight: ArrayLike,
    grad_bias: Union[ArrayLike, None],
    input_rows: int,
    input_cols: int,
    output_cols: int,
    swish_beta: float = 1.0,
    recompute_preactivation: bool = True,
    stream=None,
):
    # Convert stream to c_void_p (None becomes current stream)
    if stream is None:
        stream_ptr = c_void_p(get_current_stream_ptr())
    else:
        stream_ptr = c_void_p(stream)
    
    return CUDA_LINEAR_SWISH_BACKWARD(
        to_pointer(X),
        to_pointer(weights),
        to_pointer(bias),
        to_pointer(grad_O),
        to_pointer(d_linear_tmp),
        to_pointer(grad_X),
        to_pointer(grad_weight),
        to_pointer(grad_bias),
        input_rows,
        input_cols,
        output_cols,
        swish_beta,
        int(recompute_preactivation),
        stream_ptr,
    )


class _CUDALinearSwishTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(
            X: Tensor,
            weight: Tensor,
            bias: Tensor,
            in_rows_num,
            in_features,
            out_features,
            swish_beta,
            preactivation,
            save_preactivation,
            grad,
        ):
            grad_X = X.xp.empty_like(X.data, dtype=X.xp.float32)
            grad_weight = X.xp.empty_like(weight.data, dtype=X.xp.float32)
            grad_bias = (
                X.xp.empty_like(bias.data, dtype=X.xp.float32)
                if bias is not None
                else None
            )
            if save_preactivation:
                d_linear_tmp = preactivation
                recompute_preactivation = False
            else:
                # Temporary buffer for backward pass
                d_linear_tmp = X.xp.empty(
                    (in_rows_num, out_features), dtype=X.xp.float32
                )
                recompute_preactivation = True

            cuda_linear_swish_backward(
                X.data,
                weight.data,
                bias.data if bias is not None else None,
                grad,
                d_linear_tmp,
                grad_X,
                grad_weight,
                grad_bias,
                in_rows_num,
                in_features,
                out_features,
                swish_beta,
                recompute_preactivation,
            )

            X.apply_grad(grad_X)
            weight.apply_grad(grad_weight)
            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn


class CUDALinearSwish(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        swish_beta: float = 1.0,
        save_preactivation: bool = True,
        device: Literal["cuda"] = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.swish_beta = swish_beta
        self.save_preactivation = save_preactivation

        # Initialize weights and bias
        stdv = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (out_features, in_features)),
                dtype=np.float32,
            )
        )

        if bias == True:
            self.bias: Union[Tensor, None] = Parameter(
                neunet.tensor(
                    np.random.uniform(-stdv, stdv, (1, out_features)),
                    dtype=np.float32,
                )
            )
        else:
            self.bias = None
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        # Allocate output tensor
        if X.device != self.device:
            raise ValueError(f"Input tensor must be on {self.device}")

        output_shape = X.shape[:-1] + (self.out_features,)
        output = X.xp.empty(output_shape, dtype=X.xp.float32)
        preactivation = None
        if self.save_preactivation:
            preactivation = X.xp.empty(output_shape, dtype=X.xp.float32)

        # Compute forward pass
        input_rows = int(np.prod(X.shape[:-1]))
        cuda_linear_swish_forward(
            X.data,
            self.weight.data,
            self.bias.data if self.bias is not None else None,
            output,
            preactivation,
            input_rows,
            self.in_features,
            self.out_features,
            self.swish_beta,
            self.save_preactivation,
        )

        return _CUDALinearSwishTensor(
            output,
            (
                X,
                self.weight,
                self.bias,
                input_rows,
                self.in_features,
                self.out_features,
                self.swish_beta,
                preactivation,
                self.save_preactivation,
            ),
            "linear_swish",
            device=self.device,
        )

    def __call__(self, X):
        return self.forward(X)
