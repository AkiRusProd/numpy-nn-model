import ctypes
from ctypes import c_int

import cupy as cp
import neunet.nn as nn
from neunet.autograd import Tensor

from neunet.nn.experimental.utils import (
    CUDA_FUSED_SWISH_AND_MUL_MODULE,
    call_cuda_function,
    get_current_stream_ptr,
    get_module_path,
    load_cuda_function,
    load_dlls,
)

load_dlls()

CUDA_FUSED_SWISH_AND_MUL_DLL = get_module_path(CUDA_FUSED_SWISH_AND_MUL_MODULE)

CUDA_FUSED_SWISH_AND_MUL_FORWARD = load_cuda_function(
    CUDA_FUSED_SWISH_AND_MUL_DLL,
    "cudaFusedSwishAndMul",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        c_int,
        c_int,
        ctypes.c_void_p,
    ],
)

CUDA_FUSED_SWISH_AND_MUL_BACKWARD = load_cuda_function(
    CUDA_FUSED_SWISH_AND_MUL_DLL,
    "cudaFusedSwishAndMulBackward",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        c_int,
        c_int,
        ctypes.c_void_p,
    ],
)


def cuda_fused_swish_and_mul(
    x: cp.ndarray,
    out: cp.ndarray,
    hidden_size: int | None = None,
    beta: float = 1.0,
):
    if not all([isinstance(arg, cp.ndarray) for arg in [x, out]]):
        raise ValueError("All arguments must be cupy arrays.")
    if x.ndim < 1:
        raise ValueError("Input must have at least 1 dimension.")
    if hidden_size is None:
        hidden_size = out.shape[-1]
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0.")
    if x.shape[-1] != hidden_size * 2:
        raise ValueError("Input last dimension must be exactly 2 * hidden_size.")
    if out.shape != x.shape[:-1] + (hidden_size,):
        raise ValueError("Output shape must be input.shape[:-1] + (hidden_size,).")
    if x.dtype != cp.float32 or out.dtype != cp.float32:
        raise NotImplementedError("Only float32 is supported.")

    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    if not out.flags.c_contiguous:
        out = cp.ascontiguousarray(out)

    size = out.size
    stream_ptr = get_current_stream_ptr()

    call_cuda_function(
        CUDA_FUSED_SWISH_AND_MUL_FORWARD,
        out,
        x,
        beta,
        hidden_size,
        size,
        stream_ptr,
    )
    return out


def cuda_fused_swish_and_mul_backward(
    grad_input: cp.ndarray,
    grad_output: cp.ndarray,
    x: cp.ndarray,
    hidden_size: int | None = None,
    beta: float = 1.0,
):
    if not all([isinstance(arg, cp.ndarray) for arg in [grad_input, grad_output, x]]):
        raise ValueError("All arguments must be cupy arrays.")
    if x.ndim < 1:
        raise ValueError("Input must have at least 1 dimension.")
    if hidden_size is None:
        hidden_size = grad_output.shape[-1]
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0.")
    if x.shape[-1] != hidden_size * 2:
        raise ValueError("Input last dimension must be exactly 2 * hidden_size.")
    if grad_output.shape != x.shape[:-1] + (hidden_size,):
        raise ValueError("grad_output shape must be input.shape[:-1] + (hidden_size,).")
    if grad_input.shape != x.shape:
        raise ValueError("grad_input shape must match input shape.")
    if x.dtype != cp.float32 or grad_output.dtype != cp.float32 or grad_input.dtype != cp.float32:
        raise NotImplementedError("Only float32 is supported.")

    if not grad_output.flags.c_contiguous:
        grad_output = cp.ascontiguousarray(grad_output)
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    if not grad_input.flags.c_contiguous:
        grad_input = cp.ascontiguousarray(grad_input)

    size = grad_output.size
    stream_ptr = get_current_stream_ptr()

    call_cuda_function(
        CUDA_FUSED_SWISH_AND_MUL_BACKWARD,
        grad_input,
        grad_output,
        x,
        beta,
        hidden_size,
        size,
        stream_ptr,
    )
    return grad_input


class _CUDAFusedSwishAndMulTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(x: Tensor, beta: float, grad):
            if not all(arr.dtype == "float32" for arr in [x, grad]):
                raise NotImplementedError("Only float32 is supported.")

            grad_input = x.xp.empty_like(x.data)
            cuda_fused_swish_and_mul_backward(
                grad_input, grad, x.data, beta=beta
            )
            x.apply_grad(grad_input)

        self.grad_fn = grad_fn


class CUDAFusedSwishAndMul(nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CUDAFusedSwishAndMul, self).__init__()
        self.beta = beta

    def forward(self, x: Tensor):
        if x.dtype != "float32":
            raise NotImplementedError(f"Only float32 is supported, got {x.dtype} instead.")
        if x.device != "cuda":
            raise NotImplementedError(f"Only CUDA is supported with cupy backend, got {x.device} instead.")
        if x.ndim < 1:
            raise ValueError("Input must have at least 1 dimension.")

        if x.shape[-1] % 2 != 0:
            raise ValueError("Input last dimension must be divisible by 2.")
        hidden_size = x.shape[-1] // 2

        out = x.xp.empty(x.shape[:-1] + (hidden_size,), dtype=x.xp.float32)
        cuda_fused_swish_and_mul(x.data, out, beta=self.beta)

        return _CUDAFusedSwishAndMulTensor(
            out,
            [x, self.beta],
            "fused_swish_and_mul",
            device=x.device,
        )
