import ctypes
from ctypes import POINTER, c_float, c_int
from typing import Literal, Union

import cupy as cp
import numpy as np

import neunet
import neunet.nn as nn
from neunet.autograd import Tensor
from neunet.nn.experimental.utils import CUDA_RMSNORM_MODULE, get_module_path, load_dlls

load_dlls()

CUDA_RMSNORM_DLL = get_module_path(CUDA_RMSNORM_MODULE)

# Helper to load CUDA functions
def _load_cuda_function(module_path, function_name, arg_types):
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    return func

CUDA_RMSNORM_FORWARD = _load_cuda_function(
    CUDA_RMSNORM_DLL, "RMSNormForward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        c_int, 
        c_int,
        c_float,
        ctypes.c_void_p
    ]
)

CUDA_RMSNORM_BACKWARD = _load_cuda_function(
    CUDA_RMSNORM_DLL, "RMSNormBackward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        c_int, 
        c_int,
        ctypes.c_void_p
    ]
)  

def call_cuda_function(func, *args):
    # Helper for casting data to pointers
    def _to_pointer(obj: cp.ndarray):
        if obj is None:
            return None
        elif isinstance(obj, cp.ndarray):
            if obj.dtype == cp.float32:
                return ctypes.cast(obj.data.ptr, POINTER(c_float))
            elif obj.dtype == cp.int32:
                return ctypes.cast(obj.data.ptr, POINTER(c_int))

        return obj

    return func(*[_to_pointer(arg) for arg in args])



def rmsnorm_forward(
    X: cp.ndarray,
    weight: cp.ndarray,
    bias: cp.ndarray,
    X_norm: cp.ndarray,
    X_std: cp.ndarray,
    O: cp.ndarray,
    eps: float
):
    """
    Forward pass for RMSNorm.   
    """
    if not all([isinstance(arg, cp.ndarray) for arg in [X, weight, X_norm, X_std, O]]) \
    or (bias is not None and not isinstance(bias, cp.ndarray)):
        raise ValueError("All arguments must be cupy arrays. 'bias' can be None or a cupy array.")

    if X.shape != O.shape:
        raise ValueError("Input and output shapes must match")
    
    if not X.flags.c_contiguous:
        X = cp.ascontiguousarray(X)
    if not O.flags.c_contiguous:
        O = cp.ascontiguousarray(O)

    if not weight.flags.c_contiguous:
        weight = cp.ascontiguousarray(weight)
    if bias is not None and not bias.flags.c_contiguous:
        bias = cp.ascontiguousarray(bias)

    n_rows = np.prod(X.shape[:-1])
    n_cols = X.shape[-1]


    call_cuda_function(
        CUDA_RMSNORM_FORWARD,
        X, weight, bias, 
        O, X_norm, X_std,
        n_rows, n_cols, eps, None
    )

    return O, X_norm, X_std


def rmsnorm_backward(
    X: cp.ndarray,
    weight: cp.ndarray,
    bias: cp.ndarray,
    grad_O: cp.ndarray,
    grad_X: cp.ndarray,
    grad_weight: cp.ndarray,
    grad_bias: cp.ndarray,
    X_norm: cp.ndarray,
    X_std: cp.ndarray,
):    
    """
    Backward pass for RMSNorm.   
    """
    if not all([isinstance(arg, cp.ndarray) for arg in [X, weight, grad_O, grad_X, grad_weight, X_norm, X_std]]) \
    or (grad_bias is not None and not isinstance(grad_bias, cp.ndarray)):
        raise ValueError("All arguments must be cupy arrays. 'grad_bias' can be None or a cupy array.")

    if X.shape != grad_O.shape:
        raise ValueError("Input and output shapes must match")
    
    if grad_X.shape != X.shape:
        raise ValueError("Input and output gradients shapes must match")
    if grad_weight.shape != weight.shape:
        raise ValueError("Weight and weight gradient shapes must match")
    if grad_bias is not None and grad_bias.shape != bias.shape:
        raise ValueError("Bias and bias gradient shapes must match")
    
    if not X.flags.c_contiguous:
        X = cp.ascontiguousarray(X)
    if not grad_O.flags.c_contiguous:
        grad_O = cp.ascontiguousarray(grad_O)   

    if not grad_X.flags.c_contiguous:
        grad_X = cp.ascontiguousarray(grad_X)
    if not grad_weight.flags.c_contiguous:
        grad_weight = cp.ascontiguousarray(grad_weight) 
    if grad_bias is not None and not grad_bias.flags.c_contiguous:
        grad_bias = cp.ascontiguousarray(grad_bias)

    if not weight.flags.c_contiguous:
        weight = cp.ascontiguousarray(weight)


    n_rows = np.prod(X.shape[:-1])
    n_cols = X.shape[-1]

    call_cuda_function(
        CUDA_RMSNORM_BACKWARD,
        grad_O, X, weight,
        X_norm, X_std,
        grad_X, grad_weight, grad_bias,
        n_rows, n_cols, None
    )

    return grad_X, grad_weight, grad_bias





class _CUDARMSNormTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, weight: Tensor, bias: Tensor, X_norm, X_std, grad):
            # TODO: add assertions for data types and shapes
            grad_X = X.xp.empty_like(X.data, dtype=X.xp.float32)
            grad_weight = X.xp.empty_like(weight.data, dtype=X.xp.float32)
            grad_bias = X.xp.empty_like(bias.data, dtype=X.xp.float32) if bias is not None else None

            grad_X, grad_weight, grad_bias = rmsnorm_backward(
                X.data, weight.data, bias.data if bias is not None else None,
                grad, grad_X, grad_weight, grad_bias, X_norm, X_std
            )

            X.apply_grad(grad_X)
            weight.apply_grad(grad_weight)
            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class CUDARMSNorm(nn.Module): #layer with static backpropagation
    """
    Root Mean Squared Normalization with autograd backward pass.
    References: 
    https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    https://catalyst-team.github.io/catalyst/v20.12/_modules/catalyst/contrib/nn/modules/rms_norm.html
    """
    def __init__(self, dim: int, eps: float = 1e-6, device: Literal["cuda"] = "cuda", bias = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(neunet.ones(dim))

        if bias:
            self.bias: Union[nn.Parameter, None] = nn.Parameter(neunet.zeros(dim))
        else:
            self.bias = None

        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        if X.dtype != "float32":
            raise NotImplementedError(f"Only float32 is supported, got {X.dtype} instead.")
        if X.device != "cuda":
            raise NotImplementedError(f"Only CUDA is supported with cupy backend, got {X.device} instead.")

        X_norm = cp.empty_like(X.data)
        X_std = cp.empty_like(X.data)
        O = cp.empty_like(X.data)

        O, X_norm, X_std = rmsnorm_forward(
            X.data, self.weight.data, self.bias.data if self.bias is not None else None,
            X_norm, X_std, O, self.eps
        )

        return _CUDARMSNormTensor(O, (X, self.weight, self.bias, X_norm, X_std), "rmsnorm", device = self.device)
