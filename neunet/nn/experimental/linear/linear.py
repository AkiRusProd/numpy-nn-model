import ctypes
import os
from ctypes import POINTER, c_float, c_size_t
from typing import Literal, Union

import cupy as cp
import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


# Set DLL path for CUDA libraries
# DLL_PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin'
# os.add_dll_directory(DLL_PATH)

def find_cuda_path():
    cuda_path = os.getenv('CUDA_PATH')
    if cuda_path:
        return os.path.join(cuda_path, 'bin')
    elif os.name == 'posix':
        cuda_path = '/usr/local/cuda'
        if os.path.exists(cuda_path):
            return '/usr/local/cuda/bin'
    
    raise EnvironmentError("CUDA_PATH is not set in the environment variables.")

DLL_PATH = find_cuda_path()

# Helper to load CUDA functions
def _load_cuda_function(dll_path, function_name, argtypes):
    dll = ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = argtypes
    return func

# Load CUDA linear module functions
if os.name == 'posix':
    CUDA_LINEAR_DLL = 'neunet/nn/experimental/linear/linearcuda.so'
elif os.name == 'nt':
    CUDA_LINEAR_DLL = 'neunet/nn/experimental/linear/linearcuda.dll'
else:
    raise OSError("Unsupported operating system")

CUDA_LINEAR_FORWARD = _load_cuda_function(
    CUDA_LINEAR_DLL, 'cudaLinearModuleForward', [
        POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
        c_size_t, c_size_t, c_size_t
    ]
)
CUDA_LINEAR_BACKWARD = _load_cuda_function(
    CUDA_LINEAR_DLL, 'cudaLinearModuleBackward', [
        POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
        POINTER(c_float), POINTER(c_float),
        c_size_t, c_size_t, c_size_t
    ]
)

class ndarray:
    def __init__(self, array: Union[np.ndarray, cp.ndarray]):
        self.array = array

    def __array__(self):
        return self.array

# Helper for casting data to pointers
def _to_pointer(array: Union[ndarray, None]):
    if array is None:
        return array
    elif isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, POINTER(c_float))
    return array.ctypes.data_as(POINTER(c_float))

def cuda_linear_module_forward(X: ndarray, weights: ndarray, bias: ndarray, O: ndarray, input_rows: int, input_cols: int, output_cols: int):
    CUDA_LINEAR_FORWARD(
        _to_pointer(X), _to_pointer(weights), _to_pointer(bias), _to_pointer(O),
        input_rows, input_cols, output_cols
    )

def cuda_linear_module_backward(X: ndarray, weights: ndarray, grad_O: ndarray, grad_X: ndarray, grad_weight: ndarray, grad_bias: ndarray, input_rows: int, input_cols: int, output_cols: int):
    CUDA_LINEAR_BACKWARD(
        _to_pointer(X), _to_pointer(weights), _to_pointer(grad_O), _to_pointer(grad_X),
        _to_pointer(grad_weight), _to_pointer(grad_bias),
        input_rows, input_cols, output_cols
    )

class _CUDALinearTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, weight: Tensor, bias: Tensor, in_rows_num, in_features, out_features, grad):
            grad_X = X.xp.zeros_like(X.data, dtype=np.float32)
            grad_weight = cp.zeros_like(weight.data, dtype=np.float32)
            grad_bias = cp.zeros_like(bias.data, dtype=np.float32) if bias is not None else None

            cuda_linear_module_backward(
                X.data, weight.data, grad, grad_X,
                grad_weight, grad_bias,
                in_rows_num, in_features, out_features
            )

            X.apply_grad(grad_X)
            weight.apply_grad(grad_weight)
            if bias is not None:
                bias.apply_grad(grad_bias)

        self.grad_fn = grad_fn

class CUDALinear(Module):
    def __init__(self, in_features, out_features,  bias: bool=True, device: Literal["cpu", "cuda"] = "cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

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
        output_shape = X.shape[:-1] + (self.out_features,)
        output = X.xp.zeros(output_shape, dtype=np.float32)

        # Compute forward pass
        input_rows = np.prod(X.shape[:-1])
        cuda_linear_module_forward(
            X.data, self.weight.data, self.bias.data if self.bias is not None else None, output,
            input_rows, self.in_features, self.out_features
        )

        return _CUDALinearTensor(
            output, (X, self.weight, self.bias, input_rows, self.in_features, self.out_features),
            "linear", device=self.device
        )

    def __call__(self, X):
        return self.forward(X)
