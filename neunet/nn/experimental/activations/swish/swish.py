import ctypes
from ctypes import POINTER, c_float, c_size_t, c_int
import cupy as cp
import neunet.nn as nn
from neunet.autograd import Tensor
from neunet.nn.experimental.utils import CUDA_SWISH_MODULE, get_module_path, load_dlls

load_dlls()

CUDA_SWISH_DLL = get_module_path(CUDA_SWISH_MODULE)

# Helper to load CUDA functions
def _load_cuda_function(module_path, function_name, arg_types):
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    return func

CUDA_SWISH_FORWARD = _load_cuda_function(
    CUDA_SWISH_DLL, "cudaSwishForward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_float,
        c_int, 
        ctypes.c_void_p
    ]
)

CUDA_SWISH_BACKWARD = _load_cuda_function(
    CUDA_SWISH_DLL, "cudaSwishBackward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_float,
        c_int,
        ctypes.c_void_p
    ]
)

def call_cuda_function(func, *args):
    # Helper for casting data to pointers
    def _to_pointer(array: cp.ndarray):
        if array is None:
            return None
        elif isinstance(array, cp.ndarray):
            return ctypes.cast(array.data.ptr, POINTER(c_float))

        return array

    return func(*[_to_pointer(arg) for arg in args])

def cuda_swish_forward(x: cp.ndarray, out: cp.ndarray, beta: float):
    if not all([isinstance(arg, cp.ndarray) for arg in [x, out]]):
        raise ValueError("All arguments must be cupy arrays.")
    if x.shape != out.shape:
        raise ValueError("Input and output shapes must match")
    
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    if not out.flags.c_contiguous:
        out = cp.ascontiguousarray(out)
    
    size = x.size
    stream = cp.cuda.get_current_stream()
    
    call_cuda_function(
        CUDA_SWISH_FORWARD, 
        out, 
        x, 
        beta,
        size,
        stream.ptr
    )
    return out

def cuda_swish_backward(grad_input: cp.ndarray, grad_output: cp.ndarray, x: cp.ndarray, beta: float):
    if not all([isinstance(arg, cp.ndarray) for arg in [grad_input, grad_output, x]]):
        raise ValueError("All arguments must be cupy arrays.")
    if not (grad_input.shape == grad_output.shape == x.shape):
        raise ValueError("Shapes must match")
    
    if not grad_output.flags.c_contiguous:
        grad_output = cp.ascontiguousarray(grad_output)
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
        
    size = x.size
    stream = cp.cuda.get_current_stream()
    
    call_cuda_function(
        CUDA_SWISH_BACKWARD,
        grad_input,
        grad_output,
        x,
        beta,
        size,
        stream.ptr
    )
    return grad_input

class _CUDASwishTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(x: Tensor, beta, grad):
            if not all(arr.dtype == "float32" for arr in [x, grad]):
                 pass
            
            grad_input = x.xp.empty_like(x.data)
            cuda_swish_backward(grad_input, grad, x.data, beta)
            
            x.apply_grad(grad_input)

        self.grad_fn = grad_fn

class CUDASwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CUDASwish, self).__init__()
        self.beta = beta

    def forward(self, x: Tensor):
        if x.dtype != "float32":
             raise NotImplementedError(f"Only float32 is supported, got {x.dtype} instead.")
        if x.device != "cuda":
             raise NotImplementedError(f"Only CUDA is supported with cupy backend, got {x.device} instead.")

        out = x.xp.empty_like(x.data)
        cuda_swish_forward(x.data, out, self.beta)
        
        return _CUDASwishTensor(out, [x, self.beta], "swish", device=x.device)
