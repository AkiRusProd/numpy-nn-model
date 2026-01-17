import ctypes
from ctypes import POINTER, c_float, c_int
import cupy as cp
import numpy as np

from neunet.nn.experimental.utils import get_module_path, load_dlls

load_dlls()

CUDA_FUSED_ADAMW_MODULE = {
    "name": "fused_adamw",
    'posix': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.so',
    'nt': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.dll'
}

# Try to load the DLL. If it doesn't exist, we can't load the functions.
# This pattern assumes compilation happens before import or handle it gracefully?
# The other modules call get_module_path at top level.
try:
    CUDA_FUSED_ADAMW_DLL = get_module_path(CUDA_FUSED_ADAMW_MODULE)
except FileNotFoundError:
    CUDA_FUSED_ADAMW_DLL = None

# Helper to load CUDA functions
def _load_cuda_function(module_path, function_name, arg_types):
    if module_path is None:
        return None
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    return func

if CUDA_FUSED_ADAMW_DLL:
    CUDA_ADAMW_STEP = _load_cuda_function(
        CUDA_FUSED_ADAMW_DLL, "FusedAdamWStep", 
        [
            ctypes.POINTER(ctypes.c_float), # params
            ctypes.POINTER(ctypes.c_float), # grads
            ctypes.POINTER(ctypes.c_float), # exp_avgs
            ctypes.POINTER(ctypes.c_float), # exp_avg_sqs
            c_float, # lr
            c_float, # beta1
            c_float, # beta2
            c_float, # eps
            c_float, # weight_decay
            c_int,   # step
            c_int,   # n
            ctypes.c_void_p # stream
        ]
    )
else:
    CUDA_ADAMW_STEP = None

def call_cuda_function(func, *args):
    if func is None:
        raise RuntimeError("CUDA FusedAdamW module not compiled. Run 'build.py' in the module directory.")

    def _to_pointer(obj):
        if obj is None:
            return None
        elif isinstance(obj, cp.ndarray):
            if obj.dtype == cp.float32:
                return ctypes.cast(obj.data.ptr, POINTER(c_float))
            elif obj.dtype == cp.int32:
                return ctypes.cast(obj.data.ptr, POINTER(c_int))
            else:
                raise ValueError(f"Unsupported dtype: {obj.dtype}. Expected float32 or int32.")
        return obj

    return func(*[_to_pointer(arg) for arg in args])

class CUDAFusedAdamW:
    def __init__(self, params, lr: float=0.01, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moments
        self.m = [param.xp.zeros_like(param.data) for param in self.params]
        self.v = [param.xp.zeros_like(param.data) for param in self.params]

        self.t = 0
        
        # Verify all params are on CUDA and contiguous
        for p in self.params:
             if p.xp != cp:
                 raise ValueError("FusedAdamW only supports parameters on CUDA (CuPy backend).")

    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Ensure float32
            if param.data.dtype != cp.float32:
                 raise ValueError(f"FusedAdamW only supports float32 parameters, got {param.data.dtype}")

            if not param.data.flags.c_contiguous:
                param.data = cp.ascontiguousarray(param.data)
            
            if not grad.flags.c_contiguous:
                grad = cp.ascontiguousarray(grad)
                
            m = self.m[i]
            v = self.v[i]
            
            if not m.flags.c_contiguous:
                m = cp.ascontiguousarray(m)
                self.m[i] = m
            if not v.flags.c_contiguous:
                v = cp.ascontiguousarray(v)
                self.v[i] = v

            n = param.data.size
            
            call_cuda_function(
                CUDA_ADAMW_STEP,
                param.data,
                grad,
                m,
                v,
                self.lr,
                self.betas[0],
                self.betas[1],
                self.eps,
                self.weight_decay,
                self.t,
                n,
                None # stream
            )

    def zero_grad(self):
        for param in self.params:
            param.grad = None
