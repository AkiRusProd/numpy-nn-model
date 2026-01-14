import ctypes
from ctypes import POINTER, c_float, c_int, c_void_p
import cupy as cp

from neunet.nn.experimental.utils import get_module_path, load_dlls

load_dlls()

CUDA_FUSED_ADAMW_MODULE = {
    "name": "fused_adamw_multitensor",
    'posix': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.so',
    'nt': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.dll'
}

try:
    CUDA_FUSED_ADAMW_DLL = get_module_path(CUDA_FUSED_ADAMW_MODULE)
except FileNotFoundError:
    CUDA_FUSED_ADAMW_DLL = None

def _load_cuda_function(module_path, function_name, arg_types, res_type=None):
    if module_path is None:
        return None
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    if res_type:
        func.restype = res_type
    return func


if CUDA_FUSED_ADAMW_DLL:
    CreateFusedOptimizer = _load_cuda_function(
        CUDA_FUSED_ADAMW_DLL, "CreateFusedOptimizer", [], c_void_p
    )
    
    DestroyFusedOptimizer = _load_cuda_function(
        CUDA_FUSED_ADAMW_DLL, "DestroyFusedOptimizer", [c_void_p]
    )

    FusedAdamWStep = _load_cuda_function(
        CUDA_FUSED_ADAMW_DLL, "FusedAdamWStep",
        [
            c_void_p,                   # optimizer instance
            c_int,                      # num_tensors
            POINTER(POINTER(c_float)),  # params list (float**)
            POINTER(POINTER(c_float)),  # grads list
            POINTER(POINTER(c_float)),  # exp_avgs list
            POINTER(POINTER(c_float)),  # exp_avg_sqs list
            POINTER(c_int),             # sizes list (int*)
            c_float, c_float, c_float, c_float, c_float, # lr, beta1, beta2, eps, wd
            c_int,                      # step
            c_void_p                    # stream
        ]
    )
else:
    CreateFusedOptimizer = None
    DestroyFusedOptimizer = None
    FusedAdamWStep = None

class CUDAFusedMultiTensorAdamW:
    def __init__(self, params, lr: float=0.01, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize moments
        self.m = []
        self.v = []
        for p in self.params:
            if p.xp != cp:
                raise ValueError("FusedAdamW only supports parameters on CUDA (CuPy backend).")
            self.m.append(cp.zeros_like(p.data))
            self.v.append(cp.zeros_like(p.data))

        # Create C++ optimizer
        if CreateFusedOptimizer:
            self.opt_ptr = CreateFusedOptimizer()
        else:
            raise RuntimeError("CUDA library not loaded")

        # Cached arrays to avoid recreation at each step
        # Stored as ctypes arrays
        self._num_tensors = len(self.params)
        
        FloatPtrArray = POINTER(c_float) * self._num_tensors
        IntArray = c_int * self._num_tensors
        
        self.c_params = FloatPtrArray()
        self.c_grads = FloatPtrArray()
        self.c_exp_avgs = FloatPtrArray()
        self.c_exp_avg_sqs = FloatPtrArray()
        self.c_sizes = IntArray()

    def __del__(self):
        if hasattr(self, 'opt_ptr') and self.opt_ptr and DestroyFusedOptimizer:
            DestroyFusedOptimizer(self.opt_ptr)

    def step(self):
        self.t += 1
        
        # 1. Filter parameters that have gradients
        active_params_indices = []
        for i, p in enumerate(self.params):
            if p.grad is not None:
                active_params_indices.append(i)
        
        if not active_params_indices:
            return

        current_num_tensors = len(active_params_indices)
        
        def get_ptr(arr):
            return ctypes.cast(arr.data.ptr, POINTER(c_float))

        idx = 0
        for i in active_params_indices:
            p = self.params[i]
            g = p.grad
            m = self.m[i]
            v = self.v[i]
            
            # Ensure contiguous memory
            if not p.data.flags.c_contiguous: p.data = cp.ascontiguousarray(p.data)
            if not g.flags.c_contiguous: g = cp.ascontiguousarray(g) # Note: might create a copy, but acceptable for optimizer
            if not m.flags.c_contiguous: m = cp.ascontiguousarray(m); self.m[i] = m
            if not v.flags.c_contiguous: v = cp.ascontiguousarray(v); self.v[i] = v

            self.c_params[idx] = get_ptr(p.data)
            self.c_grads[idx] = get_ptr(g)
            self.c_exp_avgs[idx] = get_ptr(m)
            self.c_exp_avg_sqs[idx] = get_ptr(v)
            self.c_sizes[idx] = p.data.size
            idx += 1

        # Call CUDA kernel
        # Pass pointers to the start of our pointer arrays
        # ctypes array automatically converts to pointer when calling function with argtypes
        
        FusedAdamWStep(
            self.opt_ptr,
            current_num_tensors,
            ctypes.cast(self.c_params, POINTER(POINTER(c_float))),
            ctypes.cast(self.c_grads, POINTER(POINTER(c_float))),
            ctypes.cast(self.c_exp_avgs, POINTER(POINTER(c_float))),
            ctypes.cast(self.c_exp_avg_sqs, POINTER(POINTER(c_float))),
            ctypes.cast(self.c_sizes, POINTER(c_int)),
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.t,
            None # stream (optional: cp.cuda.get_current_stream().ptr)
        )

    def zero_grad(self):
        for param in self.params:
            param.grad = None
