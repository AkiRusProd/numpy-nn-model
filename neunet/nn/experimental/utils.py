import os
import ctypes

CUDA_LINEAR_MODULE = {
    "name": "linear",
    'posix': 'neunet/nn/experimental/linear/linearcuda.so',
    'nt': 'neunet/nn/experimental/linear/linearcuda.dll'
}

CUDA_CUTLASS_LINEAR_MODULE = {
    "name": "cutlass_linear",
    'posix': 'neunet/nn/experimental/linear/linearcutlass.so',
    'nt': 'neunet/nn/experimental/linear/linearcutlass.dll'
}

CUDA_SOFTMAX_MODULE = {
    "name": "softmax",
    'posix': 'neunet/nn/experimental/activations/softmax/softmaxcuda.so',
    'nt': 'neunet/nn/experimental/activations/softmax/softmaxcuda.dll'
}

CUDA_CROSS_ENTROPY_MODULE = {
    "name": "cross_entropy",
    'posix': 'neunet/nn/experimental/losses/cross_entropy_loss/cross_entropy_cuda.so',
    'nt': 'neunet/nn/experimental/losses/cross_entropy_loss/cross_entropy_cuda.dll'
}

CUDA_RMSNORM_MODULE = {
    "name": "rmsnorm",
    'posix': 'neunet/nn/experimental/rmsnorm/rmsnorm_cuda.so',
    'nt': 'neunet/nn/experimental/rmsnorm/rmsnorm_cuda.dll'
}

CUDA_SWISH_MODULE = {
    "name": "swish",
    'posix': 'neunet/nn/experimental/activations/swish/swish_cuda.so',
    'nt': 'neunet/nn/experimental/activations/swish/swish_cuda.dll'
}

CUDA_FUSED_SWISH_AND_MUL_MODULE = {
    "name": "fused_swish_and_mul",
    'posix': 'neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul_cuda.so',
    'nt': 'neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul_cuda.dll'
}

CUDA_LINEAR_SWISH_CUTLASS_MODULE = {
    "name": "linear_swish_cutlass",
    'posix': 'neunet/nn/experimental/linear_swish/linear_swish_cutlass.so',
    'nt': 'neunet/nn/experimental/linear_swish/linear_swish_cutlass.dll'
}

CUDA_FUSED_ADAMW_MULTITENSOR_MODULE = {
    "name": "fused_adamw_multitensor",
    'posix': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.so',
    'nt': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.dll'
}

CUDA_FUSED_ADAMW_MODULE = {
    "name": "fused_adamw",
    'posix': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.so',
    'nt': 'neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.dll'
}

def load_cuda_function(module_path, function_name, arg_types, restype=None):
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    if restype is not None:
        func.restype = restype
    return func

def to_pointer(obj):
    if obj is None:
        return None
    if hasattr(obj, "ctypes"):
        raise TypeError("NumPy arrays are not supported here.")
    if hasattr(obj, "data") and hasattr(obj.data, "ptr"):
        dtype = getattr(obj, "dtype", None)
        if dtype is not None and getattr(dtype, "name", None) == "int32":
            return ctypes.cast(obj.data.ptr, ctypes.POINTER(ctypes.c_int))
        return ctypes.cast(obj.data.ptr, ctypes.POINTER(ctypes.c_float))
    return obj

def call_cuda_function(func, *args):
    return func(*[to_pointer(arg) for arg in args])

def get_current_stream_ptr():
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError("CuPy is required for CUDA stream access.") from exc
    return cp.cuda.get_current_stream().ptr

def load_dlls():
    """
    Loads CUDA DLLs for Windows systems by adding the CUDA binary directory to the DLL search path.
    """
    cuda_path = os.getenv('CUDA_PATH') or os.getenv('CUDA_HOME')
    if cuda_path and os.name == 'nt':
        os.add_dll_directory(os.path.join(cuda_path, 'bin'))
        return
    elif os.name == 'posix':
        return
    raise EnvironmentError("CUDA_PATH is not set in the environment variables.")

def get_module_path(module: dict[str, str]):
    """
    Verifies the presence of the compiled CUDA module and returns its path.
    """
    module_path = module.get(os.name)
    if not module_path:
        raise OSError("Unsupported operating system")
    elif os.path.exists(module_path):
        print(f"CUDA {module['name']} module loaded.")
        return module_path
    else:
        print(
            f"CUDA module with name '{module['name']}' not found at '{module_path}'. "
            "Please compile it with 'build.py' from CUDA module in the 'neunet/nn/experimental' directory. Skipping..."
        )
        return None
