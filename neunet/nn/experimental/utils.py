import os

CUDA_LINEAR_MODULE = {
    "name": "linear",
    'posix': 'neunet/nn/experimental/linear/linearcuda.so',
    'nt': 'neunet/nn/experimental/linear/linearcuda.dll'
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
        raise FileNotFoundError(
            f"CUDA module not found at '{module_path}'. "
            "Please compile it with 'build.py' from CUDA module in the 'neunet/nn/experimental' directory."
        )