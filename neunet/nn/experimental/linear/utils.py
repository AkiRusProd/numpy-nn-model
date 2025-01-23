import os

CUDA_LINEAR_MODULES = {
    'posix': 'neunet/nn/experimental/linear/linearcuda.so',
    'nt': 'neunet/nn/experimental/linear/linearcuda.dll'
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

def get_module_path():
    """
    Verifies the presence of the compiled CUDA linear module and returns its path.
    """
    module_path = CUDA_LINEAR_MODULES.get(os.name)
    if not module_path:
        raise OSError("Unsupported operating system")
    elif os.path.exists(module_path):
        print("CUDA linear module loaded.")
        return module_path
    else:
        raise FileNotFoundError(
            f"CUDA linear module not found at '{module_path}'. "
            "Please compile it with 'python neunet/nn/experimental/linear/build.py'"
        )