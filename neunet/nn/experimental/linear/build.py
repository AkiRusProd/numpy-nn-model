import os
import subprocess
import locale

def compile():
    print("Compiling CUDA linear module...")
    cuda_source_name = "linear_cublaslt_no_manual_mem.cu"
    if cuda_source_name == "linear_cublaslt_no_manual_mem.cu":
        print("WARNING: The CUDALinear module, compiled from `linear_cublaslt_no_manual_mem.cu`, “works” with numpy arrays only in the “undefined behavior” mode.")
        print("REASON: The code does not explicitly copy the numpy array from CPU to GPU, but it works. Possibly implicit allocation.")


    if os.name == 'posix':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/linear/linearcuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            f"neunet/nn/experimental/linear/{cuda_source_name}",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcublas", "-lcurand",
            "-lcublas", "-lcublasLt", "-lcurand"
        ]
    elif os.name == 'nt':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/linear/linearcuda.dll",
            "-shared",
            f"neunet/nn/experimental/linear/{cuda_source_name}",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcublas", "-lcurand",
            "-lcublas", "-lcublasLt", "-lcurand"
        ]
    else:
        raise OSError("Unsupported operating system")

    system_encoding = locale.getpreferredencoding(False)

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        stdout = result.stdout.decode(system_encoding, errors="replace")
        stderr = result.stderr.decode(system_encoding, errors="replace")
        
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during compilation: {e}")
        if e.stdout:
            print(e.stdout.decode(system_encoding, errors="replace"))
        if e.stderr:
            print(e.stderr.decode(system_encoding, errors="replace"))
        raise

    print("CUDA linear module compiled successfully.")

compile()

# nvcc -O3 -arch=sm_75 --use_fast_math -Xptxas="-v,-O3,-warn-spills" -Xcompiler "/O2 /fp:fast /MT" -Xcompiler -DNDEBUG -shared neunet/nn/experimental/linear/linear.cu -o neunet/nn/experimental/linear/linearcuda.dll -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcurand -lcudart_static -lcublas -lcublasLt -lcurand