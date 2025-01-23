import os
import subprocess

def compile():
    print("Compiling CUDA linear module...")

    if os.name == 'posix':
        result = subprocess.run([
            "nvcc",
            "-o", "neunet/nn/experimental/linear/linearcuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/linear/linear.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcublas", "-lcurand"
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
     
    elif os.name == 'nt':
        result = subprocess.run([
            "nvcc",
            "-o", "neunet/nn/experimental/linear/linearcuda.dll",
            "-shared",
            "neunet/nn/experimental/linear/linear.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcublas", "-lcurand"
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

    else:
        raise OSError("Unsupported operating system")
    
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    
    if len(stdout) > 0:
        print(stdout)

    if len(stderr) > 0:
        print(stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args,
                                                output=result.stdout, stderr=result.stderr)   

    print("CUDA linear module compiled successfully.")

compile()