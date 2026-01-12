import locale
import os
import subprocess


def compile():
    print("Compiling CUDA swish module...")

    if os.name == 'posix':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/activations/swish/swish_cuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/activations/swish/swish.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]
    elif os.name == 'nt':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/activations/swish/swish_cuda.dll",
            "-shared",
            "neunet/nn/experimental/activations/swish/swish.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
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

    print("CUDA swish module compiled successfully.")

if __name__ == "__main__":
    compile()
