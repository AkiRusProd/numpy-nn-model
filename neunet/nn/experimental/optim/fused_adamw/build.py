import locale
import os
import subprocess


def compile():
    print("Compiling CUDA FusedAdamW module...")

    if os.name == 'posix':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.dll",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]
    elif os.name == 'nt':
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.dll",
            "-shared",
            "-allow-unsupported-compiler",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw.cu",
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

    print("CUDA FusedAdamW module compiled successfully.")

if __name__ == "__main__":
    compile()
