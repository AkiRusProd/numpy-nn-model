import locale
import os
import subprocess


def compile():
    print("Compiling CUDA fused_swish_and_mul module...")

    if os.name == "posix":
        command = [
            "nvcc",
            "-o", "neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul_cuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-use_fast_math",
        ]
    elif os.name == "nt":
        command = [
            "nvcc",
            "-allow-unsupported-compiler",
            "-o", "neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul_cuda.dll",
            "-shared",
            "neunet/nn/experimental/activations/fused_swish_and_mul/fused_swish_and_mul.cu",
            "-use_fast_math",
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

    print("CUDA fused_swish_and_mul module compiled successfully.")


if __name__ == "__main__":
    compile()
