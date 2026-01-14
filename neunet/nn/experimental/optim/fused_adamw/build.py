import locale
import os
import subprocess


def compile():
    print("Compiling CUDA FusedAdamW module...")

    if os.name == 'posix':
        command_original = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]
        
        command_multitensor = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.so",
            "-Xcompiler", "-fPIC",
            "-shared",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]
        
        commands = [command_original, command_multitensor]
    elif os.name == 'nt':
        # Compile original FusedAdamW
        command_original = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_cuda.dll",
            "-shared",
            "-allow-unsupported-compiler",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]

        # Compile MultiTensor FusedAdamW
        command_multitensor = [
            "nvcc",
            "-o", "neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor_cuda.dll",
            "-shared",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "neunet/nn/experimental/optim/fused_adamw/fused_adamw_multitensor.cu",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
        ]
        
        commands = [command_original, command_multitensor]
    else:
        raise OSError("Unsupported operating system")

    system_encoding = locale.getpreferredencoding(False)

    try:
        for cmd in commands:
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
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
