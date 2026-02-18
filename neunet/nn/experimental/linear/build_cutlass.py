import locale
import os
import subprocess

def compile_cutlass():
    print("Compiling CUTLASS linear module...")
    
    cuda_source_name = "linear_cutlass.cu" 
    output_name = "linearcutlass"
    
    cutlass_inc = os.path.join("third_party", "cutlass", "include")
    cutlass_util_inc = os.path.join("third_party", "cutlass", "tools", "util", "include")

    sm_version = "sm_89"

    if os.name == 'posix':
        out_file = f"neunet/nn/experimental/linear/{output_name}.so"
        command = [
            "nvcc", "-O3",
            "-std=c++17",
            f"-arch={sm_version}",
            "--use_fast_math",
            "-o", out_file,
            "-Xcompiler", "-fPIC",
            "-shared",
            f"neunet/nn/experimental/linear/{cuda_source_name}",
            f"-I{cutlass_inc}",
            f"-I{cutlass_util_inc}",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcudart"
        ]
    elif os.name == 'nt':
        out_file = f"neunet/nn/experimental/linear/{output_name}.dll"
        command = [
            "nvcc", "-O3",
            "-std=c++17",
            f"-arch={sm_version}",
            "--use_fast_math",
            "-allow-unsupported-compiler",
            "-o", out_file,
            "-shared",
            f"neunet/nn/experimental/linear/{cuda_source_name}",
            f"-I{cutlass_inc}",
            f"-I{cutlass_util_inc}",
            "-lcudart"
        ]
        
        command += ["-Xcompiler", "/O2 /fp:fast /DNDEBUG"]
    else:
        raise OSError("Unsupported operating system")

    print(f"Executing command: {' '.join(command)}")

    system_encoding = locale.getpreferredencoding(False)

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        stdout = result.stdout.decode(system_encoding, errors="replace")
        stderr = result.stderr.decode(system_encoding, errors="replace")
        
        if stdout: print(stdout)
        if stderr: print(stderr)
        print(f"Successfully compiled: {out_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during compilation: {e}")
        if e.stdout: print(e.stdout.decode(system_encoding, errors="replace"))
        if e.stderr: print(e.stderr.decode(system_encoding, errors="replace"))
        raise

if __name__ == "__main__":
    compile_cutlass()