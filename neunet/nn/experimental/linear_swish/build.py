import locale
import os
import subprocess

def compile_cutlass():
    print("Compiling CUTLASS linear module...")
    
    # Имя вашего нового файла на CUTLASS
    cuda_source_name = "linear_swish_cutlass_evt_full.cu" 
    output_name = "linear_swish_cutlass"
    
    # Путь к CUTLASS (относительно корня проекта)
    cutlass_inc = os.path.join("third_party", "cutlass", "include")
    # Дополнительные утилиты CUTLASS (иногда нужны для хелперов)
    cutlass_util_inc = os.path.join("third_party", "cutlass", "tools", "util", "include")

    # Архитектура для 4060 Ti (Ada Lovelace)
    sm_version = "sm_89"

    if os.name == 'posix':
        out_file = f"neunet/nn/experimental/linear_swish/{output_name}.so"
        command = [
            "nvcc", "-O3",
            "-std=c++17",
            f"-arch={sm_version}",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-o", out_file,
            "-Xcompiler", "-fPIC",
            "-shared",
            f"neunet/nn/experimental/linear_swish/{cuda_source_name}",
            f"-I{cutlass_inc}",
            f"-I{cutlass_util_inc}",
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-lcudart"
        ]
    elif os.name == 'nt':
        cuda_path = os.environ.get("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x")
        out_file = f"neunet/nn/experimental/linear_swish/{output_name}.dll"
        
        # Ваш специфичный путь к MSVC
        msvc_bin = r"Z:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
        os.environ["PATH"] = msvc_bin + ";" + os.environ.get("PATH", "")

        command = [
            "nvcc", "-O3",
            "-std=c++17",
            f"-arch={sm_version}",
            "-D__CUDA_ARCH_LIST__=890",
            "-DCUTLASS_EXCLUDE_SM100_KERNELS",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--extended-lambda",
            "-DNDEBUG",
            "-allow-unsupported-compiler",
            "-o", out_file,
            "-shared",
            f"neunet/nn/experimental/linear_swish/{cuda_source_name}",
            f"-I{cutlass_inc}",
            f"-I{cutlass_util_inc}",
            f"-I{cuda_path}\\include",
            f"-L{cuda_path}\\lib\\x64",
            "-lcudart"
        ]
        # Добавляем флаги оптимизации для MSVC
        # /bigobj - CUTLASS templates generate large object files
        # /Zc:preprocessor - conformant preprocessor for CUTLASS macros
        command += ["-Xcompiler", "/O2 /fp:fast /DNDEBUG /bigobj /Zc:preprocessor"]
    else:
        raise OSError("Unsupported operating system")

    # Try to remove old DLL if it exists (may fail if loaded in Python)
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
            print(f"Removed old {out_file}")
        except OSError as e:
            print(f"Warning: Could not remove {out_file}: {e}")
            print("  This usually means the DLL is loaded in a Python process.")
            print("  Please close all Python processes that use this module and try again.")
            raise

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