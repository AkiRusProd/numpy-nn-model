import numpy as np
import cupy as cp
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.activations import Swish
from neunet.nn.experimental import CUDASwish

def verify_correctness():
    print("Verifying correctness...")
    batch_size = 32
    features = 128
    beta = 1.5
    
    # Generate random data
    np.random.seed(42)
    data = np.random.randn(batch_size, features).astype(np.float32)
    
    # Native Swish on GPU
    t_native = Tensor(data.copy(), device='cuda', requires_grad=True)
    model_native = Swish(beta=beta)
    
    # CUDA Swish
    t_cuda = Tensor(data.copy(), device='cuda', requires_grad=True)
    model_cuda = CUDASwish(beta=beta)
    
    # Forward Pass
    out_native = model_native(t_native)
    out_cuda = model_cuda(t_cuda)
    
    # Compare Forward Results
    try:
        cp.testing.assert_allclose(out_native.data, out_cuda.data, rtol=1e-5, atol=1e-5)
        print("Forward pass matches!")
    except AssertionError as e:
        print("Forward pass mismatch!")
        print(e)
        return

    # Backward Pass
    grad_np = np.random.randn(*out_native.shape).astype(np.float32)
    grad_cuda = cp.array(grad_np)
    
    out_native.backward(grad_cuda.copy())
    out_cuda.backward(grad_cuda.copy())
    
    # Compare Backward Results
    try:
        cp.testing.assert_allclose(t_native.grad, t_cuda.grad, rtol=1e-5, atol=1e-5)
        print("Backward pass matches!")
        print("Verification passed.\n")
    except AssertionError as e:
        print("Backward pass mismatch!")
        print(e)
        return

def benchmark_speed():
    print("Benchmarking speed...")
    
    batch_size = 4096
    features = 4096
    beta = 1.0
    iterations = 100
    warmup = 10
    
    print(f"Configuration: Batch Size={batch_size}, Features={features}, Iterations={iterations}")
    
    data = np.random.randn(batch_size, features).astype(np.float32)
    grad_data = np.random.randn(batch_size, features).astype(np.float32)
    
    # Setup Models
    model_native = Swish(beta=beta)
    model_cuda = CUDASwish(beta=beta)
    
    # Setup Tensors
    t_native = Tensor(data, device='cuda', requires_grad=True)
    t_cuda = Tensor(data, device='cuda', requires_grad=True)
    
    grad_native = cp.array(grad_data)
    grad_cuda = cp.array(grad_data)

    # --- Native Forward ---
    print("\nRunning Native Forward...")
    # Warmup
    for _ in range(warmup):
        _ = model_native(t_native)
    cp.cuda.Device().synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = model_native(t_native)
    cp.cuda.Device().synchronize()
    native_fwd_time = (time.time() - start) / iterations
    
    # --- CUDA Forward ---
    print("Running CUDA Forward...")
    # Warmup
    for _ in range(warmup):
        _ = model_cuda(t_cuda)
    cp.cuda.Device().synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = model_cuda(t_cuda)
    cp.cuda.Device().synchronize()
    cuda_fwd_time = (time.time() - start) / iterations
    
    print(f"Forward Native: {native_fwd_time*1000:.4f} ms")
    print(f"Forward CUDA:   {cuda_fwd_time*1000:.4f} ms")
    print(f"Speedup Forward: {native_fwd_time / cuda_fwd_time:.2f}x")

    # --- Native Forward+Backward ---
    print("\nRunning Native Backward...")
    # Warmup
    for _ in range(warmup):
        t_native.grad = None
        out = model_native(t_native)
        out.backward(grad_native)
    cp.cuda.Device().synchronize()
    
    start = time.time()
    for _ in range(iterations):
        t_native.grad = None
        out = model_native(t_native)
        out.backward(grad_native)
    cp.cuda.Device().synchronize()
    native_fb_time = (time.time() - start) / iterations
    native_bwd_time = native_fb_time - native_fwd_time

    # --- CUDA Forward+Backward ---
    print("Running CUDA Backward...")
    # Warmup
    for _ in range(warmup):
        t_cuda.grad = None
        out = model_cuda(t_cuda)
        out.backward(grad_cuda)
    cp.cuda.Device().synchronize()
    
    start = time.time()
    for _ in range(iterations):
        t_cuda.grad = None
        out = model_cuda(t_cuda)
        out.backward(grad_cuda)
    cp.cuda.Device().synchronize()
    cuda_fb_time = (time.time() - start) / iterations
    cuda_bwd_time = cuda_fb_time - cuda_fwd_time

    print(f"Backward Native: {native_bwd_time*1000:.4f} ms")
    print(f"Backward CUDA:   {cuda_bwd_time*1000:.4f} ms")
    if cuda_bwd_time > 0:
        print(f"Speedup Backward: {native_bwd_time / cuda_bwd_time:.2f}x")
    else:
        print("Speedup Backward: inf")

if __name__ == "__main__":
    try:
        verify_correctness()
        benchmark_speed()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
