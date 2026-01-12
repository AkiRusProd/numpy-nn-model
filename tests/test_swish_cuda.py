import numpy as np
import cupy as cp
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.activations import Swish
from neunet.nn.experimental import CUDASwish

def test_swish_comparison():
    print("Testing CUDA Swish vs Native Swish...")
    
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
    print("\n--- Forward Pass Comparison ---")
    forward_diff = cp.abs(out_native.data - out_cuda.data).max()
    print(f"Max absolute difference (Forward): {forward_diff}")
    
    try:
        cp.testing.assert_allclose(out_native.data, out_cuda.data, rtol=1e-5, atol=1e-5)
        print("Forward pass matches!")
    except AssertionError as e:
        print("Forward pass mismatch!")
        print(e)

    # Backward Pass
    # Create a random gradient
    grad_np = np.random.randn(*out_native.shape).astype(np.float32)
    grad_cuda = cp.array(grad_np)
    
    out_native.backward(grad_cuda.copy())
    out_cuda.backward(grad_cuda.copy())
    
    # Compare Backward Results (Gradients)
    print("\n--- Backward Pass Comparison ---")
    grad_diff = cp.abs(t_native.grad - t_cuda.grad).max()
    print(f"Max absolute difference (Gradients): {grad_diff}")
    
    try:
        cp.testing.assert_allclose(t_native.grad, t_cuda.grad, rtol=1e-5, atol=1e-5)
        print("Backward pass matches!")

        print(f"{t_native.grad.flatten()[:5]=}")
        print(f"{t_cuda.grad.flatten()[:5]=}")

    except AssertionError as e:
        print("Backward pass mismatch!")
        print(e)

if __name__ == "__main__":
    try:
        test_swish_comparison()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
