import numpy as np
import cupy as cp
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.activations import Softmax
from neunet.nn.experimental import CUDASoftmax


@pytest.mark.parametrize(
    "batch_size,features,axis,rtol,atol",
    [
        (32, 128, -1, 1e-5, 1e-5),
        (16, 256, 1, 1e-5, 1e-5),
    ],
)
def test_softmax_comparison(batch_size, features, axis, rtol, atol):
    print(
        f"Testing CUDA Softmax vs Native Softmax: Batch={batch_size}, Features={features}, axis={axis}"
    )

    np.random.seed(42)
    data = np.random.randn(batch_size, features).astype(np.float32)

    # Native Softmax on GPU
    t_native = Tensor(data.copy(), device="cuda", requires_grad=True)
    model_native = Softmax(axis=axis)

    # CUDA Softmax
    t_cuda = Tensor(data.copy(), device="cuda", requires_grad=True)
    model_cuda = CUDASoftmax(axis=axis)

    # Forward Pass
    out_native = model_native(t_native)
    out_cuda = model_cuda(t_cuda)

    print("\n--- Forward Pass Comparison ---")
    forward_diff = cp.abs(out_native.data - out_cuda.data).max()
    print(f"Max absolute difference (Forward): {forward_diff}")

    forward_passed = cp.allclose(out_native.data, out_cuda.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("Forward pass: SUCCESS")
    else:
        print("Forward pass: FAILED")
    assert forward_passed

    # Backward Pass
    grad_np = np.random.randn(*out_native.shape).astype(np.float32)
    grad_cuda = cp.array(grad_np)

    out_native.backward(grad_cuda.copy())
    out_cuda.backward(grad_cuda.copy())

    print("\n--- Backward Pass Comparison ---")
    grad_diff = cp.abs(t_native.grad - t_cuda.grad).max()
    print(f"Max absolute difference (Gradients): {grad_diff}")

    grad_passed = cp.allclose(t_native.grad, t_cuda.grad, rtol=rtol, atol=atol)
    if grad_passed:
        print("Backward pass: SUCCESS")
    else:
        print("Backward pass: FAILED")
    assert grad_passed
