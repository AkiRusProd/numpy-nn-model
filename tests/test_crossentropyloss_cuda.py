import numpy as np
import cupy as cp
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.autograd import Tensor
from neunet.nn.losses import CrossEntropyLoss
from neunet.nn.experimental import CUDACrossEntropyLoss


@pytest.mark.parametrize(
    "batch_size,vocab_size,ignore_index,reduction,rtol,atol",
    [
        (32, 128, -100, "none", 1e-5, 1e-5),
        (16, 256, -100, "mean", 1e-5, 1e-5),
        (8, 64, -100, "sum", 1e-5, 1e-5),
    ],
)
def test_cross_entropy_comparison(
    batch_size, vocab_size, ignore_index, reduction, rtol, atol
):
    print(
        f"Testing CUDA CrossEntropy vs Native CrossEntropy: Batch={batch_size}, Vocab={vocab_size}, reduction={reduction}"
    )

    np.random.seed(42)
    logits = np.random.randn(batch_size, vocab_size).astype(np.float32)
    labels = np.random.randint(0, vocab_size, size=(batch_size,)).astype(np.int32)

    # Native CrossEntropy on GPU
    t_logits_native = Tensor(logits.copy(), device="cuda", requires_grad=True)
    t_labels_native = Tensor(labels.copy(), device="cuda", requires_grad=False, dtype=np.int32)
    loss_native_fn = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    # CUDA CrossEntropy
    t_logits_cuda = Tensor(logits.copy(), device="cuda", requires_grad=True)
    t_labels_cuda = Tensor(labels.copy(), device="cuda", requires_grad=False, dtype=np.int32)
    loss_cuda_fn = CUDACrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)


    # Forward pass
    out_native = loss_native_fn(t_logits_native, t_labels_native)
    out_cuda = loss_cuda_fn(t_logits_cuda, t_labels_cuda)

    if reduction == "none":
        out_native = out_native.sum()
        out_cuda = out_cuda.sum()

    print("\n--- Forward Pass Comparison ---")
    forward_diff = cp.abs(out_native.data - out_cuda.data).max()
    print(f"Max absolute difference (Forward): {forward_diff}")

    forward_passed = cp.allclose(out_native.data, out_cuda.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("Forward pass: SUCCESS")
    else:
        print("Forward pass: FAILED")
    assert forward_passed

    # Backward pass
    grad = cp.array(1.0, dtype=cp.float32)

    out_native.backward(grad)
    out_cuda.backward(grad)

    print("\n--- Backward Pass Comparison ---")
    grad_diff = cp.abs(t_logits_native.grad - t_logits_cuda.grad).max()
    print(f"Max absolute difference (Gradients): {grad_diff}")

    grad_passed = cp.allclose(t_logits_native.grad, t_logits_cuda.grad, rtol=rtol, atol=atol)
    if grad_passed:
        print("Backward pass: SUCCESS")
    else:
        print("Backward pass: FAILED")
    assert grad_passed
