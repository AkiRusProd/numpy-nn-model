import cupy as cp
import pytest


import os, sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet
from neunet.nn.experimental import CUDALinearSwish
from neunet.nn.layers import Linear
from neunet.nn.activations import Swish


@pytest.mark.parametrize(
    "batch_size,in_features,out_features,swish_beta,save_preactivation,rtol,atol,bwd_rtol,bwd_atol",
    [
        (128, 256, 512, 1.0, True, 1e-3, 1e-3, 1e-3, 2e-3),
        (128, 256, 512, 1.0, False, 1e-3, 1e-3, 1e-3, 2e-3),
        (64, 128, 256, 1.5, True, 1e-3, 1e-3, 1e-3, 2e-3),
        (64, 128, 256, 1.5, False, 1e-3, 1e-3, 1e-3, 2e-3),
    ],
)
def test_cutlass_linear_swish(
    batch_size,
    in_features,
    out_features,
    swish_beta,
    save_preactivation,
    rtol,
    atol,
    bwd_rtol,
    bwd_atol,
):
    device = "cuda"

    print(
        f"Testing Linear+Swish: Batch={batch_size}, In={in_features}, Out={out_features}, Beta={swish_beta}, save_preactivation={save_preactivation}"
    )

    # 1. Layer initialization
    ref_linear = Linear(in_features, out_features, bias=True, device=device)
    ref_swish = Swish(beta=swish_beta)
    cutlass_layer = CUDALinearSwish(
        in_features,
        out_features,
        bias=True,
        swish_beta=swish_beta,
        save_preactivation=save_preactivation,
        device=device,
    )

    # 2. Sync weights and bias
    cutlass_layer.weight.data = cp.copy(ref_linear.weight.data)
    cutlass_layer.bias.data = cp.copy(ref_linear.bias.data)

    # 3. Input data
    x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)
    X_ref = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    X_cutlass = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)

    # --- FORWARD PASS ---
    print("Running Forward pass...")
    out_ref = ref_swish(ref_linear(X_ref))
    out_cutlass = cutlass_layer(X_cutlass)

    forward_passed = cp.allclose(out_ref.data, out_cutlass.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("Forward pass: SUCCESS")
    else:
        diff = cp.abs(out_ref.data - out_cutlass.data).max()
        print(f"Forward pass: FAILED (Max diff: {diff})")
        print(f"   Ref range: [{out_ref.data.min():.6f}, {out_ref.data.max():.6f}]")
        print(
            f"   Cutlass range: [{out_cutlass.data.min():.6f}, {out_cutlass.data.max():.6f}]"
        )
    assert forward_passed

    # --- BACKWARD PASS ---
    print("Running Backward pass...")
    grad_output = cp.random.uniform(-1, 1, out_ref.shape).astype(cp.float32)

    out_ref.backward(grad_output)
    out_cutlass.backward(grad_output)

    # TF32 TensorOp tolerances for backward pass
    grad_X_passed = cp.allclose(X_ref.grad, X_cutlass.grad, rtol=bwd_rtol, atol=bwd_atol)
    if grad_X_passed:
        print("Backward X grad: SUCCESS")
    else:
        diff = cp.abs(X_ref.grad - X_cutlass.grad).max()
        print(f"Backward X grad: FAILED (Max diff: {diff})")
        print(f"   Ref grad range: [{X_ref.grad.min():.6f}, {X_ref.grad.max():.6f}]")
        print(
            f"   Cutlass grad range: [{X_cutlass.grad.min():.6f}, {X_cutlass.grad.max():.6f}]"
        )
    assert grad_X_passed

    grad_W_passed = cp.allclose(
        ref_linear.weight.grad, cutlass_layer.weight.grad, rtol=bwd_rtol, atol=bwd_atol
    )
    if grad_W_passed:
        print("Backward Weight grad: SUCCESS")
    else:
        diff = cp.abs(ref_linear.weight.grad - cutlass_layer.weight.grad).max()
        print(f"Backward Weight grad: FAILED (Max diff: {diff})")
        print(
            f"   Ref weight grad range: [{ref_linear.weight.grad.min():.6f}, {ref_linear.weight.grad.max():.6f}]"
        )
        print(
            f"   Cutlass weight grad range: [{cutlass_layer.weight.grad.min():.6f}, {cutlass_layer.weight.grad.max():.6f}]"
        )
    assert grad_W_passed

    grad_b_passed = cp.allclose(ref_linear.bias.grad, cutlass_layer.bias.grad, rtol=bwd_rtol, atol=bwd_atol)
    if grad_b_passed:
        print("Backward Bias grad: SUCCESS")
    else:
        diff = cp.abs(ref_linear.bias.grad - cutlass_layer.bias.grad).max()
        print(f"Backward Bias grad: FAILED (Max diff: {diff})")
        print(
            f"   Ref bias grad range: [{ref_linear.bias.grad.min():.6f}, {ref_linear.bias.grad.max():.6f}]"
        )
        print(
            f"   Cutlass bias grad range: [{cutlass_layer.bias.grad.min():.6f}, {cutlass_layer.bias.grad.max():.6f}]"
        )
    assert grad_b_passed

    print("\nALL TESTS PASSED! CUTLASS Linear+Swish implementation is correct.")
