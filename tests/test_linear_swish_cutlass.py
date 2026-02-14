import cupy as cp
import neunet

import os, sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neunet.nn.experimental import CUDALinearSwish
from neunet.nn.layers import Linear
from neunet.nn.activations import Swish

def test_cutlass_linear_swish():
    # Настройки
    batch_size = 128
    in_features = 256
    out_features = 512
    device = "cuda"
    swish_beta = 1.0
    rtol = 1e-3
    atol = 1e-3
    # TF32 TensorOp tolerances for backward pass (TF32 has ~10-bit mantissa)
    bwd_rtol = 1e-3
    bwd_atol = 2e-3

    print(f"Testing Linear+Swish Layer: Batch={batch_size}, In={in_features}, Out={out_features}, Beta={swish_beta}")

    # 1. Инициализация слоев
    # Создаем эталонный слой (Linear + Swish)
    ref_linear = Linear(in_features, out_features, bias=True, device=device)
    ref_swish = Swish(beta=swish_beta)
    
    # Создаем CUTLASS слой (fused Linear+Swish)
    cutlass_layer = CUDALinearSwish(in_features, out_features, bias=True, swish_beta=swish_beta, device=device)

    # 2. Синхронизация весов и смещений (чтобы они были идентичны)
    cutlass_layer.weight.data = cp.copy(ref_linear.weight.data)
    cutlass_layer.bias.data = cp.copy(ref_linear.bias.data)

    # 3. Подготовка входных данных
    x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)
    X_ref = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    X_cutlass = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)

    # --- FORWARD PASS ---
    print("Running Forward pass...")
    # Эталон: Linear -> Swish
    out_ref = ref_swish(ref_linear(X_ref))
    # CUTLASS: fused Linear+Swish
    out_cutlass = cutlass_layer(X_cutlass)

    # Проверка выхода
    forward_passed = cp.allclose(out_ref.data, out_cutlass.data, rtol=rtol, atol=atol)
    if forward_passed:
        print("✅ Forward pass: SUCCESS")
    else:
        diff = cp.abs(out_ref.data - out_cutlass.data).max()
        print(f"❌ Forward pass: FAILED (Max diff: {diff})")
        print(f"   Ref range: [{out_ref.data.min():.6f}, {out_ref.data.max():.6f}]")
        print(f"   Cutlass range: [{out_cutlass.data.min():.6f}, {out_cutlass.data.max():.6f}]")

    # --- BACKWARD PASS ---
    print("Running Backward pass...")
    # Генерируем случайный градиент для выхода
    grad_output = cp.random.uniform(-1, 1, out_ref.shape).astype(cp.float32)
    
    # Запускаем backprop для эталона
    out_ref.backward(grad_output)
    # Запускаем backprop для CUTLASS
    out_cutlass.backward(grad_output)

    # Сравнение градиентов по входу (dX) — uses TF32 TensorOp tolerances
    grad_X_passed = cp.allclose(X_ref.grad, X_cutlass.grad, rtol=bwd_rtol, atol=bwd_atol)
    if grad_X_passed:
        print("✅ Backward X grad: SUCCESS")
    else:
        diff = cp.abs(X_ref.grad - X_cutlass.grad).max()
        print(f"❌ Backward X grad: FAILED (Max diff: {diff})")
        print(f"   Ref grad range: [{X_ref.grad.min():.6f}, {X_ref.grad.max():.6f}]")
        print(f"   Cutlass grad range: [{X_cutlass.grad.min():.6f}, {X_cutlass.grad.max():.6f}]")

    # Сравнение градиентов по весам (dW) — uses TF32 TensorOp tolerances
    grad_W_passed = cp.allclose(ref_linear.weight.grad, cutlass_layer.weight.grad, rtol=bwd_rtol, atol=bwd_atol)
    if grad_W_passed:
        print("✅ Backward Weight grad: SUCCESS")
    else:
        diff = cp.abs(ref_linear.weight.grad - cutlass_layer.weight.grad).max()
        print(f"❌ Backward Weight grad: FAILED (Max diff: {diff})")
        print(f"   Ref weight grad range: [{ref_linear.weight.grad.min():.6f}, {ref_linear.weight.grad.max():.6f}]")
        print(f"   Cutlass weight grad range: [{cutlass_layer.weight.grad.min():.6f}, {cutlass_layer.weight.grad.max():.6f}]")

    # Сравнение градиентов по смещению (db) — uses TF32 TensorOp tolerances
    grad_b_passed = cp.allclose(ref_linear.bias.grad, cutlass_layer.bias.grad, rtol=bwd_rtol, atol=bwd_atol)
    if grad_b_passed:
        print("✅ Backward Bias grad: SUCCESS")
    else:
        diff = cp.abs(ref_linear.bias.grad - cutlass_layer.bias.grad).max()
        print(f"❌ Backward Bias grad: FAILED (Max diff: {diff})")
        print(f"   Ref bias grad range: [{ref_linear.bias.grad.min():.6f}, {ref_linear.bias.grad.max():.6f}]")
        print(f"   Cutlass bias grad range: [{cutlass_layer.bias.grad.min():.6f}, {cutlass_layer.bias.grad.max():.6f}]")

    if all([forward_passed, grad_X_passed, grad_W_passed, grad_b_passed]):
        print("\n✨ ALL TESTS PASSED! CUTLASS Linear+Swish implementation is correct.")
        return True
    else:
        print("\n⚠️ SOME TESTS FAILED. Check alignment or layouts.")
        return False

def test_cutlass_linear_swish_different_beta():
    """Тест с другим значением beta для Swish"""
    batch_size = 64
    in_features = 128
    out_features = 256
    device = "cuda"
    swish_beta = 1.5  # Другое значение beta
    rtol = 1e-3
    atol = 1e-3
    # TF32 TensorOp tolerances for backward pass
    bwd_rtol = 1e-3
    bwd_atol = 2e-3

    print(f"\nTesting Linear+Swish with beta={swish_beta}: Batch={batch_size}, In={in_features}, Out={out_features}")

    # 1. Инициализация слоев
    ref_linear = Linear(in_features, out_features, bias=True, device=device)
    ref_swish = Swish(beta=swish_beta)
    cutlass_layer = CUDALinearSwish(in_features, out_features, bias=True, swish_beta=swish_beta, device=device)

    # 2. Синхронизация весов
    cutlass_layer.weight.data = cp.copy(ref_linear.weight.data)
    cutlass_layer.bias.data = cp.copy(ref_linear.bias.data)

    # 3. Подготовка входных данных
    x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)
    X_ref = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    X_cutlass = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)

    # Forward pass
    out_ref = ref_swish(ref_linear(X_ref))
    out_cutlass = cutlass_layer(X_cutlass)

    forward_passed = cp.allclose(out_ref.data, out_cutlass.data, rtol=rtol, atol=atol)
    if forward_passed:
        print(f"✅ Forward pass with beta={swish_beta}: SUCCESS")
    else:
        diff = cp.abs(out_ref.data - out_cutlass.data).max()
        print(f"❌ Forward pass with beta={swish_beta}: FAILED (Max diff: {diff})")

    # Backward pass
    grad_output = cp.random.uniform(-1, 1, out_ref.shape).astype(cp.float32)
    out_ref.backward(grad_output)
    out_cutlass.backward(grad_output)

    all_passed = (
        forward_passed and
        cp.allclose(X_ref.grad, X_cutlass.grad, rtol=bwd_rtol, atol=bwd_atol) and
        cp.allclose(ref_linear.weight.grad, cutlass_layer.weight.grad, rtol=bwd_rtol, atol=bwd_atol) and
        cp.allclose(ref_linear.bias.grad, cutlass_layer.bias.grad, rtol=bwd_rtol, atol=bwd_atol)
    )

    if all_passed:
        print(f"✅ All tests with beta={swish_beta}: SUCCESS")
    else:
        print(f"❌ Some tests with beta={swish_beta}: FAILED")

    return all_passed

if __name__ == "__main__":
    result1 = test_cutlass_linear_swish()
    result2 = test_cutlass_linear_swish_different_beta()
    
    if result1 and result2:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)
