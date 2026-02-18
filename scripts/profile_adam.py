# import nvtx
from cupy.cuda import nvtx
import cupy as cp
import neunet
from neunet.nn.experimental.optim.fused_adamw.fused_adamw import CUDAFusedAdamW
from neunet.nn.experimental.optim.fused_adamw.fused_adamw_multitensor import CUDAFusedMultiTensorAdamW
from neunet.optim import AdamW

# Создаем много параметров, чтобы имитировать нагрузку
params = []
for _ in range(200): # 200 тензоров
    p = neunet.tensor(cp.random.randn(512, 1024).astype(cp.float32), requires_grad=True, device="cuda")
    p.grad = cp.random.randn(512, 1024).astype(cp.float32)
    params.append(p)

# Выбери оптимизатор
opt = CUDAFusedAdamW(params)
# opt = CUDAFusedMultiTensorAdamW(params)
# opt = AdamW(params)

# Warmup
nvtx.RangePush("Warmup")
try:
    for _ in range(3):
        opt.step()
finally:
    nvtx.RangePop()

cp.cuda.Device(0).synchronize()

print("Start profiling...")

# Profiling
nvtx.RangePush("Profiling Loop")
try:
    for i in range(5):
        nvtx.RangePush(f"Step {i}")
        try:
            opt.step()
        finally:
            nvtx.RangePop()
finally:
    nvtx.RangePop()

cp.cuda.Device(0).synchronize()
print("Done.")

#  ncu --nvtx --nvtx-include "Step 0/" --launch-count 1  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis -f -o profile_multitensor python profile_adam.py

#  $env:PATH += ";C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe"

# ncu --nvtx --nvtx-include "Step 0/" --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis -f -o profile_adam_single_quick --print-summary per-gpu python profile_adam.py

# ncu --nvtx --nvtx-include "Step 0/" --launch-count 1 --set full -f -o profile_adam_detailed --print-summary per-gpu python profile_adam.py