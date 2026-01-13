#include <cuda_runtime.h>
#include <math_functions.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

__global__ void adamw_kernel(
    float* params,
    const float* grads,
    float* exp_avgs,
    float* exp_avg_sqs,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float p = params[i];
    float g = grads[i];
    float m = exp_avgs[i];
    float v = exp_avg_sqs[i];

    // Weight decay
    if (weight_decay != 0.0f) {
        p -= lr * weight_decay * p;
    }

    // Update first moment
    m = beta1 * m + (1.0f - beta1) * g;
    
    // Update second moment
    v = beta2 * v + (1.0f - beta2) * g * g;

    // Bias correction
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    // Write back
    params[i] = p;
    exp_avgs[i] = m;
    exp_avg_sqs[i] = v;
}

extern "C" {
    DLLEXPORT void FusedAdamWStep(
        float* params,
        const float* grads,
        float* exp_avgs,
        float* exp_avg_sqs,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        int step,
        int n,
        cudaStream_t stream = 0
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
        adamw_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            params, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, weight_decay, step, n
        );
    }
}
