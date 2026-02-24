#include <cuda_runtime.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_swish_and_mul_kernel(
    float* output,
    const float* input,
    float beta,
    int hidden_size,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int row = idx / hidden_size;
        int col = idx - row * hidden_size;
        int base = row * (hidden_size * 2) + col;

        float gate = input[base];
        float up = input[base + hidden_size];
        output[idx] = (gate * sigmoid(beta * gate)) * up;
    }
}

__global__ void fused_swish_and_mul_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* input,
    float beta,
    int hidden_size,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int row = idx / hidden_size;
        int col = idx - row * hidden_size;
        int base = row * (hidden_size * 2) + col;

        float gate = input[base];
        float up = input[base + hidden_size];
        float grad = grad_output[idx];

        float s = sigmoid(beta * gate);
        float swish_gate = gate * s;
        float d_swish = s + beta * swish_gate * (1.0f - s);

        grad_input[base] = grad * up * d_swish;
        grad_input[base + hidden_size] = grad * swish_gate;
    }
}

extern "C" {
    DLLEXPORT void cudaFusedSwishAndMul(
        float* output,
        const float* input,
        float beta,
        int hidden_size,
        int size,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

        fused_swish_and_mul_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            output, input, beta, hidden_size, size
        );
    }

    DLLEXPORT void cudaFusedSwishAndMulBackward(
        float* grad_input,
        const float* grad_output,
        const float* input,
        float beta,
        int hidden_size,
        int size,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

        fused_swish_and_mul_backward_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            grad_input, grad_output, input, beta, hidden_size, size
        );
    }
}
