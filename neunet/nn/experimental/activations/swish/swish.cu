#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swish_forward_kernel(
    float* output, 
    const float* input,
    float beta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * sigmoid(beta * x);
    }
}

__global__ void swish_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* input,
    float beta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float s = sigmoid(beta * x);
        float f = x * s;
        // f'(x) = s + beta * f * (1 - s)
        float d_swish = s + beta * f * (1.0f - s);
        grad_input[idx] = grad_output[idx] * d_swish;

        // f'(x) = f * beta + s * (1 - f * b)
        // float d_swish = f * beta + s * (1 - f * beta);
        // grad_input[idx] = grad_output[idx] * d_swish;
    }
}

extern "C" {
    DLLEXPORT void cudaSwishForward(
        float* output, 
        const float* input,
        float beta,
        int size,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

        swish_forward_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            output, input, beta, size
        );
    }

    DLLEXPORT void cudaSwishBackward(
        float* grad_input,
        const float* grad_output,
        const float* input,
        float beta,
        int size,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

        swish_backward_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            grad_input, grad_output, input, beta, size
        );
    }
}
