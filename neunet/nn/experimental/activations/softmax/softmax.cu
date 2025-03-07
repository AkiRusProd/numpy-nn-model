#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

// __inline__ __device__ float warp_reduce_max(float val) {
//     for (int offset = 16; offset > 0; offset /= 2)
//         val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
//     return val;
// }

__inline__ __device__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// __inline__ __device__ float warp_reduce_sum(float val) {
//     for (int offset = 16; offset > 0; offset /= 2)
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     return val;
// }

__inline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

int get_threads_per_block(int slice_size) {
    int max_tpb;
    cudaDeviceGetAttribute(&max_tpb, cudaDevAttrMaxThreadsPerBlock, 0);
    
    int tpb = 32;
    while (tpb < max_tpb && tpb < slice_size) {
        tpb *= 2;
    }
    return min(tpb, max_tpb);
}

__global__ void fused_softmax_forward_kernel(
    float* output, 
    const float* input,
    int num_slices,
    int slice_size,
    int stride,
    int threads_per_block
) {
    const int slice_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int nwarps = threads_per_block / 32;
    
    extern __shared__ float smem[];
    float* slice_max = smem;
    float* slice_sum = smem + nwarps;

    // Calculate multi-dimensional indices using output_stride for slicing
    const int outer_idx = slice_idx / stride;
    const int inner_idx = slice_idx % stride;
    
    // Use input_stride for input and output_stride for output
    const float* slice_input = input + outer_idx * stride * slice_size + inner_idx;
    float* slice_output = output + outer_idx * stride * slice_size + inner_idx;

    // 1. Compute thread-local max
    float thread_max = -INFINITY;
    #pragma unroll
    for (int i = tid; i < slice_size; i += threads_per_block) {
        const int offset = i * stride;
        thread_max = fmaxf(thread_max, slice_input[offset]);
    }
    
    // 2. Warp-level reduction for max
    float warp_max = warp_reduce_max(thread_max);
    
    // 3. Store warp results to shared memory
    if (tid % 32 == 0) {
        slice_max[tid / 32] = warp_max;
    }
    __syncthreads();

    // 4. Final max reduction using first warp
    float block_max = -INFINITY;
    if (tid < 32) {
        float val = (tid < nwarps) ? slice_max[tid] : -INFINITY;
        block_max = warp_reduce_max(val);
    }
    if (tid == 0) {
        slice_max[0] = block_max;
    }
    __syncthreads();
    block_max = slice_max[0];

    // 5. Compute exp and thread-local sum
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = tid; i < slice_size; i += threads_per_block) {
        const int offset = i * stride;
        float val = expf(slice_input[offset] - block_max);
        slice_output[offset] = val;
        thread_sum += val;
    }
    
    // 6. Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // 7. Store warp sums to shared memory
    if (tid % 32 == 0) {
        slice_sum[tid / 32] = warp_sum;
    }
    __syncthreads();

    // 8. Final sum reduction using first warp
    float block_sum = 0.0f;
    if (tid < 32) {
        float val = (tid < nwarps) ? slice_sum[tid] : 0.0f;
        block_sum = warp_reduce_sum(val);
    }
    if (tid == 0) {
        slice_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = slice_sum[0];
    
    // 9. Normalize and write output
    #pragma unroll
    for (int i = tid; i < slice_size; i += threads_per_block) {
        const int offset = i * stride;
        slice_output[offset] /= block_sum;
    }
}

extern "C" {
    DLLEXPORT void cudaSoftmaxForward(
        float* output, 
        const float* input,
        int num_slices,
        int slice_size,
        int stride,
        cudaStream_t stream
    ) {
        int threads_per_block = get_threads_per_block(slice_size);

        dim3 grid(num_slices);
        dim3 block(threads_per_block);
        size_t smem_size = 2 * (threads_per_block / 32) * sizeof(float);
        
        fused_softmax_forward_kernel<<<grid, block, smem_size, stream>>>(
            output, input, num_slices, slice_size, stride, threads_per_block
        );
    }
}

__global__ void fused_softmax_backward_kernel(
    float* grad_x,
    const float* grad,
    const float* f_x,
    int num_slices,
    int slice_size,
    int stride,
    int threads_per_block
) {
    const int slice_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int nwarps = threads_per_block / 32;
    
    extern __shared__ float smem[];
    float* slice_sum = smem;

    // Calculate multi-dimensional indices
    const int outer_idx = slice_idx / stride;
    const int inner_idx = slice_idx % stride;
    
    // Use strides for each tensor
    const float* slice_grad = grad + outer_idx * stride * slice_size + inner_idx;
    const float* slice_fx = f_x + outer_idx * stride * slice_size + inner_idx;
    float* slice_gradx = grad_x + outer_idx * stride * slice_size + inner_idx;

    // 1. Thread-local sum of grad * f_x
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = tid; i < slice_size; i += threads_per_block) {
        const int offset = i * stride;
        thread_sum += slice_grad[offset] * slice_fx[offset];
    }
    
    // 2. Warp-level sum reduction
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // 3. Store warp results to shared memory
    if (tid % 32 == 0) {
        slice_sum[tid / 32] = warp_sum;
    }
    __syncthreads();

    // 4. Final sum reduction using first warp
    float block_sum = 0.0f;
    if (tid < 32) {
        float val = (tid < nwarps) ? slice_sum[tid] : 0.0f;
        block_sum = warp_reduce_sum(val);
    }
    if (tid == 0) {
        slice_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = slice_sum[0];
    
    // 5. Compute (grad[i] - sum) * f_x[i] and write to grad_x
    #pragma unroll
    for (int i = tid; i < slice_size; i += threads_per_block) {
        const int offset = i * stride;
        float g = slice_grad[offset];
        float fx = slice_fx[offset];
        slice_gradx[offset] = (g - block_sum) * fx;
    }
}

extern "C" {
    DLLEXPORT void cudaSoftmaxBackward(
        float* grad_x,
        const float* grad,
        const float* f_x,
        int num_slices,
        int slice_size,
        int stride,
        cudaStream_t stream
    ) {
        int threads_per_block = get_threads_per_block(slice_size);

        dim3 grid(num_slices);
        dim3 block(threads_per_block);
        size_t smem_size = (threads_per_block / 32) * sizeof(float);
        
        fused_softmax_backward_kernel<<<grid, block, smem_size, stream>>>(
            grad_x, grad, f_x, num_slices, slice_size, stride, threads_per_block
        );
    }
}