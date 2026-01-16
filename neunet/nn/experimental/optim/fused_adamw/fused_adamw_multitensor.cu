#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


// 65536 elements = 256 KB of data. Optimal for load balancing.
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh
constexpr int CHUNK_SIZE = 65536; 
constexpr int BLOCK_SIZE = 512;

// Naive realization
// __global__ void adamw_multi_tensor_kernel(
//     int n_total_blocks,
//     float** params_list,       // Pointers to data in VRAM
//     const float** grads_list,
//     float** exp_avgs_list,
//     float** exp_avg_sqs_list,
//     int* sizes,                // Tensor sizes
//     int* block_to_tensor,      // Map: which tensor the block processes
//     int* block_to_chunk,       // Map: which chunk of the tensor
//     float lr,
//     float beta1,
//     float beta2,
//     float eps,
//     float weight_decay,
//     int step
// ) {
//     // 1. Global block ID
//     int global_block_idx = blockIdx.x;
//     if (global_block_idx >= n_total_blocks) return;

//     // 2. Read metadata from global memory
//     int tensor_idx = block_to_tensor[global_block_idx];
//     int chunk_idx = block_to_chunk[global_block_idx];
//     int tensor_size = sizes[tensor_idx];

//     // 3. Calculate work boundaries for this block
//     int start_offset = chunk_idx * CHUNK_SIZE;
//     int items_to_process = min(CHUNK_SIZE, tensor_size - start_offset);

//     // 4. Get necessary pointers for the specific tensor
//     float* p_ptr = params_list[tensor_idx];
//     const float* g_ptr = grads_list[tensor_idx];
//     float* m_ptr = exp_avgs_list[tensor_idx];
//     float* v_ptr = exp_avg_sqs_list[tensor_idx];

//     // Pre-calculate bias correction (same for all threads)
//     // Use float (powf) to avoid using double registers
//     float bias_correction1 = 1.0f - powf(beta1, (float)step);
//     float bias_correction2 = 1.0f - powf(beta2, (float)step);

//     // 5. Loop inside chunk
//     for (int i = threadIdx.x; i < items_to_process; i += blockDim.x) {
//         int idx = start_offset + i;

//         float p = p_ptr[idx];
//         float g = g_ptr[idx];
//         float m = m_ptr[idx];
//         float v = v_ptr[idx];

//         // Weight Decay (decoupled)
//         if (weight_decay != 0.0f) {
//             p -= lr * weight_decay * p;
//         }

//         m = beta1 * m + (1.0f - beta1) * g;
//         v = beta2 * v + (1.0f - beta2) * g * g;

//         float m_hat = m / bias_correction1;
//         float v_hat = v / bias_correction2;

//         p -= lr * m_hat / (sqrtf(v_hat) + eps);

//         p_ptr[idx] = p;
//         m_ptr[idx] = m;
//         v_ptr[idx] = v;
//     }
// }

// Optimized with float4 vectorization realization
__device__ __forceinline__ void adam_update_elem(
    float& p, float g, float& m, float& v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    // Weight Decay (decoupled)
    if (weight_decay != 0.0f) {
        p -= lr * weight_decay * p;
    }

    // Adam math
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    p -= lr * m_hat / (sqrtf(v_hat) + eps);
}

__global__ void adamw_multi_tensor_kernel(
    int n_total_blocks,
    float** params_list,       // Pointers to data in VRAM
    const float** grads_list,
    float** exp_avgs_list,
    float** exp_avg_sqs_list,
    int* sizes,                // Tensor sizes
    int* block_to_tensor,      // Map: which tensor the block processes
    int* block_to_chunk,       // Map: which chunk of the tensor
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step
) {
    // 1. Global block ID
    int global_block_idx = blockIdx.x;
    if (global_block_idx >= n_total_blocks) return;

    // 2. Read metadata from global memory
    int tensor_idx = block_to_tensor[global_block_idx];
    int chunk_idx = block_to_chunk[global_block_idx];
    int tensor_size = sizes[tensor_idx];

    // 3. Calculate work boundaries for this block
    int start_offset = chunk_idx * CHUNK_SIZE;
    int items_to_process = min(CHUNK_SIZE, tensor_size - start_offset);

    // 4. Get necessary pointers for the specific tensor
    float* p_ptr = params_list[tensor_idx] + start_offset;
    const float* g_ptr = grads_list[tensor_idx] + start_offset;
    float* m_ptr = exp_avgs_list[tensor_idx] + start_offset;
    float* v_ptr = exp_avg_sqs_list[tensor_idx] + start_offset;

    // Pre-calculate bias correction (same for all threads)
    // Use float (powf) to avoid using double registers
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);


    // VECTORIZED PATH (FLOAT4)
    // Alignment check: address must be 16-byte aligned (4 floats)
    // start_offset is usually a multiple of 65536, so if the array start is aligned, the chunk is aligned too.
    bool aligned = ((uint64_t)p_ptr % 16 == 0) &&
                   ((uint64_t)g_ptr % 16 == 0) &&
                   ((uint64_t)m_ptr % 16 == 0) &&
                   ((uint64_t)v_ptr % 16 == 0);

    int processed = 0;

    if (aligned) {
        int n_vectors = items_to_process / 4; // Number of full groups of 4
        
        // Cast pointers to float4
        float4* p4_ptr = reinterpret_cast<float4*>(p_ptr);
        const float4* g4_ptr = reinterpret_cast<const float4*>(g_ptr);
        float4* m4_ptr = reinterpret_cast<float4*>(m_ptr);
        float4* v4_ptr = reinterpret_cast<float4*>(v_ptr);

        // Stride is blockDim.x, but each thread processes 1 float4 (4 floats)
        for (int i = threadIdx.x; i < n_vectors; i += blockDim.x) {
            // Vector loads (1 instruction per vector instead of 4)
            float4 p4 = p4_ptr[i];
            float4 g4 = g4_ptr[i];
            float4 m4 = m4_ptr[i];
            float4 v4 = v4_ptr[i];

            // Process components
            adam_update_elem(p4.x, g4.x, m4.x, v4.x, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_update_elem(p4.y, g4.y, m4.y, v4.y, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_update_elem(p4.z, g4.z, m4.z, v4.z, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
            adam_update_elem(p4.w, g4.w, m4.w, v4.w, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);

            // Store results
            p4_ptr[i] = p4;
            m4_ptr[i] = m4;
            v4_ptr[i] = v4;
        }
        
        processed = n_vectors * 4;
    }

    // SCALAR PATH (CLEANUP)
    // Process the remainder (if size is not divisible by 4) or all if unaligned
    for (int i = processed + threadIdx.x; i < items_to_process; i += blockDim.x) {
        float p = p_ptr[i];
        float g = g_ptr[i];
        float m = m_ptr[i];
        float v = v_ptr[i];

        adam_update_elem(p, g, m, v, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);

        p_ptr[i] = p;
        m_ptr[i] = m;
        v_ptr[i] = v;
    }
}

// HOST CLASS (MEMORY MANAGER)
// This class is needed to avoid calling cudaMalloc every time.
// It stores metadata buffers and recreates them only if the model changes.
class FusedOptimizer {
    private:
        // Buffers on GPU (Device Pointers)
        float **d_params = nullptr, **d_exp_avgs = nullptr, **d_exp_avg_sqs = nullptr;
        const float **d_grads = nullptr;
        int *d_sizes = nullptr, *d_block_to_tensor = nullptr, *d_block_to_chunk = nullptr;
    
        // Cached sizes
        size_t cached_num_tensors = 0;
        size_t cached_total_blocks = 0;    

public:
    ~FusedOptimizer() {
        cleanup();
    }

    void cleanup() {
        if (d_params) cudaFree(d_params);
        if (d_grads) cudaFree((void*)d_grads);
        if (d_exp_avgs) cudaFree(d_exp_avgs);
        if (d_exp_avg_sqs) cudaFree(d_exp_avg_sqs);
        if (d_sizes) cudaFree(d_sizes);
        if (d_block_to_tensor) cudaFree(d_block_to_tensor);
        if (d_block_to_chunk) cudaFree(d_block_to_chunk);
        
        d_params = nullptr;
        cached_num_tensors = 0;
    }

    void step(
        int num_tensors,
        float** h_params,       // Array of pointers on CPU
        const float** h_grads,
        float** h_exp_avgs,
        float** h_exp_avg_sqs,
        int* h_sizes,
        float lr, float beta1, float beta2, float eps, float weight_decay, int step_num,
        cudaStream_t stream
    ) {
        // 1. Check if metadata needs to be recreated
        // (in real tasks num_tensors changes rarely, usually only at start)
        if (num_tensors != cached_num_tensors) {
            cleanup();

            // PLANNING STAGE (On CPU)
            std::vector<int> map_tensor;
            std::vector<int> map_chunk;
            
            for (int t = 0; t < num_tensors; ++t) {
                int size = h_sizes[t];
                int chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
                for (int c = 0; c < chunks; ++c) {
                    map_tensor.push_back(t);
                    map_chunk.push_back(c);
                }
            }
            
            cached_num_tensors = num_tensors;
            cached_total_blocks = map_tensor.size();

             // ALLOCATING MEMORY FOR METADATA
            cudaMalloc(&d_params, num_tensors * sizeof(float*));
            cudaMalloc(&d_grads, num_tensors * sizeof(float*));
            cudaMalloc(&d_exp_avgs, num_tensors * sizeof(float*));
            cudaMalloc(&d_exp_avg_sqs, num_tensors * sizeof(float*));
            cudaMalloc(&d_sizes, num_tensors * sizeof(int));
            
            cudaMalloc(&d_block_to_tensor, cached_total_blocks * sizeof(int));
            cudaMalloc(&d_block_to_chunk, cached_total_blocks * sizeof(int));

            // Copy immutable chunk maps
            cudaMemcpyAsync(d_block_to_tensor, map_tensor.data(), map_tensor.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_block_to_chunk, map_chunk.data(), map_chunk.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sizes, h_sizes, num_tensors * sizeof(int), cudaMemcpyHostToDevice, stream);
        }

        // 2. Copy pointer lists (they may change if PyTorch reallocates tensors, 
        // but most often addresses are stable. For reliability, we always copy - it's fast, only ~4-8KB of data).
        cudaMemcpyAsync(d_params, h_params, num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_grads, h_grads, num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_exp_avgs, h_exp_avgs, num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_exp_avg_sqs, h_exp_avg_sqs, num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);

        // 3. Kernel launch
        adamw_multi_tensor_kernel<<<cached_total_blocks, BLOCK_SIZE, 0, stream>>>(
            cached_total_blocks,
            d_params, d_grads, d_exp_avgs, d_exp_avg_sqs,
            d_sizes,
            d_block_to_tensor,
            d_block_to_chunk,
            lr, beta1, beta2, eps, weight_decay, step_num
        );
    }
};

// EXTERNAL API
extern "C" {
    // Create optimizer instance
    DLLEXPORT void* CreateFusedOptimizer() {
        return new FusedOptimizer();
    }

    // Destroy instance
    DLLEXPORT void DestroyFusedOptimizer(void* optimizer) {
        delete static_cast<FusedOptimizer*>(optimizer);
    }

    // Optimization step
    DLLEXPORT void FusedAdamWStep(
        void* optimizer,
        int num_tensors,
        float** params,
        const float** grads,
        float** exp_avgs,
        float** exp_avg_sqs,
        int* sizes,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        int step,
        cudaStream_t stream
    ) {
        auto* opt = static_cast<FusedOptimizer*>(optimizer);
        opt->step(
            num_tensors, params, grads, exp_avgs, exp_avg_sqs, sizes,
            lr, beta1, beta2, eps, weight_decay, step, stream
        );
    }
}
