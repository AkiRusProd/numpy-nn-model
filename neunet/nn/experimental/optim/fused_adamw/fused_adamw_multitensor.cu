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
    float* p_ptr = params_list[tensor_idx];
    const float* g_ptr = grads_list[tensor_idx];
    float* m_ptr = exp_avgs_list[tensor_idx];
    float* v_ptr = exp_avg_sqs_list[tensor_idx];

    // Pre-calculate bias correction (same for all threads)
    // Use float (powf) to avoid using double registers
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    // 5. Loop inside chunk
    for (int i = threadIdx.x; i < items_to_process; i += blockDim.x) {
        int idx = start_offset + i;

        // Load
        float p = p_ptr[idx];
        float g = g_ptr[idx];
        float m = m_ptr[idx];
        float v = v_ptr[idx];

        // AdamW Math ---
        
        // Weight Decay (decoupled)
        if (weight_decay != 0.0f) {
            p -= lr * weight_decay * p;
        }

        // Update moments
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        // Bias correction
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;

        // Update weights
        p -= lr * m_hat / (sqrtf(v_hat) + eps);

        // --- Write back ---
        p_ptr[idx] = p;
        m_ptr[idx] = m;
        v_ptr[idx] = v;
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
            
            // --- PLANNING STAGE (On CPU) ---
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
