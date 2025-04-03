#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <stdio.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

// https://arxiv.org/pdf/1805.02867
// https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py
// https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py

__global__ void cross_entropy_forward_backward_kernel(
    float* logits_ptr ,
    float* loss_ptr,
    float* lse_ptr,
    int* labels_ptr,
    int logits_stride,
    int ignore_index,
    int n_cols,
    char reduction, // 'n', 'm' or 's' // 'n' - no reduction, 'm' - mean reduction, 's' - sum reduction
    int n_non_ignore // number of non-ignored indices in the batch (used for mean reduction)
) {

    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    int label_idx = labels_ptr[row];
    
    if (label_idx == ignore_index) {
        // Zero out the gradient (logits_ptr) for this sample
        int base = row * logits_stride;
        for (int i = 0; i < n_cols; i += blockDim.x) {
            int idx = i + tid;
            if (idx < n_cols) {
                logits_ptr[base + idx] = 0.0f;
            }
        }
        // Set loss and lse to zero for this sample (Im not sure if this is needed)
        if (tid == 0) {
            loss_ptr[row] = 0.0f;
        }
        return;
    }

    // Shared variables for online softmax computation
    __shared__ float shared_m, shared_d;

    // Initialize shared variables
    if (tid == 0) {
        shared_m = -FLT_MAX;
        shared_d = 0.0f;
    }
    __syncthreads();

    // Process each chunk of the logits with one block
    // Each block processes one row of logits [BS, V]
    for (int chunk_start = 0; chunk_start < n_cols; chunk_start += blockDim.x) {
        const int idx = chunk_start + tid;
        const float x = (idx < n_cols) ? logits_ptr[row * logits_stride + idx] : -FLT_MAX;

        // 1. Block-wide max reduction for current chunk
        shared_max[tid] = x;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
            __syncthreads();
        }
        const float chunk_max = shared_max[0];

        // 2. Compute new max and update sum of exponents
        const float m_new = fmaxf(shared_m, chunk_max);

        // 3. Compute exp(x - m_new) for current chunk
        const float exp_x = (idx < n_cols) ? expf(x - m_new) : 0.0f;

        // 4. Block-wide sum reduction for exponents
        shared_sum[tid] = exp_x;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }
        const float chunk_sum = shared_sum[0];

        // 5. Update shared variables
        if (threadIdx.x == 0) {
            const float factor = expf(shared_m - m_new);
            shared_d = shared_d * factor + chunk_sum;
            shared_m = m_new;
        }
        __syncthreads();
    }

    // Compute log-sum-exp (LSE)
    // Note: we do it in tid=0, because each block processes one row of logits [BS, V]
    // And in this case each block processes one value in loss [BS,] and lse [BS,], so we dont need to use other threads
    if (tid == 0) {
        float lse = 0.0f;
        if (shared_d <= 0.0f) {
            lse = -INFINITY; // Protection against log(0) when calculating LSE
        } else {
            lse = shared_m + logf(shared_d); 
        }
        lse_ptr[row] = lse;
   
        // Calculate loss
        float loss = 0.0f;
        if (label_idx >= 0 && label_idx < n_cols) {
            const float logits_label = logits_ptr[row * logits_stride + label_idx];

            loss = lse - logits_label;
        }

        // Store loss
        loss_ptr[row] = loss;
    }

    __syncthreads(); // Ensure all threads have completed before proceeding

    // Load the precomputed log-sum-exp (LSE) value
    float lse = lse_ptr[row];
    
    // Determine scaling factor based on reduction mode
    float scale = (reduction == 'm') ? (1.0f / n_non_ignore) : 1.0f;

    // Loop with coalesced access pattern
    for (int chunk_start = 0; chunk_start < n_cols; chunk_start += blockDim.x) {
        const int idx = chunk_start + tid;
        const float x = logits_ptr[row * logits_stride + idx];
        if (idx >= n_cols) {
            break; // Avoid out-of-bounds access. Note: this is needed for coalesced accesses
        }

        float probs = expf(x - lse); // Softmax
                
        // Apply correction for target class
        if (idx == label_idx) {
            probs -= 1.0f;
        }

        // Apply reduction scaling
        probs *= scale;
        logits_ptr[row * logits_stride + idx] = probs; // Update logits in-place
    }

    // Alternative approach: Loop with uncoalesced access pattern (less efficient)
    // for (int i = tid; i < n_cols; i += blockDim.x) {
    //
    //     float x = logits_ptr[row * logits_stride + i];
    //     float probs = expf(x - lse);
        
    //     if (i == label_idx) {
    //         probs -= 1.0f;
    //     }
        
    //     probs *= scale;
    //     logits_ptr[row * logits_stride + i] = probs;
    // }


}



extern "C" {
    /**
     * @brief CUDA kernel for computing the forward and backward pass of the cross-entropy loss.
     *
     * @param logits_ptr Pointer to the logits matrix [rows, n_cols] = [batch_size * seq_len, vocab_size].
     * @param loss_ptr Pointer to the output loss array [rows] = [batch_size * seq_len].
     * @param lse_ptr Pointer to the log-sum-exp (LSE) array [rows] = [batch_size * seq_len].
     * @param labels_ptr Pointer to the ground truth labels array [rows] = [batch_size * seq_len].
     * @param logits_stride Stride of the logits matrix (typically vocab_size).
     * @param ignore_index Index to ignore in the labels (e.g., -100 for padding tokens).
     * @param class_start_idx Starting index for classes (used in tensor parallelism, always 0 here, because this kernel is not optimized for it).
     * @param n_rows Number of rows in the logits matrix (batch_size * seq_len).
     * @param n_cols Number of columns in the logits matrix (vocab_size).
     * @param reduction Reduction mode: 'n' - no reduction, 'm' - mean reduction, 's' - sum reduction.
     * @param n_non_ignore Number of non-ignored indices in the batch (used for mean reduction).
     */
    DLLEXPORT void cudaCrossEntropyForwardBackward(
        float* logits_ptr,
        float* loss_ptr,
        float* lse_ptr,
        int* labels_ptr,
        int logits_stride,
        int ignore_index,
        int n_rows,
        int n_cols,
        char reduction,
        int n_non_ignore
    ) {
        // Explanation:
        // Number of blocks = number of examples in batch
        // (1 example is processed by 1 block)

        // blocks_per_grid = n_rows: Each block processes one row of the logits matrix.
        // threads_per_block = 256: Each block has 256 threads for parallel processing.
        // shared_mem_size = 2 * threads_per_block * sizeof(float): Shared memory for max and sum reductions.

        int blocks_per_grid = n_rows;
        int threads_per_block = 1024;

        size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
                
        cross_entropy_forward_backward_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            logits_ptr,
            loss_ptr,
            lse_ptr,
            labels_ptr,
            logits_stride,
            ignore_index,
            n_cols,
            reduction,
            n_non_ignore
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(1);
       }
    }
}
