#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

__global__ void rms_norm_forward_kernel(
    const float* X,    
    const float* weight, 
    const float* bias,  
    float* Y,          
    float* X_std,   // allocated in the forward pass, used in the backward pass
    float* X_norm, // allocated in the forward pass, used in the backward pass
    int n_rows,
    int n_cols,                  
    float eps) { 
    
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= n_rows) return;

    const float* x_row = X + row * n_cols;
    float* y_row = Y + row * n_cols;
    float* x_norm_row = X_norm + row * n_cols;

    // 1. Summation of squares for the row
    // sum_sq is a partial sum of squares calculated by a separate thread for its part of the row elements. 
    // For example:
    //  If there are 256 threads in a block (blockDim.x = 256), and the row has 1024 elements (n_cols = 1024), then:
    //  Each thread processes 1024 / 256 = 4 elements.
    //  Thread 0 sums elements 0, 256, 512, 768.
    //  Thread 1 sums elements 1, 257, 513, 769.
    //  And so on.
    // Result: Each thread has its own sum_sq, covering only a part of the row.

    // Note: this is loop with uncoalesced access pattern (less efficient)
    float sum_sq = 0.0f;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        if (i < n_cols) {
            float val = x_row[i];
            sum_sq += val * val;
        }
    }

    // Collect all partial amounts in one place for reduction.
    // Each thread writes its partial sum to shared memory.
    // shared[] — array in shared memory of the block (fast access for all threads of the block).
    shared[tid] = sum_sq;
    __syncthreads();

    // Sum all partial sums into one result (Parallel Reduction). 
    // The final result is in shared[0].
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // 2. Calculating the standard deviation
    // The standard deviation is calculated in the first thread (tid == 0) and stored in X_std[row].
    if (tid == 0) {
        float mean_sq = shared[0] / n_cols;
        X_std[row] = sqrtf(mean_sq);
    }
    __syncthreads();

    // 3. Normalization and scaling
    // Note: this is loop with uncoalesced access pattern (less efficient)
    float current_X_std = X_std[row] + eps;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        if (i < n_cols) {
            x_norm_row[i] = x_row[i] / current_X_std;
            y_row[i] = x_norm_row[i] * weight[i];
            if (bias != nullptr) {
                y_row[i] += bias[i];
            }
        }
    }
}

extern "C" {
    DLLEXPORT void RMSNormForward(
        /**
        * @brief RMSNorm forward kernel.
        * @param X Pointer to input tensor [rows, n_cols].
        * @param weight Pointer to weights [n_cols].
        * @param bias Pointer to biases [n_cols] (can be nullptr).
        * @param Y Pointer to output tensor [rows, n_cols].
        * @param X_std Pointer to standard deviation [rows].
        * @param X_norm Pointer to normalized input [rows, n_cols].
        * @param n_rows Number of rows (batch * seq_len).
        * @param n_cols Feature dimension.
        * @param eps Small value for numerical stability.
        * @param stream CUDA stream for asynchronous execution (default is 0).
        * @note The kernel computes the RMS normalization of the input tensor X, scales it with weights, and adds biases if provided.
        */
        const float* X,
        const float* weight,
        const float* bias,
        float* Y,
        float* X_std,
        float* X_norm,
        int n_rows,
        int n_cols,
        float eps,
        cudaStream_t stream = 0) {
        

        // Explanation:
        // Number of blocks = number of rows (n_rows): Each block processes one row of the input tensor.

        // blocks_per_grid = n_rows: Each block processes one row of the matrix.
        // threads_per_block = 1024: Each block has 1024 threads for parallel processing.
        // shared_mem_size = threads_per_block * sizeof(float): Shared memory for block-wide sum of squares reduction. 
        
        dim3 grid(n_rows);       // One block per row
        int threads_per_block = 1024;
        size_t shared_mem_size = threads_per_block * sizeof(float); // for block-wide reductions

        rms_norm_forward_kernel<<<grid, threads_per_block, shared_mem_size, stream>>>(
            X, weight, bias, Y, X_std, X_norm, n_rows, n_cols, eps
        );
    }
}


__global__ void rms_norm_backward_kernel(
    const float* grad_Y, 
    const float* X,           
    const float* weight, 
    const float* X_std,
    float* grad_X,
    int n_rows,
    int n_cols
) {
    
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= n_rows) return;

    const float* grad_out_row = grad_Y + row * n_cols;
    const float* x_row = X + row * n_cols;
    float* grad_x_row = grad_X + row * n_cols;

    float X_std_sq = X_std[row] * X_std[row];

    // 1. Sum reduction of (dX_hat * X / X_std) for a row.
    // The same logic as in the forward pass.
    float sum_val = 0.0f;
    for (int j = tid; j < n_cols; j += blockDim.x) {
        if (j < n_cols){
            float wj = weight[j];
            float goj = grad_out_row[j];
            float xj = x_row[j];
            sum_val += wj * goj * xj / X_std[row];
        }
    }

    shared[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float sum_dXhat_X_Xstd = shared[0];
    __syncthreads();

    // 2. Gradient calculation
    float N = n_cols;
    for (int j = tid; j < n_cols; j += blockDim.x) {
        if (j < n_cols){
            float wj = weight[j];
            float goj = grad_out_row[j];
            float xj = x_row[j];

            // Input gradient calculation
            float dX_hat = wj * goj;
            float term1 = dX_hat * X_std[row];
            float term2 = xj * (sum_dXhat_X_Xstd / N);
            grad_x_row[j] = (term1 - term2) / X_std_sq;
        }
    }
}

__global__ void rms_norm_backward_grad_bias_kernel(
    float* grad_bias,
    const float* grad_Y,
    int n_rows,
    int n_cols) {

    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx >= n_cols) return;

    float sum = 0.0f;
    for (int row = 0; row < n_rows; ++row) {
        sum += grad_Y[row * n_cols + feature_idx];
    }
    grad_bias[feature_idx] = sum;
}

__global__ void rms_norm_backward_grad_weight_kernel(
    const float* grad_Y, const float* X, const float* X_norm,
    float* grad_weight, int n_rows, int n_cols) 
{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx >= n_cols) return;

    float sum = 0.0f;
    for (int row = 0; row < n_rows; row++) {
        // float X_norm = X[row * n_cols + feature_idx] / (X_std[row] + eps);
        sum += grad_Y[row * n_cols + feature_idx] * X_norm[row * n_cols + feature_idx];
    }
    
    grad_weight[feature_idx] = sum;
}


extern "C" {
    DLLEXPORT void RMSNormBackward(
        /**
        * @brief RMSNorm backward kernel.
        * @param grad_Y Pointer to gradient from the next layer [n_rows, n_cols].
        * @param X Pointer to input tensor [n_rows, n_cols].
        * @param weight Pointer to weights [n_cols].
        * @param X_std Pointer to standard deviation from forward pass [n_rows].
        * @param X_norm Pointer to normalized input from forward pass [n_rows, n_cols].
        * @param grad_X Pointer to gradient of input [n_rows, n_cols].
        * @param grad_weight Pointer to gradient of weights [n_cols].
        * @param grad_bias Pointer to gradient of biases [n_cols] (can be nullptr).
        * @param n_cols Feature dimension.
        * @param n_rows Number of rows (batch * seq_len).
        * @param stream CUDA stream for asynchronous execution (default is 0).
        * @note The kernel computes the gradients for the input, weights, and biases based on the gradients from the next layer.
        */
        const float* grad_Y,
        const float* X,           
        const float* weight,      
        const float* X_std,     
        const float* X_norm,
        float* grad_X,       
        float* grad_weight,      
        float* grad_bias,  
        int n_rows,      
        int n_cols,                                     
        cudaStream_t stream = 0) {

        
        dim3 grid(n_rows); 
        int threads_per_block = 1024;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        rms_norm_backward_kernel<<<grid, threads_per_block, shared_mem_size, stream>>>(
            grad_Y, X, weight, X_std, grad_X,
            n_rows, n_cols
        );

        // int threads_per_block = 256;
        int grid_size = (n_cols + threads_per_block - 1) / threads_per_block;
        rms_norm_backward_grad_weight_kernel<<<grid_size, threads_per_block, 0, stream>>>(
            grad_Y, X, X_norm, grad_weight,  n_rows, n_cols 
        );
        if (grad_bias != nullptr) {
            rms_norm_backward_grad_bias_kernel<<<grid_size, threads_per_block, 0, stream>>>(
                grad_bias, grad_Y, n_rows, n_cols
            );
        }
    }
}