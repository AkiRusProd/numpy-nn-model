#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

using namespace std;

// CUDA kernel for adding bias to each column in the output matrix C
__global__ void addBiasKernel(float *C, const float *bias, int rowsNum, int colsNum) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rowsNum && col < colsNum) {
         if (bias != nullptr) {
            C[row * colsNum + col] += bias[col];
        }
    }
}

// CUDA kernel for summing gradients for bias
__global__ void sumBiasKernel(float *d_bias, const float *d_output, int rowsNum, int colsNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < colsNum) {
        float sum = 0.0f;
        for (int i = 0; i < rowsNum; i++) {
            sum += d_output[i * colsNum + idx];
        }
        d_bias[idx] = sum;
    }
}

// Global cacheable forward pointers to GPU memory
static float *p_input_fw = nullptr;
static float *p_weights_fw = nullptr;
static float *p_output_fw = nullptr;
static float *p_bias_fw = nullptr;

// Current sizes of cached tensors in forward
static int current_input_fw_rows = 0;
static int current_input_fw_cols = 0;
static int current_weights_fw_rows = 0;
static int current_weights_fw_cols = 0;
static int current_output_fw_rows = 0;
static int current_output_fw_cols = 0;
static int current_bias_fw_size = 0;


// Global cacheable backward pointers to GPU memory
static float *p_input_bw = nullptr;       
static float *p_weights_bw = nullptr;   
static float *p_d_output_bw = nullptr;
static float *p_d_input_bw = nullptr;
static float *p_d_weights_bw = nullptr;
static float *p_d_bias_bw = nullptr;

// Current sizes of cached tensors in backward
static int current_input_bw_rows = 0;
static int current_input_bw_cols = 0;
static int current_weights_bw_rows = 0;
static int current_weights_bw_cols = 0;
static int current_d_output_bw_rows = 0;
static int current_d_output_bw_cols = 0;
static int current_d_input_bw_rows = 0;
static int current_d_input_bw_cols = 0;
static int current_d_weights_bw_rows = 0;
static int current_d_weights_bw_cols = 0;
static int current_d_bias_bw_size = 0;

// Cached handle for cuBLAS
static cublasHandle_t cublas_handle = nullptr;


extern "C" {
    DLLEXPORT void cudaLinearModuleForward(float *input, float *weights, float *bias, float *output, int inputRowsNum, int inputColsNum, int outputColsNum) {

        // Check and allocate memory for p_input
        if (p_input_fw == nullptr || current_input_fw_rows != inputRowsNum || current_input_fw_cols != inputColsNum) {
            cudaFree(p_input_fw);
            cudaMalloc(&p_input_fw, inputRowsNum * inputColsNum * sizeof(float));
            current_input_fw_rows = inputRowsNum;
            current_input_fw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_weights
        if (p_weights_fw == nullptr || current_weights_fw_rows != inputColsNum || current_weights_fw_cols != outputColsNum) {
            cudaFree(p_weights_fw);
            cudaMalloc(&p_weights_fw, inputColsNum * outputColsNum * sizeof(float));
            current_weights_fw_rows = inputColsNum;
            current_weights_fw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_output
        if (p_output_fw == nullptr || current_output_fw_rows != inputRowsNum || current_output_fw_cols != outputColsNum) {
            cudaFree(p_output_fw);
            cudaMalloc(&p_output_fw, inputRowsNum * outputColsNum * sizeof(float));
            current_output_fw_rows = inputRowsNum;
            current_output_fw_cols = outputColsNum;
        }
    
        // Handle bias
        if (bias != nullptr) {
            if (p_bias_fw == nullptr || current_bias_fw_size != outputColsNum) {
                cudaFree(p_bias_fw);
                cudaMalloc(&p_bias_fw, outputColsNum * sizeof(float));
                current_bias_fw_size = outputColsNum;
            }
            cudaMemcpy(p_bias_fw, bias, outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            if (p_bias_fw != nullptr) {
                cudaFree(p_bias_fw);
                p_bias_fw = nullptr; // Explicitly set to nullptr
                current_bias_fw_size = 0;
            }
        }
    
        // Copy data to device
        cudaMemcpy(p_input_fw, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_weights_fw, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
    
        // Perform linear layer operation: output = input * weights + bias
        // C(rowsNum, colsNum) = A(rowsNum, width) * B(width, colsNum) + bias
        if (cublas_handle == nullptr) {
            cublasCreate(&cublas_handle);
        }

        const float alf = 1.0f;
        const float bet = 0.0f;
    
        // matrix multiplication: C = A * B^T
        cublasSgemm(cublas_handle,
            CUBLAS_OP_T,   // Transpose A (A is originally inputRowsNum x inputColsNum in row-major, becomes inputColsNum x inputRowsNum)
            CUBLAS_OP_N,   // No transpose B (B is outputColsNum x inputColsNum in row-major)
            outputColsNum, // m: rows of op(A) and C (outputColsNum)
            inputRowsNum,  // n: columns of op(B) and C (inputRowsNum)
            inputColsNum,  // k: columns of op(A) and rows of op(B) (inputColsNum)
            &alf,
            p_weights_fw,     // A: originally B (outputColsNum x inputColsNum in row-major)
            inputColsNum,  // lda: leading dimension of A (inputColsNum, since row-major)
            p_input_fw,       // B: originally A (inputRowsNum x inputColsNum in row-major)
            inputColsNum,  // ldb: leading dimension of B (inputColsNum, since row-major)
            &bet,
            p_output_fw,      // C: result matrix (inputRowsNum x outputColsNum in row-major)
            outputColsNum);// ldc: leading dimension of C (outputColsNum, since row-major)

        // Add bias to each column of C
        if (bias != nullptr) {
            // C(i, j) += bias(j)
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((inputRowsNum + 15) / 16, (outputColsNum + 15) / 16);
            // Add bias to each row for each column (without race conditions)
            addBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(p_output_fw, p_bias_fw, inputRowsNum, outputColsNum);
            cudaDeviceSynchronize();
        }
    
        // Copy result back to host
        cudaMemcpy(output, p_output_fw, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {
    
        // Check and allocate memory for p_input
        if (p_input_bw == nullptr || current_input_bw_rows != inputRowsNum || current_input_bw_cols != inputColsNum) {
            cudaFree(p_input_bw);
            cudaMalloc(&p_input_bw, inputRowsNum * inputColsNum * sizeof(float));
            current_input_bw_rows = inputRowsNum;
            current_input_bw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_weights
        if (p_weights_bw == nullptr || current_weights_bw_rows != inputColsNum || current_weights_bw_cols != outputColsNum) {
            cudaFree(p_weights_bw);
            cudaMalloc(&p_weights_bw, inputColsNum * outputColsNum * sizeof(float));
            current_weights_bw_rows = inputColsNum;
            current_weights_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_output
        if (p_d_output_bw == nullptr || current_d_output_bw_rows != inputRowsNum || current_d_output_bw_cols != outputColsNum) {
            cudaFree(p_d_output_bw);
            cudaMalloc(&p_d_output_bw, inputRowsNum * outputColsNum * sizeof(float));
            current_d_output_bw_rows = inputRowsNum;
            current_d_output_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_input
        if (p_d_input_bw == nullptr || current_d_input_bw_rows != inputRowsNum || current_d_input_bw_cols != inputColsNum) {
            cudaFree(p_d_input_bw);
            cudaMalloc(&p_d_input_bw, inputRowsNum * inputColsNum * sizeof(float));
            current_d_input_bw_rows = inputRowsNum;
            current_d_input_bw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_d_weights
        if (p_d_weights_bw == nullptr || current_d_weights_bw_rows != inputColsNum || current_d_weights_bw_cols != outputColsNum) {
            cudaFree(p_d_weights_bw);
            cudaMalloc(&p_d_weights_bw, inputColsNum * outputColsNum * sizeof(float));
            current_d_weights_bw_rows = inputColsNum;
            current_d_weights_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_bias
        if (d_bias != nullptr) {
            if (p_d_bias_bw == nullptr || current_d_bias_bw_size != outputColsNum) {
                cudaFree(p_d_bias_bw);
                cudaMalloc(&p_d_bias_bw, outputColsNum * sizeof(float));
                current_d_bias_bw_size = outputColsNum;
            }
        } else {
            if (p_d_bias_bw != nullptr) {
                cudaFree(p_d_bias_bw);
                p_d_bias_bw = nullptr;
                current_d_bias_bw_size = 0;
            }
        }
    
        // Copy data to device
        cudaMemcpy(p_input_bw, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_weights_bw, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d_output_bw, d_output, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
    
        if (cublas_handle == nullptr) {
            cublasCreate(&cublas_handle);
        }
    
        const float alf = 1.0f;
        const float bet = 0.0f;
    
        // Compute d_input = d_output * W^T
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N,     // No transpose (A is inputColsNum x outputColsNum in column-major)
            CUBLAS_OP_N,     // No transpose (B is outputColsNum x inputRowsNum in column-major)
            inputColsNum,    // m: rows of op(A) and C (inputColsNum)
            inputRowsNum,    // n: columns of op(B) and C (inputRowsNum)
            outputColsNum,   // k: columns of op(A) and rows of op(B) (outputColsNum)
            &alf,
            p_weights_bw,       // A: inputColsNum x outputColsNum (column-major)
            inputColsNum,    // lda: rows of A in column-major (inputColsNum)
            p_d_output_bw,      // B: outputColsNum x inputRowsNum (column-major)
            outputColsNum,   // ldb: rows of B in column-major (outputColsNum)
            &bet,
            p_d_input_bw,       // C: inputColsNum x inputRowsNum (column-major)
            inputColsNum);   // ldc: rows of C in column-major (inputColsNum)


        // Compute d_weights = X^T * d_output (corrected for row-major)
        cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,     // No transpose (A is inputColsNum x inputRowsNum in column-major)
                    CUBLAS_OP_T,     // Transpose B (B is outputColsNum x inputRowsNum → becomes inputRowsNum x outputColsNum)
                    inputColsNum,    // m: rows of op(A) and C
                    outputColsNum,   // n: columns of op(B) and C
                    inputRowsNum,    // k: columns of op(A) and rows of op(B)
                    &alf,
                    p_input_bw,         // A: inputColsNum x inputRowsNum (column-major)
                    inputColsNum,    // lda: rows of A in column-major (inputColsNum)
                    p_d_output_bw,      // B: outputColsNum x inputRowsNum (column-major)
                    outputColsNum,   // ldb: rows of B in column-major (outputColsNum)
                    &bet,
                    p_d_weights_bw,     // C: inputColsNum x outputColsNum (column-major)
                    inputColsNum);   // ldc: rows of C in column-major (inputColsNum)

        // Compute d_bias = sum(d_output, axis=0) if d_bias is not nullptr
        // Use custom kernel to sum along rows
        if (d_bias != nullptr && p_d_bias_bw != nullptr) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (outputColsNum + threadsPerBlock - 1) / threadsPerBlock;
            sumBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(p_d_bias_bw, p_d_output_bw, inputRowsNum, outputColsNum);
            cudaDeviceSynchronize();
        }
    
        // Copy results back to host
        cudaMemcpy(d_input, p_d_input_bw, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_weights, p_d_weights_bw, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (d_bias != nullptr && p_d_bias_bw != nullptr) {
            cudaMemcpy(d_bias, p_d_bias_bw, outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    DLLEXPORT void cleanupCudaMemory() {
        // Freeing forward caches
        cudaFree(p_input_fw); p_input_fw = nullptr;
        cudaFree(p_weights_fw); p_weights_fw = nullptr;
        cudaFree(p_output_fw); p_output_fw = nullptr;
        cudaFree(p_bias_fw); p_bias_fw = nullptr;
        current_input_fw_rows = current_input_fw_cols = 0;
        current_weights_fw_rows = current_weights_fw_cols = 0;
        current_output_fw_rows = current_output_fw_cols = 0;
        current_bias_fw_size = 0;

        // Freeing backward caches
        cudaFree(p_input_bw); p_input_bw = nullptr;
        cudaFree(p_weights_bw); p_weights_bw = nullptr;
        cudaFree(p_d_output_bw); p_d_output_bw = nullptr;
        cudaFree(p_d_input_bw); p_d_input_bw = nullptr;
        cudaFree(p_d_weights_bw); p_d_weights_bw = nullptr;
        cudaFree(p_d_bias_bw); p_d_bias_bw = nullptr;
        current_input_bw_rows = current_input_bw_cols = 0;
        current_weights_bw_rows = current_weights_bw_cols = 0;
        current_d_output_bw_rows = current_d_output_bw_cols = 0;
        current_d_input_bw_rows = current_d_input_bw_cols = 0;
        current_d_weights_bw_rows = current_d_weights_bw_cols = 0;
        current_d_bias_bw_size = 0;

        // Destroying cuBLAS handle
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }
}
