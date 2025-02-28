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
static cublasHandle_t cublas_handle_forward = nullptr;
static cublasHandle_t cublas_handle_backward = nullptr;

// CUDA streams for asynchronous execution
static cudaStream_t forwardStream = nullptr;
static cudaStream_t backwardStream = nullptr;

extern "C" {
    DLLEXPORT void cudaLinearModuleForward(float *input, float *weights, float *bias, float *output, int inputRowsNum, int inputColsNum, int outputColsNum) {
        // Create forward stream if not exists
        if (forwardStream == nullptr) {
            cudaStreamCreate(&forwardStream);
            cublasCreate(&cublas_handle_forward);
            cublasSetStream(cublas_handle_forward, forwardStream);
        }

        // Check and allocate memory for p_input
        if (p_input_fw == nullptr || current_input_fw_rows != inputRowsNum || current_input_fw_cols != inputColsNum) {
            if (p_input_fw != nullptr) {
                cudaFreeAsync(p_input_fw, forwardStream);
            }
            cudaMallocAsync(&p_input_fw, inputRowsNum * inputColsNum * sizeof(float), forwardStream);
            current_input_fw_rows = inputRowsNum;
            current_input_fw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_weights
        if (p_weights_fw == nullptr || current_weights_fw_rows != inputColsNum || current_weights_fw_cols != outputColsNum) {
            if (p_weights_fw != nullptr) {
                cudaFreeAsync(p_weights_fw, forwardStream);
            }
            cudaMallocAsync(&p_weights_fw, inputColsNum * outputColsNum * sizeof(float), forwardStream);
            current_weights_fw_rows = inputColsNum;
            current_weights_fw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_output
        if (p_output_fw == nullptr || current_output_fw_rows != inputRowsNum || current_output_fw_cols != outputColsNum) {
            if (p_output_fw != nullptr) {
                cudaFreeAsync(p_output_fw, forwardStream);
            }
            cudaMallocAsync(&p_output_fw, inputRowsNum * outputColsNum * sizeof(float), forwardStream);
            current_output_fw_rows = inputRowsNum;
            current_output_fw_cols = outputColsNum;
        }
    
        // Handle bias
        if (bias != nullptr) {
            if (p_bias_fw == nullptr || current_bias_fw_size != outputColsNum) {
                if (p_bias_fw != nullptr) {
                    cudaFreeAsync(p_bias_fw, forwardStream);
                }
                cudaMallocAsync(&p_bias_fw, outputColsNum * sizeof(float), forwardStream);
                current_bias_fw_size = outputColsNum;
            }
            cudaMemcpyAsync(p_bias_fw, bias, outputColsNum * sizeof(float), cudaMemcpyHostToDevice, forwardStream);
        } else {
            if (p_bias_fw != nullptr) {
                cudaFreeAsync(p_bias_fw, forwardStream);
                p_bias_fw = nullptr;
                current_bias_fw_size = 0;
            }
        }
    
        // Copy data to device asynchronously
        cudaMemcpyAsync(p_input_fw, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice, forwardStream);
        cudaMemcpyAsync(p_weights_fw, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice, forwardStream);
    
        // Perform linear layer operation: output = input * weights + bias
        cublasSetStream(cublas_handle_forward, forwardStream);

        const float alf = 1.0f;
        const float bet = 0.0f;
    
        // matrix multiplication: C = A * B^T
        cublasSgemm(cublas_handle_forward,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            outputColsNum,
            inputRowsNum,
            inputColsNum,
            &alf,
            p_weights_fw,
            inputColsNum,
            p_input_fw,
            inputColsNum,
            &bet,
            p_output_fw,
            outputColsNum);

        // Add bias to each column of C
        if (bias != nullptr) {
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((inputRowsNum + 15) / 16, (outputColsNum + 15) / 16);
            addBiasKernel<<<blocksPerGrid, threadsPerBlock, 0, forwardStream>>>(p_output_fw, p_bias_fw, inputRowsNum, outputColsNum);
        }
    
        // Copy result back to host asynchronously
        cudaMemcpyAsync(output, p_output_fw, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost, forwardStream);
    
        // Synchronize the stream to ensure all operations are completed
        cudaStreamSynchronize(forwardStream);
    }
    
    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {
    
        // Create backward stream if not exists
        if (backwardStream == nullptr) {
            cudaStreamCreate(&backwardStream);
            cublasCreate(&cublas_handle_backward);
            cublasSetStream(cublas_handle_backward, backwardStream);
        }

        // Check and allocate memory for p_input
        if (p_input_bw == nullptr || current_input_bw_rows != inputRowsNum || current_input_bw_cols != inputColsNum) {
            if (p_input_bw != nullptr) {
                cudaFreeAsync(p_input_bw, backwardStream);
            }
            cudaMallocAsync(&p_input_bw, inputRowsNum * inputColsNum * sizeof(float), backwardStream);
            current_input_bw_rows = inputRowsNum;
            current_input_bw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_weights
        if (p_weights_bw == nullptr || current_weights_bw_rows != inputColsNum || current_weights_bw_cols != outputColsNum) {
            if (p_weights_bw != nullptr) {
                cudaFreeAsync(p_weights_bw, backwardStream);
            }
            cudaMallocAsync(&p_weights_bw, inputColsNum * outputColsNum * sizeof(float), backwardStream);
            current_weights_bw_rows = inputColsNum;
            current_weights_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_output
        if (p_d_output_bw == nullptr || current_d_output_bw_rows != inputRowsNum || current_d_output_bw_cols != outputColsNum) {
            if (p_d_output_bw != nullptr) {
                cudaFreeAsync(p_d_output_bw, backwardStream);
            }
            cudaMallocAsync(&p_d_output_bw, inputRowsNum * outputColsNum * sizeof(float), backwardStream);
            current_d_output_bw_rows = inputRowsNum;
            current_d_output_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_input
        if (p_d_input_bw == nullptr || current_d_input_bw_rows != inputRowsNum || current_d_input_bw_cols != inputColsNum) {
            if (p_d_input_bw != nullptr) {
                cudaFreeAsync(p_d_input_bw, backwardStream);
            }
            cudaMallocAsync(&p_d_input_bw, inputRowsNum * inputColsNum * sizeof(float), backwardStream);
            current_d_input_bw_rows = inputRowsNum;
            current_d_input_bw_cols = inputColsNum;
        }
    
        // Check and allocate memory for p_d_weights
        if (p_d_weights_bw == nullptr || current_d_weights_bw_rows != inputColsNum || current_d_weights_bw_cols != outputColsNum) {
            if (p_d_weights_bw != nullptr) {
                cudaFreeAsync(p_d_weights_bw, backwardStream);
            }
            cudaMallocAsync(&p_d_weights_bw, inputColsNum * outputColsNum * sizeof(float), backwardStream);
            current_d_weights_bw_rows = inputColsNum;
            current_d_weights_bw_cols = outputColsNum;
        }
    
        // Check and allocate memory for p_d_bias
        if (d_bias != nullptr) {
            if (p_d_bias_bw == nullptr || current_d_bias_bw_size != outputColsNum) {
                if (p_d_bias_bw != nullptr) {
                    cudaFreeAsync(p_d_bias_bw, backwardStream);
                }
                cudaMallocAsync(&p_d_bias_bw, outputColsNum * sizeof(float), backwardStream);
                current_d_bias_bw_size = outputColsNum;
            }
        } else {
            if (p_d_bias_bw != nullptr) {
                cudaFreeAsync(p_d_bias_bw, backwardStream);
                p_d_bias_bw = nullptr;
                current_d_bias_bw_size = 0;
            }
        }
    
        // Copy data to device asynchronously
        cudaMemcpyAsync(p_input_bw, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice, backwardStream);
        cudaMemcpyAsync(p_weights_bw, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice, backwardStream);
        cudaMemcpyAsync(p_d_output_bw, d_output, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice, backwardStream);
    
        cublasSetStream(cublas_handle_backward, backwardStream);
    
        const float alf = 1.0f;
        const float bet = 0.0f;
    
        // Compute d_input = d_output * W^T
        cublasSgemm(cublas_handle_backward,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            inputColsNum,
            inputRowsNum,
            outputColsNum,
            &alf,
            p_weights_bw,
            inputColsNum,
            p_d_output_bw,
            outputColsNum,
            &bet,
            p_d_input_bw,
            inputColsNum);

        // Compute d_weights = X^T * d_output
        cublasSgemm(cublas_handle_backward,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    inputColsNum,
                    outputColsNum,
                    inputRowsNum,
                    &alf,
                    p_input_bw,
                    inputColsNum,
                    p_d_output_bw,
                    outputColsNum,
                    &bet,
                    p_d_weights_bw,
                    inputColsNum);

        // Compute d_bias = sum(d_output, axis=0)
        if (d_bias != nullptr && p_d_bias_bw != nullptr) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (outputColsNum + threadsPerBlock - 1) / threadsPerBlock;
            sumBiasKernel<<<blocksPerGrid, threadsPerBlock, 0, backwardStream>>>(p_d_bias_bw, p_d_output_bw, inputRowsNum, outputColsNum);
        }
    
        // Copy results back to host asynchronously
        cudaMemcpyAsync(d_input, p_d_input_bw, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyDeviceToHost, backwardStream);
        cudaMemcpyAsync(d_weights, p_d_weights_bw, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost, backwardStream);
        if (d_bias != nullptr && p_d_bias_bw != nullptr) {
            cudaMemcpyAsync(d_bias, p_d_bias_bw, outputColsNum * sizeof(float), cudaMemcpyDeviceToHost, backwardStream);
        }
    
        // Synchronize the stream to ensure all operations are completed
        cudaStreamSynchronize(backwardStream);
    }

    DLLEXPORT void cleanupCudaMemory() {
        if (p_input_fw) cudaFreeAsync(p_input_fw, forwardStream);
        if (p_weights_fw) cudaFreeAsync(p_weights_fw, forwardStream);
        if (p_output_fw) cudaFreeAsync(p_output_fw, forwardStream);
        if (p_bias_fw) cudaFreeAsync(p_bias_fw, forwardStream);
        p_input_fw = p_weights_fw = p_output_fw = p_bias_fw = nullptr;

        if (p_input_bw) cudaFreeAsync(p_input_bw, backwardStream);
        if (p_weights_bw) cudaFreeAsync(p_weights_bw, backwardStream);
        if (p_d_output_bw) cudaFreeAsync(p_d_output_bw, backwardStream);
        if (p_d_input_bw) cudaFreeAsync(p_d_input_bw, backwardStream);
        if (p_d_weights_bw) cudaFreeAsync(p_d_weights_bw, backwardStream);
        if (p_d_bias_bw) cudaFreeAsync(p_d_bias_bw, backwardStream);
        p_input_bw = p_weights_bw = p_d_output_bw = p_d_input_bw = p_d_weights_bw = p_d_bias_bw = nullptr;

        if (forwardStream) {
            cudaStreamSynchronize(forwardStream);
            cudaStreamDestroy(forwardStream);
            forwardStream = nullptr;
        }
        if (backwardStream) {
            cudaStreamSynchronize(backwardStream);
            cudaStreamDestroy(backwardStream);
            backwardStream = nullptr;
        }

        if (cublas_handle_forward) {
            cublasDestroy(cublas_handle_forward);
            cublas_handle_forward = nullptr;
        }
        if (cublas_handle_backward) {
            cublasDestroy(cublas_handle_backward);
            cublas_handle_backward = nullptr;
        }

        current_input_fw_rows = current_input_fw_cols = 0;
        current_weights_fw_rows = current_weights_fw_cols = 0;
        current_output_fw_rows = current_output_fw_cols = 0;
        current_bias_fw_size = 0;

        current_input_bw_rows = current_input_bw_cols = 0;
        current_weights_bw_rows = current_weights_bw_cols = 0;
        current_d_output_bw_rows = current_d_output_bw_cols = 0;
        current_d_input_bw_rows = current_d_input_bw_cols = 0;
        current_d_weights_bw_rows = current_d_weights_bw_cols = 0;
        current_d_bias_bw_size = 0;
    }
}