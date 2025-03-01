#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cublasLt.h>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

using namespace std;

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
static cublasLtHandle_t cublaslt_handle = nullptr;

static const size_t cublaslt_workspace_size = 4 * 1024 * 1024; // 4MB
static void *cublaslt_workspace = nullptr;




// Helper function for matrix multiplication with cuBLASLt
void matmul_cublaslt(
    float* C, const float* A, const float* B,
    int m, int n, int k,
    bool transA, bool transB,
    float alpha = 1.0f, float beta = 0.0f,
    const float* bias = nullptr
) {
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulPreference_t preference;

    // Create descriptors

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                  &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                  &opB, sizeof(opB));

    // Configure epilogue for bias in forward mode
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }

    // Create matrix layouts
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 
                              transA ? k : m, transA ? m : k, 
                              transA ? k : m);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 
                              transB ? n : k, transB ? k : n, 
                              transB ? n : k);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, m);

    // Set preference
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, 
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size));

    // Find heuristic
    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 1, &heuristic, &returnedResults
    );

    if (returnedResults == 0) {
        printf("cuBLASLt heuristic not found!\n");
        exit(EXIT_FAILURE);
    }

    // Run matmul
    cublasLtMatmul(
        cublaslt_handle, operationDesc,
        &alpha, A, Adesc, B, Bdesc,
        &beta, C, Cdesc, C, Cdesc,
        &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0
    );

    // Cleanup
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}


extern "C" {
    DLLEXPORT void cudaLinearModuleForward(float *input, float *weights, float *bias, float *output, int inputRowsNum, int inputColsNum, int outputColsNum) {

        // check alignment (some modes work unaligned but it always best to be aligned for performance)
        if(((uintptr_t)input % 16) != 0 || ((uintptr_t)weights % 16) != 0 || ((uintptr_t)bias % 16) != 0 || ((uintptr_t)output % 16) != 0) {
            printf("All cuBLASLt pointers must be aligned!\n");
            exit(EXIT_FAILURE);
        }

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

        if (cublaslt_handle == nullptr) {
            cublasLtCreate(&cublaslt_handle);
            // Allocate workspace once
            if (cublaslt_workspace == nullptr) {
                cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);
            }
        }

        matmul_cublaslt(
            p_output_fw, p_weights_fw, p_input_fw,     // Result, Weights, Input
            outputColsNum, inputRowsNum, inputColsNum, // m, n, k
            true,           // Transpose Weights (transA)
            false,          // Do not transpose Input (transB)
            1.0f,           // alpha
            0.0f,           // beta
            p_bias_fw       // bias (if not null)
        );

        cudaMemcpy(output, p_output_fw, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);

        
    }
    
    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {

        if(((uintptr_t)input % 16) != 0 || ((uintptr_t)weights % 16) != 0 || ((uintptr_t)d_output % 16) != 0 || 
            ((uintptr_t)d_input % 16) != 0 || ((uintptr_t)d_weights % 16) != 0 || ((uintptr_t)d_bias % 16) != 0) 
         {
             printf("All cuBLASLt pointers must be aligned!\n");
             exit(EXIT_FAILURE);
         }
    
    
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
    
        // Initialize cuBLAS and cuBLASLt
        if (!cublaslt_handle) {
            cublasLtCreate(&cublaslt_handle);
            cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);
        }

        // Copy data to device (same as before)
        cudaMemcpy(p_input_bw, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_weights_bw, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d_output_bw, d_output, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);

        // Compute d_input = d_output * W^T
        matmul_cublaslt(
            p_d_input_bw, p_weights_bw, p_d_output_bw,  // Grad Input, Weights, Grad Output
            inputColsNum, inputRowsNum,  outputColsNum, // m, n, k
            false,
            false
        );

        // Compute d_weights = X^T * d_output
        matmul_cublaslt(
            p_d_weights_bw, p_input_bw, p_d_output_bw, // Grad Weights, Input, Grad Output
            inputColsNum, outputColsNum, inputRowsNum, // m, n, k
            false,
            true
        );

        // Compute d_bias = sum(d_output, axis=0)
        if (d_bias != nullptr && p_d_bias_bw != nullptr) {
            int threads = 256;
            int blocks = (outputColsNum + threads - 1) / threads;
            sumBiasKernel<<<blocks, threads>>>(
                p_d_bias_bw, p_d_output_bw, inputRowsNum, outputColsNum
            );
        }

        // Copy results back (same as before)
        cudaMemcpy(d_input, p_d_input_bw, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_weights, p_d_weights_bw, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        if (d_bias != nullptr) {
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

        // Destroying cuBLASlt handle
        if (cublaslt_handle != nullptr) {
            cublasLtDestroy(cublaslt_handle);
            cublaslt_handle = nullptr;
        }
    }
}
