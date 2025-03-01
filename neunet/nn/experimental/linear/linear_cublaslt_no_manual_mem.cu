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

        if (cublaslt_handle == nullptr) {
            cublasLtCreate(&cublaslt_handle);
            // Allocate workspace once
            if (cublaslt_workspace == nullptr) {
                cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);
            }
        }

        matmul_cublaslt(
            output, weights,input,     // Result, Weights, Input
            outputColsNum, inputRowsNum, inputColsNum, // m, n, k
            true,           // Transpose Weights (transA)
            false,          // Do not transpose Input (transB)
            1.0f,           // alpha
            0.0f,           // beta
            bias       // bias (if not null)
        );

    }
    
    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {
    
        // Initialize cuBLAS and cuBLASLt
        if (!cublaslt_handle) {
            cublasLtCreate(&cublaslt_handle);
            cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);
        }

        // Compute d_input = d_output * W^T
        matmul_cublaslt(
            d_input, weights, d_output,  // Grad Input, Weights, Grad Output
            inputColsNum, inputRowsNum,  outputColsNum, // m, n, k
            false,
            false
        );

        // Compute d_weights = X^T * d_output
        matmul_cublaslt(
            d_weights, input, d_output, // Grad Weights, Input, Grad Output
            inputColsNum, outputColsNum, inputRowsNum, // m, n, k
            false,
            true
        );

        // Compute d_bias = sum(d_output, axis=0)
        if (d_bias != nullptr && d_bias != nullptr) {
            int threads = 256;
            int blocks = (outputColsNum + threads - 1) / threads;
            sumBiasKernel<<<blocks, threads>>>(
                d_bias, d_output, inputRowsNum, outputColsNum
            );
        }
    }

    DLLEXPORT void cleanupCudaMemory() {
        // Destroying cuBLASlt handle
        if (cublaslt_handle != nullptr) {
            cublasLtDestroy(cublaslt_handle);
            cublaslt_handle = nullptr;
        }
    }
}
