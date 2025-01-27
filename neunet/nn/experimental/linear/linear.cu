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
// __global__ void addBiasKernel(float *C, const float *bias, int rowsNum, int colsNum) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
//     if (idx < colsNum) {
//         for (int i = 0; i < rowsNum; i++) {
//             if (bias != nullptr) { // Check if bias is not null
//                 C[row * colsNum + col] += bias[col];
//             }
//         }
//     }
// }

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

// C(rowsNum, colsNum) = A(rowsNum, width) * B(width, colsNum) + bias
void blasMatMulWithBias(const float *A, const float *B, const float *bias, float *C, const int rowsNum, const int width, const int colsNum) {
    const float alf = 1.0f;
    const float bet = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Matrix multiplication: C = A * B
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colsNum, rowsNum, width, &alf, B, colsNum, A, width, &bet, C, colsNum);

    // Add bias to each column of C
    if (bias != nullptr) {
        // C(i, j) += bias(j)
        // int threadsPerBlock = 256;  // You can experiment with this value
        // int blocksPerGrid = (rowsNum + threadsPerBlock - 1) / threadsPerBlock;

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((rowsNum + 15) / 16, (colsNum + 15) / 16);

        // Add bias to each row for each column (without race conditions)
        addBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(C, bias, rowsNum, colsNum);
        cudaDeviceSynchronize();  // Make sure the kernel execution finishes
    }

    cublasDestroy(handle);
}


extern "C" {
    DLLEXPORT void cudaLinearModuleForward(float *input, float *weights, float *bias, float *output, int inputRowsNum, int inputColsNum, int outputColsNum) {
        float *p_input, *p_weights, *p_bias, *p_output;

        // Device memory allocation
        cudaMalloc((void**)&p_input, inputRowsNum * inputColsNum * sizeof(float));
        cudaMalloc((void**)&p_weights, inputColsNum * outputColsNum * sizeof(float));
        cudaMalloc((void**)&p_output, inputRowsNum * outputColsNum * sizeof(float));

        // Handle bias
        if (bias != nullptr) {
            cudaMalloc((void**)&p_bias, outputColsNum * sizeof(float));
            cudaMemcpy(p_bias, bias, outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            p_bias = nullptr; // Explicitly set to nullptr
        }

        // Copy data to device
        cudaMemcpy(p_input, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_weights, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);

        // Perform linear layer operation: output = input * weights + bias
        blasMatMulWithBias(p_input, p_weights, p_bias, p_output, inputRowsNum, inputColsNum, outputColsNum);

        // Copy result back to host
        cudaMemcpy(output, p_output, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(p_input);
        cudaFree(p_weights);
        cudaFree(p_output);

        if (p_bias != nullptr) {
            cudaFree(p_bias);
        }
    }



    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {

        float *p_input, *p_weights, *p_d_output, *p_d_input, *p_d_weights, *p_d_bias = nullptr;

        // Device memory allocation
        cudaMalloc((void**)&p_input, inputRowsNum * inputColsNum * sizeof(float));
        cudaMalloc((void**)&p_weights, inputColsNum * outputColsNum * sizeof(float));
        cudaMalloc((void**)&p_d_output, inputRowsNum * outputColsNum * sizeof(float));
        cudaMalloc((void**)&p_d_input, inputRowsNum * inputColsNum * sizeof(float));
        cudaMalloc((void**)&p_d_weights, inputColsNum * outputColsNum * sizeof(float));

        if (d_bias != nullptr) {
            cudaMalloc((void**)&p_d_bias, outputColsNum * sizeof(float));
        }

        // Copy data to device
        cudaMemcpy(p_input, input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_weights, weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(p_d_output, d_output, inputRowsNum * outputColsNum * sizeof(float), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alf = 1.0f;
        const float bet = 0.0f;

        // Compute d_input = d_output * W^T
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, inputColsNum, inputRowsNum, outputColsNum,
                    &alf, p_weights, outputColsNum, p_d_output, outputColsNum, &bet, p_d_input, inputColsNum);

        // Compute d_weights = A^T * d_output
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, outputColsNum, inputColsNum, inputRowsNum,
                    &alf, p_d_output, outputColsNum, p_input, inputColsNum, &bet, p_d_weights, outputColsNum);

        // Compute d_bias = sum(d_output, axis=0) if d_bias is not nullptr
        // Use custom kernel to sum along rows
        if (d_bias != nullptr) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (outputColsNum + threadsPerBlock - 1) / threadsPerBlock;
            sumBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(p_d_bias, p_d_output, inputRowsNum, outputColsNum);
            cudaDeviceSynchronize();
        }

        // Copy results back to host
        cudaMemcpy(d_input, p_d_input, inputRowsNum * inputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_weights, p_d_weights, inputColsNum * outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);

        if (d_bias != nullptr) {
            cudaMemcpy(d_bias, p_d_bias, outputColsNum * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Free device memory
        cudaFree(p_input);
        cudaFree(p_weights);
        cudaFree(p_d_output);
        cudaFree(p_d_input);
        cudaFree(p_d_weights);

        if (p_d_bias != nullptr) {
            cudaFree(p_d_bias);
        }

        cublasDestroy(handle);
    }
}
