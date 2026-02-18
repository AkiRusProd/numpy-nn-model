#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


__global__ void sumBiasKernel(float* d_bias, const float* d_output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += d_output[row * cols + col];
        }
        d_bias[col] = sum;
    }
}


using GemmRM = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;
using GemmRowRowTransposed = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::ColumnMajor, 
    float, cutlass::layout::RowMajor>;
using GemmTransposedRowRow = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;

extern "C" {

    DLLEXPORT void cudaLinearModuleForward(float *input, float *weights, float *bias, float *output, 
                                           int inputRowsNum, int inputColsNum, int outputColsNum) {
        
        // FUSED BIAS IMPLEMENTATION
        // C = alpha*(A*B) + beta*C
        // We set C = bias. To broadcast bias across rows, set stride (ldc) = 0.
        
        float alpha = 1.0f;
        float beta = (bias != nullptr) ? 1.0f : 0.0f;
        float* ptr_bias = (bias != nullptr) ? bias : output; // Safe fallback
        int64_t ldc_bias = (bias != nullptr) ? 0 : outputColsNum; 

        cutlass::gemm::GemmCoord problem_size{inputRowsNum, outputColsNum, inputColsNum};

        typename GemmRowRowTransposed::Arguments args{
            problem_size,
            {input, inputColsNum},     
            {weights, inputColsNum},   // Interpret as ColumnMajor to apply transpose
            {ptr_bias, ldc_bias},      // C: Bias with ldc=0 (broadcast)
            {output, outputColsNum},   // D: Output
            {alpha, beta}
        };

        GemmRowRowTransposed gemm_op;
        gemm_op(args);
    }

    DLLEXPORT void cudaLinearModuleBackward(
        float *input, float *weights, float *d_output, 
        float *d_input, float *d_weights, float *d_bias, 
        int inputRowsNum, int inputColsNum, int outputColsNum) {

        {
            cutlass::gemm::GemmCoord size_dx{inputRowsNum, inputColsNum, outputColsNum};
            typename GemmRM::Arguments args_dx{
                size_dx,
                {d_output, outputColsNum},
                {weights, inputColsNum},
                {d_input, inputColsNum},
                {d_input, inputColsNum},
                {1.0f, 0.0f}
            };
            GemmRM gemm_dx;
            gemm_dx(args_dx);
        }

        {
            cutlass::gemm::GemmCoord size_dw{outputColsNum, inputColsNum, inputRowsNum};
            typename GemmTransposedRowRow::Arguments args_dw{
                size_dw,
                {d_output, outputColsNum}, 
                {input, inputColsNum},     
                {d_weights, inputColsNum},
                {d_weights, inputColsNum},
                {1.0f, 0.0f}
            };
            GemmTransposedRowRow gemm_dw;
            gemm_dw(args_dw);
        }

        if (d_bias != nullptr) {
            int threads = 256;
            int blocks = (outputColsNum + threads - 1) / threads;
            sumBiasKernel<<<blocks, threads>>>(d_bias, d_output, inputRowsNum, outputColsNum);
        }
    }

    DLLEXPORT void cleanupCudaMemory() {}
}
