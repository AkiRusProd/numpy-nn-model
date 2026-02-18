#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


// d_linear = d_output * Swish'(pre_act)
// dL/dz = dL/dy * Swish'(z)
__global__ void swish_backward_kernel(
    float* d_linear_tmp,         // Input: z, output: dL/dz
    const float* d_output,       // Input: dL/dy (gradient from next layer)
    float beta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_linear_tmp[idx];
        float sigmoid = 1.0f / (1.0f + expf(-beta * z));
        
        // Swish'(z) = sigma(beta*z) * (1 + beta*z * (1 - sigma(beta*z)))
        float swish_grad = sigmoid * (1.0f + (beta * z * (1.0f - sigmoid)));
        
        // Overwrite buffer with gradient
        d_linear_tmp[idx] = d_output[idx] * swish_grad;
    }
}


// -----------------------------------------------------------------------------
// Custom Epilogue Functor for Fused Swish
// Computes: D = Swish(alpha * Accumulator + beta * Bias)
// Swish(x) = x * sigmoid(swish_beta * x) = x / (1 + exp(-swish_beta * x))
// -----------------------------------------------------------------------------
template <
  typename ElementOutput_,                             // Data type used to load and store tensors
  int Count,                                           // Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       // Accumulator data type
  typename ElementCompute_ = ElementOutput_            // Data type used to compute linear combination
>
class LinearCombinationSwish {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kCount = Count;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute swish_beta; // Added parameter for Swish
    ElementCompute const *alpha_ptr = nullptr;
    ElementCompute const *beta_ptr = nullptr;
    ElementCompute const *swish_beta_ptr = nullptr;

    CUTLASS_HOST_DEVICE
    Params(): alpha(ElementCompute(1)), beta(ElementCompute(0)), swish_beta(ElementCompute(1)) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta, ElementCompute swish_beta): 
      alpha(alpha), beta(beta), swish_beta(swish_beta) { }
  };

private:
  ElementCompute alpha;
  ElementCompute beta;
  ElementCompute swish_beta;

public:
  // Constructor
  CUTLASS_HOST_DEVICE
  LinearCombinationSwish(Params const &params)
    : alpha(params.alpha), beta(params.beta), swish_beta(params.swish_beta) {
    if (params.alpha_ptr) alpha = *params.alpha_ptr;
    if (params.beta_ptr) beta = *params.beta_ptr;
    if (params.swish_beta_ptr) swish_beta = *params.swish_beta_ptr;
  }

  // Check if source (C matrix/bias) is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
      // Intentionally empty
  }

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentSource = cutlass::Array<ElementOutput, kCount>;

  // Operator for C = alpha*AB + beta*C with Swish
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &source_accumulator,
    FragmentSource const &source_output) const {

    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount> source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> destination_converter;

    cutlass::Array<ElementCompute, kCount> converted_source = source_converter(source_output);
    cutlass::Array<ElementCompute, kCount> converted_accumulator = accumulator_converter(source_accumulator);
    cutlass::Array<ElementCompute, kCount> intermediate;

    #pragma unroll
    for (int i = 0; i < kCount; ++i) {
        // 1. Linear combination: x = alpha * Acc + beta * Bias
        ElementCompute x = alpha * converted_accumulator[i] + beta * converted_source[i];
        
        // 2. Swish activation: x / (1 + exp(-swish_beta * x))
        // Using fast math for exp if available or standard expf
        ElementCompute sigmoid_val = ElementCompute(1) / (ElementCompute(1) + expf(-swish_beta * x));
        intermediate[i] = x * sigmoid_val;
    }

    return destination_converter(intermediate);
  }

  // Operator for case when beta is 0 (Source not loaded)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &source_accumulator) const {

    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> destination_converter;

    cutlass::Array<ElementCompute, kCount> converted_accumulator = accumulator_converter(source_accumulator);
    cutlass::Array<ElementCompute, kCount> intermediate;

    #pragma unroll
    for (int i = 0; i < kCount; ++i) {
        // 1. Linear combination (beta=0)
        ElementCompute x = alpha * converted_accumulator[i];
        
        // 2. Swish activation
        ElementCompute sigmoid_val = ElementCompute(1) / (ElementCompute(1) + expf(-swish_beta * x));
        intermediate[i] = x * sigmoid_val;
    }

    return destination_converter(intermediate);
  }
};

// -----------------------------------------------------------------------------
// Bias Sum Kernel (Backward pass helper)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Gemm Type Definitions
// -----------------------------------------------------------------------------

// Standard linear GEMM types for Backward pass (no swish fusion needed there usually)
using GemmRM = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;

using GemmTransposedRowRow = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;

// FUSED GEMM Definition for Forward pass
// We explicitly specify the EpilogueOp
using EpilogueSwish = LinearCombinationSwish<float, 1, float, float>;

using GemmSwish = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,           // ElementA, LayoutA
    float, cutlass::layout::ColumnMajor,        // ElementB, LayoutB (Weights transposed via layout)
    float, cutlass::layout::RowMajor,           // ElementC, LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassSimt,                 // Operator Class (Simt for FP32 usually)
    cutlass::arch::Sm50,                        // Architecture
    cutlass::gemm::GemmShape<128, 128, 8>,      // Threadblock Shape
    cutlass::gemm::GemmShape<32, 64, 8>,        // Warp Shape
    cutlass::gemm::GemmShape<1, 1, 1>,          // Instruction Shape
    EpilogueSwish                               // <--- OUR CUSTOM EPILOGUE
>;


extern "C" {

    /*
     * Fused Linear + Swish Forward
     * output = Swish(Input * Weights^T + Bias)
     */
    DLLEXPORT void cudaLinearSwishForward(
        float *input, 
        float *weights, 
        float *bias, 
        float *output, 
        int inputRowsNum, 
        int inputColsNum, 
        int outputColsNum,
        float swish_beta // New parameter
    ) {
        
        float alpha = 1.0f;
        float beta = (bias != nullptr) ? 1.0f : 0.0f;
        float* ptr_bias = (bias != nullptr) ? bias : output; 
        int64_t ldc_bias = (bias != nullptr) ? 0 : outputColsNum; // stride 0 for broadcasting bias

        cutlass::gemm::GemmCoord problem_size{inputRowsNum, outputColsNum, inputColsNum};

        // Construct arguments. Note the last argument struct is for the Epilogue.
        typename GemmSwish::Arguments args{
            problem_size,
            {input, inputColsNum},     
            {weights, inputColsNum},   
            {ptr_bias, ldc_bias},      
            {output, outputColsNum},
            {alpha, beta, swish_beta}  // Passing params to LinearCombinationSwish
        };

        GemmSwish gemm_op;
        cutlass::Status status = gemm_op(args);
        
        if (status != cutlass::Status::kSuccess) {
            // Error handling can be added here
        }
    }

    // Keep Backward standard (unfused) or requires much more complex fusion
    DLLEXPORT void cudaLinearSwishBackward(
        float *input,           // X (saved from forward)
        float *weights,         // W (saved from forward)
        float *bias,            // b (saved from forward)
        float *d_output,        // dL/dy (from upstream)
        float *d_linear_tmp,    // TEMP buffer [M * N]
        float *d_input,         // Result: dL/dX
        float *d_weights,       // Result: dL/dW
        float *d_bias,          // Result: dL/db
        int inputRowsNum,       // M
        int inputColsNum,       // K
        int outputColsNum,      // N
        float swish_beta,
        cudaStream_t stream
    ) {
        // --- STEP 1: RECOMPUTATION ---
        // Recompute z = X * W^T + b
        // Use standard GemmRowRowTransposed (no Swish)
        {
            float alpha = 1.0f;
            float beta_gemm = (bias != nullptr) ? 1.0f : 0.0f;
            float* ptr_bias = (bias != nullptr) ? bias : d_linear_tmp; 
            int64_t ldc_bias = (bias != nullptr) ? 0 : outputColsNum;

            cutlass::gemm::GemmCoord size_recomp{inputRowsNum, outputColsNum, inputColsNum};
            
            // Use standard GemmRM or GemmRowRowTransposed for recompute
            using GemmRecompute = cutlass::gemm::device::Gemm<
                float, cutlass::layout::RowMajor, 
                float, cutlass::layout::ColumnMajor, 
                float, cutlass::layout::RowMajor>;

            typename GemmRecompute::Arguments args_recomp{
                size_recomp,
                {input, inputColsNum},
                {weights, inputColsNum},
                {ptr_bias, ldc_bias},
                {d_linear_tmp, outputColsNum}, // Write z here
                {alpha, beta_gemm}
            };
            GemmRecompute gemm_recomp;
            gemm_recomp(args_recomp, nullptr, stream);
        }

        // --- STEP 2: COMPUTE ACTIVATION GRADIENT ---
        // d_linear_tmp currently holds z. Convert it to dL/dz.
        int total_elements = inputRowsNum * outputColsNum;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        swish_backward_kernel<<<blocks, threads, 0, stream>>>(
            d_linear_tmp, d_output, swish_beta, total_elements
        );

        // --- STEP 3: INPUT GRADIENT (dX = dL/dz * W) ---
        {
            cutlass::gemm::GemmCoord size_dx{inputRowsNum, inputColsNum, outputColsNum};
            typename GemmRM::Arguments args_dx{
                size_dx,
                {d_linear_tmp, outputColsNum}, // A: dL/dz
                {weights, inputColsNum},        // B: W
                {d_input, inputColsNum},        // C: zero
                {d_input, inputColsNum},        // D: dL/dX
                {1.0f, 0.0f}
            };
            GemmRM gemm_dx;
            gemm_dx(args_dx, nullptr, stream);
        }

        // --- STEP 4: WEIGHT GRADIENT (dW = (dL/dz)^T * X) ---
        {
            cutlass::gemm::GemmCoord size_dw{outputColsNum, inputColsNum, inputRowsNum};
            typename GemmTransposedRowRow::Arguments args_dw{
                size_dw,
                {d_linear_tmp, outputColsNum}, 
                {input, inputColsNum},         
                {d_weights, inputColsNum},
                {d_weights, inputColsNum},
                {1.0f, 0.0f}
            };
            GemmTransposedRowRow gemm_dw;
            gemm_dw(args_dw, nullptr, stream);
        }

        // --- STEP 5: BIAS GRADIENT (db = sum(dL/dz)) ---
        if (d_bias != nullptr) {
            int threads = 256;
            int blocks = (outputColsNum + threads - 1) / threads;
            sumBiasKernel<<<blocks, threads, 0, stream>>>(
                d_bias, d_linear_tmp, inputRowsNum, outputColsNum
            );
        }
    }
    
    DLLEXPORT void cleanupCudaMemory() {}
}
