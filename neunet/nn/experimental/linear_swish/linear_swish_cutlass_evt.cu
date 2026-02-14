#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>

// --- EVT (Epilogue Visitor Tree) includes for fused backward ---
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

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

// =============================================================================
// FUSED Backward: Recompute + SwishBackward via CUTLASS EVT (Sm80)
// Computes: d_linear_tmp = d_output * Swish'(X * W^T + bias)
// Uses TensorOp (TF32) for the GEMM core - fast, with ~1e-4 precision vs FP32.
// =============================================================================

using namespace cute;

// --- TensorOp GEMM config for backward EVT (SM80 TF32) ---
using BwdThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using BwdWarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using BwdInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
constexpr int BwdAlignment = 4;  // 128 bits / 32 bits
constexpr int BwdNumStages = 3;
constexpr int BwdEVTEpilogueStages = 1;

// OutputTileThreadLayout maps threads to output elements for EVT visitors
using BwdOutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    BwdThreadblockShape, BwdWarpShape, float, BwdAlignment, BwdEVTEpilogueStages>;

// --- Custom Visitor: SwishBackward compute ---
// Takes z (pre-activation) and d_output, computes: d_output * Swish'(z)
// where Swish'(z) = sigma(beta*z) * (1 + beta*z * (1 - sigma(beta*z)))
struct VisitorSwishBackwardCompute {
    struct Arguments {
        float swish_beta = 1.0f;
    };
    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return args;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const&, Arguments const&) { return true; }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

    template <class ProblemShape>
    static cutlass::Status
    initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
        cutlass::CudaHostAdapter* = nullptr) { return cutlass::Status::kSuccess; }

    struct SharedStorage {};

    CUTLASS_HOST_DEVICE VisitorSwishBackwardCompute() {}
    CUTLASS_HOST_DEVICE VisitorSwishBackwardCompute(Params const& params, SharedStorage const&)
        : params_ptr(&params) {}

    Params const* params_ptr;

    struct Callbacks : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
        float swish_beta;

        CUTLASS_DEVICE Callbacks(float sb) : swish_beta(sb) {}

        template <typename ElementAccumulator, typename ElementZ, typename ElementDOut, int FragmentSize>
        CUTLASS_DEVICE cutlass::Array<float, FragmentSize>
        visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
              cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
              cutlass::Array<ElementZ, FragmentSize> const& frg_z,
              cutlass::Array<ElementDOut, FragmentSize> const& frg_dout) {

            cutlass::Array<float, FragmentSize> result;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i) {
                float z    = float(frg_z[i]);
                float dout = float(frg_dout[i]);
                float sig  = 1.0f / (1.0f + expf(-swish_beta * z));
                float sg   = sig * (1.0f + swish_beta * z * (1.0f - sig));
                result[i]  = dout * sg;
            }
            return result;
        }
    };

    template <class ProblemShape>
    CUTLASS_DEVICE auto
    get_callbacks(
        cutlass::gemm::GemmCoord threadblock_tile_offset,
        int thread_idx,
        ProblemShape problem_shape) {
        return Callbacks(params_ptr->swish_beta);
    }
};

// --- EVT tree definition ---
// Tree:  Store( SwishBwd( Add(Acc, Bias), DOutput ) )
//
// Level 0 (leaves):
//   Accum    = VisitorAccFetch              -> X * W^T  (GEMM accumulator)
//   Bias     = VisitorRowBroadcast          -> bias row vector (nullable)
//   DOutput  = VisitorAuxLoad               -> d_output matrix
//
// Level 1 (compute):
//   AddBias  = VisitorCompute<plus>         -> z = Acc + Bias
//
// Level 2 (compute):
//   SwishBwd = VisitorSwishBackwardCompute  -> d_output * Swish'(z)
//
// Level 3 (store):
//   Store    = VisitorAuxStore              -> write result to d_linear_tmp

using BwdAccum = cutlass::epilogue::threadblock::VisitorAccFetch;

using BwdBias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    BwdOutputTileThreadMap, float,
    Stride<_0, _1, int32_t>  // M=broadcast, N=contiguous, L=batch
>;

using BwdAddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTAddBias = cutlass::epilogue::threadblock::Sm80EVT<
    BwdAddBias, BwdAccum, BwdBias>;  // z = Acc + Bias

using BwdDOutput = cutlass::epilogue::threadblock::VisitorAuxLoad<
    BwdOutputTileThreadMap, float,
    Stride<int64_t, _1, int64_t>  // M=row_stride, N=contiguous, L=batch
>;

using EVTSwishBwd = cutlass::epilogue::threadblock::Sm80EVT<
    VisitorSwishBackwardCompute, EVTAddBias, BwdDOutput>;  // SwishBwd(z, d_output)

using BwdStore = cutlass::epilogue::threadblock::VisitorAuxStore<
    BwdOutputTileThreadMap, float,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>  // M=row_stride, N=contiguous, L=batch
>;

using EVTBackward = cutlass::epilogue::threadblock::Sm80EVT<
    BwdStore, EVTSwishBwd>;  // Store(SwishBwd(...))

// --- Fused backward GEMM kernel via DefaultGemmWithVisitor ---
using EVTBwdKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    float, cutlass::layout::RowMajor,          // A: input [M, K]
    cutlass::ComplexTransform::kNone, BwdAlignment,
    float, cutlass::layout::ColumnMajor,       // B: weights [N, K] as W^T [K, N]
    cutlass::ComplexTransform::kNone, BwdAlignment,
    float, cutlass::layout::RowMajor,          // C (unused by EVT, placeholder)
    BwdAlignment,
    float,                                     // ElementAccumulator
    float,                                     // ElementEpilogue
    cutlass::arch::OpClassTensorOp,            // TensorOp (TF32 for FP32)
    cutlass::arch::Sm80,                       // Architecture
    BwdThreadblockShape,
    BwdWarpShape,
    BwdInstructionShape,
    EVTBackward,                               // <--- Our EVT fusion callbacks
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    BwdNumStages,
    cutlass::arch::OpMultiplyAdd,
    BwdEVTEpilogueStages
>::GemmKernel;

using GemmSwishBackward = cutlass::gemm::device::GemmUniversalAdapter<EVTBwdKernel>;


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
        // --- STEP 1+2 (FUSED): RECOMPUTATION + SWISH BACKWARD via EVT ---
        // Fused: d_linear_tmp = d_output * Swish'(X * W^T + bias)
        // Single GEMM with EVT epilogue replaces separate recompute GEMM + swish_backward_kernel
        {
            typename EVTBackward::Arguments callback_args{
                // op_0: EVTSwishBwd
                {
                    // op_0: EVTAddBias
                    {
                        {},                                                                              // Accum (VisitorAccFetch, no args)
                        {bias, float(0), {_0{}, _1{}, int32_t(outputColsNum)}},                          // Bias (VisitorRowBroadcast: ptr, null_default, stride)
                        {}                                                                               // AddBias (VisitorCompute<plus>, no args)
                    },
                    // op_1: BwdDOutput (VisitorAuxLoad)
                    {d_output, float(0), {int64_t(outputColsNum), _1{}, int64_t(0)}},
                    // op_2: VisitorSwishBackwardCompute
                    {swish_beta}
                },
                // op_1: BwdStore (VisitorAuxStore)
                {d_linear_tmp, {int64_t(outputColsNum), _1{}, int64_t(0)}}
            };

            typename GemmSwishBackward::Arguments args_bwd(
                cutlass::gemm::GemmUniversalMode::kGemm,
                {inputRowsNum, outputColsNum, inputColsNum},     // problem_size M, N, K
                1,                                                // batch_count
                callback_args,                                    // EVT epilogue arguments
                input,                                            // ptr_A: X [M, K]
                weights,                                          // ptr_B: W [N, K] as ColumnMajor
                nullptr,                                          // ptr_C (unused by EVT)
                nullptr,                                          // ptr_D (unused by EVT)
                int64_t(inputRowsNum) * inputColsNum,             // batch_stride_A
                int64_t(outputColsNum) * inputColsNum,            // batch_stride_B
                0,                                                // batch_stride_C
                0,                                                // batch_stride_D
                inputColsNum,                                     // stride_a (lda)
                inputColsNum,                                     // stride_b (ldb)
                0,                                                // stride_c
                0                                                 // stride_d
            );

            GemmSwishBackward gemm_bwd;
            size_t workspace_size = GemmSwishBackward::get_workspace_size(args_bwd);
            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                cudaMalloc(&workspace_ptr, workspace_size);
            }
            gemm_bwd.initialize(args_bwd, workspace_ptr, stream);
            gemm_bwd.run(stream);
            if (workspace_ptr) {
                cudaFree(workspace_ptr);
            }
        }

        int total_elements = inputRowsNum * outputColsNum;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

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
