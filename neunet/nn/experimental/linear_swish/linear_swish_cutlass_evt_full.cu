#include <cuda_runtime.h>
#include <stdio.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>

// --- EVT (Epilogue Visitor Tree) includes ---
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


// =============================================================================
// Helper Kernels
// =============================================================================

// Swish backward with a separate saved_z buffer (no recompute)
// d_input[i] = d_output[i] * Swish'(saved_z[i])
__global__ void swish_backward_from_saved_kernel(
    float* __restrict__ d_input,
    const float* __restrict__ saved_z,
    const float* __restrict__ d_output,
    float beta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = saved_z[idx];
        float sigmoid = 1.0f / (1.0f + expf(-beta * z));
        float swish_grad = sigmoid * (1.0f + (beta * z * (1.0f - sigmoid)));
        d_input[idx] = d_output[idx] * swish_grad;
    }
}

// Swish backward in-place (reads z from d_input, writes result back)
__global__ void swish_backward_kernel(
    float* d_input,
    const float* d_output,
    float beta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-beta * z));
        float swish_grad = sigmoid * (1.0f + (beta * z * (1.0f - sigmoid)));
        d_input[idx] = d_output[idx] * swish_grad;
    }
}

// Bias gradient: sum across rows
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


// =============================================================================
// Custom Epilogue Functor: LinearCombinationSwish (for non-EVT forward)
// Computes: D = Swish(alpha * Accumulator + beta * Bias)
// =============================================================================
template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = ElementOutput_
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
    ElementCompute swish_beta;
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
  CUTLASS_HOST_DEVICE
  LinearCombinationSwish(Params const &params)
    : alpha(params.alpha), beta(params.beta), swish_beta(params.swish_beta) {
    if (params.alpha_ptr) alpha = *params.alpha_ptr;
    if (params.beta_ptr) beta = *params.beta_ptr;
    if (params.swish_beta_ptr) swish_beta = *params.swish_beta_ptr;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentSource = cutlass::Array<ElementOutput, kCount>;

  // With bias (beta != 0)
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
        ElementCompute x = alpha * converted_accumulator[i] + beta * converted_source[i];
        ElementCompute sigmoid_val = ElementCompute(1) / (ElementCompute(1) + expf(-swish_beta * x));
        intermediate[i] = x * sigmoid_val;
    }

    return destination_converter(intermediate);
  }

  // Without bias (beta == 0)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &source_accumulator) const {

    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> destination_converter;

    cutlass::Array<ElementCompute, kCount> converted_accumulator = accumulator_converter(source_accumulator);
    cutlass::Array<ElementCompute, kCount> intermediate;

    #pragma unroll
    for (int i = 0; i < kCount; ++i) {
        ElementCompute x = alpha * converted_accumulator[i];
        ElementCompute sigmoid_val = ElementCompute(1) / (ElementCompute(1) + expf(-swish_beta * x));
        intermediate[i] = x * sigmoid_val;
    }

    return destination_converter(intermediate);
  }
};


// =============================================================================
// GEMM Type Definitions
// =============================================================================

// Standard GEMMs for backward steps dX and dW
using GemmRM = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;

using GemmTransposedRowRow = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor, 
    float, cutlass::layout::RowMajor, 
    float, cutlass::layout::RowMajor>;

// FUSED forward (non-EVT, SIMT FP32) - used when save_z=0
using EpilogueSwish = LinearCombinationSwish<float, 1, float, float>;

using GemmSwish = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueSwish
>;


// =============================================================================
// Shared EVT configuration (SM80 TensorOp TF32)
// Used for both forward EVT (dual output) and backward EVT (recompute)
// =============================================================================

using namespace cute;

using EvtThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using EvtWarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using EvtInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
constexpr int EvtAlignment = 4;  // 128 bits / 32 bits = 4 floats
constexpr int EvtNumStages = 3;
constexpr int EvtEpilogueStages = 1;

using EvtOutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    EvtThreadblockShape, EvtWarpShape, float, EvtAlignment, EvtEpilogueStages>;


// =============================================================================
// Custom EVT Visitors
// =============================================================================

// --- Forward: Swish(z) = z * sigmoid(beta * z) ---
// Unary visitor: takes z from child, returns Swish(z)
struct VisitorSwishForwardCompute {
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

    CUTLASS_HOST_DEVICE VisitorSwishForwardCompute() {}
    CUTLASS_HOST_DEVICE VisitorSwishForwardCompute(Params const& params, SharedStorage const&)
        : params_ptr(&params) {}

    Params const* params_ptr;

    struct Callbacks : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
        float swish_beta;

        CUTLASS_DEVICE Callbacks(float sb) : swish_beta(sb) {}

        // Unary visit: frg_acc (unused), frg_z (z from child)
        template <typename ElementAccumulator, typename ElementZ, int FragmentSize>
        CUTLASS_DEVICE cutlass::Array<float, FragmentSize>
        visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
              cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
              cutlass::Array<ElementZ, FragmentSize> const& frg_z) {

            cutlass::Array<float, FragmentSize> result;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i) {
                float z   = float(frg_z[i]);
                float sig = 1.0f / (1.0f + expf(-swish_beta * z));
                result[i] = z * sig;
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

// --- Backward: d_output * Swish'(z) ---
// Binary visitor: takes z and d_output from two children
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

        // Binary visit: frg_z (z = Acc+Bias), frg_dout (d_output)
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


// =============================================================================
// Forward EVT Tree (dual output: output=Swish(z), z_output=z)
//
// Tree:  StoreOutput( SwishFwd( StoreZ( AddBias(Acc, Bias) ) ) )
//
//  Level 0 (leaves):
//    Accum  = VisitorAccFetch            -> X * W^T  (GEMM accumulator)
//    Bias   = VisitorRowBroadcast        -> bias row vector (nullable, null_default=0)
//
//  Level 1 (compute):
//    AddBias = VisitorCompute<plus>      -> z = Acc + Bias
//
//  Level 2 (store+passthrough):
//    StoreZ = VisitorAuxStore            -> store z in z_output, passthrough z
//
//  Level 3 (compute):
//    SwishFwd = VisitorSwishForwardCompute -> Swish(z)
//
//  Level 4 (store):
//    StoreOutput = VisitorAuxStore       -> store Swish(z) in output
// =============================================================================

using FwdAccum = cutlass::epilogue::threadblock::VisitorAccFetch;

using FwdBias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    EvtOutputTileThreadMap, float,
    Stride<_0, _1, int32_t>
>;

using FwdAddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTFwdAddBias = cutlass::epilogue::threadblock::Sm80EVT<
    FwdAddBias, FwdAccum, FwdBias>;  // z = Acc + Bias

using FwdStoreZ = cutlass::epilogue::threadblock::VisitorAuxStore<
    EvtOutputTileThreadMap, float,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>
>;

using EVTFwdStoreZ = cutlass::epilogue::threadblock::Sm80EVT<
    FwdStoreZ, EVTFwdAddBias>;  // store z -> z_output, passthrough z

using EVTFwdSwish = cutlass::epilogue::threadblock::Sm80EVT<
    VisitorSwishForwardCompute, EVTFwdStoreZ>;  // Swish(z)

using FwdStoreOutput = cutlass::epilogue::threadblock::VisitorAuxStore<
    EvtOutputTileThreadMap, float,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>
>;

using EVTForward = cutlass::epilogue::threadblock::Sm80EVT<
    FwdStoreOutput, EVTFwdSwish>;  // Store(Swish(z))

// --- Forward EVT GEMM Kernel ---
using EVTFwdKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    float, cutlass::layout::RowMajor,          // A: input [M, K]
    cutlass::ComplexTransform::kNone, EvtAlignment,
    float, cutlass::layout::ColumnMajor,       // B: weights [N, K] -> W^T [K, N]
    cutlass::ComplexTransform::kNone, EvtAlignment,
    float, cutlass::layout::RowMajor,          // C (unused, placeholder)
    EvtAlignment,
    float,                                     // ElementAccumulator
    float,                                     // ElementEpilogue
    cutlass::arch::OpClassTensorOp,            // TensorOp (TF32)
    cutlass::arch::Sm80,
    EvtThreadblockShape,
    EvtWarpShape,
    EvtInstructionShape,
    EVTForward,                                // <--- Forward EVT callbacks
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    EvtNumStages,
    cutlass::arch::OpMultiplyAdd,
    EvtEpilogueStages
>::GemmKernel;

using GemmSwishForwardEVT = cutlass::gemm::device::GemmUniversalAdapter<EVTFwdKernel>;


// =============================================================================
// Backward EVT Tree (recompute mode: z = X * W^T + bias)
//
// Tree:  Store( SwishBwd( Add(Acc, Bias), DOutput ) )
//
//  Level 0 (leaves):
//    Accum    = VisitorAccFetch            -> X * W^T
//    Bias     = VisitorRowBroadcast        -> bias
//    DOutput  = VisitorAuxLoad             -> d_output
//
//  Level 1 (compute):
//    AddBias  = VisitorCompute<plus>       -> z = Acc + Bias
//
//  Level 2 (compute):
//    SwishBwd = VisitorSwishBackwardCompute -> d_output * Swish'(z)
//
//  Level 3 (store):
//    Store    = VisitorAuxStore            -> d_preactivation
// =============================================================================

using BwdAccum = cutlass::epilogue::threadblock::VisitorAccFetch;

using BwdBias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    EvtOutputTileThreadMap, float,
    Stride<_0, _1, int32_t>
>;

using BwdAddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTBwdAddBias = cutlass::epilogue::threadblock::Sm80EVT<
    BwdAddBias, BwdAccum, BwdBias>;  // z = Acc + Bias

using BwdDOutput = cutlass::epilogue::threadblock::VisitorAuxLoad<
    EvtOutputTileThreadMap, float,
    Stride<int64_t, _1, int64_t>
>;

using EVTBwdSwish = cutlass::epilogue::threadblock::Sm80EVT<
    VisitorSwishBackwardCompute, EVTBwdAddBias, BwdDOutput>;  // SwishBwd(z, d_output)

using BwdStore = cutlass::epilogue::threadblock::VisitorAuxStore<
    EvtOutputTileThreadMap, float,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>
>;

using EVTBackward = cutlass::epilogue::threadblock::Sm80EVT<
    BwdStore, EVTBwdSwish>;  // Store(SwishBwd(...))

// --- Backward EVT GEMM Kernel ---
using EVTBwdKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    float, cutlass::layout::RowMajor,          // A: input [M, K]
    cutlass::ComplexTransform::kNone, EvtAlignment,
    float, cutlass::layout::ColumnMajor,       // B: weights [N, K] -> W^T [K, N]
    cutlass::ComplexTransform::kNone, EvtAlignment,
    float, cutlass::layout::RowMajor,          // C (unused, placeholder)
    EvtAlignment,
    float,                                     // ElementAccumulator
    float,                                     // ElementEpilogue
    cutlass::arch::OpClassTensorOp,            // TensorOp (TF32)
    cutlass::arch::Sm80,
    EvtThreadblockShape,
    EvtWarpShape,
    EvtInstructionShape,
    EVTBackward,                               // <--- Backward EVT callbacks
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    EvtNumStages,
    cutlass::arch::OpMultiplyAdd,
    EvtEpilogueStages
>::GemmKernel;

using GemmSwishBackward = cutlass::gemm::device::GemmUniversalAdapter<EVTBwdKernel>;


// =============================================================================
// Static workspace for EVT operations
// =============================================================================

static void* evt_forward_workspace = nullptr;
static size_t evt_forward_workspace_size = 0;

static void* evt_backward_workspace = nullptr;
static size_t evt_backward_workspace_size = 0;

// =============================================================================
// Extern "C" API
// =============================================================================

extern "C" {

    /*
     * Forward: output = Swish(X * W^T + bias)
     *
     * save_preactivation == 0: Standard forward (SIMT, FP32). Fast and accurate.
     *              Backward requires recomputing z via EVT GEMM.
     *
     * save_preactivation == 1: EVT forward (TensorOp, TF32). Also saves
     *              z = X*W^T + bias in preactivation for backward without recompute.
     *              GEMM uses TF32 (~10-bit mantissa).
     */
    DLLEXPORT void cudaLinearSwishForward(
        float *input,
        float *weights,
        float *bias,
        float *output,
        float *preactivation,        // buffer for z [M, N] (nullptr if save_z=0)
        int inputRowsNum,
        int inputColsNum,
        int outputColsNum,
        float swish_beta,
        int save_preactivation,              // 0: standard forward, 1: dual output (Swish(z) + z)
        cudaStream_t stream
    ) {
        if (save_preactivation) {
            // =================================================================
            // EVT Forward: dual output
            //   output   <- Swish(z)
            //   preactivation <- z = X * W^T + bias
            // =================================================================
            typename EVTForward::Arguments callback_args{
                // child_0: EVTFwdSwish
                {
                    // child_0: EVTFwdStoreZ
                    {
                        // child_0: EVTFwdAddBias = Sm80EVT<FwdAddBias, FwdAccum, FwdBias>
                        {
                            {},                                                       // FwdAccum (no args)
                            {bias, float(0), {_0{}, _1{}, int32_t(outputColsNum)}},   // FwdBias (ptr, null_default, stride)
                            {}                                                        // FwdAddBias (no args)
                        },
                        // node: FwdStoreZ (VisitorAuxStore) -> store z in preactivation
                        {preactivation, {int64_t(outputColsNum), _1{}, int64_t(0)}}
                    },
                    // node: VisitorSwishForwardCompute
                    {swish_beta}
                },
                // node: FwdStoreOutput (VisitorAuxStore) -> store Swish(z) in output
                {output, {int64_t(outputColsNum), _1{}, int64_t(0)}}
            };

            typename GemmSwishForwardEVT::Arguments args(
                cutlass::gemm::GemmUniversalMode::kGemm,
                {inputRowsNum, outputColsNum, inputColsNum},     // problem_size M, N, K
                1,                                                // batch_count
                callback_args,                                    // EVT epilogue arguments
                input,                                            // ptr_A: X [M, K]
                weights,                                          // ptr_B: W [N, K] as ColumnMajor
                nullptr,                                          // ptr_C (unused)
                nullptr,                                          // ptr_D (unused)
                int64_t(inputRowsNum) * inputColsNum,             // batch_stride_A
                int64_t(outputColsNum) * inputColsNum,            // batch_stride_B
                0,                                                // batch_stride_C
                0,                                                // batch_stride_D
                inputColsNum,                                     // lda
                inputColsNum,                                     // ldb
                0,                                                // ldc
                0                                                 // ldd
            );

            GemmSwishForwardEVT gemm_op;
            size_t workspace_size = GemmSwishForwardEVT::get_workspace_size(args);
            
            // Allocate or reallocate workspace if needed
            if (workspace_size > evt_forward_workspace_size) {
                if (evt_forward_workspace != nullptr) {
                    cudaFreeAsync(evt_forward_workspace, stream);
                    evt_forward_workspace = nullptr;
                    evt_forward_workspace_size = 0;
                }
                if (workspace_size > 0) {
                    cudaError_t err = cudaMallocAsync(&evt_forward_workspace, workspace_size, stream);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Malloc Async failed in forward: %s\n", cudaGetErrorString(err));
                        return;
                    }
                    evt_forward_workspace_size = workspace_size;
                }
            }
            
            gemm_op.initialize(args, evt_forward_workspace, stream);
            gemm_op.run(stream);
        } else {
            // =================================================================
            // Standard Forward (SIMT, FP32)
            //   output <- Swish(X * W^T + bias)
            // =================================================================
            float alpha = 1.0f;
            float beta_gemm = (bias != nullptr) ? 1.0f : 0.0f;
            float* ptr_bias = (bias != nullptr) ? bias : output;
            int64_t ldc_bias = (bias != nullptr) ? 0 : outputColsNum;

            cutlass::gemm::GemmCoord problem_size{inputRowsNum, outputColsNum, inputColsNum};

            typename GemmSwish::Arguments args{
                problem_size,
                {input, inputColsNum},
                {weights, inputColsNum},
                {ptr_bias, ldc_bias},
                {output, outputColsNum},
                {alpha, beta_gemm, swish_beta}
            };

            GemmSwish gemm_op;
            gemm_op(args, nullptr, stream);
        }
    }


    /*
     * Backward:
     *
     * recompute_preactivation == 1: Recompute z via fused EVT GEMM (TensorOp TF32).
     *                   d_preactivation = d_output * Swish'(X * W^T + bias)
     *                   Does not require preactivation. Requires input and weights.
     *
     * recompute_preactivation == 0: No recompute - use saved preactivation.
     *                   d_preactivation = preactivation = d_output * Swish'(preactivation)
     *                   Single element-wise kernel, no GEMM for step 1+2.
     *                   Does not require input/weights for this step.
     *
     * Steps 3-5 (dX, dW, dBias) are the same for both modes.
     */
    DLLEXPORT void cudaLinearSwishBackward(
        float *input,           // X (needed for dX, dW, and recompute if recompute_preactivation=1)
        float *weights,         // W (needed for dX, dW, and recompute if recompute_preactivation=1)
        float *bias,            // b (needed for recompute if recompute_preactivation=1)
        float *d_output,        // dL/dy
        float *tmp_buffer,    // TEMP buffer [M, N]
        float *d_input,         // Result: dL/dX
        float *d_weights,       // Result: dL/dW
        float *d_bias,          // Result: dL/db (may be nullptr)
        int inputRowsNum,       // M
        int inputColsNum,       // K
        int outputColsNum,      // N
        float swish_beta,
        int recompute_preactivation,         // 
        cudaStream_t stream
    ) {
        int total_elements = inputRowsNum * outputColsNum;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        // --- STEP 1+2: COMPUTE dL/dz = dL/dy * Swish'(z) ---

        if (!recompute_preactivation) {
            // =================================================================
            // MODE A: Use saved z (no GEMM recompute!)
            // Single element-wise kernel: FP32, as fast as possible.
            // =================================================================
            // at this stage tmp_buffer is preactivation
            swish_backward_kernel<<<blocks, threads, 0, stream>>>(
                tmp_buffer, d_output, swish_beta, total_elements
            );
            // at this stage tmp_buffer is d_preactivation
        } else {
            // =================================================================
            // MODE B: Recompute z via fused EVT GEMM (TensorOp TF32)
            // z = X * W^T + bias  ->  tmp_buffer = d_output * Swish'(z)
            // =================================================================
            // at this stage tmp_buffer is empty
            typename EVTBackward::Arguments callback_args{
                // child_0: EVTBwdSwish
                {
                    // child_0: EVTBwdAddBias = Sm80EVT<BwdAddBias, BwdAccum, BwdBias>
                    {
                        {},                                                       // BwdAccum (no args)
                        {bias, float(0), {_0{}, _1{}, int32_t(outputColsNum)}},   // BwdBias
                        {}                                                        // BwdAddBias (no args)
                    },
                    // child_1: BwdDOutput (VisitorAuxLoad)
                    {d_output, float(0), {int64_t(outputColsNum), _1{}, int64_t(0)}},
                    // node: VisitorSwishBackwardCompute
                    {swish_beta}
                },
                // node: BwdStore (VisitorAuxStore) -> tmp_buffer
                {tmp_buffer, {int64_t(outputColsNum), _1{}, int64_t(0)}}
            };

            typename GemmSwishBackward::Arguments args_bwd(
                cutlass::gemm::GemmUniversalMode::kGemm,
                {inputRowsNum, outputColsNum, inputColsNum},
                1,
                callback_args,
                input,
                weights,
                nullptr,
                nullptr,
                int64_t(inputRowsNum) * inputColsNum,
                int64_t(outputColsNum) * inputColsNum,
                0,
                0,
                inputColsNum,
                inputColsNum,
                0,
                0
            );

            GemmSwishBackward gemm_bwd;
            size_t workspace_size = GemmSwishBackward::get_workspace_size(args_bwd);
            
            // Allocate or reallocate workspace if needed
            if (workspace_size > evt_backward_workspace_size) {
                if (evt_backward_workspace != nullptr) {
                    cudaFreeAsync(evt_backward_workspace, stream);
                    evt_backward_workspace = nullptr;
                    evt_backward_workspace_size = 0;
                }
                if (workspace_size > 0) {
                    cudaError_t err = cudaMallocAsync(&evt_backward_workspace, workspace_size, stream);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Malloc Async failed in backward: %s\n", cudaGetErrorString(err));
                        return;
                    }
                    evt_backward_workspace_size = workspace_size;
                }
            }
            
            gemm_bwd.initialize(args_bwd, evt_backward_workspace, stream);
            gemm_bwd.run(stream);
            // at this stage tmp_buffer is d_preactivation
        }

        // --- STEP 3: INPUT GRADIENT (dX = dL/dz * W) ---
        {
            cutlass::gemm::GemmCoord size_dx{inputRowsNum, inputColsNum, outputColsNum};
            typename GemmRM::Arguments args_dx{
                size_dx,
                {tmp_buffer, outputColsNum},
                {weights, inputColsNum},
                {d_input, inputColsNum},
                {d_input, inputColsNum},
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
                {tmp_buffer, outputColsNum},
                {input, inputColsNum},
                {d_weights, inputColsNum},
                {d_weights, inputColsNum},
                {1.0f, 0.0f}
            };
            GemmTransposedRowRow gemm_dw;
            gemm_dw(args_dw, nullptr, stream);
        }

        // --- STEP 5: BIAS GRADIENT (db = sum(dL/dz, axis=0)) ---
        if (d_bias != nullptr) {
            int threads = 256;
            int blocks = (outputColsNum + threads - 1) / threads;
            sumBiasKernel<<<blocks, threads, 0, stream>>>(
                d_bias, tmp_buffer, inputRowsNum, outputColsNum
            );
        }
    }

    DLLEXPORT void cleanupCudaMemory() {
        if (evt_forward_workspace != nullptr) {
            cudaFree(evt_forward_workspace);
            evt_forward_workspace = nullptr;
            evt_forward_workspace_size = 0;
        }
        if (evt_backward_workspace != nullptr) {
            cudaFree(evt_backward_workspace);
            evt_backward_workspace = nullptr;
            evt_backward_workspace_size = 0;
        }
    }
}
