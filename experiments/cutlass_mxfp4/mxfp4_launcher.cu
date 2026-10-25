// MXFP4 SM120 GEMM launcher for our row-major MX4 layout, callable from Python by ctypes.
// Built for sm_120a with CUTLASS 4.8.1.  Weights are repacked once (B is N-contiguous, scales are
// interleaved); A data is used as-is (row-major, K-contiguous); A scales are repacked per call.
#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace mxfp4_launcher {

using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentAB = 32;

template <class ElementC, class ElementD, int TM, int TN, int TK>
struct Kernel {
  using ThreadBlockShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutDTag, 128 / cutlass::sizeof_bits<ElementD>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentAB,
      ElementB, LayoutBTag, AlignmentAB,
      ElementAccumulator, ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using KernelT = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                      CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<KernelT>;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

template <class T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

__device__ __forceinline__ float e2m1_value(int c) {
  const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float x = v[c & 7];
  return (c & 8) ? -x : x;
}

template <class TensorB>
__global__ void repack_b_kernel(TensorB tB, const uint8_t* q, int N, int K) {
 // B is K-contiguous (strideB (K,1)): one thread writes both nibbles of a byte (k = 2*kp, 2*kp+1).
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * (K / 2)) return;
  int n = idx / (K / 2), kp = idx % (K / 2);
  uint8_t b = q[idx];
  tB(n, 2 * kp, 0) = ElementB::DataType(e2m1_value(b & 0xF));
  tB(n, 2 * kp + 1, 0) = ElementB::DataType(e2m1_value(b >> 4));
}

template <class TensorA>
__global__ void repack_a_kernel(TensorA tA, const uint8_t* q, int M, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * (K / 2)) return;
  int m = idx / (K / 2), kp = idx % (K / 2);
  uint8_t b = q[idx];
  tA(m, 2 * kp, 0) = ElementA::DataType(e2m1_value(b & 0xF));
  tA(m, 2 * kp + 1, 0) = ElementA::DataType(e2m1_value(b >> 4));
}

template <class TensorSF>
__global__ void repack_sf_kernel(TensorSF t, const uint8_t* s, int MN, int KS) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= MN * KS) return;
  int mn = idx / KS, ks = idx % KS;
  t(mn, ks * 32, 0) = typename TensorSF::value_type(ldexpf(1.0f, int(s[idx]) - 127));
}

template <class ElementC, class ElementD, class KernelT>
void run(const void* aq, const void* as, const void* bq, const void* bs, void* out,
         int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using Gemm = typename KernelT::Gemm;
  using LayoutSFA = typename KernelT::LayoutSFA;
  using LayoutSFB = typename KernelT::LayoutSFB;
  using Sm1xx = typename KernelT::Sm1xxBlkScaledConfig;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  LayoutSFA layout_SFA = Sm1xx::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA::DataType const*>(aq), stride_A,
       reinterpret_cast<ElementB::DataType const*>(bq), stride_B,
       reinterpret_cast<ElementA::ScaleFactorType const*>(as), layout_SFA,
       reinterpret_cast<ElementB::ScaleFactorType const*>(bs), layout_SFB},
      {{alpha, beta}, reinterpret_cast<ElementC*>(out), stride_C, reinterpret_cast<ElementD*>(out), stride_D}};

  static void* workspace = nullptr;
  static size_t workspace_size = 0;
  size_t need = Gemm::get_workspace_size(args);
  if (need > workspace_size) {
    if (workspace) cudaFree(workspace);
    cudaMalloc(&workspace, need);
    workspace_size = need;
  }
  Gemm gemm;
  gemm.can_implement(args);
  gemm.initialize(args, workspace, stream);
  gemm.run(stream);
}

}  // namespace mxfp4_launcher

using namespace mxfp4_launcher;

extern "C" {

void cutlass_b_sizes(int N, int K, long* b_bytes, long* sf_bytes) {
  using Gemm = typename Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>::Gemm;
  using Sm1xx = typename Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>::Sm1xxBlkScaledConfig;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
  auto layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
  *b_bytes = (long)((size(layout_B) + 1) / 2);
  *sf_bytes = (long)size(filter_zeros(layout_SFB));
}

void cutlass_prepare_b(const void* q, const void* s, void* bq, void* bs, int N, int K, cudaStream_t stream) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  using StrideB = typename K0::Gemm::GemmKernel::StrideB;
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
  auto layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
  auto tB = make_tensor(make_iterator(reinterpret_cast<ElementB::DataType*>(bq)), layout_B);
  auto tS = make_tensor(reinterpret_cast<ElementB::ScaleFactorType*>(bs), layout_SFB);
  int pairs = N * (K / 2), scale_n = N * (K / 32);
  repack_b_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(tB, (const uint8_t*)q, N, K);
  repack_sf_kernel<<<(scale_n + 255) / 256, 256, 0, stream>>>(tS, (const uint8_t*)s, N, K / 32);
}

void cutlass_prepare_a(const void* q, void* dst, int M, int K, cudaStream_t stream) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>;
  using StrideA = typename K0::Gemm::GemmKernel::StrideA;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
  auto tA = make_tensor(make_iterator(reinterpret_cast<ElementA::DataType*>(dst)), layout_A);
  int n = M * (K / 2);
  repack_a_kernel<<<(n + 255) / 256, 256, 0, stream>>>(tA, (const uint8_t*)q, M, K);
}

void cutlass_prepare_sfa(const void* s, void* dst, int M, int K, cudaStream_t stream) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xx::tile_atom_to_shape_SFA(make_shape(M, 1, K, 1));
  auto tS = make_tensor(reinterpret_cast<ElementA::ScaleFactorType*>(dst), layout_SFA);
  int n = M * (K / 32);
  repack_sf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(tS, (const uint8_t*)s, M, K / 32);
}

void cutlass_gemm(const void* aq, const void* as, const void* bq, const void* bs, void* out,
                  int M, int N, int K, int epi, int tile, cudaStream_t stream) {
  if (epi == 0) {
    switch (tile) {
      case 2: run<cutlass::bfloat16_t, cutlass::bfloat16_t, Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 128>>(aq, as, bq, bs, out, M, N, K, 1.f, 0.f, stream); break;
      default: run<cutlass::bfloat16_t, cutlass::bfloat16_t, Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128, 128>>(aq, as, bq, bs, out, M, N, K, 1.f, 0.f, stream);
    }
  } else {
    switch (tile) {
      case 2: run<float, float, Kernel<float, float, 64, 128, 128>>(aq, as, bq, bs, out, M, N, K, 1.f, 1.f, stream); break;
      default: run<float, float, Kernel<float, float, 128, 128, 128>>(aq, as, bq, bs, out, M, N, K, 1.f, 1.f, stream);
    }
  }
}

}  // extern "C"
