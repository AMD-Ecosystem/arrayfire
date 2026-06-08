/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cublas.hpp>
#include <hipblas/hipblas.h>
#include <cudaDataType.hpp>
#include <hip/hip_runtime.h>
#include <err_cuda.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <transpose.hpp>
#include <types.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using arrayfire::common::half;
using arrayfire::common::kernel_type;
using std::is_same;
using std::vector;

namespace arrayfire {
namespace cuda {

hipblasOperation_t toCblasTranspose(af_mat_prop opt) {
    hipblasOperation_t out = HIPBLAS_OP_N;
    switch (opt) {
        case AF_MAT_NONE: out = HIPBLAS_OP_N; break;
        case AF_MAT_TRANS: out = HIPBLAS_OP_T; break;
        case AF_MAT_CTRANS: out = HIPBLAS_OP_C; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
using gemm_func_def = std::function<hipblasStatus_t(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const T *, const T *, int, const T *, int, const T *, T *, int)>;

template<typename T>
using gemmBatched_func_def = std::function<hipblasStatus_t(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const T *, const T **, int, const T **, int, const T *, T **, int, int)>;

template<typename T>
using trsm_func_def = std::function<hipblasStatus_t(
    hipblasHandle_t, hipblasSideMode_t, hipblasFillMode_t, hipblasOperation_t,
    hipblasDiagType_t, int, int, const T *, const T *, int, T *, int)>;

#define BLAS_FUNC_DEF(FUNC) \
    template<typename T>    \
    FUNC##_func_def<T> FUNC##_func();

// arrayfire's element types differ from hipBLAS's at the bit-compatible level:
// cfloat/cdouble are plain PODs (not hipComplex = HIP_vector_type, to avoid the
// vector-operator ambiguity) and __half maps to hipBLAS's hipblasHalf (uint16_t).
// So &hipblas{C,Z,H}<fn> has hipComplex*/hipblasHalf* parameters that do not
// implicitly convert to the cfloat*/cdouble*/__half* parameters of the
// FUNC##_func_def<T> typedef. The storage is layout-identical, so reinterpret the
// function pointer. blas_fnptr_cast reinterprets any function pointer to the
// signature the std::function target expects (Sig is the typedef's
// callable signature, extracted via its ::result_type-bearing operator()).
template<typename Target, typename Fn>
static inline Target blas_fnptr_cast(Fn f) {
    return reinterpret_cast<Target>(f);
}

#define BLAS_FUNC(FUNC, TYPE, PREFIX)                                       \
    template<>                                                              \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() {                             \
        return blas_fnptr_cast<FUNC##_raw_t<TYPE>>(&hipblas##PREFIX##FUNC); \
    }

// Raw (function-pointer) form of each std::function typedef, used as the
// reinterpret_cast target so the bit-compatible element-type mismatch is bridged.
template<typename T>
using gemm_raw_t = hipblasStatus_t (*)(hipblasHandle_t, hipblasOperation_t,
                                       hipblasOperation_t, int, int, int,
                                       const T *, const T *, int, const T *, int,
                                       const T *, T *, int);
template<typename T>
using gemmBatched_raw_t = hipblasStatus_t (*)(
    hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int, int, int,
    const T *, const T **, int, const T **, int, const T *, T **, int, int);

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float, S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble, Z)
BLAS_FUNC(gemm, __half, H)

BLAS_FUNC_DEF(gemmBatched)
BLAS_FUNC(gemmBatched, float, S)
BLAS_FUNC(gemmBatched, cfloat, C)
BLAS_FUNC(gemmBatched, double, D)
BLAS_FUNC(gemmBatched, cdouble, Z)
BLAS_FUNC(gemmBatched, __half, H)

template<>
gemm_func_def<schar> gemm_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemm_func_def<schar>();
}
template<>
gemmBatched_func_def<schar> gemmBatched_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemmBatched_func_def<schar>();
}

template<typename T>
using trsm_raw_t = hipblasStatus_t (*)(hipblasHandle_t, hipblasSideMode_t,
                                       hipblasFillMode_t, hipblasOperation_t,
                                       hipblasDiagType_t, int, int, const T *,
                                       const T *, int, T *, int);

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float, S)
BLAS_FUNC(trsm, cfloat, C)
BLAS_FUNC(trsm, double, D)
BLAS_FUNC(trsm, cdouble, Z)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T, bool conjugate>
struct dot_func_def_t {
    typedef hipblasStatus_t (*dot_func_def)(hipblasHandle_t, int, const T *, int,
                                           const T *, int, T *);
};

#define BLAS_FUNC_DEF(FUNC)              \
    template<typename T, bool conjugate> \
    typename FUNC##_func_def_t<T, conjugate>::FUNC##_func_def FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX)                       \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               hipblas##PREFIX##FUNC;                                  \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, float, true, S)
BLAS_FUNC(dot, double, true, D)
BLAS_FUNC(dot, float, false, S)
BLAS_FUNC(dot, double, false, D)

#undef BLAS_FUNC

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX, SUFFIX)               \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               hipblas##PREFIX##FUNC##SUFFIX;                          \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, cfloat, true, C, c)
BLAS_FUNC(dot, cdouble, true, Z, c)
BLAS_FUNC(dot, cfloat, false, C, u)
BLAS_FUNC(dot, cdouble, false, Z, u)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T>
hipblasGemmAlgo_t selectGEMMAlgorithm() {
    return HIPBLAS_GEMM_DEFAULT;
}

template<>
hipblasGemmAlgo_t selectGEMMAlgorithm<common::half>() {
    // hipBLAS has only HIPBLAS_GEMM_DEFAULT (no per-arch TENSOR_OP variant);
    // rocBLAS picks the MFMA path internally on CDNA. The cuBLAS tensor-op
    // branch is dropped.
    return HIPBLAS_GEMM_DEFAULT;
}

template<>
hipblasGemmAlgo_t selectGEMMAlgorithm<__half>() {
    return selectGEMMAlgorithm<common::half>();
}

template<typename Ti, typename To = Ti>
hipblasStatus_t gemmDispatch(BlasHandle handle, hipblasOperation_t lOpts,
                            hipblasOperation_t rOpts, int M, int N, int K,
                            const To *alpha, const Array<Ti> &lhs, dim_t lStride,
                            const Array<Ti> &rhs, dim_t rStride, const To *beta,
                            Array<To> &out, dim_t oleading) {
    auto prop = getDeviceProp(getActiveDeviceId());
    // On ROCm the typed gemm_func path (hipblasSgemm/Cgemm/.../Hgemm) is the
    // validated route for float/complex/half, but it has no schar (int8)
    // specialization (AF_ERR_TYPE). Route ONLY int8 through hipblasGemmEx;
    // hipblasGemmEx for half on hipBLAS regresses accuracy, so the typed Hgemm
    // is kept for it. On CUDA the original __CUDACC_VER_MAJOR__>=10 Ex path is
    // preserved.
#if defined(__HIP_PLATFORM_AMD__)
    if constexpr (std::is_same<Ti, schar>::value) {
        // rocBLAS rejects int8 in / float32 out|compute (HIPBLAS_STATUS_NOT_
        // SUPPORTED), but gfx9/CDNA has int8 MFMA and the supported int8 GEMM is
        // int8 x int8 -> int32 accumulate. Compute into an int32 buffer with
        // HIPBLAS_COMPUTE_32I, then cast the int32 result into the arrayfire
        // float output (out_type for s8 is f32). alpha/beta are integer scales
        // for the int path (the s8 gemm contract); round the float scalars.
        // (if constexpr so the int32 cast is only emitted for Ti == schar; the
        // float/complex instantiations skip this branch at compile time.)
        if (prop.major > 3) {
            Array<int> outI = createEmptyArray<int>(out.dims());
            int alphaI      = static_cast<int>(*alpha);
            int betaI       = static_cast<int>(*beta);
            hipblasStatus_t st = hipblasGemmEx(
                blasHandle(), lOpts, rOpts, M, N, K, &alphaI, lhs.get(),
                getType<Ti>(), lStride, rhs.get(), getType<Ti>(), rStride,
                &betaI, outI.get(), HIP_R_32I, outI.strides()[1],
                HIPBLAS_COMPUTE_32I, selectGEMMAlgorithm<Ti>());
            if (st != HIPBLAS_STATUS_SUCCESS) { return st; }
            copyArray<int, To>(out, outI);
            return HIPBLAS_STATUS_SUCCESS;
        }
    }
#elif __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3) {
        return hipblasGemmEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs.get(), getType<Ti>(),
            lStride, rhs.get(), getType<Ti>(), rStride, beta, out.get(),
            getType<To>(), out.strides()[1],
            getComputeType<To>(),  // Compute type

            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The HIPBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is HIP_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemm_func<Nt>()(blasHandle(), lOpts, rOpts, M, N, K, (Nt *)alpha,
                               (Nt *)lhs.get(), lStride, (Nt *)rhs.get(),
                               rStride, (Nt *)beta, (Nt *)out.get(), oleading);

#if !defined(__HIP_PLATFORM_AMD__) && __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename Ti, typename To = Ti>
hipblasStatus_t gemmBatchedDispatch(BlasHandle handle, hipblasOperation_t lOpts,
                                   hipblasOperation_t rOpts, int M, int N, int K,
                                   const To *alpha, const Ti **lptrs,
                                   int lStrides, const Ti **rptrs, int rStrides,
                                   const To *beta, To **optrs, int oStrides,
                                   int batchSize) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3) {
        return hipblasGemmBatchedEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, (const void **)lptrs,
            getType<Ti>(), lStrides, (const void **)rptrs, getType<Ti>(),
            rStrides, beta, (void **)optrs, getType<Ti>(), oStrides, batchSize,
            getComputeType<Ti>(),  // compute type
            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The HIPBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is HIP_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemmBatched_func<Nt>()(
            blasHandle(), lOpts, rOpts, M, N, K, (const Nt *)alpha,
            (const Nt **)lptrs, lStrides, (const Nt **)rptrs, rStrides,
            (const Nt *)beta, (Nt **)optrs, oStrides, batchSize);
#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename Ti, typename To>
void gemm(Array<To> &out, af_mat_prop optLhs, af_mat_prop optRhs, const To *alpha,
          const Array<Ti> &lhs, const Array<Ti> &rhs, const To *beta) {
    const hipblasOperation_t lOpts = toCblasTranspose(optLhs);
    const hipblasOperation_t rOpts = toCblasTranspose(optRhs);

    const int aRowDim = (lOpts == HIPBLAS_OP_N) ? 0 : 1;
    const int aColDim = (lOpts == HIPBLAS_OP_N) ? 1 : 0;
    const int bColDim = (rOpts == HIPBLAS_OP_N) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M      = lDims[aRowDim];
    const int N      = rDims[bColDim];
    const int K      = lDims[aColDim];
    const dim4 oDims = out.dims();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    dim4 oStrides = out.strides();

    if (oDims.ndims() <= 2) {
        CUBLAS_CHECK((gemmDispatch<Ti, To>(blasHandle(), lOpts, rOpts, M, N, K, alpha,
                                           lhs, lStrides[1], rhs, rStrides[1], beta,
                                           out, oStrides[1])));
    } else {
        int batchSize = oDims[2] * oDims[3];
        vector<const Ti *> lptrs(batchSize);
        vector<const Ti *> rptrs(batchSize);
        vector<To *> optrs(batchSize);

        bool is_l_d2_batched = oDims[2] == lDims[2];
        bool is_l_d3_batched = oDims[3] == lDims[3];

        bool is_r_d2_batched = oDims[2] == rDims[2];
        bool is_r_d3_batched = oDims[3] == rDims[3];

        const Ti *lptr = lhs.get();
        const Ti *rptr = rhs.get();
        To *optr    = out.get();

        for (int n = 0; n < batchSize; n++) {
            int w    = n / oDims[2];
            int z    = n - w * oDims[2];
            int loff = z * (is_l_d2_batched * lStrides[2]) +
                       w * (is_l_d3_batched * lStrides[3]);
            int roff = z * (is_r_d2_batched * rStrides[2]) +
                       w * (is_r_d3_batched * rStrides[3]);
            lptrs[n] = lptr + loff;
            rptrs[n] = rptr + roff;
            optrs[n] = optr + z * oStrides[2] + w * oStrides[3];
        }

        size_t bytes = batchSize * sizeof(Ti **);
        auto d_lptrs = memAlloc<uchar>(bytes);
        auto d_rptrs = memAlloc<uchar>(bytes);
        auto d_optrs = memAlloc<uchar>(bytes);
        CUDA_CHECK(hipMemcpyAsync(d_lptrs.get(), lptrs.data(), bytes,
                                   hipMemcpyHostToDevice, getActiveStream()));
        CUDA_CHECK(hipMemcpyAsync(d_rptrs.get(), rptrs.data(), bytes,
                                   hipMemcpyHostToDevice, getActiveStream()));
        CUDA_CHECK(hipMemcpyAsync(d_optrs.get(), optrs.data(), bytes,
                                   hipMemcpyHostToDevice, getActiveStream()));

        // Call this before the gemm call so that you don't have to wait for the
        // computation. Even though it would make more sense to put it
        // afterwards
        CUDA_CHECK(hipStreamSynchronize(getActiveStream()));

        using Nt = typename common::kernel_type<Ti>::native;
        CUBLAS_CHECK(gemmBatchedDispatch(
            blasHandle(), lOpts, rOpts, M, N, K, alpha,
            (const Ti **)d_lptrs.get(), lStrides[1], (const Ti **)d_rptrs.get(),
            rStrides[1], beta, (To **)d_optrs.get(), oStrides[1], batchSize));
    }
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    auto lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    auto rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));
    auto temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, af_mat_prop trans, bool is_upper,
          bool is_left, bool is_unit) {
    // dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = rDims[0];
    int N      = rDims[1];

    T alpha = scalar<T>(1);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();

    CUBLAS_CHECK(trsm_func<T>()(
        blasHandle(), is_left ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT,
        is_upper ? HIPBLAS_FILL_MODE_UPPER : HIPBLAS_FILL_MODE_LOWER,
        toCblasTranspose(trans),
        is_unit ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT, M, N, &alpha,
        lhs.get(), lStrides[1], rhs.get(), rStrides[1]));
}

#define INSTANTIATE_GEMM(TYPE, OUTTYPE)                                      \
    template void gemm<TYPE>(Array<OUTTYPE> & out, af_mat_prop optLhs,       \
                             af_mat_prop optRhs, const OUTTYPE *alpha,       \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const OUTTYPE *beta);

INSTANTIATE_GEMM(float, float)
INSTANTIATE_GEMM(cfloat, cfloat)
INSTANTIATE_GEMM(double, double)
INSTANTIATE_GEMM(cdouble, cdouble)
INSTANTIATE_GEMM(half, half)
INSTANTIATE_GEMM(schar, float)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)
INSTANTIATE_DOT(half)

#define INSTANTIATE_TRSM(TYPE)                                               \
    template void trsm<TYPE>(const Array<TYPE> &lhs, Array<TYPE> &rhs,       \
                             af_mat_prop trans, bool is_upper, bool is_left, \
                             bool is_unit);

INSTANTIATE_TRSM(float)
INSTANTIATE_TRSM(cfloat)
INSTANTIATE_TRSM(double)
INSTANTIATE_TRSM(cdouble)

}  // namespace cuda
}  // namespace arrayfire
