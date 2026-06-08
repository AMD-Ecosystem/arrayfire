/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP backend: sparse matmul (SpMV / SpMM) on hipSPARSE's generic API. Ported
// from src/backend/cuda/sparse_blas.cu. The cuSPARSE generic calls keep their
// cuSPARSE spelling (aliased to hipsparse* by nvrtc_shims/cusparse_v2.h) and run
// against the directly-linked roc::hipsparse (no dlopen plugin). The descriptor
// RAII comes from cusparse_descriptor_helpers.hpp (tag-keyed, the void* fix).
//
// HIP delta: hipsparseSpMV/SpMM take the COMPUTE type as a hipDataType, so this
// uses getType<T>() (NOT getComputeType<T>(); on HIP the latter returns a
// hipblasComputeType_t for the dense gemm Ex path, a different enum family).

#include <sparse_blas.hpp>

#include <common/err_common.hpp>
#include <complex.hpp>
#include <cudaDataType.hpp>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <math.hpp>
#include <platform.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

hipsparseOperation_t toCusparseTranspose(af_mat_prop opt) {
    hipsparseOperation_t out = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    switch (opt) {
        case AF_MAT_NONE: out = HIPSPARSE_OPERATION_NON_TRANSPOSE; break;
        case AF_MAT_TRANS: out = HIPSPARSE_OPERATION_TRANSPOSE; break;
        case AF_MAT_CTRANS:
            out = HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
size_t spmvBufferSize(hipsparseOperation_t opA, const T *alpha,
                      const hipsparseSpMatDescr_t matA,
                      const hipsparseDnVecDescr_t vecX, const T *beta,
                      const hipsparseDnVecDescr_t vecY) {
    size_t retVal = 0;
    CUSPARSE_CHECK(hipsparseSpMV_bufferSize(
        sparseHandle(), opA, alpha, matA, vecX, beta, vecY, getType<T>(),
        HIPSPARSE_SPMV_CSR_ALG1, &retVal));
    return retVal;
}

template<typename T>
void spmv(hipsparseOperation_t opA, const T *alpha,
          const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t vecX,
          const T *beta, const hipsparseDnVecDescr_t vecY, void *buffer) {
    CUSPARSE_CHECK(hipsparseSpMV(sparseHandle(), opA, alpha, matA, vecX, beta,
                                 vecY, getType<T>(), HIPSPARSE_SPMV_ALG_DEFAULT,
                                 buffer));
}

template<typename T>
size_t spmmBufferSize(hipsparseOperation_t opA, hipsparseOperation_t opB,
                      const T *alpha, const hipsparseSpMatDescr_t matA,
                      const hipsparseDnMatDescr_t matB, const T *beta,
                      const hipsparseDnMatDescr_t matC) {
    size_t retVal = 0;
    CUSPARSE_CHECK(hipsparseSpMM_bufferSize(
        sparseHandle(), opA, opB, alpha, matA, matB, beta, matC, getType<T>(),
        HIPSPARSE_SPMM_CSR_ALG1, &retVal));
    return retVal;
}

template<typename T>
void spmm(hipsparseOperation_t opA, hipsparseOperation_t opB, const T *alpha,
          const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB,
          const T *beta, const hipsparseDnMatDescr_t matC, void *buffer) {
    CUSPARSE_CHECK(hipsparseSpMM(sparseHandle(), opA, opB, alpha, matA, matB,
                                 beta, matC, getType<T>(),
                                 HIPSPARSE_SPMM_CSR_ALG1, buffer));
}

template<typename T>
Array<T> matmul(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    // Similar Operations to GEMM
    hipsparseOperation_t lOpts = toCusparseTranspose(optLhs);

    int lRowDim = (lOpts == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    static const int rColDim = 1;  // Unsupported : (rOpts ==
                                   // HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = lDims[lRowDim];
    int N      = rDims[rColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    T alpha      = scalar<T>(1);
    T beta       = scalar<T>(0);

    auto spMat = cusparseDescriptor<T>(lhs);

    if (rDims[rColDim] == 1) {
        auto dnVec = denVecDescriptor<T>(rhs);
        auto dnOut = denVecDescriptor<T>(out);
        size_t bufferSize =
            spmvBufferSize<T>(lOpts, &alpha, spMat, dnVec, &beta, dnOut);
        auto tempBuffer = createEmptyArray<char>(dim4(bufferSize));
        spmv<T>(lOpts, &alpha, spMat, dnVec, &beta, dnOut, tempBuffer.get());
    } else {
        hipsparseOperation_t rOpts = toCusparseTranspose(optRhs);

        auto dnMat = denMatDescriptor<T>(rhs);
        auto dnOut = denMatDescriptor<T>(out);
        size_t bufferSize =
            spmmBufferSize<T>(lOpts, rOpts, &alpha, spMat, dnMat, &beta, dnOut);
        auto tempBuffer = createEmptyArray<char>(dim4(bufferSize));
        spmm<T>(lOpts, rOpts, &alpha, spMat, dnMat, &beta, dnOut,
                tempBuffer.get());
    }

    return out;
}

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T> &lhs,       \
                                const Array<T> &rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace cuda
}  // namespace arrayfire
