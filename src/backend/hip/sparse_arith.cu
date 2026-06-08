/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP backend: sparse arithmetic. The dense-broadcast ops (arithOpD, sparse op
// dense) are pure arrayfire kernels and port verbatim from the CUDA backend.
// The sparse-op-sparse add/sub is hipSPARSE csrgeam2 (ported from
// src/backend/cuda/sparse_arith.cu). The CUDA backend reaches csrgeam2 through
// the dlopen plugin (_.cusparseScsrgeam2); on HIP we link roc::hipsparse and
// call the typed S/D/C/Z functions directly via small dispatch wrappers.
//
// HIP delta: hipSPARSE's typed complex csrgeam2 takes const hipComplex* /
// hipDoubleComplex*, but arrayfire's cfloat/cdouble are distinct (layout-
// compatible) POD structs on the compiled path (types.hpp), so the complex
// value/alpha/beta pointers are reinterpret_cast to the hip complex type at the
// call boundary (the cublas Hgemm reinterpret precedent).

#include <kernel/sparse_arith.hpp>
#include <sparse_arith.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <where.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

using namespace common;
using std::numeric_limits;

template<typename T>
T getInf() {
    return scalar<T>(numeric_limits<T>::infinity());
}

template<>
cfloat getInf() {
    return scalar<cfloat, float>(
        NAN, NAN);  // Matches behavior of complex division by 0 in CUDA
}

template<>
cdouble getInf() {
    return scalar<cdouble, double>(
        NAN, NAN);  // Matches behavior of complex division by 0 in CUDA
}

template<typename T, af_op_t op>
Array<T> arithOpD(const SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse) {
    lhs.eval();
    rhs.eval();

    Array<T> out  = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch (op) {
        case af_add_t: out = copyArray<T>(rhs); break;
        case af_sub_t:
            out = reverse ? copyArray<T>(rhs)
                          : arithOp<T, af_sub_t>(zero, rhs, rhs.dims());
            break;
        default: out = copyArray<T>(rhs);
    }
    out.eval();
    switch (lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out, lhs.getValues(),
                                            lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out, lhs.getValues(),
                                            lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs,
                       const bool reverse) {
    lhs.eval();
    rhs.eval();

    SparseArray<T> out = createArrayDataSparseArray<T>(
        lhs.dims(), lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
        lhs.getStorage(), true);
    out.eval();
    switch (lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out.getValues(), out.getRowIdx(),
                                            out.getColIdx(), rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out.getValues(), out.getRowIdx(),
                                            out.getColIdx(), rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

// hipSPARSE typed csrgeam2 dispatch. The complex element/alpha/beta pointers
// are reinterpret_cast to the hip complex type (arrayfire's cfloat/cdouble are
// layout-compatible POD structs, not hipComplex, on the compiled HIP path).
template<typename T>
hipsparseStatus_t csrgeam2BufferSize(int m, int n, const T *alpha, int nnzA,
                                     const T *valA, const int *rowA,
                                     const int *colA, const T *beta, int nnzB,
                                     const T *valB, const int *rowB,
                                     const int *colB, hipsparseMatDescr_t desc,
                                     size_t *bytes);

template<>
hipsparseStatus_t csrgeam2BufferSize<float>(
    int m, int n, const float *alpha, int nnzA, const float *valA,
    const int *rowA, const int *colA, const float *beta, int nnzB,
    const float *valB, const int *rowB, const int *colB,
    hipsparseMatDescr_t desc, size_t *bytes) {
    return hipsparseScsrgeam2_bufferSizeExt(
        sparseHandle(), m, n, alpha, desc, nnzA, valA, rowA, colA, beta, desc,
        nnzB, valB, rowB, colB, desc, nullptr, nullptr, nullptr, bytes);
}

template<>
hipsparseStatus_t csrgeam2BufferSize<double>(
    int m, int n, const double *alpha, int nnzA, const double *valA,
    const int *rowA, const int *colA, const double *beta, int nnzB,
    const double *valB, const int *rowB, const int *colB,
    hipsparseMatDescr_t desc, size_t *bytes) {
    return hipsparseDcsrgeam2_bufferSizeExt(
        sparseHandle(), m, n, alpha, desc, nnzA, valA, rowA, colA, beta, desc,
        nnzB, valB, rowB, colB, desc, nullptr, nullptr, nullptr, bytes);
}

template<>
hipsparseStatus_t csrgeam2BufferSize<cfloat>(
    int m, int n, const cfloat *alpha, int nnzA, const cfloat *valA,
    const int *rowA, const int *colA, const cfloat *beta, int nnzB,
    const cfloat *valB, const int *rowB, const int *colB,
    hipsparseMatDescr_t desc, size_t *bytes) {
    return hipsparseCcsrgeam2_bufferSizeExt(
        sparseHandle(), m, n, (const hipComplex *)alpha, desc, nnzA,
        (const hipComplex *)valA, rowA, colA, (const hipComplex *)beta, desc,
        nnzB, (const hipComplex *)valB, rowB, colB, desc, nullptr, nullptr,
        nullptr, bytes);
}

template<>
hipsparseStatus_t csrgeam2BufferSize<cdouble>(
    int m, int n, const cdouble *alpha, int nnzA, const cdouble *valA,
    const int *rowA, const int *colA, const cdouble *beta, int nnzB,
    const cdouble *valB, const int *rowB, const int *colB,
    hipsparseMatDescr_t desc, size_t *bytes) {
    return hipsparseZcsrgeam2_bufferSizeExt(
        sparseHandle(), m, n, (const hipDoubleComplex *)alpha, desc, nnzA,
        (const hipDoubleComplex *)valA, rowA, colA,
        (const hipDoubleComplex *)beta, desc, nnzB,
        (const hipDoubleComplex *)valB, rowB, colB, desc, nullptr, nullptr,
        nullptr, bytes);
}

template<typename T>
hipsparseStatus_t csrgeam2(int m, int n, const T *alpha, int nnzA,
                           const T *valA, const int *rowA, const int *colA,
                           const T *beta, int nnzB, const T *valB,
                           const int *rowB, const int *colB,
                           hipsparseMatDescr_t desc, T *valC, int *rowC,
                           int *colC, void *buffer);

template<>
hipsparseStatus_t csrgeam2<float>(int m, int n, const float *alpha, int nnzA,
                                  const float *valA, const int *rowA,
                                  const int *colA, const float *beta, int nnzB,
                                  const float *valB, const int *rowB,
                                  const int *colB, hipsparseMatDescr_t desc,
                                  float *valC, int *rowC, int *colC,
                                  void *buffer) {
    return hipsparseScsrgeam2(sparseHandle(), m, n, alpha, desc, nnzA, valA,
                              rowA, colA, beta, desc, nnzB, valB, rowB, colB,
                              desc, valC, rowC, colC, buffer);
}

template<>
hipsparseStatus_t csrgeam2<double>(int m, int n, const double *alpha, int nnzA,
                                   const double *valA, const int *rowA,
                                   const int *colA, const double *beta,
                                   int nnzB, const double *valB,
                                   const int *rowB, const int *colB,
                                   hipsparseMatDescr_t desc, double *valC,
                                   int *rowC, int *colC, void *buffer) {
    return hipsparseDcsrgeam2(sparseHandle(), m, n, alpha, desc, nnzA, valA,
                              rowA, colA, beta, desc, nnzB, valB, rowB, colB,
                              desc, valC, rowC, colC, buffer);
}

template<>
hipsparseStatus_t csrgeam2<cfloat>(int m, int n, const cfloat *alpha, int nnzA,
                                   const cfloat *valA, const int *rowA,
                                   const int *colA, const cfloat *beta,
                                   int nnzB, const cfloat *valB,
                                   const int *rowB, const int *colB,
                                   hipsparseMatDescr_t desc, cfloat *valC,
                                   int *rowC, int *colC, void *buffer) {
    return hipsparseCcsrgeam2(
        sparseHandle(), m, n, (const hipComplex *)alpha, desc, nnzA,
        (const hipComplex *)valA, rowA, colA, (const hipComplex *)beta, desc,
        nnzB, (const hipComplex *)valB, rowB, colB, desc, (hipComplex *)valC,
        rowC, colC, buffer);
}

template<>
hipsparseStatus_t csrgeam2<cdouble>(int m, int n, const cdouble *alpha,
                                    int nnzA, const cdouble *valA,
                                    const int *rowA, const int *colA,
                                    const cdouble *beta, int nnzB,
                                    const cdouble *valB, const int *rowB,
                                    const int *colB, hipsparseMatDescr_t desc,
                                    cdouble *valC, int *rowC, int *colC,
                                    void *buffer) {
    return hipsparseZcsrgeam2(
        sparseHandle(), m, n, (const hipDoubleComplex *)alpha, desc, nnzA,
        (const hipDoubleComplex *)valA, rowA, colA,
        (const hipDoubleComplex *)beta, desc, nnzB,
        (const hipDoubleComplex *)valB, rowB, colB, desc, (hipDoubleComplex *)valC,
        rowC, colC, buffer);
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs) {
    af::storage sfmt = lhs.getStorage();
    auto ldesc       = make_handle<SparseDescriptorRAII>();
    auto rdesc       = make_handle<SparseDescriptorRAII>();
    auto odesc       = make_handle<SparseDescriptorRAII>();

    const dim4 ldims      = lhs.dims();
    const int M           = ldims[0];
    const int N           = ldims[1];
    const dim_t nnzA      = lhs.getNNZ();
    const dim_t nnzB      = rhs.getNNZ();
    const int *csrRowPtrA = lhs.getRowIdx().get();
    const int *csrColPtrA = lhs.getColIdx().get();
    const int *csrRowPtrB = rhs.getRowIdx().get();
    const int *csrColPtrB = rhs.getColIdx().get();

    int baseC, nnzC = M + 1;

    auto nnzDevHostPtr = memAlloc<int>(1);
    auto outRowIdx     = createValueArray<int>(M + 1, 0);

    T alpha = scalar<T>(1);
    T beta  = op == af_sub_t ? scalar<T>(-1) : scalar<T>(1);

    T *csrValC      = nullptr;
    int *csrColIndC = nullptr;

    size_t pBufferSize = 0;
    CUSPARSE_CHECK(csrgeam2BufferSize<T>(
        M, N, &alpha, nnzA, lhs.getValues().get(), csrRowPtrA, csrColPtrA, &beta,
        nnzB, rhs.getValues().get(), csrRowPtrB, csrColPtrB, ldesc,
        &pBufferSize));

    auto tmpBuffer = memAlloc<char>(pBufferSize);
    CUSPARSE_CHECK(hipsparseXcsrgeam2Nnz(
        sparseHandle(), M, N, ldesc, nnzA, csrRowPtrA, csrColPtrA, rdesc, nnzB,
        csrRowPtrB, csrColPtrB, odesc, outRowIdx.get(), nnzDevHostPtr.get(),
        tmpBuffer.get()));

    if (NULL != nnzDevHostPtr) {
        CUDA_CHECK(hipMemcpyAsync(&nnzC, nnzDevHostPtr.get(), sizeof(int),
                                  hipMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(hipStreamSynchronize(cuda::getActiveStream()));
    } else {
        CUDA_CHECK(hipMemcpyAsync(&nnzC, outRowIdx.get() + M, sizeof(int),
                                  hipMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(hipMemcpyAsync(&baseC, outRowIdx.get(), sizeof(int),
                                  hipMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(hipStreamSynchronize(cuda::getActiveStream()));
        nnzC -= baseC;
    }
    auto outColIdx = createEmptyArray<int>(nnzC);
    auto outValues = createEmptyArray<T>(nnzC);

    CUSPARSE_CHECK(csrgeam2<T>(M, N, &alpha, nnzA, lhs.getValues().get(),
                               csrRowPtrA, csrColPtrA, &beta, nnzB,
                               rhs.getValues().get(), csrRowPtrB, csrColPtrB,
                               ldesc, outValues.get(), outRowIdx.get(),
                               outColIdx.get(), tmpBuffer.get()));

    SparseArray<T> retVal = createArrayDataSparseArray(
        ldims, outValues, outRowIdx, outColIdx, sfmt);
    return retVal;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> arithOpD<T, af_add_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_sub_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_mul_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_div_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace cuda
}  // namespace arrayfire
