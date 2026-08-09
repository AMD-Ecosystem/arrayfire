/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP backend: sparse storage conversions on hipSPARSE. Ported from
// src/backend/cuda/sparse.cu (the AF_USE_NEW_CUSPARSE_API / generic-API path):
// cusparseDenseToSparse_* and cusparseSparseToDense_* for dense<->CSR/CSC, plus
// the legacy Xcsrsort / Xcsr2coo / Xcoo2csr / Xcoosort coordinate-conversion and
// sort surface, and the COO<->dense arrayfire kernels. The CUDA backend reaches
// these through the dlopen plugin (_.cusparseXxx); on HIP we link roc::hipsparse
// and call the hipsparse functions directly. Descriptor RAII is tag-keyed
// because the descriptor types alias one void*. matB in DenseToStorage is
// created raw (its pointers are re-set after the size query) and explicitly
// destroyed before returning.

#include <sparse.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cudaDataType.hpp>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <handle.hpp>
#include <kernel/sparse.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <where.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

using namespace common;

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in) {
    Array<uint> nonZeroIdx_ = where<T>(in);
    Array<int> nonZeroIdx   = cast<int, uint>(nonZeroIdx_);

    dim_t nNZ = nonZeroIdx.elements();

    Array<int> constDim = createValueArray<int>(dim4(nNZ), in.dims()[0]);

    Array<int> rowIdx =
        arithOp<int, af_mod_t>(nonZeroIdx, constDim, nonZeroIdx.dims());
    Array<int> colIdx =
        arithOp<int, af_div_t>(nonZeroIdx, constDim, nonZeroIdx.dims());

    Array<T> values = copyArray<T>(in);
    values.modDims(dim4(values.elements()));
    values = lookup<T, int>(values, nonZeroIdx, 0);

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx,
                                         AF_STORAGE_COO);
}

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in) {
    const int M = in.dims()[0];
    const int N = in.dims()[1];

    auto matA = denMatDescriptor(in);
    hipsparseSpMatDescr_t matB;

    Array<int> d_offsets = createEmptyArray<int>(0);

    if (stype == AF_STORAGE_CSR) {
        d_offsets = createEmptyArray<int>(M + 1);
        // Create sparse matrix B in CSR format
        CUSPARSE_CHECK(hipsparseCreateCsr(
            &matB, M, N, 0, d_offsets.get(), nullptr, nullptr,
            HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO,
            getType<T>()));
    } else {
        d_offsets = createEmptyArray<int>(N + 1);
        CUSPARSE_CHECK(hipsparseCreateCsc(
            &matB, M, N, 0, d_offsets.get(), nullptr, nullptr,
            HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO,
            getType<T>()));
    }

    // allocate an external buffer if needed
    size_t bufferSize;
    CUSPARSE_CHECK(hipsparseDenseToSparse_bufferSize(
        sparseHandle(), matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize));

    auto dBuffer = memAlloc<char>(bufferSize);

    // execute Dense to Sparse analysis
    CUSPARSE_CHECK(hipsparseDenseToSparse_analysis(
        sparseHandle(), matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer.get()));
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CUSPARSE_CHECK(
        hipsparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz));

    auto d_ind    = createEmptyArray<int>(nnz);
    auto d_values = createEmptyArray<T>(nnz);
    // reset offsets, column indices, and values pointers
    if (stype == AF_STORAGE_CSR) {
        CUSPARSE_CHECK(hipsparseCsrSetPointers(matB, d_offsets.get(),
                                               d_ind.get(), d_values.get()));

    } else {
        CUSPARSE_CHECK(hipsparseCscSetPointers(matB, d_offsets.get(),
                                               d_ind.get(), d_values.get()));
    }
    // execute Dense to Sparse conversion
    CUSPARSE_CHECK(hipsparseDenseToSparse_convert(
        sparseHandle(), matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer.get()));

    if (stype == AF_STORAGE_CSR) {
        size_t pBufferSizeInBytes = 0;
        auto desc                 = make_handle<SparseDescriptorRAII>();
        CUSPARSE_CHECK(hipsparseXcsrsort_bufferSizeExt(
            sparseHandle(), M, N, nnz, d_offsets.get(), d_ind.get(),
            &pBufferSizeInBytes));
        auto pBuffer = memAlloc<char>(pBufferSizeInBytes);
        Array<int> P = createEmptyArray<int>(nnz);
        CUSPARSE_CHECK(
            hipsparseCreateIdentityPermutation(sparseHandle(), nnz, P.get()));
        CUSPARSE_CHECK(hipsparseXcsrsort(
            sparseHandle(), M, N, nnz, desc, (int *)d_offsets.get(),
            (int *)d_ind.get(), P.get(), pBuffer.get()));
        d_values = lookup(d_values, P, 0);
        CUSPARSE_CHECK(hipsparseDestroySpMat(matB));
        return createArrayDataSparseArray<T>(in.dims(), d_values, d_offsets,
                                             d_ind, stype, false);
    } else {
        CUSPARSE_CHECK(hipsparseDestroySpMat(matB));
        return createArrayDataSparseArray<T>(in.dims(), d_values, d_ind,
                                             d_offsets, stype, false);
    }
}

// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in) {
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));

    const Array<T> values   = in.getValues();
    const Array<int> rowIdx = in.getRowIdx();
    const Array<int> colIdx = in.getColIdx();

    kernel::coo2dense<T>(dense, values, rowIdx, colIdx);

    return dense;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in) {
    SparseSpMatRAII inhandle = cusparseDescriptor(in);

    Array<T> dense            = createEmptyArray<T>(in.dims());
    SparseDnMatRAII outhandle = denMatDescriptor(dense);

    size_t bufferSize = 0;
    CUSPARSE_CHECK(hipsparseSparseToDense_bufferSize(
        sparseHandle(), inhandle, outhandle,
        HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));

    auto dBuffer = memAlloc<char>(bufferSize);
    CUSPARSE_CHECK(hipsparseSparseToDense(sparseHandle(), inhandle, outhandle,
                                          HIPSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer.get()));

    return dense;
}

template<typename T, af_storage dest, af_storage src>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in) {
    using std::shared_ptr;
    in.eval();

    int nNZ                  = in.getNNZ();
    SparseArray<T> converted = createEmptySparseArray<T>(in.dims(), nNZ, dest);

    if (src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
        // Copy colIdx as is
        CUDA_CHECK(hipMemcpyAsync(
            converted.getColIdx().get(), in.getColIdx().get(),
            in.getColIdx().elements() * sizeof(int), hipMemcpyDeviceToDevice,
            getActiveStream()));

        // cusparse function to expand compressed row into coordinate
        CUSPARSE_CHECK(hipsparseXcsr2coo(
            sparseHandle(), in.getRowIdx().get(), nNZ, in.dims()[0],
            converted.getRowIdx().get(), HIPSPARSE_INDEX_BASE_ZERO));

        // Call sort
        size_t pBufferSizeInBytes = 0;
        CUSPARSE_CHECK(hipsparseXcoosort_bufferSizeExt(
            sparseHandle(), in.dims()[0], in.dims()[1], nNZ,
            converted.getRowIdx().get(), converted.getColIdx().get(),
            &pBufferSizeInBytes));
        auto pBuffer = memAlloc<char>(pBufferSizeInBytes);

        Array<int> P = createEmptyArray<int>(nNZ);
        CUSPARSE_CHECK(
            hipsparseCreateIdentityPermutation(sparseHandle(), nNZ, P.get()));

        CUSPARSE_CHECK(hipsparseXcoosortByRow(
            sparseHandle(), in.dims()[0], in.dims()[1], nNZ,
            converted.getRowIdx().get(), converted.getColIdx().get(), P.get(),
            pBuffer.get()));

        converted.getValues() = lookup<T, int>(in.getValues(), P, 0);

    } else if (src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
        // The cusparse csr sort function is not behaving correctly.
        // So the work around is to convert the COO into row major and then
        // convert it to CSR

        int M = in.dims()[0];
        int N = in.dims()[1];
        // Deep copy input into temporary COO Row Major
        SparseArray<T> cooT = createArrayDataSparseArray<T>(
            in.dims(), in.getValues(), in.getRowIdx(), in.getColIdx(),
            in.getStorage(), true);

        // Call sort to convert column major to row major
        {
            size_t pBufferSizeInBytes = 0;
            CUSPARSE_CHECK(hipsparseXcoosort_bufferSizeExt(
                sparseHandle(), M, N, nNZ, cooT.getRowIdx().get(),
                cooT.getColIdx().get(), &pBufferSizeInBytes));
            auto pBuffer = memAlloc<char>(pBufferSizeInBytes);

            Array<int> P = createEmptyArray<int>(nNZ);
            CUSPARSE_CHECK(hipsparseCreateIdentityPermutation(sparseHandle(),
                                                              nNZ, P.get()));

            CUSPARSE_CHECK(hipsparseXcoosortByRow(
                sparseHandle(), M, N, nNZ, cooT.getRowIdx().get(),
                cooT.getColIdx().get(), P.get(), pBuffer.get()));

            converted.getValues() = lookup<T, int>(in.getValues(), P, 0);
        }

        // Copy values and colIdx as is
        copyArray<int, int>(converted.getColIdx(), cooT.getColIdx());

        // cusparse function to compress row from coordinate
        CUSPARSE_CHECK(hipsparseXcoo2csr(
            sparseHandle(), cooT.getRowIdx().get(), nNZ, M,
            converted.getRowIdx().get(), HIPSPARSE_INDEX_BASE_ZERO));

        // No need to call CSRSORT

    } else {
        // Should never come here
        AF_ERROR("CUDA Backend invalid conversion combination",
                 AF_ERR_NOT_SUPPORTED);
    }

    return converted;
}

#define INSTANTIATE_TO_STORAGE(T, S)                     \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>( \
        const SparseArray<T> &in);

#define INSTANTIATE_COO_SPECIAL(T)                                 \
    template<>                                                     \
    SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>( \
        const Array<T> &in) {                                      \
        return sparseConvertDenseToCOO<T>(in);                     \
    }                                                              \
    template<>                                                     \
    Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(       \
        const SparseArray<T> &in) {                                \
        return sparseConvertCOOToDense<T>(in);                     \
    }

#define INSTANTIATE_SPARSE(T)                                               \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>( \
        const Array<T> &in);                                                \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>( \
        const Array<T> &in);                                                \
                                                                            \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(       \
        const SparseArray<T> &in);                                          \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(       \
        const SparseArray<T> &in);                                          \
                                                                            \
    INSTANTIATE_COO_SPECIAL(T)                                              \
                                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}  // namespace cuda
}  // namespace arrayfire
