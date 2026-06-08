/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for <cusparse_v2.h>. The sparse backend (sparse.cu /
// sparse_arith.cu / sparse_blas.cu / cusparse.hpp / cusparse_descriptor_helpers.hpp)
// keeps the cuSPARSE spelling and links roc::hipsparse directly (no dlopen
// plugin, unlike the NVIDIA build's cusparseModule). This header aliases the
// cuSPARSE names those files use to their hipSPARSE equivalents, so the source
// stays in CUDA spelling (the colmap minimal-footprint recipe).
//
// hipSPARSE mirrors cuSPARSE name-for-name (only the cusparse/CUSPARSE_ ->
// hipsparse/HIPSPARSE_ prefix differs), so the mapping is mechanical. This shim
// lives in nvrtc_shims/ which is on the HIP include path ONLY, so a real NVIDIA
// build of src/backend/cuda still resolves the CUDA toolkit's cusparse_v2.h.
//
// CUSPARSE_VERSION is forced high so the backend's version #if branches select
// the generic-API + csrgeam2 paths (the ones hipSPARSE 4.2 implements), matching
// AF_USE_NEW_CUSPARSE_API. The legacy named-function paths (cusparseZ* dense2csr
// / csrgeam / csrgeamNnz under CUSPARSE_VERSION < 11000/11300) are compiled out.

#pragma once

#include <hipsparse/hipsparse.h>

// Pick the generic-API + csrgeam2 code paths (>= 11400 also selects the
// SPMV_CSR_ALG1 / SPMM_CSR_ALG1 enum names hipSPARSE exposes).
#if !defined(CUSPARSE_VERSION)
#define CUSPARSE_VERSION 11400
#endif

// ---- handles / opaque descriptors -----------------------------------------
#define cusparseHandle_t       hipsparseHandle_t
#define cusparseMatDescr_t     hipsparseMatDescr_t
#define cusparseSpMatDescr_t   hipsparseSpMatDescr_t
#define cusparseDnVecDescr_t   hipsparseDnVecDescr_t
#define cusparseDnMatDescr_t   hipsparseDnMatDescr_t

// ---- enums / scalar types --------------------------------------------------
#define cusparseStatus_t       hipsparseStatus_t
#define cusparseOperation_t    hipsparseOperation_t
#define cusparseDirection_t    hipsparseDirection_t
#define cusparseIndexBase_t    hipsparseIndexBase_t
#define cusparseSpMVAlg_t      hipsparseSpMVAlg_t
#define cusparseSpMMAlg_t      hipsparseSpMMAlg_t
#define cusparseOrder_t        hipsparseOrder_t

#define CUSPARSE_STATUS_SUCCESS                   HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_NOT_INITIALIZED           HIPSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_ALLOC_FAILED              HIPSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_INVALID_VALUE             HIPSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_ARCH_MISMATCH             HIPSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_MAPPING_ERROR             HIPSPARSE_STATUS_MAPPING_ERROR
#define CUSPARSE_STATUS_EXECUTION_FAILED          HIPSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INTERNAL_ERROR            HIPSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_ZERO_PIVOT                HIPSPARSE_STATUS_ZERO_PIVOT

#define CUSPARSE_OPERATION_NON_TRANSPOSE       HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE           HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE

#define CUSPARSE_DIRECTION_ROW    HIPSPARSE_DIRECTION_ROW
#define CUSPARSE_DIRECTION_COLUMN HIPSPARSE_DIRECTION_COLUMN

#define CUSPARSE_INDEX_32I     HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_MATRIX_TYPE_GENERAL HIPSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_ORDER_COL     HIPSPARSE_ORDER_COL

#define CUSPARSE_DENSETOSPARSE_ALG_DEFAULT HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define CUSPARSE_SPARSETODENSE_ALG_DEFAULT HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
#define CUSPARSE_SPMV_CSR_ALG1   HIPSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_ALG_DEFAULT HIPSPARSE_SPMV_ALG_DEFAULT
#define CUSPARSE_SPMM_CSR_ALG1   HIPSPARSE_SPMM_CSR_ALG1

// ---- functions: lifetimes / descriptors -----------------------------------
#define cusparseCreate         hipsparseCreate
#define cusparseDestroy        hipsparseDestroy
#define cusparseCreateMatDescr  hipsparseCreateMatDescr
#define cusparseDestroyMatDescr hipsparseDestroyMatDescr
#define cusparseSetMatType      hipsparseSetMatType
#define cusparseSetMatIndexBase hipsparseSetMatIndexBase

#define cusparseCreateCsr      hipsparseCreateCsr
#define cusparseCreateCsc      hipsparseCreateCsc
#define cusparseCreateCoo      hipsparseCreateCoo
#define cusparseCreateDnVec    hipsparseCreateDnVec
#define cusparseCreateDnMat    hipsparseCreateDnMat
#define cusparseDestroySpMat   hipsparseDestroySpMat
#define cusparseDestroyDnVec   hipsparseDestroyDnVec
#define cusparseDestroyDnMat   hipsparseDestroyDnMat
#define cusparseCsrSetPointers hipsparseCsrSetPointers
#define cusparseCscSetPointers hipsparseCscSetPointers
#define cusparseSpMatGetSize   hipsparseSpMatGetSize

// ---- functions: generic SpMV / SpMM ---------------------------------------
#define cusparseSpMV           hipsparseSpMV
#define cusparseSpMV_bufferSize hipsparseSpMV_bufferSize
#define cusparseSpMM           hipsparseSpMM
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize

// ---- functions: dense<->sparse conversion (generic) -----------------------
#define cusparseDenseToSparse_bufferSize hipsparseDenseToSparse_bufferSize
#define cusparseDenseToSparse_analysis   hipsparseDenseToSparse_analysis
#define cusparseDenseToSparse_convert    hipsparseDenseToSparse_convert
#define cusparseSparseToDense            hipsparseSparseToDense
#define cusparseSparseToDense_bufferSize hipsparseSparseToDense_bufferSize

// ---- functions: legacy sort / coordinate conversion -----------------------
#define cusparseCreateIdentityPermutation hipsparseCreateIdentityPermutation
#define cusparseXcsrsort_bufferSizeExt    hipsparseXcsrsort_bufferSizeExt
#define cusparseXcsrsort                  hipsparseXcsrsort
#define cusparseXcoosort_bufferSizeExt    hipsparseXcoosort_bufferSizeExt
#define cusparseXcoosortByRow             hipsparseXcoosortByRow
#define cusparseXcsr2coo                  hipsparseXcsr2coo
#define cusparseXcoo2csr                  hipsparseXcoo2csr

// ---- functions: csrgeam2 (typed, sparse+sparse add/sub) --------------------
#define cusparseScsrgeam2                 hipsparseScsrgeam2
#define cusparseDcsrgeam2                 hipsparseDcsrgeam2
#define cusparseCcsrgeam2                 hipsparseCcsrgeam2
#define cusparseZcsrgeam2                 hipsparseZcsrgeam2
#define cusparseScsrgeam2_bufferSizeExt   hipsparseScsrgeam2_bufferSizeExt
#define cusparseDcsrgeam2_bufferSizeExt   hipsparseDcsrgeam2_bufferSizeExt
#define cusparseCcsrgeam2_bufferSizeExt   hipsparseCcsrgeam2_bufferSizeExt
#define cusparseZcsrgeam2_bufferSizeExt   hipsparseZcsrgeam2_bufferSizeExt
#define cusparseXcsrgeam2Nnz              hipsparseXcsrgeam2Nnz

#define cusparseSetStream      hipsparseSetStream
