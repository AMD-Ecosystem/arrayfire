/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/half.hpp>
#include <hip/library_types.h>  // hipDataType enum
#include <types.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
inline hipDataType getType();

template<>
inline hipDataType getType<float>() {
    return HIP_R_32F;
}

template<>
inline hipDataType getType<cfloat>() {
    return HIP_C_32F;
}

template<>
inline hipDataType getType<double>() {
    return HIP_R_64F;
}

template<>
inline hipDataType getType<cdouble>() {
    return HIP_C_64F;
}

template<>
inline hipDataType getType<common::half>() {
    return HIP_R_16F;
}

template<>
inline hipDataType getType<uchar>() {
    return HIP_R_8I;
}

template<>
inline hipDataType getType<schar>() {
    return HIP_R_8I;
}

/* only supports LStride/RStride % 4 == 0 */
template<>
inline hipDataType getType<int>() {
    return HIP_R_32I;
}

// hipblasGemmEx/GemmBatchedEx take a hipblasComputeType_t (NOT a hipDataType as
// cuBLAS's cudaDataType-based cublasGemmEx historically did). Map the backend's
// accumulate type to the matching compute enum. cfloat/cdouble accumulate in
// 32F/64F; the integer (schar) gemm accumulates into float, so 32F.
template<typename T>
inline hipblasComputeType_t getComputeType() {
    return HIPBLAS_COMPUTE_32F;
}
template<>
inline hipblasComputeType_t getComputeType<double>() {
    return HIPBLAS_COMPUTE_64F;
}
template<>
inline hipblasComputeType_t getComputeType<cdouble>() {
    return HIPBLAS_COMPUTE_64F;
}
template<>
inline hipblasComputeType_t getComputeType<common::half>() {
    return HIPBLAS_COMPUTE_32F;
}

}  // namespace cuda
}  // namespace arrayfire
