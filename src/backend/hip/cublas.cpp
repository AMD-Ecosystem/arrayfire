/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cublas.hpp>

#include <common/err_common.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {
const char* errorString(hipblasStatus_t err) {
    switch (err) {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
        default: return "UNKNOWN";
    }
}

}  // namespace cuda
}  // namespace arrayfire
