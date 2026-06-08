/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cusolverDn.hpp>
#include <debug_cuda.hpp>
#include <platform.hpp>
#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {
const char *errorString(hipsolverStatus_t err) {
    switch (err) {
        case HIPSOLVER_STATUS_SUCCESS: return "HIPSOLVER_STATUS_SUCCESS";
        case HIPSOLVER_STATUS_NOT_INITIALIZED:
            return "HIPSOLVER_STATUS_NOT_INITIALIZED";
        case HIPSOLVER_STATUS_ALLOC_FAILED:
            return "HIPSOLVER_STATUS_ALLOC_FAILED";
        case HIPSOLVER_STATUS_INVALID_VALUE:
            return "HIPSOLVER_STATUS_INVALID_VALUE";
        case HIPSOLVER_STATUS_ARCH_MISMATCH:
            return "HIPSOLVER_STATUS_ARCH_MISMATCH";
        case HIPSOLVER_STATUS_MAPPING_ERROR:
            return "HIPSOLVER_STATUS_MAPPING_ERROR";
        case HIPSOLVER_STATUS_EXECUTION_FAILED:
            return "HIPSOLVER_STATUS_EXECUTION_FAILED";
        case HIPSOLVER_STATUS_INTERNAL_ERROR:
            return "HIPSOLVER_STATUS_INTERNAL_ERROR";
        case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case HIPSOLVER_STATUS_NOT_SUPPORTED:
            return "HIPSOLVER_STATUS_NOT_SUPPORTED";
        case HIPSOLVER_STATUS_ZERO_PIVOT: return "HIPSOLVER_STATUS_ZERO_PIVOT";
        // hipSOLVER has no INVALID_LICENSE status (cuSOLVER orphan enum); it
        // does carry HANDLE_IS_NULLPTR / INVALID_ENUM / UNKNOWN that cuSOLVER
        // lacks. Map the ones that exist; the rest fall through to UNKNOWN.
        case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
            return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
        case HIPSOLVER_STATUS_INVALID_ENUM:
            return "HIPSOLVER_STATUS_INVALID_ENUM";
        default: return "UNKNOWN";
    }
}
}  // namespace cuda
}  // namespace arrayfire
