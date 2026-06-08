/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cusparse.hpp>
#include <platform.hpp>
#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {
const char* errorString(hipsparseStatus_t err) {
    switch (err) {
        case HIPSPARSE_STATUS_SUCCESS: return "HIPSPARSE_STATUS_SUCCESS";
        case HIPSPARSE_STATUS_NOT_INITIALIZED:
            return "HIPSPARSE_STATUS_NOT_INITIALIZED";
        case HIPSPARSE_STATUS_ALLOC_FAILED:
            return "HIPSPARSE_STATUS_ALLOC_FAILED";
        case HIPSPARSE_STATUS_INVALID_VALUE:
            return "HIPSPARSE_STATUS_INVALID_VALUE";
        case HIPSPARSE_STATUS_ARCH_MISMATCH:
            return "HIPSPARSE_STATUS_ARCH_MISMATCH";
        case HIPSPARSE_STATUS_MAPPING_ERROR:
            return "HIPSPARSE_STATUS_MAPPING_ERROR";
        case HIPSPARSE_STATUS_EXECUTION_FAILED:
            return "HIPSPARSE_STATUS_EXECUTION_FAILED";
        case HIPSPARSE_STATUS_INTERNAL_ERROR:
            return "HIPSPARSE_STATUS_INTERNAL_ERROR";
        case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case HIPSPARSE_STATUS_ZERO_PIVOT: return "HIPSPARSE_STATUS_ZERO_PIVOT";
        case HIPSPARSE_STATUS_NOT_SUPPORTED:
            return "HIPSPARSE_STATUS_NOT_SUPPORTED";
        case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default: return "UNKNOWN";
    }
}

}  // namespace cuda
}  // namespace arrayfire
