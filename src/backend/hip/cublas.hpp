/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/defines.hpp>
#include <hip_unique_handle.hpp>
#include <hipblas/hipblas.h>

// hipblasHandle_t is `void*` (so are hipsolverHandle_t and the hipsparse
// handles/descriptors), which collides under the type-keyed common DEFINE_HANDLER
// the moment a TU pulls two of these headers. Use the tag-keyed HIP handle
// instead (see hip_unique_handle.hpp).
DEFINE_HIP_HANDLE(BlasHandleRAII, hipblasHandle_t, hipblasCreate,
                  hipblasDestroy);

namespace arrayfire {
namespace cuda {

const char* errorString(hipblasStatus_t err);

#define CUBLAS_CHECK(fn)                                                    \
    do {                                                                    \
        hipblasStatus_t _error = fn;                                         \
        if (_error != HIPBLAS_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                            \
            snprintf(_err_msg, sizeof(_err_msg), "CUBLAS Error (%d): %s\n", \
                     (int)(_error), arrayfire::cuda::errorString(_error));  \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                            \
        }                                                                   \
    } while (0)

}  // namespace cuda
}  // namespace arrayfire
