/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <hip_unique_handle.hpp>
#include <hipsolver/hipsolver.h>

// hipsolverHandle_t is `void*` (collides with the other void* handles under the
// type-keyed common DEFINE_HANDLER); use the tag-keyed HIP handle.
DEFINE_HIP_HANDLE(SolveHandleRAII, hipsolverHandle_t, hipsolverDnCreate,
                  hipsolverDnDestroy);

namespace arrayfire {
namespace cuda {

const char* errorString(hipsolverStatus_t err);

#define CUSOLVER_CHECK(fn)                                                    \
    do {                                                                      \
        hipsolverStatus_t _error = fn;                                         \
        if (_error != HIPSOLVER_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSOLVER Error (%d): %s\n", \
                     (int)(_error), arrayfire::cuda::errorString(_error));    \
                                                                              \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace arrayfire
