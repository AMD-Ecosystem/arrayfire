/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/FFTPlanCache.hpp>
#include <common/err_common.hpp>
#include <common/unique_handle.hpp>
#include <hipfft/hipfft.h>
#include <cstdio>

DEFINE_HANDLER(hipfftHandle, hipfftCreate, hipfftDestroy);

namespace arrayfire {
namespace cuda {

typedef hipfftHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char *_cufftGetResultString(hipfftResult res);

SharedPlan findPlan(int rank, int *n, int *inembed, int istride, int idist,
                    int *onembed, int ostride, int odist, hipfftType type,
                    int batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    friend SharedPlan findPlan(int rank, int *n, int *inembed, int istride,
                               int idist, int *onembed, int ostride, int odist,
                               hipfftType type, int batch);
};

}  // namespace cuda
}  // namespace arrayfire

#define CUFFT_CHECK(fn)                                                   \
    do {                                                                  \
        hipfftResult _cufft_res = fn;                                      \
        if (_cufft_res != HIPFFT_SUCCESS) {                                \
            char cufft_res_msg[1024];                                     \
            snprintf(cufft_res_msg, sizeof(cufft_res_msg),                \
                     "cuFFT Error (%d): %s\n", (int)(_cufft_res),         \
                     arrayfire::cuda::_cufftGetResultString(_cufft_res)); \
                                                                          \
            AF_ERROR(cufft_res_msg, AF_ERR_INTERNAL);                     \
        }                                                                 \
    } while (0)
