/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/traits.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <types.hpp>

namespace af {

template<>
struct dtype_traits<cuFloatComplex> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char* getName() { return "cuFloatComplex"; }
};

template<>
struct dtype_traits<cuDoubleComplex> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char* getName() { return "cuDoubleComplex"; }
};

#if defined(__HIP_PLATFORM_AMD__) && !defined(__CUDACC_RTC__)
// On HIP the backend's cfloat/cdouble are distinct PODs (not cuFloatComplex =
// hipFloatComplex; see types.hpp), so they need their own dtype_traits. getName
// drives the runtime-JIT template instantiation name and MUST stay
// "cuFloatComplex" -- that is the type the JIT device source uses everywhere
// (the shim typedef is always in scope, unlike the project cfloat alias which is
// only visible where types.hpp is pulled).
template<>
struct dtype_traits<arrayfire::cuda::cfloat> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char* getName() { return "cuFloatComplex"; }
};

template<>
struct dtype_traits<arrayfire::cuda::cdouble> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char* getName() { return "cuDoubleComplex"; }
};
#endif

template<>
struct dtype_traits<__half> {
    enum { af_type = f16 };
    typedef __half base_type;
    static const char* getName() { return "__half"; }
};

}  // namespace af

using af::dtype_traits;
