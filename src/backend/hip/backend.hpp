/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __DH__
#undef __DH__
#endif

#ifdef __CUDACC_RTC__
#define __DH__ __device__
#else
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define __DH__ __device__ __host__
#else
#define __DH__
#endif
#endif

namespace arrayfire {
namespace cuda {}  // namespace cuda
}  // namespace arrayfire

// The CUDA backend uses `namespace detail = arrayfire::cuda;` (a namespace
// ALIAS) as the backend-indirection that common/api headers reference as
// `detail::`. Under ROCm that alias is irreconcilable with rocThrust: rocprim
// (pulled by every hipcub/rocThrust TU) opens a GLOBAL real `namespace detail`,
// and a namespace alias cannot share a name with a real namespace at the same
// scope ("redefinition of 'detail' as different kind of symbol"). Reductions /
// sort / set pull BOTH rocprim and (via Array.hpp -> common/jit/Node.hpp) the
// detail-indirection, so suppressing the alias is not an option either.
// Fix: make `detail` a REAL namespace that re-exports arrayfire::cuda via a
// using-directive. Two real `namespace detail {}` definitions MERGE (rocprim's
// internal helpers and this using-directive coexist with no name clash), while
// `detail::getFullName<T>()` still resolves through the using-directive. CUDA
// keeps the plain alias (no rocprim), byte-for-byte unchanged.
#if defined(__HIP_PLATFORM_AMD__)
namespace detail {
using namespace arrayfire::cuda;
}
#else
namespace detail = arrayfire::cuda;
#endif
