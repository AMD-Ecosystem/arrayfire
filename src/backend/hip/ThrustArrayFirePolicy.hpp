/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <thrust/memory.h>
#include <thrust/system/hip/execution_policy.h>

namespace arrayfire {
namespace cuda {
struct ThrustArrayFirePolicy
    : thrust::hip::execution_policy<ThrustArrayFirePolicy> {};

template<typename T>
thrust::pair<thrust::pointer<T, ThrustArrayFirePolicy>, std::ptrdiff_t>
get_temporary_buffer(ThrustArrayFirePolicy, std::ptrdiff_t n) {
    thrust::pointer<T, ThrustArrayFirePolicy> result(
        arrayfire::cuda::memAlloc<T>(n / sizeof(T)).release());

    return thrust::make_pair(result, n);
}

template<typename Pointer>
inline void return_temporary_buffer(ThrustArrayFirePolicy, Pointer p) {
    memFree(thrust::raw_pointer_cast(p));
}

}  // namespace cuda
}  // namespace arrayfire

// rocThrust puts the stream-policy hooks in thrust::hip_rocprim (the analogue of
// CUDA Thrust's thrust::cuda_cub), and the device-vs-host guard is
// __HIP_DEVICE_COMPILE__ (HIP does not define __CUDA_ARCH__ in the device pass).
// cudaStream_t / cudaError_t / cudaSuccess / cudaStreamSynchronize resolve to
// the hip* spellings via the force-included hip_compat.h.
THRUST_NAMESPACE_BEGIN
namespace hip_rocprim {
template<>
__DH__ inline cudaStream_t get_stream<arrayfire::cuda::ThrustArrayFirePolicy>(
    execution_policy<arrayfire::cuda::ThrustArrayFirePolicy> &) {
#if defined(__HIP_DEVICE_COMPILE__)
    return 0;
#else
    return arrayfire::cuda::getActiveStream();
#endif
}

__DH__
inline cudaError_t synchronize_stream(
    const arrayfire::cuda::ThrustArrayFirePolicy &) {
#if defined(__HIP_DEVICE_COMPILE__)
    return cudaSuccess;
#else
    return cudaStreamSynchronize(arrayfire::cuda::getActiveStream());
#endif
}

}  // namespace hip_rocprim
THRUST_NAMESPACE_END
