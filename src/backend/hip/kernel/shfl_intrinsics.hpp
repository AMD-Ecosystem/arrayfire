/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace arrayfire {
namespace cuda {
namespace kernel {

// AMD's __shfl*_sync / __ballot_sync / __all_sync / __any_sync static_assert that
// the mask is a 64-bit integer (the active wavefront is up to 64 lanes on CDNA),
// so the CUDA literal 0xffffffff fails to compile. Use a 64-bit all-lanes mask on
// HIP; the value is keyed on the platform, not the wave width, so it is correct
// on wave64 (gfx90a) and wave32 (gfx11xx) alike. HIP always provides the _sync
// intrinsics, so drop the legacy CUDA_VERSION<9000 fallbacks on this backend.
// CUDA keeps the 32-bit literal byte-for-byte (this file is only compiled on
// the AMD backend).
#if defined(__HIP_PLATFORM_AMD__)
constexpr unsigned long long FULL_MASK = 0xffffffffffffffffULL;
#else
constexpr unsigned int FULL_MASK = 0xffffffff;
#endif

//__all_sync wrapper
template<typename T>
__device__ T all_sync(T var) {
    return __all_sync(FULL_MASK, var);
}

//__any_sync wrapper
template<typename T>
__device__ T any_sync(T var) {
    return __any_sync(FULL_MASK, var);
}

//__ballot_sync wrapper
template<typename T>
__device__ auto ballot_sync(T var) {
    return __ballot_sync(FULL_MASK, var);
}

//__shfl_down_sync wrapper
template<typename T>
__device__ T shfl_down_sync(T var, int delta) {
    return __shfl_down_sync(FULL_MASK, var, delta);
}
// specialization for cfloat
template<>
inline __device__ cfloat shfl_down_sync(cfloat var, int delta) {
    cfloat res = {__shfl_down_sync(FULL_MASK, var.x, delta),
                  __shfl_down_sync(FULL_MASK, var.y, delta)};
    return res;
}
// specialization for cdouble
template<>
inline __device__ cdouble shfl_down_sync(cdouble var,
                                         int delta) {
    cdouble res = {__shfl_down_sync(FULL_MASK, var.x, delta),
                   __shfl_down_sync(FULL_MASK, var.y, delta)};
    return res;
}

//__shfl_up_sync wrapper
template<typename T>
__device__ T shfl_up_sync(T var, int delta) {
    return __shfl_up_sync(FULL_MASK, var, delta);
}
// specialization for cfloat
template<>
inline __device__ cfloat shfl_up_sync(cfloat var, int delta) {
    cfloat res = {__shfl_up_sync(FULL_MASK, var.x, delta),
                  __shfl_up_sync(FULL_MASK, var.y, delta)};
    return res;
}
// specialization for cdouble
template<>
inline __device__ cdouble shfl_up_sync(cdouble var, int delta) {
    cdouble res = {__shfl_up_sync(FULL_MASK, var.x, delta),
                   __shfl_up_sync(FULL_MASK, var.y, delta)};
    return res;
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
