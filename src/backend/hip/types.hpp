/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/kernel_type.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>

namespace arrayfire {
namespace common {
class half;
}  // namespace common
}  // namespace arrayfire

#ifdef __CUDACC_RTC__

using dim_t = long long;

#else  //__CUDACC_RTC__

#include <af/traits.hpp>

#endif  //__CUDACC_RTC__

namespace arrayfire {
namespace cuda {

// HIP's <hip/hip_fp16.h> does `using half = __half;` at GLOBAL scope, so bare
// `half` (which the backend uses to mean arrayfire::common::half, as on CUDA)
// becomes ambiguous with ::half in every TU that pulls hip_fp16.h. Re-declare
// the name inside `namespace cuda` so the nearer-scope alias wins; this matches
// the CUDA backend's meaning of bare `half`. CUDA's fp16 header does not inject
// a global `half`, so this alias is HIP-only.
#if defined(__HIP_PLATFORM_AMD__) && !defined(__CUDACC_RTC__)
using half = arrayfire::common::half;

// cfloat/cdouble must NOT be hipFloatComplex/hipDoubleComplex (=
// HIP_vector_type<float|double,2>): that type ships FRIEND operator* and
// operator/ (componentwise) with no opt-out, which TIE with arrayfire's
// complex operator* / operator/ (math.hpp) at every cfloat*cfloat site
// ("ambiguous"). cuComplex on NVIDIA is a plain struct with no operators, so
// only arrayfire's apply there. Define them as plain PODs (8/16-byte, ABI- and
// layout-compatible with hipFloatComplex/hipDoubleComplex and float2/double2),
// with implicit conversions so the cuC* helpers and the hipBLAS/hipFFT/cuComplex
// boundaries (which take hipFloatComplex) keep working; only arrayfire's complex
// operators then apply. The runtime-JIT source strings keep cuFloatComplex (a
// separate device type compiled by hipRTC), so this is host-backend-only.
struct alignas(2 * sizeof(float)) cfloat {
    float x, y;
    __host__ __device__ cfloat() = default;
    __host__ __device__ constexpr cfloat(float a, float b = 0.f) : x(a), y(b) {}
    __host__ __device__ cfloat(hipFloatComplex c) : x(c.x), y(c.y) {}
    __host__ __device__ operator hipFloatComplex() const {
        return make_hipFloatComplex(x, y);
    }
};
struct alignas(2 * sizeof(double)) cdouble {
    double x, y;
    __host__ __device__ cdouble() = default;
    __host__ __device__ constexpr cdouble(double a, double b = 0.0)
        : x(a), y(b) {}
    __host__ __device__ cdouble(hipDoubleComplex c) : x(c.x), y(c.y) {}
    __host__ __device__ operator hipDoubleComplex() const {
        return make_hipDoubleComplex(x, y);
    }
};
#else
using cdouble = cuDoubleComplex;
using cfloat  = cuFloatComplex;
#endif
using intl    = long long;
using schar   = signed char;
using uchar   = unsigned char;
using uint    = unsigned int;
using uintl   = unsigned long long;
using ushort  = unsigned short;
using ulong   = unsigned long long;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

#ifndef __CUDACC_RTC__
namespace {
template<typename T>
inline const char *shortname(bool caps = false) {
    return caps ? "Q" : "q";
}
template<>
inline const char *shortname<float>(bool caps) {
    return caps ? "S" : "s";
}
template<>
inline const char *shortname<double>(bool caps) {
    return caps ? "D" : "d";
}
template<>
inline const char *shortname<cfloat>(bool caps) {
    return caps ? "C" : "c";
}
template<>
inline const char *shortname<cdouble>(bool caps) {
    return caps ? "Z" : "z";
}
template<>
inline const char *shortname<int>(bool caps) {
    return caps ? "I" : "i";
}
template<>
inline const char *shortname<uint>(bool caps) {
    return caps ? "U" : "u";
}
template<>
inline const char *shortname<char>(bool caps) {
    return caps ? "J" : "j";
}
template<>
inline const char *shortname<schar>(bool caps) {
    return caps ? "A" : "a"; // TODO
}
template<>
inline const char *shortname<uchar>(bool caps) {
    return caps ? "V" : "v";
}
template<>
inline const char *shortname<intl>(bool caps) {
    return caps ? "X" : "x";
}
template<>
inline const char *shortname<uintl>(bool caps) {
    return caps ? "Y" : "y";
}
template<>
inline const char *shortname<short>(bool caps) {
    return caps ? "P" : "p";
}
template<>
inline const char *shortname<ushort>(bool caps) {
    return caps ? "Q" : "q";
}
template<>
inline const char *shortname<arrayfire::common::half>(bool caps) {
    return caps ? "H" : "h";
}

template<typename T>
inline const char *getFullName();

#define SPECIALIZE(T)                     \
    template<>                            \
    inline const char *getFullName<T>() { \
        return #T;                        \
    }

SPECIALIZE(float)
SPECIALIZE(double)
SPECIALIZE(cfloat)
SPECIALIZE(cdouble)
SPECIALIZE(char)
SPECIALIZE(signed char)
SPECIALIZE(unsigned char)
SPECIALIZE(short)
SPECIALIZE(unsigned short)
SPECIALIZE(int)
SPECIALIZE(unsigned int)
SPECIALIZE(unsigned long long)
SPECIALIZE(long long)

template<>
inline const char *getFullName<common::half>() {
    return "half";
}
#undef SPECIALIZE
}  // namespace
#endif  //__CUDACC_RTC__

}  // namespace cuda

namespace common {

template<typename T>
struct kernel_type;

template<>
struct kernel_type<arrayfire::common::half> {
    using data = arrayfire::common::half;

#ifdef __CUDA_ARCH__

    // These are the types within a kernel
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    using compute = __half;
#else
    using compute = float;
#endif
    using native = compute;

#else  // __CUDA_ARCH__

    // outside of a cuda kernel use float
    using compute = float;

#if defined(__NVCC__) || defined(__CUDACC_RTC__) || \
    defined(__HIP_PLATFORM_AMD__)
    // On HIP the native GPU half type is __half (hipBLAS's hipblasHalf is
    // bit-compatible and the gemm function pointers reinterpret to it); the
    // blas.cu gemm_func is instantiated for __half, so native must be __half
    // here too, not common::half.
    using native  = __half;
#else
    using native = common::half;
#endif

#endif  // __CUDA_ARCH__
};
}  // namespace common
}  // namespace arrayfire
