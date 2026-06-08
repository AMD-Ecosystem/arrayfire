/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP shim for the include "cuComplex.h". ArrayFire's CUDA backend embeds the
// CUDA toolkit's cuComplex.h as a string blob fed to NVRTC, AND its host TUs pull
// it (via types.hpp); hipRTC / ROCm has no cuComplex.h, so this shim resolves it
// under that name in both paths.
//
// Two distinct shapes, selected by __CUDACC_RTC__:
//
//  * Host / compiled backend (not RTC): just alias cuFloatComplex etc. to HIP's
//    hipFloatComplex and the hipC* helpers. hip_compat.h (force-included on every
//    compiled TU) already #defines the same names, so this branch mostly just
//    pulls hip_complex.h; the typedefs are harmless no-ops there.
//
//  * Runtime JIT (hipRTC, __CUDACC_RTC__ defined, no hip_compat.h): cuFloatComplex
//    / cuDoubleComplex are plain PODs (NOT hipFloatComplex = HIP_vector_type).
//    HIP_vector_type ships componentwise FRIEND operator* / operator/ with no
//    opt-out; a templated kernel doing a bare `a * b` on a complex T (convolve,
//    the cubic interp in approx/resize, fftconvolve, ...) would then silently get
//    a COMPONENTWISE product instead of the complex product (wrong results) or tie
//    with arrayfire's complex operator*. A POD has no operators, so the only
//    complex operator* in scope is arrayfire's (math.hpp BINOP, forwarding to the
//    cuCmulf below). Layout (two floats/doubles) is identical to hipFloatComplex /
//    float2, so buffer reinterpretation at the kernel boundary is unchanged. This
//    is the JIT analogue of the host backend's POD cfloat/cdouble in types.hpp.

#pragma once

#include <hip/hip_complex.h>

#ifndef __CUDACC_RTC__

typedef hipFloatComplex  cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;
typedef hipComplex       cuComplex;

#define make_cuComplex          make_hipFloatComplex
#define make_cuFloatComplex     make_hipFloatComplex
#define make_cuDoubleComplex    make_hipDoubleComplex
#define cuCaddf                 hipCaddf
#define cuCadd                  hipCadd
#define cuCsubf                 hipCsubf
#define cuCsub                  hipCsub
#define cuCmulf                 hipCmulf
#define cuCmul                  hipCmul
#define cuCdivf                 hipCdivf
#define cuCdiv                  hipCdiv
#define cuCabsf                 hipCabsf
#define cuCabs                  hipCabs
#define cuConjf                 hipConjf
#define cuConj                  hipConj
#define cuCrealf                hipCrealf
#define cuCreal                 hipCreal
#define cuCimagf                hipCimagf
#define cuCimag                 hipCimag
#define cuComplexFloatToDouble  hipComplexFloatToDouble
#define cuComplexDoubleToFloat  hipComplexDoubleToFloat

#else  // __CUDACC_RTC__ : runtime-JIT POD complex (see header note above)

struct alignas(2 * sizeof(float)) cuFloatComplex {
    float x, y;
};
struct alignas(2 * sizeof(double)) cuDoubleComplex {
    double x, y;
};
typedef cuFloatComplex cuComplex;

__device__ static inline cuFloatComplex make_cuComplex(float r, float i) {
    return cuFloatComplex{r, i};
}
__device__ static inline cuFloatComplex make_cuFloatComplex(float r, float i) {
    return cuFloatComplex{r, i};
}
__device__ static inline cuDoubleComplex make_cuDoubleComplex(double r,
                                                              double i) {
    return cuDoubleComplex{r, i};
}
__device__ static inline float cuCrealf(cuFloatComplex c) { return c.x; }
__device__ static inline float cuCimagf(cuFloatComplex c) { return c.y; }
__device__ static inline double cuCreal(cuDoubleComplex c) { return c.x; }
__device__ static inline double cuCimag(cuDoubleComplex c) { return c.y; }
__device__ static inline cuFloatComplex cuConjf(cuFloatComplex c) {
    return cuFloatComplex{c.x, -c.y};
}
__device__ static inline cuDoubleComplex cuConj(cuDoubleComplex c) {
    return cuDoubleComplex{c.x, -c.y};
}
__device__ static inline cuFloatComplex cuCaddf(cuFloatComplex a,
                                                cuFloatComplex b) {
    return cuFloatComplex{a.x + b.x, a.y + b.y};
}
__device__ static inline cuDoubleComplex cuCadd(cuDoubleComplex a,
                                                cuDoubleComplex b) {
    return cuDoubleComplex{a.x + b.x, a.y + b.y};
}
__device__ static inline cuFloatComplex cuCsubf(cuFloatComplex a,
                                                cuFloatComplex b) {
    return cuFloatComplex{a.x - b.x, a.y - b.y};
}
__device__ static inline cuDoubleComplex cuCsub(cuDoubleComplex a,
                                                cuDoubleComplex b) {
    return cuDoubleComplex{a.x - b.x, a.y - b.y};
}
__device__ static inline cuFloatComplex cuCmulf(cuFloatComplex a,
                                                cuFloatComplex b) {
    return cuFloatComplex{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ static inline cuDoubleComplex cuCmul(cuDoubleComplex a,
                                                cuDoubleComplex b) {
    return cuDoubleComplex{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ static inline cuFloatComplex cuCdivf(cuFloatComplex a,
                                                cuFloatComplex b) {
    float d = b.x * b.x + b.y * b.y;
    return cuFloatComplex{(a.x * b.x + a.y * b.y) / d,
                          (a.y * b.x - a.x * b.y) / d};
}
__device__ static inline cuDoubleComplex cuCdiv(cuDoubleComplex a,
                                                cuDoubleComplex b) {
    double d = b.x * b.x + b.y * b.y;
    return cuDoubleComplex{(a.x * b.x + a.y * b.y) / d,
                           (a.y * b.x - a.x * b.y) / d};
}
__device__ static inline float cuCabsf(cuFloatComplex c) {
    return hypotf(c.x, c.y);
}
__device__ static inline double cuCabs(cuDoubleComplex c) {
    return hypot(c.x, c.y);
}
__device__ static inline cuDoubleComplex cuComplexFloatToDouble(
    cuFloatComplex c) {
    return cuDoubleComplex{(double)c.x, (double)c.y};
}
__device__ static inline cuFloatComplex cuComplexDoubleToFloat(
    cuDoubleComplex c) {
    return cuFloatComplex{(float)c.x, (float)c.y};
}

// The POD cuFloatComplex / cuDoubleComplex live in the GLOBAL namespace, so the
// complex == / != must too: arrayfire's other equality operators sit in
// namespace arrayfire::cuda (math.hpp), which is unreachable by ADL from a
// global-namespace POD -- a JIT kernel comparing a complex value from another
// namespace (e.g. common::Transform<cuFloatComplex,uint,af_notzero_t>, which
// `where` over a complex array instantiates) then fails overload resolution
// ("invalid operands to binary expression"). Defining them here, beside the
// type, makes ADL find them from any namespace. math.hpp drops its
// arrayfire::cuda complex ==/!= on the RTC path so these are unambiguous.
__device__ static inline bool operator==(cuFloatComplex a, cuFloatComplex b) {
    return (a.x == b.x) && (a.y == b.y);
}
__device__ static inline bool operator!=(cuFloatComplex a, cuFloatComplex b) {
    return !(a == b);
}
__device__ static inline bool operator==(cuDoubleComplex a, cuDoubleComplex b) {
    return (a.x == b.x) && (a.y == b.y);
}
__device__ static inline bool operator!=(cuDoubleComplex a, cuDoubleComplex b) {
    return !(a == b);
}

#endif  // __CUDACC_RTC__
