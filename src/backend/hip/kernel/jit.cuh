/*******************************************************
 * Copyright (c) 2025, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef float2 cuFloatComplex;
typedef cuFloatComplex cfloat;

typedef double2 cuDoubleComplex;
typedef cuDoubleComplex cdouble;

#include <cuda_fp16.h>

// Half-precision transcendental overloads for the JIT element-wise kernels.
// The JIT emits the bare math name (sin/cos/.../fabs) for every type; on CUDA
// cuda_fp16.h supplies __half overloads so sin(__half) resolves, but HIP has no
// such overloads and __half converts to BOTH float and double, so an unqualified
// sin(__half) is ambiguous ("call to 'sin' is ambiguous"). Provide exact __half
// overloads here (global scope, where the generated kernel body resolves names):
// HIP native h* intrinsic where one exists, otherwise promote through float.
#if defined(__HIP_DEVICE_COMPILE__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
#define AF_HALF_NATIVE(NAME, HFN) \
    __device__ __inline__ __half NAME(__half x) { return HFN(x); }
#define AF_HALF_VIA_FLOAT(NAME, FFN) \
    __device__ __inline__ __half NAME(__half x) { return __float2half(FFN(__half2float(x))); }
AF_HALF_NATIVE(sin, hsin)
AF_HALF_NATIVE(cos, hcos)
AF_HALF_NATIVE(exp, hexp)
AF_HALF_NATIVE(log, hlog)
AF_HALF_NATIVE(log2, hlog2)
AF_HALF_NATIVE(log10, hlog10)
AF_HALF_NATIVE(sqrt, hsqrt)
AF_HALF_NATIVE(rsqrt, hrsqrt)
AF_HALF_NATIVE(ceil, hceil)
AF_HALF_NATIVE(floor, hfloor)
AF_HALF_NATIVE(trunc, htrunc)
AF_HALF_VIA_FLOAT(round, roundf)
AF_HALF_VIA_FLOAT(tan, tanf)
AF_HALF_VIA_FLOAT(asin, asinf)
AF_HALF_VIA_FLOAT(acos, acosf)
AF_HALF_VIA_FLOAT(atan, atanf)
AF_HALF_VIA_FLOAT(sinh, sinhf)
AF_HALF_VIA_FLOAT(cosh, coshf)
AF_HALF_VIA_FLOAT(tanh, tanhf)
AF_HALF_VIA_FLOAT(asinh, asinhf)
AF_HALF_VIA_FLOAT(acosh, acoshf)
AF_HALF_VIA_FLOAT(atanh, atanhf)
AF_HALF_VIA_FLOAT(cbrt, cbrtf)
AF_HALF_VIA_FLOAT(erf, erff)
AF_HALF_VIA_FLOAT(erfc, erfcf)
AF_HALF_VIA_FLOAT(expm1, expm1f)
AF_HALF_VIA_FLOAT(log1p, log1pf)
AF_HALF_VIA_FLOAT(tgamma, tgammaf)
AF_HALF_VIA_FLOAT(lgamma, lgammaf)
__device__ __inline__ __half fabs(__half x) { return __habs(x); }
__device__ __inline__ __half abs(__half x) { return __habs(x); }
// Binary half transcendentals (atan2/hypot) the JIT emits as bare names: same
// float<->double conversion ambiguity, promote through float.
__device__ __inline__ __half atan2(__half y, __half x) {
    return __float2half(atan2f(__half2float(y), __half2float(x)));
}
__device__ __inline__ __half hypot(__half y, __half x) {
    return __float2half(hypotf(__half2float(y), __half2float(x)));
}
#undef AF_HALF_NATIVE
#undef AF_HALF_VIA_FLOAT
#endif

// ----------------------------------------------
// COMMON OPERATIONS
// ----------------------------------------------

#define __select(cond, a, b) (cond) ? (a) : (b)
#define __not_select(cond, a, b) (cond) ? (b) : (a)
#define __circular_mod(a, b) ((a) < (b)) ? (a) : (a - b)

// ----------------------------------------------
// REAL NUMBER OPERATIONS
// ----------------------------------------------
#define __noop(a) (a)
#define __add(lhs, rhs) (lhs) + (rhs)
#define __sub(lhs, rhs) (lhs) - (rhs)
#define __mul(lhs, rhs) (lhs) * (rhs)
#define __div(lhs, rhs) (lhs) / (rhs)
#define __and(lhs, rhs) (lhs) && (rhs)
#define __or(lhs, rhs) (lhs) || (rhs)

#define __lt(lhs, rhs) (lhs) < (rhs)
#define __gt(lhs, rhs) (lhs) > (rhs)
#define __le(lhs, rhs) (lhs) <= (rhs)
#define __ge(lhs, rhs) (lhs) >= (rhs)
#define __eq(lhs, rhs) (lhs) == (rhs)
#define __neq(lhs, rhs) (lhs) != (rhs)

#define __conj(in) (in)
#define __real(in) (in)
#define __imag(in) (0)
#define __abs(in) abs(in)
// Cast the exp result to double so the `1 + ...` is unambiguous for __half (a
// bare `1 + __half` ties int->float vs int->double on HIP); float/double paths
// already computed the sigmoid in double, so this is semantically unchanged.
#define __sigmoid(in) (1.0 / (1.0 + (double)(exp(-(in)))))

#define __bitnot(in) (~(in))
#define __bitor(lhs, rhs) ((lhs) | (rhs))
#define __bitand(lhs, rhs) ((lhs) & (rhs))
#define __bitxor(lhs, rhs) ((lhs) ^ (rhs))
#define __bitshiftl(lhs, rhs) ((lhs) << (rhs))
#define __bitshiftr(lhs, rhs) ((lhs) >> (rhs))

#define __min(lhs, rhs) ((lhs) < (rhs)) ? (lhs) : (rhs)
#define __max(lhs, rhs) ((lhs) > (rhs)) ? (lhs) : (rhs)
#define __rem(lhs, rhs) ((lhs) % (rhs))
#define __mod(lhs, rhs) ((lhs) % (rhs))

#define __pow(lhs, rhs)  \
    static_cast<double>( \
        pow(static_cast<double>(lhs), static_cast<double>(rhs)));
#define __powll(lhs, rhs) \
    __double2ll_rn(pow(__ll2double_rn(lhs), __ll2double_rn(rhs)))
#define __powul(lhs, rhs) \
    __double2ull_rn(pow(__ull2double_rn(lhs), __ull2double_rn(rhs)))
#define __powui(lhs, rhs) \
    __double2uint_rn(pow(__uint2double_rn(lhs), __uint2double_rn(rhs)))
#define __powsi(lhs, rhs) \
    __double2int_rn(pow(__int2double_rn(lhs), __int2double_rn(rhs)))

#define __convert_char(val) (char)((val) != 0)
#define frem(lhs, rhs) remainder((lhs), (rhs))
#define fremf(lhs, rhs) remainderf((lhs), (rhs))

// ----------------------------------------------
// COMPLEX FLOAT OPERATIONS
// ----------------------------------------------

#define __crealf(in) ((in).x)
#define __cimagf(in) ((in).y)
#define __cabsf(in) hypotf(in.x, in.y)

__device__ cfloat __cplx2f(float x, float y) {
    cfloat res = {x, y};
    return res;
}

__device__ cfloat __cconjf(cfloat in) {
    cfloat res = {in.x, -in.y};
    return res;
}

__device__ cfloat __caddf(cfloat lhs, cfloat rhs) {
    cfloat res = {lhs.x + rhs.x, lhs.y + rhs.y};
    return res;
}

__device__ cfloat __csubf(cfloat lhs, cfloat rhs) {
    cfloat res = {lhs.x - rhs.x, lhs.y - rhs.y};
    return res;
}

__device__ cfloat __cmulf(cfloat lhs, cfloat rhs) {
    cfloat out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

__device__ cfloat __cdivf(cfloat lhs, cfloat rhs) {
    // Normalize by absolute value and multiply
    float rhs_abs     = __cabsf(rhs);
    float inv_rhs_abs = 1.0f / rhs_abs;
    float rhs_x       = inv_rhs_abs * rhs.x;
    float rhs_y       = inv_rhs_abs * rhs.y;
    cfloat out = {lhs.x * rhs_x + lhs.y * rhs_y, lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

__device__ cfloat __cminf(cfloat lhs, cfloat rhs) {
    return __cabsf(lhs) < __cabsf(rhs) ? lhs : rhs;
}

__device__ cfloat __cmaxf(cfloat lhs, cfloat rhs) {
    return __cabsf(lhs) > __cabsf(rhs) ? lhs : rhs;
}
#define __candf(lhs, rhs) __cabsf(lhs) && __cabsf(rhs)
#define __corf(lhs, rhs) __cabsf(lhs) || __cabsf(rhs)
#define __ceqf(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneqf(lhs, rhs) !__ceqf((lhs), (rhs))
#define __cltf(lhs, rhs) (__cabsf(lhs) < __cabsf(rhs))
#define __clef(lhs, rhs) (__cabsf(lhs) <= __cabsf(rhs))
#define __cgtf(lhs, rhs) (__cabsf(lhs) > __cabsf(rhs))
#define __cgef(lhs, rhs) (__cabsf(lhs) >= __cabsf(rhs))
#define __convert_cfloat(real) __cplx2f(real, 0)
#define __convert_c2c(in) (in)
#define __convert_z2c(in) __cplx2f((float)in.x, (float)in.y)

// ----------------------------------------------
// COMPLEX DOUBLE OPERATIONS
// ----------------------------------------------
#define __creal(in) ((in).x)
#define __cimag(in) ((in).y)
#define __cabs(in) hypot(in.x, in.y)

__device__ cdouble __cplx2(double x, double y) {
    cdouble res = {x, y};
    return res;
}

__device__ cdouble __cconj(cdouble in) {
    cdouble res = {in.x, -in.y};
    return res;
}

__device__ cdouble __cadd(cdouble lhs, cdouble rhs) {
    cdouble res = {lhs.x + rhs.x, lhs.y + rhs.y};
    return res;
}

__device__ cdouble __csub(cdouble lhs, cdouble rhs) {
    cdouble res = {lhs.x - rhs.x, lhs.y - rhs.y};
    return res;
}

__device__ cdouble __cmul(cdouble lhs, cdouble rhs) {
    cdouble out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

__device__ cdouble __cdiv(cdouble lhs, cdouble rhs) {
    // Normalize by absolute value and multiply
    double rhs_abs     = __cabs(rhs);
    double inv_rhs_abs = 1.0 / rhs_abs;
    double rhs_x       = inv_rhs_abs * rhs.x;
    double rhs_y       = inv_rhs_abs * rhs.y;
    cdouble out        = {lhs.x * rhs_x + lhs.y * rhs_y,
                          lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

__device__ cdouble __cmin(cdouble lhs, cdouble rhs) {
    return __cabs(lhs) < __cabs(rhs) ? lhs : rhs;
}

__device__ cdouble __cmax(cdouble lhs, cdouble rhs) {
    return __cabs(lhs) > __cabs(rhs) ? lhs : rhs;
}

template<typename T>
static __device__ __inline__ int iszero(T a) {
    return a == T(0);
}

template<typename T>
static __device__ __inline__ int __isinf(const T in) {
    return isinf(in);
}

template<>
__device__ __inline__ int __isinf<__half>(const __half in) {
#if __CUDA_ARCH__ >= 530
    return __hisinf(in);
#else
    return ::isinf(__half2float(in));
#endif
}

__device__ __inline__
__half hmod(const __half lhs, const __half rhs) {
#if __CUDA_ARCH__ >= 530
    return __hsub(lhs, __hmul(htrunc(__hdiv(lhs, rhs)), rhs));
#else
    return __float2half(fmodf(__half2float(lhs), __half2float(rhs)));
#endif
}

template<typename T>
static __device__ __inline__ int __isnan(const T in) {
    return isnan(in);
}

template<>
__device__ __inline__ int __isnan<__half>(const __half in) {
#if __CUDA_ARCH__ >= 530
    return __hisnan(in);
#else
    return ::isnan(__half2float(in));
#endif
}

#define __cand(lhs, rhs) __cabs(lhs) && __cabs(rhs)
#define __cor(lhs, rhs) __cabs(lhs) || __cabs(rhs)
#define __ceq(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneq(lhs, rhs) !__ceq((lhs), (rhs))
#define __clt(lhs, rhs) (__cabs(lhs) < __cabs(rhs))
#define __cle(lhs, rhs) (__cabs(lhs) <= __cabs(rhs))
#define __cgt(lhs, rhs) (__cabs(lhs) > __cabs(rhs))
#define __cge(lhs, rhs) (__cabs(lhs) >= __cabs(rhs))
#define __convert_cdouble(real) __cplx2(real, 0)
#define __convert_z2z(in) (in)
#define __convert_c2z(in) __cplx2((double)in.x, (double)in.y)
