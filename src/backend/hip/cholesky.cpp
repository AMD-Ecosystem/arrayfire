/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cholesky.hpp>
#include <common/err_common.hpp>

#include <copy.hpp>
#include <hipblas/hipblas.h>
#include <cusolverDn.hpp>
#include <identity.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <triangle.hpp>

#include <common/err_common.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

// hipsolverStatus_t cusolverDn<>potrf_bufferSize(
//        hipsolverHandle_t handle,
//        hipblasFillMode_t uplo,
//        int n,
//        <> *A,
//        int lda,
//        int *Lwork );
//
// hipsolverStatus_t cusolverDn<>potrf(
//        hipsolverHandle_t handle,
//        hipblasFillMode_t uplo,
//        int n,
//        <> *A, int lda,
//        <> *Workspace, int Lwork,
//        int *devInfo );

template<typename T>
struct potrf_func_def_t {
    using potrf_func_def = hipsolverStatus_t (*)(hipsolverHandle_t,
                                                hipblasFillMode_t, int, T *, int,
                                                T *, int, int *);
};

template<typename T>
struct potrf_buf_func_def_t {
    using potrf_buf_func_def = hipsolverStatus_t (*)(hipsolverHandle_t,
                                                    hipblasFillMode_t, int, T *,
                                                    int, int *);
};

#define CH_FUNC_DEF(FUNC)                                         \
    template<typename T>                                          \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func(); \
                                                                  \
    template<typename T>                                          \
    typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def FUNC##_buf_func();

#define CH_FUNC(FUNC, TYPE, PREFIX)                                         \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               hipsolverDn##PREFIX##FUNC;                                    \
    }                                                                       \
                                                                            \
    template<>                                                              \
    typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def               \
        FUNC##_buf_func<TYPE>() {                                           \
        return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def) &         \
               hipsolverDn##PREFIX##FUNC##_bufferSize;                       \
    }

CH_FUNC_DEF(potrf)
CH_FUNC(potrf, float, S)
CH_FUNC(potrf, double, D)
CH_FUNC(potrf, cfloat, C)
CH_FUNC(potrf, cdouble, Z)

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    Array<T> out = copyArray<T>(in);
    *info        = cholesky_inplace(out, is_upper);

    triangle<T>(out, out, is_upper, false);

    return out;
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    dim4 iDims = in.dims();
    int N      = iDims[0];

    int lwork = 0;

    hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_LOWER;
    if (is_upper) { uplo = HIPBLAS_FILL_MODE_UPPER; }

    CUSOLVER_CHECK(potrf_buf_func<T>()(solverDnHandle(), uplo, N, in.get(),
                                       in.strides()[1], &lwork));

    auto workspace = memAlloc<T>(lwork);
    auto d_info    = memAlloc<int>(1);

    CUSOLVER_CHECK(potrf_func<T>()(solverDnHandle(), uplo, N, in.get(),
                                   in.strides()[1], workspace.get(), lwork,
                                   d_info.get()));

    // FIXME: should return h_info
    return 0;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)
}  // namespace cuda
}  // namespace arrayfire
