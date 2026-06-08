/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <svd.hpp>

#include <common/err_common.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include "transpose.hpp"

#include <cusolverDn.hpp>

namespace arrayfire {
namespace cuda {
template<typename T>
hipsolverStatus_t gesvd_buf_func(hipsolverHandle_t /*handle*/, int /*m*/,
                                int /*n*/, int * /*Lwork*/) {
    return HIPSOLVER_STATUS_ARCH_MISMATCH;
}

template<typename T, typename Tr>
hipsolverStatus_t gesvd_func(hipsolverHandle_t /*handle*/, char /*jobu*/,
                            char /*jobvt*/, int /*m*/, int /*n*/, T * /*A*/,
                            int /*lda*/, Tr * /*S*/, T * /*U*/, int /*ldu*/,
                            T * /*VT*/, int /*ldvt*/, T * /*Work*/,
                            int /*Lwork*/, Tr * /*rwork*/, int * /*devInfo*/) {
    return HIPSOLVER_STATUS_ARCH_MISMATCH;
}

#define SVD_SPECIALIZE(T, Tr, X)                                         \
    template<>                                                           \
    hipsolverStatus_t gesvd_buf_func<T>(hipsolverHandle_t handle, int m, \
                                       int n, int *Lwork) {              \
        return hipsolverDn##X##gesvd_bufferSize(handle, m, n, Lwork);     \
    }

SVD_SPECIALIZE(float, float, S);
SVD_SPECIALIZE(double, double, D);
SVD_SPECIALIZE(cfloat, float, C);
SVD_SPECIALIZE(cdouble, double, Z);

#undef SVD_SPECIALIZE

// arrayfire's cfloat/cdouble are plain PODs (not hipFloatComplex), so the
// T* element pointers do not implicitly convert to hipSOLVER's hipFloatComplex*/
// hipDoubleComplex* parameters. NT is the hipSOLVER element type per specialization
// (identical for the real S/D cases); reinterpret the complex pointers (layout
// compatible).
#define SVD_SPECIALIZE(T, Tr, X, NT)                                          \
    template<>                                                                \
    hipsolverStatus_t gesvd_func<T, Tr>(                                      \
        hipsolverHandle_t handle, char jobu, char jobvt, int m, int n, T *A,  \
        int lda, Tr *S, T *U, int ldu, T *VT, int ldvt, T *Work, int Lwork,   \
        Tr *rwork, int *devInfo) {                                            \
        return hipsolverDn##X##gesvd(                                         \
            handle, jobu, jobvt, m, n, reinterpret_cast<NT *>(A), lda, S,     \
            reinterpret_cast<NT *>(U), ldu, reinterpret_cast<NT *>(VT), ldvt, \
            reinterpret_cast<NT *>(Work), Lwork, rwork, devInfo);             \
    }

SVD_SPECIALIZE(float, float, S, float);
SVD_SPECIALIZE(double, double, D, double);
SVD_SPECIALIZE(cfloat, float, C, hipFloatComplex);
SVD_SPECIALIZE(cdouble, double, Z, hipDoubleComplex);

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    int lwork = 0;

    CUSOLVER_CHECK(gesvd_buf_func<T>(solverDnHandle(), M, N, &lwork));

    auto lWorkspace = memAlloc<T>(lwork);
    auto rWorkspace = memAlloc<Tr>(5 * std::min(M, N));

    auto info = memAlloc<int>(1);

    gesvd_func<T, Tr>(solverDnHandle(), 'A', 'A', M, N, in.get(), M, s.get(),
                      u.get(), M, vt.get(), N, lWorkspace.get(), lwork,
                      rWorkspace.get(), info.get());
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    if (M >= N) {
        Array<T> in_copy = copyArray(in);
        svdInPlace(s, u, vt, in_copy);
    } else {
        Array<T> in_trans = transpose(in, true);
        svdInPlace(s, vt, u, in_trans);
        transpose_inplace(vt, true);
        transpose_inplace(u, true);
    }
}

#define INSTANTIATE(T, Tr)                                               \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace cuda
}  // namespace arrayfire
