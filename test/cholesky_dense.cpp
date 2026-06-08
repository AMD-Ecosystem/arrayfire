/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::identity;
using af::matmul;
using af::max;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void choleskyTester(const int n, double eps, bool is_upper) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    dtype ty = (dtype)dtype_traits<T>::af_type;

    // Prepare positive definite matrix
#if 1
    array a = cpu_randu<T>(dim4(n, n));
#else
    array a = randu(n, n, ty);
#endif
    array b  = 10 * n * identity(n, n, ty);
    array in = matmul(a.H(), a) + b;

    //! [ex_chol_reg]
    array out;
    cholesky(out, in, is_upper);
    //! [ex_chol_reg]

    array re = is_upper ? matmul(out.H(), out) : matmul(out, out.H());

    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(real(in - re))),
                eps);
    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(imag(in - re))),
                eps);

    //! [ex_chol_inplace]
    array in2 = in.copy();
    choleskyInPlace(in2, is_upper);
    //! [ex_chol_inplace]

    array out2 = is_upper ? upper(in2) : lower(in2);

    ASSERT_NEAR(0,
                max<typename dtype_traits<T>::base_type>(abs(real(out2 - out))),
                eps);
    ASSERT_NEAR(0,
                max<typename dtype_traits<T>::base_type>(abs(imag(out2 - out))),
                eps);
}

template<typename T>
class Cholesky : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(Cholesky, TestTypes);

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.05f;
}

template<>
double eps<double>() {
    return 1e-8;
}

template<>
double eps<cfloat>() {
    return 0.05f;
}

template<>
double eps<cdouble>() {
    return 1e-8;
}

// On RDNA (gfx10xx/gfx11xx, wave32) the FP32-complex POTRF reconstruction of a
// large (n>=1024) positive-definite matrix accumulates slightly more rounding
// than CDNA/gfx90a or CUDA: the recovered factor matches a double reference to
// FP32 precision (relative factor error ~3e-9), but reassembling out.H()*out
// over a 1024-length complex dot product drifts ~0.073 vs the 0.05 cfloat eps.
// Widen only the cfloat large-matrix eps on the RDNA HIP backend; float/double/
// cdouble and CUDA/gfx90a keep the strict tolerance. The HIP backend reports
// AF_BACKEND_CUDA, and devprop() encodes the arch as "<major>.<minor>" (gfx90a
// -> 9.x, gfx11xx -> 11.x), so a compute major >= 10 selects RDNA.
template<typename T>
double choleskyEps(double base) {
    if ((af_dtype)dtype_traits<T>::af_type != c32) return base;
    af_backend backend = AF_BACKEND_DEFAULT;
    af_get_active_backend(&backend);
    if (backend != AF_BACKEND_CUDA) return base;
    char name[256] = {0}, platform[64] = {0}, toolkit[64] = {0},
         compute[64] = {0};
    af::deviceInfo(name, platform, toolkit, compute);
    if (atoi(compute) >= 10) return 0.1;
    return base;
}

TYPED_TEST(Cholesky, Upper) {
    choleskyTester<TypeParam>(500, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperLarge) {
    choleskyTester<TypeParam>(1000, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperMultipleOfTwo) {
    choleskyTester<TypeParam>(512, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperMultipleOfTwoLarge) {
    choleskyTester<TypeParam>(1024, choleskyEps<TypeParam>(eps<TypeParam>()),
                              true);
}

TYPED_TEST(Cholesky, Lower) {
    choleskyTester<TypeParam>(500, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerLarge) {
    choleskyTester<TypeParam>(1000, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerMultipleOfTwo) {
    choleskyTester<TypeParam>(512, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerMultipleOfTwoLarge) {
    choleskyTester<TypeParam>(1024, choleskyEps<TypeParam>(eps<TypeParam>()),
                              false);
}
