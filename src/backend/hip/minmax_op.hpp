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
#include <common/Binary.hpp>

namespace arrayfire {
namespace cuda {

// These helpers run inside device JIT kernels (ireduce). Under NVRTC the whole
// RTC TU is device code so plain functions are implicitly device-callable;
// hipRTC treats unattributed functions as host, so abs(cfloat) (a __device__
// builtin) is unreachable from them. Mark them __DH__ so they are device code on
// the hipRTC path and host+device under nvcc/hipcc (unchanged semantics).
template<typename T>
static __DH__ double cabs(const T &in) {
    return (double)in;
}

template<>
__DH__ double cabs<char>(const char &in) {
    return (double)(in > 0);
}

template<>
__DH__ double cabs<cfloat>(const cfloat &in) {
    return (double)abs(in);
}

template<>
__DH__ double cabs<cdouble>(const cdouble &in) {
    return (double)abs(in);
}

template<af_op_t op, typename T>
struct MinMaxOp {
    T m_val;
    uint m_idx;
    __DH__ MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {
        using arrayfire::cuda::is_nan;
        if (is_nan(val)) { m_val = common::Binary<compute_t<T>, op>::init(); }
    }

    __DH__ void operator()(T val, uint idx) {
        if ((cabs(val) < cabs(m_val) ||
             (cabs(val) == cabs(m_val) && idx > m_idx))) {
            m_val = val;
            m_idx = idx;
        }
    }
};

template<typename T>
struct MinMaxOp<af_max_t, T> {
    T m_val;
    uint m_idx;
    __DH__ MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {
        using arrayfire::cuda::is_nan;
        if (is_nan(val)) { m_val = common::Binary<T, af_max_t>::init(); }
    }

    __DH__ void operator()(T val, uint idx) {
        if ((cabs(val) > cabs(m_val) ||
             (cabs(val) == cabs(m_val) && idx <= m_idx))) {
            m_val = val;
            m_idx = idx;
        }
    }
};

}  // namespace cuda
}  // namespace arrayfire
