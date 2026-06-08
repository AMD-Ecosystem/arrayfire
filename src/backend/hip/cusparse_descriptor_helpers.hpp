/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// HIP backend descriptor helpers. The CUDA backend builds the cuSPARSE generic
// descriptors through common::make_handle<cusparseXxxDescr_t> (type-keyed); on
// HIP the descriptor types are aliasing void* typedefs, so the tag-keyed
// TaggedHandle RAII from cusparse.hpp (SparseSpMatRAII / SparseDnVecRAII /
// SparseDnMatRAII) is used instead (the amgcl void*-aliasing fix). The returned
// TaggedHandle implicitly converts to the raw hipsparse*Descr_t when passed to
// the hipSPARSE SpMV/SpMM/conversion calls.

#if defined(AF_USE_NEW_CUSPARSE_API)
// CUDA Toolkit 10.0 or later

#include <cudaDataType.hpp>
#include <cusparse.hpp>

#include <utility>

namespace arrayfire {
namespace cuda {

template<typename T>
SparseSpMatRAII cusparseDescriptor(const common::SparseArray<T> &in) {
    return make_tagged_handle<hipsparseSpMatDescr_t, SparseSpMatRAII_tag>(in);
}

template<typename T>
SparseDnVecRAII denVecDescriptor(const Array<T> &in) {
    return make_tagged_handle<hipsparseDnVecDescr_t, SparseDnVecRAII_tag>(
        in.elements(), (void *)(in.get()), getType<T>());
}

template<typename T>
SparseDnMatRAII denMatDescriptor(const Array<T> &in) {
    auto dims    = in.dims();
    auto strides = in.strides();
    return make_tagged_handle<hipsparseDnMatDescr_t, SparseDnMatRAII_tag>(
        dims[0], dims[1], strides[1], (void *)in.get(), getType<T>(),
        HIPSPARSE_ORDER_COL);
}

}  // namespace cuda
}  // namespace arrayfire

#endif
