/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// HIP backend cuSPARSE compat. Unlike the NVIDIA build (which dlopens cuSPARSE
// through cusparseModule and calls _.cusparseXxx), the HIP backend links
// roc::hipsparse directly and calls the functions by their cuSPARSE spelling
// (aliased to hipsparse* by nvrtc_shims/cusparse_v2.h). This header therefore
// drops the getCusparsePlugin() indirection the CUDA cusparse.hpp carries.
//
// The descriptor RAII is the amgcl void*-aliasing fix: hipSPARSE typedefs
// hipsparseDnVecDescr_t and hipsparseDnMatDescr_t (and hipsparseHandle_t /
// hipsparseMatDescr_t) all to `void*`, so the shared type-keyed
// common::unique_handle<T> / DEFINE_HANDLER(T,...) machinery would redefine the
// same ResourceHandler<void*> for two of them in one TU (sparse.cu pulls the
// matrix descriptor AND the dense descriptors). Use the tag-keyed DEFINE_HIP_HANDLE
// TaggedHandle<Raw,Tag> instead, mirroring how platform.cpp keys the blas /
// solver / sparse HANDLES. hipsparseSpMatDescr_t is a distinct struct pointer in
// hipSPARSE 4.2 but is given a tag too for uniformity. make_handle<T>(args...)
// keeps the call-site spelling the descriptor helpers use.

#include <common/SparseArray.hpp>
#include <common/defines.hpp>
#include <cudaDataType.hpp>
#include <cusparse_v2.h>  // shim -> hipsparse/hipsparse.h (+ cusparse* aliases)
#include <err_cuda.hpp>
#include <hip_unique_handle.hpp>

#if defined(AF_USE_NEW_CUSPARSE_API)
namespace arrayfire {
namespace cuda {

template<typename T>
hipsparseStatus_t createSpMatDescr(
    hipsparseSpMatDescr_t *out, const arrayfire::common::SparseArray<T> &arr) {
    switch (arr.getStorage()) {
        case AF_STORAGE_CSR: {
            return hipsparseCreateCsr(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getRowIdx().get(), (void *)arr.getColIdx().get(),
                (void *)arr.getValues().get(), HIPSPARSE_INDEX_32I,
                HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
        case AF_STORAGE_CSC: {
            return hipsparseCreateCsc(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getColIdx().get(), (void *)arr.getRowIdx().get(),
                (void *)arr.getValues().get(), HIPSPARSE_INDEX_32I,
                HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
        case AF_STORAGE_COO: {
            return hipsparseCreateCoo(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getRowIdx().get(), (void *)arr.getColIdx().get(),
                (void *)arr.getValues().get(), HIPSPARSE_INDEX_32I,
                HIPSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

}  // namespace cuda
}  // namespace arrayfire
#endif

// Tag-keyed RAII for the cuSPARSE handle + descriptors (void* aliasing fix).
// createSpMatDescr is templated on the SparseArray, so its tag forwards a
// variadic create directly to the function template (TaggedHandle::create is
// variadic and Tag::create just forwards).
DEFINE_HIP_HANDLE(SparseDescriptorRAII, hipsparseMatDescr_t,
                  hipsparseCreateMatDescr, hipsparseDestroyMatDescr);
DEFINE_HIP_HANDLE(SparseDnVecRAII, hipsparseDnVecDescr_t, hipsparseCreateDnVec,
                  hipsparseDestroyDnVec);
DEFINE_HIP_HANDLE(SparseDnMatRAII, hipsparseDnMatDescr_t, hipsparseCreateDnMat,
                  hipsparseDestroyDnMat);

#if defined(AF_USE_NEW_CUSPARSE_API)
namespace arrayfire {
namespace cuda {
struct SparseSpMatRAII_tag {
    template<typename T>
    static int create(hipsparseSpMatDescr_t *handle,
                      const common::SparseArray<T> &arr) {
        return createSpMatDescr<T>(handle, arr);
    }
    static int destroy(hipsparseSpMatDescr_t handle) {
        return hipsparseDestroySpMat(handle);
    }
};
using SparseSpMatRAII = TaggedHandle<hipsparseSpMatDescr_t, SparseSpMatRAII_tag>;
}  // namespace cuda
}  // namespace arrayfire
#endif

namespace arrayfire {
namespace cuda {

const char *errorString(hipsparseStatus_t err);

#define CUSPARSE_CHECK(fn)                                                    \
    do {                                                                      \
        hipsparseStatus_t _error = fn;                                        \
        if (_error != HIPSPARSE_STATUS_SUCCESS) {                             \
            char _err_msg[1024];                                             \
            snprintf(_err_msg, sizeof(_err_msg), "CUSPARSE Error (%d): %s\n", \
                     (int)(_error), arrayfire::cuda::errorString(_error));   \
                                                                              \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                             \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace arrayfire
