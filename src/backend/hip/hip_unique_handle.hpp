/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// HIP descriptor-handle RAII that is keyed on a distinguishing TAG type rather
// than on the raw handle type.
//
// Why this exists: under ROCm several logically distinct handle types are the
// same `void*` typedef, so anything keyed on the handle type cannot tell them
// apart. The shared common::unique_handle<T> / DEFINE_HANDLER(T,...) machinery
// specializes common::ResourceHandler<T>, keyed on the C handle type T. In
// cuBLAS/cuSOLVER/cuSPARSE those handle types are DISTINCT opaque struct
// pointers, so each DEFINE_HANDLER produces a distinct ResourceHandler
// specialization. Under ROCm, hipblasHandle_t, hipsolverHandle_t,
// hipsparseHandle_t, hipsparseMatDescr_t, hipsparseDnVecDescr_t and
// hipsparseDnMatDescr_t are ALL `typedef void*` (hipblas.h / hipsolver.h /
// hipsparse-types.h), so every DEFINE_HANDLER over them would redefine the
// SAME ResourceHandler<void*> ("class member cannot be redeclared") the moment
// a single TU pulls two of these headers (solve.cu pulls cublas+cusolver;
// platform.cpp pulls all three).
//
// Fix: a HIP-only handle template carrying its own create/destroy as a policy
// and disambiguated by a per-logical-handle TAG type, so two void* handles get
// two distinct types. It mirrors common::unique_handle's interface (default
// ctor zero-inits, move-only, guarded reset, implicit `operator const Raw&`,
// variadic create, operator bool) so call sites (platform.cpp managers, the
// cusparse descriptor make_handle helpers) read the same. The raw ROCm handle
// type still flows to the ROCm API through the implicit conversion, so the
// many `cusolverStatus_t(*)(cusolverDnHandle_t, ...)` function-pointer
// signatures in solve/qr/lu/svd/cholesky stay the raw type untouched.
//
// The CUDA backend never includes this header (it is under src/backend/hip
// only) and keeps the type-keyed common::unique_handle, so the NVIDIA path is
// byte-for-byte unchanged.

#include <utility>

namespace arrayfire {
namespace cuda {

template<typename Raw, typename Tag>
class TaggedHandle {
    Raw handle_;

   public:
    constexpr TaggedHandle() noexcept : handle_(0) {}
    explicit constexpr TaggedHandle(Raw handle) noexcept : handle_(handle) {}
    ~TaggedHandle() noexcept { reset(); }

    void reset() noexcept {
        if (handle_) {
            Tag::destroy(handle_);
            handle_ = 0;
        }
    }

    TaggedHandle(const TaggedHandle&)            = delete;
    TaggedHandle& operator=(const TaggedHandle&) = delete;

    TaggedHandle(TaggedHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = 0;
    }
    TaggedHandle& operator=(TaggedHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_       = other.handle_;
            other.handle_ = 0;
        }
        return *this;
    }

    constexpr operator const Raw&() const noexcept { return handle_; }

    template<typename... Args>
    int create(Args... args) {
        if (!handle_) {
            int error = Tag::create(&handle_, std::forward<Args>(args)...);
            if (error) { handle_ = 0; }
            return error;
        }
        return 0;
    }

    constexpr operator bool() const noexcept { return handle_ != 0; }
};

template<typename Raw, typename Tag, typename... Args>
TaggedHandle<Raw, Tag> make_tagged_handle(Args... args) {
    TaggedHandle<Raw, Tag> h;
    h.create(std::forward<Args>(args)...);
    return h;
}

// Convenience matching common::make_handle<T>(args...) but for a TaggedHandle
// alias type T (e.g. SparseDescriptorRAII): default-construct and create. Lets
// sparse call sites read `make_handle<SparseDescriptorRAII>()` like the CUDA
// backend's `make_handle<cusparseMatDescr_t>()`.
template<typename HandleT, typename... Args>
HandleT make_handle(Args... args) {
    HandleT h;
    h.create(std::forward<Args>(args)...);
    return h;
}

}  // namespace cuda
}  // namespace arrayfire

// Define a TAG struct whose static create/destroy forward to the library's
// creator/destroyer, plus an alias NAME = TaggedHandle<RAW, Tag>. Because the
// tag type is unique per NAME, two void* RAW handles yield two distinct types.
#define DEFINE_HIP_HANDLE(NAME, RAW, HCREATOR, HDESTROYER)        \
    namespace arrayfire {                                         \
    namespace cuda {                                              \
    struct NAME##_tag {                                           \
        template<typename... Args>                                \
        static int create(RAW* handle, Args... args) {            \
            return HCREATOR(handle, std::forward<Args>(args)...); \
        }                                                         \
        static int destroy(RAW handle) { return HDESTROYER(handle); } \
    };                                                            \
    using NAME = TaggedHandle<RAW, NAME##_tag>;                   \
    }                                                             \
    }
