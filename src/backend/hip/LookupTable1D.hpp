/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>

namespace arrayfire {
namespace cuda {

// The lookup tables handed to this class are a few dozen entries and are only
// ever point sampled, so they are read straight out of global memory instead of
// through a texture object. CDNA3 and newer devices (gfx942 onwards) drop the
// fixed-function texture path and their HIP headers mark the texture fetch
// builtins unavailable, so a texture object here would not compile there and
// would buy nothing on the devices where it does.
template<typename T>
class LookupTable1D {
   public:
    LookupTable1D()                                     = delete;
    LookupTable1D(const LookupTable1D& arg)             = delete;
    LookupTable1D(const LookupTable1D&& arg)            = delete;
    LookupTable1D& operator=(const LookupTable1D& arg)  = delete;
    LookupTable1D& operator=(const LookupTable1D&& arg) = delete;

    LookupTable1D(const Array<T>& lutArray)
        : mData(lutArray), mPtr(mData.get()) {}

    const T* get() const noexcept { return mPtr; }

   private:
    // Keep a copy so that ref count doesn't go down to zero when
    // original Array<T> goes out of scope before LookupTable1D object does.
    Array<T> mData;
    const T* mPtr;
};

}  // namespace cuda
}  // namespace arrayfire
