/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace cuda {
namespace kernel {

static const uint THREADS_PER_BLOCK = 256;
static const uint THREADS_X         = 32;
static const uint THREADS_Y         = THREADS_PER_BLOCK / THREADS_X;
static const uint REPEAT            = 32;

// Wavefront size for the active arch. NVIDIA warps are always 32; AMD wavefronts
// are 64 on CDNA (gfx90a/gfx94x) and 32 on RDNA (gfx10xx/gfx11xx). __GFX9__ is
// defined only during HIP device compilation, so device code uses kWarpSize and
// host code that needs the runtime value should query
// hipGetDeviceProperties(...).warpSize. Never hardcode 32 in warp-staged kernels
// (reduce/scan/ireduce) -- use kWarpSize so the staging is correct on both wave
// widths.
#if defined(__HIP_PLATFORM_AMD__)
#if defined(__GFX9__)
static constexpr int kWarpSize = 64;  // CDNA: gfx90a, gfx94x
#else
static constexpr int kWarpSize = 32;  // RDNA: gfx10xx, gfx11xx
#endif
#else
static constexpr int kWarpSize = 32;  // CUDA
#endif
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
