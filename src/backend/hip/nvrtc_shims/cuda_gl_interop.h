/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for <cuda_gl_interop.h>. The graphics sources (plot /
// image / surface / vector_field / hist_graphics / GraphicsResourceManager)
// keep their cudaGraphics* / cudaGLGetDevices spelling, aliased to the
// hipGraphics* / hipGLGetDevices surface by hip_compat.h. They compile (HIP has
// the full GL-interop API) but are unused in the headless build (Forge OFF,
// checkGraphicsInteropCapability returns false). nvrtc_shims/ is on the HIP
// include path only, so the CUDA backend still sees the real toolkit header.

#pragma once

#include <hip/hip_gl_interop.h>
