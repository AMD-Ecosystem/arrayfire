/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for the CUDA toolkit include <driver_types.h> (pulled by
// GraphicsResourceManager.hpp for cudaGraphicsResource_t etc.). The cudaXxx
// names are aliased to the hipXxx surface by hip_compat.h; this only makes the
// include resolve and pulls the HIP runtime. nvrtc_shims/ is on the HIP compiler
// include path only, so the CUDA backend still sees the real toolkit header.

#pragma once

#include <hip/hip_runtime.h>
