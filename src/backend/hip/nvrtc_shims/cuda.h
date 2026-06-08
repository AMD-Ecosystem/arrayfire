/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for the CUDA Driver API include <cuda.h>. The afhip
// backend keeps the CUstream/CUmodule/cuModuleLoadData spellings and aliases
// them to the HIP driver API in hip_compat.h (force-included). This shim makes
// the include resolve and pulls the HIP runtime (which carries the driver API
// surface). nvrtc_shims/ is on the HIP compiler include path only.

#pragma once

#include <hip/hip_runtime.h>
