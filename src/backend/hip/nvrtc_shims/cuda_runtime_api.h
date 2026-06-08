/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for the toolkit include <cuda_runtime_api.h> (the lower
// level runtime-API header). The afhip backend keeps CUDA spelling and relies
// on the force-included hip_compat.h to alias the cudaXxx surface to hipXxx;
// this shim only makes the include resolve and pulls the HIP runtime. It lives
// in nvrtc_shims/ which is on the HIP compiler include path (HIP only), so the
// real toolkit header still wins on a NVIDIA build of src/backend/cuda.

#pragma once

#include <hip/hip_runtime_api.h>
