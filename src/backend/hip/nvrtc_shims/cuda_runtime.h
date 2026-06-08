/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for the toolkit include <cuda_runtime.h>. The afhip
// backend keeps CUDA spelling and relies on hip_compat.h (force-included on
// every HIP TU) to alias the cudaXxx runtime surface to hipXxx. This shim only
// has to make the include itself resolve and pull the HIP runtime; the symbol
// aliasing is in hip_compat.h. It lives in nvrtc_shims/ which is on the HIP
// compiler include path (HIP only), so the CUDA toolkit header still wins on a
// real NVIDIA build of src/backend/cuda.

#pragma once

#include <hip/hip_runtime.h>
