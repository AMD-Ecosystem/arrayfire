/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP shim for the runtime-JIT include "vector_types.h": HIP provides float2 /
// double2 / etc as builtins under hipRTC, so the project's vector_types include
// only needs the HIP vector header.
#pragma once
#include <hip/hip_vector_types.h>
