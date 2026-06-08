/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP shim for the runtime-JIT include "cuda_fp16.h": forward to HIP's fp16
// header. hipRTC also provides __half/__float2half/__half2float as builtins, but
// including the header keeps the JIT source's explicit half API resolvable.
#pragma once
#include <hip/hip_fp16.h>
