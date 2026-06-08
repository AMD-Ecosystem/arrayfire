/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP shim for the runtime-JIT include "math_constants.h" (CUDA toolkit header,
// absent on ROCm). The JIT source (interp.hpp, math.hpp) uses CUDART_INF,
// CUDART_INF_F and CUDART_PI. Values match the CUDA math_constants.h definitions.

#pragma once

#define CUDART_INF_F   __int_as_float(0x7f800000)
#define CUDART_NAN_F   __int_as_float(0x7fffffff)
#define CUDART_PI_F    3.141592654f

#define CUDART_INF     __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN     __longlong_as_double(0xfff8000000000000ULL)
#define CUDART_PI      3.1415926535897931e+0
