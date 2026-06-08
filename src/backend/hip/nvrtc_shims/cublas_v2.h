/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP forwarding shim for <cublas_v2.h>, included by the public af/cuda.h
// interop header (and historically by the backend). hipBLAS has no _v2 header;
// forward to it. cublasHandle_t / cublasStatus_t etc. used by af/cuda.h map to
// the hipblas* names via hip_compat.h on the backend; the public header only
// needs the type to exist, which hipblas.h provides as hipblasHandle_t. This
// shim lives in nvrtc_shims/ which is on the HIP compiler include path only, so
// a real NVIDIA build of src/backend/cuda still sees the CUDA toolkit header.

#pragma once

#include <hipblas/hipblas.h>
