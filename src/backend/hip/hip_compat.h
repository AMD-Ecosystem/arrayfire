/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// HIP/ROCm compatibility shim for the afhip backend (cloned from src/backend/cuda).
//
// The backend keeps its CUDA spelling and `namespace cuda` identity (it reports
// AF_BACKEND_CUDA); this header aliases the CUDA driver/runtime/complex symbols it
// uses to their HIP equivalents so the bulk of the .cu/.cpp/.hpp files compile
// unedited. It is force-included on every afhip translation unit via
// CMAKE_HIP_FLAGS `-include .../hip_compat.h` (MPPI force-include pattern), so the
// aliases precede any use regardless of per-file include order. The CUDA/NVIDIA
// backend never sees this header (it is under src/backend/hip only), so that path
// is byte-for-byte unchanged.
//
// Genuine signature/semantic deltas that an alias cannot bridge (the CU_CHECK
// cuGetErrorString two-arg form, the NVRTC->hipRTC engine flow, library enum
// remaps, wave64) are fixed in the respective source files, not here.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

// ---------------------------------------------------------------------------
// Driver API: types
// ---------------------------------------------------------------------------
#define CUcontext       hipCtx_t
#define CUstream        hipStream_t
#define CUevent         hipEvent_t
#define CUmodule        hipModule_t
#define CUfunction      hipFunction_t
#define CUdeviceptr     hipDeviceptr_t
#define CUresult        hipError_t
#define CUdevice        hipDevice_t
#define CUjit_option    hipJitOption
#define CUlinkState     hiprtcLinkState

// Driver API: status / enums
#define CUDA_SUCCESS                hipSuccess
#define CU_EVENT_BLOCKING_SYNC      hipEventBlockingSync
#define CU_EVENT_DISABLE_TIMING     hipEventDisableTiming

// Driver API: error strings. cuGetErrorName/cuGetErrorString take the two-arg
// (CUresult, const char**) form; HIP's hipDrvGetError* match it exactly (the
// hipGetError* runtime forms return the string directly, so they are NOT the
// right map for the CU_CHECK macro's two-arg usage).
#define cuGetErrorName              hipDrvGetErrorName
#define cuGetErrorString            hipDrvGetErrorString

// Driver API: module / launch / memcpy
#define cuModuleLoadData    hipModuleLoadData
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleGetGlobal   hipModuleGetGlobal
#define cuModuleUnload      hipModuleUnload
#define cuLaunchKernel      hipModuleLaunchKernel
#define cuMemcpyHtoDAsync   hipMemcpyHtoDAsync
#define cuMemcpyDtoHAsync   hipMemcpyDtoHAsync
#define cuMemcpyDtoDAsync   hipMemcpyDtoDAsync

// Driver API: streams / events
#define cuStreamSynchronize hipStreamSynchronize
#define cuStreamWaitEvent   hipStreamWaitEvent
#define cuEventCreate       hipEventCreateWithFlags
#define cuEventDestroy      hipEventDestroy
#define cuEventRecord       hipEventRecord
#define cuEventSynchronize  hipEventSynchronize

// ---------------------------------------------------------------------------
// Runtime API: the cudaXxx surface used by the backend (.cpp host helpers).
// ---------------------------------------------------------------------------
#define cudaError_t                     hipError_t
#define cudaSuccess                     hipSuccess
#define cudaErrorMemoryAllocation       hipErrorOutOfMemory
#define cudaErrorDevicesUnavailable     hipErrorNotReady
#define cudaGetLastError                hipGetLastError
#define cudaGetErrorString              hipGetErrorString
#define cudaStream_t                    hipStream_t
#define cudaEvent_t                     hipEvent_t
#define cudaStreamSynchronize           hipStreamSynchronize
#define cudaStreamWaitEvent             hipStreamWaitEvent
#define cudaDeviceSynchronize           hipDeviceSynchronize
#define cudaMemcpyAsync                 hipMemcpyAsync
#define cudaMemsetAsync                 hipMemsetAsync
#define cudaMemcpy                      hipMemcpy
#define cudaMemset                      hipMemset
#define cudaMemcpyKind                  hipMemcpyKind
#define cudaMemcpyHostToDevice          hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost          hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice        hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost            hipMemcpyHostToHost
#define cudaMemcpyDefault               hipMemcpyDefault
#define cudaStreamLegacy                ((hipStream_t)1)
#define cudaPeekAtLastError             hipPeekAtLastError
#define cudaErrorCudartUnloading        hipErrorDeinitialized
#define cudaStreamCreate                hipStreamCreate
#define cudaMalloc                      hipMalloc
#define cudaFree                        hipFree
#define cudaMallocHost                  hipHostMalloc
#define cudaFreeHost                    hipHostFree
#define cudaMemcpyToSymbolAsync         hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbolAsync       hipMemcpyFromSymbolAsync
#define cudaMemcpyPeerAsync             hipMemcpyPeerAsync

// Device management / version / peer-access surface.
#define cudaDeviceProp                  hipDeviceProp_t
#define cudaGetDeviceProperties         hipGetDeviceProperties
#define cudaGetDeviceCount              hipGetDeviceCount
#define cudaSetDevice                   hipSetDevice
#define cudaDeviceCanAccessPeer         hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess      hipDeviceEnablePeerAccess
#define cudaDriverGetVersion            hipDriverGetVersion
#define cudaRuntimeGetVersion           hipRuntimeGetVersion
#define cudaErrorDeviceAlreadyInUse     hipErrorContextAlreadyInUse

// Texture-object surface (LookupTable1D + interp). hipTextureObject_t and the
// resource/texture descriptor structs map 1:1; the linear-filter rejection on
// float element-read textures is handled at the bind site, not here.
#define cudaTextureObject_t             hipTextureObject_t
#define cudaTextureDesc                 hipTextureDesc
#define cudaResourceDesc                hipResourceDesc
#define cudaCreateTextureObject         hipCreateTextureObject
#define cudaDestroyTextureObject        hipDestroyTextureObject
#define cudaResourceTypeLinear          hipResourceTypeLinear
#define cudaReadModeElementType         hipReadModeElementType
#define cudaChannelFormatKindFloat      hipChannelFormatKindFloat
#define cudaChannelFormatKindSigned     hipChannelFormatKindSigned
#define cudaChannelFormatKindUnsigned   hipChannelFormatKindUnsigned
#define cudaCreateChannelDesc           hipCreateChannelDesc
#define cudaChannelFormatDesc           hipChannelFormatDesc
#define cudaFilterModePoint             hipFilterModePoint
#define cudaFilterModeLinear            hipFilterModeLinear
#define cudaAddressModeClamp            hipAddressModeClamp

// GL/graphics interop (compiled but unused in the headless build; HIP provides
// the full surface so the graphics sources link).
#define cudaGraphicsResource            hipGraphicsResource
#define cudaGraphicsResource_t          hipGraphicsResource_t
#define cudaGraphicsMapResources        hipGraphicsMapResources
#define cudaGraphicsUnmapResources      hipGraphicsUnmapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsGLRegisterBuffer    hipGraphicsGLRegisterBuffer
#define cudaGraphicsUnregisterResource  hipGraphicsUnregisterResource
#define cudaGraphicsMapFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard
#define cudaGLGetDevices                hipGLGetDevices
#define cudaGLDeviceListAll             hipGLDeviceListAll

// ---------------------------------------------------------------------------
// cuBLAS handle-management surface used by platform.cpp / af/cuda.h interop.
// The numeric BLAS calls in blas.cu/solve.cu use the native hipblas* spelling
// (hipify'd, distinct enum coverage); only these few management symbols stay in
// CUDA spelling because they are referenced by host platform code and the
// public afcu_cublasSetMathMode entry point.
// ---------------------------------------------------------------------------
#define cublasSetStream        hipblasSetStream
#define cublasSetMathMode      hipblasSetMathMode
#define cublasSetAtomicsMode   hipblasSetAtomicsMode
#define cublasMath_t           hipblasMath_t
#define CUBLAS_TF32_TENSOR_OP_MATH HIPBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_ATOMICS_ALLOWED HIPBLAS_ATOMICS_ALLOWED

// ---------------------------------------------------------------------------
// cuComplex.h -> hip_complex.h (host-side blas/fft/math helpers).
// ---------------------------------------------------------------------------
#define cuFloatComplex          hipFloatComplex
#define cuDoubleComplex         hipDoubleComplex
#define cuComplex               hipComplex
#define make_cuFloatComplex     make_hipFloatComplex
#define make_cuDoubleComplex    make_hipDoubleComplex
#define cuCaddf                 hipCaddf
#define cuCadd                  hipCadd
#define cuCsubf                 hipCsubf
#define cuCsub                  hipCsub
#define cuCmulf                 hipCmulf
#define cuCmul                  hipCmul
#define cuCdivf                 hipCdivf
#define cuCdiv                  hipCdiv
#define cuCabsf                 hipCabsf
#define cuCabs                  hipCabs
#define cuConjf                 hipConjf
#define cuConj                  hipConj
#define cuCrealf                hipCrealf
#define cuCreal                 hipCreal
#define cuCimagf                hipCimagf
#define cuCimag                 hipCimag
#define cuComplexFloatToDouble  hipComplexFloatToDouble
#define cuComplexDoubleToFloat  hipComplexDoubleToFloat
