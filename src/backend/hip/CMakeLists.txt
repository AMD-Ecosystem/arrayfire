# Copyright (c) 2026, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

# HIP/ROCm backend for ArrayFire, cloned from src/backend/cuda. It builds as the
# library `afcuda` and reports AF_BACKEND_CUDA so the unified dispatcher and the
# gtest suite treat it as the CUDA-model backend on AMD; the NVIDIA path
# (src/backend/cuda) is untouched. The defining work is the runtime-JIT engine
# moved from NVRTC + the CUDA driver link step to hipRTC (compile_module.cpp);
# the JIT-source headers the CUDA backend pulled from the CUDA toolkit
# (cuComplex.h / math_constants.h / vector_types.h / cuda_fp16.h) are replaced by
# HIP shims under nvrtc_shims/ so the embedded-header blobs carry HIP-valid
# content under the same variable names compile_module.cpp expects.
#
# STATUS: headless-core bring-up in progress. The hipRTC engine is proven
# (agent_space/af_hiprtc_poc) and ported; the per-file CUDA-library symbol
# translation (hipBLAS/hipSOLVER/hipSPARSE/hipFFT/rocThrust) and the device_manager
# NVML port are the remaining work before the full source list compiles. See
# projects/arrayfire/notes.md for the file-by-file roadmap.

include(FileToString)
include(InternalUtils)

if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

# ---------------------------------------------------------------------------
# HIP / ROCm libraries (headless core: no cuDNN/MIOpen, no Forge graphics).
# ---------------------------------------------------------------------------
find_package(hip       REQUIRED)
find_package(hipblas   REQUIRED)
find_package(hipfft    REQUIRED)
find_package(hipsolver REQUIRED)
find_package(hipsparse REQUIRED)
find_package(hiprand   QUIET)
find_package(rocrand   QUIET)
# rocThrust/hipCUB are header-only on the HIP compiler's default include path
# (/opt/rocm/include); they need no link target for LANGUAGE-HIP sources.

find_library(HIPRTC_LIBRARY hiprtc HINTS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)

# ---------------------------------------------------------------------------
# JIT (element-wise) source header: the kernel/jit.cuh string blob.
# ---------------------------------------------------------------------------
file(GLOB jit_src "kernel/jit.cuh")
file_to_string(
    SOURCES ${jit_src}
    VARNAME jit_files
    EXTENSION "hpp"
    OUTPUT_DIR "kernel_headers"
    TARGETS jit_kernel_targets
    NAMESPACE "arrayfire cuda"
    WITH_EXTENSION
    )

# ---------------------------------------------------------------------------
# nvrtc_kernel_headers: the headers the runtime compiler injects. The CUDA
# backend takes cuComplex.h / cuda_fp16.h(.hpp) / math_constants.h /
# vector_types.h / vector_functions.h from the CUDA toolkit; HIP has none of
# them, so we substitute the shims in nvrtc_shims/ (same basenames -> same
# generated VARNAMEs cuComplex_h / cuda_fp16_h / math_constants_h / ... that
# compile_module.cpp references). All other entries are the project's own headers.
# ---------------------------------------------------------------------------
set(nvrtc_src
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/cuda_fp16.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/cuda_fp16.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/cuComplex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/math_constants.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/vector_types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims/vector_functions.h

  ${PROJECT_SOURCE_DIR}/src/api/c/optypes.hpp
  ${PROJECT_SOURCE_DIR}/include/af/defines.h
  ${PROJECT_SOURCE_DIR}/include/af/traits.hpp
  ${PROJECT_BINARY_DIR}/include/af/version.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assign_kernel_param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dims_param.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/interp.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/shared.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/minmax_op.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utility.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/Binary.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/Transform.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/half.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/internal_enums.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/kernel_type.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/anisotropic_diffusion.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx1.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/approx2.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/assign.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bilateral.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/canny.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve1.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve2.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve3.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/convolve_separable.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/copy.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diagonal.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/diff.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/exampleFunction.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/fftconvolve.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/flood_fill.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/gradient.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/histogram.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/hsv_rgb.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/identity.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iir.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/index.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/iota.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/ireduce.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lookup.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/lu_split.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/match_template.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/meanshift.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/medfilt.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/memcopy.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/moments.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/morph.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/pad_array_borders.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/range.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/resize.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/reorder.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/rotate.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/select.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sobel.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sparse_arith.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/susan.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/tile.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transform.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/transpose_inplace.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/triangle.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/unwrap.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/where.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/wrap.cuh
  )

file_to_string(
    SOURCES ${nvrtc_src}
    VARNAME nvrtc_files
    EXTENSION "hpp"
    OUTPUT_DIR "nvrtc_kernel_headers"
    TARGETS nvrtc_kernel_targets
    NAMESPACE "arrayfire cuda"
    WITH_EXTENSION
    NULLTERM
    )

# ---------------------------------------------------------------------------
# scan-by-key and thrust-sort-by-key generated per-type TUs (mirror the CUDA
# backend's two included sub-CMakeLists; they configure_file the _impl.cu into
# one .cu per (type,op) and append to scan_by_key_sources / thrust_sort_sources).
# ---------------------------------------------------------------------------
include(kernel/scan_by_key/CMakeLists.txt)
include(kernel/thrust_sort_by_key/CMakeLists.txt)

# ---------------------------------------------------------------------------
# afhip is built AS the library `afcuda`: it reports AF_BACKEND_CUDA, so the
# unified loader (which dlopens libafcuda) and the gtest suite treat it as the
# CUDA-model backend on AMD. Source list mirrors src/backend/cuda; the cuDNN
# group stays behind AF_WITH_CUDNN (OFF for the headless core -> MIOpen later).
# ---------------------------------------------------------------------------
add_library(afcuda
    ${thrust_sort_sources}

    blas.cu
    blas.hpp
    cudaDataType.hpp
    cufft.cu
    cufft.hpp
    fft.cu
    sparse.cu
    sparse.hpp
    sparse_arith.cu
    sparse_arith.hpp
    sparse_blas.cu
    sparse_blas.hpp
    cusparse.cpp
    cusparse.hpp
    cusparse_descriptor_helpers.hpp
    solve.cu
    solve.hpp

    EnqueueArgs.hpp
    all.cu
    anisotropic_diffusion.cpp
    any.cu
    approx.cpp
    bilateral.cpp
    canny.cpp
    count.cu
    Event.cpp
    Event.hpp
    exampleFunction.cpp
    fast.cu
    harris.cu
    histogram.cpp
    homography.cu
    hsv_rgb.cpp
    match_template.cpp
    max.cu
    mean.cu
    meanshift.cpp
    medfilt.cpp
    min.cu
    moments.cpp
    nearest_neighbour.cu
    orb.cu
    pad_array_borders.cpp
    product.cu
    random_engine.cu
    regions.cu
    resize.cpp
    rotate.cpp
    set.cu
    sift.cu
    sobel.cpp
    sort.cu
    sort_by_key.cu
    sort_index.cu
    sum.cu
    topk.cu
    transform.cpp
    transpose.cpp
    transpose_inplace.cpp

    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/atomics.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/config.hpp
    kernel/convolve.hpp
    kernel/convolve_separable.cpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fast_lut.hpp
    kernel/fftconvolve.hpp
    kernel/flood_fill.hpp
    kernel/gradient.hpp
    kernel/harris.hpp
    kernel/histogram.hpp
    kernel/homography.hpp
    kernel/hsv_rgb.hpp
    kernel/identity.hpp
    kernel/iir.hpp
    kernel/index.hpp
    kernel/interp.hpp
    kernel/iota.hpp
    kernel/ireduce.hpp
    kernel/lookup.hpp
    kernel/lu_split.hpp
    kernel/match_template.hpp
    kernel/mean.hpp
    kernel/meanshift.hpp
    kernel/medfilt.hpp
    kernel/memcopy.hpp
    kernel/moments.hpp
    kernel/morph.hpp
    kernel/nearest_neighbour.hpp
    kernel/orb.hpp
    kernel/orb_patch.hpp
    kernel/pad_array_borders.hpp
    kernel/random_engine.hpp
    kernel/random_engine_mersenne.hpp
    kernel/random_engine_philox.hpp
    kernel/random_engine_threefry.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/reduce_by_key.hpp
    kernel/regions.hpp
    kernel/reorder.hpp
    kernel/resize.hpp
    kernel/rotate.hpp
    kernel/scan_dim.hpp
    kernel/scan_dim_by_key.hpp
    kernel/scan_dim_by_key_impl.hpp
    kernel/scan_first.hpp
    kernel/scan_first_by_key.hpp
    kernel/scan_first_by_key_impl.hpp
    kernel/select.hpp
    kernel/shared.hpp
    kernel/shfl_intrinsics.hpp
    kernel/sift.hpp
    kernel/sobel.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sparse.hpp
    kernel/sparse_arith.hpp
    kernel/susan.hpp
    kernel/thrust_sort_by_key.hpp
    kernel/thrust_sort_by_key_impl.hpp
    kernel/tile.hpp
    kernel/topk.hpp
    kernel/transform.hpp
    kernel/transpose.hpp
    kernel/transpose_inplace.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/where.hpp
    kernel/wrap.hpp

    Array.cpp
    Array.hpp
    Kernel.cpp
    Kernel.hpp
    LookupTable1D.hpp
    Module.hpp
    Param.hpp
    ThrustAllocator.cuh
    ThrustArrayFirePolicy.hpp
    anisotropic_diffusion.hpp
    approx.hpp
    arith.hpp
    assign.cpp
    assign.hpp
    backend.hpp
    bilateral.hpp
    binary.hpp
    blas.hpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    complex.hpp
    compile_module.cpp
    convolve.cpp
    convolve.hpp
    convolveNN.cpp
    copy.cpp
    copy.hpp
    cublas.cpp
    cublas.hpp

    $<$<BOOL:${AF_WITH_CUDNN}>: cudnn.cpp
                             cudnn.hpp
                             cudnnModule.cpp
                             cudnnModule.hpp>

    cufft.hpp
    cusolverDn.cpp
    cusolverDn.hpp
    device_manager.cpp
    device_manager.hpp
    debug_cuda.hpp
    thrust_utils.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    driver.cpp
    err_cuda.hpp
    exampleFunction.hpp
    fast.hpp
    fast_pyramid.cpp
    fast_pyramid.hpp
    fft.hpp
    fftconvolve.cpp
    fftconvolve.hpp
    flood_fill.cpp
    flood_fill.hpp
    GraphicsResourceManager.cpp
    GraphicsResourceManager.hpp
    gradient.cpp
    gradient.hpp
    harris.hpp
    hist_graphics.cpp
    hist_graphics.hpp
    histogram.hpp
    homography.hpp
    hsv_rgb.hpp
    identity.cpp
    identity.hpp
    iir.cpp
    iir.hpp
    image.cpp
    image.hpp
    index.cpp
    index.hpp
    inverse.cpp
    inverse.hpp
    iota.cpp
    iota.hpp
    ireduce.cpp
    ireduce.hpp
    jit.cpp
    join.cpp
    join.hpp
    logic.hpp
    lookup.cpp
    lookup.hpp
    lu.cpp
    lu.hpp
    match_template.hpp
    math.hpp
    mean.hpp
    meanshift.hpp
    medfilt.hpp
    memory.cpp
    memory.hpp
    minmax_op.hpp
    moments.hpp
    morph.cpp
    morph.hpp
    nearest_neighbour.hpp
    orb.hpp
    platform.cpp
    platform.hpp
    plot.cpp
    plot.hpp
    print.hpp
    qr.cpp
    qr.hpp
    random_engine.hpp
    range.cpp
    range.hpp
    reduce.hpp
    reduce_impl.hpp
    regions.hpp
    reorder.cpp
    reorder.hpp
    resize.hpp
    reshape.cpp
    rotate.hpp
    scalar.hpp
    scan.cpp
    scan.hpp
    scan_by_key.cpp
    scan_by_key.hpp
    select.cpp
    select.hpp
    set.hpp
    shift.cpp
    shift.hpp
    sift.hpp
    sobel.hpp
    solve.hpp
    sort.hpp
    sort_by_key.hpp
    sort_index.hpp
    sparse.hpp
    sparse_arith.hpp
    sparse_blas.hpp
    surface.cpp
    surface.hpp
    susan.cpp
    susan.hpp
    svd.cpp
    svd.hpp
    tile.cpp
    tile.hpp
    threadsMgt.hpp
    topk.hpp
    traits.hpp
    transform.hpp
    transpose.hpp
    triangle.cpp
    triangle.hpp
    types.hpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    utility.cpp
    utility.hpp
    vector_field.cpp
    vector_field.hpp
    where.cpp
    where.hpp
    wrap.cpp
    wrap.hpp

    jit/BufferNode.hpp
    jit/ShiftNode.hpp
    jit/kernel_generators.hpp

    ${scan_by_key_sources}
  )

# Every .cu/.cuh kernel TU compiles through the HIP toolchain; the host .cpp/.hpp
# stay plain C++ (clang). Mark the .cu sources LANGUAGE HIP (the colmap recipe).
get_target_property(_afhip_sources afcuda SOURCES)
foreach(_src ${_afhip_sources})
  if(_src MATCHES "\\.cu$")
    set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

set_target_properties(afcuda PROPERTIES
  HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

# Force shared-library link mode on Windows: the clang driver does not always
# infer the DLL subsystem for this backend (seen with the Visual Studio
# generator selecting clang), which surfaces as a missing-subsystem link error.
if(WIN32)
  target_link_options(afcuda PRIVATE "-shared")
endif()

# Force-include hip_compat.h on every TU (MPPI force-include pattern) so the
# cudaXxx / CUxxx / cuComplex aliases precede any use regardless of per-file
# include order. The host .cpp of this backend call the CUDA runtime/driver API
# directly (cudaStream_t, cudaGetLastError, CUmodule, ...); in the CUDA build
# nvcc supplies those, so on HIP the host CXX TUs (g++) need the same aliasing.
# HIP's runtime headers are host-parseable. (SHELL: keeps "-include <path>" as
# two argv tokens.)
target_compile_options(afcuda PRIVATE
  "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat.h>"
  "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat.h>"
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-deprecated-declarations>)

target_compile_definitions(afcuda
  PRIVATE
    AF_CUDA
    CUDA_NO_HALF
    AF_USE_NEW_CUSPARSE_API
    __HIP_PLATFORM_AMD__
    # HIP_NO_HALF suppresses <hip/hip_fp16.h>'s GLOBAL `using half = __half;`,
    # which otherwise makes bare `half` ambiguous with arrayfire's
    # common::half throughout the force-included shared headers and the api/c
    # layer (which is compiled into this backend via the c_api_interface). __half
    # / __half2 stay defined; the backend uses __half directly and aliases bare
    # `half` to common::half inside namespace cuda (types.hpp).
    HIP_NO_HALF
    $<$<BOOL:${AF_WITH_CUDNN}>:WITH_CUDNN>)

add_library(ArrayFire::afcuda ALIAS afcuda)

add_dependencies(afcuda ${jit_kernel_targets} ${nvrtc_kernel_targets})

target_include_directories(afcuda
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${ArrayFire_SOURCE_DIR}/src/api/c
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/jit
    ${CMAKE_CURRENT_BINARY_DIR})

# nvrtc_shims/ carries HIP forwarding headers for the toolkit includes the
# backend keeps in CUDA spelling (cuda_runtime.h, cuda.h, cublas_v2.h,
# cuComplex.h, cuda_fp16.h, math_constants.h, vector_types.h). It is on the HIP
# include path ONLY (the CUDA backend never sees it), so the NVIDIA build still
# resolves the real toolkit headers.
target_include_directories(afcuda
  SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_shims
    ${ROCM_PATH}/include)

target_link_libraries(afcuda
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    # hip::host only -- NOT hip::device. hip::device injects "-x hip
    # --offload-arch" as INTERFACE_COMPILE_OPTIONS scoped to $<COMPILE_LANGUAGE:CXX>,
    # which would force every host .cpp in this target through the HIP compiler.
    # The .cu kernel TUs get their HIP-ness from LANGUAGE HIP + CMAKE_HIP_COMPILER;
    # host .cpp must stay plain C++ (the colmap isolation invariant).
    hip::host
    roc::hipblas
    hip::hipfft
    roc::hipsolver
    roc::hipsparse
    ${HIPRTC_LIBRARY}
    ${CMAKE_DL_LIBS})

af_split_debug_info(afcuda ${AF_INSTALL_LIB_DIR})

install(TARGETS afcuda
  EXPORT ArrayFireCUDATargets
  COMPONENT cuda
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR})
