/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/compile_module.hpp>  //compileModule & loadModuleFromDisk
#include <common/kernel_cache.hpp>    //getKernel(Module&, ...)

#include <Module.hpp>
#include <common/Logger.hpp>
#include <common/deterministicHash.hpp>
#include <common/internal_enums.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <kernel_headers/jit_cuh.hpp>
#include <nvrtc_kernel_headers/Binary_hpp.hpp>
#include <nvrtc_kernel_headers/Param_hpp.hpp>
#include <nvrtc_kernel_headers/Transform_hpp.hpp>
#include <nvrtc_kernel_headers/assign_kernel_param_hpp.hpp>
#include <nvrtc_kernel_headers/backend_hpp.hpp>
#include <nvrtc_kernel_headers/cuComplex_h.hpp>
#include <nvrtc_kernel_headers/cuda_fp16_h.hpp>
#include <nvrtc_kernel_headers/cuda_fp16_hpp.hpp>
#include <nvrtc_kernel_headers/defines_h.hpp>
#include <nvrtc_kernel_headers/dims_param_hpp.hpp>
#include <nvrtc_kernel_headers/half_hpp.hpp>
#include <nvrtc_kernel_headers/internal_enums_hpp.hpp>
#include <nvrtc_kernel_headers/interp_hpp.hpp>
#include <nvrtc_kernel_headers/kernel_type_hpp.hpp>
#include <nvrtc_kernel_headers/math_constants_h.hpp>
#include <nvrtc_kernel_headers/math_hpp.hpp>
#include <nvrtc_kernel_headers/minmax_op_hpp.hpp>
#include <nvrtc_kernel_headers/optypes_hpp.hpp>
#include <nvrtc_kernel_headers/shared_hpp.hpp>
#include <nvrtc_kernel_headers/traits_hpp.hpp>
#include <nvrtc_kernel_headers/types_hpp.hpp>
#include <nvrtc_kernel_headers/utility_hpp.hpp>
#include <nvrtc_kernel_headers/vector_functions_h.hpp>
#include <nvrtc_kernel_headers/vector_types_h.hpp>
#include <nvrtc_kernel_headers/version_h.hpp>
#include <optypes.hpp>
#include <platform.hpp>
#include <af/defines.h>
#include <af/version.h>

#include <hip/hiprtc.h>

#include <algorithm>
#include <array>
#if defined(_WIN32)
#include <filesystem>
#else
#include <dirent.h>
#endif

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using arrayfire::common::getCacheDirectory;
using arrayfire::common::makeTempFilename;
using arrayfire::common::removeFile;
using arrayfire::common::renameFile;
using arrayfire::cuda::getComputeCapability;
using arrayfire::cuda::getDeviceProp;
using detail::Module;
using nonstd::span;
using std::accumulate;
using std::array;
using std::back_insert_iterator;
using std::begin;
using std::end;
using std::extent;
using std::find_if;
using std::make_pair;
using std::ofstream;
using std::pair;
using std::string;
using std::to_string;
using std::transform;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// HIP port: arrayfire's CUDA backend runtime-compiles kernels with NVRTC, emits
// PTX, then assembles+links it through the CUDA driver (cuLinkCreate /
// cuLinkAddData(CU_JIT_INPUT_PTX) / cuLinkComplete) before cuModuleLoadData.
// hipRTC compiles directly to a loadable gfx code object (hiprtcGetCode), so the
// PTX + cuLink* assemble step does not exist here: hiprtcGetCode feeds straight
// into hipModuleLoadData. The arch flag is --offload-arch=<gcnArchName> (feature
// suffix stripped) rather than --gpu-architecture=compute_XX. Proven end to end
// in agent_space/af_hiprtc_poc before this port. Three NVRTC->hipRTC deltas:
//   1. hipRTC rejects an empty-string ("") header source; dummy header bodies
//      must be a non-empty placeholder.
//   2. hipRTC rejects --device-as-default-execution-space (device is the default
//      execution space under HIP); it is dropped.
//   3. an injected header that #includes <hip/...> needs -I<rocm>/include at
//      runtime-compile time, so the ROCm include dir is appended to the options.

#define HIPRTC_CHECK(fn)                                                   \
    do {                                                                   \
        hiprtcResult res = (fn);                                           \
        if (res == HIPRTC_SUCCESS) break;                                  \
        array<char, 4096> rtc_err_msg;                                     \
        snprintf(rtc_err_msg.data(), rtc_err_msg.size(),                   \
                 "hipRTC Error(%d): %s\n", res, hiprtcGetErrorString(res)); \
        AF_ERROR(rtc_err_msg.data(), AF_ERR_INTERNAL);                     \
    } while (0)

#define HIPRTC_COMPILE_CHECK(fn)                             \
    do {                                                     \
        hiprtcResult res = (fn);                             \
        if (res == HIPRTC_SUCCESS) break;                    \
        size_t logSize;                                      \
        hiprtcGetProgramLogSize(prog, &logSize);             \
        vector<char> log(logSize + 1);                       \
        hiprtcGetProgramLog(prog, log.data());               \
        log[logSize] = '\0';                                 \
        array<char, 4096> rtc_err_msg;                       \
        snprintf(rtc_err_msg.data(), rtc_err_msg.size(),     \
                 "hipRTC Error(%d): %s\nLog: \n%s\n", res,   \
                 hiprtcGetErrorString(res), log.data());     \
        AF_ERROR(rtc_err_msg.data(), AF_ERR_INTERNAL);       \
    } while (0)

spdlog::logger *getLogger() {
    static std::shared_ptr<spdlog::logger> logger(
        arrayfire::common::loggerFactory("jit"));
    return logger.get();
}

string getKernelCacheFilename(const int device, const string &key) {
    // Key the on-disk code-object cache on the gfx arch (per-device code objects
    // are arch-specific) instead of the NVIDIA compute capability.
    string archName = getDeviceProp(device).gcnArchName;
    if (auto pos = archName.find(':'); pos != string::npos) {
        archName = archName.substr(0, pos);
    }

    return "KER" + key + "_HIP_" + archName + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".bin";
}

namespace arrayfire {
namespace common {

Module compileModule(const string &moduleKey, span<const string> sources,
                     span<const string> opts, span<const string> kInstances,
                     const bool sourceIsJIT) {
    hiprtcProgram prog;
    using namespace arrayfire::cuda;
    // hipRTC rejects an empty-string ("") header body (INVALID_INPUT), unlike
    // NVRTC; every dummy/placeholder header source must be non-empty.
    static const char *const kEmptyHeader = "/* intentionally empty */\n";
    if (sourceIsJIT) {
        constexpr const char *header_names[] = {
            "utility",        "cuda_fp16.hpp",      "cuda_fp16.h",
            "vector_types.h", "vector_functions.h",
        };
        constexpr size_t numHeaders = extent<decltype(header_names)>::value;
        array<const char *, numHeaders> headers = {
            kEmptyHeader, cuda_fp16_hpp, cuda_fp16_h,
            vector_types_h, vector_functions_h,
        };
        static_assert(headers.size() == numHeaders,
                      "headers array contains fewer sources than header_names");
        HIPRTC_CHECK(hiprtcCreateProgram(&prog, sources[0].c_str(),
                                         moduleKey.c_str(), numHeaders,
                                         headers.data(), header_names));
    } else {
        constexpr static const char *includeNames[] = {
            "math.h",          // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "stdbool.h",       // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "stdlib.h",        // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "vector_types.h",  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "utility",         // DUMMY ENTRY TO SATISFY utility inclusion
            "backend.hpp",
            "cuComplex.h",
            "jit.cuh",
            "math.hpp",
            "optypes.hpp",
            "Param.hpp",
            "shared.hpp",
            "types.hpp",
            "cuda_fp16.hpp",
            "cuda_fp16.h",
            "common/Binary.hpp",
            "common/Transform.hpp",
            "common/half.hpp",
            "common/kernel_type.hpp",
            "af/traits.hpp",
            "interp.hpp",
            "math_constants.h",
            "af/defines.h",
            "af/version.h",
            "utility.hpp",
            "assign_kernel_param.hpp",
            "dims_param.hpp",
            "common/internal_enums.hpp",
            "minmax_op.hpp",
            "vector_functions.h",
        };

        constexpr size_t numHeaders = extent<decltype(includeNames)>::value;
        static const array<string, numHeaders> sourceStrings = {{
            string(kEmptyHeader),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(kEmptyHeader),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(kEmptyHeader),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(kEmptyHeader),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(kEmptyHeader),  // DUMMY ENTRY TO SATISFY utility inclusion
            string(backend_hpp, backend_hpp_len),
            string(cuComplex_h, cuComplex_h_len),
            string(jit_cuh, jit_cuh_len),
            string(math_hpp, math_hpp_len),
            string(optypes_hpp, optypes_hpp_len),
            string(Param_hpp, Param_hpp_len),
            string(shared_hpp, shared_hpp_len),
            string(types_hpp, types_hpp_len),
            string(cuda_fp16_hpp, cuda_fp16_hpp_len),
            string(cuda_fp16_h, cuda_fp16_h_len),
            string(Binary_hpp, Binary_hpp_len),
            string(Transform_hpp, Transform_hpp_len),
            string(half_hpp, half_hpp_len),
            string(kernel_type_hpp, kernel_type_hpp_len),
            string(traits_hpp, traits_hpp_len),
            string(interp_hpp, interp_hpp_len),
            string(math_constants_h, math_constants_h_len),
            string(defines_h, defines_h_len),
            string(version_h, version_h_len),
            string(utility_hpp, utility_hpp_len),
            string(assign_kernel_param_hpp, assign_kernel_param_hpp_len),
            string(dims_param_hpp, dims_param_hpp_len),
            string(internal_enums_hpp, internal_enums_hpp_len),
            string(minmax_op_hpp, minmax_op_hpp_len),
            string(vector_functions_h, vector_functions_h_len),
        }};

        static const char *headers[] = {
            sourceStrings[0].c_str(),  sourceStrings[1].c_str(),
            sourceStrings[2].c_str(),  sourceStrings[3].c_str(),
            sourceStrings[4].c_str(),  sourceStrings[5].c_str(),
            sourceStrings[6].c_str(),  sourceStrings[7].c_str(),
            sourceStrings[8].c_str(),  sourceStrings[9].c_str(),
            sourceStrings[10].c_str(), sourceStrings[11].c_str(),
            sourceStrings[12].c_str(), sourceStrings[13].c_str(),
            sourceStrings[14].c_str(), sourceStrings[15].c_str(),
            sourceStrings[16].c_str(), sourceStrings[17].c_str(),
            sourceStrings[18].c_str(), sourceStrings[19].c_str(),
            sourceStrings[20].c_str(), sourceStrings[21].c_str(),
            sourceStrings[22].c_str(), sourceStrings[23].c_str(),
            sourceStrings[24].c_str(), sourceStrings[25].c_str(),
            sourceStrings[26].c_str(), sourceStrings[27].c_str(),
            sourceStrings[28].c_str(), sourceStrings[29].c_str()};
        static_assert(extent<decltype(headers)>::value == numHeaders,
                      "headers array contains fewer sources than includeNames");
        HIPRTC_CHECK(hiprtcCreateProgram(&prog, sources[0].c_str(),
                                         moduleKey.c_str(), numHeaders, headers,
                                         includeNames));
    }

    int device = getActiveDeviceId();
    // --offload-arch=<gcnArchName> with the :sramecc+:xnack- feature suffix
    // stripped (e.g. "gfx90a:sramecc+:xnack-" -> "gfx90a"). This replaces
    // NVRTC's --gpu-architecture=compute_XX. The same string keys the disk cache.
    string archName = getDeviceProp(device).gcnArchName;
    if (auto pos = archName.find(':'); pos != string::npos) {
        archName = archName.substr(0, pos);
    }
    const string archOpt = "--offload-arch=" + archName;
    // -I<rocm>/include so injected headers that #include <hip/...> resolve.
    const string rocmInclude = []() -> string {
        if (const char *p = std::getenv("ROCM_PATH")) return string(p) + "/include";
        return string("/opt/rocm/include");
    }();
    const string rocmIncludeOpt = "-I" + rocmInclude;
    // hipRTC compiles with clang but does not add clang's own builtin-header
    // directory (stddef.h, stdint.h, ...), so a JIT source that transitively
    // pulls a libc header (af/defines.h -> <stdlib.h> -> <stddef.h>) fails
    // "'stddef.h' file not found". Point -isystem at the ROCm clang resource
    // include dir (lib/llvm/lib/clang/<ver>/include) so the builtin headers
    // resolve. The .cpp is host-compiled by g++, so the clang version is
    // discovered from the filesystem rather than __clang_major__.
    const string rocmRoot = rocmInclude.substr(0, rocmInclude.rfind("/include"));
    const string clangResourceOpt = [&]() -> string {
        const string base = rocmRoot + "/lib/llvm/lib/clang";
        string ver;
#if defined(_WIN32)
        try {
            for (const auto &entry :
                 std::filesystem::directory_iterator(
                     base, std::filesystem::directory_options::skip_permission_denied)) {
                const string name = entry.path().filename().string();
                if (!name.empty() && name[0] != '.') {
                    ver = name;
                    break;
                }
            }
        } catch (const std::filesystem::filesystem_error &) {}
#else
        if (DIR *d = opendir(base.c_str())) {
            for (dirent *e; (e = readdir(d)) != nullptr;) {
                if (e->d_name[0] != '.') {
                    ver = e->d_name;
                    break;
                }
            }
            closedir(d);
        }
#endif
        if (!ver.empty()) return "-isystem" + base + "/" + ver + "/include";
        return "-isystem" + base + "/include";
    }();
    vector<const char *> compiler_options = {
        archOpt.c_str(),
        "--std=c++17",
        rocmIncludeOpt.c_str(),
        clangResourceOpt.c_str(),
        // NVRTC auto-defines __CUDACC_RTC__, which arrayfire's headers key on to
        // skip their host-only #includes (af/defines.h guards <stdlib.h> /
        // "af/compilers.h" etc. behind `#ifndef __CUDACC_RTC__`) and to pick the
        // RTC type set (types.hpp dim_t). hipRTC does NOT define it, so define it
        // here; this is what makes the embedded af/* headers self-contained under
        // hipRTC (and is why most dummy header bodies are then unused).
        "-D__CUDACC_RTC__",
        // The embedded device headers (common/half.hpp's half<->int conversions,
        // the fp16 min/max) gate their device intrinsic path on __CUDA_ARCH__,
        // which hipRTC does not define -- without it they fall to the host path
        // ("call to __host__ function from __device__ function"). Everything in a
        // hipRTC compile is device code, and HIP provides the __half2short_rn /
        // __half2ll_rn / __hlt intrinsics, so define __CUDA_ARCH__ to a gfx9-class
        // value (>=530 also enables the native-fp16 branches).
        "-D__CUDA_ARCH__=900",
        // Marks a hipRTC (vs NVRTC) runtime compile, so the embedded headers can
        // tell the two RTC paths apart -- e.g. AF_CONSTEXPR must be empty on
        // hipRTC because the half ctors call __float2half-style intrinsics that
        // are not constexpr there (clang errors -Winvalid-constexpr).
        "-D__HIP_RTC__",
#ifdef AF_WITH_FAST_MATH
        "-ffast-math",
        "-DAF_WITH_FAST_MATH",
#endif
    };
    // The compile opts come from the shared DefineValue / DefineKeyValue macros
    // as single strings like " -D NAME=value" (NVRTC accepts a flag and its
    // argument joined by a space as ONE option; clang/hipRTC does NOT -- it reads
    // "-D NAME=value" as a filename and fails "cannot specify -o when generating
    // multiple output files"). Split each opt on whitespace into separate tokens
    // so clang sees "-D" and "NAME=value" as distinct argv entries. The token
    // strings must outlive hiprtcCompileProgram, so they are owned here.
    vector<string> opt_tokens;
    if (!sourceIsJIT) {
        for (const string &s : opts) {
            size_t i = 0;
            while (i < s.size()) {
                while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
                size_t j = i;
                while (j < s.size() && !std::isspace((unsigned char)s[j])) ++j;
                if (j > i) opt_tokens.push_back(s.substr(i, j - i));
                i = j;
            }
        }
        for (const string &t : opt_tokens) {
            compiler_options.push_back(t.c_str());
        }

        for (auto &instantiation : kInstances) {
            HIPRTC_CHECK(hiprtcAddNameExpression(prog, instantiation.c_str()));
        }
    }

    auto compile = high_resolution_clock::now();
    HIPRTC_COMPILE_CHECK(hiprtcCompileProgram(prog, compiler_options.size(),
                                              compiler_options.data()));
    auto compile_end = high_resolution_clock::now();

    // hipRTC emits a loadable gfx code object directly (no PTX, no cuLink* step):
    // hiprtcGetCode -> hipModuleLoadData.
    auto link = high_resolution_clock::now();
    size_t codeSize = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    vector<char> code(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

    CUmodule modOut = nullptr;
    CU_CHECK(cuModuleLoadData(&modOut, code.data()));
    auto link_end = high_resolution_clock::now();

    Module retVal(modOut);
    if (!sourceIsJIT) {
        for (auto &instantiation : kInstances) {
            // memory owned by the hiprtcProgram until it is destroyed
            const char *name = nullptr;
            HIPRTC_CHECK(
                hiprtcGetLoweredName(prog, instantiation.c_str(), &name));
            retVal.add(instantiation, string(name, strlen(name)));
        }
    }

#ifdef AF_CACHE_KERNELS_TO_DISK
    // save kernel in cache
    const string &cacheDirectory = getCacheDirectory();
    if (!cacheDirectory.empty()) {
        const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                                 getKernelCacheFilename(device, moduleKey);
        const string tempFile =
            cacheDirectory + AF_PATH_SEPARATOR + makeTempFilename();
        try {
            // write module hash(everything: names, code & options) and CUBIN
            // data
            ofstream out(tempFile, std::ios::binary);
            if (!sourceIsJIT) {
                size_t mangledNamesListSize = retVal.map().size();
                out.write(reinterpret_cast<const char *>(&mangledNamesListSize),
                          sizeof(mangledNamesListSize));
                for (auto &iter : retVal.map()) {
                    size_t kySize   = iter.first.size();
                    size_t vlSize   = iter.second.size();
                    const char *key = iter.first.c_str();
                    const char *val = iter.second.c_str();
                    out.write(reinterpret_cast<const char *>(&kySize),
                              sizeof(kySize));
                    out.write(key, iter.first.size());
                    out.write(reinterpret_cast<const char *>(&vlSize),
                              sizeof(vlSize));
                    out.write(val, iter.second.size());
                }
            }

            // compute code-object hash
            const size_t cubinHash = deterministicHash(code.data(), codeSize);

            out.write(reinterpret_cast<const char *>(&cubinHash),
                      sizeof(cubinHash));
            out.write(reinterpret_cast<const char *>(&codeSize),
                      sizeof(codeSize));
            out.write(code.data(), codeSize);
            out.close();

            // try to rename temporary file into final cache file, if this fails
            // this means another thread has finished compiling this kernel
            // before the current thread.
            if (!renameFile(tempFile, cacheFile)) { removeFile(tempFile); }
        } catch (const std::ios_base::failure &e) {
            AF_TRACE("{{{:<30} : failed saving binary to {} for {}, {}}}",
                     moduleKey, cacheFile, getDeviceProp(device).name,
                     e.what());
        }
    }
#endif

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    // skip the --std flag in the trace; it does not change
    auto listOpts = [](vector<const char *> &in) {
        return accumulate(begin(in) + 2, end(in), string(in[0]),
                          [](const string &lhs, const string &rhs) {
                              return lhs + ", " + rhs;
                          });
    };
    AF_TRACE("{{ {:<20} : compile:{:>5} ms, link:{:>4} ms, {{ {} }}, {} }}",
             moduleKey,
             duration_cast<milliseconds>(compile_end - compile).count(),
             duration_cast<milliseconds>(link_end - link).count(),
             listOpts(compiler_options), getDeviceProp(device).name);
    return retVal;
}

Module loadModuleFromDisk(const int device, const string &moduleKey,
                          const bool isJIT) {
    const string &cacheDirectory = getCacheDirectory();
    if (cacheDirectory.empty()) return Module{nullptr};

    const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                             getKernelCacheFilename(device, moduleKey);

    CUmodule modOut = nullptr;
    Module retVal{nullptr};
    try {
        std::ifstream in(cacheFile, std::ios::binary);
        if (!in) {
            AF_TRACE("{{{:<20} : Unable to open {} for {}}}", moduleKey,
                     cacheFile, getDeviceProp(device).name);
            removeFile(cacheFile);  // Remove if exists
            return Module{nullptr};
        }
        in.exceptions(std::ios::failbit | std::ios::badbit);

        if (!isJIT) {
            size_t mangledListSize = 0;
            in.read(reinterpret_cast<char *>(&mangledListSize),
                    sizeof(mangledListSize));
            for (size_t i = 0; i < mangledListSize; ++i) {
                size_t keySize = 0;
                in.read(reinterpret_cast<char *>(&keySize), sizeof(keySize));
                vector<char> key;
                key.reserve(keySize);
                in.read(key.data(), keySize);

                size_t itemSize = 0;
                in.read(reinterpret_cast<char *>(&itemSize), sizeof(itemSize));
                vector<char> item;
                item.reserve(itemSize);
                in.read(item.data(), itemSize);

                retVal.add(string(key.data(), keySize),
                           string(item.data(), itemSize));
            }
        }

        size_t cubinHash = 0;
        in.read(reinterpret_cast<char *>(&cubinHash), sizeof(cubinHash));
        size_t cubinSize = 0;
        in.read(reinterpret_cast<char *>(&cubinSize), sizeof(cubinSize));
        vector<char> cubin(cubinSize);
        in.read(cubin.data(), cubinSize);
        in.close();

        // check CUBIN binary data has not been corrupted
        const size_t recomputedHash =
            deterministicHash(cubin.data(), cubinSize);
        if (recomputedHash != cubinHash) {
            AF_ERROR("Module on disk seems to be corrupted", AF_ERR_LOAD_SYM);
        }

        CU_CHECK(cuModuleLoadData(&modOut, cubin.data()));

        AF_TRACE("{{{:<20} : loaded from {} for {} }}", moduleKey, cacheFile,
                 getDeviceProp(device).name);

        retVal.set(modOut);
    } catch (const std::ios_base::failure &e) {
        AF_TRACE("{{{:<20} : Unable to read {} for {}}}", moduleKey, cacheFile,
                 getDeviceProp(device).name);
        removeFile(cacheFile);
    } catch (const AfError &e) {
        if (e.getError() == AF_ERR_LOAD_SYM) {
            AF_TRACE(
                "{{{:<20} : Corrupt binary({}) found on disk for {}, removed}}",
                moduleKey, cacheFile, getDeviceProp(device).name);
        } else {
            if (modOut != nullptr) { CU_CHECK(cuModuleUnload(modOut)); }
            AF_TRACE(
                "{{{:<20} : cuModuleLoadData failed with content from {} for "
                "{}, {}}}",
                moduleKey, cacheFile, getDeviceProp(device).name, e.what());
        }
        removeFile(cacheFile);
    }
    return retVal;
}

arrayfire::cuda::Kernel getKernel(const Module &mod, const string &nameExpr,
                                  const bool sourceWasJIT) {
    std::string name  = (sourceWasJIT ? nameExpr : mod.mangledName(nameExpr));
    CUfunction kernel = nullptr;
    CU_CHECK(cuModuleGetFunction(&kernel, mod.get(), name.c_str()));
    return {nameExpr, mod.get(), kernel};
}

}  // namespace common
}  // namespace arrayfire
