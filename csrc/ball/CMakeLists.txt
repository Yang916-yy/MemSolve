cmake_minimum_required(VERSION 3.25)

project(memsolve_equilibrium_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(
    MEMSOLVE_CUDA_ARCHITECTURE
    "120"
    CACHE STRING
    "One CUDA architecture for a strict MemSolve equilibrium extension"
)
set(
    MEMSOLVE_CUDA_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../build/cuda/lib"
    CACHE PATH
    "Directory for the built MemSolve CUDA extension"
)

set(_memsolve_supported_architectures 80 86 87 89 90 100 120)
if(NOT MEMSOLVE_CUDA_ARCHITECTURE IN_LIST _memsolve_supported_architectures)
    message(FATAL_ERROR
        "MemSolve supports only Ampere-and-newer CUDA architectures: "
        "${_memsolve_supported_architectures}. Got ${MEMSOLVE_CUDA_ARCHITECTURE}."
    )
endif()

# Keep PyTorch's imported CUDA flags aligned with this exact device-LTO artifact.
string(REGEX REPLACE "^([0-9]+)([0-9])$" "\\1.\\2" _memsolve_torch_cuda_arch "${MEMSOLVE_CUDA_ARCHITECTURE}")
set(ENV{TORCH_CUDA_ARCH_LIST} "${_memsolve_torch_cuda_arch}")

if(NOT DEFINED CUDAToolkit_ROOT)
    message(FATAL_ERROR "Set CUDAToolkit_ROOT to the CUDA 13.2 toolkit root.")
endif()

file(REAL_PATH "${CUDAToolkit_ROOT}" _memsolve_cuda_toolkit_root)
file(REAL_PATH "${CMAKE_CUDA_COMPILER}" _memsolve_cuda_compiler)
get_filename_component(_memsolve_cuda_compiler_root "${_memsolve_cuda_compiler}" DIRECTORY)
get_filename_component(_memsolve_cuda_compiler_root "${_memsolve_cuda_compiler_root}" DIRECTORY)

if(NOT _memsolve_cuda_compiler_root STREQUAL _memsolve_cuda_toolkit_root)
    message(FATAL_ERROR
        "CMAKE_CUDA_COMPILER (${_memsolve_cuda_compiler}) is not from "
        "CUDAToolkit_ROOT (${_memsolve_cuda_toolkit_root})."
    )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.2 OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
    message(FATAL_ERROR
        "MemSolve CUDA builds require CUDA 13.2 exactly; found "
        "${CMAKE_CUDA_COMPILER_VERSION}."
    )
endif()

find_package(CUDAToolkit 13.2 EXACT REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR ${_memsolve_cuda_toolkit_root} CACHE PATH CUDA_13_2_root FORCE)
set(CUDA_NVCC_EXECUTABLE ${CMAKE_CUDA_COMPILER} CACHE FILEPATH CUDA_13_2_nvcc FORCE)
# PyTorch configures its own CUDA flags from TORCH_CUDA_ARCH_LIST above. The
# target-level CUDA_ARCHITECTURES property below remains the source of truth.
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
unset(CMAKE_CUDA_ARCHITECTURES)
find_package(Torch REQUIRED CONFIG)
find_package(mathdx 26.06.1 EXACT REQUIRED CONFIG COMPONENTS cublasdx cusolverdx)

if(NOT TARGET mathdx::cusolverdx_fatbin)
    message(FATAL_ERROR
        "MathDx cuSolverDx fatbin support requires CUDA 13.2 and was not found."
    )
endif()

add_library(
    memsolve_equilibrium
    SHARED
    binding.cpp
    forward.cu
)

target_link_libraries(
    memsolve_equilibrium
    PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    mathdx::cusolverdx_fatbin
)

target_compile_options(memsolve_equilibrium PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(
    memsolve_equilibrium
    # CMake uses 75/120, while MathDx and cudaDeviceProp use 750/1200.
    PRIVATE MEMSOLVE_CUDA_TARGET_SM=${MEMSOLVE_CUDA_ARCHITECTURE}0
)

set_target_properties(
    memsolve_equilibrium
    PROPERTIES
        CUDA_ARCHITECTURES "${MEMSOLVE_CUDA_ARCHITECTURE}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ON
        LIBRARY_OUTPUT_DIRECTORY "${MEMSOLVE_CUDA_OUTPUT_DIR}"
        PREFIX ""
        OUTPUT_NAME "memsolve_equilibrium_sm${MEMSOLVE_CUDA_ARCHITECTURE}"
)
