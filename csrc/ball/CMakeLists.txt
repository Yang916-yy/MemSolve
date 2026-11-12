cmake_minimum_required(VERSION 3.25)

project(memsolve_equilibrium_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MEMSOLVE_USE_MATHDX "Build compact solvers with NVIDIA MathDx cuSOLVERDx" OFF)
option(
    MEMSOLVE_EMBED_PTX
    "Embed PTX on highest architecture for forward JIT compatibility when MEMSOLVE_USE_MATHDX=OFF"
    ON
)

set(
    MEMSOLVE_CUDA_ARCHITECTURES
    ""
    CACHE STRING
    "Semicolon-separated CUDA architectures for the multi-arch MemSolve fatbinary (MEMSOLVE_USE_MATHDX=OFF)"
)
set(
    MEMSOLVE_CUDA_ARCHITECTURE
    ""
    CACHE STRING
    "One CUDA architecture for a strict MemSolve equilibrium extension"
)
set(
    MEMSOLVE_CUDA_OUTPUT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../build/cuda/lib"
    CACHE PATH
    "Directory for the built MemSolve CUDA extension"
)

if(DEFINED CUDAToolkit_ROOT)
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
    set(CUDA_TOOLKIT_ROOT_DIR ${_memsolve_cuda_toolkit_root} CACHE PATH "CUDA toolkit root" FORCE)
endif()
set(CUDA_NVCC_EXECUTABLE ${CMAKE_CUDA_COMPILER} CACHE FILEPATH "CUDA nvcc compiler" FORCE)

if(MEMSOLVE_USE_MATHDX)
    if("${MEMSOLVE_CUDA_ARCHITECTURE}" STREQUAL "")
        set(_memsolve_single_arch "120")
    else()
        set(_memsolve_single_arch "${MEMSOLVE_CUDA_ARCHITECTURE}")
    endif()

    set(_memsolve_supported_architectures 80 86 87 89 90 90a 100 100a 120 120a)
    if(NOT _memsolve_single_arch IN_LIST _memsolve_supported_architectures)
        message(FATAL_ERROR
            "MemSolve MathDx builds support only Ampere-and-newer CUDA architectures: "
            "${_memsolve_supported_architectures}. Got ${_memsolve_single_arch}."
        )
    endif()

    string(REGEX REPLACE "a$" "" _memsolve_numeric_architecture "${_memsolve_single_arch}")
    # Keep PyTorch's imported CUDA flags aligned with this exact device-LTO artifact.
    string(REGEX REPLACE "^([0-9]+)([0-9])(a?)$" "\\1.\\2\\3" _memsolve_torch_cuda_arch "${_memsolve_single_arch}")
    set(ENV{TORCH_CUDA_ARCH_LIST} "${_memsolve_torch_cuda_arch}")

    if(NOT DEFINED CUDAToolkit_ROOT)
        message(FATAL_ERROR "Set CUDAToolkit_ROOT to the CUDA 13.2 toolkit root.")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.2 OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
        message(FATAL_ERROR
            "MemSolve MathDx CUDA builds require CUDA 13.2 exactly; found "
            "${CMAKE_CUDA_COMPILER_VERSION}."
        )
    endif()

    find_package(CUDAToolkit 13.2 EXACT REQUIRED)
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    unset(CMAKE_CUDA_ARCHITECTURES)
    find_package(Torch REQUIRED CONFIG)
    find_package(mathdx 26.06.1 EXACT REQUIRED CONFIG COMPONENTS cusolverdx)

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
        CUDA::cublasLt
        CUDA::cudart
        mathdx::cusolverdx_fatbin
    )

    set(_memsolve_runtime_candidates 80 86 87 89 90 100 120)
    list(FIND _memsolve_runtime_candidates "${_memsolve_numeric_architecture}" _memsolve_sm_bit)
    math(EXPR _memsolve_supported_sm_mask "1 << ${_memsolve_sm_bit}")

    target_compile_options(memsolve_equilibrium PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_definitions(
        memsolve_equilibrium
        PRIVATE
            MEMSOLVE_USE_MATHDX=1
            MEMSOLVE_CUDA_TARGET_SM=${_memsolve_numeric_architecture}0
            MEMSOLVE_SUPPORTED_SM_MASK=${_memsolve_supported_sm_mask}
    )

    set_target_properties(
        memsolve_equilibrium
        PROPERTIES
            CUDA_ARCHITECTURES "${_memsolve_single_arch}"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            INTERPROCEDURAL_OPTIMIZATION ON
            LIBRARY_OUTPUT_DIRECTORY "${MEMSOLVE_CUDA_OUTPUT_DIR}"
            PREFIX ""
            OUTPUT_NAME "memsolve_equilibrium_sm${_memsolve_numeric_architecture}"
    )
else()
    if(NOT "${MEMSOLVE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(_memsolve_arch_list ${MEMSOLVE_CUDA_ARCHITECTURES})
    elseif(NOT "${MEMSOLVE_CUDA_ARCHITECTURE}" STREQUAL "")
        set(_memsolve_arch_list ${MEMSOLVE_CUDA_ARCHITECTURE})
    else()
        set(_memsolve_arch_list 80 90a)
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR
            "MemSolve CUDA builds require CUDA 12.0 or newer; found "
            "${CMAKE_CUDA_COMPILER_VERSION}."
        )
    endif()

    set(_memsolve_supported_architectures 80 86 87 89 90 90a 100 100a 103 103a 110 120 120a)
    set(_memsolve_cmake_cuda_archs "")
    set(_memsolve_numeric_archs "")
    set(_memsolve_max_numeric_arch 0)

    foreach(_arch IN LISTS _memsolve_arch_list)
        if(NOT _arch IN_LIST _memsolve_supported_architectures)
            message(FATAL_ERROR
                "MemSolve supports only Ampere-and-newer CUDA architectures: "
                "${_memsolve_supported_architectures}. Got ${_arch}."
            )
        endif()
        string(REGEX REPLACE "[af]$" "" _num_arch "${_arch}")
        list(APPEND _memsolve_cmake_cuda_archs "${_arch}-real")
        list(APPEND _memsolve_numeric_archs "${_num_arch}")
        if(_num_arch GREATER _memsolve_max_numeric_arch)
            set(_memsolve_max_numeric_arch "${_num_arch}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _memsolve_numeric_archs)
    if(MEMSOLVE_EMBED_PTX AND _memsolve_max_numeric_arch GREATER 0)
        list(APPEND _memsolve_cmake_cuda_archs "${_memsolve_max_numeric_arch}-virtual")
    endif()
    list(REMOVE_DUPLICATES _memsolve_cmake_cuda_archs)

    # Prevent TorchConfig.cmake from probing GPUs or injecting duplicate -gencode flags;
    # target-level CUDA_ARCHITECTURES below is the single source of truth.
    set(ENV{TORCH_CUDA_ARCH_LIST} "8.0")
    find_package(CUDAToolkit REQUIRED)
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    unset(CMAKE_CUDA_ARCHITECTURES)
    set(_memsolve_saved_cuda_flags "${CMAKE_CUDA_FLAGS}")
    find_package(Torch REQUIRED CONFIG)
    set(CMAKE_CUDA_FLAGS "${_memsolve_saved_cuda_flags}")

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
        CUDA::cublasLt
        CUDA::cudart
    )

    set(_memsolve_runtime_candidates 80 86 87 89 90 100 120)
    set(_memsolve_supported_sm_mask 0)
    set(_memsolve_cand_bit 0)
    foreach(_cand IN LISTS _memsolve_runtime_candidates)
        set(_cand_supported OFF)
        if(MEMSOLVE_EMBED_PTX AND _memsolve_max_numeric_arch GREATER 0 AND NOT _cand LESS _memsolve_max_numeric_arch)
            set(_cand_supported ON)
        endif()
        math(EXPR _cand_major "${_cand} / 10")
        foreach(_arch IN LISTS _memsolve_arch_list)
            string(REGEX REPLACE "[af]$" "" _raw_num "${_arch}")
            if(_raw_num STREQUAL "103" OR _raw_num STREQUAL "110")
                set(_norm_num "100")
            else()
                set(_norm_num "${_raw_num}")
            endif()
            if(_arch MATCHES "[af]$")
                if(_cand EQUAL _norm_num)
                    set(_cand_supported ON)
                endif()
            else()
                math(EXPR _raw_major "${_raw_num} / 10")
                if(_cand EQUAL _norm_num OR (_cand_major EQUAL _raw_major AND NOT _cand LESS _raw_num))
                    set(_cand_supported ON)
                endif()
            endif()
        endforeach()
        if(_cand_supported)
            math(EXPR _memsolve_supported_sm_mask "${_memsolve_supported_sm_mask} | (1 << ${_memsolve_cand_bit})")
        endif()
        math(EXPR _memsolve_cand_bit "${_memsolve_cand_bit} + 1")
    endforeach()

    target_compile_options(memsolve_equilibrium PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_definitions(
        memsolve_equilibrium
        PRIVATE
            MEMSOLVE_USE_MATHDX=0
            MEMSOLVE_CUDA_TARGET_SM=0
            MEMSOLVE_SUPPORTED_SM_MASK=${_memsolve_supported_sm_mask}
    )

    set_target_properties(
        memsolve_equilibrium
        PROPERTIES
            CUDA_ARCHITECTURES "${_memsolve_cmake_cuda_archs}"
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS OFF
            INTERPROCEDURAL_OPTIMIZATION OFF
            LIBRARY_OUTPUT_DIRECTORY "${MEMSOLVE_CUDA_OUTPUT_DIR}"
            PREFIX ""
            OUTPUT_NAME "memsolve_equilibrium"
    )

    # Create hardlinks (falling back to copy) for memsolve_equilibrium_sm<arch>.so
    # so existing per-SM lookup (including Path.resolve() name checks) works seamlessly.
    foreach(_num_arch IN LISTS _memsolve_numeric_archs)
        add_custom_command(
            TARGET memsolve_equilibrium
            POST_BUILD
            COMMAND sh -c
                "rm -f \"$2\" && (ln \"$1\" \"$2\" 2>/dev/null || cp -f \"$1\" \"$2\")"
                _
                "$<TARGET_FILE:memsolve_equilibrium>"
                "${MEMSOLVE_CUDA_OUTPUT_DIR}/memsolve_equilibrium_sm${_num_arch}.so"
            VERBATIM
        )
    endforeach()
endif()
