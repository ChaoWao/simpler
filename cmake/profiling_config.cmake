# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_SIMPLER_PROFILING_CONFIG_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(simpler_configure_profiling target)
    foreach(name IN ITEMS
            SIMPLER_DFX
            SIMPLER_ORCH_PROFILING
            SIMPLER_SCHED_PROFILING
            SIMPLER_TENSORMAP_PROFILING)
        if(DEFINED ${name} AND NOT "${${name}}" MATCHES "^[01]$")
            message(FATAL_ERROR "${name} must be 0 or 1")
        endif()
    endforeach()

    if(NOT DEFINED SIMPLER_DFX)
        set(SIMPLER_DFX 1)
    endif()
    if(NOT DEFINED SIMPLER_ORCH_PROFILING)
        set(SIMPLER_ORCH_PROFILING 0)
    endif()
    if(NOT DEFINED SIMPLER_SCHED_PROFILING)
        set(SIMPLER_SCHED_PROFILING 0)
    endif()
    if(NOT DEFINED SIMPLER_TENSORMAP_PROFILING)
        set(SIMPLER_TENSORMAP_PROFILING 0)
    endif()

    if(SIMPLER_ORCH_PROFILING AND NOT SIMPLER_DFX)
        message(FATAL_ERROR "SIMPLER_ORCH_PROFILING requires SIMPLER_DFX=1")
    endif()
    if(SIMPLER_SCHED_PROFILING AND NOT SIMPLER_DFX)
        message(FATAL_ERROR "SIMPLER_SCHED_PROFILING requires SIMPLER_DFX=1")
    endif()
    if(SIMPLER_TENSORMAP_PROFILING AND NOT SIMPLER_ORCH_PROFILING)
        message(FATAL_ERROR "SIMPLER_TENSORMAP_PROFILING requires SIMPLER_ORCH_PROFILING=1")
    endif()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${generated_dir}")
    configure_file(
        "${_SIMPLER_PROFILING_CONFIG_DIR}/profiling_build_config.h.in"
        "${generated_dir}/simpler_profiling_build_config.h"
        @ONLY
    )
    target_include_directories(${target} PRIVATE "${generated_dir}")
endfunction()
