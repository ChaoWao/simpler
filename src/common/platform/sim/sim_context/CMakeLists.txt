# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(cpu_sim_context CXX)

add_library(cpu_sim_context SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu_sim_context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../log/host_log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../log/unified_log_host.cpp"
)

target_include_directories(cpu_sim_context
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    # Host logger and LOG_* headers used by this self-contained consumer.
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../log/include"
)

target_compile_options(cpu_sim_context PRIVATE
    -fPIC
    -O3
    -g
    -Wall
    -Wextra
)
target_compile_features(cpu_sim_context PRIVATE cxx_std_17)

target_link_libraries(cpu_sim_context PRIVATE pthread)


set_target_properties(cpu_sim_context PROPERTIES
    OUTPUT_NAME "cpu_sim_context"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Apply compiler sanitizers to this host-compiled target. No-op unless
# SIMPLER_SANITIZERS is set at configure time, which runtime_compiler only
# passes for host toolchains (sim + onboard host). See cmake/sanitizers.cmake.
if(SIMPLER_SANITIZERS)
    include("${SIMPLER_CMAKE_DIR}/sanitizers.cmake")
    simpler_apply_sanitizers(cpu_sim_context)
endif()
