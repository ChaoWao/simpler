# Standalone L2 orchestrator package: bench + profile for the four-line
# orchestration sequence in runtime_maker.cpp:538-541. Host-only, no CANN,
# no Ascend SDK, no NPU.
#
# The TU list mirrors the `host` target of
# src/a2a3/runtime/host_build_graph/build_config.py, minus `host/` itself
# (that is where runtime_maker and its CANN dependencies live):
#
#   BUILD_CONFIG["host"]["source_dirs"] =
#       ["host", "runtime/orchestrator_core", "runtime/shared", "orchestration"]
#
# scripts/check_extraction.sh asserts this list still matches the source repo.

cmake_minimum_required(VERSION 3.15)
# C is required, and not incidentally: qwen3_dynamic_tensormap.h is C and cannot
# be compiled as C++ (C99 compound-literal lvalues; see esl_shim/esl_c_abi.h).
project(l2_orchestrator CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(WIN32)
    message(FATAL_ERROR "This package requires POSIX. Build on Linux.")
endif()

# Neither simpler nor this package sets a build type by default, which would
# leave the bench measuring an -O0 engine. Pin it: an unoptimised orchestrator
# is not the thing anyone wants a number for.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
# Engine TUs are compiled straight out of the enclosing checkout, not copied:
# the bench measures the orchestrator of THIS revision, and an edit to
# ../src is picked up by the next build with nothing to keep in sync.
set(S ${ROOT}/../src)
set(R ${S}/a2a3/runtime/host_build_graph)

# SIMPLER_ORCH_PROFILING gates the engine's own per-STEP cycle counters inside
# submit_task_common (Level 3 of the profile report). It requires SIMPLER_DFX=1,
# and both add work to the measured path — so the default is off and the
# throughput numbers come from the clean build.
option(L2_ORCH_PROFILING "Build the engine's own per-STEP cycle counters (Level 3)" OFF)
# SIMPLER_TENSORMAP_PROFILING adds the engine's own TensorMap lookup counters
# (bucket chain length, overlap checks/hits, insert count) — Level 4 of the
# report. profiling_config.h #errors unless SIMPLER_ORCH_PROFILING is also on,
# so this implies it.
option(L2_TENSORMAP_PROFILING "Build the engine's TensorMap lookup counters (Level 4)" OFF)

add_library(l2_engine STATIC
    ${R}/runtime/orchestrator_core/pto_orchestrator.cpp
    ${R}/runtime/orchestrator_core/pto_ring_buffer.cpp
    ${R}/runtime/orchestrator_core/pto_runtime2.cpp
    ${R}/runtime/shared/pto_runtime2_init.cpp
    ${R}/runtime/shared/pto_shared_memory.cpp
    ${R}/runtime/shared/pto_tensormap.cpp
    ${R}/runtime/shared/runtime.cpp
    ${R}/orchestration/common.cpp
    ${ROOT}/src/host_shim/host_shim.cpp
)
target_include_directories(l2_engine PUBLIC
    ${R}/runtime
    ${R}/common
    ${R}/orchestration
    ${S}/a2a3/runtime
    ${S}
    ${S}/common
    ${S}/common/log/include
    ${S}/common/task_interface
    ${S}/common/platform/include
    ${S}/a2a3/platform/include
)
if(L2_TENSORMAP_PROFILING)
    set(L2_ORCH_PROFILING ON)
endif()
if(L2_ORCH_PROFILING)
    target_compile_definitions(l2_engine PUBLIC SIMPLER_DFX=1 SIMPLER_ORCH_PROFILING=1)
    message(STATUS "SIMPLER_ORCH_PROFILING=1 — Level 3 built; this build is NOT the one to quote throughput from")
endif()
if(L2_TENSORMAP_PROFILING)
    target_compile_definitions(l2_engine PUBLIC SIMPLER_TENSORMAP_PROFILING=1)
    message(STATUS "SIMPLER_TENSORMAP_PROFILING=1 — Level 4 (TensorMap chain stats) built")
endif()

# Where qwen3_dynamic_tensormap.h lives. Defaults to this package's own
# directory; override to point elsewhere.
set(L2_QWEN3_DYN_CASE_DIR "${ROOT}" CACHE PATH
    "Directory containing qwen3_dynamic_tensormap.h")
if(NOT EXISTS "${L2_QWEN3_DYN_CASE_DIR}/qwen3_dynamic_tensormap.h")
    message(FATAL_ERROR
        "qwen3_dynamic_tensormap.h not found in ${L2_QWEN3_DYN_CASE_DIR}.\n"
        "Pass -DL2_QWEN3_DYN_CASE_DIR=<dir> to point at it.")
endif()

# The payload is separate from the engine so a bench-side change cannot reach
# the engine TUs.
add_library(l2_payloads STATIC
    ${ROOT}/bench/payload_qwen3_dyn.cpp
    ${ROOT}/bench/payload_qwen3_dyn_case.c      # C, on purpose — see below
    ${ROOT}/bench/esl_shim/esl_shim_impl.cpp
)
target_include_directories(l2_payloads PUBLIC ${ROOT}/bench)
target_link_libraries(l2_payloads PUBLIC l2_engine)

# esl_shim/ is deliberately NOT on any target-wide include path: it provides
# headers named `mem_pool.h` and `tensormap.h`, and only the two TUs below may
# see them. Everything else in the package must keep resolving those names the
# way it does today (i.e. not at all).
set_source_files_properties(${ROOT}/bench/esl_shim/esl_shim_impl.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${ROOT}/bench/esl_shim"
)

# The case's SPMD tier is a compile-time knob of the case itself:
#
#     #ifndef QWEN3_SPMD_TIER
#     #define QWEN3_SPMD_TIER 4
#     #endif
#
# and it changes the DAG, not just its scheduling. `qwen3_blocks_per_task()`
# returns min(total_chunks, {1,2,4,8,1<<30}[tier]), so the tier decides how many
# SPMD chunks are folded into ONE task. This package pins tier 0 — one chunk per
# task, 3096 tasks / 3096 subtasks, the most task-dense variant and therefore the
# one that puts the most pressure on the orchestrator per unit of device work.
# The #ifndef guard in the case exists precisely so the build can set it, which
# is why pinning it here does not modify the case file.
#
# The entry symbol is renamed at the same time so the case's C-linkage
# `aicpu_orchestration_entry` cannot collide with the engine-side name.
set_source_files_properties(${ROOT}/bench/payload_qwen3_dyn_case.c PROPERTIES
    INCLUDE_DIRECTORIES "${L2_QWEN3_DYN_CASE_DIR};${ROOT}/bench/esl_shim"
    COMPILE_DEFINITIONS "aicpu_orchestration_entry=qwen3_dyn_orchestration_entry;QWEN3_SPMD_TIER=0"
)

add_executable(l2_bench ${ROOT}/bench/l2_bench.cpp ${ROOT}/bench/l2_profile.cpp)
target_link_libraries(l2_bench PRIVATE l2_payloads)

add_executable(l2_orch_main ${ROOT}/apps/l2_orch_main.cpp)
target_include_directories(l2_orch_main PRIVATE ${ROOT}/bench)
target_link_libraries(l2_orch_main PRIVATE l2_payloads)

enable_testing()
add_test(NAME smoke COMMAND l2_orch_main)
# Both entry points are covered because the throughput path and the profile path
# drive the entry differently (the latter through the replacement ops table).
add_test(NAME bench COMMAND l2_bench --mode=throughput --repeat=1)
add_test(NAME profile COMMAND l2_bench --mode=profile)
# The prefault diagnostics touch the SM and arena before the engine initialises
# them, so a regression there would corrupt the run rather than just skew a
# number. Cover both, and assert the graph still comes out the same size.
add_test(NAME prefault COMMAND l2_bench --mode=throughput --repeat=1 --prefault-all)
set_tests_properties(prefault PROPERTIES
    PASS_REGULAR_EXPRESSION "kernel submits +3096 +framework allocs +779")
