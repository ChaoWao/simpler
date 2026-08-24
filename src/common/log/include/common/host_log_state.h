/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <stdint.h>

#define SIMPLER_HOST_LOG_STATE_ABI_VERSION 2U

/* Matches CallConfig::output_prefix, which is where the path comes from. */
#define SIMPLER_HOST_LOG_DIR_CAPACITY 1024

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Process-owned state shared by the private HostLogger copy compiled into
 * every host-side DSO. The fields are plain integers and fixed char arrays so
 * modules built by different host compiler versions share a C ABI;
 * host_log.cpp performs all scalar accesses with compiler atomic builtins.
 *
 * clock_anchor_pid is positive after a successful anchor write and temporarily
 * negative while one writer owns the claim for that PID. Linux PIDs are
 * positive and bounded well below INT32_MAX.
 *
 * log_directory is where this logger writes, one fully-buffered file per
 * process. It is empty until a caller that knows the run's artifact directory
 * supplies it, and every record goes to stderr while it is. The destination is
 * a property of the logger, so it applies to every record from every caller —
 * there is no per-record or per-call-site routing.
 *
 * The first non-empty path wins: log_directory_bound is release-stored after the
 * path is filled and acquire-loaded before it is read, so a reader sees either no
 * directory or the whole one, and a reader that has already opened the file never
 * has the path change under it.
 */
typedef struct SimplerHostLogState {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t threshold;
    int32_t clock_anchor_pid;
    int32_t log_directory_bound;
    char log_directory[SIMPLER_HOST_LOG_DIR_CAPACITY];
} SimplerHostLogState;

typedef int (*SimplerHostLogBindStateFn)(SimplerHostLogState *state);

/* The only cross-DSO logger symbol. Each consumer exports its own copy and its
 * loader resolves it from that consumer's handle before the module is used. */
int simpler_host_log_bind_state(SimplerHostLogState *state);

#ifdef __cplusplus
}
#endif
