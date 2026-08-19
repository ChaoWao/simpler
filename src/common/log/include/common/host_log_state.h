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

#define SIMPLER_HOST_LOG_STATE_ABI_VERSION 1U

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Process-owned state shared by the private HostLogger copy compiled into
 * every host-side DSO. The fields stay plain integers so modules built by
 * different host compiler versions share a C ABI; host_log.cpp performs all
 * accesses with compiler atomic builtins.
 *
 * clock_anchor_pid is positive after a successful anchor write and temporarily
 * negative while one writer owns the claim for that PID. Linux PIDs are
 * positive and bounded well below INT32_MAX.
 */
typedef struct SimplerHostLogState {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t threshold;
    int32_t clock_anchor_pid;
} SimplerHostLogState;

typedef int (*SimplerHostLogBindStateFn)(SimplerHostLogState *state);

/* The only cross-DSO logger symbol. Each consumer exports its own copy and its
 * loader resolves it from that consumer's handle before the module is used. */
int simpler_host_log_bind_state(SimplerHostLogState *state);

#ifdef __cplusplus
}
#endif
