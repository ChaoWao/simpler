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

/**
 * @file host_tensor_access.h
 * @brief Tensor-byte access for the host orchestrator, over device buffers.
 *
 * `Tensor::buffer.addr` is a device address. host_build_graph runs the
 * orchestrator on the host, so `get_tensor_data` / `set_tensor_data` cannot
 * assume the CPU executing them can load that address — whether it can is a
 * platform capability, not a property of the runtime. This is the seam where
 * that capability is resolved, so the orchestrator core never dereferences a
 * device address itself.
 *
 * A region is registered per staged tensor with the host address serving it:
 *
 *   - `host_view == dev_base`  — the device buffer is mapped into the host
 *     address space (a2a3 `halHostRegister(DEV_SVM_MAP_HOST)`; sim, where a
 *     device pointer is already a host pointer). Reads and writes land on the
 *     device bytes directly.
 *   - `host_view != dev_base`  — no host mapping exists. The region is served
 *     from the staging buffer holding the same bytes, and a write is pushed
 *     back through the device-copy hook so the device observes it.
 *
 * An address no registered region covers is a failure, never a raw
 * dereference: only tensors the runtime staged have a host view at all, so a
 * GM-heap tensor or a pass-through child-memory buffer resolves to nothing.
 *
 * Registrations are valid only for one orchestration run — the window between
 * staging and the first dispatched task. A mirror is a copy, and nothing has
 * executed yet to make it stale; once tasks run, a stale mirror would be
 * indistinguishable from live device memory. `host_tensor_access_reset` bounds
 * that window at both ends.
 *
 * The read/write pair carries weak fallbacks in the runtime translation unit
 * (`orchestrator_core/pto_runtime2.cpp`) that dereference `dev_addr` directly,
 * so the AICPU build — which compiles this path but never runs an
 * orchestrator — resolves without this .cpp. libhost_runtime.so links the
 * strong definitions from `host/host_tensor_access.cpp`.
 */

#ifndef SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_TENSOR_ACCESS_H_
#define SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_TENSOR_ACCESS_H_

#include <stddef.h>
#include <stdint.h>

/**
 * Read `bytes` at device address `dev_addr` into `dst`.
 *
 * @return false when no registered region covers the whole span; `dst` is
 *         untouched.
 */
bool host_tensor_read(uint64_t dev_addr, void *dst, uint64_t bytes);

/**
 * Write `bytes` from `src` to device address `dev_addr`, leaving the bytes
 * visible to the device.
 *
 * @return false when no registered region covers the whole span, or when the
 *         push-back to the device fails.
 */
bool host_tensor_write(uint64_t dev_addr, const void *src, uint64_t bytes);

/**
 * Drop every registration and latch the hook used to push mirror-mode writes
 * to the device (`HostApi::copy_to_device`; null when no write can need one).
 *
 * Called around one orchestration run: before the first
 * `host_tensor_access_add`, and again once the run has finished, after which
 * every access fails until the next run registers its own tensors.
 */
void host_tensor_access_reset(int (*copy_to_device)(void *dev_ptr, const void *host_ptr, size_t size));

/**
 * Register `[dev_base, dev_base + size)` as reachable from the host at
 * `host_view`.
 *
 * @return false for an empty region or a null `host_view`.
 */
bool host_tensor_access_add(uint64_t dev_base, uint64_t size, void *host_view);

#endif  // SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_HOST_TENSOR_ACCESS_H_
