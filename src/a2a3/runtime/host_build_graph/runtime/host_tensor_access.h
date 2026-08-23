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
 * @brief ChipTensor-byte access for the host orchestrator, over device buffers.
 *
 * `ChipTensor::buffer.addr` is a device address. host_build_graph runs the
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
 * Registrations are owned by one orchestration run — the window between
 * staging and the first dispatched task. A mirror is a copy, and nothing has
 * executed yet to make it stale; once tasks run, a stale mirror would be
 * indistinguishable from live device memory. `HostTensorAccessor` bounds that
 * window and releases its mappings on every exit path.
 *
 * The read/write pair carries weak fallbacks in the runtime translation unit
 * (`orchestrator_core/runtime_core.cpp`) that dereference `dev_addr` directly,
 * so the AICPU build — which compiles this path but never runs an
 * orchestrator — resolves without this .cpp. libhost_runtime.so links the
 * strong definitions from `host/host_tensor_access.cpp`.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

struct HostApi;  // common/host_api.h — fwd-declared so this header stays out of platform includes

/**
 * The registered regions of one orchestration run, and the mappings that run
 * installed to serve them.
 *
 * One accessor per run, mutated only by the thread running that run's
 * orchestration. The region and mapping tables are plain vectors with no lock,
 * so concurrent `add` / `close` on one accessor is a data race; concurrent runs
 * are isolated by each owning a separate accessor, which is what makes two runs
 * unable to see or drop each other's regions.
 *
 * `add` is the only producer of mappings and the only caller of
 * `register_device_memory_to_host`; `close` unregisters exactly the mappings
 * this accessor installed and nothing else. Both are reached on every return
 * path — `close` is idempotent and the destructor calls it — so a mapping
 * cannot outlive the run that made it.
 *
 * A null `api` makes every `add` fail, so a registered region always implies a
 * usable `api`; `write`'s mirror push-back relies on that and does not re-check.
 *
 * The state lives behind `Impl` because this header is also compiled by the
 * AICPU build (through `orchestrator_core/runtime_core.cpp`, which resolves the
 * weak read/write fallbacks below), and that build has no `<vector>`. Keep the
 * standard containers in `host/host_tensor_access.cpp`.
 */
class HostTensorAccessor {
public:
    explicit HostTensorAccessor(const HostApi *api);
    ~HostTensorAccessor();

    HostTensorAccessor(const HostTensorAccessor &) = delete;
    HostTensorAccessor &operator=(const HostTensorAccessor &) = delete;

    /**
     * Register `[dev_base, dev_base + size)`, preferring a host mapping of the
     * device buffer and falling back to `fallback_host_view` (the staging copy)
     * when the platform cannot map it.
     *
     * @return false for an empty region, a null `api`, or when neither a
     *         mapping nor a fallback view is available.
     */
    bool add(uint64_t dev_base, uint64_t size, void *fallback_host_view);
    bool read(uint64_t dev_addr, void *dst, uint64_t bytes) const;
    bool write(uint64_t dev_addr, const void *src, uint64_t bytes) const;

    /** Drop every region and unregister every mapping this accessor installed. */
    void close() noexcept;

    /** Mappings installed by `add` and not yet dropped by `close`. */
    size_t mapping_count() const noexcept;

    /** Total bytes covered by those mappings; excludes fallback staging views. */
    uint64_t mapped_bytes() const noexcept;

private:
    struct Impl;
    Impl *impl_;
};

/**
 * Read `bytes` at device address `dev_addr` into `dst`.
 *
 * @return false when no registered region covers the whole span; `dst` is
 *         untouched.
 */
bool host_tensor_read(HostTensorAccessor *accessor, uint64_t dev_addr, void *dst, uint64_t bytes);

/**
 * Write `bytes` from `src` to device address `dev_addr`, leaving the bytes
 * visible to the device.
 *
 * @return false when no registered region covers the whole span, or when the
 *         push-back to the device fails.
 */
bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes);
