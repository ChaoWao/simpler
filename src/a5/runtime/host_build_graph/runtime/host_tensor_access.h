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
 * (`orchestrator_core/pto_runtime2.cpp`) that dereference `dev_addr` directly,
 * so the AICPU build — which compiles this path but never runs an
 * orchestrator — resolves without this .cpp. libhost_runtime.so links the
 * strong definitions from `host/host_tensor_access.cpp`.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

struct HostApi;  // common/host_api.h — fwd-declared so this header stays out of platform includes

// What a region backs, and therefore which lookups may resolve to it. The two
// kinds never overlap: a staged tensor is its own device allocation and the GM
// heap is another, so a device address belongs to at most one region.
enum class HostTensorRegionKind : uint8_t {
    StagedTensor,
    GmHeap,
};

// Why a fill did not happen. The cases are distinct enough to lead a reader in
// different directions — a bad element width is a malformed create-info, an
// unresolved span is an address outside the heap, and a failed push is a device
// copy error — so the caller reports which one occurred rather than guessing.
//
// A region without a host view is not among them: that is the ordinary state on
// a platform with no host-map path, where `fill` stages the bytes and pushes
// them, and the outcome is an ordinary Ok or PushFailed.
enum class HostTensorFillStatus : uint8_t {
    Ok,
    BadElementSize,
    NoRegion,
    PushFailed,
};

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
 * `add` and `add_heap` are the only producers of mappings and the only callers
 * of `register_device_memory_to_host`; `close` unregisters exactly the mappings
 * this accessor installed and nothing else. Both are reached on every return
 * path — `close` is idempotent and the destructor calls it — so a mapping
 * cannot outlive the run that made it.
 *
 * A null `api` makes every registration fail, so a registered region always
 * implies a usable `api`; `write`'s mirror push-back relies on that and does
 * not re-check.
 *
 * Regions come in two kinds and the lookups are kind-scoped. `read` / `write`
 * serve the orchestration scalar-access API and see staged-tensor regions only,
 * so a device address outside a staged tensor — a GM-heap buffer, a
 * pass-through child-memory buffer — resolves to nothing there, which is what
 * makes runtime-created outputs unreadable during host orchestration. `fill`
 * serves `TensorCreateInfo::set_initial_value`, whose target is always a
 * runtime allocation, and sees GM-heap regions only.
 *
 * The state lives behind `Impl` because this header is also compiled by the
 * AICPU build (through `orchestrator_core/pto_runtime2.cpp`, which resolves the
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

    /**
     * Register the GM heap `[dev_base, dev_base + size)` as the region serving
     * runtime allocations, reachable through `fill` alone.
     *
     * Recording the span is all this does. Mapping it is deferred to the first
     * `fill` that needs one, because the heap runs to hundreds of MB (2 GiB on
     * the dsv4 case) and a run that sets no initial value never reads or writes
     * it — paying `halHostRegister` over that span at every bind would buy
     * nothing. Where no host-map path exists (a5 onboard, whose
     * `DeviceRunnerBase` default returns nullptr) the mapping stays absent and
     * `fill` stages the bytes and pushes them with `copy_to_device` instead.
     */
    bool add_heap(uint64_t dev_base, uint64_t size);

    bool read(uint64_t dev_addr, void *dst, uint64_t bytes) const;
    bool write(uint64_t dev_addr, const void *src, uint64_t bytes) const;

    /**
     * Fill `[dev_addr, dev_addr + bytes)` with `elem_size`-wide `value`,
     * leaving the bytes visible to the device. `bytes` need not be a multiple
     * of `elem_size`; the tail is a partial element.
     *
     * Maps the GM heap on the first call that needs it, so a run that sets no
     * initial value never pays for a mapping of the whole heap.
     *
     * The write lands immediately, never deferred to a flush: the GM heap is
     * reclaimed and re-let within one orchestration, so two fills can name the
     * same address and only the order they were issued in is correct.
     */
    HostTensorFillStatus fill(uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size) const;

    /** Drop every region and unregister every mapping this accessor installed. */
    void close() noexcept;

    /** Mappings installed by `add` and not yet dropped by `close`. */
    size_t mapping_count() const noexcept;

    /** Total bytes covered by those mappings; excludes fallback staging views. */
    uint64_t mapped_bytes() const noexcept;

private:
    bool add_region(uint64_t dev_base, uint64_t size, void *fallback_host_view, HostTensorRegionKind kind);
    void ensure_region_mapped(struct HostTensorRegion *region) const;
    bool fill_staged(uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size) const;

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
 * @return false when no staged-tensor region covers the whole span, or when the
 *         push-back to the device fails.
 */
bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes);

/**
 * Fill `bytes` at device address `dev_addr` with `elem_size`-wide `value`,
 * leaving the bytes visible to the device.
 *
 * @return what stopped the fill, or `Ok`.
 */
HostTensorFillStatus
host_tensor_fill(HostTensorAccessor *accessor, uint64_t dev_addr, uint64_t bytes, uint64_t value, uint64_t elem_size);

/** Short, stable label for `status`, for a diagnostic naming what failed. */
const char *host_tensor_fill_status_name(HostTensorFillStatus status);
