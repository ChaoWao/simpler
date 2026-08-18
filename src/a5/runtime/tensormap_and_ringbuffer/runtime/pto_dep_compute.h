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
 * @file pto_dep_compute.h
 * @brief Dependency computation primitives shared by runtime submit_task and dep_gen replay.
 *
 * Two header-only template entry points:
 *
 *   compute_task_fanin     — STEP 3 in submit_task: per-tensor creator retention (Step A)
 *                            + tensormap.lookup for INPUT/TRACKED_INPUT/INOUT (Step B). Calls back into
 *                            user-supplied `emit` for each producer it identifies.
 *
 *   register_task_accesses — STEP 4 in submit_task: register TRACKED_INPUT readers and
 *                            INOUT/OUTPUT_EXISTING writers. No callbacks.
 *
 * STEP 1 (explicit_deps) is intentionally left at the runtime call site because its
 * `last_task_alive` shortcut + unchecked slot lookup is subtly different from the
 * `slot_state->task->task_id == producer` reuse check in STEP 3. Unifying them would
 * require two emit semantics or a marginal behavior change in transients — not worth
 * the minor structural overlap. Replay handles STEP 1 with a one-line loop of its own.
 *
 * The Emit callback contract:
 *   bool emit(PTO2TaskId producer, DepFlags kind);
 *     - kind is DEP_WAIT|DEP_RETAIN for a Step-A creator edge (the consumer reads
 *       the producer's allocated buffer, so the producer is retained) and DEP_WAIT
 *       for a Step-B modifier edge (ordering only; the buffer was allocated
 *       elsewhere). Duplicate producers OR-accumulate their flags.
 *     - return true to continue (whether or not the producer was actually recorded —
 *       producer-not-alive / dedup-hit / etc. all return true silently)
 *     - return false to signal fatal (e.g. fanin spill overflow); caller bails
 *
 * Performance: Emit is a template parameter, not std::function. Both runtime
 * (lambda capturing fanin_builder + sm_header) and replay (lambda capturing edge
 * vector) instantiate at the call site and inline through. Do NOT replace with
 * std::function — it would break the inlining and add ~5 ns/call to the orch hot path.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "pto_task_id.h"
#include "pto_tensormap.h"
#include "pto_types.h"  // TensorRef
#include "tensor.h"

/**
 * View struct for inputs to compute_task_fanin / register_task_accesses.
 *
 * Both runtime and replay assemble one of these from their own data sources
 * (runtime: from Arg accessors; replay: from SubmitTraceEntry fields). All
 * pointer arrays must remain valid for the duration of the call.
 */
struct DepInputs {
    int32_t tensor_count;
    const TensorRef *tensors;        // length = tensor_count (union; OUTPUT slots' .ptr is unused)
    const TensorArgType *arg_types;  // length = tensor_count
    int32_t explicit_dep_count;
    const PTO2TaskId *explicit_deps;  // length = explicit_dep_count (validity checked by caller)
};

template <typename Emit>
[[nodiscard]] inline bool
emit_writer_dependency(Emit &emit, PTO2TaskId task_id, DepFlags flags) {
    if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, DepFlags>) {
        return emit(task_id, flags);
    } else if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, DepFlags, TensorAccessKind>) {
        return emit(task_id, flags, TensorAccessKind::WRITER);
    } else if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, TensorAccessKind>) {
        return emit(task_id, TensorAccessKind::WRITER);
    } else {
        return emit(task_id);
    }
}

template <typename Emit>
[[nodiscard]] inline bool emit_reader_dependency(Emit &emit, PTO2TaskId task_id, DepFlags flags) {
    if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, DepFlags>) {
        return emit(task_id, flags);
    } else if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, DepFlags, TensorAccessKind>) {
        return emit(task_id, flags, TensorAccessKind::READER);
    } else if constexpr (std::is_invocable_r_v<bool, Emit &, PTO2TaskId, TensorAccessKind>) {
        return emit(task_id, TensorAccessKind::READER);
    } else {
        return emit(task_id);
    }
}

template <typename ReaderEmit>
[[nodiscard]] __attribute__((noinline, cold)) bool
compute_reader_fanin(const DepInputs &inputs, PTO2TensorMap &tensor_map, ReaderEmit &reader_emit) {
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype != TensorArgType::INOUT && ptype != TensorArgType::OUTPUT_EXISTING) {
            continue;
        }
        const ChipTensor *tensor = &inputs.tensors[i].ref();
        if (tensor->manual_dep) {
            continue;
        }
        bool fatal = false;
        tensor_map.lookup(
            *tensor, TensorAccessKind::READER,
            [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
                if (!emit_reader_dependency(reader_emit, entry.access_task_id, DEP_WAIT)) {
                    fatal = true;
                    return false;
                }
                if (overlap_status == OverlapStatus::COVERED) {
                    tensor_map.remove_reader_entry(entry);
                }
                return true;
            }
        );
        if (fatal) {
            return false;
        }
    }
    return true;
}

/**
 * Compute fanin for a task being submitted (STEP 3: Step A creator retention +
 * Step B tensormap modifier lookup).
 *
 * For each non-OUTPUT tensor:
 *   - If owner_task_id is valid, emit(owner)
 *   - For INPUT/TRACKED_INPUT/INOUT (and not manual_dep), tensor_map.lookup(*tensor) and emit
 *     each matching producer. INOUT+COVERED triggers tensor_map.remove_entry(entry).
 *
 * @return true on success (or producer-skipped-silently); false if emit signaled
 *         fatal — caller should propagate (after any fatal bookkeeping done by emit).
 */
template <typename Emit, typename ReaderEmit>
[[nodiscard]] inline bool
compute_task_fanin(
    const DepInputs &inputs, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit, ReaderEmit reader_emit
) {
    if (in_manual_scope) {
        return true;
    }

    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since
            // they have no dependencies.
            continue;
        }

        const ChipTensor *tensor = &inputs.tensors[i].ref();

        // Step A: creator retention — reading a tensor retains its allocator, so
        // the creator edge carries both ordering and lifetime.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit_writer_dependency(emit, owner, DEP_WAIT | DEP_RETAIN)) {
                return false;
            }
        }

        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT &&
            ptype != TensorArgType::TRACKED_INPUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit_writer_dependency(emit, entry.access_task_id, DEP_WAIT)) {
                fatal = true;
                return false;
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                tensor_map.remove_writer_entry(entry);
            }
            return true;
        });
        if (fatal) {
            return false;
        }
    }
    return tensor_map.current_readers() == 0 || compute_reader_fanin(inputs, tensor_map, reader_emit);
}

template <typename Emit>
[[nodiscard]] inline bool
compute_task_fanin(const DepInputs &inputs, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit) {
    return compute_task_fanin(inputs, tensor_map, in_manual_scope, emit, emit);
}

/**
 * Register a task's accesses in the tensormap (STEP 4 in submit_task).
 *
 * No-op when in_manual_scope.
 */
inline void
register_task_accesses(const DepInputs &inputs, PTO2TaskId task_id, PTO2TensorMap &tensor_map, bool in_manual_scope) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const ChipTensor *tensor = &inputs.tensors[i].ref();
            if (!tensor->manual_dep) {
                tensor_map.insert(*tensor, task_id);
            }
        } else if (ptype == TensorArgType::TRACKED_INPUT) {
            const ChipTensor *tensor = &inputs.tensors[i].ref();
            if (!tensor->manual_dep) {
                tensor_map.insert(*tensor, task_id, TensorAccessKind::READER);
            }
        }
    }
}

/**
 * Count the tensormap entries register_task_accesses() will insert for this task.
 *
 * Mirrors register_task_accesses() exactly, so the returned value is the precise number of
 * new_entry() calls that step makes. The orchestrator uses it to reserve pool
 * capacity before inserting. Returns 0 in a manual scope (no registration).
 */
inline int32_t count_registrable_accesses(const DepInputs &inputs, bool in_manual_scope) {
    if (in_manual_scope) {
        return 0;
    }
    int32_t needed = 0;
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING ||
            ptype == TensorArgType::TRACKED_INPUT) {
            if (!inputs.tensors[i].ref().manual_dep) {
                needed++;
            }
        }
    }
    return needed;
}
