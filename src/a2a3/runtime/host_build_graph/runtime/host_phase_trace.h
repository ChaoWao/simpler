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

/**
 * Timing of this runtime's prepare path: the bind stage's segments and the host
 * orchestrator's submit-level operations.
 *
 * One record call feeds two independent sinks, and the split is what lets each
 * survive without the other:
 *
 *   - **Per-kind counters**, owned here. They produce the `LOG_TIMING`
 *     breakdown and are exact whether or not records are being kept, so they are
 *     the channel that works at `--rounds > 1`, where the artifact collectors are
 *     switched off entirely.
 *   - **The platform's host phase pool**, when a per-event reader is enabled.
 *     Records go straight into platform-allocated memory (see
 *     host/host_phase_records.h); this runtime holds only the pool pointer.
 *
 * Timestamps are host monotonic nanoseconds — the clock the `[STRACE]` host
 * spans use — so records nest inside `chip.run.bind` with no alignment step
 * and share the enclosing run's (pid, inv) identity.
 */

/**
 * Whether this runtime asks for its own prepare-path breakdown.
 *
 * True only when `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` starts with `1`, `t` or
 * `T`, read once on first use. This is the producer's own condition; the runner
 * ORs it with the chip-swimlane level when deciding whether to arm the pool, so
 * a level-4 run collects records with the variable unset, and this variable
 * collects them with the level at 0.
 *
 * "Breakdown" rather than "timing" because the bind stage's *duration* is
 * already published as the `chip.run.bind` `[STRACE]` marker; what this adds
 * is the division of that one number into its parts.
 */
bool host_phase_timing_enabled();

/**
 * Host monotonic nanoseconds, or 0 when this pass records nothing — which is
 * what keeps the orchestrator's per-operation hooks down to two calls returning
 * a constant.
 */
uint64_t host_phase_now_ns();

/**
 * Append one record and fold it into its kind's counters.
 *
 * `kind` is a HostPhaseKind; it is passed as a plain integer because the
 * orchestrator core declares a weak fallback for this function and is also
 * compiled for the AICPU, where the platform's host headers are absent.
 *
 * @param payload  task id for the kinds that submit a task, else a detail count
 * @param index    submit_idx for orchestrator operations, 0 for bind segments
 */
void host_phase_record(uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index);

/**
 * Record one bind segment along with the attributes its log line carries.
 *
 * The attributes are formatted here rather than at flush so the values need not
 * outlive the segment, and they are not written to the log here so the log write
 * stays off the path being measured.
 *
 * @param payload  the segment's one machine-readable quantity, for readers that
 *                 cannot parse the attribute text — byte count on a transfer
 *                 segment, 0 where the segment moves nothing
 */
void host_phase_record_bind(uint32_t kind, uint64_t start_ns, const char *attrs, uint64_t payload = 0);

/**
 * Arm a pass: take the pool the runner offers, and clear the counters.
 *
 * Must run before the first bind segment, which is before the orchestration
 * phase and therefore well before the device collector exists.
 *
 * @param host_api  this run's `const HostApi *`, void-typed to keep the
 *                  platform headers out of the orchestrator core's include set
 */
void host_phase_trace_begin(const void *host_api);

/**
 * Report how many tasks this pass submitted, at the point that becomes known —
 * inside the orchestration segment, well before the pass ends. A reader compares
 * it against the records whose kind submits a task.
 */
void host_phase_trace_note_submitted(uint64_t submitted_tasks);

/**
 * Close the pass: hand the submitted-task count and the run's invocation id to
 * the runner, then emit the per-kind `LOG_TIMING` breakdown from the counters.
 *
 * The records stay in the pool for its readers.
 */
void host_phase_trace_end();
