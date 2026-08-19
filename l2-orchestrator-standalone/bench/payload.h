/*
 * The payload — the `entry` half of the four-line sequence.
 *
 * In simpler, `entry_points->entry` is a symbol dlsym'd out of a compiled
 * orchestration .so and its type is `void (*)(const ChipTaskArgs &)`
 * (runtime_maker.cpp:347). The payload here is exactly that function, wrapping
 * qwen3_dynamic_tensormap.h compiled unmodified as C through the esl_shim.
 */

#pragma once

#include <cstdint>
#include <cstdio>

#include "pto_types.h"

namespace l2_bench {

// Owns the ChipTensors the entry args point at: ChipTaskArgs stores TensorRefs
// into a ChipStorageTaskArgs, so that storage must outlive every call.
struct PayloadArgs {
    ChipStorageTaskArgs storage;
    ChipTaskArgs args;

    void finalize() { args.create_from_chip_args(storage); }
};

void qwen3_dyn_entry(const ChipTaskArgs &orch_args);
void qwen3_dyn_build_args(PayloadArgs &out);
void qwen3_dyn_report(std::FILE *f);

}  // namespace l2_bench
