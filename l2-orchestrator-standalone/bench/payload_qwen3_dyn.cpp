/*
 * Adapter side of the qwen3_dynamic_tensormap payload.
 *
 * The case itself is compiled by payload_qwen3_dyn_case.c (as C); the esl_proxy
 * API it calls is implemented by esl_shim/esl_shim_impl.cpp (as C++, on the L2
 * orchestration API). This file only adapts the entry signature and reports the
 * case's own accounting back to the driver.
 *
 * WHAT THE CASE'S orch_args IS HERE
 * ---------------------------------
 * The case builds its own 20 entry tensors with
 * `tensor_from_base_layout(orch_args + i, <explicit shape>, ...)`, so orch_args
 * is a base ADDRESS and the case owns all shape knowledge. It reads no TaskArgs
 * container, which is why build_args below leaves the ChipTaskArgs empty rather
 * than fabricating 20 tensors the case never looks at.
 *
 * Consequence worth stating: the 20 externals sit at consecutive BYTE addresses
 * (orch_args+0 .. orch_args+19) and therefore overlap heavily. In esl_proxy that
 * is harmless because every one of them is passed with an `_ro` tag and never
 * enters the tensormap; the same holds here (NO_DEP), and esl_shim_impl.cpp
 * asserts the invariant at every `_ro` call instead of trusting it.
 */

#include <cstdint>

#include "esl_shim/esl_c_abi.h"
#include "payload.h"

// The case's entry, renamed at build time (see payload_qwen3_dyn_case.c). Its
// parameter is the case's own `const uint64_t orch_args`, not a ChipTaskArgs.
extern "C" void qwen3_dyn_orchestration_entry(const uint64_t orch_args);
// Defined in payload_qwen3_dyn_case.c from the macro the case itself saw.
extern "C" const int qwen3_dyn_spmd_tier;

namespace l2_bench {

// Host backing the case addresses its externals from. Never read or written —
// the orchestrator only does address arithmetic on external tensors — but the
// address must be real so ChipTensor's bounds assertions have something sane to
// work with. ChipTensor::view validates offsets against the recorded SHAPE, not
// against any allocation size, so no multi-GiB fixture is needed for a case
// whose largest external is k_cache at [2949120, 128].
alignas(64) static uint8_t g_ext_backing[1 << 16];

void qwen3_dyn_entry(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // see the header note: the case builds its own externals
    qwen3_dyn_orchestration_entry(reinterpret_cast<uint64_t>(g_ext_backing));
}

void qwen3_dyn_build_args(PayloadArgs &out) {
    // Deliberately empty. Leaving it visibly empty documents that this case
    // takes no entry tensors through the args container.
    out.storage.clear();
    out.finalize();
}

/*
 * The case's own accounting, which the engine's task count alone cannot show.
 *
 * `tasks` counts kernel submits only, so `tasks + allocs` is what the engine
 * reports as its task count — on L2 an alloc_tensors IS a task, unlike in
 * esl_proxy where it is a pool-tail bump.
 *
 * The tracked/no-dep split is the load-bearing number for anyone reading an edge
 * count off this run: only tracked args can produce a TensorMap edge, because
 * esl_proxy's `_ro` variants create no dependency (see esl_shim_impl.cpp).
 */
void qwen3_dyn_report(std::FILE *f) {
    const esl_stats s = esl_get_stats();
    std::fprintf(
        f, "\nqwen3_dynamic_tensormap.h — the case's own accounting (QWEN3_SPMD_TIER=%d)\n", qwen3_dyn_spmd_tier
    );
    // The tier decides how many SPMD chunks fold into one task, so the task
    // count below is meaningless without it. The build pins tier 0 — one chunk
    // per task, so tasks == subtasks — and this line reads the value the case
    // itself saw, which is what makes the pin verifiable rather than assumed.
    std::fprintf(
        f, "  kernel submits      %6llu      framework allocs   %6llu      (engine tasks = %llu)\n",
        static_cast<unsigned long long>(s.tasks), static_cast<unsigned long long>(s.allocs),
        static_cast<unsigned long long>(s.tasks + s.allocs)
    );
    std::fprintf(
        f, "  SPMD subtasks       %6llu      scalar args        %6llu\n",
        static_cast<unsigned long long>(s.subtasks), static_cast<unsigned long long>(s.scalars)
    );
    std::fprintf(
        f, "  dep-tracked args    %6llu      _ro (NO_DEP) args  %6llu\n",
        static_cast<unsigned long long>(s.tracked_args), static_cast<unsigned long long>(s.no_dep_args)
    );
    std::fprintf(
        f, "  sum of the case's DUR_* (virtual AICore time, NOT measured): %.3f us\n",
        static_cast<double>(s.duration_ns) / 1000.0
    );
}

}  // namespace l2_bench
