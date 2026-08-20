/*
 * Three-level profiler for the L2 orchestration sequence.
 *
 * WHERE THE TIMESTAMPS COME FROM — no engine source is modified.
 *
 * Level 1  The four calls, bracketed by the driver:
 *              rt_scope_begin / entry / rt_scope_end / rt_orchestration_done
 *          This is the only level whose numbers are free of instrument cost.
 *
 * Level 2  Every call the entry makes back into the runtime, via a REPLACEMENT
 *          OPS TABLE. `rt->ops` is a plain `const PTO2RuntimeOps *` that
 *          runtime_finalize_after_wire points at the runtime's own s_runtime_ops
 *          (pto_runtime2.cpp:366-380), and the orchestration API reaches the
 *          engine exclusively through it (pto_orchestration_api.h:175, 125, 245,
 *          254, 262 — every one is `rt->ops->…`). Copying that table, wrapping
 *          each entry with a clock read, and pointing rt->ops at the copy
 *          therefore intercepts 100% of the entry->engine traffic without
 *          touching a line of engine code and without a build flag.
 *
 *          This is a strictly better seam than L3's set_test_hook: it needs no
 *          cooperation from the engine at all, and it cannot miss a call site.
 *
 * Level 3  The engine's OWN per-step cycle counters, which exist in the source
 *          already: submit_task_common laps CYCLE_COUNT_LAP into g_orch_*_cycle
 *          at each of its six STEP boundaries, and orchestrator_get_profiling()
 *          returns and resets them (pto_orchestrator.cpp:2043). They are
 *          compiled out unless SIMPLER_ORCH_PROFILING=1, so this level is
 *          present only in the -DL2_ORCH_PROFILING=1 build and the report says
 *          so rather than printing zeros.
 *
 * READ p50/p99, NOT mean. One OS scheduling stall inflates a mean past
 * recognition while leaving the percentiles untouched; the mean column exists
 * because total/count must reconcile, not because it is robust.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// Runtime side only. pto_orchestration_api.h carries its own partial
// PTO2Runtime / PTO2RuntimeOps definitions for the orchestration .so, and the
// two headers cannot coexist in one TU — including both is a hard redefinition
// error. The driver is the runtime side (same as runtime_maker.cpp), the
// payloads are the .so side, and PTO2RuntimeOps is field-for-field identical
// across the two declarations by contract.
#include "pto_runtime2.h"
#include "profiling_config.h"

#if SIMPLER_ORCH_PROFILING
#include "pto_orchestrator.h"
#endif
#if SIMPLER_TENSORMAP_PROFILING
#include "pto_tensormap.h"
#endif

namespace l2_bench {

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}

// Samples for one named step. Kept as raw sample vectors rather than running
// moments so percentiles are exact and the distribution can be re-derived.
struct Step {
    std::vector<uint64_t> ns;

    void add(uint64_t v) { ns.push_back(v); }
    size_t count() const { return ns.size(); }
    uint64_t total() const {
        uint64_t t = 0;
        for (uint64_t v : ns) t += v;
        return t;
    }
    uint64_t pct(double q) const {
        if (ns.empty()) return 0;
        std::vector<uint64_t> s = ns;
        std::sort(s.begin(), s.end());
        size_t i = static_cast<size_t>(q * static_cast<double>(s.size() - 1) + 0.5);
        return s[std::min(i, s.size() - 1)];
    }
    uint64_t mean() const { return ns.empty() ? 0 : total() / ns.size(); }
    uint64_t max() const { return ns.empty() ? 0 : *std::max_element(ns.begin(), ns.end()); }
};

class Profiler {
public:
    // --- Level 1 -----------------------------------------------------------
    void phase(const char *name, uint64_t ns) { phases_[name].add(ns); }

    // --- Level 2 -----------------------------------------------------------
    void record_op(const char *op, uint64_t ns) { ops_[op].add(ns); }

    // Per-submit shape, so a submit's cost can be regressed against how much
    // work it was actually given rather than averaged over a mixed population.
    void record_submit_shape(int32_t tensors, int32_t scalars, int16_t blocks, uint64_t ns) {
        by_tensor_count_[tensors].add(ns);
        subtasks_ += static_cast<uint64_t>(blocks);
        scalars_total_ += static_cast<uint64_t>(scalars);
    }

    void set_totals(int32_t tasks, uint64_t clean_total_ns, uint64_t inst_total_ns) {
        tasks_ = tasks;
        clean_total_ns_ = clean_total_ns;
        inst_total_ns_ = inst_total_ns;
    }

    void set_clock_cost(uint64_t ns) { clock_cost_ns_ = ns; }

#if SIMPLER_ORCH_PROFILING
    // Level 3 must be snapshotted right after the CLEAN run: the engine's
    // counters are process-global accumulators that orchestrator_get_profiling()
    // resets on read, so reading them at report time would fold the clean and
    // instrumented runs together and double every number.
    void set_engine_steps(const PTO2OrchProfilingData &d) {
        engine_steps_ = d;
        have_engine_steps_ = true;
    }
#endif

#if SIMPLER_TENSORMAP_PROFILING
    // Level 4. Same reset-on-read caveat as Level 3: snapshot after the clean run.
    void set_tensormap_stats(const PTO2TensorMapProfilingData &d) {
        tm_stats_ = d;
        have_tm_stats_ = true;
    }
#endif

    uint64_t subtasks() const { return subtasks_; }

    void report(std::FILE *f) const;
    void report_json(std::FILE *f) const;

private:
    void table(std::FILE *f, const char *title, const std::map<std::string, Step> &rows, uint64_t denom) const;

    std::map<std::string, Step> phases_;
    std::map<std::string, Step> ops_;
    std::map<int32_t, Step> by_tensor_count_;
    uint64_t subtasks_{0};
    uint64_t scalars_total_{0};
    int32_t tasks_{0};
    uint64_t clean_total_ns_{0};
    uint64_t inst_total_ns_{0};
    uint64_t clock_cost_ns_{0};
#if SIMPLER_ORCH_PROFILING
    PTO2OrchProfilingData engine_steps_{};
    bool have_engine_steps_{false};
#endif
#if SIMPLER_TENSORMAP_PROFILING
    PTO2TensorMapProfilingData tm_stats_{};
    bool have_tm_stats_{false};
#endif
};

// ---------------------------------------------------------------------------
// The replacement ops table.
//
// install() copies rt->ops, wraps the six entries the orchestration surface
// actually calls, and repoints rt->ops at the copy. remove() restores the
// original. The wrapped functions forward to the ORIGINAL table, so the engine
// path taken is byte-identical to production plus two clock reads.
//
// Single-threaded by construction: an orchestration entry runs on exactly one
// thread (the orchestrator is single-writer by design — see
// PTO2OrchestratorState's "single-thread access by orchestrator, no atomic
// needed"), so a plain global needs no synchronisation.
// ---------------------------------------------------------------------------
class OpsInterceptor {
public:
    OpsInterceptor(PTO2Runtime *rt, Profiler *prof) : rt_(rt), saved_(rt->ops) {
        g_original = rt->ops;
        g_prof = prof;

        table_ = *saved_;  // copy every field, then override the measured ones
        table_.submit_task = &wrap_submit_task;
        table_.alloc_tensors = &wrap_alloc_tensors;
        table_.submit_dummy_task = &wrap_submit_dummy_task;
        table_.scope_begin = &wrap_scope_begin;
        table_.scope_end = &wrap_scope_end;
        table_.orchestration_done = &wrap_orchestration_done;
        // is_fatal is deliberately NOT wrapped: the inline API calls it before
        // every single op, so wrapping it would add a clock pair per call that
        // measures nothing and inflates everything else.
        rt->ops = &table_;
    }

    ~OpsInterceptor() {
        rt_->ops = saved_;
        g_original = nullptr;
        g_prof = nullptr;
    }

    OpsInterceptor(const OpsInterceptor &) = delete;
    OpsInterceptor &operator=(const OpsInterceptor &) = delete;

private:
    static const PTO2RuntimeOps *g_original;
    static Profiler *g_prof;

    static TaskOutputTensors wrap_submit_task(
        PTO2Runtime *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args
    ) {
        const uint64_t t0 = now_ns();
        TaskOutputTensors r = g_original->submit_task(rt, mixed_kernels, args);
        const uint64_t dt = now_ns() - t0;
        g_prof->record_op("submit_task", dt);
        g_prof->record_submit_shape(args.tensor_count(), args.scalar_count(), args.launch_spec.block_num(), dt);
        return r;
    }

    static TaskOutputTensors wrap_alloc_tensors(PTO2Runtime *rt, const CoreTaskArgs &args) {
        const uint64_t t0 = now_ns();
        TaskOutputTensors r = g_original->alloc_tensors(rt, args);
        g_prof->record_op("alloc_tensors", now_ns() - t0);
        return r;
    }

    static TaskOutputTensors wrap_submit_dummy_task(PTO2Runtime *rt, const CoreTaskArgs &args) {
        const uint64_t t0 = now_ns();
        TaskOutputTensors r = g_original->submit_dummy_task(rt, args);
        g_prof->record_op("submit_dummy_task", now_ns() - t0);
        return r;
    }

    static void wrap_scope_begin(PTO2Runtime *rt) {
        const uint64_t t0 = now_ns();
        g_original->scope_begin(rt);
        g_prof->record_op("scope_begin", now_ns() - t0);
    }

    static void wrap_scope_end(PTO2Runtime *rt) {
        const uint64_t t0 = now_ns();
        g_original->scope_end(rt);
        g_prof->record_op("scope_end", now_ns() - t0);
    }

    static void wrap_orchestration_done(PTO2Runtime *rt) {
        const uint64_t t0 = now_ns();
        g_original->orchestration_done(rt);
        g_prof->record_op("orchestration_done", now_ns() - t0);
    }

    PTO2Runtime *rt_;
    const PTO2RuntimeOps *saved_;
    PTO2RuntimeOps table_{};
};

// Cost of one steady_clock read, so the instrument's own weight is a reported
// number rather than a silent addition to every row above.
inline uint64_t measure_clock_cost(int iters = 20000) {
    volatile uint64_t sink = 0;
    const uint64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) sink += now_ns();
    const uint64_t t1 = now_ns();
    (void)sink;
    return (t1 - t0) / static_cast<uint64_t>(iters);
}

}  // namespace l2_bench
