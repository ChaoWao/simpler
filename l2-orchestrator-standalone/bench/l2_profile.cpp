/*
 * Report rendering for l2_profile.h, plus the interceptor's statics.
 */

#include "l2_profile.h"

namespace l2_bench {

const PTO2RuntimeOps *OpsInterceptor::g_original = nullptr;
Profiler *OpsInterceptor::g_prof = nullptr;

void Profiler::table(std::FILE *f, const char *title, const std::map<std::string, Step> &rows, uint64_t denom) const {
    std::fprintf(f, "\n%s\n", title);
    std::fprintf(
        f, "  %-28s %8s %10s %10s %10s %10s %10s %7s\n", "step", "count", "total_ms", "mean_ns", "p50_ns", "p99_ns",
        "max_ns", "%"
    );
    std::fprintf(f, "  %s\n", std::string(108, '-').c_str());
    for (const auto &kv : rows) {
        const Step &s = kv.second;
        const double pct = denom > 0 ? 100.0 * static_cast<double>(s.total()) / static_cast<double>(denom) : 0.0;
        std::fprintf(
            f, "  %-28s %8zu %10.3f %10llu %10llu %10llu %10llu %6.2f%%\n", kv.first.c_str(), s.count(),
            static_cast<double>(s.total()) / 1e6, static_cast<unsigned long long>(s.mean()),
            static_cast<unsigned long long>(s.pct(0.5)), static_cast<unsigned long long>(s.pct(0.99)),
            static_cast<unsigned long long>(s.max()), pct
        );
    }
}

void Profiler::report(std::FILE *f) const {
    // `tasks_` is the engine's own ring-slot count, which includes the framework
    // alloc tasks; `subtasks_` sums block_num over kernel submits ONLY, because
    // that is the only path the ops-table interceptor sees a launch_spec on.
    // Printing the two side by side without saying so reads as "N tasks expand
    // into M subtasks" and makes M < N look like a bug. Break the count out
    // instead, so every number states its own population.
    const auto op_count = [this](const char *name) -> uint64_t {
        auto it = ops_.find(name);
        return it == ops_.end() ? 0 : static_cast<uint64_t>(it->second.count());
    };
    const uint64_t submits = op_count("submit_task") + op_count("submit_dummy_task");
    const uint64_t allocs = op_count("alloc_tensors");

    std::fprintf(f, "%s\n", std::string(110, '=').c_str());
    std::fprintf(f, "L2 Orchestrator profile — qwen3_dynamic_tensormap.h (QWEN3_SPMD_TIER=0)\n");
    std::fprintf(
        f, "  %llu kernel submits (%llu SPMD subtasks) + %llu framework allocs = %d engine tasks\n",
        static_cast<unsigned long long>(submits), static_cast<unsigned long long>(subtasks_),
        static_cast<unsigned long long>(allocs), tasks_
    );
    // A mismatch means the clean and instrumented runs built different graphs,
    // which would silently invalidate every per-call number below.
    if (submits + allocs != static_cast<uint64_t>(tasks_)) {
        std::fprintf(
            f,
            "  WARNING: %llu + %llu != %d — the instrumented run did not build the same\n"
            "  graph as the clean one, so the tables below are not comparable.\n",
            static_cast<unsigned long long>(submits), static_cast<unsigned long long>(allocs), tasks_
        );
    }
    std::fprintf(f, "engine: a2a3 / host_build_graph, PTO2OrchestratorState (L2)\n");
    std::fprintf(f, "%s\n", std::string(110, '=').c_str());

    uint64_t phase_total = 0;
    for (const auto &kv : phases_) phase_total += kv.second.total();

    std::fprintf(f, "\nLEVEL 1 — the four lines (driver-bracketed, uninstrumented run)\n");
    std::fprintf(f, "  %-28s %14s %14s\n", "step", "ns", "% of block");
    std::fprintf(f, "  %s\n", std::string(58, '-').c_str());
    // Fixed order: this is a sequence, and sorting it alphabetically would
    // destroy the one property a reader needs from it.
    for (const char *name : {"rt_scope_begin", "entry", "rt_scope_end", "rt_orchestration_done"}) {
        auto it = phases_.find(name);
        if (it == phases_.end()) continue;
        const uint64_t t = it->second.total();
        const double pct = phase_total > 0 ? 100.0 * static_cast<double>(t) / static_cast<double>(phase_total) : 0.0;
        std::fprintf(f, "  %-28s %14llu %13.2f%%\n", name, static_cast<unsigned long long>(t), pct);
    }
    std::fprintf(f, "  %-28s %14llu\n", "TOTAL", static_cast<unsigned long long>(phase_total));
    if (tasks_ > 0) {
        std::fprintf(
            f, "\n  per task: %.2f us    throughput: %.0f tasks/s\n",
            static_cast<double>(phase_total) / 1000.0 / tasks_,
            static_cast<double>(tasks_) / (static_cast<double>(phase_total) / 1e9)
        );
    }

    if (clean_total_ns_ > 0 && inst_total_ns_ > 0) {
        const double delta =
            100.0 * (static_cast<double>(inst_total_ns_) - static_cast<double>(clean_total_ns_)) /
            static_cast<double>(clean_total_ns_);
        std::fprintf(f, "\nINSTRUMENT COST — quote throughput from the clean run, shape from the tables below\n");
        std::fprintf(f, "  uninstrumented total   %10.3f ms\n", static_cast<double>(clean_total_ns_) / 1e6);
        std::fprintf(
            f, "  instrumented   total   %10.3f ms   (%+.1f%%)\n", static_cast<double>(inst_total_ns_) / 1e6, delta
        );
        std::fprintf(f, "  one steady_clock read  %10llu ns\n", static_cast<unsigned long long>(clock_cost_ns_));
    }

    std::fprintf(
        f,
        "\nREAD p50/p99, NOT mean. A single OS scheduling stall moves a mean by multiples\n"
        "while leaving p50 and p99 untouched. The mean column reconciles total/count; it\n"
        "is not the robust statistic.\n"
    );

    uint64_t ops_total = 0;
    for (const auto &kv : ops_) ops_total += kv.second.total();
    table(f, "LEVEL 2 — every entry->engine call (intercepted ops table)", ops_, ops_total);

    if (!by_tensor_count_.empty()) {
        std::fprintf(f, "\nSUBMIT LATENCY BY TENSOR-ARG COUNT — the per-tensor slope of STEP 3/4\n");
        std::fprintf(f, "  %-12s %8s %10s %10s %12s\n", "tensors", "count", "p50_ns", "p99_ns", "p50/tensor");
        std::fprintf(f, "  %s\n", std::string(56, '-').c_str());
        for (const auto &kv : by_tensor_count_) {
            const uint64_t p50 = kv.second.pct(0.5);
            std::fprintf(
                f, "  %-12d %8zu %10llu %10llu %12.1f\n", kv.first, kv.second.count(),
                static_cast<unsigned long long>(p50), static_cast<unsigned long long>(kv.second.pct(0.99)),
                kv.first > 0 ? static_cast<double>(p50) / kv.first : 0.0
            );
        }
    }

#if SIMPLER_ORCH_PROFILING
    if (!have_engine_steps_) {
        std::fprintf(f, "\nLEVEL 3 — no snapshot was taken; call set_engine_steps() after the clean run.\n");
    } else {
        const PTO2OrchProfilingData &d = engine_steps_;
        // The clock the shim feeds get_sys_cnt_aicpu() is cntvct_el0, so the
        // divisor must be cntfrq_el0 — NOT PLATFORM_PROF_SYS_CNT_FREQ, which is
        // the device's 50 MHz counter and is 2x off on this host.
        const uint64_t hz = device_time_frequency_hz();
        const auto to_ns = [hz](uint64_t cycles) -> double {
            return hz > 0 ? static_cast<double>(cycles) * 1e9 / static_cast<double>(hz) : 0.0;
        };
        uint64_t sum = d.alloc_cycle + d.sync_cycle + d.lookup_cycle + d.insert_cycle + d.args_cycle + d.fanin_cycle +
                       d.scope_end_cycle;
        std::fprintf(f, "\nLEVEL 3 — the engine's own step counters inside submit_task_common\n");
        std::fprintf(
            f, "  (SIMPLER_ORCH_PROFILING=1, uninstrumented run; submit_count=%lld, clock=%llu Hz)\n",
            static_cast<long long>(d.submit_count), static_cast<unsigned long long>(hz)
        );
        std::fprintf(f, "  %-46s %12s %12s %8s\n", "step", "cycles", "total_ms", "%");
        std::fprintf(f, "  %s\n", std::string(82, '-').c_str());
        const struct {
            const char *name;
            uint64_t cycles;
        } rows[] = {
            {"STEP 1  prepare_task (slot + heap alloc)", d.alloc_cycle},
            {"STEP 2  sync_tensormap", d.sync_cycle},
            {"STEP 3  infer deps: TensorMap lookup", d.lookup_cycle},
            {"STEP 4  register outputs: TensorMap insert", d.insert_cycle},
            {"STEP 5  payload/descriptor GM write", d.args_cycle},
            {"STEP 6  publish fanin_count", d.fanin_cycle},
            {"        end_scope (outside submit)", d.scope_end_cycle},
        };
        for (const auto &r : rows) {
            std::fprintf(
                f, "  %-46s %12llu %12.3f %7.2f%%\n", r.name, static_cast<unsigned long long>(r.cycles),
                to_ns(r.cycles) / 1e6, sum > 0 ? 100.0 * static_cast<double>(r.cycles) / static_cast<double>(sum) : 0.0
            );
        }
        std::fprintf(
            f, "  waits: alloc %llu cycles, fanin %llu cycles\n", static_cast<unsigned long long>(d.alloc_wait_cycle),
            static_cast<unsigned long long>(d.fanin_wait_cycle)
        );
    }

#if SIMPLER_TENSORMAP_PROFILING
    if (have_tm_stats_) {
        const PTO2TensorMapProfilingData &t = tm_stats_;
        const double avg_chain =
            t.lookup_count > 0 ? static_cast<double>(t.lookup_chain_total) / static_cast<double>(t.lookup_count) : 0.0;
        std::fprintf(f, "\nLEVEL 4 — inside STEP 3: what the TensorMap lookup actually walks\n");
        std::fprintf(f, "  lookups                       %10llu\n", (unsigned long long)t.lookup_count);
        std::fprintf(f, "  inserts                       %10llu\n", (unsigned long long)t.insert_count);
        std::fprintf(f, "  bucket entries walked (total)  %10llu\n", (unsigned long long)t.lookup_chain_total);
        std::fprintf(f, "  avg chain length              %13.2f\n", avg_chain);
        std::fprintf(f, "  MAX chain length              %10d\n", t.lookup_chain_max);
        std::fprintf(f, "  overlap checks                %10llu\n", (unsigned long long)t.overlap_checks);
        std::fprintf(
            f, "  overlap hits                  %10llu   (%.1f%% of checks)\n", (unsigned long long)t.overlap_hits,
            t.overlap_checks > 0 ? 100.0 * (double)t.overlap_hits / (double)t.overlap_checks : 0.0
        );
        // The map hashes on buffer.addr ALONE (pto_tensormap.h:522), so every
        // sub-view of one buffer shares a bucket. A chain far longer than the
        // average therefore means one heavily-subdivided buffer is serialising
        // every lookup that touches it — that is a data-structure problem, not a
        // per-call constant, and it is what makes STEP 3 scale with SPMD width.
        if (avg_chain > 0.0) {
            std::fprintf(
                f, "  wasted walk (checks that found no overlap): %.1f%%\n",
                t.overlap_checks > 0 ? 100.0 * (1.0 - (double)t.overlap_hits / (double)t.overlap_checks) : 0.0
            );
        }
    }
#endif
#else
    std::fprintf(
        f,
        "\nLEVEL 3 — not built. The engine's per-STEP counters are compiled out at\n"
        "SIMPLER_ORCH_PROFILING=0 (the default). Rebuild with -DL2_ORCH_PROFILING=1\n"
        "to get the six-step breakdown inside submit_task_common.\n"
    );
#endif
    std::fprintf(f, "\n");
}

void Profiler::report_json(std::FILE *f) const {
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"payload\": \"qwen3-dyn\",\n");
    std::fprintf(f, "  \"engine\": \"a2a3/host_build_graph L2 PTO2OrchestratorState\",\n");
    // Named for their populations, not "tasks"/"subtasks": engine_tasks counts
    // ring slots (kernel submits + framework allocs), spmd_subtasks sums
    // block_num over kernel submits only.
    std::fprintf(f, "  \"engine_tasks\": %d,\n", tasks_);
    std::fprintf(f, "  \"spmd_subtasks\": %llu,\n", static_cast<unsigned long long>(subtasks_));
    std::fprintf(f, "  \"clean_total_ns\": %llu,\n", static_cast<unsigned long long>(clean_total_ns_));
    std::fprintf(f, "  \"instrumented_total_ns\": %llu,\n", static_cast<unsigned long long>(inst_total_ns_));
    std::fprintf(f, "  \"clock_read_ns\": %llu,\n", static_cast<unsigned long long>(clock_cost_ns_));
    std::fprintf(f, "  \"orch_profiling_built\": %s,\n", SIMPLER_ORCH_PROFILING ? "true" : "false");

    const auto emit_steps = [&](const char *key, const std::map<std::string, Step> &rows, bool last) {
        std::fprintf(f, "  \"%s\": {\n", key);
        for (auto it = rows.begin(); it != rows.end(); ++it) {
            std::fprintf(
                f, "    \"%s\": {\"count\": %zu, \"total_ns\": %llu, \"mean_ns\": %llu, \"p50_ns\": %llu, "
                   "\"p99_ns\": %llu, \"max_ns\": %llu}%s\n",
                it->first.c_str(), it->second.count(), static_cast<unsigned long long>(it->second.total()),
                static_cast<unsigned long long>(it->second.mean()),
                static_cast<unsigned long long>(it->second.pct(0.5)),
                static_cast<unsigned long long>(it->second.pct(0.99)),
                static_cast<unsigned long long>(it->second.max()), std::next(it) == rows.end() ? "" : ","
            );
        }
        std::fprintf(f, "  }%s\n", last ? "" : ",");
    };

    emit_steps("phases", phases_, false);
    emit_steps("ops", ops_, false);

    std::fprintf(f, "  \"submit_by_tensor_count\": {\n");
    for (auto it = by_tensor_count_.begin(); it != by_tensor_count_.end(); ++it) {
        std::fprintf(
            f, "    \"%d\": {\"count\": %zu, \"p50_ns\": %llu, \"p99_ns\": %llu}%s\n", it->first, it->second.count(),
            static_cast<unsigned long long>(it->second.pct(0.5)),
            static_cast<unsigned long long>(it->second.pct(0.99)),
            std::next(it) == by_tensor_count_.end() ? "" : ","
        );
    }
    std::fprintf(f, "  }\n}\n");
}

}  // namespace l2_bench
