/*
 * L2 orchestration bench driver.
 *
 * Measures the four-line sequence
 *
 *     rt_scope_begin(rt); entry(orch_l2); rt_scope_end(rt); rt_orchestration_done(rt);
 *
 * against qwen3_dynamic_tensormap.h at QWEN3_SPMD_TIER=0, on a fresh runtime
 * per repetition.
 *
 * Modes
 *   --mode=throughput   N clean repetitions, wall clock only. No interception,
 *                       so these are the numbers to quote.
 *   --mode=profile      One clean run (for the honest total) followed by one
 *                       instrumented run through the replacement ops table, so
 *                       the instrument's own cost is reported instead of folded
 *                       in silently. See bench/l2_profile.h.
 *
 * WHY A FRESH RUNTIME PER REPETITION
 * ----------------------------------
 * Nothing ever completes here — there is no scheduler and no device — so
 * last_task_alive never advances and the ring's watermark reclaim never fires.
 * A second run on the same runtime would therefore start with the task window,
 * GM heap and TensorMap entry pool already consumed by the first, and would
 * measure back-pressure rather than steady-state submit. Tearing down and
 * rebuilding is the only way each repetition sees the same initial conditions.
 *
 * That same fact bounds what the payload can ask for: task_window must exceed
 * its total task count and heap_bytes must exceed its total allocation, because
 * neither is ever recycled. --task-window and --heap-mb exist for that, and
 * exhausting either latches a fatal that this driver reports rather than
 * silently reporting a small, fast graph.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "l2_harness.h"
#include "l2_profile.h"
#include "payload.h"
#include "pto_runtime2.h"

using namespace l2_bench;

namespace {

struct Options {
    std::string mode = "throughput";
    int repeat = 5;
    // Defaults hold the whole graph: 3875 engine tasks, nothing reclaimed.
    uint64_t task_window = 8192;
    uint64_t heap_mb = 2048;
    bool prefault_sm = false;
    bool prefault_arena = false;
    std::string out;
};

[[noreturn]] void usage(int code) {
    std::printf(
        "l2_bench — bench and profile the L2 orchestration sequence\n"
        "           payload: qwen3_dynamic_tensormap.h, QWEN3_SPMD_TIER=0\n\n"
        "  --mode=throughput|profile     default throughput\n"
        "  --repeat=N         repetitions in throughput mode (default 5)\n"
        "  --task-window=N    ring task slots (default 8192; never reclaimed here)\n"
        "  --heap-mb=N        GM heap stand-in in MiB (default 2048; never reclaimed here)\n"
        "  --prefault-sm      DIAGNOSTIC: pre-touch the SM so cold-slot cost is excluded\n"
        "  --prefault-arena   DIAGNOSTIC: pre-touch the arena (TensorMap buckets + pool)\n"
        "  --prefault-all     both of the above\n"
        "  --out=PATH         write the JSON report here (profile mode)\n"
    );
    std::exit(code);
}

bool arg_u64(const std::string &a, const char *key, uint64_t &out) {
    const std::string prefix = std::string("--") + key + "=";
    if (a.rfind(prefix, 0) != 0) return false;
    out = std::strtoull(a.c_str() + prefix.size(), nullptr, 10);
    return true;
}

bool arg_int(const std::string &a, const char *key, int &out) {
    uint64_t v = 0;
    if (!arg_u64(a, key, v)) return false;
    out = static_cast<int>(v);
    return true;
}

bool arg_str(const std::string &a, const char *key, std::string &out) {
    const std::string prefix = std::string("--") + key + "=";
    if (a.rfind(prefix, 0) != 0) return false;
    out = a.substr(prefix.size());
    return true;
}

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") usage(0);
        else if (arg_str(a, "mode", o.mode)) {}
        else if (arg_str(a, "out", o.out)) {}
        else if (arg_int(a, "repeat", o.repeat)) {}
        else if (arg_u64(a, "task-window", o.task_window)) {}
        else if (arg_u64(a, "heap-mb", o.heap_mb)) {}
        else if (a == "--prefault-sm") o.prefault_sm = true;
        else if (a == "--prefault-arena") o.prefault_arena = true;
        else if (a == "--prefault-all") { o.prefault_sm = true; o.prefault_arena = true; }
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(2);
        }
    }
    if (o.mode != "throughput" && o.mode != "profile") {
        std::fprintf(stderr, "--mode must be throughput|profile\n");
        std::exit(2);
    }
    if (o.repeat < 1) o.repeat = 1;
    return o;
}

struct RunResult {
    uint64_t scope_begin_ns{0};
    uint64_t entry_ns{0};
    uint64_t scope_end_ns{0};
    uint64_t done_ns{0};
    uint64_t total_ns{0};
    int32_t tasks{0};
    bool fatal{false};
    int32_t error_code{0};
};

// One complete lifecycle: build a runtime, run the four lines, tear it down.
// `prof` non-null installs the replacement ops table for the duration of the
// entry only — the Level-1 numbers around it stay comparable either way.
RunResult run_once(const Options &o, Profiler *prof) {
    HarnessConfig hc;
    hc.task_window = o.task_window;
    hc.heap_bytes = o.heap_mb << 20;
    hc.prefault_sm = o.prefault_sm;
    hc.prefault_arena = o.prefault_arena;
    Harness h(hc);

    PayloadArgs args;
    qwen3_dyn_build_args(args);

    PTO2Runtime *rt = h.rt();
    RunResult r;
    const uint64_t t0 = now_ns();

    {
        // Scoped so the table is restored before the harness tears down.
        std::unique_ptr<OpsInterceptor> interceptor;
        if (prof != nullptr) interceptor.reset(new OpsInterceptor(rt, prof));

        const uint64_t a = now_ns();
        rt_scope_begin(rt);
        const uint64_t b = now_ns();
        qwen3_dyn_entry(args.args);
        const uint64_t c = now_ns();
        rt_scope_end(rt);
        const uint64_t d = now_ns();
        rt_orchestration_done(rt);
        const uint64_t e = now_ns();

        r.scope_begin_ns = b - a;
        r.entry_ns = c - b;
        r.scope_end_ns = d - c;
        r.done_ns = e - d;
    }

    r.total_ns = now_ns() - t0;
    r.tasks = h.active_task_count();
    r.fatal = h.fatal();
    r.error_code = h.error_code();
    return r;
}

void check(const RunResult &r) {
    if (r.fatal) {
        std::fprintf(
            stderr,
            "l2_bench: the orchestrator latched a fatal (error_code=%d) — every submit after\n"
            "that point was a silent no-op, so the task count and timings below are NOT a\n"
            "measurement of the workload. Raise --task-window / --heap-mb and re-run.\n",
            r.error_code
        );
        std::exit(1);
    }
    if (r.tasks <= 0) {
        std::fprintf(stderr, "l2_bench: no tasks were claimed — the payload built an empty graph\n");
        std::exit(1);
    }
}

int mode_throughput(const Options &o) {
    std::vector<uint64_t> totals;
    std::vector<uint64_t> entries;
    RunResult last;
    for (int i = 0; i < o.repeat; ++i) {
        last = run_once(o, nullptr);
        check(last);
        totals.push_back(last.total_ns);
        entries.push_back(last.entry_ns);
    }
    std::sort(totals.begin(), totals.end());
    std::sort(entries.begin(), entries.end());
    const uint64_t med = totals[totals.size() / 2];
    const uint64_t med_entry = entries[entries.size() / 2];

    std::printf("payload=qwen3-dyn  tasks=%d  repeat=%d\n", last.tasks, o.repeat);
    std::printf(
        "four-line block   median %10.3f ms   min %10.3f ms   max %10.3f ms\n", med / 1e6, totals.front() / 1e6,
        totals.back() / 1e6
    );
    std::printf("  of which entry() median %10.3f ms  (%.1f%%)\n", med_entry / 1e6, 100.0 * med_entry / med);
    std::printf(
        "per task          %8.2f us          throughput %12.0f tasks/s\n", med / 1000.0 / last.tasks,
        last.tasks / (med / 1e9)
    );
    qwen3_dyn_report(stdout);
    return 0;
}

int mode_profile(const Options &o) {
    // Clean pass FIRST, so the instrumented pass cannot be credited with a warm
    // allocator or warm page cache the clean one did not have.
    const RunResult clean = run_once(o, nullptr);
    check(clean);

    Profiler prof;
#if SIMPLER_ORCH_PROFILING
    // Drain the engine's global step counters NOW, while they hold only the
    // clean run. They reset on read, so the instrumented run below starts from
    // zero and never contaminates this snapshot.
    prof.set_engine_steps(orchestrator_get_profiling());
#endif
#if SIMPLER_TENSORMAP_PROFILING
    prof.set_tensormap_stats(pto2_tensormap_get_profiling());
#endif
    prof.set_clock_cost(measure_clock_cost());
    const RunResult inst = run_once(o, &prof);
    check(inst);

    // Level 1 rows come from the CLEAN run — the interception distorts entry()
    // and nothing else, so mixing the two would misattribute its cost.
    prof.phase("rt_scope_begin", clean.scope_begin_ns);
    prof.phase("entry", clean.entry_ns);
    prof.phase("rt_scope_end", clean.scope_end_ns);
    prof.phase("rt_orchestration_done", clean.done_ns);
    prof.set_totals(clean.tasks, clean.total_ns, inst.total_ns);

    prof.report(stdout);
    qwen3_dyn_report(stdout);
    if (!o.out.empty()) {
        std::FILE *f = std::fopen(o.out.c_str(), "w");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
            return 1;
        }
        prof.report_json(f);
        std::fclose(f);
        std::fprintf(stderr, "l2_bench: profile -> %s\n", o.out.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    const Options o = parse(argc, argv);
    try {
        if (o.mode == "throughput") return mode_throughput(o);
        return mode_profile(o);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "l2_bench: failed: %s\n", e.what());
        return 1;
    }
}
