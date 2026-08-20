/*
 * Smoke driver: the four lines, nothing else.
 *
 *     rt_scope_begin(rt);
 *     entry_points->entry(orch_l2);
 *     rt_scope_end(rt);
 *     rt_orchestration_done(rt);
 *
 * Same shape as runtime_maker.cpp:538-541, with the dlopen'd entry replaced by
 * a linked-in one. Prints the task count the orchestrator actually claimed and
 * whether mark_done published its signal — enough to tell "the engine ran" from
 * "the engine latched a fatal on the first submit and no-op'd the rest", which
 * is otherwise indistinguishable from a very fast run.
 */

#include <cstdio>
#include <exception>

#include "l2_harness.h"
#include "payload.h"
#include "pto_runtime2.h"

using namespace l2_bench;

int main() {
    try {
        // Same graph the bench builds, so a smoke failure here is a real
        // failure there: nothing is reclaimed, so both must hold in full.
        HarnessConfig hc;
        hc.task_window = 8192;
        hc.heap_bytes = 2048ULL << 20;
        Harness h(hc);

        PayloadArgs args;
        qwen3_dyn_build_args(args);

        PTO2Runtime *rt = h.rt();
        rt_scope_begin(rt);
        qwen3_dyn_entry(args.args);
        rt_scope_end(rt);
        rt_orchestration_done(rt);

        if (h.fatal()) {
            std::fprintf(stderr, "l2_orch_main: orchestrator latched fatal, error_code=%d\n", h.error_code());
            return 1;
        }
        const int32_t tasks = h.active_task_count();
        if (tasks <= 0) {
            std::fprintf(stderr, "l2_orch_main: no tasks were claimed\n");
            return 1;
        }
        if (!h.orchestration_done()) {
            std::fprintf(stderr, "l2_orch_main: mark_done did not publish orchestrator_done\n");
            return 1;
        }
        std::printf("l2_orch_main: ok  tasks=%d  orchestrator_done=1\n", tasks);
        std::printf("l2_orch_main: this is graph CONSTRUCTION only — no task is executed, no H2D happens.\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "l2_orch_main: failed: %s\n", e.what());
        return 1;
    }
}
