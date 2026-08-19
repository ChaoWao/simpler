/*
 * Host-side definitions for the handful of symbols the extracted L2 TUs
 * reference but do not define.
 *
 * In simpler these live in the AICPU / platform binaries. None of them is part
 * of the orchestrator's logic: four are the scope-stats collector (a DFX
 * sampler the orchestrator calls only behind is_scope_stats_enabled(), which is
 * weak-false in host builds), five are the unified-log sinks, and
 * get_sys_cnt_aicpu is the cycle source the SIMPLER_ORCH_PROFILING build reads.
 *
 * The log sinks write to stderr and are silent unless L2_ORCH_VERBOSE is set,
 * so a bench run's output stays parseable. The scope-stats entries are inert:
 * this package never enables the collector, so they exist only to close the
 * link.
 */

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "aicpu/device_time.h"

namespace {

bool verbose() {
    static const bool on = std::getenv("L2_ORCH_VERBOSE") != nullptr;
    return on;
}

void emit(const char *level, const char *func, const char *fmt, va_list args) {
    if (!verbose()) return;
    std::fprintf(stderr, "[%s] %s: ", level, func);
    std::vfprintf(stderr, fmt, args);
    std::fputc('\n', stderr);
}

}  // namespace

extern "C" {

void unified_log_error(const char *func, const char *fmt, ...) {
    // Errors print regardless of L2_ORCH_VERBOSE: a fatal latched inside the
    // orchestrator silently turns every later submit into a no-op, which would
    // otherwise show up only as an implausibly good throughput number.
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[ERROR] %s: ", func);
    std::vfprintf(stderr, fmt, args);
    std::fputc('\n', stderr);
    va_end(args);
}

#define L2_SHIM_LOG(name, level)                        \
    void name(const char *func, const char *fmt, ...) { \
        va_list args;                                   \
        va_start(args, fmt);                            \
        emit(level, func, fmt, args);                   \
        va_end(args);                                   \
    }

L2_SHIM_LOG(unified_log_warn, "WARN")
L2_SHIM_LOG(unified_log_timing, "TIMING")
L2_SHIM_LOG(unified_log_info, "INFO")
L2_SHIM_LOG(unified_log_debug, "DEBUG")

#undef L2_SHIM_LOG

void scope_stats_begin(
    int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t
) {}
void scope_stats_end(
    int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t
) {}
void scope_stats_on_fatal() {}
void scope_stats_set_pending_site(const char *, int) {}

}  // extern "C"

// The engine's own CYCLE_COUNT_* macros (SIMPLER_ORCH_PROFILING=1) read this.
// device_time_now_ticks() is cntvct_el0 on aarch64 and steady_clock elsewhere,
// so the per-step cycle accounting is sourced identically to the device build.
// Declared with C++ linkage in aicpu/device_time.h, so it must be defined that
// way too — an extern "C" definition is a different symbol.
uint64_t get_sys_cnt_aicpu() { return device_time_now_ticks(); }
