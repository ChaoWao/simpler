/*
 * Compiles qwen3_dynamic_tensormap.h — UNMODIFIED — as C.
 *
 * The case is a single-translation-unit orchestration source that happens to be
 * named .h; including it is how it is meant to be consumed. It is compiled as C
 * because it is C: its shapes are C99 compound literals, which g++ rejects
 * outright (see the header of esl_shim/esl_c_abi.h for the full account).
 *
 * Its `#include "mem_pool.h"` and `#include "tensormap.h"` resolve to
 * esl_shim/, which forwards to the C ABI that esl_shim_impl.cpp implements on
 * top of the L2 orchestrator.
 *
 * The entry is renamed at build time via
 * -Daicpu_orchestration_entry=qwen3_dyn_orchestration_entry, because the other
 * qwen3 payload in this package already exports that symbol with a different
 * signature and both have C linkage. The case file itself is untouched — the
 * fidelity check asserts it is byte-identical to the repo-root original.
 */

#include "qwen3_dynamic_tensormap.h"

/*
 * The SPMD tier this binary actually compiled the case at, read from the macro
 * as the case itself saw it. Exported so the report cannot disagree with the
 * binary: a tier printed from anywhere else could drift from the -D that shaped
 * the DAG. The case's own guard is
 *
 *     #ifndef QWEN3_SPMD_TIER
 *     #define QWEN3_SPMD_TIER 4
 *     #endif
 *
 * so this picks up the build's -D when given and the case's default otherwise.
 */
const int qwen3_dyn_spmd_tier = QWEN3_SPMD_TIER;
