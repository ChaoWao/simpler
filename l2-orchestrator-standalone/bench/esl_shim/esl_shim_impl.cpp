/*
 * esl_proxy C ABI implemented on the L2 orchestration API.
 *
 * THE TAG MAPPING — read this before trusting any edge count
 * ---------------------------------------------------------
 * Taken from esl_proxy's source, not inferred. In
 * esl_proxy/include/algorithm/tensormap.h the `_ro` variants push NOTHING onto
 * the pending list that tm_submit later looks up and inserts:
 *
 *     tm_in_ptr:       add_tensor_addr(); tm_pending_push(t, TM_PEND_IN)
 *     tm_out_ptr:      add_tensor_addr(); tm_pending_push(t, TM_PEND_OUT)
 *     tm_inout_ptr:    add_tensor_addr(); tm_pending_push(t, TM_PEND_INOUT)
 *     tm_in_ro_ptr:    add_tensor_addr()                       <-- no push
 *     tm_out_ro_ptr:   add_tensor_addr()                       <-- no push
 *     tm_inout_ro_ptr: add_tensor_addr()                       <-- no push
 *
 * So `_ro` means "hand this tensor to the kernel, create NO dependency". It is
 * not a narrower access grant. Hence:
 *
 *   | esl_proxy | dependency role         | L2 tag          |
 *   | tm_in     | tensormap lookup (RaW)  | INPUT           |
 *   | tm_out    | tensormap insert        | OUTPUT_EXISTING |
 *   | tm_inout  | lookup + insert         | INOUT           |
 *   | tm_*_ro   | none                    | NO_DEP          |
 *
 * L2's NO_DEP exists for exactly this ("No-dependency existing tensor: skips
 * OverlapMap lookup, depends on creator only", pto_types.h): compute_task_fanin
 * skips its Step-B lookup, register_task_outputs skips its insert, and the arg
 * still reaches the kernel and the descriptor.
 *
 * tm_out maps to OUTPUT_EXISTING rather than OUTPUT because the case allocates
 * its buffers up front with alloc_tensors and then writes into them. Only
 * OUTPUT_EXISTING and INOUT get registered in the TensorMap; a runtime-created
 * OUTPUT is explicitly skipped ("Runtime-created OUTPUT tensors are not looked
 * up in the TensorMap since they have no dependencies"), so mapping tm_out to
 * OUTPUT would allocate a second buffer and register no producer — silently
 * erasing every RaW edge in the case.
 *
 * NOTE — this DISAGREES with the sibling L3 package's table in
 * bench/qwen3_l3_replay.h, which maps `tm_in_ro -> INPUT` and
 * `tm_out_ro -> OUTPUT_EXISTING`, i.e. gives the `_ro` args real edges. Against
 * esl_proxy's source that is wrong and inflates the L3 replay's edge count.
 *
 * ONE RESIDUAL DIFFERENCE, GUARDED
 * --------------------------------
 * NO_DEP still performs L2's Step A (creator retention): a NO_DEP tensor with a
 * valid owner_task_id yields an edge to its creator, which esl_proxy would not
 * produce. In this case every `_ro` arg is an entry external or a view of one,
 * and externals have no creator — so Step A contributes nothing here. That is
 * asserted at every `_ro` call rather than assumed.
 */

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "esl_c_abi.h"
#include "pto_orchestration_api.h"

// The blob-to-ChipTensor equivalence the whole C boundary rests on.
static_assert(sizeof(EslTensor) == sizeof(ChipTensor), "EslTensor must match ChipTensor size");
static_assert(alignof(EslTensor) == alignof(ChipTensor), "EslTensor must match ChipTensor alignment");
static_assert(std::is_standard_layout_v<ChipTensor>, "ChipTensor must be standard-layout to cross the C boundary");
static_assert(
    std::is_trivially_copyable_v<ChipTensor>,
    "ChipTensor must be trivially copyable: the case copies Tensors by value"
);

namespace {

const ChipTensor &as_tensor(const EslTensor *t) { return *reinterpret_cast<const ChipTensor *>(t); }

EslTensor from_tensor(const ChipTensor &t) {
    EslTensor out;
    std::memcpy(&out, &t, sizeof(out));
    return out;
}

DataType to_l2_dtype(dtype_t d) { return d == BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT32; }

// The pending-task state the C header cannot hold: one CoreTaskArgs per ring
// slot, indexed exactly as esl_proxy indexes g_basic_buf.
struct Pending {
    CoreTaskArgs args;
    uint32_t count{1};
    task_type_t type{TASK_TYPE_VECTOR};
    uint32_t duration{0};
    bool open{false};
};

Pending g_pending[RING_SIZE];
esl_stats g_stats{};

// The case stores its tensors in its own locals and hands us pointers; L2's Arg
// also stores pointers, and both must stay valid until submit. The case's
// locals outlive their tm_submit in every instance, but a NO_DEP/INPUT arg
// pointing at a dead local would corrupt the DAG invisibly, so keep a per-slot
// copy and register that instead. Ownership is then ours and the lifetime
// question disappears.
struct ArgStore {
    ChipTensor tensors[MAX_TENSOR_ARGS];
    int32_t n{0};

    ChipTensor *add(const ChipTensor &t) {
        if (n >= MAX_TENSOR_ARGS) return nullptr;
        tensors[n] = t;
        return &tensors[n++];
    }
    void clear() { n = 0; }
};

ArgStore g_store[RING_SIZE];

[[noreturn]] void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "esl_shim: ");
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    std::abort();
}

ChipTensor *stash(uint32_t tid, const EslTensor *t) {
    ChipTensor *p = g_store[tid & RING_MASK].add(as_tensor(t));
    if (p == nullptr) {
        die("task %u exceeded MAX_TENSOR_ARGS (%d) tensor args", tid, MAX_TENSOR_ARGS);
    }
    return p;
}

void assert_no_creator(const ChipTensor &t, const char *who) {
    if (t.owner_task_id.is_valid()) {
        die(
            "%s received a tensor with a valid owner_task_id. An `_ro` arg creates no\n"
            "  dependency in esl_proxy, but L2's NO_DEP still retains its creator, so this\n"
            "  run's DAG would no longer match the case's semantics.",
            who
        );
    }
}

}  // namespace

// ---------------------------------------------------------------------------

extern "C" {

struct esl_task_desc g_basic_buf[RING_SIZE];
uint32_t g_task_id = 0;

void esl_reset_state(void) {
    std::memset(g_basic_buf, 0, sizeof(g_basic_buf));
    for (int i = 0; i < RING_SIZE; ++i) {
        g_pending[i].args.reset();
        g_pending[i].count = 1;
        g_pending[i].type = TASK_TYPE_VECTOR;
        g_pending[i].duration = 0;
        g_pending[i].open = false;
        g_store[i].clear();
    }
    g_task_id = 0;
    g_stats = esl_stats{};
}

struct esl_stats esl_get_stats(void) { return g_stats; }

void tm_deps_init(void) {
    // esl_proxy builds its own TensorMap here. On L2 the TensorMap belongs to the
    // runtime the harness already stood up, so there is nothing to construct —
    // but per-run state must be cleared, because the bench rebuilds the runtime
    // for every repetition and re-enters the case.
    esl_reset_state();
}

int new_task(uint32_t task_id, uint32_t type, uint32_t count, uint32_t duration) {
    const uint32_t slot = task_id & RING_MASK;

    struct esl_task_desc &d = g_basic_buf[slot];
    d.id = task_id;
    d.type = static_cast<task_type_t>(type);
    d.count = count;
    d.duration = duration;
    // esl_proxy sets SPMD_SYNC whenever count > 1 (ring_buf.h:162).
    d.mode = count > 1 ? ORG_MODE_SPMD_SYNC : ORG_MODE_SINGLE;
    d.tensor_cnt = 0;
    d.scalar_cnt = 0;

    Pending &p = g_pending[slot];
    p.args.reset();
    p.count = count;
    p.type = static_cast<task_type_t>(type);
    p.duration = duration;
    p.open = true;
    p.args.launch_spec.set_block_num(static_cast<int16_t>(count));
    g_store[slot].clear();
    return 1;
}

void add_scalar(uint32_t task_id, int64_t value) {
    const uint32_t slot = task_id & RING_MASK;
    g_pending[slot].args.add_scalar(value);
    g_basic_buf[slot].scalar_cnt++;
    g_stats.scalars++;
}

EslTensor tensor_from_base_layout(uint64_t base, const uint32_t shapes[], uint32_t ndims, dtype_t dtype) {
    return from_tensor(make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(base)), shapes, ndims, to_l2_dtype(dtype)
    ));
}

EslTensor esl_view_at(const EslTensor *t, uint32_t off0, uint32_t off1, uint32_t n0, uint32_t n1) {
    // esl_proxy's view_at and ChipTensor::view compute the same thing: advance
    // start_offset by sum(offset[i] * stride[i]) and keep the parent's strides.
    // This is an argument reshuffle, not a reimplementation.
    const uint32_t view_shapes[2] = {n0, n1};
    const uint32_t view_offsets[2] = {off0, off1};
    return from_tensor(as_tensor(t).view(view_shapes, view_offsets));
}

EslTensor alloc_tensors(uint32_t shape[], int dim, int bytes) {
    // In esl_proxy this bumps a pool tail. On L2 it is a real hidden alloc task:
    // it claims a ring slot, cuts the GM heap, registers the buffer in the
    // TensorMap and sets owner_task_id so consumers retain their creator. That
    // difference is the point — an alloc here costs what it costs in the engine
    // under test, and shows up in the profile as `alloc_tensors`.
    const uint32_t shapes[2] = {shape[0], shape[1]};
    TensorCreateInfo ci(shapes, static_cast<uint32_t>(dim), to_l2_dtype(static_cast<dtype_t>(bytes)));
    TaskOutputTensors out = alloc_tensors(ci);
    if (out.empty()) {
        die("alloc_tensors([%u, %u]) returned nothing — the GM heap or ring is exhausted", shape[0], shape[1]);
    }
    g_stats.allocs++;
    return from_tensor(out.get_ref(0));
}

void tm_in_ptr(uint32_t tid, const EslTensor *t) {
    g_pending[tid & RING_MASK].args.add_input(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.tracked_args++;
}

void tm_out_ptr(uint32_t tid, const EslTensor *t) {
    g_pending[tid & RING_MASK].args.add_output(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.tracked_args++;
}

void tm_inout_ptr(uint32_t tid, const EslTensor *t) {
    g_pending[tid & RING_MASK].args.add_inout(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.tracked_args++;
}

void tm_in_ro_ptr(uint32_t tid, const EslTensor *t) {
    assert_no_creator(as_tensor(t), "tm_in_ro");
    g_pending[tid & RING_MASK].args.add_no_dep(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.no_dep_args++;
}

void tm_out_ro_ptr(uint32_t tid, const EslTensor *t) {
    assert_no_creator(as_tensor(t), "tm_out_ro");
    g_pending[tid & RING_MASK].args.add_no_dep(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.no_dep_args++;
}

void tm_inout_ro_ptr(uint32_t tid, const EslTensor *t) {
    assert_no_creator(as_tensor(t), "tm_inout_ro");
    g_pending[tid & RING_MASK].args.add_no_dep(*stash(tid, t));
    g_basic_buf[tid & RING_MASK].tensor_cnt++;
    g_stats.no_dep_args++;
}

void tm_submit_ptr(uint32_t tid) {
    const uint32_t slot = tid & RING_MASK;
    Pending &p = g_pending[slot];
    if (!p.open) die("tm_submit(%u) without a matching new_task", tid);
    if (p.args.has_error) {
        die("task %u built an invalid Arg: %s", tid, p.args.error_msg ? p.args.error_msg : "(unknown)");
    }
    // The C-visible descriptor and the real pending state must agree; they can
    // only diverge if the case wrote g_basic_buf directly (set_block_num), which
    // would not reach the engine. Fail loudly instead of submitting a task whose
    // SPMD width silently differs from what the case asked for.
    if (g_basic_buf[slot].count != p.count) {
        die(
            "task %u: g_basic_buf.count=%u disagrees with the submitted block_num=%u.\n"
            "  The case mutated the descriptor directly; that path does not reach the engine.",
            tid, g_basic_buf[slot].count, p.count
        );
    }

    MixedKernels mk;
    // CUBE -> AIC, VECTOR/MIX -> AIV0. Kernel ids are opaque to the
    // orchestrator: it stores them in the descriptor and the AICore would
    // resolve them, so any distinct valid value carries identical cost.
    if (p.type == TASK_TYPE_CUBE) {
        mk.aic_kernel_id = 1;
    } else {
        mk.aiv0_kernel_id = 2;
    }

    (void)rt_submit_task(mk, p.args);

    g_stats.tasks++;
    g_stats.subtasks += p.count;
    g_stats.duration_ns += p.duration;
    p.open = false;
}

}  // extern "C"
