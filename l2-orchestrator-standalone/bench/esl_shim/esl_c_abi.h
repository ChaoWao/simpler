/*
 * esl_proxy C ABI, reimplemented over the L2 orchestrator.
 *
 * This header is included from BOTH sides:
 *   - the C translation unit that compiles qwen3_dynamic_tensormap.h unmodified
 *   - the C++ translation unit that implements these functions on L2
 *
 * WHY THE CASE IS COMPILED AS C, NOT C++
 * --------------------------------------
 * The case passes its tensor shapes as C99 compound literals:
 *
 *     tensor_from_base_layout(orch_args + 0, (uint32_t[]){90, 5120}, 2, BFLOAT16)
 *
 * In C a compound literal is an lvalue with enclosing-block lifetime, so
 * array-to-pointer decay is well defined. In C++ it is a temporary, and g++
 * rejects the decay outright ("taking address of temporary array"). No dialect
 * or permissiveness flag accepts it — checked across -std=c++17 / gnu++17 /
 * gnu++11 / gnu++03, with and without -fpermissive. The file is C, so it is
 * compiled as C, and this header is the boundary.
 *
 * WHY `Tensor` IS AN OPAQUE BLOB
 * ------------------------------
 * The C side needs `Tensor` to be a complete type (the case declares locals,
 * copies them, and takes their address) but it never reads a single member —
 * verified by grep: no `.shapes`, `.buffer`, or any other field access in the
 * whole case. So the C side gets a blob, and the C++ side reinterprets it as
 * ChipTensor. Both are 128 bytes at 64-byte alignment, and ChipTensor is
 * standard-layout and trivially copyable, which is what makes the case's
 * by-value copies (`Tensor v = view(t, ...)`) mean exactly what they mean for a
 * ChipTensor. esl_shim_impl.cpp static_asserts all of it rather than trusting
 * this comment.
 *
 * A blob is deliberately stronger than mirroring esl_proxy's struct field by
 * field: it assumes only size and alignment, so it cannot silently rot if a
 * ChipTensor field is reordered.
 */

#ifndef ESL_SHIM_C_ABI_H
#define ESL_SHIM_C_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Tensor -------------------------------------------------------------- */

typedef struct EslTensor {
    unsigned char opaque[128];
} __attribute__((aligned(64))) EslTensor;

#ifndef __cplusplus
/* The case spells it `Tensor`. The C++ side must NOT: `Tensor` is already a
 * distinct wire struct in src/common/task_interface/buffer.h, and aliasing the
 * name there is a hard redefinition error. */
typedef EslTensor Tensor;
#endif

/* esl_proxy's dtype_t IS the element width in bytes (its tensor.h:16-20).
 * FLOAT32 and INT32 are both 4 and therefore indistinguishable — carried over
 * as-is, since width is all the orchestrator uses a dtype for. */
typedef enum {
    BFLOAT16 = 2,
    FLOAT32 = 4,
    INT32 = 4
} dtype_t;

/* --- task.h ------------------------------------------------------------- */

typedef enum {
    TASK_TYPE_CUBE = 0,
    TASK_TYPE_VECTOR = 1,
    /* esl_proxy really defines MIX == VECTOR == 1 (task.h:19-20), so the case's
     * one TASK_TYPE_MIX task is indistinguishable from a VECTOR one on the
     * esl_proxy side too. Preserved, not "fixed". */
    TASK_TYPE_MIX = 1,
    TASK_TYPE_CNT = 3
} task_type_t;

typedef enum {
    ORG_MODE_SINGLE = 0,
    ORG_MODE_GROUP = 1,
    ORG_MODE_SPMD_SYNC = 2,
    ORG_MODE_SPMD_ASYNC = 3
} org_mode_t;

#define RING_SIZE 4096
#define RING_MASK (RING_SIZE - 1)

/* --- ring_buf.h ---------------------------------------------------------- */

/*
 * Only the fields the case itself touches. The case defines two static helpers
 * (set_task_type / set_block_num) that write .type / .mode / .count directly;
 * neither is called anywhere in it, but both must type-check, which is the only
 * reason this struct is visible at all.
 *
 * It is NOT the pending-task state. That lives in esl_shim_impl.cpp as a
 * CoreTaskArgs per ring slot, because a C++ Arg cannot appear in a C header.
 * new_task() writes both. Consequence, stated plainly: if a future case DID
 * call set_block_num(), it would update this struct and not the CoreTaskArgs,
 * so the block count would not reach the engine. esl_shim_impl.cpp's submit
 * cross-checks the two and aborts on disagreement rather than silently
 * submitting the wrong SPMD width.
 */
struct esl_task_desc {
    uint32_t id;
    task_type_t type;
    org_mode_t mode;
    uint32_t index;
    uint32_t count;
    uint32_t duration;
    uint32_t tensor_cnt;
    uint32_t scalar_cnt;
};

extern struct esl_task_desc g_basic_buf[RING_SIZE];
extern uint32_t g_task_id;

int new_task(uint32_t task_id, uint32_t type, uint32_t count, uint32_t duration);
void add_scalar(uint32_t task_id, int64_t value);

/* --- tensor.h ----------------------------------------------------------- */

EslTensor tensor_from_base_layout(uint64_t base, const uint32_t shapes[], uint32_t ndims, dtype_t dtype);
EslTensor esl_view_at(const EslTensor *t, uint32_t off0, uint32_t off1, uint32_t n0, uint32_t n1);

#ifndef __cplusplus
/* esl_proxy's own spelling: a macro so the call site keeps by-value syntax
 * while the source Tensor is passed by address (its tensor.h:139).
 *
 * C-side ONLY. `view` is also a member function of ChipTensor
 * (src/common/task_interface/tensor.h:281), and an object-like macro named
 * `view` mangles that declaration for every C++ TU that sees both headers.
 * The C++ implementation calls esl_view_at / ChipTensor::view directly. */
#define view(t, off0, off1, n0, n1) esl_view_at(&(t), (off0), (off1), (n0), (n1))
#endif

/* --- tensormap.h -------------------------------------------------------- */

void tm_deps_init(void);

void tm_in_ptr(uint32_t tid, const EslTensor *t);
void tm_out_ptr(uint32_t tid, const EslTensor *t);
void tm_inout_ptr(uint32_t tid, const EslTensor *t);
void tm_in_ro_ptr(uint32_t tid, const EslTensor *t);
void tm_out_ro_ptr(uint32_t tid, const EslTensor *t);
void tm_inout_ro_ptr(uint32_t tid, const EslTensor *t);
void tm_submit_ptr(uint32_t tid);

#define tm_in(tid, t) tm_in_ptr((tid), &(t))
#define tm_in_ro(tid, t) tm_in_ro_ptr((tid), &(t))
#define tm_out(tid, t) tm_out_ptr((tid), &(t))
#define tm_out_ro(tid, t) tm_out_ro_ptr((tid), &(t))
#define tm_inout(tid, t) tm_inout_ptr((tid), &(t))
#define tm_inout_ro(tid, t) tm_inout_ro_ptr((tid), &(t))
#define tm_submit(tid) tm_submit_ptr((tid))

/* --- mem_pool.h --------------------------------------------------------- */

/* Signature is esl_proxy's verbatim, `int dim` / `int bytes` included: `bytes`
 * is the dtype (see dtype_t above) and `dim` is the rank at every call site. */
EslTensor alloc_tensors(uint32_t shape[], int dim, int bytes);

/* --- driver-side accounting (not part of esl_proxy) --------------------- */

/*
 * Read back after the entry returns so the report can state the case's own
 * numbers — task count, SPMD subtask total (esl_proxy's g_subtask_cnt), and the
 * sum of the DUR_* the case attaches — without the case being modified to
 * export them.
 */
struct esl_stats {
    uint64_t tasks;
    uint64_t subtasks;
    uint64_t duration_ns;
    uint64_t allocs;
    uint64_t tracked_args;  /* tm_in / tm_out / tm_inout       — build edges */
    uint64_t no_dep_args;   /* tm_*_ro                          — build none  */
    uint64_t scalars;
};

struct esl_stats esl_get_stats(void);
void esl_reset_state(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESL_SHIM_C_ABI_H */
