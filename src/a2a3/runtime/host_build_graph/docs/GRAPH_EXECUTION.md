# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. A Graph is
a **composite incore**: like an ordinary AIC / AIV / MIX / SPMD task it is
submitted once and completes once, but internally it drives several hardware
units in a fixed dependency order. The first call records that internal task
DAG; every later call submits **one** `GRAPH` task instead of re-submitting the
whole sub-DAG on the host. The device Scheduler expands the recorded topology
with the current call's tensor addresses and runs it.

Because a Graph is just another task kind, its submission arguments are
`L0TaskArgs` — the same type every other incore takes — not the orchestration
entry frame.

| Aspect | ordinary incore | GRAPH |
| ------ | --------------- | ----- |
| task kind | `KERNEL` / `DUMMY` | `GRAPH` |
| ring task-window slots | 1 | **1** (outer only; internal nodes take none) |
| heap allocations | 1 | **1** (all internal intermediates in one block) |
| boundary dependency wiring | `compute_task_fanin` + tensormap | identical |
| completion accounting | 1 stream task | **1** (internal nodes count 0) |

## Two kinds of scalar

A Graph has two kinds of scalar, on two separate channels. The distinction
drives the API, the cache key, and what may be patched:

- **Construction parameters (config: `int` / `float` / `bool`).** Consumed by
  the Graph function's own control flow at build time — loop counts, branch
  selection, how many tasks are submitted, which kernel. They determine the
  graph's *structure*, so a different config value is a different graph. They
  are part of the identity and are never patched.
- **Execution scalars (TaskArg scalars).** Values passed into kernels via
  `add_scalar(...)` — runtime data that does not change structure. They travel
  in `L0TaskArgs`, are literal (step 1) or patched per call (step 2), and are
  **never** part of the identity.

The rule of thumb: **in the key = "if it changes, it must be a different
graph"; patched = "if it changes, it is still the same graph"**.

## Scope

This document describes **step 1**, the minimal safe core:

- **Dynamic tensor addresses; config in the key; execution scalars are
  literals.** Boundary tensor addresses are patched every call; construction
  parameters enter the cache key by value; an execution scalar must be a
  function-body literal.
- **Fixed boundary structure.** The supported scenario has determinate input
  shapes. A boundary whose structural metadata differs from the recording is
  *not* a second graph — it is an unsupported call that falls back to the
  ordinary path (see [Boundary contract](#boundary-contract)).
- Single worker, non-nested, caller-owned boundary storage.

**Deferred to step 2:** dynamic execution scalars — carrying a per-invocation
TaskArg scalar (e.g. a Qwen `layer_id` / `token_position`) through `L0TaskArgs`
and patching it at expansion time. See [Roadmap](#roadmap).

## API

A Graph has two argument channels: **construction parameters**, folded into the
key and forwarded to the build function, and **`L0TaskArgs`**, the execution I/O
boundary.

```cpp
// L0TaskArgs is the execution boundary; trailing parameters are config.
void layer(const L0TaskArgs &io, int variant) {
    const Tensor &input  = io.tensor(0).ref();
    const Tensor &weight = io.tensor(1).ref();
    const Tensor &output = io.tensor(2).ref();

    uint32_t shape[1] = {input.shapes[0]};
    TensorCreateInfo intermediate(shape, 1, DataType::FLOAT32);

    L0TaskArgs mm;
    mm.add_input(input, weight);
    mm.add_output(intermediate);         // runtime-allocated intermediate
    mm.add_scalar(uint32_t{16});         // execution scalar (literal in step 1)
    Tensor hidden = rt_submit_aic_task(
        variant == 0 ? FUNC_MATMUL : FUNC_MATMUL_T, mm).get_ref(0);

    L0TaskArgs act;
    act.add_input(hidden);
    act.add_output(output);              // writes the caller's boundary buffer
    rt_submit_aiv_task(FUNC_ACT, act);
}

void submit_layer(const Tensor &in, const Tensor &w, const Tensor &out) {
    L0TaskArgs io;
    io.add_input(in, w);
    io.add_output(out);                  // caller-owned, already allocated
    rt_submit_graph("layer", &layer, io, /*variant=*/0);
}
```

`rt_submit_graph(name, fn, io, config...)` builds the key from `name` (or the
function-pointer identity), the orchestration-callable hash, and the config
values **by value**, then calls `fn(io, config...)`. A Graph with no config is
just `rt_submit_graph(&fn, io)`.

There are no public `GraphBindings`, `Patch`, or `GraphArgs` types.

## Cache key

The key is deliberately minimal — identity is **what the function is** plus
**how it was configured**:

```text
key = hash(name | function-pointer identity,
           orchestration-callable hash,
           config values by value)
```

Nothing about the tensors enters the key: addresses are dynamic and patched,
and structural metadata is handled as a contract instead (below). Execution
scalar values are never in the key — in step 1 they are literals already covered
by the function identity, and in step 2 they are patched, so their value cannot
be part of identity.

## Boundary contract

The caller owns two invariants across every call of a given Graph. The runtime
records the recorded boundary's signature in the definition and re-checks it
cheaply on each cache hit, but it does not treat proving them as its job — a
violation is a caller error, surfaced defensively.

| Contract | Detection | Action on mismatch | Why |
| -------- | --------- | ------------------ | --- |
| **Structural metadata** — `shapes[]`, `strides[]`, `ndims`, `dtype`, size, `tag`, `manual_dep`, `is_contiguous` | field compare against the recorded signature | **`LOG_WARN` and fall back to the ordinary submit path** for this call (the cache is not used) | The recording baked `total_output_size`, `node_offsets[]`, and `required_heap` from these values. Replaying with a different shape would slice a block sized for the old one — a memory-safety fault, not a bounded wrong value. |
| **Alias partition** — which boundary slots share a buffer | recompute `rep[i] = min{ j<=i : args[j] shares args[i]'s buffer }` and compare | **`LOG_WARN` and proceed with the cached graph** | An internal tensor is matched to its boundary slot by address; when two slots alias, first-match resolves it by lowest index, valid only for the recorded alias pattern. The blast radius is bounded (a mis-bound pointer, not a wrong size) and the case is rare. |

Both are the same philosophy — caller contract plus cheap runtime validation —
differing only in blast radius: what can corrupt memory refuses the cache, what
is bounded only warns.

## Memory model

Four distinct memory domains are involved. Knowing which is which explains both
what a Graph saves and where it can fail.

| Memory | Allocated by | When | Lifetime |
| ------ | ------------ | ---- | -------- |
| **Boundary tensors** (graph I/O) | **the caller** | **before** the submit | caller-owned |
| **Internal intermediates** (node outputs created from `TensorCreateInfo`) | host, **one** `alloc(required_heap)` on the outer GRAPH task | at submit | tied to the outer task — reclaimed only after *all* nodes complete |
| Node outputs written straight to a boundary buffer (`OUTPUT_EXISTING`) | not allocated | — | caller-owned |
| **Internal nodes' TaskArgs storage** (descriptor + payload + slot state) | AICPU-local execution pool | at device-side localize | tied to the execution block; reusable across calls |

### The boundary is caller-owned storage

Boundary tensors must already exist when the Graph is submitted. A boundary
argument tagged `OUTPUT` — i.e. a runtime-allocated `TensorCreateInfo` — is
**rejected as uncacheable** and the call falls back to the ordinary path. Only
`INPUT` / `INOUT` / `OUTPUT_EXISTING` / `NO_DEP` are supported at the boundary.
This is what makes per-call address patching possible: the caller supplies fresh
addresses each time and the expansion simply re-points at them.

### Internal intermediates: one block, sliced by recorded offsets

During recording, each node's runtime-allocated output size is accumulated and
its offset saved:

```text
node_offsets[i] = required_heap
required_heap  += align_up(node_i_total_output_size)
```

At submit, the outer GRAPH task performs a single `alloc(required_heap)`; at
expansion each node's packed buffer is a slice `outer_base + node_offsets[i]`,
and an internal tensor's address is `producer_packed_base + packed_offset`.

**One block is structurally required, not an optimization.** The ring heap
allocator is owned by the *host* during orchestration; the device Scheduler that
materializes nodes has no allocator to call. Every byte the graph will use on
device must therefore be reserved by the host at submit time, in one
pre-computed block. This is also why `node_offsets[]` is baked into the
definition — and why a boundary shape change must refuse the cache rather than
silently reuse offsets computed for a different size.

Only runtime-allocated outputs (`OUTPUT`) count toward `required_heap`; a node
writing directly into a boundary buffer contributes nothing.

**Trade-off — peak heap.** The block's lifetime is tied to the outer task, so
internal intermediates cannot be reclaimed early: peak usage is the *sum* of all
intermediate outputs rather than a live-range optimum. Submission preflight
rejects the graph when `required_heap` exceeds available heap, falling back to
the ordinary path. Long chains whose early outputs die quickly are poor Graph
candidates.

### Internal nodes cost no task-window slots and are never uploaded

An ordinary task's descriptor, payload, and slot state live in the ring task
window in shared memory, and are uploaded with it. A Graph's internal nodes do
not: only the outer task occupies a window slot, and each node's storage

```cpp
struct alignas(64) GraphNodeStorage {
    TaskDescriptor task;
    TaskPayload    payload;   // ~4 KiB: worst-case tensor/scalar/fanin arrays
    TaskSlotState  slot;
};
```

is built on the device, in an AICPU-local execution block. This is where the
scaling win comes from — `N` window slots collapse to 1, and `N` payloads leave
the upload path entirely; only the compact definition plus the boundary args are
uploaded.

**The AICPU-local pool is the new bounded resource.** It is capped (16 MiB / 64
blocks), and at roughly 4 KiB per node a 1024-node graph consumes about 4 MiB,
so only a few large graphs can be resident concurrently. Localization can
genuinely fail under pressure; that path must fail fast (see
[Failure](#failure-is-fail-fast-never-a-wedge)), never leave a submitted outer
task unable to activate. Shrinking `GraphNodeStorage` is the highest-value
follow-up — see [Roadmap](#roadmap).

## Two representations

The DAG has two forms with different jobs and therefore different storage rules
(see `.claude/rules/codestyle.md` §8).

**Recording form — host-only, C++/STL.** Built while the Graph function runs on
the host. It never leaves the host, so it carries no POD constraint: standard
containers sized to the actual graph, with no worst-case fixed-array globals.

```cpp
struct RecordedNode {
    int32_t   kernel_id[SUBTASK_SLOT_COUNT];
    ActiveMask active_mask;
    int16_t   logical_block_num, total_required_subtasks;
    int32_t   task_timing_slot, total_output_size;
    uint64_t  record_packed_base;
    std::vector<Tensor>          tensors;
    std::vector<TensorSourceRef> tensor_sources;
    std::vector<TensorArgType>   tensor_arg_types;
    std::vector<uint64_t>        scalars;      // execution scalars (literals)
    std::vector<FaninRef>        fanins;
};
```

**Definition form — position-independent POD.** The recording is compacted into
one contiguous block that is uploaded to the device. Every internal reference is
a `uint32_t` offset from the block base, **never a raw pointer**, so a single
`memcpy` relocates the whole image with zero fix-up and any consumer that knows
its own base can read it.

```cpp
struct GraphDefinition {          // POD: memcpy-able, position-independent
    uint64_t full_key;
    int32_t  task_count, edge_count, root_count;
    uint32_t total_bytes;
    ReplayPlan replay_plan;
    uint32_t off_fanout_offsets;  // uint32_t[task_count+1]  CSR
    uint32_t off_fanout_indices;  // uint16_t[edge_count]
    uint32_t off_fanin_counts;    // uint16_t[task_count]
    uint32_t off_root_indices;    // uint16_t[root_count]
    uint32_t off_node_offsets;    // uint64_t[task_count]     packed-heap offsets
    uint32_t off_nodes;           // NodeDef[task_count]
    uint32_t off_tensors;         // Tensor[sum tensor_count]
    uint32_t off_tensor_sources;  // TensorSourceRef[sum tensor_count]
    uint32_t off_scalars;         // uint64_t[sum scalar_count]
    uint32_t off_boundary_sig;    // recorded boundary signature + alias rep[]

    template <class T> const T *ptr(uint32_t off) const {
        return reinterpret_cast<const T *>(
            reinterpret_cast<const char *>(this) + off);
    }
};
static_assert(std::is_trivially_copyable_v<GraphDefinition> &&
              std::is_standard_layout_v<GraphDefinition>);
```

Because the definition is immutable, one uploaded copy can be shared by every
invocation of that graph.

## Record, materialize, activate

Three stages turn a blueprint into running work:

| Stage | What it is |
| ----- | ---------- |
| **definition** | The immutable recorded blueprint — topology plus static data, with no per-call addresses. |
| **materialize** | The device Scheduler builds the *runnable* objects for this call: for each node it fills a real descriptor / payload / slot and patches the dynamic parts — tensor addresses from the boundary or from a producer node's slice, literal scalars copied in. "Materialize a graph" = turn its definition into live, dispatchable scheduler tasks. |
| **activate** | Once materialized *and* the outer task's external dependencies are met, the saved root nodes are pushed to the ready queue and execution begins. |

**Record (cache miss).** `rt_submit_graph` runs the function normally; every
submitted task is captured *and* executes as usual, so a miss costs nothing
beyond the recording. At scope end the recording is compacted into a definition.

**Submit (cache hit).** The Orchestrator allocates `required_heap`, computes only
the Graph's *external* dependencies, wires the boundary exactly like an ordinary
task, and places one `GRAPH` descriptor in the task window. The upload carries
the compact definition plus this call's boundary args — no expanded node array.

**Materialize.** On first observation of the outer task a Scheduler thread takes
an AICPU-local execution block, `memcpy`s the definition into it, and
materializes **four nodes per scheduling slice** while the outer Graph waits for
external fanin. Slicing keeps expansion from monopolizing the scheduler.

**Activate — exactly once.** Preparation and external readiness arrive
independently, possibly on different threads. They meet at a single atomic gate:
each side sets its bit with `fetch_or`, and whichever call observes *both* bits
now set performs the activation.

```cpp
enum : uint8_t { GATE_PREPARED = 1, GATE_EXTERNAL = 2, GATE_BOTH = 3 };

if ((gate.fetch_or(GATE_PREPARED, std::memory_order_acq_rel) | GATE_PREPARED)
    == GATE_BOTH) {
    activate_roots(exec);
}
```

Because `fetch_or` is an atomic read-modify-write, exactly one side can observe
the transition to `BOTH`. There is no interleaving in which each side misses the
other, so a graph can never sit prepared-but-never-activated.

**Execute and complete.** A root node has no internal fanin and is routed at
activation; every other node is held and released along the saved CSR fanout
edges as its producers complete. The outer Graph completes only after all
internal nodes retire, so downstream tasks observe it as one normal dependency
producer, and the whole graph counts as **one** completed stream task.

### Failure is fail-fast, never a wedge

If materialization cannot proceed — AICPU-local allocation or definition copy
fails under pool pressure — the Scheduler latches an error and fails the run with
a clear code. It must never leave the outer Graph counted as submitted yet unable
to activate, which would hang until the op-execute timeout.

## Lifetime, ownership, concurrency

Host orchestration for one worker is single-threaded, so the recording state, the
definition cache, and the host submission pool are **owned per worker** (per
runtime instance), not process-global; two workers can never corrupt each other's
capture. The AICPU-local execution pool is the only multi-threaded pool and is
guarded by a spinlock — the asymmetry is deliberate and stated, not accidental.

The execution pool prefers a block previously used by the same Graph key; that
graph-affine path keeps the local definition and static node fields and refreshes
only task IDs, tensor addresses, packed-buffer bases, and scheduling state.
Completed executions are reclaimed once their internal nodes retire.

If a task inside a recording Graph latches a fatal error, the capture state is
reset **unconditionally and idempotently at run end**, and `graph_begin` force-
resets any leftover active flag. Otherwise a stale "recording active" flag would
survive the run and make every later Graph report a nested-scope violation,
disabling the feature permanently.

## Unsupported constructs

Detected during recording or at the boundary check; the call falls back to the
ordinary submit path — correct, just not accelerated — with one warning:

- boundary structural metadata differing from the recording;
- runtime-allocated (`OUTPUT`) boundary tensors;
- nested Graph scopes;
- explicit dependencies crossing the Graph boundary;
- dispatch predicates;
- an internal tensor whose source cannot be classified;
- fanin / argument overflow, or `required_heap` exceeding available heap.

Current limits: 16 definitions, 1024 nodes per definition, 32 boundary tensors.

## Naming

New identifiers in this subsystem carry no `PTO2` prefix (`.claude/rules/`
`codestyle.md` §9): `GraphDefinition`, `GraphExecution`, `GraphSubmission`,
`GRAPH_KEY`, `GRAPH_CACHE_SCHEMA_VERSION`. Disambiguation comes from a namespace,
not a tag.

## DFX lanes

`--enable-l2-swimlane 4` produces five logical lanes in the converted Perfetto
JSON:

- `Host Orchestrator`: one envelope for the host build/submit interval, with
  `task_submit(tN)` children for cache-miss nodes and one `graph_submit(tN)`
  child per cache-hit Graph invocation;
- `Graph Execution`: one envelope per outer Graph task, from its first prepare
  slice through its last visible node;
- `AICPU Scheduler`: `graph_prepare` is reported separately from `dispatch`, so
  expansion cost is visible without inflating dispatch time;
- `Scheduler View` and `Worker View`: the existing lanes.

Host and device counters are separate clock domains. The converter places Host
Orchestrator immediately before device time zero, then rebases the trace so it
starts at trace time zero and every device event follows — preserving ordering
without implying clock synchronization.

## Roadmap

**Dynamic execution scalars (step 2).** In step 1 an execution scalar must be a
function-body literal, so a Graph whose kernel scalars vary per call (the Qwen
decode pattern, where `layer_id` and `token_position` change every layer/token)
cannot pass them through and stay cached. Step 2 lets an execution scalar read
from `L0TaskArgs` carry its boundary source index, and patches it at expansion
exactly like a tensor address. The key is unchanged — execution scalar values
were never in it — and construction parameters stay on the config channel.

**Variable-length node storage.** `GraphNodeStorage` currently embeds worst-case
payload arrays (`MAX_TENSOR_ARGS` = 32) at roughly 4 KiB per node, while a
typical node has two or three tensors. Packing each node's payload to its actual
counts with the same offset-based technique used for the definition would shrink
this by close to an order of magnitude, directly raising how many graphs fit in
the AICPU-local pool. The cost is one extra indirection when addressing a node's
tensor array during expansion.

**Runtime-allocated boundary outputs.** Chaining graphs (graph A's output feeding
graph B) requires the runtime to allocate a boundary output and return its
address to the caller. Step 1 rejects that form at the boundary.

## Schema version

`GRAPH_CACHE_SCHEMA_VERSION` is folded into the key so that changing the
definition layout invalidates old entries. The definition cache is per worker and
starts empty each run, so today this has no observable runtime effect; it earns
its keep only if the cache ever becomes persistent or shared across binaries.
Treat it as reserved for that future, not as active protection.
