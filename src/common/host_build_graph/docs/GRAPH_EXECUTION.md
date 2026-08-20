# Graph Execution

Graph Execution is available only in the `host_build_graph` runtime. A Graph is
a composite incore task: it is submitted and completed once like an AIC, AIV,
MIX, or SPMD task, but contains a recorded task DAG.

Every invocation places exactly one `GRAPH` task in the host task window. On a
first miss, the caller immediately submits an outer task shell keyed by Graph
identity while a worker records the DAG off the ring. Internal submissions
build host-only node metadata and assign output addresses from a private bit-63
virtual range instead of consuming task-window slots or heap. Later calls for
the same in-flight identity submit more shells without waiting for recording.
Before leaving that consecutive Graph batch, the caller joins the worker and
fills every shell's heap range and Definition content hash. Cached invocations
submit the same one `GRAPH` task directly. In both cases the device Scheduler
expands the saved topology and dispatches the internal nodes; the Host
Orchestrator never submits those nodes as ring tasks.

Boundary contracts are checked before an in-flight shell is accepted. Once a
shell has entered the task/dependency sequence, an unsupported construct found
by asynchronous recording is terminal for that orchestration; it cannot be
replayed as ordinary tasks without rolling back already assigned task IDs and
TensorMap producers.

## API

A Graph uses `CoreTaskArgs`, the existing incore argument type:

```cpp
void graph_function(const CoreTaskArgs &args, int variant) {
    const ChipTensor &input = args.tensor(0).ref();
    const ChipTensor &weight = args.tensor(1).ref();
    const ChipTensor &output = args.tensor(2).ref();

    const std::array<uint32_t, 1> shape{input.shapes[0]};
    TensorCreateInfo intermediate(
        shape.data(), static_cast<uint32_t>(shape.size()), input.dtype
    );

    CoreTaskArgs matmul_args;
    matmul_args.add_input(input, weight);
    matmul_args.add_output(intermediate);
    matmul_args.copy_scalars_from(args, 0, 1);  // current invocation's value
    TaskOutputTensors matmul = rt_submit_aic_task(
        variant == 0 ? FUNC_MATMUL : FUNC_MATMUL_TRANSPOSED,
        matmul_args
    );

    CoreTaskArgs activation_args;
    activation_args.add_input(matmul.get_ref(0));
    activation_args.add_output(output);
    rt_submit_aiv_task(FUNC_ACTIVATION, activation_args);
}

void submit_layer(const CoreTaskArgs &args) {
    rt_submit_graph(&graph_function, args, /*variant=*/0);
}
```

The function pointer is the default Graph identity. Trailing integral,
`float`, `double`, and `bool` construction parameters are forwarded to the
Graph function and hashed by value into the cache key. They are separate from
execution scalars in `CoreTaskArgs`: changing a construction parameter selects a
different Definition rather than patching an existing one.

An explicit identity is available for call sites that need a stable name:

```cpp
rt_submit_graph(
    GRAPH_KEY("qwen_decoder_layer_v1"),
    &graph_function,
    args,
    /*variant=*/0
);
```

An explicit `GRAPH_KEY` must be unique for every distinct Graph function in an
orchestration callable. The explicit-key overload deliberately excludes the
Graph function pointer from the cache identity so the key remains stable; using
the same key for different functions can select the wrong recorded topology.

There are no public `GraphArgs`, `GraphBindings`, `Patch`, or `ScalarRef`
types. The boundary is represented by `CoreTaskArgs`.

Boundary scalars are pass-through bindings. Forward them directly with
`node_args.add_scalar(args.scalar(i))` or `copy_scalars_from(args, i, count)`
so recording can retain their source indices.

Ordinary C++ value transformations do not retain boundary provenance. Both
`node_args.add_scalar(args.scalar(i) + 1)` and copying `args.scalar(i)` into a
local arithmetic variable before calling `add_scalar` produce an ordinary
static node scalar. That value is stored in the Definition, and later cache
hits reuse the first invocation's value without a warning. The runtime cannot
distinguish such a derived value from an intentional static literal after the
C++ expression has produced a plain arithmetic value. Compute the derived value
before constructing the Graph boundary and pass it as another boundary scalar,
perform the transformation in a kernel, or use a construction parameter when
the value changes the Graph structure.

Access through a non-const `scalar()` invalidates inherited boundary provenance
conservatively, because returning a mutable reference cannot distinguish a read
from a later write. A Graph containing such an invalidated binding is not
cached, which prevents replay from silently replacing the transformed value
with the unmodified boundary value.

## Supported dynamic and static data

- Boundary ChipTensor addresses may change for every invocation.
- Boundary scalar values may change for every invocation. Their count is fixed
  by the recorded boundary contract. Unused boundary scalars are allowed and do
  not create internal scalar patches.
- A Graph boundary contains at least one ChipTensor.
- Construction parameters are part of Graph identity and may control the
  function's task count, kernel selection, or other structural choices.
- Boundary ChipTensor shape, stride, dtype, size, direction, contiguity, and alias
  partition must match the first invocation.
- Internal task scalars with no boundary source are fixed Definition data.
- Boundary storage is caller-owned. `INPUT`, `INOUT`, `OUTPUT_EXISTING`, and
  `NO_DEP` are supported. A boundary `TensorCreateInfo` tagged `OUTPUT` is not.
- Early-resolve hints apply while recording the first invocation. Replayed
  internal nodes use the saved completion topology without the hint.
- A recorded task may depend on a Graph-external producer when that producer
  is the creator of a boundary ChipTensor. The outer Graph owns that dependency on
  replay; arbitrary cross-boundary explicit dependencies remain unsupported.
- A recorded task may carry a dispatch predicate. Submit resolves a predicate
  into an absolute GM address, which no Definition can hold, so the Definition
  stores the operand tensor's classified source plus the element index within
  it and materialize resolves the pair per execution. The operand may be a
  boundary ChipTensor or another node's output; the predicate itself creates no
  dependency, exactly as on the ordinary path, so the caller still declares one
  on the operand's producer.

Structural or alias mismatch logs a warning and executes the Graph function
normally for that invocation. It never reuses heap offsets recorded for a
different shape. Debug builds also assert at these unsupported boundaries so
development catches a violated fixed-shape contract immediately; the ordinary
path remains the defensive release-build behavior.

## Qwen decoder-layer example

The upper layer packages all ChipTensor I/O in `CoreTaskArgs`; the wrapper has no
separate `hidden`, `weight`, or `output` parameters:

```cpp
void qwen_decoder_layer(const CoreTaskArgs &args) {
    const ChipTensor &hidden = args.tensor(0).ref();
    const ChipTensor &attention_weight = args.tensor(1).ref();
    const ChipTensor &mlp_weight = args.tensor(2).ref();
    const ChipTensor &output = args.tensor(3).ref();

    const std::array<uint32_t, 1> hidden_shape{hidden.shapes[0]};
    TensorCreateInfo attention_out(
        hidden_shape.data(), static_cast<uint32_t>(hidden_shape.size()), hidden.dtype
    );

    CoreTaskArgs attention_args;
    attention_args.add_input(hidden, attention_weight);
    attention_args.add_output(attention_out);
    attention_args.copy_scalars_from(args, 0, 1);  // dynamic token position
    TaskOutputTensors attention =
        rt_submit_aic_task(FUNC_ATTENTION, attention_args);

    MixedKernels mlp;
    mlp.aic_kernel_id = FUNC_MLP_AIC;
    mlp.aiv0_kernel_id = FUNC_MLP_AIV;

    CoreTaskArgs mlp_args;
    mlp_args.add_input(attention.get_ref(0), mlp_weight);
    mlp_args.add_output(output);
    rt_submit_task(mlp, mlp_args);
}

void submit_qwen_decoder_layer(const CoreTaskArgs &args) {
    rt_submit_graph(&qwen_decoder_layer, args);
}

void decode_three_layers(
    const std::array<ChipTensor, 3> &hidden,
    const std::array<ChipTensor, 3> &attention_weight,
    const std::array<ChipTensor, 3> &mlp_weight,
    const std::array<ChipTensor, 3> &output,
    const std::array<uint32_t, 3> &token_position
) {
    for (std::size_t layer = 0; layer < hidden.size(); ++layer) {
        CoreTaskArgs args;
        args.add_input(
            hidden[layer],
            attention_weight[layer],
            mlp_weight[layer]
        );
        args.add_output(output[layer]);
        args.add_scalar(token_position[layer]);
        submit_qwen_decoder_layer(args);
    }
}
```

All three layers submit one Graph task each. The first starts background
recording, while layers two and three immediately submit outer task shells for
the same in-flight identity. The first following non-Graph operation (or
orchestration completion) joins recording and finalizes all three shells with
the new Definition. Each invocation patches the current layer's
`token_position`; it is a dynamic boundary scalar refreshed on every submission
and is not part of the Graph key.

## Definition

Recording uses host-only C++ state:

- `std::vector` for nodes, tensors, scalars, fanins, and pending uploads;
- `std::unordered_map` for the per-run Definition cache;
- `std::unique_ptr` for the active recording, guarded by a mutex and completion
  condition while the recording worker publishes the Definition.

The cache stores at most 16 Definitions and allocates each entry to its actual
serialized size. No fixed maximum-size recording array is copied on a cache
hit.

Recorded output addresses start at `GRAPH_RECORD_VIRTUAL_BASE = 1ULL << 63`.
They exist only to classify `OWN_OUTPUT` and `INTERNAL` Tensor sources and are
converted to offsets in the Definition; they are never dereferenced or placed
on the wire. Classification is by address-range containment alone, so it is only
sound while no real heap address can fall in that range:
`PTO2TaskAllocator::init()` asserts the whole configured heap lies below
`GRAPH_RECORD_VIRTUAL_BASE`, which makes an overlapping device GM heap a loud
failure at setup instead of a silently misclassified Tensor source. Recording
therefore leaves the shared task allocator unchanged.

### First-miss host threading

`graph_begin` computes the Graph identity before the body is recorded. On a
cache miss, the calling thread allocates a zero-heap outer task shell, records
its boundary dependency edges, and returns. A worker receives a deep copy of
the boundary arguments, records the internal nodes in the private virtual
address range, and builds and hashes the Definition. The first call waits only
until that private job has been installed in the worker queue; it does not wait
for the operating system to schedule the worker or for `graph_prepare` to bind
the private recording state. The hash-keyed `RECORDING` entry and zero-heap
outer shell already exist before the job is enqueued, so later same-identity
submissions can safely proceed immediately. The worker remains parked on a
condition variable for the lifetime of the loaded orchestration SO and is
reused by later runs, so only its first miss pays thread creation. Unloading the
SO stops and joins the idle worker before its code is unmapped.

The queue handoff publishes the already-created recording to `graph_prepare`.
Prepare therefore does not reacquire the Definition-state mutex: until the
worker ends or aborts, later same-identity submissions only read the immutable
boundary signature under that mutex. Avoiding the redundant acquire prevents
the short main-thread submit loop from starving the worker before it can enter
its private recording state.

Calls for the same identity while recording is in flight follow the same shell
submission path on the calling thread. Their task IDs and TensorMap producers
therefore enter the ordinary program-order sequence while the worker is still
executing `record_node` and `build_definition`.

`rt_graph_commit` is a barrier before a non-Graph operation or orchestration
completion. It waits for the current worker job, reserves each shell's real heap
block in task order using the Definition's `required_heap`, patches the task
descriptor and Definition content hash, and then permits ordinary submission to continue. A
scope transition is deliberately not a barrier: the main thread has already
submitted the outer Graph shell into that scope, while scopes executed by the
recording worker are no-ops on the real scope stack. This lets consecutive
Graph calls in separate outer scopes continue submitting while recording is in
flight. No unrelated heap allocation can appear between a deferred shell and
this finalization. Cache-hit submissions after the barrier already know
`required_heap` and use the ordinary immediate path.

Host phase records therefore show `graph_submit` on the main lane overlapping
`record_node` and `build_definition` on the recording-worker lane.

At `graph_end`, recording is compacted into one contiguous, pointer-free POD
Definition. It contains:

- node order and AIC/AIV/MIX/SPMD kernel metadata;
- `root_indices` plus both directions of the immutable topology:
  fanin CSR and fanout CSR;
- one packed-heap offset per node;
- each node's ChipTensor source:
  `BOUNDARY_EXACT`, `BOUNDARY_VIEW`, `INTERNAL`, or `OWN_OUTPUT`;
- fixed scalar values plus boundary-scalar source indices;
- fixed boundary signatures and alias representatives.

The header also carries a content hash of the complete Definition image. The
device execution pool requires this hash, the Graph key, and the node count to
all match before reusing a resident Definition. A new run may record different
metadata under the same function identity, so key-only reuse is not safe.

All references are 32-bit offsets from the Definition base. Cross-boundary
Tensors use the fixed-width `GraphTensor` wire POD rather than the
64-byte-aligned C++ `ChipTensor` object. The upload is therefore one contiguous
copy with no raw Host pointers and no relocation pass.

Before materialization, the Scheduler recomputes the Definition content hash
and validates section ranges, topology indices, node heap offsets, the outer
heap extent, ChipTensor metadata, and ChipTensor-source bounds. Invalid wire data is
rejected before an offset participates in pointer arithmetic.

There is no cache schema version. The cache is per run and starts empty, so a
persistent-format version would currently have no effect.

## Cache hit and memory

For a cache hit, the Host Orchestrator:

1. validates the fixed boundary contract;
2. reserves one task-window slot;
3. reserves one heap block large enough for every internal intermediate, plus
   the Definition's `execution_storage_bytes` for the execution the device
   materializes into;
4. computes only external fanin and boundary tensormap effects;
5. emits one outer `GRAPH` task;
6. stages the exact-size POD submission image for upload after orchestration.

Internal nodes consume no ring task-window slots. Their descriptor, payload,
and slot state live in the tail of the outer `GRAPH` task's own heap block,
past `required_heap`: one `PTO2TaskAllocator::alloc` covers both halves the
task owns, so the storage is reclaimed with the task's packed outputs and
needs no separate device allocation, retention keying or release path.

The execution address is therefore not on the wire — both sides compute
`packed_buffer_base + required_heap` and read the size from the Definition. The
Scheduler validates that the region lies inside the outer task's allocation and
then placement-constructs `GraphExecution` there; it never allocates execution
storage from the AICPU process heap, and it never reads the block's prior
contents. Every submission materializes from the Definition, so a resubmission
of the same Graph rebuilds rather than replaying the previous expansion: the
bytes it starts from are whatever that heap region last held.

## Scheduler flow

Host orchestration builds the complete task image before device execution. At
the end of orchestration, the Host uploads every exact-size Graph POD image,
relocates the task and payload pointers in the shared-memory image, copies the
complete shared-memory/runtime-arena image to the device, and then launches the
resident Scheduler.

All AICPU threads classify disjoint slices of the completed task window behind
one startup barrier. A Graph task enters preparation and external-fanin
classification during that scan, so Graph execution is interleaved with other
ready tasks at the same scheduling level once the Scheduler starts.

This design does not overlap orchestration and scheduling within one run.
Prepared-successor pipelining can overlap preparation of run N+1 with device
execution of run N, while Graph cache hits reduce repeated orchestration work
inside a run.

A Graph is placed in two independent control flows:

- `graph_prepare_queue`: materialize the saved nodes even while external fanin
  is still pending;
- `graph_ready_queue`: signal that the outer Graph's external fanin is ready.

Core-owning Scheduler threads pop at most one item from each queue per loop. A
prepare call expands at most four nodes and requeues unfinished work,
interleaving Graph expansion with normal scheduling.

Preparation and external readiness set two bits in one atomic activation gate.
Whichever operation sets the second bit activates the saved root nodes exactly
once.

Internal dependency readiness borrows the completion-state polling idea, but
dependency wiring remains an Orchestrator responsibility:

- recording constructs both fanin and fanout CSR in the immutable Definition;
- materialization builds each node's runnable state from the Definition,
  resolving its Tensor addresses against the boundary image and its producers'
  packed windows;
- materialization registers each non-root on one producer selected from its
  saved fanin CSR;
- a node's release/acquire `task_state` is its Graph-local completion flag, so
  internal nodes need neither ring completion flags nor task-window slots;
- producer completion closes and drains only its current wake-list rather than
  traversing the saved fanout CSR;
- a woken consumer scans its saved fanin CSR and either enters its shape queue
  or registers on the next incomplete producer;
- `WAKE_LIST_SENTINEL` closes the completion/registration race: a failed
  registration observes completion and immediately rescans.

The runtime wake-list registration is a transient polling subscription, not
dependency discovery or Graph rewiring. Fanout CSR remains in the Definition
as part of the complete recorded topology and for DFX, but readiness does not
walk it.

```text
outer GRAPH
  -> activate root_indices[]
  -> producer completion drains its current wake-list
  -> each waiter polls saved fanin completion state
  -> ready waiter enters its ordinary shape queue
     or registers on another incomplete producer
  -> final internal completion completes the outer GRAPH
```

Internal nodes count as zero outer ring tasks. The final node completes
the one outer Graph task, publishes the outer ring completion flag, wakes
external consumers, and contributes one to the host-visible completion count.

Localization or materialization failure is fail-fast: the Scheduler latches an
error instead of leaving an already-submitted outer Graph unable to complete.

## Current unsupported cases

Conditions detected before an outer shell is accepted use the ordinary path:

- an empty Graph boundary;
- variable ChipTensor shape or metadata;
- changed boundary aliasing;
- runtime-allocated boundary outputs;
- more than 32 boundary Tensors;
- more than 16 Definitions;
- insufficient task-window or known cache-hit heap capacity.

The following constructs are discovered only while the worker records the
first Definition. Because its outer shell is already in the task/dependency
sequence, they assert in debug builds and fail the orchestration in release
builds:

- nested Graph recording;
- cross-boundary explicit dependencies that are not represented by a boundary
  ChipTensor's creator;
- an unclassifiable internal ChipTensor source, including a dispatch
  predicate's operand tensor;
- a dispatch predicate whose operand is the predicated node's own output;
- a dispatch predicate whose index vector leaves the operand tensor's extent;
- a boundary-derived scalar accessed through mutable `scalar()`;
- runtime allocation inside the Graph body;
- more than 1024 internal nodes;
- insufficient heap capacity while deferred shells are finalized.

An AICPU execution-pool or materialization failure happens after the outer
Graph has already been submitted. It therefore latches a Scheduler fatal error
instead of falling back; leaving the outer task pending would otherwise wedge
completion.

Explicit dependencies between recorded internal nodes are preserved when they
are otherwise supported; ordinary ChipTensor dependencies are always preserved.

## DFX

With L2 swimlane level 4:

- `Graph Execution` spans an outer Graph execution;
- `AICPU Scheduler` shows bounded `graph_prepare` slices separately from normal
  dispatch;
- existing Scheduler and Worker lanes show the expanded internal tasks.

The scene coverage under `tests/st/{a2a3,a5}/host_build_graph/graph_execution`
includes AIV fanin/fanout DAGs, architecture-native AIC/AIV decoder-style DAGs,
and three-slot multi-block MIX/SPMD Graphs. Every scene invokes the same fixed
Graph three times: one recording execution followed by two outer-Graph
submissions. The full-model example at
`examples/a2a3/host_build_graph/qwen3_14b_decode` records one Qwen3-14B decoder
layer and replays its Definition for the remaining 39 layers.
`tests/st/{a2a3,a5}/host_build_graph/graph_predicated_dispatch` covers dispatch
predicates on both operand sources, giving each invocation its own gate buffer
and gate scalar so a replay that reused the recorded operand address would read
the recording invocation's gates.
