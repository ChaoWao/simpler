# `host_build_graph` Runtime Design

## 1. Execution Model

`host_build_graph` separates graph construction from device execution:

```text
host register: materialize + dlopen orchestration SO
        ↓
host run/bind: stage external tensors, execute orchestration to completion
        ↓
host: relocate the prebuilt graph image and copy it to device memory
        ↓
device: attach the image, classify tasks, dispatch with AICPU schedulers
        ↓
host: collect outputs and destroy/reset per-run state
```

The device has no orchestration thread. Every launched AICPU thread participates
in scheduling its assigned AICore workers; the highest-index thread first
attaches the prebuilt runtime and publishes the boot barrier.

This ordering is the defining constraint of the runtime. The host constructs the
whole graph before any device task can complete.

## 2. Callable and Run Lifecycle

### 2.1 Registration

`register_callable_impl` materializes the orchestration bytes as a temporary
shared object, opens it with `dlopen(RTLD_LOCAL)`, and resolves:

- `aicpu_orchestration_config`;
- `aicpu_orchestration_entry`; and
- `framework_bind_runtime`.

The resolved handle and entry points belong to the registered callable and stay
alive across prepared runs. Unregister/context teardown closes the handle and
removes its temporary file.

Child AIC/AIV callables are uploaded separately. Their resolved device function
addresses populate the per-run dispatch table.

### 2.2 Host Graph Construction

For each run, the host:

1. validates and stages external tensors into a run-owned host accessor;
2. reserves one backing arena for runtime/shared-memory subregions;
3. binds the runtime to the orchestration DSO;
4. calls the orchestration entry synchronously;
5. finalizes task counts and the graph image;
6. rewrites per-slot task/payload pointers to device addresses; and
7. copies the shared-memory image and arena to the device.

An orchestration fatal stops this sequence and is propagated through
`orch_error_code`.

### 2.3 Device Execution and Teardown

The boot thread attaches the already-populated arena without resetting it. All
threads classify/dispatch their core partitions and then shut those cores down.
The last arriving thread destroys the attached runtime before publishing cleanup
eligibility. Exactly one returning AICPU thread claims that eligibility and
resets executor/scheduler state for the next run.

Publishing cleanup only after destruction prevents `deinit()` from racing the
runtime arena or this run's host accessor.

## 3. Prebuilt Graph Image

The shared image uses three per-slot structures:

| Structure | Purpose |
| --------- | ------- |
| `PTO2TaskDescriptor` | Full task ID, kernel IDs, packed-buffer addresses |
| `PTO2TaskPayload` | Tensors, scalars, predicate, local-ID fanins, dispatch metadata |
| `PTO2TaskSlotState` | Active mask, attributes, block/subtask counters, completion state, task/payload bindings |

The host/device boundary is POD and position-independent. Fanins are integer
producer IDs, not pointers. The only per-slot pointers are rebound to their final
device addresses before H2D.

### 3.1 Bounded H2D Upload

The shared-memory mirror is sized to ring capacity (task window) but a run only
writes `[0, total_tasks)`, and the device boots scheduler-only and reads no SM slot
past `total_tasks`. So the SM H2D shipped each run is bounded, not capacity-sized —
the contract that keeps `bind` proportional to the workload.

- **Shared memory** — the header is zeroed on the host; `descriptors`, `payloads`,
  `slot_states` and `completion_flags` are each written per task at submit and
  H2D-uploaded bounded to `[0, total_tasks)`. Per-slot reset is init-on-write in
  `orch::prepare_task` as each slot is claimed — there is no window-wide reset.

- **Runtime arena** — the **orchestrator block** (`fanin_seen_epoch` /
  `scope_tasks` / TensorMap, ~8.5 MB) is **not shipped at all**: it is host-only
  dep-computation scratch, and the AICPU scheduler holds zero references to it. (The
  scalar `inline_completed_tasks` the scheduler does read lives in the runtime
  header, inside the region that still ships whole.) Everything from the scheduler
  block onward — the ready-queue slot pools, runtime header, and completion mailbox
  — ships **whole**. The ready queues are *not* bounded to `total_tasks`: graph
  execution replays a cached GRAPH task that the device Scheduler expands into
  on-device nodes, and those nodes push into the ready queues past the host task
  count, so the queue slots must all carry a valid Vyukov sequence on the device.

`bind_callable_to_runtime_impl` `always_assert`s `orch_start <= orch_end` before
uploading, so a future `runtime_reserve_layout` reorder that moved the scheduler
block ahead of the orchestrator block faults instead of shipping a misaligned image.

## 4. Whole-Graph Capacity

The runtime uses one task ring, one graph heap, and one TensorMap pool. They are
capacity-bounded storage, not streaming flow-control buffers:

- `last_task_alive` does not advance during a run;
- `heap_tail` does not retire task output buffers during a run;
- task slots and heap bytes are never recycled mid-run; and
- TensorMap entries are not reclaimed while host construction is active.

`completed_watermark` records the contiguous prefix of completed device tasks.
It supports completion/consumer metadata only; it does not reclaim the task ring
or heap.

There is no post-run sweep that makes graph space reusable. Runtime destruction
releases the complete arena, and the next run starts from a newly initialized
image.

### 4.1 Allocation Failure

The graph must fit the configured task window, heap, fanin capacity, and
TensorMap pool. When an allocation cannot progress, a wall-clock backstop
latches a fatal instead of waiting for a scheduler that has not started.

Representative allocator output is:

```text
FATAL: Task Allocator Deadlock - Heap Exhausted!
  Task ring:  current=..., last_alive=..., active=.../...
  Heap ring:  top=..., tail=..., size=..., available=...
  Requested:  ... bytes
```

This is host-orchestration logging. The allocator records the corresponding
runtime error and unwinds; it does not terminate the process directly.

## 5. Submission and Dependencies

### 5.1 Mixed Tasks and Logical Blocks

An active mask selects AIC, AIV0, and AIV1 lanes. `block_num` is a logical SPMD
width and may exceed the physical device width when sync-start is not requested;
the scheduler dispatches those blocks in waves.

Before a slot is allocated or published, submission requires:

```text
block_num >= 1
block_num * popcount(active_mask) <= INT16_MAX
```

The product is stored in the 16-bit `total_required_subtasks` field. Sync-start
adds a separate co-residency limit: AIV tasks use the available AIV count, while
AIC/MIX tasks use the available cluster count.

### 5.2 TensorMap and Fanins

TensorMap maps tensor regions to producer task IDs. For every task:

1. INPUT/INOUT regions look up overlapping producers.
2. Explicit and discovered producers are deduplicated into
   `fanin_local_ids[]`.
3. OUTPUT/INOUT regions register the new task as producer.
4. Each producer tracks its highest consumer local ID for completion metadata.

There is no fanout adjacency or dependency pool. A per-slot completion flag is
the readiness truth on device.

## 6. Boot Classification and Wake Lists

Submit does not push tasks into ready queues. After the graph arrives on device,
boot classification scans every submitted task exactly once:

- a task whose fanins are all complete is routed to its shape queue;
- otherwise it registers on the first unmet producer's intrusive wake list; and
- producer completion reclassifies every detached waiter.

Completion flags are monotonic, so this consumer-pull scheme cannot miss a
producer transition and does not require periodic dependency polling.

The dispatchable shapes are `AIC`, `AIV`, and `MIX`; dependency-only `DUMMY`
tasks use a dedicated queue and complete without AICore dispatch.

Early producer propagation is currently disabled in HBG. The shared scheduler
retains early-staging code for parity with `tensormap_and_ringbuffer`, but HBG's
boot classifier and wake lists are the active readiness path.

## 7. Dispatch and Completion

- AIC/AIV dispatch claims ranges of logical block indices from
  `next_block_idx` and requeues unfinished wide tasks.
- MIX dispatch selects cluster offsets whose used lanes share one valid
  placement. The tracker uses a 128-bit bitset because the flattened offset is
  `cluster * 3`, reaching above bit 63 on supported devices.
- Sync-start cohorts stage locally when possible; wider ownership spans use a
  generation-tagged global drain before launch.
- Every lane completion increments `completed_subtasks`. The task completes once
  that count equals `block_num * popcount(active_mask)`.
- Completion sets the task's flag, advances the contiguous
  `completed_watermark`, and reclassifies its wake-list consumers.

The drain's `pending_task` stays valid for the complete attempt: all participant
threads load it before the coordinator can pass the stage-done barrier and clear
it. A recovery return for a null pointer would describe an unreachable state and
could strand the drain protocol, so the active path relies on that invariant.

## 8. Scalar Access During Construction

`get_tensor_data` and `set_tensor_data` operate on registered host views of
external tensors. They cannot wait for a submitted device producer because the
device scheduler starts only after orchestration returns. Runtime-created graph-
heap outputs also have no host view.

Producer references are checked against the complete bound descriptor ID before
a slot is used, preventing masked-slot aliasing. See
[SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md) for the supported contract.

## 9. Errors and Diagnostics

The runtime latches orchestration and scheduler errors in shared memory and maps
them to the negative run status observed by the host. Important validation paths
include:

- invalid arguments (`-5`);
- sync-start residency violations (`-7`);
- tensor wait timeout (`-8`); and
- scheduler timeout (`-100`).

Device logs contain scheduler records only. Host graph-construction diagnostics
remain host-side. See [device_log_profiling.md](device_log_profiling.md).

## 10. Verification

Runtime C++ changes require rebuilding the editable package, then running both
simulation variants:

```bash
pip install --no-build-isolation -e .
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim --runtime host_build_graph
```

Pure scheduler/core-tracker and lifecycle primitives also have C++ unit tests
under `tests/ut/cpp`.
