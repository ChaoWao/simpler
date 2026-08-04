# L3 — Host-level multi-chip examples

**L3 = HOST**: one host machine that drives multiple L2 chips plus M
SubWorkers (plain Python callables), coordinated by an Orchestrator running in
the host process. This is where you first see the *DAG* model — you submit a
task per chip — each pinned to its target chip worker id — and each task
carries a dependency graph via `orchestrator` APIs.

See [`docs/hierarchical-level-runtime.md`](../../../docs/hierarchical-level-runtime.md)
for the full L0–L6 diagram and [`docs/task-flow.md`](../../../docs/task-flow.md)
for data-flow end to end.

## Minimum Worker lifecycle

L3 adds two steps before `init()`:

```python
from simpler.worker import Worker

worker = Worker(
    level=3,
    platform="a2a3sim",
    runtime="tensormap_and_ringbuffer",
    device_ids=[0, 1],        # two chips
    num_sub_workers=1,        # one Python post-processing callable
)

# 1. Register sub-worker callables BEFORE init (level >= 3 only).
#    Returns an opaque handle you pass to orchestrator.submit_sub(...) later.
postprocess_handle = worker.register(
    lambda args: print("post-process received", args)
)

worker.init()                 # forks chip child processes + sub children,
                              # then starts the C++ scheduler

def my_orch(orch, args, cfg):
    # orch is the Orchestrator. Submit one task per chip + any sub work.
    # orch.submit_next_level(..., worker=chip_id) targets one chip.
    # orch.submit_sub(postprocess_handle, sub_args) schedules a Python callable.
    ...

try:
    worker.run(my_orch, my_args, my_config)
finally:
    worker.close()            # shuts down child processes and releases shm
```

Two things to know before reading the example:

1. **This example registers callables before `init()`**. That keeps startup
   simple and lets chip children pre-warm their callable state before the
   first DAG dispatch.
2. **The orchestration function is a *plain Python function*, not a C++
   kernel.** It runs in the host process and calls `orch.submit_*(...)` to
   hand work to chip children. The children get the submitted `ChipCallable`
   through shared-memory mailboxes.

## What each example demonstrates

Roughly in reading order — each row assumes the ones above it. Every directory
has its own README with the run commands and what the golden check proves.

### Start here

| Directory | New concept |
| --------- | ----------- |
| [`multi_chip_dispatch/`](multi_chip_dispatch/) | Two chips + one SubWorker. An orchestration fn dispatches a `ChipCallable` to each chip, then submits a Python callable to collect/verify results. The smallest correct L3 program. |
| [`child_memory/`](child_memory/) | `orch.malloc` + `ChipTensor(child_memory=True)` to load a weight once and reuse it across multiple kernel invocations on the same chip. |
| [`per_task_runtime_env/`](per_task_runtime_env/) | One L3 launch where each L2 task binds its own ring sizes through `CallConfig.runtime_env`. |

### Communication domains and collectives

| Directory | New concept |
| --------- | ----------- |
| [`allreduce/`](allreduce/) | The minimal collective: allocate a domain, submit one task per rank, golden-check the sum. |
| [`ffn_tp_parallel/`](ffn_tp_parallel/) | Two dependent stages per rank — AIC matmul then AIV cross-rank reduce — with the producer/consumer edge inferred from a shared buffer address rather than declared. |
| [`domain_rank_map/`](domain_rank_map/) | What `orch.allocate_domain` actually hands each chip: dense per-domain ranks, `KeyError` for non-members, distinct buffers for overlapping domains. |
| [`dual_domain_overlap/`](dual_domain_overlap/) | Two overlapping domains where worker 1 holds a different rank in each, plus `submit_next_level_group`. |
| [`ep_dispatch_combine/`](ep_dispatch_combine/) | Production-shape MoE dispatch/combine: three chained AIV kernels, a five-phase routing protocol, and a hand-laid-out window. The largest example here. |

### Driving a task that is already running

Both submit one long-lived L2 task and then exchange data with it in flight —
a serving loop rather than a batch DAG. Each needs `aicpu_thread_num = 2`.

| Directory | New concept |
| --------- | ----------- |
| [`worker_chip_message_queue/`](worker_chip_message_queue/) | Request/response streaming over the L3-L2 queue: input and output arenas, `peek`/`read_into`/`release`, cooperative `request_stop`. Responses are not paired one-to-one with requests. |
| [`worker_chip_orch_comm_stream/`](worker_chip_orch_comm_stream/) | The raw form underneath: a shared payload region plus counters, driven by `notify` / `test` / `wait` on a monotonic sequence. |

**Collective algorithm tests** (allreduce, allgather, reduce_scatter, broadcast, all_to_all) have moved to `tests/st/worker/collectives/`. See the scene tests there for the full algorithm corpus including multi-mode allreduce (onephase, twophase, ring, bidirectional_ring, ibing).

## Prerequisites

Same as L2 (see [`../l2/README.md`](../l2/README.md)): venv + `pip install .`.

Additionally, L3 runs real child processes via `fork()`. On macOS you *can*
run the L3 sim path, but fork + Python state can surface issues that don't
appear on Linux. When in doubt, run L3 examples on a Linux host.
