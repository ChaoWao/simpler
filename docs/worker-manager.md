# Worker Manager — Pool, Threading, and Dispatch

Callable identity update: WorkerThread task frames now carry the submitted
callable's 32-byte digest; the child resolves that digest to its private local
slot. Older callable-id snippets below are historical shorthand for
target-local internals. See
[callable-identity-registration.md](callable-identity-registration.md).

`WorkerManager`, `WorkerThread`, and `WorkerEndpoint` together implement the
**execution layer** of a `Worker` engine. In today's local implementation,
`WorkerManager` owns two pools of `WorkerThread`s (one for next-level workers,
one for sub workers); each `WorkerThread` owns a `LocalMailboxEndpoint` that
drives a shared-memory mailbox consumed by a forked Python child. The child
runs the real worker (a `ChipWorker` for NEXT_LEVEL, a Python callable for
SUB) in its own address space.

The remote L3 design keeps this local fork/shm path behind
`LocalMailboxEndpoint` and reserves the same `WorkerEndpoint` boundary for a
framed `RemoteL3Endpoint` for cross-host NEXT_LEVEL children. A remote endpoint
is not another child loop that polls the `MAILBOX_SIZE`-byte mailbox; it uses the
contracts in
[remote-l3-worker-design.md](remote-l3-worker-design.md).
The current code includes that `RemoteL3Endpoint` boundary, a socket-backed
simulation transport, and the daemon/session runner used by
`Worker.add_remote_worker()` for sim remote L3 endpoints. HCOMM hardware
profiles are still pending.

For the high-level role of this layer among the three engine components, see
[hierarchical-level-runtime.md](hierarchical-level-runtime.md). For what
runs on the other side of the local mailbox, see [task-flow.md](task-flow.md).
For where dispatched tasks come from, see [scheduler.md](scheduler.md).

---

## 1. `WorkerManager`

```cpp
class WorkerManager {
public:
    // Registration (before init). `mailbox` is a MAILBOX_SIZE-byte
    // MAP_SHARED region; the real worker (a `ChipWorker` for NEXT_LEVEL,
    // a Python callable for SUB) lives in the forked child.
    void add_next_level(void *mailbox);
    void add_next_level_at(int32_t worker_id, void *mailbox);
    void add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint);
    void add_sub       (void *mailbox);

    // Lifecycle
    void start(Ring *ring, OnCompleteFn on_complete,
               OnAcceptFn on_accept);              // starts all WorkerThreads
    void stop();

    // Scheduler API
    WorkerThread *get_worker_by_id(WorkerType type, int32_t worker_id) const;
    std::vector<int32_t> next_level_worker_ids() const;
    WorkerThread *pick_idle_sub_excluding(
        const std::vector<WorkerThread *> &exclude) const;

private:
    struct LocalNextLevelEntry {
        int32_t worker_id;
        void *mailbox;
    };
    std::vector<LocalNextLevelEntry> next_level_entries_;
    std::vector<void *> sub_entries_;
    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;
};
```

`add_next_level_at(...)` is used by the Python `Worker` facade when local L4
children share the NEXT_LEVEL worker id space with remote L3 workers.
Python local Worker children use explicit worker ids rather than deriving
the public worker id from the local worker vector index.

### Responsibilities

- **Pool ownership**: two `std::vector` pools, sized at init from `add_*`
  calls
- **Directed NEXT_LEVEL lookup**: `get_worker_by_id` resolves the exact stable
  target selected by the user; the Scheduler never asks the manager to choose
  another NEXT_LEVEL worker
- **SUB-only idle selection**: `pick_idle_sub_excluding` chooses an idle SUB
  worker not already used by the same SUB group

Callable and remote-buffer eligibility is validated against the exact target
during Orchestrator submission. It is not scheduling metadata and is not
stored on the task slot.

---

## 2. `WorkerThread`

One WorkerThread per registered mailbox (i.e. per forked child worker).

```cpp
struct WorkerDispatch {
    TaskSlot task_slot;
    int32_t  group_index = 0;    // 0 for non-group; 0..N-1 for group members
};

class WorkerThread {
public:
    void start(Ring *ring,
               const std::function<void(WorkerCompletion)> &on_complete,
               const std::function<void(WorkerDispatch)> &on_accept,
               std::unique_ptr<WorkerEndpoint> endpoint);
    void stop();
    void dispatch(WorkerDispatch d);       // slot id + group sub-index
    bool idle() const;
    const WorkerEndpointCaps &caps() const;
    int32_t worker_id() const;

private:
    Ring *ring_;                       // reads slot state via ring->slot_state(id)
    std::unique_ptr<WorkerEndpoint> endpoint_;
    std::thread thread_;
    std::queue<WorkerDispatch> queue_;
    std::mutex mu_;
    std::condition_variable cv_;

    void loop();
    WorkerCompletion dispatch_process(WorkerDispatch d,
                                      const std::function<void()> &on_accept);
};
```

The WorkerThread's `std::thread` pumps the internal queue and calls
`endpoint->run_with_accept(...)` once per dispatch. `LocalMailboxEndpoint`
drives the shm handshake — one mailbox round trip per dispatch. The forked child loop
that consumes the mailbox lives in Python (`_chip_process_loop` /
`_sub_worker_loop` in `python/simpler/worker.py`); the parent does not fork
children.

`WorkerDispatch` carries only `{slot_id, group_index}`; the thread reads
`slot.callable` / `slot.task_args` / `slot.config` on each dispatch via
`ring->slot_state(slot_id)`. For a group slot with `group_size() == N`,
the Scheduler pushes N `WorkerDispatch` entries (one per member) onto N
exact target threads for NEXT_LEVEL, or N freely selected SUB threads. Each
thread's `group_index` selects which
`task_args_list[i]` view to hand to the worker. There is no
`WorkerPayload` — the per-dispatch carrier is just the slot id plus the
group sub-index.

---

## 3. Dispatch via shm mailbox

Each `LocalMailboxEndpoint` drives a `MAILBOX_SIZE`-byte `MAP_SHARED` region.
The Python facade forks one child per mailbox **before**
`WorkerManager::start()` (so the parent has only the Python main thread when
fork runs, avoiding the classical "fork in a multi-threaded process" hazard)
and the child polls the mailbox for the lifetime of the worker.

### 3.1 Parent-side dispatch

```cpp
// `run(ring, d)` forwards here with an empty hook; the acceptance-aware
// overload is what WorkerThread calls.
WorkerCompletion LocalMailboxEndpoint::run_with_accept(
    Ring *ring, const WorkerDispatch &d, const std::function<void()> &on_accept
) {
    TaskSlotState &s = *ring->slot_state(d.task_slot);
    char *m = static_cast<char *>(mailbox_);

    // Write task data: reserved callable field, config, digest prefix, then
    // length-prefixed TaskArgs blob. Tags are stripped; only
    // [digest][T][S][tensors][scalars] crosses the fork boundary.
    uint64_t reserved = 0;
    memcpy(m + MAILBOX_OFF_CALLABLE, &reserved, sizeof(reserved));
    memcpy(m + MAILBOX_OFF_CONFIG, &s.config, sizeof(CallConfig));
    memcpy(m + MAILBOX_OFF_TASK_CALLABLE_HASH, s.callable.digest.data(), 32);
    const TaskArgs &args = s.is_group() ? s.task_args_list[d.group_index] : s.task_args;
    write_blob(m + MAILBOX_OFF_TASK_ARGS_BLOB, args);

    // Signal child
    write_state(mailbox_, MailboxState::TASK_READY);

    // Spin-poll for completion. The task's latency runs through this wait, so
    // it never sleeps (codestyle rule 5). A child that dies without publishing
    // TASK_DONE would spin here forever, so its liveness is sampled on a
    // kChildLivenessPollPeriod steady-clock period rather than an iteration
    // count, which no longer maps to a bounded time at spin speed.
    auto next_liveness_check = steady_clock::now() + kChildLivenessPollPeriod;
    bool acceptance_observed = false;
    while (true) {
        MailboxState state = read_state(mailbox_);
        // Launch acceptance is a sticky word, not a state: the child sets it
        // before TASK_DONE and nothing clears it until the next TASK_READY, so
        // a task that finishes between two polls cannot lose the ACK. Read
        // after the state so a TASK_DONE observation implies it is visible.
        if (!acceptance_observed && read_accepted(mailbox_)) {
            acceptance_observed = true;
            if (on_accept) on_accept();
        }
        if (state == MailboxState::TASK_DONE) break;
        auto now = steady_clock::now();
        if (now >= next_liveness_check) {
            next_liveness_check = now + kChildLivenessPollPeriod;
            std::string death = check_child_death();
            if (!death.empty()) {
                completion.outcome = EndpointOutcome::ENDPOINT_FAILURE;
                completion.error_message = "LocalMailboxEndpoint::run: " + death;
                return completion;
            }
        }
    }

    int err = read_error(mailbox_);
    write_state(mailbox_, MailboxState::IDLE);
    return err == 0
        ? WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::SUCCESS, ""}
        : WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::TASK_FAILURE,
                           read_error_msg(mailbox_)};
}
```

Parent-side cost per dispatch:

- One reserved `uint64`, one `CallConfig`, one 32-byte digest, and one
  TaskArgs blob
- One signal (`write_state`)
- Spin-poll loop, never a sleep — a task's latency passes through this wait
  ([codestyle](../.claude/rules/codestyle.md) rule 5). Child liveness is sampled
  on a steady-clock period, so a dead child ends the wait with
  `ENDPOINT_FAILURE` instead of spinning the parent forever
- A sticky launch-acceptance word, observed before completion and cleared only
  when the parent publishes the next `TASK_READY`
- One explicit completion outcome: success, task failure, or endpoint failure

Total ~nanoseconds overhead; the wait is dominated by actual kernel execution.

### 3.2 Child loop

The child loop lives in Python — see `_chip_process_loop` and
`_sub_worker_loop` in `python/simpler/worker.py`. Each child polls
`MAILBOX_OFF_STATE`, decodes the digest-prefixed args blob on `TASK_READY`,
resolves the digest to its private local slot/callable, writes back any error,
and publishes `TASK_DONE`. An A2A3 onboard chip child also exposes its mailbox
state word to the native runner, which publishes `TASK_ACCEPTED` after both
kernel launches succeed. SUB, remote, A5, and failure paths use WorkerThread's
completion-time acceptance fallback.
The child inherits the parent's full address space at fork time, so:

- ChipCallable objects (pre-fork allocated) are COW-visible at the same VA
- The Python callable registry is COW-visible
- Tensor data in `torch.share_memory_()` regions is fully shared (MAP_SHARED)

### 3.3 Mailbox layout

```text
offset 0:                         int32   state
offset 4:                         int32   error
offset 8:                         uint64  reserved task callable field
                                          or control sub-command
offset 16:                        CallConfig config
MAILBOX_OFF_TASK_CALLABLE_HASH:   uint8[32] callable digest
MAILBOX_OFF_TASK_ARGS_BLOB:       bytes [int32 T][int32 S]
                                        [Tensor x T][uint64_t x S]
tail:                             fixed-size NUL-terminated error message
```

The current mailbox size is the C++ `MAILBOX_SIZE` constant exported through
the nanobind module; Python derives its offsets from the same binding where
possible so the two sides cannot drift silently.

### 3.4 Shutdown

`WorkerManager::shutdown_children()` writes `SHUTDOWN` to every registered
endpoint; for `LocalMailboxEndpoint` this writes `SHUTDOWN` to the mailbox.
Each child loop sees it on its next poll and exits. The Python facade owns the
child PIDs and calls `waitpid()` after writing `SHUTDOWN` to its own mailbox
copy. The parent's `WorkerThread::stop()` only joins the C++ dispatcher thread
— it does not own the child process.

---

## 4. Local vs. Remote Endpoints

The mailbox protocol is the local endpoint contract. Adding another local
forked worker kind still follows the existing pattern:

1. Define the worker entry point.
2. Write a child-process loop that polls the mailbox, decodes the args blob,
   and invokes that entry point.
3. Register the mailbox via `manager.add_next_level_at(worker_id, mailbox)`
   for an explicit NEXT_LEVEL worker id, `manager.add_next_level(mailbox)` to
   allocate the next stable local id, or `manager.add_sub(mailbox)`.

Remote L3 is different. It cannot reuse the mailbox wire format because the
remote side does not share virtual addresses, fork-time COW registries, POSIX
shm names, or parent-visible child PIDs. The remote design introduces a
transport-neutral endpoint under `WorkerThread`: `LocalMailboxEndpoint` wraps
this local mailbox path, while `RemoteL3Endpoint` sends framed TASK, CONTROL,
COMPLETION, HEALTH, and SHUTDOWN messages over the negotiated transport.

The implemented `RemoteL3Endpoint` sends TASK and CONTROL frames, waits for
COMPLETION and CONTROL_REPLY frames through `RemoteL3Transport`, and monitors
an independent simulation health lane. Python remote worker specs open a
session through `simpler-remote-worker`; the endpoint is schedulable only after
the session runner reports `HELLO READY`.

### 4.1 Nested fork ordering (L4+ Worker children)

When an L4 Worker has L3 Worker children, `init()` is eager and the fork
sequence nests recursively — the whole tree is READY when `L4.init()` returns:

```text
L4 parent process
  ├─ _init_hierarchical(): Worker(4) + HeapRing mmap (before fork)
  └─ _start_hierarchical() (inside init()):
       ├─ fork L3 child  ────────►  L3 child process:
       │                              inner_worker.init()  ← eager, recursive
       │                                ├─ Worker(3) + L3 HeapRing
       │                                └─ _start_hierarchical() forks L3's
       │                                   sub/chip children and blocks on their
       │                                   INIT_READY, THEN publishes INIT_READY
       │                              _child_worker_loop()  (dispatch only)
       ├─ await every L3 child's INIT_READY (whole subtree ready)
       └─ register mailbox with L4's Worker
```

Each inner Worker inits **inside its forked child process** so its own
children are forked from the correct parent. Because the inner `init()` is
eager and blocks on its descendants, a child publishes `INIT_READY` only after
its whole subtree is ready, so readiness propagates recursively up to
`L4.init()`. The L4 parent never sees L3's sub/chip grandchildren — they're
L3's responsibility; if startup fails, they are reclaimed via the child's
process-group cancellation domain, not the resource_tracker.

**Key invariant**: `Worker(N)` and its HeapRing are created before any
fork at level N. Children inherit the `MAP_SHARED` mmap at the same virtual
address. C++ scheduler threads start only after all forks at that level.

---

## 5. Why this layering

Three decisions that led here:

### 5.1 Why not fork per task?

Forking per submit eliminates the mailbox and serialization, but costs
~1-10 ms per fork (COW page-table setup for a large parent image). For
thousands of tasks per DAG, the overhead dominates. Pre-forked pool amortizes
fork across many dispatches.

### 5.2 Why slot pool on parent heap, not shm?

The scheduling state (TaskSlotState.fanin_count, fanout_consumers,
fanout_mu) is parent-only — Scheduler and Orchestrator read/write it, but
children never do. Putting the slot in shm would force cross-process atomics
and shm-safe containers for no benefit. See
[task-flow.md](task-flow.md) §11 for full rationale.

### 5.3 Why one WorkerThread per child?

Alternative: N children share one dispatch queue. Rejected because:

- `WorkerThread` is the natural execution unit. Directed NEXT_LEVEL work waits
  in child `i`'s ready FIFO if that child is busy; SUB work may use another
  idle SUB child
- Simpler mental model: one child = one thread that drives it
- Zero contention on queue access (only one producer, one consumer per queue)

---

## 6. Related

- [hierarchical-level-runtime.md](hierarchical-level-runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `ChipWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
