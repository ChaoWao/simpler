# `host_build_graph` Device-Log Profiling

## Execution Boundary

`host_build_graph` runs its orchestration shared object on the host before the
device launch. The device boots scheduler-only, so its device log contains no
orchestrator-thread block and no device-side `dlopen` timing.

Host graph-construction timing belongs in host-side diagnostics. Ordinary DAG
runs schedule on resident AICore workers; AICPU logs cover lifecycle timing only.
Graph replay continues to emit the compatibility AICPU scheduler records.

## Finding the Log

On hardware, AICPU `LOG_INFO` records are written by CANN's dlog subsystem:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Find the newest file and filter the current scheduler and lifecycle records:

```bash
ls -t "$HOME/ascend/log/debug/device-<device_id>"/device-*.log | head -1
grep -E "A5 HBG AICore scheduler|AicoreLifecycle|sched_start=" <logfile>
```

## Resident Scheduler Summary

After a successful resident run, Host validation emits aggregate timing and
protocol counters copied from every active worker:

```text
A5 HBG AICore scheduler HOST TIMING: payload=... kernel=... completion=... backoff=... cycles
A5 HBG AICore scheduler COUNTERS: bootstrap_tasks=... ready_enqueues=... ready_pops=...
```

| Field | Meaning |
| ----- | ------- |
| `bootstrap_tasks` | Executable tasks classified during Resolver bootstrap |
| `ready_enqueues` / `ready_pops` | Ordinary tasks published to and claimed from Ready inboxes |
| `completion_enqueues` | Completed AIC/AIV subtasks reported to their Resolver |
| `completion_resolves` | Graph tasks retired after completion aggregation |
| `ready_to_kernel_*` | Aggregate and maximum Ready-to-kernel latency |
| `backoff` / `idle_iterations` | Resident-loop idle cost and iterations |

Validation also rejects nonempty Ready/completion state, occupied dispatch slots,
open wake lists, incomplete bootstrap, or inconsistent execution counts. These
errors are correctness failures, not profiling warnings.

## Compatibility Scheduler Records

Only Graph replay uses the AICPU scheduler phase records. With
`SIMPLER_SCHED_PROFILING` enabled, they include:

```text
Thread 0: === Scheduler Phase Breakdown: total=3460.100us, 352 tasks ===
Thread 0:   complete       : ...
Thread 0:     poll         : ... hit=... miss=... hit_rate=...%
Thread 0:   dispatch       : ...
Thread 0:     pop          : ... work=... wait=... atomics=...
Thread 0:     setup        : ...
Thread 0:   idle           : ...
Thread 0:   avg/complete   : ...
```

- `complete` polls core completion and retires mixed tasks.
- `dispatch` selects ready work, builds payloads, and publishes core commands.
- `pop` separates ready-queue work from contention wait.
- `setup` covers payload/MMIO preparation.
- `idle` is accounted time when the loop made no progress.

Dependency fanout/fanin aggregates are derived from `deps.json` by the DFX
analysis tools; current HBG device logs do not print the obsolete adjacency-list
statistics described by older guides.

## Timeout Record

A scheduler timeout uses an explicit end marker:

```text
Thread 0: sched_start=... sched_end(timeout)=... sched_cost=...
```

Use the accompanying runtime-status and stall-classification records to identify
the failure. Do not infer an orchestration stall from a missing Thread-3 block;
that block never exists in this runtime.

## Related Tools

- `python -m simpler_setup.tools.strace_timing` summarizes host/device run markers.
- `simpler_setup.tools.core_swimlane` reconstructs intra-core execution from dumps.
- The repository's DFX analysis workflow combines scheduler timing with
  dependency and task records.
