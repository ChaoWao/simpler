# Host runtime trace markers — `[STRACE]`

`simpler_run()` spans several host-side stages (`bind`, `runner_run`,
`validate`) plus, inside `runner_run`'s enqueue-through-drain lifetime, an
on-NPU AICPU window. TMR subdivides that window into preamble / SO-load /
graph-build / post-orch; HBG emits the whole device wall without those
device-orchestrator phases. The two headline walls (`host_wall` / `device_wall`, see
[l2-timing.md](l2-timing.md)) cannot show *where* the time goes.

`[STRACE]` markers are simpler's answer — host-side trace spans emitted to the
log, analogous to Android atrace/systrace. A consumer (e.g. pypto-serving)
reads the per-stage breakdown **from the log**, with **no code change** on its
side and no API contract: `run()` returns `None`, so markers (not a return
value) are the channel, and the log is the one sink the L3 parent and its L2
children share.

`[STRACE]` rides on the compile-time `SIMPLER_HOST_STRACE` macro (default on, in
`src/common/task_interface/profiling_config.h` — separate from the
`SIMPLER_DFX` gate on the device Orch/Sched markers) and is emitted at
`LOG_TIMING` (the default threshold) — **no new env var or flag**. In a
`SIMPLER_HOST_STRACE`-off build the RAII macros compile to nothing.

## Marker grammar

Every host log record starts with a `CLOCK_MONOTONIC` nanosecond timestamp:

```text
[mono_ns=<ns>][T0x<thread>][<level>] <func>: ...
```

Each process emits one TIMING-level mapping from that clock to wall time when
its logger starts:

```text
[CLOCK_ANCHOR] v=1 pid=<pid> mono_ns=<ns> wall_ns=<ns>
```

For host-clock records, consumers recover an approximate absolute timestamp
with `wall_ns + record_mono_ns - mono_ns`. Their event ordering and duration
calculations remain entirely on the monotonic clock and are unaffected by
wall-clock corrections. Records tagged `clk=dev` use the separate device-clock
domain described below and do not use this anchor.

`strace_timing.py` applies that mapping to both `--trace-out` and `--swimlane`.
The visible Perfetto axis remains monotonic; each mapped host event exposes the
exact decimal `wall_ts_ns` and a UTC `wall_time` in its arguments, while the JSON
top level retains the source mappings in `clockAnchors`. Nanosecond epoch values
are strings because JSON consumers commonly use IEEE-754 numbers, which cannot
represent current epoch nanoseconds exactly. Old logs without an anchor retain
their existing output, and `clk=dev` records never receive host wall time.

One line per span, emitted on scope exit
(`src/common/log/include/common/strace.h`):

```text
[STRACE] v=1 pid=<n> tid=<n> inv=<n> hid=<hex> depth=<n> name=<dotted> ts=<ns> dur=<ns> [k=v ...]
```

| Field | Meaning |
| ----- | ------- |
| `v` | format version; the parser branches on it. Lets device-side markers align later by reusing the prefix + adding fields. |
| `pid` `tid` | process / thread id — L3 parent and each L2 child are distinct pids, so they land on separate lanes. |
| `inv` | 64-bit process-wide `simpler_run` invocation id (allocated from an atomic, so `(pid, inv)` is unique even across concurrent calls) — **a grouping key only** (gathers one call's spans), NOT a token index. Set once per call. |
| `hid` | callable content hash (ELF Build-ID 64), stable across slot reuse / processes / runs. The parser buckets by `hid`; the most-frequent bucket is decode (one invocation per token), a once-seen bucket is prefill. |
| `depth` | thread-local nesting depth (`++` on enter, `--` on exit). The parser rebuilds the call tree from `depth` — **not** from timestamp containment. |
| `name` | dotted span name (self-locating even without the tree). |
| `ts` `dur` | start + duration in ns. Maps 1:1 onto a Chrome-trace `"X"` event. For host spans `ts` is `CLOCK_MONOTONIC` (`steady_clock`), same-host cross-process comparable. For `clk=dev` device spans (see below) `ts` is instead a **device-clock** start offset on a per-invocation origin — comparable to the other device spans (so the orch∪sched window is recoverable), not the host clock. |
| `k=v ...` | optional per-span attributes (e.g. `ntensor=4`); a parser that doesn't recognize one ignores it. |

Span names and attributes percent-encode control bytes and record delimiters.
They are length-capped (with `~` marking truncation) so each marker remains a
single atomic pipe write even when forked workers share captured stderr.
`strace_timing.py` decodes both on the way back in, so a consumer reading its
output sees the original text; a consumer reading the raw log does not.

## Span tree

```text
simpler_run                                   (= host_wall)
├─ simpler_run.bind
│  ├─ simpler_run.bind.args        (ntensor=N: per-tensor device_malloc + H2D)
│  └─ simpler_run.bind.prebuilt    (prebuilt runtime-arena cache hit or build + upload)
├─ simpler_run.runner_run          (device enqueue + completion drain)
│  └─ simpler_run.runner_run.device_wall      (whole on-NPU AICPU wall)
│     └─ .{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}
│           TMR device-domain (clk=dev): AICPU subdivision of the on-NPU wall
└─ simpler_run.validate
```

The `device_wall` span exists for both runtimes. Its
`.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}`
children are TMR-only; HBG orchestration runs on the host and stamps none of
those phases. All emitted device spans are tagged `clk=dev`. They are not host
`steady_clock` spans: the AICPU stamps raw sys-counter cycles into a host-allocated buffer
(whose address rides on `KernelArgs::device_wall_data_base`), the host reads it
back after stream-sync, converts cycles → ns, and emits the marker. `orch`/
`sched` are the orchestrator/scheduler windows that formerly only appeared as
device-log lines. A phase that was never stamped
(0 ns) is skipped — e.g. `so_load` is ~0 on a cached-callable run. See
[device-phases.md](device-phases.md) for the device-side mechanism.

The phased native-run interface preserves this same marker contract. Prepare
allocates one `inv` and records the host-wall start; prepare, the child progress
path's launch/drain lifecycle, and finalize bind that `(inv, hid)` while
emitting their spans. Finalize releases the runner claim, destroys the per-run
state, and then emits the stored `simpler_run` wall, so the root includes that
cleanup tail.
No trace scope or synthetic nesting remains active between C API calls. For
direct phased use the host wall is the full prepare-to-finalize lifetime,
including time the caller spends polling or doing other host work; blocking
`simpler_run` is the same phases composed back-to-back.

| Depth | Span names |
| ----- | ---------- |
| 0 | `simpler_run` |
| 1 | `simpler_run.bind`, `simpler_run.runner_run`, `simpler_run.claim_release`, `simpler_run.validate` |
| 2 | `simpler_run.bind.args`, `simpler_run.bind.prebuilt`, `simpler_run.runner_run.device_wall` |
| 3 | TMR phase spans `simpler_run.runner_run.device_wall.{preamble,so_load,graph_build,config_validate,arena_wire,sm_reset,post_orch,orch,sched}` and optional `task_slot_*` spans |

## L3/L4 host scheduler spans

Every hierarchical worker that drives next-level children emits these spans
through the logger compiled into `_task_interface` — an L3 with chips and an
L4 pod alike, since the orchestrator and scheduler code they run is the same:

| Span | Host decision point |
| ---- | ------------------- |
| `l3.graph_build` | serialized Python graph callback |
| `l3.submit` | next-level task publication after slot allocation |
| `l3.dispatch` | scheduler handoff to a worker thread |
| `l3.frame_submit` | local child mailbox-frame publication |
| `l3.activate` | prepared-frame activation |
| `l3.complete` | terminal child progress handling |
| `l3.post_fence_retirement` | run erase + quiescent compaction, after the completion fence |

Their attributes carry the available `run_id`, `task_slot`, `group_index`,
`worker_id`, `dispatch_id`, endpoint kind, and the dispatch's pipeline lease
(`slot_id` / `generation`).

Because the names do not encode which level emitted them, a pod run puts the L4
process and each of its L3 processes on lanes that differ only by pid. The
per-level vocabulary that resolves this is tracked in
[#1793](https://github.com/hw-native-sys/simpler/issues/1793).

One process contributes at most two host lanes, because the scheduler runs on
one thread: the facade thread emits `l3.graph_build` and `l3.submit`, and the
scheduler thread emits the other four. `role=worker` on `l3.frame_submit`,
`l3.activate` and `l3.complete` names the worker a dispatch targets, not the
thread that ran it.

The spans reach the logger over the fixed POD `SimplerHostSpan` ABI in
`common/host_span.h`. `_task_interface` compiles the host logger directly, so
there is no nullable sink and host-span support cannot disappear because a
separate logger DSO was absent. Every other host consumer also compiles a
private logger implementation, then binds it to the
`SimplerHostLogState` owned by `_task_interface`. The hierarchical parent seeds
that state before `fork()`; a chip child re-seeds its inherited copy and passes
the same pointer to every runtime module it loads. The threshold and one-anchor
coordination are therefore shared within each process without relying on
`RTLD_GLOBAL` logger symbols.

## Reading the markers — `strace_timing.py`

```bash
# TPOT table (per-callable, decode = most-invoked hid bucket)
python -m simpler_setup.tools.strace_timing path/to/host_or_device.log

# also emit the established per-invocation call-tree JSON
python -m simpler_setup.tools.strace_timing path/to/log --trace-out strace.json

# emit the L3/L4 host scheduler timeline on real OS pid/tid lanes
python -m simpler_setup.tools.strace_timing path/to/log --swimlane host_swimlane.json
```

The tool groups by `(pid, inv)`, rebuilds each invocation's tree from `depth`,
buckets by `hid`, and prints each callable's mean `simpler_run` plus per-stage
means. With `--trace-out` it writes one `ph:"X"` event per span on a synthetic
per-invocation lane, so each call renders as an isolated nested tree in
[Perfetto](https://ui.perfetto.dev) / `chrome://tracing`.

`--swimlane` is a separate view. Host slices keep their real OS pid/tid, and
task submission-to-dispatch handoffs render as flow arrows.

**One exception, because a K-deep pipeline is not K threads.** The direct-chip
lane drives prepare(N+1) and finalize(N) from the *same* OS thread, so a 40-run
K=2 stress produces 40 overlapping run lifetimes on one tid. Perfetto nests
slices by timestamp containment within a track, so flattening them there puts
run N+1's spans *inside* run N's root — false nesting that hides the very
overlap the view exists to show. A thread whose depth-0 spans overlap is
therefore split into one lane per pipeline slot (`pipeline slot 0 (tid …)`),
which reads as a plain sequence because a run holds its slot exclusively. The
overlap then shows where it belongs: across lanes. Each slice keeps `os_tid` in
its args, and a thread that ran sequentially — the L3 scheduler, which carries a
slot but never interleaves — keeps its real tid.

Chrome Trace JSON has only one visible timestamp axis, so putting the raw
per-invocation device clock beside `CLOCK_MONOTONIC` would create a multi-day
empty interval. The converter therefore keeps `clk=dev` records, with their
original ns timestamps, in the top-level `unalignedDeviceSpans` array instead of
`traceEvents`; it does not guess a clock offset. Perfetto opens directly on the
host activity, while the existing tables, tree, and `--trace-out` still provide
the device-phase timing views.

## Async pipeline proof

The phased native lane claims that run N+1's preparation runs *concurrently*
with run N's device execution. That claim is checkable from a captured log
without any new marker family: the windows are already in the `simpler_run`
tree, and the root span already carries the identity that tells two runs apart.

| Property | Read from |
| -------- | --------- |
| successor's preparation | `simpler_run.bind` — its arena build + host orchestration |
| predecessor's device work | `simpler_run.runner_run` |
| when a successor may launch | `simpler_run.claim_release` |
| which run each belongs to | root `simpler_run` attrs, joined by `(pid, inv)` |

Only `claim_release` was added for this: it wraps `release_native_run` inside
finalize, the point a successor's launch becomes admissible, and no other span
marks that boundary. `l3.post_fence_retirement` covers the L3 orchestrator's
`release_run` tail for the same reason.

The identity is `run_id / dispatch_id / run_epoch / slot_id / generation`. Each
field means one thing: `run_id` and `dispatch_id` are zero on the direct-chip
lane, which allocates neither, and `run_epoch` is a per-process monotonic counter
that is always set — so it is what orders runs when the other two are absent.
`NativeDispatchIdentity.sequence` makes that choice in the parser, where it is
visible, rather than in the record.

```bash
python -m simpler_setup.tools.strace_timing path/to/log --assert-native-overlap
```

Per adjacent run on one child process, the command requires `bind(N+1)` to
**overlap** `runner_run(N)` — the intervals intersect — and `runner_run(N+1)` not
to start before `claim_release(N)`. It exits nonzero on a missing identity, a
missing span, or an ordering violation.

Reading `bind` rather than the whole prepare is deliberate: `bind` sits inside
prepare, so an overlap it reports is one the prepare certainly had.

`--require-hidden-prepare` adds the stronger claim that the preparation also
*finishes* inside the predecessor's device window — fully hidden rather than
merely concurrent. That is a statement about pipeline depth and it is sensitive
to host scheduling, so it is opt-in and the scene test asserts only the overlap
property.

## Why markers, not a return value

Android's atrace writes to the ftrace `trace_marker` sink and systrace renders
it; nobody changes their code to be observed. `[STRACE]` mirrors that: the
runtime emits, tooling renders, the caller is untouched. Concretely, `run()`
returns `None`: an L3 `DistributedWorker.run` has no single device wall, and a
return-value channel could not carry each L2 child's host/device breakdown up
anyway. The log can. This is also why device phases are emitted as markers from
the host C++ rather than threaded back through any return struct to Python.
