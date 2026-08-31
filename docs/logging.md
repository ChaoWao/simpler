# Log System

Architecture and contracts for the host and device logging subsystem.

For the user-facing level model and CLI flags, see
[testing.md § Log levels](testing.md#log-levels). This document covers the
implementation, cross-DSO state ABI, build wiring, and output formats.

## Mental model

```text
logging.getLogger("simpler").setLevel(N)
                  │
                  ▼  Worker.init() snapshots + normalizes the threshold
       _task_interface owns SimplerHostLogState
                  │
                  ├─ its own HostLogger reads that state
                  ├─ state is inherited before hierarchical worker forks
                  └─ ChipWorker passes the state pointer to loaded host modules
                       │
                       ├─ libcpu_sim_context.so (sim, RTLD_GLOBAL for PTO hooks)
                       ├─ libhost_runtime.so (RTLD_LOCAL)
                       ├─ generated host orchestration SOs
                       └─ sim AICore SOs

Each host module:
       private host_log.cpp + unified_log_host.cpp
                  │
                  └─ simpler_host_log_bind_state(state*)

Device logging:
       dev_vlog_* compatibility interface
            ├─ sim: bound HostLogger → same state, envelope, and destination
            └─ onboard: separate CANN backend sampled during device init
```

One threshold controls `DEBUG / INFO / TIMING / WARN / ERROR`; `NUL` suppresses
all output. The values match Python logging (`10 / 20 / 25 / 30 / 40 / 60`).
CANN has no TIMING level, so onboard setup maps both TIMING and WARN to CANN
WARN. The default TIMING threshold keeps host `[STRACE]` markers without
opening CANN's INFO stream.

## File layout

```text
src/common/log/
├── include/
│   ├── common/
│   │   ├── host_log_binding.h     loader-side binder resolution helper
│   │   ├── host_log_state.h       cross-DSO state ABI + binder declaration
│   │   ├── host_span.h            host-span data ABI
│   │   ├── log_level.h            shared levels + CANN mapping
│   │   └── unified_log.h          LOG_* ABI used by host and device code
│   └── host_log.h                 private HostLogger implementation interface
├── host_log.cpp                   host envelope, filter, clock anchor, STRACE grammar
└── unified_log_host.cpp           unified_log_* adapters → HostLogger

src/common/platform/
├── include/aicpu/device_log.h               device backend declarations
├── shared/aicpu/unified_log_device.cpp      LOG_* ABI → dev_vlog_* adapter
├── onboard/aicpu/device_log.cpp             onboard CANN backend
└── sim/aicpu/device_log.cpp                 dev_vlog_* → bound HostLogger adapter
```

There is no standalone `libsimpler_log.so`. Host consumers compile the two
host logger translation units directly. This makes each DSO self-contained:
loading a runtime can no longer fail because a logger artifact or a global
logger symbol was missing.

Each bindable private logger starts with a `NUL` fallback threshold and stays
silent until its loader binds the process-owned state. The `_task_interface`
owner is seeded with the configured user-facing threshold (`TIMING` by default)
before it forks workers or loads runtime modules. A missed binding therefore
appears as an observably silent module instead of output filtered at the wrong
level. A standalone executable that compiles the logger and has no loader must
seed its own state before logging; `query_device_hal` does this explicitly.

## Three-layer ABI

### Layer 1 — consumer macros

`common/unified_log.h` defines the macros consumers use:

```cpp
LOG_DEBUG(fmt, ...)
LOG_INFO(fmt, ...)
LOG_TIMING(fmt, ...)
LOG_WARN(fmt, ...)
LOG_ERROR(fmt, ...)
```

Each macro injects the source location and passes `__FUNCTION__` separately.

Host spans use `SimplerHostSpan` and `unified_log_host_span`. Callers supply
what happened—name, duration, invocation identity, and attributes—while
`HostLogger::log_host_span` owns the PID, TID, timestamp envelope, escaping,
and the single `[STRACE]` format string. Before constructing attributes, hot
call sites query `unified_log_host_span_enabled`; the logger rechecks the
TIMING threshold before validating or encoding a record.

### Layer 2 — unified logging ABI

The macros expand to five `extern "C"` functions declared by
`common/unified_log.h`:

```cpp
void unified_log_error(const char *func, const char *fmt, ...);
void unified_log_warn(const char *func, const char *fmt, ...);
void unified_log_timing(const char *func, const char *fmt, ...);
void unified_log_info(const char *func, const char *fmt, ...);
void unified_log_debug(const char *func, const char *fmt, ...);
```

Host targets compile `unified_log_host.cpp`; AICPU targets compile
`unified_log_device.cpp`. Host definitions have hidden visibility and are
resolved inside the target that contains them. The device implementation
forwards to its local backend.

### Layer 3 — backend primitives

`HostLogger::vlog` is the host-side authority for level gating. It formats a
complete record and performs one `write(2)` under its module-local mutex.
`HostLogger::log_host_span` additionally bounds and escapes machine-readable
fields so a STRACE record fits the portable `PIPE_BUF` floor. Its early gate
keeps direct ABI and legacy STRACE callers from paying those encoding costs
when TIMING is disabled; `vlog` remains the final check if the threshold changes
between the caller's query and emission.

The mutex serializes writers only within the DSO that owns it. Writers from
different DSOs, or from forked processes sharing a pipe, are indivisible only
when their single `write(2)` is no larger than that pipe's `PIPE_BUF`.
Machine-readable `[STRACE]` records satisfy that bound. Longer human-readable
records are best-effort and may interleave across module boundaries. Blocking
and drop accounting for those writes remain part of issue #1792 item 6.

The AICPU `dev_vlog_*` interface remains source-compatible on both platforms.
Sim implements it as a thin `va_list` adapter into its bound `HostLogger`, so it
shares the live threshold, envelope, destination, and fallback with the other
host-side modules in that process. Only real-silicon AICPU retains a separate
backend, because its records go through CANN dlog rather than a host process.

## Cross-DSO host state

Each host DSO has a private `HostLogger` object, but every copy in one process
reads the same `SimplerHostLogState`:

```c
typedef struct SimplerHostLogState {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t threshold;
    int32_t clock_anchor_pid;
} SimplerHostLogState;

int simpler_host_log_bind_state(SimplerHostLogState *state);
```

The native `_task_interface` extension owns the state for the process. The
fields are plain fixed-width integers to keep the ABI compiler-independent;
`host_log.cpp` accesses mutable fields with atomic builtins. ABI version and
size are checked before a module accepts the pointer.

Only `simpler_host_log_bind_state` is exported from a host logging consumer.
`HostLogger` and every `unified_log_*` definition are hidden. This avoids
accidentally recreating the old global-symbol singleton through ELF
interposition while still giving loaders one stable binding entry point.

`clock_anchor_pid` is also shared. Consequently the private logger copies
coordinate one successful `[CLOCK_ANCHOR]` per process. A negative PID is a
temporary writer claim; a failed stderr write releases the claim so the next
record can retry.

### Load and bind order

Python seeds the extension-owned state before C++ loads consumers:

```python
_initialize_host_log(level)
self._impl.init(host_path, aicpu_path, aicore_path, dispatcher_path,
                device_id, prewarm_config, enable_sdma, sim_context_path)
```

`ChipWorker::init` then performs the module-specific work:

1. On sim, load `libcpu_sim_context.so` with `RTLD_GLOBAL` once per path and
   retain it in a process-wide registry. Global visibility remains necessary
   for PTO simulator hooks, not for logging. Resolve its binder on the first
   load and pass the shared state.
2. Load `libhost_runtime.so` with `RTLD_LOCAL`.
3. Resolve the runtime's binder from its handle and pass the shared state before
   resolving or calling its regular C API.
4. Call `simpler_init(...)` to configure CANN, attach the device, and take
   ownership of executor binaries.
5. When the runtime later loads a generated host orchestration SO or sim
   AICore SO, resolve that SO's binder from its own handle and pass the same
   state before invoking its entry point.

RAII guards close partially loaded DSOs when initialization fails. A successful
host-runtime handle remains owned by `ChipWorker` and is closed during
finalization. Successful sim-context handles remain in the process registry;
this preserves their process-lifetime simulator state and pthread keys across
sequential `ChipWorker` instances.

## Output formats

### Host

```text
[mono_ns=MONOTONIC_NS][T0xTID][LEVEL] func: [file.cpp:line] message
```

The prefix clock is monotonic nanoseconds, so envelope ordering and host span
timestamps share one clock and are unaffected by wall-clock corrections.
`T0x...` is `pthread_self()`.

The first TIMING-enabled record in a process emits a mapping to Unix wall time:

```text
[mono_ns=...][T0x...][TIMING] clock_anchor: [CLOCK_ANCHOR] v=1 pid=<pid> mono_ns=<ns> wall_ns=<ns>
```

For a record at `record_ns`, the corresponding wall time is approximately
`wall_ns + record_ns - mono_ns`. The anchor is TIMING so it is present whenever
the default-threshold host trace is present. A forked child emits its own
anchor because coordination is keyed by PID.

`strace_timing.py` parses these anchors and adds wall-clock metadata to Chrome
trace/swimlane JSON while leaving event timestamps monotonic and relative. See
[Host trace](dfx/host-trace.md) for the rendering contract.

### AICPU sim

```text
[mono_ns=MONOTONIC_NS][T0xTID][LEVEL] func: [file.cpp:line] message
```

Sim AICPU runs on a host CPU inside the host process and uses the same envelope
and destination as every other bound host module. The `dev_vlog_*` names remain
as the compatibility boundary `unified_log_device.cpp` consumes. Onboard AICPU
uses the CANN dlog format. Device TIMING uses CANN WARN and adds a `[TIMING]`
message tag.

## Configuration flow

| Stage | Action | Source |
| ----- | ------ | ------ |
| Python import | Register `TIMING` / `NUL`; default the `simpler` logger to TIMING | `python/simpler/_log.py` |
| `Worker.init()` | Normalize the Python logger level, seed native state before the first fork, and point the `simpler` logger at the host logger | `python/simpler/worker.py` |
| `ChipWorker.init()` | Re-seed inherited native state in a chip child, then enter C++ | `python/simpler/task_interface.py` |
| `_ChipWorker.init()` | Load sim context and host runtime, then bind each module's logger state | `src/common/worker/chip_worker.cpp` |
| `simpler_init` | Onboard maps the bound threshold to CANN; attach and take executor binaries | `src/common/platform/{onboard,sim}/host/c_api_shared.cpp` |
| Nested host load | Bind generated host orchestration/AICore logger state before entry | runtime maker / sim device runner |
| AICPU init | Sim binds the live host state; onboard snapshots CANN policy | platform AICPU init |

The Python level is still sampled during worker initialization. Calling
`logger.setLevel(...)` does not itself call the native setter; recreate or
reinitialize the worker to apply a new Python configuration. Within a process,
all bound host modules observe a native state update immediately instead of
requiring threshold fan-out to every DSO.

### The Python logger is a client, not a second system

Seeding the native threshold also installs a handler on the `simpler` logger that
forwards each record through `unified_log_*`. Before that, Python and C++ agreed
only on a threshold: a Python record carried `[%(levelname)s] %(message)s` — no
timestamp, no thread id — so it could not be ordered against a C++ record no
matter where either was written, and it went to whatever handler happened to be
on the root logger rather than to the host logger's own output. Now one envelope,
one clock and one destination cover every record in the process.

Two properties this deliberately keeps:

- **Propagation is untouched**, so a record still reaches the root logger as
  well. That is what keeps an interactive console readable and what keeps
  `caplog.at_level(..., logger="simpler")` working in the tests that assert on
  warnings. The host log is the complete copy; the console is a view of it.
- **The handler is installed where the threshold is seeded, not at import.**
  `import simpler` must keep working when `_task_interface` is missing or stale —
  the build-stamp guard in `simpler/task_interface.py` raises on every
  source-tree move until a rebuild — so putting the extension on the import path
  of the logging surface would lose the logger exactly when it is needed to say
  why. Records logged before a worker is initialized stay on the root logger's
  handler.

A record's level is rounded toward the milder name on the way through (`35`
becomes `WARN`), the opposite of how a *threshold* is normalized: asking for a
threshold of 35 must not silently admit warnings, but a record at 35 is a warning
someone gave a custom number.

### Forked chip subprocesses

The hierarchical parent seeds native state before `fork()` and passes the
normalized level explicitly to `_chip_process_loop`. The child re-seeds its
inherited copy before loading runtime modules. This covers both chip-owning L3
workers and higher-level processes that emit scheduler spans without loading a
chip runtime.

### Onboard AICPU severity is CANN-owned

Onboard AICPU reads severity from CANN's dlog rather than from the host state.
`simpler_init` calls `dlog_setlevel(-1, HostLogger.cann_level(), 0)` before
device attach unless `ASCEND_GLOBAL_LOG_LEVEL` is already set; the environment
therefore wins over the Python logger.

| Simpler threshold | CANN level |
| ----------------- | ---------- |
| `debug` | DEBUG |
| `info` | INFO |
| `timing` | WARN |
| `warn` | WARN |
| `error` | ERROR |
| `null` | NUL |

The AICPU samples CANN during its init kernel and latches the result. Configure
`ASCEND_GLOBAL_LOG_LEVEL` before `Worker.init()` if direct CANN control is
needed.

## Build orchestration

There is no logger build step or logger field in `RuntimeBinaries`. Instead:

- `_task_interface`, all host runtimes, sim-context, and sim AICore targets add
  `host_log.cpp` and `unified_log_host.cpp` to their source lists.
- Sim AICPU targets add `host_log.cpp` while keeping `unified_log_device.cpp`;
  the device ABI delegates to HostLogger there.
- Host-compiled generated orchestration SOs receive the same sources through
  `KernelCompiler.get_orchestration_cache_inputs`; those sources therefore
  participate in the scene-test cache key.
- AICPU orchestration built for silicon keeps the device logging ABI and does
  not compile the host logger.
- `libcpu_sim_context.so` remains a standalone process-global simulator helper
  and is built once for sim platforms.

## Where to look

| You want to … | Look at |
| ------------- | ------- |
| Change the user-facing level model | `python/simpler/_log.py` and `docs/testing.md` |
| Change host output or STRACE grammar | `src/common/log/host_log.cpp` |
| Change the shared-state ABI | `src/common/log/include/common/host_log_state.h` |
| Change sim AICPU adaptation | `src/common/platform/sim/aicpu/device_log.cpp` |
| Change onboard CANN tagging | `src/common/platform/onboard/aicpu/device_log.cpp` |
| Add a host logging consumer | compile both host logger sources, include `src/common/log/include`, and bind state during module init |
| Add a level | `log_level.h`, `_log.py`, `simpler_setup/log_config.py`, and AICPU `set_log_level` |
