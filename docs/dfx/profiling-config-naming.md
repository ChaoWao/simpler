# Profiling / DFX Configuration Naming Rules

This is the authoritative naming convention for the compile-time macros and
runtime environment variables that configure profiling, DFX, and timeout
behavior. These names are configuration surface — they are read by CI,
onboard scripts, and external consumers (e.g. pypto-serving), so they must be
consistent and self-describing.

See [profiling-framework.md](profiling-framework.md) for the collector
architecture, and [dfx/host-trace.md](host-trace.md) for the `[STRACE]`
host-trace facility.

## The four rules

### 1. Compile macros are bare names; env vars carry a value-shape suffix

A reader must be able to tell from the name alone whether a knob is a
compile-time `#define` (set via header or `-D`) or a runtime environment
variable (set via `export` / `env:`).

- **Compile macros**: bare feature name — `SIMPLER_<FEATURE>`.
- **Env vars**: must end in a value-shape suffix that signals the runtime
  value kind:
  - `_ENABLE` — boolean on/off
  - `_LEVEL` — numeric verbosity tier
  - `_TYPE` — enum selector
  - `_US` / `_MS` — numeric with a time unit

| Name | Kind | Why |
| ---- | ---- | --- |
| `SIMPLER_HOST_STRACE` | macro | bare → compile switch |
| `SIMPLER_DEVICE_STRACE_ENABLE` | env | `_ENABLE` → runtime boolean |
| `SIMPLER_PMU_EVENT_TYPE` | env | `_TYPE` → runtime enum |
| `SIMPLER_OP_EXECUTE_TIMEOUT_US` | env | `_US` → runtime numeric (microseconds) |

### 2. The name reflects what the knob actually gates

Do not use "PROFILING" as a catch-all.

- `SIMPLER_DFX` is the **device DFX instrumentation build switch** (the
  umbrella). It gates a broad set of device instrumentation infrastructure
  (cycle counters, chip swimlane, scope stats, PMU, device-phase timing), not
  just "profiling logs."
- The device **sub-tier** macros keep `PROFILING` because they gate actual
  profiling counters: `SIMPLER_ORCH_PROFILING` (orchestrator task/cycle
  counters), `SIMPLER_SCHED_PROFILING` (scheduler dispatch hit/miss + cycle),
  `SIMPLER_TENSORMAP_PROFILING` (tensor-map hash-table chain/overlap stats).
- `SIMPLER_HOST_STRACE` gates the `[STRACE]` facility. Onboard, it also gates
  capture work whose only consumer is a device-domain `[STRACE]` marker — so it
  is `HOST_STRACE`, not `PROFILING`.

### 3. Runtime-subsystem-specific knobs carry an owner prefix; platform knobs do not

- **Platform-layer** knobs (collectors shared by every runtime) take **no**
  subsystem qualifier: `SIMPLER_DFX_FLAG_PMU`, `SIMPLER_PMU_EVENT_TYPE`.
- **Runtime-subsystem-specific** knobs carry the owning subsystem as a prefix:
  - `HBG_` — host_build_graph runtime
  - `TMR_` — the orchestrator/scheduler dispatch subsystem

This keeps the per-run DFX flag namespace (`SIMPLER_DFX_FLAG_*` for platform
collectors; `SIMPLER_HBG_DFX_FLAG_*` / `SIMPLER_TMR_DFX_FLAG_*` for future
runtime-specific ones) unambiguous about ownership.

### 4. Everything carries the `SIMPLER_` project prefix

`PTO2_` is the device runtime's internal namespace (89 identifiers:
`PTO2_MAX_RING_DEPTH`, `PTO2_ERROR_*`, `PTO2Runtime`, …). Configuration
surface exposed to users / CI / external consumers is unified under
`SIMPLER_`, regardless of which internal namespace implements it.

## Compile-time gates vs runtime emission

A `[STRACE]` line or profiling data point reaching the host log is governed
by **independent layers**:

| Layer | Control | Default |
| ----- | ------- | ------- |
| Compile-time (does the code exist?) | macros (`SIMPLER_DFX`, `SIMPLER_HOST_STRACE`, `SIMPLER_*_PROFILING`) | umbrella on, sub-tiers off |
| Per-run (does this run collect X?) | `SIMPLER_DFX_FLAG_*` bitmask via `CallConfig` | none selected |
| Runtime emission (does it actually emit?) | env (`SIMPLER_DEVICE_STRACE_ENABLE`, log level) | on |
| Runtime detail tier | `get_chip_swimlane_level()` | AICPU_TIMING |

`SIMPLER_HOST_STRACE` (compile) gates whether `[STRACE]` markers exist at all;
`SIMPLER_DEVICE_STRACE_ENABLE` (runtime env) independently gates device-domain
markers. Onboard device-phase capture also requires the `LOG_TIMING` level to
be visible. The capture predicate is shared by buffer setup/reset, readback,
and emission, so disabled markers do not leave marker-only transfers behind.

## Reference: the configuration surface

### Compile macros (`profiling_config.h`)

| Macro | Default | Gates |
| ----- | ------- | ----- |
| `SIMPLER_DFX` | 1 | device DFX instrumentation umbrella (cycle / swimlane / scope-stats / PMU / device-phase + the three sub-tiers) |
| `SIMPLER_ORCH_PROFILING` | 0 | orchestrator-phase counters (requires `SIMPLER_DFX`) |
| `SIMPLER_SCHED_PROFILING` | 0 | scheduler hot-path counters (requires `SIMPLER_DFX`) |
| `SIMPLER_TENSORMAP_PROFILING` | 0 | tensor-map hash-table counters (requires `SIMPLER_ORCH_PROFILING`) |
| `SIMPLER_HOST_STRACE` | 1 | `[STRACE]` macros (`strace.h`) and onboard capture used only by device-domain markers; independent of `SIMPLER_DFX` |

### Env vars

| Env | Layer | Value | Gates |
| --- | ----- | ----- | ----- |
| `SIMPLER_DEVICE_STRACE_ENABLE` | host general | bool | device-domain `[STRACE]` capture/emission onboard; emission in sim |
| `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE` | host runtime | bool | serial orch→scheduler transition (TMR subsystem) |
| `SIMPLER_PMU_EVENT_TYPE` | host platform | enum | which PMU event the PMU collector samples |
| `SIMPLER_OP_EXECUTE_TIMEOUT_US` | host platform | µs | op-execute timeout (overrides the `platform_config.h` compile default) |
| `SIMPLER_STREAM_SYNC_TIMEOUT_MS` | host platform | ms | stream-sync timeout |
| `SIMPLER_SCHEDULER_TIMEOUT_MS` | host platform | ms | scheduler no-progress timeout |

### Per-run DFX flag bitmask (`platform_config.h`)

The `SIMPLER_DFX_FLAG_*` constants select which DFX collectors a given run
collects; accessed via `SIMPLER_GET/SET/CLEAR_DFX_FLAG`:

`SIMPLER_DFX_FLAG_NONE`, `_DUMP_ARGS`, `_CHIP_SWIMLANE`, `_PMU`, `_DEP_GEN`,
`_SCOPE_STATS`.

Platform collectors carry no subsystem qualifier. Future runtime-specific
collectors take `SIMPLER_HBG_DFX_FLAG_*` / `SIMPLER_TMR_DFX_FLAG_*`.

## Adding a new knob

1. Decide **compile macro vs runtime env**. If the feature must be absent from
   the binary entirely (size / hot-path codegen), use a macro. If it should be
   toggleable per process without a rebuild, use an env var.
2. Apply the rules above to pick the name (project prefix, owner prefix if
   runtime-specific, value-suffix if env).
3. If a macro, define it with an `#ifndef` guard in `profiling_config.h`.
4. If an env, declare the name once as a `constexpr const char *` and read it
   via `getenv(SYMBOL)` (never a bare string literal — the indirection makes
   the rename surface discoverable).
5. Add a row to the relevant reference table above.
