# Command-line flags and tools

How you select what runs, turn diagnostics on, and read the artifacts back.

## Running tests and examples

```bash
pytest examples tests/st --platform a2a3sim            # simulation, no device
pytest examples tests/st --platform a2a3 --device 4-7  # hardware
python examples/my_example/test_my_example.py -p a2a3sim   # standalone, no pytest
```

### Selecting what runs

| Flag | Meaning |
| ---- | ------- |
| `--platform` | Target platform: `a2a3sim`, `a2a3`, `a5sim`, `a5`. Required |
| `--device` | Device id or range, e.g. `0` or `4-7`. Default `0` |
| `--runtime` | Restrict to one runtime (`tensormap_and_ringbuffer`, `host_build_graph`) |
| `--level` | Restrict to one level, e.g. `--level 2` |
| `--case` | Case selector, repeatable: `Foo`, `ClassA::Foo`, or `ClassA::` for a whole class |
| `--manual` | Manual-case handling: `exclude` (default), `include`, `only` |
| `--rounds` | Run each case N times. Default `1`; use this so first-run effects do not dominate a measurement |
| `--skip-golden` | Skip golden comparison — benchmark mode |
| `--require-pto-isa` | Fail the session immediately if PTO-ISA cannot be resolved, instead of deferring to the per-test lazy path |
| `--sanitizer` | Run against a sanitizer build; see [Compiler Sanitizers](../../sanitizers.md) |

### Diagnostics

All off by default. Each also exists as a `CallConfig` field for direct-`Worker`
programs — see the [Python API reference](python-api.md).

| Flag | Meaning |
| ---- | ------- |
| `--enable-l2-swimlane` | Per-task timing. Bare flag = level 4 (full); `1` AICore timing, `2` + dispatch/fanout, `3` + scheduler phases, `4` + orchestration. **L2 only** |
| `--enable-swimlane-overhead` | Adds the 8 Overhead Analysis counter tracks. Requires `--enable-l2-swimlane` **and** a `deps.json` — add `--enable-dep-gen` if absent |
| `--enable-pmu` | AICore hardware counters. Bare flag = `PIPE_UTILIZATION` (2); pass an event type to override, e.g. `--enable-pmu 4` |
| `--dump-args` | Capture per-task arguments. `0` off, `1` partial (only args marked via `Arg::dump(...)`) |
| `--enable-dep-gen` | Capture the dependency graph (first round only) |
| `--enable-scope-stats` | Per-scope peaks to `<output_prefix>/scope_stats.jsonl` |

Which flag answers which question is tabulated in
[How-to: profile a kernel](../how-to/profile-a-kernel.md).

## Analysis tools

Shipped in the wheel under `simpler_setup.tools`, run with `python -m`. Full
flags in [`simpler_setup/tools/README.md`](../../../simpler_setup/tools/README.md).

| Tool | Input | Output |
| ---- | ----- | ------ |
| `strace_timing` | run log | host/device wall-clock breakdown; `--rounds-table` for per-round |
| `swimlane_converter` | `l2_swimlane_records_*.json` | Perfetto trace + per-function task summary; `--overhead` adds the scheduler-overhead counters (needs `deps.json`) |
| `sched_overhead_analysis` | swimlane + deps | scheduler bottleneck vs starvation verdict |
| `l0_swimlane` | args dump | intra-core pipeline trace for one task |
| `deps_viewer` | `deps.json` | the dependency graph |
| `critical_path` | `deps.json` | critical path through the graph |
| `dump_viewer` | args dump | captured per-task arguments |
| `scope_stats_plot` | `scope_stats.jsonl` | per-scope resource peaks |

```bash
python -m simpler_setup.tools.strace_timing <run.log> --rounds-table
python -m simpler_setup.tools.swimlane_converter <l2_swimlane_records_*.json>
```

## Environment variables

| Variable | Effect |
| -------- | ------ |
| `SIMPLER_OP_EXECUTE_TIMEOUT_US` | Overrides the op-execute timeout (default 45 s) |
| `SIMPLER_STREAM_SYNC_TIMEOUT_MS` | Overrides the stream-sync timeout (default 50 s) |
| `SIMPLER_SCHEDULER_TIMEOUT_MS` | Overrides the scheduler timeout (default 10 s) |
| `ASCEND_PROCESS_LOG_PATH` | Redirects the device log into a directory you own; the directory must already exist |
| `ASCEND_HOME_PATH` | CANN toolkit location; required for hardware platforms |

The three timeouts are validated against each other at startup, and CI runs
deliberately shorter values than the defaults — read a CI timeout against the CI
values. Details in
[local-timeout-defaults](../../troubleshooting/local-timeout-defaults.md).

## See also

- [Testing](../../testing.md) — the full testing guide, including how the suites are composed
- [CI Pipeline](../../ci.md) — what each pipeline stage runs
- [How-to: debug a failed run](../how-to/debug-a-failed-run.md)
