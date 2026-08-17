# HBG Qwen3-14B host per-step bind decomposition (a2a3 + a5, fine-grained STRACE)

- **Date**: 2026-08-15 (a2a3 data 2026-08-14; a5 data 2026-08-17)
- **Branch**: `perf/hbg-bind-strace-fine-grained` @ `bd6867be`
- **Workload**: `examples/a2a3/host_build_graph/qwen3_14b_decode` (40-layer
  graph execution, batch16 / seq3500, `--skip-golden`)
- **Method**: 50 rounds on one exclusively-held even die (device 0) via
  `task-submit`; round 0 (first graph record + arena build) dropped; per-span
  median with ±3 MAD outlier trim. `[STRACE]` markers captured by pytest `-s`
  stderr redirect, parsed with `strace_timing` + per-span aggregation.

## Question

Where does the host spend time in each HBG bind, specifically: (a) how long
the host-side orchestration (graph record/replay) takes, and (b) how long the
post-graph H2D upload takes.

## Instrumentation

`src/{a2a3,a5}/runtime/host_build_graph/host/runtime_maker.cpp` gained ten
`[STRACE]` spans (identical on both arches):

| span | covers |
| ---- | ------ |
| `simpler_run.bind.args` | per-tensor `device_malloc` + H2D staging + host-view register |
| `simpler_run.bind.prebuilt` | arena build: layout/commit, static arena, heap/SM acquire, host orch, arena H2D |
| `simpler_run.bind.prebuilt.arena_h2d` | prebuilt runtime-arena image H2D (orch block skipped) |
| `simpler_run.bind.host_orch` | whole host orchestration window |
| `simpler_run.bind.host_orch.sm_init` | dep_gen capture start, host SM mirror alloc + memset, orchestrator re-init, SM handle init, graph state alloc, finalize_after_wire |
| `simpler_run.bind.host_orch.orch_entry` | orch entry body: scope begin, graph record (round 0) or definition replay (steady), scope end, orchestration done |
| `simpler_run.bind.host_orch.graph_upload` | graph submission POD upload (`upload_graph_submissions`) |
| `simpler_run.bind.host_orch.relocate` | `relocate_host_orch_image` (per-slot pointer host→device rewrite) |
| `simpler_run.bind.host_orch.sm_h2d` | live-prefix SM H2D (header+descriptors, payloads, slot states, completion flags) |
| `simpler_run.bind.host_orch.tensor_view_unmap` | `HostTensorAccessor::close()` — `halHostUnregister` of every staged tensor's host view |

## Result (steady state, median ms, n≈49 after trim)

| span | med ms | share of `simpler_run` |
| ---- | ------ | ---------------------- |
| `simpler_run` (host wall) | 2028 | 100% |
| `bind.args` | 1257 | 62% |
| `bind.prebuilt` | 262 | 13% |
| `host_orch.tensor_view_unmap` | **256** | **12.6%** |
| `host_orch.orch_entry` | **2.9** | 0.14% |
| `host_orch.sm_h2d` | 1.7 | <0.1% |
| `host_orch.sm_init` | 0.04 | <0.1% |
| `host_orch.relocate` | 0.03 | <0.1% |
| `prebuilt.arena_h2d` | 0.12 | <0.1% |
| `runner_run` | 34.8 | 1.7% |
| `device_wall` (clk=dev) | 33.3 | 1.6% |
| `validate` | 738 | 36% |

## Conclusions

1. **Graph construction/replay is not a bottleneck.** The orch entry body is
   ~2.9 ms steady state (round-0 record ≈ 5 ms). dlopen/dlsym of the orch .so
   happen once per callable in `register_callable_impl`, never per run.
2. **Post-graph H2D is negligible** (~1.8 ms total across arena/graph/SM).
3. **The `host_orch` window's cost is its teardown, not its work**: 256 ms of
   262 ms is `tensor_view_unmap` — serial `halHostUnregister` of every staged
   tensor's SVM host view (dozens of buffers, 38 GiB total). Each unregister
   is a kernel-mode unmap with TLB shootdown.
4. **`bind.args` (1.26 s) dominates the host wall** — per-round full-argument
   staging (device_malloc + H2D of all inputs + `halHostRegister` host views)
   for tensors whose addresses do not change across rounds.
5. `validate` (0.74 s D2H copy-back) is scene-test-specific (output
   verification), not part of a serving loop.

## a5 cross-arch comparison (2026-08-17)

Same method on a5 silicon (Ascend950PR, device 2 exclusively held via
`task-submit`, 50 rounds, round 0 dropped, per-span median with ±3 MAD trim).
The workload is identical by construction: both arch examples load the same
`simpler_setup/goldens/qwen3_14b_decode` fixture (BATCH=16 / seq 3500 / 40
layers).

| span | a5 med ms | a5 share | a2a3 med ms | a2a3 share | a5/a2a3 |
| ---- | --------: | -------: | ----------: | ---------: | ------: |
| `simpler_run` (host wall) | 9807 | 100% | 2028 | 100% | 4.9x |
| `bind.args` | **7113** | **72.3%** | 1257 | 62% | 5.7x |
| `bind.prebuilt` | 46.1 | 0.5% | 262 | 13% | 0.18x |
| `bind.prebuilt.arena_h2d` | 1.47 | <0.1% | 0.12 | <0.1% | 12x |
| `host_orch` | 43.6 | 0.4% | 262 | 13% | 0.17x |
| `host_orch.sm_init` | 0.10 | <0.1% | 0.04 | <0.1% | 2.6x |
| `host_orch.orch_entry` | 25.4 | 0.3% | 2.9 | 0.14% | 8.8x |
| `host_orch.relocate` | 0.04 | <0.1% | 0.03 | <0.1% | 1.4x |
| `host_orch.sm_h2d` | 13.1 | 0.1% | 1.7 | <0.1% | 7.7x |
| `host_orch.tensor_view_unmap` | **~0** | 0.0% | **256** | **12.6%** | ≈0 |
| `runner_run` | 27.7 | 0.3% | 34.8 | 1.7% | 0.80x |
| `runner_run.device_wall` (clk=dev) | 26.4 | 0.3% | 33.3 | 1.6% | 0.79x |
| `validate` | 2625 | 26.7% | 738 | 36% | 3.6x |

Cross-arch findings:

1. **`bind.args` dominates on a5 even more than on a2a3** (72.3% vs 62%;
   7.1 s vs 1.26 s). Every host↔device transfer span is several times more
   expensive on a5 in this setup (`validate` D2H 3.6x, `sm_h2d` 7.7x,
   `arena_h2d` 12x), so the gap is a lower host↔device copy bandwidth, not
   an `bind.args`-specific defect. The arg-staging-skip optimization below
   pays more on a5 than on a2a3.
2. **`tensor_view_unmap` ≈ 0 on a5 is structural, not a fast path.**
   `DeviceRunnerBase::register_device_memory_to_host`
   (`src/common/platform/onboard/host/device_runner_base.h`) returns nullptr
   unless the arch overrides it — a2a3 overrides it with
   `halHostRegister(DEV_SVM_MAP_HOST)`; a5 onboard has no host-map path, so
   `HostTensorAccessor::add` falls back to the staging host pointer,
   `close()` has no mappings to release, and the 256 ms a2a3 teardown does
   not exist. The host-view-mapping-reuse optimization is therefore **a2a3
   only**; a5's `host_orch` window is already cheap (43.6 ms), with its cost
   being `orch_entry` replay (25.4 ms, 8.8x a2a3 — still only 0.3% of wall).
3. **a5 `device_wall` is faster than a2a3's** (26.4 vs 33.3 ms): the chip
   executes the graph quicker; the host-side transfer path is where a5 loses.
4. `bind.args` drifts upward over a run: ~6.2 s (rounds 2–18) rising to
   ~6.9–7.6 s (round 21 onward), a ~15% creep within 50 rounds; the median
   lands in the later, slower regime.

## Optimization directions (not pursued here)

- **Reuse host-view mappings across rounds** (a2a3 only): cache
  `halHostRegister` results keyed by `dev_ptr` and skip unregister when the
  next round re-registers the same buffer — would remove ~256 ms/step (12.6%
  of host wall). The mapping lifetime would move from per-run to
  per-caller-owned-tensor. Not applicable on a5, which has no host-map path
  and thus no unmap cost.
- **Skip re-staging unchanged tensors in `bind.args`** (addresses stable
   across rounds for qwen3 weights/KV) — would remove most of the 1.26 s on
   a2a3 and most of the 7.1 s on a5, where this span is 72% of the host
   wall. Needs a caller-visible contract for buffer lifetime, so it is a
   design change, not a local fix.

## Reproduce (a2a3)

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
mkdir -p tmp/hbg_qwen_strace/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/tmp/hbg_qwen_strace/ascend"
task-submit --timeout 7200 --max-time 7200 --device 0 --run "\
  .venv/bin/python -m pytest examples/a2a3/host_build_graph/qwen3_14b_decode \
    --platform a2a3 --device \$TASK_DEVICE --manual include \
    --rounds 50 --skip-golden -x -q -s > tmp/hbg_qwen_strace/rounds50.log 2>&1"
python -m simpler_setup.tools.strace_timing tmp/hbg_qwen_strace/rounds50.log --tree
```

`-s` is required: `[STRACE]` markers are written by the host logger straight
to stderr (`write_stderr`), so pytest's default capture swallows them for
passing tests.

## Reproduce (a5)

Same command with `a5` paths on a5 silicon (the 2026-08-17 data used device
2). Two a5 specifics observed on the collection box:

- After a rebuild, sync `libhost_runtime.so` from
  `build/cache/a5/onboard/host_build_graph/host/` to both `build/lib/...` and
  the venv's `simpler_setup/_assets/build/lib/...` before running — a stale
  loaded `.so` silently produces span-less logs that still look healthy.
- In containers where `npu-smi` cannot initialize (dcmi -8005, shared
  devices), the precheck script fails even on matching silicon; confirm the
  chip (e.g. `npu-smi info` inside `task-submit`) before proceeding.
