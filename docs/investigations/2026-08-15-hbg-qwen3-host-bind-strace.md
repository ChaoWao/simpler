# HBG Qwen3-14B host per-step bind decomposition (a2a3, fine-grained STRACE)

- **Date**: 2026-08-15 (data collected 2026-08-14)
- **Branch**: `perf/hbg-orch` @ `3f4686c1` + bind-path STRACE instrumentation
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

## Optimization directions (not pursued here)

- **Reuse host-view mappings across rounds**: cache `halHostRegister` results
  keyed by `dev_ptr` and skip unregister when the next round re-registers the
  same buffer — would remove ~256 ms/step (12.6% of host wall). The mapping
  lifetime would move from per-run to per-caller-owned-tensor.
- **Skip re-staging unchanged tensors in `bind.args`** (addresses stable
   across rounds for qwen3 weights/KV) — would remove most of the 1.26 s.
   Needs a caller-visible contract for buffer lifetime, so it is a design
   change, not a local fix.

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

The a5 counterpart (`examples/a5/host_build_graph/qwen3_14b_decode`) carries
the same ten spans; run the same command with `a5` paths on a5 silicon for
the cross-arch comparison.
