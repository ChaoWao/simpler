# DeepSeek-V4 FLASH decode on host_build_graph

The `tensormap_and_ringbuffer` DeepSeek-V4 case
(`examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode/`) run under
`host_build_graph`: same 43-layer network, same 367 kernels, same fixture, same
comm-window protocol. Only the runtime changes — HBG compiles the orchestration
with the host `g++`, runs it on the host CPU instead of the AICPU, and ships the
built shared-memory image to the device, which then boots scheduler-only.

The case exists to measure **host-side graph construction** and to prove the
device can execute what the host built. It is deliberately not a numerics test.

## What differs from the TMR case

`kernels/orchestration/decode_fwd_hostbuild.cpp` is the TMR orchestration with
two edits, **51 lines in all**, kept as the non-Graph baseline.
`kernels/orchestration/decode_fwd_graph.cpp` — the file the test points at —
carries the same two edits and additionally recasts the 20-iteration decoder
layer loop (40 of the 43 layers) as one `rt_submit_graph` per iteration: the
layer's task set becomes the Graph body (a free function reading its per-layer
views, scales and indices through `GraphTaskArgs`, positionally), and the host
records a 744-node Definition once instead of submitting the loop's ~15600
tasks individually. The runtime is untouched.

| Edit | Sites | Why |
| ---- | ----- | --- |
| `get_tensor_data(recv_count_out, …)` → `HBG_RECV_ROWS_PER_EXPERT` | 10 | HBG builds the whole graph before the device runs anything, so a read of a **task-produced** tensor has no value to return. The constant holds the per-expert tile loops at their real trip count (`ceil(16/16) == 1`, which is what the `h_i8 [512, 2048]` layout budgets per expert). |
| `set_initial_value(0)` dropped | 6 | The fill target is a GM-heap **device** address and the host orchestrator cannot store to it. Leaving the call in place segfaults the chip subprocess — see "Runtime gaps" below. |

The other **31** `get_tensor_data` reads are left alone: they read external
tensors (`ext_num_tokens_per_owner`, `hc_attn_scale_*`, `hc_ffn_scale_*`), which
the runtime stages with a host view and which therefore return real values.

Everything else — submit order, dependencies, scope nesting,
`valid_rows = min(n_rows - t0, 16)` — is byte-identical to the TMR source inside
the Graph body. The graph keeps the size and shape of the real one (the 15971
device-side tasks; 1131 host-submitted with the Graph collapse) but not the
fixture's routing, hence `skip_golden`.

## Status: host construction works, device execution stalls 12 tasks from the end

With the Graph form the host records the 744-node Definition and boots with
**1131 tasks on host** (down from 15991 in the submit-everything form) — this is
measured, on both ranks. The device-side replay of a Definition that large is
**not yet exercised**: an unskipped Graph-form run fails earlier, in Graph
activation (`sched_error_code=5 INVALID_ARGS` from the scheduler's graph
queues), before the tail. The numbers below are the **non-Graph baseline**
(hostbuild orchestration, no skip): the device executes
**15959 of 15971 tasks** and stalls:

```text
TASK ring=0 task_id=15959 state=RUNNING fanin_met=2/2 kernels=[aic:355 aiv0:-1 aiv1:-1]
     running_on=[cores=[core=0(aic) core=1(aic) core=3(aic) … 13 AIC cores]]
SUMMARY completed=15959/15971 scan_ready=0 scan_waiting=11 scan_running=1
CLUSTER cluster_id=0 aic=core0(busy kernel=355 task=15959 cond_reg_state=ack) …
```

`sub_class=S1:running-stalled`; both ranks; bit-identical across every run.

Task 15959 is `hc_head_linear` (kernel 355) — an AIC-only SPMD matmul in the
head section, `block_num=16`, submitted with no predicate and no
`require_sync_start`. Thirteen of its sixteen blocks sit at `cond_reg_state=ack`
(dispatch acknowledged, FIN never raised) and the remaining eleven tasks of the
network wait behind it.

### What has been ruled out

Each row was tested by changing exactly that one thing and re-running the full
network on hardware. The stall reproduced at the identical task, core count and
`completed` value every time.

| Ruled out | How |
| --------- | --- |
| pto-isa version | Stalls identically on `83d01313`, `0cefc9a5` and `f51c92f6`. (The *earlier* stall at task 2307 — issue #1839 — is a genuine ISA regression and **is** fixed by `f51c92f6`; it is a different stall.) |
| The orchestration edits above | An earlier variant that instead used dispatch predicates and a static tile grid stalls at the same task. So does one with the predicate rewrite removed. |
| `set_initial_value` | Present and absent both stall identically. |
| Runtime modifications | A `host_tensor_fill` seam was written to support `set_initial_value` on HBG and later reverted; the stall is identical with it, without it, and on a pristine tree. |
| The kernel itself | `hc_head_linear` completes normally under TMR on the same branch, ISA and build. |
| Kernel-internal spin | All loop bounds in `hc_head_linear` are compile-time constants; `t_dim`/`t_linear` are unused in the body. `ptoas_auto_sync_tail(kBarrierAll)` expands to `pipe_barrier(PIPE_ALL)` — intra-core, not cross-core. There is no cross-block synchronization to deadlock on. |
| Bad tensor addresses | A host-side probe at the submit site prints `t_dim=8 t_linear=16 block_num=16`, `x_flat [8,16384] @0x12c0c0742000`, `hc_head_fn [4,16384] @0x12c0c0702000`, `mixes_raw [16,16] @0x12cd75153400 size=1024`. Shapes and sizes are self-consistent and match the constants the kernel hardcodes. |
| Slow-not-hung | Raising `SIMPLER_SCHEDULER_TIMEOUT_MS` from 10 s to 60 s (with `SIMPLER_OP_EXECUTE_TIMEOUT_US`/`SIMPLER_STREAM_SYNC_TIMEOUT_MS` raised to keep the required ordering) leaves the stall unchanged; the device log confirms the window grew from exactly 10 s to exactly 60 s. |
| Early-dispatch gating | A gated dispatch parks a core at its doorbell, which would look exactly like this. It cannot fire: `force_gate=true` is passed only from `stage_consumer_blocks`, whose only source `early_dispatch_queues` has no producer outside its own drain (Milestone 1 leaves early-dispatch stubbed). |
| Pointer relocation | No `cannot relocate` / `outside both SM and arena windows` diagnostic; the SM H2D reports no error. |
| Resource exhaustion | The ring and heap allocators report a fatal on exhaustion. None fired: `orch_error_code=0`, orchestration completed, the graph uploaded, 15959 tasks ran. |

### What is still open

The contradiction is that thirteen mutually independent, fixed-trip-count blocks
of one SPMD matmul, holding sane addresses, sit in `ack` forever — while the
same kernel completes under TMR.

Two threads worth pulling:

1. **What the cores actually received.** `cond_reg_state=ack` says the core took
   the dispatch and is executing. A hang with no spin loop in the kernel points
   at an MTE access that never returns, which would follow from a malformed
   dispatch payload rather than from the tensor descriptors the probe printed.
   `core_swimlane` / `insight_trace` can show where inside the kernel the core
   stopped.
2. **Why thirteen and not sixteen.** Block dispatch is deliberately partial —
   `claim_block_range` takes as many blocks as there are free cores and re-pushes
   the task for the rest, and `enter_drain_mode` only applies to
   `require_sync_start` tasks. Thirteen is stable across runs, so either three
   blocks completed and thirteen did not (odd, since the blocks are symmetric) or
   only thirteen were ever placed and the remaining three can never get a core.
   Distinguishing these two is the cheapest next measurement.

## Runtime gaps this case exposed

Neither is required for this case as it stands, but both are real and neither has
test coverage on `host_build_graph`:

- **`TensorCreateInfo::set_initial_value()` segfaults.** `alloc_tensors` hands
  out GM-heap **device** addresses while `fill_tensor_initial_value` performs a
  plain host `memcpy`. Under TMR the orchestrator runs on the AICPU where GM is
  directly addressable, so the same code is fine. Backtrace:
  `PTO2OrchestratorState::alloc_tensors+0x798` ← `aicpu_orchestration_entry`,
  faulting on a device VA. No HBG scene test uses the API, which is why it went
  unnoticed.
- **`get_tensor_data` on a task-produced tensor burns its full timeout.** The
  wait can never be satisfied in this runtime — the device does not execute until
  orchestration finishes — yet `wait_for_tensor_ready` spins the whole 15 s before
  failing. The condition is decidable at the call.

## Running

```bash
# standalone (2 dies; wrap in task-submit on a shared box)
python examples/a2a3/host_build_graph/deepseek_v4_flash_decode/\
test_deepseek_v4_flash_decode.py -p a2a3 -d <d0>,<d1> --manual only

# pytest
pytest examples/a2a3/host_build_graph/deepseek_v4_flash_decode \
    --platform a2a3 --device <d0>,<d1> --manual only
```

`manual` because the 367-kernel compile takes minutes; `skip_golden` because the
routing is stood in, not computed.

To exercise only the host side while the stall below is unresolved, set
`SIMPLER_SKIP_DEVICE_RUN=1`. `simpler_launch_run` then completes the run before
any execution claim — orchestration, graph recording, image relocation and the
SM H2D all ran during prepare, the kernel launch and its completion wait do not
happen — and `simpler_finalize_run` still releases the run's resources. The
check sits at the launch entry because the multi-chip subprocess drives a run
through the split `prepare/launch/wait/finalize` entry points and never calls
`simpler_run`. No outputs are produced, so a run under this variable is a timing
harness, not a test. The variable is temporary and goes away with the stall.

To read the stall diagnostics, raise the device log level — the per-task dump is
`LOG_INFO` and the default threshold does not open CANN's INFO stream:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1     # before Worker.init()
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/<run>/ascend"   # dir must pre-exist
```

## Provenance

Kernels, fixture and orchestration come from the TMR case; see its
[README](../../tensormap_and_ringbuffer/deepseek_v4_flash_decode/README.md) for
network shape, regeneration steps and cost. Two orchestration files are specific
to this case: `kernels/orchestration/decode_fwd_hostbuild.cpp` (the TMR
orchestration with the two edits in the table above — the host-orchestration
baseline the investigation was run against) and
`kernels/orchestration/decode_fwd_graph.cpp` (the same program recast as a
Graph, which the test points at). The Graph variant is derivable from the
hostbuild one: the 20-iteration decoder layer loop becomes one
`rt_submit_graph` per iteration with the layer's task set as the Graph body and
its per-layer views, scales and indices crossing the boundary positionally.
