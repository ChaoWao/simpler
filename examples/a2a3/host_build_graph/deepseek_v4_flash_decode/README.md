# DeepSeek-V4 FLASH decode on host_build_graph

The `tensormap_and_ringbuffer` DeepSeek-V4 case
(`examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode/`) run under
`host_build_graph`: same 43-layer network, same 368 kernels, same fixture, same
comm-window protocol. Only the runtime changes — HBG compiles the orchestration
with the host `g++`, runs it on the host CPU instead of the AICPU, and ships the
built shared-memory image to the device, which then boots scheduler-only.

The case exists to measure **host-side graph construction** and to prove the
device can execute what the host built. It is deliberately not a numerics test.

## What differs from the TMR case

`kernels/orchestration/decode_fwd_hostbuild.cpp` is the TMR orchestration with
one runtime-specific rewrite, kept as the non-Graph baseline.
`kernels/orchestration/decode_fwd_graph.cpp` — the file the test points at —
carries the same rewrite and additionally recasts the 20-iteration decoder
layer loop (40 of the 43 layers) as one `rt_submit_graph` per iteration: the
layer's task set becomes the Graph body (a free function reading its per-layer
views, scales and indices through `GraphTaskArgs`, positionally), and the host
records a 744-node Definition once instead of submitting the loop's ~15600
tasks individually. The runtime is untouched.

| Edit | Sites | Why |
| ---- | ----- | --- |
| `get_tensor_data(recv_count_out, …)` → `HBG_RECV_ROWS_PER_EXPERT` | 10 | HBG builds the whole graph before the device runs anything, so a read of a **task-produced** tensor has no value to return. The constant holds the per-expert tile loops at their real trip count (`ceil(16/16) == 1`, which is what the `h_i8 [512, 2048]` layout budgets per expert). |

The other **31** `get_tensor_data` reads are left alone: they read external
tensors (`ext_num_tokens_per_owner`, `hc_attn_scale_*`, `hc_ffn_scale_*`), which
the runtime stages with a host view and which therefore return real values.

The six former orchestration-side initializations are now identical under both
runtimes. Each `sh_gate_up_act_q*` producer clears its own two padded
`h_tile_i8` rows, while a dedicated AIV seed task clears `mixes_raw` before the
split-K `hc_head_linear` AtomicAdds. The host therefore never writes a GM-heap
device address.

Everything else — submit order, dependencies, scope nesting, and the
orchestration-side `valid_rows = min(n_rows - t0, 16)` — is byte-identical to
the TMR source inside the Graph body. The graph keeps the size and shape of the
real one (the 15971 device-side tasks; 1131 host-submitted with the Graph
collapse) but not the fixture's routing, hence `skip_golden`.

## Status: the non-Graph tail completes; Graph replay remains blocked earlier

With the Graph form the host records the 744-node Definition and boots with
**1131 tasks on host** (down from 15991 in the submit-everything form) — this is
measured, on both ranks. The device-side replay of a Definition that large is
**not yet exercised**: an unskipped Graph-form run still fails earlier, in Graph
activation (`sched_error_code=5 INVALID_ARGS` from the scheduler's graph
queues), before reaching the tail.

The **non-Graph baseline** (hostbuild orchestration, no skip) now completes the
full two-rank device body after bounding `hc_head_linear` to the rows present in
`x_flat`. Both ranks reported `outcome=0` and the case reported `PASSED` in
`task_20260817_204500_135435017977` and
`task_20260817_210320_91023923204`, with no MTE exception, error 507018,
`TIMEOUT_EXIT`, or task-15959 stall.

Those artifacts came from a predecessor that reaped the close-path child on a
10 s budget, so their wrappers exited 1 only after PASS when `worker.close`
overran it. `_CLOSE_CHILD_REAP_TIMEOUT_S` is 60 s here, but no final combined
launcher exit 0 is claimed without a rerun. The `skip_golden` result
establishes liveness, not numerical correctness.

### Original non-Graph failure

Before the fix, the device executed **15959 of 15971 tasks** and stopped making
progress:

```text
TASK ring=0 task_id=15959 state=RUNNING fanin_met=2/2 kernels=[aic:355 aiv0:-1 aiv1:-1]
     running_on=[cores=[core=0(aic) core=1(aic) core=3(aic) … 13 AIC cores]]
SUMMARY completed=15959/15971 scan_ready=0 scan_waiting=11 scan_running=1
CLUSTER cluster_id=0 aic=core0(busy kernel=355 task=15959 cond_reg_state=ack) …
```

Task 15959 is `hc_head_linear` (kernel 355) — an AIC-only SPMD matmul in the
head section, `block_num=16`, submitted with no predicate and no
`require_sync_start`. The thirteen AIC entries shown in the TASK line are only
the prefix that fits in its fixed 192-byte `running_on` buffer; the per-cluster
records show all 16 executors acknowledged without FIN. CANN physical `blk` IDs
are not the child kernel's logical `block_idx`. CANN subsequently reported MTE
out-of-range faults inside kernel 355, so the failure was not a 13-of-16
dispatch deadlock.

### What has been ruled out

Each row was tested by changing exactly that one thing and re-running the full
network on hardware. The failure reproduced at the identical task and
`completed` value every time.

| Ruled out | How |
| --------- | --- |
| pto-isa version | Stalls identically on `83d01313`, `0cefc9a5` and `f51c92f6`. (The *earlier* stall at task 2307 — issue #1839 — is a genuine ISA regression and **is** fixed by `f51c92f6`; it is a different stall.) |
| The orchestration edits above | An earlier variant that instead used dispatch predicates and a static tile grid stalls at the same task. So does one with the predicate rewrite removed. |
| `set_initial_value` | Present and absent both fail identically, so it was not the MTE cause. The calls have since been replaced with device-side initialization kernels. |
| Runtime modifications | A `host_tensor_fill` seam was written to support `set_initial_value` on HBG and later reverted; the stall is identical with it, without it, and on a pristine tree. |
| A different TMR kernel binary | The HBG and TMR `hc_head_linear` binaries are byte-identical. TMR completing this kernel was an allocation-layout mask, not evidence that its memory accesses were valid. |
| Kernel-internal spin | In the failing binary all loop bounds are compile-time constants and `t_dim`/`t_linear` are unused in the body. `ptoas_auto_sync_tail(kBarrierAll)` expands to `pipe_barrier(PIPE_ALL)` — intra-core, not cross-core. There is no cross-block synchronization to deadlock on. |
| Malformed submitted descriptors or base pointers | A host-side probe prints `t_dim=8 t_linear=16 block_num=16`, `x_flat [8,16384] @0x12c0c0742000`, `hc_head_fn [4,16384] @0x12c0c0702000`, `mixes_raw [16,16] @0x12cd75153400 size=1024`. These descriptors are self-consistent; the bug was the kernel's later hard-coded 16-row view of the 8-row `x_flat`. |
| Slow-not-hung | Raising `SIMPLER_SCHEDULER_TIMEOUT_MS` from 10 s to 60 s (with `SIMPLER_OP_EXECUTE_TIMEOUT_US`/`SIMPLER_STREAM_SYNC_TIMEOUT_MS` raised to keep the required ordering) leaves the stall unchanged; the device log confirms the window grew from exactly 10 s to exactly 60 s. |
| Early-dispatch gating | A gated dispatch parks a core at its doorbell, which would look exactly like this. It cannot fire: `force_gate=true` is passed only from `stage_consumer_blocks`, whose only source `early_dispatch_queues` has no producer outside its own drain (Milestone 1 leaves early-dispatch stubbed). |
| Pointer relocation | No `cannot relocate` / `outside both SM and arena windows` diagnostic; the SM H2D reports no error. |
| Resource exhaustion | The ring and heap allocators report a fatal on exhaustion. None fired: `orch_error_code=0`, orchestration completed, the graph uploaded, 15959 tasks ran. |

### Root cause and fix

`x_flat [8,16384]` is a valid 512 KiB allocation, but both TLOAD views were
hard-coded as `Shape<...,16,256>` with a 16384-element row stride. They covered
an almost 1 MiB address range and read rows 8–15 out of bounds. HBG's exact-size
per-tensor allocation exposed the over-read as MTE OOR; TMR's retained bump
allocation kept it inside a larger mapping and masked the same kernel bug.

The fix derives

```text
row_base = (block_idx / 16) * 16
valid_rows = clamp(t_dim - row_base, 0, 16)
```

The two `x_flat` views and TLOAD tiles, dependent matmul/accumulator tiles, and
the `mixes_raw` AtomicAdd store now all use `valid_rows`. It is eight for this
invocation, so no instruction addresses a non-existent input row.

## Runtime gap this case exposed

This gap is independent of the `hc_head_linear` MTE fault:

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

`manual` because the 368-kernel compile takes minutes; `skip_golden` because the
routing is stood in, not computed.

To exercise only the host side without launching the device body, set
`SIMPLER_SKIP_DEVICE_RUN=1`. `simpler_launch_run` then completes the run before
any execution claim — orchestration, graph recording, image relocation and the
SM H2D all ran during prepare, the kernel launch and its completion wait do not
happen — and `simpler_finalize_run` still releases the run's resources. The
check sits at the launch entry because the multi-chip subprocess drives a run
through the split `prepare/launch/wait/finalize` entry points and never calls
`simpler_run`. No outputs are produced, so a run under this variable is a timing
harness, not a test.

To collect scheduler diagnostics for a regression, raise the device log level —
the per-task dump is `LOG_INFO` and the default threshold does not open CANN's
INFO stream:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1     # before Worker.init()
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/<run>/ascend"   # dir must pre-exist
```

## Provenance

Kernels, fixture and orchestration come from the TMR case; see its
[README](../../tensormap_and_ringbuffer/deepseek_v4_flash_decode/README.md) for
network shape, regeneration steps and cost. Two orchestration files are specific
to this case: `kernels/orchestration/decode_fwd_hostbuild.cpp` (the TMR
orchestration with the rewrite in the table above — the host-orchestration
baseline the investigation was run against) and
`kernels/orchestration/decode_fwd_graph.cpp` (the same program recast as a
Graph, which the test points at). The Graph variant is derivable from the
hostbuild one: the 20-iteration decoder layer loop becomes one
`rt_submit_graph` per iteration with the layer's task set as the Graph body and
its per-layer views, scales and indices crossing the boundary positionally.
