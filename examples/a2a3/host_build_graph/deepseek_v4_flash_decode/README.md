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

`kernels/orchestration/decode_fwd_graph.cpp` — the file the test points at — is
the TMR orchestration with one runtime-specific rewrite, additionally recasting
the 20-iteration decoder layer loop (40 of the 43 layers) as one
`rt_submit_graph` per iteration: the
layer's task set becomes the Graph body (a free function reading its per-layer
views, scales and indices through `GraphTaskArgs`, positionally), and the host
records a 744-node Definition once instead of submitting the loop's ~15600
tasks individually. The runtime is untouched.

| Edit | Sites | Why |
| ---- | ----- | --- |
| `get_tensor_data(recv_count_out, …)` → `HBG_RECV_ROWS_PER_EXPERT` | 10 | HBG builds the whole graph before the device runs anything, so a read of a **task-produced** tensor has no value to return. The constant holds the per-expert tile loops at their real trip count (`ceil(16/16) == 1`, which is what the `h_i8 [512, 2048]` layout budgets per expert). |

The only other `get_tensor_data` read left is `ext_num_tokens_per_owner`: its
value feeds tile counts and launch block numbers — orchestration control flow
that must run where the graph is built. The 30 former `hc_attn_scale_*` /
`hc_ffn_scale_*` reads moved data, not control flow, and are gone from both
runtimes: each `split_pre_post*` / `comb_sinkhorn*` kernel now takes the scale
view as an extra tensor input and reads its elements from GM itself.

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

## Status: the host records the Definition; device replay is blocked in activation

With the Graph form the host records the 744-node Definition and boots with
**1131 tasks on host** (down from 15991 when every task is submitted
individually) — this is measured, on both ranks. The device-side replay of a
Definition that large is **not yet exercised**: an unskipped run fails in Graph
activation (`sched_error_code=5 INVALID_ARGS` from the scheduler's graph
queues), so the recorded bodies never replay.

`skip_golden` therefore establishes host-side construction, not numerical
correctness.

### Why `hc_head_linear` carries a row-tail bound

`x_flat [8,16384]` is a valid 512 KiB allocation, but the kernel's two TLOAD
views were hard-coded as `Shape<...,16,256>` with a 16384-element row stride.
They covered an almost 1 MiB address range and read rows 8-15 out of bounds.
HBG allocates each tensor at its exact size, which surfaces the over-read as an
MTE out-of-range fault; TMR's retained bump allocation keeps it inside a larger
mapping and masks the same kernel bug, so the kernel completes there.

The kernel derives

```text
row_base = (block_idx / 16) * 16
valid_rows = clamp(t_dim - row_base, 0, 16)
```

and uses it for the two `x_flat` views and TLOAD tiles, the dependent
matmul/accumulator tiles, and the `mixes_raw` AtomicAdd store. It is eight for
this invocation, so no instruction addresses a non-existent input row. With that
bound in place the full two-rank device body ran to completion under a
submit-every-task form of this orchestration, both ranks `outcome=0`
(`task_20260817_204500_135435017977`, `task_20260817_210320_91023923204`).

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
network shape, regeneration steps and cost. One orchestration file is specific
to this case: `kernels/orchestration/decode_fwd_graph.cpp`, the TMR
orchestration carrying the rewrite in the table above and recast as a Graph —
the 20-iteration decoder layer loop becomes one `rt_submit_graph` per iteration
with the layer's task set as the Graph body and its per-layer views, scales and
indices crossing the boundary positionally.
