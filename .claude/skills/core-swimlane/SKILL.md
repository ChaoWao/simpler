---
name: core-swimlane
description: Produce a Core swimlane (the L0/intra-core AICore pipeline) for one task — a single kernel or a mix — via the dump-driven `simpler_setup.tools.core_swimlane` tool. Use when the user asks to "run/produce a core swimlane", "trace a task's intra-core pipeline", profile why one AICore task is slow inside the core(s), or needs help choosing the tool's manual flags (`--func-id`, `--restore-arg`, `--set-arg`, `--spmd-block-num`, `--case`). The tool captures real per-task args from an args dump and auto-generates the `msprof op simulator` replay — no hand-authored workspace. For a hand-authored single-`kernel_entry` replay use [insight-trace](../insight-trace/SKILL.md); for cross-task / scheduler / dependency timing use the chip swimlane.
---

# Core Swimlane — Intra-core Pipeline Trace for a Task

`python -m simpler_setup.tools.core_swimlane` dumps a task's real `args[]`,
reconstructs them, generates a combined `msprof op simulator` replay of the
**whole task** (a mix runs AIC + AIV0 + AIV1 in one op), and exports an
Insight `trace.json` whose lanes are the cluster's pipes. Full reference:
[docs/dfx/core-swimlane-profiling.md](../../../docs/dfx/core-swimlane-profiling.md).
This skill is the **operating procedure** — above all the manual decisions:
whether a structured control tensor needs `--restore-arg`, and the
slot/value for any `--set-arg`.

## When to use

- **Use** when one task (single kernel or mix) is slow and you need the
  per-pipe (`MTE2` / `MTE1` / `CUBE` / `FIXP` / `SCALAR` / `VECTOR`)
  intra-core picture, or to confirm AIC↔AIV overlap inside a mix.
- **Not** for cross-task dependencies / scheduler / dispatch / finish
  timing — that is the **chip swimlane**. Core swimlane traces ONE task in isolation
  with no AICPU, so inter-task ordering is out of scope (doc §9, tier C).
- **vs `insight-trace`**: that skill hand-authors a wrapper around one
  `kernel_entry`; `core_swimlane` automates the whole thing from a real dump
  (real args, mix-together, SPMD context synthesised). Reach for
  `insight-trace` only when there is no test/dump to drive the capture.

## Run

```bash
source .venv/bin/activate
source "$ASCEND_HOME_PATH/set_env.sh"          # CANN env (msprof on PATH)
# Sim dump (no NPU); task-submit locks a device for the step-5 collect.
task-submit --device auto --max-time 1800 --run \
  "python -m simpler_setup.tools.core_swimlane --platform a2a3sim \
     --func-id <set> --test <test_file.py>"
```

Onboard `a2a3` instead of `a2a3sim`: run
`.claude/skills/onboard-arch-precheck/check.sh a2a3` first (the dump then
runs on the locked device). The five internal steps and all flags are in
doc §3.2 / §3.3.

**Payload-restoration scope:** `--restore-arg` is supported only with
`--platform a2a3sim` or `a2a3`. On `a5sim` / `a5`, tensor payload bytes from
args dump are not trustworthy because of
[#1560](https://github.com/hw-native-sys/simpler/issues/1560), so do not use
them as tensor truth or pass `--restore-arg`. Metadata-only and zero-filled
A5 Core swimlane replays remain usable. A5 still writes level-3 `args.bin` for tensors
marked with `CoreTaskArgs::dump(...)`,
but a Core swimlane trace restored from it is untrustworthy until #1560 is fixed;
this limitation does not make the whole Core swimlane tool a2a3-only.

## Choosing the manual flags (the hard part)

### `--func-id` — the task's member set

You wrote the orchestration, so the members are known. `--func-id 0` traces
the single-kernel task `{0}`; `--func-id 0,1,2` traces that 3-way mix.
It must equal a dispatched task's func_id **set** — for a same-AIV-on-both-
lanes SPMD mix the dump records a duplicate (`[0,1,1]`), so pass
`--func-id 0,1` (`set([0,1,1]) == {0,1}`). Wrong set → the tool lists the
func_id shapes present in the dump; pick one of those.

### `--restore-arg SLOT` — real structured control data

This is currently an **a2a3/a2a3sim-only** capability. Do not use A5 payload
for restoration while
[#1560](https://github.com/hw-native-sys/simpler/issues/1560) is open.

Use this repeatable flag when a tensor contains non-uniform runtime-built
state that changes control flow or addresses, such as a tiling struct.
Mark that tensor in orchestration with `CoreTaskArgs::dump(...)` before running
the tool. The fresh capture stays at `--dump-args 3` (hybrid): every argument appears
in JSON, while the existing Level-1 task mask writes only marked tensor bytes
to `args.bin`. `--restore-arg` consumes the requested `before_dispatch` bytes
and leaves all other replay tensors zero. A reused `--dump-json` must have its
sibling `args.bin` and the requested record payload.

Do not mark weights, activations, or KV pools merely for replay. A restored
slot cannot also use `--set-arg`.

### `--set-arg SLOT=VALUE` — only when a loop count must shrink

First classify where the kernel's loop trip count comes from:

| Trip count from | `--set-arg`? | Rule |
| --------------- | ------------ | ---- |
| **Tensor shape** (e.g. `shapes[0] / TILE_ELEMS`) | **No** | shape is the real dump value; changing it distorts. (mixed_example / single-kernel rows need no `--set-arg`.) |
| **A scalar arg** (e.g. `n_blocks`) | **Yes** — set the count directly | camodel would run the full loop; shrink to ≥ 3–4 (doc §7.2: floor 3, prefer 4). |
| **A control-tensor's content** (e.g. `context_lens`) | **Yes** — fill the buffer | the kernel *derives* the count from the data; fill so the derived count ≈ 4 (need `block_size` to back out the value). Integer dtypes only. |
| **A structured control tensor** (e.g. `FAInferTilingData`) | **No** — use `--restore-arg` on a2a3/a2a3sim | One uniform integer cannot reproduce a struct; restore its captured bytes on the supported platform family. |
| **The SPMD `block_num`** | **No** — use `--spmd-block-num` | block_num lives in the synthesised slot-48 context, which `--set-arg` cannot reach. |

Then find the slot — it is **per-kernel, never fixed**. Discover it:

1. Run once with `--no-collect`; step 3 prints the **arg-slot table**
   (every slot: index / kind / shape / scalar value).
2. Identify which slot is the loop bound by cross-referencing **any** of:
   the kernel's `args[N]` reads, the kernel-top **args-layout comment**
   (paged-attention kernels have one, e.g.
   `args[15] = total_logical_blocks scalar`), or the orchestration's
   `add_input` / `add_scalar` **order** (the i-th `add_*` is slot `i`).
3. Set the value per the table above.
4. Re-run, then **self-check** (below).

Verified examples (slots read from source):

| Test | Loop bound | Flag |
| ---- | ---------- | ---- |
| `paged_attention_unroll` | `aic_qk_matmul.cpp` `args[4] = n_blocks` (scalar) | `--set-arg 4=4` |
| `batch_paged_attention` | `context_lens` **tensor** (slot 1; 2nd `params_sf` `add_input`); the SF kernel (func 1) derives per-batch blocks from its content | `--set-arg 1=512` |

### `--spmd-block-num N` — SPMD grid width

`block_num` is written into the synthesised slot-48 `LocalContext`. Default
is 1, which models a single-block replay and does not take multi-block
branches such as `block_idx + 1 < block_num`. Keep that default only when the
kernel ignores `block_num`. For a kernel that branches or grid-strides on it,
read the selected task's real logical width from its orchestration and pass it
explicitly. That value may come from `rt_available_cluster_count()`, or it may
be task-specific. For example, the default lowest task-id in
`spmd_multiblock_aiv` uses `--spmd-block-num 4`, while the default lowest task
in `spmd_multiblock_mix` uses `--spmd-block-num 2`. When the real value is
unavailable, a value of at least 2 is only a bounded approximation for
exercising the next-block branch; record that limitation in the analysis.
`block_idx` is always synthesised to `0` (a representative block) and is
**not** a flag — it has no instruction-stream branches (doc §8).

### `--case NAME` — pin a small case on a multi-case test

When the test declares several `CASES[*]`, omitting `--case` auto-pins the
**first** case that lists your `--platform` (a deterministic single-case
dump). That first case is **not** guaranteed to be the smallest, and the
replay rebuilds every tensor at its **real dumped shape**: a production-size
case (long sequence, big batch, large KV cache) makes the camodel — a
cycle-accurate, whole-chip, serial simulator — **crawl or look hung** on the
oversized buffers. So **pin the small one yourself** with `--case <name>`
(accepts `ClassName::Case`) whenever the first-platform case is not the
smallest. Pick the case with the smallest shapes. `--set-arg` shrinks a
*loop count*; `--case` shrinks the *tensor shapes* — reach for `--case`
first when a replay stalls. Single-case tests need no `--case`.

Pick a case that is **scaled down, not reshaped** — same tile geometry
(M/K/N, head_dim, tile size), just fewer blocks / shorter sequence — so the
per-block pipeline stays identical to production (you lose only iteration
*count*, which does not change the pipeline shape). A case with different
tile shapes traces only itself, not production.

### Oversized CAModel checklist

On the supported a2a3/a2a3sim path, marking only the required tensors with
`CoreTaskArgs::dump(...)` limits the level-3 `args.bin`; the replay still
allocates and zero-initializes every tensor at its dumped shape. For a large
weight or KV tensor:

1. Run with `--dump-json <manifest> --no-collect` first and inspect the printed
   slot shapes.
2. Use `--case <bounded-case>` to reduce the actual shape-driving extents
   (layers, pages, cache rows, or sequence blocks). Lowering runtime `seq_len`
   alone does not help when shapes use fixed `MAX_SEQ` or layer constants.
3. Use `--set-arg` only when the remaining cost is a loop count; it does not
   resize tensors. On a2a3/a2a3sim, use `--restore-arg` only to preserve
   structured control state; it does not make the replay smaller. Do not use
   A5 payload restoration while #1560 is open.
4. Keep the default `--msprof-timeout 120` for ordinary cases. For a large,
   repetitive workload where a partial pipeline is sufficient, set a shorter
   timeout such as `--msprof-timeout 4`. This is the behavior recommended by
   the [Ascend CANN official QA/parameter guidance for msProf `--timeout`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/devaids/optool/atlasopdev_16_0082.html):
   msProf terminates simulation at the limit and parses the data produced so
   far. A partial trace can show the observed pipeline segment, but cannot
   establish whole-kernel duration or tail completeness. If the target kernel
   produced no profiler data before timeout, there may be no `trace.json`.

If no bounded case exists, make a focused workload-local case with the same
target task, func-id set, argument order, tiling path, and tile geometry but
smaller shape-driving extents. Keep a diagnostic-only case out of the PR unless
it is intended as permanent regression coverage. Full details are in doc §3.6.

## Self-check after every run

A known msprof/camodel export bug can truncate the last loop iteration(s).
Verify `MMAD == FIX_L0C_TO_DST == n_blocks` in the trace; if they disagree,
the tail was cut — do not draw timing conclusions, re-run or change the loop
count (doc §7.4). Read the auto-generated `*_trace_perfetto.json`, not the
raw Insight `trace.json`, for sub-laned per-instruction overlap (doc §3.4).

## Coverage

Representative command per task shape (single AIC / single AIV / 1+1 mix /
2-AIV mix / 3-way mix / SPMD single-source / SPMD coop mix / same-AIV-both-
lanes / paged-attn scalar & control-tensor loops) is in doc §3.8.
