# L2 Orchestrator standalone package

**In-repo, one payload.** This package lives inside the simpler checkout and
compiles the engine **straight out of `../src`** — there is no copy of the
runtime here, so an edit to `../src` is measured by the next build with nothing
to keep in sync. The only engine-side file the package owns is
`src/host_shim/host_shim.cpp`, which defines the 9 symbols the AICPU binary
would otherwise provide.

It measures exactly one workload: `qwen3_dynamic_tensormap.h` at
`QWEN3_SPMD_TIER=0`, which lives in this directory. Results for the engine as of
`perf/hbg-orch` @ 72c3163d are in
`reports/perf-report-perf-hbg-orch-qwen3-dyn.md`.

Bench and profile for **one sequence** — the four lines that build an L2 task
graph, at `src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp:538-541`:

```c
rt_scope_begin(rt);
entry_points->entry(orch_l2);
rt_scope_end(rt);
rt_orchestration_done(rt);
```

The engine under measurement is **L2 `PTO2OrchestratorState`** (a2a3 /
`host_build_graph`). This is *not* the L3 `Orchestrator` — that one is in
`../l3-orchestrator-standalone`. The two are different engines at different
levels, not two views of one thing.

No CANN, no Ascend SDK, no NPU, no `.so` to dlopen.

## Why this can run on a host at all

`host_build_graph` already runs the L2 orchestrator on the host: it builds the
graph into a **host SM mirror** and only then H2Ds the populated image to the
device, which boots scheduler-only (`run_host_orchestration`, runtime_maker.cpp:487-499).
This package stops before the H2D. Three substitutions, all documented in
`bench/l2_harness.h`:

| simpler | here | why it is sound |
| --- | --- | --- |
| device GM heap | host `aligned_alloc` | the orchestrator only does address arithmetic on it — the AICore would dereference it, and there is no AICore here |
| host SM mirror | same, a plain buffer | not a substitution: the host-orch path already uses one |
| `entry_points->bind` + dlsym'd entry | direct call, linked in | same `framework_bind_runtime` function (orchestration/common.cpp:42); dlopen only exists to cross the .so boundary |

**What is measured is graph CONSTRUCTION.** Nothing executes. There is no
scheduler, no completion, no H2D, no pointer relocation.

## Build and run

```bash
cmake -B build -S .
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure     # 4 tests
./build/l2_orch_main                            # smoke: the four lines, 3875 tasks
```

If a binary dies with ``version `GLIBCXX_3.4.30' not found``, an older
`libstdc++.so.6` is ahead of the system one on the loader path:

```bash
export LD_LIBRARY_PATH="$(dirname "$(readlink -f "$(g++ -print-file-name=libstdc++.so)")"):$LD_LIBRARY_PATH"
```

Everything at once:

```bash
./scripts/profile_l2.sh          # throughput + per-step profile + cold/warm attribution
./scripts/check_extraction.sh    # assert the harness still matches the repo runtime
```

## The payload

`qwen3_dynamic_tensormap.h` — the esl_proxy case, compiled **unmodified**, via
the C-ABI shim in `bench/esl_shim/`. 3096 kernel submits + 779 framework allocs
= 3875 engine tasks.

```bash
./build/l2_bench --mode=throughput --repeat=5
./build-prof/l2_bench --mode=profile
```

### `QWEN3_SPMD_TIER` is pinned to 0, and it changes the DAG

The case guards its tier with `#ifndef`, so the build sets it — `CMakeLists.txt`
compiles the case with `-DQWEN3_SPMD_TIER=0` and offers no knob for the rest.

`qwen3_blocks_per_task()` returns `min(total_chunks, {1,2,4,8,1<<30}[tier])`, i.e.
the tier decides how many SPMD chunks fold into **one** task. The subtask total
is tier-invariant; the task count is not — tier 0 gives 3096 kernel submits,
tier 4 (the case's own default) gives 522, both for the same 3096 subtasks.

Tier 0 is pinned because one chunk per task is the most task-dense variant, so
it puts the most orchestration pressure on the engine per unit of device work.
**A task count quoted without its tier is meaningless** — the report prints
`QWEN3_SPMD_TIER=N` on the accounting line for that reason, read from the macro
as the case itself saw it (`payload_qwen3_dyn_case.c` exports it) so the report
cannot drift from the binary, and `check_extraction.sh` asserts the pin.

This is the natural home for that case: it is an **L2** case (`tm_in` / `tm_out`
/ `tm_submit`, `aicpu_orchestration_entry`), and L2's TensorMap does the
**view-overlap detection** the case's SPMD sub-view writes are built around
(`pto_tensormap.h:23`). The sibling L3 package had to disclaim exactly this — L3
keys on whole-buffer identity with no byte-range refinement, so its edge set is
coarser by construction. No such disclaimer is needed here.

**The case file is not translated.** It lives in this directory and is
`#include`d as it stands; `check_extraction.sh` asserts exactly one copy exists,
so no stale duplicate can be picked up instead. Three build-level
accommodations, none of which touch the file:

1. **It is compiled as C**, because it is C. Its shapes are C99 compound
   literals — `tensor_from_base_layout(orch_args + 0, (uint32_t[]){90, 5120}, 2, BFLOAT16)`
   — which are lvalues in C but temporaries in C++, and g++ rejects the decay
   ("taking address of temporary array"). No dialect or permissiveness flag
   accepts it: checked across `-std=c++17`, `gnu++17`, `gnu++11`, `gnu++03`,
   with and without `-fpermissive`.
2. `-Daicpu_orchestration_entry=qwen3_dyn_orchestration_entry`, so the case's
   C-linkage entry cannot collide with the engine-side name of the same symbol.
3. `Tensor` on the C side is an **opaque 128-byte / 64-byte-aligned blob**, which
   the C++ shim reinterprets as `ChipTensor`. The case never reads a single
   `Tensor` member (verified by grep), and `ChipTensor` is exactly 128 bytes,
   64-byte aligned, standard-layout and trivially copyable — all four
   `static_assert`ed in `esl_shim_impl.cpp`. A blob assumes only size and
   alignment, so unlike a field-by-field mirror it cannot silently rot.

### The tag mapping, and where the L3 package got it wrong

Read from esl_proxy's source, not inferred. In its `tensormap.h` the `_ro`
variants push **nothing** onto the pending list that `tm_submit` later looks up
and inserts:

```c
tm_in_ptr:       add_tensor_addr(); tm_pending_push(t, TM_PEND_IN)
tm_in_ro_ptr:    add_tensor_addr()                          /* no push */
tm_out_ro_ptr:   add_tensor_addr()                          /* no push */
```

So `_ro` means "hand this tensor to the kernel, create **no** dependency" — it is
not a narrower access grant. Hence:

| esl_proxy | dependency role | L2 tag |
| --- | --- | --- |
| `tm_in` | TensorMap lookup (RaW) | `INPUT` |
| `tm_out` | TensorMap insert | `OUTPUT_EXISTING` |
| `tm_inout` | lookup + insert | `INOUT` |
| `tm_*_ro` | none | `NO_DEP` |

L2's `NO_DEP` exists for precisely this case ("skips OverlapMap lookup, depends
on creator only"). Two details that are easy to get wrong and are handled:

- `tm_out` must map to `OUTPUT_EXISTING`, **not** `OUTPUT`. The case allocates
  its buffers up front, and only `OUTPUT_EXISTING`/`INOUT` get registered in the
  TensorMap — a runtime-created `OUTPUT` is explicitly skipped. Mapping `tm_out`
  to `OUTPUT` would allocate a second buffer and register no producer, silently
  erasing every RaW edge in the case.
- `NO_DEP` still performs L2's Step-A creator retention, which esl_proxy has no
  equivalent of. Every `_ro` arg in this case is an entry external or a view of
  one, and externals have no creator, so Step A contributes nothing — and that
  invariant is **asserted at every `_ro` call**, not assumed.

This **disagrees with the sibling L3 package's table** in
`bench/qwen3_l3_replay.h`, which maps `tm_in_ro -> INPUT` and
`tm_out_ro -> OUTPUT_EXISTING`, i.e. gives the `_ro` args real edges. Against
esl_proxy's source that is wrong, and it inflates the L3 replay's edge count.
804 of this case's 2,766 tensor args are `_ro`.

### Measured at tier 0, and cross-validated against the L3 replay

```
payload=qwen3-dyn  tasks=3875
four-line block   median 11.340 ms     per task 2.93 us      341,702 tasks/s
  QWEN3_SPMD_TIER=0
  kernel submits    3096    framework allocs    779    (engine tasks = 3875)
  SPMD subtasks     3096    scalar args        7326
  dep-tracked args  9162    _ro (NO_DEP) args  2874
  sum of the case's DUR_*: 110168.700 us  (virtual AICore time, NOT measured)
```

**Five** numbers match the L3 package's independently hand-translated replay of
the same case at the same tier, exactly — `reports/lat-t0.json` there reports
`task_cnt=3096`, `alloc_cnt=779`, `subtask_cnt=3096`, `task+alloc=3875`, and
`timeline.busy_ns=110168700`. Two unrelated paths (a hand translation to L3, and
the unmodified case through this shim) agreeing to the digit is the strongest
available evidence that the shim reproduces the case's structure rather than
approximating it. The agreement holds at tier 4 as well (522 / 779 / 3096 /
12572220).

`tasks=3875` is `3096 + 779` because on L2 an `alloc_tensors` **is** a real task —
it claims a ring slot, cuts the GM heap and registers the buffer — whereas in
esl_proxy it is a pool-tail bump. That difference is visible in the profile
rather than hidden.

### Where the time actually goes — separate allocation cost from compute first

**A raw Level-3 reading is misleading, and this is the most important thing on
this page.** With a cold SM, STEP 5 appears to be 66% of submit cost. It is not:
96% of that is the **first touch of the shared memory**, not the engine writing
descriptors. `--prefault-sm` / `--prefault-arena` isolate the three effects:

| step | cold (raw) | +prefault SM | +prefault arena | +both |
| --- | --- | --- | --- | --- |
| STEP 1 prepare_task | 0.501 ms | 0.334 | 0.469 | **0.333** |
| STEP 3 infer deps: TensorMap lookup | 2.552 ms | 2.231 | 2.558 | **2.201** |
| STEP 4 register outputs | 0.296 ms | 0.310 | **0.122** | **0.122** |
| STEP 5 payload/descriptor GM write | **6.461 ms** | **0.275** | 6.300 | **0.296** |

Read the rows, not the totals:

- **STEP 5** collapses 24× on `--prefault-sm` and does not move on
  `--prefault-arena`. So it is SM first-touch. The engine's actual descriptor +
  payload write is **0.28 ms**, not 6.5 ms.
- **STEP 4** halves on `--prefault-arena` only — the TensorMap buckets and entry
  pool live in the arena.
- **STEP 3** barely moves (−14%). It is **real CPU work**, and once the
  allocation effects are removed it is **74%** of submit cost.

So the steady-state breakdown at tier 0 is:

| step | time | share |
| --- | --- | --- |
| **STEP 3 TensorMap lookup** | **2.201 ms** | **74%** |
| STEP 1 prepare_task | 0.333 ms | 11% |
| STEP 5 payload/descriptor write | 0.296 ms | 10% |
| STEP 4 register outputs | 0.122 ms | 4% |
| STEP 2 / 6 | 0.017 ms | 1% |

**Do not quote the cold numbers as engine cost.** They are dominated by a
per-run allocation that production shares (see below), which is a different
problem with a different fix.

### Why STEP 3 costs what it does — Level 4

`-DL2_TENSORMAP_PROFILING=1` turns on the engine's own lookup counters:

```
lookups                      6612
bucket entries walked       88784
avg chain length            13.43
MAX chain length              120
overlap checks              86970
overlap hits                28260   (32.5% of checks)
wasted walk                             67.5%
```

Root cause is at `pto_tensormap.h:522` — the map **hashes on `buffer.addr`
alone**, so every SPMD sub-view of one buffer shares a bucket and each lookup
walks the whole chain. `MAX chain = 120` is exactly q_proj's 20 chunks × 6 tiles.

The arithmetic says where the cost sits:

```
2.201 ms / 6612 lookups  = 333 ns per lookup
2.201 ms / 88784 walked  = 24.8 ns per chain entry
```

24.8 ns is about a cache miss. `check_overlap` already fast-rejects in O(1) on a
byte-range test (`pto_tensormap.h:240-249`), so the cost is **not** the
comparison — it is chasing `next_in_bucket` through an entry pool that is
allocated in submit order and therefore scattered.

The tier makes this visible as a superlinear term:

| | tier 0 | tier 4 | ratio |
| --- | --- | --- | --- |
| lookups | 6612 | 1194 | 5.5× |
| **entries walked** | **88784** | **3215** | **27×** |
| MAX chain | 120 | 17 | 7× |

Entries walked grows 27× while lookups grow 5.5×: chain length itself scales
with SPMD width.

### End-to-end effect of the two levers

| | cold | prefaulted |
| --- | --- | --- |
| tier 0 | 11.606 ms | 5.030 ms |
| tier 4 | 3.259 ms | 1.129 ms |

Together: 11.6 ms → 1.13 ms, a **90%** reduction — but the two levers are not
equivalent. Prefaulting is a *diagnostic*; the production fix is to pool the
buffer (below). The tier is a *workload* knob that changes device-side SPMD
granularity, so it is a trade, not a free win.

### Optimization candidates, with measured headroom

Ranked by what the numbers above support. None of these is implemented here —
this package measures, it does not patch the engine.

1. **Make the per-buffer entry scan dense (≈1.5 ms, 51% of steady-state cost).**
   Not "make each compare cheaper" — the L1 reject is already O(1). Reduce the
   pointer chasing: either keep a compact parallel array of each entry's
   `(start_offset, extent, version)` so the scan is a linear sweep before any
   full-entry touch, or keep each buffer's chain **sorted by `start_offset`** and
   stop once `entry.start >= in_end`. Sorted insertion is O(chain) but there are
   4002 inserts against 88784 walks. Upper bound: walking only the 28260 real
   overlaps costs 0.70 ms instead of 2.20 ms.

2. **Pool the host SM mirror across runs (≈6.2 ms per run).**
   `runtime_maker.cpp:487` does `new uint8_t[sm_size]` **every run** and memsets
   only the header segment. At the production default
   `PTO2_TASK_WINDOW_SIZE = 16384` that is a fresh **~77.6 MB** allocation, so
   large-block `new`/`free` returns the pages to the OS and the next run faults
   them in again. Safe to reuse a dirty buffer: init-on-write is already the
   engine's invariant. The device-side SM and arena are already pooled
   (`acquire_pooled_gm_sm`); the host mirror is not, which reads as an omission.

3. **Shrink `PTO2TaskPayload` (4864 B/slot, of which `tensors[32] × 128 B` = 4096 B).**
   This case peaks at 12 tensor args (the `qwen3` payload at 17), yet every slot
   reserves 32. Costs SM footprint — hence #2's residual — and slot-to-slot
   locality (a submit writes ~512 B but consecutive slots' hot regions are 4864 B
   apart). It is a host↔device wire struct, so it must stay POD and contiguous —
   but variable-length slots with offset indices are exactly what the project's
   own codestyle rule 8 endorses, and the same rule warns against sizing fixed
   arrays to a worst case. ABI change; measure the real maximum across all
   examples first.

4. **Raise the SPMD tier (11.6 → 3.3 ms cold, zero code change).**
   A workload knob, not an engine fix: folding chunks cuts the number of
   TensorMap lookups but changes device-side SPMD granularity and load balance.

**Caveat that bounds candidate 1's estimate:** nothing is reclaimed here, so
chain length only ever grows. Production wraps the ring and reclaims entries at
the watermark, so steady-state chains should be shorter and 1.5 ms is an **upper
bound at this graph size**, not a production figure. Measuring the steady state
requires the reclaim paths, which this harness never enters.

## Modes

```bash
./build/l2_bench --mode=throughput --repeat=5
./build-prof/l2_bench --mode=profile
# cold/warm attribution — needed before reading any Level-3 percentage
./build-tm/l2_bench --mode=profile --prefault-all
./build/l2_bench --help
```

- `throughput` — N clean repetitions, wall clock only. **Quote these.**
- `profile` — one clean run then one instrumented run, so the instrument's own
  cost is a reported number rather than folded in silently. **Pass
  `--prefault-all` before reading any per-STEP percentage** (see below).

## The four profile levels

No engine source is modified. Each level has a different seam:

**Level 1 — the four lines.** Driver-bracketed `steady_clock`. Free of
instrument cost.

**Level 2 — every `entry` → engine call.** `rt->ops` is a plain
`const PTO2RuntimeOps *` and the orchestration API reaches the engine
*exclusively* through it (`pto_orchestration_api.h:175, 125, 245, 254, 262` —
every one is `rt->ops->…`). So the profiler copies that table, wraps each entry
with a clock read, and repoints `rt->ops` at the copy. This intercepts **100%**
of the traffic, needs no cooperation from the engine, and cannot miss a call
site — a strictly better seam than L3's `set_test_hook`.

**Level 3 — the engine's own per-STEP counters.** `submit_task_common` already
laps `CYCLE_COUNT_LAP` into `g_orch_*_cycle` at each of its six STEP boundaries,
and `orchestrator_get_profiling()` returns them. Compiled out unless
`SIMPLER_ORCH_PROFILING=1`:

```bash
cmake -B build-prof -S . -DL2_ORCH_PROFILING=1 && cmake --build build-prof --parallel 8
```

Two traps this package handles, both of which produce plausible-looking garbage
if missed:

- The counters are **process-global accumulators that reset on read**, so they
  are snapshotted right after the clean run — reading at report time would fold
  both runs together and double every number.
- The cycle→time divisor is **`cntfrq_el0` (100 MHz here), not
  `PLATFORM_PROF_SYS_CNT_FREQ` (50 MHz)**. The latter describes the device
  counter, not the host clock the shim reads, and using it makes every Level-3
  duration 2× too large.

**Level 4 — the engine's own TensorMap lookup counters.** Bucket chain length
(avg and max), overlap checks vs hits, insert count. This is what turns
"STEP 3 is 74%" into a root cause. Implies Level 3:

```bash
cmake -B build-tm -S . -DL2_TENSORMAP_PROFILING=1 && cmake --build build-tm --parallel 8
```

**Cold/warm attribution — `--prefault-sm` / `--prefault-arena` / `--prefault-all`.**
Diagnostics, not optimizations: they pre-touch the SM and/or the runtime arena so
first-touch cost lands outside the measured window instead of being attributed to
whichever step happened to touch the page. Without them, STEP 5 reads as 66-86%
of submit cost and the real bottleneck is invisible. Semantically safe — every
byte the engine reads from either region is written by the init phases first.

## Measured, this host (Kunpeng-920 aarch64, `-O3`, taskset 0-3)

3875 engine tasks, `--repeat=20`, median:

| condition | four-line block | per task | throughput |
| --- | --- | --- | --- |
| cold | 11.41 ms | 2.94 us | 340 k tasks/s |
| `--prefault-all` | 4.71 – 4.90 ms | 1.22 – 1.26 us | 790 – 820 k tasks/s |

Level 1 shows where the block's time is: `entry` **99.95%**,
`rt_orchestration_done` 0.03%, `rt_scope_begin` and `rt_scope_end` together
under 0.02%. The two scope calls are O(1) stack pushes and, in
`host_build_graph`, `on_scope_end` is literally `{}`
(`scheduler/pto_scheduler.h:866`) — so the outer scope pair costs nothing and
all cost is inside `entry`.

Level 3 — **cold vs prefaulted**, because the raw reading is dominated by SM
first-touch and says almost nothing about the engine:

| step | cold | prefaulted | prefaulted share | first-touch share of cold |
| --- | --- | --- | --- | --- |
| STEP 5 payload/descriptor GM write | 6.523 ms | 0.294 ms | 9.9% | 95% |
| **STEP 3 infer deps: TensorMap lookup** | 2.554 ms | **2.213 ms** | **74.7%** | 13% |
| STEP 1 prepare_task (slot + heap) | 0.506 ms | 0.314 ms | 10.6% | 38% |
| STEP 4 register outputs | 0.299 ms | 0.120 ms | 4.0% | 60% |
| STEP 2 sync_tensormap | 0.030 ms | 0.012 ms | 0.4% | 60% |
| STEP 6 publish fanin_count | 0.014 ms | 0.008 ms | 0.3% | 43% |

**Cold and warm answer different questions.** Cold, STEP 5 looks like the
bottleneck at 66% of submit cost — but 95% of that is the kernel faulting in SM
pages, a cost paid once per runtime in production rather than per orchestration.
Prefaulted, the engine bottleneck is **STEP 3, the TensorMap dependency lookup,
at 75%**. Level 4 explains why: this case writes many SPMD sub-views of shared
buffers, the map hashes on `buffer.addr` alone, and they collapse into a handful
of buckets — max chain 120, 67.5% of overlap checks find nothing.

That is specific to workloads shaped like this one. A workload that allocates
whole buffers instead keeps its max chain at 1 and pays in the ring/heap
allocator (STEP 1) rather than in the lookup, so a fix aimed at the chain would
do nothing for it.

## Caveat that bounds every number here

**Nothing is ever reclaimed.** There is no scheduler and no completion, so
`last_task_alive` never advances and the ring's watermark reclaim never fires.
Consequences:

- `--task-window` must exceed the payload's **total** task count, and
  `--heap-mb` its **total** allocation. The defaults (8192 slots, 2048 MiB) hold
  this graph with headroom.
- Every repetition tears down and rebuilds the whole runtime. Reusing one would
  measure back-pressure, not steady-state submit.
- The reclaim paths (`sync_tensormap`'s eviction, `ensure_tensormap_capacity`'s
  back-pressure spin, the 500 ms deadlock backstop) are **never exercised**.
  This bench measures the fast path only.

A payload that exhausts either resource latches a fatal, after which every
submit is a silent no-op — which would otherwise look like a small, very fast
graph. The driver checks `orch_error_code` after every run and refuses to print
numbers when it is set.

## What is in the box

| path | role |
| --- | --- |
| `../src/a2a3/runtime/host_build_graph/` | the L2 engine — 8 TUs, compiled in place, not copied |
| `src/host_shim/host_shim.cpp` | the only hand-written engine-side file: 9 symbols (5 log sinks, 4 scope-stats stubs) plus `get_sys_cnt_aicpu` |
| `qwen3_dynamic_tensormap.h` | the case, compiled unmodified as C |
| `bench/l2_harness.h` | host-only runtime assembly |
| `bench/l2_profile.{h,cpp}` | the 4-level profiler and the ops-table interceptor |
| `bench/payload_qwen3_dyn*.{cpp,c}` | the payload: entry adapter plus the case's TU |
| `bench/esl_shim/` | the esl_proxy C ABI, implemented on the L2 orchestration API |
| `apps/l2_orch_main.cpp` | smoke driver: the four lines, nothing else |
| `scripts/` | `profile_l2.sh`, `check_extraction.sh`, `cold_warm_table.py` |

The TU list is the `host` target of
`src/a2a3/runtime/host_build_graph/build_config.py` minus `host/` itself, which
is where `runtime_maker` and its CANN dependencies live. `check_extraction.sh`
asserts that correspondence.

## Confirm there is no CANN

```bash
ldd build/l2_bench | grep -Ei 'ascend|hcom|runtime|acl'   # must print nothing
```
