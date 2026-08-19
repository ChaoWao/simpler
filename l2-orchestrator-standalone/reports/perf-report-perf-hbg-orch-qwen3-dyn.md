# L2 orchestration performance — branch `perf/hbg-orch`, payload `qwen3-dyn`

Measured 2026-08-19 with the pristine l2-orchestrator-standalone harness
(commit f06646d of the extraction, before any of its later optimization work),
against this branch's engine sources compiled **in place from `../src`** at
`perf/hbg-orch` @ 72c3163d. The four lines under test
are `../src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp:538-541`
(`host/` is the one directory this package does not compile, because that is
where runtime_maker's CANN dependencies live):

```c
rt_scope_begin(rt);
entry_points->entry(orch_l2);
rt_scope_end(rt);
rt_orchestration_done(rt);
```

driven by `qwen3_dynamic_tensormap.h` (QWEN3_SPMD_TIER=0), compiled unmodified
as C via the esl_shim.

Engine `file:line` citations below are into `../src` at that revision; bench
paths are relative to this package. Line numbers move when the runtime is
edited — re-run the grep next to each claim rather than trusting the number
after a change.

## Environment

| | |
|---|---|
| CPU | Kunpeng-920, 192 cores, aarch64 |
| Kernel | 5.10.0-60.18.0.50.r865_35.hce2 |
| Compiler | g++ 11.4.0, `-O3` (CMake Release) |
| Pinning | `taskset -c 0-3` (harness is single-threaded) |
| Bench args | `--task-window=8192 --heap-mb=2048 --repeat=20`, median quoted |

Graph size cross-check (must match the L3 package's independent translation,
and does): 3096 kernel submits + 779 framework allocs = 3875 engine tasks,
3096 SPMD subtasks, 9162 dep-tracked args, 7326 scalars. ctest 4/4, including
the prefault graph-size assertion.

## Where the code is

Every number in this report comes from one of these places.

### The measured path

| what | file | symbol / line |
|---|---|---|
| the four lines | `../src/…/host_build_graph/host/runtime_maker.cpp` | 538-541 |
| the API the entry calls | `../src/…/host_build_graph/orchestration/pto_orchestration_api.h` | `rt_submit_task` 177-180, `rt_alloc_tensors` 127-130 |
| **the submit body — where STEPs 1-6 live** | `../src/…/runtime/orchestrator_core/pto_orchestrator.cpp` | `submit_task_common` 1112 |
| the alloc body | same | `PTO2OrchestratorState::alloc_tensors` 1896 |
| slot + heap reservation (STEP 1) | same | `prepare_task` 808 |
| TensorMap lookup / insert (STEP 3/4) | `../src/…/runtime/pto_tensormap.h` | `lookup` 521, `link_entry` 647 |
| the hash (STEP 3's bucket choice) | same | `hash` 639 |
| descriptor + payload GM write (STEP 5) | `pto_orchestrator.cpp` | 1250-1284 |

### Where the counters come from

Nothing in the engine was modified to produce this report; each level reads an
existing seam.

| level | seam | file |
|---|---|---|
| 1 — the four lines | driver brackets each call with `steady_clock` | `bench/l2_bench.cpp:162-176` |
| 2 — every entry→engine call | copies `rt->ops` and repoints it; the orchestration API reaches the engine *only* through that table | `bench/l2_profile.h:177` (`OpsInterceptor`) |
| 3 — per-STEP cycles | the engine's own `CYCLE_COUNT_LAP` accumulators, compiled in by `SIMPLER_ORCH_PROFILING=1` | macros at `pto_orchestrator.cpp:160-170`; laps at 1150 / 1165 / 1235 / 1248 / 1284 / 1298 |
| 4 — TensorMap chain stats | the engine's own `g_lookup_*` counters, compiled in by `SIMPLER_TENSORMAP_PROFILING=1` | `pto_tensormap.h:87-91`, incremented inside `lookup` |

The six Level-3 lap counters map to the STEP names one-for-one, in source
order: `g_orch_alloc_cycle` (STEP 1, line 1150), `g_orch_sync_cycle` (STEP 2,
1165), `g_orch_lookup_cycle` (STEP 3, 1235), `g_orch_insert_cycle` (STEP 4,
1248), `g_orch_args_cycle` (STEP 5, 1284), `g_orch_fanin_cycle` (STEP 6, 1298).
Levels 3 and 4 both **reset on read**, so `bench/l2_bench.cpp:242-245` snapshots
them immediately after the clean run — before the instrumented run can
contaminate them.

### What the harness substitutes

Three seams, all in `bench/l2_harness.h`; none of them is on the measured path
in a way that changes the engine's work:

| simpler | here | line |
|---|---|---|
| device GM heap | host `aligned_alloc` at the device's 1024 B granularity | 149 |
| dlopen + dlsym'd entry | direct call through the same `framework_bind_runtime` | 192 |
| SM mirror / arena first touch | optional `memset` before init, for the cold/warm split | 144, 154 |

The payload's own side: `bench/payload_qwen3_dyn.cpp` adapts the entry
signature, `bench/payload_qwen3_dyn_case.c` compiles the case, and
`bench/esl_shim/esl_shim_impl.cpp:22-24` holds the esl_proxy→L2 tag mapping
(`tm_in`→`INPUT`, `tm_out`→`OUTPUT_EXISTING`, `tm_inout`→`INOUT`,
`tm_*_ro`→`NO_DEP`) that decides which args create edges at all.

## Headline throughput

| condition | four-line block (median) | per task | throughput |
|---|---|---|---|
| cold (fresh runtime, pages untouched) | 11.41 ms | 2.94 µs | ~340 k tasks/s |
| `--prefault-all` (first-touch excluded) | 4.71–4.90 ms | 1.22–1.26 µs | ~790–820 k tasks/s |

Roughly 58% of the cold block is first-touch paging of the SM mirror and the
TensorMap arena, not engine work — see the cold/warm attribution below.

Measured before the package was cut down to this one payload, the `qwen3`
example ran 1.94 µs/task and the synthetic DAG 2.07 µs/task on the same build.
This case is the slowest per task of the three, and Level 4 explains why.

## Level 1 — the four lines themselves

| step | ns | % of block |
|---|---:|---:|
| rt_scope_begin | 810 | 0.01% |
| entry | 11 788 550 | 99.96% |
| rt_scope_end | 160 | 0.00% |
| rt_orchestration_done | 3 620 | 0.03% |

Everything is `entry()`; the scope bracket and the done-flag write are noise.

## Level 2 — every entry→engine call (intercepted ops table, cold run)

| call | count | total ms | p50 ns | p99 ns | % of entry |
|---|---:|---:|---:|---:|---:|
| alloc_tensors | 779 | 1.41 | 1 560 | 4 480 | 13.8% |
| submit_task | 3096 | 8.77 | 2 550 | 8 150 | 86.2% |

Submit latency by tensor-arg count (p50), from the same run:

| tensor args | submits | p50 ns | p99 ns | ns per tensor |
|---:|---:|---:|---:|---:|
| 3 | 1320 | 2 780 | 8 310 | 927 |
| 4 | 1440 | 1 730 | 7 820 | 433 |
| 5 | 240 | 4 290 | 11 810 | 858 |
| 6 | 6 | 1 970 | 3 800 | 328 |
| 12 | 90 | 4 850 | 7 290 | 404 |

**Arg count does not predict submit cost here** — 4 args is faster than 3, and
6 args is faster than 5. Whatever dominates is a property of *which* buffers
those args view, not how many there are, which is what Level 4 pins down. (The
per-arg slope is real and was isolated on the synthetic payload before it was
removed; in this workload it is confounded with chain length and cannot be read
off this table.)

## Level 3 — inside `submit_task_common` (engine's own cycle counters)

Cold vs `--prefault-all`, per-STEP totals over the whole graph:

| step | cold ms | warm ms | warm % | first-touch share of cold |
|---|---:|---:|---:|---:|
| 1 prepare_task (slot + heap alloc) | 0.506 | 0.314 | 10.6% | 38% |
| 2 sync_tensormap | 0.030 | 0.012 | 0.4% | 60% |
| 3 infer deps: TensorMap lookup | 2.554 | 2.213 | **74.7%** | 13% |
| 4 register outputs: TensorMap insert | 0.299 | 0.120 | 4.0% | 60% |
| 5 payload/descriptor GM write | 6.523 | 0.294 | 9.9% | **95%** |
| 6 publish fanin_count | 0.014 | 0.008 | 0.3% | 43% |
| TOTAL | 9.926 | 2.961 | | |

Read cold and warm as two different questions. Cold, STEP 5 looks like the
bottleneck (66%), but 95% of it is the kernel faulting in SM pages — a cost
that in production is paid once per runtime, not per orchestration. Once
prefaulted, the real engine bottleneck is **STEP 3, the TensorMap dependency
lookup, at 75% of submit cost**. No alloc/fanin waits were ever entered
(nothing is reclaimed in this harness).

Why STEP 5's cold number is that large is arithmetic, not mystery.
`sizeof(PTO2TaskPayload)` is **4864 B** (`pto_runtime2_types.h:265`), of which
`tensors[32] × sizeof(ChipTensor) = 32 × 128 = 4096 B`. At `--task-window=8192`
the payload ring alone is **39.8 MB** of fresh mapping, and STEP 5 is the first
thing to touch each slot (`pto_orchestrator.cpp:1250`, a deliberate batched
write — the comment there says it is deferred out of the allocation phase to
avoid scattered GM writes). So STEP 5 is charged for the page faults of every
slot it is merely the first writer of. At the production default
`PTO2_TASK_WINDOW_SIZE = 16384` (`pto_runtime2_types.h:74`) that doubles.

## Level 4 — why STEP 3 dominates

| counter | value |
|---|---:|
| lookups | 6 612 |
| inserts | 4 002 |
| bucket entries walked | 88 590 |
| avg chain length | 13.4 |
| max chain length | **120** |
| overlap checks | 86 970 |
| overlap hits | 28 260 (32.5%) |
| wasted walk (no overlap found) | **67.5%** |

**The long chains are the design working as specified, not a hash that got
unlucky.** `pto_tensormap.h:30-34` states the constraint outright:

> CRITICAL: Hash only by base_ptr. For overlap detection to work, ALL
> sub-regions of the same base tensor MUST be in the SAME hash bucket.

`hash()` (`pto_tensormap.h:639`) is a golden-ratio multiplicative hash over
`tensor.buffer.addr`, and a sub-view keeps its parent's `buffer.addr` while
differing in `start_offset` — so every sub-view of one buffer lands in one
bucket *by construction*, which is exactly what lets `lookup`
(`pto_tensormap.h:521`) find byte-range overlaps at all. The walk then does
`if (tensor.buffer.addr == cur_entry->buffer_addr)` before calling
`check_overlap`, so the chain cost is the price of that guarantee.

Two counters pin down which of the two possible causes it is:

- **It is not table pressure.** `PTO2_TENSORMAP_NUM_BUCKETS` is **4096**
  (`pto_runtime2_types.h:87`) and this case makes only **4002 inserts** — about
  one entry per bucket if addresses were distinct. Observed average chain:
  **13.4**. More buckets would change nothing.
- **It is same-address sub-views.** 86 970 of the 88 590 entries walked
  (**98.2%**) passed the `buffer_addr` equality test, i.e. nearly every entry on
  a walked chain is another view of the very buffer being probed. Only 1.8% of
  the walk is unrelated buffers colliding.

So the 67.5% wasted walk is `check_overlap` returning `NO_OVERLAP` for
sub-views of the same buffer that do not intersect this probe's byte range.
Attacking it means giving the bucket a secondary order on `start_offset` — an
interval structure, or simply keeping the chain sorted so the scan can stop
early — not resizing the table. This cost is specific to workloads shaped like
this one: a payload that allocates whole buffers keeps its max chain at 1 and
pays in the ring/heap allocator (STEP 1) instead.

## Branch vs simpler-main baseline

The original f06646d package, whose engine TUs were byte-copies of
simpler-main, was rebuilt and run interleaved with this one on the same
machine, same flags:

| engine | prefaulted median (5 interleaved rounds) |
|---|---|
| simpler-main copies (f06646d) | 4.64 – 4.81 ms |
| `perf/hbg-orch` (this repo) | 4.64 – 4.82 ms (one 5.64 ms outlier) |

**No measurable difference.** The branch's engine diffs (the
`CoreTaskArgs` → `GraphTaskArgs` graph-recording rework in
`pto_orchestrator.cpp` / `pto_orchestration_api.h`) are on the graph
record/replay path, which this payload never enters, and the per-STEP and
Level-4 counters are identical to the baseline within noise
(STEP 3 warm: 2.197 ms baseline vs 2.187 ms branch; chain stats identical).

## Comparing an optimization against this baseline

To measure an edit to the runtime under `../src`:

```bash
scripts/check_extraction.sh          # assert the harness still matches the runtime
scripts/profile_l2.sh --repeat 20    # rebuilds all three variants, rewrites reports/
```

Since the engine is compiled in place, an edit to `../src` is measured by the
next build — but it is also a change to the repo's runtime, so run the repo's
own tests before believing a number, not just this package's four.

Compare against this file, and quote the **prefaulted, pinned** number — the
cold one is dominated by first-touch paging and moves with the machine's page
state rather than with your change. An engine edit that changes the graph is a
correctness failure first: `ctest` asserts 3096 submits / 779 allocs, and a
latched `orch_error_code` turns further submits into silent no-ops, which would
otherwise read as a small, very fast graph.

## Caveats (unchanged from the harness's own README)

- Nothing is ever reclaimed: no scheduler, no completions, so ring watermark
  reclaim, tensormap eviction and back-pressure never fire. Numbers are the
  pure submit path.
- Quote throughput from the clean build only; the Level 3/4 build adds ~4–5%
  instrument cost and is used only for the shape of the breakdown.
- Full machine-readable data: `profile-qwen3-dyn.{txt,json}` in this directory,
  regenerated by `scripts/profile_l2.sh --repeat 20`.
