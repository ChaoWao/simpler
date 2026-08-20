# Handoff — l2-orchestrator-standalone

Context summary for resuming work after a session reset. For how to *use* the
package, read `README.md` instead; this file records **why it is the way it is**
and what is not done.

## What this package is

A host-only bench + profile harness for **one sequence** — the four lines that
build an L2 task graph, at
`simpler-main/src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp:538-541`:

```c
rt_scope_begin(rt);
entry_points->entry(orch_l2);
rt_scope_end(rt);
rt_orchestration_done(rt);
```

Engine under test: **L2 `PTO2OrchestratorState`** (a2a3 / `host_build_graph`).
Not the L3 `Orchestrator` — that is the sibling `../l3-orchestrator-standalone`,
a different engine at a different level.

## How it came about (the question chain that produced it)

1. "What are l3-orchestrator-standalone's inputs/outputs?" → its input is C++ API
   calls, not a file; its output is DAG/scheduling behaviour, not numbers.
2. "How does it use simpler's `runtime_maker`?" → **it does not**, zero
   references. `runtime_maker` is L2 glue driving `PTO2OrchestratorState`.
3. "Is that four-line snippet L2 or L3?" → **L2**. `host_build_graph` runs the
   *same L2 orchestrator* on the host; execution site changed, level did not.
4. "Expand those four lines fully" → 1 stack push + T×(6-step submit) + A×alloc
   + 1 pop + 1 release store.
5. "Build a package to bench/profile them, like the L3 one" → this package.
6. "Run it with the repo-root `qwen3_dynamic_tensormap.h` case" → the
   `qwen3-dyn` payload.
7. "Wasn't that case 3096 tasks?" → yes, **at tier 0**. The build now pins tier
   0; tier 4 (522 tasks) was the case's own default.
8. "Only tier 0 of qwen3-dyn is useful" → the synth and qwen3 payloads, the
   other four tiers and the sweep mode were removed. The package builds one
   workload.

## Non-obvious decisions, and why

**Why it can run on a host at all.** `host_build_graph` already runs the L2
orchestrator on the host into a host SM mirror, then H2Ds the image
(`run_host_orchestration`, runtime_maker.cpp:487-499). This package stops before
the H2D. Three substitutions, all in `bench/l2_harness.h`: GM heap → host
`aligned_alloc` (orchestrator only does address arithmetic on it, never
dereferences); host SM → same thing the real path already uses;
`entry_points->bind` → direct `framework_bind_runtime` (same function, dlopen
only exists to cross the .so boundary).

**The extraction boundary was free.** `build_config.py`'s `host` target already
compiles exactly `runtime/orchestrator_core` + `runtime/shared` +
`orchestration`. Dropping `host/` (= runtime_maker + CANN) leaves 8 TUs that
compiled on the first try. Only 9 symbols were missing, all diagnostics →
`src/host_shim/host_shim.cpp`.

**`pto_orchestration_api.h` and `pto_runtime2.h` cannot share a TU.** Both define
`PTO2Runtime` / `PTO2RuntimeOps` (by design: .so side vs runtime side). The
driver is the runtime side (like runtime_maker); the payloads are the .so side.
Do not "fix" this by including both.

**The Level-2 profiling seam is the ops table.** `rt->ops` is a plain
`const PTO2RuntimeOps *` and the orchestration API reaches the engine
*exclusively* through it. Copy the table, wrap each entry with a clock read,
repoint `rt->ops`. Intercepts 100% of entry→engine traffic, needs no engine
cooperation, cannot miss a call site. Strictly better than L3's `set_test_hook`.

**Two Level-3 traps, both handled.** The engine's `g_orch_*_cycle` counters are
process-global accumulators that **reset on read**, so they are snapshotted right
after the clean run (reading at report time folds both runs and doubles
everything). And the cycle→time divisor is `cntfrq_el0` (100 MHz here), **not**
`PLATFORM_PROF_SYS_CNT_FREQ` (50 MHz, the device counter) — using the latter made
STEP 5 report 73 ms inside a 21 ms run.

## The `qwen3-dyn` payload — the subtle parts

The repo-root case is compiled **unmodified and in place** (not copied, not
translated). `scripts/check_extraction.sh` asserts no in-package duplicate exists.

**It is compiled as C, and must be.** Its shapes are C99 compound literals
(`(uint32_t[]){90, 5120}`), which are lvalues in C but temporaries in C++. g++
rejects the decay outright. Verified across `-std=c++17 / gnu++17 / gnu++11 /
gnu++03`, with and without `-fpermissive` — **no flag accepts it**, and there is
no clang on this box. Hence `bench/payload_qwen3_dyn_case.c` + the C ABI in
`bench/esl_shim/esl_c_abi.h`.

**`Tensor` on the C side is an opaque 128-byte blob.** The case never reads a
single `Tensor` member (verified by grep), and `ChipTensor` is exactly 128 bytes /
64-byte aligned / standard-layout / trivially copyable — all four
`static_assert`ed in `esl_shim_impl.cpp`. A blob assumes only size+alignment, so
unlike a field-by-field mirror it cannot silently rot. Note `Tensor` is *already*
a distinct struct in `src/common/task_interface/buffer.h`, so aliasing that name
in C++ is a hard error — that is why the C++ side uses `EslTensor` throughout.

**The tag mapping, read from esl_proxy's source** (the `esl_proxy` checkout):
the `_ro` variants call only `add_tensor_addr()` and push **nothing** onto the
pending list `tm_submit` looks up/inserts. So `_ro` = "pass to kernel, create no
dependency", not a narrower access grant:

| esl_proxy | L2 tag |
| --- | --- |
| `tm_in` | `INPUT` |
| `tm_out` | `OUTPUT_EXISTING` (**not** `OUTPUT` — see below) |
| `tm_inout` | `INOUT` |
| `tm_*_ro` | `NO_DEP` |

`tm_out` → `OUTPUT_EXISTING` is load-bearing: only `OUTPUT_EXISTING`/`INOUT` get
registered in the TensorMap, and a runtime-created `OUTPUT` is explicitly
skipped. Mapping it to `OUTPUT` would allocate a second buffer, register no
producer, and **silently erase every RaW edge in the case**.

**This disagrees with the L3 package.** `bench/qwen3_l3_replay.h` maps
`tm_in_ro -> INPUT` and `tm_out_ro -> OUTPUT_EXISTING`, giving `_ro` args real
edges. Against esl_proxy's source that is wrong and inflates the L3 replay's edge
count. 2874 of this case's tensor args are `_ro` at tier 0. **The L3 package has
not been corrected.**

**`NO_DEP` still does creator retention** (L2 Step A), which esl_proxy has no
equivalent of. Every `_ro` arg here is an external or a view of one, and
externals have no creator — so it contributes nothing. `assert_no_creator()`
enforces this at every `_ro` call rather than assuming it.

**`QWEN3_SPMD_TIER` changes the DAG, not just scheduling.** The subtask total is
tier-invariant (3096); the task count is not. Default here is **0**.

| tier | kernel submits | subtasks | engine tasks |
| --- | --- | --- | --- |
| **0** (default) | **3096** | 3096 | **3875** |
| 1 / 2 / 3 | 1602 / 864 / 678 | 3096 | 2381 / 1643 / 1457 |
| 4 (case's own default) | 522 | 3096 | 1301 |

The tier is exported from the case's own macro
(`payload_qwen3_dyn_case.c: const int qwen3_dyn_spmd_tier = QWEN3_SPMD_TIER;`) so
the report cannot drift from the binary.

## Cross-validation (the strongest evidence the shim is faithful)

The unmodified case through this shim vs the L3 package's **independently
hand-translated** replay, at every tier:

| tier | L2 kernel submits | L3 `task_cnt` |
| --- | --- | --- |
| 0 | 3096 | 3096 |
| 1 | 1602 | 1602 |
| 2 | 864 | 864 |
| 3 | 678 | 678 |
| 4 | 522 | 522 |

At tier 0 five numbers match exactly: 3096 submits, 779 allocs, 3096 subtasks,
3875 engine tasks, and the `DUR_*` sum 110168.700 us == L3's
`timeline.busy_ns = 110168700`. Tier 4 agrees too (522 / 779 / 3096 / 12572220).

## Findings worth remembering

**CORRECTION — an earlier reading in this file and in conversation was wrong.**
I originally reported "dependency inference is not the cost; writing the
descriptor is (STEP 5 = 66-86%)". That was a **measurement artifact of this
harness**, not a property of the engine. STEP 5's cold cost is ~96% **first touch
of the shared memory**. Anyone quoting the cold per-STEP percentages as engine
cost is quoting paging. `--prefault-sm` / `--prefault-arena` / `--prefault-all`
exist to separate the three effects; Level 4 (`-DL2_TENSORMAP_PROFILING=1`)
explains what is left.

Steady-state (prefaulted) per-STEP shares, and they differ **by payload**:

| step | `qwen3-dyn` tier 0 | `qwen3` (40 layers) |
| --- | --- | --- |
| STEP 1 prepare_task (ring slot + heap) | 11.0% | **43.2%** |
| STEP 3 TensorMap lookup | **74.4%** | 24.7% |
| STEP 5 payload/descriptor write | 9.9% | 26.5% |
| STEP 4 register outputs | 4.1% | 1.7% |
| cold → warm total | 9.75 → 2.97 ms | 19.68 → 1.71 ms |

- **The two payloads have different bottlenecks, and Level 4 says why.**
  `qwen3-dyn` writes many SPMD sub-views of shared buffers; the TensorMap hashes
  on `buffer.addr` **alone** (`pto_tensormap.h:522`), so they collapse into one
  bucket — avg chain 13.2, **MAX 120** (q_proj's 20 chunks × 6 tiles), **67.5% of
  overlap checks find nothing**. `qwen3` allocates whole buffers, so its **MAX
  chain is 1 and 0% is wasted**; its cost is the allocator instead.
- **STEP 3 is memory-latency bound, not compute bound.** 2.201 ms / 88784 walked
  entries = **24.8 ns per entry** ≈ a cache miss. `check_overlap` already
  fast-rejects in O(1) on a byte range (`pto_tensormap.h:240-249`), so the cost is
  chasing `next_in_bucket` through an entry pool allocated in submit order.
- **Chain length scales with SPMD width — a superlinear term.** tier 0 → 4:
  lookups fall 5.5× but entries walked fall **27×**.
- **The per-run SM allocation is a real production cost, not just a harness one.**
  `runtime_maker.cpp:487` does `new uint8_t[sm_size]` **every run** and memsets
  only the header. At the production default `PTO2_TASK_WINDOW_SIZE = 16384` that
  is a fresh **~77.6 MB**, so large-block new/free hands pages back to the OS and
  the next run faults them in again. ~6.2 ms per run at this graph size.
  `sizeof(PTO2TaskPayload) = 4864 B`, of which `tensors[32] × 128 B` = 4096 B.
- **Submit cost tracks tensor-arg count, not graph size.** Established with
  synth sweeps that held task count constant while moving one knob (`depth`
  8→128 and `width` 1→32 flat at ~2.0 us/task; `tensors` 1→24 rising 1.81→2.39
  us then saturating; `scalars` 1→16 flat). That payload is gone, so the claim
  is no longer reproducible here — what remains is the profile's SUBMIT LATENCY
  BY TENSOR-ARG COUNT table, which shows the same effect confounded with which
  TensorMap bucket each arg set hashes into.
- **The four scope/done calls are free.** `entry` is ~99.97% of the block;
  `on_scope_end` is literally `{}` in `host_build_graph`
  (`scheduler/pto_scheduler.h:866`).

## Optimization candidates (measured, none implemented)

Ranked by headroom the data supports. This package measures; it does not patch
the engine. Checked `simpler-main/docs/investigations/` first — the existing
entries cover scheduler dispatch and host worker dispatch, **none touches the
orchestrator submit path**, so none of these was previously rejected.

1. **Dense per-buffer entry scan — ≈1.5 ms, 51% of `qwen3-dyn`'s steady-state.**
   Not "cheaper compares" (the L1 reject is already O(1)) but less pointer
   chasing: a compact parallel array of `(start_offset, extent, version)` scanned
   linearly before touching any full entry, or keep each buffer's chain **sorted
   by `start_offset`** and stop once `entry.start >= in_end`. Sorted insertion is
   O(chain) but there are 4002 inserts against 88784 walks. Upper bound: walking
   only the 28260 real overlaps costs 0.70 ms instead of 2.20 ms.
2. **Pool the host SM mirror across runs — ≈6.2 ms per run.** Reusing a dirty
   buffer is safe: init-on-write is already the engine's invariant. The device
   SM and arena are already pooled (`acquire_pooled_gm_sm`); the host mirror is
   not, which reads as an omission rather than a decision.
3. **Shrink `PTO2TaskPayload`.** 4096 of 4864 B is `tensors[32]`, but this case
   peaks at 12 args and `qwen3` at 17. Cuts SM footprint (hence #2's residual)
   and slot-to-slot locality. It is a host↔device wire struct so it must stay POD
   and contiguous — variable-length slots with offset indices are what codestyle
   rule 8 endorses, and the same rule warns against worst-case fixed arrays. ABI
   change; measure the true max across all examples first.
4. **Raise the SPMD tier** — workload knob, zero code change, 11.6 → 3.3 ms cold.
   Changes device-side SPMD granularity and load balance, so it is a trade.

**Bound on candidate 1:** nothing is reclaimed here, so chain length only grows.
Production wraps the ring and reclaims entries at the watermark, so steady-state
chains should be shorter and 1.5 ms is an **upper bound at this graph size**.
Measuring the steady state needs the reclaim paths this harness never enters.

## The caveat that bounds every number

**Nothing is ever reclaimed.** No scheduler, no completions → `last_task_alive`
never advances → the ring's watermark reclaim never fires. Therefore:

- `--task-window` must exceed the payload's **total** task count and `--heap-mb`
  its **total** allocation. Not sized per steady state.
- Every repetition rebuilds the whole runtime; reusing one would measure
  back-pressure.
- **The reclaim paths are never exercised** — `sync_tensormap`'s eviction,
  `ensure_tensormap_capacity`'s back-pressure spin, the 500 ms deadlock backstop.
  This bench measures the fast path only.

A payload that exhausts either resource latches a fatal, after which every submit
is a silent no-op. The driver checks `orch_error_code` after every run and refuses
to print numbers when it is set — without that check, exhaustion looks like a
small, very fast graph.

## Verification status (all re-run at tier 0)

| check | result |
| --- | --- |
| clean build, default flags | 0 warnings, 0 errors |
| `-Wall -Wextra` | warnings only in the repo's engine code; 0 in hand-written code |
| ASAN + UBSAN, both modes | **clean** (LeakSanitizer cannot run in this container — needs ptrace) |
| `ctest` | 4/4 pass (incl. a prefault test asserting the graph size is unchanged) |
| `check_extraction.sh` | 3/3 pass; **negative-tested** — an edited engine file is reported and TU-list drift exits 1 |
| `ldd` | no CANN (libc / libstdc++ / libm only) |

The tier sweep and the multi-payload ASAN matrix were verified before those
payloads and tiers were removed; only tier 0 of qwen3-dyn is buildable now.

## Not done / open

1. **Not committed.** `l2-orchestrator-standalone/` is untracked (`??`). 91 files
   would be added, no build artifacts (verified after the `.gitignore` fix below).
2. **`.gitignore` gap found and fixed** in this session: `build/` and `**/build/`
   do not match `build-prof/`, which `scripts/profile_l2.sh` creates and whose
   `l2_bench` is an extension-less executable no other rule caught. Added
   `build-*/` and `**/build-*/`. **This fix is also untracked.**
3. **No back-pressure test.** The only real coverage gap: deliberately undersize
   `--task-window` and assert it fails with the expected `orch_error_code` rather
   than crashing. Currently such a run is a bench *failure*, not a test case. This
   is also what blocks measuring the reclaim paths, and therefore what bounds the
   optimization estimates above.
6. **ASAN/UBSAN and `-Wall -Wextra` are not wired into any script or CI.** They
   were run by hand this session (both clean). Reproduce with:
   `cmake -B /tmp/san -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g" -DCMAKE_C_FLAGS="-fsanitize=address,undefined -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"`
   then run with `ASAN_OPTIONS=detect_leaks=0`.
4. **The L3 package's `_ro` tag mapping is still wrong** (see above) and its edge
   counts are still inflated. Not corrected — it would change
   `docs/qwen3-l3-equivalence-report.md` and the `reports/sim-*.json` evidence.
5. **`tensormap_and_ringbuffer` not covered.** The repo-root case's *native*
   runtime is TRB (device AICPU), whose four lines are at
   `a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:707-709` and
   `:810`. Harder to extract: that orchestrator runs on-device, its
   `on_scope_end` does real work (`release_producer_scope` per task), and there
   is no existing host-side run path.
