# 2026-08 — hbg: upload Graph Definitions once as shared device objects

## Question

Breaking down the host side of `examples/a2a3/host_build_graph/qwen3_14b_decode`
(40 Graph submissions of one 277-node, 130,192-byte Definition) showed two
dominant costs outside pure orchestration:

- the orchestration window rebuilt and zero-filled a 132,752-byte submission
  image per layer — 98.1% of those bytes a byte-identical Definition copy
  (KNOWN_ISSUES at the time);
- the upload stage shipped all 40 images: 5.31 MB across 40
  `device_malloc` + `rtMemcpy` round trips, a stable 14.4–15.9 ms per run.

Both trace to one design choice: the Definition travels *inside* every
submission image.

## Change

`254f924e` (measured by `4d434174`/`b8095e39`, enabled by `f868ac52`):

- each distinct Definition uploads **once** as a
  `[GraphDefinitionHeader][Definition image]` object retained by
  `acquire_graph_definition_buffer`, keyed by content identity;
- `GraphSubmission` carries `definition_addr` + `definition_hash` instead of
  the inline image; a submission is now 2,568 bytes;
- device localize validates the shared object through a one-time verify gate
  (first localizer FNV-hashes, peers spin on the state word) and binds
  topology against the shared image in place — the per-occurrence embedded
  Definition copy in execution storage is gone.

## Result (qwen3, 5 serial runs each, median)

| Stage | Before | After | Δ |
| ----- | ------ | ----- | - |
| orch image build (incl. zero-fill) | 931 µs (826) | 23.6 µs (10.0) | −97.5% |
| submission bytes | 5,310,080 | 232,944 | −95.6% |
| orch window total | 1,834 µs | 836 µs | −54% |
| **H2D upload time** | **14.66 ms** | **12.10 ms** | **−17%** |

> These runs were taken with `SIMPLER_SKIP_DEVICE_RUN=1`, which stops a run
> after prepare so the host-side stages above are measured without the device
> execution behind them. The numbers agree with a `--rounds 3` reproduction
> (see the amendment below), which measures the same stage without that knob.

## Why the H2D time gain is far below the byte gain

Bytes fell 95.6% but the upload *time* fell only 17%: the stage's cost was
never bandwidth-dominated. The effective rate is absurd on both sides —
0.36 GB/s before, 0.02 GB/s after — which is the signature of fixed
per-call costs dominating data movement. The optimized stage still makes 41
allocation-and-copy pairs: one 130,192-byte shared Definition object and 40
2,568-byte reference submissions.

Each upload pays:

1. **`rtMalloc` per object** — `MemoryAllocator::alloc` calls CANN
   `rtMalloc(RT_MEMORY_HBM)` (a driver round trip) plus a mutex-guarded map
   insert; the one Definition object and 40 submissions make 41 allocations
   per run, freed again at teardown.
2. **`rtMemcpy` (sync, `RT_MEMCPY_HOST_TO_DEVICE`) per object** — each
   call is a blocking submit-and-wait on the copy stream: host builds the
   descriptor, pushes to the driver, and blocks for completion. Each 2.5 KB
   reference submission never occupies the link long enough for bandwidth to
   matter.

So the model is `time ≈ N × (malloc + memcpy latency) + bytes / BW`, and at
these sizes the first term dominates by two orders of magnitude. The byte
reduction could only remove the (already small) second term.

**The model above is right; the per-call figure this entry originally derived
from it was not.** See the amendment.

## Amendment 2026-08-18 — the residual is one-time cost, not per-call latency

The original text divided the post-change 12.10 ms by the 41 alloc-and-copy
pairs to get **≈ 295 µs per call**, and proposed batching the 40 reference
submissions with an expected result of "well under 1 ms". A `--rounds 3` run of
the same case on the same machine separates one-time from steady-state cost and
shows the division was over the wrong numerator.

| `graph_upload` | round 1 (cold) | round 3 (steady) | one-time |
| -------------- | -------------- | ---------------- | -------- |
| before (`f4ed1045`) | 14.756 ms | 0.916 ms | 13.840 ms |
| after (`9e32a99b`) | 12.877 ms | **0.685 ms** | 12.192 ms |

Cold agrees with the table above (14.66 / 12.10 ms), so the measurements
match; only the attribution differs.

**What the residual actually is.** 12.192 ms of the post-change 12.877 ms is a
*one-time* cost: the first-touch `rtMalloc` + `aclrtMemset` of the 40
execution-storage blocks (~1.33 MB each, ~53 MB total) inside
`acquire_graph_execution_buffer`. Those 40 allocations are not among the 41
alloc-and-copy pairs, so dividing 12.10 ms by 41 charges them to calls that did
not make them. Steady state — where the retained blocks are reused and only the
41 pairs remain — is **0.685 ms**, i.e. **≈ 17 µs per pair**, not 295 µs.

**Where this change's own cold-start gain came from.** The round-split table's
own cold delta is **−1.879 ms** (14.756 → 12.877); the −2.56 ms in the five-run
median table is the same gain on a different run set, so the decomposition below
is of the −1.879 ms and both its rows come from that one table. Splitting the
delta by round separates two effects of similar size, which is why a
single-round experiment cannot attribute it:

| source | Δ | share |
| ------ | - | ----- |
| execution storage shrinking by 130,192 B per block — the one-time column, 13.840 → 12.192 ms (side effect of removing the embedded Definition: 40 × less to `rtMalloc` and `aclrtMemset`) | −1.648 ms | **88%** |
| upload bytes 5.31 MB → 0.233 MB — the steady-state column, 0.916 → 0.685 ms (the effect the change targeted) | −0.231 ms | 12% |

"5.21 MB less allocated-and-zeroed" and "5.08 MB less transferred" are the same
magnitude, so they are indistinguishable in a cold-only measurement.

**Consequence for the proposed follow-up.** Batching the 40 reference
submissions attacks `N` in the model, which is real but bounded by the
steady-state figure: 41 pairs × ~17 µs ≈ 0.645 ms of per-call cost, plus
~0.04 ms of actual data movement. Collapsing 41 pairs to 2 recovers **at most
~0.6 ms** — not ~11 ms — and the stage is already under 1 ms once the one-time
cost is excluded. Worth doing, but it is not where the 12 ms lives.

**Where the 12 ms does live.** The one-time 12.19 ms is the 53 MB of
execution-storage allocation and zeroing. Its lifetime is exactly that of the
outer GRAPH task's packed output buffer, so it can come from the same
`PTO2TaskAllocator::alloc` call as the outputs instead of a separate retained
`rtMalloc` — removing both the allocations and the memsets, and making cold
start converge with steady state. That is a separate change; this amendment only
records why it, and not batching, is the one that moves the 12 ms.

## Notes

- The first-cut device verify gate returned "busy-looking" nulls to peer
  submissions and surfaced as `sched_error_code=5 INVALID_ARGS`; the fix is
  the spin-wait on `verify_state` (dispatch-path legal: spin, no sleep).
- `graph record` (245–687 µs in the original experiment) remains a cold-miss
  cost. The current cache belongs to the registered callable, but every fresh
  process used for cold benchmarking still starts with no Definitions.

## Amendment 2026-08-20 — concurrent cold recording and compact shells

The follow-up overlaps distinct-key recording and removes repeated static tensor
metadata from the cold path:

- callable registration reserves empty host recording state for the first eight
  keys; it creates neither a Definition nor a device cache entry;
- one cold orchestration may build several Definitions concurrently. Every
  distinct Definition and per-occurrence submission is staged into the run's
  Graph block, and each submission is patched to the Definition in that block;
- each submission uses a 40-byte `GraphInvocationTensor` for dynamic boundary
  state. Static shape, stride, dtype and layout metadata stay in the retained
  Definition and are reconstructed and exact-validated on device;
- the Graph submission images themselves already share one callable-owned host
  allocation, and fixed-capacity metadata arrays remove temporary map/vector
  allocation from the measured upload window;
- a fixed 16-entry main-thread key table validates each Graph boundary once per
  run, then lets repeated same-key shells read recorder/Definition publication
  atomically without the recording mutex or another boundary scan;
- the pre-sized submission byte store uses a separate used cursor, so appending
  a shell overwrites its exact image instead of value-initializing the reserved
  capacity again;
- each recording owns a block arena for copied `ChipTensor` arguments, avoiding
  one general-heap allocation per recorded node while preserving stable tensor
  addresses for borrowed outputs;
- callable registration reserves empty recording/Definition-image storage for
  the first eight keys. A cold worker packs sections directly into its final
  image allocation rather than assembling thirteen temporary vectors, and the
  finished recording arena is retired with the callable instead of being freed
  under the publication mutex;
- a recording worker claims only one distinct key in each orchestration batch,
  so a short recording cannot consume a second job while another prewarmed
  worker remains parked; each job is assigned to that worker's private slot
  and condition variable, avoiding a pool-wide wakeup between consecutive
  cold Graph submissions;
- non-sanitizer host runtime and orchestration builds define `NDEBUG`. Sanitizer
  and C++ unit-test builds retain the reference Definition builder and its
  byte-for-byte comparison, while production cold misses execute only the
  direct builder;
- a cache-only pass reuses the callable-owned host Definition cache, but stages
  the referenced Definitions at their addresses in the new run's bind image;
- main's one-bind-image path (#1947) restacks the four live shared-memory ranges,
  the device-read arena prefix and the Graph submission block into one host
  image. One `arena_h2d` transfers that image; there is no separate `sm_h2d`.

After integration with #1947, a cold run's copy model is one bind-image H2D:
`[copied arena][compact SM][Definition objects][Graph shells]`. The original
branch's separate page-locked Definition/submission pack is therefore
superseded; retaining another allocation and upload path would only duplicate
the main implementation. DSV4 results below are from fresh processes whose
callable Definition cache starts empty; cache-hit rounds are not a substitute
for those measurements.

Before #1947 landed, rebasing onto `upstream/main` at `2f3376ab`, rebuilding
every runtime and reinstalling the editable wheel produced the following
fresh-process DSV4 `DecodeFwdEP2TP2` sample. It is retained as historical
evidence for the recording overlap, not as the current-main upload baseline:

| Rank | `host_orch` | `graph_upload` | `sm_h2d` | `arena_h2d` | Sum |
| ---- | ----------- | -------------- | -------- | ----------- | --- |
| 0 | 866 µs | 107 µs | 99 µs | 35 µs | **1,106 µs** |
| 1 | 1,007 µs | 103 µs | 94 µs | 39 µs | **1,243 µs** |
| Mean | 936 µs | 105 µs | 96 µs | 37 µs | **1,174 µs** |

The per-event timeline shows that 83 and 82 of the 86 main-thread Graph
submissions overlap worker recording. The last Graph submission also precedes
the last Definition publication by 149 and 124 µs respectively; the remaining
tail is the required final publication barrier before the packed upload, not an
intermediate submit-side wait.

Startup and externally contended runs are kept as controls rather than folded
into this table. In the same final-rebase session a contended pass measured
2.64/0.98 ms in `host_orch`, with the extra time concentrated in recorder
scheduling. It still recorded seven distinct workers and the same cold-miss
event counts, so it diagnoses host contention rather than a serial-path
fallback.

## Amendment 2026-08-21 — current main and the one-bind Graph block

The branch was then rebased onto `upstream/main` at `a5c6093d`. That base
includes #1939's producer-lifetime dependency fix and #1947's one-bind-image
layout. The DSV4 cold workload now records 1,679 nodes and eight Definitions,
instead of the earlier 1,388 nodes and seven Definitions, so its `host_orch`
number is not directly comparable with the pre-rebase table above.

The first post-rebase sample still uploaded all eight Definition objects
separately before copying the bind image. Folding those objects into the Graph
block reduced the copy-bearing stages as follows; each row is one rank from a
fresh-process hardware pass:

| Rank | Before `graph_upload + arena_h2d` | One bind image | Delta |
| ---- | --------------------------------- | -------------- | ----- |
| 0 | 1,066.261 us | 264.793 us | -75.2% |
| 1 | 1,112.880 us | 311.813 us | -72.0% |

The final sample's complete host preparation breakdown is:

| Rank | `host_orch` | `graph_upload` | `sm_h2d` | `arena_h2d` | Sum |
| ---- | ----------- | -------------- | -------- | ----------- | --- |
| 0 | 2,041.889 us | 3.841 us | 0 | 260.952 us | **2,306.682 us** |
| 1 | 2,617.405 us | 5.760 us | 0 | 306.053 us | **2,929.218 us** |
| Mean | 2,329.647 us | 4.801 us | 0 | 283.503 us | **2,617.950 us** |

Both ranks submitted 86 shells on the main thread, recorded 1,679 nodes,
built eight Definitions on eight distinct worker threads, and dropped no trace
records. On one rank the last shell preceded the final Definition by 788 us; on
the other it followed it by 9 us. The only remaining tail after the last
Definition was 20--46 us, which is the final publication/packing barrier rather
than the earlier erroneous submit-side wait. The visible run-to-run
`host_orch` variation is concentrated in worker scheduling gaps; task-submit
exposed 320 CPUs to the job, so it is not an accidental one-CPU affinity limit.
