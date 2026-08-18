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

## Result (qwen3, `SIMPLER_SKIP_DEVICE_RUN=1`, 5 serial runs each, median)

| Stage | Before | After | Δ |
| ----- | ------ | ----- | - |
| orch image build (incl. zero-fill) | 931 µs (826) | 23.6 µs (10.0) | −97.5% |
| submission bytes | 5,310,080 | 232,944 | −95.6% |
| orch window total | 1,834 µs | 836 µs | −54% |
| **H2D upload time** | **14.66 ms** | **12.10 ms** | **−17%** |

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
   matter. The 12.1 ms / 41 ≈ 295 µs figure is the average across one large
   and 40 small calls, not a uniform per-submission measurement, but it bounds
   the scale of the driver and synchronization overhead dominating the stage.

So the model is `time ≈ N × (malloc + memcpy latency) + bytes / BW`, and at
these sizes the first term dominates by two orders of magnitude. The byte
reduction could only remove the (already small) second term — consistent
with the measured 2.5 ms saving landing near 40 × 130 KB / 0.36 GB/s of
actual removed traffic.

**The follow-up this points at**: batch the 40 reference submissions into one
packed upload (they are POD and contiguous by construction; one malloc, one
memcpy, per-device offsets), which attacks the *N* in the dominant term.
Expected to take the stage from ~12 ms to well under 1 ms. Not done in this
change to keep the wire-v2 commit reviewable and independently revertible.

## Notes

- The first-cut device verify gate returned "busy-looking" nulls to peer
  submissions and surfaced as `sched_error_code=5 INVALID_ARGS`; the fix is
  the spin-wait on `verify_state` (dispatch-path legal: spin, no sleep).
- `graph record` (245–687 µs) remains per-run and untouched — the per-run
  Definition cache discard is a separate, still-open item (KNOWN_ISSUES).
