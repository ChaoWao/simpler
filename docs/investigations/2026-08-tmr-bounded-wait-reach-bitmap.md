# Arbitrary-depth WAIT-edge reduction on the tmr orchestrator: bounded bitmap vs the 1-hop ceiling

**Date**: 2026-08-25
**Verdict**: implemented as a bounded (BL=64) reachability bitmap —
`reduce_wait_edges` on both arches (#1376), pending review/merge. Supersedes
the abandoned 1-hop PR #1830 and overturns the earlier "no reliable view past
one hop" premise recorded when #1830 was scoped.

## Question

`tensormap_and_ringbuffer` builds each consumer's fanin incrementally at
submit time from tensormap overlap and creator retention; there is no
materialized task graph. Issue #1375 asked for redundant ordering edges to
be dropped: if `A→B→C` already orders A before C, the direct `A→C` WAIT
carries no readiness information. Two prior answers existed:

- **PR #1830 (abandoned)** implemented the 1-hop case only: edge `P→C` is
  cleared when another of C's *direct* producers Q has P in its own fanin
  with `WAIT|RETAIN`, and Q is still live. Onboard and replay measurements
  showed **no qwen3/dsv4 gain**, and the approach paid
  `O(count² · fanin(Q))` per submit with a `count ≤ CHIP_FANIN_INLINE_CAP`
  guard that skipped dense fanins entirely. It was never merged.
- The scoping investigation that produced #1830 concluded arbitrary depth
  was unsound on this runtime: a producer's identity past one hop cannot
  be trusted because ring slots are reused, and a persistent per-slot
  ancestor closure was judged unaffordable on the AICPU submit path.

Issue #1376 then asked: is there a bounded representation that buys
arbitrary depth *inside a window* at a price the dispatch path can pay?

## What was tried

The reachability-bitmap design from #1376, evaluated against the real
graphs before committing to a window size:

- Each task publishes, at its own submit, a frozen 64-bit bitmap `R[t]`:
  bit `i` set means "the task submitted `i+1` positions before me has a
  WAIT path to me". Publication is once-per-submit and immutable — there
  is no incremental maintenance.
- `reduce_wait_edges` (two passes over the deduped fanin): pass 1 folds
  `direct` (one bit per WAIT producer) and `via` (`R[q] << d` per direct
  WAIT producer q — distance addition is a left shift); pass 2 clears
  DEP_WAIT on candidates whose bit `via` proves covered. `d > 64` keeps
  the edge conservatively; `d == 64` sets the direct bit without a shift.
- The safety argument replaces the issue's "generation validation": every
  builder producer is pinned by *this submit's* `fanout_count++` claim
  (all edges, any DepFlags), and slot rebind requires the CONSUMED flip
  which requires `fanout_refcount == fanout_count` — so the producer's
  `wait_reach` entry and seq provably belong to the claimed task for the
  whole pass. The bitmap is frozen at the producer's own submit, so there
  is no staleness dimension beyond slot identity, and the pin settles
  identity. This is strictly stronger than #1830's position, which read
  Q's *fanin pointers* (stale after Q's completion) and therefore needed
  the live-Q + RETAIN-cover conditions.
- `wait_reduction_sim` (shipped as
  `simpler_setup/tools/wait_reduction_sim.py`) replays the exact online
  algorithm over `deps.json` captures and compares it with the full-DAG
  transitive reduction — the upper bound the hbg host-resident pass
  achieves on its materialized graph.

## Result

Cost per submit drops from #1830's `O(count² · fanin(Q))` (capped at 64
inline entries, dense fanins skipped) to `O(fanin)` single-word ops with
no cap: one 16-byte `WaitReachEntry` (bitmap + global seq) per slot,
16 KiB per 1024-slot ring slice, 1 MiB total at the default window on
both arches — orchestrator-private arena storage beside
`fanin_seen_epoch`.

Coverage gains over 1-hop, all locked by cpput on both arches:

- depth ≥ 3 (`A→B→C→D` + `A→D` reduces; 1-hop structurally cannot see it),
- cross-ring candidates (a global `submit_seq` makes distances
  ring-independent),
- spill-region candidates (no inline cap),
- WAIT-only covering edges count as reachability witnesses (no RETAIN
  needed on the covering hop).

Representative replay (`paged_attention_unroll`, 1280 WAIT pairs): upper
bound 256 redundant edges; **BL=64 removes 256/256 (100%)** — the
single-word window already saturates this graph's redundancy; p99
producer→consumer distance is 3.

Onboard dep-gen captures from 2026-08-25 give the production-model result
required by acceptance #9:

| capture | tasks | WAIT pairs | full-DAG upper bound | BL=64 | BL=128 | BL=256 |
| ------- | ----: | ---------: | -------------------: | ----: | -----: | -----: |
| qwen3_14b_decode, B16/S3500 | 11,166 | 23,605 | 40 | 1 (2.50%) | 1 (2.50%) | 1 (2.50%) |
| deepseek_v4_flash_decode, EP2/TP2 | 15,971 | 45,917 | 21,698 | 10,065 (46.39%) | 13,678 (63.04%) | 20,214 (93.16%) |

Qwen's result is not a misleadingly small window: 39/40 redundant edges are
cross-ring long edges and remain outside BL=256, even though BL=256 contains
99.8% of all WAIT pairs. Its producer-distance CDF is p50/p90/p99/max =
51/119/143/10,894. DeepSeek-V4 is genuinely window-sensitive:
p50/p90/p99/max = 16/186/4,429/15,960; BL64/128/256 leave
11,633/8,020/1,484 redundant window misses respectively. Cross-ring misses
are 7,187/4,433/1,337.

Each removal also removes one readiness-fanout node and one dependency-pool
entry in the simulator's resource estimate. The demote/drop split is
8,947/1,118 for DeepSeek-V4 at BL64. Treat the split, but not the total
removal count, conservatively: dep-gen cannot preserve the kind of an
explicit dependency yet (#1827), making 39 Qwen pairs and 43 DeepSeek-V4
pairs uncertain between RETAIN-only demotion and pure drop.

The matching 10-round onboard A/B used merge-base `d11689f6`, PTO-ISA pin
`cd4a3d3f7a1a27fcfe536f617e9bca3008929664`, and the same locked devices for
each baseline/current pair:

| workload | metric | merge-base | BL64 bitmap | change |
| -------- | ------ | ---------: | ----------: | -----: |
| qwen3_14b_decode | Effective | 35,873.0 us | 35,844.3 us | -0.08% |
| qwen3_14b_decode | Orch | 8,875.7 us | 8,608.1 us | -3.02% |
| deepseek_v4_flash_decode | max-rank Effective | 293,746.0 us | 232,966.5 us | -20.69% |
| deepseek_v4_flash_decode | max-rank Orch | 17,626.0 us | 17,674.4 us | +0.27% |

DeepSeek-V4's distributed scheduler time is noisy across ranks, so its
end-to-end row is computed per round as `max(rank0, rank1)` and only then
averaged over 10 rounds. The conclusion is deliberately narrow: BL64 is
neutral on Qwen, neutral in DeepSeek-V4 orchestration cost, and materially
reduces the latter's scheduler-critical window on this run.

## Why this shape

- The bitmap is *frozen at publication*, which is what makes the identity
  argument simple: unlike fanin pointers, a frozen value has no staleness
  dimension. The pin argument then closes the only remaining hole (slot
  reuse).
- BL=64 is one native word: the shift-merge is a single instruction and
  the `d == BL` UB case is a natural no-shift boundary. Wider windows
  (128/256) need multiword shifts and 2–4× the side storage. DeepSeek-V4 does
  show a real coverage gap, but BL64 already removes 10,065 edges and produced
  a measurable end-to-end gain without increasing the 1 MiB default side
  storage. Widen only after an onboard A/B shows that the extra simulated
  removals repay multiword submit-path work and 2–4 MiB storage.
- `fanin_wait_count` (payload padding, layout unchanged) splits the
  readiness denominator from the storage count, so RETAIN-only survivors
  and dropped `DEP_NONE` entries keep their pin-release accounting intact.

## When to reconsider

- If a future capture shows `BL=64 removed ≪ upper_bound` while
  `pct_pairs_within_window` is the binding constraint, widen the window —
  the two-pass structure generalizes to N words unchanged.
- Full-DAG reduction on tmr remains out of scope by the same argument as
  before: the orchestrator never materializes the graph. hbg does, and
  its host-side exact reduction is the right home for unbounded depth.
