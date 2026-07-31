# Why qwen3_14b_decode held a device for 406 s — and the four answers that were wrong

**Date**: 2026-07-30
**Verdict**: root cause found (torch thread oversubscription in goldens, fixed in
[#1601](https://github.com/hw-native-sys/simpler/pull/1601)); golden caching,
golden vectorisation, and a nightly golden split all considered and dropped

## Question

`qwen3_14b_decode` was the one scene test excluded from the general a2a3 sweep.
`ci.yml` carried a `HEAVY_IGNORE="--ignore=.../qwen3_14b_decode"` plus a step of
its own with a 1800 s session budget, on the grounds that at "~5 min" it would
eat more than half the sweep's 600 s budget — the budget that makes the sweep a
hang detector.

The case is an ordinary `@scene_test(level=2)` class. Nothing about it is
structurally special, so the question was where its time actually went, and
whether the exclusion could be removed rather than managed.

The intuition a future contributor will reach for is the one that misled this
investigation four times over: **the expensive thing must be the big thing.**
The case has a 38 GiB fixture, 40 fused layers, 36 kernels including a
template-heavy vendor tree, and a 40-layer host reference. Every one of those is
a plausible culprit and only one of them was right.

## What was tried

### Separating queue wait from work

Wall-clock per step comes from the Actions API; it does **not** distinguish
waiting for a device from using one. The `npu-lock` lines in the job log do:

```bash
gh api "repos/hw-native-sys/simpler/actions/jobs/<JOB_ID>" \
  --jq '.steps[] | "\(.name)\t\(.started_at)\t\(.completed_at)"'

gh run view --repo hw-native-sys/simpler --job <JOB_ID> --log \
  | grep -E '\[npu-lock\]|task_[0-9]{8}_'
```

`task-submit` localises its messages, so match on the ASCII parts. Submission
carries a `task_<date>_<id>` token. The three `[npu-lock]` line shapes are
distinguishable without reading the words:

| Line ends with | Meaning |
| -------------- | ------- |
| `)...` | the wait for that device started |
| `(pid=NNNN)` | the lock was granted |
| neither | the locks were released |

Queue time is submission to the first `pid=`; card-held time is that `pid=` to
the release. Note the log fetch goes to
`productionresultssa10.blob.core.windows.net`, which this box's whitelist proxy
rejects with 403 — unset `HTTPS_PROXY` for that call.

### Timing the host phases off-device

`generate_inputs` and `compute_golden` are pure torch and need neither CANN nor
a device; `simpler_setup` imports with `sys.path` pointing at the source tree:

```python
import sys; sys.path.insert(0, "python"); sys.path.insert(0, ".")
from simpler_setup.goldens import qwen3_14b_decode as Q
args = Q.generate_inputs(1234, 3500)      # fixture
Q.compute_golden(args)                    # 40-layer reference
```

### Timing compilation per source

`KernelCompiler` needs only `PTO_ISA_ROOT` and CANN's `ccec` — no built runtime,
no device. The vendor CCE kernels additionally need the `_CANN_INCLUDE_DIRS`
list the test builds from `$ASCEND_HOME_PATH`; without them they fail on
`'basic_api/kernel_basic_intf.h' file not found` and silently drop out of any
total.

```bash
export PTO_ISA_ROOT=<repo>/build/pto-isa
source /usr/local/Ascend/cann/set_env.sh
```

## Result

### The job, and the case inside it

`st-onboard-a2a3`, ci run 30507320146, job 90761014912 — 2297 s total:

| Where the job's 2297 s went | Wall | Share |
| --------------------------- | ---- | ----- |
| **queued waiting for dies** (8 separate `task-submit` acquisitions) | **1187 s** | **52%** |
| holding a device | 1005 s | 44% |
| checkout, venv, `pip install '.[test]'` | ~105 s | 5% |

The sweep's first acquisition alone waited **976 s** for four free dies. Of the
eight acquisitions, the five trailing smoke steps spent 210 s re-queueing for
75 s of work.

`qwen3_14b_decode` held a device for **406 s**, decomposed by measuring each
phase separately on the same class of box (320 cores):

| Phase | Wall | Needs a device? |
| ----- | ---- | --------------- |
| kernel compilation — 36 incores + 1 orchestration | 59 s | no |
| `generate_inputs`, the 38 GiB fixture | 13 s | no |
| **`compute_golden`, 40 layers** | **359 s** | **no** |
| host→device upload, device run, comparison | remainder | yes — device busy for **tens of ms** |
| | ~431 s measured vs 406 s in CI (6%) | |

Compilation breaks down as ~1.1–1.3 s per ordinary incore, 6.6 s for the
orchestration, and 8.4 s for the vendor FAI kernel's AIV variant (1330 KiB) —
the single outlier.

### The root cause

None of the golden's 359 s was arithmetic. Its reference walks **3584 small
slice operations per layer**, and torch sizes its intra-op pool from the core
count, so on a many-core host every one of those calls forks and joins hundreds
of threads to move a few KiB:

| torch threads | per layer |
| ------------- | --------- |
| 320 (default = `nproc`) | 6.35 s |
| 64 | 1.31 s |
| 16 | 1.13 s |
| 8 | 1.07 s |
| 4 | 1.05 s |
| 2 | 1.59 s |
| 1 | 1.82 s |

Five to eight times the cost of the work, flat from 4 to 16, turning back up
below 4. Nothing in `simpler_setup/`, `conftest.py` or the workflows had ever
set a thread count, and all three goldens under `simpler_setup/goldens/` are
built from per-tile Python loops — qwen is merely the densest (8 `for`-range
loops against 2 each), so every golden was paying this on every many-core host.

Capping to 8 around the two `compute_golden` call sites takes the golden to
~38–43 s and the case to ~115 s. The results are unaffected: the same fixture
golden-computed at 320 and at 8 threads is **bit-identical** across `out`,
`k_cache` and `v_cache` (`torch.equal` true) — worth checking rather than
assuming, since thread count can reorder float reductions.

### The four wrong answers

Each was plausible, each was killed by a specific measurement. They are recorded
because the next person will reach for them in the same order.

| Hypothesis | What killed it |
| ---------- | -------------- |
| The 38 GiB fixture dominates | `generate_inputs` is **13 s**. The 38 GiB is `torch.cat([w] * 40)` replication of one layer's weights, so only ~2 GB is ever generated; the rest is memcpy. |
| Kernel compilation dominates | **59 s** measured per source. Real, and now the largest remaining term, but never the majority. |
| Per-case process/worker startup dominates | The floor is real — `test_hello_worker` (`Worker.init()` + `close()`, no kernels) is 4.1 s, and the eleven allreduce collectives take 17.9–20.6 s each despite exercising five different algorithms, which does say fixed overhead. But 57 cases summing to 615 s cannot explain a 1482 s step. |
| The 320-thread golden figure is an artifact of this box | Half true and the most instructive error. The README reported "~49 s golden", which matches a 4–16-thread host, so the 359 s looked like a local artifact and was retracted. It was not: summing the independently measured phases (59 + 13 + 359 = 431) against the 406 s CI window closes to 6%, which only works if CI's golden is also the slow one. Both numbers were real; they differ by thread count, and the README's had never described CI. |

An earlier hypothesis — that the unexplained time was device contention — was
correct in general and wrong for this case: contention is 52% of the *job*, but
qwen's own acquisition waited 0 s because it asks for a single die.

## Why not (now)

Four fixes were considered and dropped once the thread curve was measured.

**Caching the golden.** Deterministic from `(seed, seq_len)`, and the payload is
small — decode writes one token per sequence per layer, so `out` plus the
touched KV slots is ~2.8 MB against a 14 GiB pool. Dropped: with the cap the
golden costs ~40 s, which does not justify a cache key that must hash the
reference implementation's own source. A key over inputs alone goes stale
silently when `_one_layer` changes and then confirms a wrong device result —
worse than no cache.

**Vectorising the gather.** The 3584 slice operations per layer could be one
`index_select` over `block_table`. Dropped: the cap already removes 85% of the
cost without touching a reference implementation whose readability is the point,
and it fixes the other two goldens at the same time.

**Splitting the golden to a nightly workflow.** Implemented, then reverted
before merge. Dropped: it moved ~40 s of work at the cost of a workflow with no
owner — a red nightly has no PR to attach to, so `discipline.md` §5 does not
reach it — and it put the only check that 40 layers compose correctly (per-layer
indexing into the stacked weights, per-layer KV pool offsets; no smaller test
covers either) on a 24-hour delay with a full day of merges to bisect.

**`skip_golden` on the case in CI.** Dropped for the same coverage reason,
without the nightly's compensation: the 40-layer dispatch would keep running
with nothing checking it was right.

## When to reconsider

- **The cap value.** 8 is a middle of a flat 4–16 range measured on a 320-core
  aarch64 host with torch 2.7.1+cpu. If goldens grow tiles large enough that
  their per-op work stops being dwarfed by pool overhead, re-measure the curve
  before assuming 8 still fits.
- **Golden caching or vectorisation.** Worth re-opening only if a golden's
  post-cap cost becomes a material share of its case again — say, if a future
  case's reference exceeds a couple of minutes with the cap in place.
- **Compilation.** It is now the largest term in the case (59 s of ~115 s) and
  has no cross-run cache: `_compile_cache` is a module-level dict cleared at
  session end, so every CI run recompiles every kernel. The key machinery
  already exists (`l3_compile_cache_key()` composes a stable key) and only ever
  reaches an in-memory dict. Tracked with the device-lock question in
  [#1604](https://github.com/hw-native-sys/simpler/issues/1604).
- **The device lock.** All ~115 s of the case is CPU work performed while
  holding a card, because `ci.yml` wraps the whole pytest session in
  `task-submit`. Also [#1604](https://github.com/hw-native-sys/simpler/issues/1604);
  note it pulls against the competing idea of holding **one** lock for the whole
  job, which would reclaim the 210 s of re-queueing but lengthen the hold.

## References

- [#1601](https://github.com/hw-native-sys/simpler/pull/1601) — the thread cap,
  and qwen rejoining the sweep (`HEAVY_IGNORE` and the dedicated step deleted)
- [#1604](https://github.com/hw-native-sys/simpler/issues/1604) — compile cache
  and CPU/NPU phase split
- [#1589](https://github.com/hw-native-sys/simpler/pull/1589),
  [#1590](https://github.com/hw-native-sys/simpler/pull/1590) — the change-detection
  work that surfaced this, and [`ci-change-detection.md`](../../.claude/rules/ci-change-detection.md)
- ci run 30507320146, job 90761014912 — the measured `st-onboard-a2a3` job
- [`running-onboard.md`](../../.claude/rules/running-onboard.md) — device locking
  and why the `npu-lock` timestamps exist to be read
