# 2026-09 — Sizing the host-log queue: the claim budget is not the constraint

**Verdict: do not change `kProducerClaimAttempts`.** It has ~12× headroom over
the worst attempt count any workload here produced, and lowering it converts
successful writes into drops for a worst case that never occurs. The constraint
on loss is `kQueueCapacity` — the writer's drain rate — and that is structural,
not incidental.

Measured on `d42d465bc` (#2029, item 6 of #1792), on a 320-core aarch64 host at
load average ~80.

## The question, and why it looks like a knob

`src/common/log/host_log.cpp` gives a producer a fixed budget for the lock-free
MPSC position claim:

```cpp
constexpr size_t kQueueCapacity = 4096;
constexpr size_t kProducerClaimAttempts = 1024;
```

The budget exists for boundedness, not speed: a CAS retry loop is itself a latent
unbounded wait, so without a ceiling "never blocks" would describe the happy path
rather than the worst case. Exhausting it is a counted drop
(`SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED`), never a wait.

`1024` reads like a suspicious magic number, and it is the most knob-shaped thing
on the screen when a run reports drops. Before #2029 split the drop counter by
cause there was no way to do better than guess, because the total does not say
which step lost the record.

## Probe 1 (saturation) — measured the wrong regime

N threads each emitting 100k records with no pacing, output to `/dev/null` so the
writer is as fast as it can be:

| threads | emitted | dropped | `queue_full` | `claim_exhausted` |
| ------- | ------- | ------- | ------------ | ----------------- |
| 4 | 400,000 | 182,279 (45.6%) | 182,279 | **0** |
| 16 | 1,600,000 | 1,377,572 (86.1%) | 1,377,572 | **0** |
| 64 | 6,400,000 | 6,007,746 (93.9%) | 6,007,746 | **0** |

Every loss is `queue_full`; the budget is never exhausted. But this **does not**
establish that the budget is generous, and reading it that way was the first
wrong conclusion here. When the queue is full a producer exits on
`difference < 0` **before spending any attempt**, so a saturated run exercises the
early-exit path and barely visits the claim loop at all. The claim loop only burns
attempts when the queue *has* room and producers contend — that is, when the
writer is keeping up.

Two distinct questions were being conflated:

- *Does the budget cause drops?* — answered by the counter. No.
- *How many attempts does a producer actually spend?* — not answered by the
  counter at all, and it is the one that decides whether the loop holds the
  calling thread for long and whether a smaller bound would fit.

## Probe 2 (paced) — the attempts-to-win distribution

Same shape, but each producer sleeps ~2 µs between records so the queue keeps
room and the claim loop is the path under test. Throwaway instrumentation
recorded the attempt index on which each producer won its slot.

| threads | won on 1st try | tail | **max attempts to win** |
| ------- | -------------- | ---- | ----------------------- |
| 4 | 78% | ≤4 | **4** |
| 16 | 57% | ≤16 | **14** |
| 64 | 65% | ≤32: 784 · ≤64: 3 | **37** |

Unpaced 64 threads, for contrast: **max 86**.

So the realistic cost is **one attempt**, and the worst observed across every
shape tried is 86 against a bound of 1024.

## What that means for the calling thread

One attempt is an acquire load plus a compare, and a CAS on the winning path. On
a contended cache line that is roughly 50–100 ns here.

| case | attempts | cost |
| ---- | -------- | ---- |
| typical | 1 | tens of ns |
| worst observed | 86 | ~9 µs |
| theoretical bound | 1024 | ~100 µs |

The ~100 µs ceiling is real but never approached. Worth stating precisely, since
"bounded" is easy to misread: **exhausting the budget drops the record rather
than delaying it**, so the bound is not "how long a caller waits" — it is how
much CPU the caller will burn before giving up on that record. Nothing in the
loop waits on anything external: no syscall, no lock, no condition variable.

## Why a smaller bound is worse, with numbers

A bound of 16 was the specific alternative considered.

- 16 threads already reached **14**, i.e. the edge.
- 64 threads paced reached **37**; unpaced, **86**.
- From the 64-thread paced histogram, **787 records (0.06%)** needed more than 16
  attempts. With a bound of 16 those become `claim_exhausted` drops — records
  that in fact succeeded after ~2 µs.

What it buys is a worst-case CPU burn of ~1.6 µs instead of ~100 µs, for a worst
case that never happened. Real drops traded for a hypothetical saving.

A value covering everything observed with margin would be around 128–256, so
`1024` is genuinely larger than needed. It is still not worth changing: the
current value costs nothing (never reached, no memory, no effect on any
successful claim), while lowering it risks exactly the trade above, and the
observed maximum already moved 37 → 86 purely by changing the workload shape.

## Two properties recorded, not fixed

Neither is a requirement of item 6 — which asked for "never block; drop and
count", and gets both — but both are properties of *this* implementation and are
cheaper to write down than to rediscover.

**Head-of-line blocking.** `pop()` is strictly in-order: a producer that claims
position *P* and is preempted before publishing `sequence = P+1` leaves the writer
parked on *P*, so records at *P+1* and beyond cannot drain even though they are
published. The queue then fills behind the gap and *other* producers start
dropping. The window is one ≤512-byte `memcpy` plus two stores, so it is small —
but it is not zero, and on an oversubscribed host preemption is routine. The
writer handles it correctly (it sleeps on the semaphore rather than spinning, and
`WriterSleepsWhenAdjacentProducersPublishOutOfOrder` pins that), but *the writer
not burning CPU* and *the queue not backing up* are different properties and only
the first is tested.

**Fairness.** The `difference > 0` branch reloads the position and retries, so a
thread that keeps losing is starved into a drop rather than deadlocking. Under
contention the losses fall on the slowest producers, which is the opposite of the
useful bias for diagnostics: the thread that is stuck is often the one worth
observing.

## Method notes

- **The drop counter's breakdown is what made any of this answerable.** A single
  total cannot distinguish "the queue is too small" from "the claim budget is too
  tight" from "the destination is broken", and those want three different fixes.
  `_host_log_dropped_records_by_reason()` returns it.
- **A saturating benchmark is the wrong instrument for the claim path**, because
  saturation routes producers through the early exit. Pace the producers to hold
  the regime you mean to measure.
- The drop rates above (45–94%) come from deliberately pathological workloads
  with no pacing and are **not** representative. What they establish is the
  queue's role: it absorbs bursts, not sustained overload — which is what its own
  comment claims ("large enough to absorb ordinary bursts") and now has a number
  behind.
- Everything here is one machine and two workload shapes. The defensible reading
  is "1024 has ~12× headroom over the worst observed and 16 does not", not "1024
  is optimal".

## Related

- #1792 item 6 — the requirement this implements
- #2029 — the queue, the drop breakdown, and the `[HOSTLOG_DROPS]` log record
- [docs/logging.md](../logging.md) — the destination, the counters, the record
- [docs/dfx/host-trace.md](../dfx/host-trace.md) — what a reader is told about an
  incomplete log
