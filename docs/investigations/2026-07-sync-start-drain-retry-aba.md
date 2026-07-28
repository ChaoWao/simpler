# `sync_start` drain retry ABA across reusable barrier state

**Date**: 2026-07-27
**Verdict**: fixed by tagging drain attempts, reducing acknowledgements through
a generation-tagged tree, and using scheduler thread 0 as the coordinator. The
deterministic runtime hook was used to validate the interleaving and then
removed; hook-free unit and scene tests remain as regression coverage.

## Question

Issue [#1455](https://github.com/hw-native-sys/simpler/issues/1455)
identified an ABA race in the `sync_start` drain retry protocol. The original
protocol reused these values across attempts without a generation:

- `drain_ack_mask`
- `drain_worker_elected`
- `drain_stage_go`
- `drain_stage_done_mask`

When the elected drain worker found fewer available resources than the pending
task required, it cleared the acknowledgement and election state so scheduler
threads could resume completion polling. A thread from the old attempt could
already have crossed the acknowledgement barrier and then be delayed before
election. If it resumed after the reset, every value it observed was locally
valid, but some values belonged to the next attempt.

The failure ended in a split wait: the stale owner waited for the old attempt's
`stage_done` mask while its peers waited for the new attempt's `ack_mask`.
Neither side could make progress.

The investigation asked:

1. Can that stale-participant interleaving be reproduced deterministically?
2. Which attempt-tagged protocol prevents state from different retries from
   being combined?
3. Can the fix avoid adding a shared atomic hot spot or a per-spin O(N) scan?

## What was tried

### Deterministic runtime hook

The first hook was tested on upstream commit
`d0bc661a5b41b4925f2d72c77e86228cc351cf23`. It was enabled with
`SIMPLER_DRAIN_ABA_TEST=1` and used four AICPU threads, which give the
tensormap-and-ringbuffer runtime three active scheduler threads.

The scene submitted:

1. one long-running MIX holder block;
2. one 24-block MIX task with `sync_start`.

The holder left only 23 clusters available, forcing the first drain attempt to
retry. The hook parked scheduler thread 1 after the old attempt's
acknowledgement barrier but before election. After another scheduler reset the
drain state, the hook released thread 1 into the original protocol. Once the
resource count reached 24, that stale thread could become the owner and publish
`stage_go` using state assembled from two attempts.

The important property was that the hook did not synthesize the final timeout.
It only selected the interleaving; the original election and staging protocol
created the split barrier.

The hook was adapted as the production protocol changed. In the final
generation-tagged version it:

- published an independent per-thread arrival token;
- parked thread 1 after the acknowledgement barrier;
- waited for another scheduler to advance `drain_attempt`;
- waited until the peer schedulers had acknowledged the new attempt;
- released thread 1 while it still held the old local attempt.

The fixed protocol then had to reject that stale local attempt before it could
coordinate staging. A dedicated `NOT_EXERCISED` failure made the test fail if
the resource retry never occurred, preventing a silent pass where the intended
interleaving was not reached.

### Coordination designs

The following production designs were built and measured:

| Design | Coordination shape | Outcome |
| ------ | ------------------ | ------- |
| Per-thread generation tokens with O(N) scan | Every scheduler publishes independently; the coordinator scans all tokens | Correct, but O(N) work on each barrier spin |
| Packed generation and ack mask | One shared atomic stores the attempt and mask | Rejected because shared LL/SC updates contended |
| One-way tree without root broadcast | Fixed coordinator; followers proceed without waiting for the root token | Rejected because polling moved to the shared `stage_go` cache line |
| Generation-tagged tree with rotating coordinator | Tree reduction plus `attempt % N` coordinator | No stable performance benefit |
| Generation-tagged tree with fixed coordinator | Tree reduction and root broadcast; scheduler thread 0 coordinates staging | Retained |

The retained tree has one token per scheduler. Each scheduler waits for at most
two child subtree tokens, publishes one subtree token, and waits for the root
token before staging. Fan-in work per scheduler is O(1), the critical path is
O(log N), and no shared read-modify-write operation is required.

Scheduler indices are contiguous in all three runtime variants, so thread 0 is
present whenever a scheduler exists. Fixing the coordinator also removes the
election state and prevents a stale scheduler from becoming an owner in a later
attempt.

### Validation and measurement

Correctness checks covered the A2/A3 tensormap-and-ringbuffer and
host-build-graph runtimes and the A5 tensormap-and-ringbuffer runtime.
Generation-token unit tests were added for both architecture families.

Hardware scene tests covered:

- the deterministic ABA hook;
- normal and stress `sync_start` drains;
- one, two, three, and four active schedulers;
- a non-power-of-two three-scheduler acknowledgement tree.

Performance comparisons used one locked A2/A3 device and ABBA ordering. Each
ABBA segment ran 100 rounds. The final stress follow-up ran five independent
`A100-B100-B100-A100` cycles, giving 1,000 samples per version.

## Result

### Reproduced failure

The original hook run reached the intended cross-attempt state:

```text
attempt 1 owner: available=23 < block_num=24, reset for retry
stale thread 1: local_attempt=1, current_attempt=2
stale thread 1: publishes stage_go with ack_mask=0x5
threads 0 and 2: wait for attempt 2 ack_mask=0x7
stale thread 1: waits for attempt 1 stage_done=0x7
```

The run then failed with AICPU timeout `507018`. This established that the
generation-less reusable fields could combine state from different attempts;
it was not only a theoretical stale-load concern.

### Correctness after the fix

With the generation-tagged tree and fixed coordinator:

- the deterministic hook passed on A2/A3 hardware;
- normal and stress `sync_start` scene tests passed;
- one-, two-, three-, and four-scheduler configurations passed;
- A2/A3 and A5 scheduler-state unit tests passed;
- static review found no cross-attempt election, staging, teardown, or
  memory-ordering gap.

The final fixed-coordinator hardware tasks included:

- ABA hook: `task_20260727_031658_60345132353`
- three-scheduler normal: `task_20260727_031726_6337306055`
- three-scheduler stress: `task_20260727_031758_65330616500`
- two-scheduler MIX spill: `task_20260727_031827_66165821039`
- one-scheduler normal: `task_20260727_031859_67079319390`
- four-scheduler host-build-graph: `task_20260727_031941_68308727426`

### Performance

Relative to the earlier PR version, the rejected alternatives changed mean
Device time as follows:

| Design | Normal | Stress |
| ------ | -----: | -----: |
| Packed shared atomic | +16.42% | +19.02% |
| Tree without root broadcast | +10.34% | +7.60% |
| Fixed coordinator | -5.67% | +1.51% |
| Rotating coordinator | +3.02% | -1.32% |

Positive values are slower. The rotating result did not show a consistent win
across normal and stress cases, while the fixed coordinator removed election
work and preserved the tree's distributed polling.

After rebasing, a direct 100-round ABBA comparison against upstream runtime
commit `a53cc168` measured:

| Case | Device | Effective | Scheduler | 10% trimmed Device |
| ---- | -----: | --------: | --------: | -----------------: |
| Normal | -0.91% | -0.83% | -0.52% | -0.14% |
| Stress | -2.93% | -3.02% | -2.99% | -4.56% |

The final PR was subsequently rebased from `a53cc168` to `1c475a70`; the only
upstream difference was a CI workflow edit, so the runtime source and measured
binary were unchanged.

The five-cycle, 1,000-sample stress follow-up compared upstream `1c475a70`
against PR commit `34a250e`. Its pooled Effective time was 4.07% slower, but
the five paired deltas ranged from -3.25% to +18.90%. Treating each process
cycle as the correlated unit gave a 95% interval of [-5.41%, +14.80%]. The
stress distribution was multimodal, so neither the earlier approximately 3%
improvement nor the later approximately 4% regression was established as a
stable effect.

Plain non-drain cases ranged from -3.19% to +1.80% across Device, Effective,
Orchestrator, Scheduler, and trimmed Device measurements, with no consistent
regression.

## Why not (now)

The generation-less acknowledgement and election protocol is not safe to
restore: the deterministic run demonstrated a reachable split barrier, and the
attempt tag directly distinguishes the state that the old protocol conflated.

The packed atomic and no-root-broadcast designs are not retained because they
introduced clear performance regressions. Rotation added state and
generation-dependent ownership without a stable gain.

The deterministic hook itself was valuable during the investigation, but its
permanent implementation required an environment-controlled behavior gate,
runtime launch plumbing, scheduler hot-path branches, a production error code,
and test-only token scanning. The hook was A2/A3-specific while the production
fix spans three runtime variants. Once the interleaving and fix were recorded,
the permanent regression barrier was better provided by hook-free
generation-token unit tests plus normal and stress scene tests.

## When to reconsider

- A future scheduler change reintroduces reusable drain coordination state that
  is not tagged by `drain_attempt`.
- A failure shows that unit tests and repeated normal/stress scene tests do not
  cover a new cross-attempt transition.
- The project gains a fault-injection framework that can suspend scheduler
  participants without adding product configuration, status codes, or
  test-only branches to the runtime.
- Hardware with a substantially larger scheduler count makes the O(log N)
  tree visible in profiles; compare a distributed tree against a contention-
  free alternative on that hardware before changing the protocol.

## References

- [Issue #1455](https://github.com/hw-native-sys/simpler/issues/1455) -
  original ABA analysis, deterministic reproducer, and split-barrier logs.
- [PR #1461](https://github.com/hw-native-sys/simpler/pull/1461) - production
  fix, review discussion, and performance follow-ups.
- Upstream reproduction commit:
  `d0bc661a5b41b4925f2d72c77e86228cc351cf23`.
- Final measured upstream base:
  `1c475a7016f0bbc8a3d6cb7515acff6c14769905`.
- Final pre-hook-removal PR commit:
  `34a250e4fa5a65a40ad2e839c87ab54f920f1623`.
- [Environment-variable and macro gating](../../.claude/rules/env-macro-gating.md).
