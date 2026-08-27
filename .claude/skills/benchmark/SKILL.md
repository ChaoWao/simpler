---
name: benchmark
description: Benchmark runtime performance on hardware. If the current branch has commits ahead of upstream/main or uncommitted changes, compares against the fork point (merge-base). Otherwise benchmarks current state only. Use when the user asks to benchmark, measure performance, or compare latency.
---

# Benchmark Workflow

Benchmark runtime performance on Ascend hardware. Automatically detects whether to run a single benchmark or a comparison.

## Modes

| Condition | Mode | What happens |
| --------- | ---- | ------------ |
| 0 commits ahead AND no uncommitted changes | **Single** | Benchmark current state and report its runtime-specific timing columns |
| >= 1 commits ahead OR uncommitted changes | **Compare** | Benchmark merge-base (worktree) AND current workspace, show comparison table |

## Input

Optional benchmark arguments forwarded to `tools/benchmark_rounds.sh`:

```text
/benchmark
/benchmark -d 4 -n 50
/benchmark -d 4 -d 6
/benchmark --serial-orch-sched
```

Extra arguments (`-n`, `-r`, `--serial-orch-sched`, etc.) are forwarded to
`tools/benchmark_rounds.sh`.

### Device arguments (`-d`)

The `-d` flag specifies NPU device IDs.

**Hard rule: one benchmark process per device at any time.** Never run two benchmark processes on the same `-d` device concurrently — not two runtimes, not baseline + current, nothing. This prevents resource contention and ensures stable measurements.

On a shared hardware host, the complete single or compare run must execute
inside one `task-submit` allocation. Let that allocation own every device for
the full sequence; do not submit one job per example or run the script bare.

| `-d` count | Compare mode behavior |
| ---------- | --------------------- |
| One device (`-d 4`) | **Sequential**: baseline first, then current, both on the same device. Multiple runtimes also run serially on that device. |
| Two devices (`-d 4 -d 6`) | **Parallel per-runtime**: for each runtime, baseline on first device and current on second device can run in parallel (different devices). Multiple runtimes still run serially — finish one runtime on both devices before starting the next. |
| Zero (not specified) | Let `task-submit --device auto` allocate one device (see Step 2) |

**Defaults** (when not specified): use `benchmark_rounds.sh` defaults (device 0, 100 rounds, a2a3, tensormap_and_ringbuffer).

## Runtime Selection

`tools/benchmark_rounds.sh` supports `-r <runtime>`:

- `tensormap_and_ringbuffer` (default)
- `host_build_graph`

Each architecture/runtime quadrant has its own list at the top of the script.
TMR reports Host / Device / Effective / Orch / Sched. HBG reports Host / Device
because its orchestration runs on the host and has no device-side Orch/Sched
windows. `--serial-orch-sched` is TMR-only and must be rejected for HBG.

## Step 1: Detect Mode

```bash
git fetch upstream main --quiet
COMMITS_AHEAD=$(git rev-list HEAD --not upstream/main --count 2>/dev/null || echo "0")
HAS_CHANGES=$(git status --porcelain)

if [ "$COMMITS_AHEAD" -eq 0 ] && [ -z "$HAS_CHANGES" ]; then
  MODE="single"
else
  MODE="compare"
  MERGE_BASE=$(git merge-base upstream/main HEAD)
fi
```

## Step 2: Device Isolation

When `task-submit` is available, allocate the requested device IDs or use
`--device auto`; run the architecture gate as the first command inside that
allocation, then pass `$TASK_DEVICE` to every benchmark command. Hold one
allocation for the whole baseline/current sequence. When `task-submit` is
unavailable, follow
`.claude/lib/onboard-detection.md` and clearly report that the fallback run is
unlocked.

Before submitting, inspect the queue:

```bash
task-submit --list
```

Remove each `-d` option from the forwarded `BENCH_ARGS` after collecting its
value in `REQUESTED_DEVICES`. Build the allocation arguments once and reuse
them for the single `task-submit` call:

```bash
case "${#REQUESTED_DEVICES[@]}" in
  0) TASK_SUBMIT_DEVICE_ARGS=(--device auto --device-num 1) ;;
  1) TASK_SUBMIT_DEVICE_ARGS=(--device "${REQUESTED_DEVICES[0]}") ;;
  2) TASK_SUBMIT_DEVICE_ARGS=(--device "${REQUESTED_DEVICES[0]},${REQUESTED_DEVICES[1]}") ;;
  *) echo "ERROR: benchmark accepts at most two -d devices"; exit 1 ;;
esac
```

## Step 3: Confirm PTO-ISA Pin

PTO-ISA is selected by the repo-root `pto_isa.pin`. Record the pin in the
benchmark notes so baseline and current runs can be compared against the same
source revision:

```bash
PTO_ISA_PIN=$(tr -d '[:space:]' < pto_isa.pin)
```

## Step 4: Prepare — Compute Absolute Paths

The Bash tool resets its working directory to the project root on every call. Relative paths like `cd worktree && ...` are fragile and easy to forget. **Compute absolute paths once, then use them everywhere.**

```bash
PROJECT_ROOT="$(pwd)"                    # e.g. /home/user/simpler
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKTREE_ABS="${PROJECT_ROOT}/tmp/worktree_baseline_${TIMESTAMP}"
PAYLOAD_SCRIPT="${PROJECT_ROOT}/tmp/benchmark_payload_${TIMESTAMP}.sh"
mkdir -p "${PROJECT_ROOT}/tmp"
```

Store `PROJECT_ROOT` and `WORKTREE_ABS` as shell variables in every Bash call that needs them (the Bash tool does not persist variables across calls). Use this pattern:

```bash
# Correct — self-contained, uses absolute path
WORKTREE_ABS="/home/user/simpler/tmp/worktree_baseline_20260331_102302"
"${WORKTREE_ABS}/tools/benchmark_rounds.sh" -d 2 ...
```

**Do NOT use `cd` + relative `./tools/...`** — this is the #1 source of silent errors (running the wrong workspace).

## Step 5: Run Benchmarks

### Single Mode

```bash
task-submit --timeout 7200 --max-time 7200 "${TASK_SUBMIT_DEVICE_ARGS[@]}" \
  --run "set -o pipefail && \
    .claude/skills/onboard-arch-precheck/check.sh '$PLATFORM' && \
    BENCH_DEVICE=\${TASK_DEVICE%%,*} && \
    ./tools/benchmark_rounds.sh $BENCH_ARGS -d \$BENCH_DEVICE -r '$RUNTIME' \
      2>&1 | tee 'tmp/benchmark_${TIMESTAMP}.txt'"
```

Use `--serial-orch-sched` to run each case once in the default overlapped mode
and once with `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1`, then emit serial-vs-parallel
Delta/Change tables.

### Compare Mode

Use a **git worktree** for the baseline so the current workspace is never disturbed.
Prepare the worktree and venv before taking an NPU lock. Steps 5b and 5c below
are the inner payload of one self-contained shell script; never execute either
onboard command outside that script's single `task-submit` allocation. Embed
the computed paths and selected platform/runtime as literal assignments at the
top of the payload, then map the allocated devices and run the architecture
check:

```bash
set -euo pipefail

PROJECT_ROOT="/absolute/path/to/current"
WORKTREE_ABS="/absolute/path/to/baseline"
PLATFORM="a2a3"
RUNTIME="tensormap_and_ringbuffer"
TIMESTAMP="YYYYmmdd_HHMMSS"
BENCH_ARGS=("-n" "100")  # all forwarded args except -d/--device and -r/--runtime
IFS=',' read -r -a LOCKED_DEVICES <<< "$TASK_DEVICE"
BASELINE_DEVICE="${LOCKED_DEVICES[0]}"
CURRENT_DEVICE="${LOCKED_DEVICES[1]:-${LOCKED_DEVICES[0]}}"
"$PROJECT_ROOT/.claude/skills/onboard-arch-precheck/check.sh" "$PLATFORM"
```

After composing the payload from the sequential or parallel block below,
submit that script exactly once:

```bash
task-submit --timeout 7200 --max-time 7200 "${TASK_SUBMIT_DEVICE_ARGS[@]}" \
  --run "bash '$PAYLOAD_SCRIPT'"
```

#### CRITICAL: Worktree needs its own build environment

The worktree is a fresh checkout at merge-base — it has **no pre-built runtime binaries** and no compiled nanobind extension. Two things must be built:

1. **Runtime `.so` binaries** (`build/lib/`) — loaded via ctypes by `bindings.py`
2. **Nanobind `_task_interface` extension** — compiled C++ Python bindings

Pure Python files under `simpler_setup/` (e.g. `scene_test.py`, `kernel_compiler.py`) are resolved via `sys.path` from the worktree when an editable install is active there, so they correctly come from the worktree. But `_task_interface.*.so` is installed into site-packages by `pip install -e .` and is **shared system-wide**. Without isolation, the worktree would use the main workspace's nanobind extension — which may have incompatible API changes.

**Solution: always create a venv in the worktree** (~26s overhead). This builds both the nanobind extension AND runtime binaries, fully isolating the baseline.

#### 5a. Create worktree, venv, and build

Inline the **absolute** worktree path (copy-paste the value, do not rely on shell variables persisting):

```bash
# Create worktree
git worktree add "$WORKTREE_ABS" "$MERGE_BASE" --quiet

# Create venv with system site-packages (for torch, numpy, etc.)
python3 -m venv "${WORKTREE_ABS}/.venv" --system-site-packages

# Install into venv — builds nanobind extension + runtime binaries
"${WORKTREE_ABS}/.venv/bin/pip" install -e "${WORKTREE_ABS}" -q 2>&1 | tail -3
```

This gives the worktree its own `_task_interface.*.so` in `.venv/lib/python3.*/site-packages/`, completely independent from the main workspace.

#### 5b. Run baseline

Activate the venv so `benchmark_rounds.sh` (which calls `python3`) picks up the worktree's nanobind extension and Python bindings:

```bash
# WORKTREE_ABS must be the literal absolute path.
(
  cd "$WORKTREE_ABS"
  source .venv/bin/activate
  pwd
  ./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$BASELINE_DEVICE" -r "$RUNTIME" \
    2>&1 | tee "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"
)
```

**Always print `pwd` after `cd` to verify you are in the correct directory.** If it does not print the worktree path, something went wrong — do not proceed.

#### 5c. Run current

```bash
cd "$PROJECT_ROOT"
./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$CURRENT_DEVICE" -r "$RUNTIME" \
  2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"
```

#### 5d. Cleanup

```bash
git worktree remove "$WORKTREE_ABS" --force
```

If `git worktree remove` fails (e.g., cwd was inside the deleted worktree), use:

```bash
git -C "$PROJECT_ROOT" worktree remove "$WORKTREE_ABS" --force
```

#### Parallel execution (two devices)

When two devices are available, run baseline and current **for the same runtime** in parallel on separate devices. The venv ensures the worktree has its own nanobind extension, so both workspaces are fully independent.

```bash
# For each runtime (serially):
for RUNTIME in "${RUNTIMES_TO_BENCH[@]}"; do
  # Baseline on device A (from worktree with venv), current on device B (from main) — parallel
  (cd "$WORKTREE_ABS" && source .venv/bin/activate && pwd && ./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$BASELINE_DEVICE" -r "$RUNTIME") &
  BASELINE_PID=$!
  (cd "$PROJECT_ROOT" && ./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$CURRENT_DEVICE" -r "$RUNTIME") &
  CURRENT_PID=$!
  PARALLEL_RC=0
  wait "$BASELINE_PID" || PARALLEL_RC=$?
  wait "$CURRENT_PID" || PARALLEL_RC=$?
  if [[ $PARALLEL_RC -ne 0 ]]; then
    exit "$PARALLEL_RC"
  fi
done
```

**Never launch the next runtime until the current one finishes on all devices.**

#### Sequential execution (one device)

```bash
# 1. Worktree + venv already created in step 5a

# 2. For each runtime (serially — one device, one process at a time):
#    Baseline first (from worktree with venv activated in a subshell)
(
  cd "$WORKTREE_ABS"
  source .venv/bin/activate
  pwd
  ./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$BASELINE_DEVICE" -r "$RUNTIME" \
    2>&1 | tee "${PROJECT_ROOT}/tmp/benchmark_baseline_${TIMESTAMP}_${RUNTIME}.txt"
)

#    Then current (from main workspace, no baseline venv)
cd "$PROJECT_ROOT"
./tools/benchmark_rounds.sh "${BENCH_ARGS[@]}" -d "$CURRENT_DEVICE" -r "$RUNTIME" \
  2>&1 | tee "tmp/benchmark_current_${TIMESTAMP}_${RUNTIME}.txt"

# 3. Cleanup
git -C "$PROJECT_ROOT" worktree remove "$WORKTREE_ABS" --force
```

## Step 6: Report Results

Parse every `Avg <Metric>:` field present in the runtime's output. Missing
runtime-inapplicable columns are not zero measurements and must not be printed.

| Metric | Source | What it captures |
| ------ | ------ | ---------------- |
| Host | `[STRACE]` `chip.run` span | steady_clock around dispatch (Python overhead included); rendered from markers by `strace_timing --rounds-table` |
| Device | `[STRACE]` `chip.run.runner_run.device_wall` span | full on-NPU AICPU run wall (`AicpuPhase::RunWall`, `max(end) − min(start)` across threads); on TMR this whole run + teardown is strictly larger than the windows below |
| Effective | TMR orch/sched markers' device-domain `ts`+`dur` | TMR only: `max(orch_end,sched_end) − min(orch_start,sched_start)` — the orch∪sched merged window |
| Orch | `[STRACE]` `…device_wall.orch` span (`--rounds-table`) | TMR only: device orchestrator (graph-build) window |
| Sched | `[STRACE]` `…device_wall.sched` span (`--rounds-table`) | TMR only: scheduler dispatch/execution window |

The scene test only *emits* `[STRACE]` markers to stderr; `benchmark_rounds.sh`
tees the run and renders the Host/Device/Effective/Orch/Sched table with
`python -m simpler_setup.tools.strace_timing <log> --rounds-table`. All columns
come from the markers (onboard and sim) — no CANN device log is read.

Use Effective as the TMR headline metric. HBG has no equivalent phase window:
report Host and Device independently and do not synthesize an overall score
from one of them.

For a per-stage breakdown of `Host`/`Device` (host `bind`/`runner_run`/`validate`
plus TMR's AICPU `preamble`/`so_load`/`graph_build`
(`config_validate`/`arena_wire`/`sm_reset` prep sub-phases)/`post_orch`
subdivision), parse the `[STRACE]` markers with
`simpler_setup/tools/strace_timing.py` (add `--tree` for the nested view) — see
[docs/dfx/host-trace.md](../../../docs/dfx/host-trace.md). Same `SIMPLER_HOST_STRACE`
gate, no extra flag (set `SIMPLER_DEVICE_STRACE_ENABLE=0` to drop only the device
`clk=dev` markers).

### Single Mode

Use the runtime's actual column set; the five-column example below is TMR. An
HBG table contains only Host and Device.

```text
Benchmark at: <short SHA>
Args: -d 4 -n 100

Example                          Host (us)   Device (us)   Effective (us)    Orch (us)   Sched (us)
-------------------------------  ---------   -----------   --------------   ----------   ----------
alternating_matmul_add           480000.0        9050.0         1235.5          820.3       1235.4
benchmark_bgemm                  370000.0        7100.0          892.1          650.2        892.0
...
```

### Compare Mode

Show a comparison table per metric (one row per metric per example), **grouped
by runtime**. For TMR, `Effective` is the headline metric used in the overall
summary and the other four are context. HBG has no combined headline metric:
show Host and Device as independent rows and summarize regressions separately.

```text
Merge-base: <short SHA>  →  HEAD: <short SHA> (+ uncommitted)
Args: -d 4 -n 100
Device: baseline=4, current=4  (or baseline=4, current=6)

### tensormap_and_ringbuffer

Example                      Base (us)   HEAD (us)   Delta (us)   Change (%)
---------------------------  ---------   ---------   ----------   ----------
alternating_matmul_add         1240.1      1235.5        -4.6       -0.37%
  (host)                     480000.0    470000.0    -10000.0       -2.08%
  (device)                     9000.0      8800.0       -200.0       -2.22%
  (orch)                        830.0       820.3        -9.7       -1.17%
  (sched)                      1240.0      1235.4        -4.6       -0.37%
benchmark_bgemm                 890.3       892.1        +1.8       +0.20%
  (host)                     370000.0    370500.0      +500.0       +0.14%
  (device)                     7100.0      7080.0       -20.0       -0.28%
  (orch)                        650.0       650.2        +0.2       +0.03%
  (sched)                       890.2       892.0        +1.8       +0.20%
...

Overall: X of Y examples improved, Z regressed   (based on Effective)
```

If baseline and current ran on **different devices**, add a note:

> Note: Baseline and current ran on different NPU devices (4 vs 6). Results within ±2% may reflect device-to-device variance rather than code changes. For definitive comparison, re-run on the same device with `/benchmark -d <single_device>`.

**Interpretation:**

| Change (%) | Assessment |
| ---------- | ---------- |
| < -2% | Notable improvement |
| -2% to +2% | Within noise margin |
| > +2% | Potential regression — flag for review |

If any example shows > 5% regression, highlight it explicitly.

## Error Handling

| Error | Action |
| ----- | ------ |
| `task-submit` cannot allocate the requested device count | Report the queue/allocation error; do not run outside the lock |
| Benchmark script fails | Report which examples failed; continue with remaining |
| No timing data | Warn: "No timing markers — ensure `SIMPLER_HOST_STRACE` is enabled" |
| All examples fail | Check: did you run `pip install -e .` in the worktree venv? |
| Worktree creation fails | Fall back to stash/checkout approach or report error |
| `Pre-built runtime binaries not found` | The venv `pip install -e .` should have built these; re-run it |
| `ModuleNotFoundError: _task_interface` | Venv not activated; add `source .venv/bin/activate &&` before the command |

## Checklist

- [ ] Mode detected (single vs compare)
- [ ] Architecture precheck passed and one `task-submit` allocation owns the run
- [ ] PTO-ISA pinned to CI commit
- [ ] `PROJECT_ROOT` and `WORKTREE_ABS` absolute paths computed
- [ ] (Compare mode) Worktree created, venv built with `pip install -e .`
- [ ] (Compare mode) Baseline completed — venv activated, `pwd` confirmed worktree path before running
- [ ] Current completed in main workspace
- [ ] Worktree cleaned up (compare mode)
- [ ] Results table uses TMR's five columns or HBG's Host / Device columns
- [ ] (Compare mode) Device difference noted if applicable
- [ ] (Compare mode) Regressions > 2% flagged
