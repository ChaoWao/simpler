# A5 AICPU Affinity Preflight — Design

Authoritative AICPU core selection for Ascend950 (a5) is **not** done on the
production hot path. A separate preflight tool measures the user AICPU pool,
ranks orchestrator and scheduler CPUs, and writes a per-device plan. The
onboard runtime only **reads** that plan and applies the existing affinity
gate + contiguous AICore ownership.

Related hardware/OCCUPY background: [hardware.md](hardware.md).
Tooling entry points: `python -m simpler_setup.tools.rtt_die_preflight`,
`simpler_setup/tools/aicpu_device_query/`.

## Problem

A5 has two dies of AICores and a small user AICPU pool (typically six
OCCUPY bits). Production wants:

- one orchestrator (O) and up to four schedulers (S0–S3)
- **logical** contract users care about: S0/S1 own die0, S2/S3 own die1
- launch count still equal to `popcount(OCCUPY)` so the affinity gate sees
  the full pool

Earlier approaches mixed concerns:

1. **FG/PG / `build_aicpu_launch_plan`** — topology-driven packing inside
   `DeviceRunner` / `aicpu_topology_probe` (fast, but not latency-aware).
2. **Per-run RTT / die scoring on the hot path** — too heavy and couples
   measurement SO into `host_runtime`.
3. **Thick JSON “plan readers” in `src`** — duplicated schema and ranking
   policy next to the tool that already owns them.

The result was either wrong placement relative to cross-die cost, or too
much strategy code living in production.

## Design principles

1. **Tool owns measured selection.** Enumeration, handshake orch election, COND
   die scoring, physical picks, logical packing, and tool-authored fallback
   plans live only in the preflight toolchain. Runtime never re-ranks; on a
   missing `.cpus` file it may only apply OCCUPY-bit contiguous `[S…, O]`.
2. **Python generates; C++ only reads.** `ChipWorker.init` (a5 onboard)
   ensures the per-device `.cpus` side file exists (runs `--probe` on miss,
   with start/skip/done/fail printed to the console). `DeviceRunner` never
   `system()`-spawns Python.
3. **Plan is authoritative when present.** Schema validation and ranking
   live in Python. Runtime does not re-rank a loaded CPU list.
4. **Miss → OCCUPY contiguous.** If the side file is still absent at
   prepare, runtime builds `[S…, O]` from OCCUPY bit order (active in
   `[2, 5]`). No FG/PG / scenario selection on the affinity path.
5. **Users see logical sched↔die only.** Physical `cpu_id` values are
   diagnostic. AICore ownership stays contiguous on logical S0–S3.
6. **Per-device independence.** Plans are keyed by `(soc_name, device_id)`.
   L3 multi-card binds each card separately (one `.cpus` file per device).

## Layering

```text
┌─────────────────────────────────────────────────────────┐
│  ChipWorker.init (Python, a5 onboard)                   │
│  .cpus missing → rtt_die_preflight --probe (printed)    │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│  Preflight toolchain (selection authority)              │
│  aicpu_device_query SO/host  +  rtt_die_preflight.py     │
│  enum → handshake orch → COND die → pack → write plan   │
└────────────────────────────┬────────────────────────────┘
                             │  aicpu_affinity_plan.json
                             │  aicpu_affinity_plan.<id>.cpus
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Production DeviceRunner (thin consumer)                │
│  OCCUPY → launch_count                                  │
│  read .cpus → AffinityPlanCache memo → runtime fields   │
│  miss → OCCUPY contiguous [S…, O] (no Python spawn)     │
└────────────────────────────┬────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Device execution                                       │
│  affinity gate filter  +  contiguous AICore ownership   │
│  (S0/S1 die0, S2/S3 die1 when active==5)                │
└─────────────────────────────────────────────────────────┘
```

`aicpu_topology_probe` remains for **tools / UT / `--json` diagnostics** and
for OCCUPY/soc helpers. It is **not** the production `ALLOWED_CPUS` path
when a plan is present.

pypto L2/L3 and simpler ST all attach via `simpler.task_interface.ChipWorker.init`,
so they share the same Python preflight hook.

## Measurement and packing (tool only)

Public CLI:

```bash
python -m simpler_setup.tools.rtt_die_preflight --device N --probe
```

Backend: `simpler_setup/tools/aicpu_device_query` via `--rtt-json` (same
dispatcher bootstrap as production AICPU upload).

Phases:

| Phase | What | Output |
| ----- | ---- | ------ |
| 1 | Serial enum of user pool | `user_pool_cpus` |
| 2 | Multi-thread atomic-flag pairwise handshake (1000 iters) | elect orch = min avg latency to peers |
| 3 | Non-orch COND LDR sums per die (100 samples/core), serialized turns | `die0_sum_ticks` / `die1_sum_ticks` |
| 4 | Phys picks in die order `{0,1,1,0}`, pack to logical `[P0,P3,P1,P2,orch]` | `allowed_cpus = [S0,S1,S2,S3,O]` |

Packing single source: `pack_schedulers_from_die_scores` in
`rtt_die_preflight.py`. There is **no** duplicate pack helper in
production headers.

### Fallbacks

| Condition | Plan source | Behavior |
| --------- | ----------- | -------- |
| `pool < 5` | `pool-too-small-contiguous` | tool skips handshake/COND; contiguous OCCUPY order; ≥1S+1O |
| probe failed but topo `--json` usable | `probe-failed-contiguous` | same contiguous write |
| offline | `--allowed-cpus` / `--fallback-occupy` | manual / occupy list |
| `.cpus` still missing at prepare | `runtime-occupy-contiguous` | DeviceRunner OCCUPY bit-order contiguous; does not write side file |

## Plan artifacts

### Authoritative JSON

Path: `build/config/aicpu_affinity_plan.json` (override with
`SIMPLER_AICPU_AFFINITY_PLAN`). Schema **v3**, bucketed by
`socs.<soc>.devices.<device_id>`.

Contains `allowed_cpus`, `plan_source`, `schedulers[]`, scores, etc.
Python validates; humans and tools inspect this file. Default under
`build/` — not committed to the repo.

### Runtime companion (line-oriented)

Sibling: `aicpu_affinity_plan.<device_id>.cpus`

```text
soc=Ascend950PR_9599
source=atomic-flag-orch+cond-die-v1
pool_too_small=0
cpus=3,4,5,6,8
```

Why a second file: production must not embed a JSON schema parser. The
companion is machine-trivial (`key=value` + CSV). Loader:
`src/a5/platform/onboard/host/affinity_allowed_file.*`
(`load_affinity_cpus_side_file`).

## Runtime minimal surface

On `ChipWorker.init` (a5 onboard, before C++ attach):

1. If `aicpu_affinity_plan.<device_id>.cpus` exists → print skip.
2. Else print start, run `python -m …rtt_die_preflight --probe`, print done/fail.
   Failure does not abort init.

On prepare:

1. Device-side OCCUPY query → `aicpu_launch_count = popcount(OCCUPY)`
   (gate must launch the full user pool).
2. `aclrtGetSocName` + load `.cpus` for this `device_id` into
   `DeviceRunner::AffinityPlanCache` (in-runner memo for one
   `(soc, device_id)` key).
3. On miss: OCCUPY contiguous `[S…, O]` (source
   `runtime-occupy-contiguous`); never spawn Python.
4. Write `runtime.aicpu_allowed_cpus` / `aicpu_thread_num`.
   Logical scheduler index then owns contiguous AICore ranges
   (`[t*N/A, (t+1)*N/A)` in `scheduler_cold_path.cpp`).

`clear_aicpu_caches()` drops OCCUPY + `AffinityPlanCache` on recovery,
reset, and finalize.

Device side:

- `platform_aicpu_affinity_gate_filter` — launched threads outside
  `ALLOWED_CPUS` become sinks; survivors keep order as exec roles
  (S0..Sk, O).
- Contiguous AICore ownership uses logical sched index only.
  With four schedulers: S0/S1→first half (die0), S2/S3→second half (die1).

## Triggers and multi-device

| Event | Behavior |
| ----- | -------- |
| `.cpus` present at ChipWorker.init | print skip; no probe |
| `.cpus` missing at ChipWorker.init | Python `--probe` once; start/done/fail printed |
| Plan hit at prepare | AffinityPlanCache; no DSMI CPU_TOPO selection |
| Plan miss at prepare | OCCUPY contiguous fallback |
| L3 N cards | one `.cpus` per `device_id`; each chip child runs ChipWorker.init |

## Non-goals

- Do not move device-SO measurement back into `host_runtime`.
- Do not let C++ spawn Python for affinity.
- Do not change a2a3.
- Do not delete `aicpu_topology_probe` (tools/UT still use it).
- Do not require operators to understand physical pick order; only the
  logical S0–S3 ↔ die contract is user-facing.

## File map

| Role | Path |
| ---- | ---- |
| Ranking / write / fallback | `simpler_setup/tools/rtt_die_preflight.py` |
| Device enum / handshake / COND | `simpler_setup/tools/aicpu_device_query/` |
| Python init hook | `python/simpler/task_interface.py` (`_ensure_a5_affinity_cpus`) |
| Side-file reader | `src/a5/platform/onboard/host/affinity_allowed_file.*` |
| Thin consumer | `src/a5/platform/onboard/host/device_runner.cpp` |
| Affinity gate | `src/common/platform/onboard/aicpu/platform_aicpu_affinity.cpp` |
| Tool/UT topo (not prod ALLOWED_CPUS) | `src/a5/platform/onboard/host/aicpu_topology_probe.*` |

## Acceptance sketch

- a5 `ChipWorker.init` without `.cpus` → probe prints start/done and writes
  artifacts; subsequent prepares only read disk.
- Plan present → no Python spawn from C++, no FG/PG for `ALLOWED_CPUS`.
- Plan absent at prepare → OCCUPY contiguous; prepare still succeeds when
  OCCUPY has ≥2 CPUs.
- Python UT covers orch/pack/fallback; AICore contiguous ownership is
  inlined in scheduler cold path (`[t*N/A, (t+1)*N/A)`).
