# A5 AICPU Core-Selection Strategy for All Scenarios

## 1. Objective and scope

This document defines AICPU core-selection logic for all currently known A5
topology scenarios: default FG, one failed cluster (PG1), and two failed
clusters (PG2). Scheduler SMT availability is a topology property of these
scenarios, not a separate scenario. A5 supports two through five active AICPU
threads: one Orchestrator and one through four Schedulers. `1O+4S` remains the
automatic default; Sections 4-6 define the detailed rules.

This document is the design contract for A5 AICPU core selection. The runtime
implements device-side occupancy query, scenario classification
(`kFg` / `kPg1` / `kPg2` / `kUnknown`), the Section 4 policies for every
supported active count (`1O+1S` through `1O+4S`), and the unknown-topology
fallback. Section 4 also lists placement variants reserved for later hardware
performance testing.

> **Architecture boundary:** AICPU PG is designed and implemented only for A5.
> A2/A3 AICPU does not enter PG. It continues to use the existing non-PG
> selection path and does not call the PG classification or policy branches in
> this document.

## 2. Terms and boundaries

| Term | Meaning |
| ---- | ------- |
| O | Orchestrator |
| S | Scheduler for AICore work |
| AICPU physical CPU | May expose one or two SMT logical CPUs |
| Dedicated O physical CPU | The SMT sibling of O is not assigned to another O/S role |
| S close to O | Prefer the same cluster, then the same die, and cross dies last |
| Dedicated S physical CPU | One Scheduler uses one physical CPU without sharing an SMT sibling with another O/S role |

## 3. Topology input and scenario detection

### 3.1 Known A5 physical structure

- One chip contains two dies.
- Each die contains two AICPU clusters.
- Each AICPU cluster contains two physical CPUs.
- The chip contains four AICPU clusters and eight physical CPUs in total.
- PG failure granularity is a complete cluster, removing two physical CPUs.
- PG1 may lose any one cluster.
- PG2 loses one cluster from each die.

### 3.2 Runtime input

The runtime does not detect failed cores itself and does not infer topology
from contiguous CPU IDs. The driver isolates failed clusters and CPUs. The
runtime queries DSTI, DSMI, HAL, or equivalent driver topology interfaces for:

- the O/S-schedulable logical CPU set;
- `cpu_id`, `phy_cpu_id`, and `hyperthread_id`;
- `cluster_id`;
- `die_id`.

The authoritative per-logical-CPU contract is `AicpuLogicalCpu` in
`src/a5/platform/onboard/host/aicpu_topology_probe.h`.
`hyperthread_id == 0` identifies the primary thread. Logical CPUs with the same
`phy_cpu_id` are SMT siblings. The `os_schedulable_cpus` set contains only CPUs
present in the driver `OCCUPY` bitmap; Control, Data, and system-reserved CPUs
are not candidates for O/S placement.

`scheduler_smt_enabled` records whether the driver-reported schedulable pool
contains an SMT sibling pair. It is an observed topology property and not a
temporary Runtime software switch. In particular, `kFg` does not imply that
Scheduler SMT is disabled.

The runtime may identify the scenario from any of the following sources, in
priority order:

1. Prefer the scenario-type query interface in Section 3.3 and directly use
   `kFg`, `kPg1`, or `kPg2`.
2. If the interface does not return a type, use the surviving-cluster count
   and cross-die distribution.
3. Record schedulable SMT sibling state independently in
   `scheduler_smt_enabled`; it does not change the scenario type.

When cluster/die metadata is present (the common host path), classification
**does not** require logical AICPU counts of 16/12/8. Those counts describe one
BIOS mapping; other valid layouts (for example a 9-logical FG SKU) still map by
surviving clusters:

| Scenario | Surviving clusters | Distribution | SMT property |
| -------- | -----------------: | ------------ | ------------ |
| FG | 4 | Both dies complete (2+2 clusters) | Recorded independently; does not affect `kFg` classification |
| PG1 | 3 | One die 2 clusters, the other 1 | Recorded; default 1O+4S uses SMT pairs |
| PG2 | 2 | One surviving cluster on each die | Recorded; SMT sharing is allowed for 1O+4S |

Return `kUnknown` only when the cluster/die shape cannot identify a scenario
uniquely. A complete four-cluster, two-die layout is FG regardless of its
current schedulable SMT state.

Whether Data 1 and Data 2 are exposed as two independent logical CPUs or one
merged logical CPU remains a BIOS/driver question. Core selection consumes only
the final schedulable set returned by the driver.

### 3.3 Scenario-type query interface

The runtime combines these low-level signals:

- `tools/cann-examples/query/query.cpp` queries CPU topology through
  `halGetDeviceInfoByBuff(SYSTEM, CPU_TOPO)`, with
  `dsmi_get_device_info(SOC_INFO, CPU_TOPO)` as a fallback. It prints
  `cpu_id`, `phy_cpu_id`, and `hyperthread_id`.
- `tools/cann-examples/aicpu-device-query/` queries device-side `OCCUPY`,
  `PF_OCCUPY`, and `OS_SCHED` bitmaps.

The runtime already uses the same signals through `probe_aicpu_topology()` and
represents each schedulable logical CPU with the following authoritative type:

```cpp
struct AicpuLogicalCpu {
    int32_t cpu_id;
    int32_t phy_cpu_id;
    int32_t hyperthread_id;
    int32_t cluster_id;
    int32_t die_id;
};
```

`compute_allowed_cpus()` remains the generic topology-aware packer for packaged
entries that explicitly request the generic policy. Known scenarios use
`classify_aicpu_scenario()` plus `compute_scenario_allowed_cpus()` for every
supported active count from two through five.

The production AICPU SO exposes a one-thread topology-query entry before the
normal affinity-gated launch. It returns device-side `OCCUPY`, `PF_OCCUPY`, and
`OS_SCHED`; Host CPU_TOPO supplies physical, SMT, cluster, and die metadata.

```cpp
enum class AicpuScenarioType {
    kNotApplicable,  // A2/A3
    kFg,             // A5, 4 clusters; SMT state is recorded separately
    kPg1,            // A5, 3 surviving clusters
    kPg2,            // A5, one surviving cluster on each die
    kUnknown,
};

struct AicpuTopology {
    AicpuTopologySource source;  // driver, verified JSON, or OCCUPY-only
    AicpuScenarioType scenario_type;
    bool scheduler_smt_enabled;
    uint32_t logical_cpu_count;
    std::vector<int32_t> surviving_cluster_ids;
    std::vector<AicpuLogicalCpu> os_schedulable_cpus;
};

struct AicpuLaunchPlan {
    int32_t effective_active_count;
    int32_t stable_reachable_count;
    int32_t launch_count;
    std::vector<int32_t> allowed_cpus;  // [S..., O]
};

AicpuTopology QueryAicpuTopology(uint32_t device_id);
```

`tools/cann-examples/aicpu-device-query/host/build/query_device_hal <device_id>
--json` combines the device masks with Host CPU_TOPO and emits the
classification and launch decision for tests and DFX:

```json
{
  "architecture": "a5",
  "scenario_type": "PG1",
  "scheduler_smt_enabled": true,
  "logical_cpu_count": 12,
  "surviving_clusters": [0, 1, 2],
  "os_schedulable_cpus": [
    {
      "cpu_id": 6,
      "phy_cpu_id": 3,
      "hyperthread_id": 0,
      "cluster_id": 1,
      "die_id": 0
    }
  ],
  "launch_plan": {
    "effective_active_count": 5,
    "stable_reachable_count": 6,
    "launch_count": 6,
    "allowed_cpus": [6, 7, 8, 9, 10]
  }
}
```

The JSON array serializes every `AicpuLogicalCpu` field rather than maintaining
a second ID-only representation. Device-mask validity, topology source,
selection policy, and requested/effective/launch counts are also serialized so
tests and DFX can reproduce every placement decision. The interface combines
these signals using the priority in Section 3.2 and returns `kUnknown` when
classification is ambiguous.

## 4. Core-selection strategies for all scenarios

Topology policies for `1O+NS`, where `N` is one through four:

| Scenario | `1O+NS` policy |
| -------- | -------------- |
| FG | O uses the primary thread of the last suitable physical CPU. Select N primary threads on dedicated physical CPUs, preferring O's cluster, then O's die, then the other die. The scenario remains FG whether or not schedulable SMT siblings are present |
| PG1 | O uses the primary thread of the last suitable physical CPU on the die with the most schedulable threads and leaves its sibling idle. Among the remaining CPUs, select one primary thread per physical CPU first, ordered by O's cluster, O's die, then the other die. Consume SMT siblings only when distinct physical CPUs cannot satisfy N. If the remaining logical CPUs are insufficient, return insufficient capacity |
| PG2 | Order CPUs by die, cluster, physical CPU, hyperthread, and logical CPU; use the last eligible logical CPU as O without requiring an idle sibling. Order the remaining candidates by proximity to O and take the first N. An S may use O's SMT sibling, and two S roles may share another physical CPU |

### 4.1 Concrete Compute topologies and selections

The following fixtures are the concrete FG, PG1, and PG2 layouts shown in the
design diagrams and covered by the unit tests. `Tn` means logical CPU
`cpu_id == n`; two rows with the same physical CPU are SMT siblings. These
tables show only the Compute CPUs in the driver `OCCUPY` bitmap, because only
that pool can be selected for S/O roles. Scenario classification still uses
the complete CPU_TOPO, including non-Compute CPUs, to determine the surviving
cluster shape.

The affinity result is always written as `[S..., O]`.

#### FG concrete layout

| Logical CPU | Physical CPU | HT | Cluster | Die |
| ----------- | ------------ | -: | ------: | --: |
| T4 | CPU2 | 0 | 1 | 0 |
| T6 | CPU3 | 0 | 1 | 0 |
| T8 | CPU4 | 0 | 2 | 1 |
| T10 | CPU5 | 0 | 2 | 1 |
| T12 | CPU6 | 0 | 3 | 1 |
| T14 | CPU7 | 0 | 3 | 1 |

All six schedulable Compute CPUs are primary threads on distinct physical
CPUs. O is T14. S starts with T12 in O's cluster, then T8 and T10 on O's die,
and finally T4 on the other die.

| Active threads | Exact selection |
| -------------: | --------------- |
| 2 (`1S+1O`) | `[S:T12, O:T14]` |
| 3 (`2S+1O`) | `[S:T12, S:T8, O:T14]` |
| 4 (`3S+1O`) | `[S:T12, S:T8, S:T10, O:T14]` |
| 5 (`4S+1O`) | `[S:T12, S:T8, S:T10, S:T4, O:T14]` |

#### PG1 concrete layout

| Logical CPU | Physical CPU | HT | Cluster | Die |
| ----------- | ------------ | -: | ------: | --: |
| T4 | CPU2 | 0 | 1 | 0 |
| T6 | CPU3 | 0 | 1 | 0 |
| T7 | CPU3 | 1 | 1 | 0 |
| T8 | CPU4 | 0 | 2 | 1 |
| T9 | CPU4 | 1 | 2 | 1 |
| T10 | CPU5 | 0 | 2 | 1 |
| T11 | CPU5 | 1 | 2 | 1 |

O is T10, while its SMT sibling T11 is deliberately left idle. The policy
first selects the primary Scheduler threads T8, T4, and T6 on distinct
physical CPUs. T9 is selected only for `4S+1O`, after those primary threads
are exhausted.

| Active threads | Exact selection |
| -------------: | --------------- |
| 2 (`1S+1O`) | `[S:T8, O:T10]` |
| 3 (`2S+1O`) | `[S:T8, S:T4, O:T10]` |
| 4 (`3S+1O`) | `[S:T8, S:T4, S:T6, O:T10]` |
| 5 (`4S+1O`) | `[S:T8, S:T4, S:T6, S:T9, O:T10]` |

In the five-thread result, T8/T9 are the only S/S SMT pair; T4 and T6 are
Schedulers on dedicated physical CPUs; T10 is O. T7 is an unused sibling of
T6, and T11 is O's unused sibling. A fully non-SMT `4S+1O` mapping is
impossible because only four Compute physical CPUs remain and one is reserved
for O. Implementations derive the mirrored result from topology metadata.

#### PG2 concrete layout

| Logical CPU | Physical CPU | HT | Cluster | Die |
| ----------- | ------------ | -: | ------: | --: |
| T3 | CPU1 | 1 | 0 | 0 |
| T4 | CPU2 | 0 | 2 | 1 |
| T5 | CPU2 | 1 | 2 | 1 |
| T6 | CPU3 | 0 | 2 | 1 |
| T7 | CPU3 | 1 | 2 | 1 |

PG2 chooses the last eligible logical CPU, T7, as O. Scheduler selection first
fills T4, T5, and T6 in O's cluster and then uses T3 on the other die.

| Active threads | Exact selection |
| -------------: | --------------- |
| 2 (`1S+1O`) | `[S:T4, O:T7]` |
| 3 (`2S+1O`) | `[S:T4, S:T5, O:T7]` |
| 4 (`3S+1O`) | `[S:T4, S:T5, S:T6, O:T7]` |
| 5 (`4S+1O`) | `[S:T4, S:T5, S:T6, S:T3, O:T7]` |

T4/T5 are an S/S SMT pair on CPU2. T6 and O:T7 are an S/O SMT pair on CPU3.
T3 is the Scheduler selected from the surviving cluster on the other die.

Strategies for later exploration:

| Scenario | Strategies to explore |
| -------- | --------------------- |
| FG | Compare the implemented `1O+1S` through `1O+4S` counts, single-cluster packing, multi-cluster spreading on one die, and cross-die placement. When schedulable SMT siblings are present, also compare compact SMT and hybrid SMT variants |
| PG1 | Compare the minimum-SMT-sharing default with compact SMT only on a machine that exposes the required PG1 topology; this machine cannot perform that hardware comparison |
| PG2 | Compare the implemented shared-physical-CPU counts with dedicated-O variants and alternative Scheduler distribution between the two surviving clusters |

## 5. Global core-selection principles

The scenario-specific rules in Section 4 take precedence over the global
placement preferences in this section. PG1 delays Scheduler SMT sharing until
distinct physical CPUs are exhausted; PG2 explicitly permits physical sharing.

Orchestrator:

- The affinity array is always `[S0 ... SN-1, O]`; its last entry records O's
  logical CPU.
- A "suitable physical CPU" is a physical CPU in the driver-provided O/S pool
  with at least one available primary logical thread. The driver has already
  excluded Control, Data, and system-reserved threads.
- Sort candidates deterministically by die, cluster, and physical CPU. The
  final candidate satisfying the condition above is the "last suitable
  physical CPU."
- Except in PG2, O uses `hyperthread_id == 0` on the last suitable physical CPU
  and leaves its SMT sibling idle.
- If primary-thread metadata is unavailable, use the lower logical CPU ID only
  after confirming that the active BIOS mapping defines it as the primary
  thread.
- In PG2, O uses the last eligible logical CPU. O does not require an isolated
  physical CPU, and its SMT sibling remains eligible for Scheduler placement.

Scheduler placement and affinity:

- Prefer O's cluster, then O's die, and cross dies last.
- Prefer a dedicated physical CPU for each Scheduler except where the scenario
  policy permits SMT sharing. PG1 uses a sibling only after available primary
  threads on other physical CPUs; PG2 may assign a Scheduler to O's sibling and may assign two
  Schedulers to both threads of another physical CPU when necessary.
- Alternative multi-cluster spreading and SMT density are exploration items.

Policy-specific capacity requirements:

- FG requires at least N+1 suitable physical CPUs: O owns one and each
  Scheduler uses another primary thread.
- PG1 requires one suitable physical CPU for O and at least N remaining
  schedulable logical CPUs. It prefers distinct physical CPUs but does not
  require complete SMT pairs.
- PG2 requires at least N+1 schedulable logical CPUs across the identified PG2
  topology; physical-CPU sharing is permitted.

Determinism and state consistency:

- When several locations are equivalent, use a fixed die, cluster, physical
  CPU, and logical CPU ordering.
- Actual launch count, active O/S count, Runtime state, DFX state, and the
  affinity array must match.
- Use only schedulable logical CPUs returned by the driver.
- Every selected `cpu_id` must be unique and must belong to
  `os_schedulable_cpus`.
- All primary threads, SMT pairs, physical CPUs, clusters, and dies required by
  the selected scenario must exist in the probe result.
- If any common or scenario-specific requirement cannot be satisfied, return
  insufficient capacity without emitting a partial affinity array.

## 6. Common selection flow

1. Query device-side `OCCUPY` / `PF_OCCUPY` / `OS_SCHED`, then call
   `probe_aicpu_topology()` to merge Host CPU_TOPO metadata, build
   `os_schedulable_cpus`, and classify the scenario. A2/A3 stays on the
   existing non-PG path and does not enter this flow.
2. Group logical CPUs by `phy_cpu_id`, and retain their `cluster_id`, `die_id`,
   and `hyperthread_id`.
3. Select a scenario policy from `kFg`, `kPg1`, or `kPg2`, or take the
   `kUnknown` fallback in step 8.
4. Resolve `aicpu_thread_num=0` to the automatic default of five. Reject manual
   values outside 2-5. For every known scenario and active count, validate the
   common requirements and the policy-specific capacity in Section 5.
5. Place O and S using the selected scenario policy in Section 4. The scenario
   policy overrides the global placement preferences where Section 4 states an
   exception.
6. Verify that all selected IDs are unique and schedulable, then emit
   `[S0 ... SN-1, O]` atomically. Synchronize the actual launch count, Runtime,
   DFX, and affinity state with that complete array.
7. If validation or placement fails, return insufficient capacity without
   emitting a partial affinity array.
8. For `kUnknown`, sort valid metadata by die, cluster, physical CPU,
   hyperthread, and logical CPU. OCCUPY-only inputs have no physical metadata,
   so their order reduces to logical CPU ID. A manual request selects exactly
   that many CPUs and fails on insufficient capacity; automatic mode selects up
   to five and may shrink to the available count. At least two CPUs are
   required. The last selection is O and the preceding selections are S. Emit
   a Host warning when this fallback is reached because live CPU_TOPO was
   unavailable. Driver-provided but unclassified topology remains observable
   through the `kUnknown` scenario without being reported as a query failure.

Before the generic `kUnknown` fallback, a packaged JSON entry may preserve a
verified CPU_TOPO-less signature. Its SoC and every constraint declared by the
entry must match. The 9599 entry is host-independent but requires exact
device-side `OCCUPY=0x1f8`; the 9579 entry also constrains host architecture.
Entries marked `generic` use `compute_allowed_cpus()` and honor an explicit
active count.

The fallback's active count is the number selected, while the AICPU launch
count remains the full device-side `OCCUPY` population so the filter gate sees
one representative on every schedulable CPU. This launch count may exceed the
active limit of five and is independently capped at 14. Unknown automatic mode
may shrink the active count; a manual count is never silently changed.
