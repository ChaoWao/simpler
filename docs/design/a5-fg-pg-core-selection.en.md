# A5 AICPU Core-Selection Strategy for All Scenarios

## 1. Objective and scope

This document defines AICPU core-selection logic for all currently known A5
topology scenarios: default FG, FG+SMT, one failed cluster (PG1), and two
failed clusters (PG2).

FG and FG+SMT must be identified separately, but they use the same default
policy. FG+SMT only adds SMT layouts for later exploration. Every scenario uses
`1O+4S` as its default basis; Sections 4-6 define the detailed rules.

This document defines code design only. The repository already implements the
generic `probe_aicpu_topology()` topology probe and `compute_allowed_cpus()`
packing algorithm. It does not yet implement the PG scenario classification or
the scenario-specific policies proposed here. Section 4 lists other Scheduler
counts and layouts for later hardware performance testing.

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

FG+SMT is selected through driver configuration. After changing that
configuration, the tool queries the effective state. `scheduler_smt_enabled`
records this queried state; it is not a temporary Runtime software switch.

The runtime may identify the scenario from any of the following sources, in
priority order:

1. Prefer the scenario-type query interface in Section 3.3 and directly use
   `kFg`, `kFgSmt`, `kPg1`, or `kPg2`.
2. If the interface does not return a type, use the surviving-cluster count,
   cross-die distribution, and schedulable SMT sibling state.
3. If cluster information is unavailable, match the driver-reported logical
   AICPU count against the current A5 BIOS topology configuration and use SMT
   sibling state to distinguish FG from FG+SMT.

| Scenario | Logical AICPU count | Surviving clusters | SMT state | Distribution |
| -------- | ------------------: | -----------------: | --------- | ------------ |
| FG | 16 (T0-T15) | 4 | Scheduler SMT siblings disabled | Both dies are complete |
| FG+SMT | 16 (T0-T15) | 4 | Scheduler SMT siblings available | Both dies are complete |
| PG1 | 12 (T0-T11) | 3 | As reported by the driver | One complete die and one surviving cluster on the other die |
| PG2 | 8 (T0-T7) | 2 | As reported by the driver | One surviving cluster on each die |

The logical AICPU count is the total number of logical CPUs exposed by
BIOS/driver, not the number of active Control, Data, and Compute roles. For
example, PG1 exposes 12 logical AICPUs while the diagram contains 10 active
roles. Count-based detection is valid only after confirming that the device
uses the A5 BIOS mapping above. Return `kUnknown` if the count does not match
the mapping or cannot identify a scenario uniquely. FG and FG+SMT have the
same logical-AICPU and cluster counts, so the runtime must distinguish them
using driver-reported schedulable SMT sibling information rather than counts
alone.

Whether Data 1 and Data 2 are exposed as two independent logical CPUs or one
merged logical CPU remains a BIOS/driver question. Core selection consumes only
the final schedulable set returned by the driver.

### 3.3 Scenario-type query interface

Two existing tools provide reusable low-level signals:

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

The existing `compute_allowed_cpus()` applies a generic topology-aware packing
algorithm to this probe output. Neither it nor the query tools currently provide
the PG scenario classifier or the scenario-specific policies proposed below.

> **TODO (not implemented):** Implement `QueryAicpuTopology()` under
> `tools/cann-examples/aicpu-topology-query/`. The interface and output
> format below are proposals and cannot be called by the current version.

```cpp
enum class AicpuScenarioType {
    kNotApplicable,  // A2/A3
    kFg,             // A5, 4 clusters, Scheduler SMT disabled
    kFgSmt,          // A5, 4 clusters, Scheduler SMT enabled
    kPg1,            // A5, 3 surviving clusters
    kPg2,            // A5, one surviving cluster on each die
    kUnknown,
};

struct AicpuTopology {
    AicpuScenarioType scenario_type;
    bool scheduler_smt_enabled;
    uint32_t logical_cpu_count;
    std::vector<int32_t> surviving_cluster_ids;
    std::vector<AicpuLogicalCpu> os_schedulable_cpus;
};

AicpuTopology QueryAicpuTopology(uint32_t device_id);
```

The tool should also provide JSON output for tests and DFX:

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
  ]
}
```

The JSON array serializes every `AicpuLogicalCpu` field rather than maintaining
a second ID-only representation. This lets tests and DFX reproduce every
placement decision. The interface combines these signals using the priority in
Section 3.2 and returns `kUnknown` when classification is ambiguous.

## 4. Core-selection strategies for all scenarios

Default policies:

| Scenario | Default `1O+4S` policy |
| -------- | ---------------------- |
| FG | O owns the last suitable physical CPU; every Scheduler uses a dedicated physical CPU close to O; prefer one die |
| FG+SMT | Same as FG; O uses the first logical thread of the last suitable physical CPU and leaves its sibling idle; Schedulers do not use the second SMT thread by default |
| PG1 | Use the fixed SMT compensation layout without searching other layouts. O uses the primary thread of the last suitable physical CPU and leaves its sibling idle. From the remaining candidates, prefer the same cluster as O, then the same die, then the other die, and select two physical CPUs that each provide a complete schedulable SMT pair; the four threads of those pairs become 4S. If this layout is unavailable, return insufficient capacity. The same topology-relative rule handles every failed-cluster position without relying on logical CPU numbering |
| PG2 | The scenario policy overrides the global isolation preferences. Order CPUs by die, cluster, physical CPU, and logical CPU; use the last eligible logical CPU as O without requiring an idle sibling. An S may use O's SMT sibling, and two S roles may use both threads of another physical CPU when necessary. Place Schedulers in O's cluster first and the remaining Schedulers on the other die |

Under the current BIOS mapping, `[T6, T7, T8, T9, T10]`, with T6-T9 as
Schedulers and T10 as O, is a non-normative example for one right-side PG1
cluster failure. Implementations must derive the equivalent layout from
`cluster_id`, `die_id`, `phy_cpu_id`, and `hyperthread_id` for either failure
orientation.

Strategies for later exploration:

| Scenario | Strategies to explore |
| -------- | --------------------- |
| FG | Compare `1O+2S`, `1O+3S`, and `1O+4S`, plus single-cluster packing, multi-cluster spreading on one die, and cross-die placement |
| FG+SMT | In addition to the FG options, compare one or two Scheduler logical threads per physical CPU, compact SMT, hybrid SMT, and multi-cluster spreading |
| PG1 | Validate only the performance and stability of the fixed SMT compensation layout; do not explore other SMT densities or placements |
| PG2 | Compare default shared-physical-CPU `1O+4S`, dedicated-O `1O+3S`/`1O+2S`, and Scheduler distribution between the two surviving clusters |

## 5. Global core-selection principles

The scenario-specific rules in Section 4 take precedence over the global
placement preferences in this section. In particular, PG1 explicitly uses
Scheduler SMT pairs, and PG2 explicitly permits physical-CPU sharing.

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
  policy permits SMT sharing. PG1 uses exactly two complete Scheduler SMT pairs.
  PG2 may assign a Scheduler to O's sibling and may assign two Schedulers to
  both threads of another physical CPU when necessary.
- Multi-cluster spreading, SMT density, and other Scheduler counts are
  exploration items.

Policy-specific capacity requirements:

| Scenario | Requirements for the default `1O+4S` layout |
| -------- | ------------------------------------------- |
| FG | At least five suitable physical CPUs; O owns one physical CPU and leaves its sibling idle; four Schedulers use four other distinct physical CPUs |
| FG+SMT | Same as FG; schedulable SMT siblings may exist, but the default layout does not use a second Scheduler thread on any physical CPU |
| PG1 | One suitable physical CPU provides O's primary thread and an idle sibling; two other physical CPUs each provide a complete schedulable SMT pair, supplying 4S |
| PG2 | At least five schedulable logical CPUs; two surviving clusters are present on different dies; O/S and Scheduler/Scheduler physical-CPU sharing is permitted |

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

1. Call the existing `probe_aicpu_topology()` to obtain the driver-filtered
   `os_schedulable_cpus`. After the unified interface is implemented, A5 uses
   `QueryAicpuTopology()` to add scenario classification. Until then, the
   existing `compute_allowed_cpus()` remains a generic packing algorithm rather
   than an implementation of these scenario policies. A2/A3 directly enters
   the existing non-PG path.
2. Group logical CPUs by `phy_cpu_id`, and retain their `cluster_id`, `die_id`,
   and `hyperthread_id`.
3. Select a scenario policy from `kFg`, `kFgSmt`, `kPg1`, or `kPg2`.
4. Validate the common requirements and every policy-specific capacity
   requirement in Section 5 before assigning any role. A count of five logical
   CPUs alone is sufficient only for PG2, subject to its required cross-die
   cluster distribution.
5. Place O and S using the selected scenario policy in Section 4. The scenario
   policy overrides the global placement preferences where Section 4 states an
   exception.
6. Verify that all five selected IDs are unique and schedulable, then emit
   `[S0 ... S3, O]` atomically. Synchronize the actual launch count, Runtime,
   DFX, and affinity state with that complete array.
7. If validation or placement fails, return insufficient capacity without
   emitting a partial affinity array.
8. For `kUnknown`, disable automatic affinity and emit only the raw probe data.
