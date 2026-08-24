# HBG: host-side transitive reduction of the recorded task DAG

**Status**: design
**Target**: `host_build_graph` (a2a3 + a5, mirrored)
**Baseline**: upstream/main `66ba5c4a`

## Problem

`host_build_graph` records a task DAG into a `GraphRecording` and packs it into
the device-resident Definition's fanin CSR. Every HBG edge carries exactly one
semantics — **ordering/readiness** — because Graph Execution is
whole-graph-resident: node slots are never reclaimed or rebound mid-run
(`scheduler.h`: *"on_task_release is gone… host-orch never reclaimed slots on
device"*), and outputs live until the execution object is torn down. There is
no per-edge resource-lifetime semantic to preserve.

DAGs from generated orchestrations (qwen3-class decode: ~5240 tasks; dsv4:
43-layer MoE) contain redundant ordering edges: a direct edge P→C whose
ordering is already implied by a longer path P→…→C through other nodes of C's
own fanin. On TMR these could only be cleared 1-hop and at a measured per-submit
cost (PR #1830: Orch +2.4–3.8%, no Effective gain on chain-shaped corpora). HBG
pays nothing per submit — the whole graph exists on the host at once — so the
reduction can be exact and run once per Definition build.

Cost today, per redundant edge, paid on the device at every execution of the
Definition:

- `graph_first_unmet_producer` scans the consumer's CSR row from the front on
  every wake-list re-registration (`scheduler.h`);
- `drain_graph_wake_list` re-runs that scan for every waiter on the producer's
  wake list each time the producer completes;
- the `fanout_offsets`/`fanout_indices` section of the image grows with the
  raw edge count.

## Non-goals

- No change to what `deps.json` records: dep_gen capture is wired to the
  *recording* path (annotate hooks in `record_submit`), not to the packed
  Definition. `deps.json` keeps the as-constructed edge set — same convention
  the TMR side documents in `docs/dfx/dep-gen.md` ("Flags are the
  as-constructed set").
- No cross-Definition reduction. #1968's per-block Definitions connect through
  the outer Graph shell's external dependencies; a diamond spanning two blocks
  is not visible to either block's recording. Intra-Definition only, stated as
  a scope limit here.
- No behavior gate / env knob (per `.claude/rules/env-macro-gating.md`): the
  reduction preserves reachability, and reachability is the only semantics an
  HBG edge has, so it is unconditionally correct.

## Design

### Where

One pass in `graph_build_definition` (`orchestrator_core/orchestrator.cpp`),
after the size-computation loop and **before** the CSR fill loop, operating on
`recording.nodes[*].fanin_offset/fanin_count` + `recording.internal_fanins`.
The recording itself is left untouched — dep_gen reads it afterwards (same
function in a different call chain? no: dep_gen hooks live in `record_submit`,
which has already run by build time), and the packed image is what the device
consumes. Reduction is a build-time projection of the recording, not a
mutation of it.

`graph_build_definition` runs once per recording on the recorder thread
(`graph_end`, behind `ORCH_PHASE(BuildDefinition)`), and the resulting
Definition is cached and shared across invocations (#1968 single-upload).
Steady-state replay cost of the reduction is therefore zero.

### Algorithm

Nodes are recorded in topological order already — the CSR fill loop checks
`producer >= i → return false` — so no topo-sort is needed.

Exact transitive reduction on a DAG via the standard reachability method,
iterated in reverse topological order:

```text
state: bitmask reach[N][N/64]         # GRAPH_MAX_NODES = 1024 → 1024 × 16 words = 128 KB scratch
for i = N-1 .. 0:                     # reverse topo order
    for each producer p in row(i):    # after reduction of later rows
        if reach[i] ⊇ contains(p):    # p already reachable via another path
            drop edge p→i
        else:
            reach[i] |= reach[p]
            reach[i].set(p)
```

Correctness: processing in reverse topological order means `reach[p]` is final
(a node's descendants are all later nodes) when row *i* consumes it. An edge
p→i is dropped iff `p` is reachable from some *other* producer row-entry of *i*
through its (already reduced) descendant sets — i.e. the edge is a transitive
shortcut. Reachability is preserved by construction; since ordering is the only
edge semantics, the reduced graph admits exactly the same executions.

Scratch: `GRAPH_MAX_NODES = 1024` nodes → 1024 rows × 16 × `uint64` = 128 KB,
host-side `std::vector`, freed after the pass. At the median Definition size
(seven Definitions over 43 layers, #1968 — hundreds of nodes each) this is far
below the recording itself. For `node_count` below ~256 the dense bitset is
allocated to the actual row count, not the cap.

Complexity: O(V · E / 64) word-ORs — for 1024 nodes with 8 edges/node average,
~130 K word operations, well under a millisecond on the host. This is the
"persistent per-slot ancestor closure" the TMR investigation
(`docs/investigations/2026-08-tmr-transitive-reduction-depth.md`) could not
afford on the AICPU; on the host at build time it is free.

### What changes in the packed image

- `definition.edge_count` shrinks to the reduced count.
- `fanin_offsets/fanin_indices` rows drop the removed producer entries.
- `fanout_offsets/fanout_indices` are rebuilt from the reduced edge set (they
  are pure derived data: `bind_graph_topology` only validates them; no
  scheduler code reads them — the wake machinery walks fanin via
  `graph_first_unmet_producer`).
- `root_count` is unaffected (a root has no producers; reduction cannot create
  or destroy one).
- The Definition content hash changes for graphs with redundant edges —
  expected: the hash identifies image bytes, and the image legitimately
  changed. Existing cached Definitions (from a previous process run) are
  keyed by `full_key` + hash; a rebuild with different bytes is a different
  Definition object, uploaded once. No migration concern.

### What must NOT change

- The recording (`GraphRecording`), hence `deps.json`, hence the dep_gen
  differential gate and every downstream tool (deps_viewer, swimlane join).
- `GraphRecordedNode::fanin_count` on the recording side (the reduction works
  on a local per-node view or a copy of the ranges — see Implementation).
- Boundary/external dependency handling: producers outside the recording
  window were already dropped at record time; nothing to reduce against.

## Implementation sketch

In `graph_build_definition`, between the counting loop and the layout calls:

```cpp
// Returns the reduced edge list as (producer, consumer) pairs in row order,
// or empty when the recording has no redundant shortcut edge.
std::vector<uint32_t> graph_reduce_transitive_edges(const GraphRecording &recording);
```

- The fill loop then reads the reduced rows instead of
  `recording.internal_fanins[fanin_offset + f]` directly. Simplest shape: the
  helper returns a **new** flat producer array plus per-node offsets, and the
  existing loop consumes those; when the helper finds nothing to drop it
  returns the identity projection so the loop is unchanged in shape.
- `total_fanins` for the layout pass is taken from the reduced count.
- Log one `LOG_DEBUG` line with nodes/edges before→after when the drop count
  is nonzero (gated to the existing debug channel — cold path, not per-submit).

Mirrored identically in `src/a5/.../orchestrator_core/orchestrator.cpp`
(the two files are byte-identical today; the edit applies verbatim).

## Testing

1. **Unit test** (`tests/ut/cpp/common/test_hbg_graph_reduction.cpp`, linked
   like `test_hbg_graph_cache`): construct small recordings by hand —
   - the diamond `A→B→C` + `A→C`: `A→C` is dropped, `A→B`, `B→C` kept;
   - a 3-hop chain + shortcut `A→B→C→D` + `A→D`: `A→D` dropped by the
     *transitive* path, proving arbitrary depth (the case TMR could not do);
   - two independent producers to one consumer: both kept;
   - a diamond where the shortcut is the *only* path from one producer:
     kept (no false drop);
   - duplicate edges / self-edge guards (self-edges are rejected earlier by
     `producer >= i`).
2. **bind_graph_topology invariants**: the reduced image must still pass the
   existing CSR validation (offsets monotone, indices < consumer, edge_count
   consistent both sides, root_count unchanged). The unit test asserts this by
   running `bind_graph_topology` on the packed image — the same gate the
   device runs.
3. **dep_gen differential**: an ST run with `--enable-dep-gen` before/after
   must produce byte-identical `deps.json` (reduction does not touch the
   recording). This is the guard that the recording/Definition boundary stays
   clean.
4. **Scene tests**: `test-all-sim` for both arches (a2a3sim, a5sim) — the
   reduced Definitions must replay every existing graph-carrying scene test
   identically.

## Measurement plan

After landing: `/benchmark -r host_build_graph` (a2a3). HBG's Device column
covers the scheduler dispatch window where `graph_first_unmet_producer` /
`drain_graph_wake_list` run. Expectation on qwen3-class graphs: modest
improvement on the completion-path scans, zero change in host phases
(Definition build is cached). If no example improves beyond noise, record the
null result in `docs/investigations/` per the house rule — the mechanism is
still right (removing work the device does per wake), and the entry documents
by how much.

## Risks

- **Reduced image changes Definition hashes**: intentional, benign (see above).
- **Dense fanin rows near `PTO2_MAX_FANIN`**: reduction only shrinks rows;
  nothing can overflow a cap by being reduced.
- **Predicate nodes**: dispatch predicates gate execution, not edges; a
  predicated-off node's edges still exist in the CSR and still reduce like any
  other node's. The predicate evaluation order is untouched.
- **`sync_start` cohorts**: cohort membership is built from slot states at
  execution time, not from the CSR; unaffected.

## Alternatives considered

- **Reduce at record time (drop in `add_fanin`)**: wrong place — the recording
  must stay as-constructed for dep_gen, and the reducibility question is a
  whole-graph property, not knowable per-submit (the same reason TMR is 1-hop).
- **Reduce on the device at bind time**: pays the walk on the AICPU at every
  first-execution of a Definition and complicates the verifier; the host
  already owns the image build.
- **1-hop only (mirror TMR)**: strictly dominated on the host — the exact
  algorithm is the same code shape and catches strictly more redundant edges.
