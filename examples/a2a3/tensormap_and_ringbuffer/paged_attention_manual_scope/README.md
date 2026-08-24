# paged_attention_manual_scope — the same computation, dependencies declared

Identical arithmetic to [`../paged_attention/`](../paged_attention/), with
automatic dependency wiring replaced by explicit task-to-task edges.

**All four kernels are byte-identical to the baseline's.** The only file that
differs is `kernels/orchestration/paged_attention_orch.cpp`, by 45 of 292
lines. If you want to know what "manual scope" costs you in practice, that diff
is the answer — nothing about the computation changes, only who states the
ordering.

```bash
diff -r ../paged_attention/kernels ./kernels     # only the orch file differs
```

## What changes

`SIMPLER_SCOPE()` + `SIMPLER_SCOPE_GUARD()` becomes `SIMPLER_SCOPE(PTO2ScopeMode::MANUAL)`,
and each submit names its predecessors. See
[`docs/manual-scope.md`](../../../../docs/manual-scope.md) for the mode itself.

The variant deliberately uses **both dependency APIs**, so one file shows the
choice:

| API | Shape | Suited to |
| --- | ----- | --------- |
| Primitive | `TaskId deps[] = {...}; params.set_dependencies(deps, n);` | Codegen, and any task whose dependency set is fixed and known at the call site. The caller owns the buffer; the `Arg` stores `(ptr, count)`. |
| Convenience | `CoreTaskArgsWithDeps<> params; params.add_dep(id);` | Hand-written orchestration where the set is assembled conditionally across branches. The wrapper owns a stack-sized buffer and binds it on submit. |

`SF` and `PV` use the primitive form (one predecessor each). `UP` uses the
convenience form, because its edges are conditional: always the `PV` output,
plus the previous group's `UP` when there is one, plus the allocation task on
the last iteration — that last edge exists only to keep the scratch buffers
alive until their final consumer, not to order anything.

Note what is *not* declared: `UP` reads `SF`'s `mi` / `li`, but `SF → PV → UP`
already orders that transitively, so only the `PV` edge is stated. Manual mode
means you own the edges, including the ones you can leave out.

## Cases

The same seven as the baseline, with one addition: `Case1` sets
`runtime_env: {ring_task_window: 32768}`. Long-context cases submit more than
16384 in-flight tasks into a single MANUAL scope, and the default per-ring task
window is 16384 — it can fill before the oldest task retires and wedge the
orchestrator with `FLOW_CONTROL_DEADLOCK` (code 3). Doubling the window buys
the headroom. Worth knowing before you put a long loop in one manual scope: the
scope's in-flight task count becomes a sizing constraint you have to check.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/paged_attention_manual_scope --platform a2a3 --device 0
```

Onboard only. Wrap in `task-submit` on a shared box.
