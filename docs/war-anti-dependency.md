# WAR (Write-After-Read) Anti-Dependencies

The runtime can track unfinished tensor readers and writers in TensorMap. A
pure reader that may overlap a later write must use `add_tracked_input()` so
the writer can discover it and create the WAR edge.

```text
W0: INOUT X  ──RAW──>  R0: TRACKED_INPUT X  ──WAR──>  W1: INOUT X
```

On A2/A3 and A5, both `host_build_graph` and `tensormap_and_ringbuffer` use the
same opt-in rule: plain `add_input()` queries prior writers but does not publish
a reader. Use `add_tracked_input()` when a later overlapping write must wait.

## Access semantics

| Argument | Queries | Registers | Meaning |
| -------- | ------- | --------- | ------- |
| `INPUT` | overlapping writers (RAW) | nothing | ordinary read-only access |
| `TRACKED_INPUT` | overlapping writers (RAW) | reader | read-only access that must order a later write |
| `INOUT` | overlapping writers (RAW/WAW) and readers (WAR) | writer | read-modify-write |
| existing-tensor `add_output` | overlapping readers (WAR) | writer | pure overwrite (`OUTPUT_EXISTING`) |
| runtime-created `add_output` | nothing | nothing | fresh allocation (`OUTPUT`) |
| `NO_DEP` | creator only | nothing | retains the allocator but skips TensorMap lookup/publication |

Independent input tasks do not depend on each other and can still execute in
parallel. Accesses are registered only after the task's complete fanin has been
computed, preventing aliases within one task from creating a self-dependency.

The annotation belongs on the reader, not on the later writer. A writer cannot
retroactively discover an earlier plain input that left no reader entry.

`OUTPUT_EXISTING` retains its existing unordered-writer contract: it waits for
readers but does not acquire WAW edges on older writers. Consequently, a fully
covered reader entry can be retired after its WAR edge is created, while an
unordered writer entry must remain discoverable.

## `INOUT` versus pure overwrite

Use `INOUT` when the final value can depend on the target's old contents, such
as accumulation, an in-place operator, a partial update that preserves other
elements, or conditional writeback. Use existing-tensor `add_output(X)` when a
task fully determines the bytes it writes without reading the previous value.

## Host access

- `get_tensor_data()` is a host read and waits only for overlapping writers.
- `set_tensor_data()` is a host write and waits for overlapping writers, the
  existing writer-consumer drain, and every overlapping tracked reader task to
  complete. TMR performs the wait synchronously; HBG emits an equivalent host
  write graph node. Neither waits for a reader's downstream consumers.

Use `add_tracked_input()` for any pure reader followed by an overlapping
`set_tensor_data()` call.

## Explicit overrides

Manual scopes skip automatic dependency computation. Tensors marked
`manual_dep` and `add_no_dep()` retain a valid creator but skip automatic
reader/writer lookup and registration. In these modes, use
`CoreTaskArgs::set_dependencies()` or `CoreTaskArgsWithDeps::add_dep()` to state
the required task ordering explicitly. An explicit task edge is not a
buffer-keyed access record, so it cannot make a task visible to a later host
`set_tensor_data()` call.

## Capacity and diagnostics

Tracked-reader and writer entries share the existing 65,536-entry pool and
retire at the existing `CONSUMED` watermark. The indexes have separate bucket
heads so reader fan-out does not make subsequent readers scan older readers.
HBG keeps 128 fanins inline and spills additional deduplicated fanins into its
scheduler pool; fanins are never silently truncated.

Dependency capture labels TensorMap edges with `hazard` (`RAW`, `WAW`, or
`WAR`) and `access_kind` (`READER` or `WRITER`). TensorMap also retains
reader/writer live counts and high-water marks for capacity diagnosis.

## See also

- [Dependency generation DFX](dfx/dep-gen.md)
- [Manual scopes](manual-scope.md)
