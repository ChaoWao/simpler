# Scalar Data Access During Host Graph Construction

`host_build_graph` runs the orchestration function synchronously on the host,
before any AICPU scheduler or AICore kernel starts. `get_tensor_data` and
`set_tensor_data` therefore access the host view used to stage the graph's
external tensors; they do not interleave host code with device execution.

## Supported Uses

| Tensor state | `get_tensor_data` | `set_tensor_data` |
| ------------ | ----------------- | ----------------- |
| External tensor with no submitted producer | Reads the staged host value | Updates the host view before submission; otherwise emits a host-write node |
| External control/output tensor not referenced by a task | Reads immediately | Writes immediately |
| Runtime-created output | Unsupported: no registered host view | Emits a host-write node after its producer |
| Tensor owned by a submitted device task | Unsupported during graph construction | Emits a host-write node after conflicting accesses |
| Tensor with an invalid or stale owner task ID | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |

A write before the first submission changes the data staged for the device. A
later write becomes a dependency-aware scheduler node: it waits for overlapping
writers, their direct consumers, and tracked readers, then publishes itself as
the next writer for subsequent tasks.

## API

```cpp
uint32_t index[1] = {0};

int32_t value = get_tensor_data<int32_t>(control, 1, index);
set_tensor_data<int32_t>(layout, 1, index, value + 1);
```

`control` must be an external tensor staged by the host. `layout` may also be
external, or it may be a runtime-created tensor when the write occurs after its
allocation task has been submitted.

## Why Device-Produced Values Cannot Be Read Here

The execution order is:

1. The host loads and calls the orchestration shared object.
2. Orchestration builds the entire task graph and returns.
3. The host relocates and copies the graph image to device memory.
4. AICPU schedulers boot and dispatch the graph.

A producer submitted in step 1 cannot become `COMPLETED` until step 4. A host
read that waits from orchestration therefore cannot make progress. A host write
does not wait synchronously; it is represented in the graph and executes in
step 4.

Runtime-created output buffers live in the graph heap and have no host-view
registration. Host orchestration must not dereference them, but a host-write
node may modify them on the device after their producer completes.

## Ownership Validation

Before a wait slot is used, the runtime verifies:

- the task ID is valid and belongs to the single HBG ring;
- the selected ring slot has a bound task descriptor; and
- the descriptor's full task ID matches the tensor's owner/producer ID.

The full-ID check prevents a masked ring-slot lookup from aliasing an unused or
different task. A failure latches `PTO2_ERROR_INVALID_ARGS` and the run returns
status `-5`; reads return zero and writes stop only after that fatal status is
recorded.

## Practical Rules

- Use `get_tensor_data` only on external, host-staged tensors with no device
  producer.
- Mark pure readers with `add_tracked_input()` when a later host write overlaps.
- Use tensor dependencies to order device tasks; a host write is a graph node,
  not a synchronous device barrier.
- Pass values needed for graph construction as orchestration inputs or scalars.
- Keep device-produced values on the device or return them after the run; they
  cannot drive host control flow during graph construction.
