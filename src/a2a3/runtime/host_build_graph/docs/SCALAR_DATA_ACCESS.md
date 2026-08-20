# Scalar Data Access During Host Graph Construction

`host_build_graph` runs the orchestration function synchronously on the host,
before any AICPU scheduler or AICore kernel starts. `get_tensor_data` and
`set_tensor_data` therefore access the host view used to stage the graph's
external tensors; they do not interleave host code with device execution.

## Supported Uses

| Tensor state | `get_tensor_data` | `set_tensor_data` |
| ------------ | ----------------- | ----------------- |
| External tensor with no submitted producer | Reads the staged host value | Updates the staged host value |
| External control/output tensor not referenced by a task | Reads immediately | Writes immediately |
| Runtime-created output | Unsupported: not a staged tensor | Unsupported: not a staged tensor. Set the value with `TensorCreateInfo::set_initial_value` instead |
| Tensor owned by a submitted device task | Unsupported during graph construction | Unsupported during graph construction |
| Tensor with an invalid or stale owner task ID | Fails with `INVALID_ARGS` | Fails with `INVALID_ARGS` |

The supported write changes the data that will be copied to the device. Every
task in the graph observes that final staged value; submit order does not turn
the write into a barrier between kernels.

## API

```cpp
uint32_t index[1] = {0};

int32_t value = get_tensor_data<int32_t>(control, 1, index);
set_tensor_data<int32_t>(layout, 1, index, value + 1);
```

Both tensors in this example must be external tensors staged by the host. A
common use is to read an input control value or publish runtime geometry into an
external layout tensor that no submitted task owns.

## Why Device-Produced Values Cannot Be Read Here

The execution order is:

1. The host loads and calls the orchestration shared object.
2. Orchestration builds the entire task graph and returns.
3. The host relocates and copies the graph image to device memory.
4. AICPU schedulers boot and dispatch the graph.

A producer submitted in step 1 cannot become `COMPLETED` until step 4. Waiting
for that producer from the orchestration call cannot make progress. The runtime
keeps a timeout as a defensive failure backstop, but it is not a supported
synchronization mechanism.

Runtime-created output buffers live in the GM heap rather than among the staged
tensors, so `get_tensor_data` / `set_tensor_data` do not resolve them. Host
orchestration must never dereference such an address itself; the initial-value
path below is the one way it reaches those bytes.

## Initial Values on a Runtime Allocation

`TensorCreateInfo::set_initial_value(v)` fills the whole buffer the runtime
allocates for that output, before any task can observe it. It is the way to give
a runtime-created buffer a defined starting content — a fixed-size tile whose
producer writes only a prefix, for instance, so its consumer reads `v` in the
remainder rather than whatever the heap last held.

```cpp
TensorCreateInfo tile(shape, 2, DataType::INT8);
tile.set_initial_value(0);
TaskOutputTensors outs = alloc_tensors(tile);
```

The fill is a host-side write reaching device memory, so how it gets there
depends on the platform, but the semantics do not:

| Platform | How the bytes reach the heap |
| -------- | ---------------------------- |
| a2a3 onboard | The heap is mapped into the host address space (`halHostRegister(DEV_SVM_MAP_HOST)`) and the store lands on it directly — no copy |
| a5 onboard | No host-map path, so the pattern is staged in a bounded buffer and pushed with `copy_to_device` — the same mechanism `set_tensor_data` already uses there |
| simulation | A device pointer is already a host pointer |

The mapping is made by the first fill that needs it, not at bind: the heap runs
to hundreds of MB and a run that sets no initial value never touches it, so an
orchestration that uses none costs nothing here.

The write lands at the call that asked for it, never deferred: the heap is
reclaimed and re-let within one orchestration, so two fills can name the same
address and only their issue order is correct.

One constraint follows from *when* the fill happens:

- **A Graph body must not ask for one.** Recording addresses an internal node's
  outputs in the private range based at `GRAPH_RECORD_VIRTUAL_BASE`, and every
  submission of the resulting Definition materializes its own from a heap block
  whose prior contents it never reads — so a value written during recording
  reaches none of the replays. The recording is marked unsupported and
  `graph_commit` latches `PTO2_ERROR_INVALID_ARGS`; the body is not re-run on the
  ordinary path, because its outer shell was published before the body was
  recorded (see [GRAPH_EXECUTION.md](../../../../common/host_build_graph/docs/GRAPH_EXECUTION.md)).
  Allocate the buffer outside the Graph body and pass it across the boundary, or
  have a task write the padding the consumer relies on.

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

- Use scalar access only on external, host-staged tensors with no device
  producer or outstanding device consumer.
- Use tensor dependencies to order device tasks; do not use host scalar access
  as a device synchronization barrier.
- Pass values needed for graph construction as orchestration inputs or scalars.
- Keep device-produced values on the device or return them after the run.
- Give a runtime allocation its starting content with
  `TensorCreateInfo::set_initial_value`, not with `set_tensor_data`.
