# scalar_data — the orchestration reads and writes tensor data itself

Every other example here treats the orchestration as a *scheduler*: it submits
kernels and touches no data. This one has the orchestration poke individual
tensor elements with `get_tensor_data` / `set_tensor_data`, interleaved with
kernel submits, and checks that the runtime orders the two correctly.

Read it for one question: **when the orchestration writes a tensor a kernel is
using, who waits?**

## The answer, and the trap

`set_tensor_data` waits for overlapping writers, their existing consumers,
and readers explicitly published with `add_tracked_input`.

| Reader annotation | Published in TensorMap? | Does a later overlapping `set_tensor_data` wait? |
| ----------------- | ----------------------- | ------------------------------------------------ |
| `add_tracked_input` | Yes | **Yes.** The host write waits for the reader task to complete. |
| `add_input` | No | **No.** The host write cannot discover an unfinished pure reader. |

The annotation belongs on the reader: a later writer cannot recover an earlier
plain input that was never published. Keep ordinary `add_input` for reads that
cannot overlap a later write, and use `add_tracked_input` when they can.

Step 11 submits a real `kernel_add` reader before overwriting its input. Step 12
uses `noop` to isolate external tracked-reader registration from kernel work;
plain `add_input` would leave the host write with no reader entry to wait on.

## The thirteen steps

`kernels/orchestration/scalar_data_orch.cpp` walks through them in order, each
landing a value in `check[0..8]`:

| Steps | What is exercised | Lands in |
| ----- | ----------------- | -------- |
| 1–3 | `get_tensor_data` on a runtime-created tensor at two indices | `check[0]` = 2.0, `check[1]` = 102.0 |
| 4–5 | `add_output(TensorCreateInfo, 77.0f)` — a runtime-created output with an **initial value** | `check[2]` = 77.0 |
| 6–7 | `add_inout` on the same tensor: the buffer already exists, so the submit only registers a dependency; the value survives | `check[3]` = 77.0 |
| 8 | Plain arithmetic in the orchestration on a value it read back | `check[4]` = 79.0 |
| 9 | `set_tensor_data` → `get_tensor_data` round-trip | `check[5]` = 42.0 |
| 10 | Orchestration → AICore RAW: write 10.0, then a kernel reads it | `check[6]` = 12.0 |
| 11 | Internal tracked reader followed by a host write | `check[7]` = 88.0 |
| 12 | External tracked-reader registration followed by a host write | `check[8]` = 55.0 |
| 13 | `result = a + b`, external output through `add_output` | `result` |

`kernel_noop` exists precisely because several steps need a task that
*registers a dependency* without touching data.

## Cases

One, `default`, for `platforms=["a2a3"]`. The default AICPU configuration uses
auto selection.

Note the comment on the `check` tensor in `generate_args`: it is **exactly 9
slots**, matching `check[0..8]`. Output-tensor slots are not seeded from the
host, so an unwritten slot would contain undefined device memory and make the
golden comparison nondeterministic.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/scalar_data --platform a2a3 --device 0
```

Onboard only. Wrap in `task-submit` on a shared box.

Each step also logs its observed-vs-expected value with `LOG_INFO`, so a
failure tells you which step diverged before you look at `check`.

## See also

- [`docs/war-anti-dependency.md`](../../../../docs/war-anti-dependency.md) — write-after-read hazards and how the runtime orders them
- [`docs/orchestrator.md`](../../../../docs/orchestrator.md) — TensorMap, producer/consumer entries, `fanout_refcount`
