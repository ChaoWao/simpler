# Using simpler

Documentation for **building on** simpler — writing kernels and orchestration,
running them on a device or the simulator, measuring them, and diagnosing
failures. If you are changing simpler's own runtime internals, start from the
[documentation index](../README.md) instead; the architecture and internals
pages are written for that audience.

## The path through

| Step | Go to |
| ---- | ----- |
| 1. Install and get a device visible | [Getting Started](../getting-started.md), [Installation and Runtime Environment](../install.md) |
| 2. Prove your setup works, no kernels involved | [`examples/workers/l2/hello_worker/`](../../examples/workers/l2/hello_worker/README.md) — constructs a `Worker`, `init()`, `malloc`/`free`, `close()` |
| 3. Compile and run your first kernel | [How-to: write and run a kernel](how-to/write-and-run-a-kernel.md) |
| 4. Spread work across chips | [How-to: run on multiple chips](how-to/run-on-multiple-chips.md) |
| 5. Find out where the time goes | [How-to: profile a kernel](how-to/profile-a-kernel.md) |
| 6. Something failed | [How-to: debug a failed run](how-to/debug-a-failed-run.md) |

## Reference

| Document | What it covers |
| -------- | -------------- |
| [Python API](reference/python-api.md) | The types and methods you call: `Worker`, callables, task args, `CallConfig` |
| [Command-line flags and tools](reference/cli.md) | pytest flags for platform/device/diagnostics, and the analysis CLIs |

## Which package do I import from?

Both `simpler` and `simpler_setup` are part of the surface you write against:

- **`simpler`** — the runtime itself. `Worker`, and the task/callable types in
  `simpler.task_interface`.
- **`simpler_setup`** — compilation and test scaffolding: `KernelCompiler`,
  `SceneTestCase` / `scene_test`, `Tensor`, `TaskArgsBuilder`,
  `ensure_pto_isa_root`, `make_tensor_arg`. Also the analysis CLIs under
  `simpler_setup.tools`.

You cannot compile a kernel without `simpler_setup`, so treat both as yours to
use. `from simpler import Worker` works; the task and callable types come from
`simpler.task_interface`. One name to watch:
`simpler.task_interface.Tensor` is a device tensor descriptor while
`simpler_setup.Tensor` is a scene-test argument spec — same name, different
types.

## Where the worked examples live

The example trees are the tutorial layer, and several carry their own
walkthrough README:

| Tree | What is in it |
| ---- | ------------- |
| [`examples/workers/l2/`](../../examples/workers/l2/README.md) | Single-chip Worker API: `hello_worker`, `vector_add`, `worker_malloc`, `per_task_runtime_env` |
| [`examples/workers/l3/`](../../examples/workers/l3/README.md) | Host-level multi-chip: `allreduce`, `multi_chip_dispatch`, `child_memory` |
| `examples/a2a3/`, `examples/a5/` | Per-arch, per-runtime scene tests — the `@scene_test` style you will most often copy |

[`examples/workers/l2/vector_add/main.py`](../../examples/workers/l2/vector_add/main.py)
is the most useful single file to read: it walks the whole pipeline explicitly
(compile → wrap into callables → allocate → copy → run → verify) with a diagram
in its module docstring, precisely because `@scene_test` normally hides those
stages.
