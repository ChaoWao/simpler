# How-to: write and run a kernel

Two ways to get a kernel onto a device. Pick by what you are doing:

| You want | Use | Why |
| -------- | --- | --- |
| A test or example that CI will run | **`@scene_test`** | Compilation, arg building, golden comparison and per-platform cases are handled for you |
| To see or control every stage yourself | **The `Worker` API directly** | Nothing is hidden; you own compile, malloc, copy, run |

Both need the same three source pieces:

```text
my_example/
  kernels/orchestration/my_orch.cpp   # runs on AICPU (or host) — submits tasks
  kernels/aiv/my_kernel.cpp           # runs on an AICore vector unit
  test_my_example.py                  # or main.py for the direct-API style
```

The orchestration source is what builds the task graph; the in-core sources are
the per-task kernels it submits. See
[AICore Kernel Programming](../../aicore-kernel-programming.md) for what may go
inside a kernel body.

## Option A — `@scene_test` (recommended)

Declare the sources and the per-case matrix; the framework compiles and runs
them. Copy
[`examples/a2a3/tensormap_and_ringbuffer/vector_example/`](../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/)
as your starting point.

```python
from simpler.task_interface import ArgDirection as D
from simpler_setup import SceneTestCase, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMyExample(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",
            "function_name": "aicpu_orchestration_entry",   # must match the exported symbol
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,                                # matches rt_submit_aiv_task(0, ...)
                "source": "kernels/aiv/my_kernel.cpp",
                "core_type": "aiv",                          # "aiv" or "aic"
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],                # sim needs no hardware
            "config": {"aicpu_thread_num": 4},
            "params": {},
        },
    ]

    def generate_args(self, params):
        ...        # build the input tensors

    def compute_golden(self, args, params):
        ...        # return the expected output; the framework compares for you
```

Two things to get right, because both fail confusingly:

- **`function_name` must match the symbol** your orchestration `.cpp` exports
  with `__attribute__((visibility("default")))`.
- **`func_id` must match the id** the orchestration passes to its submit call.
  `func_id=0` corresponds to the first `rt_submit_aiv_task(0, ...)`.

Run it:

```bash
# simulation — no device needed
pytest examples/my_example --platform a2a3sim

# hardware
pytest examples/my_example --platform a2a3 --device 4-7
```

Standalone (same file, no pytest):

```bash
python examples/my_example/test_my_example.py -p a2a3sim
```

`level` and `runtime` are decorator arguments; **platforms are declared per case
in `CASES`**, not on the decorator. Per-case knobs go under `"config"` —
`aicpu_thread_num`, and `runtime_env` for ring sizing on
`tensormap_and_ringbuffer`. See the [Python API reference](../reference/python-api.md)
for the full `CallConfig` field list.

## Option B — the `Worker` API directly

Use this when you need to see the stages, or when your program is not a test.
The canonical worked example is
[`examples/workers/l2/vector_add/main.py`](../../../examples/workers/l2/vector_add/main.py);
the shape is:

```python
from simpler.task_interface import (
    ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs,
    CoreCallable, DataType, Tensor,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

# 1. Compile. pto_isa_root is the managed sibling header checkout.
kc = KernelCompiler(platform=platform)
kernel_bytes = kc.compile_incore(
    source_path=".../kernels/aiv/my_kernel.cpp",
    core_type="aiv",
    pto_isa_root=ensure_pto_isa_root(),
    extra_include_dirs=kc.get_orchestration_include_dirs("tensormap_and_ringbuffer"),
)
# On real hardware only, extract .text before wrapping:
if not platform.endswith("sim"):
    from simpler_setup.elf_parser import extract_text_section
    kernel_bytes = extract_text_section(kernel_bytes)

orch_bytes = kc.compile_orchestration(
    runtime_name="tensormap_and_ringbuffer",
    source_path=".../kernels/orchestration/my_orch.cpp",
)

# 2. Wrap into callables. children maps func_id -> CoreCallable.
core = CoreCallable.build(
    signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
    binary=kernel_bytes,
)
chip = ChipCallable.build(
    signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
    func_name="my_orchestration",
    binary=orch_bytes,
    children=[(0, core)],
)

# 3. Register BEFORE init(), then init.
worker = Worker(level=2, platform=platform,
                runtime="tensormap_and_ringbuffer", device_id=device_id)
handle = worker.register(chip)
worker.init()
try:
    # 4. Device memory + H2D.
    dev_a, dev_out = worker.malloc(nbytes), worker.malloc(nbytes)
    worker.copy_to(dev_a, host_a.data_ptr(), nbytes)

    # 5. Task args, in the same order as the signature.
    args = ChipStorageTaskArgs()
    args.add_tensor(Tensor.make(dev_a, (rows, cols), DataType.FLOAT32))
    args.add_tensor(Tensor.make(dev_out, (rows, cols), DataType.FLOAT32))

    # 6. Run, then D2H.
    worker.run(handle, args, CallConfig())
    worker.copy_from(host_out.data_ptr(), dev_out, nbytes)
    worker.free(dev_a); worker.free(dev_out)
finally:
    worker.close()          # always; a leaked device stays locked
```

Ordering rules that are not optional:

- **`register()` happens before `init()`.** Construction only stashes config;
  `init()` is where runtime binaries are resolved and the device is opened.
- **`close()` belongs in a `finally`.** Skipping it leaves the device held, and
  the next job on that device hangs.
- **Task-arg order must match the callable `signature`**, positionally.

`run()` blocks and returns `None`. For a non-blocking submit use `submit()`,
which returns a handle with `done()` / `wait(timeout)` / `result(timeout)`.

## After it runs

- Timing: [How-to: profile a kernel](profile-a-kernel.md)
- A failure or a hang: [How-to: debug a failed run](debug-a-failed-run.md)
- Multiple chips: [How-to: run on multiple chips](run-on-multiple-chips.md)
