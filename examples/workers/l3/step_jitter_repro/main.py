#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce multi-chip step jitter with Simpler APIs only.

Each logical step is one four-member ``submit_next_level_group``.  The parent
keeps two ``Worker.submit`` handles in flight and retires the oldest handle
only after the successor has been submitted, matching a serving-style
continuous depth-two pipeline.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, CoreCallable, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = Path(__file__).resolve().parent
VECTOR_KERNEL = HERE / "kernels" / "aiv" / "repeated_vector_add.cpp"
ORCHESTRATION = HERE / "kernels" / "orchestration" / "repeated_vector_add.cpp"
ROWS = 128
COLS = 128


def _parse_devices(value: str) -> list[int]:
    devices = []
    for item in value.split(","):
        if "-" in item:
            first, last = (int(part) for part in item.split("-", maxsplit=1))
            devices.extend(range(first, last + 1))
        elif item:
            devices.append(int(item))
    if len(devices) != 4 or len(set(devices)) != 4:
        raise argparse.ArgumentTypeError(f"exactly four distinct devices are required, got {devices}")
    return devices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", required=True, type=_parse_devices)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2])
    parser.add_argument("--kernel-repeats", type=int, default=4096)
    return parser.parse_args()


def _build_callable(platform: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    kernel = compiler.compile_incore(
        source_path=str(VECTOR_KERNEL),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel = extract_text_section(kernel)
    orchestration = compiler.compile_orchestration(runtime_name=runtime, source_path=str(ORCHESTRATION))
    core = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN],
        binary=kernel,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN],
        func_name="repeated_vector_add",
        binary=orchestration,
        children=[(0, core)],
    )


def _task_args(worker: Worker, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], repeats: int) -> TaskArgs:
    a, b, out = tensors
    args = TaskArgs()
    args.add_tensor(make_tensor_arg(worker, a), TensorArgType.INPUT)
    args.add_tensor(make_tensor_arg(worker, b), TensorArgType.INPUT)
    args.add_tensor(make_tensor_arg(worker, out), TensorArgType.OUTPUT_EXISTING)
    args.add_scalar(repeats)
    return args


def run(config: argparse.Namespace) -> None:
    if config.warmup < 0 or config.rounds <= 0 or config.kernel_repeats <= 0:
        raise ValueError("warmup must be non-negative; rounds and kernel-repeats must be positive")

    devices = config.device
    workers = list(range(len(devices)))
    total = config.warmup + config.rounds
    print(
        f"[repro] devices={devices} warmup={config.warmup} rounds={config.rounds} "
        f"depth={config.depth} kernel_repeats={config.kernel_repeats}",
        flush=True,
    )

    torch.manual_seed(20260824)
    slots = []
    for _slot in range(config.depth):
        per_device = []
        for _device in devices:
            per_device.append(
                (
                    torch.full((ROWS, COLS), 1.0, dtype=torch.float32).share_memory_(),
                    torch.full((ROWS, COLS), 2.0, dtype=torch.float32).share_memory_(),
                    torch.zeros((ROWS, COLS), dtype=torch.float32).share_memory_(),
                )
            )
        slots.append(per_device)

    worker = Worker(
        level=3,
        platform=config.platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=devices,
        num_sub_workers=0,
    )
    chip_handle = worker.register(_build_callable(config.platform))
    worker.init()
    started = time.monotonic()
    handles = []
    try:
        slot_args = [
            [_task_args(worker, tensors, config.kernel_repeats) for tensors in per_device] for per_device in slots
        ]
        call_config = CallConfig()

        for step in range(total):
            while len(handles) >= config.depth:
                handles.pop(0).wait()
            args_for_step = slot_args[step % config.depth]

            def graph(orch, _args, cfg, group_args=args_for_step):
                orch.submit_next_level_group(chip_handle, group_args, cfg, workers=workers)

            handles.append(worker.submit(graph, config=call_config))

        for handle in handles:
            handle.wait()

        elapsed = time.monotonic() - started
        for slot in slots:
            for _a, _b, out in slot:
                if not torch.allclose(out, torch.full_like(out, 3.0), rtol=0.0, atol=0.0):
                    raise AssertionError("vector-add output mismatch")
        print(f"[repro] PASS total_steps={total} elapsed_s={elapsed:.3f}", flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    run(_parse_args())
