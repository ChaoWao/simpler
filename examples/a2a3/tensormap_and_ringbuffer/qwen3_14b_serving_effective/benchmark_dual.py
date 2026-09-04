#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run Serving-equivalent Qwen decode directly through Simpler."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


BATCH = 16
STEPS = 127
SAMPLED_IDS_PAD = 8
HIDDEN = 5120
PADDED_VOCAB = 152064
RING_TASK_WINDOW = 131072
RING_DEP_POOL = 131072
RING_HEAP = [1073741824, 1073741824, 1073741824, 8589934592]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-work-dir", type=Path, required=True)
    parser.add_argument("--artifact-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "dual"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--steady-skip", type=int, default=5)
    parser.add_argument("--inter-run-wait", type=float, default=5.0)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--qualification-steps", type=int)
    parser.add_argument("--fixture-module", type=Path, default=here / "fixture.py")
    parser.add_argument("--weights-module", type=Path, default=here / "weights.py")
    return parser.parse_args()


def _base_name(name: str) -> str:
    return name.split("__ssa_", 1)[0]


def _shared_slot_state(fixture: Any) -> list[dict[str, torch.Tensor]]:
    block_table = fixture.metadata["block_table"].reshape(-1)
    slots = []
    for _ in range(2):
        slots.append(
            {
                "seq_lens": fixture.metadata["seq_lens_after_first_token"]
                .clone()
                .share_memory_(),
                "block_table": block_table.clone().share_memory_(),
                "initial_block_table": block_table.clone(),
                "slot_mapping": fixture.metadata["next_slot_mapping"]
                .clone()
                .share_memory_(),
                "sampled_ids_host": torch.zeros(
                    (BATCH, SAMPLED_IDS_PAD), dtype=torch.int32
                ).share_memory_(),
            }
        )
    return slots


def _update_slot(
    slot: dict[str, torch.Tensor], golden: dict[str, torch.Tensor], step: int
) -> None:
    slot["seq_lens"].copy_(golden["seq_lens"][step])
    slot["slot_mapping"].copy_(golden["slot_mapping"][step])
    positions = slot["seq_lens"] - 1
    if not torch.equal(slot["slot_mapping"].remainder(128), positions.remainder(128)):
        raise RuntimeError(f"slot mapping offset mismatch at decode step {step}")
    logical_blocks = positions.div(128, rounding_mode="floor")
    page_ids = slot["slot_mapping"].div(128, rounding_mode="floor")
    for row in range(BATCH):
        slot["block_table"][row * 32 + int(logical_blocks[row])] = page_ids[row]
    slot["sampled_ids_host"].zero_()


def _ordered_args(param_names: list[str], values: dict[str, Any]) -> tuple[Any, ...]:
    missing = [name for name in param_names if name not in values]
    if missing:
        raise ValueError(f"decode artifact parameters have no runtime value: {missing}")
    return tuple(values[name] for name in param_names)


def _slot_value(value: Any, slot_id: int) -> Any:
    return value[slot_id] if isinstance(value, list) else value


def _sampled_value(resident: dict[str, Any], step: int, mode: str) -> Any:
    index = step % 2 if mode == "single" else step
    return resident["sampled"][index]


def _run_sequence(
    *,
    rt: Any,
    compiled: Any,
    run_config: Any,
    param_names: list[str],
    resident: dict[str, Any],
    slots: list[dict[str, torch.Tensor]],
    golden: dict[str, torch.Tensor],
    mode: str,
    steps: int,
    sampled_ids_host_abi: bool,
) -> dict[str, Any]:
    initial_host = torch.zeros((BATCH, SAMPLED_IDS_PAD), dtype=torch.int32)
    initial_host[:, 0] = golden["decode_input_token_ids"][0]
    rt.copy_to(
        resident["initial_tokens"].data_ptr,
        initial_host.data_ptr(),
        initial_host.nbytes,
    )
    for slot in slots:
        slot["block_table"].copy_(slot["initial_block_table"])
    completions = []
    token_rows: list[list[int]] = []
    started = time.perf_counter()

    def read_sampled_ids(slot_id: int, step: int) -> list[int]:
        if sampled_ids_host_abi:
            host = slots[slot_id]["sampled_ids_host"]
        else:
            device_sampled = _sampled_value(resident, step, mode)
            host = torch.empty((BATCH, SAMPLED_IDS_PAD), dtype=torch.int32)
            rt.copy_from(
                host.data_ptr(),
                device_sampled.data_ptr,
                host.nbytes,
            )
        return [int(value) for value in host[:, 0].tolist()]

    def submit(step: int):
        slot_id = 0 if mode == "single" else step % 2
        _update_slot(slots[slot_id], golden, step)
        token_input = (
            resident["initial_tokens"]
            if step == 0
            else _sampled_value(resident, step - 1, mode)
        )
        values = {
            **resident["weights"],
            **slots[slot_id],
            "rope_cos": resident["rope_cos"],
            "rope_sin": resident["rope_sin"],
            "k_cache": resident["k_cache"],
            "v_cache": resident["v_cache"],
            "out": _slot_value(resident["out"], slot_id),
            "sampled_ids_in": token_input,
            "sampled_ids": _sampled_value(resident, step, mode),
            "next_hidden": _slot_value(resident["next_hidden"], slot_id),
        }
        return slot_id, rt.submit(
            compiled,
            *_ordered_args(param_names, values),
            config=run_config,
        )

    if mode == "single":
        for step in range(steps):
            slot_id, handle = submit(step)
            handle.result()
            completions.append(time.perf_counter())
            row = read_sampled_ids(slot_id, step)
            token_rows.append(row)
    else:
        pending = []
        completed = []
        for step in range(min(2, steps)):
            slot_id, handle = submit(step)
            pending.append((step, slot_id, handle))
        next_step = 2
        while pending:
            step, slot_id, handle = pending.pop(0)
            handle.result()
            completions.append(time.perf_counter())
            completed.append((step, slot_id))
            if next_step < steps:
                next_slot, next_handle = submit(next_step)
                pending.append((next_step, next_slot, next_handle))
                next_step += 1
        token_rows.extend(
            read_sampled_ids(slot_id, step) for step, slot_id in completed
        )

    expected = golden["decode_output_token_ids"][:steps].tolist()
    if token_rows != expected:
        for step, (actual, wanted) in enumerate(zip(token_rows, expected, strict=True)):
            if actual != wanted:
                raise RuntimeError(f"golden token mismatch at decode step {step}")
        raise RuntimeError("golden token sequence mismatch")
    intervals = [
        (right - left) * 1000.0 for left, right in zip(completions, completions[1:])
    ]
    return {
        "wall_s": time.perf_counter() - started,
        "completion_intervals_ms": intervals,
        "token_rows": token_rows,
        "valid": True,
    }


def main() -> int:
    args = _parse_args()
    if args.qualification_steps is not None:
        if args.profile_only or not 1 <= args.qualification_steps <= STEPS:
            raise ValueError(
                "qualification_steps must be in [1, 127] and cannot profile"
            )
        args.warmup_runs = 0
        args.measured_runs = 1
        args.inter_run_wait = 0.0
        decode_steps = args.qualification_steps
    elif args.profile_only:
        if args.warmup_runs != 1:
            raise ValueError("profile-only requires one complete warmup sequence")
        args.measured_runs = 1
        decode_steps = STEPS
    elif args.warmup_runs != 1 or args.measured_runs != 5:
        raise ValueError("formal execution requires exactly 1 warmup + 5 measured runs")
    else:
        decode_steps = STEPS
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    fixture_module = _load_module(
        args.fixture_module.resolve(), "_serving_effective_fixture"
    )
    weights_module = _load_module(
        args.weights_module.resolve(), "_serving_effective_weights"
    )
    fixture = fixture_module.load_fixture(
        args.fixture,
        expected_versions=None,
    )
    fixture.require_executable()
    fixture.verify_external_inputs(args.model_dir.resolve())
    golden = fixture.load_golden()

    artifact_source = (args.artifact_source or fixture.artifact_dir).resolve()
    artifact_manifest = json.loads((artifact_source / "manifest.json").read_text())
    if args.artifact_work_dir.exists():
        raise FileExistsError(args.artifact_work_dir)
    shutil.copytree(artifact_source, args.artifact_work_dir)
    decode_dir = (
        args.artifact_work_dir / artifact_manifest["programs"]["decode"]["path"]
    )
    distributed_meta = json.loads((decode_dir / "distributed_meta.json").read_text())
    param_names = [_base_name(param["name"]) for param in distributed_meta["params"]]
    if len(param_names) not in {25, 26}:
        raise ValueError(f"unsupported decode ABI with {len(param_names)} parameters")
    sampled_ids_host_abi = "sampled_ids_host" in param_names

    from pypto.ir.distributed_compiled_program import (
        DistributedCompiledProgram,
        DistributedConfig,
    )
    from pypto.runtime import RunConfig

    distributed = DistributedConfig(
        device_ids=[args.device_id],
        runtime="tensormap_and_ringbuffer",
        aicpu_thread_num=int(
            distributed_meta["distributed_config"]["aicpu_thread_num"]
        ),
    )
    compiled = DistributedCompiledProgram.from_dir(
        decode_dir,
        platform="a2a3",
        distributed_config=distributed,
    )
    run_config = RunConfig(
        platform="a2a3",
        device_id=args.device_id,
        ring_task_window=RING_TASK_WINDOW,
        ring_dep_pool=RING_DEP_POOL,
        ring_heap=RING_HEAP,
    )
    slots = _shared_slot_state(fixture)

    with compiled.prepare(config=run_config) as rt:
        device_weights = {}
        for name, host in weights_module.iter_kernel_weights(args.model_dir.resolve()):
            device_weights[name] = rt.alloc_tensor(
                tuple(host.shape), host.dtype, init=host
            )
            del host
            gc.collect()
        rope_cos_host, rope_sin_host = weights_module.rope_tables(
            args.model_dir.resolve()
        )
        rope_cos = rt.alloc_tensor(
            tuple(rope_cos_host.shape), rope_cos_host.dtype, init=rope_cos_host
        )
        rope_sin = rt.alloc_tensor(
            tuple(rope_sin_host.shape), rope_sin_host.dtype, init=rope_sin_host
        )
        del rope_cos_host, rope_sin_host
        k_cache = fixture.allocate_kv(rt, "key")
        v_cache = fixture.allocate_kv(rt, "value")
        resident = {
            "weights": device_weights,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "out": (
                rt.alloc_tensor((BATCH, PADDED_VOCAB), torch.float32)
                if args.mode == "single"
                else [
                    rt.alloc_tensor((BATCH, PADDED_VOCAB), torch.float32)
                    for _ in range(2)
                ]
            ),
            "next_hidden": (
                rt.alloc_tensor((BATCH, HIDDEN), torch.bfloat16)
                if args.mode == "single"
                else [
                    rt.alloc_tensor((BATCH, HIDDEN), torch.bfloat16) for _ in range(2)
                ]
            ),
            "initial_tokens": rt.alloc_tensor((BATCH, SAMPLED_IDS_PAD), torch.int32),
            "sampled": [
                rt.alloc_tensor((BATCH, SAMPLED_IDS_PAD), torch.int32)
                for _ in range(2 if args.mode == "single" else decode_steps)
            ],
        }
        runs = []
        for kind, count in (
            ("warmup", args.warmup_runs),
            ("measured", args.measured_runs),
        ):
            for run_index in range(count):
                result = _run_sequence(
                    rt=rt,
                    compiled=compiled,
                    run_config=run_config,
                    param_names=param_names,
                    resident=resident,
                    slots=slots,
                    golden=golden,
                    mode=args.mode,
                    steps=decode_steps,
                    sampled_ids_host_abi=sampled_ids_host_abi,
                )
                result.update({"kind": kind, "run": run_index + 1})
                runs.append(result)
                is_last = kind == "measured" and run_index + 1 == count
                if not is_last:
                    time.sleep(args.inter_run_wait)

    output = {
        "schema": 1,
        "campaign_kind": (
            "qualification"
            if args.qualification_steps is not None
            else "diagnostic_profiling"
            if args.profile_only
            else "formal_performance"
        ),
        "official_performance": not args.profile_only
        and args.qualification_steps is None,
        "scenario": {
            "runtime": "tensormap_and_ringbuffer",
            "mode": args.mode,
            "batch_size": BATCH,
            "decode_dispatches": decode_steps,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "retain_all_measured_runs": True,
            "steady_skip_dispatches": args.steady_skip,
            "inter_run_wait_s": args.inter_run_wait,
            "fixture": str(fixture.root),
            "fixture_artifact_source": str(fixture.artifact_dir),
            "artifact_source": str(artifact_source),
            "artifact_work_dir": str(args.artifact_work_dir),
            "ring_task_window": RING_TASK_WINDOW,
            "ring_dep_pool": RING_DEP_POOL,
            "ring_heap": RING_HEAP,
            "kv_reuse": "decode positions are overwritten before attention reads them",
        },
        "runs": runs,
        "correctness": {
            "all_runs_valid": all(run["valid"] for run in runs),
            "cross_run_tokens_identical": len(
                {json.dumps(run["token_rows"]) for run in runs}
            )
            == 1,
            "golden_tokens_identical": True,
        },
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
