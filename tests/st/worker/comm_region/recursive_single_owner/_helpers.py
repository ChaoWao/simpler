# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private L3/L4 kernel and host-handshake fixtures for delegated-region ST."""

from __future__ import annotations

import ctypes
import os
import struct
from collections.abc import Callable
from typing import Any

from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, scalar_to_uint64
from simpler.worker import Worker
from simpler.worker_chip_orch_comm import NotifyOp, WaitCmp

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "recursive_single_owner_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_transform.cpp")
_HEADER = struct.Struct("<QII")
_HEADER_BYTES = 64
_NUMEL = 128 * 128
_NBYTES = _NUMEL * 4
_INPUT_OFFSET = _HEADER_BYTES
_OUTPUT_OFFSET = _INPUT_OFFSET + _NBYTES
_PAYLOAD_BYTES = _OUTPUT_OFFSET + _NBYTES
_DATA_READY_COUNTER = 0
_COMPLETION_COUNTER = 64
_COUNTER_BYTES = 128
_SCALAR = ctypes.c_float(7.0)

SUCCESS_PLATFORMS = ["a2a3sim", "a2a3", "a5sim", "a5"]
FAULT_PLATFORMS = ["a2a3sim", "a5sim"]


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)
    extra_common = [str(kc.project_root / "src" / "common")]
    aiv = kc.compile_incore(
        _AIV_SRC,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=inc_dirs,
    )
    if not platform.endswith("sim"):
        aiv = extract_text_section(aiv)
    orch = kc.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=_ORCH_SRC,
        extra_include_dirs=extra_common,
    )
    return ChipCallable.build(
        signature=[],
        func_name="recursive_single_owner_orchestration",
        binary=orch,
        children=[(0, CoreCallable.build(signature=[D.IN, D.OUT], binary=aiv))],
    )


def _float_view(handle):
    return (ctypes.c_float * _NUMEL).from_address(int(handle.base))


def _byte_view(handle):
    return (ctypes.c_uint8 * int(handle.nbytes)).from_address(int(handle.base))


def _write_header(header_tensor, seq: int, opcode: int) -> None:
    buf = _byte_view(header_tensor)
    for i in range(_HEADER_BYTES):
        buf[i] = 0
    _HEADER.pack_into(buf, 0, seq, opcode, 0)


def _fill_input(input_tensor, round_idx: int) -> list[float]:
    values = _float_view(input_tensor)
    expected = []
    for i in range(_NUMEL):
        value = float(round_idx * 1000 + (i % 251)) / 16.0
        values[i] = value
        expected.append(value + float(_SCALAR.value))
    return expected


def _assert_output_matches(output_tensor, expected: list[float]) -> None:
    values = _float_view(output_tensor)
    for i, want in enumerate(expected):
        got = float(values[i])
        assert abs(got - want) <= 1e-5, f"output[{i}] expected {want}, got {got}"


def _handshake(orch_handle, region, round_idx: int) -> None:
    header = orch_handle.alloc([_HEADER_BYTES], DataType.UINT8)
    host_input = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
    host_output = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
    data_ready = region.counter(_DATA_READY_COUNTER)
    completion = region.counter(_COMPLETION_COUNTER)
    expected = _fill_input(host_input, round_idx)
    region.payload_write(_INPUT_OFFSET, host_input, nbytes=_NBYTES)
    _write_header(header, round_idx, 1)
    region.payload_write(0, header, nbytes=_HEADER.size)
    data_ready.notify(round_idx, NotifyOp.Set)
    snapshot = completion.test(round_idx, WaitCmp.GE)
    if not snapshot.matched:
        completion.wait(round_idx, WaitCmp.GE, timeout=5.0)
    region.payload_read(_OUTPUT_OFFSET, host_output, nbytes=_NBYTES)
    _assert_output_matches(host_output, expected)
    stop_seq = round_idx + 1
    _write_header(header, stop_seq, 2)
    region.payload_write(0, header, nbytes=_HEADER.size)
    data_ready.notify(stop_seq, NotifyOp.Set)


def _chip_task_args(region) -> TaskArgs:
    task_args = TaskArgs()
    for scalar in region.descriptor_scalars():
        task_args.add_scalar(int(scalar))
    task_args.add_scalar(_INPUT_OFFSET)
    task_args.add_scalar(_OUTPUT_OFFSET)
    task_args.add_scalar(_NUMEL)
    task_args.add_scalar(DataType.FLOAT32.value)
    task_args.add_scalar(_NBYTES)
    task_args.add_scalar(scalar_to_uint64(_SCALAR))
    task_args.add_scalar(_DATA_READY_COUNTER)
    task_args.add_scalar(_COMPLETION_COUNTER)
    return task_args


def _stream_config() -> CallConfig:
    config = CallConfig()
    config.aicpu_thread_num = 2
    return config


def _install_release_probe(worker: Worker) -> list[dict[str, object]]:
    releases: list[dict[str, object]] = []
    original = worker._dispatch_delegated_release

    def _probe(*, session_instance_id, transaction_id, provider_path):
        releases.append(
            {
                "session_instance_id": bytes(session_instance_id),
                "transaction_id": int(transaction_id),
                "provider_path": bytes(provider_path),
            }
        )
        return original(
            session_instance_id=session_instance_id,
            transaction_id=transaction_id,
            provider_path=provider_path,
        )

    worker._dispatch_delegated_release = _probe
    return releases


def _assert_delegated_live_region(region, provider_path: bytes, transaction_id: int) -> None:
    instance = region._instance
    assert instance._delegated_release_edge is True
    assert instance._delegated_allocation_committed is True
    assert instance._delegated_provider_path == provider_path
    assert instance._delegated_transaction_id == transaction_id
    assert instance._payload_local_view is not None
    assert instance._counter_local_view is not None
    assert int(instance._payload_local_view.logical_bytes) == _PAYLOAD_BYTES
    assert int(instance._counter_local_view.logical_bytes) == _COUNTER_BYTES
    assert int(instance._payload_local_view.local_base) != int(instance._counter_local_view.local_base)


def _assert_clean_session(worker: Worker, *, next_transaction_id: int, releases: list[dict[str, object]]) -> None:
    registry = worker._region_instance_registry
    assert registry._next_delegated_transaction_id == next_transaction_id
    assert registry._instances == {}
    assert registry._delegated_admission_closed is False
    assert [item["transaction_id"] for item in releases] == list(range(1, next_transaction_id))
    assert worker._delegated_session_fatal is None


def run_two_lifecycles(
    worker: Worker,
    *,
    provider_path: str,
    submit_chip: Callable[[Any, Any, CallConfig], None],
) -> None:
    path = provider_path.encode("ascii")
    seen: list[int] = []
    releases = _install_release_probe(worker)

    def orch(orch_handle, _args, cfg):
        assert not hasattr(orch_handle, "create_region")
        region = orch_handle._worker._create_delegated_worker_chip_region(provider_path, _PAYLOAD_BYTES, _COUNTER_BYTES)
        transaction_id = int(region._instance._delegated_transaction_id)
        seen.append(transaction_id)
        _assert_delegated_live_region(region, path, transaction_id)
        submit_chip(orch_handle, region, cfg)
        _handshake(orch_handle, region, 1)

    worker.run(orch, args=None, config=_stream_config())
    worker.run(orch, args=None, config=_stream_config())
    assert seen == [1, 2]
    _assert_clean_session(worker, next_transaction_id=3, releases=releases)


def make_l3_submit(chip_handle) -> Callable[[Any, Any, CallConfig], None]:
    def _submit(orch_handle, region, cfg):
        orch_handle.submit_next_level(chip_handle, _chip_task_args(region), cfg, worker=0)

    return _submit


def make_l3_forward(chip_handle):
    def l3_forward(orch, args, cfg):
        chip_args = TaskArgs()
        for index in range(args.scalar_count()):
            chip_args.add_scalar(int(args.scalar(index)))
        orch.submit_next_level(chip_handle, chip_args, cfg, worker=0)

    return l3_forward


def make_l4_submit(l3_handle, l3_worker_id: int) -> Callable[[Any, Any, CallConfig], None]:
    def _submit(orch_handle, region, cfg):
        orch_handle.submit_next_level(l3_handle, _chip_task_args(region), cfg, worker=int(l3_worker_id))

    return _submit
