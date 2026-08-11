# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import gc
import importlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional, cast

import pytest
from simpler import worker as worker_module
from simpler import worker_chip_orch_comm
from simpler.buffer import mint_owner_instance_id, wrap_fork_inherited
from simpler.task_interface import DataType
from simpler.worker import (
    _IDLE,
    _OFF_STATE,
    Worker,
    _buffer_field_addr,
    _mailbox_store_i32,
)
from simpler.worker_chip_orch_comm import (
    NotifyOp,
    SignalTestResult,
    WaitCmp,
)

from simpler_setup.runtime_builder import RuntimeBuilder

_task_interface_ext = cast(Any, importlib.import_module("_task_interface"))


class _FakeDirectCWorker:
    def __init__(
        self,
        *,
        payload_base: int = 0xDEAD_0000,
        access_profile: int = int(worker_chip_orch_comm.WorkerChipRegionAccessProfile.SIM_POSIX_SHM),
        device_id: int = 0,
        shareable_handle: int = 0xABCDEF,
        magic_version: int = 0x4C334C3200020000,
        region_id: Optional[int] = None,
        mapping_bytes: Optional[int] = None,
    ):
        self.create_calls: list[tuple[int, str, str]] = []
        self.release_calls: list[tuple[int, int]] = []
        self.next_region_id = 1
        self.payload_base = int(payload_base)
        self.access_profile = int(access_profile)
        self.device_id = int(device_id)
        self.shareable_handle = int(shareable_handle)
        self.magic_version = int(magic_version)
        self.region_id = region_id
        self.mapping_bytes = mapping_bytes

    def control_worker_chip_region_create(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
        self.create_calls.append((int(worker_id), str(request_shm_name), str(reply_shm_name)))
        req_shm = SharedMemory(name=request_shm_name)
        reply_shm = SharedMemory(name=reply_shm_name)
        req_buf = req_shm.buf
        reply_buf = reply_shm.buf
        assert req_buf is not None
        assert reply_buf is not None
        try:
            req = worker_chip_orch_comm._REGION_CREATE_REQUEST.unpack_from(req_buf, 0)
            payload_bytes = int(req[2])
            counter_bytes = int(req[3])
            counter_offset = ((payload_bytes + 63) // 64) * 64
            region_id = int(self.region_id) if self.region_id is not None else self.next_region_id
            if self.region_id is None:
                self.next_region_id += 1
            backing_name = f"sim-direct-{region_id}".encode()
            if self.access_profile == int(worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM):
                backing_name = b""
            shareable_handle = (
                self.shareable_handle
                if self.access_profile == int(worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM)
                else 0
            )
            worker_chip_orch_comm._REGION_CREATE_REPLY.pack_into(
                reply_buf,
                0,
                self.magic_version,
                region_id,
                self.payload_base,
                payload_bytes,
                self.payload_base + counter_offset,
                counter_bytes,
                self.access_profile,
                0,
                self.device_id,
                backing_name + b"\x00" * (worker_chip_orch_comm._CTRL_SHM_TOKEN_BYTES - len(backing_name)),
                counter_offset + counter_bytes if self.mapping_bytes is None else int(self.mapping_bytes),
                shareable_handle,
            )
        finally:
            del req_buf
            del reply_buf
            req_shm.close()
            reply_shm.close()

    def control_worker_chip_region_release(self, worker_id: int, region_id: int) -> None:
        self.release_calls.append((int(worker_id), int(region_id)))


class _EndpointFailingOrch:
    def _begin_run(self) -> int:
        return 1

    def _scope_begin(self) -> None:
        pass

    def _scope_end(self) -> None:
        pass

    def _close_run_submission(self, run_id: int) -> None:
        assert run_id == 1

    def _fail_run_submission(self, run_id: int) -> None:
        assert run_id == 1

    def _wait_run(self, run_id: int) -> None:
        assert run_id == 1
        raise RuntimeError(
            "child failed: L3-L2 endpoint error op=signal_wait kind=3 region=2 "
            "counter_addr=0x200000 counter_operand=7 observed_counter=0 msg=wait timed out"
        )

    def _release_run(self, run_id: int) -> None:
        assert run_id == 1


class _NamedOnboardRegionExport:
    def __init__(self) -> None:
        self.device_addr = 0xD00D_0000
        self.mapping_bytes = 65536
        self.shareable_handle = 0xABCDEF
        self.registry_handle = 77


class _FakeChipWorkerForRegionCreate:
    """Collaborator double for the in-process L3-L2 region-create handlers.

    Distinct from ``_harness.FakeChipWorker``, which stands in for the class the
    forked chip child instantiates: this one is never bound as
    ``worker.ChipWorker``, so it implements only the device-memory surface a
    region create touches.
    """

    device_id = 2

    def __init__(self) -> None:
        self.copy_calls: list[tuple[int, int]] = []

    def copy_to(self, dst: int, _src: int, size: int) -> None:
        self.copy_calls.append((int(dst), int(size)))


def _write_ctrl_shm_name(buf: memoryview, offset: int, name: str) -> None:
    encoded = name.encode("utf-8")
    assert len(encoded) < worker_module._CTRL_SHM_NAME_BYTES
    buf[offset : offset + worker_module._CTRL_SHM_NAME_BYTES] = b"\x00" * worker_module._CTRL_SHM_NAME_BYTES
    buf[offset : offset + len(encoded)] = encoded


def _make_started_sim_worker() -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker()
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = fake_c_worker
    worker._chip_shms = [shm]
    return worker, shm, fake_c_worker


def _make_started_onboard_worker(platform: str = "a2a3") -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[2], platform=platform, runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker(
        access_profile=int(worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM),
        device_id=2,
    )
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = fake_c_worker
    worker._chip_shms = [shm]
    return worker, shm, fake_c_worker


def test_sim_direct_region_uses_lifecycle_control_and_worker_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_sim",
            lambda token, mapping_bytes, owner_token: calls.append(("import", token, mapping_bytes, owner_token)) or 99,
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_payload_write",
            lambda handle, offset, src, nbytes: calls.append(("write", handle, offset, src, nbytes)),
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_payload_read",
            lambda handle, offset, dst, nbytes: calls.append(("read", handle, offset, dst, nbytes)),
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_counter_test",
            lambda handle, offset, value, cmp: (calls.append(("test", handle, offset, value, cmp)) or (True, 7)),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        payload = wrap_fork_inherited(0x1234_0000, 16, mint_owner_instance_id(), 1, "L3")
        region.payload_write(0, payload, nbytes=8)
        region.payload_read(8, payload, nbytes=8)
        result = region.counter(64).test(7, WaitCmp.EQ)
        region.counter(64).notify(3, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200020000, 1, 0xDEAD_0000, 64, 0xDEAD_0040, 128]
        assert 99 not in region.descriptor_scalars()
        worker_host_mapping = region._worker_host_mapping
        assert worker_host_mapping is not None
        assert worker_host_mapping.handle != region.descriptor.payload_base
        assert worker_host_mapping.counter_offset == 64
        assert calls[0] == ("import", "sim-direct-1", 192, worker._owner_id)
        assert calls[1][0:3] == ("write", 99, 0)
        assert calls[2][0:3] == ("read", 99, 8)
        assert calls[3] == ("test", 99, 128, 7, int(WaitCmp.EQ))
        assert calls[4] == ("notify", 99, 128, 3, int(NotifyOp.Set))
        assert result == SignalTestResult(matched=True, observed=7)
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_region_imports_vmm_shareable_handle_and_uses_worker_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda device_id, shareable_handle, mapping_bytes, owner_token: calls.append(
                ("import_onboard", device_id, shareable_handle, mapping_bytes, owner_token)
            )
            or 123,
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        region.counter(64).notify(9, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200020000, 1, 0xDEAD_0000, 64, 0xDEAD_0040, 128]
        assert 123 not in region.descriptor_scalars()
        worker_host_mapping = region._worker_host_mapping
        assert worker_host_mapping is not None
        assert worker_host_mapping.access_profile == worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM
        assert worker_host_mapping.counter_offset == 64
        assert calls[0] == ("import_onboard", 2, 0xABCDEF, 192, worker._owner_id)
        assert calls[1] == ("notify", 123, 128, 9, int(NotifyOp.Set))
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_create_import_failure_rolls_back_l2_host_region(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_sim",
            lambda _token, _mapping_bytes, _owner_token: (_ for _ in ()).throw(RuntimeError("import failed")),
        )

        with pytest.raises(RuntimeError, match="import failed"):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.create_calls
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_worker_chip_regions == []
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_create_decode_failure_rolls_back_l2_host_region():
    worker, shm, fake_c_worker = _make_started_sim_worker()
    fake_c_worker.access_profile = 99
    try:
        with pytest.raises(ValueError, match="99 is not a valid"):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_worker_chip_regions == []
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize(
    ("reply_updates", "match"),
    [
        ({"magic_version": 0xBAD}, "magic_version is invalid"),
        ({"region_id": 0}, "region_id must be nonzero"),
        (
            {"access_profile": int(worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM)},
            "access_profile must be sim_posix_shm",
        ),
    ],
)
def test_direct_create_validation_failure_rolls_back_l2_host_region(reply_updates, match):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    for name, value in reply_updates.items():
        setattr(fake_c_worker, name, value)
    expected_region_id = int(reply_updates.get("region_id", 1))
    try:
        with pytest.raises(RuntimeError, match=match):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.release_calls == ([(0, expected_region_id)] if expected_region_id else [])
        assert worker._live_worker_chip_regions == []
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_mapping_bytes_too_small_rolls_back_l2_host_region(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    fake_c_worker.mapping_bytes = 191
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda *args: calls.append(args) or 123,
        )

        with pytest.raises(RuntimeError, match="onboard_vmm reply mapping_bytes is smaller"):
            worker._create_worker_chip_region(0, 64, 128)

        assert calls == []
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_worker_chip_regions == []
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_mapping_allows_granularity_aligned_mapping(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    fake_c_worker.mapping_bytes = 65536
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda *args: calls.append(args) or 123,
        )

        region = worker._create_worker_chip_region(0, 64, 128)

        assert calls == [(2, 0xABCDEF, 65536, worker._owner_id)]
        assert region._worker_host_mapping is not None
        assert region._worker_host_mapping.total_bytes == 192
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize("interrupted_publication", ["worker", "run"])
def test_direct_region_create_rolls_back_partially_published_region(monkeypatch, interrupted_publication):
    class _AppendThenInterrupt(list):
        def append(self, item) -> None:
            super().append(item)
            raise KeyboardInterrupt(f"interrupted {interrupted_publication} publication")

    worker, shm, fake_c_worker = _make_started_sim_worker()
    resources = worker_module._RunResources()
    worker._building_run_resources = resources
    close_calls: list[int] = []
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55)
    monkeypatch.setattr(
        worker_chip_orch_comm,
        "_worker_host_mapped_region_close",
        lambda handle: close_calls.append(int(handle)),
    )
    if interrupted_publication == "worker":
        worker._live_worker_chip_regions = _AppendThenInterrupt()
    else:
        resources.worker_chip_regions = _AppendThenInterrupt()

    try:
        with pytest.raises(KeyboardInterrupt, match=f"interrupted {interrupted_publication} publication"):
            worker._create_worker_chip_region(0, 64, 128)

        assert worker._live_worker_chip_regions == []
        assert resources.worker_chip_regions == []
        assert resources.requires_ordered_cleanup is False
        assert close_calls == [55]
        assert fake_c_worker.release_calls == [(0, 1)]
    finally:
        worker._building_run_resources = None
        worker._live_worker_chip_regions.clear()
        resources.worker_chip_regions.clear()
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_region_create_mapping_rollback_failure_poisons_worker(monkeypatch):
    class _AppendThenInterrupt(list):
        def append(self, item) -> None:
            super().append(item)
            raise KeyboardInterrupt("interrupted publication")

    worker, shm, fake_c_worker = _make_started_sim_worker()
    worker._live_worker_chip_regions = _AppendThenInterrupt()
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55)
    monkeypatch.setattr(
        worker_chip_orch_comm,
        "_worker_host_mapped_region_close",
        lambda _handle: (_ for _ in ()).throw(RuntimeError("mapping close failed")),
    )

    try:
        with pytest.raises(RuntimeError, match="rollback could not close the L3 Host mapping") as excinfo:
            worker._create_worker_chip_region(0, 64, 128)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_worker_chip_regions == []
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
    finally:
        worker._live_worker_chip_regions.clear()
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_unadopted_native_mapping_cleanup_failure_poisons_worker(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    cleanup_errors = iter(("", "native owner cleanup failed"))
    consumed_owner_tokens: list[str] = []
    acknowledgements: list[tuple[str, str]] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return next(cleanup_errors)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_ack_cleanup_error",
        lambda owner_token, observed: acknowledgements.append((owner_token, observed)),
    )
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda _token, _size, _owner_token: (_ for _ in ()).throw(KeyboardInterrupt("interrupted native adoption")),
    )

    try:
        with pytest.raises(RuntimeError, match="rollback could not close the L3 Host mapping") as excinfo:
            worker._create_worker_chip_region(0, 64, 128)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "native owner cleanup failed" in str(excinfo.value.__cause__)
        assert consumed_owner_tokens == [worker._owner_id, worker._owner_id]
        assert acknowledgements == [(worker._owner_id, "native owner cleanup failed")]
        assert fake_c_worker.release_calls == [(0, 1)]
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_interrupted_cleanup_ack_happens_after_region_rollback(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    cleanup_errors = iter(("", "native owner cleanup failed"))
    ack_interrupt = KeyboardInterrupt("interrupted cleanup acknowledgement")

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda _owner_token: next(cleanup_errors),
    )

    def interrupt_ack(_owner_token: str, _observed: str) -> None:
        assert fake_c_worker.release_calls == [(0, 1)]
        raise ack_interrupt

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", interrupt_ack)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda _token, _size, _owner_token: (_ for _ in ()).throw(KeyboardInterrupt("interrupted native adoption")),
    )

    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker._create_worker_chip_region(0, 64, 128)

        assert caught.value is ack_interrupt
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._ordered_cleanup_error is worker._worker_host_mapped_cleanup_error
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_deferred_native_cleanup_error_only_poisons_owning_worker_on_admission(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    peer._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    consumed_owner_tokens: list[str] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with peer._operation_lease("submit"):
        pass

    assert peer._ordered_cleanup_error is None
    with pytest.raises(RuntimeError, match="no further work is admitted") as excinfo:
        with owner._operation_lease("submit"):
            pass

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "owner mapping cleanup failed" in str(excinfo.value.__cause__.__cause__)
    assert consumed_owner_tokens == [peer._owner_id, owner._owner_id]


def test_close_consumes_only_its_deferred_native_cleanup_error(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    consumed_owner_tokens: list[str] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    peer.close()
    with pytest.raises(RuntimeError, match="native L3 Host mapping") as excinfo:
        owner.close()

    assert "owner mapping cleanup failed" in str(excinfo.value.__cause__)
    assert consumed_owner_tokens == [peer._owner_id, peer._owner_id, owner._owner_id, owner._owner_id]


def test_cleanup_error_survives_interrupted_peek_boundary(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    peer._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    interrupt = KeyboardInterrupt("interrupted native cleanup-error lookup")
    interrupt_owner_once = True

    def peek_cleanup_error(owner_token: str) -> str:
        nonlocal interrupt_owner_once
        if owner_token == owner._owner_id and interrupt_owner_once:
            interrupt_owner_once = False
            raise interrupt
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass

    assert caught.value is interrupt
    assert owner._ordered_cleanup_error is None
    assert errors == {owner._owner_id: "owner mapping cleanup failed"}
    with peer._operation_lease("submit"):
        pass
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass
    assert owner._ordered_cleanup_error is owner._worker_host_mapped_cleanup_error
    assert errors == {}


def test_cleanup_error_ack_interrupt_happens_after_poison_publication(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    interrupt = KeyboardInterrupt("interrupted cleanup-error acknowledgement")
    interrupt_ack_once = True

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        nonlocal interrupt_ack_once
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)
        if interrupt_ack_once:
            interrupt_ack_once = False
            raise interrupt

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass

    assert caught.value is interrupt
    assert owner._ordered_cleanup_error is owner._worker_host_mapped_cleanup_error
    assert errors == {}
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass


def test_later_native_cleanup_error_is_retained_in_sticky_poison(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors: dict[str, str] = {owner._owner_id: "cleanup A"}

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    errors[owner._owner_id] = "cleanup B"
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    assert owner._worker_host_mapped_cleanup_error is not None
    assert owner._worker_host_mapped_cleanup_error.__cause__ is not None
    assert str(owner._worker_host_mapped_cleanup_error.__cause__) == "cleanup A; cleanup B"
    assert errors == {}


def test_interrupted_ack_replay_does_not_duplicate_cleanup_detail(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors: dict[str, str] = {owner._owner_id: "cleanup A"}
    interrupt = KeyboardInterrupt("interrupted before native acknowledgement")
    interrupt_once = True

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        nonlocal interrupt_once
        if interrupt_once:
            interrupt_once = False
            raise interrupt
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass
    assert caught.value is interrupt

    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    assert owner._worker_host_mapped_cleanup_error is not None
    assert owner._worker_host_mapped_cleanup_error.__cause__ is not None
    assert str(owner._worker_host_mapped_cleanup_error.__cause__) == "cleanup A"
    assert errors == {}


def test_late_cleanup_error_after_successful_close_replays_stably(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    errors: dict[str, str] = {}

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    worker.close()
    errors[worker._owner_id] = "late owner mapping cleanup failed"

    with pytest.raises(RuntimeError, match="native L3 Host mapping") as first:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()

    assert replayed.value is first.value
    assert errors == {}


def test_concurrent_close_publishes_joiner_cleanup_error_to_every_caller(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors: dict[str, str] = {}
    teardown_started = threading.Event()
    joiner_errors: list[BaseException] = []

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    def teardown_tree() -> None:
        errors[worker._owner_id] = "joiner observed mapping cleanup failed"
        teardown_started.set()
        worker._worker = None

    monkeypatch.setattr(worker, "_teardown_ready_tree", teardown_tree)

    def join_close() -> None:
        assert teardown_started.wait(5.0)
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            joiner_errors.append(exc)

    joiner = threading.Thread(target=join_close)
    joiner.start()
    try:
        with pytest.raises(RuntimeError, match="native L3 Host mapping") as owner_error:
            worker.close()
    finally:
        joiner.join(5.0)

    assert not joiner.is_alive()
    assert joiner_errors == [owner_error.value]


def test_cleanup_error_after_final_consume_waits_for_next_close_attempt(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors: dict[str, str] = {}
    late_recorded = threading.Event()
    joiner_waiting = threading.Event()
    joiner_errors: list[BaseException] = []
    has_live_calls = 0

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    def has_live_resources() -> bool:
        nonlocal has_live_calls
        has_live_calls += 1
        if has_live_calls == 1:
            return True
        if has_live_calls == 2:
            errors[worker._owner_id] = "late owner mapping cleanup failed"
            late_recorded.set()
            assert joiner_waiting.wait(5.0)
        return False

    monkeypatch.setattr(worker, "_has_live_resources", has_live_resources)
    real_close_wait = worker._hierarchical_start_cv.wait
    joiner: threading.Thread

    def close_wait(timeout=None):
        if threading.current_thread() is joiner:
            joiner_waiting.set()
        return real_close_wait(timeout=timeout)

    monkeypatch.setattr(worker._hierarchical_start_cv, "wait", close_wait)

    def join_close() -> None:
        assert late_recorded.wait(5.0)
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            joiner_errors.append(exc)

    joiner = threading.Thread(target=join_close)
    joiner.start()
    try:
        worker.close()
    finally:
        joiner.join(5.0)

    assert not joiner.is_alive()
    assert joiner_errors == []
    assert worker._close_completion is not None
    assert worker._close_completion.error is None
    assert errors == {worker._owner_id: "late owner mapping cleanup failed"}

    with pytest.raises(RuntimeError, match="native L3 Host mapping") as first:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()
    assert replayed.value is first.value


def test_wrong_thread_close_does_not_consume_owner_cleanup_error(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors = {worker._owner_id: "owner mapping cleanup failed"}
    peeked: list[str] = []
    foreign_errors: list[BaseException] = []

    def peek_cleanup_error(owner_token: str) -> str:
        peeked.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    def close_from_foreign_thread() -> None:
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            foreign_errors.append(exc)

    foreign = threading.Thread(target=close_from_foreign_thread)
    foreign.start()
    foreign.join(5.0)

    assert len(foreign_errors) == 1
    assert "thread that init()'d it" in str(foreign_errors[0])
    assert peeked == []
    assert errors == {worker._owner_id: "owner mapping cleanup failed"}
    with pytest.raises(RuntimeError, match="native L3 Host mapping"):
        worker.close()
    assert errors == {}


def test_cleanup_error_survives_close_drain_timeout_retry(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    worker._active_ops = 1
    errors = {worker._owner_id: "owner mapping cleanup failed"}

    monkeypatch.setattr(worker_module, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.001)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    with pytest.raises(TimeoutError):
        worker.close()
    assert worker._worker_host_mapped_cleanup_error is not None
    assert errors == {}

    worker._active_ops = 0
    with pytest.raises(RuntimeError, match="native L3 Host mapping") as retry:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()

    assert replayed.value is retry.value


def test_native_mapping_cleanup_errors_are_keyed_by_owner_token():
    owner_token = "owner-a"
    peer_token = "owner-b"
    _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token)
    _task_interface_ext._worker_host_mapped_region_take_cleanup_error(peer_token)

    _task_interface_ext._worker_host_mapped_region_record_cleanup_error_for_test(
        owner_token, "owner mapping cleanup failed"
    )

    assert _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(peer_token) == ""
    observed = _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(owner_token)
    assert observed == "owner mapping cleanup failed"
    _task_interface_ext._worker_host_mapped_region_record_cleanup_error_for_test(owner_token, "later cleanup failed")
    _task_interface_ext._worker_host_mapped_region_ack_cleanup_error(owner_token, observed)
    assert _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(owner_token) == "later cleanup failed"
    _task_interface_ext._worker_host_mapped_region_ack_cleanup_error(owner_token, "later cleanup failed")
    assert _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token) == ""


def test_onboard_region_create_handler_uses_named_export_fields(monkeypatch):
    req_shm = SharedMemory(create=True, size=worker_chip_orch_comm._REGION_CREATE_REQUEST_BYTES)
    reply_shm = SharedMemory(create=True, size=worker_chip_orch_comm._REGION_CREATE_REPLY_BYTES)
    req_buf = cast(memoryview, req_shm.buf)
    reply_buf = cast(memoryview, reply_shm.buf)
    ctrl_storage = bytearray(worker_module._OFF_ARGS + 2 * worker_module._CTRL_SHM_NAME_BYTES)
    ctrl_buf = memoryview(ctrl_storage)
    cw = _FakeChipWorkerForRegionCreate()
    typed_cw = cast(Any, cw)
    close_calls: list[int] = []
    try:
        worker_chip_orch_comm.WorkerChipRegionCreateRequest(
            magic_version=0x4C334C3200020000,
            request_bytes=worker_chip_orch_comm._REGION_CREATE_REQUEST_BYTES,
            payload_bytes=64,
            counter_bytes=128,
        ).encode_into(req_buf)
        _write_ctrl_shm_name(ctrl_buf, worker_module._OFF_ARGS, req_shm.name)
        _write_ctrl_shm_name(ctrl_buf, worker_module._OFF_ARGS + worker_module._CTRL_SHM_NAME_BYTES, reply_shm.name)
        monkeypatch.setattr(
            worker_module,
            "_l3_child_onboard_region_create",
            lambda _nbytes: _NamedOnboardRegionExport(),
        )
        monkeypatch.setattr(
            worker_module,
            "_l3_child_onboard_region_close",
            lambda handle: close_calls.append(int(handle)),
        )

        store = worker_module._HostWorkerChipRegionStore()
        worker_module._handle_ctrl_worker_chip_region_create(typed_cw, ctrl_buf, "a2a3", store)

        reply = worker_chip_orch_comm.decode_region_create_reply(reply_buf)
        assert reply.desc.scalars() == [0x4C334C3200020000, 1, 0xD00D_0000, 64, 0xD00D_0040, 128]
        assert reply.access_profile == worker_chip_orch_comm.WorkerChipRegionAccessProfile.ONBOARD_VMM
        assert reply.device_id == 2
        assert reply.mapping_bytes == 65536
        assert reply.shareable_handle == 0xABCDEF
        assert cw.copy_calls == [(0xD00D_0040, 128)]

        region = store.regions.pop(1)
        worker_module._release_host_worker_chip_region(region)
        assert close_calls == [77]
    finally:
        ctrl_buf.release()
        del req_buf
        del reply_buf
        req_shm.close()
        req_shm.unlink()
        reply_shm.close()
        reply_shm.unlink()


def test_region_create_handler_rejects_abi_mismatch():
    req_shm = SharedMemory(create=True, size=worker_chip_orch_comm._REGION_CREATE_REQUEST_BYTES)
    reply_shm = SharedMemory(create=True, size=worker_chip_orch_comm._REGION_CREATE_REPLY_BYTES)
    req_buf = cast(memoryview, req_shm.buf)
    ctrl_storage = bytearray(worker_module._OFF_ARGS + 2 * worker_module._CTRL_SHM_NAME_BYTES)
    ctrl_buf = memoryview(ctrl_storage)
    store = worker_module._HostWorkerChipRegionStore()
    try:
        _write_ctrl_shm_name(ctrl_buf, worker_module._OFF_ARGS, req_shm.name)
        _write_ctrl_shm_name(ctrl_buf, worker_module._OFF_ARGS + worker_module._CTRL_SHM_NAME_BYTES, reply_shm.name)
        bad_requests = (
            worker_chip_orch_comm.WorkerChipRegionCreateRequest(
                magic_version=0xDEAD,
                request_bytes=worker_chip_orch_comm._REGION_CREATE_REQUEST_BYTES,
                payload_bytes=64,
                counter_bytes=128,
            ),
            worker_chip_orch_comm.WorkerChipRegionCreateRequest(
                magic_version=0x4C334C3200020000,
                request_bytes=worker_chip_orch_comm._REGION_CREATE_REQUEST_BYTES + 8,
                payload_bytes=64,
                counter_bytes=128,
            ),
        )
        for bad_request in bad_requests:
            bad_request.encode_into(req_buf)
            with pytest.raises(RuntimeError, match="CTRL_WORKER_CHIP_REGION_CREATE"):
                worker_module._handle_ctrl_worker_chip_region_create(
                    cast(Any, _FakeChipWorkerForRegionCreate()), ctrl_buf, "a2a3", store
                )
        assert store.regions == {}
    finally:
        ctrl_buf.release()
        del req_buf
        req_shm.close()
        req_shm.unlink()
        reply_shm.close()
        reply_shm.unlink()


def test_worker_host_mapped_counter_wait_releases_gil_for_python_notifier():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "counter-wait-test")
        handle = int(owner)

        def notify() -> None:
            time.sleep(0.05)
            _task_interface_ext._worker_host_mapped_counter_notify(handle, 0, 1, int(NotifyOp.Set))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(notify)
            status, error_kind, observed, matched, message = _task_interface_ext._worker_host_mapped_counter_wait(
                handle, 0, 1, int(WaitCmp.EQ), 1_000_000_000
            )
            future.result(timeout=1.0)

        assert (status, error_kind, observed, matched, message) == (0, 0, 1, True, "")
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_worker_host_mapped_sim_payload_and_counter_helpers_roundtrip():
    shm = SharedMemory(create=True, size=128)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 128, "roundtrip-test")
        handle = int(owner)
        src_t = ctypes.c_uint8 * 8
        src = src_t(*range(10, 18))
        dst = src_t()

        _task_interface_ext._worker_host_mapped_payload_write(handle, 16, ctypes.addressof(src), 8)
        _task_interface_ext._worker_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
        assert bytes(dst) == bytes(range(10, 18))

        _task_interface_ext._worker_host_mapped_counter_notify(handle, 64, 3, int(NotifyOp.Set))
        assert _task_interface_ext._worker_host_mapped_counter_test(handle, 64, 3, int(WaitCmp.EQ)) == (True, 3)
        _task_interface_ext._worker_host_mapped_counter_notify(handle, 64, 4, int(NotifyOp.Add))
        assert _task_interface_ext._worker_host_mapped_counter_test(handle, 64, 7, int(WaitCmp.GE)) == (True, 7)
        assert _task_interface_ext._worker_host_mapped_counter_wait(handle, 64, 7, int(WaitCmp.EQ), 1_000_000) == (
            0,
            0,
            7,
            True,
            "",
        )

        _task_interface_ext._worker_host_mapped_region_close(handle)
        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_worker_host_mapped_region_close_makes_sim_handle_unusable():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "closed-handle-test")
        handle = int(owner)
        _task_interface_ext._worker_host_mapped_region_close(handle)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(handle, 0, 0, int(WaitCmp.EQ))
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_worker_host_mapped_import_owner_closes_unadopted_mapping():
    shm = SharedMemory(create=True, size=64)
    raw_handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "unadopted-owner-test")
        raw_handle = int(owner)
        del owner
        gc.collect()

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(raw_handle, 0, 0, int(WaitCmp.EQ))
    finally:
        if raw_handle:
            _task_interface_ext._worker_host_mapped_region_close(raw_handle)
        shm.close()
        shm.unlink()


def test_sim_import_registry_failure_releases_pre_registry_mapping():
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("requires Linux procfs resource accounting")

    shm = SharedMemory(create=True, size=64)
    shm_token = shm.name.lstrip("/")
    owner_token = "registry-failure-test"

    def mapped_resource_counts() -> tuple[int, int]:
        fd_count = 0
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd_name}")
            except OSError:
                continue
            fd_count += shm_token in target
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            map_count = sum(shm_token in line for line in maps_file)
        return fd_count, map_count

    try:
        baseline = mapped_resource_counts()
        _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token)
        _task_interface_ext._worker_host_mapped_region_fail_next_registry_insert_for_test()

        with pytest.raises(RuntimeError, match="injected mapped-region registry insertion failure"):
            _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, owner_token)

        gc.collect()
        assert mapped_resource_counts() == baseline
        assert _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token) == ""
    finally:
        shm.close()
        shm.unlink()


def test_worker_host_mapped_concurrent_closes_wait_for_in_flight_counter_wait():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "concurrent-close-test")
        handle = int(owner)
        close_entered = [threading.Event(), threading.Event()]
        close_done = [threading.Event(), threading.Event()]

        def wait_for_counter():
            return _task_interface_ext._worker_host_mapped_counter_wait(handle, 0, 1, int(WaitCmp.EQ), 1_000_000_000)

        def close_mapping(index: int) -> None:
            close_entered[index].set()
            _task_interface_ext._worker_host_mapped_region_close(handle)
            close_done[index].set()

        with ThreadPoolExecutor(max_workers=3) as executor:
            wait_future = executor.submit(wait_for_counter)
            deadline = time.monotonic() + 1.0
            while _task_interface_ext._worker_host_mapped_region_active_leases(handle) != 1:
                assert time.monotonic() < deadline, "counter wait never acquired its mapped-region lease"
                time.sleep(0.001)

            close_futures = [executor.submit(close_mapping, index) for index in range(2)]
            assert all(event.wait(1.0) for event in close_entered)
            assert not any(event.wait(0.05) for event in close_done), (
                "a concurrent close returned while a native operation still held the region"
            )

            cast(memoryview, shm.buf)[:4] = b"\x01\x00\x00\x00"
            assert wait_future.result(timeout=1.0) == (0, 0, 1, True, "")
            for close_future in close_futures:
                close_future.result(timeout=1.0)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(handle, 0, 1, int(WaitCmp.EQ))
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_sim_direct_transfer_failure_poisons_only_region(monkeypatch):
    worker, shm, _fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_payload_write",
            lambda _handle, _offset, _src, _nbytes: (_ for _ in ()).throw(RuntimeError("copy failed")),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        payload = wrap_fork_inherited(0x1234_0000, 16, mint_owner_instance_id(), 1, "L3")
        with pytest.raises(RuntimeError, match="copy failed"):
            region.payload_write(0, payload, nbytes=8)
        with pytest.raises(RuntimeError, match="poisoned"):
            region.descriptor_scalars()
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_cleanup_closes_worker_host_mapping_before_l2_host_release(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    events: list[tuple[str, int]] = []
    original_release = fake_c_worker.control_worker_chip_region_release

    def release(worker_id: int, region_id: int) -> None:
        events.append(("release", int(region_id)))
        original_release(worker_id, region_id)

    try:
        fake_c_worker.control_worker_chip_region_release = release
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 77
        )
        monkeypatch.setattr(
            worker_chip_orch_comm,
            "_worker_host_mapped_region_close",
            lambda handle: events.append(("close", int(handle))),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        region.free()
        worker._cleanup_worker_chip_regions()

        assert events == [("close", 77), ("release", 1)]
        with pytest.raises(RuntimeError, match="expired"):
            region.descriptor_scalars()
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_worker_region_payload_roundtrip(platform):
    try:
        RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")

    worker = Worker(
        level=3,
        device_ids=[0],
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        num_sub_workers=0,
    )
    worker.init()
    try:

        def orch(orch_handle, _args, _cfg):
            host = orch_handle.alloc([16], DataType.UINT8)
            buf_t = ctypes.c_uint8 * 16
            buf = buf_t.from_address(int(host.base))
            for i in range(16):
                buf[i] = (i + 41) & 0xFF
            region = orch_handle.create_worker_chip_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            region.payload_write(0, host)
            for i in range(16):
                buf[i] = 0
            region.payload_read(0, host)
            assert bytes(buf) == bytes((i + 41) & 0xFF for i in range(16))

        worker.run(orch)
    finally:
        worker.close()


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_worker_counter_wait_timeout_does_not_poison_region_and_free_is_idempotent(platform):
    try:
        RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")

    worker = Worker(
        level=3,
        device_ids=[0],
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        num_sub_workers=0,
    )
    worker.init()
    try:

        def orch(orch_handle, _args, _cfg):
            region = orch_handle.create_worker_chip_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            with pytest.raises(TimeoutError, match="observed=0"):
                region.counter(0).wait(1, WaitCmp.EQ, timeout=0.001)
            assert region.descriptor_scalars()[1] != 0
            region.free()
            region.free()

        worker.run(orch)
    finally:
        worker.close()
