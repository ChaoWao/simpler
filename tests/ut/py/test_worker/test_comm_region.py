# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the private W3.5 region materializer."""

import dataclasses

import pytest
from simpler import comm_endpoints as ce
from simpler.comm_region import (
    MaterializationContext,
    MaterializationRefusal,
    RefusalReason,
    RegionInstanceState,
    validate_single_owner_region_shape,
)
from simpler.worker import Worker, _Lifecycle
from simpler.worker_chip_orch_comm import NotifyOp, WaitCmp


def _ready(worker: Worker) -> Worker:
    worker._lifecycle = _Lifecycle.READY
    return worker


def _l3(device_ids=(0,)) -> Worker:
    return _ready(Worker(level=3, device_ids=list(device_ids)))


def _l4_with_local_l3(device_ids=(0,)) -> Worker:
    child = Worker(level=3, device_ids=list(device_ids), num_sub_workers=0)
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_worker(child)
    return _ready(worker)


def _context(worker: Worker, members, topology, layout=None) -> MaterializationContext:
    registry = worker._get_endpoint_registry()
    resolved = registry.resolve_region_spec(members, topology)
    plan = ce.BackendResolver(registry, worker._get_region_access_service()).plan(
        resolved,
        layout or ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )
    return MaterializationContext(
        worker=worker,
        registry=registry,
        plan=plan,
        layout=layout or ce.RegionLayoutSpec(64, 128),
    )


def _accepted_context(worker: Worker | None = None) -> MaterializationContext:
    worker = worker or _l3(device_ids=[8, 9])
    return _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
    )


def _assert_refusal(ctx: MaterializationContext, reason: RefusalReason) -> None:
    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(ctx)
    assert excinfo.value.reason is reason


def _attachments(part: ce.RegionPartPlan) -> dict[ce.EndpointIdentity, ce.MemberAttachmentPlan]:
    return {attachment.member: attachment for attachment in part.attachments}


def test_shape_validation_accepts_l3_host_to_local_l2_aicpu_copy_plan():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)

    assert shape.consumer.path == "L3"
    assert shape.consumer.deployment is ce.HOST_CPU
    assert shape.provider.path == "L3/L2[1]"
    assert shape.provider.deployment is ce.DEVICE_AICPU
    assert shape.worker_id == 1


def test_shape_validation_rejects_l4_plan_as_delegation_work():
    worker = _l4_with_local_l3(device_ids=[4])
    ctx = _context(
        worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)),
    )

    _assert_refusal(ctx, RefusalReason.NEEDS_DELEGATION)


def test_shape_validation_rejects_aicore_provider_until_it_has_a_materializer():
    worker = _l3(device_ids=[0])
    ctx = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICORE)),
    )

    _assert_refusal(ctx, RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT)


def test_shape_validation_rejects_unsupported_plan_and_non_vmm_backing():
    unsupported = _context(
        _l3(device_ids=[0]),
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3", ce.HOST_CPU)),
    )
    _assert_refusal(unsupported, RefusalReason.UNSUPPORTED_PLAN)

    ctx = _accepted_context()
    posix_payload = dataclasses.replace(ctx.plan.payload, backend_kind=ce.BackendKind.POSIX_SHM)
    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=posix_payload)),
        RefusalReason.UNSUPPORTED_BACKEND_KIND,
    )


def test_shape_validation_rejects_direct_map_and_extra_consumers():
    ctx = _accepted_context()
    host = validate_single_owner_region_shape(ctx).consumer.identity
    payload_by_member = _attachments(ctx.plan.payload)
    payload_by_member[host] = dataclasses.replace(
        payload_by_member[host],
        adapter_kind=ce.AdapterKind.DIRECT_MAP,
        adapter_profile=ce.AdapterProfile.HOST_SVM_MAP,
    )
    direct_map_payload = dataclasses.replace(ctx.plan.payload, attachments=tuple(payload_by_member.values()))
    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=direct_map_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )

    worker = _l3(device_ids=[0])
    worker._region_access_service = ce.StaticRegionAccessService(
        {
            (
                ce.BackendKind.VMM_WINDOW,
                part,
                ce.AdapterKind.OWNER_DELEGATED_COPY,
                ce.AdapterProfile.HOST_VMM_COPY,
            ): True
            for part in (ce.RegionPartKind.PAYLOAD, ce.RegionPartKind.COUNTER)
        }
        | {
            (
                ce.BackendKind.VMM_WINDOW,
                part,
                ce.AdapterKind.DEVICE_PEER,
                ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
            ): True
            for part in (ce.RegionPartKind.PAYLOAD, ce.RegionPartKind.COUNTER)
        }
    )
    extra = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICPU)),
    )
    _assert_refusal(extra, RefusalReason.UNSUPPORTED_MEMBER_SHAPE)


class _FakeCounter:
    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def notify(self, value: int, op=NotifyOp.Set) -> None:
        self._calls.append(("notify", value, op))

    def test(self, cmp_value: int, cmp: WaitCmp):
        self._calls.append(("test", cmp_value, cmp))
        return True

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: int):
        self._calls.append(("wait", cmp_value, cmp, timeout))
        return None


class _FakeRegion:
    def __init__(self, calls: list[tuple], *, fail_mapping_close: bool = False) -> None:
        self._calls = calls
        self._fail_mapping_close = fail_mapping_close
        self._worker_id = 1
        self.region_id = 42

    def payload_write(self, offset: int, host_buffer, nbytes=None) -> None:
        self._calls.append(("payload_write", offset, host_buffer, nbytes))

    def payload_read(self, offset: int, host_buffer, nbytes=None) -> None:
        self._calls.append(("payload_read", offset, host_buffer, nbytes))

    def counter(self, offset: int) -> _FakeCounter:
        self._calls.append(("counter", offset))
        return _FakeCounter(self._calls)

    def _close_worker_host_mapping(self) -> None:
        self._calls.append(("mapping_close",))
        if self._fail_mapping_close:
            raise RuntimeError("mapping close failed")

    def _expire(self) -> None:
        self._calls.append(("expire",))


class _FakeNativeWorker:
    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def control_worker_chip_region_release(self, worker_id: int, region_id: int) -> None:
        self._calls.append(("release", worker_id, region_id))


def test_worker_materializes_region_instance_and_closes_single_region(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    calls: list[tuple] = []
    fake_region = _FakeRegion(calls)
    worker._worker = _FakeNativeWorker(calls)

    def create_region(worker_id: int, payload_bytes: int, counter_bytes: int):
        calls.append(("create", worker_id, payload_bytes, counter_bytes))
        worker._live_worker_chip_regions.append(fake_region)
        return fake_region

    monkeypatch.setattr(worker, "_create_worker_chip_region", create_region)

    instance = worker._materialize_region_instance(
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )

    assert instance.state is RegionInstanceState.LIVE
    assert instance.worker_id == 1
    instance.payload_write(4, "src", nbytes=8)
    instance.payload_read(12, "dst")
    assert instance.counter(16).test(7, WaitCmp.EQ) is True

    instance.close()
    instance.close()
    assert instance.state is RegionInstanceState.CLOSED
    assert worker._live_worker_chip_regions == []
    assert calls == [
        ("create", 1, 64, 128),
        ("payload_write", 4, "src", 8),
        ("payload_read", 12, "dst", None),
        ("counter", 16),
        ("test", 7, WaitCmp.EQ),
        ("mapping_close",),
        ("release", 1, 42),
        ("expire",),
    ]


def test_live_region_instance_rollback_reuses_single_region_cleanup(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    calls: list[tuple] = []
    fake_region = _FakeRegion(calls)
    worker._worker = _FakeNativeWorker(calls)

    def create_region(worker_id: int, payload_bytes: int, counter_bytes: int):
        calls.append(("create", worker_id, payload_bytes, counter_bytes))
        worker._live_worker_chip_regions.append(fake_region)
        return fake_region

    monkeypatch.setattr(worker, "_create_worker_chip_region", create_region)
    instance = worker._materialize_region_instance(
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )

    instance.rollback()
    instance.rollback()

    assert instance.state is RegionInstanceState.ROLLED_BACK
    assert worker._live_worker_chip_regions == []
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close",),
        ("release", 1, 42),
        ("expire",),
    ]
    with pytest.raises(RuntimeError, match="not live"):
        instance.payload_write(0, "src")


def test_region_instance_close_failure_marks_failed_and_keeps_tracking(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    calls: list[tuple] = []
    fake_region = _FakeRegion(calls, fail_mapping_close=True)
    worker._worker = _FakeNativeWorker(calls)

    def create_region(worker_id: int, payload_bytes: int, counter_bytes: int):
        calls.append(("create", worker_id, payload_bytes, counter_bytes))
        worker._live_worker_chip_regions.append(fake_region)
        return fake_region

    monkeypatch.setattr(worker, "_create_worker_chip_region", create_region)
    instance = worker._materialize_region_instance(
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )

    with pytest.raises(RuntimeError, match="mapping close failed"):
        instance.close()

    assert instance.state is RegionInstanceState.CLOSE_FAILED
    assert worker._live_worker_chip_regions == [fake_region]
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close",),
        ("release", 1, 42),
    ]


def test_materialize_create_failure_propagates_without_live_tracking(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    calls: list[tuple] = []

    def create_region(worker_id: int, payload_bytes: int, counter_bytes: int):
        calls.append(("create", worker_id, payload_bytes, counter_bytes))
        raise RuntimeError("create failed")

    monkeypatch.setattr(worker, "_create_worker_chip_region", create_region)

    with pytest.raises(RuntimeError, match="create failed"):
        worker._materialize_region_instance(
            [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
            ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
            ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
        )

    assert worker._live_worker_chip_regions == []
    assert calls == [("create", 1, 64, 128)]
