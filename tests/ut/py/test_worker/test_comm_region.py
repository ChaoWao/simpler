# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the private region materializer."""

import dataclasses
from typing import Any, Optional, Union

import pytest
from simpler import comm_endpoints as ce
from simpler.comm_region import (
    MaterializationContext,
    MaterializationError,
    MaterializationRefusal,
    RefusalReason,
    RegionInstanceState,
    validate_single_owner_region_shape,
)
from simpler.orchestrator import _callback_frame_for, _callback_run
from simpler.worker import Worker, _Lifecycle, _RunResources
from simpler.worker_chip_orch_comm import NotifyOp, WaitCmp


def _ready(worker: Worker) -> Worker:
    worker._lifecycle = _Lifecycle.READY
    return worker


def _l3(device_ids=(0,)) -> Worker:
    worker = _ready(Worker(level=3, device_ids=list(device_ids)))
    worker._worker = object()
    return worker


def _l4_with_local_l3(device_ids=(0,)) -> Worker:
    child = Worker(level=3, device_ids=list(device_ids), num_sub_workers=0)
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_worker(child)
    return _ready(worker)


def _context(worker: Worker, members, topology, layout=None) -> MaterializationContext:
    layout = layout or ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128)
    registry = worker._get_endpoint_registry()
    resolved = registry.resolve_region_spec(members, topology)
    plan = ce.BackendResolver(registry, worker._get_region_access_service()).plan(resolved, layout)
    return MaterializationContext(
        worker=worker,
        registry=registry,
        plan=plan,
        layout=layout,
    )


def _accepted_context(worker: Optional[Worker] = None) -> MaterializationContext:
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


def _materialize_default_region(worker: Worker):
    return worker._materialize_region_instance(
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )


def _manual_two_member_context(
    worker: Worker,
    provider: ce.EndpointRecord,
    consumer: ce.EndpointRecord,
    layout: Optional[ce.RegionLayoutSpec] = None,
) -> MaterializationContext:
    layout = layout or ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128)
    registry = ce.EndpointRegistry(
        root_level=3,
        session_instance_id=worker._owner_instance_id,
        registry_epoch=worker._endpoint_registry_epoch,
        records=(consumer, provider),
    )
    provider_attachment = ce.MemberAttachmentPlan(
        member=provider.identity,
        role=ce.AttachmentRole.PROVIDER,
        adapter_kind=None,
        adapter_profile=None,
    )
    consumer_attachment = ce.MemberAttachmentPlan(
        member=consumer.identity,
        role=ce.AttachmentRole.CONSUMER,
        adapter_kind=ce.AdapterKind.OWNER_DELEGATED_COPY,
        adapter_profile=ce.AdapterProfile.HOST_VMM_COPY,
    )
    part_attachments = (provider_attachment, consumer_attachment)
    plan = ce.BackendPlan(
        ordered_members=(consumer.identity, provider.identity),
        payload=ce.RegionPartPlan(
            part=ce.RegionPartKind.PAYLOAD,
            backend_kind=ce.BackendKind.VMM_WINDOW,
            attachments=part_attachments,
        ),
        counter=ce.RegionPartPlan(
            part=ce.RegionPartKind.COUNTER,
            backend_kind=ce.BackendKind.VMM_WINDOW,
            attachments=part_attachments,
        ),
        topology_plan=ce.SingleOwnerPlan(provider_endpoint=provider.identity),
    )
    return MaterializationContext(worker=worker, registry=registry, plan=plan, layout=layout)


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


def test_shape_validation_rejects_non_l2_child_provider_path():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L1[0]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=0,
    )

    _assert_refusal(_manual_two_member_context(worker, provider, consumer), RefusalReason.NEEDS_DELEGATION)


def test_shape_validation_translates_worker_chip_id_value_error():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L2[1]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=0,
    )

    _assert_refusal(_manual_two_member_context(worker, provider, consumer), RefusalReason.UNSUPPORTED_MEMBER_SHAPE)


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
    decisions: dict[
        tuple[Any, ce.RegionPartKind, ce.AdapterKind, ce.AdapterProfile],
        Union[ce.RegionAccessDecision, bool],
    ] = {}
    for part in (ce.RegionPartKind.PAYLOAD, ce.RegionPartKind.COUNTER):
        decisions[
            (
                ce.BackendKind.VMM_WINDOW,
                part,
                ce.AdapterKind.OWNER_DELEGATED_COPY,
                ce.AdapterProfile.HOST_VMM_COPY,
            )
        ] = True
        decisions[
            (
                ce.BackendKind.VMM_WINDOW,
                part,
                ce.AdapterKind.DEVICE_PEER,
                ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
            )
        ] = True
    worker._region_access_service = ce.StaticRegionAccessService(decisions)
    extra = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICPU)),
    )
    _assert_refusal(extra, RefusalReason.UNSUPPORTED_MEMBER_SHAPE)


def test_shape_validation_rejects_remote_endpoint_member():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    remote_provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L2[0]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=1,
    )

    _assert_refusal(
        _manual_two_member_context(worker, remote_provider, consumer),
        RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
    )


def test_shape_validation_rejects_device_peer_consumer_attachment():
    ctx = _accepted_context()
    host = validate_single_owner_region_shape(ctx).consumer.identity
    payload_by_member = _attachments(ctx.plan.payload)
    payload_by_member[host] = dataclasses.replace(
        payload_by_member[host],
        adapter_kind=ce.AdapterKind.DEVICE_PEER,
        adapter_profile=ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
    )
    device_peer_payload = dataclasses.replace(ctx.plan.payload, attachments=tuple(payload_by_member.values()))

    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=device_peer_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )


def test_shape_validation_rejects_duplicate_attachment_members():
    ctx = _accepted_context()
    duplicate_payload = dataclasses.replace(
        ctx.plan.payload,
        attachments=ctx.plan.payload.attachments + (ctx.plan.payload.attachments[-1],),
    )

    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=duplicate_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )


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


class _FakeRegion:
    def __init__(self, calls: list[tuple], *, fail_mapping_close: bool = False) -> None:
        self._calls = calls
        self._fail_mapping_close = fail_mapping_close
        self._worker_id = 1
        self.region_id = 42
        self._expired = False
        self._released = False

    def payload_write(self, offset: int, host_buffer, nbytes=None) -> None:
        self._calls.append(("payload_write", offset, host_buffer, nbytes))

    def payload_read(self, offset: int, host_buffer, nbytes=None) -> None:
        self._calls.append(("payload_read", offset, host_buffer, nbytes))

    def counter(self, offset: int) -> _FakeCounter:
        self._calls.append(("counter", offset))
        return _FakeCounter(self._calls)

    def _close_worker_host_mapping(self) -> None:
        self._calls.append(("mapping_close", self._expired))
        if self._fail_mapping_close:
            raise RuntimeError("mapping close failed")

    def free(self) -> None:
        self._calls.append(("free",))
        self._released = True

    def _expire(self) -> None:
        self._calls.append(("expire",))
        self._expired = True


class _FakeNativeWorker:
    def __init__(self, calls: list[tuple], *, fail_release: bool = False) -> None:
        self._calls = calls
        self._fail_release = fail_release

    def control_worker_chip_region_release(self, worker_id: int, region_id: int) -> None:
        self._calls.append(("release", worker_id, region_id))
        if self._fail_release:
            raise RuntimeError("release failed")


@pytest.fixture
def region_worker(monkeypatch):
    def build(*, fail_mapping_close: bool = False, fail_release: bool = False, device_ids=(8, 9)):
        worker = _l3(device_ids=device_ids)
        calls: list[tuple] = []
        fake_region = _FakeRegion(calls, fail_mapping_close=fail_mapping_close)
        worker._worker = _FakeNativeWorker(calls, fail_release=fail_release)

        def create_region(worker_id: int, payload_bytes: int, counter_bytes: int):
            calls.append(("create", worker_id, payload_bytes, counter_bytes))
            worker._live_worker_chip_regions.append(fake_region)
            resources = worker._building_run_resources
            if resources is not None:
                resources.worker_chip_regions.append(fake_region)
                resources.requires_ordered_cleanup = True
            return fake_region

        monkeypatch.setattr(worker, "_create_worker_chip_region", create_region)
        return worker, calls, fake_region

    return build


def test_worker_materializes_region_instance_and_closes_single_region(region_worker):
    worker, calls, _fake_region = region_worker()
    instance = _materialize_default_region(worker)

    assert instance.state is RegionInstanceState.LIVE
    assert instance.worker_id == 1
    with worker._control_reservation("test_region_instance"):
        instance.payload_write(4, "src", nbytes=8)
        instance.payload_read(12, "dst")
        instance.counter(0).notify(3)
        instance.counter(8).wait(3, WaitCmp.GE, timeout=10)
        assert instance.counter(16).test(7, WaitCmp.EQ) is True

        instance.close()
        instance.close()
    assert instance.state is RegionInstanceState.CLOSED
    assert worker._live_worker_chip_regions == []
    assert calls == [
        ("create", 1, 64, 128),
        ("payload_write", 4, "src", 8),
        ("payload_read", 12, "dst", None),
        ("counter", 0),
        ("notify", 3, NotifyOp.Set),
        ("counter", 8),
        ("wait", 3, WaitCmp.GE, 10),
        ("counter", 16),
        ("test", 7, WaitCmp.EQ),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_live_region_instance_rollback_reuses_single_region_cleanup(region_worker):
    worker, calls, _fake_region = region_worker()
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        instance.rollback()
        instance.rollback()

    assert instance.state is RegionInstanceState.ROLLED_BACK
    assert worker._live_worker_chip_regions == []
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]
    with worker._control_reservation("test_region_instance"):
        instance.close()
    assert instance.state is RegionInstanceState.ROLLED_BACK
    with pytest.raises(MaterializationError, match="not live"):
        instance.payload_write(0, "src")


def test_region_instance_close_failure_marks_failed_and_poisons_worker(region_worker):
    worker, calls, _fake_region = region_worker(fail_mapping_close=True)
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        with pytest.raises(RuntimeError, match="mapping close failed"):
            instance.close()

    assert instance.state is RegionInstanceState.CLOSE_FAILED
    assert worker._live_worker_chip_regions == []
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")
    with pytest.raises(RuntimeError, match="not live"):
        instance.payload_write(0, "src")
    with pytest.raises(RuntimeError, match="mapping close failed"):
        instance.close()
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_region_instance_rollback_failure_replays_cached_error_for_close(region_worker):
    worker, calls, _fake_region = region_worker(fail_mapping_close=True)
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        with pytest.raises(RuntimeError, match="mapping close failed") as first_excinfo:
            instance.rollback()
        with pytest.raises(RuntimeError, match="mapping close failed") as second_excinfo:
            instance.rollback()
        with pytest.raises(RuntimeError, match="mapping close failed") as close_excinfo:
            instance.close()

    assert instance.state is RegionInstanceState.ROLLBACK_FAILED
    assert second_excinfo.value is first_excinfo.value
    assert close_excinfo.value is first_excinfo.value
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_close_worker_chip_region_preserves_release_error_cause(region_worker):
    worker, _calls, fake_region = region_worker(fail_mapping_close=True, fail_release=True)

    with pytest.raises(RuntimeError, match="mapping close failed") as excinfo:
        worker._close_worker_chip_region(fake_region, poison_on_error=True)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "release failed"


def test_callback_region_close_before_submit_retired_from_run_cleanup(region_worker):
    worker, calls, _fake_region = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with _callback_run(17, worker):
            instance = _materialize_default_region(worker)
            instance.close()
    finally:
        worker._building_run_resources = None

    assert instance.state is RegionInstanceState.CLOSED
    assert worker._live_worker_chip_regions == []
    assert resources.worker_chip_regions == []

    worker._cleanup_worker_chip_regions(resources)
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_callback_region_run_cleanup_then_later_close_is_idempotent(region_worker):
    worker, calls, _fake_region = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with _callback_run(18, worker):
            instance = _materialize_default_region(worker)
            frame = _callback_frame_for(worker)
            assert frame is not None
            frame.has_submitted_task = True
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                instance.close()
    finally:
        worker._building_run_resources = None

    worker._cleanup_worker_chip_regions(resources)

    assert instance.state is RegionInstanceState.LIVE
    assert worker._live_worker_chip_regions == []
    assert resources.worker_chip_regions == []

    with worker._control_reservation("test_region_instance"):
        instance.close()

    assert instance.state is RegionInstanceState.CLOSED
    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_retire_worker_chip_region_tracking_does_not_retry_base_exception():
    class FailingTrackingList(list):
        def __init__(self, values):
            super().__init__(values)
            self.assignments = 0

        def __setitem__(self, key, value):
            self.assignments += 1
            raise KeyboardInterrupt("tracking interrupted")

    worker = _l3(device_ids=[8, 9])
    region = object()
    failing_tracking = FailingTrackingList([region])
    succeeding_tracking = [region]
    resources = _RunResources(worker_chip_regions=failing_tracking)
    worker._live_worker_chip_regions = succeeding_tracking

    with pytest.raises(KeyboardInterrupt, match="tracking interrupted"):
        worker._retire_worker_chip_region_tracking(region, resources)

    assert failing_tracking.assignments == 1
    assert succeeding_tracking == []


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


def test_live_region_instance_access_requires_control_context(region_worker):
    worker, calls, _fake_region = region_worker()
    instance = _materialize_default_region(worker)

    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.payload_write(0, "src")
    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.counter(0)
    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.close()

    with worker._control_reservation("test_region_instance"):
        instance.close()

    assert calls == [
        ("create", 1, 64, 128),
        ("mapping_close", False),
        ("release", 1, 42),
        ("free",),
        ("expire",),
    ]


def test_shape_validation_rejects_foreign_registry_context():
    owner = _accepted_context(_l3(device_ids=[0, 1]))
    foreign_worker = _l3(device_ids=[0, 1])

    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(dataclasses.replace(owner, worker=foreign_worker))

    assert excinfo.value.reason is RefusalReason.REGISTRY_MISMATCH


def test_shape_validation_does_not_translate_worker_readiness_errors():
    ctx = _accepted_context()
    ctx.worker._worker = None

    with pytest.raises(RuntimeError, match="requires Worker.init"):
        validate_single_owner_region_shape(ctx)


def test_shape_validation_rejects_stale_registry_epoch():
    ctx = _accepted_context(_l3(device_ids=[0, 1]))
    ctx.worker._invalidate_endpoint_registry()

    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(ctx)

    assert excinfo.value.reason is RefusalReason.REGISTRY_MISMATCH
