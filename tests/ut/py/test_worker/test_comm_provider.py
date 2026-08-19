# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for neutral comm-provider values and typed errors."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import simpler.comm_provider as comm_provider_module
from simpler.buffer import BackendKind
from simpler.comm_provider import (
    POSIX_SHM_TOKEN_MAX_BYTES,
    DeviceAllocationTarget,
    HostAllocationTarget,
    PosixShmImport,
    ProviderCleanupFailure,
    ProviderPartResourceState,
    ProviderRegionResourceState,
    ProviderRegionStoreState,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationContext,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionControlError,
    RegionControlErrorKind,
    RegionCleanupCause,
    RegionEnvironmentKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartAllocationSpec,
    RegionPartExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
    VmmShareableHandleImport,
    validate_independent_local_views,
)

_COMM_PROVIDER_PATH = Path(comm_provider_module.__file__).resolve()
_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1


def _payload_spec(logical_bytes: int = 64, backing=BackendKind.VMM_WINDOW) -> RegionPartAllocationSpec:
    return RegionPartAllocationSpec(planned_backing_kind=backing, logical_bytes=logical_bytes)


def _counter_spec(logical_bytes: int = 8, backing=BackendKind.VMM_WINDOW) -> RegionPartAllocationSpec:
    return RegionPartAllocationSpec(planned_backing_kind=backing, logical_bytes=logical_bytes)


def _posix_export(logical_bytes: int, shm_name: str, *, mapping_bytes: int | None = None) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes if mapping_bytes is None else mapping_bytes,
        import_capability=PosixShmImport(shm_name=shm_name),
    )


def _vmm_export(logical_bytes: int, handle: int = 7) -> RegionPartExportDescriptor:
    return RegionPartExportDescriptor(
        planned_backing_kind=BackendKind.VMM_WINDOW,
        logical_bytes=logical_bytes,
        mapping_bytes=logical_bytes,
        import_capability=VmmShareableHandleImport(device_id=0, shareable_handle=handle),
    )


def test_comm_provider_module_does_not_import_worker_chip_or_w5a():
    tree = ast.parse(_COMM_PROVIDER_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_substrings = (
        "worker_chip",
        "worker.py",
        "mailbox",
        "w5a",
        "orchestrator",
        "comm_region",
    )
    for name in imported:
        lowered = name.replace("_", "").lower()
        assert "workerchip" not in lowered
        assert not any(token in name for token in forbidden_substrings)
        assert name not in {"simpler.worker", "worker"}


@pytest.mark.parametrize(
    ("enum_cls", "name", "value"),
    [
        (RegionPartKind, "INVALID", 0),
        (RegionPartKind, "PAYLOAD", 1),
        (RegionPartKind, "COUNTER", 2),
        (RegionControlErrorKind, "NONE", 0),
        (RegionControlErrorKind, "BAD_MAGIC_VERSION", 1),
        (RegionControlErrorKind, "BAD_MESSAGE_SIZE", 2),
        (RegionControlErrorKind, "INVALID_ENUM_VALUE", 3),
        (RegionControlErrorKind, "RESERVED_NONZERO", 4),
        (RegionControlErrorKind, "INVALID_FIELD_VALUE", 5),
        (RegionControlErrorKind, "STORE_LIFECYCLE", 6),
        (RegionControlErrorKind, "INTERNAL_INVARIANT", 7),
        (RegionControlErrorKind, "BACKEND_FAILURE", 8),
        (RegionOperationKind, "NONE", 0),
        (RegionOperationKind, "MATERIALIZE", 1),
        (RegionOperationKind, "ZERO_BYTES", 2),
        (RegionOperationKind, "DESCRIBE", 3),
        (RegionOperationKind, "LOCAL_VIEW", 4),
        (RegionOperationKind, "RELEASE", 5),
        (RegionCleanupCause, "NONE", 0),
        (RegionCleanupCause, "BACKEND_ERROR", 1),
        (RegionCleanupCause, "INTERRUPTED", 2),
        (RegionCleanupCause, "BACKEND_STATE_MISMATCH", 3),
        (ProviderReleaseStatus, "RELEASED", 1),
        (ProviderReleaseStatus, "ALREADY_GONE", 2),
        (ProviderReleaseStatus, "UNKNOWN_RESOURCE", 3),
        (ProviderReleaseStatus, "CLEANUP_INCOMPLETE", 4),
    ],
)
def test_typed_code_values_match_design_tables(enum_cls, name, value):
    member = enum_cls[name]
    assert int(member) == value
    assert enum_cls(value) is member


@pytest.mark.parametrize(
    "enum_cls",
    [
        RegionPartKind,
        RegionControlErrorKind,
        RegionOperationKind,
        RegionCleanupCause,
        ProviderReleaseStatus,
        BackendKind,
    ],
)
def test_unknown_nonzero_enum_values_are_rejected(enum_cls):
    with pytest.raises(ValueError):
        enum_cls(99)


def test_state_enum_names_are_exactly_the_accepted_set():
    assert [state.name for state in ProviderRegionStoreState] == [
        "OPEN",
        "CLOSING",
        "CLOSE_FAILED",
        "CLOSED",
    ]
    assert [state.name for state in ProviderRegionResourceState] == [
        "CREATING",
        "ACTIVE",
        "CLEANUP_PENDING",
    ]
    assert [state.name for state in ProviderPartResourceState] == [
        "SHELL",
        "MATERIALIZING",
        "READY",
        "CLEANUP_PENDING",
        "RELEASED",
    ]


def test_allocation_spec_accepts_independent_part_sizes_and_backing_kinds():
    spec = RegionAllocationSpec(
        payload=_payload_spec(96, BackendKind.VMM_WINDOW),
        counter=_counter_spec(12, BackendKind.POSIX_SHM),
    )
    assert spec.part(RegionPartKind.PAYLOAD).logical_bytes == 96
    assert spec.part(RegionPartKind.COUNTER).logical_bytes == 12
    assert spec.payload.planned_backing_kind is BackendKind.VMM_WINDOW
    assert spec.counter.planned_backing_kind is BackendKind.POSIX_SHM


@pytest.mark.parametrize("logical_bytes", [0, -1, _UINT64_MAX + 1, True])
def test_part_spec_rejects_non_positive_or_overflow_sizes(logical_bytes):
    with pytest.raises((ValueError, TypeError)):
        RegionPartAllocationSpec(planned_backing_kind=BackendKind.VMM_WINDOW, logical_bytes=logical_bytes)


@pytest.mark.parametrize("logical_bytes", [1, 2, 3, 5, 6, 7, 9])
def test_counter_spec_requires_multiple_of_four(logical_bytes):
    payload = _payload_spec()
    counter = RegionPartAllocationSpec(planned_backing_kind=BackendKind.VMM_WINDOW, logical_bytes=logical_bytes)
    with pytest.raises(ValueError, match="multiple of 4"):
        RegionAllocationSpec(payload=payload, counter=counter)


def test_allocation_context_preserves_typed_targets():
    sim = RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.SIM,
        target=DeviceAllocationTarget(device_id=3),
    )
    reserved = RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.ONBOARD,
        target=HostAllocationTarget(),
    )
    assert sim.target.device_id == 3
    assert isinstance(reserved.target, HostAllocationTarget)
    with pytest.raises(ValueError):
        DeviceAllocationTarget(device_id=_INT32_MAX + 1)


def test_export_descriptor_keeps_capability_and_omits_local_addresses():
    descriptor = RegionExportDescriptor(
        payload=_posix_export(64, "/pto_payload_a"),
        counter=_vmm_export(8, handle=11),
    )
    assert dataclasses.fields(RegionPartExportDescriptor)
    field_names = {field.name for field in dataclasses.fields(RegionPartExportDescriptor)}
    assert "local_base" not in field_names
    assert "local_addr" not in field_names
    assert isinstance(descriptor.payload.import_capability, PosixShmImport)
    assert isinstance(descriptor.counter.import_capability, VmmShareableHandleImport)
    result = RegionAllocationResult(provider_resource_id=1, export_descriptor=descriptor)
    assert result.provider_resource_id == 1
    with pytest.raises(ValueError):
        RegionAllocationResult(provider_resource_id=0, export_descriptor=descriptor)


@pytest.mark.parametrize("shm_name", ["", "a" * (POSIX_SHM_TOKEN_MAX_BYTES + 1), "npu\u4e00", "name\x00x"])
def test_posix_shm_token_rejects_empty_overlong_non_ascii_and_nul(shm_name):
    with pytest.raises((ValueError, TypeError)):
        PosixShmImport(shm_name=shm_name)


def test_posix_shm_token_accepts_bounded_ascii_and_mapping_may_exceed_logical_bytes():
    token = "p" * POSIX_SHM_TOKEN_MAX_BYTES
    export = _posix_export(8, token, mapping_bytes=64)
    assert export.mapping_bytes == 64
    with pytest.raises(ValueError):
        _posix_export(16, "ok", mapping_bytes=15)


def test_local_view_accepts_base_zero_and_rejects_uint64_span_overflow():
    view = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=8)
    assert view.local_base == 0
    with pytest.raises(ValueError, match="overflowed uint64"):
        RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=_UINT64_MAX, logical_bytes=1)
    with pytest.raises(ValueError, match="64-byte aligned"):
        RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=32, logical_bytes=8)


def test_independent_span_rules_allow_noncontiguous_nonoverlapping_bases():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=100)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=8)
    validate_independent_local_views(payload, counter)


def test_independent_span_rules_reject_overlap_and_swapped_parts():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=128)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=64, logical_bytes=8)
    with pytest.raises(ValueError, match="must not overlap"):
        validate_independent_local_views(payload, counter)
    with pytest.raises(ValueError, match="part PAYLOAD"):
        validate_independent_local_views(counter, payload)


def test_adjacent_spans_do_not_overlap():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=64)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=64, logical_bytes=8)
    validate_independent_local_views(payload, counter)


def test_release_result_allows_at_most_one_failure_per_part():
    payload_failure = ProviderCleanupFailure(
        part=RegionPartKind.PAYLOAD,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.BACKEND_ERROR,
    )
    counter_failure = ProviderCleanupFailure(
        part=RegionPartKind.COUNTER,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.INTERRUPTED,
    )
    result = ProviderReleaseResult(
        provider_resource_id=9,
        status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
        failures=(payload_failure, counter_failure),
    )
    assert len(result.failures) == 2
    with pytest.raises(ValueError, match="at most one failure per part"):
        ProviderReleaseResult(
            provider_resource_id=9,
            status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
            failures=(payload_failure, payload_failure),
        )
    with pytest.raises(ValueError):
        ProviderReleaseResult(
            provider_resource_id=9,
            status=ProviderReleaseStatus.RELEASED,
            failures=(payload_failure,),
        )


def test_allocation_error_carries_debt_bit_and_no_cleanup_cause():
    error = RegionAllocationError(
        provisional_resource_id=4,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.COUNTER,
        failed_operation=RegionOperationKind.ZERO_BYTES,
        cleanup_debt_remaining=True,
    )
    assert error.provisional_resource_id == 4
    assert error.cleanup_debt_remaining is True
    assert error.failed_part is RegionPartKind.COUNTER
    assert error.failed_operation is RegionOperationKind.ZERO_BYTES
    assert not hasattr(error, "typed_cause")
    assert not hasattr(error, "cleanup_cause")
    with pytest.raises(ValueError):
        RegionAllocationError(
            provisional_resource_id=0,
            control_kind=RegionControlErrorKind.BACKEND_FAILURE,
            failed_part=RegionPartKind.PAYLOAD,
            failed_operation=RegionOperationKind.MATERIALIZE,
            cleanup_debt_remaining=False,
        )
    with pytest.raises(ValueError):
        RegionAllocationError(
            provisional_resource_id=1,
            control_kind=RegionControlErrorKind.BAD_MESSAGE_SIZE,
            failed_part=RegionPartKind.PAYLOAD,
            failed_operation=RegionOperationKind.MATERIALIZE,
            cleanup_debt_remaining=False,
        )


def test_control_error_rejects_none_kind_and_unknown_values():
    error = RegionControlError(RegionControlErrorKind.STORE_LIFECYCLE, "store closed")
    assert error.kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert error.failed_part is RegionPartKind.INVALID
    with pytest.raises(ValueError):
        RegionControlError(RegionControlErrorKind.NONE)
    with pytest.raises(ValueError):
        RegionControlError(99)
