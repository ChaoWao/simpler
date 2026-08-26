# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture the pre-W5a W4.5 outcome and Global CommDomain numeric baselines."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from simpler.buffer import BackendKind
from simpler.comm_endpoints import AdapterKind, AdapterProfile
from simpler.comm_provider import (
    ProviderCleanupFailure,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionCleanupCause,
    RegionControlErrorKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
    PosixShmImport,
    VmmShareableHandleImport,
)
from simpler.comm_provider_control import (
    ALLOCATE_REPLY_BYTES,
    RELEASE_REPLY_BYTES,
    encode_allocate_allocation_error_reply,
    encode_allocate_request_error_reply,
    encode_allocate_success_reply,
    encode_release_error_reply,
    encode_release_result_reply,
)
from simpler.global_comm_domain import _ADAPTER_KIND_IDS, _ADAPTER_PROFILE_IDS

from tests.ut.py.test_worker.w5a_migration_baseline import (
    FROZEN_ADAPTER_KIND_LE_U32,
    FROZEN_ADAPTER_KIND_U32,
    FROZEN_ADAPTER_PROFILE_LE_U32,
    FROZEN_ADAPTER_PROFILE_U32,
    L3_COMPATIBILITY_SIM_ONBOARD_CASES,
    L3_COMPATIBILITY_SIM_ONBOARD_PLATFORMS,
    L3_COMPATIBILITY_UT_MODULES,
    OCCUPIED_WORKER_CONTROL_COMMANDS,
    RETIRED_W4_5_CONTROL_COMMANDS,
    W4_5_ALLOCATE_ALLOCATION_ERROR_OUTCOMES,
    W4_5_ALLOCATE_OUTCOME_BYTES,
    W4_5_ALLOCATE_REPLY_PREFIX_BYTES,
    W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES,
    W4_5_ALLOCATE_SUCCESS_OUTCOME,
    W4_5_CTRL_REGION_ALLOCATE,
    W4_5_CTRL_REGION_RELEASE,
    W4_5_RELEASE_CLEANUP_INCOMPLETE_OUTCOMES,
    W4_5_RELEASE_CLEAN_OUTCOMES,
    W4_5_RELEASE_ERROR_OUTCOMES,
    W4_5_RELEASE_OUTCOME_BYTES,
    W4_5_RELEASE_REPLY_PREFIX_BYTES,
    W5A_CTRL_DELEGATED_REGION,
    W5A_UNKNOWN_TRANSACTION_OUTCOME,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _outcome(buf: bytearray, prefix: int) -> bytes:
    return bytes(buf[prefix:])


def _success_result() -> tuple[RegionAllocationResult, RegionPartLocalView, RegionPartLocalView]:
    result = RegionAllocationResult(
        provider_resource_id=11,
        export_descriptor=RegionExportDescriptor(
            payload=RegionPartExportDescriptor(
                planned_backing_kind=BackendKind.VMM_WINDOW,
                logical_bytes=64,
                mapping_bytes=64,
                import_capability=PosixShmImport(shm_name="/pto_payload_a"),
            ),
            counter=RegionPartExportDescriptor(
                planned_backing_kind=BackendKind.VMM_WINDOW,
                logical_bytes=8,
                mapping_bytes=8,
                import_capability=VmmShareableHandleImport(device_id=2, shareable_handle=21),
            ),
        ),
    )
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=64)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=8)
    return result, payload, counter


def test_frozen_outcome_sizes_match_w4_5_trailing_bytes():
    assert W4_5_ALLOCATE_REPLY_PREFIX_BYTES == 16
    assert W4_5_RELEASE_REPLY_PREFIX_BYTES == 16
    assert W4_5_ALLOCATE_OUTCOME_BYTES == ALLOCATE_REPLY_BYTES - W4_5_ALLOCATE_REPLY_PREFIX_BYTES
    assert W4_5_RELEASE_OUTCOME_BYTES == RELEASE_REPLY_BYTES - W4_5_RELEASE_REPLY_PREFIX_BYTES
    assert len(W4_5_ALLOCATE_SUCCESS_OUTCOME) == W4_5_ALLOCATE_OUTCOME_BYTES
    assert len(W5A_UNKNOWN_TRANSACTION_OUTCOME) == W4_5_RELEASE_OUTCOME_BYTES
    assert W5A_UNKNOWN_TRANSACTION_OUTCOME == bytes(W4_5_RELEASE_OUTCOME_BYTES)


def test_live_allocate_success_outcome_matches_frozen_vector():
    result, payload, counter = _success_result()
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_success_reply(buf, result, payload, counter)
    assert _outcome(buf, W4_5_ALLOCATE_REPLY_PREFIX_BYTES) == W4_5_ALLOCATE_SUCCESS_OUTCOME


@pytest.mark.parametrize("kind_name", list(W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES))
def test_live_allocate_request_error_outcomes_match_frozen_vectors(kind_name):
    kind = RegionControlErrorKind[kind_name]
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_request_error_reply(buf, kind)
    outcome = _outcome(buf, W4_5_ALLOCATE_REPLY_PREFIX_BYTES)
    assert outcome == W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES[kind_name]
    assert outcome[24:] == b"\x00" * (W4_5_ALLOCATE_OUTCOME_BYTES - 24)


@pytest.mark.parametrize(
    ("key", "error"),
    [
        (
            "backend_counter_zero_debt_0",
            RegionAllocationError(
                provisional_resource_id=7,
                control_kind=RegionControlErrorKind.BACKEND_FAILURE,
                failed_part=RegionPartKind.COUNTER,
                failed_operation=RegionOperationKind.ZERO_BYTES,
                cleanup_debt_remaining=False,
            ),
        ),
        (
            "backend_counter_zero_debt_1",
            RegionAllocationError(
                provisional_resource_id=7,
                control_kind=RegionControlErrorKind.BACKEND_FAILURE,
                failed_part=RegionPartKind.COUNTER,
                failed_operation=RegionOperationKind.ZERO_BYTES,
                cleanup_debt_remaining=True,
            ),
        ),
        (
            "internal_payload_materialize_debt_0",
            RegionAllocationError(
                provisional_resource_id=9,
                control_kind=RegionControlErrorKind.INTERNAL_INVARIANT,
                failed_part=RegionPartKind.PAYLOAD,
                failed_operation=RegionOperationKind.MATERIALIZE,
                cleanup_debt_remaining=False,
            ),
        ),
    ],
)
def test_live_allocate_allocation_error_outcomes_match_frozen_vectors(key, error):
    buf = bytearray(ALLOCATE_REPLY_BYTES)
    encode_allocate_allocation_error_reply(buf, error)
    outcome = _outcome(buf, W4_5_ALLOCATE_REPLY_PREFIX_BYTES)
    assert outcome == W4_5_ALLOCATE_ALLOCATION_ERROR_OUTCOMES[key]
    assert outcome[24:] == b"\x00" * (W4_5_ALLOCATE_OUTCOME_BYTES - 24)


@pytest.mark.parametrize("status_name", list(W4_5_RELEASE_CLEAN_OUTCOMES))
def test_live_clean_release_outcomes_match_frozen_vectors(status_name):
    buf = bytearray(RELEASE_REPLY_BYTES)
    encode_release_result_reply(
        buf,
        ProviderReleaseResult(provider_resource_id=13, status=ProviderReleaseStatus[status_name]),
    )
    assert _outcome(buf, W4_5_RELEASE_REPLY_PREFIX_BYTES) == W4_5_RELEASE_CLEAN_OUTCOMES[status_name]


@pytest.mark.parametrize(
    ("key", "failures"),
    [
        (
            "payload",
            (
                ProviderCleanupFailure(
                    part=RegionPartKind.PAYLOAD,
                    backend_operation=RegionOperationKind.RELEASE,
                    typed_cause=RegionCleanupCause.BACKEND_ERROR,
                ),
            ),
        ),
        (
            "counter",
            (
                ProviderCleanupFailure(
                    part=RegionPartKind.COUNTER,
                    backend_operation=RegionOperationKind.RELEASE,
                    typed_cause=RegionCleanupCause.INTERRUPTED,
                ),
            ),
        ),
        (
            "both",
            (
                ProviderCleanupFailure(
                    part=RegionPartKind.PAYLOAD,
                    backend_operation=RegionOperationKind.RELEASE,
                    typed_cause=RegionCleanupCause.BACKEND_ERROR,
                ),
                ProviderCleanupFailure(
                    part=RegionPartKind.COUNTER,
                    backend_operation=RegionOperationKind.ZERO_BYTES,
                    typed_cause=RegionCleanupCause.BACKEND_STATE_MISMATCH,
                ),
            ),
        ),
    ],
)
def test_live_cleanup_incomplete_outcomes_match_frozen_vectors(key, failures):
    buf = bytearray(RELEASE_REPLY_BYTES)
    encode_release_result_reply(
        buf,
        ProviderReleaseResult(
            provider_resource_id=13,
            status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
            failures=failures,
        ),
    )
    assert _outcome(buf, W4_5_RELEASE_REPLY_PREFIX_BYTES) == W4_5_RELEASE_CLEANUP_INCOMPLETE_OUTCOMES[key]


@pytest.mark.parametrize(
    ("key", "kind", "resource_id"),
    [
        ("INVALID_FIELD_VALUE", RegionControlErrorKind.INVALID_FIELD_VALUE, 13),
        ("INVALID_FIELD_VALUE_id0", RegionControlErrorKind.INVALID_FIELD_VALUE, 0),
        ("STORE_LIFECYCLE", RegionControlErrorKind.STORE_LIFECYCLE, 13),
        ("STORE_LIFECYCLE_id0", RegionControlErrorKind.STORE_LIFECYCLE, 0),
        ("INTERNAL_INVARIANT", RegionControlErrorKind.INTERNAL_INVARIANT, 13),
        ("INTERNAL_INVARIANT_id0", RegionControlErrorKind.INTERNAL_INVARIANT, 0),
    ],
)
def test_live_release_error_outcomes_match_frozen_vectors(key, kind, resource_id):
    buf = bytearray(RELEASE_REPLY_BYTES)
    encode_release_error_reply(buf, kind, provider_resource_id=resource_id)
    assert _outcome(buf, W4_5_RELEASE_REPLY_PREFIX_BYTES) == W4_5_RELEASE_ERROR_OUTCOMES[key]


def test_global_comm_domain_adapter_numeric_bytes_match_frozen_baseline():
    live_kinds = {None: 0, **{kind.name: value for kind, value in _ADAPTER_KIND_IDS.items()}}
    live_profiles = {None: 0, **{profile.name: value for profile, value in _ADAPTER_PROFILE_IDS.items()}}
    assert live_kinds == FROZEN_ADAPTER_KIND_U32
    assert live_profiles == FROZEN_ADAPTER_PROFILE_U32
    for kind in AdapterKind:
        value = FROZEN_ADAPTER_KIND_U32[kind.name]
        assert value.to_bytes(4, "little") == FROZEN_ADAPTER_KIND_LE_U32[kind.name]
    for profile in AdapterProfile:
        value = FROZEN_ADAPTER_PROFILE_U32[profile.name]
        assert value.to_bytes(4, "little") == FROZEN_ADAPTER_PROFILE_LE_U32[profile.name]


def test_control_command_26_is_delegated_region_and_16_17_remain_compat():
    worker_source = (_REPO_ROOT / "python" / "simpler" / "worker.py").read_text(encoding="utf-8")
    source = ast.parse(worker_source)
    assigned: dict[str, int] = {}
    for node in source.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("_CTRL_"):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            assigned[target.id] = node.value.value

    assert assigned["_CTRL_REGION_ALLOCATE"] == W4_5_CTRL_REGION_ALLOCATE
    assert assigned["_CTRL_REGION_RELEASE"] == W4_5_CTRL_REGION_RELEASE
    assert assigned["_CTRL_DELEGATED_REGION"] == W5A_CTRL_DELEGATED_REGION
    assert RETIRED_W4_5_CONTROL_COMMANDS == (16, 17)
    command_ids = {
        name: value
        for name, value in assigned.items()
        if name
        in {
            "_CTRL_MALLOC",
            "_CTRL_FREE",
            "_CTRL_COPY_TO",
            "_CTRL_COPY_FROM",
            "_CTRL_PREPARE",
            "_CTRL_REGISTER",
            "_CTRL_UNREGISTER",
            "_CTRL_ALLOC_DOMAIN",
            "_CTRL_RELEASE_DOMAIN",
            "_CTRL_COMM_INIT",
            "_CTRL_PY_REGISTER",
            "_CTRL_PY_UNREGISTER",
            "_CTRL_PY_IMPORT_REGISTER",
            "_CTRL_IMPORT_RELEASE",
            "_CTRL_REGION_ALLOCATE",
            "_CTRL_REGION_RELEASE",
            "_CTRL_COMMITTED_DEVICE_MEMORY",
            "_CTRL_GLOBAL_DOMAIN_NODE",
            "_CTRL_DEVICE_MEMORY_INFO",
            "_CTRL_DELEGATED_REGION",
        }
    }
    assert set(command_ids.values()) == OCCUPIED_WORKER_CONTROL_COMMANDS
    assert assigned["_CTRL_REGION_ALLOCATE"] == 16
    assert assigned["_CTRL_REGION_RELEASE"] == 17


def test_l3_compatibility_baseline_paths_exist_with_declared_platforms():
    for relative in (*L3_COMPATIBILITY_UT_MODULES, *L3_COMPATIBILITY_SIM_ONBOARD_CASES):
        path = _REPO_ROOT / relative
        assert path.is_file(), relative
    for relative in L3_COMPATIBILITY_SIM_ONBOARD_CASES:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for platform in L3_COMPATIBILITY_SIM_ONBOARD_PLATFORMS:
            assert f'"{platform}"' in text, f"{relative} missing {platform}"
