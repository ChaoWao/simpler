# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frozen W4.5 / Global CommDomain numeric baselines for the W5a migration.

These vectors are the pre-W5a wire facts. W5a runtime must not import the old
provider-control codec; later codec tests compare new outcomes against this
module instead.
"""

from __future__ import annotations

W4_5_ALLOCATE_REPLY_PREFIX_BYTES = 16
W4_5_RELEASE_REPLY_PREFIX_BYTES = 16
W4_5_ALLOCATE_OUTCOME_BYTES = 216
W4_5_RELEASE_OUTCOME_BYTES = 32
W5A_ALLOCATE_REPLY_HEADER_BYTES = 40
W5A_RELEASE_REPLY_HEADER_BYTES = 40

W4_5_CTRL_REGION_ALLOCATE = 16
W4_5_CTRL_REGION_RELEASE = 17
W5A_CTRL_DELEGATED_REGION = 26
RETIRED_W4_5_CONTROL_COMMANDS = (W4_5_CTRL_REGION_ALLOCATE, W4_5_CTRL_REGION_RELEASE)

# Occupied Worker control_payload sub-commands after delegated-region lands command 26.
OCCUPIED_WORKER_CONTROL_COMMANDS = frozenset(
    {
        0,  # _CTRL_MALLOC
        1,  # _CTRL_FREE
        2,  # _CTRL_COPY_TO
        3,  # _CTRL_COPY_FROM
        4,  # _CTRL_PREPARE
        5,  # _CTRL_REGISTER
        6,  # _CTRL_UNREGISTER
        7,  # _CTRL_ALLOC_DOMAIN
        8,  # _CTRL_RELEASE_DOMAIN
        9,  # _CTRL_COMM_INIT
        10,  # _CTRL_PY_REGISTER
        11,  # _CTRL_PY_UNREGISTER
        12,  # _CTRL_PY_IMPORT_REGISTER
        13,  # _CTRL_IMPORT_RELEASE
        16,  # _CTRL_REGION_ALLOCATE
        17,  # _CTRL_REGION_RELEASE
        18,  # _CTRL_COMMITTED_DEVICE_MEMORY
        24,  # _CTRL_GLOBAL_DOMAIN_NODE
        25,  # _CTRL_DEVICE_MEMORY_INFO
        26,  # _CTRL_DELEGATED_REGION
    }
)

# Canonical SUCCESS fixture: resource 11, POSIX payload "/pto_payload_a" (64 B,
# local_base 0x1000) and VMM counter handle 21 on device 2 (8 B, local_base 0x2000).
W4_5_ALLOCATE_SUCCESS_OUTCOME = bytes.fromhex(
    "0b0000000000000000000000000000000000000000000000"
    "0200000002000000400000000000000040000000000000000e00000000000000"
    "2f70746f5f7061796c6f61645f61000000000000000000000000000000000000"
    "0000000000000000020000000100000008000000000000000800000000000000"
    "1000000000000000020000000000000015000000000000000000000000000000"
    "0000000000000000000000000000000001000000000000000010000000000000"
    "4000000000000000020000000000000000200000000000000800000000000000"
)

_ALLOCATE_INACTIVE_TAIL_HEX = "00" * (W4_5_ALLOCATE_OUTCOME_BYTES - 24)

W4_5_ALLOCATE_REQUEST_ERROR_OUTCOMES = {
    "BAD_MAGIC_VERSION": bytes.fromhex("000000000000000001000000000000000000000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX),
    "BAD_MESSAGE_SIZE": bytes.fromhex("000000000000000002000000000000000000000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX),
    "INVALID_ENUM_VALUE": bytes.fromhex("000000000000000003000000000000000000000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX),
    "RESERVED_NONZERO": bytes.fromhex("000000000000000004000000000000000000000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX),
    "INVALID_FIELD_VALUE": bytes.fromhex("000000000000000005000000000000000000000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX),
}

W4_5_ALLOCATE_ALLOCATION_ERROR_OUTCOMES = {
    "backend_counter_zero_debt_0": bytes.fromhex(
        "070000000000000008000000020000000200000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX
    ),
    "backend_counter_zero_debt_1": bytes.fromhex(
        "070000000000000008000000020000000200000001000000" + _ALLOCATE_INACTIVE_TAIL_HEX
    ),
    "internal_payload_materialize_debt_0": bytes.fromhex(
        "090000000000000007000000010000000100000000000000" + _ALLOCATE_INACTIVE_TAIL_HEX
    ),
}

W4_5_RELEASE_CLEAN_OUTCOMES = {
    "RELEASED": bytes.fromhex("0d00000000000000000000000000000000000000000000000000000000000000"),
    "ALREADY_GONE": bytes.fromhex("0d00000000000000000000000000000000000000000000000000000000000000"),
    "UNKNOWN_RESOURCE": bytes.fromhex("0d00000000000000000000000000000000000000000000000000000000000000"),
}

W4_5_RELEASE_CLEANUP_INCOMPLETE_OUTCOMES = {
    "payload": bytes.fromhex("0d00000000000000010000000000000005000000010000000000000000000000"),
    "counter": bytes.fromhex("0d00000000000000020000000000000000000000000000000500000002000000"),
    "both": bytes.fromhex("0d00000000000000030000000000000005000000010000000200000003000000"),
}

W4_5_RELEASE_ERROR_OUTCOMES = {
    "INVALID_FIELD_VALUE": bytes.fromhex("0d00000000000000000000000500000000000000000000000000000000000000"),
    "INVALID_FIELD_VALUE_id0": bytes.fromhex("0000000000000000000000000500000000000000000000000000000000000000"),
    "STORE_LIFECYCLE": bytes.fromhex("0d00000000000000000000000600000000000000000000000000000000000000"),
    "STORE_LIFECYCLE_id0": bytes.fromhex("0000000000000000000000000600000000000000000000000000000000000000"),
    "INTERNAL_INVARIANT": bytes.fromhex("0d00000000000000000000000700000000000000000000000000000000000000"),
    "INTERNAL_INVARIANT_id0": bytes.fromhex("0000000000000000000000000700000000000000000000000000000000000000"),
}

# W5a-only release tag. Missing table identity always uses this all-zero outcome.
W5A_UNKNOWN_TRANSACTION_OUTCOME = bytes(W4_5_RELEASE_OUTCOME_BYTES)

# Little-endian u32 identities currently owned by Global CommDomain attachment records.
FROZEN_ADAPTER_KIND_U32 = {
    None: 0,
    "DIRECT_MAP": 1,
    "DEVICE_PEER": 2,
    "OWNER_DELEGATED_COPY": 3,
    "EXPLICIT_TRANSFER": 4,
    "COLLECTIVE": 5,
}
FROZEN_ADAPTER_PROFILE_U32 = {
    None: 0,
    "HOST_SVM_MAP": 1,
    "HOST_VMM_COPY": 2,
    "DEVICE_VMM_PEER_IMPORT": 3,
    "DEVICE_FABRIC_V2_PEER_IMPORT": 4,
    "HOST_SHM_MAP": 5,
    "REMOTE_COPY": 6,
}
FROZEN_ADAPTER_KIND_LE_U32 = {name: value.to_bytes(4, "little") for name, value in FROZEN_ADAPTER_KIND_U32.items()}
FROZEN_ADAPTER_PROFILE_LE_U32 = {
    name: value.to_bytes(4, "little") for name, value in FROZEN_ADAPTER_PROFILE_U32.items()
}

L3_COMPATIBILITY_UT_MODULES = (
    "tests/ut/py/test_worker/test_comm_provider.py",
    "tests/ut/py/test_worker/test_comm_provider_control.py",
    "tests/ut/py/test_worker/test_comm_region.py",
    "tests/ut/py/test_worker/test_worker_chip_orch_comm.py",
    "tests/ut/py/test_worker/test_worker_chip_message_queue.py",
    "tests/ut/py/test_worker/test_host_worker.py",
    "tests/ut/py/test_worker/test_provider_region_onboard.py",
)
L3_COMPATIBILITY_SIM_ONBOARD_CASES = (
    "examples/workers/l3/worker_chip_orch_comm_stream/test_worker_chip_orch_comm_stream.py",
    "examples/workers/l3/worker_chip_message_queue/test_worker_chip_message_queue.py",
)
L3_COMPATIBILITY_SIM_ONBOARD_PLATFORMS = ("a2a3sim", "a2a3", "a5sim", "a5")
