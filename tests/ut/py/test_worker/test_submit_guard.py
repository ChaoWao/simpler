# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Submit-time argument guard: access ⊆ granted, and no overlapping writes within one task.

Both are checked where the values are final. `access ⊆ granted` cannot be trusted from add time
because a tag stays mutable afterwards; overlapping writes cannot be caught later because the two
args belong to one task node, so there is no order between them to express.
"""

import pytest
from simpler.buffer import AccessMode, BackendKind, mint_owner_instance_id, wrap_fork_inherited
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

_OID = mint_owner_instance_id()


def _ro_handle(nbytes=256):
    """A read-only host backing — what a copy-on-write fork-inherited page grants."""
    return wrap_fork_inherited(0x10000, nbytes, _OID, buffer_id=1, access=AccessMode.READ)


def _rw_handle(nbytes=256):
    return wrap_fork_inherited(
        0x20000, nbytes, _OID, buffer_id=2, access=AccessMode.READWRITE, backend_kind=BackendKind.FORK_SHM
    )


def _run(orch_fn):
    w = Worker(level=3, num_sub_workers=1)
    sub = w.register(lambda args: None)
    w.init()
    try:
        w.run(lambda o, a, c: orch_fn(o, sub))
    finally:
        w.close()


def test_set_tag_cannot_smuggle_a_write_past_the_add_time_check():
    ro = _ro_handle()

    def orch(o, sub):
        ta = TaskArgs()
        ta.add_tensor(ro.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)  # accepted
        ta.set_tag(0, TensorArgType.OUTPUT_EXISTING)  # mutated afterwards, unchecked here
        o.submit_sub(sub, ta)

    with pytest.raises(ValueError, match="does not grant"):
        _run(orch)


def test_overlapping_writes_in_one_task_rejected():
    rw = _rw_handle()

    def orch(o, sub):
        ta = TaskArgs()
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        # [32, 96) overlaps [0, 64)
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=32), TensorArgType.OUTPUT_EXISTING)
        o.submit_sub(sub, ta)

    with pytest.raises(ValueError, match="overlapping bytes"):
        _run(orch)


def test_disjoint_slices_of_one_backing_are_legal():
    rw = _rw_handle()

    def orch(o, sub):
        ta = TaskArgs()
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=64), TensorArgType.OUTPUT_EXISTING)
        o.submit_sub(sub, ta)

    _run(orch)  # must not raise — sharding one buffer across args is the point of byte offsets


def test_overlapping_reads_are_legal():
    rw = _rw_handle()

    def orch(o, sub):
        ta = TaskArgs()
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
        ta.add_tensor(rw.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=32), TensorArgType.INPUT)
        o.submit_sub(sub, ta)

    _run(orch)  # two readers of the same bytes conflict with nothing


def test_same_offsets_on_different_backings_are_legal():
    a = _rw_handle()
    b = wrap_fork_inherited(
        0x30000, 256, _OID, buffer_id=3, access=AccessMode.READWRITE, backend_kind=BackendKind.FORK_SHM
    )

    def orch(o, sub):
        ta = TaskArgs()
        ta.add_tensor(a.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        ta.add_tensor(b.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        o.submit_sub(sub, ta)

    _run(orch)  # identity differs, so the byte ranges are unrelated


def test_a_plain_tensor_cannot_be_named_as_an_output():
    """A copy-on-write tensor's writes never reach the parent, so it may not be tagged as written."""
    import torch  # noqa: PLC0415

    w = Worker(level=3, num_sub_workers=0)
    plain = torch.zeros(16, dtype=torch.float32)
    shared = torch.zeros(16, dtype=torch.float32).share_memory_()

    plain_t = w.make_tensor_arg(plain, shapes=(16,), dtype=DataType.FLOAT32)
    shared_t = w.make_tensor_arg(shared, shapes=(16,), dtype=DataType.FLOAT32)

    ta = TaskArgs()
    with pytest.raises(ValueError, match="not granted by the buffer"):
        ta.add_tensor(plain_t, TensorArgType.OUTPUT_EXISTING)
    ta.add_tensor(plain_t, TensorArgType.INPUT)  # reading it is fine
    ta.add_tensor(shared_t, TensorArgType.OUTPUT_EXISTING)  # share_memory_() earns the write


def test_fork_cow_handle_cannot_claim_a_write_grant():
    # Structural, not just policy: the tag and the grant are checked against each other on decode,
    # so a hand-built descriptor cannot present a copy-on-write backing as writable.
    from simpler.buffer import BackendKind, BufferDescriptor  # noqa: PLC0415

    with pytest.raises(ValueError, match="FORK_COW grants READ only"):
        BufferDescriptor(
            identity=_ro_handle().identity,
            address_space=_ro_handle().address_space,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.FORK_COW,
            nbytes=64,
        )
