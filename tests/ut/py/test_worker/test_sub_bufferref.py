# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sub-worker task args over the BufferRef wire (P1-B B3).

A Python sub callable is a compute leaf: it receives its args as MappedArgs (each backing mapped into
the sub child, map-once) and computes with torch.frombuffer(arg.buffer, ...). Writes land in the
shared backing the owner sees — no C++ Tensor involved. This is exactly the case a closure cannot
serve: a post-init create_buffer shm is not mapped in the pre-forked sub child, so the buffer must
arrive via args and be mapped from its Ref.
"""

import ctypes

import pytest
import torch
from simpler.buffer_handle import BackendKind, BufferRef
from simpler.task_interface import (
    CallConfig,
    ChipStorageTaskArgs,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
    read_args_from_blob,
)
from simpler.worker import Worker

_F32 = 0  # DataType.FLOAT32 value


def test_alloc_shared_tensor_returns_managed_fork_handle():
    hw = Worker(level=3, num_sub_workers=1)
    hw.init()
    try:
        h = hw.alloc_shared_tensor((4, 8), DataType.FLOAT32)
        assert h.nbytes == 4 * 8 * 4  # prod(shape) * element_size
        # Backed by the orchestrator's HeapRing VA (FORK_SHM, no POSIX shm), runtime-managed.
        assert h.backend_kind == BackendKind.FORK_SHM
        assert h.shm is None and h.base != 0
    finally:
        hw.close()


def test_create_buffer_at_l2_needs_no_child():
    # An L2 leaf has no forked children — it materializes the ref in-process itself — so create_buffer
    # must not require a child. The handle is a usable POSIX-shm backing.
    w = Worker(level=2)
    h = w._create_buffer_locked(64)
    try:
        assert h.nbytes == 64
        assert h.shm is not None
        t = torch.frombuffer(h.shm.buf, dtype=torch.float32, count=4)
        t.fill_(3.0)
        assert t.tolist() == [3.0, 3.0, 3.0, 3.0]
        t = None
        # The level guard now admits L2: the public create_buffer no longer TypeErrors on level, it
        # reaches the READY check (this Worker is uninitialized).
        with pytest.raises(RuntimeError, match="READY"):
            w.create_buffer(64)
    finally:
        h.close()


def test_l2_run_materializes_bufferref_to_tensor_blob():
    # An L2 leaf consumes its own BufferRef args: _run_l2_materialized resolves each ref to a local
    # base and hands the runtime a Tensor blob (write_blob format), exactly like a chip child, minus
    # the mailbox. Capture that blob via a fake ChipWorker and decode it to prove the materialization.
    captured = {}

    class _FakeImpl:
        def run_from_blob(self, cid, ptr, cap, cfg):
            captured["cid"] = cid
            captured["blob"] = ctypes.string_at(ptr, cap)

    class _FakeChip:
        _impl = _FakeImpl()

    w = Worker(level=2)
    w._chip_worker = _FakeChip()  # type: ignore[assignment]
    h = w._create_buffer_locked(16)  # 4 x f32
    try:
        shm = h.shm
        assert shm is not None
        torch.frombuffer(shm.buf, dtype=torch.float32, count=4).fill_(7.0)
        ta = TaskArgs()
        ta.add_ref(h.ref(shapes=(4,), dtype=_F32), TensorArgType.INPUT)
        w._run_l2_materialized(3, ta, CallConfig())

        assert captured["cid"] == 3
        decode_buf = ctypes.create_string_buffer(captured["blob"], len(captured["blob"]))
        args = read_args_from_blob(ctypes.addressof(decode_buf))
        assert args.tensor_count() == 1
        t = args.tensor(0)
        assert tuple(t.shapes[: t.ndims]) == (4,)
        assert t.data != 0  # resolved to a real local base
        # The materialized base maps the same physical pages the owner wrote through.
        mapped = torch.frombuffer((ctypes.c_float * 4).from_address(t.data), dtype=torch.float32, count=4)
        assert mapped.tolist() == [7.0, 7.0, 7.0, 7.0]
    finally:
        h.close()
        w._close_l2_import_registry()


def test_l2_run_passes_legacy_chipstorage_through():
    # A pre-BufferRef caller hands worker.run a ChipStorageTaskArgs (has tensor(), no ref()); it must
    # route straight to the runtime, not the BufferRef materialize path.
    routed = {}

    class _FakeChip:
        def _run_slot(self, cid, args, cfg):
            routed["cid"] = cid
            routed["args"] = args

    w = Worker(level=2)
    w._chip_worker = _FakeChip()  # type: ignore[assignment]
    cs = ChipStorageTaskArgs()
    cs.add_tensor(Tensor.make(0x1000, (4,), DataType.FLOAT32))
    w._run_l2_materialized(5, cs, CallConfig())
    assert routed["cid"] == 5 and routed["args"] is cs
    assert w._l2_import_registry is None  # BufferRef path never touched


def test_make_ref_arg_memoizes_handle_per_storage():
    # Every ref over the same torch storage shares one FORK_SHM handle/identity (so deps key on it);
    # a view's byte_offset places it within that shared backing.
    w = Worker(level=3, num_sub_workers=0)  # no init needed: make_ref_arg only reads owner-side state
    t = torch.zeros(16, dtype=torch.float32)
    r1 = BufferRef.unpack(w.make_ref_arg(t, shapes=(16,), dtype=_F32).pack())
    r2 = BufferRef.unpack(w.make_ref_arg(t[4:12], shapes=(8,), dtype=_F32).pack())  # slice of same storage
    assert r1.handle.backend_kind == BackendKind.FORK_SHM
    assert r1.handle.identity.pack() == r2.handle.identity.pack()  # one memoized handle
    assert r1.byte_offset == 0
    assert r2.byte_offset == 16  # t[4:12] starts 4 float32 = 16 B into the storage
    assert len(w._fork_tensor_handles) == 1


def test_make_ref_arg_share_memory_output_across_fork():
    # A pre-fork share_memory_() tensor is MAP_SHARED: a forked child's writes land in the same
    # physical pages the parent reads. make_ref_arg names it (FORK_SHM), the sub child writes through
    # the inherited VA, and the parent sees the result — the fork-inherited read-write path.
    t = torch.zeros(4, dtype=torch.float32).share_memory_()  # allocated + shared BEFORE init/fork

    def sub_fn(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a.add_(1.0)

    hw = Worker(level=3, num_sub_workers=1)
    handle = hw.register(sub_fn)
    hw.init()  # forks the sub child, inheriting t's MAP_SHARED mapping at the same VA
    try:
        t.fill_(5.0)

        def orch(o, _args, cfg):
            sa = TaskArgs()
            sa.add_ref(hw.make_ref_arg(t, shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(handle, sa)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        hw.close()


def test_alloc_shared_tensor_managed_intermediate_deps_and_share():
    # A runtime-managed HeapRing intermediate (alloc_shared_tensor) named as a BufferRef: sub-A writes
    # it (INOUT → depends on the alloc slot + becomes producer), sub-B reads it (INPUT → depends on the
    # producer) and copies it into a create_buffer output the parent reads. The dep wires via the ref's
    # canonical identity; the ring VA is MAP_SHARED so B sees A's write; the slot auto-reclaims at scope
    # end (drain completes, no hang). Result crosses back via the shared `out` buffer, not a dict (the
    # subs run in forked processes).
    def producer(args):
        torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4).fill_(9.0)

    def consumer(args):
        src = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        torch.frombuffer(args[1].buffer, dtype=torch.float32, count=4).copy_(src)

    hw = Worker(level=3, num_sub_workers=2)
    ph = hw.register(producer)
    ch = hw.register(consumer)
    hw.init()
    out_t = None
    try:
        out_h = hw.create_buffer(16)  # 4 x f32 — the parent-observable output
        shm = out_h.shm
        assert shm is not None
        out_t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        out_t.zero_()

        def orch(o, _args, cfg):
            inter = hw.alloc_shared_tensor((4,), DataType.FLOAT32)  # worker captured in the closure
            pa = TaskArgs()
            pa.add_ref(inter.ref(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(ph, pa)
            ca = TaskArgs()
            ca.add_ref(inter.ref(shapes=(4,), dtype=_F32), TensorArgType.INPUT)
            ca.add_ref(out_h.ref(shapes=(4,), dtype=_F32), TensorArgType.OUTPUT_EXISTING)
            o.submit_sub(ch, ca)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(out_t, torch.full((4,), 9.0)), out_t.tolist()
    finally:
        out_t = None
        hw.close()


def test_sub_worker_mapped_arg_readwrite():
    def sub_fn(args):
        a = torch.frombuffer(args[0].buffer, dtype=torch.float32, count=4)
        a.add_(1.0)  # write through the mapped shared buffer

    hw = Worker(level=3, num_sub_workers=1)
    handle = hw.register(sub_fn)
    hw.init()
    t = None
    try:
        buf_h = hw.create_buffer(16)  # 4 x float32, POSIX shm allocated post-init
        shm = buf_h.shm
        assert shm is not None
        t = torch.frombuffer(shm.buf, dtype=torch.float32, count=4)
        t.fill_(5.0)

        def orch(o, args, cfg):
            sa = TaskArgs()
            sa.add_ref(buf_h.ref(shapes=(4,), dtype=_F32), TensorArgType.INOUT)
            o.submit_sub(handle, sa)

        hw.run(orch, args=None, config=CallConfig())
        assert torch.allclose(t, torch.full((4,), 6.0)), t.tolist()
    finally:
        t = None
        hw.close()
