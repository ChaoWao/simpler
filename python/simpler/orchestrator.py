# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestrator — DAG builder passed to a Worker submit/run callback.

A thin Python facade over the C++ ``Orchestrator``. The Worker creates one
Orchestrator handle at init, retrieves the C++ object via ``Worker.get_orchestrator()``,
and passes the handle to the user's orch function::

    def my_orch(orch, args, cfg):
        # chip_handle/sub_handle come from Worker.register(...)
        # build the args object yourself as Tensors; tags drive dependency inference
        a = TaskArgs()
        a.add_tensor(input_handle.tensor(shape, dtype),  TensorArgType.INPUT)
        a.add_tensor(output_handle.tensor(shape, dtype), TensorArgType.OUTPUT)
        orch.submit_next_level(chip_handle, a, cfg, worker=0)  # handle from Worker.register(chip_callable)

        sub_args = TaskArgs()
        sub_args.add_tensor(output_handle.tensor(shape, dtype), TensorArgType.INPUT)
        orch.submit_sub(sub_handle, sub_args)

    handle = w.submit(my_orch, my_args, my_config)
    handle.wait()

Scope and submission-close lifecycle is managed by ``Worker.submit()``;
completion is managed by its ``RunHandle``. ``Worker.run()`` remains the
blocking ``submit(...).wait()`` compatibility entry point.
"""

from __future__ import annotations

import contextlib
import operator
from collections.abc import Iterator, Sequence
from typing import Any

from _task_interface import _Orchestrator as _COrchestrator  # pyright: ignore[reportMissingImports]

from .buffer_handle import AccessMode, BufferHandle, CanonicalIdentity, wrap_fork_inherited
from .callable_identity import CallableHandle
from .task_interface import (
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CommDomainHandle,
    DataType,
    RemoteAddressSpace,
    TaskArgs,
    _empty_remote_sidecar_for,
    _remote_sidecar_for,
    _RemoteTaskArgsSidecar,
    _validate_remote_sidecar_access,
    get_element_size,
)


def _require_handle(
    callable_or_handle: Any,
    *,
    kind: str,
    worker: Any = None,
    expected_namespace: str | None = None,
) -> tuple[bytes, str, str, tuple[int, ...]]:
    """Validate a submit argument is a registered CallableHandle.

    Raises a clear migration error when the caller still passes a
    ``ChipCallable`` directly — every chip callable must be registered
    via ``Worker.register(callable)`` *before* ``init()`` so each chip
    child can pre-warm it on its own device.
    """
    if isinstance(callable_or_handle, ChipCallable) or hasattr(callable_or_handle, "buffer_ptr"):
        raise TypeError(
            f"{kind} now takes a CallableHandle, not a ChipCallable. "
            "Register the callable before init() via "
            "`handle = worker.register(chip_callable)` and pass `handle` here."
        )
    if not isinstance(callable_or_handle, CallableHandle):
        raise TypeError(f"{kind} expects a CallableHandle returned by Worker.register")
    if worker is not None:
        state = worker._resolve_handle(callable_or_handle, expected_namespace=expected_namespace)
        return state.digest, state.kind, state.target_namespace, state.eligible_worker_ids
    if expected_namespace is not None and callable_or_handle.target_namespace != expected_namespace:
        raise TypeError(
            f"{kind} cannot run {callable_or_handle.target_namespace}; expected {expected_namespace} "
            f"for {callable_or_handle.hashid}"
        )
    return callable_or_handle.digest, callable_or_handle.kind, callable_or_handle.target_namespace, ()


def _require_next_level_worker_id(value: Any, *, argument: str) -> int:
    """Return an exact integer worker ID without accepting coercible values."""
    if isinstance(value, bool):
        raise TypeError(f"{argument} must be an integer NEXT_LEVEL worker id")
    try:
        worker_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{argument} must be an integer NEXT_LEVEL worker id") from exc
    if worker_id < 0:
        raise ValueError(f"{argument} must be a non-negative NEXT_LEVEL worker id")
    return worker_id


def _split_next_level_args(args: TaskArgs) -> tuple[TaskArgs, _RemoteTaskArgsSidecar | None]:
    if isinstance(args, TaskArgs):
        return args, _remote_sidecar_for(args)
    raise TypeError("NEXT_LEVEL submit expects TaskArgs")


def _reject_remote_sidecar_args(args: object, *, kind: str) -> None:
    if isinstance(args, TaskArgs) and _remote_sidecar_for(args) is not None:
        raise TypeError(f"RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits, not {kind}")


def _reject_device_args(args: object, *, kind: str) -> None:
    """Refuse a DEVICE-space tensor bound for a host endpoint.

    A Python sub-worker maps its args into its own process and hands them to torch, so a device
    address there is dereferenced as a host pointer. Rejecting at submit gives the caller an error
    naming the argument instead of a segfault inside the child.
    """
    if not isinstance(args, TaskArgs):
        return
    from .buffer_handle import AddressSpace, Tensor  # noqa: PLC0415

    for i in range(args.tensor_count()):
        desc = Tensor.unpack(args.ref(i)).handle
        if desc.address_space == AddressSpace.DEVICE:
            raise ValueError(
                f"{kind}: argument {i} is a DEVICE-space tensor ({desc.backend_kind.name}); a Python "
                f"sub-worker runs on the host and can only take HOST-space tensors"
            )


def _remote_data_eligible_worker_ids(
    remote_sidecar: _RemoteTaskArgsSidecar | None,
    callable_worker_ids: tuple[int, ...],
) -> list[int]:
    worker_ids = [int(worker_id) for worker_id in callable_worker_ids]
    if remote_sidecar is None:
        return worker_ids

    allowed = set(worker_ids)
    for tensor_sidecar in getattr(remote_sidecar, "tensors", ()):
        if tensor_sidecar is None or not getattr(tensor_sidecar, "present", False):
            continue
        desc = tensor_sidecar.desc
        if RemoteAddressSpace(int(desc.address_space)) == RemoteAddressSpace.HOST_INLINE:
            continue
        handle = getattr(tensor_sidecar, "handle", None)
        consumable_worker_id = int(getattr(handle, "worker_id", desc.owner_worker_id))
        allowed.intersection_update({consumable_worker_id})

    final_worker_ids = [worker_id for worker_id in worker_ids if worker_id in allowed]
    if not final_worker_ids:
        raise ValueError("remote tensor sidecars leave no eligible remote worker")
    return final_worker_ids


class Orchestrator:
    """DAG builder. Valid only inside the orch function passed to Worker.run().

    Wraps a borrowed reference to the C++ Orchestrator owned by the parent
    Worker. The Python ``Worker`` keeps a strong reference to the parent
    C++ Worker for the entire orch-fn execution, so the borrowed reference
    stays valid.
    """

    def __init__(self, c_orchestrator: _COrchestrator, worker: Any | None = None) -> None:
        self._o = c_orchestrator
        # Back-reference to the Python Worker so dynamic-allocate APIs
        # (allocate_domain / release_domain) can dispatch CTRL_* through the
        # Worker's chip mailboxes.  None when the Orchestrator is constructed
        # in isolation for tests.
        self._worker = worker

    def _expected_next_level_namespace(self) -> str | None:
        if self._worker is None:
            return None
        if getattr(self._worker, "_next_level_workers", []):
            return "LOCAL_PYTHON"
        if getattr(self._worker, "_chip_shms", []):
            return "LOCAL_CHIP"
        return None

    # ------------------------------------------------------------------
    # User-facing submit API
    # ------------------------------------------------------------------

    def submit_next_level(self, callable_handle: Any, args: TaskArgs, config: CallConfig | None = None, *, worker: int):
        """Submit a NEXT_LEVEL task by registered callable handle.

        ``callable_handle`` must be returned by ``Worker.register``. Tags inside ``args`` drive deps.
        ``worker`` is the exact stable NEXT_LEVEL worker id that runs the
        task. For L3 chip dispatch, these are the existing chip worker ids.
        """
        cfg = config if config is not None else CallConfig()
        cpp_worker_id = _require_next_level_worker_id(worker, argument="worker")
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        if target_namespace != "REMOTE_TASK_DISPATCHER" and self._worker is not None:
            self._worker._require_local_next_level_target(cpp_worker_id, api="submit_next_level")
        c_args, explicit_remote_sidecar = _split_next_level_args(args)
        if target_namespace == "REMOTE_TASK_DISPATCHER":
            remote_sidecar = (
                explicit_remote_sidecar if explicit_remote_sidecar is not None else _empty_remote_sidecar_for(c_args)
            )
        else:
            if explicit_remote_sidecar is not None:
                raise TypeError("RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits")
            remote_sidecar = None
        _validate_remote_sidecar_access(c_args, remote_sidecar)
        final_worker_ids = _remote_data_eligible_worker_ids(remote_sidecar, eligible_worker_ids)
        worker = self._worker
        # Do the (fallible) kind4 provenance analysis BEFORE capturing remote slot
        # refs, so an exception here can never leave captured refs neither
        # released nor adopted (which would defer a remote free forever). Capture
        # is the last step before the rollback try.
        child_ptrs = worker._child_ptrs_in_args(c_args) if worker is not None else []
        prov_guard: Any = contextlib.nullcontext()
        if child_ptrs and worker is not None:
            prov_guard = worker._child_prov_lock
        captured_refs = worker._capture_remote_sidecar_refs(remote_sidecar) if worker is not None else []
        try:
            with prov_guard:
                if child_ptrs and worker is not None:
                    worker._child_prov_check_dispatch(child_ptrs, cpp_worker_id, api="submit_next_level")
                self._o.submit_next_level(
                    digest, kind, target_namespace, c_args, cfg, cpp_worker_id, final_worker_ids, remote_sidecar
                )
        except BaseException:
            if self._worker is not None:
                self._worker._release_remote_slot_refs(captured_refs)
            raise
        if self._worker is not None:
            self._worker._adopt_remote_slot_refs(captured_refs)

    def submit_next_level_group(  # noqa: PLR0912 -- linear per-member sidecar + eligibility + kind4-provenance passes, one branch each
        self,
        callable_handle: Any,
        args_list: list,
        config: CallConfig | None = None,
        *,
        workers: list,
    ):
        """Submit a group of NEXT_LEVEL tasks (N TaskArgs → N worker selections, 1 DAG node).

        ``workers`` contains the exact stable NEXT_LEVEL worker id for each
        member. For L3 chip dispatch, these are the existing chip worker ids.
        """
        cfg = config if config is not None else CallConfig()
        worker_ids = [_require_next_level_worker_id(value, argument="workers entries") for value in workers]
        if len(worker_ids) != len(args_list):
            raise ValueError("workers length must match args_list length")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("workers must not contain duplicate NEXT_LEVEL worker ids")
        expected_namespace = (
            None
            if isinstance(callable_handle, CallableHandle)
            and callable_handle.target_namespace == "REMOTE_TASK_DISPATCHER"
            else self._expected_next_level_namespace()
        )
        digest, kind, target_namespace, eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_next_level_group",
            worker=self._worker,
            expected_namespace=expected_namespace,
        )
        if target_namespace != "REMOTE_TASK_DISPATCHER" and self._worker is not None:
            for worker_id in worker_ids:
                self._worker._require_local_next_level_target(worker_id, api="submit_next_level_group")
        c_args_list = []
        explicit_remote_sidecars = []
        has_explicit_remote_sidecar = False
        for args in args_list:
            c_args, sidecar = _split_next_level_args(args)
            c_args_list.append(c_args)
            explicit_remote_sidecars.append(sidecar)
            has_explicit_remote_sidecar = has_explicit_remote_sidecar or sidecar is not None
        if target_namespace == "REMOTE_TASK_DISPATCHER":
            remote_sidecars = [
                sidecar if sidecar is not None else _empty_remote_sidecar_for(c_args)
                for c_args, sidecar in zip(c_args_list, explicit_remote_sidecars)
            ]
        else:
            if has_explicit_remote_sidecar:
                raise TypeError("RemoteTensorRef is only supported for RemoteCallable NEXT_LEVEL submits")
            remote_sidecars = None
        if remote_sidecars is not None:
            for c_args, remote_sidecar in zip(c_args_list, remote_sidecars):
                _validate_remote_sidecar_access(c_args, remote_sidecar)
        worker_id_sets = (
            [
                _remote_data_eligible_worker_ids(remote_sidecar, eligible_worker_ids)
                for remote_sidecar in remote_sidecars
            ]
            if remote_sidecars is not None
            else [list(eligible_worker_ids) for _ in args_list]
            if eligible_worker_ids
            else []
        )
        # Per-member kind4 dispatch guard: each member's child_memory pointers
        # must be live on that member's exact submitted target.
        # Run this (fallible) analysis BEFORE capturing remote slot refs, so an
        # exception here can never strand captured refs outside the rollback try.
        worker = self._worker
        member_checks: list[tuple[list[tuple[int, int]], int]] = []
        if worker is not None:
            for g, c_args in enumerate(c_args_list):
                child_ptrs = worker._child_ptrs_in_args(c_args)
                if not child_ptrs:
                    continue
                member_checks.append((child_ptrs, worker_ids[g]))
        prov_guard: Any = (
            worker._child_prov_lock if (worker is not None and member_checks) else contextlib.nullcontext()
        )
        captured_refs: list[Any] = []
        if self._worker is not None and remote_sidecars is not None:
            for sidecar in remote_sidecars:
                captured_refs.extend(self._worker._capture_remote_sidecar_refs(sidecar))
        try:
            with prov_guard:
                for child_ptrs, target_worker_id in member_checks:
                    assert worker is not None  # member_checks is only populated when worker is present
                    worker._child_prov_check_dispatch(child_ptrs, target_worker_id, api="submit_next_level_group")
                self._o.submit_next_level_group(
                    digest, kind, target_namespace, c_args_list, cfg, worker_ids, worker_id_sets, remote_sidecars
                )
        except BaseException:
            if self._worker is not None:
                self._worker._release_remote_slot_refs(captured_refs)
            raise
        if self._worker is not None:
            self._worker._adopt_remote_slot_refs(captured_refs)

    def submit_sub(self, callable_handle: Any, args: TaskArgs | None = None):
        """Submit a SUB task by registered callable handle.

        ``args`` may be omitted for a tag-less task (no dependencies, no outputs).
        """
        if args is None:
            args = TaskArgs()
        digest, kind, target_namespace, _eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        _reject_remote_sidecar_args(args, kind="orch.submit_sub")
        _reject_device_args(args, kind="orch.submit_sub")
        self._o.submit_sub(digest, kind, target_namespace, args)

    def submit_sub_group(self, callable_handle: Any, args_list: list):
        """Submit a group of SUB tasks (N TaskArgs → N workers, 1 DAG node)."""
        digest, kind, target_namespace, _eligible_worker_ids = _require_handle(
            callable_handle,
            kind="orch.submit_sub_group",
            worker=self._worker,
            expected_namespace="LOCAL_PYTHON",
        )
        for args in args_list:
            _reject_remote_sidecar_args(args, kind="orch.submit_sub_group")
            _reject_device_args(args, kind="orch.submit_sub_group")
        self._o.submit_sub_group(digest, kind, target_namespace, args_list)

    # ------------------------------------------------------------------
    # Dynamic CommDomain allocation (collective; blocks orch_fn for the
    # duration of the alloc / release handshake)
    # ------------------------------------------------------------------

    def allocate_domain(
        self,
        *,
        name: str,
        workers: Sequence[int],
        window_size: int,
        buffers: Sequence[CommBufferSpec] = (),
    ) -> CommDomainHandle:
        """Collectively allocate a fresh CommDomain across `workers`.

        Driven from the orch thread.  Dispatches CTRL_ALLOC_DOMAIN to each
        participating chip in parallel and blocks until all have completed
        the IPC handshake (HCCL: aclrtMalloc + IPC import; sim: shm + ftruncate).
        Returns a ``CommDomainHandle`` whose ``contexts[chip_idx]`` exposes
        the per-chip ``ChipDomainContext`` (``device_ctx``, ``local_window_base``,
        ``buffers`` by name — each a device ``VMM_WINDOW`` BufferHandle).

        ``name`` is a local identifier (uniqueness checked against currently-live
        handles); peers do not need to agree on the string.  ``workers`` must be
        a subset of the Worker's ``device_ids`` indices; their order defines
        dense domain ranks.  ``buffers`` are carved sequentially inside the
        window in declaration order; their ``nbytes`` sum must fit within
        ``window_size`` — this is validated on the orch thread before any
        chip-side allocation is dispatched, so an oversized request raises
        ``ValueError`` here without leaking a backend allocation.

        Use the handle as a context manager for auto-release:

            with orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096) as tp:
                for chip_idx in tp.workers:
                    orch.submit_next_level(chip_handle, ..., worker=chip_idx)
        """
        if self._worker is None:
            raise RuntimeError("allocate_domain requires an Orchestrator bound to a Worker")
        return self._worker._allocate_domain(
            name=str(name),
            workers=tuple(int(w) for w in workers),
            window_size=int(window_size),
            buffers=list(buffers),
        )

    def release_domain(self, handle: CommDomainHandle) -> None:
        """Collective release.  Equivalent to ``handle.release()``."""
        handle.release()

    def create_l3_l2_region(self, *, worker_id: int, payload_bytes: int, counter_bytes: int):
        """Create an L3-L2 communication region on one NEXT_LEVEL chip worker."""
        if self._worker is None:
            raise RuntimeError("create_l3_l2_region requires an Orchestrator bound to a Worker")
        return self._worker._create_l3_l2_region(int(worker_id), int(payload_bytes), int(counter_bytes))

    def create_l3_l2_queue(self, *, worker_id: int, depth: int, input_arena_bytes: int, output_arena_bytes: int):
        """Create an L3-L2 message queue backed by one L3-L2 communication region."""
        if self._worker is None:
            raise RuntimeError("create_l3_l2_queue requires an Orchestrator bound to a Worker")
        from .l3_l2_message_queue import create_l3_l2_queue  # noqa: PLC0415

        return create_l3_l2_queue(
            self,
            worker_id=int(worker_id),
            depth=int(depth),
            input_arena_bytes=int(input_arena_bytes),
            output_arena_bytes=int(output_arena_bytes),
        )

    # ------------------------------------------------------------------
    # Nested scope (Strict-1 per-scope rings)
    # ------------------------------------------------------------------
    #
    # Tasks and allocations inside a nested ``with orch.scope():`` bind to a
    # deeper heap ring (``min(depth, MAX_RING_DEPTH-1)``) so their
    # memory reclaims independently of the outer scope. ``scope_end`` is
    # non-blocking — it releases scope refs and returns; call
    # the ``RunHandle`` returned by ``Worker.submit`` for a synchronous wait.
    #
    # Usage::
    #
    #     def my_orch(orch, args):
    #         with orch.scope():
    #             orch.submit_next_level(a, ..., worker=0)
    #             orch.submit_next_level(b, ..., worker=0)
    #         orch.submit_next_level(c, ..., worker=0)  # outer-scope ring

    def scope_begin(self) -> None:
        """Open a nested scope explicitly.

        Prefer the ``scope()`` context manager, which pairs the end for you. Every
        ``scope_begin()`` must be matched by a ``scope_end()``.
        """
        self._o.scope_begin()

    def scope_end(self) -> None:
        """Close the scope opened by the matching ``scope_begin()``."""
        self._o.scope_end()

    @contextlib.contextmanager
    def scope(self) -> Iterator[Orchestrator]:
        """Open a nested scope for the ``with`` block.

        Tasks submitted inside the block use a deeper heap ring so they
        reclaim independently of the outer scope (see Strict-1 in
        ``.claude/plans/HIERARCHICAL_RUNTIME_REFACTOR.md``).
        """
        self._o.scope_begin()
        try:
            yield self
        finally:
            self._o.scope_end()

    # A Worker is the only allocator. The Orchestrator exposes thin wrappers that delegate to the
    # bound Worker's implementation so an orchestration fn can allocate / copy / free without reaching
    # for the Worker — each forwards to the Worker's no-lease in-run path (the run already holds it).

    def alloc_child_tensor(self, worker_id: int, shapes: tuple[int, ...], dtype: DataType) -> BufferHandle:
        """Allocate device memory on next-level ``worker_id`` sized for ``shapes`` × ``dtype``; returns a
        DEVICE_MALLOC ``BufferHandle``. Delegates to ``Worker.alloc_child_tensor``."""
        if self._worker is None:
            raise RuntimeError("orch.alloc_child_tensor requires a Worker context")
        return self._worker.alloc_child_tensor(int(worker_id), tuple(shapes), dtype)

    def free(self, handle: BufferHandle) -> None:
        """Free a device ``BufferHandle`` (from ``alloc_child_tensor``). Delegates to ``Worker.free``."""
        if self._worker is None:
            raise RuntimeError("orch.free requires a Worker context")
        self._worker.free(handle)

    def copy_to(self, dst: BufferHandle, src) -> None:
        """H2D: copy host ``src`` into device handle ``dst``. Delegates to ``Worker.copy_to``."""
        if self._worker is None:
            raise RuntimeError("orch.copy_to requires a Worker context")
        self._worker.copy_to(dst, src)

    def copy_from(self, dst, src: BufferHandle) -> None:
        """D2H: copy device handle ``src`` into host ``dst``. Delegates to ``Worker.copy_from``."""
        if self._worker is None:
            raise RuntimeError("orch.copy_from requires a Worker context")
        self._worker.copy_from(dst, src)

    def alloc(self, shape: Sequence[int], dtype: DataType) -> BufferHandle:
        """Allocate a runtime-managed intermediate buffer; returns a ``BufferHandle``.

        The backing is a MAP_SHARED slab (visible to forked child workers), auto-reclaimed once every
        downstream consumer has completed and the run's scope ends — no manual free. Name it in a task
        arg with ``handle.tensor(shape, dtype)``: its canonical identity dependency-wires to this
        alloc's synthetic producer slot (tag it OUTPUT/INOUT on the producer, INPUT on the consumer).

        Use this for chip-A → chip-B intermediate buffers instead of pre-allocating with
        ``torch.share_memory_()`` — the runtime owns the lifecycle. Equivalent to
        ``worker.alloc_shared_tensor``, additionally registered as an L3-L2 orch-comm host buffer so it
        may back an L3-L2 message-queue payload.
        """
        assert self._worker is not None, "orch.alloc requires an L3+ orchestration context"
        shape_t = tuple(int(s) for s in shape)
        nbytes = get_element_size(dtype)
        for s in shape_t:
            nbytes *= s
        oid, buffer_id, path = (
            self._worker._owner_instance_id,
            self._worker._next_buffer_id(),
            f"L{self._worker.level}",
        )
        identity = CanonicalIdentity(oid, buffer_id)
        # alloc keys the synthetic producer slot by the ref's canonical identity (not a raw VA), so a
        # consumer named via handle.tensor(...) dependency-wires to it. Same managed backing as
        # worker.alloc_shared_tensor; additionally registered as an L3-L2 orch-comm host buffer.
        va = int(self._o.alloc(list(shape_t), dtype, identity.pack()))
        handle = wrap_fork_inherited(va, int(nbytes), oid, buffer_id, path, access=AccessMode.READWRITE)
        self._worker._register_l3_l2_orch_comm_host_buffer(handle)
        return handle

    # ------------------------------------------------------------------
    # Internal (called by Worker.submit)
    # ------------------------------------------------------------------

    def _scope_begin(self) -> None:
        self._o._scope_begin()

    def _scope_end(self) -> None:
        self._o._scope_end()

    def _begin_run(self) -> int:
        return int(self._o._begin_run())

    def _close_run_submission(self, run_id: int) -> None:
        self._o._close_run_submission(run_id)

    def _fail_run_submission(self, run_id: int, message: str = "") -> None:
        self._o._fail_run_submission(run_id, message)

    def _wait_run(self, run_id: int) -> None:
        self._o._wait_run(run_id)

    def _wait_run_accepted(self, run_id: int) -> None:
        self._o._wait_run_accepted(run_id)

    def _run_accepted(self, run_id: int) -> bool:
        return bool(self._o._run_accepted(run_id))

    def _wait_run_for(self, run_id: int, timeout_seconds: float) -> bool:
        return bool(self._o._wait_run_for(run_id, timeout_seconds))

    def _run_done(self, run_id: int) -> bool:
        return bool(self._o._run_done(run_id))

    def _release_run(self, run_id: int) -> None:
        self._o._release_run(run_id)
