# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private region materialization helpers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .comm_endpoints import (
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    BackendKind,
    BackendPlan,
    EndpointRecord,
    EndpointRegistry,
    MemberAttachmentPlan,
    RegionLayoutSpec,
    RegionPartPlan,
    SingleOwnerPlan,
    UnsupportedRegionPlan,
    parse_endpoint_path,
)

_GENERATION_COUNTER = itertools.count(1)


class RegionInstanceState(str, Enum):
    PLANNED = "PLANNED"
    OWNER_CREATED = "OWNER_CREATED"
    CONSUMER_ATTACHED = "CONSUMER_ATTACHED"
    LIVE = "LIVE"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


_TERMINAL_FAILURE_STATES = frozenset(
    (
        RegionInstanceState.ROLLBACK_FAILED,
        RegionInstanceState.CLOSE_FAILED,
    )
)


class RefusalReason(str, Enum):
    UNSUPPORTED_PLAN = "UNSUPPORTED_PLAN"
    NEEDS_DELEGATION = "NEEDS_DELEGATION"
    UNSUPPORTED_MEMBER_SHAPE = "UNSUPPORTED_MEMBER_SHAPE"
    UNSUPPORTED_PROVIDER_DEPLOYMENT = "UNSUPPORTED_PROVIDER_DEPLOYMENT"
    UNSUPPORTED_BACKEND_KIND = "UNSUPPORTED_BACKEND_KIND"
    UNSUPPORTED_ATTACHMENT = "UNSUPPORTED_ATTACHMENT"
    REGISTRY_MISMATCH = "REGISTRY_MISMATCH"


class MaterializationError(RuntimeError):
    pass


class MaterializationRefusal(MaterializationError):
    def __init__(self, reason: RefusalReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class MaterializationContext:
    worker: Any
    registry: EndpointRegistry
    plan: BackendPlan | UnsupportedRegionPlan
    layout: RegionLayoutSpec


@dataclass(frozen=True)
class SingleOwnerRegionShape:
    provider: EndpointRecord
    consumer: EndpointRecord
    worker_id: int


class RegionInstance:
    def __init__(self, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> None:
        self.plan = ctx.plan
        self.layout = ctx.layout
        self.provider = shape.provider
        self.consumer = shape.consumer
        self.worker_id = int(shape.worker_id)
        self.generation = next(_GENERATION_COUNTER)
        self.diagnostic_label = (
            f"{self.consumer.path} {self.consumer.deployment.value} -> "
            f"{self.provider.path} {self.provider.deployment.value} "
            f"payload={int(self.layout.payload_bytes)} counter={int(self.layout.counter_bytes)}"
        )
        self._worker = ctx.worker
        self._cleanup_resources = getattr(ctx.worker, "_building_run_resources", None)
        self._region = None
        self._state = RegionInstanceState.PLANNED
        self._cleanup_error: BaseException | None = None

    @property
    def state(self) -> RegionInstanceState:
        return self._state

    @classmethod
    def planned(cls, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> RegionInstance:
        return cls(ctx, shape)

    def _adopt_worker_chip_region(self, region: Any) -> None:
        self._region = region

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_write")
        region = self._region
        assert region is not None
        region.payload_write(offset, host_buffer, nbytes)

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_read")
        region = self._region
        assert region is not None
        region.payload_read(offset, host_buffer, nbytes)

    def counter(self, offset: int):
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.counter")
        region = self._region
        assert region is not None
        return region.counter(offset)

    def close(self) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state in _TERMINAL_FAILURE_STATES and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._state is RegionInstanceState.ROLLED_BACK:
            return
        if self._state is RegionInstanceState.PLANNED:
            self._state = RegionInstanceState.CLOSED
            return
        if self._region is None:
            self._state = RegionInstanceState.CLOSED
            return
        self._worker._require_region_control_context("region_instance.close")
        self._state = RegionInstanceState.CLOSING
        try:
            self._worker._close_worker_chip_region(
                self._region,
                self._cleanup_resources,
                poison_on_error=True,
            )
        except BaseException as exc:
            self._cleanup_error = exc
            self._state = RegionInstanceState.CLOSE_FAILED
            raise
        self._state = RegionInstanceState.CLOSED

    def rollback(self) -> None:
        if self._state in (RegionInstanceState.CLOSED, RegionInstanceState.ROLLED_BACK):
            return
        if self._state in _TERMINAL_FAILURE_STATES and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._region is None:
            self._state = RegionInstanceState.ROLLED_BACK
            return
        self._worker._require_region_control_context("region_instance.rollback")
        self._state = RegionInstanceState.ROLLING_BACK
        try:
            self._worker._close_worker_chip_region(
                self._region,
                self._cleanup_resources,
                poison_on_error=True,
            )
        except BaseException as exc:
            self._cleanup_error = exc
            self._state = RegionInstanceState.ROLLBACK_FAILED
            raise
        self._state = RegionInstanceState.ROLLED_BACK

    def _rollback_after_failed_materialization(self) -> None:
        self.rollback()

    def _ensure_live(self) -> None:
        if self._state is not RegionInstanceState.LIVE:
            raise MaterializationError(f"region instance is not live: {self._state.value}")
        if self._region is None:
            raise MaterializationError("region instance has no adopted worker-chip region")


def validate_single_owner_region_shape(ctx: MaterializationContext) -> SingleOwnerRegionShape:  # noqa: PLR0912
    _validate_registry_matches_worker(ctx)
    plan = ctx.plan
    if isinstance(plan, UnsupportedRegionPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, plan.message)
    if not isinstance(plan, BackendPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materializer expects a BackendPlan")
    if int(getattr(ctx.worker, "level", -1)) != 3:
        raise MaterializationRefusal(
            RefusalReason.NEEDS_DELEGATION,
            "Only L3-local worker-chip regions can be materialized directly; higher-level roots require delegation",
        )
    if not isinstance(plan.topology_plan, SingleOwnerPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "Only SingleOwner region plans are supported")
    provider = _record_for(ctx, plan.topology_plan.provider_endpoint)
    if not provider.path.startswith("L3/"):
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider path requires delegated materialization")
    if provider.deployment is not DEVICE_AICPU:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT,
            "Only DEVICE_AICPU providers are supported for worker-chip regions",
        )
    try:
        provider_path = parse_endpoint_path(provider.path, root_level=ctx.registry.root_level)
    except ValueError as exc:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint") from exc
    if len(provider_path.segments) != 2:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
    root, child = provider_path.segments
    if root.level != 3 or root.index is not None or child.level != 2 or child.index is None:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
    worker_id = int(child.index)
    try:
        ctx.worker._validate_worker_chip_id(worker_id)
    except ValueError as exc:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            f"provider worker_id {worker_id} is outside the current L3 device list",
        ) from exc
    member_records = tuple(_record_for(ctx, member) for member in plan.ordered_members)
    if len(member_records) != 2:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Only one host consumer and one device provider are supported",
        )
    host_consumers = [member for member in member_records if member.deployment is HOST_CPU]
    if len(host_consumers) != 1 or host_consumers[0].path != "L3":
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "The current L3 HOST_CPU endpoint must be the only consumer",
        )
    consumer = host_consumers[0]
    if provider.identity not in plan.ordered_members or consumer.identity not in plan.ordered_members:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Ordered members must contain the provider and consumer endpoints",
        )
    if not ctx.registry.same_node(provider, consumer):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Worker-chip region materialization only supports local endpoints",
        )
    _validate_part(plan.payload, provider, consumer)
    _validate_part(plan.counter, provider, consumer)
    return SingleOwnerRegionShape(provider=provider, consumer=consumer, worker_id=worker_id)


def materialize_region_instance(ctx: MaterializationContext) -> RegionInstance:
    shape = validate_single_owner_region_shape(ctx)
    instance = RegionInstance.planned(ctx, shape)
    try:
        instance._state = RegionInstanceState.OWNER_CREATED
        region = ctx.worker._create_worker_chip_region(
            shape.worker_id,
            int(ctx.layout.payload_bytes),
            int(ctx.layout.counter_bytes),
        )
        instance._adopt_worker_chip_region(region)
        instance._state = RegionInstanceState.LIVE
        return instance
    except BaseException:
        instance._rollback_after_failed_materialization()
        raise


def _record_for(ctx: MaterializationContext, endpoint: Any) -> EndpointRecord:
    try:
        return ctx.registry.record_for(endpoint)
    except ValueError as exc:
        raise MaterializationRefusal(RefusalReason.REGISTRY_MISMATCH, str(exc)) from exc


def _validate_registry_matches_worker(ctx: MaterializationContext) -> None:
    worker_instance_id = getattr(ctx.worker, "_owner_instance_id", None)
    worker_epoch = getattr(ctx.worker, "_endpoint_registry_epoch", None)
    if worker_instance_id != ctx.registry.session_instance_id or worker_epoch != ctx.registry.registry_epoch:
        raise MaterializationRefusal(
            RefusalReason.REGISTRY_MISMATCH,
            "Region materialization requires a registry from the current worker endpoint epoch",
        )


def _validate_part(part: RegionPartPlan, provider: EndpointRecord, consumer: EndpointRecord) -> None:
    if part.backend_kind is not BackendKind.VMM_WINDOW:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_BACKEND_KIND,
            "Only VMM_WINDOW-backed worker-chip region parts are supported",
        )
    attachments = {attachment.member: attachment for attachment in part.attachments}
    if len(attachments) != len(part.attachments) or set(attachments) != {provider.identity, consumer.identity}:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Part attachments must match exactly the provider and host consumer",
        )
    _validate_provider_attachment(attachments[provider.identity])
    _validate_consumer_attachment(attachments[consumer.identity])


def _validate_provider_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.PROVIDER
        or attachment.adapter_kind is not None
        or attachment.adapter_profile is not None
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Provider attachment must be a bare PROVIDER attachment",
        )


def _validate_consumer_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.CONSUMER
        or attachment.adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY
        or attachment.adapter_profile is not AdapterProfile.HOST_VMM_COPY
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Host consumer attachment must use OWNER_DELEGATED_COPY/HOST_VMM_COPY",
        )
