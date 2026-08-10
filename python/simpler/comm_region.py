# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private W3.5 region materialization helpers."""

from __future__ import annotations

import itertools
import re
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
)

_GENERATION_COUNTER = itertools.count(1)
_LOCAL_L2_PATH_RE = re.compile(r"^L3/L2\[(?P<worker_id>[0-9]+)\]$")


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
        self._region = None
        self._state = RegionInstanceState.PLANNED

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
        self._region.payload_write(offset, host_buffer, nbytes)

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._region.payload_read(offset, host_buffer, nbytes)

    def counter(self, offset: int):
        self._ensure_live()
        return self._region.counter(offset)

    def close(self) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state in (RegionInstanceState.PLANNED, RegionInstanceState.ROLLED_BACK):
            self._state = RegionInstanceState.CLOSED
            return
        if self._region is None:
            self._state = RegionInstanceState.CLOSED
            return
        self._state = RegionInstanceState.CLOSING
        try:
            self._worker._close_worker_chip_region(self._region)
        except BaseException:
            self._state = RegionInstanceState.CLOSE_FAILED
            raise
        self._state = RegionInstanceState.CLOSED

    def rollback(self) -> None:
        if self._state in (RegionInstanceState.CLOSED, RegionInstanceState.ROLLED_BACK):
            return
        if self._region is None:
            self._state = RegionInstanceState.ROLLED_BACK
            return
        self._state = RegionInstanceState.ROLLING_BACK
        try:
            self._worker._close_worker_chip_region(self._region)
        except BaseException:
            self._state = RegionInstanceState.ROLLBACK_FAILED
            raise
        self._state = RegionInstanceState.ROLLED_BACK

    def _rollback_after_failed_materialization(self) -> None:
        self.rollback()

    def _ensure_live(self) -> None:
        if self._state is not RegionInstanceState.LIVE:
            raise RuntimeError(f"region instance is not live: {self._state.value}")
        if self._region is None:
            raise RuntimeError("region instance has no adopted worker-chip region")


def validate_single_owner_region_shape(ctx: MaterializationContext) -> SingleOwnerRegionShape:  # noqa: PLR0912
    plan = ctx.plan
    if isinstance(plan, UnsupportedRegionPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, plan.message)
    if not isinstance(plan, BackendPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materializer expects a BackendPlan")
    if int(getattr(ctx.worker, "level", -1)) != 3:
        raise MaterializationRefusal(
            RefusalReason.NEEDS_DELEGATION,
            "W3.5 can materialize only an L3-local worker-chip region; higher roots need W5 delegation",
        )
    if not isinstance(plan.topology_plan, SingleOwnerPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "W3.5 supports SingleOwner plans only")
    provider = _record_for(ctx, plan.topology_plan.provider_endpoint)
    if not provider.path.startswith("L3/"):
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider path requires delegated materialization")
    if provider.deployment is not DEVICE_AICPU:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT,
            "W3.5 supports only DEVICE_AICPU providers for worker-chip regions",
        )
    match = _LOCAL_L2_PATH_RE.match(provider.path)
    if match is None:
        raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
    worker_id = int(match.group("worker_id"))
    device_ids = tuple(getattr(ctx.worker, "_config", {}).get("device_ids", ()))
    if worker_id < 0 or worker_id >= len(device_ids):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            f"provider worker_id {worker_id} is outside the current L3 device list",
        )
    member_records = tuple(_record_for(ctx, member) for member in plan.ordered_members)
    if len(member_records) != 2:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "W3.5 supports exactly one host consumer and one device provider",
        )
    host_consumers = [member for member in member_records if member.deployment is HOST_CPU]
    if len(host_consumers) != 1 or host_consumers[0].path != "L3":
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "W3.5 requires the current L3 HOST_CPU endpoint as the only consumer",
        )
    consumer = host_consumers[0]
    if provider.identity not in plan.ordered_members or consumer.identity not in plan.ordered_members:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "W3.5 ordered members must contain the provider and consumer endpoints",
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


def _validate_part(part: RegionPartPlan, provider: EndpointRecord, consumer: EndpointRecord) -> None:
    if part.backend_kind is not BackendKind.VMM_WINDOW:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_BACKEND_KIND,
            "W3.5 supports only VMM_WINDOW-backed worker-chip region parts",
        )
    attachments = {attachment.member: attachment for attachment in part.attachments}
    if set(attachments) != {provider.identity, consumer.identity}:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "W3.5 part attachments must match exactly the provider and host consumer",
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
            "W3.5 provider attachment must be a bare PROVIDER attachment",
        )


def _validate_consumer_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.CONSUMER
        or attachment.adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY
        or attachment.adapter_profile is not AdapterProfile.HOST_VMM_COPY
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "W3.5 host consumer attachment must use OWNER_DELEGATED_COPY/HOST_VMM_COPY",
        )
