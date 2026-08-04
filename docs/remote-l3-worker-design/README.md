# Remote L3 Worker Design References

Detailed protocol, data-movement, delivery, and audit references live in this
directory. Start with the
[Remote L3 Worker Design](../remote-l3-worker-design.md) for the architecture
and current implementation status, then use the references below for a
specific layer or delivery stage.

## Protocol and data movement

| Document | What it covers |
| -------- | -------------- |
| [Remote L3 Protocol](protocol.md) | Versioned wire frames, payloads, ordering, controls, and bounds validation |
| [Remote L3 Buffers and Transports](buffers-and-transports.md) | Remote buffer ownership and lifetime, tensor sidecars, transport profiles, and simulation behavior |

## Delivery status

| Document | What it covers |
| -------- | -------------- |
| [Remote L3 Implementation Plan](implementation-plan.md) | Incremental delivery sequence, required tests, hardware-gated work, and open decisions |
| [Remote L3 Implementation Record](implementation-record.md) | Feature-by-feature implementation status and verification evidence |

## PR split and audit

| Document | What it covers |
| -------- | -------------- |
| [PR Split and Audit Plan](pr-split-and-audit-plan.md) | Strategy for splitting and reviewing the remote L3 stack without losing contract coverage |
| [PR Split and Audit Artifacts](pr-split-and-audit-artifacts.md) | Recorded split inputs, compliance matrix, risk register, and verification log |

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [Callable Identity Registration](../callable-identity-registration.md) | Stable callable identity and target-private execution slots used by remote routing |
| [Dynamic Callable Registration over IPC](../callable-ipc-dynamic-register.md) | Registration lifecycle and control semantics shared with hierarchical workers |
| [Python Callable Serialization](../python-callable-serialization.md) | Serialization contract for negotiated remote Python callable payloads |
| [Communication Domains](../comm-domain.md) | Deferred-release communication resources that remote domains build on |
