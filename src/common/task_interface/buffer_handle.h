/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

/**
 * BufferHandle / BufferRef ABI — typed, opaque cross-layer buffer identity.
 *
 * Three types:
 *   - CanonicalIdentity      : owner_instance_id + buffer_id + generation. The key both the owner
 *                              registry and every consumer import cache use, invariant across every
 *                              edge. Fixed-length with no length field, so hashing and comparison
 *                              cannot read past it.
 *   - BufferHandleDescriptor : the owner's self-describing wire descriptor — backing properties plus
 *                              a length-delimited backend body. Embedded whole in every BufferRef
 *                              built over the handle.
 *   - BufferRef              : the blob-carried wire element. Embeds the full BufferHandleDescriptor
 *                              plus a view (byte_offset, shape, strides, dtype) — self-describing, so
 *                              a consumer materializes it lazily on receipt with no prior handshake.
 *                              No materialized address.
 *
 * There is no wire version. Every endpoint of a run comes from one `pip install`, so a version field
 * would guard a skew that cannot arise; the skew that CAN arise (a stale compiled extension against
 * newer Python) is caught by SIMPLER_BUILD_COMMIT. The leading `magic` on the descriptor and on the
 * blob envelope are discriminators against non-descriptor bytes, not versions.
 *
 * Endianness: all multi-byte integers little-endian. owner_instance_id is an opaque byte sequence
 * (bytewise-compared, no integer/endianness meaning). An unknown backend / address_space / access
 * value is rejected, never silently accepted.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "data_type.h"

// Leading sentinel of a BufferHandleDescriptor, NOT a version. A descriptor decoder needs a cheap
// leading discriminator because `TaskArgs.add_tensor` accepts raw bytes; the sentinel rejects most
// non-descriptor input before any field is trusted. It does not by itself prove the bytes are a
// descriptor — every other field is still validated. There is no multi-version wire: every endpoint
// of a run is built from one `pip install`, and build skew is caught by SIMPLER_BUILD_COMMIT.
inline constexpr uint16_t BUFFER_DESCRIPTOR_MAGIC = 0x5342;  // 'BS' little-endian

// Leading sentinel of the BufferRef blob ENVELOPE (the length-prefixed container in task_args.h),
// distinct from the descriptor sentinel above. Same role, same reason: not a version, just a cheap
// discriminator against a buffer that is not a blob. `read_bufferref_blob` rejects any other value.
inline constexpr uint32_t BUFFERREF_BLOB_MAGIC = 0x424F4C42;  // "BLOB" in memory order

// owner_instance_id is a fixed-width opaque nonce (compared bytewise; no integer/endianness meaning).
// It is the SOLE source of cross-incarnation uniqueness, so it must be generated with a full-width
// random draw — a structured value (timestamp/pid) collides between two Workers built in the same
// process and second.
inline constexpr uint32_t OWNER_INSTANCE_ID_BYTES = 8;

// Backend body upper bound. Only POSIX_SHM uses more than 8 bytes (a shm name); every other backend
// stores a single u64 address.
inline constexpr uint32_t DESC_MAX_BYTES = 32;

// AddressSpace (HOST/DEVICE) is shared with Tensor and lives in data_type.h.

// The backing's granted permission. A per-arg TensorArgType requests read/write and is validated
// against this at submit (requested must be a subset of granted).
enum class AccessMode : uint8_t {
    READ = 0,
    WRITE = 1,
    READWRITE = 2,
};

// Materialization backend of a handle. The consumer resolves a BufferRef to a local address via the
// import registry keyed by canonical identity; this tag selects how. REMOTE_SIDECAR is reserved for
// P2 and rejected on decode in P1. Values are frozen; 5.. reserved (unknown tag => reject).
enum class BackendKind : uint8_t {
    FORK_SHM = 0,
    POSIX_SHM = 1,
    VMM_WINDOW = 2,
    REMOTE_SIDECAR = 3,
    DEVICE_MALLOC = 4,
};

/**
 * Canonical allocation identity — globally unique across owner incarnations, unchanged across every
 * edge. `buffer_id` is unique only within one owner incarnation; `owner_instance_id` (a per-incarnation
 * nonce) disambiguates it, and `generation` detects buffer_id slot reuse (ABA). The key of both the
 * owner registry and every consumer import registry.
 *
 * FIXED-LENGTH BY DESIGN: no field here bounds a read. Hashing and comparison therefore cannot run
 * off the end of the struct whatever bytes arrive on the wire — the property is structural, not
 * something a validator has to enforce.
 *
 * `_pad` is excluded from comparison and hashing, so a decoded identity with dirty padding still
 * matches the same backing (two views of one backing must never key differently).
 *
 * `generation` starts at 1; every reuse of a `buffer_id` slot increments it. 0 is reserved to mean
 * uninitialized and is rejected on decode.
 */
struct CanonicalIdentity {
    uint8_t owner_instance_id[OWNER_INSTANCE_ID_BYTES];
    uint64_t buffer_id;
    uint32_t generation;
    uint8_t _pad[12];
};

static_assert(std::is_trivially_copyable_v<CanonicalIdentity>);
static_assert(sizeof(CanonicalIdentity) == 32, "CanonicalIdentity is wire ABI");
static_assert(offsetof(CanonicalIdentity, owner_instance_id) == 0);
static_assert(offsetof(CanonicalIdentity, buffer_id) == 8);
static_assert(offsetof(CanonicalIdentity, generation) == 16);

inline bool operator==(const CanonicalIdentity &a, const CanonicalIdentity &b) {
    return a.buffer_id == b.buffer_id && a.generation == b.generation &&
           std::memcmp(a.owner_instance_id, b.owner_instance_id, OWNER_INSTANCE_ID_BYTES) == 0;
}
inline bool operator!=(const CanonicalIdentity &a, const CanonicalIdentity &b) { return !(a == b); }

// Hash for use as an unordered_map key (consumer import registry). Folds exactly the fields
// `operator==` compares — `_pad` is excluded so padding can never perturb the bucket.
struct CanonicalIdentityHash {
    size_t operator()(const CanonicalIdentity &k) const {
        auto mix = [](size_t h, uint64_t v) {
            return (h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)));
        };
        size_t h = 0;
        for (uint32_t i = 0; i < OWNER_INSTANCE_ID_BYTES; ++i)
            h = mix(h, k.owner_instance_id[i]);
        h = mix(h, k.buffer_id);
        h = mix(h, k.generation);
        return h;
    }
};

/**
 * The owner's self-describing handle descriptor — embedded whole in every BufferRef built over the
 * handle. A consumer materializes it lazily on first receipt (no separate export handshake) and
 * caches `canonical identity -> local base` (map-once). `backend_kind` + `body[0, body_len)` carry
 * the per-backend materialization (POSIX shm name, fork-inherited VA, device VA, ...).
 * `magic` leads as a cheap discriminator; `address_space` / `access` / `backend_kind` are raw u8 so
 * an unknown value can be rejected without invoking undefined enum behavior.
 *
 * `owner_worker_path_id` is a DIAGNOSTIC id only — it names the owning worker in logs and
 * post-mortems and takes part in no routing, visibility or identity decision. Its side table lives in
 * the owning process; an id a consumer cannot resolve prints as `<path#N>` and is never an error.
 */
struct BufferHandleDescriptor {
    uint16_t magic;
    uint8_t address_space;
    uint8_t access;
    uint8_t backend_kind;
    uint8_t _pad0[3];
    CanonicalIdentity identity;
    uint64_t nbytes;
    uint32_t owner_worker_path_id;
    uint16_t body_len;
    uint8_t _pad1[2];
    char body[DESC_MAX_BYTES];
};

static_assert(std::is_trivially_copyable_v<BufferHandleDescriptor>);
static_assert(sizeof(BufferHandleDescriptor) == 88, "BufferHandleDescriptor is wire ABI");
static_assert(offsetof(BufferHandleDescriptor, magic) == 0);
static_assert(offsetof(BufferHandleDescriptor, address_space) == 2);
static_assert(offsetof(BufferHandleDescriptor, access) == 3);
static_assert(offsetof(BufferHandleDescriptor, backend_kind) == 4);
static_assert(offsetof(BufferHandleDescriptor, identity) == 8);
static_assert(offsetof(BufferHandleDescriptor, nbytes) == 40);
static_assert(offsetof(BufferHandleDescriptor, owner_worker_path_id) == 48);
static_assert(offsetof(BufferHandleDescriptor, body_len) == 52);
static_assert(offsetof(BufferHandleDescriptor, body) == 56);

/**
 * The blob-carried, self-describing wire element: a full embedded handle descriptor plus a strided
 * view onto it. Because the descriptor travels with the ref, a consumer needs no prior handshake —
 * it materializes the embedded `handle` (backend selects how) on first receipt, keyed by
 * `handle.identity`, and reuses the cached base for later refs to the same identity.
 *
 * Invariants:
 *   - Carries NO materialized address. The consumer materializes `handle` to a local base, then
 *     `Tensor.buffer.addr = base`, `Tensor.start_offset = byte_offset / dtype_bytes`.
 *   - `byte_offset` is a BYTE offset of the view origin; a multiple of the dtype size (validated at
 *     materialization).
 *   - `strides[i] > 0` strictly (broadcast / negative step unsupported), carried explicitly — a
 *     singleton dimension's stride is never normalized away.
 */
struct BufferRef {
    BufferHandleDescriptor handle;
    uint64_t byte_offset;
    uint32_t ndims;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    DataType dtype;
    uint8_t _pad[3];
};

/**
 * Reject any BufferRef whose fields are not self-consistent, BEFORE any of them is trusted.
 *
 * This is the single implementation behind all three trust boundaries — the builder
 * (`TaskArgs.add_tensor`, which accepts raw bytes), blob decode on receipt, and materialization —
 * so the three can never drift apart. Throws `std::invalid_argument` naming the field.
 *
 * Every remaining length-like field is bounded here: `body_len` against `DESC_MAX_BYTES` and `ndims`
 * against `MAX_TENSOR_DIMS`, mirroring what the fixed-length `CanonicalIdentity` gets structurally.
 *
 * `REMOTE_SIDECAR` is a legal wire value and passes: an arg bound for a remote worker rides the wire
 * with no local backing, and it is *materialization* that refuses it in P1.
 */
inline void validate_buffer_ref(const BufferRef &r) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };
    const BufferHandleDescriptor &h = r.handle;

    if (h.magic != BUFFER_DESCRIPTOR_MAGIC) reject("invalid Tensor: descriptor magic");
    if (h.address_space > static_cast<uint8_t>(AddressSpace::DEVICE))
        reject("invalid Tensor: address_space out of range");
    if (h.access > static_cast<uint8_t>(AccessMode::READWRITE)) reject("invalid Tensor: access out of range");
    if (h.backend_kind > static_cast<uint8_t>(BackendKind::DEVICE_MALLOC))
        reject("invalid Tensor: backend_kind out of range");
    if (h.body_len > DESC_MAX_BYTES) reject("invalid Tensor: body_len exceeds DESC_MAX_BYTES");
    if (h.identity.generation == 0) reject("invalid Tensor: generation 0 is reserved (uninitialized)");

    // address_space x backend_kind capability gate. REMOTE_SIDECAR is legal in either space.
    const auto backend = static_cast<BackendKind>(h.backend_kind);
    const bool device_space = h.address_space == static_cast<uint8_t>(AddressSpace::DEVICE);
    if (backend != BackendKind::REMOTE_SIDECAR) {
        const bool device_backend = backend == BackendKind::VMM_WINDOW || backend == BackendKind::DEVICE_MALLOC;
        if (device_backend != device_space) reject("invalid Tensor: unsupported address_space x backend_kind");
    }

    if (r.ndims == 0 || r.ndims > static_cast<uint32_t>(MAX_TENSOR_DIMS)) reject("invalid Tensor: ndims out of range");
    if (r.dtype >= DataType::DATA_TYPE_NUM) reject("invalid Tensor: unknown dtype");
    const uint64_t elem = get_element_size(r.dtype);
    if (elem == 0) reject("invalid Tensor: unknown dtype");
    if (r.byte_offset % elem != 0) reject("invalid Tensor: byte_offset is not a multiple of the dtype size");

    // Byte extent of the (possibly strided) view: the last addressable element, +1 element.
    // Computed in u64 with per-dim bounds so a hostile shape/stride cannot wrap it.
    uint64_t last_elem = 0;
    for (uint32_t i = 0; i < r.ndims; ++i) {
        if (r.shapes[i] == 0) reject("invalid Tensor: shape dimension is zero");
        if (r.strides[i] == 0) reject("invalid Tensor: stride must be > 0 (broadcast and negative step unsupported)");
        last_elem += static_cast<uint64_t>(r.shapes[i] - 1) * static_cast<uint64_t>(r.strides[i]);
    }
    const uint64_t extent_bytes = (last_elem + 1) * elem;
    if (r.byte_offset > h.nbytes || extent_bytes > h.nbytes - r.byte_offset) {
        reject("invalid Tensor: view extends past the backing (byte_offset + extent > nbytes)");
    }
}

static_assert(std::is_trivially_copyable_v<BufferRef>, "BufferRef must be trivially copyable for blob memcpy");
static_assert(sizeof(BufferRef) == 144, "BufferRef is wire ABI");
static_assert(offsetof(BufferRef, handle) == 0);
static_assert(offsetof(BufferRef, byte_offset) == 88);
static_assert(offsetof(BufferRef, ndims) == 96);
static_assert(offsetof(BufferRef, shapes) == 100);
static_assert(offsetof(BufferRef, strides) == 120);
static_assert(offsetof(BufferRef, dtype) == 140);
