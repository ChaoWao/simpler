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
 * Buffer / Tensor ABI — typed, opaque cross-layer buffer identity.
 *
 * Three types:
 *   - CanonicalIdentity      : owner_instance_id + buffer_id + generation. The key both the owner
 *                              registry and every consumer import cache use, invariant across every
 *                              edge. Fixed-length with no length field, so hashing and comparison
 *                              cannot read past it.
 *   - BufferDescriptor       : the owner's self-describing wire descriptor — backing properties plus
 *                              a length-delimited backend body. Embedded whole in every Tensor
 *                              built over the buffer.
 *   - Tensor                 : the argument an L3+ TaskArgs carries, and the element its mailbox
 *                              blob transports. Embeds the full BufferDescriptor plus a view
 *                              (byte_offset, shape, strides, dtype) — self-describing, so a consumer
 *                              materializes it lazily on receipt with no prior handshake. Carries no
 *                              address; the consuming endpoint materializes it into the
 *                              GM-address-bearing `ChipTensor` of tensor.h, the distinct device POD
 *                              the L2 runtime reads.
 *
 * The blob that carries these lives in task_args.h — it is the existing TaskArgs mailbox format,
 * with Tensor as its element. This header owns the types and the gate every receive boundary runs,
 * not their transport.
 *
 * There is no wire version. Every endpoint of a run comes from one `pip install`, so a version field
 * would guard a skew that cannot arise; the skew that CAN arise (a stale compiled extension against
 * newer Python) is caught by SIMPLER_BUILD_COMMIT. The descriptor's leading `magic` is a
 * discriminator against non-descriptor bytes, not a version.
 *
 * Endianness: all multi-byte integers little-endian. owner_instance_id is an opaque byte sequence
 * (bytewise-compared, no integer/endianness meaning). An unknown backend / address_space / access
 * value is rejected, never silently accepted.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "data_type.h"

// Leading sentinel of a BufferDescriptor, NOT a version. A descriptor decoder needs a cheap
// leading discriminator because `TaskArgs.add_tensor` accepts raw bytes; the sentinel rejects most
// non-descriptor input before any field is trusted. It does not by itself prove the bytes are a
// descriptor — every other field is still validated. There is no multi-version wire: every endpoint
// of a run is built from one `pip install`, and build skew is caught by SIMPLER_BUILD_COMMIT.
inline constexpr uint16_t BUFFER_DESCRIPTOR_MAGIC = 0x5342;  // 'BS' little-endian

// owner_instance_id is a fixed-width opaque nonce (compared bytewise; no integer/endianness meaning).
// It is the SOLE source of cross-incarnation uniqueness, so it must be generated with a full-width
// random draw — a structured value (timestamp/pid) collides between two Workers built in the same
// process and second.
inline constexpr uint32_t OWNER_INSTANCE_ID_BYTES = 8;

// Backend body upper bound. Only POSIX_SHM uses more than 8 bytes (a shm name); every other backend
// stores a single u64 address.
inline constexpr uint32_t DESC_MAX_BYTES = 32;

// Body width of every address-bearing backend: one u64 little-endian base. Exact, not a maximum —
// a shorter body reads as a truncated pointer indistinguishable from a real one.
inline constexpr uint32_t BACKEND_ADDRESS_BODY_BYTES = 8;

// The backing's granted permission. A per-arg TensorArgType requests read/write and is validated
// against this at submit (requested must be a subset of granted).
enum class AccessMode : uint8_t {
    READ = 0,
    WRITE = 1,
    READWRITE = 2,
};

// Materialization backend of a buffer. The consumer resolves a Tensor to a local address via the
// import registry keyed by canonical identity; this tag selects how. REMOTE_SIDECAR is reserved for
// P2 and rejected on decode in P1. Values are frozen; 6.. reserved (unknown tag => reject).
//
// FORK_SHM and FORK_COW materialize identically — the body is a base VA the child already has,
// inherited across the fork — but the kernel's write semantics are opposite, so they are separate
// tags rather than one tag plus a hint. A child's write to a MAP_SHARED page lands in the physical
// page the parent reads; a write to a copy-on-write page splits it into a private copy the parent
// never sees, silently. FORK_COW therefore grants READ only, and that is enforced on decode.
enum class BackendKind : uint8_t {
    FORK_SHM = 0,
    POSIX_SHM = 1,
    VMM_WINDOW = 2,
    REMOTE_SIDECAR = 3,
    DEVICE_MALLOC = 4,
    FORK_COW = 5,
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
 * The owner's self-describing buffer descriptor — embedded whole in every Tensor built over the
 * buffer. A consumer materializes it lazily on first receipt (no separate export handshake) and
 * caches `canonical identity -> local base` (map-once). `backend_kind` + `body[0, body_len)` carry
 * the per-backend materialization (POSIX shm name, fork-inherited VA, device VA, ...).
 * `magic` leads as a cheap discriminator; `address_space` / `access` / `backend_kind` are raw u8 so
 * an unknown value can be rejected without invoking undefined enum behavior.
 *
 * `owner_worker_path_id` is a DIAGNOSTIC id only — it names the owning worker in logs and
 * post-mortems and takes part in no routing, visibility or identity decision. Its side table lives in
 * the owning process; an id a consumer cannot resolve prints as `<path#N>` and is never an error.
 */
struct BufferDescriptor {
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

static_assert(std::is_trivially_copyable_v<BufferDescriptor>);
static_assert(sizeof(BufferDescriptor) == 88, "BufferDescriptor is wire ABI");
static_assert(offsetof(BufferDescriptor, magic) == 0);
static_assert(offsetof(BufferDescriptor, address_space) == 2);
static_assert(offsetof(BufferDescriptor, access) == 3);
static_assert(offsetof(BufferDescriptor, backend_kind) == 4);
static_assert(offsetof(BufferDescriptor, identity) == 8);
static_assert(offsetof(BufferDescriptor, nbytes) == 40);
static_assert(offsetof(BufferDescriptor, owner_worker_path_id) == 48);
static_assert(offsetof(BufferDescriptor, body_len) == 52);
static_assert(offsetof(BufferDescriptor, body) == 56);

// Field-wise, so neither the struct padding nor the unused tail of `body` takes part: two decodes of
// one backing describe the same buffer whatever bytes sit outside the fields that carry meaning.
// `body_len` is clamped rather than trusted: this is the one length field in the header, and an
// unvalidated descriptor reaching a comparison must not be able to bound a read past `body`.
inline bool operator==(const BufferDescriptor &a, const BufferDescriptor &b) {
    const size_t body_len = a.body_len < DESC_MAX_BYTES ? a.body_len : DESC_MAX_BYTES;
    return a.identity == b.identity && a.address_space == b.address_space && a.access == b.access &&
           a.backend_kind == b.backend_kind && a.nbytes == b.nbytes &&
           a.owner_worker_path_id == b.owner_worker_path_id && a.body_len == b.body_len &&
           std::memcmp(a.body, b.body, body_len) == 0;
}
inline bool operator!=(const BufferDescriptor &a, const BufferDescriptor &b) { return !(a == b); }

/**
 * The blob-carried, self-describing wire element: a full embedded buffer descriptor plus a strided
 * view onto it. Because the descriptor travels with the tensor, a consumer needs no prior handshake
 * — it materializes the embedded `buffer` (backend selects how) on first receipt, keyed by
 * `buffer.identity`, and reuses the cached base for later tensors over the same identity.
 *
 * Invariants:
 *   - Carries NO materialized address. The consumer materializes `buffer` to a local base, then
 *     `Tensor.buffer.addr = base`, `Tensor.start_offset = byte_offset / dtype_bytes`.
 *   - `byte_offset` is a BYTE offset of the view origin; a multiple of the dtype size (validated at
 *     materialization).
 *   - `strides[i] > 0` strictly (broadcast / negative step unsupported), carried explicitly — a
 *     singleton dimension's stride is never normalized away.
 */
struct Tensor {
    BufferDescriptor buffer;
    uint64_t byte_offset;
    uint32_t ndims;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    DataType dtype;
    uint8_t _pad[3];
};

// Saturating u64 arithmetic. Every input below is wire-supplied, and `shapes[i]` and `strides[i]`
// are each u32, so one product alone reaches ~2^64: an unsaturated sum would wrap to a SMALL extent
// and pass the bound check in validate_tensor while the view really spans exabytes.
inline uint64_t tensor_mul_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_mul_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a != 0 && b > UINT64_MAX / a) ? UINT64_MAX : a * b;
#endif
}
inline uint64_t tensor_add_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_add_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
#endif
}

// Sentinel `tensor_extent_bytes` returns when a view's extent does not fit 64 bits. No real backing
// is 16 EiB, so a Tensor whose extent lands here describes no representable view and is rejected.
inline constexpr uint64_t TENSOR_EXTENT_UNREPRESENTABLE = UINT64_MAX;

// Byte extent of a (possibly strided) view: the last addressable element, plus one element.
// Saturates at TENSOR_EXTENT_UNREPRESENTABLE rather than wrapping, so an extent a hostile
// shape/stride cannot express is rejected instead of comparing as a small, in-bounds value.
// Callers that have not yet validated `r` must not trust the result for anything but a bound.
inline uint64_t tensor_extent_bytes(const Tensor &r) {
    uint64_t last_elem = 0;
    for (uint32_t i = 0; i < r.ndims && i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        if (r.shapes[i] == 0) continue;
        last_elem = tensor_add_sat(
            last_elem, tensor_mul_sat(static_cast<uint64_t>(r.shapes[i] - 1), static_cast<uint64_t>(r.strides[i]))
        );
    }
    return tensor_mul_sat(tensor_add_sat(last_elem, 1), get_element_size(r.dtype));
}

/**
 * Reject any BufferDescriptor whose fields are not self-consistent, BEFORE any of them is trusted.
 *
 * The descriptor half of the tensor gate below, split out because a descriptor also arrives on its
 * own — construction from Python and blob descriptor extraction both hand one over with no view
 * attached. `body_len` is bounded against `DESC_MAX_BYTES` here, mirroring what the fixed-length
 * `CanonicalIdentity` gets structurally, and the body itself is checked against the schema its
 * `backend_kind` implies. Throws `std::invalid_argument` naming the field.
 *
 * `REMOTE_SIDECAR` is a legal wire value and passes: an arg bound for a remote worker rides the wire
 * with no local backing, and it is *materialization* that refuses it in P1.
 */
inline void validate_buffer_descriptor(const BufferDescriptor &h) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };

    if (h.magic != BUFFER_DESCRIPTOR_MAGIC) reject("invalid BufferDescriptor: magic");
    if (h.address_space > static_cast<uint8_t>(AddressSpace::DEVICE))
        reject("invalid BufferDescriptor: address_space out of range");
    if (h.access > static_cast<uint8_t>(AccessMode::READWRITE)) reject("invalid BufferDescriptor: access out of range");
    if (h.backend_kind > static_cast<uint8_t>(BackendKind::FORK_COW))
        reject("invalid BufferDescriptor: backend_kind out of range");
    if (h.body_len > DESC_MAX_BYTES) reject("invalid BufferDescriptor: body_len exceeds DESC_MAX_BYTES");
    if (h.identity.generation == 0) reject("invalid BufferDescriptor: generation 0 is reserved (uninitialized)");

    // address_space x backend_kind capability gate. REMOTE_SIDECAR is legal in either space.
    const auto backend = static_cast<BackendKind>(h.backend_kind);
    const bool device_space = h.address_space == static_cast<uint8_t>(AddressSpace::DEVICE);
    if (backend != BackendKind::REMOTE_SIDECAR) {
        const bool device_backend = backend == BackendKind::VMM_WINDOW || backend == BackendKind::DEVICE_MALLOC;
        if (device_backend != device_space)
            reject("invalid BufferDescriptor: unsupported address_space x backend_kind (capability matrix)");
    }

    // A copy-on-write page splits on the consumer's first write into a private copy the owner never
    // sees, so a write grant over FORK_COW would be silently unobservable rather than an error.
    if (backend == BackendKind::FORK_COW && h.access != static_cast<uint8_t>(AccessMode::READ)) {
        reject("invalid BufferDescriptor: FORK_COW grants READ only (a child's write would not reach the owner)");
    }

    // Per-backend body schema. `backend_kind` says how the consumer reads `body`, so a body that
    // does not fit the reading is not a smaller backing — it is a different value. A DEVICE_MALLOC
    // body of 3 bytes reads as a TRUNCATED pointer with nothing to distinguish it from a real one.
    switch (backend) {
    case BackendKind::FORK_SHM:
    case BackendKind::FORK_COW:
    case BackendKind::DEVICE_MALLOC:
    case BackendKind::VMM_WINDOW: {
        if (h.body_len != BACKEND_ADDRESS_BODY_BYTES)
            reject("invalid BufferDescriptor: an address-bearing backend body must be exactly 8 bytes");
        uint64_t base = 0;
        std::memcpy(&base, h.body, BACKEND_ADDRESS_BODY_BYTES);
        // No backing is mapped or allocated at 0, so a zero body is an unfilled one.
        if (base == 0) reject("invalid BufferDescriptor: address-bearing backend body is a null base");
        break;
    }
    case BackendKind::POSIX_SHM: {
        // The consumer opens this by name, and the Python side decodes it as UTF-8 before the open.
        // Restricting the name to printable ASCII other than '/' makes that decode total instead of
        // a second, weaker schema: it is a UTF-8 subset, so bytes that pass here always decode. '/'
        // is excluded because shm_open forbids it inside a name; space and the control range because
        // the name is rendered into paths and diagnostics.
        if (h.body_len == 0) reject("invalid BufferDescriptor: POSIX_SHM body must carry a shm name");
        for (uint16_t i = 0; i < h.body_len; ++i) {
            const auto c = static_cast<unsigned char>(h.body[i]);
            if (c < 0x21 || c > 0x7E || c == '/') {
                reject("invalid BufferDescriptor: POSIX_SHM shm name must be printable ASCII without '/'");
            }
        }
        break;
    }
    case BackendKind::REMOTE_SIDECAR:
        // The authoritative descriptor rides in the per-task sidecar, so nothing belongs here.
        if (h.body_len != 0) reject("invalid BufferDescriptor: REMOTE_SIDECAR carries no body in P1");
        break;
    }

    // The unused tail of `body` crosses a process boundary with the descriptor, so whatever the
    // owner's memory held there would cross with it.
    //
    // The rule covers the regions a PRODUCER chooses not to fill — this tail, and a Tensor's
    // shape/stride slots past `ndims` (see validate_tensor). It does NOT cover the `_pad` fields:
    // those are alignment slack no producer selects, and `CanonicalIdentity::_pad` in particular is
    // deliberately tolerated so two decodes of one backing still key alike. Equality stays field-wise
    // for that reason, so a descriptor is never canonical byte-for-byte, only field-for-field.
    for (uint32_t i = h.body_len; i < DESC_MAX_BYTES; ++i) {
        if (h.body[i] != 0) reject("invalid BufferDescriptor: reserved bytes past body_len must be zero");
    }
}

/**
 * Reject any Tensor whose fields are not self-consistent, BEFORE any of them is trusted.
 *
 * This is the single implementation behind the trust boundaries that build or decode a Tensor —
 * construction and the builder (`TaskArgs.add_tensor`, which accepts raw bytes) today, blob decode
 * on receipt when the wire carries a Tensor — so they can never drift apart. Throws
 * `std::invalid_argument` naming the field.
 *
 * Materialization is NOT behind this gate yet: `ImportRegistry.materialize` takes an already-decoded
 * descriptor and adds no endpoint check, so nothing there refuses a DEVICE backing handed to a host
 * endpoint. That gate is a separate change; do not read this comment as covering it.
 *
 * `ndims` is bounded against `MAX_TENSOR_DIMS` here; the descriptor's own length-like field is
 * bounded by `validate_buffer_descriptor` above, which this runs first.
 */
inline void validate_tensor(const Tensor &r) {
    auto reject = [](const char *what) {
        throw std::invalid_argument(what);
    };
    const BufferDescriptor &h = r.buffer;

    validate_buffer_descriptor(h);

    if (r.ndims == 0 || r.ndims > static_cast<uint32_t>(MAX_TENSOR_DIMS)) reject("invalid Tensor: ndims out of range");
    if (r.dtype >= DataType::DATA_TYPE_NUM) reject("invalid Tensor: unknown dtype");
    const uint64_t elem = get_element_size(r.dtype);
    if (elem == 0) reject("invalid Tensor: unknown dtype");
    if (r.byte_offset % elem != 0) reject("invalid Tensor: byte_offset is not a multiple of the dtype size");

    for (uint32_t i = 0; i < r.ndims; ++i) {
        if (r.shapes[i] == 0) reject("invalid Tensor: shape dimension is zero");
        if (r.strides[i] == 0) reject("invalid Tensor: stride must be > 0 (broadcast and negative step unsupported)");
    }
    // Slots past `ndims` are the same case as the descriptor's body tail: a producer chose not to
    // fill them, and the blob memcpy carries all 144 bytes across the process boundary regardless.
    // Requiring them zero is also what makes two encodings of one view agree over the active fields
    // AND the inactive ones, so a decoded element can be compared as bytes. (`_pad` is excluded, as
    // in the descriptor — it is alignment slack, not a slot anyone selects.)
    for (uint32_t i = r.ndims; i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        if (r.shapes[i] != 0 || r.strides[i] != 0) {
            reject("invalid Tensor: shape/stride slots past ndims must be zero");
        }
    }
    const uint64_t extent_bytes = tensor_extent_bytes(r);
    if (extent_bytes == TENSOR_EXTENT_UNREPRESENTABLE) {
        reject("invalid Tensor: view extent overflows 64 bits (shape/stride not representable)");
    }
    if (r.byte_offset > h.nbytes || extent_bytes > h.nbytes - r.byte_offset) {
        reject("invalid Tensor: view extends past the backing (byte_offset + extent > nbytes)");
    }
}

// Do two views of the SAME backing touch a common byte? Compared as bounding ranges
// [byte_offset, byte_offset + extent): a strided view's gaps are treated as occupied, so this is
// conservative — it never misses a real overlap, and may report one for two interleaved views.
// The end offsets saturate, so an unvalidated extent cannot wrap a range into a false "disjoint".
inline bool tensors_overlap(const Tensor &a, const Tensor &b) {
    if (!(a.buffer.identity == b.buffer.identity)) return false;
    const uint64_t a_end = tensor_add_sat(a.byte_offset, tensor_extent_bytes(a));
    const uint64_t b_end = tensor_add_sat(b.byte_offset, tensor_extent_bytes(b));
    return a.byte_offset < b_end && b.byte_offset < a_end;
}

static_assert(std::is_trivially_copyable_v<Tensor>, "Tensor must be trivially copyable for blob memcpy");
static_assert(sizeof(Tensor) == 144, "Tensor is wire ABI");
static_assert(offsetof(Tensor, buffer) == 0);
static_assert(offsetof(Tensor, byte_offset) == 88);
static_assert(offsetof(Tensor, ndims) == 96);
static_assert(offsetof(Tensor, shapes) == 100);
static_assert(offsetof(Tensor, strides) == 120);
static_assert(offsetof(Tensor, dtype) == 140);
