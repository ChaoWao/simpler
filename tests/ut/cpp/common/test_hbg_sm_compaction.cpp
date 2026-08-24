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
/**
 * The orchestrator writes a ring-pitched shared-memory mirror; the image that
 * ships is pitched to the submitted count so its four live prefixes are
 * contiguous and travel as one copy. compact_live_image is the restack, and it
 * owes the device three things: the live payloads, a header carrying no host
 * addresses, and slot-state bindings that resolve inside the image rather than
 * back into the mirror.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "shared_memory.h"

namespace {

constexpr uint64_t WINDOW = 64;  // stands in for the ring capacity
constexpr uint64_t SUBMITTED = 5;

// The per-task argument shape these tests write. Deliberately far below every cap, so
// the compacted pools are a small fraction of the mirror's and the difference is
// visible in the shipped byte count. The fanin stride is what the orchestrator
// advances by — a whole number of cache lines, since a region has to start on one.
constexpr int32_t TENSORS_PER_TASK = 2;
constexpr int32_t SCALARS_PER_TASK = 3;
constexpr int32_t FANIN_PER_TASK = 4;
constexpr int32_t FANIN_STRIDE = static_cast<int32_t>(ARG_POOL_ALIGN / sizeof(int32_t));

// A buffer aligned the way both the arena mirror and the device SM base are;
// ChipTaskSlotState is alignas(64) and every segment offset is a multiple of
// PTO2_ALIGN_SIZE.
class AlignedImage {
public:
    explicit AlignedImage(uint64_t bytes, uint8_t fill = 0) :
        storage_(bytes + PTO2_ALIGN_SIZE, std::byte{0}) {
        base_ = reinterpret_cast<char *>(
            (reinterpret_cast<uintptr_t>(storage_.data()) + PTO2_ALIGN_SIZE - 1) &
            ~static_cast<uintptr_t>(PTO2_ALIGN_SIZE - 1)
        );
        if (fill != 0) std::memset(base_, fill, bytes);
    }

    char *base() { return base_; }
    const char *base() const { return base_; }

private:
    std::vector<std::byte> storage_;
    char *base_{nullptr};
};

// A mirror in the state the orchestrator leaves it: SUBMITTED slots bound to
// their own payload and descriptor, distinguishable per-slot content, and header
// pointers naming the mirror's own arrays.
class Mirror {
public:
    Mirror() :
        image_(pto2_sm_layout::ring_segment_offsets(WINDOW).end) {
        const auto off = pto2_sm_layout::ring_segment_offsets(WINDOW);
        auto *header = reinterpret_cast<PTO2SharedMemoryHeader *>(image_.base());
        auto &ring = header->ring;
        ring.task_window_size = WINDOW;
        ring.task_window_mask = static_cast<int32_t>(WINDOW - 1);
        ring.fc.current_task_index.store(static_cast<int32_t>(SUBMITTED), std::memory_order_relaxed);
        ring.task_descriptors = descriptors();
        ring.task_payloads = payloads();
        ring.slot_states = slot_states();
        ring.completion_flags = completion_flags();
        (void)off;

        // Each live slot takes a packed region in each pool, exactly as the
        // orchestrator's bump cursors hand them out, and gets content that identifies
        // the slot so the compaction can be checked element by element.
        for (uint64_t i = 0; i < SUBMITTED; ++i) {
            descriptors()[i].task_id = TaskId::make(0, static_cast<uint32_t>(i));
            payloads()[i].tensor_count = TENSORS_PER_TASK;
            payloads()[i].scalar_count = SCALARS_PER_TASK;
            payloads()[i].fanin_count = FANIN_PER_TASK;
            payloads()[i].bind_regions(
                tensor_pool() + i * TENSORS_PER_TASK, scalar_pool() + i * SCALARS_PER_TASK,
                fanin_pool() + i * FANIN_STRIDE
            );
            for (int32_t j = 0; j < TENSORS_PER_TASK; ++j) {
                payloads()[i].tensor_data()[j].buffer.addr = 0x1000 + i * 0x10 + j;
            }
            for (int32_t j = 0; j < SCALARS_PER_TASK; ++j) {
                payloads()[i].scalar_data()[j] = 0x3000 + i * 0x10 + j;
            }
            for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
                payloads()[i].fanin_data()[j] = static_cast<int32_t>(0x50 + i * 0x10 + j);
            }
            slot_states()[i].last_consumer_local_id = static_cast<int32_t>(i);
            slot_states()[i].graph_node_index = static_cast<int32_t>(200 + i);
            slot_states()[i].bind_buffers(&payloads()[i], &descriptors()[i]);
            completion_flags()[i].store(static_cast<uint8_t>(i & 1), std::memory_order_relaxed);
        }
        // A slot past the submitted prefix, to prove it does not travel.
        descriptors()[SUBMITTED].task_id = TaskId::make(0, 0xBEEF);
    }

    const char *base() const { return image_.base(); }

    PTO2TaskDescriptor *descriptors() {
        return reinterpret_cast<PTO2TaskDescriptor *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).descriptors
        );
    }
    PTO2TaskPayload *payloads() {
        return reinterpret_cast<PTO2TaskPayload *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).payloads
        );
    }
    ChipTensor *tensor_pool() {
        return reinterpret_cast<ChipTensor *>(image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).tensor_pool);
    }
    uint64_t *scalar_pool() {
        return reinterpret_cast<uint64_t *>(image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).scalar_pool);
    }
    int32_t *fanin_pool() {
        return reinterpret_cast<int32_t *>(image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).fanin_pool);
    }
    ChipTaskSlotState *slot_states() {
        return reinterpret_cast<ChipTaskSlotState *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).slot_states
        );
    }
    std::atomic<uint8_t> *completion_flags() {
        return reinterpret_cast<std::atomic<uint8_t> *>(
            image_.base() + pto2_sm_layout::ring_segment_offsets(WINDOW).completion_flags
        );
    }

private:
    AlignedImage image_;
};

// What a bind of `submitted` tasks put in the pools, given the per-task shape the
// Mirror writes. This is the same arithmetic the orchestrator's cursors perform.
inline pto2_sm_layout::BindUsage usage_for(uint64_t submitted) {
    return pto2_sm_layout::BindUsage{
        submitted,
        submitted * static_cast<uint64_t>(FANIN_STRIDE),
        submitted * static_cast<uint64_t>(TENSORS_PER_TASK),
        submitted * static_cast<uint64_t>(SCALARS_PER_TASK),
    };
}

struct Compacted {
    AlignedImage image;
    uint64_t bytes;
    pto2_sm_layout::BindUsage used;

    explicit Compacted(Mirror &mirror, uint64_t submitted = SUBMITTED) :
        image(pto2_sm_layout::ring_segment_offsets(pto2_sm_layout::image_extents(usage_for(submitted))).end, 0xAA),
        bytes(0),
        used(usage_for(submitted)) {
        bytes = pto2_sm_layout::compact_live_image(mirror.base(), WINDOW, used, image.base());
    }

    pto2_sm_layout::PTO2RingSegmentOffsets off(uint64_t submitted = SUBMITTED) const {
        return pto2_sm_layout::ring_segment_offsets(pto2_sm_layout::image_extents(usage_for(submitted)));
    }

    PTO2TaskPayload *payload_at(uint64_t i) {
        return reinterpret_cast<PTO2TaskPayload *>(image.base() + off().payloads) + i;
    }
    PTO2TaskDescriptor *descriptors() {
        return reinterpret_cast<PTO2TaskDescriptor *>(image.base() + off().descriptors);
    }
    PTO2TaskPayload *payloads() { return payload_at(0); }
    ChipTaskSlotState *slot_states() { return reinterpret_cast<ChipTaskSlotState *>(image.base() + off().slot_states); }
    std::atomic<uint8_t> *completion_flags() {
        return reinterpret_cast<std::atomic<uint8_t> *>(image.base() + off().completion_flags);
    }
};

}  // namespace

// The point of the restack: the whole image is one range, and it is far smaller
// than the mirror it came from.
TEST(HbgSmCompaction, ShipsOnlyTheLivePrefix) {
    Mirror mirror;
    Compacted compacted(mirror);

    EXPECT_EQ(compacted.bytes, compacted.off().end);
    EXPECT_LT(compacted.bytes, pto2_sm_layout::ring_segment_offsets(WINDOW).end);
}

TEST(HbgSmCompaction, CarriesEveryLiveSlotsContent) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(compacted.descriptors()[i].task_id.local(), i) << "slot " << i;
        EXPECT_EQ(compacted.payload_at(i)->tensor_count, TENSORS_PER_TASK) << "slot " << i;
        EXPECT_EQ(compacted.payload_at(i)->tensor_data()[0].buffer.addr, 0x1000 + i * 0x10) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].last_consumer_local_id, static_cast<int32_t>(i)) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].graph_node_index, static_cast<int32_t>(200 + i)) << "slot " << i;
        EXPECT_EQ(compacted.completion_flags()[i].load(std::memory_order_relaxed), static_cast<uint8_t>(i & 1))
            << "slot " << i;
    }
    // The header's pitch-independent fields come across; the mirror slot past the
    // prefix does not.
    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_window_size, WINDOW);
    EXPECT_EQ(ring.fc.current_task_index.load(std::memory_order_relaxed), static_cast<int32_t>(SUBMITTED));
}

// The load-bearing one. Restacking changes the distance between a slot state and
// its payload, so a binding copied verbatim would resolve to the mirror — a host
// address, in device memory.
TEST(HbgSmCompaction, RebindsEverySlotInsideTheImage) {
    Mirror mirror;
    Compacted compacted(mirror);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(compacted.slot_states()[i].payload.get(), compacted.payload_at(i)) << "slot " << i;
        EXPECT_EQ(compacted.slot_states()[i].task.get(), &compacted.descriptors()[i]) << "slot " << i;
    }
}

// A delta is invariant under a whole-image move, which is what makes the single
// copy to the device safe.
TEST(HbgSmCompaction, BindingsSurviveTheCopyToTheDevice) {
    Mirror mirror;
    Compacted compacted(mirror);

    AlignedImage landed(compacted.bytes);
    std::memcpy(landed.base(), compacted.image.base(), compacted.bytes);
    // The image's own layout, not the mirror's. The four slot-pitched offsets happen
    // to agree between the two, but reading them off the mirror overload here would
    // stop being true the moment a pool moved ahead of them.
    const auto off = compacted.off();
    auto *slots = reinterpret_cast<ChipTaskSlotState *>(landed.base() + off.slot_states);
    auto *payloads = reinterpret_cast<PTO2TaskPayload *>(landed.base() + off.payloads);
    auto *descriptors = reinterpret_cast<PTO2TaskDescriptor *>(landed.base() + off.descriptors);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        EXPECT_EQ(slots[i].payload.get(), &payloads[i]) << "slot " << i;
        EXPECT_EQ(slots[i].task.get(), &descriptors[i]) << "slot " << i;
    }
}

// The header's data pointers name the mirror. The device resolves its own in
// attach_populated, so shipping them would only put host addresses in device
// memory.
TEST(HbgSmCompaction, LeavesNoHostPointerInTheHeader) {
    Mirror mirror;
    Compacted compacted(mirror);

    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_descriptors, nullptr);
    EXPECT_EQ(ring.task_payloads, nullptr);
    EXPECT_EQ(ring.slot_states, nullptr);
    EXPECT_EQ(ring.completion_flags, nullptr);
}

// A bind that submits nothing still ships its header and still attaches. Its pools
// are empty, so what travels is the header plus one slot's worth of pitch — far below
// the mirror, whose pools are dimensioned for the whole window.
TEST(HbgSmCompaction, ZeroSubmittedShipsTheHeaderAlone) {
    Mirror mirror;
    Compacted compacted(mirror, /*submitted=*/0);

    EXPECT_EQ(compacted.bytes, compacted.off(0).end);
    EXPECT_LT(compacted.bytes, pto2_sm_layout::ring_segment_offsets(1).end);
    auto &ring = reinterpret_cast<const PTO2SharedMemoryHeader *>(compacted.image.base())->ring;
    EXPECT_EQ(ring.task_window_size, WINDOW);
    EXPECT_EQ(ring.task_descriptors, nullptr);
}

// The pools ship what the bind used, not what the mirror is dimensioned for. That
// is the same property the payload stride used to carry, moved to the pools: fewer
// bytes, and every argument still resolves.
TEST(HbgSmCompaction, PackedPoolsShipFewerBytesAndStillResolve) {
    Mirror mirror;
    Compacted compacted(mirror);

    // The mirror's pools cover WINDOW tasks at their full caps; the image's cover
    // SUBMITTED tasks at the shape they actually used.
    EXPECT_LT(compacted.bytes, pto2_sm_layout::ring_segment_offsets(WINDOW).end);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        PTO2TaskPayload *shipped = compacted.payload_at(i);
        EXPECT_EQ(shipped->tensor_count, TENSORS_PER_TASK) << "slot " << i;
        EXPECT_EQ(shipped->scalar_count, SCALARS_PER_TASK) << "slot " << i;
        EXPECT_EQ(shipped->fanin_count, FANIN_PER_TASK) << "slot " << i;
        // Every element resolves through the re-taken delta, and lands on this slot's
        // own region rather than a neighbour's.
        for (int32_t j = 0; j < TENSORS_PER_TASK; ++j) {
            EXPECT_EQ(shipped->tensor_data()[j].buffer.addr, 0x1000 + i * 0x10 + j) << "slot " << i << " arg " << j;
        }
        for (int32_t j = 0; j < SCALARS_PER_TASK; ++j) {
            EXPECT_EQ(shipped->scalar_data()[j], 0x3000 + i * 0x10 + j) << "slot " << i << " arg " << j;
        }
        for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
            EXPECT_EQ(shipped->fanin_data()[j], static_cast<int32_t>(0x50 + i * 0x10 + j))
                << "slot " << i << " arg " << j;
        }
        EXPECT_EQ(compacted.slot_states()[i].payload.get(), shipped) << "slot " << i;
    }
}

// A region's delta is only correct for the layout it was taken in, so the restack has
// to re-take it. Proof: every shipped region lies inside the image's own pools, which
// the mirror's addresses could not satisfy.
TEST(HbgSmCompaction, RebindsEveryArgumentRegionInsideTheImage) {
    Mirror mirror;
    Compacted compacted(mirror);

    const auto off = compacted.off();
    const char *base = compacted.image.base();
    const char *tensor_begin = base + off.tensor_pool;
    const char *tensor_end = tensor_begin + compacted.used.tensor_elems * sizeof(ChipTensor);
    const char *scalar_begin = base + off.scalar_pool;
    const char *scalar_end = scalar_begin + compacted.used.scalar_elems * sizeof(uint64_t);
    const char *fanin_begin = base + off.fanin_pool;
    const char *fanin_end = fanin_begin + compacted.used.fanin_elems * sizeof(int32_t);

    for (uint64_t i = 0; i < SUBMITTED; ++i) {
        PTO2TaskPayload *shipped = compacted.payload_at(i);
        const char *t = reinterpret_cast<const char *>(shipped->tensor_data());
        const char *s = reinterpret_cast<const char *>(shipped->scalar_data());
        const char *f = reinterpret_cast<const char *>(shipped->fanin_data());
        EXPECT_GE(t, tensor_begin) << "slot " << i;
        EXPECT_LT(t, tensor_end) << "slot " << i;
        EXPECT_GE(s, scalar_begin) << "slot " << i;
        EXPECT_LT(s, scalar_end) << "slot " << i;
        EXPECT_GE(f, fanin_begin) << "slot " << i;
        EXPECT_LT(f, fanin_end) << "slot " << i;
    }
}

// An outer GRAPH task binds only a fanin region: its boundary arguments travel inside
// the submission image and both argument counts are 0. The restack must carry that
// unbound state through rather than resolving it to the pool base, which is what the
// consumers' count guards assume.
TEST(HbgSmCompaction, UnboundRegionsStayUnbound) {
    Mirror mirror;
    // Slot 2 stands in for the outer GRAPH task.
    constexpr uint64_t GRAPH_SLOT = 2;
    mirror.payloads()[GRAPH_SLOT].tensor_count = 0;
    mirror.payloads()[GRAPH_SLOT].scalar_count = 0;
    mirror.payloads()[GRAPH_SLOT].bind_regions(nullptr, nullptr, mirror.fanin_pool() + GRAPH_SLOT * FANIN_STRIDE);

    Compacted compacted(mirror);

    PTO2TaskPayload *shipped = compacted.payload_at(GRAPH_SLOT);
    EXPECT_EQ(shipped->tensor_data(), nullptr);
    EXPECT_EQ(shipped->scalar_data(), nullptr);
    // Its fanin region still resolves, and still inside the image.
    const char *f = reinterpret_cast<const char *>(shipped->fanin_data());
    EXPECT_GE(f, compacted.image.base() + compacted.off().fanin_pool);
    EXPECT_LT(f, compacted.image.base() + compacted.off().tensor_pool);
    for (int32_t j = 0; j < FANIN_PER_TASK; ++j) {
        EXPECT_EQ(shipped->fanin_data()[j], static_cast<int32_t>(0x50 + GRAPH_SLOT * 0x10 + j)) << "arg " << j;
    }
    // Its neighbours are untouched by the unbound slot.
    EXPECT_EQ(compacted.payload_at(1)->tensor_data()[0].buffer.addr, 0x1000 + 1 * 0x10);
    EXPECT_EQ(compacted.payload_at(3)->tensor_data()[0].buffer.addr, 0x1000 + 3 * 0x10);
}

// A slot names its payload, and a payload names its regions, by an int32 delta, so
// neither the mirror nor the image may span more than that reach. `init_per_ring` and
// `attach_populated` each reject a layout that does; what is checked here is the
// arithmetic they test — that the default capacity is comfortably inside the bound, and
// that the layout is what decides where the bound falls. Growing a segment enough to
// put the default capacity out of reach fails here.
TEST(HbgSmCompaction, LayoutStaysWithinDeltaReach) {
    Mirror mirror;
    Compacted compacted(mirror);
    EXPECT_LT(compacted.bytes, static_cast<uint64_t>(INT32_MAX));

    constexpr uint64_t REACH = static_cast<uint64_t>(INT32_MAX);
    EXPECT_LE(pto2_sm_layout::ring_segment_offsets(PTO2_TASK_WINDOW_SIZE).end, REACH);

    // The bound is a property of the layout, not of the capacity alone: there is a
    // capacity past which the mirror no longer fits, and it is above the default.
    uint64_t window = PTO2_TASK_WINDOW_SIZE;
    while (window <= (UINT64_MAX / 2) && pto2_sm_layout::ring_segment_offsets(window).end <= REACH) {
        window *= 2;
    }
    EXPECT_GT(window, static_cast<uint64_t>(PTO2_TASK_WINDOW_SIZE));
    EXPECT_GT(pto2_sm_layout::ring_segment_offsets(window).end, REACH);
}
