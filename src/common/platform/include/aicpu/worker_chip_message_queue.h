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

#include <stddef.h>
#include <stdint.h>
#include <new>
#include <string.h>

#include "aicpu/worker_chip_orch_endpoint.h"

static constexpr uint32_t WORKER_CHIP_QUEUE_MAGIC = 0x4C335132u;  // "L3Q2"
static constexpr uint16_t WORKER_CHIP_QUEUE_ABI_MAJOR = 1;
static constexpr uint16_t WORKER_CHIP_QUEUE_ABI_MINOR = 1;
static constexpr uint64_t WORKER_CHIP_QUEUE_DESC_SLOT_BYTES = 32;
static constexpr uint64_t WORKER_CHIP_QUEUE_DESC_RING_ALIGNMENT = 8;
static constexpr uint64_t WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT = 64;
static constexpr uint64_t WORKER_CHIP_QUEUE_COUNTER_STRIDE = 64;
static constexpr uint64_t WORKER_CHIP_QUEUE_INPUT_DESC_TAIL_OFFSET = 0;
static constexpr uint64_t WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET = 64;
static constexpr uint64_t WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET = 128;
static constexpr uint64_t WORKER_CHIP_QUEUE_OUTPUT_DESC_HEAD_OFFSET = 192;
static constexpr uint64_t WORKER_CHIP_QUEUE_WORKER_ABORT_FLAG_OFFSET = 256;
static constexpr uint64_t WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET = 320;
static constexpr uint64_t WORKER_CHIP_QUEUE_COUNTER_BYTES = 384;
static constexpr uint64_t WORKER_CHIP_QUEUE_MAX_DEPTH = 1ull << 30;
static constexpr uint64_t WORKER_CHIP_QUEUE_MAGIC_VERSION = worker_chip_orch_comm_pack_magic_version(
    WORKER_CHIP_QUEUE_MAGIC, WORKER_CHIP_QUEUE_ABI_MAJOR, WORKER_CHIP_QUEUE_ABI_MINOR
);

struct WorkerChipQueueDescSlot {
    uint64_t seq;
    uint64_t opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
};

static_assert(
    sizeof(WorkerChipQueueDescSlot) == WORKER_CHIP_QUEUE_DESC_SLOT_BYTES, "WorkerChipQueueDescSlot ABI size changed"
);
static_assert(offsetof(WorkerChipQueueDescSlot, seq) == 0, "WorkerChipQueueDescSlot::seq offset changed");
static_assert(offsetof(WorkerChipQueueDescSlot, opcode) == 8, "WorkerChipQueueDescSlot::opcode offset changed");
static_assert(
    offsetof(WorkerChipQueueDescSlot, payload_offset) == 16, "WorkerChipQueueDescSlot::payload_offset changed"
);
static_assert(
    offsetof(WorkerChipQueueDescSlot, payload_nbytes) == 24, "WorkerChipQueueDescSlot::payload_nbytes changed"
);

enum class WorkerChipQueueOpcode : uint64_t {
    INVALID = 0,
    DATA = 1,
    STOP = 2,
    ERROR = 3,
};

enum class WorkerChipQueueErrorKind : uint32_t {
    NONE = 0,
    BAD_ARGUMENT = 1,
    BAD_DESCRIPTOR = 2,
    INVALID_DESCRIPTOR = 3,
    OUT_OF_SPACE = 4,
    OWNERSHIP = 5,
    REMOTE_ABORTED = 6,
    ENDPOINT_ERROR = 7,
};

enum class WorkerChipQueueTimeoutStatus : uint32_t {
    ORDINARY_TIMEOUT = 0,
    REMOTE_ABORTED = 1,
};

enum class WorkerChipQueueOp : uint32_t {
    INIT = 1,
    TIMEOUT = 2,
    INPUT_TRY_PEEK = 3,
    INPUT_RELEASE = 4,
    OUTPUT_TRY_RESERVE = 5,
    OUTPUT_PUBLISH = 6,
};

inline const char *worker_chip_queue_op_to_string(WorkerChipQueueOp op) {
    switch (op) {
    case WorkerChipQueueOp::INIT:
        return "init";
    case WorkerChipQueueOp::TIMEOUT:
        return "timeout";
    case WorkerChipQueueOp::INPUT_TRY_PEEK:
        return "input.try_peek";
    case WorkerChipQueueOp::INPUT_RELEASE:
        return "input.release";
    case WorkerChipQueueOp::OUTPUT_TRY_RESERVE:
        return "output.try_reserve";
    case WorkerChipQueueOp::OUTPUT_PUBLISH:
        return "output.publish";
    default:
        return "unknown";
    }
}

struct WorkerChipQueueError {
    WorkerChipQueueErrorKind kind;
    WorkerChipQueueOp op;
    uint64_t region_id;
    char message[256];
};

struct WorkerChipQueueLayout {
    uint64_t depth;
    uint64_t input_desc_offset;
    uint64_t output_desc_offset;
    uint64_t input_arena_offset;
    uint64_t output_arena_offset;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t payload_bytes;
    uint64_t input_desc_tail_offset;
    uint64_t input_desc_head_offset;
    uint64_t output_desc_tail_offset;
    uint64_t output_desc_head_offset;
    uint64_t worker_abort_flag_offset;
    uint64_t chip_abort_flag_offset;
    uint64_t counter_bytes;
};

struct WorkerChipQueueArgs {
    uint64_t magic_version;
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t payload_bytes;
    uint64_t counter_bytes;
};

struct WorkerChipQueueInputHandle {
    uint64_t seq;
    WorkerChipQueueOpcode opcode;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    WorkerChipOrchPayloadView payload;
};

struct WorkerChipQueueOutputReservation {
    uint64_t seq;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
    WorkerChipOrchPayloadView payload;
    bool valid;
};

static inline uint64_t worker_chip_queue_magic_version() { return WORKER_CHIP_QUEUE_MAGIC_VERSION; }

static inline bool worker_chip_queue_is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static inline uint64_t worker_chip_queue_align_up(uint64_t value, uint64_t align) {
    if (align == 0) {
        return value;
    }
    uint64_t remainder = value % align;
    return remainder == 0 ? value : value + (align - remainder);
}

static inline bool worker_chip_queue_align_up_checked(uint64_t value, uint64_t align, uint64_t *out) {
    if (out == nullptr || align == 0) {
        return false;
    }
    uint64_t remainder = value % align;
    uint64_t bump = remainder == 0 ? 0 : align - remainder;
    if (worker_chip_orch_comm_add_overflows(value, bump)) {
        return false;
    }
    *out = value + bump;
    return true;
}

static inline bool worker_chip_queue_valid_opcode(WorkerChipQueueOpcode opcode) {
    return opcode == WorkerChipQueueOpcode::DATA || opcode == WorkerChipQueueOpcode::STOP ||
           opcode == WorkerChipQueueOpcode::ERROR;
}

static inline bool worker_chip_queue_make_layout(
    uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, WorkerChipQueueLayout &out
) {
    if (!worker_chip_queue_is_power_of_two(depth) || depth > WORKER_CHIP_QUEUE_MAX_DEPTH || input_arena_bytes == 0 ||
        output_arena_bytes == 0 || input_arena_bytes % WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0 ||
        output_arena_bytes % WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0) {
        return false;
    }

    uint64_t desc_ring_bytes = depth * WORKER_CHIP_QUEUE_DESC_SLOT_BYTES;
    uint64_t input_desc_offset = 0;
    if (worker_chip_orch_comm_add_overflows(input_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t output_desc_offset = input_desc_offset + desc_ring_bytes;
    if (worker_chip_orch_comm_add_overflows(output_desc_offset, desc_ring_bytes)) {
        return false;
    }
    uint64_t desc_end = output_desc_offset + desc_ring_bytes;
    uint64_t input_arena_offset = 0;
    if (!worker_chip_queue_align_up_checked(desc_end, WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT, &input_arena_offset)) {
        return false;
    }
    if (worker_chip_orch_comm_add_overflows(input_arena_offset, input_arena_bytes)) {
        return false;
    }
    uint64_t input_arena_end = input_arena_offset + input_arena_bytes;
    uint64_t output_arena_offset = 0;
    if (!worker_chip_queue_align_up_checked(
            input_arena_end, WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT, &output_arena_offset
        )) {
        return false;
    }
    if (worker_chip_orch_comm_add_overflows(output_arena_offset, output_arena_bytes)) {
        return false;
    }
    uint64_t payload_bytes = output_arena_offset + output_arena_bytes;

    out = WorkerChipQueueLayout{
        depth,
        input_desc_offset,
        output_desc_offset,
        input_arena_offset,
        output_arena_offset,
        input_arena_bytes,
        output_arena_bytes,
        payload_bytes,
        WORKER_CHIP_QUEUE_INPUT_DESC_TAIL_OFFSET,
        WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET,
        WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET,
        WORKER_CHIP_QUEUE_OUTPUT_DESC_HEAD_OFFSET,
        WORKER_CHIP_QUEUE_WORKER_ABORT_FLAG_OFFSET,
        WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET,
        WORKER_CHIP_QUEUE_COUNTER_BYTES,
    };
    return output_desc_offset % WORKER_CHIP_QUEUE_DESC_RING_ALIGNMENT == 0 &&
           input_arena_offset % WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT == 0 &&
           output_arena_offset % WORKER_CHIP_QUEUE_PAYLOAD_ARENA_ALIGNMENT == 0;
}

static inline bool worker_chip_queue_validate_region(
    const WorkerChipOrchRegionDesc &desc, const WorkerChipQueueArgs &args, WorkerChipQueueLayout *out_layout
) {
    WorkerChipQueueLayout layout{};
    if (args.magic_version != worker_chip_queue_magic_version() ||
        worker_chip_orch_comm_validate_desc(desc) != WorkerChipOrchCommValidationError::OK ||
        !worker_chip_queue_make_layout(args.depth, args.input_arena_bytes, args.output_arena_bytes, layout)) {
        return false;
    }
    if (args.payload_bytes != layout.payload_bytes || args.counter_bytes != layout.counter_bytes ||
        desc.payload_bytes != layout.payload_bytes || desc.counter_bytes != layout.counter_bytes) {
        return false;
    }
    if (out_layout != nullptr) {
        *out_layout = layout;
    }
    return true;
}

static inline void worker_chip_queue_encode_desc(
    WorkerChipQueueDescSlot *slot, uint64_t seq, WorkerChipQueueOpcode opcode, uint64_t payload_offset,
    uint64_t payload_nbytes
) {
    if (slot == nullptr) {
        return;
    }
    slot->seq = seq;
    slot->opcode = static_cast<uint64_t>(opcode);
    slot->payload_offset = payload_offset;
    slot->payload_nbytes = payload_nbytes;
}

static inline bool
worker_chip_queue_reconstruct_counter(int32_t observed_low32, uint64_t depth, uint64_t &local_value) {
    if (depth > WORKER_CHIP_QUEUE_MAX_DEPTH) {
        return false;
    }
    uint32_t local_low32 = static_cast<uint32_t>(local_value);
    int32_t delta = static_cast<int32_t>(static_cast<uint32_t>(observed_low32) - local_low32);
    if (delta < 0 || static_cast<uint64_t>(delta) > depth) {
        return false;
    }
    local_value += static_cast<uint64_t>(delta);
    return true;
}

namespace worker_chip_message_queue {

static inline uint64_t magic_version() { return ::worker_chip_queue_magic_version(); }

static inline bool is_power_of_two(uint64_t value) { return ::worker_chip_queue_is_power_of_two(value); }

static inline uint64_t align_up(uint64_t value, uint64_t align) { return ::worker_chip_queue_align_up(value, align); }

static inline bool align_up_checked(uint64_t value, uint64_t align, uint64_t *out) {
    return ::worker_chip_queue_align_up_checked(value, align, out);
}

static inline bool valid_opcode(WorkerChipQueueOpcode opcode) { return ::worker_chip_queue_valid_opcode(opcode); }

static inline bool
make_layout(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes, WorkerChipQueueLayout &out) {
    return ::worker_chip_queue_make_layout(depth, input_arena_bytes, output_arena_bytes, out);
}

static inline bool validate_region(
    const WorkerChipOrchRegionDesc &desc, const WorkerChipQueueArgs &args, WorkerChipQueueLayout *out_layout
) {
    return ::worker_chip_queue_validate_region(desc, args, out_layout);
}

static inline void encode_desc(
    WorkerChipQueueDescSlot *slot, uint64_t seq, WorkerChipQueueOpcode opcode, uint64_t payload_offset,
    uint64_t payload_nbytes
) {
    ::worker_chip_queue_encode_desc(slot, seq, opcode, payload_offset, payload_nbytes);
}

static inline bool reconstruct_counter(int32_t observed_low32, uint64_t depth, uint64_t &local_value) {
    return ::worker_chip_queue_reconstruct_counter(observed_low32, depth, local_value);
}

}  // namespace worker_chip_message_queue

template <uint64_t MaxInflight = 1>
class WorkerChipQueueEndpoint {
    static_assert(MaxInflight > 0, "MaxInflight must be positive");

public:
    static constexpr uint64_t kStopEntrySlots = 1;
    static constexpr uint64_t kEntryCapacity = MaxInflight + kStopEntrySlots;

    class InputQueue {
        struct ActiveInputEntry {
            uint64_t seq;
            WorkerChipQueueOpcode opcode;
            uint64_t payload_offset;
            uint64_t payload_nbytes;
            WorkerChipOrchPayloadView payload;
            bool completed;
        };

    public:
        explicit InputQueue(WorkerChipQueueEndpoint *parent) :
            parent_(parent) {}

        InputQueue(const InputQueue &) = delete;
        InputQueue &operator=(const InputQueue &) = delete;
        InputQueue(InputQueue &&) = delete;
        InputQueue &operator=(InputQueue &&) = delete;

        friend class WorkerChipQueueEndpoint<MaxInflight>;

    private:
        bool initialize() {
            for (uint64_t i = 0; i < kEntryCapacity; ++i) {
                active_entries_[i] = ActiveInputEntry{};
            }
            active_head_ = 0;
            active_count_ = 0;
            active_non_stop_count_ = 0;
            input_head_ = 0;
            input_tail_ = 0;
            input_payload_head_ = 0;
            input_payload_acquire_head_ = 0;
            input_acquire_ = input_head_;
            stop_observed_ = false;
            drained_ = false;
            return true;
        }

    public:
        bool peek(uint64_t timeout_ns, WorkerChipQueueInputHandle &out) {
            uint64_t start = device_time_now_ticks();
            uint64_t frequency_hz = device_time_frequency_hz();
            uint64_t spins = 0;
            while (true) {
                if (try_peek(out)) {
                    return true;
                }
                if (parent_->error_.kind != WorkerChipQueueErrorKind::NONE) {
                    return false;
                }
                spins += 1;
                if (timeout_ns == 0 || (spins & 1023ull) == 0) {
                    uint64_t now = device_time_now_ticks();
                    if (timeout_ns == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout_ns) {
                        parent_->disambiguate_timeout();
                        return false;
                    }
                }
            }
        }

        bool try_peek(WorkerChipQueueInputHandle &out) {
            out = WorkerChipQueueInputHandle{0, WorkerChipQueueOpcode::INVALID, 0, 0, WorkerChipOrchPayloadView{0, 0}};
            if (!parent_->ensure_live()) {
                return false;
            }
            const WorkerChipQueueLayout &layout = parent_->layout_;
            WorkerChipOrchEndpoint &endpoint = parent_->endpoint_;
            if constexpr (MaxInflight == 1) {
                if (active_count_ != 0) {
                    parent_->poison(
                        WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::INPUT_TRY_PEEK,
                        "input handle already active"
                    );
                    return false;
                }
            }
            if (!parent_->refresh_counter(
                    layout.input_desc_tail_offset, input_tail_, layout.depth, WorkerChipQueueOp::INPUT_TRY_PEEK
                )) {
                return false;
            }
            if (stop_observed_) {
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                        "input descriptor published after STOP"
                    );
                }
                return false;
            }
            if (input_tail_ == input_acquire_) {
                return false;
            }
            if (input_tail_ - input_head_ > layout.depth || input_acquire_ < input_head_ ||
                input_acquire_ > input_tail_) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    "input descriptor state invalid"
                );
                return false;
            }

            WorkerChipQueueDescSlot slot{};
            uint64_t slot_index = input_acquire_ & (layout.depth - 1);
            uint64_t slot_offset = layout.input_desc_offset + slot_index * sizeof(WorkerChipQueueDescSlot);
            if (!parent_->read_desc_slot(slot_offset, &slot, WorkerChipQueueOp::INPUT_TRY_PEEK)) {
                return false;
            }
            uint64_t expected_seq = input_acquire_ + 1;
            if (slot.seq != expected_seq) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    "input descriptor seq mismatch"
                );
                return false;
            }
            WorkerChipQueueOpcode opcode = static_cast<WorkerChipQueueOpcode>(slot.opcode);
            if (!worker_chip_message_queue::valid_opcode(opcode)) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    "invalid input opcode"
                );
                return false;
            }
            if (opcode == WorkerChipQueueOpcode::STOP && (slot.payload_offset != 0 || slot.payload_nbytes != 0)) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    "STOP descriptor must be zero-byte"
                );
                return false;
            }
            bool counts_against_window =
                opcode == WorkerChipQueueOpcode::DATA || opcode == WorkerChipQueueOpcode::ERROR;
            if (counts_against_window && active_non_stop_count_ >= MaxInflight) {
                return false;
            }
            if (active_count_ >= kEntryCapacity) {
                parent_->poison(
                    WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::INPUT_TRY_PEEK, "input window state full"
                );
                return false;
            }

            WorkerChipOrchPayloadView view{0, 0};
            if (slot.payload_nbytes == 0) {
                if (slot.payload_offset != 0) {
                    parent_->poison(
                        WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                        "zero-byte descriptor uses nonzero payload offset"
                    );
                    return false;
                }
            } else if (!parent_->payload_in_arena(
                           slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset, layout.input_arena_bytes
                       )) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    "input payload out of arena"
                );
                return false;
            } else if (!parent_->payload_matches_head(
                           input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes,
                           layout.input_arena_offset, layout.input_arena_bytes, WorkerChipQueueOp::INPUT_TRY_PEEK
                       )) {
                return false;
            } else if (!endpoint.payload_read(slot.payload_offset, slot.payload_nbytes, view)) {
                parent_->poison(
                    WorkerChipQueueErrorKind::ENDPOINT_ERROR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                    endpoint.error().message
                );
                return false;
            } else {
                parent_->advance_payload_head(
                    input_payload_acquire_head_, slot.payload_offset, slot.payload_nbytes, layout.input_arena_offset,
                    layout.input_arena_bytes, WorkerChipQueueOp::INPUT_TRY_PEEK
                );
                if (parent_->error_.kind != WorkerChipQueueErrorKind::NONE) {
                    return false;
                }
            }

            out = WorkerChipQueueInputHandle{slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view};
            uint64_t insert_index = (active_head_ + active_count_) % kEntryCapacity;
            active_entries_[insert_index] =
                ActiveInputEntry{slot.seq, opcode, slot.payload_offset, slot.payload_nbytes, view, false};
            active_count_ += 1;
            if (counts_against_window) {
                active_non_stop_count_ += 1;
            }
            input_acquire_ += 1;
            if (opcode == WorkerChipQueueOpcode::STOP) {
                stop_observed_ = true;
                if (input_tail_ != input_acquire_) {
                    parent_->poison(
                        WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::INPUT_TRY_PEEK,
                        "input descriptor published after STOP"
                    );
                    return false;
                }
            }
            return true;
        }

        bool release(const WorkerChipQueueInputHandle &handle) {
            if (!parent_->ensure_live()) {
                return false;
            }
            ActiveInputEntry *entry = entry_for_seq(handle.seq);
            if (entry == nullptr || handle.opcode != entry->opcode || handle.payload_offset != entry->payload_offset ||
                handle.payload_nbytes != entry->payload_nbytes || handle.payload.gm_addr != entry->payload.gm_addr ||
                handle.payload.nbytes != entry->payload.nbytes) {
                parent_->poison(
                    WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::INPUT_RELEASE, "input handle is not active"
                );
                return false;
            }
            if (entry->completed) {
                parent_->poison(
                    WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::INPUT_RELEASE,
                    "input handle already released"
                );
                return false;
            }
            entry->completed = true;
            return release_completed_prefix();
        }

        bool drained() const { return drained_; }

    private:
        ActiveInputEntry *entry_for_seq(uint64_t seq) {
            uint64_t first_seq = input_head_ + 1;
            if (seq < first_seq) {
                return nullptr;
            }
            uint64_t ordinal = seq - first_seq;
            if (ordinal >= active_count_) {
                return nullptr;
            }
            uint64_t index = (active_head_ + ordinal) % kEntryCapacity;
            return active_entries_[index].seq == seq ? &active_entries_[index] : nullptr;
        }

        bool release_completed_prefix() {
            while (active_count_ != 0 && active_entries_[active_head_].completed) {
                ActiveInputEntry entry = active_entries_[active_head_];
                if (entry.payload_nbytes != 0) {
                    parent_->advance_payload_head(
                        input_payload_head_, entry.payload_offset, entry.payload_nbytes,
                        parent_->layout_.input_arena_offset, parent_->layout_.input_arena_bytes,
                        WorkerChipQueueOp::INPUT_RELEASE
                    );
                    if (parent_->error_.kind != WorkerChipQueueErrorKind::NONE) {
                        return false;
                    }
                }
                input_head_ += 1;
                if (entry.opcode == WorkerChipQueueOpcode::DATA || entry.opcode == WorkerChipQueueOpcode::ERROR) {
                    active_non_stop_count_ -= 1;
                }
                if (entry.opcode == WorkerChipQueueOpcode::STOP) {
                    drained_ = true;
                }
                active_entries_[active_head_] = ActiveInputEntry{};
                active_head_ = (active_head_ + 1) % kEntryCapacity;
                active_count_ -= 1;
                if (!parent_->notify_counter(
                        parent_->layout_.input_desc_head_offset, static_cast<int32_t>(input_head_),
                        WorkerChipQueueOp::INPUT_RELEASE
                    )) {
                    return false;
                }
            }
            return true;
        }

        WorkerChipQueueEndpoint *parent_;
        ActiveInputEntry active_entries_[kEntryCapacity]{};
        uint64_t active_head_{0};
        uint64_t active_count_{0};
        uint64_t active_non_stop_count_{0};
        uint64_t input_head_{0};
        uint64_t input_tail_{0};
        uint64_t input_payload_head_{0};
        uint64_t input_payload_acquire_head_{0};
        uint64_t input_acquire_{0};
        bool stop_observed_{false};
        bool drained_{false};
    };

    class OutputQueue {
    public:
        explicit OutputQueue(WorkerChipQueueEndpoint *parent) :
            parent_(parent) {}

        bool initialize() {
            output_head_ = 0;
            output_tail_ = 0;
            output_payload_head_ = 0;
            output_payload_tail_ = 0;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
            return true;
        }

        bool reserve(uint64_t nbytes, uint64_t timeout_ns, WorkerChipQueueOutputReservation &out) {
            uint64_t start = device_time_now_ticks();
            uint64_t frequency_hz = device_time_frequency_hz();
            uint64_t spins = 0;
            while (true) {
                if (try_reserve(nbytes, out)) {
                    return true;
                }
                if (parent_->error_.kind != WorkerChipQueueErrorKind::NONE) {
                    return false;
                }
                spins += 1;
                if (timeout_ns == 0 || (spins & 1023ull) == 0) {
                    uint64_t now = device_time_now_ticks();
                    if (timeout_ns == 0 || sys_cnt_elapsed_ns(start, now, frequency_hz) >= timeout_ns) {
                        parent_->disambiguate_timeout();
                        return false;
                    }
                }
            }
        }

        bool try_reserve(uint64_t nbytes, WorkerChipQueueOutputReservation &out) {
            out = WorkerChipQueueOutputReservation{0, 0, 0, WorkerChipOrchPayloadView{0, 0}, false};
            if (!parent_->ensure_live()) {
                return false;
            }
            const WorkerChipQueueLayout &layout = parent_->layout_;
            if (reservation_active_) {
                parent_->poison(
                    WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::OUTPUT_TRY_RESERVE,
                    "output reservation already active"
                );
                return false;
            }
            if (nbytes > layout.output_arena_bytes) {
                return false;
            }
            uint64_t old_head = output_head_;
            if (!parent_->refresh_counter(
                    layout.output_desc_head_offset, output_head_, layout.depth, WorkerChipQueueOp::OUTPUT_TRY_RESERVE
                )) {
                return false;
            }
            if (output_head_ != old_head &&
                !replay_output_releases(old_head, output_head_, WorkerChipQueueOp::OUTPUT_TRY_RESERVE)) {
                return false;
            }
            if (output_tail_ - output_head_ >= layout.depth) {
                return false;
            }

            uint64_t payload_offset = 0;
            WorkerChipOrchPayloadView view{0, 0};
            if (nbytes != 0) {
                uint64_t arena_base = layout.output_arena_offset;
                uint64_t arena_bytes = layout.output_arena_bytes;
                uint64_t arena_pos = output_payload_tail_ % arena_bytes;
                if (arena_pos + nbytes > arena_bytes) {
                    // Payloads are never split across arena wrap. The skipped tail bytes are retired in the
                    // monotonic virtual cursor even if this reservation later finds the arena full.
                    output_payload_tail_ += arena_bytes - arena_pos;
                    arena_pos = 0;
                }
                if (output_payload_tail_ + nbytes - output_payload_head_ > arena_bytes) {
                    return false;
                }
                payload_offset = arena_base + arena_pos;
                view = WorkerChipOrchPayloadView{parent_->endpoint_.descriptor().payload_base + payload_offset, nbytes};
                output_payload_tail_ += nbytes;
            }

            reservation_active_ = true;
            reservation_seq_ = output_tail_ + 1;
            reservation_offset_ = payload_offset;
            reservation_nbytes_ = nbytes;
            out = WorkerChipQueueOutputReservation{reservation_seq_, payload_offset, nbytes, view, true};
            return true;
        }

        bool publish(const WorkerChipQueueOutputReservation &reservation, WorkerChipQueueOpcode opcode) {
            if (!parent_->ensure_live()) {
                return false;
            }
            if (!reservation_active_ || !reservation.valid || reservation.seq != reservation_seq_ ||
                reservation.payload_offset != reservation_offset_ ||
                reservation.payload_nbytes != reservation_nbytes_) {
                parent_->poison(
                    WorkerChipQueueErrorKind::OWNERSHIP, WorkerChipQueueOp::OUTPUT_PUBLISH, "unknown output reservation"
                );
                return false;
            }
            if (opcode == WorkerChipQueueOpcode::STOP || !worker_chip_message_queue::valid_opcode(opcode)) {
                parent_->poison(
                    WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, WorkerChipQueueOp::OUTPUT_PUBLISH,
                    "invalid output opcode"
                );
                return false;
            }
            WorkerChipQueueDescSlot slot{};
            worker_chip_message_queue::encode_desc(
                &slot, 0, opcode, reservation.payload_offset, reservation.payload_nbytes
            );
            uint64_t slot_index = output_tail_ & (parent_->layout_.depth - 1);
            uint64_t slot_offset = parent_->layout_.output_desc_offset + slot_index * sizeof(WorkerChipQueueDescSlot);
            if (!parent_->write_desc_slot(slot_offset, slot, reservation.seq, WorkerChipQueueOp::OUTPUT_PUBLISH)) {
                return false;
            }
            output_tail_ += 1;
            reservation_active_ = false;
            reservation_seq_ = 0;
            reservation_offset_ = 0;
            reservation_nbytes_ = 0;
            return parent_->notify_counter(
                parent_->layout_.output_desc_tail_offset, static_cast<int32_t>(output_tail_),
                WorkerChipQueueOp::OUTPUT_PUBLISH
            );
        }

    private:
        bool replay_output_releases(uint64_t old_head, uint64_t new_head, WorkerChipQueueOp op) {
            uint64_t cursor = old_head;
            while (cursor < new_head) {
                WorkerChipQueueDescSlot slot{};
                uint64_t slot_index = cursor & (parent_->layout_.depth - 1);
                uint64_t slot_offset =
                    parent_->layout_.output_desc_offset + slot_index * sizeof(WorkerChipQueueDescSlot);
                if (!parent_->read_desc_slot(slot_offset, &slot, op)) {
                    return false;
                }
                if (slot.seq != cursor + 1) {
                    parent_->poison(
                        WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, op, "output release replay seq mismatch"
                    );
                    return false;
                }
                if (slot.payload_nbytes != 0) {
                    parent_->advance_payload_head(
                        output_payload_head_, slot.payload_offset, slot.payload_nbytes,
                        parent_->layout_.output_arena_offset, parent_->layout_.output_arena_bytes, op
                    );
                    if (parent_->error_.kind != WorkerChipQueueErrorKind::NONE) {
                        return false;
                    }
                }
                cursor += 1;
            }
            return true;
        }

        WorkerChipQueueEndpoint *parent_;
        uint64_t output_head_{0};
        uint64_t output_tail_{0};
        uint64_t output_payload_head_{0};
        uint64_t output_payload_tail_{0};
        bool reservation_active_{false};
        uint64_t reservation_seq_{0};
        uint64_t reservation_offset_{0};
        uint64_t reservation_nbytes_{0};
    };

    WorkerChipQueueEndpoint(const WorkerChipOrchRegionDesc &desc, const WorkerChipQueueArgs &args) :
        endpoint_(desc),
        input_queue_(this),
        output_queue_(this) {
        if (endpoint_.error().kind != WorkerChipEndpointErrorKind::NONE ||
            !worker_chip_message_queue::validate_region(desc, args, &layout_)) {
            set_error(
                WorkerChipQueueErrorKind::BAD_DESCRIPTOR, WorkerChipQueueOp::INIT, desc.region_id,
                "invalid queue descriptor"
            );
            return;
        }
        if (MaxInflight > layout_.depth) {
            set_error(
                WorkerChipQueueErrorKind::BAD_ARGUMENT, WorkerChipQueueOp::INIT, desc.region_id, "invalid input window"
            );
            return;
        }
        input_queue_.initialize();
        output_queue_.initialize();
    }

    WorkerChipQueueEndpoint(const WorkerChipQueueEndpoint &) = delete;
    WorkerChipQueueEndpoint &operator=(const WorkerChipQueueEndpoint &) = delete;
    WorkerChipQueueEndpoint(WorkerChipQueueEndpoint &&) = delete;
    WorkerChipQueueEndpoint &operator=(WorkerChipQueueEndpoint &&) = delete;

    const WorkerChipQueueError &error() const { return error_; }
    const WorkerChipQueueLayout &layout() const { return layout_; }
    InputQueue &input() { return input_queue_; }
    OutputQueue &output() { return output_queue_; }

    WorkerChipQueueTimeoutStatus disambiguate_timeout() {
        if (error_.kind != WorkerChipQueueErrorKind::NONE) {
            return error_.kind == WorkerChipQueueErrorKind::REMOTE_ABORTED ?
                       WorkerChipQueueTimeoutStatus::REMOTE_ABORTED :
                       WorkerChipQueueTimeoutStatus::ORDINARY_TIMEOUT;
        }
        WorkerChipOrchSignalTestResult result{};
        uint64_t addr = 0;
        if (!endpoint_.counter_addr(layout_.worker_abort_flag_offset, addr) ||
            !endpoint_.signal_test(addr, 1, WorkerChipOrchWaitCmp::GE, result)) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, WorkerChipQueueOp::TIMEOUT, endpoint_.error().message);
            return WorkerChipQueueTimeoutStatus::ORDINARY_TIMEOUT;
        }
        if (result.matched) {
            set_error(
                WorkerChipQueueErrorKind::REMOTE_ABORTED, WorkerChipQueueOp::TIMEOUT, endpoint_.descriptor().region_id,
                "remote abort"
            );
            return WorkerChipQueueTimeoutStatus::REMOTE_ABORTED;
        }
        return WorkerChipQueueTimeoutStatus::ORDINARY_TIMEOUT;
    }

private:
    bool ensure_live() {
        if (error_.kind == WorkerChipQueueErrorKind::NONE) {
            return true;
        }
        return false;
    }

    void set_error(WorkerChipQueueErrorKind kind, WorkerChipQueueOp op, uint64_t region_id, const char *message) {
        if (error_.kind != WorkerChipQueueErrorKind::NONE) {
            return;
        }
        error_ = WorkerChipQueueError{kind, op, region_id, ""};
        worker_chip_orch_comm::copy_error_message(error_.message, sizeof(error_.message), message);
    }

    void poison(WorkerChipQueueErrorKind kind, WorkerChipQueueOp op, const char *message) {
        set_error(kind, op, endpoint_.descriptor().region_id, message);
        if (kind != WorkerChipQueueErrorKind::REMOTE_ABORTED) {
            uint64_t addr = 0;
            if (endpoint_.counter_addr(layout_.chip_abort_flag_offset, addr)) {
                endpoint_.signal_notify(addr, 1, WorkerChipOrchNotifyOp::Set);
            }
        }
    }

    bool notify_counter(uint64_t offset, int32_t value, WorkerChipQueueOp op) {
        uint64_t addr = 0;
        if (!endpoint_.counter_addr(offset, addr) ||
            !endpoint_.signal_notify(addr, value, WorkerChipOrchNotifyOp::Set)) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        return true;
    }

    bool refresh_counter(uint64_t offset, uint64_t &local, uint64_t depth, WorkerChipQueueOp op) {
        uint64_t addr = 0;
        WorkerChipOrchSignalTestResult result{};
        if (!endpoint_.counter_addr(offset, addr) ||
            !endpoint_.signal_test(addr, static_cast<int32_t>(local), WorkerChipOrchWaitCmp::NE, result)) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        if (!result.matched) {
            return true;
        }
        if (!worker_chip_message_queue::reconstruct_counter(result.observed, depth, local)) {
            poison(WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, op, "counter reconstruction failed");
            return false;
        }
        return true;
    }

    bool read_desc_slot(uint64_t slot_offset, WorkerChipQueueDescSlot *slot, WorkerChipQueueOp op) {
        WorkerChipOrchPayloadView view{};
        if (!endpoint_.payload_read(slot_offset, sizeof(WorkerChipQueueDescSlot), view)) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        memcpy(
            slot, reinterpret_cast<const void *>(static_cast<uintptr_t>(view.gm_addr)), sizeof(WorkerChipQueueDescSlot)
        );
        return true;
    }

    bool
    write_desc_slot(uint64_t slot_offset, const WorkerChipQueueDescSlot &slot, uint64_t seq, WorkerChipQueueOp op) {
        WorkerChipQueueDescSlot fields = slot;
        fields.seq = 0;
        if (!endpoint_.payload_write(slot_offset + offsetof(WorkerChipQueueDescSlot, opcode), &fields.opcode, 24)) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        if (!endpoint_.payload_write(slot_offset + offsetof(WorkerChipQueueDescSlot, seq), &seq, sizeof(seq))) {
            poison(WorkerChipQueueErrorKind::ENDPOINT_ERROR, op, endpoint_.error().message);
            return false;
        }
        return true;
    }

    static bool payload_in_arena(uint64_t offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
        if (nbytes == 0 || worker_chip_orch_comm::add_overflows(offset, nbytes)) {
            return false;
        }
        return offset >= arena_offset && offset + nbytes <= arena_offset + arena_bytes;
    }

    static uint64_t
    payload_expected_offset(uint64_t cursor, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes) {
        uint64_t arena_pos = cursor % arena_bytes;
        return arena_pos + nbytes > arena_bytes ? arena_offset : arena_offset + arena_pos;
    }

    bool payload_matches_head(
        uint64_t cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        WorkerChipQueueOp op
    ) {
        if (nbytes == 0) {
            return true;
        }
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (payload_offset != expected_offset) {
            poison(WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return false;
        }
        return true;
    }

    void advance_payload_head(
        uint64_t &cursor, uint64_t payload_offset, uint64_t nbytes, uint64_t arena_offset, uint64_t arena_bytes,
        WorkerChipQueueOp op
    ) {
        uint64_t arena_pos = cursor % arena_bytes;
        uint64_t expected_offset = payload_expected_offset(cursor, nbytes, arena_offset, arena_bytes);
        if (expected_offset != payload_offset) {
            poison(WorkerChipQueueErrorKind::INVALID_DESCRIPTOR, op, "payload replay offset mismatch");
            return;
        }
        if (arena_pos + nbytes > arena_bytes) {
            cursor += arena_bytes - (cursor % arena_bytes);
        }
        cursor += nbytes;
    }

    WorkerChipOrchEndpoint endpoint_;
    WorkerChipQueueLayout layout_{};
    WorkerChipQueueError error_{WorkerChipQueueErrorKind::NONE, WorkerChipQueueOp::INIT, 0, ""};
    InputQueue input_queue_;
    OutputQueue output_queue_;
};
