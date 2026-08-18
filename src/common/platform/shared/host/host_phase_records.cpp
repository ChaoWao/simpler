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

#include "host/host_phase_records.h"

#include <unistd.h>

#include <fstream>

#include "common/unified_log.h"

namespace simpler::dfx {

HostPhaseRecordPool *HostPhaseRecordStore::arm(bool collect_records) {
    pool_ = HostPhaseRecordPool{};
    armed_ = false;
    finished_ = false;
    submitted_tasks_ = 0;
    // A pass's identity must not survive into the next one: write_records_jsonl
    // asks only whether the store is armed, so a pass written before finish()
    // would otherwise carry the previous invocation id and join to the wrong span.
    invocation_id_ = 0;
    if (!collect_records) {
        buffers_.clear();
        buffers_.shrink_to_fit();
        return nullptr;
    }

    // Allocated once and reused for every later pass: a pass's records are only
    // valid until the next arm(), so the storage can be too.
    buffers_.resize(static_cast<size_t>(PLATFORM_HOST_PHASE_BUFFERS));
    for (auto &buffer : buffers_) {
        buffer.count = 0;
    }

    // Buffer 0 is active and the rest fill the free queue in index order, so the
    // reader recovers fill order from the buffer index alone — the free queue
    // holds exactly PLATFORM_PROF_SLOT_COUNT spares, which is why the pool is
    // one buffer larger than that.
    pool_.head.current_buf_ptr = reinterpret_cast<uint64_t>(&buffers_[0]);
    pool_.head.current_buf_seq = 1;
    for (size_t i = 1; i < buffers_.size(); ++i) {
        pool_.free_queue.buffer_ptrs[(i - 1) % PLATFORM_PROF_SLOT_COUNT] = reinterpret_cast<uint64_t>(&buffers_[i]);
    }
    pool_.free_queue.head = 0;
    pool_.free_queue.tail = static_cast<uint32_t>(buffers_.size() - 1);
    armed_ = true;
    return &pool_;
}

void HostPhaseRecordStore::finish(uint64_t submitted_tasks, uint64_t invocation_id) {
    if (!armed_) return;
    submitted_tasks_ = submitted_tasks;
    invocation_id_ = invocation_id;
    finished_ = true;
    if (pool_.head.dropped_record_count > 0) {
        LOG_WARN(
            "Host phase pool dropped %u of %u records (capacity %zu); the per-event views are truncated while the "
            "per-kind totals stay exact",
            static_cast<unsigned>(pool_.head.dropped_record_count),
            static_cast<unsigned>(pool_.head.total_record_count), capacity()
        );
    }
}

int HostPhaseRecordStore::write_records_jsonl(const std::string &path) const {
    if (!armed_) return 0;
    const std::vector<HostPhaseRecord> all = records();
    if (all.empty() && dropped_records() == 0) return 0;

    // Appended one object per pass: a --rounds run emits a pass per round, and
    // each object carries the identity needed to join it to that round's spans.
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        LOG_ERROR("Host phase records: cannot open '%s' for append", path.c_str());
        return -1;
    }
    // One line per pass (JSON Lines), matching scope_stats.jsonl: a reader can
    // take a pass at a time without buffering the file or tracking brace depth.
    out << "{\"pid\": " << static_cast<int>(getpid()) << ", \"inv\": " << invocation_id_
        << ", \"clock\": \"host_monotonic_ns\", \"dropped\": " << dropped_records() << ", \"records\": [";
    bool first = true;
    for (const auto &record : all) {
        if (!first) out << ", ";
        out << "{\"phase\": \"" << host_phase_kind_name(static_cast<HostPhaseKind>(record.kind))
            << "\", \"start_ns\": " << record.start_ns << ", \"end_ns\": " << record.end_ns
            << ", \"detail\": " << record.payload << "}";
        first = false;
    }
    out << "]}\n";
    out.flush();
    if (!out.good()) {
        LOG_ERROR("Host phase records: write to '%s' failed", path.c_str());
        return -1;
    }
    LOG_DEBUG("Host phase records: appended %zu records to %s", all.size(), path.c_str());
    return 0;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::records() const {
    std::vector<HostPhaseRecord> out;
    if (!armed_) return out;
    out.reserve(pool_.head.total_record_count);
    for (const auto &buffer : buffers_) {
        const uint32_t count = buffer.count < PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER ?
                                   buffer.count :
                                   static_cast<uint32_t>(PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(buffer.records[i]);
        }
    }
    return out;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::submit_records() const {
    std::vector<HostPhaseRecord> out;
    for (const auto &record : records()) {
        if (host_phase_kind_submits_task(static_cast<HostPhaseKind>(record.kind))) {
            out.push_back(record);
        }
    }
    return out;
}

std::vector<HostPhaseRecord> HostPhaseRecordStore::device_upload_records() const {
    std::vector<HostPhaseRecord> out;
    for (const auto &record : records()) {
        if (host_phase_kind_is_device_upload(static_cast<HostPhaseKind>(record.kind))) {
            out.push_back(record);
        }
    }
    return out;
}

}  // namespace simpler::dfx
