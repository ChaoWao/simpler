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

#include "host/args_dump_collector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

void *test_alloc(size_t size) { return std::calloc(1, size); }

int test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

class TestArgsDumpCollector : public ArgsDumpCollector {
public:
    void *get_dump_shm_host_ptr() const { return shm_host_; }
};

}  // namespace

TEST(ArgsDumpCollectorTest, MergesConcurrentShardRecordsIntoManifest) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("args_dump_collector_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(test_dir));

    constexpr int kRecordsPerShard = 32;
    constexpr int kShardCount = DumpModule::kMaxCollectorThreads;
    ArgsDumpCollector collector;
    ASSERT_EQ(
        collector.initialize(kShardCount, 0, test_alloc, nullptr, test_free, test_dir.string(), DumpArgsLevel::HYBRID),
        0
    );

    std::vector<DumpMetaBuffer> buffers(kShardCount);
    std::atomic<int> ready_workers{0};
    std::atomic<bool> start_workers{false};
    std::vector<std::thread> workers;
    workers.reserve(kShardCount);
    for (int shard = 0; shard < kShardCount; shard++) {
        workers.emplace_back([&collector, &buffers, &ready_workers, &start_workers, shard] {
            DumpMetaBuffer &buffer = buffers[shard];
            buffer.count = kRecordsPerShard;
            for (int record = 0; record < kRecordsPerShard; record++) {
                ArgsDumpRecord &entry = buffer.records[record];
                entry.task_id = static_cast<uint64_t>(shard * kRecordsPerShard + record);
                entry.role = static_cast<uint8_t>(ArgsDumpRole::INPUT);
                entry.stage = static_cast<uint8_t>(ArgsDumpStage::BEFORE_DISPATCH);
                entry.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
                entry.arg_index = static_cast<uint32_t>(record);
                entry.func_ids[0] = static_cast<uint16_t>(shard);
                entry.func_count = 1;
            }

            DumpReadyBufferInfo info{};
            info.thread_index = 0;
            info.host_buffer_ptr = &buffer;
            ready_workers.fetch_add(1, std::memory_order_release);
            while (!start_workers.load(std::memory_order_acquire)) {}
            collector.on_buffer_collected(info, shard);
        });
    }
    while (ready_workers.load(std::memory_order_acquire) != kShardCount) {}
    start_workers.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }

    ASSERT_EQ(collector.export_dump_files(), 0);
    const std::filesystem::path manifest_path = test_dir / "args_dump" / "args_dump.json";
    std::ifstream manifest_file(manifest_path);
    ASSERT_TRUE(manifest_file.is_open());
    const std::string manifest{std::istreambuf_iterator<char>(manifest_file), std::istreambuf_iterator<char>()};
    EXPECT_NE(manifest.find("\"dump_args_level\": 3"), std::string::npos);
    EXPECT_NE(manifest.find("\"bin_file\": null"), std::string::npos);
    EXPECT_NE(manifest.find("\"total_args\": " + std::to_string(kShardCount * kRecordsPerShard)), std::string::npos);
    EXPECT_EQ(manifest.find("\"dropped_overwrite\""), std::string::npos);
    EXPECT_EQ(manifest.find("\"overwritten\""), std::string::npos);

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(test_dir);
}

TEST(ArgsDumpCollectorTest, BackpressureReleaseWaitsForAllPublishedPayloads) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("args_dump_count_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(test_dir));

    constexpr int kArenaCount = 2;
    constexpr uint64_t kPayloadSize = sizeof(uint64_t);
    TestArgsDumpCollector collector;
    ASSERT_EQ(
        collector.initialize(kArenaCount, 0, test_alloc, nullptr, test_free, test_dir.string(), DumpArgsLevel::FULL), 0
    );

    auto *device_base = collector.get_dump_shm_device_ptr();
    ASSERT_NE(device_base, nullptr);
    auto *host_base = collector.get_dump_shm_host_ptr();
    ASSERT_NE(host_base, nullptr);
    DumpDataHeader *header = get_dump_header(host_base);
    ASSERT_NE(header, nullptr);
    EXPECT_FALSE(collector.backpressure_release_ready());
    header->backpressure.fq_freeze_active = 1;
    std::vector<DumpMetaBuffer> buffers(kArenaCount);
    for (int arena_index = 0; arena_index < kArenaCount; arena_index++) {
        DumpBufferState *state = get_dump_buffer_state(device_base, arena_index);
        ASSERT_NE(state, nullptr);
        auto *arena = reinterpret_cast<uint64_t *>(state->arena_base);
        ASSERT_NE(arena, nullptr);
        arena[0] = static_cast<uint64_t>(arena_index + 1);
        state->arena_write_offset = kPayloadSize;
        state->published_payload_count = 1;

        DumpMetaBuffer &buffer = buffers[arena_index];
        buffer.count = 1;
        ArgsDumpRecord &record = buffer.records[0];
        record.task_id = static_cast<uint64_t>(arena_index);
        record.role = static_cast<uint8_t>(ArgsDumpRole::INPUT);
        record.stage = static_cast<uint8_t>(ArgsDumpStage::BEFORE_DISPATCH);
        record.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        record.dtype = static_cast<uint8_t>(DataType::UINT64);
        record.ndims = 1;
        record.shapes[0] = 1;
        record.strides[0] = 1;
        record.payload_offset = 0;
        record.payload_size = kPayloadSize;
    }

    EXPECT_FALSE(collector.backpressure_release_ready());

    DumpReadyBufferInfo first{};
    first.thread_index = 0;
    first.host_buffer_ptr = &buffers[0];
    collector.on_buffer_collected(first, 0);
    EXPECT_FALSE(collector.backpressure_release_ready());

    DumpReadyBufferInfo second{};
    second.thread_index = 1;
    second.host_buffer_ptr = &buffers[1];
    collector.on_buffer_collected(second, 1);
    ASSERT_EQ(collector.export_dump_files(), 0);
    EXPECT_TRUE(collector.backpressure_release_ready());
    for (int arena_index = 0; arena_index < kArenaCount; arena_index++) {
        DumpBufferState *state = get_dump_buffer_state(device_base, arena_index);
        EXPECT_EQ(state->completed_payload_count, state->published_payload_count);
    }

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(test_dir);
}

TEST(ArgsDumpCollectorTest, BackpressureReleaseDoesNotOffsetPayloadsAcrossThreads) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("args_dump_thread_count_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(test_dir));

    TestArgsDumpCollector collector;
    ASSERT_EQ(collector.initialize(2, 0, test_alloc, nullptr, test_free, test_dir.string(), DumpArgsLevel::FULL), 0);

    auto *device_base = collector.get_dump_shm_device_ptr();
    ASSERT_NE(device_base, nullptr);
    auto *host_base = collector.get_dump_shm_host_ptr();
    ASSERT_NE(host_base, nullptr);
    DumpDataHeader *header = get_dump_header(host_base);
    ASSERT_NE(header, nullptr);
    header->backpressure.fq_freeze_active = 1;
    DumpBufferState *thread0_state = get_dump_buffer_state(device_base, 0);
    DumpBufferState *thread1_state = get_dump_buffer_state(device_base, 1);
    ASSERT_NE(thread0_state, nullptr);
    ASSERT_NE(thread1_state, nullptr);

    constexpr uint64_t kPayloadSize = sizeof(uint64_t);
    auto *arena = reinterpret_cast<uint64_t *>(thread0_state->arena_base);
    ASSERT_NE(arena, nullptr);
    arena[0] = 1;
    thread0_state->arena_write_offset = kPayloadSize;

    DumpMetaBuffer buffer{};
    buffer.count = 1;
    ArgsDumpRecord &record = buffer.records[0];
    record.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    record.dtype = static_cast<uint8_t>(DataType::UINT64);
    record.ndims = 1;
    record.shapes[0] = 1;
    record.strides[0] = 1;
    record.payload_size = kPayloadSize;

    DumpReadyBufferInfo info{};
    info.thread_index = 0;
    info.host_buffer_ptr = &buffer;
    collector.on_buffer_collected(info, 0);
    ASSERT_EQ(collector.export_dump_files(), 0);

    // Model a skewed global snapshot: one payload was written for thread 0,
    // while the sampled publication belongs to thread 1. Equal totals must
    // not acknowledge either thread.
    thread0_state->published_payload_count = 0;
    thread1_state->published_payload_count = 1;
    EXPECT_FALSE(collector.backpressure_release_ready());
    EXPECT_EQ(thread0_state->completed_payload_count, 0u);
    EXPECT_EQ(thread1_state->completed_payload_count, 0u);

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(test_dir);
}
