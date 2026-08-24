# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit coverage for direct-to-final-storage Qwen input generation."""

import torch

from simpler_setup.goldens import qwen3_14b_decode as qwen


class _RecordingAllocator:
    def __init__(self):
        self.tensors = []

    def empty(self, shape, *, dtype=None):
        tensor = torch.empty(shape, dtype=dtype)
        self.tensors.append(tensor)
        return tensor


def test_qwen_args_are_generated_in_allocator_storage(monkeypatch):
    dimensions = {
        "BATCH": 2,
        "HIDDEN": 8,
        "KV_HIDDEN": 4,
        "HEAD_DIM": 4,
        "HALF_DIM": 2,
        "INTERMEDIATE": 6,
        "MAX_SEQ": 8,
        "BLOCK_SIZE": 2,
        "MAX_BLOCKS_PER_SEQ": 4,
        "CACHE_ROWS": 16,
    }
    for name, value in dimensions.items():
        monkeypatch.setattr(qwen, name, value)

    allocator = _RecordingAllocator()
    args = qwen.generate_inputs(seed=7, seq_len=5, n_layers=2, allocator=allocator)
    allocated_ptrs = {tensor.data_ptr() for tensor in allocator.tensors}

    assert args.tensor_names() == [*qwen.INPUT_NAMES, "out"]
    assert all(spec.value.data_ptr() in allocated_ptrs for spec in args.specs)
    assert args.wq.shape == (2 * dimensions["HIDDEN"], dimensions["HIDDEN"])
    assert torch.equal(args.wq[: dimensions["HIDDEN"]], args.wq[dimensions["HIDDEN"] :])
    assert torch.count_nonzero(args.out) == 0
