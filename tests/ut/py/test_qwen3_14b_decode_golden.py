# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

from simpler_setup.goldens import qwen3_14b_decode as qwen


def _use_tiny_regime(monkeypatch):
    values = {
        "BATCH": 2,
        "HIDDEN": 4,
        "KV_HIDDEN": 2,
        "HEAD_DIM": 2,
        "HALF_DIM": 1,
        "INTERMEDIATE": 3,
        "MAX_SEQ": 4,
        "MAX_BLOCKS_PER_SEQ": 2,
        "BLOCK_SIZE": 2,
        "CACHE_ROWS": 8,
    }
    for name, value in values.items():
        monkeypatch.setattr(qwen, name, value)


def _legacy_tensors(seed: int, seq_len: int, n_layers: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0, bias=0.0):
        return torch.empty(shape).normal_(0.0, std, generator=g) + bias

    def s0(t):
        return torch.cat([t] * n_layers, dim=0).contiguous()

    seq_lens = torch.full([qwen.BATCH], seq_len, dtype=torch.int32)
    block_table, slot_mapping = qwen._paged_block_table_slot_mapping(seq_lens)
    posv = torch.arange(qwen.MAX_SEQ).float().unsqueeze(1)
    inv_freq = 1.0 / (qwen.ROPE_THETA ** (torch.arange(0, qwen.HALF_DIM).float() / qwen.HALF_DIM))
    ang = posv * inv_freq.unsqueeze(0)

    return {
        "hidden_states": rn([qwen.BATCH, qwen.HIDDEN]).to(torch.bfloat16),
        "input_rms_weight": s0(rn([1, qwen.HIDDEN], 0.1, 1.0).float()),
        "wq": s0(rn([qwen.HIDDEN, qwen.HIDDEN], 0.02).to(torch.bfloat16)),
        "wk": s0(rn([qwen.HIDDEN, qwen.KV_HIDDEN], 0.02).to(torch.bfloat16)),
        "wv": s0(rn([qwen.HIDDEN, qwen.KV_HIDDEN], 0.02).to(torch.bfloat16)),
        "q_norm_weight": s0(rn([1, qwen.HEAD_DIM], 0.1, 1.0).float()),
        "k_norm_weight": s0(rn([1, qwen.HEAD_DIM], 0.1, 1.0).float()),
        "seq_lens": seq_lens,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "rope_cos": torch.cat([ang.cos(), ang.cos()], dim=1).float(),
        "rope_sin": torch.cat([ang.sin(), ang.sin()], dim=1).float(),
        "k_cache": s0(rn([qwen.CACHE_ROWS, qwen.HEAD_DIM], 0.01).to(torch.bfloat16)),
        "v_cache": s0(rn([qwen.CACHE_ROWS, qwen.HEAD_DIM], 0.02, 0.3).to(torch.bfloat16)),
        "wo": s0(rn([qwen.HIDDEN, qwen.HIDDEN], 0.0006).to(torch.bfloat16)),
        "w_gate": s0(rn([qwen.HIDDEN, qwen.INTERMEDIATE], 0.02).to(torch.bfloat16)),
        "w_up": s0(rn([qwen.HIDDEN, qwen.INTERMEDIATE], 0.02).to(torch.bfloat16)),
        "w_down": s0(rn([qwen.INTERMEDIATE, qwen.HIDDEN], 0.0004).to(torch.bfloat16)),
        "post_rms_weight": s0(rn([1, qwen.HIDDEN], 0.1, 1.0).float()),
        "out": torch.zeros([qwen.BATCH, qwen.HIDDEN], dtype=torch.bfloat16),
    }


def test_streaming_fixture_is_byte_identical_to_legacy_generator(monkeypatch):
    _use_tiny_regime(monkeypatch)
    expected = _legacy_tensors(seed=17, seq_len=3, n_layers=2)
    streamed = dict(qwen.param_tensors(seed=17, seq_len=3, n_layers=2))

    assert list(streamed) == [*qwen.INPUT_NAMES, "out"]
    for name, tensor in streamed.items():
        assert torch.equal(tensor, expected[name]), name

    materialized = qwen.generate_inputs(seed=17, seq_len=3, n_layers=2)
    assert materialized.tensor_names() == [*qwen.INPUT_NAMES, "out"]
    for name in materialized.tensor_names():
        assert torch.equal(getattr(materialized, name), expected[name]), name


def test_param_specs_match_streamed_fixture(monkeypatch):
    _use_tiny_regime(monkeypatch)
    specs = qwen.param_specs(n_layers=2)
    tensors = dict(qwen.param_tensors(seed=17, seq_len=3, n_layers=2))
    torch_dtypes = {
        "BFLOAT16": torch.bfloat16,
        "FLOAT32": torch.float32,
        "INT32": torch.int32,
    }

    assert [spec.name for spec in specs] == [*qwen.INPUT_NAMES, "out"]
    for spec in specs:
        assert tensors[spec.name].shape == spec.shape
        assert tensors[spec.name].dtype == torch_dtypes[spec.dtype]
