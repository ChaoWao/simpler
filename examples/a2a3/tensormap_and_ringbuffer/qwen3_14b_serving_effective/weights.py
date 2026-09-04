#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B checkpoint conversion for the decode callable ABI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


NUM_LAYERS = 40
HEAD_DIM = 128
HIDDEN = 5120
PADDED_VOCAB = 152064
MAX_SEQ_LEN = 4096


class CheckpointReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def load(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def load_optional(self, name: str, default: torch.Tensor) -> torch.Tensor:
        return self.load(name) if name in self.weight_map else default


def _stack_layers(
    reader: CheckpointReader,
    suffix: str,
    *,
    dtype: torch.dtype,
    transpose: bool,
) -> torch.Tensor:
    layers = []
    for layer in range(NUM_LAYERS):
        tensor = reader.load(f"model.layers.{layer}.{suffix}")
        tensor = tensor.transpose(0, 1) if transpose else tensor.view(1, -1)
        layers.append(tensor.to(dtype).contiguous())
    return torch.cat(layers, dim=0).contiguous()


def iter_kernel_weights(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    reader = CheckpointReader(model_dir)
    rules = (
        ("input_rms_weight", "input_layernorm.weight", torch.float32, False),
        ("wq", "self_attn.q_proj.weight", torch.bfloat16, True),
        ("wk", "self_attn.k_proj.weight", torch.bfloat16, True),
        ("wv", "self_attn.v_proj.weight", torch.bfloat16, True),
        ("wo", "self_attn.o_proj.weight", torch.bfloat16, True),
        ("w_gate", "mlp.gate_proj.weight", torch.bfloat16, True),
        ("w_up", "mlp.up_proj.weight", torch.bfloat16, True),
        ("w_down", "mlp.down_proj.weight", torch.bfloat16, True),
        ("post_rms_weight", "post_attention_layernorm.weight", torch.float32, False),
    )
    for target, suffix, dtype, transpose in rules:
        yield target, _stack_layers(reader, suffix, dtype=dtype, transpose=transpose)

    for target, suffix in (
        ("q_norm_weight", "q_norm.weight"),
        ("k_norm_weight", "k_norm.weight"),
    ):
        layers = []
        for layer in range(NUM_LAYERS):
            layers.append(
                reader.load_optional(
                    f"model.layers.{layer}.self_attn.{suffix}",
                    torch.ones(HEAD_DIM, dtype=torch.float32),
                )
                .view(1, -1)
                .float()
            )
        yield target, torch.cat(layers, dim=0).contiguous()

    yield (
        "final_norm_weight",
        reader.load("model.norm.weight").view(1, -1).float().contiguous(),
    )

    embed = reader.load("model.embed_tokens.weight").to(torch.bfloat16).contiguous()
    if embed.shape[0] < PADDED_VOCAB:
        padding = torch.zeros(
            (PADDED_VOCAB - embed.shape[0], HIDDEN), dtype=embed.dtype
        )
        embed = torch.cat((embed, padding), dim=0).contiguous()
    yield "embed_weight", embed

    lm_name = (
        "lm_head.weight"
        if "lm_head.weight" in reader.weight_map
        else "model.embed_tokens.weight"
    )
    lm_head = reader.load(lm_name).to(torch.bfloat16).contiguous()
    if lm_head.shape[0] < PADDED_VOCAB:
        padding = lm_head[:1].expand(PADDED_VOCAB - lm_head.shape[0], -1).clone()
        lm_head = torch.cat((lm_head, padding), dim=0).contiguous()
    yield "lm_head_weight", lm_head


def rope_tables(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    config = json.loads((model_dir / "config.json").read_text())
    theta = float(config.get("rope_theta", 1_000_000.0))
    half = HEAD_DIM // 2
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    angles = positions * inv_freq.unsqueeze(0)
    return (
        torch.cat((angles.cos(), angles.cos()), dim=1).contiguous(),
        torch.cat((angles.sin(), angles.sin()), dim=1).contiguous(),
    )
