# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixture generator for the DeepSeek-V4 FLASH full-decode SceneTestCase (EP2/TP2, 2 dies).

Harvested from pypto-lib ``models/deepseek_v4_flash_mtp/decode_fwd.py`` — its
``build_tensor_specs()`` at the harvest defaults (start_pos=8192, num_tokens=8,
EP=2, TP=2): one packed decode step of the full 43-layer DeepSeek-V4 Flash
network (embed -> 43 x attention+MoE layers -> hyper-connection head -> final
RMSNorm -> TP2 LM head with sampling). The 43 forward layers are 2 SWA
(ratio 0) + 21 CSA (ratio 4) + 20 HCA (ratio 128) attention kinds; every layer
runs the distributed MoE (64 experts at EP2 = 32 local experts per rank, the
256-expert model scaled by moe.py's ``--ep`` override). Regime: B=4 requests x
S=2 tokens = T=8 token rows, every request at start_pos=8192 (kv_seq_lens 8193).

The scene test is a smoke/completion case (skip_golden — upstream decode_fwd
has no golden either), so this module only builds the input/output host
tensors; there is deliberately no ``compute_golden``.

Per-rank ``_r0`` / ``_r1`` naming: the pypto fixture stacks every parameter as
``[N_RANKS=2, *tail]`` and slices ``value[r]`` per rank at submit time. The
scene-test framework re-hosts tensors and cannot pass slices (views) of one
stacked storage, so each base name expands into TWO tensors ``<name>_r0`` /
``<name>_r1`` of shape ``tail``, preserving the per-rank semantics of the
original init: replicated inits give both ranks equal content (r1 is a clone so
storages never alias); genuinely per-rank inits (routed-expert EP shards,
hc_head_fn, final_norm_w, the two TP2 lm_head vocab shards) draw independently
per rank. ``PARAM_NAMES`` lists the 80 base names in the generated program's
parameter order; the builder receives r0 then r1 per name in that order. One
exception: ``num_tokens_per_owner`` stays a single whole ``[N_RANKS]`` vector —
the chip program reads every rank's count, so the orchestration forwards the
full vector to each rank instead of a per-rank slice.

Upstream init semantics preserved here:
- Layer-stacked weights are ``torch.cat`` of per-layer draws along the stack
  dim. Leaf inits that redraw per call (e.g. wq_a, wkv, wo_a, the CSA/HCA
  compressor weights, gate_w, hc_attn_fn/hc_ffn_fn) get fresh randomness per
  layer; leaf inits captured once in the leaf spec builder (the INT8-quantized
  wq_b/wo_b/csa_idx_wq_b pairs, csa_weights_proj, the MoE routed/shared expert
  weights) repeat one draw across all layers.
- The KV / compressor-state pools are ALL-ZERO: decode_fwd overrides each
  pool's single-layer init to ``torch.zeros`` before layer-tiling, so the
  packed pools start empty and are written in place on device (InOut).
- Metadata (block tables, position_ids, kv_seq_lens, block_counts, tid2eid,
  input_ids, freqs) is transcribed exactly — these must be valid indices.

Determinism: all randomness flows through one ``torch.Generator`` seeded from
``seed``; the draw order is the fixed entry order below.

Memory: at full size this materializes ~100+ GB of host tensors (dominated by
the 43x-tiled routed-expert INT8 stacks). Intended for large-memory hosts.
"""

from __future__ import annotations

import math

import torch

from simpler_setup.scene_test import TaskArgsBuilder, TensorArg

# ── World / decode regime (decode_fwd harvest defaults) ──
N_RANKS = 2  # EP world size (2 dies)
LM_HEAD_TP_SIZE = 2
B = 4  # decode requests per step
S = 2  # tokens per request ([previous, current])
T = B * S  # 8 token rows
START_POS = 8192  # every request decodes at the 8k target position
COMMIT_TOKENS = 1  # kv_seq_lens = start_pos + commit_tokens

# ── Model architecture (DeepSeek-V4 FLASH preset) ──
D = 4096  # hidden size
H = 64  # attention heads
HEAD_DIM = 512  # MLA value-head dim
ROPE_HEAD_DIM = 64
Q_LORA = 1024
O_LORA = 1024
O_GROUPS = 8
O_GROUP_IN = H * HEAD_DIM // O_GROUPS  # 4096
HC_MULT = 4  # hyper-connection stack width
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
HC_DIM = HC_MULT * D  # 16384
VOCAB = 129280
VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE  # 64640
MAX_LOGIT_ROWS = T  # 8
SAMPLED_IDS_PAD = 8
MAX_SEQ_LEN = 16384

# ── Layer stacking (43 = 2 SWA + 21 CSA + 20 HCA) ──
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20

# ── MoE (EP2: moe.py scales the 256-expert model to n_routed = 256 // 8 * EP) ──
N_EXPERTS_GLOBAL = 64
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS  # 32 local experts per rank
MOE_INTER = 2048
TOPK = 6

# ── Paged KV / compressor-state pools (per-layer physical capacities) ──
BLOCK_SIZE = 128
ORI_BLOCK_NUM = 128
ORI_TABLE_MAX_BLOCKS = MAX_SEQ_LEN // BLOCK_SIZE  # 128
CMP_BLOCK_NUM = 32  # shared HCA/CSA compressed-KV pool
IDX_BLOCK_NUM = 64
CSA_IDX_N_HEADS = 64
CSA_IDX_HEAD_DIM = 128
CSA_MAIN_OUT_DIM = 1024  # ratio-4 overlap compressor: 2 * HEAD_DIM
CSA_STATE_BLOCK_NUM = 65
CSA_STATE_BLOCK_SIZE = 4
CSA_STATE_DIM = 2 * CSA_MAIN_OUT_DIM  # 2048
CSA_STATE_MAX_BLOCKS = MAX_SEQ_LEN // CSA_STATE_BLOCK_SIZE  # 4096
CSA_INNER_OUT_DIM = 256  # 2 * CSA_IDX_HEAD_DIM
CSA_INNER_STATE_BLOCK_NUM = 65
CSA_INNER_STATE_DIM = 2 * CSA_INNER_OUT_DIM  # 512
CSA_INNER_STATE_MAX_BLOCKS = CSA_STATE_MAX_BLOCKS  # 4096
HCA_MAIN_OUT_DIM = 512  # ratio-128 compressor: 1 * HEAD_DIM
HCA_STATE_BLOCK_NUM = 64
HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_DIM = 2 * HCA_MAIN_OUT_DIM  # 1024
HCA_STATE_MAX_BLOCKS = MAX_SEQ_LEN // HCA_STATE_BLOCK_SIZE  # 2048
CSA_COMPRESS_RATIO = 4
HCA_COMPRESS_RATIO = 128
N_CACHE_GROUPS = 6

# ── INT8 quantization (config.py) ──
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4

# ── RoPE / YaRN (FLASH preset) ──
ROPE_THETA = 1.0e4  # ratio-0 (SWA) profile
COMPRESS_ROPE_THETA = 1.6e5  # compress-ratio profiles (with YaRN)
ROPE_FACTOR = 16.0
BETA_FAST = 32
BETA_SLOW = 1
ORIGINAL_MAX_SEQ = 65536  # original_max_position_embeddings (YaRN reference)

# ── Real-layer hyper-connection gate constants (decode_swa / moe leaf builders) ──
SWA_HC_ATTN_SCALE = (2.076026, 0.018729, 0.245936)
SWA_HC_ATTN_BASE = (
    3.9083,
    -2.0399,
    -2.2033,
    -2.017,
    -2.4443,
    -10.3158,
    -8.9943,
    -6.3581,
    9.8577,
    -9.5177,
    -24.8724,
    -22.8929,
    -21.545,
    0.7791,
    -3.386,
    1.1948,
    -20.9605,
    -0.7702,
    1.4218,
    -4.8994,
    1.5177,
    -29.7663,
    -30.1413,
    -1.2413,
)
MOE_HC_FFN_SCALE = (0.11334, 0.035901, 0.058183)
MOE_HC_FFN_BASE = (
    2.4153,
    -2.0252,
    -2.0019,
    -2.1947,
    -1.5430,
    -3.0228,
    -6.8248,
    0.5894,
    2.1916,
    -7.2132,
    -3.0938,
    -2.1119,
    -3.0161,
    3.3293,
    -3.2224,
    -4.0226,
    -2.0428,
    -3.3478,
    3.0893,
    -3.4166,
    -1.8144,
    -3.8147,
    -3.1307,
    1.7862,
)
HC_HEAD_SCALE = 0.076099
HC_HEAD_BASE = (5.9166, -3.6223, -2.9324, -3.3124)

# 80 base names in the generated program's parameter order (decode_fwd
# build_tensor_specs ``ordered_names`` + the appended head/LM tail); matches
# the harvested distributed_meta.json param list with ``__ssa_v0`` stripped.
PARAM_NAMES: list[str] = [
    "embed_weight",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "block_counts",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "sampled_ids",
    "num_tokens_per_owner",
    "logit_row_indices",
]


# ---------------------------------------------------------------------------
# Init helpers (transcribed from the pypto-lib leaf spec builders)
# ---------------------------------------------------------------------------


def _quant_per_output_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """decode_swa.quant_w_per_output_channel: symmetric INT8, scale per output column."""
    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, w.shape[1])
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()


def _quant_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """decode_swa.quant_w_per_row: symmetric INT8, scale per row."""
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()


def _gen_routed_int8(g: torch.Generator, shape, dequant_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """expert_routed.gen_routed_weight: simulate the MXFP4 (e2m1 + per-32-group E8M0)
    routed-expert grid, then re-quantize per output channel to INT8 + FP32 scale."""
    fp4_mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    fp4_mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    fp4_max, tiny = 6.0, 1e-20
    group = 32
    chunk_elems = 1 << 25  # elements per pass; unchunked this walks GiB-sized temporaries

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    big = torch.randn(*shape, generator=g).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)
    step = max(1, chunk_elems // (out * inn))
    for i0 in range(0, n_lead, step):
        w = big[i0 : i0 + step]
        wg = w.reshape(-1, out, inn // group, group)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / fp4_max).clamp_min(tiny))))
        idx = torch.bucketize(absw.div_(grp_scale), fp4_mid).clamp_max_(7)
        wq = (torch.sign(wg) * fp4_mag[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0 : i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0 : i0 + step] = chan_scale
    del big
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def _gen_mxfp8_int8(g: torch.Generator, shape, dequant_std: float, chan_cv: float) -> tuple[torch.Tensor, torch.Tensor]:
    """expert_shared.gen_shared_weight / decode_indexer.gen_shared_weight: simulate the
    MXFP8 (e4m3 + 128x128-block E8M0) grid with a log-space per-output-channel gain,
    then re-quantize per output channel to INT8 + FP32 scale."""
    fp8_max, tiny = 448.0, 1e-20
    out, inn = shape
    w = torch.randn(*shape, generator=g) * torch.exp(chan_cv * torch.randn(out, 1, generator=g))
    wb = w.reshape(out // 128, 128, inn // 128, 128)
    blk_scale = torch.exp2(torch.ceil(torch.log2((wb.abs().amax(dim=(1, 3), keepdim=True) / fp8_max).clamp_min(tiny))))
    wq = ((wb / blk_scale).to(torch.float8_e4m3fn).float() * blk_scale).reshape(out, inn)
    amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def _hadamard(dim: int) -> torch.Tensor:
    """decode_csa.init_hadamard_idx: Sylvester Hadamard matrix scaled by 1/sqrt(dim)."""
    h = torch.ones((1, 1))
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (dim**0.5)


def _block_table(batch: int, table_blocks: int, physical_blocks: int) -> torch.Tensor:
    """utils.block_table (permuted=False): interleave request-local logical pages
    inside the fixed global physical pool."""
    physical_cols = torch.arange(table_blocks, dtype=torch.int32) % physical_blocks
    request_offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1)
    return (physical_cols.unsqueeze(0) * batch + request_offsets) % physical_blocks


def _rope_tables(base: float, original_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """utils.precompute_freqs_cos_sin: [MAX_SEQ_LEN, ROPE_HEAD_DIM] bf16 cos/sin with the
    half-width angles duplicated into both halves; original_seq_len > 0 enables the YaRN
    frequency correction used by the compress-ratio profiles."""
    dim = ROPE_HEAD_DIM
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def correction_dim(num_rotations: int) -> float:
            return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        low: float = max(math.floor(correction_dim(BETA_FAST)), 0)
        high: float = min(math.ceil(correction_dim(BETA_SLOW)), dim - 1)
        if low == high:
            high = high + 0.001
        ramp = torch.clamp((torch.arange(half_dim, dtype=torch.float32) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        inv_freq = inv_freq / ROPE_FACTOR * (1 - smooth) + inv_freq * smooth
    angles = torch.outer(torch.arange(MAX_SEQ_LEN, dtype=torch.float32), inv_freq)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(torch.bfloat16)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(torch.bfloat16)
    return cos, sin


def _tid2eid() -> torch.Tensor:
    """decode_fwd layer-stacked tid2eid: per layer, token t's k-th route is
    (t * TOPK + k + layer * TOPK) % N_EXPERTS_GLOBAL; layers concatenated along rows."""
    token_ids = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
    topk_ids = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
    rows = [(token_ids * TOPK + topk_ids + layer * TOPK) % N_EXPERTS_GLOBAL for layer in range(FWD_NUM_LAYERS)]
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------


def generate_inputs(seed: int = 1234) -> TaskArgsBuilder:
    """Deterministic input/output host tensors for the packed 43-layer decode.

    Returns a TaskArgsBuilder with 159 TensorArgs: for each of the 80 base
    names in PARAM_NAMES, ``<name>_r0`` then ``<name>_r1`` (per-rank halves of
    the upstream ``[2, *tail]`` stacked tensor), except ``num_tokens_per_owner``
    which stays one whole ``[N_RANKS]`` vector. No golden is computed — the
    scene test runs skip_golden.
    """
    g = torch.Generator().manual_seed(seed)

    def rn(*shape) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    def layer_cat(layers: int, shape, std: float, dtype: torch.dtype, mean: float = 0.0) -> torch.Tensor:
        """torch.cat of `layers` fresh (mean + std * randn(shape)) draws along dim 0 —
        the layer-stacking of leaf inits that redraw per call."""
        return torch.cat([(rn(*shape) * std + mean).to(dtype) for _ in range(layers)], dim=0)

    def tile(t: torch.Tensor, layers: int) -> torch.Tensor:
        """Repeat one per-layer tensor across the layer-stack dim (dim 0) — the
        layer-stacking of leaf inits captured once in the leaf spec builder."""
        return t.repeat(layers, *([1] * (t.dim() - 1)))

    # Pair makers: every entry produces (r0, r1) with non-aliasing storages.
    def rep(fn):
        """Replicated across ranks: both ranks get the same content (r1 cloned)."""

        def make():
            t = fn()
            return t, t.clone()

        return make

    def per_rank(fn):
        """Independent draw per rank."""

        def make():
            return fn(), fn()

        return make

    def zeros(shape, dtype: torch.dtype):
        def make():
            return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)

        return make

    def ones(shape, dtype: torch.dtype):
        def make():
            return torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)

        return make

    # ── Single-draw weights reused across the layer stacks (upstream draws these
    # once in the leaf spec builder and re-reads the same tensor per layer) ──
    wq_b_i8, wq_b_sc = _quant_per_output_channel((rn(Q_LORA, H * HEAD_DIM) / Q_LORA**0.5).to(torch.bfloat16))
    wo_b_i8, wo_b_sc = _quant_per_row((rn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5).to(torch.bfloat16))
    # Indexer wq_b follows the shared-expert MXFP8 grid; built [out, in] then transposed.
    idx_wq_b_t, idx_wq_b_sc = _gen_mxfp8_int8(g, (CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM, Q_LORA), 0.108, 0.56)
    idx_wq_b_i8 = idx_wq_b_t.t().contiguous()
    weights_proj = (rn(D, CSA_IDX_N_HEADS) * 0.2313).to(torch.bfloat16)
    hadamard = _hadamard(CSA_IDX_HEAD_DIM).to(torch.bfloat16)
    # Shared expert weights (replicated across ranks); dequant stds from moe.py.
    sw1_i8, sw1_sc = _gen_mxfp8_int8(g, (MOE_INTER, D), 7.65e-3, 0.50)
    sw3_i8, sw3_sc = _gen_mxfp8_int8(g, (MOE_INTER, D), 7.39e-3, 0.50)
    sw2_i8, sw2_sc = _gen_mxfp8_int8(g, (D, MOE_INTER), 2.39e-2, 0.33)
    # Routed expert weights: independent EP shard per rank; dequant stds from moe.py.
    routed_draws = [
        (
            _gen_routed_int8(g, (N_LOCAL, MOE_INTER, D), 1.08e-2),  # w1
            _gen_routed_int8(g, (N_LOCAL, MOE_INTER, D), 1.10e-2),  # w3
            _gen_routed_int8(g, (N_LOCAL, D, MOE_INTER), 2.54e-2),  # w2
        )
        for _ in range(N_RANKS)
    ]

    def routed_pair(expert_idx: int, part: int):
        """(r0, r1) for one routed tensor: rank r's single-layer shard tiled 43x."""

        def make():
            return (
                tile(routed_draws[0][expert_idx][part], FWD_NUM_LAYERS),
                tile(routed_draws[1][expert_idx][part], FWD_NUM_LAYERS),
            )

        return make

    # ── RoPE tables: [swa profile, compress profile] stacked on dim 0 ──
    swa_cos, swa_sin = _rope_tables(ROPE_THETA, 0)
    csa_cos, csa_sin = _rope_tables(COMPRESS_ROPE_THETA, ORIGINAL_MAX_SEQ)

    # ── Forward metadata (exact transcription; identical on both ranks) ──
    starts = torch.full((B,), START_POS, dtype=torch.int32)
    position_ids = (starts.unsqueeze(1) + torch.arange(S, dtype=torch.int32).unsqueeze(0)).reshape(-1).contiguous()
    kv_seq_lens = (starts.to(torch.int64) + COMMIT_TOKENS).to(torch.int32)
    # block_counts row: [ori, cmp, idx, hca_state, csa_state, inner_state] physical
    # blocks; the state pools shrink to per-request shares (block_num // B).
    block_counts = (
        torch.tensor(
            [
                ORI_BLOCK_NUM,
                CMP_BLOCK_NUM,
                IDX_BLOCK_NUM,
                HCA_STATE_BLOCK_NUM // B,
                CSA_STATE_BLOCK_NUM // B,
                CSA_INNER_STATE_BLOCK_NUM // B,
            ],
            dtype=torch.int32,
        )
        .view(1, N_CACHE_GROUPS)
        .repeat(B, 1)
    )
    active_rows = max(min(T, MAX_LOGIT_ROWS), 0)
    logit_row_indices = torch.full((MAX_LOGIT_ROWS,), -1, dtype=torch.int32)
    logit_row_indices[:active_rows] = torch.arange(active_rows, dtype=torch.int32)

    # ── Entry table: (name, tail shape, dtype, pair maker), in PARAM_NAMES order.
    # tail = upstream stacked shape minus the leading N_RANKS dim. ──
    entries = [
        (
            "embed_weight",
            (VOCAB, D),
            torch.bfloat16,
            rep(lambda: torch.randn(VOCAB, D, dtype=torch.bfloat16, generator=g)),
        ),
        # Attention stack (43 layers, all from the SWA leaf builder).
        (
            "hc_attn_fn",
            (FWD_NUM_LAYERS * MIX_HC, HC_DIM),
            torch.float32,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (MIX_HC, HC_DIM), 0.039, torch.float32)),
        ),
        (
            "hc_attn_scale",
            (FWD_NUM_LAYERS * 3,),
            torch.float32,
            rep(lambda: tile(torch.tensor(SWA_HC_ATTN_SCALE), FWD_NUM_LAYERS)),
        ),
        (
            "hc_attn_base",
            (FWD_NUM_LAYERS * MIX_HC,),
            torch.float32,
            rep(lambda: tile(torch.tensor(SWA_HC_ATTN_BASE), FWD_NUM_LAYERS)),
        ),
        ("attn_norm_w", (FWD_NUM_LAYERS * D,), torch.bfloat16, ones((FWD_NUM_LAYERS * D,), torch.bfloat16)),
        (
            "wq_a",
            (FWD_NUM_LAYERS * D, Q_LORA),
            torch.bfloat16,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (D, Q_LORA), D**-0.5, torch.bfloat16)),
        ),
        ("wq_b", (FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM), torch.int8, rep(lambda: tile(wq_b_i8, FWD_NUM_LAYERS))),
        ("wq_b_scale", (FWD_NUM_LAYERS * H * HEAD_DIM,), torch.float32, rep(lambda: tile(wq_b_sc, FWD_NUM_LAYERS))),
        (
            "wkv",
            (FWD_NUM_LAYERS * D, HEAD_DIM),
            torch.bfloat16,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (D, HEAD_DIM), D**-0.5, torch.bfloat16)),
        ),
        ("gamma_cq", (FWD_NUM_LAYERS * Q_LORA,), torch.bfloat16, ones((FWD_NUM_LAYERS * Q_LORA,), torch.bfloat16)),
        ("gamma_ckv", (FWD_NUM_LAYERS * HEAD_DIM,), torch.bfloat16, ones((FWD_NUM_LAYERS * HEAD_DIM,), torch.bfloat16)),
        # Cache pools are zero-initialized upstream and written in place on device.
        (
            "kv_cache",
            (FWD_NUM_LAYERS * ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM),
            torch.bfloat16,
            zeros((FWD_NUM_LAYERS * ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), torch.bfloat16),
        ),
        (
            "attn_sink",
            (FWD_NUM_LAYERS * H,),
            torch.float32,
            zeros((FWD_NUM_LAYERS * H,), torch.float32),
        ),  # SWA leaf init: zeros(H)
        (
            "wo_a",
            (FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN),
            torch.bfloat16,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (O_GROUPS, O_LORA, O_GROUP_IN), O_GROUP_IN**-0.5, torch.bfloat16)),
        ),
        ("wo_b", (FWD_NUM_LAYERS * D, O_GROUPS * O_LORA), torch.int8, rep(lambda: tile(wo_b_i8, FWD_NUM_LAYERS))),
        ("wo_b_scale", (FWD_NUM_LAYERS * D,), torch.float32, rep(lambda: tile(wo_b_sc, FWD_NUM_LAYERS))),
        # HCA compressor stack (20 layers); stds from the decode_hca leaf builder.
        (
            "hca_cmp_wkv",
            (HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(HCA_NUM_LAYERS, (HCA_MAIN_OUT_DIM, D), 0.0246, torch.bfloat16)),
        ),
        (
            "hca_cmp_wgate",
            (HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(HCA_NUM_LAYERS, (HCA_MAIN_OUT_DIM, D), 0.0316, torch.bfloat16)),
        ),
        (
            "hca_cmp_ape",
            (HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM),
            torch.float32,
            rep(lambda: layer_cat(HCA_NUM_LAYERS, (HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM), 0.0340, torch.float32)),
        ),
        (
            "hca_cmp_norm_w",
            (HCA_NUM_LAYERS * HEAD_DIM,),
            torch.bfloat16,
            rep(lambda: layer_cat(HCA_NUM_LAYERS, (HEAD_DIM,), 0.0549, torch.bfloat16, mean=0.1001)),
        ),
        (
            "hca_compress_state",
            (HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_STATE_DIM),
            torch.float32,
            zeros((HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_STATE_DIM), torch.float32),
        ),
        # CSA compressor + indexer stack (21 layers); stds from the decode_csa leaf builder.
        (
            "csa_cmp_wkv",
            (CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_MAIN_OUT_DIM, D), 0.0245, torch.bfloat16)),
        ),
        (
            "csa_cmp_wgate",
            (CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_MAIN_OUT_DIM, D), 0.0388, torch.bfloat16)),
        ),
        (
            "csa_cmp_ape",
            (CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM),
            torch.float32,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM), 0.1243, torch.float32)),
        ),
        (
            "csa_cmp_norm_w",
            (CSA_NUM_LAYERS * HEAD_DIM,),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (HEAD_DIM,), 0.1929, torch.bfloat16, mean=0.9666)),
        ),
        (
            "csa_compress_state",
            (CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_STATE_DIM),
            torch.float32,
            zeros((CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_STATE_DIM), torch.float32),
        ),
        (
            "csa_idx_wq_b",
            (CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM),
            torch.int8,
            rep(lambda: tile(idx_wq_b_i8, CSA_NUM_LAYERS)),
        ),
        (
            "csa_idx_wq_b_scale",
            (CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM,),
            torch.float32,
            rep(lambda: tile(idx_wq_b_sc, CSA_NUM_LAYERS)),
        ),
        (
            "csa_weights_proj",
            (CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS),
            torch.bfloat16,
            rep(lambda: tile(weights_proj, CSA_NUM_LAYERS)),
        ),
        (
            "csa_hadamard_idx",
            (CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM),
            torch.bfloat16,
            rep(lambda: tile(hadamard, CSA_NUM_LAYERS)),
        ),
        (
            "csa_inner_wkv",
            (CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_INNER_OUT_DIM, D), 0.0293, torch.bfloat16)),
        ),
        (
            "csa_inner_wgate",
            (CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_INNER_OUT_DIM, D), 0.0512, torch.bfloat16)),
        ),
        (
            "csa_inner_ape",
            (CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM),
            torch.float32,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM), 0.1528, torch.float32)),
        ),
        (
            "csa_inner_norm_w",
            (CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM,),
            torch.bfloat16,
            rep(lambda: layer_cat(CSA_NUM_LAYERS, (CSA_IDX_HEAD_DIM,), 0.2610, torch.bfloat16, mean=0.6850)),
        ),
        (
            "csa_inner_compress_state",
            (CSA_NUM_LAYERS * CSA_INNER_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM),
            torch.float32,
            zeros(
                (CSA_NUM_LAYERS * CSA_INNER_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM), torch.float32
            ),
        ),
        (
            "cmp_kv",
            (FWD_NUM_LAYERS * CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM),
            torch.bfloat16,
            zeros((FWD_NUM_LAYERS * CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), torch.bfloat16),
        ),
        (
            "idx_kv_cache",
            (CSA_NUM_LAYERS * IDX_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM),
            torch.int8,
            zeros((CSA_NUM_LAYERS * IDX_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM), torch.int8),
        ),
        (
            "idx_kv_scale",
            (CSA_NUM_LAYERS * IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1),
            torch.float32,
            zeros((CSA_NUM_LAYERS * IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1), torch.float32),
        ),
        # MoE stack (43 layers, from the moe leaf builder).
        (
            "hc_ffn_fn",
            (FWD_NUM_LAYERS * MIX_HC, HC_DIM),
            torch.float32,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (MIX_HC, HC_DIM), 0.0635, torch.float32)),
        ),
        (
            "hc_ffn_scale",
            (FWD_NUM_LAYERS * 3,),
            torch.float32,
            rep(lambda: tile(torch.tensor(MOE_HC_FFN_SCALE), FWD_NUM_LAYERS)),
        ),
        (
            "hc_ffn_base",
            (FWD_NUM_LAYERS * MIX_HC,),
            torch.float32,
            rep(lambda: tile(torch.tensor(MOE_HC_FFN_BASE), FWD_NUM_LAYERS)),
        ),
        ("norm_w", (FWD_NUM_LAYERS * D,), torch.bfloat16, ones((FWD_NUM_LAYERS * D,), torch.bfloat16)),
        (
            "gate_w",
            (FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D),
            torch.float32,
            rep(lambda: layer_cat(FWD_NUM_LAYERS, (N_EXPERTS_GLOBAL, D), D**-0.5, torch.float32)),
        ),
        (
            "gate_bias",
            (FWD_NUM_LAYERS * N_EXPERTS_GLOBAL,),
            torch.float32,
            zeros((FWD_NUM_LAYERS * N_EXPERTS_GLOBAL,), torch.float32),
        ),
        ("tid2eid", (FWD_NUM_LAYERS * VOCAB, TOPK), torch.int32, rep(_tid2eid)),
        ("routed_w1", (FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D), torch.int8, routed_pair(0, 0)),
        ("routed_w1_scale", (FWD_NUM_LAYERS * N_LOCAL, MOE_INTER), torch.float32, routed_pair(0, 1)),
        ("routed_w3", (FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D), torch.int8, routed_pair(1, 0)),
        ("routed_w3_scale", (FWD_NUM_LAYERS * N_LOCAL, MOE_INTER), torch.float32, routed_pair(1, 1)),
        ("routed_w2", (FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER), torch.int8, routed_pair(2, 0)),
        ("routed_w2_scale", (FWD_NUM_LAYERS * N_LOCAL, D), torch.float32, routed_pair(2, 1)),
        ("shared_w1", (FWD_NUM_LAYERS * MOE_INTER, D), torch.int8, rep(lambda: tile(sw1_i8, FWD_NUM_LAYERS))),
        ("shared_w1_scale", (FWD_NUM_LAYERS * MOE_INTER,), torch.float32, rep(lambda: tile(sw1_sc, FWD_NUM_LAYERS))),
        ("shared_w3", (FWD_NUM_LAYERS * MOE_INTER, D), torch.int8, rep(lambda: tile(sw3_i8, FWD_NUM_LAYERS))),
        ("shared_w3_scale", (FWD_NUM_LAYERS * MOE_INTER,), torch.float32, rep(lambda: tile(sw3_sc, FWD_NUM_LAYERS))),
        ("shared_w2", (FWD_NUM_LAYERS * D, MOE_INTER), torch.int8, rep(lambda: tile(sw2_i8, FWD_NUM_LAYERS))),
        ("shared_w2_scale", (FWD_NUM_LAYERS * D,), torch.float32, rep(lambda: tile(sw2_sc, FWD_NUM_LAYERS))),
        # RoPE tables: [ratio-0 profile, compress profile] stacked on dim 0.
        (
            "freqs_cos",
            (2, MAX_SEQ_LEN, ROPE_HEAD_DIM),
            torch.bfloat16,
            rep(lambda: torch.stack((swa_cos, csa_cos), dim=0)),
        ),
        (
            "freqs_sin",
            (2, MAX_SEQ_LEN, ROPE_HEAD_DIM),
            torch.bfloat16,
            rep(lambda: torch.stack((swa_sin, csa_sin), dim=0)),
        ),
        # Forward metadata (identical on both ranks).
        (
            "block_table",
            (B, ORI_TABLE_MAX_BLOCKS),
            torch.int32,
            rep(lambda: _block_table(B, ORI_TABLE_MAX_BLOCKS, ORI_BLOCK_NUM)),
        ),
        ("position_ids", (T,), torch.int32, rep(position_ids.clone)),
        ("kv_seq_lens", (B,), torch.int32, rep(kv_seq_lens.clone)),
        (
            "hca_compress_state_block_table",
            (B, HCA_STATE_MAX_BLOCKS),
            torch.int32,
            rep(lambda: _block_table(B, HCA_STATE_MAX_BLOCKS, HCA_STATE_BLOCK_NUM // B)),
        ),
        (
            "csa_compress_state_block_table",
            (B, CSA_STATE_MAX_BLOCKS),
            torch.int32,
            rep(lambda: _block_table(B, CSA_STATE_MAX_BLOCKS, CSA_STATE_BLOCK_NUM // B)),
        ),
        (
            "csa_inner_compress_state_block_table",
            (B, CSA_INNER_STATE_MAX_BLOCKS),
            torch.int32,
            rep(lambda: _block_table(B, CSA_INNER_STATE_MAX_BLOCKS, CSA_INNER_STATE_BLOCK_NUM // B)),
        ),
        (
            "cmp_block_table",
            (B, CMP_BLOCK_NUM),
            torch.int32,
            rep(lambda: _block_table(B, CMP_BLOCK_NUM, CMP_BLOCK_NUM)),
        ),
        (
            "idx_block_table",
            (B, IDX_BLOCK_NUM),
            torch.int32,
            rep(lambda: _block_table(B, IDX_BLOCK_NUM, IDX_BLOCK_NUM)),
        ),
        ("block_counts", (B, N_CACHE_GROUPS), torch.int32, rep(block_counts.clone)),
        ("input_ids", (T,), torch.int64, rep(lambda: torch.arange(T, dtype=torch.int64))),
        # Hyper-connection head + final norm.
        ("hc_head_fn", (HC_MULT, HC_DIM), torch.float32, per_rank(lambda: rn(HC_MULT, HC_DIM) * 0.0519)),
        ("hc_head_scale", (1,), torch.float32, rep(lambda: torch.full((1,), HC_HEAD_SCALE, dtype=torch.float32))),
        ("hc_head_base", (HC_MULT,), torch.float32, rep(lambda: torch.tensor(HC_HEAD_BASE, dtype=torch.float32))),
        ("final_norm_w", (D,), torch.bfloat16, per_rank(lambda: (rn(D) * 0.1 + 1.0).to(torch.bfloat16))),
        # Outputs + LM head tail.
        ("pre_hc_hidden_out", (T, HC_MULT, D), torch.float32, zeros((T, HC_MULT, D), torch.float32)),
        # TP2 vocab shards: rank r keeps shard r % TP_SIZE, i.e. independent halves.
        (
            "lm_head_weight",
            (VOCAB_PER_TP, D),
            torch.bfloat16,
            per_rank(lambda: (rn(VOCAB_PER_TP, D) / D**0.5).to(torch.bfloat16)),
        ),
        ("hidden_out", (T, D), torch.bfloat16, zeros((T, D), torch.bfloat16)),
        ("logits", (MAX_LOGIT_ROWS, VOCAB), torch.float32, zeros((MAX_LOGIT_ROWS, VOCAB), torch.float32)),
        (
            "sampled_ids",
            (MAX_LOGIT_ROWS, SAMPLED_IDS_PAD),
            torch.int32,
            zeros((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), torch.int32),
        ),
        # Passed whole, not split: the chip program reads every rank's count, and
        # the generated host_orch forwards the full [N_RANKS] vector to each rank.
        ("num_tokens_per_owner", (N_RANKS,), torch.int32, None),
        ("logit_row_indices", (MAX_LOGIT_ROWS,), torch.int32, rep(logit_row_indices.clone)),
    ]

    if [name for name, _, _, _ in entries] != PARAM_NAMES:
        raise AssertionError("fixture entry order diverged from PARAM_NAMES")

    specs = []
    for name, tail, dtype, make in entries:
        if name == "num_tokens_per_owner":
            specs.append(TensorArg(name, torch.full((N_RANKS,), T, dtype=torch.int32)))
            continue
        pair = make()
        for rank_id, tensor in enumerate(pair):
            if tuple(tensor.shape) != tuple(tail) or tensor.dtype != dtype:
                raise AssertionError(
                    f"{name}_r{rank_id}: got {tuple(tensor.shape)}/{tensor.dtype}, expected {tuple(tail)}/{dtype}"
                )
            specs.append(TensorArg(f"{name}_r{rank_id}", tensor))
    return TaskArgsBuilder(*specs)
