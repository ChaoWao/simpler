// Copyright (c) PyPTO Contributors. CANN Open Software License Agreement v2.0.
// RoPE + QK-norm prologue: the pypto/ptoas-GENERATED rope_qkv kernel, copied
// verbatim from the decode_fwd rope_qkv scope codegen (golden-correct) and
// wrapped for reuse inside the fused paged_attention_rope_cce extern. Do not
// hand-edit the generated body; regenerate from decode_fwd rope_qkv if the
// math changes. VEC-only (pto Vec tiles); the caller guards the invocation.
#ifndef PYPTO_QWEN_ROPE_QKV_GENERATED_HPP
#define PYPTO_QWEN_ROPE_QKV_GENERATED_HPP

#include <cstdint>

#ifdef __DAV_C220_VEC__
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

namespace qwen_rope_gen {
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static __aicore__ inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static __aicore__ inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

static __aicore__ void rope_qkv(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ int32_t* v4, __gm__ float* v5, __gm__ int32_t* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ float* v12, __gm__ float* v13, int64_t v14, int64_t v15, int32_t v16, int32_t v17) {
  SaturationMode v18 = SaturationMode::OFF;
  RoundMode v19 = RoundMode::CAST_ROUND;
  const int64_t v20 = 2048;
  const float v21 = 9.99999997E-7f;
  const float v22 = 0.0078125f;
  const int64_t v23 = 64;
  const int64_t v24 = 40;
  const int64_t v25 = 5;
  const int64_t v26 = 8;
  const int64_t v27 = 32;
  const int64_t v28 = 2;
  const int64_t v29 = 4;
  const int64_t v30 = 896;
  const float v31 = 0.0f;
  const int64_t v32 = 1408;
  const int64_t v33 = 5120;
  const int64_t v34 = 1024;
  const int64_t v35 = 16;
  const int64_t v36 = 640;
  const int64_t v37 = 1;
  const int64_t v38 = 128;
  const int64_t v39 = 63232;
  const int64_t v40 = 62720;
  const int64_t v41 = 8192;
  const int64_t v42 = 62976;
  const int64_t v43 = 16384;
  const int64_t v44 = 62208;
  const int64_t v45 = 0;
  const int64_t v46 = 61952;
  const int64_t v47 = 61696;
  const int64_t v48 = 61440;
  const int64_t v49 = 61184;
  const int64_t v50 = 32768;
  const int64_t v51 = 32256;
  const int64_t v52 = 48896;
  const int64_t v53 = 32512;
  const int64_t v54 = 57088;
  const int64_t v55 = 31744;
  const int64_t v56 = 40704;
  const int64_t v57 = 31488;
  const int64_t v58 = 31232;
  const int64_t v59 = 30976;
  const int64_t v60 = 30720;
  const int64_t v61 = 27136;
  const int64_t v62 = 21504;
  const int64_t v63 = 20992;
  const int64_t v64 = 20480;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %k_norm_w_inline55__tile
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
  // pto: %k_norm_w_inline55__tile
  uint64_t v66 = (uint64_t) v64;
  TASSIGN(v65, v66);
  // pto: %k_norm_w_inline55__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 128> v67 = pto::Shape<1, 1, 1, 1, 128>();
  // pto: %k_norm_w_inline55__ssa_v0_pview
  pto::Stride<128, 128, 128, 128, 1> v68 = pto::Stride<128, 128, 128, 128, 1>();
  // pto: %k_norm_w_inline55__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v10 + ((v45 + v45 * v38) + v45 * v37), v67, v68);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v65, v69);
  // pto: %q_norm_w_inline246__tile
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
  // pto: %q_norm_w_inline246__tile
  uint64_t v71 = (uint64_t) v63;
  TASSIGN(v70, v71);
  // pto: %q_norm_w_inline246__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 128> v72 = pto::Shape<1, 1, 1, 1, 128>();
  // pto: %q_norm_w_inline246__ssa_v0_pview
  pto::Stride<128, 128, 128, 128, 1> v73 = pto::Stride<128, 128, 128, 128, 1>();
  // pto: %q_norm_w_inline246__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v13 + ((v45 + v45 * v38) + v45 * v37), v72, v73);
  TLOAD(v70, v74);
  // pto: %rope_core_inline264__tile
  int64_t v75 = (int64_t) v16;
  // pto: %q_red_pad_inline73__tile
  Tile<TileType::Vec, float, 1, 1408, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 1408, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v32);
  // pto: %q_red_pad_inline73__tile
  uint64_t v77 = (uint64_t) v62;
  TASSIGN(v76, v77);
  TEXPANDS(v76, v31);
  // pto: %k_red_pad_inline268__tile
  Tile<TileType::Vec, float, 1, 896, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 1, 896, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v30);
  // pto: %k_red_pad_inline268__tile
  uint64_t v79 = (uint64_t) v61;
  TASSIGN(v78, v79);
  TEXPANDS(v78, v31);
  for (int64_t i80 = v45; i80 < v29; i80 += v28) {
    // pto: %106
    int64_t v81 = (int64_t) ((uint64_t) i80 * (uint64_t) v27);
    // pto: %107
    int64_t v82 = (int64_t) ((uint64_t) v75 + (uint64_t) v81);
    // pto: %110, %109
    int64_t v83 = (int64_t) ((uint64_t) v75 + (uint64_t) ((int64_t) ((uint64_t) v81 + (uint64_t) v27)));
    // pto: %111
    if (v82 < v38) {
      // pto: %112
      int64_t v84 = v82 / v35;
      // pto: %114, %113
      int64_t v85 = (int64_t) ((uint64_t) v82 - (uint64_t) ((int64_t) ((uint64_t) v84 * (uint64_t) v35)));
      // pto: %ctx_len_inline54__tile
      int32_t v86 = v4[v85];
      // pto: %inv_rms_b_inline212__tile
      float v87 = v5[v85];
      // pto: %115, %116
      int64_t v88 = (int64_t) ((uint64_t) ((int64_t) v86) - (uint64_t) v37);
      // pto: %117
      int32_t v89 = v6[v85];
      // pto: %122
      int64_t v90 = (int64_t) ((uint64_t) v84 * (uint64_t) v38);
      // pto: %126, %118, %125, %127
      int64_t v91 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v14 + (uint64_t) ((int64_t) ((uint64_t) ((int64_t) v89) * (uint64_t) v26)))) + (uint64_t) v84);
      // pto: %cos_lo_inline82__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %cos_lo_inline82__tile
      uint64_t v93 = (uint64_t) v60;
      TASSIGN(v92, v93);
      // pto: %rope_cos__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 64> v94 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %rope_cos__ssa_v0_pview
      pto::Stride<128, 128, 128, 128, 1> v95 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %rope_cos__ssa_v0_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v96 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v7 + ((v45 + v88 * v38) + v45 * v37), v94, v95);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v92, v96);
      // pto: %cos_hi_inline158__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %cos_hi_inline158__tile
      uint64_t v98 = (uint64_t) v59;
      TASSIGN(v97, v98);
      // pto: %131
      pto::Shape<1, 1, 1, 1, 64> v99 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %131
      pto::Stride<128, 128, 128, 128, 1> v100 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %131
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v101 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v7 + ((v45 + v88 * v38) + v23 * v37), v99, v100);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v97, v101);
      // pto: %sin_lo_inline57__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %sin_lo_inline57__tile
      uint64_t v103 = (uint64_t) v58;
      TASSIGN(v102, v103);
      // pto: %rope_sin__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 64> v104 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %rope_sin__ssa_v0_pview
      pto::Stride<128, 128, 128, 128, 1> v105 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %rope_sin__ssa_v0_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v106 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v8 + ((v45 + v88 * v38) + v45 * v37), v104, v105);
      TLOAD(v102, v106);
      // pto: %sin_hi_inline88__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %sin_hi_inline88__tile
      uint64_t v108 = (uint64_t) v57;
      TASSIGN(v107, v108);
      // pto: %132
      pto::Shape<1, 1, 1, 1, 64> v109 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %132
      pto::Stride<128, 128, 128, 128, 1> v110 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %132
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v111 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v8 + ((v45 + v88 * v38) + v23 * v37), v109, v110);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v107, v111);
      // pto: %t__tile
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %t__tile
      uint64_t v113 = (uint64_t) v56;
      TASSIGN(v112, v113);
      // pto: %k_proj_inline145__rv_v3_pview
      pto::Shape<1, 1, 1, 1, 128> v114 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %k_proj_inline145__rv_v3_pview
      pto::Stride<1024, 1024, 1024, 1024, 1> v115 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %k_proj_inline145__rv_v3_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v116 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v9 + ((v45 + v85 * v34) + v90 * v37), v114, v115);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v112, v116);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %0
      uint64_t v118 = (uint64_t) v55;
      TASSIGN(v117, v118);
      // pto: %v_proj_inline207__rv_v3_pview
      pto::Shape<1, 1, 1, 1, 128> v119 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %v_proj_inline207__rv_v3_pview
      pto::Stride<1024, 1024, 1024, 1024, 1> v120 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %v_proj_inline207__rv_v3_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v121 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v11 + ((v45 + v85 * v34) + v90 * v37), v119, v120);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v117, v121);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %1
      Tile<TileType::Vec, float, 1, 640, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 1, 640, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v36);
      // pto: %1
      uint64_t v123 = (uint64_t) v54;
      TASSIGN(v122, v123);
      // pto: %q_proj_inline157__rv_v5_pview
      pto::Shape<1, 1, 1, 1, 640> v124 = pto::Shape<1, 1, 1, 1, 640>();
      // pto: %q_proj_inline157__rv_v5_pview
      pto::Stride<5120, 5120, 5120, 5120, 1> v125 = pto::Stride<5120, 5120, 5120, 5120, 1>();
      // pto: %q_proj_inline157__rv_v5_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 640>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v126 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 640>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v12 + ((v45 + v85 * v33) + (int64_t) ((uint64_t) v84 * (uint64_t) v36) * v37), v124, v125);
      TLOAD(v122, v126);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %2
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v34);
      // pto: %2
      uint64_t v128 = (uint64_t) v53;
      TASSIGN(v127, v128);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCONCAT(v127, v112, v78);
      // pto: %3
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %3
      uint64_t v130 = (uint64_t) v53;
      TASSIGN(v129, v130);
      // pto: %k_raw_inline122__tile
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %k_raw_inline122__tile
      uint64_t v132 = (uint64_t) v53;
      TASSIGN(v131, v132);
      pipe_barrier(PIPE_V);
      TMULS(v131, v129, v87);
      // pto: %4
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %4
      uint64_t v134 = (uint64_t) v56;
      TASSIGN(v133, v134);
      pipe_barrier(PIPE_V);
      TMUL(v133, v131, v131);
      // pto: %tmp_tile
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %tmp_tile
      uint64_t v136 = (uint64_t) v52;
      TASSIGN(v135, v136);
      // pto: %k_ss_inline196__tile
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v37);
      // pto: %k_ss_inline196__tile
      uint64_t v138 = (uint64_t) v51;
      TASSIGN(v137, v138);
      pipe_barrier(PIPE_V);
      TROWSUM(v137, v133, v135);
      // pto: %t__rm_a0_tmp_v0
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__rm_a0_tmp_v0
      uint64_t v140 = (uint64_t) v51;
      TASSIGN(v139, v140);
      // pto: %t__row_major_tmp_v1
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__row_major_tmp_v1
      uint64_t v142 = (uint64_t) v56;
      TASSIGN(v141, v142);
      pipe_barrier(PIPE_V);
      TMULS(v141, v139, v22);
      // pto: %t__rm_a0_tmp_v2
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__rm_a0_tmp_v2
      uint64_t v144 = (uint64_t) v56;
      TASSIGN(v143, v144);
      // pto: %t__row_major_tmp_v3
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__row_major_tmp_v3
      uint64_t v146 = (uint64_t) v56;
      TASSIGN(v145, v146);
      pipe_barrier(PIPE_V);
      TADDS(v145, v143, v21);
      // pto: %t__rm_a0_tmp_v4
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__rm_a0_tmp_v4
      uint64_t v148 = (uint64_t) v56;
      TASSIGN(v147, v148);
      // pto: %t__row_major_tmp_v5
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %t__row_major_tmp_v5
      uint64_t v150 = (uint64_t) v56;
      TASSIGN(v149, v150);
      pipe_barrier(PIPE_V);
      TSQRT(v149, v147);
      // pto: %k_inv_inline78__rm_a0_tmp_v6
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %k_inv_inline78__rm_a0_tmp_v6
      uint64_t v152 = (uint64_t) v56;
      TASSIGN(v151, v152);
      // pto: %k_inv_inline78__row_major_tmp_v7
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %k_inv_inline78__row_major_tmp_v7
      uint64_t v154 = (uint64_t) v52;
      TASSIGN(v153, v154);
      pipe_barrier(PIPE_V);
      TRECIP(v153, v151);
      // pto: %k_inv_inline78__tile
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v37);
      // pto: %k_inv_inline78__tile
      uint64_t v156 = (uint64_t) v52;
      TASSIGN(v155, v156);
      // pto: %8
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %8
      uint64_t v158 = (uint64_t) v53;
      TASSIGN(v157, v158);
      TCOLEXPANDMUL(v157, v131, v65);
      // pto: %k_normed_inline49__tile
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %k_normed_inline49__tile
      uint64_t v160 = (uint64_t) v53;
      TASSIGN(v159, v160);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v159, v157, v155);
      // pto: %slice_view
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161;
      // pto: %slice_view
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = v161;
      // pto: %slice_view
      uint64_t v163 = (uint64_t) v53;
      TASSIGN(v162, v163);
      // pto: %k_lo_inline243__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %k_lo_inline243__tile
      uint64_t v165 = (uint64_t) v53;
      TASSIGN(v164, v165);
      // pto: %k_hi_inline58__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %k_hi_inline58__tile
      uint64_t v167 = (uint64_t) v50;
      TASSIGN(v166, v167);
      // pto: %9
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %9
      uint64_t v169 = (uint64_t) v56;
      TASSIGN(v168, v169);
      pipe_barrier(PIPE_V);
      TEXTRACT(v164, v162, v45, v45);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v168, v164, v92);
      // pto: %10
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %10
      uint64_t v171 = (uint64_t) v52;
      TASSIGN(v170, v171);
      TEXTRACT(v166, v162, v45, v23);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v170, v166, v102);
      // pto: %rot_lo_inline45__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %rot_lo_inline45__tile
      uint64_t v173 = (uint64_t) v56;
      TASSIGN(v172, v173);
      pipe_barrier(PIPE_V);
      TSUB(v172, v168, v170);
      // pto: %11
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %11
      uint64_t v175 = (uint64_t) v52;
      TASSIGN(v174, v175);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v174, v166, v97);
      // pto: %12
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %12
      uint64_t v177 = (uint64_t) v53;
      TASSIGN(v176, v177);
      TCOLEXPANDMUL(v176, v164, v107);
      // pto: %rot_hi_inline221__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %rot_hi_inline221__tile
      uint64_t v179 = (uint64_t) v52;
      TASSIGN(v178, v179);
      pipe_barrier(PIPE_V);
      TADD(v178, v174, v176);
      // pto: %13
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %13
      uint64_t v181 = (uint64_t) v53;
      TASSIGN(v180, v181);
      pipe_barrier(PIPE_V);
      TCONCAT(v180, v172, v178);
      // pto: %14
      Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %14
      uint64_t v183 = (uint64_t) v51;
      TASSIGN(v182, v183);
      pipe_barrier(PIPE_V);
      TCVT(v182, v180, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %15
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %15
      uint64_t v185 = (uint64_t) v53;
      TASSIGN(v184, v185);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TMULS(v184, v117, v87);
      // pto: %v_row_bf16_inline202__tile
      Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %v_row_bf16_inline202__tile
      uint64_t v187 = (uint64_t) v55;
      TASSIGN(v186, v187);
      pipe_barrier(PIPE_V);
      TCVT(v186, v184, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %16
      Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v20);
      // pto: %16
      uint64_t v189 = (uint64_t) v53;
      TASSIGN(v188, v189);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TCONCAT(v188, v122, v76);
      // pto: %17
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %17
      uint64_t v191 = (uint64_t) v53;
      TASSIGN(v190, v191);
      // pto: %q_raw_inline151__tile
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %q_raw_inline151__tile
      uint64_t v193 = (uint64_t) v53;
      TASSIGN(v192, v193);
      pipe_barrier(PIPE_V);
      TMULS(v192, v190, v87);
      // pto: %18
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %18
      uint64_t v195 = (uint64_t) v56;
      TASSIGN(v194, v195);
      pipe_barrier(PIPE_V);
      TMUL(v194, v192, v192);
      // pto: %19
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %19
      uint64_t v197 = (uint64_t) v52;
      TASSIGN(v196, v197);
      // pto: %q_ss_inline61__tile
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v37);
      // pto: %q_ss_inline61__tile
      uint64_t v199 = (uint64_t) v54;
      TASSIGN(v198, v199);
      pipe_barrier(PIPE_V);
      TROWSUM(v198, v194, v196);
      // pto: %t__rm_a0_tmp_v8
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__rm_a0_tmp_v8
      uint64_t v201 = (uint64_t) v54;
      TASSIGN(v200, v201);
      // pto: %t__row_major_tmp_v9
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__row_major_tmp_v9
      uint64_t v203 = (uint64_t) v56;
      TASSIGN(v202, v203);
      pipe_barrier(PIPE_V);
      TMULS(v202, v200, v22);
      // pto: %t__rm_a0_tmp_v10
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__rm_a0_tmp_v10
      uint64_t v205 = (uint64_t) v56;
      TASSIGN(v204, v205);
      // pto: %t__row_major_tmp_v11
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__row_major_tmp_v11
      uint64_t v207 = (uint64_t) v56;
      TASSIGN(v206, v207);
      pipe_barrier(PIPE_V);
      TADDS(v206, v204, v21);
      // pto: %t__rm_a0_tmp_v12
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__rm_a0_tmp_v12
      uint64_t v209 = (uint64_t) v56;
      TASSIGN(v208, v209);
      // pto: %t__row_major_tmp_v13
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %t__row_major_tmp_v13
      uint64_t v211 = (uint64_t) v56;
      TASSIGN(v210, v211);
      pipe_barrier(PIPE_V);
      TSQRT(v210, v208);
      // pto: %q_inv_inline192__rm_a0_tmp_v14
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %q_inv_inline192__rm_a0_tmp_v14
      uint64_t v213 = (uint64_t) v56;
      TASSIGN(v212, v213);
      // pto: %q_inv_inline192__row_major_tmp_v15
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %q_inv_inline192__row_major_tmp_v15
      uint64_t v215 = (uint64_t) v52;
      TASSIGN(v214, v215);
      pipe_barrier(PIPE_V);
      TRECIP(v214, v212);
      // pto: %q_inv_inline192__tile
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v37);
      // pto: %q_inv_inline192__tile
      uint64_t v217 = (uint64_t) v52;
      TASSIGN(v216, v217);
      // pto: %23
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %23
      uint64_t v219 = (uint64_t) v53;
      TASSIGN(v218, v219);
      TCOLEXPANDMUL(v218, v192, v70);
      // pto: %q_heads_inline237__tile
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %q_heads_inline237__tile
      uint64_t v221 = (uint64_t) v53;
      TASSIGN(v220, v221);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v220, v218, v216);
      // pto: %q_lo_inline136__tile_textract
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %q_lo_inline136__tile_textract
      uint64_t v223 = (uint64_t) v56;
      TASSIGN(v222, v223);
      pipe_barrier(PIPE_V);
      TEXTRACT(v222, v220, v45, v45);
      // pto: %24
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %24
      uint64_t v225 = (uint64_t) v56;
      TASSIGN(v224, v225);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v224, v222, v92);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %q_hi_inline142__tile_textract
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %q_hi_inline142__tile_textract
      uint64_t v227 = (uint64_t) v52;
      TASSIGN(v226, v227);
      TEXTRACT(v226, v220, v45, v23);
      // pto: %25
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %25
      uint64_t v229 = (uint64_t) v52;
      TASSIGN(v228, v229);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v228, v226, v102);
      // pto: %q_rot_lo_inline285__tile
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %q_rot_lo_inline285__tile
      uint64_t v231 = (uint64_t) v56;
      TASSIGN(v230, v231);
      pipe_barrier(PIPE_V);
      TSUB(v230, v224, v228);
      // pto: %26
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %26
      uint64_t v233 = (uint64_t) v52;
      TASSIGN(v232, v233);
      pipe_barrier(PIPE_V);
      TEXTRACT(v232, v220, v45, v23);
      // pto: %27
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %27
      uint64_t v235 = (uint64_t) v52;
      TASSIGN(v234, v235);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v234, v232, v97);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      // pto: %28
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %28
      uint64_t v237 = (uint64_t) v54;
      TASSIGN(v236, v237);
      TEXTRACT(v236, v220, v45, v45);
      // pto: %29
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %29
      uint64_t v239 = (uint64_t) v53;
      TASSIGN(v238, v239);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v238, v236, v107);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      // pto: %q_rot_hi_inline39__tile
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %q_rot_hi_inline39__tile
      uint64_t v241 = (uint64_t) v52;
      TASSIGN(v240, v241);
      pipe_barrier(PIPE_V);
      TADD(v240, v234, v238);
      // pto: %30
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %30
      uint64_t v243 = (uint64_t) v53;
      TASSIGN(v242, v243);
      pipe_barrier(PIPE_V);
      TCONCAT(v242, v230, v240);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %137
      Tile<TileType::Vec, float, 5, 128, BLayout::RowMajor, 5, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244;
      // pto: %137
      Tile<TileType::Vec, float, 5, 128, BLayout::RowMajor, 5, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = v244;
      // pto: %137
      uint64_t v246 = (uint64_t) v53;
      TASSIGN(v245, v246);
      // pto: %32
      Tile<TileType::Vec, bfloat16_t, 5, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, bfloat16_t, 5, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v38);
      // pto: %32
      uint64_t v248 = (uint64_t) v53;
      TASSIGN(v247, v248);
      pipe_barrier(PIPE_V);
      TCVT(v247, v245, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %k_cache__iter_v3_pview
      pto::Shape<1, 1, 1, 1, 128> v249 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %k_cache__iter_v3_pview
      pto::Stride<128, 128, 128, 128, 1> v250 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %k_cache__iter_v3_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v251 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + ((v45 + v91 * v38) + v45 * v37), v249, v250);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v251, v182);
      // pto: %v_cache__iter_v3_pview
      pto::Shape<1, 1, 1, 1, 128> v252 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %v_cache__iter_v3_pview
      pto::Stride<128, 128, 128, 128, 1> v253 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %v_cache__iter_v3_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v254 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + ((v45 + v91 * v38) + v45 * v37), v252, v253);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v254, v186);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %q_tnd_flat_inline134__iter_v1_pview
      pto::Shape<1, 1, 1, 5, 128> v255 = pto::Shape<1, 1, 1, 5, 128>();
      // pto: %q_tnd_flat_inline134__iter_v1_pview
      pto::Stride<640, 640, 640, 128, 1> v256 = pto::Stride<640, 640, 640, 128, 1>();
      // pto: %129, %130, %128, %q_tnd_flat_inline134__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 5, 128>, pto::Stride<640, 640, 640, 128, 1>, pto::Layout::ND> v257 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 5, 128>, pto::Stride<640, 640, 640, 128, 1>, pto::Layout::ND>(v2 + ((v45 + (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v85 * (uint64_t) v24)) + (uint64_t) ((int64_t) ((uint64_t) v84 * (uint64_t) v25))) * v38) + v45 * v37), v255, v256);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v257, v247);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    // pto: %138
    if (v83 < v38) {
      // pto: %139
      int64_t v258 = v83 / v35;
      // pto: %141, %140
      int64_t v259 = (int64_t) ((uint64_t) v83 - (uint64_t) ((int64_t) ((uint64_t) v258 * (uint64_t) v35)));
      // pto: %142
      int32_t v260 = v4[v259];
      // pto: %143
      float v261 = v5[v259];
      // pto: %146, %147
      int64_t v262 = (int64_t) ((uint64_t) ((int64_t) v260) - (uint64_t) v37);
      // pto: %148
      int32_t v263 = v6[v259];
      // pto: %153
      int64_t v264 = (int64_t) ((uint64_t) v258 * (uint64_t) v38);
      // pto: %157, %149, %156, %158
      int64_t v265 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v14 + (uint64_t) ((int64_t) ((uint64_t) ((int64_t) v263) * (uint64_t) v26)))) + (uint64_t) v258);
      // pto: %33
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %33
      uint64_t v267 = (uint64_t) v49;
      TASSIGN(v266, v267);
      // pto: %162
      pto::Shape<1, 1, 1, 1, 64> v268 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %162
      pto::Stride<128, 128, 128, 128, 1> v269 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %162
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v270 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v7 + ((v45 + v262 * v38) + v45 * v37), v268, v269);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v266, v270);
      // pto: %34
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %34
      uint64_t v272 = (uint64_t) v48;
      TASSIGN(v271, v272);
      // pto: %163
      pto::Shape<1, 1, 1, 1, 64> v273 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %163
      pto::Stride<128, 128, 128, 128, 1> v274 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %163
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v275 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v7 + ((v45 + v262 * v38) + v23 * v37), v273, v274);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v271, v275);
      // pto: %35
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %35
      uint64_t v277 = (uint64_t) v47;
      TASSIGN(v276, v277);
      // pto: %164
      pto::Shape<1, 1, 1, 1, 64> v278 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %164
      pto::Stride<128, 128, 128, 128, 1> v279 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %164
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v280 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v8 + ((v45 + v262 * v38) + v45 * v37), v278, v279);
      TLOAD(v276, v280);
      // pto: %36
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %36
      uint64_t v282 = (uint64_t) v46;
      TASSIGN(v281, v282);
      // pto: %165
      pto::Shape<1, 1, 1, 1, 64> v283 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %165
      pto::Stride<128, 128, 128, 128, 1> v284 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %165
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v285 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v8 + ((v45 + v262 * v38) + v23 * v37), v283, v284);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v281, v285);
      // pto: %37
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %37
      uint64_t v287 = (uint64_t) v45;
      TASSIGN(v286, v287);
      // pto: %166
      pto::Shape<1, 1, 1, 1, 128> v288 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %166
      pto::Stride<1024, 1024, 1024, 1024, 1> v289 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %166
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v290 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v9 + ((v45 + v259 * v34) + v264 * v37), v288, v289);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v286, v290);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %38
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %38
      uint64_t v292 = (uint64_t) v44;
      TASSIGN(v291, v292);
      // pto: %168
      pto::Shape<1, 1, 1, 1, 128> v293 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %168
      pto::Stride<1024, 1024, 1024, 1024, 1> v294 = pto::Stride<1024, 1024, 1024, 1024, 1>();
      // pto: %168
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v295 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v11 + ((v45 + v259 * v34) + v264 * v37), v293, v294);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v291, v295);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %39
      Tile<TileType::Vec, float, 1, 640, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 1, 640, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v36);
      // pto: %39
      uint64_t v297 = (uint64_t) v43;
      TASSIGN(v296, v297);
      // pto: %170
      pto::Shape<1, 1, 1, 1, 640> v298 = pto::Shape<1, 1, 1, 1, 640>();
      // pto: %170
      pto::Stride<5120, 5120, 5120, 5120, 1> v299 = pto::Stride<5120, 5120, 5120, 5120, 1>();
      // pto: %170
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 640>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v300 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 640>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v12 + ((v45 + v259 * v33) + (int64_t) ((uint64_t) v258 * (uint64_t) v36) * v37), v298, v299);
      TLOAD(v296, v300);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %40
      Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v34);
      // pto: %40
      uint64_t v302 = (uint64_t) v42;
      TASSIGN(v301, v302);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TCONCAT(v301, v286, v78);
      // pto: %41
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %41
      uint64_t v304 = (uint64_t) v42;
      TASSIGN(v303, v304);
      // pto: %42
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %42
      uint64_t v306 = (uint64_t) v42;
      TASSIGN(v305, v306);
      pipe_barrier(PIPE_V);
      TMULS(v305, v303, v261);
      // pto: %43
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %43
      uint64_t v308 = (uint64_t) v45;
      TASSIGN(v307, v308);
      pipe_barrier(PIPE_V);
      TMUL(v307, v305, v305);
      // pto: %44
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %44
      uint64_t v310 = (uint64_t) v41;
      TASSIGN(v309, v310);
      // pto: %45
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v37);
      // pto: %45
      uint64_t v312 = (uint64_t) v40;
      TASSIGN(v311, v312);
      pipe_barrier(PIPE_V);
      TROWSUM(v311, v307, v309);
      // pto: %46
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %46
      uint64_t v314 = (uint64_t) v40;
      TASSIGN(v313, v314);
      // pto: %47
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %47
      uint64_t v316 = (uint64_t) v45;
      TASSIGN(v315, v316);
      pipe_barrier(PIPE_V);
      TMULS(v315, v313, v22);
      // pto: %49
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %49
      uint64_t v318 = (uint64_t) v45;
      TASSIGN(v317, v318);
      // pto: %50
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %50
      uint64_t v320 = (uint64_t) v45;
      TASSIGN(v319, v320);
      pipe_barrier(PIPE_V);
      TADDS(v319, v317, v21);
      // pto: %52
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %52
      uint64_t v322 = (uint64_t) v45;
      TASSIGN(v321, v322);
      // pto: %53
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %53
      uint64_t v324 = (uint64_t) v45;
      TASSIGN(v323, v324);
      pipe_barrier(PIPE_V);
      TSQRT(v323, v321);
      // pto: %55
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %55
      uint64_t v326 = (uint64_t) v45;
      TASSIGN(v325, v326);
      // pto: %56
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v26);
      // pto: %56
      uint64_t v328 = (uint64_t) v41;
      TASSIGN(v327, v328);
      pipe_barrier(PIPE_V);
      TRECIP(v327, v325);
      // pto: %57
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v37);
      // pto: %57
      uint64_t v330 = (uint64_t) v41;
      TASSIGN(v329, v330);
      // pto: %58
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %58
      uint64_t v332 = (uint64_t) v42;
      TASSIGN(v331, v332);
      TCOLEXPANDMUL(v331, v305, v65);
      // pto: %59
      Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v26, v38);
      // pto: %59
      uint64_t v334 = (uint64_t) v42;
      TASSIGN(v333, v334);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v333, v331, v329);
      // pto: %171
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335;
      // pto: %171
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = v335;
      // pto: %171
      uint64_t v337 = (uint64_t) v42;
      TASSIGN(v336, v337);
      // pto: %61
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %61
      uint64_t v339 = (uint64_t) v42;
      TASSIGN(v338, v339);
      // pto: %62
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %62
      uint64_t v341 = (uint64_t) v39;
      TASSIGN(v340, v341);
      // pto: %63
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %63
      uint64_t v343 = (uint64_t) v45;
      TASSIGN(v342, v343);
      pipe_barrier(PIPE_V);
      TEXTRACT(v338, v336, v45, v45);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v342, v338, v266);
      // pto: %64
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %64
      uint64_t v345 = (uint64_t) v41;
      TASSIGN(v344, v345);
      TEXTRACT(v340, v336, v45, v23);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v344, v340, v276);
      // pto: %65
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %65
      uint64_t v347 = (uint64_t) v45;
      TASSIGN(v346, v347);
      pipe_barrier(PIPE_V);
      TSUB(v346, v342, v344);
      // pto: %66
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %66
      uint64_t v349 = (uint64_t) v41;
      TASSIGN(v348, v349);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v348, v340, v271);
      // pto: %67
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %67
      uint64_t v351 = (uint64_t) v42;
      TASSIGN(v350, v351);
      TCOLEXPANDMUL(v350, v338, v281);
      // pto: %68
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v23);
      // pto: %68
      uint64_t v353 = (uint64_t) v41;
      TASSIGN(v352, v353);
      pipe_barrier(PIPE_V);
      TADD(v352, v348, v350);
      // pto: %69
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %69
      uint64_t v355 = (uint64_t) v42;
      TASSIGN(v354, v355);
      pipe_barrier(PIPE_V);
      TCONCAT(v354, v346, v352);
      // pto: %70
      Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %70
      uint64_t v357 = (uint64_t) v40;
      TASSIGN(v356, v357);
      pipe_barrier(PIPE_V);
      TCVT(v356, v354, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %71
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %71
      uint64_t v359 = (uint64_t) v42;
      TASSIGN(v358, v359);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TMULS(v358, v291, v261);
      // pto: %72
      Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v38);
      // pto: %72
      uint64_t v361 = (uint64_t) v44;
      TASSIGN(v360, v361);
      pipe_barrier(PIPE_V);
      TCVT(v360, v358, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %73
      Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v20);
      // pto: %73
      uint64_t v363 = (uint64_t) v42;
      TASSIGN(v362, v363);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TCONCAT(v362, v296, v76);
      // pto: %74
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %74
      uint64_t v365 = (uint64_t) v42;
      TASSIGN(v364, v365);
      // pto: %75
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %75
      uint64_t v367 = (uint64_t) v42;
      TASSIGN(v366, v367);
      pipe_barrier(PIPE_V);
      TMULS(v366, v364, v261);
      // pto: %76
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %76
      uint64_t v369 = (uint64_t) v45;
      TASSIGN(v368, v369);
      pipe_barrier(PIPE_V);
      TMUL(v368, v366, v366);
      // pto: %77
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %77
      uint64_t v371 = (uint64_t) v41;
      TASSIGN(v370, v371);
      // pto: %78
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v37);
      // pto: %78
      uint64_t v373 = (uint64_t) v43;
      TASSIGN(v372, v373);
      pipe_barrier(PIPE_V);
      TROWSUM(v372, v368, v370);
      // pto: %79
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %79
      uint64_t v375 = (uint64_t) v43;
      TASSIGN(v374, v375);
      // pto: %80
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %80
      uint64_t v377 = (uint64_t) v45;
      TASSIGN(v376, v377);
      pipe_barrier(PIPE_V);
      TMULS(v376, v374, v22);
      // pto: %82
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %82
      uint64_t v379 = (uint64_t) v45;
      TASSIGN(v378, v379);
      // pto: %83
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %83
      uint64_t v381 = (uint64_t) v45;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TADDS(v380, v378, v21);
      // pto: %85
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %85
      uint64_t v383 = (uint64_t) v45;
      TASSIGN(v382, v383);
      // pto: %86
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %86
      uint64_t v385 = (uint64_t) v45;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TSQRT(v384, v382);
      // pto: %88
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %88
      uint64_t v387 = (uint64_t) v45;
      TASSIGN(v386, v387);
      // pto: %89
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v35);
      // pto: %89
      uint64_t v389 = (uint64_t) v41;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TRECIP(v388, v386);
      // pto: %90
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v37);
      // pto: %90
      uint64_t v391 = (uint64_t) v41;
      TASSIGN(v390, v391);
      // pto: %91
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %91
      uint64_t v393 = (uint64_t) v42;
      TASSIGN(v392, v393);
      TCOLEXPANDMUL(v392, v366, v70);
      // pto: %92
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %92
      uint64_t v395 = (uint64_t) v42;
      TASSIGN(v394, v395);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v394, v392, v390);
      // pto: %93
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %93
      uint64_t v397 = (uint64_t) v45;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TEXTRACT(v396, v394, v45, v45);
      // pto: %94
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %94
      uint64_t v399 = (uint64_t) v45;
      TASSIGN(v398, v399);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v398, v396, v266);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %95
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %95
      uint64_t v401 = (uint64_t) v41;
      TASSIGN(v400, v401);
      TEXTRACT(v400, v394, v45, v23);
      // pto: %96
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %96
      uint64_t v403 = (uint64_t) v41;
      TASSIGN(v402, v403);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v402, v400, v276);
      // pto: %97
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %97
      uint64_t v405 = (uint64_t) v45;
      TASSIGN(v404, v405);
      pipe_barrier(PIPE_V);
      TSUB(v404, v398, v402);
      // pto: %98
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %98
      uint64_t v407 = (uint64_t) v41;
      TASSIGN(v406, v407);
      pipe_barrier(PIPE_V);
      TEXTRACT(v406, v394, v45, v23);
      // pto: %99
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %99
      uint64_t v409 = (uint64_t) v41;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v408, v406, v271);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %100
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %100
      uint64_t v411 = (uint64_t) v43;
      TASSIGN(v410, v411);
      TEXTRACT(v410, v394, v45, v45);
      // pto: %101
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %101
      uint64_t v413 = (uint64_t) v42;
      TASSIGN(v412, v413);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v412, v410, v281);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %102
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v23);
      // pto: %102
      uint64_t v415 = (uint64_t) v41;
      TASSIGN(v414, v415);
      pipe_barrier(PIPE_V);
      TADD(v414, v408, v412);
      // pto: %103
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v35, v38);
      // pto: %103
      uint64_t v417 = (uint64_t) v42;
      TASSIGN(v416, v417);
      pipe_barrier(PIPE_V);
      TCONCAT(v416, v404, v414);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %174
      Tile<TileType::Vec, float, 5, 128, BLayout::RowMajor, 5, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418;
      // pto: %174
      Tile<TileType::Vec, float, 5, 128, BLayout::RowMajor, 5, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = v418;
      // pto: %174
      uint64_t v420 = (uint64_t) v42;
      TASSIGN(v419, v420);
      // pto: %105
      Tile<TileType::Vec, bfloat16_t, 5, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, bfloat16_t, 5, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v38);
      // pto: %105
      uint64_t v422 = (uint64_t) v42;
      TASSIGN(v421, v422);
      pipe_barrier(PIPE_V);
      TCVT(v421, v419, v19, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      // pto: %k_cache__phi_v6_pview
      pto::Shape<1, 1, 1, 1, 128> v423 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %k_cache__phi_v6_pview
      pto::Stride<128, 128, 128, 128, 1> v424 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %k_cache__phi_v6_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v425 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + ((v45 + v265 * v38) + v45 * v37), v423, v424);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v425, v356);
      // pto: %v_cache__phi_v6_pview
      pto::Shape<1, 1, 1, 1, 128> v426 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %v_cache__phi_v6_pview
      pto::Stride<128, 128, 128, 128, 1> v427 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %v_cache__phi_v6_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v428 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + ((v45 + v265 * v38) + v45 * v37), v426, v427);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v428, v360);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      // pto: %q_tnd_flat_inline134__phi_v4_pview
      pto::Shape<1, 1, 1, 5, 128> v429 = pto::Shape<1, 1, 1, 5, 128>();
      // pto: %q_tnd_flat_inline134__phi_v4_pview
      pto::Stride<640, 640, 640, 128, 1> v430 = pto::Stride<640, 640, 640, 128, 1>();
      // pto: %160, %161, %159, %q_tnd_flat_inline134__phi_v4_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 5, 128>, pto::Stride<640, 640, 640, 128, 1>, pto::Layout::ND> v431 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 5, 128>, pto::Stride<640, 640, 640, 128, 1>, pto::Layout::ND>(v2 + ((v45 + (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v259 * (uint64_t) v24)) + (uint64_t) ((int64_t) ((uint64_t) v258 * (uint64_t) v25))) * v38) + v45 * v37), v429, v430);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v431, v421);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

}  // namespace qwen_rope_gen
#endif  // __DAV_C220_VEC__

#endif  // PYPTO_QWEN_ROPE_QKV_GENERATED_HPP
