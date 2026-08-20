# DeepSeek-V4 FLASH full decode network (EP2 / TP2, 2 dies)

The complete DeepSeek-V4 FLASH 43-layer decode forward as one scene test:
pypto-generated chip orchestration + 368 incore kernels per rank, dispatched to
two dies that cooperate through a communication domain (expert-parallel MoE
dispatch/combine and a TP-sharded LM head). This is the first pypto-harvested
**distributed** network in the repo — the single-chip counterpart is
[`qwen3_14b_decode/`](../qwen3_14b_decode/README.md), and the raw-`Worker`
multi-chip counterpart is
[`examples/workers/l3/ep_dispatch_combine/`](../../../workers/l3/ep_dispatch_combine/).

## Network shape

| Aspect | Value |
| ------ | ----- |
| Layers | 43 (2 SWA + 21 CSA + 20 HCA sparse attentions), one MoE per layer |
| Tail | `hc_head -> rms_norm -> lm_head_with_sampling` (TP2 over vocab 129280) |
| Parallelism | attention data-parallel per rank; MoE expert-parallel (EP2, 64 routed experts global / 32 per rank, top-6 + 1 shared); LM head TP2 |
| Tokens | batch 4 x 2 rows per rank (T=8), `start_pos` 8192 |
| KV | paged, 128-token blocks; per-layer compress/index state pools (CSA/HCA) |
| Quantization | W8A8 INT8 experts and projections with per-channel scales |
| Comm window | 12 buffers (~6.6 MiB): MoE meta/payload/route + arrival counters + LM-head hidden/logits windows |

Per rank the chip program receives 92 tensors (79 per-rank shards, 12
comm-window views, 1 shared counts vector) plus 13 scalars (rank id + 12
`CommContext` pointers), then runs the whole decode step as one dispatch.

## Validation semantics

The case is `skip_golden`: it is a completion/smoke test. Upstream pypto-lib
drives the same fixture with `golden_fn=None` — a full-network torch reference
does not exist there either; numeric coverage lives with the standalone kernel
harnesses in pypto-lib (`models/deepseek_v4_flash_mtp/*.py`), and real-weight
end-to-end accuracy lives in pypto-serving. What this case pins down in simpler
CI terms: the harvested distributed program compiles, both ranks' graphs
dispatch, the cross-die window protocol (TPUT/TNOTIFY arrivals, LM-head
all-gather) drains, and the run terminates cleanly.

It is marked `manual`: compiling 368 incore kernels plus the 7.8k-line chip
orchestration takes several minutes, so it does not run in the default sweep.

The six buffer initializations formerly expressed as orchestration-side
`set_initial_value(0)` calls now run on device. Each of the five
`sh_gate_up_act_q*` kernels clears the two padded `h_tile_i8` rows owned by its
logical block. The dedicated `hc_head_mixes_zero` seed kernel clears one
up-to-16-row chunk of the dynamic `mixes_raw` split-K destination per logical
block; an explicit task dependency orders all `hc_head_linear` `AtomicAdd`
stores after it.

The 30 orchestration-side `get_tensor_data` reads of `hc_attn_scale_*` /
`hc_ffn_scale_*` are likewise gone. They moved data, not control flow: the
values went straight into task scalars for the `split_pre_post*` /
`comb_sinkhorn*` kernels. Each of those kernels now takes the scale view as an
extra tensor input and reads scale0/scale1 (`split_pre_post*`) or scale2
(`comb_sinkhorn*`) from GM itself. The remaining reads — `recv_count_out` and
`ext_num_tokens_per_owner` — feed loop trip counts and launch block numbers,
i.e. orchestration control flow, and stay on the orchestrator.

## Status: blocked on the #1644 pto-isa pin bump

The case PASSES on every runtime from the wire-ABI cutover (#1729, the earliest
commit whose scene-test plumbing can host it) through #1771 — all of which pin
pto-isa `83d01313`, the commit these kernels were generated against. From
`7a1b9b11` ("CI: bump pinned pto-isa for PTOAS v0.55", pto-isa -> `0cefc9a5`)
onward, including current main, both ranks stall mid-network: two kernels spin
forever on comm-window arrival counters, the AICPU orchestrator times out
reading `recv_count_out` (`TENSOR_WAIT_TIMEOUT`, "producer not completed"), and
the scheduler exits `S1:running-stalled`. The stall layer moves with fixture
content (a timing-dependent miss, not a fixed logic error), and the identical
program + content passes under pypto's own runner against pto-isa `83d01313`
even with every weight streamed as a host tensor. Bisect evidence: PASS at
PR #1771 and FAIL at PR #1644 on otherwise identical simpler code. Candidate
pto-isa changes in `83d01313..0cefc9a5` touch exactly the GM-polling path
these kernels spin on
(`6ffe3221` "Qualify dcci cache line arguments", `2d9d4288` "normalize cache
line constants").

## Running

```bash
# standalone (2 dies; wrap in task-submit on a shared box)
python examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode/test_deepseek_v4_flash_decode.py \
    -p a2a3 -d <d0>,<d1> --manual only

# pytest
pytest examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode \
    --platform a2a3 --device <d0>,<d1> --manual only
```

The fixture materializes on the order of 100 GiB of host tensors (weights for
both ranks); the machine needs the RAM headroom, and generation takes a few
minutes.

## Provenance

Harvested with `RunConfig` defaults (`--start-pos 8192`, `--num-tokens 8`,
`--ep 2 --tp 2`) from:

| Repo | Commit / version |
| ---- | ---------------- |
| pypto-lib | `ef88d34` (`models/deepseek_v4_flash_mtp/decode_fwd.py`) |
| pypto | `289290e6` |
| simpler (pypto `runtime/` pin at harvest) | `3165cc89` |
| ptoas | `v0.57` |
| pto-isa used for generation | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` |

The harvested files start from pypto codegen output plus three mechanical,
reproducible transforms — the license header, the repo's `clang-format`, and
the whole-word renames `L2TaskArgs -> ChipTaskArgs`, `L0TaskArgs -> CoreTaskArgs`,
`Tensor -> ChipTensor` (pypto's codegen still emits the pre-rename identifiers
of its pinned runtime `3165cc89`; simpler renamed them on main per the role-based
naming rule, with no compat alias). The intentional post-harvest edits are the
`hc_head_linear` row-tail bound, the device-side initialization, and the
kernel-side scale loads described above; regeneration must reapply all three.
`test_deepseek_v4_flash_decode.py`'s
`_KERNELS` / `_ORCH_SIG` tables are transcribed from the harvest's
`kernel_config.py`, and `_ARG_STEPS` / the comm-domain layout from its
generated `host_orch.py` (per-rank stacked slices become the `_r0`/`_r1` host
tensors because the scene-test rehost passes whole buffers, and pypto's
device-resident weight upload becomes plain host-tensor args — both are
mechanical translations, not behavior changes; residency only matters for
repeated dispatch).

### To regenerate

```python
# In a pypto-lib checkout with pypto installed (see multi-repo-setup):
#   PYTHONPATH=$PWD python models/deepseek_v4_flash_mtp/decode_fwd.py -p a2a3 --compile-only
# then, from build_output/_jit_l3_decode_fwd_<ts>/:
#   - next_levels/decode_fwd/kernels/{aic,aiv}/*.cpp  -> kernels/{aic,aiv}/   (+ license header)
#   - next_levels/decode_fwd/orchestration/decode_fwd.cpp -> kernels/orchestration/
#   - apply the whole-word renames (sed -E 's/\bL2TaskArgs\b/ChipTaskArgs/g;
#     s/\bL0TaskArgs\b/CoreTaskArgs/g; s/\bTensor\b/ChipTensor/g') and clang-format,
#     while pypto's pinned runtime predates the role-based renames
#   - next_levels/decode_fwd/kernel_config.py -> _KERNELS / _ORCH_SIG tables
#   - orchestration/host_orch.py -> _ARG_STEPS / _WINDOW_BUFFERS tables + orch fn
#   - build_tensor_specs() in decode_fwd.py -> simpler_setup/goldens/deepseek_v4_flash_decode.py
```

## Cost (measured on a2a3, 2 dies)

Upstream pypto-lib runs the identical program in ~145 s end-to-end (compile
25 s with a warm ptoas cache, fixture 36 s, runtime 85 s). In simpler the
kernel compile is cold per cache key; expect several minutes on first run and
a cached compile afterwards.
