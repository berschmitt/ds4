# DS4 RTX Pro 6000 discrete-GPU experiment

Short, current plan written against [PRINCIPLES.md](../PRINCIPLES.md). This is a home-lab science workflow: preserve correctness, measure before optimizing, and keep changes easy to compare with upstream.

## Current host

- Machine: `bst-originpc-ai`
- OS: Ubuntu 24.04 LTS bare metal
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, `sm_120`, 96 GB VRAM
- CUDA driver/toolchain observed: driver 595.71.05, CUDA runtime 13.2, nvcc 13.0
- Remote workflow: [remote-rtx-pro-6000-access-contract.md](remote-rtx-pro-6000-access-contract.md)

## Required runtime policy

Recommended environment for this 96 GB discrete card:

```bash
export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128
export DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1
```

Why:

- The default q8->f16 cache reserve is conservative for unified-memory DGX Spark style systems and leaves useful RTX Pro 6000 VRAM unused.
- Lowering the reserve from the default to 128 MB improves prefill by allowing more q8 f16 cache entries.
- Skipping `attn_output_*` q8 f16 caching lets the limited spare VRAM cache smaller, more reused q8 tensors instead.

Canonical reference CSVs live in `speed-bench/`:

- `rtx_pro_6000_default_reserve.csv`
- `rtx_pro_6000_reserve_512mb.csv`
- `rtx_pro_6000_reserve_128mb.csv`
- `rtx_pro_6000.csv` - current recommended policy baseline

## Current baseline

Command:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 \
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1 \
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128
```

Current canonical baseline, before the small decode Q norm+RoPE fusion:

- `ctx=2048`: 541.33 prefill t/s, 43.25 gen t/s
- `ctx=4096`: 528.38 prefill t/s, 42.47 gen t/s
- `ctx=32768`: 494.83 prefill t/s, 37.90 gen t/s

Latest post-fusion sweep was kept as a run artifact instead of overwriting the canonical CSV because prefill was noisier/lower despite the patch being decode-only:

- Run: `~/ds4/codex-runs/20260514-201147-decode-fusion-full-sweep`
- `ctx=4096`: 522.28 prefill t/s, 42.71 gen t/s
- `ctx=32768`: 490.80 prefill t/s, 38.10 gen t/s

## Merged RTX Pro 6000 changes

- `a865750` / related speed-bench commits: recommended q8 f16 cache policy and CSV baselines.
- `1d35fbb`: pair decode q and kv q8 matvecs. Small but repeatable decode gain.
- `5347041`: fuse decode Q head RMS norm and RoPE via existing CUDA helper. Small repeatable decode gain, with `DS4_METAL_DISABLE_Q_HEAD_ROPE_FUSION=1` as the reference fallback.

Validation status for `5347041`:

- CUDA build passed on `sm_120`.
- Smoke generation passed.
- `make test CUDA_ARCH=sm_120` has the same known `logprob-vectors / long_memory_archive` 7-failure pattern seen before this patch; no new failure shape observed.
- Nsight confirms `head_rms_norm_rope_tail_kernel` is active.

## Current diagnosis

The initial WSL2 path was not representative. Bare-metal Ubuntu removed the catastrophic WSL2 behavior and brought generation to roughly 38-43 t/s depending on context.

Remaining decode performance is not dominated by one obvious allocation fallback. The q8 f16 cache still fills only a few GiB because the 80.76 GiB model plus context/scratch leaves limited spare VRAM, but the measured fallback shape is stable and partially mitigated by the 128 MB reserve plus no-attention-output policy.

Nsight on a decode-focused fixed-token bench:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 \
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1 \
nsys profile -o /tmp/ds4-bench-decode64 --trace=cuda,nvtx,osrt --force-overwrite=true \
  ./ds4-bench -m ds4flash.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 64 --ctx-max 64 --step-incr 2048 --gen-tokens 512
```

Observed run:

- Run: `~/ds4/codex-runs/20260514-201904-decode64-bench-nsys`
- `ctx=64`: 51.61 gen t/s
- CUDA launches: about 799k for the profiled process
- Largest GPU kernel buckets: MoE decode, f16 ordered matmuls, attention decode, q8 matvecs, HC expansion, RMS norms

This points to general decode kernel fragmentation plus real q8/MoE/matvec kernel time. More one-launch micro-fusions will help only marginally unless they sit on a high-count, high-time path.

## Next engineering targets

1. **CUDA Graph replay feasibility**
   - Potentially the largest host-launch reduction.
   - Hard because token position, cache counters, compression/indexer topology, and occasional ratio-dependent branches change across tokens.
   - Needs a prototype with graph node parameter updates or device-side parameter buffers, not a broad rewrite.

2. **MoE/shared decode kernels**
   - The decode-focused profile shows MoE and shared/expert matvec kernels as major GPU-time buckets.
   - Safer next step is to inspect existing fused MoE/shared helpers before adding new kernels.

3. **Attention/indexer at longer context**
   - `ctx=64` highlights pure decode overhead; `ctx=4096+` adds indexed attention/indexer cost.
   - Any long-context change must be checked at both `ctx=4096` and `ctx=32768`.

4. **Avoid risky KV RoPE fusion for now**
   - FP8 KV quantization plus raw-cache store is already fused.
   - KV RoPE is intentionally standalone for exactness; fusing it with FP8/store is possible but higher risk than the current evidence justifies.

## Correctness gate

Token-byte parity vs the official DeepSeek V4 Flash API remains the hard gate. Existing known CUDA failures must be tracked separately from new regressions:

```bash
make test CUDA_ARCH=sm_120
```

For now, a candidate patch is acceptable only if:

- It builds for `CUDA_ARCH=sm_120`.
- Smoke generation works.
- `make test CUDA_ARCH=sm_120` does not introduce a new failure shape beyond the known `long_memory_archive` logprob-vector failures.
- A/B decode benchmarks show a repeatable gain, not a single lucky run.
