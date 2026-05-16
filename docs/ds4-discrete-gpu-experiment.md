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
- On Blackwell CUDA (`sm_12x`), one-token f16 matvecs now default to the generic 256-thread path. Set `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` to restore the old ordered 32-thread path for A/B.

Canonical reference CSVs live in `speed-bench/`:

- `rtx_pro_6000_default_reserve.csv`
- `rtx_pro_6000_reserve_512mb.csv`
- `rtx_pro_6000_reserve_128mb.csv`
- `rtx_pro_6000.csv` - current recommended policy baseline
- `rtx_pro_6000_official_65536_20260514.csv` - upstream-shaped `ctx=2048..65536` sweep on current `main`
- `rtx_pro_6000_f16_default_official_65536_20260515.csv` - upstream-shaped sweep with the Blackwell generic f16 default patch

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

Official upstream-shaped sweep on current `main` after the merged MoE half-warp decode LUT:

- Run: `~/ds4/codex-runs/20260514-225106-official-speedbench-main`
- CSV: `speed-bench/rtx_pro_6000_official_65536_20260514.csv`
- `ctx=2048`: 509.62 prefill t/s, 44.78 gen t/s
- `ctx=32768`: 466.99 prefill t/s, 39.10 gen t/s
- `ctx=65536`: 440.25 prefill t/s, 36.38 gen t/s

Upstream-shaped sweep with the Blackwell generic f16 default patch:

- Run: `~/ds4/codex-runs/20260515-003703-f16-default-official-sweep`
- CSV: `speed-bench/rtx_pro_6000_f16_default_official_65536_20260515.csv`
- `ctx=2048`: 497.56 prefill t/s, 46.74 gen t/s
- `ctx=32768`: 459.76 prefill t/s, 40.74 gen t/s
- `ctx=65536`: 437.33 prefill t/s, 37.88 gen t/s

## Performance target

Primary target: make RTX Pro 6000 generation materially faster than the official DGX Spark result, not merely faster in absolute terms.

Upstream README reference row:

- DGX Spark GB10, q2, 7047-token prompt, `--ctx 32768`, greedy `-n 256`: 343.81 prefill t/s, 13.75 generation t/s.

Hardware-ratio framing:

- RTX Pro 6000 memory bandwidth: 1792 GB/s vs DGX Spark 273 GB/s, about 6.56x.
- RTX Pro 6000 CUDA cores: 24064 vs DGX Spark 6144, about 3.92x.
- RTX Pro 6000 FP4 sparse peak: about 4030 TOPS vs DGX Spark 1000 TOPS, about 4.03x.

Sources:

- RTX Pro 6000 specs: NVIDIA RTX Blackwell architecture whitepaper, table 1 / table 4: <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf>
- DGX Spark specs: NVIDIA DGX Spark User Guide, hardware overview / performance specifications: <https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf>

Generation target bands:

- Below 50 t/s: not enough; likely leaving major RTX Pro 6000 performance unused.
- 55-65 t/s: first acceptable target, roughly matching broad compute-ratio scaling from DGX Spark.
- 70-80 t/s: strong result, likely indicating a real software bottleneck was removed.
- 85-90 t/s: stretch target, close to pure memory-bandwidth scaling from the DGX Spark row.

Prefill target bands:

- 700+ t/s: first meaningful improvement over the current roughly 500 t/s class baseline.
- 1000-1300 t/s: good RTX-class result.
- 1500+ t/s: stretch target.

Long-context behavior remains important, but it is a second-stage acceptance tier. First prove the RTX Pro 6000 path is fast at the upstream benchmark shape; then verify that the same direction still behaves at much longer contexts.

## Merged RTX Pro 6000 changes

- `a865750` / related speed-bench commits: recommended q8 f16 cache policy and CSV baselines.
- `1d35fbb`: pair decode q and kv q8 matvecs. Small but repeatable decode gain.
- `5347041`: fuse decode Q head RMS norm and RoPE via existing CUDA helper. Small repeatable decode gain, with `DS4_METAL_DISABLE_Q_HEAD_ROPE_FUSION=1` as the reference fallback.
- `da9a129`: half-warp MoE decode LUT gate/up kernel. Repeatable RTX Pro 6000 decode gain with `DS4_CUDA_MOE_NO_DECODE_LUT_H16=1` as the reference fallback:
  - `ctx=4096`: about 43.8 gen t/s vs 42.6 fallback.
  - `ctx=32768`: about 38.6 gen t/s vs 37.7 fallback.
  - `ctx=131072`: 31.51 gen t/s vs 30.81 fallback in a one-run smoke.
- Pending f16 default patch: use generic one-token f16 matvecs by default on Blackwell CUDA, with `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` as the old-path fallback:
  - Discovery run: `~/ds4/codex-runs/20260515-000626-ctx32768-router-f16-ab`.
  - Repeat run: `~/ds4/codex-runs/20260515-001905-f16-generic-repeat`.
  - Validation run: `~/ds4/codex-runs/20260515-002919-f16-default-validation`.
  - `ctx=4096`: 45.90 gen t/s default vs 43.82 old ordered fallback.
  - `ctx=32768`: 40.33 gen t/s default vs 38.59 old ordered fallback.

Validation status for `5347041`:

- CUDA build passed on `sm_120`.
- Smoke generation passed.
- `make test CUDA_ARCH=sm_120` has the same known `logprob-vectors / long_memory_archive` 7-failure pattern seen before this patch; no new failure shape observed.
- Nsight confirms `head_rms_norm_rope_tail_kernel` is active.

## Rejected RTX Pro 6000 probes

- `codex/cuda-end-sync-experiment`: made CUDA command-boundary synchronization optional. No meaningful throughput gain; profiler `cudaDeviceSynchronize` time was mostly waiting for GPU work.
- `codex/hc-rms-mix-fusion`: fused HC RMS norm plus 24-wide HC mix projection into one CUDA block. Correctness smoke passed, but decode collapsed to roughly 10 t/s because the kernel lost row-level parallelism.
- `codex/hc-rms-mix-row-fusion`: row-parallel HC RMS plus mix projection. Preserved performance and measured about +1.3% at `ctx=4096`, but the gain is too small for the extra kernel surface and redundant RMS work. Treat the HC RMS/mix route as closed unless a larger related redesign appears.
- Built-in attention/indexer fallback toggles at `ctx=32768`, 512 generated tokens, run `~/ds4/codex-runs/20260514-234525-ctx32768-attn-indexer-ab`:
  - Baseline: 510.44 prefill t/s, 38.62 gen t/s.
  - `DS4_CUDA_NO_INDEXED_HEADS8=1`: no generation gain, prefill much worse.
  - `DS4_CUDA_INDEXED_TWOPASS=1`: no generation gain, prefill worse.
  - `DS4_CUDA_NO_INDEXED_TOPK_SORT=1`: no generation gain.
  - `DS4_CUDA_NO_INDEXER_DIRECT_ONE=1`: generation dropped to 35.59 t/s.
  - `DS4_CUDA_NO_INDEXER_WMMA=1`: no generation gain, prefill collapsed to 143.91 t/s.
  - `DS4_CUDA_NO_TOPK8192=1`: no generation gain.
  - `DS4_CUDA_NO_TOPK_CHUNKED=1`: clearly slow fallback; do not pursue unless there is a specific top-k replacement patch to test.
- `codex/indexed-attn-heads8-decode-ab`: allowed single-token decode to opt into the existing grouped `attention_indexed_mixed_heads8_online_kernel` via `DS4_CUDA_INDEXED_HEADS8_DECODE=1`. Smoke passed, but performance did not justify the path; run `~/ds4/codex-runs/20260516-033124-indexed-heads8-decode-ab`:
  - `ctx=64`, 3 runs: baseline 56.77 gen t/s, candidate 56.76 gen t/s.
  - `ctx=4096`, 3 runs: baseline 45.75 gen t/s, candidate 43.77 gen t/s.
  - Interpretation: grouping heads with the existing online kernel is neutral at tiny context and slower once indexed attention matters. Do not pursue this as a simple guard change; a useful attention/indexer patch needs a decode-specific kernel design.

Operational note: future fallback-disabling probes should use shorter generation counts first. Slow fallback checks are useful negative results, but they should not occupy the GPU for long unless the early result shows a plausible gain.

## Current diagnosis

The initial WSL2 path was not representative. Bare-metal Ubuntu removed the catastrophic WSL2 behavior and brought generation to roughly 38-43 t/s depending on context.

Remaining decode performance is not dominated by one obvious allocation fallback. The q8 f16 cache still fills only a few GiB because the 80.76 GiB model plus context/scratch leaves limited spare VRAM, but the measured fallback shape is stable and partially mitigated by the 128 MB reserve plus no-attention-output policy.

Upstream status as of 2026-05-14: upstream has new server/API commits plus `04b6fda` (`cuda: use managed KV cache for huge contexts`). That CUDA change is scoped to very large KV caches and should not affect the current 32k/65k benchmark phase. Treat it as a separate long-context stability item, not as part of the RTX Pro 6000 throughput work, and verify carefully before merging because managed KV can trade performance for allocation survivability.

At the official benchmark shape, RTX Pro 6000 is already much faster than DGX Spark in absolute terms, but not in proportion to the hardware delta. At `ctx=32768`, the Blackwell f16 default patch reaches 40.74 t/s versus the upstream DGX Spark row at 13.75 t/s: about 3.0x, still below the broad 4x compute-ratio target band.

`ctx=32768` decode stage profile:

- Run: `~/ds4/codex-runs/20260514-225658-ctx32768-decode-stage-profile`
- Profiled result with overhead: 509.13 prefill t/s, 34.84 gen t/s.
- Per-token profile: about 28.67 ms/token.
- Top stage buckets: attention about 22.6%, compressor/indexer about 18.0%, routed MoE about 12.7%, ffn/attn HC pre about 18.7% combined, attention output about 9.2%, q path about 6.9%.

Decode-only delayed Nsight at `ctx=32768`:

- Run: `~/ds4/codex-runs/20260514-234008-ctx32768-decode-only-nsys`
- Top kernel buckets include indexed attention, f16 ordered matmuls, dense attention, MoE decode LUT, grouped q8 matvecs, HC q8 matvecs, RMS norms, and indexer top-k chunk/merge kernels.
- CUDA API launch count remains very high, but this is not only host overhead: the GPU kernels themselves are fragmented and many are very small.

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

1. **Router / f16 matvec A/B**
   - `matmul_f16_ordered_chunks_kernel` is a major decode-only Nsight bucket.
   - First run environment-level A/B for f16 router and pair-matmul paths.
   - Only write a specialized router kernel if the A/B shows this path has real upside.

2. **Attention/indexer replacement, not fallback toggles**
   - Built-in fallback toggles did not improve generation at `ctx=32768`.
   - Reusing the existing grouped heads8 online indexed-attention kernel for single-token decode was slower at `ctx=4096`.
   - Any useful work here needs a real top-k / indexed-attention kernel replacement or fusion, not disabling the current fast paths.

3. **MoE/shared decode kernels**
   - The decode-focused profile shows MoE and shared/expert matvec kernels as major GPU-time buckets.
   - Safer next step is to inspect existing fused MoE/shared helpers before adding new kernels.

4. **CUDA Graph replay feasibility**
   - Potentially a meaningful host-launch reduction, but unlikely to close the whole gap by itself.
   - Hard because token position, cache counters, compression/indexer topology, and occasional ratio-dependent branches change across tokens.
   - Needs a prototype with graph node parameter updates or device-side parameter buffers, not a broad rewrite.

5. **Avoid risky KV RoPE fusion for now**
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
