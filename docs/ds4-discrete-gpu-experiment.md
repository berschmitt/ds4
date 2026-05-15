# DS4 RTX Pro 6000 discrete-GPU experiment

Short, current plan written against [PRINCIPLES.md](../PRINCIPLES.md). This is a home-lab science workflow: preserve correctness, measure before optimizing, and keep changes easy to compare with upstream.

Fork hygiene and patch-carrying policy live in [fork-maintenance-policy.md](fork-maintenance-policy.md). That document is the source of truth for which local changes are adopted, candidates, historical artifacts, or dropped.

## Current host

- Machine: `bst-originpc-ai`
- OS: Ubuntu 24.04 LTS bare metal
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, `sm_120`, 96 GB VRAM
- CUDA driver/toolchain observed: driver 595.71.05, CUDA runtime 13.2, nvcc 13.0
- Remote workflow: [remote-rtx-pro-6000-access-contract.md](remote-rtx-pro-6000-access-contract.md)

## Working context

- This remains a home-lab science fork: measure first, keep negative results, and preserve an easy upstream merge/rebase path.
- Current benchmark and development runs happen on bare-metal `bst-originpc-ai` over Tailscale, not the older WSL2 port-forward path.
- Use the visible `tmux` session `codex-ds4` plus timestamped logs under `~/ds4/codex-runs/latest` for remote experiments.
- Token-byte parity and long-context exactness are correctness gates. Performance patches are only useful if they preserve the known correctness shape.
- Cite or link sources for external performance claims; otherwise treat them as ideas to reproduce locally.

## Required runtime policy

Correctness-safe default for this 96 GB discrete card:

```bash
# Do not set DS4_CUDA_Q8_F16_CACHE_RESERVE_MB for correctness-gated runs.
# Do not use the old 128 MB reserve policy as a default.
```

Why:

- `./ds4_test --long-context` passes with the default q8->f16 cache reserve.
- Lowering `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` improves prefill in short benchmarks, but it breaks the long-context correctness gate.
- `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` is not useful as a default when the reserve is left alone, because the q8->f16 cache is already budget-exhausted before it can populate.
- On Blackwell CUDA (`sm_12x`), one-token f16 matvecs now default to the generic 256-thread path. Set `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` to restore the old ordered 32-thread path for A/B.

Canonical reference CSVs live in `speed-bench/`:

- `rtx_pro_6000_default_reserve.csv`
- `rtx_pro_6000_reserve_512mb.csv`
- `rtx_pro_6000_reserve_128mb.csv`
- `rtx_pro_6000.csv` - historical low-reserve policy baseline, not correctness-safe
- `rtx_pro_6000_official_65536_20260514.csv` - upstream-shaped `ctx=2048..65536` sweep on current `main`
- `rtx_pro_6000_f16_default_official_65536_20260515.csv` - upstream-shaped sweep with the Blackwell generic f16 default patch

## Current baseline

Command:

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128
```

Current correctness-safe baseline is the upstream-shaped sweep with the Blackwell generic f16 default patch:

- Run: `~/ds4/codex-runs/20260515-003703-f16-default-official-sweep`
- CSV: `speed-bench/rtx_pro_6000_f16_default_official_65536_20260515.csv`
- `ctx=2048`: 497.56 prefill t/s, 46.74 gen t/s
- `ctx=32768`: 459.76 prefill t/s, 40.74 gen t/s
- `ctx=65536`: 437.33 prefill t/s, 37.88 gen t/s

Historical low-reserve baseline, before the small decode Q norm+RoPE fusion. Keep this only as a performance artifact; it is not correctness-safe:

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

## Related projects and future tracks

Track adjacent projects for implementation ideas, external performance references, and possible future deployment paths. These are not direct baselines for the current DS4 single-GPU GGUF path unless the hardware, quantization, runtime, and benchmark shape match.

- [LnaLang4U](https://github.com/lna-lab/LnaLang4U): DeepSeek V4 Flash system around SGLang, currently framed around 4x RTX Pro 6000 96 GB with tensor parallelism. Useful as a future SGLang deployment reference and as a source of sm_120 / sparse-kernel ideas. Its reported CUDA Graphs ON/OFF delta is large (`57` tok/s vs `9.6` tok/s for 200-token single-request output), so treat CUDA Graph replay as a serious DS4 feasibility target even though DS4's native C/CUDA runtime will not necessarily see the same framework-overhead reduction as SGLang/PyTorch.
- [deepseek-v4-flash-sm120](https://github.com/0xSero/deepseek-v4-flash-sm120): sm_120 enablement patch for SGLang / FlashMLA kernels on RTX Pro 6000 Blackwell. Useful if we test SGLang or need to understand Blackwell-specific sparse decode support outside DS4.
- [antirez/llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash): experimental llama.cpp fork that added DSv4 support for the same GGUF/model family before DS4 became the focused inference engine. Treat this mostly as a historical and integration reference: useful for understanding the model format, GGUF conversion assumptions, and how DS4 grew out of the earlier llama.cpp route, but not the main RTX Pro 6000 performance target.
- [DataCamp RunPod llama.cpp tutorial](https://www.datacamp.com/tutorial/how-to-run-deepseek-v4-flash-locally): practical user-facing walkthrough for running DeepSeek V4 Flash on an RTX Pro 6000 RunPod with a modified `llama.cpp` branch and an FP4/FP8 GGUF. Useful as an operational reference for how painful the current non-DS4 ecosystem is, and as another RTX Pro 6000 local-inference data point. Do not use it as a clean performance baseline: the model file is about 146 GB and the `llama-server --fit on` path can distribute layers across GPU and CPU memory rather than keeping the whole workload in VRAM.
- [Reddit BlackwellPerformance MTP self-speculation report](https://www.reddit.com/r/BlackwellPerformance/comments/1t9ewvn/deepseekv4flash_w4a16fp8_with_mtp_selfspeculation/): vLLM-based W4A16+FP8 + MTP self-speculation report on 2x RTX Pro 6000 Max-Q, including very long-context numbers. Treat this as a reasonable RTX Pro 6000-class benchmark reference point for a 2-card, roughly 4-bit deployment path. This gives useful perspective against DS4's current 1-card, roughly 2-bit deployment goal: not the same workload, but a helpful throughput-per-hardware comparison. It is also evidence that speculative decode plus alternate quant/runtime stacks may unlock much higher generation throughput.
- [NVIDIA Developer Forum DS4-on-Spark thread](https://forums.developer.nvidia.com/t/fully-custom-cuda-native-deepseek-4-flash-optimized-for-1x-spark-antirez-ds4/369791/10): useful DS4-specific external reference. The thread includes 1x Spark llama-benchy measurements, a roofline-style decode discussion, a warning that DS4's own average generation metric and llama-benchy steady-state `tg` metric are not the same, and an early `TP=2` DS4 tensor-parallel branch by `SeraphimSerapis`. The TP branch was explicitly described as "not working yet" when posted, so track it as a source of ideas rather than a dependency.

## Storage and KV offload note

Use the current Corsair MP700 PRO XT first for any future disk-KV/offload work. It is a very fast PCIe 5.0 NVMe drive, with Corsair listing the 2 TB model at up to 14,900 MB/s sequential read and 14,500 MB/s sequential write: <https://www.corsair.com/us/en/p/storage-drives/CSSD-F20GBMP700PXNH/mp700-pro-xt-2tb-pcie-5-0-x4-nvme-m-2-ssd>

Intel Optane P5800X is still interesting for a very specific role: latency-sensitive KV/offload paging where small random reads or tail latency block decode. Intel positions P5800X around predictable low latency and heavy-write endurance, with the product page listing 100 DWPD: <https://www.intel.com/content/www/us/en/products/sku/201840/intel-optane-ssd-dc-p5800x-series-3-2tb-2-5in-pcie-x4-3d-xpoint.html>. StorageReview measured very strong random 4K read behavior, peaking around 1.4M IOPS at about 85.5 us latency in its VDBench test: <https://www.storagereview.com/review/intel-optane-ssd-p5800x-review>

Decision rule:

- If DS4/SGLang/vLLM offload uses large async prefetches and high queue depth, the MP700 PRO XT should be tried first and may be faster.
- If profiling shows decode stalls on small random KV page misses, low queue depth reads, fsync-like behavior, or bad tail latency, Optane becomes a credible experiment.
- Do not buy or redesign around Optane until the current NVMe path is measured with the actual offload engine. The right benchmark is not just sequential bandwidth; it is DS4/SGLang/vLLM token throughput plus fio-style QD1/QD4/QD32 random read latency for the page size the runtime actually uses.

## Merged RTX Pro 6000 changes

- `a865750` / related speed-bench commits: low-reserve q8 f16 cache policy and CSV baselines. The policy is now revoked as a correctness default; keep the CSVs as historical probes only.
- `1d35fbb`: pair decode q and kv q8 matvecs. Small but repeatable decode gain.
- `5347041`: fuse decode Q head RMS norm and RoPE via existing CUDA helper. Small repeatable decode gain, with `DS4_METAL_DISABLE_Q_HEAD_ROPE_FUSION=1` as the reference fallback.
- `da9a129`: half-warp MoE decode LUT gate/up kernel. Repeatable RTX Pro 6000 decode gain with `DS4_CUDA_MOE_NO_DECODE_LUT_H16=1` as the reference fallback:
  - `ctx=4096`: about 43.8 gen t/s vs 42.6 fallback.
  - `ctx=32768`: about 38.6 gen t/s vs 37.7 fallback.
  - `ctx=131072`: 31.51 gen t/s vs 30.81 fallback in a one-run smoke.
- `a486c90`: use generic one-token f16 matvecs by default on Blackwell CUDA, with `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` as the old-path fallback:
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

Operational note: future fallback-disabling probes should use shorter generation counts first. Slow fallback checks are useful negative results, but they should not occupy the GPU for long unless the early result shows a plausible gain.
- Generic paired f16 kernel experiment, run `~/ds4/codex-runs/20260515-005807-f16-pair-generic-ab`: build and smoke passed, but generation was unchanged at `ctx=32768` (`40.34` t/s with pair vs `40.34` without pair). Rejected as extra kernel surface without meaningful decode gain.
- Single-token indexed attention through grouped heads8 kernel, run `~/ds4/codex-runs/20260515-011513-indexed-heads8-single-ab`: smoke passed, but `ctx=32768` generation dropped (`38.73` t/s patched vs `40.30` t/s old per-head path). Rejected.
- Env-gated non-indexed decode heads8 single-token route, run `~/ds4/codex-runs/20260515-012123-decode-heads8-single-ab`: `ctx=32768` moved from `40.33` to `40.47` t/s. Too small to justify making default without a larger related attention rewrite.
- CUB `top_k=512` fast path for `n_comp<=8704`, run `~/ds4/codex-runs/20260515-014029-topk8704-ab`: strong speed signal at `ctx=32768`, 512 generated tokens (`42.85` / `42.73` gen t/s vs `40.31` / `40.30` with `DS4_CUDA_NO_TOPK8704=1`). Rejected because validation changed the correctness shape:
  - Validation run `~/ds4/codex-runs/20260515-014752-topk8704-validate`: `make test` had 8 failures, adding a new `long-context` wrong-assignment failure.
  - Disable check `~/ds4/codex-runs/20260515-015233-topk8704-longctx-disable-check`: same `long-context` failure even with `DS4_CUDA_NO_TOPK8704=1`.
  - Clean check `~/ds4/codex-runs/20260515-015459-clean-longctx-check`: code reverted to `a486c90`, `./ds4_test --long-context` returned OK.
  - Conclusion: top-k/indexer is high leverage for generation speed, but this form is correctness-invalid and possibly compile-layout/numerical-sensitivity prone. Future top-k work must preserve the old selection/order behavior at token-byte parity, not just the top-k set.
- 8192+tail top-k candidate merge, run `~/ds4/codex-runs/20260515-030000-topk8192-tail-ab`: speed signal repeated at `ctx=32768`, 512 generated tokens (`42.08` / `42.06` gen t/s vs `40.33` / `40.31` with `DS4_CUDA_NO_TOPK8192_TAIL=1`). Rejected because the clean validation run `~/ds4/codex-runs/20260515-031000-topk8192-tail-longctx` failed `./ds4_test --long-context` with the same Alice wrong-assignment regression (`got 50 expected 52`). Conclusion: the near-8192 top-k region is a real speed lever, but even a more conservative candidate-merge shortcut is not correctness-safe.
- CUB per-4096-chunk top-k sorter, run `~/ds4/codex-runs/20260515-033300-topk-chunk-cub-ab`: smaller but repeatable speed signal at `ctx=32768`, 512 generated tokens (`41.54` / `41.54` gen t/s vs `40.33` / `40.28` baseline). Rejected because validation run `~/ds4/codex-runs/20260515-034300-topk-chunk-cub-longctx` failed `./ds4_test --long-context` with the same Alice wrong-assignment regression (`got 50 expected 52`). Conclusion: speculative faster top-k sorters are closed for now; any future top-k work needs a selection-comparison harness against the current path before benchmarking.
- Postscript on the top-k rejections: `~/ds4/codex-runs/20260515-042542-longctx-env-matrix` showed that `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128` fails `./ds4_test --long-context` even without a top-k fast path. Treat earlier top-k long-context failures that used the low-reserve policy as contaminated. The speed signal remains interesting, but it must be revalidated under the correctness-safe default reserve.
- Temporary 8192+tail comparison harness, run `~/ds4/codex-runs/20260515-045007-topk8192-tail-safe-default` and `~/ds4/codex-runs/20260515-045822-topk-active-correctness-matrix`: not acceptable as a validation harness. It showed zero selected-index mismatches in a `ctx=32768` compare bench, but the patched binary failed `./ds4_test --long-context` even with no top-k env var. Reverted. Any future top-k harness must avoid perturbing the current binary's long-context behavior before it can be trusted.
- MoE decode half-warp LUT row128 experiment, run `~/ds4/codex-runs/20260515-020046-moe-h16-row128-ab`: build and smoke passed, but generation dropped repeatably at `ctx=32768`, 512 generated tokens (`38.53` / `38.53` gen t/s vs baseline `40.31` / `40.31`). Rejected. Halving the number of gate/up blocks did not compensate for lower per-block efficiency.
- q8 decode fallback switches, run `~/ds4/codex-runs/20260515-024932-q8-switch-ab`: all tested switch-offs were worse at `ctx=32768`, 512 generated tokens.
  - Baseline: `512.58` prefill t/s, `40.39` gen t/s.
  - `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1`: `500.76` prefill t/s, `40.10` gen t/s.
  - `DS4_CUDA_NO_Q8_BATCH_WARP=1`: `434.00` prefill t/s, `40.26` gen t/s.
  - `DS4_CUDA_NO_Q8_DP4A=1`: `325.80` prefill t/s, `38.88` gen t/s.
  - Conclusion: the existing q8 fast paths are helping; this does not look like a simple q8 fallback switch problem.
- CUDA split-flush A/B, run `~/ds4/codex-runs/20260515-035705-split-flush-ab`: disabling the mid-token split flush with `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` did not improve real benchmark-context generation.
  - `ctx=4096`: default split `45.95` gen t/s, no split `45.93` gen t/s.
  - `ctx=32768`: default split `40.32` gen t/s, no split `40.33` gen t/s.
  - Conclusion: short-prompt token profiles can mislead here because `encode=` includes a CUDA synchronize after the early layer split. Do not treat split disabling as a throughput lever for the upstream benchmark shape.
- Single-token indexed attention grouped online variants, run `~/ds4/codex-runs/20260515-041034-single-indexed-heads-ab`: build and smoke passed, but generation got worse.
  - Baseline: `508.50` prefill t/s, `40.32` gen t/s.
  - Heads4 online route: `498.21` prefill t/s, `34.32` gen t/s.
  - Heads8 online route: `495.48` prefill t/s, `37.84` gen t/s.
  - Conclusion: grouping heads reduces KV reloads but starves single-token decode parallelism. Keep the current per-head indexed attention path unless a new design preserves enough block-level parallelism.

## Current diagnosis

The initial WSL2 path is not a clean baseline. It combined WSL2 with an earlier draft of the CUDA backend, while the bare-metal Ubuntu measurements use later upstream CUDA changes. Treat the WSL2 result as historical evidence that the first setup was bad, not as proof that WSL2 alone caused the catastrophic behavior. Current bare-metal Ubuntu generation is roughly 38-43 t/s depending on context.

Remaining decode performance is not dominated by one obvious allocation fallback. The q8 f16 cache pressure is real, but lowering the reserve is not a valid correctness-preserving mitigation on this model/hardware combination. Leave the reserve unset for correctness-gated runs. There is not yet a correctness-safe prefill-recovery policy.

Fallback audit at `ctx=32768`:

- Run: `~/ds4/codex-runs/20260515-040252-fallback-audit`
- Result: `506.31` prefill t/s, `40.76` gen t/s.
- Filtered stderr showed no hard CUDA allocation failures. The recurring fallback was q8 f16 cache budget exhaustion only: 1056 budget messages after 192 q8 f16 cache entries and full model tensor-span caching.
- Conclusion: upstream's allocation-fallback hypothesis is directionally plausible for q8 f16 cache pressure, but current RTX Pro 6000 evidence does not show an unknown buffer allocation failure path. The attempted reserve reduction improved prefill but broke correctness.

Reserve correctness boundary:

- Run: `~/ds4/codex-runs/20260515-043330-reserve-correctness-boundary`
- Default reserve: `./ds4_test --long-context` OK in this run, first q8 f16 cache attempt exhausted with `cached=0.00 GiB`, `reserve=4.75 GiB`.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096`: failed long-context with Alice wrong-assignment (`got 50 expected 52`).
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=3072`: failed the same way.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=2816`: failed after q8 f16 cache reached about `0.14 GiB`.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=2560`, `2048`, and `128`: all failed the same way.
- Conclusion: do not set `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` for correctness-gated runs. Treat all low-reserve speed CSVs as historical performance probes, not deployable policy.

Long-context stability follow-up:

- Run: `~/ds4/codex-runs/20260515-051459-longctx-correctness-matrix`
  - Current `a486c90` default failed twice with Alice wrong-assignment.
  - `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` also failed.
  - `DS4_CUDA_NO_Q8_F16_CACHE=1` passed once, but this did not repeat reliably.
- Run: `~/ds4/codex-runs/20260515-052530-noq8-cache-repeat`
  - `DS4_CUDA_NO_Q8_F16_CACHE=1`: fail, fail, pass.
  - `DS4_CUDA_Q8_F16_CACHE_MB=0`: fail, fail.
  - Bench with `DS4_CUDA_NO_Q8_F16_CACHE=1`: `341.69` prefill t/s, `40.31` gen t/s at `ctx=32768`, 512 generated tokens.
- Run: `~/ds4/codex-runs/20260515-054343-parent-longctx-check2`
  - Parent commit `50d8228` passed default twice and passed `DS4_CUDA_NO_Q8_F16_CACHE=1` once.
- Run: `~/ds4/codex-runs/20260515-055915-f16-code-revert-validate`
  - Reverting only the `ds4_cuda.cu` code delta from `a486c90` gave a mixed result: pass, fail, then generic opt-in pass.
  - Bench after code revert: `341.62` prefill t/s, `38.61` gen t/s at `ctx=32768`, 512 generated tokens.
- Current interpretation: low-reserve q8 f16 caching is definitely not deployable, but the Alice long-context check is also marginal on the CUDA path. Do not promote `DS4_CUDA_NO_Q8_F16_CACHE=1` or a code revert as a fix. Next correctness work should capture the generated long-context output/logit divergence around the Alice fact, not keep toggling cache policy blindly.

Upstream status as of 2026-05-14: upstream has new server/API commits plus `04b6fda` (`cuda: use managed KV cache for huge contexts`). That CUDA change is scoped to very large KV caches and should not affect the current 32k/65k benchmark phase. Treat it as a separate long-context stability item, not as part of the RTX Pro 6000 throughput work, and verify carefully before merging because managed KV can trade performance for allocation survivability.

At the official benchmark shape, RTX Pro 6000 is already much faster than DGX Spark in absolute terms, but not in proportion to the hardware delta. At `ctx=32768`, the Blackwell f16 default patch reaches 40.74 t/s versus the upstream DGX Spark row at 13.75 t/s: about 3.0x, still below the broad 4x compute-ratio target band.

## Upstream PR watchlist

As of 2026-05-15, upstream has many open PRs. Relevant ones for this fork:

- [#121 cuda: skip ordered f16 matmul on Blackwell](https://github.com/antirez/ds4/pull/121): aligns with our independently validated Blackwell f16 default patch. Track for upstream merge/rebase cleanup.
- [#145 cuda: add launch-bounded tile8 MoE down path](https://github.com/antirez/ds4/pull/145): opt-in `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN` path layered on #121. Worth a quick RTX Pro 6000 A/B because it targets a real decode bucket, but only keep if it moves `ctx=32768`, 512-token generation by a repeatable margin.
- [#153 cuda: add direct-model partial weight cache](https://github.com/antirez/ds4/pull/153): important for 24 GB direct-model users. Not a primary RTX Pro 6000 target because our card can hold the full 80.76 GiB model image in VRAM, but the dense-weight prioritization is useful background for future cache policy thinking.
- [#158 cuda: fix HMM path on coherent unified-memory systems](https://github.com/antirez/ds4/pull/158): GB10 / coherent unified-memory path, not a discrete RTX Pro 6000 path. Track only to avoid accidentally importing unified-memory assumptions.
- [#159 fix(kv-cache): guard never-hit entries from immediate self-eviction](https://github.com/antirez/ds4/pull/159) and [#134 Guard KV cache against page-cache pressure](https://github.com/antirez/ds4/pull/134): server disk-KV cache robustness. Useful for long-running agent use, not decode throughput.
- [#107 Add RTX 6000 speed](https://github.com/antirez/ds4/pull/107): external RTX 6000 README benchmark reference. Treat as methodology-sensitive; our current bare-metal RTX Pro 6000 numbers are higher on current code.

No open upstream PR in this scan appears to implement NVIDIA CUDA Graph capture/replay for DS4.

`ctx=32768` decode stage profile:

- Run: `~/ds4/codex-runs/20260514-225658-ctx32768-decode-stage-profile`
- Profiled result with overhead: 509.13 prefill t/s, 34.84 gen t/s.
- Per-token profile: about 28.67 ms/token.
- Top stage buckets: attention about 22.6%, compressor/indexer about 18.0%, routed MoE about 12.7%, ffn/attn HC pre about 18.7% combined, attention output about 9.2%, q path about 6.9%.

Current per-token graph timing caveat:

- Run: `~/ds4/codex-runs/20260515-035300-token-profile-ctx32768`
- Terminology: DS4's "graph backend" is active on CUDA, but this is DS4's internal fixed inference tape / graph runtime. It is not NVIDIA CUDA Graph capture/replay. There is currently no `cudaGraphLaunch` path to turn on with an environment variable.
- Result with profiling overhead: `506.80` prefill t/s, `40.92` gen t/s.
- Average over 32 generated tokens: 24.400 ms/token total.
- Breakdown: encode 9.732 ms/token (39.88%), execute 14.635 ms/token (59.98%), read 0.034 ms/token (0.14%).
- Caveat: on CUDA, `encode=` is not pure CPU launch time because the decode path can call `ds4_gpu_flush_commands()` after the early layer split, and CUDA implements that as `cudaDeviceSynchronize()`.
- Follow-up run `~/ds4/codex-runs/20260515-035705-split-flush-ab` showed that disabling the split does not improve real `ds4-bench` generation at `ctx=4096` or `ctx=32768`.
- Conclusion: CUDA Graph replay remains plausible because launch count is high, but the token-profile `execute-only` ceiling estimate was too optimistic. Prioritize it behind confirmed GPU-time buckets unless a capture probe shows a much cleaner path than expected.

Decode-only delayed Nsight at `ctx=32768`:

- Run: `~/ds4/codex-runs/20260514-234008-ctx32768-decode-only-nsys`
- Top kernel buckets include indexed attention, f16 ordered matmuls, dense attention, MoE decode LUT, grouped q8 matvecs, HC q8 matvecs, RMS norms, and indexer top-k chunk/merge kernels.
- CUDA API launch count remains very high, but this is not only host overhead: the GPU kernels themselves are fragmented and many are very small.

Indexer stage profile at `ctx=32768`, 16 generated tokens:

- Run: `~/ds4/codex-runs/20260515-040650-indexer-stage-profile`
- Result with profiling overhead: `507.00` prefill t/s, `40.34` gen t/s.
- Decode-indexer totals: `decode_attention` 65.694 ms, `decode_topk` 31.362 ms, `decode_score` 10.516 ms.
- Approximate per-token decode-indexer profile: indexed attention about 4.1 ms/token, top-k about 2.0 ms/token, score about 0.7 ms/token.
- Conclusion: top-k is a real lever but not the whole indexer cost. The indexed attention gather path is at least as important, and future top-k work needs a correctness harness before another sorter shortcut.

Patched decode-only delayed Nsight after the Blackwell generic f16 default:

- Run: `~/ds4/codex-runs/20260515-004855-patched-ctx32768-decode-nsys`
- Top buckets: indexed attention about 14.4%, generic f16 matvecs about 13.8%, dense attention about 8.3%, MoE decode LUT about 8.2%, grouped q8 about 6.8%, HC q8 expansion about 5.1%, large q8 matvec about 5.0%, q8 warp matvec about 4.8%, RMS norm about 4.5%, indexer top-k chunk/merge about 7.2% combined.
- The f16 patch reduced the old f16 ordered+pair cost but did not eliminate f16 matvecs as a major bucket.

MoE profile at `ctx=32768`, 64 generated tokens:

- Run: `~/ds4/codex-runs/20260515-034900-moe-profile-current`
- Result with profiling overhead: `507.91` prefill t/s, `39.42` gen t/s.
- Decode MoE calls: 2752 calls, 235.650 ms total, about 3.68 ms/token.
- Decode MoE breakdown: gate/up 156.193 ms (66.28%), down 54.898 ms (23.30%), x quant 11.008 ms (4.67%), mid quant 11.008 ms (4.67%), sort effectively zero.
- Conclusion: MoE gate/up is the MoE subtarget, but total MoE time is not large enough to explain the full gap to the 55-65 t/s target by itself.

Temporary f16 matmul shape stats:

- Run: `~/ds4/codex-runs/20260515-011009-f16-matmul-stats`
- `ctx=32768`: 511.15 prefill t/s, 40.42 gen t/s.
- Summary: 15 f16 matmul shapes, 157,618 calls, about 18.8T estimated FMAs across prefill plus generation.
- Largest shapes are `4096x1024`, `1024x8192`, `4096x256`, `4096x512`, and `16384x24`. This confirms f16 work is concentrated in known compressor/indexer/router/HC projections, not an unknown allocation fallback.

Nsight on a decode-focused fixed-token bench:

```bash
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

Goal for the next phase: improve RTX Pro 6000 generation speed at the upstream benchmark shape first, then re-check longer contexts. Do not spend time on changes that only move short-prompt smoke tests.

1. **Long-context exactness probe**
   - The Alice wrong-assignment failure is now the gating issue for both prefill-cache recovery and decode speed patches.
   - Stop treating `./ds4_test --long-context` as only a pass/fail gate. Add or use instrumentation that captures the generated text/logit path around the Alice fact so we can identify where the CUDA path diverges.
   - Until this is understood, do not promote low-reserve q8 f16 cache policy or top-k shortcuts as correctness-safe.

2. **CUDA Graph replay feasibility**
   - LnaLang4U reports a large CUDA Graphs ON/OFF gap on RTX Pro 6000-class hardware, and DS4 still launches many small decode kernels per token.
   - Do not assume DS4 can get the same gain as SGLang/PyTorch: DS4 is native C/CUDA and already avoids a lot of framework overhead.
   - First step: make a minimal decode capture/replay feasibility probe for a fixed benchmark shape, or prove exactly which token-varying state prevents capture.
   - Success criterion is practical: a repeatable generation gain at `ctx=32768`, not a synthetic launch-count reduction.

3. **Benchmark calibration against external references**
   - Run or reproduce an apples-to-apples DS4 server benchmark where possible, especially `llama-benchy`-style steady-state `tg`, because DS4's average generation metric and forum/README metrics are not always the same.
   - Keep the 2-card, roughly 4-bit RTX Pro 6000 reports as perspective, not as a direct target for the 1-card roughly 2-bit DS4 path.
   - Maintain the target bands: below 50 t/s is not enough; 55-65 t/s is the first acceptable generation band; 70-80 t/s is the strong target.

4. **Selective q8 f16 cache recovery**
   - Prefill is important, and the low-reserve policy showed the upside. The problem is exactness, not lack of speed.
   - After the exactness probe exists, re-test q8 f16 caching by tensor group/label rather than blunt reserve size.
   - Goal: recover safe prefill wins while preserving the Alice long-context behavior.

5. **Indexed attention / indexer redesign**
   - Highest combined decode target after the patched Nsight profile: indexed attention, dense attention, top-k chunk/merge, and indexer score/direct kernels.
   - Built-in fallback toggles and single-token grouped-head routing did not help.
   - Three top-k sorter shortcuts produced real speed signals, but their long-context failures may have been contaminated by the now-invalid low-reserve policy.
   - Next useful work here is a correctness-preserving harness under the default reserve. First prove the harness is a no-op by passing `./ds4_test --long-context` with no fast-path env var, then compare selected index/order behavior against the current chunked path before accepting any top-k or attention-gather speedup.

6. **MoE decode kernel inspection**
   - MoE decode LUT and related routed/shared kernels remain large GPU-time buckets.
   - First concrete task: inspect current fused MoE helpers and their launch count before writing new kernels.
   - Keep only changes that improve `ctx=32768`, 512-token generation by a repeatable margin.

7. **q8 decode matvec path**
   - Multiple q8 buckets together are material, and the fp16 cache budget issue is not safely mitigated by reserve tuning.
   - First concrete task: split q8 decode time by call shape/label if needed, then target the largest repeated decode-only shapes.
   - Avoid chasing more q8->fp16 cache policy unless the exactness issue is understood; low reserve values currently fail `./ds4_test --long-context`.

8. **Huge-context stability and disk-KV pass**
   - Defer until upstream-benchmark generation speed and the Alice exactness issue are better understood.
   - Revisit upstream managed-KV-cache changes separately; they may help huge contexts but can trade off discrete-GPU performance.
   - If KV offload becomes necessary for maximum context, measure the MP700 PRO XT path before considering Optane.

Acceptance rule: keep a performance patch only if it builds cleanly, passes smoke, preserves the known correctness-gate shape, and shows a repeatable generation gain at `ctx=32768` with 512 generated tokens. A gain below roughly 2% should be treated as documentation-only evidence unless it unlocks a larger follow-up.

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
