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
# Use low-reserve q8->f16 cache tuning only as an explicit per-run benchmark knob.
```

Why:

- `./ds4_test --long-context` passes with the default q8->f16 cache reserve.
- After rebasing onto upstream `c9dd949`, `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128` also passes `./ds4_test --long-context`.
- However, low reserves are still not clean enough for a global runtime policy: `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 make test CUDA_ARCH=sm_120` OOMs during `logprob-vectors`, while `512` MB changes the logprob-vector failure shape.
- `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` is not useful as a default when the reserve is left alone, because the q8->f16 cache is already budget-exhausted before it can populate.
- On Blackwell CUDA (`sm_12x`), one-token f16 matvecs now default to the generic 256-thread path. Set `DS4_CUDA_USE_ORDERED_F16_MATMUL=1` to restore the old ordered 32-thread path for A/B.

Canonical reference CSVs live in `speed-bench/`:

- `rtx_pro_6000_default_reserve.csv`
- `rtx_pro_6000_reserve_512mb.csv`
- `rtx_pro_6000_reserve_128mb.csv`
- `rtx_pro_6000.csv` - historical low-reserve policy baseline; useful for prefill comparison, not a global default
- `rtx_pro_6000_official_65536_20260514.csv` - upstream-shaped `ctx=2048..65536` sweep on current `main`
- `rtx_pro_6000_f16_default_official_65536_20260515.csv` - upstream-shaped sweep with the Blackwell generic f16 default patch
- `rtx_pro_6000_post_topk8704_default_65536_20260518.csv` - post-topk8704 upstream-shaped sweep with correctness-safe default reserve
- `rtx_pro_6000_post_topk8704_r128_no_attn_output_65536_20260518.csv` - post-topk8704 sweep with per-run prefill cache knobs
- `rtx_pro_6000_post_topk8704_focused_32768_gen512_20260518.csv` - post-topk8704 focused generation run at the upstream context shape

## Current baseline

Command:

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

Pre-topk post-`c9dd949` official-shaped sweeps:

- Run: `~/ds4/codex-runs/20260517-post-c9dd949-bench`
- Worktree: `~/ds4/codex-worktrees/post-c9dd949-bench-20260517-191237`
- Default CSV: `speed-bench/rtx_pro_6000_post_c9dd949_default_65536_20260517.csv`
- Tuned CSV: `speed-bench/rtx_pro_6000_post_c9dd949_r128_no_attn_output_65536_20260517.csv`

Default reserve:

- `ctx=2048`: 370.82 prefill t/s, 46.75 gen t/s
- `ctx=32768`: 342.81 prefill t/s, 40.61 gen t/s
- `ctx=65536`: 328.06 prefill t/s, 37.74 gen t/s

Per-run tuned reserve (`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`, `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`):

- `ctx=2048`: 488.87 prefill t/s, 46.58 gen t/s
- `ctx=32768`: 459.44 prefill t/s, 40.67 gen t/s
- `ctx=65536`: 438.34 prefill t/s, 37.81 gen t/s

Interpretation: the tuned cache policy still gives about 32-34% prefill gain in the 65k-max benchmark shape, but generation is unchanged. The gain is smaller than older 32k-max sweeps because the larger max context buffer leaves less free VRAM for q8->fp16 cache.

Current post-topk8704 official-shaped sweeps:

- Run: `~/ds4/codex-runs/20260518-000141-post-topk8704-full-sweep`
- Code: `main` at `fe56aba`, including adopted CUB `top_k=512` fast path for `8192 < n_comp <= 8704`
- Default CSV: `speed-bench/rtx_pro_6000_post_topk8704_default_65536_20260518.csv`
- Tuned CSV: `speed-bench/rtx_pro_6000_post_topk8704_r128_no_attn_output_65536_20260518.csv`
- Focused CSV: `speed-bench/rtx_pro_6000_post_topk8704_focused_32768_gen512_20260518.csv`

Default reserve:

- `ctx=2048`: 371.34 prefill t/s, 46.78 gen t/s
- `ctx=32768`: 342.95 prefill t/s, 43.03 gen t/s
- `ctx=65536`: 328.28 prefill t/s, 37.77 gen t/s

Per-run tuned reserve (`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`, `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`):

- `ctx=2048`: 489.00 prefill t/s, 46.59 gen t/s
- `ctx=32768`: 459.18 prefill t/s, 43.07 gen t/s
- `ctx=65536`: 438.11 prefill t/s, 37.82 gen t/s

Focused `ctx=32768`, `gen_tokens=512`, tuned reserve: 493.76 prefill t/s, 42.61 gen t/s.

Interpretation: the adopted topk8704 patch moved the upstream-shape generation baseline from about `40.6` to about `43.0` t/s at `ctx=32768`. The low-reserve cache policy still matters for prefill but still does not materially move generation. The generation cliff above `ctx=32768` remains: both default and tuned sweeps drop from about `43` t/s at `32768` to about `40` t/s at `34816`, then decay toward `37.8` t/s at `65536`.

External Q4 decode reference branch:

- Source: [`ngc-shj:perf/clean`](https://github.com/antirez/ds4/compare/main...ngc-shj:ds4:perf/clean), fetched locally as `ngc-shj/perf-clean`.
- Branch snapshot: `b3e92eb`.
- Local RTX run: `~/ds4/codex-runs/20260518-055406-ngc-shj-perf-clean-q4-probe`.
- Follow-up memory-policy run: `~/ds4/codex-runs/20260518-055846-ngc-shj-q4-no-f16-probe`.
- Reserve scan: `~/ds4/codex-runs/20260518-060219-ngc-shj-q4-reserve-scan`.
- Component scan: `~/ds4/codex-runs/20260518-061807-ngc-shj-q4-component-scan`.

Observed on RTX Pro 6000:

- The branch builds for `CUDA_ARCH=sm_120`, but produces warnings in CUDA graph/speculative-prefix code. Do not merge wholesale.
- Q4 preload size: `3.55 GiB`, logged as `345 Q8_0 + 358 F16` tensors.
- Q4 with the current low-reserve q8->f16 cache policy fails: the two caches compete for the last VRAM margin and prefill hits CUDA OOM.
- Q4 with `DS4_CUDA_NO_Q8_F16_CACHE=1` fits and smoke generation jumps from about `53.1` t/s to `72.8` t/s on a tiny CLI decode.
- In the focused `ctx=32768`, `gen_tokens=256` bench, Q4 without q8->f16 cache gives `351.19` prefill t/s and `47.68` gen t/s.
- In the reserve scan with `gen_tokens=128`, stable Q4 variants cluster around `47.9` gen t/s. Reserves from `1024` to `4096` MB effectively leave q8->f16 cache empty and prefill around `342` t/s; reserves `768` and `512` MB OOM during prefill.
- Component scan at `ctx=32768`, `gen_tokens=256`:
  - full Q4: `350.37` prefill t/s, `47.61` gen t/s.
  - `DS4_CUDA_F16_NO_Q4=1`: `341.93` prefill t/s, `40.82` gen t/s. F16-derived Q4 is the largest contributor.
  - `DS4_CUDA_Q8_NO_Q4_GENERIC=1`: `341.92` prefill t/s, `44.72` gen t/s.
  - `DS4_CUDA_Q8_NO_Q4_SINGLE=1`: `341.83` prefill t/s, `45.98` gen t/s.
  - `DS4_CUDA_Q8_NO_Q4_HCEXP=1`: `342.13` prefill t/s, `46.30` gen t/s.
  - `DS4_CUDA_Q8_NO_Q4_ATTN_OUT=1`: `342.85` prefill t/s, `46.49` gen t/s.
  - `DS4_CUDA_Q8_NO_Q4_PAIR=1`: `341.79` prefill t/s, `47.58` gen t/s, effectively neutral.

Interpretation: Q4 decode cache is the first external idea that produced a material generation win on this card: about `+11%` in the `ctx=32768` long bench. However, the full Q4 preload does not coexist with the q8->f16 prefill cache on a 96 GB card, so the exact branch trades away a large prefill win. Treat this as a design reference, not a merge candidate. The component scan says the first useful port target is F16-derived decode matvecs, not attention output. After that, generic single Q8->Q4 matters more than Q4 pair; attention-output and HC-expand are real but smaller.

Selective F16-derived Q4 port:

- Branch: `codex/q4-f16-decode`.
- Commits:
  - `9109322`: opt-in F16->Q4 decode cache and dispatch.
  - `c885036`: subset controls via `DS4_CUDA_Q4_F16_FILTER` and `DS4_CUDA_Q4_F16_CACHE_ONLY=1`.
- Build/smoke/bench run: `~/ds4/codex-runs/20260518-072457-q4-f16-decode-stage1`.
- Lazy-allocation probe: `~/ds4/codex-runs/20260518-073000-q4-f16-lazy-probe`.
- Subset scan: `~/ds4/codex-runs/20260518-073502-q4-f16-subset-scan`.
- Long-context subset validation: `~/ds4/codex-runs/20260518-074105-q4-f16-subset-long`.
- Adoption-style repeat: `~/ds4/codex-runs/20260518-075133-q4-f16-adoption-repeat`.
- Default correctness gate: `~/ds4/codex-runs/20260518-075729-q4-f16-default-test`.

All numbers below used `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128` and `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` on the RTX Pro 6000.

At `ctx=4096`, `gen_tokens=512`:

| F16 Q4 policy | Q4 cache | Prefill t/s | Gen t/s | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 0 | 567.93 | 45.81 | no Q4 |
| all F16 Q4 | 0.30 GiB | 536.44 | 51.01 | biggest decode gain, clear prefill cost |
| `hc_` | 0.02 GiB | 559.62 | 48.66 | best cheap decode win |
| `hc_,ffn_gate_inp` | 0.04 GiB | 551.58 | 49.04 | small extra decode, more prefill loss |
| `hc_,compressor` | 0.18 GiB | 538.58 | 50.04 | halfway to all-Q4, but prefill cost approaches all-Q4 |

At `ctx=32768`, `gen_tokens=256`:

| F16 Q4 policy | Q4 cache | Prefill t/s | Gen t/s | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 0 | 506.35 | 42.86 | no Q4 |
| all F16 Q4 | 0.30 GiB | 472.92 | 47.50 | `+10.8%` gen, `-6.6%` prefill |
| `hc_` | 0.02 GiB | 495.31 | 45.59 | `+6.4%` gen, `-2.2%` prefill |
| `hc_,ffn_gate_inp` | 0.04 GiB | 493.24 | 45.70 | tiny gain over `hc_`, slightly more prefill loss |
| `hc_,compressor` | 0.18 GiB | 479.75 | 46.54 | middle ground, but not as efficient as `hc_` |

Lazy Q4 allocation without startup preload is not a good policy on this 96 GB card. It preserves most prefill (`502.36` t/s at `ctx=32768`) but most F16->Q4 allocations fail during decode after the q8->f16 cache fills VRAM, and generation only reaches `44.06` t/s. This confirms the issue is explicit memory scheduling, not just kernel choice.

Current interpretation: F16-derived Q4 is worth keeping as an opt-in experiment. The best-balanced preset is:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 \
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1 \
DS4_CUDA_Q4_DECODE=1 \
DS4_CUDA_Q4_F16_CACHE_ONLY=1 \
DS4_CUDA_Q4_F16_FILTER=hc_
```

This is not ready as a default. It must still pass smoke plus the known correctness-gate shape. The broader all-F16-Q4 policy is useful for decode benchmarking, but it explicitly trades away too much prefill to be the default RTX Pro 6000 setting.

Repeat at `ctx=32768`, `gen_tokens=512` confirmed the ranking:

| F16 Q4 policy | Q4 cache | Prefill t/s | Gen t/s | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 0 | 508.37 | 42.66 | no Q4 |
| `hc_` | 0.02 GiB | 496.09 | 45.43 | `+6.5%` gen, `-2.4%` prefill |
| all F16 Q4 | 0.30 GiB | 474.03 | 47.37 | `+11.0%` gen, `-6.8%` prefill |

This makes `hc_` the only plausible near-term preset. It is not a huge win, but it is the first selective Q4 result that moves generation without giving away the whole prefill cache.

Default `make test CUDA_ARCH=sm_120` on the branch, with no Q4 env vars, preserves the known failure shape: `long-context`, `tool-call-quality`, `metal-kernels`, and `server` pass; `logprob-vectors / long_memory_archive` reports the same 7 failures. This supports treating the branch as inert-by-default.

Q8-derived Q4 follow-up:

- Code points:
  - `e5a32cc`: opt-in Q8->Q4 cache and generic single-decode dispatch.
  - `f8245ab`: opt-in Q8->Q4 HC-expand dispatch.
  - `46b66d1`: split Q8 Q4 preload from generic single dispatch so HC/attention output can be isolated.
  - `f7d078c`: opt-in fused `attn_output_a` low-projection Q4 path.
  - `ee8d246`: opt-in after-prefill Q8->Q4 preload mode. This releases the optional q8->f16 prefill cache only after a successful prefill, then builds the selected Q8-derived Q4 decode cache.
- All Q8-derived Q4 paths are gated separately by `DS4_CUDA_Q4_Q8_DECODE=1` and are not part of the recommended preset.
- Runs:
  - Generic single-Q8 subset scan: `~/ds4/codex-runs/20260518-080959-q8-q4-stage2-scan`.
  - HC-expand scan: `~/ds4/codex-runs/20260518-082313-q8-q4-hcexp-fixed`.
  - Attention-output scan: `~/ds4/codex-runs/20260518-082807-q8-q4-attn-output-scan`.
  - After-prefill memory-policy scan: `~/ds4/codex-runs/20260518-083957-q8-q4-after-prefill`.
  - After-prefill repeat plus `ctx=32768`: `~/ds4/codex-runs/20260518-084332-q8-q4-after-prefill-repeat`.

`ctx=4096`, `gen_tokens=512`, all with `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128` and `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`:

| Policy | Prefill t/s | Gen t/s | Notes |
|---|---:|---:|---|
| baseline | 570.21 | 45.87 | no Q4 |
| F16 `hc_` | 564.72 | 48.73 | current balanced preset |
| F16 `hc_` + Q8 `attn_q_b` single | 512.78 | 49.97 | small gen gain, large prefill loss |
| F16 `hc_` + Q8 `attn_q_a` single | 549.95 | 48.68 | no useful gen gain |
| F16 `hc_` + Q8 `attn_kv` single | 556.30 | 48.92 | no useful gen gain |
| F16 `hc_` + Q8 shared-expert singles | 537-539 | 48.75-48.78 | no useful gen gain |
| F16 `hc_` + Q8 q-proj singles | 497.48 | 49.78 | poor tradeoff |
| F16 `hc_` + Q8 `attn_output_b` HC-expand | 510.34 | 49.88 | small gen gain, large prefill loss |
| F16 `hc_` + Q8 `ffn_down_shexp` HC-expand | 539.84 | 48.94 | no useful gen gain |
| F16 `hc_` + Q8 `attn_output_a` fused low | 511.02 | 49.78 | fused path works, but poor tradeoff |
| F16 `hc_` + Q8 `attn_output_a,b` startup preload | 468.07 | 50.88 | highest gen in this scan, but gives up too much prefill |

Interpretation: the selective Q8-derived Q4 paths are technically working, but they are not recommended. They mostly recover another `+1` to `+2` generation t/s while cutting prefill by `50` to `95` t/s. This confirms the same memory-policy problem seen in the full external Q4 branch: preloading larger Q8-derived Q4 tensors competes with the q8->f16 prefill cache on a 96 GB card. Keep the Q8 Q4 code as opt-in research scaffolding for now, but do not promote any Q8 Q4 preset unless a later memory policy preserves prefill.

After-prefill Q8-derived Q4 memory policy:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1
DS4_CUDA_Q4_DECODE=1
DS4_CUDA_Q4_F16_CACHE_ONLY=1
DS4_CUDA_Q4_F16_FILTER=hc_
DS4_CUDA_Q4_Q8_DECODE=1
DS4_CUDA_Q4_Q8_CACHE_ONLY=1
DS4_CUDA_Q8_NO_Q4_SINGLE=1
DS4_CUDA_Q4_Q8_FILTER=.attn_output_a.weight,.attn_output_b.weight
DS4_CUDA_Q4_Q8_AFTER_PREFILL=1
```

This is the first Q8-derived Q4 result that keeps most of the prefill speed. The startup-preload version reached similar generation speed but cut prefill to `473.53` t/s at `ctx=4096`. The after-prefill version lets prefill use the q8->f16 cache first, releases that optional cache, then builds the Q8-derived Q4 decode cache for the selected attention-output tensors.

`ctx=4096`, `gen_tokens=1024`, three repeats:

| Policy | Avg prefill t/s | Avg gen t/s | Notes |
|---|---:|---:|---|
| baseline | 558.56 | 45.95 | no Q4 |
| F16 `hc_` | 555.38 | 48.77 | previous balanced preset |
| F16 `hc_` + after-prefill Q8 `attn_output_a,b` | 552.71 | 51.01 | `+11.0%` gen vs baseline, `-1.0%` prefill vs F16 `hc_` |

Single `ctx=32768`, `gen_tokens=512` validation:

| Policy | Prefill t/s | Gen t/s | Notes |
|---|---:|---:|---|
| baseline | 498.86 | 42.65 | no Q4 |
| F16 `hc_` | 493.39 | 45.43 | previous balanced preset |
| F16 `hc_` + after-prefill Q8 `attn_output_a,b` | 493.13 | 47.26 | `+10.8%` gen vs baseline, essentially flat prefill vs F16 `hc_` |

This should become the next experimental preset, not a default yet. The caveat is phase state: `DS4_CUDA_Q4_Q8_AFTER_PREFILL=1` disables q8->f16 rebuilds after the first prefill, so it is appropriate for one-shot benchmark/prototype runs but may hurt later fresh prefills in a long-lived server process. A production version needs explicit prefill/decode phase ownership before promotion.

Previous post-upstream-sync default-policy baseline. This used the synced fork at `89f3a0d` with no low-reserve q8 f16 cache overrides:

- Run: `~/ds4/codex-runs/20260517-043821-sync-baseline`
- Worktree: `~/ds4/codex-worktrees/20260517-043821-sync-baseline`
- CSV: `~/ds4/codex-runs/20260517-043821-sync-baseline/bench-sync-default.csv`
- `ctx=2048`: 364.03 prefill t/s, 46.74 gen t/s
- `ctx=32768`: 339.78 prefill t/s, 40.69 gen t/s

This remains the safer full-test reference because it avoids the low-reserve q8 f16 cache policy. After the `c9dd949` sync, the Alice long-context issue is no longer open, but low-reserve settings still change the `logprob-vectors` failure shape or hit CUDA OOM.

Previous upstream-shaped sweep with the Blackwell generic f16 default patch and higher q8 f16 cache reuse:

- Run: `~/ds4/codex-runs/20260515-003703-f16-default-official-sweep`
- CSV: `speed-bench/rtx_pro_6000_f16_default_official_65536_20260515.csv`
- `ctx=2048`: 497.56 prefill t/s, 46.74 gen t/s
- `ctx=32768`: 459.76 prefill t/s, 40.74 gen t/s
- `ctx=65536`: 437.33 prefill t/s, 37.88 gen t/s

Historical low-reserve baseline, before the small decode Q norm+RoPE fusion. Keep this as a performance artifact and per-run tuning reference, not as a global policy:

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

## Adopted RTX Pro 6000 probes

- CUB `top_k=512` fast path for `8192 < n_comp <= 8704`, branch `codex/topk8704-cub-post-c9dd949`, commit `94b6d70`.
  - Runtime kill switch: `DS4_CUDA_NO_TOPK8704=1`.
  - Rerun after upstream `c9dd949` in `~/ds4/codex-runs/20260517-224346-topk8704-cub-rerun`.
  - Built cleanly for `sm_120`.
  - A/B at `ctx=32768`, 512 generated tokens, with the normal per-run prefill knobs (`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`, `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`):
    - Enabled: `42.79`, `42.68` gen t/s.
    - Disabled with `DS4_CUDA_NO_TOPK8704=1`: `40.24`, `40.24` gen t/s.
    - Net result: about `+6.1%` generation throughput for the upstream-benchmark context shape.
- Correctness gates:
  - `DS4_TEST_MODEL=$HOME/ds4/ds4flash.gguf ./ds4_test --long-context`: OK with default reserve.
  - `DS4_TEST_MODEL=$HOME/ds4/ds4flash.gguf make test CUDA_ARCH=sm_120`: same known default-reserve shape as synced main: `logprob-vectors / long_memory_archive`, `7` assertion failures; no new failure shape.
- Decision: keep as a small fork patch. The earlier May 15 top-k rejection is now treated as a contaminated historical result because it predated upstream `c9dd949` and overlapped with low-reserve correctness failures.
- Post-adoption stage profile:
  - Run: `~/ds4/codex-runs/20260517-231138-post-topk-stage-profile`.
  - Result with profiling overhead: `511.03` prefill t/s, `38.04` gen t/s.
  - Decode stage totals: attention `24.73%`, routed MoE `13.89%`, compressor/indexer `12.52%`, attention output `9.97%`, `ffn_hc_pre` + `attn_hc_pre` `18.75%` combined.
  - Next target should be attention or compressor/indexer; PR #145's down path is not on the one-token decode path.
- Post-adoption delayed Nsight decode profile:
  - Run: `~/ds4/codex-runs/20260518-001421-post-topk8704-decode-nsys`.
  - Profile command used `--delay=75 --duration=55` with `ctx=32768`, `gen_tokens=3072`, and the normal per-run prefill knobs.
  - Bench result under profiling: `504.68` prefill t/s, `41.33` gen t/s.
  - Largest GPU kernel buckets: generic f16 matvec `16.8%`, indexed attention `16.8%`, dense attention `10.4%`, MoE gate/up decode LUT `10.0%`, HC q8 expansion `6.2%`, q8 matvec `5.9%`, RMS norm `5.4%`, grouped q8 `4.6%`, HC split/norm `4.2%`, q8 pair matvec `4.0%`, MoE down sum6 `3.3%`, topk8704 CUB `2.4%`, indexer direct score `2.4%`.
  - CUDA API summary still shows extreme launch fragmentation during the capture window: about `4.0M` `cudaLaunchKernel` calls and `4,588` `cudaDeviceSynchronize` calls. This does not mean synchronization is the root cause by itself; it confirms decode is many small GPU kernels plus high launch count.

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
- Pre-`c9dd949` CUB `top_k=512` fast path for `n_comp<=8704`, run `~/ds4/codex-runs/20260515-014029-topk8704-ab`: strong speed signal at `ctx=32768`, 512 generated tokens (`42.85` / `42.73` gen t/s vs `40.31` / `40.30` with `DS4_CUDA_NO_TOPK8704=1`). Rejected at the time because validation changed the correctness shape:
  - Validation run `~/ds4/codex-runs/20260515-014752-topk8704-validate`: `make test` had 8 failures, adding a new `long-context` wrong-assignment failure.
  - Disable check `~/ds4/codex-runs/20260515-015233-topk8704-longctx-disable-check`: same `long-context` failure even with `DS4_CUDA_NO_TOPK8704=1`.
  - Clean check `~/ds4/codex-runs/20260515-015459-clean-longctx-check`: code reverted to `a486c90`, `./ds4_test --long-context` returned OK.
  - Later interpretation: this was contaminated by the pre-`c9dd949` long-context bug and/or the low-reserve cache policy. The post-`c9dd949` retest above is the active decision.
- 8192+tail top-k candidate merge, run `~/ds4/codex-runs/20260515-030000-topk8192-tail-ab`: speed signal repeated at `ctx=32768`, 512 generated tokens (`42.08` / `42.06` gen t/s vs `40.33` / `40.31` with `DS4_CUDA_NO_TOPK8192_TAIL=1`). Rejected because the clean validation run `~/ds4/codex-runs/20260515-031000-topk8192-tail-longctx` failed `./ds4_test --long-context` with the same Alice wrong-assignment regression (`got 50 expected 52`). Conclusion: the near-8192 top-k region is a real speed lever, but even a more conservative candidate-merge shortcut is not correctness-safe.
- CUB per-4096-chunk top-k sorter, run `~/ds4/codex-runs/20260515-033300-topk-chunk-cub-ab`: smaller but repeatable speed signal at `ctx=32768`, 512 generated tokens (`41.54` / `41.54` gen t/s vs `40.33` / `40.28` baseline). Rejected because validation run `~/ds4/codex-runs/20260515-034300-topk-chunk-cub-longctx` failed `./ds4_test --long-context` with the same Alice wrong-assignment regression (`got 50 expected 52`). Conclusion: speculative faster top-k sorters are closed for now; any future top-k work needs a selection-comparison harness against the current path before benchmarking.
- Postscript on the top-k rejections: `~/ds4/codex-runs/20260515-042542-longctx-env-matrix` showed that `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128` fails `./ds4_test --long-context` even without a top-k fast path. Treat earlier top-k long-context failures that used the low-reserve policy as contaminated. The speed signal remains interesting, but it must be revalidated under the correctness-safe default reserve.
- Temporary 8192+tail comparison harness, run `~/ds4/codex-runs/20260515-045007-topk8192-tail-safe-default` and `~/ds4/codex-runs/20260515-045822-topk-active-correctness-matrix`: not acceptable as a validation harness. It showed zero selected-index mismatches in a `ctx=32768` compare bench, but the patched binary failed `./ds4_test --long-context` even with no top-k env var. Reverted. Any future top-k harness must avoid perturbing the current binary's long-context behavior before it can be trusted.
- MoE decode half-warp LUT row128 experiment, run `~/ds4/codex-runs/20260515-020046-moe-h16-row128-ab`: build and smoke passed, but generation dropped repeatably at `ctx=32768`, 512 generated tokens (`38.53` / `38.53` gen t/s vs baseline `40.31` / `40.31`). Rejected. Halving the number of gate/up blocks did not compensate for lower per-block efficiency.
- MoE decode gate/up pair2 experiment, branch `codex/moe-decode-gate-pair2`, run `~/ds4/codex-runs/20260517-232852-moe-gate-pair2-ab`: build passed, but generation regressed.
  - Pair2: `41.10`, `41.07` gen t/s.
  - Default: `42.65`, `42.65` gen t/s.
  - Conclusion: sharing the quantized input/LUT setup across two selected experts loses too much block-level parallelism. Do not carry.
- MoE decode policy scan, run `~/ds4/codex-runs/20260518-050009-moe-policy-scan`: no built-in MoE fallback switch exposed a new generation win at `ctx=32768`, 128 generated tokens.
  - Base: `43.07` gen t/s.
  - `DS4_CUDA_MOE_NO_DECODE_LUT_GATE=1`: `32.91` gen t/s. The decode LUT gate path is critical.
  - `DS4_CUDA_MOE_NO_DECODE_LUT_H16=1`: `41.86` gen t/s. The half-warp LUT path is a real but smaller win.
  - `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1`: `42.98` gen t/s. Direct down-sum6 is roughly neutral in this short screen, not an obvious missing win.
  - Decode-only MoE profile from the same run, split to `tokens=1`: 2752 routed-MoE calls averaged `0.004 ms` xq, `0.057 ms` gate/up, `0.004 ms` mid quant, `0.020 ms` down, `0.086 ms` total per layer call. Across 43 layers this is about `3.7 ms/token`, with gate/up the larger MoE bucket.
  - Conclusion: current MoE decode fast paths are carrying throughput. Future MoE work should target a concrete new gate/up or direct-down kernel idea, not fallback toggles.
- Forced ordered one-token f16 matmul on post-topk8704 `main`, run `~/ds4/codex-runs/20260518-002007-f16-ordered-policy-ab`: confirms the Blackwell default should stay generic.
  - Base: average `42.64` gen t/s over 3 runs.
  - `DS4_CUDA_USE_ORDERED_F16_MATMUL=1`: average `40.78` gen t/s over 3 runs.
  - Conclusion: even though generic f16 matvecs remain a large Nsight bucket, the old ordered path is worse on `sm_120`.
- Opt-in single-token indexed attention through grouped heads8, branch `codex/indexed-attn-single-token-heads8`, run `~/ds4/codex-runs/20260518-003203-indexed-single-heads8-ab`: build passed, but generation regressed.
  - Base: average `42.64` gen t/s over 3 runs.
  - `DS4_CUDA_INDEXED_SINGLE_HEADS8=1`: average `40.83` gen t/s over 3 runs.
  - Conclusion: the existing grouped indexed-attention kernel is not a drop-in decode win. Its lower block count loses enough parallelism that the generic per-head path remains faster for single-token decode.
- Broader single-token indexed attention grouping probe, branch `codex/indexed-attn-single-groups`, run `~/ds4/codex-runs/20260518-041314-indexed-attn-single-groups`: build and smoke passed, but all grouped variants regressed at `ctx=32768`, 512 generated tokens.
  - Base: average `42.67` gen t/s over 3 runs.
  - `DS4_CUDA_INDEXED_SINGLE_GROUP4=1`: average `36.02` gen t/s.
  - `DS4_CUDA_INDEXED_SINGLE_GROUP8=1`: average `39.90` gen t/s.
  - `DS4_CUDA_INDEXED_SINGLE_GROUP16=1`: average `40.82` gen t/s.
  - Conclusion: reducing KV reloads by grouping heads is not enough; the lost block-level parallelism dominates. Future indexed-attention work needs a new one-token design, not a grouped prefill/batch route.
- Decode selected-top-k sorting probe, branch `codex/indexed-decode-topk-sort`, run `~/ds4/codex-runs/20260518-044728-indexed-decode-topk-sort`: build and smoke passed, but opt-in sorting of the 512 selected compressed rows during decode did not help.
  - Base: average `42.69` gen t/s over 3 runs.
  - `DS4_CUDA_INDEXED_SORT_DECODE_TOPK=1`: average `42.57` gen t/s.
  - Conclusion: local row-order cleanup inside the existing decode top-k path is not the missing win. The next top-k/indexer work should be a correctness-preserving selection harness or a different algorithmic path, not a small decode sorting pass.
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
- Visible top-k direct-copy fast path inside one-token indexed attention, branch `codex/indexed-attn-visible-topk-fastpath`, run `~/ds4/codex-runs/20260517-231626-indexed-attn-visible-ab`: build passed, but generation regressed badly.
  - Enabled: `37.34`, `37.27` gen t/s.
  - Disabled with `DS4_CUDA_NO_INDEXED_ATTENTION_VISIBLE_FASTPATH=1`: `42.72`, `42.72` gen t/s.
  - Conclusion: do not bypass the current shared-memory atomic/filter setup. It likely preserves useful ordering/filter behavior or better scheduling despite looking wasteful.

## Current diagnosis

The initial WSL2 path is not a clean baseline. It combined WSL2 with an earlier draft of the CUDA backend, while the bare-metal Ubuntu measurements use later upstream CUDA changes. Treat the WSL2 result as historical evidence that the first setup was bad, not as proof that WSL2 alone caused the catastrophic behavior. Current bare-metal Ubuntu generation is roughly 38-43 t/s depending on context.

Remaining decode performance is not dominated by one obvious allocation fallback. The q8 f16 cache pressure is real, but lowering the reserve is not a valid correctness-preserving mitigation on this model/hardware combination. Leave the reserve unset for correctness-gated runs. There is not yet a correctness-safe prefill-recovery policy.

Fallback audit at `ctx=32768`:

- Run: `~/ds4/codex-runs/20260515-040252-fallback-audit`
- Result: `506.31` prefill t/s, `40.76` gen t/s.
- Filtered stderr showed no hard CUDA allocation failures. The recurring fallback was q8 f16 cache budget exhaustion only: 1056 budget messages after 192 q8 f16 cache entries and full model tensor-span caching.
- Conclusion from this pre-`c9dd949` run: upstream's allocation-fallback hypothesis was directionally plausible for q8 f16 cache pressure, but the evidence did not show an unknown buffer allocation failure path. Later upstream fixes changed the long-context result, but low-reserve settings still need full-test validation.

Q8 decode label profile:

- Branch: `codex/q8-decode-stats`
- Run: `~/ds4/codex-runs/20260518-051804-q8-decode-stats`
- Method: opt-in `DS4_CUDA_Q8_DECODE_STATS=1` CUDA-event timing around q8 decode callsites. This intentionally synchronizes and slows the run, so use the totals for attribution, not throughput.
- Bench under instrumentation: `ctx=32768`, 64 generated tokens, `508.36` prefill t/s, `39.16` gen t/s.
- Top q8 decode buckets over the instrumented run:
  - `q8_hc_expand` `8192->4096`: `81.931 ms` total, `0.029771 ms` average, 2752 calls.
  - `attn_output_a` `4096->8192`: `79.211 ms` total, `0.028783 ms` average, 2752 calls.
  - `q8_0` `1024->32768`: `75.477 ms` total, `0.027426 ms` average, 2752 calls.
  - `q8_pair` `4096->4096`: `45.344 ms` total, `0.016477 ms` average, 2752 calls.
  - `shared_down_hc_expand` `2048->4096`: `34.351 ms` total, `0.012482 ms` average, 2752 calls.
  - `q8_pair` `4096->1536`: `29.543 ms` total, `0.010735 ms` average, 2752 calls.
  - `q8_0` `4096->129280`: `22.677 ms` total, `0.348881 ms` average, 65 calls.
- Interpretation: the largest named q8 bucket is attention output, split almost evenly between `attn_output_a` and the fused `attn_output_b`/HC-expand path. The q-b projection is the next single large bucket. Shared expert q8 is material but smaller. This narrows q8 work toward attention-output and q-b paths rather than generic q8 toggles.

Reserve correctness boundary:

- Run: `~/ds4/codex-runs/20260515-043330-reserve-correctness-boundary`
- Default reserve: `./ds4_test --long-context` OK in this run, first q8 f16 cache attempt exhausted with `cached=0.00 GiB`, `reserve=4.75 GiB`.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096`: failed long-context with Alice wrong-assignment (`got 50 expected 52`).
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=3072`: failed the same way.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=2816`: failed after q8 f16 cache reached about `0.14 GiB`.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=2560`, `2048`, and `128`: all failed the same way.
- Conclusion from this pre-`c9dd949` boundary run: do not set `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` for correctness-gated runs. After the later upstream sync, 128 MB passes `long-context`, but full `make test` still shows that low reserves are not deployable as global policy.

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
- Current interpretation after the later `c9dd949` sync: the Alice long-context check was fixed upstream, so these runs are historical evidence rather than the active gate. Do not promote `DS4_CUDA_NO_Q8_F16_CACHE=1` or a code revert as a fix. The active correctness target is now the remaining `logprob-vectors` failure shape.
- Debug run: `~/ds4/codex-runs/20260517-045021-longctx-debug` using branch `codex/long-context-debug`.
  - Test-only envs: `DS4_TEST_LONG_DEBUG=1` and `DS4_TEST_LONG_OUTPUT_FILE=$RUN_DIR/long-context-output.txt`.
  - Output shape was otherwise correct and compact: `Bob=34`, `Alice=50`, `Clara=71`, ..., `Priya=97`.
  - This narrows the Alice failure: the format and all other assignments are correct, but Alice's value is decoded as `50` instead of `52`.
- Token trace run: `~/ds4/codex-runs/20260517-045637-longctx-token-trace` using `DS4_TEST_LONG_TOKEN_TRACE=1`.
  - At generation step 7, immediately after `Alice=`, the top logits were near-tied: token `50` at `40.2729`, token `52` at `40.0574`.
  - This makes the failure look like a small accumulated numeric/path difference that flips a near-tie, not a gross output-format bug.
  - Next useful probe is a CPU/GPU or alternate-kernel logit comparison for that exact prefix/step, not another whole-run cache toggle.

Upstream status as of 2026-05-17: fork `main` was rebased onto upstream `c9dd949` and force-pushed to `origin/main`.

Interesting upstream changes from the sync:

- `5bc1e6d` (`Apply Flash graph correctness fixes`) partially merges official DeepSeek V4 Flash graph fixes: shared expert SwiGLU limit clamp, ratio-4 indexer Hadamard rotation, and FP4 activation-simulation round trips before top-k scoring. This is directly relevant to CUDA correctness.
- `c9dd949` (`cuda: fix compressed prefill RoPE positions`) fixes compressed prefill/replay RoPE positions with `pos0 + t * ratio`. This was the most likely fix for the Alice long-context near-tie/wrong-assignment behavior.
- `04b6fda` (`cuda: use managed KV cache for huge contexts`) is scoped to very large KV/context allocations and logs when active. It should not affect the current 32k/65k benchmark phase. Treat it as a separate long-context stability item because managed KV can trade performance for allocation survivability.
- `ds4-eval` received benchmark/reporting/control improvements and new COMPSEC eval cases. This is useful for correctness instrumentation around the Alice long-context failure.
- Server/runtime work landed around CORS, cache reporting, working-directory control, tool/reasoning replay, and min-p default sampling. Useful for deployment polish, not raw RTX decode throughput.
- No upstream commit in this sync appears to implement CUDA Graph replay or a major RTX Pro 6000 decode-throughput fix.

Post-sync verification on the RTX Pro 6000 host:

- Run directory: `~/ds4/codex-runs/20260517-upstream-sync`
- Fresh worktree: `~/ds4/codex-worktrees/upstream-sync-20260517-184936`
- `make cuda CUDA_ARCH=sm_120 -j$(nproc)`: OK
- `./ds4_test --long-context`: OK with the default reserve.
- `make test CUDA_ARCH=sm_120`: still fails in `logprob-vectors / long_memory_archive` with 7 assertion failures; `long-context`, `tool-call-quality`, `metal-kernels`, and `server` pass.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 ./ds4_test --long-context`: OK.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 make test CUDA_ARCH=sm_120`: fails in `logprob-vectors` with CUDA OOM during session creation (`2` failures). This is a different failure shape from default.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512 make test CUDA_ARCH=sm_120`: avoids OOM but fails `logprob-vectors / short_code_completion` with `1` selected-token mismatch. This is also a different failure shape from default.
- `DS4_CUDA_NO_Q8_F16_CACHE=1 make test CUDA_ARCH=sm_120`: same default `logprob-vectors / long_memory_archive` 7-failure shape. This means the remaining default-reserve parity issue is not solely caused by q8->fp16 cache reuse.
- Smoke bench with `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`, `ctx=4096`, `gen_tokens=256`: `534.17` prefill t/s, `45.82` gen t/s.
- Minimal upstream-facing repro note: `docs/upstream-logprob-vector-repro.md`.

At the official benchmark shape, RTX Pro 6000 is already much faster than DGX Spark in absolute terms, but not in proportion to the hardware delta. After the adopted top-k 8704 patch, `ctx=32768` reaches about 42.6-42.8 generation t/s versus the upstream DGX Spark row at 13.75 t/s: about 3.1x, still below the broad 4x compute-ratio target band.

## Upstream PR watchlist

As of 2026-05-15, upstream has many open PRs. Relevant ones for this fork:

- [#121 cuda: skip ordered f16 matmul on Blackwell](https://github.com/antirez/ds4/pull/121): aligns with our independently validated Blackwell f16 default patch. Track for upstream merge/rebase cleanup.
- [#145 cuda: add launch-bounded tile8 MoE down path](https://github.com/antirez/ds4/pull/145): opt-in `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN` path layered on #121. Static inspection shows this is gated through the atomic tiled down route (`use_atomic_down`, normally prefill/batch) while single-token decode uses the direct `sum6` down path. Treat this as a possible prefill/batch throughput idea, not a primary generation-speed target.
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

1. **Selective Q4 decode cache port**
   - `ngc-shj/perf-clean` proves Q4 decode can move generation on the RTX Pro 6000: about `47.9` gen t/s at `ctx=32768` vs the current `42-43` t/s band.
   - Do not port the branch wholesale. It is large, includes server batching and CUDA Graph scaffolding, and has warning-prone speculative-prefix code.
   - Do not use the full preload policy directly. Full Q4 preload (`3.55 GiB`) plus q8->f16 cache does not fit cleanly on this 96 GB card; lowering the q8->f16 reserve to `768` or `512` MB OOMs, while higher reserves leave q8->f16 effectively empty and lose prefill.
   - First port candidate: Q4 cache for F16-derived decode matvecs. The component scan drops from `47.61` to `40.82` gen t/s when F16-derived Q4 is disabled, making this the largest contributor.
   - May 18 status: `codex/q4-f16-decode` implements this candidate behind `DS4_CUDA_Q4_DECODE=1`. Full F16-Q4 reaches `47.50` gen t/s at `ctx=32768`, but the best balanced subset is currently `DS4_CUDA_Q4_F16_FILTER=hc_` with `45.59` gen t/s and much lower prefill cost.
   - Second port candidate result: generic single Q8->Q4 dispatch works technically, but is not a useful preset. The best narrow case (`attn_q_b`) moved generation from `48.73` to `49.97` t/s on top of F16 `hc_`, while prefill fell from `564.72` to `512.78` t/s.
   - Third port candidate result: `attn_output_a` and fused `attn_output_b`/HC-expand Q4 also work technically, but show the same tradeoff. Best combined short-context result reached `50.88` gen t/s, but prefill fell to `468.07` t/s.
   - After-prefill memory policy result: delaying selected Q8->Q4 attention-output preload until after prefill preserves almost all F16 `hc_` prefill while keeping the decode gain. At `ctx=32768`, F16 `hc_` was `493.39` prefill / `45.43` gen, while after-prefill Q8 `attn_output_a,b` was `493.13` prefill / `47.26` gen.
   - Q4 pair is still effectively neutral in the external component benchmark and has not been ported locally.
   - Keep everything behind `DS4_CUDA_Q4_DECODE=1` and add narrower disable switches so quality/performance can be bisected by tensor family.
   - Current experimental adoption candidate is now F16 `hc_` plus after-prefill Q8 `attn_output_a,b`. Keep it opt-in until phase-state behavior is safe for long-lived server use.

2. **Logprob-vector parity probe**
   - The Alice long-context failure is resolved on the post-`c9dd949` sync, but `make test CUDA_ARCH=sm_120` still fails `logprob-vectors / long_memory_archive` under the default reserve.
   - Low-reserve q8 f16 cache settings now pass `./ds4_test --long-context`, but they change the full-test failure shape (`128` MB OOMs in logprob vectors; `512` MB flips a `short_code_completion` token).
   - Next correctness work should instrument the logprob-vector cases, especially cache/no-cache and tensor-label differences, rather than continuing to use Alice as the primary gate.

3. **CUDA Graph replay feasibility**
   - LnaLang4U reports a large CUDA Graphs ON/OFF gap on RTX Pro 6000-class hardware, and DS4 still launches many small decode kernels per token.
   - Do not assume DS4 can get the same gain as SGLang/PyTorch: DS4 is native C/CUDA and already avoids a lot of framework overhead.
   - First step: make a minimal decode capture/replay feasibility probe for a fixed benchmark shape, or prove exactly which token-varying state prevents capture.
   - Success criterion is practical: a repeatable generation gain at `ctx=32768`, not a synthetic launch-count reduction.
   - Probe run: `~/ds4/codex-runs/20260516-042711-cuda-graph-phase-probe`.
   - Single reusable `cudaGraphExec` failed quickly in earlier testing. A slot cache with 4 or 8 phase slots replayed 127 tokens, then failed on token 128.
   - A 128-slot update cache completed `ctx=4096, gen=256`: `instantiated=128 updated=128 launched=256 failed=0`.
   - It is not a speed patch as written: same-length control was 46.18 tok/s, while the 128-slot capture/update probe was 42.00 tok/s because it still captures/updates every token.
   - Post-topk8704 Nsight captured about `4.0M` kernel launches in a delayed decode window. The profile is still dominated by real GPU work, but launch count is high enough that CUDA Graph replay remains a serious track.
   - Useful conclusion: CUDA Graph replay remains plausible only with a proper 128-phase graph cache plus direct node parameter updates/static device parameter buffers. The capture/update probe itself should not be merged, and the next graph attempt should not recapture/update every token.

4. **Benchmark calibration against external references**
   - Run or reproduce an apples-to-apples DS4 server benchmark where possible, especially `llama-benchy`-style steady-state `tg`, because DS4's average generation metric and forum/README metrics are not always the same.
   - Keep the 2-card, roughly 4-bit RTX Pro 6000 reports as perspective, not as a direct target for the 1-card roughly 2-bit DS4 path.
   - Maintain the target bands: below 50 t/s is not enough; 55-65 t/s is the first acceptable generation band; 70-80 t/s is the strong target.

5. **Selective q8 f16 cache recovery**
   - Prefill is important, and the low-reserve policy showed the upside. The problem is full-test behavior and VRAM headroom, not the old Alice long-context output.
   - Re-test q8 f16 caching by tensor group/label rather than blunt reserve size. The current blunt `128` MB policy is good for benchmark prefill but too aggressive for global use.
   - Goal: recover safe prefill wins while preserving the default logprob-vector failure shape, with no OOMs.

6. **Indexed attention / indexer redesign**
   - Highest combined decode target after the post-topk8704 Nsight profile: indexed attention, dense attention, top-k8704, and indexer score/direct kernels.
   - Built-in fallback toggles and existing grouped-head single-token routing did not help. The May 18 grouped-head probes were all slower, with the best grouped variant at `40.82` vs `42.67` gen t/s.
   - Decode-only sorting of the selected top-k rows was also slightly slower (`42.57` vs `42.69` gen t/s), so do not chase local row-order cleanup as a standalone patch.
   - Post-`c9dd949`, the narrow CUB top-k fast path for `8192 < n_comp <= 8704` is adopted after preserving the current `make test` failure shape.
   - Next useful work here is either a correctness-preserving harness for larger `n_comp` top-k ranges or a purpose-built one-token indexed-attention kernel. Do not reuse the existing grouped prefill/batch kernel as-is; the A/B says it gives up too much parallelism.

7. **MoE decode kernel inspection**
   - MoE decode LUT and related routed/shared kernels remain large GPU-time buckets.
   - PR #145's tile8 down rowspan path is not a decode target because one-token decode takes the direct `sum6` down route.
   - May 18 switch scan confirmed the current decode LUT gate path is critical (`32.91` gen t/s without it vs `43.07` base), the half-warp LUT path matters (`41.86` without it), and direct down-sum6 is roughly neutral in a short screen (`42.98` without it).
   - Decode-only MoE profile puts routed MoE at about `3.7 ms/token`, mostly gate/up. That is worth tracking, but it is not the whole generation gap.
   - Next concrete work here requires a new kernel idea for gate/up or direct down; more fallback toggles are low value.
   - Keep only changes that improve `ctx=32768`, 512-token generation by a repeatable margin.

8. **q8 decode matvec path**
   - Multiple q8 buckets together are material, and the fp16 cache budget issue is not safely mitigated by reserve tuning.
   - May 18 q8 label profiling shows attention-output q8 is the largest named bucket: `attn_output_a` plus fused `attn_output_b`/HC-expand together were about `161 ms` over a 64-token instrumented decode. The q-b projection `1024->32768` was next at about `75 ms`.
   - First concrete target should be attention-output q8 or q-b, but only with a specific kernel/cache idea. Generic q8 fallback toggles have already been worse.
   - Avoid chasing more q8->fp16 cache policy unless the exactness issue is understood; low reserve values now pass `long-context` after `c9dd949`, but they still perturb full `logprob-vectors` behavior.

9. **Huge-context stability and disk-KV pass**
   - Defer until upstream-benchmark generation speed and the remaining logprob-vector exactness issue are better understood.
   - Revisit upstream managed-KV-cache changes separately; they may help huge contexts but can trade off discrete-GPU performance.
   - If KV offload becomes necessary for maximum context, measure the MP700 PRO XT path before considering Optane.

Acceptance rule: keep a performance patch only if it builds cleanly, passes smoke, preserves the known correctness-gate shape, and shows a repeatable generation gain at `ctx=32768` with 512 generated tokens. A gain below roughly 2% should be treated as documentation-only evidence unless it unlocks a larger follow-up.

## Correctness gate

Token-byte parity vs the official DeepSeek V4 Flash API remains the hard gate. Existing known CUDA failures must be tracked separately from new regressions:

```bash
make test CUDA_ARCH=sm_120
```

Known synced-main CUDA failure shape from `~/ds4/codex-runs/20260517-upstream-sync` after rebasing onto upstream `c9dd949`:

- `long-context`: OK.
- `logprob-vectors`: `long_memory_archive` selected-token mismatches and official-top-token-missing checks, `7` assertion failures total.
- `tool-call-quality`, `metal-kernels`, and `server`: OK.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`: `long-context` OK, but full `make test` OOMs during `logprob-vectors`.
- `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512`: avoids OOM, but changes the logprob-vector failure shape to a `short_code_completion` selected-token mismatch.
- `DS4_CUDA_NO_Q8_F16_CACHE=1`: same default `long_memory_archive` 7-failure shape.

Current Q4 branch default gate, commit `ee8d246`, run log `~/ds4/codex-runs/20260518-084332-q8-q4-after-prefill-repeat/make-test-default.out`: default `make test CUDA_ARCH=sm_120` preserves the same known shape. `long-context`, `tool-call-quality`, `metal-kernels`, and `server` pass; `logprob-vectors / long_memory_archive` reports the same 7 assertion failures.

For now, a candidate patch is acceptable only if:

- It builds for `CUDA_ARCH=sm_120`.
- Smoke generation works.
- `make test CUDA_ARCH=sm_120` does not introduce a new failure shape beyond the known default-reserve `long_memory_archive` logprob-vector failures.
- A/B decode benchmarks show a repeatable gain, not a single lucky run.
