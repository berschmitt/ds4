# Upstream CUDA logprob-vector repro note

This is a concise repro packet for the remaining CUDA correctness issue on the RTX Pro 6000 host. It is not a local fix plan; correctness parity belongs upstream unless we find a very small, obvious local regression.

## Environment

- Host: `bst-originpc-ai`
- OS: Ubuntu 24.04 LTS bare metal
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, `sm_120`, 96 GB VRAM
- Driver: 595.71.05
- CUDA runtime reported by `nvidia-smi`: 13.2
- CUDA toolkit used by `nvcc`: 13.0
- Build command: `make cuda CUDA_ARCH=sm_120 -j$(nproc)`
- Fork revision tested: `3cee92d`
- Upstream base: `c9dd949` (`cuda: fix compressed prefill RoPE positions`)
- Run directory: `~/ds4/codex-runs/20260517-post-c9dd949-bench`
- Worktree: `~/ds4/codex-worktrees/post-c9dd949-bench-20260517-191237`

## Correctness Results

Default reserve:

```bash
make test CUDA_ARCH=sm_120
```

Observed:

- `long-context`: OK
- `tool-call-quality`: OK
- `metal-kernels`: OK
- `server`: OK
- `logprob-vectors`: fails only on `long_memory_archive`
- Failure shape: 7 assertions, selected-token mismatches at steps 0-3 and official-top-token-missing checks at steps 1-3

With q8->fp16 cache disabled:

```bash
DS4_CUDA_NO_Q8_F16_CACHE=1 make test CUDA_ARCH=sm_120
```

Observed:

- Same default `logprob-vectors / long_memory_archive` 7-failure shape.
- This suggests the remaining default-reserve parity issue is not solely caused by q8->fp16 cache reuse.

With low q8->fp16 cache reserve:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 make test CUDA_ARCH=sm_120
```

Observed:

- `long-context`: OK
- `logprob-vectors`: CUDA OOM during session creation, 2 failures
- This is a different failure shape from default.

With a less aggressive low reserve:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512 make test CUDA_ARCH=sm_120
```

Observed:

- Avoids OOM.
- `logprob-vectors` changes failure shape to one `short_code_completion` selected-token mismatch.

## Performance Context

Fresh post-`c9dd949` official-shaped sweeps were run with:

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

Default reserve:

- `ctx=2048`: 370.82 prefill t/s, 46.75 gen t/s
- `ctx=32768`: 342.81 prefill t/s, 40.61 gen t/s
- `ctx=65536`: 328.06 prefill t/s, 37.74 gen t/s

Per-run tuned reserve:

```bash
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128 \
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1 \
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

- `ctx=2048`: 488.87 prefill t/s, 46.58 gen t/s
- `ctx=32768`: 459.44 prefill t/s, 40.67 gen t/s
- `ctx=65536`: 438.34 prefill t/s, 37.81 gen t/s

Interpretation:

- Low-reserve q8->fp16 cache tuning is a clear prefill lever.
- It does not materially improve generation.
- It should remain a per-run benchmark knob until the logprob-vector failure shape and VRAM headroom behavior are understood.
