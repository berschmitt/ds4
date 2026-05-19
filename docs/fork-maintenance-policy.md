# Fork maintenance policy

This fork should stay close to upstream DS4 unless a local change has a clear, measured reason to exist.

The default posture is upstream-first:

- Track `antirez/ds4` regularly.
- Prefer upstream solutions over local variants when they solve the same problem.
- Keep local changes small, isolated, and easy to replay.
- Keep negative results documented, but do not carry failed experiments as active code.
- Treat correctness as the hard gate. A faster path that changes token behavior is not a deployable optimization.

## Local patch tiers

Use these tiers when deciding what belongs on the active fork branch.

| Tier | Meaning | Where it lives |
| --- | --- | --- |
| Adopted patch | Correctness-safe, measured, still useful after rebase. | Active branch, as a small commit. |
| Candidate patch | Promising but needs retest against current upstream. | Separate topic branch; cherry-pick only for A/B. |
| Local operations doc | Specific to this machine or workflow. | `docs/`, clearly marked local. |
| Historical artifact | Useful measurement or rejected result. | Experiment doc or archived branch/tag, not active runtime policy. |
| Dropped patch | Too small, too risky, or superseded by upstream. | Do not replay during sync. |

## Current patch ledger

| Change | Current decision | Why |
| --- | --- | --- |
| Blackwell one-token f16 matvec default (`a486c90`) | Keep concept, prefer upstream PR #121 shape | Clear RTX Pro 6000 generation gain. May 18 retest after topk8704 still shows forced ordered f16 is slower (`40.78` vs `42.64` gen t/s). Upstream has the same idea in `upstream-pr/121`, so avoid carrying a divergent interface long term. |
| Half-warp MoE decode LUT (`8258aeb`, `da9a129`) | Candidate | Measured repeatable but modest decode gain. Retest after rebasing onto current upstream before keeping. |
| Pair decode q/kv q8 matvec (`1d35fbb`) | Drop unless retest shows larger value | Small decode gain, touches shared graph/API surface. Not worth carrying by default. |
| Decode Q head RMS norm + RoPE fusion (`5347041`) | Drop unless part of a larger fusion plan | Small decode gain, touches shared graph/API surface. Keep result documented, but do not replay automatically. |
| Shared gate/up SwiGLU fusion probe | Historical only | May 19 A/B on the Q4 branch moved generation only about `0.1-0.2` t/s and the fastest prototype skipped auxiliary writes. Do not replay; future MoE work needs a materially different kernel idea. |
| CUB top-k 8704 fast path (`94b6d70`) | Adopted patch | Post-`c9dd949` retest shows a repeatable `ctx=32768`, 512-token generation gain (`42.79` / `42.68` t/s vs `40.24` / `40.24` disabled), passes `long-context`, and preserves the known default `make test` failure shape. |
| Low-reserve q8->f16 cache policy (`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=128`) | Per-run benchmark knob only | After upstream `c9dd949`, 128 MB passes `./ds4_test --long-context`, but full `make test` OOMs in `logprob-vectors`; 512 MB avoids OOM but changes the logprob failure shape. Do not install globally yet. |
| RTX Pro 6000 benchmark CSVs | Keep as historical artifacts | Useful for tracking what happened, but not a runtime policy. |
| Bare-metal deployment / remote access docs | Keep local | Operationally important for this home-lab host; not intended as upstreamable code. |
| Broader rejected top-k/indexer shortcuts | Historical only | Narrow `8704` is now adopted after post-`c9dd949` retest. Broader/tail/chunk variants still need a non-perturbing comparison harness before adoption. |
| Existing grouped-head indexed attention for single-token decode | Drop | May 18 opt-in retest on branch `codex/indexed-attn-single-token-heads8` regressed generation (`40.83` vs `42.64` gen t/s). Do not replay this patch; a future attention win needs a purpose-built one-token kernel or a broader redesign. |
| CUDA Graph capture/update probes | Historical only | May 18 stability probes prove capture/replay is API-viable, but the decode topology changes across context/frontier state and per-token capture/update is slower. Do not carry probe code as an active patch; revisit only after staticizing the decode launch topology. |

## Upstream sync workflow

Use this workflow whenever upstream has meaningful changes:

1. Record current state.
   - `git status --short`
   - `git rev-parse --short HEAD`
   - current benchmark/correctness caveats in `docs/ds4-discrete-gpu-experiment.md`

2. Fetch and inspect upstream.
   - `git fetch upstream --prune`
   - `git log --oneline --graph HEAD..upstream/main`
   - `git diff --stat HEAD..upstream/main`

3. Protect the current branch before reshaping it.
   - Create an archive branch or tag before a major rebase.
   - Do not delete old experiment branches until their result is documented.

4. Start from upstream when the divergence is large.
   - Prefer a fresh branch from `upstream/main`.
   - Replay only adopted or actively retested candidate patches.
   - Use small cherry-picks or equivalent manual patches; avoid replaying whole historical doc/benchmark commits unless needed.

5. Validate in this order.
   - Run the post-sync validation batch on plain upstream.
   - Replay adopted/candidate local patches.
   - Run the same post-sync validation batch again.
   - Compare correctness and performance before deciding which patches remain.
   - Update the patch ledger with keep/drop rationale.

6. Push only after the branch tells a clean story.
   - Active branch should contain upstream plus a minimal local delta.
   - Historical findings belong in docs or archive refs, not as confusing runtime defaults.

## Acceptance rule for performance patches

A performance patch is worth carrying only if it meets all of these:

- It builds cleanly on `sm_120`.
- It does not introduce a new correctness failure shape.
- It has a fallback switch while under evaluation.
- It shows a repeatable gain at the relevant benchmark shape, especially `ctx=32768` with 512 generated tokens.
- It is large enough to justify rebase cost. Below roughly 2% generation gain, treat it as evidence rather than default code unless it unlocks a larger change.

## Post-sync validation batch

After each meaningful update from `antirez/ds4`, validate two states:

1. **Plain upstream on our RTX Pro 6000 host.**
2. **Plain upstream plus the local patches we intend to carry.**

This separates upstream regressions from fork regressions. If plain upstream fails, document it as upstream/current behavior before blaming a local patch. If only the patched state fails, the local patch is not adopted.

Run the batch inside the visible `codex-ds4` tmux session and write logs under `~/ds4/codex-runs/<timestamp>-post-sync-*`.

Minimum correctness batch:

```bash
make cuda CUDA_ARCH=sm_120 -j$(nproc)
./ds4 -p "Say hi in three words." --nothink -n 16 --ctx 4096
make test CUDA_ARCH=sm_120
```

If `make test` has known upstream failures, record the exact failure shape. A local patch is acceptable only if it does not add a new failure shape.

Minimum performance batch:

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 4096 --step-incr 2048 --gen-tokens 512

./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 --ctx-max 32768 --step-incr 2048 --gen-tokens 512
```

Run the full official-shaped sweep when any of these are true:

- Before pushing a refreshed `main`.
- After a CUDA, KV-cache, attention/indexer, MoE, scheduler, or server-generation change.
- When the minimum batch shows a meaningful performance shift.

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

Performance deviation rule:

- Treat single-run noise below about 2% as informational.
- Re-run if generation drops by 2% or more at `ctx=32768`.
- Block promotion and investigate if a repeated run drops generation by 3% or more, or if prefill drops by 5% or more without an understood reason.
- Always record both the upstream baseline and patched result in the experiment doc or a post-sync run note.

## Practical branch model

Keep `main` boring:

- `main`: current usable fork, close to upstream.
- `archive/*`: frozen historical branches before major resets or rebases.
- `codex/*`: focused experiment branches.
- `upstream-pr/*`: fetched upstream PR branches used for comparison.

When in doubt, archive first, then simplify.
