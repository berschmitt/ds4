# DS4 Fork Principles

Home-lab **science / exploration** fork of [DS4](https://github.com/antirez/ds4) (DeepSeek V4 Flash inference engine). Upstream is alpha-quality, targets Metal + CUDA, and evolves rapidly — we track it and absorb improvements where we can. Merge-back is nice-to-have, not an objective.

Target: x86_64 PC, RTX Pro 6000 (96 GB GDDR7, Blackwell `sm_120`), AMD 9950X3D, 64 GB RAM. Interim runtime WSL2 + Ubuntu 24.04; bare-metal Ubuntu 24.04 follows. Same binary on both is preferred, but OK to diverge if it solves a problem better. Single user, Tailscale-only.

## Don't regress

- **Token-byte parity vs the official DeepSeek V4 Flash API.** `ds4_test --all` is the gate.
- **Upstream-defined formats** (KVC disk header + DSML replay map). Bump the version byte if changes are needed.
- **Operational secrets** (HF tokens, server `--api-key`, agent client credentials) don't appear in logs, traces, or pasted output.

## Working rules

- **Measure before changing perf code.** Diagnostics (`nvidia-smi dmon`, `bandwidthTest`, `nsys`) before edits. Cite sources for perf claims.
- **Minimal diffs vs upstream.** Smaller is easier to rebase. Prefer changing a constant over adding a code path.
- **`/review` cumulative diff before commit.** Annotated tags at meaningful checkpoints; push immediately.
