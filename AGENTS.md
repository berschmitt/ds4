# DS4 fork — Codex context

Home-lab science / exploration fork of [DS4](https://github.com/antirez/ds4) (DeepSeek V4 Flash inference engine). Target: discrete RTX Pro 6000 (96 GB GDDR7, Blackwell `sm_120`) on bare-metal Ubuntu 24.04. WSL2 remains historical setup context, not the current benchmark path.

- **Principles & invariants**: [PRINCIPLES.md](PRINCIPLES.md). Token-byte parity vs the official DeepSeek V4 Flash API (`ds4_test --all`) is the hard gate.
- **Active plan**: [docs/ds4-discrete-gpu-experiment.md](docs/ds4-discrete-gpu-experiment.md). This is the current living experiment log, performance target, and next-step plan.
- **Remote execution**: SSH over Tailscale to `bst@bst-originpc-ai.tail520464.ts.net`. Use the visible `tmux` session `codex-ds4` and timestamped logs under `~/ds4/codex-runs/latest`; see [docs/remote-rtx-pro-6000-access-contract.md](docs/remote-rtx-pro-6000-access-contract.md).
- **Auto-loaded memory**: `C:\Users\Bertrand\.Codex\projects\d--Dropbox-Dev-ds4\memory\MEMORY.md` indexes user / feedback / project notes (calibration honesty, no-overkill, diagnose-before-optimize, etc.).
- **Tone**: home-lab exploration, not production. Default minimal. Cite sources for perf claims; measure before changing perf code.
