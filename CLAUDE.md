# DS4 fork — Claude Code context

Home-lab science / exploration fork of [DS4](https://github.com/antirez/ds4) (DeepSeek V4 Flash inference engine). Target: discrete RTX Pro 6000 (96 GB GDDR7, Blackwell `sm_120`) on WSL2 + Ubuntu 24.04, transitioning to bare-metal Ubuntu 24.04 once a PCIe 5 NVMe arrives.

- **Principles & invariants**: [PRINCIPLES.md](PRINCIPLES.md). Token-byte parity vs the official DeepSeek V4 Flash API (`ds4_test --all`) is the hard gate.
- **Active plan**: [docs/ds4-discrete-gpu-experiment.md](docs/ds4-discrete-gpu-experiment.md) (focused, ~60 lines, in-repo). Longer historical version at `C:\Users\Bertrand\.claude\plans\so-i-just-forked-composed-perlis.md`.
- **Remote execution**: `ssh ds4-remote 'cmd'` — alias in `~/.ssh/config` pointing at `bst@bst-originpc:2222` over Tailscale into WSL2. Build, test, bench, and run all happen there; this PC is for editing and driving.
- **Auto-loaded memory**: `C:\Users\Bertrand\.claude\projects\d--Dropbox-Dev-ds4\memory\MEMORY.md` indexes user / feedback / project notes (calibration honesty, no-overkill, diagnose-before-optimize, etc.).
- **Tone**: home-lab exploration, not production. Default minimal. Cite sources for perf claims; measure before changing perf code.
