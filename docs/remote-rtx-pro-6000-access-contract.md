# Remote RTX Pro 6000 access contract

This document defines how Codex may directly use the bare-metal Ubuntu DS4 box without requiring command copy/paste through chat.

This is a home-lab science workflow. The goal is faster DS4 experimentation while keeping the machine observable, reversible, and boringly safe.

## Host

- Machine: `bst-originpc-ai`
- SSH target: `bst@bst-originpc-ai.tail520464.ts.net`
- Access path: SSH over Tailscale only
- DS4 repo path: `~/ds4`
- Shared live session: `tmux` session named `codex-ds4`
- Log root: `~/ds4/codex-runs/`
- Current run pointer: `~/ds4/codex-runs/latest`

## How to watch

Attach read-only to the shared session:

```bash
ssh bst@bst-originpc-ai.tail520464.ts.net
tmux attach -t codex-ds4 -r
```

Detach from `tmux` with:

```text
Ctrl-b d
```

Watch the latest log directory:

```bash
tail -f ~/ds4/codex-runs/latest/run.log
```

For detailed command output, each experiment should also write stdout/stderr files in the same run directory.

## Default allowed scope

Codex may run these directly, without asking for each individual command:

- Read-only status checks: `git status`, `git log`, `nvidia-smi`, `pgrep`, `ls`, `du`, `cat`, `grep`, `rg`.
- DS4 repo operations inside `~/ds4`: build, test, benchmark, profile, and collect logs.
- Non-destructive `git fetch` / branch inspection for DS4.
- Creating timestamped log directories under `~/ds4/codex-runs/`.
- Managing the `codex-ds4` tmux session used for live visibility.
- Short DS4 experiments that are expected to complete in minutes and use normal inference load.

## Approval required first

Codex must ask before doing any of the following:

- `sudo`, `apt`, driver, kernel, CUDA, systemd, firewall, SSH, Tailscale, network, or bootloader changes.
- Killing processes, including stale DS4 jobs.
- Rebooting, shutting down, suspending, or changing GPU persistence / ECC / power settings.
- Destructive git operations such as `git reset --hard`, `git clean`, deleting branches/tags, or force-pushing unless explicitly requested.
- Long-running stress, thermal, or overnight jobs.
- Anything outside `~/ds4` unless clearly read-only and needed for diagnosis.
- Communicating externally on the user's behalf through email, social media, chat apps, GitHub comments/issues, or similar channels.

## Run discipline

Before a DS4 experiment, Codex should record:

- UTC/local timestamp.
- Git branch and commit.
- Important environment variables.
- Exact command being run.
- GPU status before the run.

During a run:

- Prefer running commands inside the `codex-ds4` tmux session so they are visible live.
- Also write logs under a timestamped directory below `~/ds4/codex-runs/`.
- Keep `~/ds4/codex-runs/latest` pointing at the most recent run directory.

After a run:

- Summarize the result in chat.
- Mention the log path.
- Do not delete logs unless asked.
- If a run fails, preserve stderr/stdout for diagnosis.

## Safety checks

Before starting GPU work, Codex should check:

```bash
nvidia-smi
pgrep -af 'ds4|nsys'
```

If another DS4 or profiling job is already active, Codex should not start a competing GPU job. Wait, report the conflict, or ask before stopping anything.

## Baseline DS4 environment

Current correctness-safe RTX Pro 6000 defaults:

```bash
# Do not set DS4_CUDA_Q8_F16_CACHE_RESERVE_MB for correctness-gated runs.
# Low-reserve q8->f16 cache policies are historical performance probes only.
```

Build target:

```bash
make cuda CUDA_ARCH=sm_120 -j$(nproc)
```

Token-byte parity remains the hard correctness gate:

```bash
make test CUDA_ARCH=sm_120
```

Known current caveat: the CUDA long-context correctness path is marginal and includes the Alice wrong-assignment investigation tracked in `docs/ds4-discrete-gpu-experiment.md`; this should be treated as a correctness investigation item, not ignored.

## Current setup state

As of 2026-05-15, the shared tmux/log setup exists:

- `tmux` session: `codex-ds4`
- Latest log symlink: `~/ds4/codex-runs/latest`

If the session disappears after reboot, recreate it with:

```bash
cd ~/ds4
RUN_DIR="$HOME/ds4/codex-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$HOME/ds4/codex-runs/latest"
tmux new-session -d -s codex-ds4 -c "$HOME/ds4"
tmux set-option -t codex-ds4 history-limit 50000
tmux pipe-pane -t codex-ds4 -o "cat >> $RUN_DIR/tmux.log"
printf 'codex-ds4 session ready\nrun_dir=%s\n' "$RUN_DIR" | tee "$RUN_DIR/run.log"
```
