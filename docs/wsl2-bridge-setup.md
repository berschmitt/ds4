# WSL2 SSH bridge setup — BST-ORIGINPC

Documents how the SSH bridge from `bst-gtr9pro` (dev PC) to `bst-originpc`'s WSL2 instance was configured. Reference if WSL2 is rebuilt, Windows is reinstalled, or the same setup is needed elsewhere.

## What this provides

`ssh ds4-remote` from `bst-gtr9pro` lands directly in `bst@BST-ORIGINPC`'s WSL2 Ubuntu 24.04 shell over Tailscale. ed25519 key auth, no password prompt.

## Architecture

- **bst-gtr9pro** (dev PC, Windows) — SSH client + private key + `ssh config` alias.
- **bst-originpc** (inference PC, Windows) — WSL2 Ubuntu 24.04 running sshd on port 2222. Tailscale on Windows host; WSL2 networking in **mirrored** mode so the host's interfaces (including Tailscale's) are visible inside WSL2.
- **Hyper-V Firewall** — explicit inbound allow rule for port 2222 to WSL2.

## Setup on bst-originpc (the inference PC)

### 1. Enable WSL2 mirrored networking

GUI: Start menu → "WSL Settings" → **Networking** → **Networking mode → Mirrored** → Save.

Equivalent `.wslconfig` at `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

**Mandatory** — the change is only applied on next WSL launch. From PowerShell:

```powershell
wsl --shutdown
```

Then open a fresh WSL session. Verify mirrored is active:

```powershell
Test-NetConnection -ComputerName localhost -Port 2222
```

If `TcpTestSucceeded : True`, mirrored is on and Windows ↔ WSL2 share loopback. If it fails, mirrored isn't active yet (most common cause: GUI saved but WSL not shut down).

### 2. Add Hyper-V firewall rule for inbound port 2222

**Elevation required.** Start menu → right-click "PowerShell" → "Run as administrator". A regular PowerShell returns `Access is denied` (Windows System Error 5).

Single-line command (multi-line backtick form is paste-fragile in plain PowerShell consoles — the leading `PS C:\>` prompt gets pasted alongside continuation lines and confuses the parser):

```powershell
New-NetFirewallHyperVRule -Name "WSL-SSH-2222" -DisplayName "WSL2 SSH (port 2222)" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 2222 -Action Allow
```

The VMCreatorId is the standard WSL identifier (Microsoft-documented). The rule scopes the allow to WSL2 traffic only.

Verify:

```powershell
Get-NetFirewallHyperVRule -Name "WSL-SSH-2222"
```

### 3. Install sshd inside WSL2 on port 2222

Inside the WSL2 shell:

```sh
sudo apt update && sudo apt install -y openssh-server
sudo sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
```

Ubuntu 24.04 uses socket-activated SSH (`ssh.socket`), so the `Port` directive in `sshd_config` is ignored. Override the socket unit:

```sh
sudo mkdir -p /etc/systemd/system/ssh.socket.d
sudo tee /etc/systemd/system/ssh.socket.d/override.conf > /dev/null <<'EOF'
[Socket]
ListenStream=
ListenStream=2222
EOF
sudo systemctl daemon-reload
sudo systemctl restart ssh.socket
```

Verify:

```sh
ss -tlnp 2>/dev/null | grep ssh
# Expected: LISTEN ... :2222 ... sshd
```

### 4. Authorize bst-gtr9pro's public key

Inside WSL2:

```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIl2Q3hAWzADuYHr3BdPMXPWjTQFVGsxDwehJUv//S4S claude-code-ds4-bridge@bst-gtr9pro' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

(Replace the key string with the actual public key from bst-gtr9pro's `~/.ssh/ds4_remote.pub` if re-deploying.)

## Setup on bst-gtr9pro (the dev PC)

### 1. Generate ed25519 keypair

Git Bash or equivalent:

```sh
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/ds4_remote -N "" -C "claude-code-ds4-bridge@bst-gtr9pro"
```

Public key lands at `~/.ssh/ds4_remote.pub`.

### 2. SSH config alias

Add to `%USERPROFILE%\.ssh\config`:

```text
Host ds4-remote
    HostName bst-originpc
    User bst
    Port 2222
    IdentityFile ~/.ssh/ds4_remote
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

`bst-originpc` resolves via Tailscale MagicDNS (both PCs on same tailnet).

## Test the bridge

From bst-gtr9pro:

```sh
ssh ds4-remote 'echo ok && hostname && uname -a'
```

Expected: `ok` followed by the WSL2 hostname and a Linux uname line. No password prompt.

## Non-interactive PATH for CUDA / GPU tools

Ubuntu's default `~/.bashrc` returns early for non-interactive shells *before* user PATH additions run. Result: `ssh ds4-remote 'nvcc --version'` fails with `nvcc: command not found`, and `ssh ds4-remote 'nvidia-smi ...'` fails the same way, even though both work interactively.

Two paths are involved on WSL2 with CUDA installed:
- `/usr/local/cuda/bin` — where `nvcc` and other CUDA toolkit tools live (from the CUDA Toolkit package).
- `/usr/lib/wsl/lib` — where `nvidia-smi` and the WSL GPU host-passthrough libraries (`libcuda.so`, `libcudadebugger.so.1`, `libnvcuvid.so`, etc.) live. WSL2 places them here because they shim through to the Windows host's NVIDIA driver.

Fix — prepend both above the interactive guard in `~/.bashrc` (inside WSL2):

```sh
# Idempotent: adds ~5 lines at the top of ~/.bashrc
grep -q "WSL GPU PATH" ~/.bashrc || (printf "%s\n" "# WSL GPU tools in PATH (nvidia-smi etc., for non-interactive SSH)" "if [ -d /usr/lib/wsl/lib ]; then export PATH=/usr/lib/wsl/lib:\$PATH; fi" "" | cat - ~/.bashrc > ~/.bashrc.new && mv ~/.bashrc.new ~/.bashrc)
grep -q "CUDA in PATH" ~/.bashrc || (printf "%s\n" "# CUDA in PATH (above interactive guard so non-interactive SSH picks it up)" "if [ -d /usr/local/cuda/bin ]; then export PATH=/usr/local/cuda/bin:\$PATH; fi" "" | cat - ~/.bashrc > ~/.bashrc.new && mv ~/.bashrc.new ~/.bashrc)
```

Verify (from a **fresh** SSH session — the same session that modifies `.bashrc` already has the old PATH loaded):

```sh
ssh ds4-remote 'echo "PATH: $PATH"; nvcc --version | tail -3; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

Expected: PATH includes `/usr/local/cuda/bin` and `/usr/lib/wsl/lib`; nvcc version prints; nvidia-smi reports the GPU name and total VRAM.

## Rollback

Disable mirrored mode — comment out in `.wslconfig`:

```ini
[wsl2]
# networkingMode=mirrored
```

`wsl --shutdown`. Returns to default NAT mode. sshd inside WSL2 is preserved; in NAT mode you'd need a `netsh portproxy` on the Windows host to bridge inbound 2222 to WSL2's internal IP.

Remove the Hyper-V firewall rule:

```powershell
Remove-NetFirewallHyperVRule -Name "WSL-SSH-2222"
```

## Update history

- 2026-05-12 — Initial setup; **first tried mirrored** for fewer moving parts. Hyper-V firewall gates external access; Windows Firewall + key auth + tailnet-only network are the layered defenses.
- 2026-05-12 — **Reverted to NAT + `netsh portproxy`.** Mirrored mode worked for basic SSH (echo, file ops, version checks) but reproducibly caused TCP RST during ds4 inference workload, then wedged WSL2 networking such that subsequent connections timed out. Twice in one session — required Windows reboot each time to recover. NAT had been rock-stable in earlier sessions for the same workload. Verdict: mirrored is incompatible with ds4 inference on this specific Windows/WSL2/driver combination as of this date. NAT + portproxy is the working configuration.
  - Cost of NAT: WSL2's internal IP changes on each boot, so the portproxy rule must be refreshed after `wsl --shutdown`. Worth it for stability.
- 2026-05-12 — Friction observed during first setup, now reflected in doc body:
  - `New-NetFirewallHyperVRule` requires elevated PowerShell; plain PowerShell fails with `Access is denied`.
  - Multi-line backtick PowerShell form is paste-fragile; single-line version is now the documented form.
  - Mirrored mode enabled via WSL Settings GUI requires `wsl --shutdown` before it takes effect — added `Test-NetConnection localhost -port 2222` as the verification step.
  - Ubuntu 24.04 uses socket-activated SSH; `sshd_config`'s `Port` directive is ignored. The `ssh.socket` override unit is the canonical way to change the listening port.
  - Public-key auth required `~/.ssh/authorized_keys` to exist with correct perms (700 on `~/.ssh`, 600 on `authorized_keys`) — sshd refuses keys silently otherwise. First connection attempt failed with `Permission denied (publickey,password)` because the key wasn't installed; the directory `~/.ssh/` didn't exist at all.
  - Non-interactive SSH PATH is **separate** from interactive PATH; needed to add both `/usr/local/cuda/bin` (nvcc) and `/usr/lib/wsl/lib` (nvidia-smi) above the interactive guard in `~/.bashrc`.
