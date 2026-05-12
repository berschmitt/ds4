# Ubuntu 24.04 LTS Server install (dual-boot) for bare-metal DS4 experiments

Install manual for transitioning BST-ORIGINPC from WSL2 to bare-metal Ubuntu 24.04 LTS Server on a new PCIe 5 NVMe drive. Target: stable RTX Pro 6000 + DS4 experimentation, with room to add vLLM and SGLang later in separate Python venvs.

## Hardware target

- AMD Ryzen 9 9950X3D (Zen 5, x86_64)
- NVIDIA RTX Pro 6000 Blackwell Workstation Edition (96 GB GDDR7, `sm_120`)
- 64 GB system RAM
- New PCIe 5 NVMe drive — destination for the Linux install
- Existing Windows install on a separate drive — dual-boot, untouched

## Goals

- Clean, reproducible install: fresh OS → DS4 runs, no back-and-forth.
- Full-disk encrypted ext4 root with passphrase prompt at boot (LUKS).
- Server, not Desktop — headless, accessed via Tailscale SSH.
- CUDA Toolkit 13.x for Blackwell `sm_120` native support.
- Per-project Python isolation: system Python stays minimal; vLLM / SGLang each get their own venv later.

## Pre-install checklist

- New PCIe 5 NVMe physically installed and visible in BIOS.
- Ubuntu 24.04 LTS Server ISO from <https://releases.ubuntu.com/24.04/> (use `ubuntu-24.04.X-live-server-amd64.iso`).
- Boot medium: either a physical USB stick (created with Rufus on Windows or `dd` from another Linux) **or** an IP-KVM with virtual mass-storage support (GL.iNet Comet KVM, JetKVM, PiKVM, etc.) — load the ISO into the KVM's "mounted as USB" feature and the target PC sees it as a regular USB boot device. KVM path is preferred since the whole install runs remotely without touching the box.
- BIOS set to UEFI boot. Secure Boot can stay off — simpler with NVIDIA open drivers; on requires MOK enrollment.
- Tailscale auth key ready (`https://login.tailscale.com/admin/settings/keys`), so the box joins the tailnet on first boot.

## Dual-boot architecture (two-drive setup)

This is the **easy case** of dual-boot: Windows lives entirely on one physical drive, Linux lives entirely on the new PCIe 5 NVMe. The drives are independent — Windows has its own EFI partition and bootloader; Linux gets its own EFI partition and GRUB. Neither install touches the other drive's partition table or bootloader.

**Boot strategy — recommended: BIOS-managed**

- BIOS UEFI boot order: **new NVMe (Linux) first**, Windows drive second.
- Default boot: Linux comes up automatically.
- To boot Windows occasionally: press the motherboard's "boot menu" key at POST (usually **F12**, sometimes F8 / F10 / F11 depending on board), pick the Windows drive from the list. No GRUB Windows entry needed; nothing to maintain.
- Pros: each OS bootloader stays in its own drive, no cross-drive dependency. If you ever remove the Linux drive, Windows still boots; if you remove the Windows drive, Linux still boots.
- Cons: one key-press at POST when you want Windows. Minor.

**Alternative — GRUB-managed (optional)**

If you'd rather have GRUB show a Linux/Windows menu at every boot:

```sh
sudo apt install -y os-prober
if grep -q "^GRUB_DISABLE_OS_PROBER=" /etc/default/grub; then
  sudo sed -i 's/^GRUB_DISABLE_OS_PROBER=.*/GRUB_DISABLE_OS_PROBER=false/' /etc/default/grub
else
  echo 'GRUB_DISABLE_OS_PROBER=false' | sudo tee -a /etc/default/grub
fi
sudo update-grub
```

GRUB's `os-prober` scans other drives, detects the Windows EFI bootloader, and adds it as a menu entry. Convenience over robustness — if Windows Update touches its EFI partition, GRUB might lose the entry until you re-run `update-grub`.

### Safety steps before installing Linux

- Note the **size and model** of your Windows drive (Settings → System → About → "Device specifications", or Disk Management). Lets you positively identify it during Linux install and **not** pick it.
- Optional: shrink the Windows partition if you want a shared data partition. Not needed for our setup since the two OSes live on separate drives.
- Make a backup of anything important on the Windows drive — not because the install will touch it, but because user-error during disk selection is a real risk. Belt-and-suspenders.

### After Linux install: verify Windows still boots

Reboot into BIOS boot menu (F12), pick the Windows drive. Should boot Windows normally. If it does, dual-boot is working. Then set Linux drive as default in BIOS for daily use.

If Windows doesn't boot after Linux install, you didn't actually break it — its EFI partition is intact on its own drive. Most likely BIOS boot order changed; reselect the Windows drive in BIOS UEFI menu and it'll come up.

## Installation

### 1. Boot the installer

Boot from USB; pick the standard server install when prompted. The installer (subiquity) handles partitioning, encryption, and OpenSSH in one flow.

### 2. Storage — full-disk LUKS + ext4

- Pick the **new PCIe 5 NVMe drive** as the install target. Verify by size/model — do NOT touch the Windows drive.
- Tick **"Set up this disk as an LVM group"** and **"Encrypt the LVM group with LUKS"**.
- Set a strong passphrase. This is what you'll type at every boot.

Default layout subiquity creates:
- `/boot` (~2 GB, ext4, unencrypted) — necessary so GRUB can start.
- LUKS-encrypted LVM physical volume containing one logical volume for `/` (ext4).
- No swap partition by default — add a swap file later only if needed.

### 3. User and SSH

- Username: `bst`
- Hostname: `bst-originpc` (matches existing Tailscale naming).
- Install OpenSSH server: **yes**.
- Import SSH keys from GitHub if available; otherwise paste your dev-workstation public key after first boot.

### 4. Finish, remove USB, reboot

First boot: LUKS passphrase prompt before GRUB → kernel hand-off → login. Type passphrase; box comes up.

## Post-install — system foundations

All commands run as `bst`, with `sudo` where shown.

### Update + basics

```sh
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git curl wget htop tmux unzip pkg-config software-properties-common ca-certificates gnupg
```

### Tailscale

```sh
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Visit the printed URL, authorize the device. SSH access reaches this box via the system OpenSSH server (installed during the base install, hardened in the Basic Hardening section below) routed over the tailnet — we deliberately don't use Tailscale's built-in `--ssh` flag so there's exactly one SSH stack to reason about.

Test from another tailnet host: `ssh bst@bst-originpc` should connect.

### Automatic security updates, no auto-reboot

Best of both: security CVE patches apply automatically; non-security upgrades and reboots stay manual.

```sh
sudo apt install -y unattended-upgrades
# Enable the daily unattended-upgrade run (default on Ubuntu 24.04, made explicit + idempotent):
cat | sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
# Ensure auto-reboot stays OFF (this is the default but make it explicit in the config):
sudo sed -i 's|^//\s*Unattended-Upgrade::Automatic-Reboot "false";|Unattended-Upgrade::Automatic-Reboot "false";|' /etc/apt/apt.conf.d/50unattended-upgrades
grep -E "Automatic-Reboot " /etc/apt/apt.conf.d/50unattended-upgrades
sudo unattended-upgrade --dry-run -d 2>&1 | tail -20   # preview what would apply
```

Behavior after this:
- Security updates from the `noble-security` apt origin apply daily, in the background.
- Non-security package upgrades (feature releases, version bumps) do NOT auto-install.
- The system never reboots automatically. Apply pending kernel updates with `sudo reboot` on a cadence you choose.

For periodic full upgrades, do them manually when you want predictable state changes:

```sh
sudo apt update && sudo apt full-upgrade -y
```

### Basic hardening

The Tailscale tailnet is the primary trust boundary, but a few sshd settings remove obvious foot-guns:

```sh
sudo sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?X11Forwarding .*/X11Forwarding no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?KbdInteractiveAuthentication .*/KbdInteractiveAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?UsePAM .*/UsePAM yes/' /etc/ssh/sshd_config
grep -q "^AllowUsers " /etc/ssh/sshd_config || echo "AllowUsers bst" | sudo tee -a /etc/ssh/sshd_config
sudo systemctl restart ssh
```

What this does: root SSH allowed via key only, never password; no password auth for any user; X11 forwarding off (headless box, unused attack surface); keyboard-interactive off (closes the PAM-password backdoor); `UsePAM yes` kept on (default — needed for account validation, session setup, login records; the password path through PAM is already closed by `KbdInteractiveAuthentication no`); `AllowUsers bst` whitelists the only user. Verify your dev-workstation key works **before** closing your current session — if not, fix `~/.ssh/authorized_keys` while you still have a route in.

Minimal UFW (optional, defense-in-depth on top of tailnet):

```sh
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on tailscale0
sudo ufw enable
sudo ufw status verbose
```

Net effect: inbound only on the Tailscale interface; LAN-side inbound dropped. Skip UFW if you'd rather rely on the tailnet boundary alone — both are valid.

Notes on what's intentionally **not** added: no `fail2ban` (no public-facing surface to brute-force against), no AIDE / OSSEC (overkill for a single-user science box), no kernel sysctl hardening (default Ubuntu 24.04 is already reasonable for this use).

## NVIDIA driver + CUDA Toolkit

Blackwell `sm_120` requires CUDA Toolkit 12.6+ and the **open** kernel driver. The proprietary `nvidia-driver-XXX` does not support `sm_120` on Linux.

### Add NVIDIA's CUDA repo for Ubuntu 24.04

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

If the `wget` 404s, NVIDIA may have published a newer keyring version. Check the current "Network Installer" instructions for Ubuntu 24.04 at <https://developer.nvidia.com/cuda-downloads> and substitute the URL shown there.

### Install toolkit + open driver

```sh
sudo apt install -y cuda-toolkit-13-0 nvidia-open
```

`nvidia-open` is the metapackage pulling the latest open kernel driver matching the toolkit.

**If Secure Boot is enabled** (Windows UEFI mode in BIOS): the installer signs the open kernel module with a temporary MOK (Machine Owner Key) and asks you to set a one-time password. On the next boot, shim shows a blue screen ("Perform MOK management") — pick **Enroll MOK** → **Continue** → enter the password you just set → reboot. After enrollment, the kernel trusts the module and `nvidia-smi` works. Skip-able by turning Secure Boot off in BIOS before installing the driver.

### Reboot so the kernel module loads

```sh
sudo reboot
```

(Re-enter LUKS passphrase at boot. If Secure Boot is on, the MOK enrollment screen appears here too.)

### Verify

```sh
nvidia-smi
nvcc --version
# Confirm Resizable BAR is active — should show one BAR sized at 64 GB or so (not 256 MiB)
GPU_BDF=$(lspci | grep -i 'vga.*nvidia' | awk '{print $1}')
sudo lspci -vv -s $GPU_BDF | grep -E "Region|BAR" | head -10
```

`nvidia-smi` should show "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" with 97887 MiB total. `nvcc --version` should report 13.x. The `lspci -vv` output should show a "Region X: Memory at ... (64-bit, prefetchable) [size=N]" where N is multiple of 32 GB — that's Resizable BAR exposing full VRAM as a single mapping. If you only see 256 MiB sized regions, ReBAR didn't activate; recheck BIOS "Re-Size BAR Support" and "Above 4G Decoding".

### CUDA on PATH for all shells

```sh
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
sudo chmod +x /etc/profile.d/cuda.sh
```

Log out + back in (or open a new shell) so it applies. Bare-metal equivalent of the WSL2 `.bashrc` workaround — cleaner since `/etc/profile.d/` covers all users and all shell modes (interactive and non-interactive).

## Python — `uv` for all venvs and installs

Ubuntu 24.04 ships Python 3.12. We use `uv` (Rust-implemented venv + package manager from Astral, ~10× faster than pip) for every Python environment. No `pip`, no vanilla `python3 -m venv` — one tool throughout, cleaner and faster.

```sh
sudo apt install -y python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
# Re-source shell so the uv binary is on PATH:
source ~/.bashrc
uv --version
```

`python3-dev` is the only system Python package needed — it provides headers for any C-extension Python package that compiles from source. `uv` itself doesn't depend on `python3-pip` or `python3-venv` at the system level.

Per-project pattern (used by every engine venv below):

```sh
cd <project-dir>
uv venv .venv         # create venv
source .venv/bin/activate
uv pip install <packages>   # install
```

## DS4 fork — checkout and build

```sh
cd ~
git clone https://github.com/berschmitt/ds4.git
cd ds4
make CUDA_ARCH=sm_120 -j$(nproc)
```

(Adjust the clone URL if the fork moves.)

### Pull the model

```sh
./download_model.sh q2-imatrix
```

Puts the q2-imatrix GGUF (~81 GB) under `./gguf/` and symlinks `./ds4flash.gguf`. Confirm the NVMe has space — q2-imatrix is 81 GB; q4 won't fit in 96 GB VRAM anyway.

### Verify

```sh
./ds4_test --all
./ds4 -p "Say hi in three words." --nothink -n 8 --ctx 4096
```

`ds4_test --all` must pass token-byte parity vs the official DeepSeek V4 Flash API logprob vectors. The smoke generation should produce a coherent reply with no crashes — bare metal removes the WSL2 wedge risk we saw earlier.

## Per-project venv pattern for vLLM / SGLang (later)

vLLM in its own venv:

```sh
mkdir -p ~/engines/vllm && cd ~/engines/vllm
uv venv .venv
source .venv/bin/activate
uv pip install vllm
```

SGLang in its own venv:

```sh
mkdir -p ~/engines/sglang && cd ~/engines/sglang
uv venv .venv
source .venv/bin/activate
uv pip install "sglang[all]"
```

Substitute the exact install command from each engine's current docs (versions and extras evolve fast). Activating one venv at a time keeps each engine's dependencies fully isolated from the others and from system Python.

## What's intentionally not in this install

- No GUI / desktop environment — server box, accessed over SSH.
- No swap file — 64 GB RAM is enough for DS4 (mmap'd model pages on demand from NVMe). Add a swap file only if specific workloads run out.
- No Docker / nvidia-container-toolkit — vLLM and SGLang both run fine in bare venvs; revisit if containerization becomes useful later.
- No `fail2ban` / AIDE / OSSEC / sysctl hardening — overkill for a single-user, Tailscale-only science box. Default Ubuntu 24.04 is already reasonable for this use.

## Update history

- 2026-05-12 — Initial draft, written while still on WSL2 and before NVMe drive arrival. Body describes the intended clean path; will be validated against the actual install once the drive is in. Friction items observed during install go into this history section; the body stays clean.
