# Ubuntu 24.04 LTS Server install (dual-boot) for bare-metal DS4 experiments

Install manual for transitioning BST-ORIGINPC from WSL2 to bare-metal Ubuntu 24.04 LTS Server on a new PCIe 5 NVMe drive. Target: stable RTX Pro 6000 + DS4 experimentation, with room to add vLLM and SGLang later in separate Python venvs.

Structured in two parts:
- **Part 1 — OS + System Foundation**: one-time setup of the box itself (OS install, encryption, networking, security, NVIDIA driver, Python tooling). Engine-agnostic.
- **Part 2 — Inference-Specific**: per-engine installation (DS4 now; vLLM / SGLang later on demand). Each engine in its own venv.

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
- **Hardware-level configuration lives in BIOS, not OS.** Memory EXPO, CPU boost / PBO, fan curves, ReBAR — set once in BIOS, applies whether you boot Linux or Windows. Only reach for OS-level tooling (`fancontrol`, `liquidctl`, etc.) when BIOS-level can't express what you need.

## Pre-install checklist

- New PCIe 5 NVMe physically installed and visible in BIOS.
- Ubuntu 24.04 LTS Server ISO from <https://releases.ubuntu.com/24.04/> (use `ubuntu-24.04.X-live-server-amd64.iso`).
- Boot medium: either a physical USB stick (created with Rufus on Windows or `dd` from another Linux) **or** an IP-KVM with virtual mass-storage support (GL.iNet Comet KVM, JetKVM, PiKVM, etc.) — load the ISO into the KVM's "mounted as USB" feature and the target PC sees it as a regular USB boot device. KVM path is preferred since the whole install runs remotely without touching the box.
- BIOS set to UEFI boot. Secure Boot can stay off — simpler with NVIDIA open drivers; on requires MOK enrollment.
- Tailscale auth key ready (`https://login.tailscale.com/admin/settings/keys`), so the box joins the tailnet on first boot.

---

# Part 1 — OS + System Foundation

Everything below up to and including the Python (uv) section establishes the box itself — boot, security posture, NVIDIA driver, Python tooling. None of it is specific to which inference engine(s) you eventually run. Once Part 1 is complete you have a fully secured, GPU-ready Ubuntu Server that's ready to host any LLM stack.

## Dual-boot architecture (two-drive setup)

This is the **easy case** of dual-boot: Windows lives entirely on one physical drive, Linux lives entirely on the new PCIe 5 NVMe. The drives are independent — Windows has its own EFI partition and bootloader; Linux gets its own EFI partition and GRUB. Neither install touches the other drive's partition table or bootloader.

**Default behavior**: subiquity on Ubuntu Server 24.04 runs `os-prober` automatically at install time and adds the Windows Boot Manager as a GRUB menu entry alongside Ubuntu. So out of the box you get:

- Boot the box → GRUB menu shows `*Ubuntu` (default), `Advanced options for Ubuntu`, `Windows Boot Manager (on /dev/nvme1n1p1)`, `UEFI Firmware Settings`.
- 5-second timeout auto-boots Ubuntu; press Enter to skip the wait; arrow-key to Windows Boot Manager if you want Windows.
- Alternative: at POST press **F8** (boot menu key on the ASUS X670E ROG Crosshair Hero; **Del** / **F2** enters BIOS Setup) to pick a drive directly, bypassing GRUB.

**If you want GRUB to forget Windows** (e.g., it stops detecting properly after a Windows Update touches the EFI partition, or you want strict drive isolation): edit `/etc/default/grub` to set `GRUB_DISABLE_OS_PROBER=true`, then `sudo update-grub`. After that, the BIOS F8 boot menu is the only way to reach Windows. Each OS bootloader stays in its own drive — if you ever remove a drive, the other OS still boots independently either way.

### Safety steps before installing Linux

- Note the **size and model** of your Windows drive (Settings → System → About → "Device specifications", or Disk Management). Lets you positively identify it during Linux install and **not** pick it.
- Optional: shrink the Windows partition if you want a shared data partition. Not needed for our setup since the two OSes live on separate drives.
- Make a backup of anything important on the Windows drive — not because the install will touch it, but because user-error during disk selection is a real risk. Belt-and-suspenders.

### After Linux install: verify Windows still boots

Reboot into BIOS boot menu (F12), pick the Windows drive. Should boot Windows normally. If it does, dual-boot is working. Then set Linux drive as default in BIOS for daily use.

If Windows doesn't boot after Linux install, you didn't actually break it — its EFI partition is intact on its own drive. Most likely BIOS boot order changed; reselect the Windows drive in BIOS UEFI menu and it'll come up.

## Installation

### 1. Boot the installer

Boot from USB (or KVM virtual USB). At the GRUB menu, pick **"Ubuntu Server with the HWE kernel"** — the second option, not the default. HWE (Hardware Enablement) ships a newer kernel (6.11+ class) with better support for recent silicon (Zen 5, Blackwell GPUs). The default GA kernel (6.8-era) also works but is conservatively older for this hardware.

At the "Choose the type of installation" screen:
- **Base**: `(X) Ubuntu Server` (the default, not "minimized"). Minimized strips interactive-admin tools intended for cloud/container environments where no humans log in; we want the standard tooling.
- **`[ ] Search for third-party drivers`**: leave **unchecked**. This option installs the NVIDIA proprietary driver, which does **not** support Blackwell `sm_120`. We install the **open** kernel module from NVIDIA's apt repo after the base OS is up.

At the network step:
- DHCP (default) is fine. Same physical NIC means the same MAC across Windows and Linux boots, so DHCP from the router gives the same lease either way. Tailscale provides stable hostname-based access regardless of LAN IP changes.
- If you want a static IP matching Windows, read Windows's current settings via `ipconfig /all` and enter them manually here.

The installer (subiquity) handles partitioning, encryption, and OpenSSH in one flow.

### 2. Storage — full-disk LUKS + ext4

- Pick the **new PCIe 5 NVMe drive** as the install target. Verify by size/model — do NOT touch the Windows drive.
- Tick **"Set up this disk as an LVM group"** and **"Encrypt the LVM group with LUKS"**.
- Set a strong passphrase. This is what you'll type at every boot. Recommended: store it in a password manager (Bitwarden, 1Password, etc.) and paste it through your IP-KVM's paste-text feature at boot — no typing a long passphrase by hand, no recovery-key gymnastics.
- "Also create a recovery key" can be left unchecked when the passphrase lives in a password manager. (Caveat if you do want one: the installer stores it inside the encrypted target at `/var/log/installer/`, so it's only useful if you also copy `~/recovery-key.txt` out of the live ISO environment to external storage **before** rebooting after install.)

Default layout subiquity creates:
- `/boot/efi` (~1 GB, fat32, unencrypted) — UEFI bootloader.
- `/boot` (~2 GB, ext4, unencrypted) — necessary so GRUB can start.
- LUKS-encrypted LVM physical volume containing one logical volume for `/` (ext4).
- No swap partition by default — add a swap file later only if needed.

**Resize root before clicking Done.** Subiquity defaults the root LV (`ubuntu-lv`) to only 100 GB even when the encrypted volume group has 1.8 TB — leaves ~1.7 TB free inside the VG, unused. For a single-purpose box (DS4 model is 81 GB, KV caches and logs accumulate, future engine models add up) take it all now: at the "Storage configuration" review screen, select `ubuntu-lv` under "USED DEVICES" → **Edit** → set Size to the maximum (clear the field, or enter the VG size shown above). Then `Done`.

(Alternative: accept the 100 GB default and extend later via `sudo lvextend -l +100%FREE ubuntu-vg/ubuntu-lv && sudo resize2fs /dev/ubuntu-vg/ubuntu-lv`. Doing it at install is simpler.)

### 3. User and SSH

**Watch out**: subiquity's Profile screen sometimes renders empty (fields not visible) and lets you `Done` past it without setting anything — would leave the installed system with no user account. If the screen looks blank, hit Back from a later screen and revisit it until the four fields are populated.

- Username: `bst`
- Hostname: `bst-originpc-ai` — distinct from the Windows-side `bst-originpc` so Tailscale registers them as two separate devices (otherwise it auto-suffixes the second one and you lose control of the name). Same MAC, so DHCP still gives the same LAN IP regardless. The `-ai` suffix captures the box's purpose.
- Install OpenSSH server: **yes**.
- **`[X] Allow password authentication over SSH`** stays checked here. Subiquity enforces that you can't disable password auth at install time without also importing at least one authorized key (otherwise the system would be unreachable on first boot). The password path gets closed later via Basic Hardening after the key is in place.
- **Skip the GitHub key import** if you maintain separate keys for server admin vs GitHub identity (recommended — keeps the GitHub-uploaded pubkey out of `authorized_keys` on your servers). Leave `AUTHORIZED KEYS` empty here, and paste your lab-admin public key manually after first boot — *before* running Basic Hardening:

```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA... your-lab-admin-key-comment' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

If you don't bother with key segmentation, Import → GitHub → `<your-username>` pulls all keys from `https://github.com/<your-username>.keys` and pre-authorizes them. Simpler but ties GitHub-side keys to server-admin access.

### 3.5. Ubuntu Pro — skip

`(X) Skip for now`. Ubuntu Pro is free for personal use (up to 5 machines) and adds Extended Security Maintenance (10-year support), FIPS/STIG/HIPAA modules, and Livepatch (kernel updates without reboot). For a single home-lab science box none of those earn the extra setup — standard 24.04 LTS support runs to 2029, longer than the hardware will sit in this configuration. `sudo pro attach` is available later if Livepatch becomes interesting.

### 3.6. Featured server snaps — skip all

The installer offers a list of "popular" snaps (microk8s, nextcloud, lxd, prometheus, canonical-livepatch, aws-cli, doctl, etc.). **Leave everything unchecked.** None of them belong in this setup; anything we actually need (Tailscale, CUDA, uv, the inference engines) gets installed explicitly post-boot via the sections below. Snap-installing things we don't use just adds attack surface and auto-update churn.

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

Then in the Tailscale admin console at <https://login.tailscale.com/admin/machines>:

1. **Click the new device** (`bst-originpc-ai`).
2. **Disable key expiry** (option in the device detail panel, or click the "180d" badge to toggle off). Without this, Tailscale auth expires every 180 days by default — surprising lockout for a long-running server.
3. **Apply a tag** (e.g. `tag:server`) consistent with your tailnet's existing ACL policy. Tagged devices are governed by the tag's ACL rules rather than the user that authorized them — cleaner for headless boxes that outlive a personal session.

Test from another tailnet host: `ssh bst@bst-originpc-ai` should connect (password auth still on at this stage; use the user password from Bitwarden).

### Authorize the lab-admin SSH key

Before running Basic Hardening (which closes the password path), add your lab-admin pubkey to `authorized_keys` and confirm key auth works end-to-end. On the new box (console or interim password-SSH session):

```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA... your-lab-admin-key-comment' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
cat ~/.ssh/authorized_keys
```

Then from your **dev workstation** (literally a different machine — the one that holds the lab-admin *private* key, e.g. `bst-gtr9pro`. Do NOT run this from the new box itself, which would loopback-SSH to its own sshd without the matching private key):

```sh
ssh bst@bst-originpc-ai 'echo ok && hostname'
```

Should return `ok` and `bst-originpc-ai` **without** prompting for a password. If it asks for a password, the pubkey isn't being matched — fix it before proceeding to Basic Hardening, otherwise you can lock yourself out.

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
sudo unattended-upgrade --dry-run 2>&1 | tail -20   # preview what would apply (omit -d / debug to avoid pages of pinning noise)
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

### CUDA on PATH for all shells

Do this **before** the verify step below — otherwise `nvcc` returns "command not found" even though it's installed at `/usr/local/cuda/bin/nvcc`.

```sh
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
sudo chmod +x /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh   # apply to current shell without logging out
```

`/etc/profile.d/` covers all users and all shell modes (interactive and non-interactive over SSH) — cleaner than the per-user `.bashrc` workaround we needed on WSL2. New shells from now on get CUDA on PATH automatically.

### Verify

```sh
nvidia-smi
nvcc --version
# Confirm Resizable BAR is active — should show one BAR sized at 64 GB or so (not 256 MiB)
GPU_BDF=$(lspci | grep -i 'vga.*nvidia' | awk '{print $1}')
sudo lspci -vv -s $GPU_BDF | grep -E "Region|BAR" | head -10
```

`nvidia-smi` should show "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" with 97887 MiB total. `nvcc --version` should report 13.x. The `lspci -vv` output should show a "Region X: Memory at ... (64-bit, prefetchable) [size=N]" where N is multiple of 32 GB — that's Resizable BAR exposing full VRAM as a single mapping. If you only see 256 MiB sized regions, ReBAR didn't activate; recheck BIOS "Re-Size BAR Support" and "Above 4G Decoding".

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

---

# Part 2 — Inference-Specific

Now the box is foundation-ready, install only the engines you intend to use. Each engine gets its own venv via `uv` so their dependencies stay isolated from each other and from system Python. Start with DS4 (the current focus); add vLLM / SGLang later, on demand.

## DS4 fork — checkout and build

```sh
cd ~
git clone https://github.com/berschmitt/ds4.git
cd ds4
make cuda CUDA_ARCH=sm_120 -j$(nproc)
```

(Adjust the clone URL if the fork moves.)

Note: upstream made target selection explicit (commit `8aaf3d1`) — bare `make` now prints a help menu. Use `make cuda CUDA_ARCH=sm_120` for our Blackwell build, `make cuda-spark` for DGX Spark, or `make cuda-generic` for auto-detect.

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

## Driving the box from Claude Code / VS Code

Day-to-day development happens from your dev workstation (`bst-gtr9pro`), with the box acting as the GPU + storage host. Easiest pattern: **VS Code Remote-SSH** into the box, then run Claude Code in that remote-context VS Code window. Claude Code's tools (Bash, Read, Edit, Write, Grep, Glob) then operate directly on files inside the box — no SSH-prefix wrappers, no copy-paste.

One-time setup on the dev workstation:

1. Install the **Remote - SSH** extension in VS Code (`ms-vscode-remote.remote-ssh`).
2. Open the Command Palette → `Remote-SSH: Connect to Host...` → pick `bst-originpc-ai` (Remote-SSH reads `~/.ssh/config` so the alias just works).
3. VS Code installs its server component on the box automatically (`~/.vscode-server/`), uses your SSH key.
4. Once connected, **File → Open Folder → `/home/bst/ds4`**.
5. Install the Claude Code extension in this Remote-SSH window if it didn't carry over from your local install.

After that, every Claude Code session you start from this VS Code window runs with the box as its working environment: filesystem operations hit the box's disk, Bash runs on the box's shell, GPU is local from Claude Code's perspective. No latency from chaining through SSH commands.

Tailscale handles the network; mDNS (`bst-originpc-ai`) or MagicDNS hostname (`bst-originpc-ai.tail520464.ts.net`) both resolve.

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

---

# Appendices

## Fan curves

Default ASUS ROG fan curves (when never configured) are aggressive — fans ramp early at low loads. All four active fan/pump channels on this box turn out to be motherboard-controlled (verified empirically — changing each BIOS Q-Fan curve has audible effect), so the entire problem is solvable in BIOS at the hardware level. No Linux daemon needed; the super-I/O chip (NCT6798D on X670E) enforces curves regardless of OS.

**Do this now** (5 minutes in BIOS Setup, solves the loud-fans complaint for both OSes):

1. Reboot, **Del** at POST → BIOS Setup → Advanced → Monitor → Q-Fan Control.
2. For each motherboard header in use: pick **Manual** profile, set the mode to **PWM Mode** (explicit, instead of Auto Detect — gives cleaner low-duty control than DC mode's higher stiction floor), and set the 4 curve points.
3. Save & Exit. The super-I/O (NCT6798D on X670E) enforces the curves at the hardware level, including across OS reboots and reinstalls.

Run **Q-Fan Tuning** first (one level up in the Q-Fan Control menu, top of screen). It sweeps each connected fan to find its real stall threshold, then sets per-channel minimum-duty floors so curve points below the floor won't ever stall the fan. On this box the tuner reported:

- CPU_FAN min: 0% (H150i pump's firmware handles its own floor; accepts 0% PWM)
- CHA_FAN1 min: 24% (real stall threshold — case fan stalls below this)
- CHA_FAN2/3/4 min: 20% (default, empty headers — nothing to test)
- W_PUMP+ / AIO_PUMP: not tested by Q-Fan Tuning (it skips pump headers)

Then translate Fan Xpert 4 curves into BIOS (4-point approximation, respecting the discovered floors). Actual values settled on for this box:

| Header | ~35 °C | 70 °C | 80 °C | 90 °C |
|---|---:|---:|---:|---:|
| CPU_FAN (H150i pump tach + PWM input) | 0 % | 5 % | ~25 % | 100 % |
| CHA_FAN1 (case fan; floor = 24% per tuner) | 25 % | 25 % | 40 % | 100 % |
| W_PUMP+ | 25 % | 25 % | 30 % | 100 % |
| AIO_PUMP | 25 % | 25 % | 30 % | 100 % |

**Why the low-end percentages differ from the Fan Xpert curves on Windows**: the BIOS Q-Fan UI enforces stricter per-channel minimums (after Q-Fan Tuning) than Fan Xpert allows. Even when Fan Xpert had a channel set to ~10% on Windows, BIOS won't let you set the same curve point that low — it clamps to the tuner-discovered floor (or a hard-coded conservative floor for pump headers). So the BIOS curves end up higher at the low end than the equivalent Windows curves. That's a UI/firmware limitation, not a problem — and for pump channels, the higher minimum is actually safer (ensures coolant flow).

All four active channels above had clear audible effect when their curves were edited on Windows — replicate all four faithfully in BIOS, don't skip any based on "0 RPM displayed in Fan Xpert" alone (that reading doesn't reliably mean nothing is connected; some headers feed pumps or fan-splitters whose tach comes back via a different path).

CHA_FAN2 / CHA_FAN3 / CHA_FAN4: nothing connected on this box (verified on Windows — they were disabled in Fan Xpert). Skip them in BIOS too; no curve to set, no impact.

Verify after applying: empirical ear test. Idle should be near-silent; ramp should be audible only under sustained inference load.

If BIOS curves ever prove not fine-grained enough later (e.g. you want case fans tied to GPU temp specifically), `lm-sensors` + `fancontrol` + the `nct6775` kernel module is the Linux-side fallback — but adds a daemon to maintain. BIOS Q-Fan alone is the robust default and what's recommended here. `liquidctl` for Corsair H-series AIOs may still be useful later for *monitoring* coolant temp and pump RPM (`liquidctl status`), but isn't needed for fan control on this build since BIOS handles it.

## What's intentionally not in this install

- No GUI / desktop environment — server box, accessed over SSH.
- No swap file — 64 GB RAM is enough for DS4 (mmap'd model pages on demand from NVMe). Add a swap file only if specific workloads run out.
- No Docker / nvidia-container-toolkit — vLLM and SGLang both run fine in bare venvs; revisit if containerization becomes useful later.
- No `fail2ban` / AIDE / OSSEC / sysctl hardening — overkill for a single-user, Tailscale-only science box. Default Ubuntu 24.04 is already reasonable for this use.

## Update history

- 2026-05-12 — Initial draft, written while still on WSL2 and before NVMe drive arrival. Body describes the intended clean path; will be validated against the actual install once the drive is in. Friction items observed during install go into this history section; the body stays clean.
