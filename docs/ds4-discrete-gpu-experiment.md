# DS4 cudaMalloc-for-discrete-GPU experiment

Short, focused plan written against [PRINCIPLES.md](../PRINCIPLES.md). A longer 600-line historical version lives outside the repo at `C:\Users\Bertrand\.claude\plans\so-i-just-forked-composed-perlis.md`.

## What we observed on WSL2 (RTX Pro 6000, 64 GB host)

- `--ctx 4096`: gen ~4 t/s, prefill ~3–10 t/s. (DGX Spark baseline: 343 t/s prefill / 13.75 t/s gen at 7 k tokens.)
- Default `--ctx 32768` OOMs in model arena ([ds4_cuda.cu:773-789](../ds4_cuda.cu)) — 1792 MiB chunks fragment after ~74 GiB.
- ~95 GiB usable VRAM (WSLg compositor takes ~400 MiB). 81 GiB model leaves ~14 GiB for KV+scratch.
- Weight upload ~4.4 GB/s (suspected SSD-bound, not PCIe; `bandwidthTest` will tell).

## Hypothesis

Universal `cudaMallocManaged` at [ds4_cuda.cu:1130](../ds4_cuda.cu) for KV / activation / scratch tensors may be the dominant cost on a discrete GPU. Managed memory routes through Windows HMM on WSL2 and can fault-migrate pages on every kernel touch.

Could also be: per-kernel launch overhead, suboptimal wmma kernels on Blackwell, PCIe throttling. The diagnostic below discriminates.

## Setup: SSH bridge from this PC into BST-ORIGINPC's WSL2

Full setup recipe in [wsl2-bridge-setup.md](wsl2-bridge-setup.md). Short version: mirrored networking + Hyper-V firewall allow for port 2222 + WSL2 sshd (socket-activated, listening on 2222) + ed25519 key auth + `~/.ssh/config` alias `ds4-remote`. After setup, `ssh ds4-remote 'cmd'` runs `cmd` inside the remote's WSL2 shell over Tailscale, no password prompt.

## Diagnose (this is the experiment)

Three measurements:

```sh
# (1) GPU util pattern during inference
ssh ds4-remote 'bash -c "
cd ~/ds4
nvidia-smi dmon -s mu -c 90 > /tmp/dmon.log 2>&1 &
DMON_PID=\$!
sleep 1
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=512 ./ds4 --ctx 4096 \
  -p \"Write 200 words about memory bandwidth.\" --nothink -n 128 2>&1
wait \$DMON_PID 2>/dev/null
echo ---DMON---
cat /tmp/dmon.log
"'

# (2) PCIe sanity check
ssh ds4-remote '/usr/local/cuda/extras/demo_suite/bandwidthTest --memory=pinned --htod --dtoh'

# (3) Kernel timing — install nsys interactively once, then profile
# (once on the remote, takes sudo password): sudo apt install -y nsight-systems
ssh ds4-remote 'cd ~/ds4 && DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=512 \
  nsys profile -o /tmp/p --trace=cuda,nvtx,osrt --force-overwrite=true \
  ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 4096 --ctx-max 4096 --gen-tokens 32 && \
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum /tmp/p.nsys-rep | head -60'
```

**Decision tree** (one of these, based on the data):
- *dmon shows oscillating util + mem-bw spikes, or nsys shows mid-inference H2D copies* → managed-memory hypothesis supported. Do the code change.
- *dmon shows steady low util between kernels* → per-kernel launch overhead is the bottleneck. Code change probably doesn't help much; wait for bare metal.
- *dmon shows high+steady util at peak memory bandwidth* → kernels are working as well as they can on this hardware; remaining gap is kernel quality (wmma path on Blackwell). Out of scope for this fork.

## If supported: code change

Five edits in [ds4_cuda.cu](../ds4_cuda.cu):

1. **Line 27** — add `void *host_shadow;` to `ds4_gpu_tensor` struct.
2. **Line 774** — change arena chunk default `1792` → `512` (kills the `--ctx 32768` OOM).
3. **Line 1126** — `cudaMallocManaged` → `cudaMalloc` in `ds4_gpu_tensor_alloc`.
4. **Line 1149** — free `host_shadow` in `ds4_gpu_tensor_free`.
5. **Line 1159** — stage DtoH copy in `ds4_gpu_tensor_contents` (single caller: [ds4.c:8348](../ds4.c)).

(`ds4_gpu_tensor` is private to `ds4_cuda.cu` — see [ds4_gpu.h:16](../ds4_gpu.h) forward declaration. Metal and CPU paths are unaffected.)

## Validate

```sh
ssh ds4-remote 'cd ~/ds4 && make clean && make CUDA_ARCH=sm_120 -j && \
  ./ds4_test --all && \
  ./ds4 --ctx 32768 -p "Explain quantum entanglement in one sentence." --nothink && \
  ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128'
```

Pass: `ds4_test --all` doesn't regress vs. running it pre-change; gen t/s improves vs. the ~4 t/s observation. If anything looks wrong, `git checkout -- ds4_cuda.cu` and revisit.
