# vLLM server — Docker

Alternative deployment path: run the vLLM inference server in a container
instead of directly on the host. Same `server.py`, same endpoints, same
client experience — just wrapped in a reproducible image with a single-
file deployment recipe.

## When to use Docker vs systemd

| Need | Recommendation |
|---|---|
| Already working on the dev box, minimal moving parts | **systemd** (the existing setup) |
| Reproducible build, easy to ship to another machine | **Docker** |
| Multiple vLLM instances on one host (different ports) | **Docker** |
| Cleanest VRAM cleanup on restart | **Either** (both kill subprocesses cleanly on stop) |
| CI/CD or automated deployments | **Docker** |

The two paths use the same port (8000) so they're mutually exclusive at
runtime — switching from one to the other is just stop one, start the other.

---

## Prerequisites

On the CentOS host:

1. **Docker Engine** installed
   ```bash
   sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER     # log out / back in for group to take effect
   ```

2. **NVIDIA Container Toolkit** — required for `--gpus` support
   ```bash
   sudo dnf install nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Verify GPU passthrough** before building anything:
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.2.1-runtime-ubuntu24.04 nvidia-smi
   ```
   You should see both your GPUs listed. If this doesn't work, the build
   succeeds but the container fails at runtime.

---

## Layout

| File | Where it lives | Role |
|---|---|---|
| `Dockerfile` | `docker/` | Image recipe |
| `entrypoint.sh` | `docker/` | Translates `VLLM_DEFAULT_MODEL` env → `--model` CLI |
| `docker-compose.yml` | `docker/` | One-command deployment, GPU + volume + IPC config |
| `.dockerignore` | **project root** | Excludes `.venv/`, `client/`, etc. from build context |
| `README.md` | `docker/` | This file |

---

## Build

```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine
docker compose -f docker/docker-compose.yml build
```

First build takes **~5–10 minutes** — CUDA 13.2 runtime base image is
~2 GB to download, vLLM + torch wheels add another ~3 GB on top. Subsequent
builds are fast as long as `requirements.txt` doesn't change (Docker reuses
the cached layer).

Final image size: **~6–8 GB**. This is fundamental to running vLLM —
PyTorch + CUDA libs alone are several GB.

---

## Run

```bash
# Default model: qwen, foreground (logs visible in terminal)
docker compose -f docker/docker-compose.yml up

# Detached
docker compose -f docker/docker-compose.yml up -d

# Different model — one-shot via env var
VLLM_DEFAULT_MODEL=mistral docker compose -f docker/docker-compose.yml up -d

# Different model — persistent via .env file in docker/ directory
echo "VLLM_DEFAULT_MODEL=mistral" > docker/.env
docker compose -f docker/docker-compose.yml up -d
```

The server is reachable on `http://192.168.88.23:8000` from the Windows
client — **no Streamlit changes**, same URL as the native or systemd setups.

---

## Operations

```bash
# Follow logs
docker compose -f docker/docker-compose.yml logs -f

# Status
docker compose -f docker/docker-compose.yml ps

# Stop (keeps the image, removes the container)
docker compose -f docker/docker-compose.yml down

# Restart (after editing server.py, requires rebuild)
docker compose -f docker/docker-compose.yml build && \
  docker compose -f docker/docker-compose.yml up -d

# Open a shell inside the running container (debug)
docker exec -it vllm-server bash

# Check what's actually in the image (e.g. verify vllm version)
docker exec vllm-server pip show vllm
```

---

## Model swap (in-container)

The in-process `/admin/load_model` endpoint continues to work normally —
it's the same `server.py`, just running inside a container. The container's
restart policy doesn't interfere.

If the in-process swap ever leaks VRAM (the failure mode we instrumented
the VRAM probes for), the bulletproof reset is a container restart:

```bash
# Change boot-time model + restart
echo "VLLM_DEFAULT_MODEL=mistral" > docker/.env
docker compose -f docker/docker-compose.yml restart
```

`docker compose restart` sends SIGTERM to the container's main process
(python), which propagates to the EngineCore subprocess, then SIGKILL after
10s if it doesn't exit. On stop the container's namespace is reset and all
GPU allocations are released by the OS before the new container starts.

---

## Cache strategy

The compose file bind-mounts the host's existing caches:

| Host | Container | Why |
|---|---|---|
| `~/.cache/huggingface` | `/cache/huggingface` | Pre-downloaded models (~11 GB) skip re-download |
| `~/.cache/vllm` | `/cache/vllm` | CUDA-graph compile cache survives container restarts |

This means cold start is ~30s, same as the native setup. Without these
mounts, the first run would re-download both models inside the container
volume.

If you want a clean-state container (e.g. to test that the image builds and
runs end-to-end without depending on host state), edit `docker-compose.yml`
to use **named volumes** instead of bind mounts:

```yaml
volumes:
  - hf-cache:/cache/huggingface
  - vllm-cache:/cache/vllm

# At the end of the file:
volumes:
  hf-cache:
  vllm-cache:
```

The first run with named volumes downloads models into the volume (~11 GB,
~10 minutes). Subsequent runs are fast.

---

## Coexistence with the systemd unit

Both deployments target port 8000, so only one can run at a time. Switching:

```bash
# Native (systemd) → Docker
systemctl --user stop vllm-server
docker compose -f docker/docker-compose.yml up -d

# Docker → Native (systemd)
docker compose -f docker/docker-compose.yml down
systemctl --user start vllm-server
```

If you want **systemd to manage Docker** — Docker's reproducibility plus
systemd's restart semantics — the unit becomes much simpler than the
direct-Python version. Create `~/.config/systemd/user/vllm-server-docker.service`:

```ini
[Unit]
Description=vLLM server (Docker)
After=docker.service

[Service]
Type=simple
WorkingDirectory=%h/Desktop/Pycharm_Projects/vLLM_Inference_Engine
ExecStart=/usr/bin/docker compose -f docker/docker-compose.yml up
ExecStop=/usr/bin/docker compose -f docker/docker-compose.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

This isn't shipped in the bundle — only the direct-Python systemd unit is.
Add it later if you want the combination.

---

## Troubleshooting

### "could not select device driver "" with capabilities: [[gpu]]"

NVIDIA Container Toolkit isn't installed or wasn't configured for Docker.

```bash
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:13.2.1-runtime-ubuntu24.04 nvidia-smi
```

### "Free memory on cuda:0 is less than utilization" on container start

Same VRAM-leak failure mode as native. Container stop usually clears it,
but if a previous run was killed harshly:

```bash
docker compose -f docker/docker-compose.yml down
sleep 5
nvidia-smi    # confirm GPU 1 is mostly free
docker compose -f docker/docker-compose.yml up -d
```

### "Cannot find module vllm.v1.metrics.loggers" or similar import errors

The vLLM version inside the container doesn't match what `server.py`
expects. Verify:

```bash
docker exec vllm-server pip show vllm
# Expect: Version: 0.20.2
```

If you've upgraded `vllm` in `server/requirements.txt`, you must rebuild
the image:

```bash
docker compose -f docker/docker-compose.yml build --no-cache
```

### Build fails at `pip install vllm`

vLLM 0.20.2's wheels target specific CUDA versions. If the runtime CUDA in
the base image doesn't have a matching prebuilt wheel, pip falls back to
source build (slow, often broken). Pin to CUDA 12.x base instead:

```dockerfile
# Edit line 11 of Dockerfile:
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04
```

The CUDA inside the container can be older than the host driver — driver
595+ supports CUDA 13.x and earlier.

### `docker compose` command not found, only `docker-compose` works

You have the legacy Python-based docker-compose. The commands work the same,
just hyphenated:

```bash
docker-compose -f docker/docker-compose.yml up
```

(Modern Docker bundles `docker compose` as a plugin. The legacy v1 is
end-of-life but still functional.)

### Host's HF cache path differs from `~/.cache/huggingface`

Edit the `volumes:` section of `docker-compose.yml`:

```yaml
volumes:
  - /your/actual/path/to/hf/cache:/cache/huggingface
  - ${HOME}/.cache/vllm:/cache/vllm
```
