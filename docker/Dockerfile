# vLLM inference server image.
#
# Base matches the host's CUDA runtime (13.2.1) so the container's
# libcudart.so.13 is compatible with the host's NVIDIA driver. Ubuntu
# 24.04 ships Python 3.12 natively, matching the host venv we developed
# server.py against.
#
# This image runs server.py exactly as the systemd-managed setup does;
# the operational UX (env-var picks the model) is intentionally parallel.

FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 (default 'python3' on Ubuntu 24.04), pip, venv.
# ca-certificates for HF Hub TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install into a venv to dodge PEP 668 system-pip friction on Ubuntu 24.04.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python deps first for better layer caching — only re-runs when
# requirements.txt changes, not on every server.py edit.
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Server source.
COPY server/ /app/server/

# Entrypoint script handles VLLM_DEFAULT_MODEL env-var → CLI translation,
# matching the systemd wrapper's behaviour. Operations are identical:
# write VLLM_DEFAULT_MODEL=mistral in either a systemd EnvironmentFile
# or a docker .env file and the server picks it up the same way.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---- Runtime environment --------------------------------------------------

# Shared cache root.
#   /cache/huggingface  ←  models (pre-downloaded on host, bind-mounted)
#   /cache/vllm         ←  CUDA-graph compile cache (saves ~30s cold start)
# server.py uses HF_HOME implicitly via huggingface-hub; we set both.
ENV XDG_CACHE_HOME=/cache
ENV HF_HOME=/cache/huggingface

# GPU selection. The container only gets one GPU (selected via
# `docker compose --gpus device=1` on the host), which appears inside
# the container at index 0. server.py's os.environ.setdefault won't
# override these because they're already set.
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Unbuffered stdout/stderr so `docker logs` shows output in real time
# instead of waiting for buffer flushes.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
