#!/bin/bash
#
# entrypoint.sh — Docker container entrypoint for the vLLM server.
#
# Translates the VLLM_DEFAULT_MODEL environment variable into a --model
# CLI argument for server.py. This mirrors the systemd wrapper's
# behaviour so operations are consistent: write VLLM_DEFAULT_MODEL into
# a docker .env file or an EnvironmentFile and the server picks the
# same model on either deployment path.
#
# Extra CLI args passed to `docker run` (or `docker compose run`) are
# forwarded to server.py after --model, so you can override --host,
# --port, --gpu-memory-utilization without rebuilding.

set -e

MODEL="${VLLM_DEFAULT_MODEL:-qwen}"

echo "[docker-entrypoint] Starting vLLM server"
echo "[docker-entrypoint]   Model:                 $MODEL"
echo "[docker-entrypoint]   CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "[docker-entrypoint]   XDG_CACHE_HOME:        $XDG_CACHE_HOME"
echo "[docker-entrypoint]   Python:                $(which python)"

# Exec → python replaces the shell, so signals from docker (SIGTERM on
# stop) reach the server directly and the EngineCore subprocess gets
# cleaned up cleanly.
exec python /app/server/server.py --model "$MODEL" --host 0.0.0.0 "$@"
