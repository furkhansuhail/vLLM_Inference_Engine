#!/usr/bin/env bash
#
# server-wrapper.sh — invoked by systemd's ExecStart.
#
# Activates the project venv and exec's the Python server. We `exec` so
# that the python process replaces this shell — systemd's MainPID then
# points directly at python, which means signals (SIGTERM on stop)
# reach the real server and EngineCore subprocess cleanly.
#
# Model selection comes from VLLM_DEFAULT_MODEL (set in the unit, or
# overridden via server.env). Falls back to "qwen" if unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Sanity-check the venv exists. If someone moves/renames it, this
# fails fast with a clear journalctl message rather than a confusing
# Python ImportError.
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[wrapper] ERROR: venv not found at $VENV_ACTIVATE" >&2
    echo "[wrapper] Expected: cd $PROJECT_ROOT && python -m venv .venv" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

MODEL="${VLLM_DEFAULT_MODEL:-qwen}"

echo "[wrapper] Project root : $PROJECT_ROOT"
echo "[wrapper] Python       : $(which python)"
echo "[wrapper] Model        : $MODEL"
echo "[wrapper] CUDA_VISIBLE_DEVICES (will be set by server.py) : 1"
echo "[wrapper] Starting server..."

# exec → python replaces this shell as the systemd MainPID
exec python server/server.py --model "$MODEL"
