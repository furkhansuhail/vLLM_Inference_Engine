#!/usr/bin/env bash
#
# preflight.sh — invoked by systemd's ExecStartPre before every start.
#
# Two jobs:
#   1. Reap orphan vLLM processes from a previous crashed/Ctrl-C'd run.
#      The EngineCore subprocess (vLLM V1) can outlive its parent if
#      shutdown isn't clean, holding ~5-8 GiB of VRAM. Without this,
#      the next start fails with the "Free memory on cuda:0 is less
#      than utilization" error we hit earlier.
#   2. Sanity-check VRAM is available on GPU 1 (the RTX 3080).
#      We don't fail on insufficient VRAM — just warn loudly. The
#      service's Restart=on-failure will retry, and the 5s gap between
#      restarts often releases enough VRAM to succeed on attempt 2.

set -euo pipefail

echo "[preflight] Running as: $(whoami) on $(hostname)"
echo "[preflight] PWD: $PWD"

# ---- 1. Kill orphans ------------------------------------------------------
#
# Target pattern is narrow: processes owned by the current user with
# either the project path or "EngineCore" in the command line. This
# avoids accidentally killing unrelated python services.

PATTERN='vLLM_Inference_Engine|EngineCore'

ORPHANS=$(pgrep -u "$USER" -f "$PATTERN" 2>/dev/null || true)
# Filter out our own PID (preflight is itself a process, though shouldn't
# match the pattern anyway — defensive belt-and-suspenders).
ORPHANS=$(echo "$ORPHANS" | grep -v "^$$\$" || true)

if [ -n "$ORPHANS" ]; then
    echo "[preflight] Found orphan PIDs (matching '$PATTERN'):"
    # shellcheck disable=SC2086  # intentional word-splitting: ORPHANS is a list of PIDs
    ps -o pid,etime,cmd -p $ORPHANS 2>/dev/null || true
    echo "[preflight] Sending SIGTERM..."
    # shellcheck disable=SC2086
    kill -TERM $ORPHANS 2>/dev/null || true
    sleep 2

    REMAINING=$(pgrep -u "$USER" -f "$PATTERN" 2>/dev/null || true)
    REMAINING=$(echo "$REMAINING" | grep -v "^$$\$" || true)
    if [ -n "$REMAINING" ]; then
        echo "[preflight] Still alive after SIGTERM, force-killing..."
        # shellcheck disable=SC2086
        kill -KILL $REMAINING 2>/dev/null || true
        sleep 1
    fi
    echo "[preflight] Orphan cleanup complete."
else
    echo "[preflight] No orphan vLLM processes found."
fi

# ---- 2. VRAM sanity check -------------------------------------------------
#
# vLLM at gpu_memory_utilization=0.85 needs ~8.2 GiB free on a 9.65 GiB
# card. We warn (not fail) below 9000 MiB to give a margin.

if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --id=1 --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [ -n "$FREE_MB" ]; then
        if [ "$FREE_MB" -lt 9000 ]; then
            echo "[preflight] WARNING: only ${FREE_MB} MiB free on GPU 1; vLLM may fail to start."
            echo "[preflight] Current GPU compute processes:"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
        else
            echo "[preflight] GPU 1: ${FREE_MB} MiB free — OK."
        fi
    else
        echo "[preflight] nvidia-smi did not return memory info for GPU 1; skipping check."
    fi
else
    echo "[preflight] nvidia-smi not in PATH; skipping VRAM check."
fi

echo "[preflight] Done."
exit 0
