#!/usr/bin/env bash
#
# install.sh — one-shot installer for the vllm-server user systemd unit.
#
# What it does (in order):
#   1. Verifies prerequisites (project root, venv, python).
#   2. Enables linger for the current user so the service survives
#      logout. This is the only step that needs sudo.
#   3. Copies the unit file into ~/.config/systemd/user/.
#   4. Makes the wrapper and preflight scripts executable.
#   5. Reloads systemd and enables the unit so it starts at user login.
#
# Idempotent — safe to re-run after editing any of the files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================================="
echo "  vLLM systemd unit installer"
echo "==================================================================="
echo "  User         : $USER"
echo "  Home         : $HOME"
echo "  Project root : $PROJECT_ROOT"
echo "  Script dir   : $SCRIPT_DIR"
echo "==================================================================="

# ---- 1. Prerequisites -----------------------------------------------------

if [ ! -f "$PROJECT_ROOT/server/server.py" ]; then
    echo "ERROR: $PROJECT_ROOT/server/server.py not found."
    echo "       Check that you're running this from the systemd/ dir of the project."
    exit 1
fi
if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "ERROR: $PROJECT_ROOT/.venv/bin/python not found."
    echo "       Create the venv first: cd $PROJECT_ROOT && python -m venv .venv"
    echo "                              && source .venv/bin/activate && pip install -r server/requirements.txt"
    exit 1
fi
for f in "$SCRIPT_DIR/vllm-server.service" \
         "$SCRIPT_DIR/server-wrapper.sh" \
         "$SCRIPT_DIR/preflight.sh"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

# ---- 2. Linger ------------------------------------------------------------
#
# Without linger, user systemd services stop when the user logs out.
# `loginctl enable-linger` makes them persist across logout/reboot.

if loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
    echo "[install] Linger already enabled for $USER."
else
    echo "[install] Enabling linger for $USER (requires sudo)..."
    sudo loginctl enable-linger "$USER"
fi

# ---- 3. Install unit file -------------------------------------------------

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

UNIT_DST="$SYSTEMD_USER_DIR/vllm-server.service"
cp "$SCRIPT_DIR/vllm-server.service" "$UNIT_DST"
echo "[install] Installed: $UNIT_DST"

# ---- 4. Make scripts executable -------------------------------------------

chmod +x "$SCRIPT_DIR/server-wrapper.sh" "$SCRIPT_DIR/preflight.sh"
echo "[install] Made wrapper + preflight executable."

# ---- 5. Reload + enable ---------------------------------------------------

systemctl --user daemon-reload
systemctl --user enable vllm-server.service

echo
echo "==================================================================="
echo "  Installed successfully."
echo "==================================================================="
echo "  Start:    systemctl --user start vllm-server"
echo "  Status:   systemctl --user status vllm-server"
echo "  Logs:     journalctl --user -u vllm-server -f"
echo "  Stop:     systemctl --user stop vllm-server"
echo "  Restart:  systemctl --user restart vllm-server"
echo
echo "  Change boot-time model:"
echo "    echo 'VLLM_DEFAULT_MODEL=mistral' > $PROJECT_ROOT/server.env"
echo "    systemctl --user restart vllm-server"
echo
echo "  Full operations docs: $SCRIPT_DIR/README.md"
echo "==================================================================="
