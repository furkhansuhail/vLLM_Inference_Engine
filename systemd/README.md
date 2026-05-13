# vLLM server — systemd unit

User-level systemd service that runs the vLLM FastAPI server on CentOS.

**What you get over `python server/server.py --model qwen`:**
- Auto-starts on user login (and survives logout via `loginctl` linger)
- Auto-restarts on crashes, with a 5-second backoff so VRAM has time to drain
- Pre-flight orphan-process reaper — the failure mode that bit us earlier
  ("Free memory on cuda:0 is less than utilization") can't happen because
  the preflight kills lingering EngineCore subprocesses before each start
- All output goes to `journalctl`, so logs are searchable and persistent

---

## Layout

These four files live in `~/Desktop/Pycharm_Projects/vLLM_Inference_Engine/systemd/`:

| File | Purpose |
|---|---|
| `vllm-server.service` | The systemd unit. Installed into `~/.config/systemd/user/` |
| `server-wrapper.sh` | Activates the venv, exec's python with the chosen model |
| `preflight.sh` | Kills orphan vLLM processes, checks VRAM. Runs before each start. |
| `install.sh` | One-shot installer (run once, ever) |

---

## One-time install

```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine/systemd
./install.sh
```

The installer will prompt for sudo **once** to enable lingering for your
user account (so the service survives logout). Everything else is
user-level.

---

## Daily operations

```bash
# Start (only needed the first time after install — systemd starts
# automatically at login afterwards).
systemctl --user start vllm-server

# Check status (running? failed? PID? last few log lines?)
systemctl --user status vllm-server

# Follow logs live (Ctrl-C to exit; doesn't stop the service)
journalctl --user -u vllm-server -f

# Last 100 log lines (e.g. after a crash)
journalctl --user -u vllm-server -n 100

# Only since boot
journalctl --user -u vllm-server -b

# Stop
systemctl --user stop vllm-server

# Restart (e.g. after editing server.py)
systemctl --user restart vllm-server
```

---

## Changing the model loaded at startup

The service reads `VLLM_DEFAULT_MODEL` from `server.env` (in the project
root) on each start. To make `mistral` the boot-time default:

```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine
echo "VLLM_DEFAULT_MODEL=mistral" > server.env
systemctl --user restart vllm-server
```

This affects only the model loaded when the service (re)starts. **Live
swaps via the Streamlit UI's "🔄 Swap to ..." button still work** —
they hit `/admin/load_model` and swap in-process without restarting
the service. The `server.env` value just decides which model you come
back up on next time systemd starts the process from cold.

---

## When to use `systemctl restart` vs `/admin/load_model`

| Situation | Use |
|---|---|
| Normal model swap (qwen ↔ mistral) | `/admin/load_model` via Streamlit UI. Faster (~35s), preserves counters. |
| In-process swap leaked VRAM (server returns "Free memory on cuda:0 is less than utilization" on the second swap) | `systemctl --user restart vllm-server`. Bulletproof — preflight kills any lingering processes, OS frees VRAM, fresh start. |
| Server is in `failed` state per `/health` | `systemctl --user reset-failed vllm-server && systemctl --user start vllm-server` |
| Pulled new code or edited `server.py` | `systemctl --user restart vllm-server` |
| Demo audience wants a clean baseline between cases | Either works; restart is more thorough |

If you want the boot model to change at the same time as the restart:

```bash
echo "VLLM_DEFAULT_MODEL=mistral" > server.env
systemctl --user restart vllm-server
```

---

## Recovering from stuck states

### Service shows `failed` status

```bash
journalctl --user -u vllm-server -n 50    # read the error
systemctl --user reset-failed vllm-server
systemctl --user start vllm-server
```

### "Free memory on cuda:0 is less than utilization" on start

Preflight should catch this, but if it slips through:

```bash
# Find anything still holding GPU memory
nvidia-smi

# Kill everything python-related you don't recognize as belonging to
# someone else's session:
pkill -KILL -u $USER -f "server.py"
pkill -KILL -u $USER -f "EngineCore"

# Wait for VRAM to actually drain
sleep 5
nvidia-smi          # should now show GPU 1 mostly empty

systemctl --user start vllm-server
```

### Someone ran `python server/server.py` manually and is now competing for port 8000

```bash
pkill -f "server/server.py"
systemctl --user restart vllm-server
```

### Service won't start at all, journalctl is empty

```bash
# Try running the wrapper script directly to surface errors
~/Desktop/Pycharm_Projects/vLLM_Inference_Engine/systemd/server-wrapper.sh
```

This bypasses systemd and runs in your terminal so you can see what's
breaking. Common causes: venv path moved, model not in HF cache.

---

## Uninstall

```bash
systemctl --user stop vllm-server
systemctl --user disable vllm-server
rm ~/.config/systemd/user/vllm-server.service
systemctl --user daemon-reload

# Optional: also disable linger
sudo loginctl disable-linger $USER
```

---

## Implementation notes

**Why user-level instead of system-level systemd?** No sudo needed for
everyday ops, GPU device access is already user-owned, easier to iterate
on the unit file. The only system-level operation needed is enabling
linger, which install.sh handles once.

**Why `KillMode=mixed`?** vLLM V1 spawns the EngineCore as a subprocess.
On stop, we want SIGTERM to the main python process (so its signal
handler can shut down EngineCore cleanly), then SIGKILL to anything
still alive after `TimeoutStopSec=30`. `mixed` does exactly that.

**Why `RestartSec=5`?** Empirically the EngineCore subprocess needs 2-5
seconds after SIGKILL before `nvidia-smi` reports the VRAM as free. A
5-second restart delay means by the time the next preflight runs, VRAM
is reclaimed.

**Why not `Type=notify`?** Would be more precise (systemd would know
exactly when the server is ready to serve requests, not just running)
but requires `python-systemd` and a `sd_notify(READY=1)` call inside
init_engine. Not worth the complexity — clients already poll `/health`
to know when the server is actually ready.
