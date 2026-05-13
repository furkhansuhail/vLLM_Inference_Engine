# vLLM Distributed Inference Demo

Demonstrates vLLM's **PagedAttention** and **Continuous Batching** through
three test cases, with strict client/server separation:

```
   Windows client (Streamlit dashboard)  ⟷  CentOS server (FastAPI + vLLM 0.20.2 on RTX 3080)
```

Three test cases (A/B/C) fire three prompts concurrently and stream tokens
back over Server-Sent Events. The dashboard captures per-token timing,
inter-token latencies, and live `/metrics` polls of scheduler state to
make the mechanics of modern LLM serving visible — not just produce text.

For the original architectural rationale, see `HANDOFF_1.md` (the design
document this project was built from). This README is the current
operational snapshot.

---

## Directory structure

```
vLLM_Inference_Engine/
├── .dockerignore                       # excludes from docker build context
├── .gitignore                          # excludes from git
├── README.md                           # this file
├── HANDOFF_1.md                        # original design doc (read for rationale)
│
├── server/                             # → deployed on CentOS
│   ├── server.py                       # FastAPI + vLLM AsyncLLM + V1 stat logger
│   │                                   #   + /admin/load_model endpoint
│   ├── requirements.txt
│   ├── prefetch_models.py              # one-time HF cache populator
│   └── smoke_test.py                   # standalone vLLM + GPU verification
│
├── client/                             # → deployed on Windows
│   ├── streamlit_app.py                # dashboard: single-run + compare tabs
│   ├── inference_client.py             # async SSE client + retry + metrics polling
│   ├── test_cases.py                   # Case A/B/C prompt definitions
│   └── client_requirements.txt
│
├── setup/                              # → host-side hardware sanity checks
│   └── hardware_sanity_check.py
│
├── System_Architecture_Design/
│   └── SystemArchitecture              # architecture diagram source
│
├── systemd/                            # operational: native-host deployment
│   ├── vllm-server.service             # user-level systemd unit
│   ├── server-wrapper.sh               # exec wrapper (activates venv + runs python)
│   ├── preflight.sh                    # kills orphan EngineCore processes, checks VRAM
│   ├── install.sh                      # one-shot installer
│   └── README.md                       # operations guide
│
└── docker/                             # operational: containerized deployment
    ├── Dockerfile                      # CUDA 13.2 + Python 3.12 + vLLM 0.20.2
    ├── entrypoint.sh                   # env-var → CLI translation
    ├── docker-compose.yml              # GPU passthrough + cache volumes + IPC
    └── README.md                       # build/run/troubleshoot guide
```

---

## What's implemented

| Feature | Status | Where |
|---|---|---|
| Three test cases (A/B/C) firing concurrent prompts over SSE | ✅ | `client/test_cases.py`, `client/streamlit_app.py` |
| V1 stat logger capturing scheduler state | ✅ | `server/server.py` (`_MetricsCaptureImpl`) |
| `/admin/load_model` endpoint for in-process model swap | ✅ | `server/server.py` |
| Live `/metrics` polling during batch (200 ms intervals) | ✅ | `client/inference_client.py` (`poll_metrics`), `client/streamlit_app.py` |
| Connection-phase retry (3 attempts, exponential backoff) | ✅ | `client/inference_client.py` (`RetryConfig`) |
| CSV export of per-request, per-token, scheduler snapshots | ✅ | `client/streamlit_app.py` |
| Model swap UI (sidebar dropdown + Swap button) | ✅ | `client/streamlit_app.py` |
| Side-by-side model comparison view (tab) | ✅ | `client/streamlit_app.py` |
| systemd user unit with orphan-reaper preflight | ✅ | `systemd/` |
| Docker image + compose with GPU passthrough | ✅ | `docker/` |
| TLS / authentication | ⏭ deferred (LAN-only demo) | — |

---

## Quick start — three deployment paths

The three options run the same `server.py` and are mutually exclusive at
runtime (all use port 8000). Pick one:

### 1. Manual (development)

```bash
# CentOS
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine
source .venv/bin/activate
python server/server.py --model qwen
```

```powershell
# Windows
cd C:\Users\<user>\PycharmProjects\vLLM_Client
.venv\Scripts\activate
streamlit run streamlit_app.py
```

### 2. systemd (operational, native)

One-time install:
```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine/systemd
chmod +x *.sh
./install.sh
```

Daily use:
```bash
systemctl --user start vllm-server
systemctl --user status vllm-server
journalctl --user -u vllm-server -f
```

Full operations guide: `systemd/README.md`.

### 3. Docker (operational, containerized)

One-time install (Docker Engine + NVIDIA Container Toolkit on host).

Build + run:
```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f
```

Full guide: `docker/README.md`.

---

## Operations cheat sheet

| Action | Command |
|---|---|
| Switch model at runtime (any deployment) | Streamlit sidebar → dropdown → 🔄 Swap |
| Switch boot-time model (systemd) | `echo "VLLM_DEFAULT_MODEL=mistral" > server.env && systemctl --user restart vllm-server` |
| Switch boot-time model (Docker) | `echo "VLLM_DEFAULT_MODEL=mistral" > docker/.env && docker compose -f docker/docker-compose.yml restart` |
| Side-by-side comparison | Streamlit → "Compare models" tab → Start comparison |
| Follow server logs (systemd) | `journalctl --user -u vllm-server -f` |
| Follow server logs (Docker) | `docker compose -f docker/docker-compose.yml logs -f` |
| Check loaded model + uptime | `curl -s http://192.168.88.23:8000/health \| python -m json.tool` |
| Live scheduler stats | `curl -s http://192.168.88.23:8000/metrics \| python -m json.tool` |

---

## Known caveats

1. **In-process model swap may leak VRAM** depending on how vLLM 0.20.2's
   AsyncLLM.shutdown() behaves with the specific EngineCore subprocess
   on your build. The server logs four `VRAM` probes around each
   teardown (`pre-teardown`, `post-teardown`, `pre-init`, `post-init`)
   so leaks are visible. If they happen, switch to the systemd or
   Docker restart pattern for model swaps — both kill the process and
   let the OS reclaim VRAM.

2. **Wi-Fi latency floor** of ~80 ms on the demo network dominates TTFT
   measurements. Plug a USB-Ethernet dongle into the same switch as
   the Windows client for clean numbers (sub-2 ms RTT).

3. **Mid-stream connection drops are NOT auto-retried.** Only the
   initial connection phase retries (3 attempts on
   ConnectError / 502 / 503 / 504). Mid-stream stalls or drops surface
   as errors because the server discards the generation when our TCP
   connection drops — a retry would only waste tokens.

4. **Compare mode swap dependency.** Step 6 (side-by-side comparison)
   cascades from step 7 (in-process model swap). If the swap leaks VRAM,
   the second phase of compare mode fails at engine re-init. The
   Streamlit state machine surfaces a FAILED state with the swap_error
   message; the fix path is the same as caveat #1 above.

---

## Where this came from

This project was built incrementally from the design captured in
`HANDOFF_1.md`. The handoff documented the architecture and a list of
"missing" features; each was implemented and verified at the syntax
level. The runtime verification is on the actual hardware and is the
user's responsibility — every step's commit-and-test instructions are
preserved in the chat history that produced this codebase.
