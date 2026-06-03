# vLLM Distributed Inference Demo

## Purpose

**This system exists to make the inner workings of vLLM visible.** It is not
a product or a chatbot — it is a teaching and demonstration harness whose
entire goal is to *show, with live telemetry, how vLLM serves large language
models efficiently*. Specifically, it is designed to highlight two of vLLM's
core mechanisms:

- **PagedAttention** — vLLM's KV-cache memory manager, which stores each
  sequence's attention cache in fixed-size physical blocks (like virtual
  memory paging) so that multiple long, divergent sequences can share one
  GPU memory pool with almost no fragmentation.
- **Continuous batching** — vLLM's scheduler, which adds and removes requests
  from the running GPU batch *on every decode step* rather than waiting for a
  whole batch to finish, keeping the GPU saturated even when requests start
  and finish at different times.

Everything in the codebase — the three test cases, the per-token timestamps,
the live scheduler polling, the waterfall and KV-cache charts — is in service
of making these two mechanisms observable, rather than leaving them as
abstract claims.

## Deployment topology: strict client/server separation

The system is **deliberately split across two physical machines** over a LAN,
and the two halves have intentionally different dependencies:

```
   ┌─────────────────────────────┐         HTTP + SSE          ┌────────────────────────────────────────────┐
   │   Windows machine (CLIENT)  │  ───────────────────────>   │      CentOS machine (SERVER)               │
   │                             │      POST /v1/generate      │                                            │
   │   Streamlit dashboard       │  <───────────────────────   │   FastAPI + vLLM 0.20.2 (AsyncLLMEngine)   │
   │   async httpx SSE client    │     token stream (SSE)      │   running on an RTX 3080 (GPU 1, 10 GB)    │
   │   no torch, no vLLM         │                             │   PagedAttention + continuous batching     │
   └─────────────────────────────┘                             └────────────────────────────────────────────┘
```

- **The server runs on CentOS.** It is the only machine that needs a GPU,
  CUDA, `torch`, and `vllm`. It loads the model weights into VRAM, runs the
  vLLM engine, and exposes a small FastAPI surface. It is pinned to GPU 1
  (an RTX 3080, 10 GB) via `CUDA_VISIBLE_DEVICES=1` so the host's other GPU
  stays free. Default endpoint on the demo network: `http://192.168.88.23:8000`.

- **The client runs on Windows.** It has **no GPU dependencies at all** — its
  `client_requirements.txt` is intentionally minimal (`streamlit`, `httpx`,
  `pandas`, `plotly`) with *no torch and no vLLM*. It speaks plain HTTP/SSE to
  the CentOS server and renders the results. This separation is the whole
  point: all heavy inference lives server-side, and any number of lightweight
  clients can drive it remotely.

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

## How the system works (end to end)

This section traces a single demo run from button-press to rendered charts,
so the role of each file is clear.

### 1. The server boots and loads a model (CentOS)

`server/server.py` is a FastAPI app wrapping a vLLM `AsyncLLMEngine`. On
startup it:

1. Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1` **before**
   importing torch/vLLM, pinning all inference to the RTX 3080.

2. Builds the engine for the chosen model (`--model qwen` or `--model mistral`)
   via the FastAPI `lifespan` hook. First-call init cost is ~20–60s while
   weights load into VRAM and CUDA graphs are captured.

3. Attaches a **custom V1 stat logger** (`_MetricsCaptureImpl`). vLLM 0.20's
   V1 architecture runs the scheduler in a separate `EngineCore` subprocess,
   so the server captures the latest `SchedulerStats` in-process via this
   logger's `record()` method. That lets `/metrics` read scheduler state
   synchronously with no per-request IPC.

The model registry currently holds two models: **Qwen2.5-3B-Instruct** (FP16,
unquantized) and **Mistral-7B-Instruct-v0.3-AWQ** (INT4 AWQ quantized). Both
are capped at `max_model_len=8192` and run at `gpu_memory_utilization=0.85`
to fit the 10 GB card.

### 2. The server exposes four endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/v1/generate` | POST | Submit a prompt; tokens stream back as Server-Sent Events. Applies the model's chat template server-side, then drives `engine.generate()` and emits `metadata` → `token`* → `done`/`error` events. Aborts the generation if the client disconnects mid-stream. |
| `/health` | GET | Readiness probe. Reports `status` (`ready`/`loading`/`swapping`/`failed`), the loaded model, uptime, and any swap error. |
| `/metrics` | GET | JSON snapshot of server counters (requests completed/failed, total tokens, 30-second rolling throughput) plus the live scheduler block: running/waiting request counts, GPU block usage, KV-cache usage %, and prefix-cache stats. |
| `/admin/load_model` | POST | Requests an in-process model swap. Returns `202` immediately and runs teardown + reload as a background task; the client polls `/health` until `status='ready'`. |

### 3. The client fires a test case (Windows)

When the user picks a case and clicks run in the Streamlit dashboard
(`client/streamlit_app.py`):

1. The three prompts of the chosen case (`client/test_cases.py`) are fired
   **concurrently** at `/v1/generate` using the async SSE client
   (`client/inference_client.py`, `stream_generate()`).
2. Each streamed token is timestamped on arrival. The `StreamResult`
   dataclass records submit time, first-byte time, first-token time, every
   token event, and completion time — so **TTFT (time-to-first-token), ITL
   (inter-token latency), and throughput fall out for free** without extra
   bookkeeping.
3. In parallel, `poll_metrics()` hits `/metrics` every 200 ms, capturing
   `MetricsSnapshot`s of scheduler state *while the batch is running* — this
   is what makes the batching behavior observable rather than inferred.
4. Connection-phase failures retry up to 3 times with exponential backoff
   (`RetryConfig`). Mid-stream drops are deliberately **not** retried,
   because the server discards an in-flight generation when the TCP
   connection drops — a retry would just waste tokens.

### 4. The dashboard renders the mechanics

After the batch completes, the dashboard plots:

- A **prefill/decode waterfall** showing when each request started producing
  tokens and how the three overlapped in the same batch.
- An **inter-token-latency chart** per request.
- A **scheduler timeline** overlaying the 200 ms `/metrics` polls — running
  vs. waiting request counts and **GPU KV-cache usage climbing and draining**
  as sequences enter and leave the batch.
- A **Compare models** tab that runs the same case on both models in
  sequence, performing an automatic `/admin/load_model` swap between phases,
  and renders the two runs side by side.

All of this is exportable to CSV (per-request, per-token, and scheduler
snapshots) for offline analysis.

## What each test case is meant to highlight

The three cases (`client/test_cases.py`) are not random prompts — each is
constructed to stress a different facet of the scheduler:

| Case | Prompts | What it demonstrates |
|---|---|---|
| **A — Simple / Overlap** | Three short prompts on nearly identical topics with high word overlap | Continuous batching efficiency on uniform short work, and (if active) vLLM's **automatic prefix caching** reducing prefill time for the 2nd and 3rd requests. |
| **B — Medium / Mixed** | Two short networking prompts + one long multi-paragraph outlier | How continuous batching keeps the short requests moving while the long one continues, and how throughput drops as the batch thins out once the short ones finish. |
| **C — Complex / Divergent** | Three divergent ~400-word long-form prompts | The **PagedAttention showcase** — three independent long sequences sharing one physical block pool with no fragmentation. GPU cache usage visibly climbs in the metrics panel. |

## What the whole system is meant to highlight

In one sentence: **this project turns vLLM's two headline optimizations into
something you can watch happen in real time, across a realistic
client/server boundary.**

- **PagedAttention is made visible** through the live KV-cache usage metric
  and the divergent-sequence Case C — you see three long generations coexist
  in one block pool.
- **Continuous batching is made visible** through the scheduler timeline
  (running/waiting counts changing step by step) and the waterfall chart,
  especially in the mixed-workload Case B.
- **The client/server split is made concrete** — a zero-GPU Windows client
  driving a CUDA/vLLM CentOS server over HTTP/SSE shows how production
  inference is actually deployed: heavy compute centralized on a GPU host,
  lightweight remote clients consuming a streaming API.
- **Operational realities are made honest** — model swapping, VRAM teardown
  probes, retry semantics, and the caveats below all reflect the real
  behavior of vLLM 0.20.2 on this hardware rather than an idealized demo.

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

## Running the demo: start the server, then the client

The server (CentOS) must be up and reporting `ready` before the client
(Windows) can do anything useful. Start them in that order.

--------------------------------------------------

### 1. Start the server (CentOS)

```bash
cd ~/Desktop/Pycharm_Projects/vLLM_Inference_Engine
source .venv/bin/activate

# One-time only: populate the HuggingFace cache (idempotent, ~11 GB total).
python server/prefetch_models.py

# Optional but recommended the first time: verify the engine + GPU + model
# stack loads and generates before involving the web layer.
python server/smoke_test.py --only qwen

# Start the server. Pick the boot-time model with --model.
python server/server.py --model qwen
#   --model qwen | mistral      which model to load at boot
#   --host 0.0.0.0              bind address (default; exposes on the LAN)
#   --port 8000                 listen port (default)
#   --gpu-memory-utilization    fraction of the 10 GB card to use (default 0.85)
```

First boot loads weights into VRAM and captures CUDA graphs — expect
**~20–60s** before it's ready. The log line `Engine ready: <repo_id>`
means it's serving.

Confirm from the server host:
```bash
curl -s http://localhost:8000/health | python -m json.tool
# status should be "ready" with the model_key you launched
```

> For the operational (auto-start) paths instead of a manual launch, see the
> systemd and Docker options under **Quick start** above.

### 2. Run the client (Windows)

```powershell
cd C:\Users\\PycharmProjects\vLLM_Client
.venv\Scripts\activate

# One-time only: install the (GPU-free) client dependencies.
pip install -r client_requirements.txt

# Launch the dashboard. Opens in your browser at http://localhost:8501
streamlit run streamlit_app.py
```

In the dashboard sidebar, set **Server** to the CentOS host's address
(default on the demo network: `http://192.168.88.23:8000`). The sidebar
shows a live model-state panel that should read **ready** once it can reach
the server. Then pick a test case (A/B/C) and run it.

### Quick connectivity check

If the client can't reach the server, verify from the Windows machine:
```powershell
curl http://192.168.88.23:8000/health
```
A `ready` response means the path is clear. If it times out, check that the
server bound to `0.0.0.0` (not `127.0.0.1`) and that the CentOS firewall
allows inbound TCP on port 8000.
