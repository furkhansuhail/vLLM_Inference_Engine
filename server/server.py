#!/usr/bin/env python3
"""
server.py — FastAPI wrapper around vLLM AsyncLLMEngine with SSE streaming.

Endpoints:
    POST /v1/generate   Submit a prompt, receive tokens as Server-Sent Events.
    GET  /health        Readiness probe + loaded model info.
    GET  /metrics       JSON snapshot of server + engine counters.

Model is fixed at startup via --model. To switch, restart the server.
<<<<<<< Updated upstream
Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES with PCI bus ordering.
=======
Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES.
>>>>>>> Stashed changes

Usage:
    python server.py --model qwen
    python server.py --model mistral --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

# Must set BEFORE any CUDA-touching import (torch, vllm).
<<<<<<< Updated upstream
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID forces CUDA to index GPUs in the same order
# nvidia-smi shows them (by PCI slot), instead of CUDA's default "fastest
# first" ranking. Without this, on machines with mixed GPUs (e.g. RTX 4070
# Laptop + RTX 3080), `CUDA_VISIBLE_DEVICES=1` may select the *wrong* card.
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # RTX 3080
=======
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
>>>>>>> Stashed changes

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vllm-server")


# ---- Model registry (must match smoke_test.py / prefetch_models.py) --------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    quantization: Optional[str]
    max_model_len: int


MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        quantization=None,
        max_model_len=8192,
    ),
    "mistral": ModelSpec(
        key="mistral",
        repo_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        quantization="awq",
        max_model_len=8192,
    ),
}


# ---- Schemas ---------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt (chat template applied server-side).")
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    request_id: Optional[str] = Field(None, description="Client-supplied ID. Server generates if omitted.")


class HealthResponse(BaseModel):
    status: str               # "ready" | "loading"
    model_key: Optional[str]
    model_repo: Optional[str]
    uptime_seconds: float


# ---- Server state ----------------------------------------------------------

class ServerState:
    """Mutable singleton holding engine + counters. Accessed from handlers."""
<<<<<<< Updated upstream
    engine = None                                 # vllm.v1.engine.async_llm.AsyncLLM
    tokenizer = None
    model_spec: Optional[ModelSpec] = None
    started_at: float = 0.0
    # Captured V1 stat-logger instance (set during init_engine via factory).
    # Untyped Optional[object] to dodge an import-order issue: StatLoggerBase
    # can't be imported at module load time (would touch CUDA before env
    # vars apply).
    metrics_capture: Optional[object] = None
=======
    engine = None                                 # vllm.AsyncLLMEngine
    tokenizer = None
    model_spec: Optional[ModelSpec] = None
    started_at: float = 0.0
>>>>>>> Stashed changes

    # Cheap counters surfaced via /metrics
    requests_completed: int = 0
    requests_failed: int = 0
    tokens_generated_total: int = 0
    # Rolling window: list of (completion_time, token_count, duration_seconds)
    recent_completions: list[tuple[float, int, float]] = []


state = ServerState()


# ---- Engine init -----------------------------------------------------------

async def init_engine(spec: ModelSpec, gpu_memory_utilization: float) -> None:
    """Build the AsyncLLMEngine. First-call cost: 20-60s."""
    log.info("Initializing vLLM engine: model=%s quant=%s max_len=%d",
             spec.repo_id, spec.quantization, spec.max_model_len)

<<<<<<< Updated upstream
    # Deferred imports keep --help instant and ensure CUDA env vars took effect.
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from vllm.v1.metrics.loggers import StatLoggerBase
    from vllm.v1.metrics.stats import SchedulerStats, IterationStats

    # ---- V1 stat-logger plumbing -------------------------------------------
    #
    # vLLM V1 runs its scheduler in a separate EngineCore process. The only
    # supported way to observe scheduler state from the front-end is to
    # register a StatLoggerBase whose record() method is called on every
    # engine step with the current SchedulerStats / IterationStats. We stash
    # the latest snapshot on the logger instance so /metrics can read it
    # synchronously — no IPC at request time.
    #
    # Known V1 quirk (vllm-project/vllm#20175): StatLoggerBase.log() is never
    # called automatically in V1 — only record() is. All capture happens in
    # record().

    class _MetricsCaptureImpl(StatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0):
            self.vllm_config = vllm_config
            self.engine_index = engine_index
            self.last_scheduler_stats: Optional[SchedulerStats] = None
            self.last_iteration_stats: Optional[IterationStats] = None
            self.last_update_monotonic: float = 0.0
            self.num_gpu_blocks: Optional[int] = None
            self.block_size: Optional[int] = None

        def record(self, scheduler_stats, iteration_stats,
                   mm_cache_stats=None, engine_idx: int = 0) -> None:
            if scheduler_stats is not None:
                self.last_scheduler_stats = scheduler_stats
            if iteration_stats is not None:
                self.last_iteration_stats = iteration_stats
            self.last_update_monotonic = time.monotonic()
            # Lazy-capture the static block-pool size on the first tick that
            # has it. cache_config.num_gpu_blocks is only populated after
            # KV-cache profiling completes, which can be slightly post-init.
            if self.num_gpu_blocks is None:
                try:
                    cc = self.vllm_config.cache_config
                    if cc.num_gpu_blocks:
                        self.num_gpu_blocks = int(cc.num_gpu_blocks)
                        self.block_size = int(cc.block_size)
                except Exception:
                    pass

        def log_engine_initialized(self) -> None:
            log.info("V1 stat logger attached (engine_index=%d)", self.engine_index)

    def _stat_logger_factory(vllm_config, engine_index: int):
        cap = _MetricsCaptureImpl(vllm_config, engine_index)
        state.metrics_capture = cap
        return cap
=======
    # Deferred import keeps --help instant and ensures CUDA_VISIBLE_DEVICES took effect.
    from vllm import AsyncLLMEngine, AsyncEngineArgs
>>>>>>> Stashed changes

    engine_args = AsyncEngineArgs(
        model=spec.repo_id,
        quantization=spec.quantization,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=spec.max_model_len,
        enforce_eager=False,             # use CUDA graphs
        trust_remote_code=False,
<<<<<<< Updated upstream
    )

    state.engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        stat_loggers=[_stat_logger_factory],
    )
    state.tokenizer = state.engine.get_tokenizer()
=======
        disable_log_requests=True,       # we do our own request logging
    )

    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    state.tokenizer = await state.engine.get_tokenizer()
>>>>>>> Stashed changes
    state.model_spec = spec
    state.started_at = time.time()
    log.info("Engine ready: %s", spec.repo_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    spec: ModelSpec = app.state.model_spec
    gpu_mem: float = app.state.gpu_memory_utilization
    await init_engine(spec, gpu_mem)
    try:
        yield
    finally:
        log.info("Shutting down.")


app = FastAPI(lifespan=lifespan, title="vLLM Inference Server")


# ---- /health ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    if state.engine is None or state.model_spec is None:
        return HealthResponse(status="loading", model_key=None, model_repo=None, uptime_seconds=0.0)
    return HealthResponse(
        status="ready",
        model_key=state.model_spec.key,
        model_repo=state.model_spec.repo_id,
        uptime_seconds=time.time() - state.started_at,
    )


# ---- /metrics --------------------------------------------------------------

def _scheduler_snapshot() -> dict:
<<<<<<< Updated upstream
    """Read the latest captured scheduler stats from our V1 StatLogger.

    Returns roughly the V0 shape where V1 has an analog, plus extras V1
    gives us (prefix-cache stats, step counter, age of last update).
    num_swapped_requests is omitted because V1 doesn't use CPU swap — it
    preempts and recomputes instead.
    """
    cap = state.metrics_capture
    if cap is None:
        return {"error": "metrics capture not initialized"}
    s = getattr(cap, "last_scheduler_stats", None)
    if s is None:
        # Engine is idle and hasn't ticked since startup. Send a request and
        # this becomes populated on the next engine step.
        return {
            "status": "idle (no scheduler stats yet — send a request)",
            "stats_age_seconds": None,
        }

    age = time.monotonic() - cap.last_update_monotonic
    usage_frac = float(getattr(s, "kv_cache_usage", 0.0))

    # Derive absolute block counts when the pool size has been captured.
    total = cap.num_gpu_blocks
    if total is not None:
        used = int(round(usage_frac * total))
        free = total - used
    else:
        used = free = None

    # Surface prefix-cache stats as a raw dict of scalars — the field names
    # of PrefixCacheStats aren't documented as stable, so we filter to safe
    # types and let downstream code pick what it needs.
    pcs = getattr(s, "prefix_cache_stats", None)
    prefix_cache_block: Optional[dict] = None
    if pcs is not None:
        try:
            prefix_cache_block = {
                k: v for k, v in vars(pcs).items()
                if not k.startswith("_")
                and isinstance(v, (int, float, str, bool, type(None)))
            }
        except Exception:
            prefix_cache_block = None

    return {
        "num_running_requests": int(getattr(s, "num_running_reqs", 0)),
        "num_waiting_requests": int(getattr(s, "num_waiting_reqs", 0)),
        "gpu_blocks_total": total,
        "gpu_blocks_used": used,
        "gpu_blocks_free": free,
        "gpu_cache_usage_perc": round(100.0 * usage_frac, 1),
        "block_size": cap.block_size,
        "step_counter": int(getattr(s, "step_counter", 0)),
        "stats_age_seconds": round(age, 3),
        "prefix_cache_stats": prefix_cache_block,
    }
=======
    """Best-effort read of vLLM's internal scheduler / block-manager state.

    The internal API path is version-sensitive — wrap defensively so a vLLM
    upgrade doesn't break the metrics endpoint entirely.
    """
    try:
        sched = state.engine.engine.scheduler[0]  # type: ignore[attr-defined]
        bm = sched.block_manager
        total = bm.get_num_total_gpu_blocks()
        free = bm.get_num_free_gpu_blocks()
        used = total - free
        return {
            "num_running_requests": len(sched.running),
            "num_waiting_requests": len(sched.waiting),
            "num_swapped_requests": len(sched.swapped),
            "gpu_blocks_total": total,
            "gpu_blocks_used": used,
            "gpu_blocks_free": free,
            "gpu_cache_usage_perc": round(100.0 * used / total, 1) if total else 0.0,
        }
    except Exception as e:
        return {"error": f"scheduler stats unavailable: {type(e).__name__}: {e}"}
>>>>>>> Stashed changes


@app.get("/metrics")
async def metrics():
    now = time.time()
    spec = state.model_spec

    # Rolling throughput over the last 30 seconds of completed requests.
    cutoff = now - 30.0
    state.recent_completions = [c for c in state.recent_completions if c[0] >= cutoff]
    if state.recent_completions:
        total_tokens = sum(c[1] for c in state.recent_completions)
        total_time = sum(c[2] for c in state.recent_completions)
        throughput = total_tokens / total_time if total_time > 0 else 0.0
    else:
        throughput = 0.0

    return {
        "model_key": spec.key if spec else None,
        "model_repo": spec.repo_id if spec else None,
        "uptime_seconds": (now - state.started_at) if state.started_at else 0.0,
        "requests_completed": state.requests_completed,
        "requests_failed": state.requests_failed,
        "tokens_generated_total": state.tokens_generated_total,
        "throughput_tokens_per_sec_30s": round(throughput, 2),
        "scheduler": _scheduler_snapshot(),
    }


# ---- /v1/generate (SSE) ----------------------------------------------------

def _sse_event(payload: dict) -> bytes:
    """Format a dict as a single SSE event frame."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@app.post("/v1/generate")
async def generate(req: GenerateRequest, http_request: Request):
    if state.engine is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    request_id = req.request_id or str(uuid.uuid4())
    log.info("→ %s: max_tokens=%d temp=%.2f top_p=%.2f",
             request_id, req.max_tokens, req.temperature, req.top_p)

    # Apply the model's chat template so the model sees properly-formatted input.
    messages = [{"role": "user", "content": req.prompt}]
    templated = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    from vllm import SamplingParams
    sampling = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    async def event_stream() -> AsyncGenerator[bytes, None]:
        t_start = time.perf_counter()
        previous_text = ""
        token_count = 0
        finish_reason: Optional[str] = None

        # Open the stream with a metadata frame so the client gets the server's
        # received-at timestamp for clock-skew alignment in telemetry.
        yield _sse_event({
            "type": "metadata",
            "request_id": request_id,
            "model_key": state.model_spec.key,
            "server_received_at": time.time(),
        })

        try:
            results = state.engine.generate(templated, sampling, request_id)
            async for output in results:
                # Client closed the connection — free the GPU slot immediately.
                if await http_request.is_disconnected():
                    log.info("← %s: client disconnected, aborting", request_id)
                    await state.engine.abort(request_id)
                    return

                completion = output.outputs[0]
                # vLLM yields cumulative text; emit only the delta.
                new_text = completion.text[len(previous_text):]
                previous_text = completion.text
                token_count = len(completion.token_ids)
                finish_reason = completion.finish_reason

                if new_text:
                    yield _sse_event({
                        "type": "token",
                        "request_id": request_id,
                        "text": new_text,
                        "token_count": token_count,
                    })

                if finish_reason is not None:
                    duration = time.perf_counter() - t_start
                    yield _sse_event({
                        "type": "done",
                        "request_id": request_id,
                        "finish_reason": finish_reason,
                        "total_tokens": token_count,
                        "duration_seconds": round(duration, 4),
                    })
                    state.requests_completed += 1
                    state.tokens_generated_total += token_count
                    state.recent_completions.append((time.time(), token_count, duration))
                    tps = token_count / duration if duration > 0 else 0
                    log.info("← %s: %d tokens in %.2fs (%.1f tok/s, finish=%s)",
                             request_id, token_count, duration, tps, finish_reason)
                    return

        except asyncio.CancelledError:
            log.info("← %s: cancelled", request_id)
            await state.engine.abort(request_id)
            raise
        except Exception as e:
            log.exception("← %s: failed", request_id)
            state.requests_failed += 1
            yield _sse_event({
                "type": "error",
                "request_id": request_id,
                "message": f"{type(e).__name__}: {e}",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- Entry point -----------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM FastAPI server with SSE streaming.")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True,
                        help="Which model to load.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind interface (0.0.0.0 = all). Default: 0.0.0.0.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="Fraction of GPU memory vLLM may use (default 0.85).")
    args = parser.parse_args()

    spec = MODELS[args.model]
    app.state.model_spec = spec
    app.state.gpu_memory_utilization = args.gpu_memory_utilization

    import uvicorn
    log.info("Starting server on %s:%d, model=%s (%s)",
             args.host, args.port, spec.key, spec.repo_id)
<<<<<<< Updated upstream
    log.info("CUDA_DEVICE_ORDER=%s  CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_DEVICE_ORDER"),
             os.environ.get("CUDA_VISIBLE_DEVICES"))
=======
    # log_level="info" gives uvicorn its own access logs (one line per request).
>>>>>>> Stashed changes
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
<<<<<<< Updated upstream
    raise SystemExit(main())


# #!/usr/bin/env python3
# """
# server.py — FastAPI wrapper around vLLM AsyncLLMEngine with SSE streaming.
#
# Endpoints:
#     POST /v1/generate   Submit a prompt, receive tokens as Server-Sent Events.
#     GET  /health        Readiness probe + loaded model info.
#     GET  /metrics       JSON snapshot of server + engine counters.
#
# Model is fixed at startup via --model. To switch, restart the server.
# Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES with PCI bus ordering.
#
# Usage:
#     python server.py --model qwen
#     python server.py --model mistral --host 0.0.0.0 --port 8000
# """
#
# from __future__ import annotations
#
# # Must set BEFORE any CUDA-touching import (torch, vllm).
# #
# # CUDA_DEVICE_ORDER=PCI_BUS_ID forces CUDA to index GPUs in the same order
# # nvidia-smi shows them (by PCI slot), instead of CUDA's default "fastest
# # first" ranking. Without this, on machines with mixed GPUs (e.g. RTX 4070
# # Laptop + RTX 3080), `CUDA_VISIBLE_DEVICES=1` may select the *wrong* card.
# import os
# os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # RTX 3080
#
# import argparse
# import asyncio
# import json
# import logging
# import time
# import uuid
# from contextlib import asynccontextmanager
# from dataclasses import dataclass
# from typing import AsyncGenerator, Optional
#
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, Field
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )
# log = logging.getLogger("vllm-server")
#
#
# # ---- Model registry (must match smoke_test.py / prefetch_models.py) --------
#
# @dataclass(frozen=True)
# class ModelSpec:
#     key: str
#     repo_id: str
#     quantization: Optional[str]
#     max_model_len: int
#
#
# MODELS: dict[str, ModelSpec] = {
#     "qwen": ModelSpec(
#         key="qwen",
#         repo_id="Qwen/Qwen2.5-3B-Instruct",
#         quantization=None,
#         max_model_len=8192,
#     ),
#     "mistral": ModelSpec(
#         key="mistral",
#         repo_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
#         quantization="awq",
#         max_model_len=8192,
#     ),
# }
#
#
# # ---- Schemas ---------------------------------------------------------------
#
# class GenerateRequest(BaseModel):
#     prompt: str = Field(..., description="User prompt (chat template applied server-side).")
#     max_tokens: int = Field(256, ge=1, le=4096)
#     temperature: float = Field(0.7, ge=0.0, le=2.0)
#     top_p: float = Field(0.9, ge=0.0, le=1.0)
#     request_id: Optional[str] = Field(None, description="Client-supplied ID. Server generates if omitted.")
#
#
# class HealthResponse(BaseModel):
#     status: str               # "ready" | "loading"
#     model_key: Optional[str]
#     model_repo: Optional[str]
#     uptime_seconds: float
#
#
# # ---- Server state ----------------------------------------------------------
#
# class ServerState:
#     """Mutable singleton holding engine + counters. Accessed from handlers."""
#     engine = None                                 # vllm.AsyncLLMEngine
#     tokenizer = None
#     model_spec: Optional[ModelSpec] = None
#     started_at: float = 0.0
#
#     # Cheap counters surfaced via /metrics
#     requests_completed: int = 0
#     requests_failed: int = 0
#     tokens_generated_total: int = 0
#     # Rolling window: list of (completion_time, token_count, duration_seconds)
#     recent_completions: list[tuple[float, int, float]] = []
#
#
# state = ServerState()
#
#
# # ---- Engine init -----------------------------------------------------------
#
# async def init_engine(spec: ModelSpec, gpu_memory_utilization: float) -> None:
#     """Build the AsyncLLMEngine. First-call cost: 20-60s."""
#     log.info("Initializing vLLM engine: model=%s quant=%s max_len=%d",
#              spec.repo_id, spec.quantization, spec.max_model_len)
#
#     # Deferred import keeps --help instant and ensures CUDA env vars took effect.
#     from vllm import AsyncLLMEngine, AsyncEngineArgs
#
#     engine_args = AsyncEngineArgs(
#         model=spec.repo_id,
#         quantization=spec.quantization,
#         dtype="auto",
#         gpu_memory_utilization=gpu_memory_utilization,
#         max_model_len=spec.max_model_len,
#         enforce_eager=False,             # use CUDA graphs
#         trust_remote_code=False,
#     )
#
#     state.engine = AsyncLLMEngine.from_engine_args(engine_args)
#     # state.tokenizer = await state.engine.get_tokenizer()
#     state.tokenizer = state.engine.get_tokenizer()
#     state.model_spec = spec
#     state.started_at = time.time()
#     log.info("Engine ready: %s", spec.repo_id)
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     spec: ModelSpec = app.state.model_spec
#     gpu_mem: float = app.state.gpu_memory_utilization
#     await init_engine(spec, gpu_mem)
#     try:
#         yield
#     finally:
#         log.info("Shutting down.")
#
#
# app = FastAPI(lifespan=lifespan, title="vLLM Inference Server")
#
#
# # ---- /health ---------------------------------------------------------------
#
# @app.get("/health", response_model=HealthResponse)
# async def health():
#     if state.engine is None or state.model_spec is None:
#         return HealthResponse(status="loading", model_key=None, model_repo=None, uptime_seconds=0.0)
#     return HealthResponse(
#         status="ready",
#         model_key=state.model_spec.key,
#         model_repo=state.model_spec.repo_id,
#         uptime_seconds=time.time() - state.started_at,
#     )
#
#
# # ---- /metrics --------------------------------------------------------------
#
# def _scheduler_snapshot() -> dict:
#     """Best-effort read of vLLM's internal scheduler / block-manager state.
#
#     The internal API path is version-sensitive — wrap defensively so a vLLM
#     upgrade doesn't break the metrics endpoint entirely.
#     """
#     try:
#         sched = state.engine.engine.scheduler[0]  # type: ignore[attr-defined]
#         bm = sched.block_manager
#         total = bm.get_num_total_gpu_blocks()
#         free = bm.get_num_free_gpu_blocks()
#         used = total - free
#         return {
#             "num_running_requests": len(sched.running),
#             "num_waiting_requests": len(sched.waiting),
#             "num_swapped_requests": len(sched.swapped),
#             "gpu_blocks_total": total,
#             "gpu_blocks_used": used,
#             "gpu_blocks_free": free,
#             "gpu_cache_usage_perc": round(100.0 * used / total, 1) if total else 0.0,
#         }
#     except Exception as e:
#         return {"error": f"scheduler stats unavailable: {type(e).__name__}: {e}"}
#
#
# @app.get("/metrics")
# async def metrics():
#     now = time.time()
#     spec = state.model_spec
#
#     # Rolling throughput over the last 30 seconds of completed requests.
#     cutoff = now - 30.0
#     state.recent_completions = [c for c in state.recent_completions if c[0] >= cutoff]
#     if state.recent_completions:
#         total_tokens = sum(c[1] for c in state.recent_completions)
#         total_time = sum(c[2] for c in state.recent_completions)
#         throughput = total_tokens / total_time if total_time > 0 else 0.0
#     else:
#         throughput = 0.0
#
#     return {
#         "model_key": spec.key if spec else None,
#         "model_repo": spec.repo_id if spec else None,
#         "uptime_seconds": (now - state.started_at) if state.started_at else 0.0,
#         "requests_completed": state.requests_completed,
#         "requests_failed": state.requests_failed,
#         "tokens_generated_total": state.tokens_generated_total,
#         "throughput_tokens_per_sec_30s": round(throughput, 2),
#         "scheduler": _scheduler_snapshot(),
#     }
#
#
# # ---- /v1/generate (SSE) ----------------------------------------------------
#
# def _sse_event(payload: dict) -> bytes:
#     """Format a dict as a single SSE event frame."""
#     return f"data: {json.dumps(payload)}\n\n".encode("utf-8")
#
#
# @app.post("/v1/generate")
# async def generate(req: GenerateRequest, http_request: Request):
#     if state.engine is None or state.tokenizer is None:
#         raise HTTPException(status_code=503, detail="Engine not ready")
#
#     request_id = req.request_id or str(uuid.uuid4())
#     log.info("→ %s: max_tokens=%d temp=%.2f top_p=%.2f",
#              request_id, req.max_tokens, req.temperature, req.top_p)
#
#     # Apply the model's chat template so the model sees properly-formatted input.
#     messages = [{"role": "user", "content": req.prompt}]
#     templated = state.tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True,
#     )
#
#     from vllm import SamplingParams
#     sampling = SamplingParams(
#         temperature=req.temperature,
#         top_p=req.top_p,
#         max_tokens=req.max_tokens,
#     )
#
#     async def event_stream() -> AsyncGenerator[bytes, None]:
#         t_start = time.perf_counter()
#         previous_text = ""
#         token_count = 0
#         finish_reason: Optional[str] = None
#
#         # Open the stream with a metadata frame so the client gets the server's
#         # received-at timestamp for clock-skew alignment in telemetry.
#         yield _sse_event({
#             "type": "metadata",
#             "request_id": request_id,
#             "model_key": state.model_spec.key,
#             "server_received_at": time.time(),
#         })
#
#         try:
#             results = state.engine.generate(templated, sampling, request_id)
#             async for output in results:
#                 # Client closed the connection — free the GPU slot immediately.
#                 if await http_request.is_disconnected():
#                     log.info("← %s: client disconnected, aborting", request_id)
#                     await state.engine.abort(request_id)
#                     return
#
#                 completion = output.outputs[0]
#                 # vLLM yields cumulative text; emit only the delta.
#                 new_text = completion.text[len(previous_text):]
#                 previous_text = completion.text
#                 token_count = len(completion.token_ids)
#                 finish_reason = completion.finish_reason
#
#                 if new_text:
#                     yield _sse_event({
#                         "type": "token",
#                         "request_id": request_id,
#                         "text": new_text,
#                         "token_count": token_count,
#                     })
#
#                 if finish_reason is not None:
#                     duration = time.perf_counter() - t_start
#                     yield _sse_event({
#                         "type": "done",
#                         "request_id": request_id,
#                         "finish_reason": finish_reason,
#                         "total_tokens": token_count,
#                         "duration_seconds": round(duration, 4),
#                     })
#                     state.requests_completed += 1
#                     state.tokens_generated_total += token_count
#                     state.recent_completions.append((time.time(), token_count, duration))
#                     tps = token_count / duration if duration > 0 else 0
#                     log.info("← %s: %d tokens in %.2fs (%.1f tok/s, finish=%s)",
#                              request_id, token_count, duration, tps, finish_reason)
#                     return
#
#         except asyncio.CancelledError:
#             log.info("← %s: cancelled", request_id)
#             await state.engine.abort(request_id)
#             raise
#         except Exception as e:
#             log.exception("← %s: failed", request_id)
#             state.requests_failed += 1
#             yield _sse_event({
#                 "type": "error",
#                 "request_id": request_id,
#                 "message": f"{type(e).__name__}: {e}",
#             })
#
#     return StreamingResponse(
#         event_stream(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",
#             "Connection": "keep-alive",
#         },
#     )
#
#
# # ---- Entry point -----------------------------------------------------------
#
# def main() -> int:
#     parser = argparse.ArgumentParser(description="vLLM FastAPI server with SSE streaming.")
#     parser.add_argument("--model", choices=list(MODELS.keys()), required=True,
#                         help="Which model to load.")
#     parser.add_argument("--host", default="0.0.0.0",
#                         help="Bind interface (0.0.0.0 = all). Default: 0.0.0.0.")
#     parser.add_argument("--port", type=int, default=8000)
#     parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
#                         help="Fraction of GPU memory vLLM may use (default 0.85).")
#     args = parser.parse_args()
#
#     spec = MODELS[args.model]
#     app.state.model_spec = spec
#     app.state.gpu_memory_utilization = args.gpu_memory_utilization
#
#     import uvicorn
#     log.info("Starting server on %s:%d, model=%s (%s)",
#              args.host, args.port, spec.key, spec.repo_id)
#     log.info("CUDA_DEVICE_ORDER=%s  CUDA_VISIBLE_DEVICES=%s",
#              os.environ.get("CUDA_DEVICE_ORDER"),
#              os.environ.get("CUDA_VISIBLE_DEVICES"))
#     uvicorn.run(app, host=args.host, port=args.port, log_level="info")
#     return 0
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())
#
#
# # #!/usr/bin/env python3
# # """
# # server.py — FastAPI wrapper around vLLM AsyncLLMEngine with SSE streaming.
# #
# # Endpoints:
# #     POST /v1/generate   Submit a prompt, receive tokens as Server-Sent Events.
# #     GET  /health        Readiness probe + loaded model info.
# #     GET  /metrics       JSON snapshot of server + engine counters.
# #
# # Model is fixed at startup via --model. To switch, restart the server.
# # Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES.
# #
# # Usage:
# #     python server.py --model qwen
# #     python server.py --model mistral --host 0.0.0.0 --port 8000
# # """
# #
# # from __future__ import annotations
# #
# # # Must set BEFORE any CUDA-touching import (torch, vllm).
# # import os
# # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# #
# # import argparse
# # import asyncio
# # import json
# # import logging
# # import time
# # import uuid
# # from contextlib import asynccontextmanager
# # from dataclasses import dataclass
# # from typing import AsyncGenerator, Optional
# #
# # from fastapi import FastAPI, HTTPException, Request
# # from fastapi.responses import StreamingResponse
# # from pydantic import BaseModel, Field
# #
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# # )
# # log = logging.getLogger("vllm-server")
# #
# #
# # # ---- Model registry (must match smoke_test.py / prefetch_models.py) --------
# #
# # @dataclass(frozen=True)
# # class ModelSpec:
# #     key: str
# #     repo_id: str
# #     quantization: Optional[str]
# #     max_model_len: int
# #
# #
# # MODELS: dict[str, ModelSpec] = {
# #     "qwen": ModelSpec(
# #         key="qwen",
# #         repo_id="Qwen/Qwen2.5-3B-Instruct",
# #         quantization=None,
# #         max_model_len=8192,
# #     ),
# #     "mistral": ModelSpec(
# #         key="mistral",
# #         repo_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
# #         quantization="awq",
# #         max_model_len=8192,
# #     ),
# # }
# #
# #
# # # ---- Schemas ---------------------------------------------------------------
# #
# # class GenerateRequest(BaseModel):
# #     prompt: str = Field(..., description="User prompt (chat template applied server-side).")
# #     max_tokens: int = Field(256, ge=1, le=4096)
# #     temperature: float = Field(0.7, ge=0.0, le=2.0)
# #     top_p: float = Field(0.9, ge=0.0, le=1.0)
# #     request_id: Optional[str] = Field(None, description="Client-supplied ID. Server generates if omitted.")
# #
# #
# # class HealthResponse(BaseModel):
# #     status: str               # "ready" | "loading"
# #     model_key: Optional[str]
# #     model_repo: Optional[str]
# #     uptime_seconds: float
# #
# #
# # # ---- Server state ----------------------------------------------------------
# #
# # class ServerState:
# #     """Mutable singleton holding engine + counters. Accessed from handlers."""
# #     engine = None                                 # vllm.AsyncLLMEngine
# #     tokenizer = None
# #     model_spec: Optional[ModelSpec] = None
# #     started_at: float = 0.0
# #
# #     # Cheap counters surfaced via /metrics
# #     requests_completed: int = 0
# #     requests_failed: int = 0
# #     tokens_generated_total: int = 0
# #     # Rolling window: list of (completion_time, token_count, duration_seconds)
# #     recent_completions: list[tuple[float, int, float]] = []
# #
# #
# # state = ServerState()
# #
# #
# # # ---- Engine init -----------------------------------------------------------
# #
# # async def init_engine(spec: ModelSpec, gpu_memory_utilization: float) -> None:
# #     """Build the AsyncLLMEngine. First-call cost: 20-60s."""
# #     log.info("Initializing vLLM engine: model=%s quant=%s max_len=%d",
# #              spec.repo_id, spec.quantization, spec.max_model_len)
# #
# #     # Deferred import keeps --help instant and ensures CUDA_VISIBLE_DEVICES took effect.
# #     from vllm import AsyncLLMEngine, AsyncEngineArgs
# #
# #     engine_args = AsyncEngineArgs(
# #         model=spec.repo_id,
# #         quantization=spec.quantization,
# #         dtype="auto",
# #         gpu_memory_utilization=gpu_memory_utilization,
# #         max_model_len=spec.max_model_len,
# #         enforce_eager=False,             # use CUDA graphs
# #         trust_remote_code=False,
# #         disable_log_requests=True,       # we do our own request logging
# #     )
# #
# #     state.engine = AsyncLLMEngine.from_engine_args(engine_args)
# #     state.tokenizer = await state.engine.get_tokenizer()
# #     state.model_spec = spec
# #     state.started_at = time.time()
# #     log.info("Engine ready: %s", spec.repo_id)
# #
# #
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     spec: ModelSpec = app.state.model_spec
# #     gpu_mem: float = app.state.gpu_memory_utilization
# #     await init_engine(spec, gpu_mem)
# #     try:
# #         yield
# #     finally:
# #         log.info("Shutting down.")
# #
# #
# # app = FastAPI(lifespan=lifespan, title="vLLM Inference Server")
# #
# #
# # # ---- /health ---------------------------------------------------------------
# #
# # @app.get("/health", response_model=HealthResponse)
# # async def health():
# #     if state.engine is None or state.model_spec is None:
# #         return HealthResponse(status="loading", model_key=None, model_repo=None, uptime_seconds=0.0)
# #     return HealthResponse(
# #         status="ready",
# #         model_key=state.model_spec.key,
# #         model_repo=state.model_spec.repo_id,
# #         uptime_seconds=time.time() - state.started_at,
# #     )
# #
# #
# # # ---- /metrics --------------------------------------------------------------
# #
# # def _scheduler_snapshot() -> dict:
# #     """Best-effort read of vLLM's internal scheduler / block-manager state.
# #
# #     The internal API path is version-sensitive — wrap defensively so a vLLM
# #     upgrade doesn't break the metrics endpoint entirely.
# #     """
# #     try:
# #         sched = state.engine.engine.scheduler[0]  # type: ignore[attr-defined]
# #         bm = sched.block_manager
# #         total = bm.get_num_total_gpu_blocks()
# #         free = bm.get_num_free_gpu_blocks()
# #         used = total - free
# #         return {
# #             "num_running_requests": len(sched.running),
# #             "num_waiting_requests": len(sched.waiting),
# #             "num_swapped_requests": len(sched.swapped),
# #             "gpu_blocks_total": total,
# #             "gpu_blocks_used": used,
# #             "gpu_blocks_free": free,
# #             "gpu_cache_usage_perc": round(100.0 * used / total, 1) if total else 0.0,
# #         }
# #     except Exception as e:
# #         return {"error": f"scheduler stats unavailable: {type(e).__name__}: {e}"}
# #
# #
# # @app.get("/metrics")
# # async def metrics():
# #     now = time.time()
# #     spec = state.model_spec
# #
# #     # Rolling throughput over the last 30 seconds of completed requests.
# #     cutoff = now - 30.0
# #     state.recent_completions = [c for c in state.recent_completions if c[0] >= cutoff]
# #     if state.recent_completions:
# #         total_tokens = sum(c[1] for c in state.recent_completions)
# #         total_time = sum(c[2] for c in state.recent_completions)
# #         throughput = total_tokens / total_time if total_time > 0 else 0.0
# #     else:
# #         throughput = 0.0
# #
# #     return {
# #         "model_key": spec.key if spec else None,
# #         "model_repo": spec.repo_id if spec else None,
# #         "uptime_seconds": (now - state.started_at) if state.started_at else 0.0,
# #         "requests_completed": state.requests_completed,
# #         "requests_failed": state.requests_failed,
# #         "tokens_generated_total": state.tokens_generated_total,
# #         "throughput_tokens_per_sec_30s": round(throughput, 2),
# #         "scheduler": _scheduler_snapshot(),
# #     }
# #
# #
# # # ---- /v1/generate (SSE) ----------------------------------------------------
# #
# # def _sse_event(payload: dict) -> bytes:
# #     """Format a dict as a single SSE event frame."""
# #     return f"data: {json.dumps(payload)}\n\n".encode("utf-8")
# #
# #
# # @app.post("/v1/generate")
# # async def generate(req: GenerateRequest, http_request: Request):
# #     if state.engine is None or state.tokenizer is None:
# #         raise HTTPException(status_code=503, detail="Engine not ready")
# #
# #     request_id = req.request_id or str(uuid.uuid4())
# #     log.info("→ %s: max_tokens=%d temp=%.2f top_p=%.2f",
# #              request_id, req.max_tokens, req.temperature, req.top_p)
# #
# #     # Apply the model's chat template so the model sees properly-formatted input.
# #     messages = [{"role": "user", "content": req.prompt}]
# #     templated = state.tokenizer.apply_chat_template(
# #         messages, tokenize=False, add_generation_prompt=True,
# #     )
# #
# #     from vllm import SamplingParams
# #     sampling = SamplingParams(
# #         temperature=req.temperature,
# #         top_p=req.top_p,
# #         max_tokens=req.max_tokens,
# #     )
# #
# #     async def event_stream() -> AsyncGenerator[bytes, None]:
# #         t_start = time.perf_counter()
# #         previous_text = ""
# #         token_count = 0
# #         finish_reason: Optional[str] = None
# #
# #         # Open the stream with a metadata frame so the client gets the server's
# #         # received-at timestamp for clock-skew alignment in telemetry.
# #         yield _sse_event({
# #             "type": "metadata",
# #             "request_id": request_id,
# #             "model_key": state.model_spec.key,
# #             "server_received_at": time.time(),
# #         })
# #
# #         try:
# #             results = state.engine.generate(templated, sampling, request_id)
# #             async for output in results:
# #                 # Client closed the connection — free the GPU slot immediately.
# #                 if await http_request.is_disconnected():
# #                     log.info("← %s: client disconnected, aborting", request_id)
# #                     await state.engine.abort(request_id)
# #                     return
# #
# #                 completion = output.outputs[0]
# #                 # vLLM yields cumulative text; emit only the delta.
# #                 new_text = completion.text[len(previous_text):]
# #                 previous_text = completion.text
# #                 token_count = len(completion.token_ids)
# #                 finish_reason = completion.finish_reason
# #
# #                 if new_text:
# #                     yield _sse_event({
# #                         "type": "token",
# #                         "request_id": request_id,
# #                         "text": new_text,
# #                         "token_count": token_count,
# #                     })
# #
# #                 if finish_reason is not None:
# #                     duration = time.perf_counter() - t_start
# #                     yield _sse_event({
# #                         "type": "done",
# #                         "request_id": request_id,
# #                         "finish_reason": finish_reason,
# #                         "total_tokens": token_count,
# #                         "duration_seconds": round(duration, 4),
# #                     })
# #                     state.requests_completed += 1
# #                     state.tokens_generated_total += token_count
# #                     state.recent_completions.append((time.time(), token_count, duration))
# #                     tps = token_count / duration if duration > 0 else 0
# #                     log.info("← %s: %d tokens in %.2fs (%.1f tok/s, finish=%s)",
# #                              request_id, token_count, duration, tps, finish_reason)
# #                     return
# #
# #         except asyncio.CancelledError:
# #             log.info("← %s: cancelled", request_id)
# #             await state.engine.abort(request_id)
# #             raise
# #         except Exception as e:
# #             log.exception("← %s: failed", request_id)
# #             state.requests_failed += 1
# #             yield _sse_event({
# #                 "type": "error",
# #                 "request_id": request_id,
# #                 "message": f"{type(e).__name__}: {e}",
# #             })
# #
# #     return StreamingResponse(
# #         event_stream(),
# #         media_type="text/event-stream",
# #         headers={
# #             "Cache-Control": "no-cache",
# #             "X-Accel-Buffering": "no",
# #             "Connection": "keep-alive",
# #         },
# #     )
# #
# #
# # # ---- Entry point -----------------------------------------------------------
# #
# # def main() -> int:
# #     parser = argparse.ArgumentParser(description="vLLM FastAPI server with SSE streaming.")
# #     parser.add_argument("--model", choices=list(MODELS.keys()), required=True,
# #                         help="Which model to load.")
# #     parser.add_argument("--host", default="0.0.0.0",
# #                         help="Bind interface (0.0.0.0 = all). Default: 0.0.0.0.")
# #     parser.add_argument("--port", type=int, default=8000)
# #     parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
# #                         help="Fraction of GPU memory vLLM may use (default 0.85).")
# #     args = parser.parse_args()
# #
# #     spec = MODELS[args.model]
# #     app.state.model_spec = spec
# #     app.state.gpu_memory_utilization = args.gpu_memory_utilization
# #
# #     import uvicorn
# #     log.info("Starting server on %s:%d, model=%s (%s)",
# #              args.host, args.port, spec.key, spec.repo_id)
# #     # log_level="info" gives uvicorn its own access logs (one line per request).
# #     uvicorn.run(app, host=args.host, port=args.port, log_level="info")
# #     return 0
# #
# #
# # if __name__ == "__main__":
# #     raise SystemExit(main())
=======
    raise SystemExit(main())
>>>>>>> Stashed changes
