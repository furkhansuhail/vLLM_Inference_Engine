#!/usr/bin/env python3
"""
server.py — FastAPI wrapper around vLLM AsyncLLMEngine with SSE streaming.

Endpoints:
    POST /v1/generate         Submit a prompt, receive tokens as SSE.
    GET  /health              Readiness probe, loaded model, swap state.
    GET  /metrics             JSON snapshot of server + engine counters.
    POST /admin/load_model    Tear down current engine, load a different one.

Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES with PCI bus ordering.

Usage:
    python server.py --model qwen
    python server.py --model mistral --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

# Must set BEFORE any CUDA-touching import (torch, vllm).
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # RTX 3080

import argparse
import asyncio
import gc
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


# ---- Model registry --------------------------------------------------------

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


class LoadModelRequest(BaseModel):
    model_key: str = Field(..., description="One of: " + ", ".join(MODELS.keys()))


class HealthResponse(BaseModel):
    status: str                                 # ready | loading | swapping | failed
    model_key: Optional[str] = None
    model_repo: Optional[str] = None
    uptime_seconds: float = 0.0
    available_models: list[str] = []

    # Populated only during a swap or after a failed swap.
    swap_target: Optional[str] = None
    swap_elapsed_seconds: Optional[float] = None
    swap_error: Optional[str] = None


# ---- Server state ----------------------------------------------------------

class ServerState:
    """Mutable singleton holding engine + counters. Accessed from handlers."""
    engine = None
    tokenizer = None
    model_spec: Optional[ModelSpec] = None
    started_at: float = 0.0
    # Captured V1 stat-logger instance (set during init_engine via factory).
    metrics_capture: Optional[object] = None

    # Counters
    requests_completed: int = 0
    requests_failed: int = 0
    tokens_generated_total: int = 0
    recent_completions: list[tuple[float, int, float]] = []

    # Swap state
    swap_in_progress: bool = False
    swap_target_key: Optional[str] = None
    swap_started_at: Optional[float] = None
    swap_last_error: Optional[str] = None    # cleared on next successful load


state = ServerState()


# ---- VRAM probe ------------------------------------------------------------
#
# Used around teardown to detect VRAM leaks. If post-shutdown VRAM is not
# substantially higher than pre-shutdown, the EngineCore subprocess didn't
# release its allocation and the next init_engine() will fail with the
# "Free memory on cuda:0 is less than utilization" error we already saw.

def _log_vram_state(label: str) -> None:
    """Log free/total VRAM on cuda:0. Returns nothing; safe to call anywhere."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info(0)
        log.info(
            "VRAM %-18s : %.2f GiB free / %.2f GiB total (%.0f%% used)",
            label, free / 1e9, total / 1e9,
            100.0 * (1.0 - free / total),
        )
    except Exception:
        pass


# ---- Engine init -----------------------------------------------------------

async def init_engine(spec: ModelSpec, gpu_memory_utilization: float) -> None:
    """Build the AsyncLLMEngine. First-call cost: 20-60s."""
    log.info("Initializing vLLM engine: model=%s quant=%s max_len=%d",
             spec.repo_id, spec.quantization, spec.max_model_len)
    _log_vram_state("pre-init")

    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from vllm.v1.metrics.loggers import StatLoggerBase
    from vllm.v1.metrics.stats import SchedulerStats, IterationStats

    # ---- V1 stat-logger plumbing -------------------------------------------
    # vLLM V1 runs its scheduler in a separate EngineCore process. We
    # capture the latest SchedulerStats via a custom StatLoggerBase so
    # /metrics can read it synchronously, no IPC at request time.
    # See vllm-project/vllm#20175: log() is never called in V1, only
    # record() — so all capture happens in record().

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

    engine_args = AsyncEngineArgs(
        model=spec.repo_id,
        quantization=spec.quantization,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=spec.max_model_len,
        enforce_eager=False,
        trust_remote_code=False,
    )

    state.engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        stat_loggers=[_stat_logger_factory],
    )
    state.tokenizer = state.engine.get_tokenizer()
    state.model_spec = spec
    state.started_at = time.time()
    _log_vram_state("post-init")
    log.info("Engine ready: %s", spec.repo_id)


# ---- Engine teardown -------------------------------------------------------

async def shutdown_engine() -> None:
    """Tear down the current vLLM engine and reclaim VRAM.

    vLLM V1 spawns the scheduler as a child process (EngineCore). For
    VRAM to actually drain we need: (1) call any available shutdown()
    method to signal a clean exit, (2) drop all Python references, (3)
    force gc + cuda.empty_cache(), (4) wait a few seconds for the child
    process to actually exit. Without (4) we get "Free memory on cuda:0
    is less than utilization" on the next init.
    """
    if state.engine is None:
        return

    from_key = state.model_spec.key if state.model_spec else "?"
    log.info("Tearing down engine (was: %s)...", from_key)
    _log_vram_state("pre-teardown")

    # (1) Signal the engine to shut down cleanly. The API name has shifted
    #     between vLLM versions; try the most likely candidates and don't
    #     panic if none exist — references-drop + GC still helps.
    for method_name in ("shutdown", "close", "stop"):
        method = getattr(state.engine, method_name, None)
        if method is None:
            continue
        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
            log.info("Called engine.%s()", method_name)
            break
        except Exception:
            log.exception("engine.%s() raised (continuing)", method_name)

    # (2) Drop references.
    state.engine = None
    state.tokenizer = None
    state.metrics_capture = None
    state.model_spec = None

    # (3) Force cleanup.
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # (4) Give the EngineCore subprocess time to actually exit.
    await asyncio.sleep(3.0)

    _log_vram_state("post-teardown")
    log.info("Engine teardown complete.")


async def _perform_swap(target_key: str) -> None:
    """Background task: tear down current engine, load new one.

    Sets state.swap_last_error on failure; clients see this via /health.
    On failure, state.engine is None — the server is in a "no model
    loaded" state and the user can retry with a different model.
    """
    try:
        await shutdown_engine()

        # New model = fresh counters. Avoid mixing per-model throughput
        # stats in the same /metrics window.
        state.requests_completed = 0
        state.requests_failed = 0
        state.tokens_generated_total = 0
        state.recent_completions = []

        spec = MODELS[target_key]
        gpu_mem = app.state.gpu_memory_utilization
        await init_engine(spec, gpu_mem)
        log.info("Swap complete: now serving %s", target_key)
        state.swap_last_error = None
    except Exception as e:
        log.exception("Engine swap to %s failed", target_key)
        state.swap_last_error = f"{type(e).__name__}: {e}"
        # state.engine etc. are None — server is in "unloaded" state.
    finally:
        state.swap_in_progress = False
        state.swap_target_key = None
        state.swap_started_at = None


# ---- Lifespan --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    spec: ModelSpec = app.state.model_spec
    gpu_mem: float = app.state.gpu_memory_utilization
    await init_engine(spec, gpu_mem)
    try:
        yield
    finally:
        log.info("Shutting down.")
        try:
            await shutdown_engine()
        except Exception:
            log.exception("Error during lifespan shutdown")


app = FastAPI(lifespan=lifespan, title="vLLM Inference Server")


# ---- /health ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    available = list(MODELS.keys())

    if state.swap_in_progress:
        elapsed = (
            time.time() - state.swap_started_at
            if state.swap_started_at else 0.0
        )
        return HealthResponse(
            status="swapping",
            model_key=None,
            model_repo=None,
            uptime_seconds=0.0,
            available_models=available,
            swap_target=state.swap_target_key,
            swap_elapsed_seconds=round(elapsed, 1),
            swap_error=state.swap_last_error,
        )

    if state.engine is None or state.model_spec is None:
        # Either we're still in initial startup, or a swap just failed.
        return HealthResponse(
            status="failed" if state.swap_last_error else "loading",
            model_key=None,
            model_repo=None,
            uptime_seconds=0.0,
            available_models=available,
            swap_error=state.swap_last_error,
        )

    return HealthResponse(
        status="ready",
        model_key=state.model_spec.key,
        model_repo=state.model_spec.repo_id,
        uptime_seconds=time.time() - state.started_at,
        available_models=available,
        swap_error=state.swap_last_error,    # surface even on success for context
    )


# ---- /admin/load_model -----------------------------------------------------

@app.post("/admin/load_model", status_code=202)
async def load_model(req: LoadModelRequest):
    """Tear down current engine and load a different model.

    Returns 202 Accepted immediately and runs the swap in the background.
    Poll /health for completion: status will be 'swapping' during the
    operation, then 'ready' with the new model_key, or 'failed' with
    swap_error set if it didn't work.

    Returns 409 Conflict if a swap is already in progress.
    Returns 400 Bad Request for an unknown model_key.
    """
    if req.model_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {req.model_key}. Available: {list(MODELS.keys())}",
        )

    if state.swap_in_progress:
        raise HTTPException(
            status_code=409,
            detail=f"Already swapping to {state.swap_target_key}",
        )

    # No-op if requested model is already loaded.
    if state.model_spec is not None and state.model_spec.key == req.model_key:
        return {
            "status": "already_loaded",
            "model_key": req.model_key,
        }

    from_key = state.model_spec.key if state.model_spec else None

    # Set swap state BEFORE returning so any immediately-following /health
    # poll sees the swap in progress (not the stale "ready" state).
    state.swap_in_progress = True
    state.swap_target_key = req.model_key
    state.swap_started_at = time.time()
    state.swap_last_error = None

    asyncio.create_task(_perform_swap(req.model_key))

    return {
        "status": "swap_started",
        "from": from_key,
        "to": req.model_key,
        "estimated_seconds": 35,    # ~3s teardown + ~30s load
    }


# ---- /metrics --------------------------------------------------------------

def _scheduler_snapshot() -> dict:
    """Read the latest captured scheduler stats from our V1 StatLogger."""
    cap = state.metrics_capture
    if cap is None:
        return {"error": "metrics capture not initialized"}
    s = getattr(cap, "last_scheduler_stats", None)
    if s is None:
        return {
            "status": "idle (no scheduler stats yet — send a request)",
            "stats_age_seconds": None,
        }

    age = time.monotonic() - cap.last_update_monotonic
    usage_frac = float(getattr(s, "kv_cache_usage", 0.0))

    total = cap.num_gpu_blocks
    if total is not None:
        used = int(round(usage_frac * total))
        free = total - used
    else:
        used = free = None

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


@app.get("/metrics")
async def metrics():
    now = time.time()
    spec = state.model_spec

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
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@app.post("/v1/generate")
async def generate(req: GenerateRequest, http_request: Request):
    if state.swap_in_progress:
        raise HTTPException(status_code=503, detail=f"Engine is swapping to {state.swap_target_key}")
    if state.engine is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    request_id = req.request_id or str(uuid.uuid4())
    log.info("→ %s: max_tokens=%d temp=%.2f top_p=%.2f",
             request_id, req.max_tokens, req.temperature, req.top_p)

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

        yield _sse_event({
            "type": "metadata",
            "request_id": request_id,
            "model_key": state.model_spec.key if state.model_spec else None,
            "server_received_at": time.time(),
        })

        try:
            results = state.engine.generate(templated, sampling, request_id)
            async for output in results:
                if await http_request.is_disconnected():
                    log.info("← %s: client disconnected, aborting", request_id)
                    await state.engine.abort(request_id)
                    return

                completion = output.outputs[0]
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
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    spec = MODELS[args.model]
    app.state.model_spec = spec
    app.state.gpu_memory_utilization = args.gpu_memory_utilization

    import uvicorn
    log.info("Starting server on %s:%d, model=%s (%s)",
             args.host, args.port, spec.key, spec.repo_id)
    log.info("CUDA_DEVICE_ORDER=%s  CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_DEVICE_ORDER"),
             os.environ.get("CUDA_VISIBLE_DEVICES"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())