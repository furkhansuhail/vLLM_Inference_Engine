#!/usr/bin/env python3
"""
server.py — FastAPI wrapper around vLLM AsyncLLMEngine with SSE streaming.

Endpoints:
    POST /v1/generate   Submit a prompt, receive tokens as Server-Sent Events.
    GET  /health        Readiness probe + loaded model info.
    GET  /metrics       JSON snapshot of server + engine counters.

Model is fixed at startup via --model. To switch, restart the server.
Pinned to GPU 1 (RTX 3080) via CUDA_VISIBLE_DEVICES.

Usage:
    python server.py --model qwen
    python server.py --model mistral --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

# Must set BEFORE any CUDA-touching import (torch, vllm).
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

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
    engine = None                                 # vllm.AsyncLLMEngine
    tokenizer = None
    model_spec: Optional[ModelSpec] = None
    started_at: float = 0.0

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

    # Deferred import keeps --help instant and ensures CUDA_VISIBLE_DEVICES took effect.
    from vllm import AsyncLLMEngine, AsyncEngineArgs

    engine_args = AsyncEngineArgs(
        model=spec.repo_id,
        quantization=spec.quantization,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=spec.max_model_len,
        enforce_eager=False,             # use CUDA graphs
        trust_remote_code=False,
        disable_log_requests=True,       # we do our own request logging
    )

    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    state.tokenizer = await state.engine.get_tokenizer()
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
    # log_level="info" gives uvicorn its own access logs (one line per request).
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())