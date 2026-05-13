"""
inference_client.py — Async HTTP/SSE client for the vLLM server.

Designed to be reusable: the Streamlit app, a future headless CSV-dumping
CLI, and a future load-testing harness can all share this client.

Every relevant timestamp is captured (submit, first byte, first token,
each token, complete), so TTFT, ITL, and throughput fall out of the
StreamResult dataclass without extra bookkeeping.

Also provides:
  * `poll_metrics()`     — cancellable /metrics polling loop, used by
                           the dashboard to capture mid-batch scheduler state.
  * `RetryConfig`        — configures the connection-phase retry behaviour
                           of stream_generate().
  * `load_model()`       — request a server-side model swap.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx


# ---- Data structures ------------------------------------------------------

@dataclass
class TokenEvent:
    """One streamed token event from the server."""
    timestamp: float          # client-side wall clock when received
    text: str                 # token text delta
    token_count: int          # server-reported cumulative count


@dataclass(frozen=True)
class RetryConfig:
    """Configures connection-phase retry behaviour for stream_generate().

    Only the *initial connection phase* is retried — failures after the
    first byte has arrived (mid-stream drops, stalls) are surfaced as
    errors rather than retried, because the server discards the
    in-flight generation when our TCP connection drops, and a fresh
    attempt would silently lose the partial tokens we already showed
    the user.
    """
    max_attempts: int = 3
    backoff_initial: float = 0.5
    backoff_factor: float = 2.0
    retry_on_status: tuple[int, ...] = (502, 503, 504)

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")


@dataclass
class StreamResult:
    """Accumulated result of a single streamed request."""
    request_id: str
    prompt: str
    case: str

    submitted_at: float = 0.0
    first_byte_at: Optional[float] = None
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None

    tokens: list[TokenEvent] = field(default_factory=list)
    text: str = ""
    total_tokens: int = 0
    finish_reason: Optional[str] = None

    server_received_at: Optional[float] = None
    model_key: Optional[str] = None

    attempts: int = 1
    attempt_errors: list[str] = field(default_factory=list)

    error: Optional[str] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.submitted_at) * 1000.0

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.submitted_at

    @property
    def throughput_tokens_per_sec(self) -> Optional[float]:
        d = self.duration_seconds
        if d is None or d <= 0 or self.total_tokens == 0:
            return None
        return self.total_tokens / d

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        if len(self.tokens) < 2:
            return []
        return [
            (self.tokens[i].timestamp - self.tokens[i - 1].timestamp) * 1000.0
            for i in range(1, len(self.tokens))
        ]


@dataclass
class MetricsSnapshot:
    """One poll of /metrics with a client wall-clock timestamp."""
    timestamp: float
    num_running: int = 0
    num_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_blocks_used: Optional[int] = None
    gpu_blocks_total: Optional[int] = None
    step_counter: int = 0
    stats_age_seconds: Optional[float] = None
    raw: dict = field(default_factory=dict)


# ---- Streaming inference call ---------------------------------------------

async def _consume_sse(
    response: httpx.Response,
    result: StreamResult,
    on_token: Optional[Callable[[StreamResult], None]],
) -> str:
    """Drain SSE events. Returns 'done' / 'error' / 'closed'."""
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        if result.first_byte_at is None:
            result.first_byte_at = time.time()
        try:
            event = json.loads(line[len("data: "):])
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "metadata":
            result.server_received_at = event.get("server_received_at")
            result.model_key = event.get("model_key")
        elif event_type == "token":
            now = time.time()
            if result.first_token_at is None:
                result.first_token_at = now
            text = event.get("text", "")
            result.text += text
            result.tokens.append(TokenEvent(
                timestamp=now, text=text,
                token_count=event.get("token_count", 0),
            ))
            if on_token is not None:
                on_token(result)
        elif event_type == "done":
            result.completed_at = time.time()
            result.total_tokens = event.get("total_tokens", 0)
            result.finish_reason = event.get("finish_reason")
            return "done"
        elif event_type == "error":
            result.error = event.get("message", "unknown server error")
            result.completed_at = time.time()
            return "error"
    return "closed"


async def stream_generate(
    client: httpx.AsyncClient,
    server_url: str,
    prompt: str,
    case: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    request_id: Optional[str] = None,
    on_token: Optional[Callable[[StreamResult], None]] = None,
    retry: Optional[RetryConfig] = None,
) -> StreamResult:
    """Submit one prompt and consume the SSE stream end-to-end.

    Retries connection-phase failures up to retry.max_attempts. See
    RetryConfig docstring for what counts as retryable. Mid-stream
    failures (after first_byte_at is set) are surfaced rather than
    retried because the server discards the in-flight generation on
    disconnect.
    """
    if retry is None:
        retry = RetryConfig()

    rid = request_id or str(uuid.uuid4())
    result = StreamResult(request_id=rid, prompt=prompt, case=case)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "request_id": rid,
    }

    result.submitted_at = time.time()
    timeouts = httpx.Timeout(300.0, connect=5.0, read=30.0)

    for attempt in range(1, retry.max_attempts + 1):
        result.attempts = attempt
        attempt_err: Optional[str] = None

        try:
            async with client.stream(
                "POST",
                f"{server_url}/v1/generate",
                json=payload,
                timeout=timeouts,
            ) as response:
                if response.status_code in retry.retry_on_status:
                    try:
                        body = (await response.aread()).decode("utf-8", "replace")[:200]
                    except Exception:
                        body = ""
                    attempt_err = (
                        f"HTTP {response.status_code}"
                        + (f": {body.strip()}" if body.strip() else "")
                    )
                else:
                    response.raise_for_status()
                    terminal = await _consume_sse(response, result, on_token)
                    if terminal in ("done", "error"):
                        return result
                    if result.first_byte_at is not None:
                        result.error = "stream closed without 'done' or 'error' event"
                        result.completed_at = time.time()
                        return result
                    attempt_err = "server closed empty stream"

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            attempt_err = f"{type(e).__name__}: {e}"

        except httpx.ReadTimeout as e:
            if result.first_byte_at is not None:
                result.error = f"stream stalled (no data for >30s): {e}"
                result.completed_at = time.time()
                return result
            attempt_err = f"ReadTimeout: {e}"

        except httpx.RemoteProtocolError as e:
            if result.first_byte_at is not None:
                result.error = f"stream dropped mid-flight: {e}"
                result.completed_at = time.time()
                return result
            attempt_err = f"RemoteProtocolError: {e}"

        except httpx.HTTPStatusError as e:
            result.error = f"HTTP {e.response.status_code}: {e}"
            result.completed_at = time.time()
            return result

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.completed_at = time.time()
            return result

        if attempt_err is not None:
            result.attempt_errors.append(f"attempt {attempt}: {attempt_err}")

        if attempt < retry.max_attempts:
            backoff = retry.backoff_initial * (retry.backoff_factor ** (attempt - 1))
            await asyncio.sleep(backoff)
        else:
            result.error = (
                f"failed after {attempt} attempts; last error: {attempt_err}"
            )
            result.completed_at = time.time()
            return result

    return result


# ---- Server probes --------------------------------------------------------

async def get_health(server_url: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{server_url}/health")
        r.raise_for_status()
        return r.json()


async def get_metrics(server_url: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{server_url}/metrics")
        r.raise_for_status()
        return r.json()


# ---- Model swap -----------------------------------------------------------

async def load_model(server_url: str, model_key: str) -> dict:
    """Request a server-side model swap. Returns the 202 response body.

    The server replies immediately and runs the swap as a background
    task; poll get_health() until status='ready' and model_key matches
    to confirm completion. Expect ~30-35s total (3s teardown + ~30s
    load). If the swap fails, get_health() will show status='failed'
    with a `swap_error` field describing the failure.

    Raises HTTPStatusError on 4xx/5xx (e.g. 409 if a swap is already in
    progress, 400 if model_key is unknown).
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{server_url}/admin/load_model",
            json={"model_key": model_key},
        )
        r.raise_for_status()
        return r.json()


# ---- Live metrics polling -------------------------------------------------

async def poll_metrics(
    server_url: str,
    *,
    interval_seconds: float = 0.2,
    snapshots: Optional[list[MetricsSnapshot]] = None,
) -> list[MetricsSnapshot]:
    """Poll /metrics in a loop and accumulate snapshots until cancelled."""
    if snapshots is None:
        snapshots = []

    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            while True:
                tick_start = time.time()
                try:
                    r = await client.get(f"{server_url}/metrics")
                    if r.status_code == 200:
                        data = r.json()
                        sched = data.get("scheduler", {})
                        if "num_running_requests" in sched:
                            snapshots.append(MetricsSnapshot(
                                timestamp=tick_start,
                                num_running=int(sched.get("num_running_requests", 0)),
                                num_waiting=int(sched.get("num_waiting_requests", 0)),
                                gpu_cache_usage_perc=float(sched.get("gpu_cache_usage_perc", 0.0)),
                                gpu_blocks_used=sched.get("gpu_blocks_used"),
                                gpu_blocks_total=sched.get("gpu_blocks_total"),
                                step_counter=int(sched.get("step_counter", 0)),
                                stats_age_seconds=sched.get("stats_age_seconds"),
                                raw=data,
                            ))
                except (httpx.HTTPError, ValueError, KeyError):
                    pass

                elapsed = time.time() - tick_start
                await asyncio.sleep(max(0.0, interval_seconds - elapsed))
        except asyncio.CancelledError:
            return snapshots
