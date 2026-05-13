"""
inference_client.py — Async HTTP/SSE client for the vLLM server.

Designed to be reusable: the Streamlit app, a future headless CSV-dumping
CLI, and a future load-testing harness can all share this client.

Every relevant timestamp is captured (submit, first byte, first token,
each token, complete), so TTFT, ITL, and throughput fall out of the
StreamResult dataclass without extra bookkeeping.

Also provides `poll_metrics()` — a cancellable loop that hits /metrics on
a fixed interval and accumulates timestamped snapshots, used by the
dashboard to capture mid-batch scheduler state instead of just the
post-batch snapshot.
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


@dataclass
class StreamResult:
    """Accumulated result of a single streamed request."""
    request_id: str
    prompt: str
    case: str

    # Timing — all client-side wall clocks, seconds since epoch.
    submitted_at: float = 0.0
    first_byte_at: Optional[float] = None    # first SSE event of any kind
    first_token_at: Optional[float] = None   # first 'token' event specifically
    completed_at: Optional[float] = None

    # Content
    tokens: list[TokenEvent] = field(default_factory=list)
    text: str = ""
    total_tokens: int = 0
    finish_reason: Optional[str] = None

    # From server metadata event
    server_received_at: Optional[float] = None
    model_key: Optional[str] = None

    # Error from network or server-side
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token, in milliseconds."""
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.submitted_at) * 1000.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Submit-to-complete wall clock duration."""
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
        """List of gaps between consecutive token arrivals, in ms."""
        if len(self.tokens) < 2:
            return []
        return [
            (self.tokens[i].timestamp - self.tokens[i - 1].timestamp) * 1000.0
            for i in range(1, len(self.tokens))
        ]


@dataclass
class MetricsSnapshot:
    """One poll of /metrics with a client wall-clock timestamp.

    Only populated when the server's `scheduler` block has real numbers
    (not the idle placeholder or an error). Mirrors the keys the V1 stat
    logger exposes; raw payload kept for debug.
    """
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
) -> StreamResult:
    """Submit one prompt and consume the SSE stream end-to-end.

    `on_token` is invoked after each 'token' event with the in-progress
    result; the Streamlit UI uses this to update its live token displays.
    """
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

    try:
        async with client.stream(
            "POST",
            f"{server_url}/v1/generate",
            json=payload,
            timeout=httpx.Timeout(180.0, connect=10.0),
        ) as response:
            response.raise_for_status()

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
                        timestamp=now,
                        text=text,
                        token_count=event.get("token_count", 0),
                    ))
                    if on_token is not None:
                        on_token(result)

                elif event_type == "done":
                    result.completed_at = time.time()
                    result.total_tokens = event.get("total_tokens", 0)
                    result.finish_reason = event.get("finish_reason")
                    return result

                elif event_type == "error":
                    result.error = event.get("message", "unknown server error")
                    result.completed_at = time.time()
                    return result

    except httpx.HTTPError as e:
        result.error = f"{type(e).__name__}: {e}"
        result.completed_at = time.time()
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.completed_at = time.time()

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


# ---- Live metrics polling -------------------------------------------------

async def poll_metrics(
    server_url: str,
    *,
    interval_seconds: float = 0.2,
    snapshots: Optional[list[MetricsSnapshot]] = None,
) -> list[MetricsSnapshot]:
    """Poll /metrics in a loop and accumulate snapshots until cancelled.

    Designed to run as an asyncio task concurrent with stream_generate()
    calls. The caller cancels it via task.cancel() when the batch is
    done. CancelledError is swallowed and the accumulated snapshots are
    returned; the caller can also observe them in real time via the
    `snapshots` list it passes in (mutated in place).

    Network errors during a poll are silently skipped — a single failed
    /metrics request shouldn't kill the polling loop or take down the
    batch run.

    Interval is drift-resistant: each iteration sleeps for the remainder
    of the interval after the HTTP round-trip, not a fixed sleep.
    """
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
                        # Only record real readings. Skip idle placeholders
                        # ({"status": "idle..."}) and error envelopes
                        # ({"error": "..."}).
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
                    # Network blip, malformed JSON, or unexpected shape —
                    # keep polling.
                    pass

                elapsed = time.time() - tick_start
                await asyncio.sleep(max(0.0, interval_seconds - elapsed))
        except asyncio.CancelledError:
            return snapshots