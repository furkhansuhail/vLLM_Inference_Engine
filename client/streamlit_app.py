"""
streamlit_app.py — Dashboard for the distributed vLLM inference demo.

Two tabs:

  Single run        Fires the three prompts of a chosen test case (A/B/C)
                    concurrently against the server, streams tokens live,
                    plots timing/scheduler telemetry, exports CSVs.

  Compare models    Runs the same case on BOTH registered models
                    sequentially (with an automatic /admin/load_model
                    swap between phases) and renders an overlaid /
                    side-by-side comparison view.

Sidebar contains the live model state panel + manual swap UI + sampling
and telemetry controls, shared across both tabs.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import datetime
import time
from typing import Optional

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference_client import (
    MetricsSnapshot,
    StreamResult,
    get_health,
    load_model,
    poll_metrics,
    stream_generate,
)
from test_cases import ALL_CASES, TestCase


# ---- Constants ------------------------------------------------------------

# Compare mode runs LEFT first, then swaps and runs RIGHT. Consistent
# ordering across runs makes screenshots / CSVs comparable across sessions.
COMPARE_LEFT_MODEL = "qwen"
COMPARE_RIGHT_MODEL = "mistral"


# ---- Page setup -----------------------------------------------------------

st.set_page_config(
    page_title="vLLM Inference Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
#  SWAP-POLLING GRACE PERIOD
# ============================================================================
#
# When the server processes /admin/load_model, AsyncLLMEngine.from_engine_args
# blocks the FastAPI event loop synchronously for ~20-30 s while it loads the
# model. During that window /health requests time out — the server is genuinely
# unresponsive even though the swap is succeeding in the background.
#
# Without intervention the client would (a) show "Cannot reach server" and
# (b) stop auto-refreshing, so it never notices when the server comes back.
# The grace period flag below tells the sidebar to keep polling for some
# time after any swap is initiated, regardless of whether /health is
# currently reachable.

SWAP_POLL_GRACE_SECONDS = 90.0


def _initiate_swap_polling(duration_seconds: float = SWAP_POLL_GRACE_SECONDS) -> None:
    """Mark a swap as just-initiated. The sidebar uses this to (a) keep
    auto-refreshing and (b) display a friendly "loading model" status
    instead of an alarming "cannot reach server" error during the
    ~20-30 s window when the server's event loop is blocked on
    AsyncLLMEngine.from_engine_args."""
    now = time.time()
    st.session_state.swap_polling_until = now + duration_seconds
    st.session_state.swap_polling_started_at = now


# ============================================================================
#  SIDEBAR — live model panel + sampling + telemetry settings (shared)
# ============================================================================

with st.sidebar:
    st.header("Server")
    server_url = st.text_input(
        "URL",
        value="http://192.168.88.23:8000",
        help="The CentOS server running server.py.",
    )

    # Fetch /health on every rerun — server is source of truth for state.
    health: Optional[dict] = None
    health_err: Optional[str] = None
    try:
        health = asyncio.run(get_health(server_url))
    except Exception as e:
        health_err = f"{type(e).__name__}: {e}"

    status_box = st.empty()
    # Did we recently initiate a swap? If so, a /health timeout is expected,
    # not an alarm condition — the server's event loop is blocked on
    # AsyncLLMEngine.from_engine_args for ~20-30 s.
    swap_grace_period = (
        st.session_state.get("swap_polling_until", 0.0) > time.time()
    )

    if health_err is not None:
        if swap_grace_period:
            elapsed = time.time() - st.session_state.get(
                "swap_polling_started_at", time.time()
            )
            status_box.warning(
                f"🟡 **LOADING MODEL** — server initializing\n\n"
                f"Briefly unreachable while loading (~20-30 s is normal).\n\n"
                f"Elapsed: {elapsed:.0f} s"
            )
        else:
            status_box.error(f"🔴 Cannot reach server\n\n`{health_err}`")
    else:
        status = health.get("status", "unknown")
        model_key = health.get("model_key")
        if status == "ready":
            status_box.success(
                f"🟢 **READY** — `{model_key}`\n\n"
                f"Uptime: {health.get('uptime_seconds', 0):.0f} s"
            )
        elif status == "swapping":
            target = health.get("swap_target", "?")
            elapsed = health.get("swap_elapsed_seconds", 0.0)
            status_box.warning(
                f"🟡 **SWAPPING** → `{target}`\n\n"
                f"Elapsed: {elapsed:.0f} s (typical ~35 s)"
            )
        elif status == "loading":
            status_box.info("🟡 **LOADING** — initial model boot")
        elif status == "failed":
            err = health.get("swap_error", "unknown")
            status_box.error(f"🔴 **FAILED**\n\n`{err}`")
        else:
            status_box.warning(f"🟡 **{status.upper()}**")

    refresh_col, _ = st.columns([1, 1])
    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Compare mode owns the model state during a comparison; hide the
    # manual swap UI then so the user can't fight the state machine.
    compare_active = (
        st.session_state.get("compare_state", "IDLE")
        not in ("IDLE", "DONE", "FAILED")
    )

    if health is not None and health.get("status") != "swapping" and not compare_active:
        available = health.get("available_models", []) or []
        current_model = health.get("model_key")
        if available:
            default_idx = available.index(current_model) if current_model in available else 0
            target_model = st.selectbox(
                "Switch model",
                options=available,
                index=default_idx,
                help="Pick a model to load on the server. A Swap button appears if different.",
            )

            if target_model != current_model:
                if st.button(
                    f"🔄 Swap to `{target_model}`",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        asyncio.run(load_model(server_url, target_model))
                        _initiate_swap_polling()
                        st.rerun()
                    except httpx.HTTPStatusError as e:
                        st.error(f"Swap rejected: HTTP {e.response.status_code}")
                    except Exception as e:
                        st.error(f"Could not issue swap: {type(e).__name__}: {e}")
    elif compare_active:
        st.caption("🔒 Model controls disabled while comparison is in progress.")

    # Auto-refresh during a server-side swap (whether from compare mode
    # or a manual swap). The actual rerun fires after the sidebar block
    # closes so other widget state is fully captured first.
    #
    # We auto-refresh in two situations:
    #   1. /health says status="swapping" (normal case, server responsive)
    #   2. We're inside the post-swap grace period (server may be unresponsive
    #      while AsyncLLMEngine.from_engine_args blocks the event loop)
    auto_refresh_pending = (
        (health is not None and health.get("status") == "swapping")
        or swap_grace_period
    )

    st.divider()

    st.header("Test case")
    case_key = st.selectbox(
        "Pick a case",
        options=["A", "B", "C"],
        format_func=lambda k: ALL_CASES[k].name,
    )
    case: TestCase = ALL_CASES[case_key]
    st.caption(case.description)

    st.divider()

    st.header("Sampling")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    max_tokens_override = st.number_input(
        "Max tokens (per request)",
        min_value=32, max_value=2048, value=case.max_tokens, step=32,
    )

    st.divider()

    st.header("Telemetry")
    poll_interval_ms = st.slider(
        "Metrics poll interval (ms)",
        min_value=50, max_value=1000, value=200, step=50,
        help=(
            "How often to hit /metrics during the batch. 200 ms is a good "
            "default — fast enough to capture KV cache ramp/drain on Case C, "
            "slow enough not to add server load."
        ),
    )

    server_ready = (
        health is not None
        and health.get("status") == "ready"
        and health.get("model_key") is not None
    )


# After the sidebar context closes, fire the auto-refresh.
if auto_refresh_pending:
    time.sleep(1.0)
    st.rerun()


# ============================================================================
#  SHARED RENDERING HELPERS
# ============================================================================

def render_streaming(placeholders, idx: int, result: StreamResult, batch_start: float) -> None:
    """Update one column's live panel as tokens arrive."""
    elapsed = max(time.time() - batch_start, 1e-6)
    tok_count = len(result.tokens)
    rate = tok_count / elapsed if elapsed > 0 else 0.0

    placeholders[idx]["response"].markdown(
        "<div style='font-size:0.92em; line-height:1.45; "
        "max-height:300px; overflow-y:auto; padding:8px; "
        "background:rgba(127,127,127,0.06); border-radius:6px;'>"
        f"{result.text}<span style='opacity:0.5'>▌</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    ttft_str = f"`{result.ttft_ms:.0f} ms`" if result.ttft_ms else "`…`"
    placeholders[idx]["metric"].markdown(
        f"⏱ TTFT: {ttft_str}  \n"
        f"📝 Tokens: `{tok_count}`  \n"
        f"⚡ Live rate: `{rate:.1f} tok/s`"
    )


def render_final(placeholders, idx: int, result: StreamResult) -> None:
    """Final state of one column after the request completes."""
    if result.error:
        placeholders[idx]["response"].error(f"Request failed: {result.error}")
    else:
        placeholders[idx]["response"].markdown(
            "<div style='font-size:0.92em; line-height:1.45; "
            "max-height:300px; overflow-y:auto; padding:8px; "
            "background:rgba(127,127,127,0.06); border-radius:6px;'>"
            f"{result.text}"
            "</div>",
            unsafe_allow_html=True,
        )

    ttft = result.ttft_ms or 0
    dur = result.duration_seconds or 0
    rate = result.throughput_tokens_per_sec or 0
    retry_line = f"🔁 Attempts: `{result.attempts}`  \n" if result.attempts > 1 else ""

    placeholders[idx]["metric"].markdown(
        f"⏱ TTFT: `{ttft:.0f} ms`  \n"
        f"⏳ Duration: `{dur:.2f} s`  \n"
        f"📝 Tokens: `{result.total_tokens}`  \n"
        f"⚡ Throughput: `{rate:.1f} tok/s`  \n"
        f"{retry_line}"
        f"🏁 Finish: `{result.finish_reason or '—'}`"
    )


def create_streaming_placeholders():
    """Create the 3-column placeholder grid used by both single-run and compare runs."""
    cols = st.columns(3, gap="medium")
    placeholders = []
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"Request {i + 1}")
            placeholders.append({
                "prompt": st.empty(),
                "response": st.empty(),
                "metric": st.empty(),
            })
    return placeholders


# ---- Chart builders -------------------------------------------------------

def build_waterfall(results: list[StreamResult], batch_start: float) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        label = f"Req {i + 1}"
        submitted_rel = r.submitted_at - batch_start
        first_token_rel = (r.first_token_at - batch_start) if r.first_token_at else submitted_rel
        completed_rel = (r.completed_at - batch_start) if r.completed_at else first_token_rel
        prefill_dur = max(first_token_rel - submitted_rel, 0)
        decode_dur = max(completed_rel - first_token_rel, 0)
        fig.add_trace(go.Bar(
            y=[label], x=[prefill_dur], base=[submitted_rel],
            orientation="h", name="Prefill / TTFT", marker_color="#FF9F40",
            hovertemplate=f"{label} prefill: {prefill_dur * 1000:.0f} ms<extra></extra>",
            showlegend=(i == 0),
        ))
        fig.add_trace(go.Bar(
            y=[label], x=[decode_dur], base=[first_token_rel],
            orientation="h", name="Decode", marker_color="#4CAF50",
            hovertemplate=f"{label} decode: {decode_dur:.2f} s<extra></extra>",
            showlegend=(i == 0),
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Time since batch start (s)",
        height=220,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_compare_waterfall(
    left_results: list[StreamResult], left_batch_start: float, left_label: str,
    right_results: list[StreamResult], right_batch_start: float, right_label: str,
) -> go.Figure:
    """Overlaid waterfall comparing the same case on two models.

    Each request gets two bar-pairs at adjacent y-positions: muted color
    for the left model, saturated for the right. Both timelines are
    normalized to their own batch start (t=0), so the visual question is
    "how much time does prefill+decode take on each model", not "which
    model started first".
    """
    fig = go.Figure()

    # Color palette: paired prefill / decode for each model. Muted vs
    # saturated. Side-by-side lookalike pairs read as "this is the same
    # phase on the other model."
    LEFT_PREFILL = "#FFD7A0"
    LEFT_DECODE = "#A5D6A7"
    RIGHT_PREFILL = "#E65100"
    RIGHT_DECODE = "#1B5E20"

    def _add_bars(results, batch_start, label_prefix, prefill_color, decode_color, y_offset):
        for i, r in enumerate(results):
            label = f"Req {i + 1} ({label_prefix})"
            submitted_rel = r.submitted_at - batch_start
            first_token_rel = (r.first_token_at - batch_start) if r.first_token_at else submitted_rel
            completed_rel = (r.completed_at - batch_start) if r.completed_at else first_token_rel
            prefill_dur = max(first_token_rel - submitted_rel, 0)
            decode_dur = max(completed_rel - first_token_rel, 0)
            fig.add_trace(go.Bar(
                y=[label], x=[prefill_dur], base=[submitted_rel],
                orientation="h",
                name=f"{label_prefix} prefill",
                marker_color=prefill_color,
                hovertemplate=f"{label} prefill: {prefill_dur * 1000:.0f} ms<extra></extra>",
                showlegend=(i == 0),
            ))
            fig.add_trace(go.Bar(
                y=[label], x=[decode_dur], base=[first_token_rel],
                orientation="h",
                name=f"{label_prefix} decode",
                marker_color=decode_color,
                hovertemplate=f"{label} decode: {decode_dur:.2f} s<extra></extra>",
                showlegend=(i == 0),
            ))

    _add_bars(left_results, left_batch_start, left_label, LEFT_PREFILL, LEFT_DECODE, 0)
    _add_bars(right_results, right_batch_start, right_label, RIGHT_PREFILL, RIGHT_DECODE, 0)

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Time since batch start (s)",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(categoryorder="category descending"),
    )
    return fig


def build_itl_chart(results: list[StreamResult]) -> go.Figure | None:
    rows = []
    for i, r in enumerate(results):
        for j, itl in enumerate(r.inter_token_latencies_ms):
            rows.append({"Request": f"Req {i + 1}", "Token index": j, "ITL (ms)": itl})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for req_name in df["Request"].unique():
        sub = df[df["Request"] == req_name]
        fig.add_trace(go.Scatter(
            x=sub["Token index"], y=sub["ITL (ms)"], mode="lines", name=req_name,
        ))
    fig.update_layout(
        xaxis_title="Token index", yaxis_title="Inter-token latency (ms)",
        height=260, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_scheduler_timeline(
    snapshots: list[MetricsSnapshot],
    batch_start: float,
) -> Optional[go.Figure]:
    if not snapshots:
        return None
    xs = [(s.timestamp - batch_start) for s in snapshots]
    running = [s.num_running for s in snapshots]
    waiting = [s.num_waiting for s in snapshots]
    cache = [s.gpu_cache_usage_perc for s in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=running, name="Running", mode="lines",
        line=dict(shape="hv", color="#4CAF50", width=2.5),
        hovertemplate="t=%{x:.2f}s — %{y} running<extra></extra>",
    ))
    if any(w > 0 for w in waiting):
        fig.add_trace(go.Scatter(
            x=xs, y=waiting, name="Waiting", mode="lines",
            line=dict(shape="hv", color="#FF9800", width=1.5, dash="dot"),
            hovertemplate="t=%{x:.2f}s — %{y} waiting<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=cache, name="KV cache (%)", mode="lines",
        line=dict(color="#2196F3", width=2), yaxis="y2",
        hovertemplate="t=%{x:.2f}s — %{y:.1f}%<extra></extra>",
    ))
    max_running = max(running) if running else 0
    max_cache = max(cache) if cache else 0
    fig.update_layout(
        xaxis_title="Time since batch start (s)",
        yaxis=dict(title="Requests", side="left",
                   range=[0, max(max_running, 3) + 0.5], tick0=0, dtick=1),
        yaxis2=dict(title="KV cache (%)", side="right", overlaying="y",
                    range=[0, max(max_cache * 1.15, 1.0)], ticksuffix="%"),
        height=260, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---- Telemetry table builders --------------------------------------------

def build_per_request_rows(results, case_key, batch_start, model_key=None):
    rows = []
    for i, r in enumerate(results, start=1):
        row = {
            "request_idx": i,
            "request_id": r.request_id,
            "case": case_key,
            "model_key": model_key or r.model_key or "",
            "prompt": r.prompt,
            "submitted_ms": round((r.submitted_at - batch_start) * 1000.0, 2),
            "first_token_ms": (
                round((r.first_token_at - batch_start) * 1000.0, 2)
                if r.first_token_at else None
            ),
            "completed_ms": (
                round((r.completed_at - batch_start) * 1000.0, 2)
                if r.completed_at else None
            ),
            "ttft_ms": round(r.ttft_ms, 2) if r.ttft_ms is not None else None,
            "duration_seconds": (
                round(r.duration_seconds, 4) if r.duration_seconds is not None else None
            ),
            "total_tokens": r.total_tokens,
            "throughput_tokens_per_sec": (
                round(r.throughput_tokens_per_sec, 2)
                if r.throughput_tokens_per_sec is not None else None
            ),
            "finish_reason": r.finish_reason or "",
            "attempts": r.attempts,
            "attempt_errors": " | ".join(r.attempt_errors) if r.attempt_errors else "",
            "error": r.error or "",
        }
        rows.append(row)
    return rows


def build_per_token_rows(results, batch_start, model_key=None):
    rows = []
    for i, r in enumerate(results, start=1):
        prev_ts: Optional[float] = None
        for j, tok in enumerate(r.tokens):
            t_rel_ms = (tok.timestamp - batch_start) * 1000.0
            delta_ms = (tok.timestamp - prev_ts) * 1000.0 if prev_ts else 0.0
            rows.append({
                "request_idx": i,
                "request_id": r.request_id,
                "case": r.case,
                "model_key": model_key or r.model_key or "",
                "token_idx": j,
                "timestamp_ms": round(t_rel_ms, 2),
                "delta_from_prev_token_ms": round(delta_ms, 2),
                "cumulative_token_count": tok.token_count,
                "text": tok.text,
            })
            prev_ts = tok.timestamp
    return rows


def build_snapshot_rows(snapshots, batch_start, model_key=None):
    return [
        {
            "model_key": model_key or "",
            "t_seconds": round(s.timestamp - batch_start, 3),
            "num_running": s.num_running,
            "num_waiting": s.num_waiting,
            "gpu_cache_usage_perc": round(s.gpu_cache_usage_perc, 2),
            "gpu_blocks_used": s.gpu_blocks_used,
            "gpu_blocks_total": s.gpu_blocks_total,
            "step_counter": s.step_counter,
            "stats_age_seconds": s.stats_age_seconds,
        }
        for s in snapshots
    ]


# ---- Aggregate stats for compare mode -------------------------------------

def _aggregate_metrics(results, snapshots, batch_start) -> dict:
    """Compute the headline metrics for a single batch run."""
    total_tokens = sum(r.total_tokens for r in results)
    last_complete = max((r.completed_at or batch_start) for r in results)
    batch_duration = max(last_complete - batch_start, 1e-6)
    aggregate_throughput = total_tokens / batch_duration
    ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
    mean_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0
    failures = sum(1 for r in results if r.error)
    total_retries = sum(max(0, r.attempts - 1) for r in results)
    peak_cache = max((s.gpu_cache_usage_perc for s in snapshots), default=0.0)
    return {
        "total_tokens": total_tokens,
        "batch_duration": batch_duration,
        "aggregate_throughput": aggregate_throughput,
        "mean_ttft": mean_ttft,
        "failures": failures,
        "total_retries": total_retries,
        "peak_cache_pct": peak_cache,
    }


# ---- Common batch runner (used by both tabs) ------------------------------

async def fire_batch_async(
    server_url: str,
    case: TestCase,
    max_tokens: int,
    temperature: float,
    top_p: float,
    poll_interval_ms: int,
    on_token_callbacks: Optional[list] = None,
) -> tuple[float, list[StreamResult], list[MetricsSnapshot]]:
    """Fire the three concurrent streams + a /metrics polling task.

    on_token_callbacks: list of 3 callbacks (one per request) or None.
    Returns (batch_start, results, snapshots).
    """
    start = time.time()
    metrics_snapshots: list[MetricsSnapshot] = []

    poll_task = asyncio.create_task(
        poll_metrics(
            server_url,
            interval_seconds=poll_interval_ms / 1000.0,
            snapshots=metrics_snapshots,
        )
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            stream_tasks = [
                stream_generate(
                    client, server_url, prompt,
                    case=case.key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    on_token=(on_token_callbacks[i] if on_token_callbacks else None),
                )
                for i, prompt in enumerate(case.prompts)
            ]
            results = await asyncio.gather(*stream_tasks)
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

    return start, results, metrics_snapshots


# ============================================================================
#  TABS
# ============================================================================

st.title("⚡ Distributed vLLM Inference Demo")
st.caption(
    "Three prompts fired concurrently over Server-Sent Events; "
    "scheduler state polled live; output streamed token-by-token."
)

tab_single, tab_compare = st.tabs(["🏃 Single run", "⚖️ Compare models"])


# ============================================================================
#  TAB 1 — SINGLE RUN
# ============================================================================

with tab_single:
    st.markdown(
        f"**{case.name}** — runs on whatever model the server currently has loaded."
    )
    with st.expander("Show the three prompts that will be fired", expanded=False):
        for i, p in enumerate(case.prompts, 1):
            st.markdown(f"**Prompt {i}**")
            st.code(p, language=None)

    run_btn = st.button(
        f"▶ Run case {case_key}",
        type="primary",
        use_container_width=False,
        disabled=not server_ready or compare_active,
        help=(
            "Comparison in progress — see Compare models tab"
            if compare_active else
            (None if server_ready else "Server isn't ready — see status panel.")
        ),
        key="run_btn_single",
    )

    single_placeholders = create_streaming_placeholders()
    single_analysis_placeholder = st.empty()

    if run_btn:
        for i, prompt in enumerate(case.prompts):
            truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
            single_placeholders[i]["prompt"].caption(f"Prompt: _{truncated}_")
            single_placeholders[i]["response"].markdown("_(waiting for tokens…)_")
            single_placeholders[i]["metric"].empty()
        single_analysis_placeholder.empty()

        def make_cb_single(idx: int):
            def cb(r: StreamResult) -> None:
                # batch_start grabbed from closure on the call below
                render_streaming(single_placeholders, idx, r, _single_batch_start[0])
            return cb

        _single_batch_start = [time.time()]
        with st.spinner(f"Running case {case.key} — three concurrent streams…"):
            batch_start, results, metrics_snapshots = asyncio.run(fire_batch_async(
                server_url, case,
                int(max_tokens_override), float(temperature), float(top_p),
                poll_interval_ms,
                on_token_callbacks=[make_cb_single(i) for i in range(3)],
            ))
            _single_batch_start[0] = batch_start

        for i, r in enumerate(results):
            render_final(single_placeholders, i, r)

        # ---- Analysis ----
        with single_analysis_placeholder.container():
            st.subheader("📊 Batch analysis")
            active_models = sorted({(r.model_key or "?") for r in results})
            st.caption(f"Model: `{', '.join(active_models)}`")

            metrics = _aggregate_metrics(results, metrics_snapshots, batch_start)

            m = st.columns(6)
            m[0].metric("Total tokens", f"{metrics['total_tokens']:,}")
            m[1].metric("Batch duration", f"{metrics['batch_duration']:.2f} s")
            m[2].metric("Aggregate throughput", f"{metrics['aggregate_throughput']:.1f} tok/s")
            m[3].metric("Mean TTFT", f"{metrics['mean_ttft']:.0f} ms")
            m[4].metric("Failed", f"{metrics['failures']}/3")
            m[5].metric("Retries", f"{metrics['total_retries']}")

            if metrics["total_retries"] > 0 or metrics["failures"] > 0:
                retry_msgs = []
                for i, r in enumerate(results, start=1):
                    if r.attempts > 1 or r.error:
                        status_ic = "❌ failed" if r.error else "✅ recovered"
                        retry_msgs.append(
                            f"- **Request {i}**: {status_ic} after {r.attempts} attempt(s)"
                            + (f" — `{r.error}`" if r.error else "")
                        )
                st.info("🔁 **Connection retries during this batch:**\n\n" + "\n".join(retry_msgs))

            st.markdown(
                "**Timing waterfall** — orange = prefill, green = decode. "
                "Overlapping bars indicate continuous batching."
            )
            st.plotly_chart(build_waterfall(results, batch_start), use_container_width=True)

            st.markdown("**Inter-token latency**")
            itl_fig = build_itl_chart(results)
            if itl_fig is not None:
                st.plotly_chart(itl_fig, use_container_width=True)
            else:
                st.info("Not enough tokens to compute ITL.")

            st.markdown(
                f"**Scheduler timeline** — polled every `{poll_interval_ms} ms`. "
                "Green = running requests; blue = KV cache usage."
            )
            tl = build_scheduler_timeline(metrics_snapshots, batch_start)
            if tl is not None:
                st.plotly_chart(tl, use_container_width=True)
                peak_running = max((s.num_running for s in metrics_snapshots), default=0)
                peak_cache = metrics["peak_cache_pct"]
                peak_blocks_used = max(
                    (s.gpu_blocks_used for s in metrics_snapshots if s.gpu_blocks_used is not None),
                    default=0,
                )
                blocks_total = next(
                    (s.gpu_blocks_total for s in metrics_snapshots if s.gpu_blocks_total is not None),
                    None,
                )
                s_cols = st.columns(4)
                s_cols[0].metric("Peak concurrent", f"{peak_running}")
                s_cols[1].metric("Peak KV cache", f"{peak_cache:.1f}%")
                s_cols[2].metric(
                    "Peak blocks used",
                    f"{peak_blocks_used}" + (f" / {blocks_total}" if blocks_total else ""),
                )
                s_cols[3].metric("Polls captured", f"{len(metrics_snapshots)}")
            else:
                st.warning("No mid-batch /metrics snapshots captured.")

            # ---- Telemetry export ----
            st.markdown("---")
            st.subheader("⬇️ Telemetry export")
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_tag = active_models[0] if len(active_models) == 1 else "mixed"
            fname_prefix = f"vllm_demo_case_{case.key}_{model_tag}_{ts_str}"

            per_request_df = pd.DataFrame(
                build_per_request_rows(results, case.key, batch_start, model_key=model_tag)
            )
            per_token_df = pd.DataFrame(
                build_per_token_rows(results, batch_start, model_key=model_tag)
            )
            snapshot_df = pd.DataFrame(
                build_snapshot_rows(metrics_snapshots, batch_start, model_key=model_tag)
            )

            dl_cols = st.columns(3)
            with dl_cols[0]:
                st.download_button(
                    label=f"📋 Per-request ({len(per_request_df)} rows)",
                    data=per_request_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{fname_prefix}_per_request.csv",
                    mime="text/csv", use_container_width=True, key="dl_per_request_single",
                )
            with dl_cols[1]:
                st.download_button(
                    label=f"🪙 Per-token ({len(per_token_df)} rows)",
                    data=per_token_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{fname_prefix}_per_token.csv",
                    mime="text/csv", use_container_width=True,
                    disabled=per_token_df.empty, key="dl_per_token_single",
                )
            with dl_cols[2]:
                st.download_button(
                    label=f"📈 Snapshots ({len(snapshot_df)} rows)",
                    data=snapshot_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{fname_prefix}_scheduler_snapshots.csv",
                    mime="text/csv", use_container_width=True,
                    disabled=snapshot_df.empty, key="dl_scheduler_single",
                )

            with st.expander("📋 Per-request summary"):
                disp = per_request_df.copy()
                if "prompt" in disp.columns:
                    disp["prompt"] = disp["prompt"].apply(
                        lambda p: (p[:80] + "…") if isinstance(p, str) and len(p) > 80 else p
                    )
                st.dataframe(disp, use_container_width=True, hide_index=True)

            with st.expander(f"🪙 Per-token timeline ({len(per_token_df)} tokens)"):
                if per_token_df.empty:
                    st.caption("(No tokens received.)")
                else:
                    opts = ["All"] + [f"Request {i}" for i in sorted(per_token_df["request_idx"].unique())]
                    pick = st.selectbox("Filter by request", options=opts, key="per_token_filter_single")
                    if pick == "All":
                        view_df = per_token_df
                    else:
                        view_df = per_token_df[per_token_df["request_idx"] == int(pick.split()[1])]
                    st.dataframe(view_df, use_container_width=True, hide_index=True, height=420)

            with st.expander(f"📈 Scheduler snapshots ({len(snapshot_df)} polls)"):
                if snapshot_df.empty:
                    st.caption("(No snapshots captured.)")
                else:
                    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
    else:
        if compare_active:
            st.warning("⚖️ A model comparison is currently in progress. Switch to the **Compare models** tab to follow it.")
        elif not server_ready:
            st.warning("Server isn't ready yet — see the **Model** panel in the sidebar.")
        else:
            st.info(
                "Click **Run case X** above. Three prompts will fire concurrently; "
                "tokens will stream into the panels above; scheduler timeline and "
                "CSV exports appear afterwards."
            )


# ============================================================================
#  TAB 2 — COMPARE MODELS
# ============================================================================
#
# State machine (lives in st.session_state):
#
#   IDLE → INIT → [SWAPPING_M1] → RUNNING_M1 → SWAPPING_M2 → RUNNING_M2 → DONE
#                                                                            ↓
#                                       FAILED ← (any swap or load_model fails)
#
# DONE renders the comparison view. FAILED renders the error + reset button.
# Both DONE and FAILED have a "New comparison" button that returns to IDLE.

COMPARE_STATES_RUNNING = {"INIT", "SWAPPING_M1", "RUNNING_M1", "SWAPPING_M2", "RUNNING_M2"}


def _reset_compare_state():
    for k in list(st.session_state.keys()):
        if k.startswith("compare_"):
            del st.session_state[k]


def _set_compare_state(new_state: str):
    st.session_state.compare_state = new_state


with tab_compare:
    compare_state = st.session_state.get("compare_state", "IDLE")

    st.markdown(
        f"**{case.name}** — runs the SAME case on `{COMPARE_LEFT_MODEL}` first, "
        f"then swaps to `{COMPARE_RIGHT_MODEL}` and runs again. Results are "
        "displayed side-by-side with an overlaid waterfall when both phases finish."
    )

    # ---- Control row (state-dependent) ----
    control_box = st.container()
    with control_box:
        if compare_state == "IDLE":
            cols = st.columns([2, 1])
            with cols[0]:
                st.caption(
                    f"Total time ≈ run × 2 + swap × {2 if (health and health.get('model_key') != COMPARE_LEFT_MODEL) else 1}. "
                    f"For Case C, expect 2–3 minutes."
                )
            with cols[1]:
                start_compare_btn = st.button(
                    "⚖️ Start comparison",
                    type="primary",
                    use_container_width=True,
                    disabled=not server_ready,
                    help=None if server_ready else "Server isn't ready.",
                )
            if start_compare_btn:
                _reset_compare_state()
                st.session_state.compare_state = "INIT"
                st.session_state.compare_case_key = case_key
                st.session_state.compare_settings = {
                    "max_tokens": int(max_tokens_override),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "poll_interval_ms": int(poll_interval_ms),
                }
                st.session_state.compare_started_at = time.time()
                st.session_state.compare_results = {}
                st.session_state.compare_error = None
                st.rerun()

        elif compare_state in COMPARE_STATES_RUNNING:
            elapsed = time.time() - st.session_state.get("compare_started_at", time.time())
            stage_label = {
                "INIT": "Setting up",
                "SWAPPING_M1": f"Swapping to `{COMPARE_LEFT_MODEL}`",
                "RUNNING_M1": f"Running case on `{COMPARE_LEFT_MODEL}`",
                "SWAPPING_M2": f"Swapping to `{COMPARE_RIGHT_MODEL}`",
                "RUNNING_M2": f"Running case on `{COMPARE_RIGHT_MODEL}`",
            }.get(compare_state, compare_state)
            st.info(f"⏱ Elapsed: {elapsed:.0f} s — current stage: **{stage_label}**")
            if st.button("❌ Cancel comparison", key="cancel_compare"):
                _reset_compare_state()
                st.rerun()

        elif compare_state == "DONE":
            elapsed = time.time() - st.session_state.get("compare_started_at", time.time())
            st.success(f"✅ Comparison complete (total elapsed: {elapsed:.0f} s)")
            if st.button("🔁 New comparison", key="new_compare"):
                _reset_compare_state()
                st.rerun()

        elif compare_state == "FAILED":
            err = st.session_state.get("compare_error", "unknown error")
            st.error(f"❌ Comparison failed: {err}")
            if st.button("🔁 Start over", key="reset_compare"):
                _reset_compare_state()
                st.rerun()

    # ---- State machine ----

    if compare_state == "INIT":
        # Decide whether we need a swap before running on M1.
        current = health.get("model_key") if health else None
        if current == COMPARE_LEFT_MODEL:
            _set_compare_state("RUNNING_M1")
        else:
            try:
                asyncio.run(load_model(server_url, COMPARE_LEFT_MODEL))
                _initiate_swap_polling()
                _set_compare_state("SWAPPING_M1")
            except Exception as e:
                st.session_state.compare_error = (
                    f"Could not initiate swap to {COMPARE_LEFT_MODEL}: {type(e).__name__}: {e}"
                )
                _set_compare_state("FAILED")
        st.rerun()

    elif compare_state == "SWAPPING_M1":
        s = health.get("status") if health else None
        if s == "ready" and health.get("model_key") == COMPARE_LEFT_MODEL:
            _set_compare_state("RUNNING_M1")
            st.rerun()
        elif s == "failed":
            st.session_state.compare_error = health.get("swap_error", "swap failed")
            _set_compare_state("FAILED")
            st.rerun()
        else:
            # Show progress and wait for the next refresh tick. The
            # sidebar's swap-detect already schedules auto-refresh, so
            # we DON'T sleep+rerun here (would double up).
            pass

    elif compare_state == "RUNNING_M1":
        st.markdown(f"### Phase 1 of 2 — running on `{COMPARE_LEFT_MODEL}`")
        m1_placeholders = create_streaming_placeholders()

        # Reset visual state
        for i, prompt in enumerate(case.prompts):
            truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
            m1_placeholders[i]["prompt"].caption(f"Prompt: _{truncated}_")
            m1_placeholders[i]["response"].markdown("_(waiting for tokens…)_")
            m1_placeholders[i]["metric"].empty()

        settings = st.session_state.compare_settings

        _m1_batch_start = [time.time()]
        def make_cb_m1(idx: int):
            def cb(r: StreamResult) -> None:
                render_streaming(m1_placeholders, idx, r, _m1_batch_start[0])
            return cb

        with st.spinner(f"Running case {case.key} on {COMPARE_LEFT_MODEL}..."):
            batch_start, results, snapshots = asyncio.run(fire_batch_async(
                server_url, case,
                settings["max_tokens"], settings["temperature"], settings["top_p"],
                settings["poll_interval_ms"],
                on_token_callbacks=[make_cb_m1(i) for i in range(3)],
            ))
            _m1_batch_start[0] = batch_start

        for i, r in enumerate(results):
            render_final(m1_placeholders, i, r)

        st.session_state.compare_results["m1"] = {
            "model": COMPARE_LEFT_MODEL,
            "batch_start": batch_start,
            "results": results,
            "snapshots": snapshots,
        }

        # Trigger swap to M2.
        try:
            asyncio.run(load_model(server_url, COMPARE_RIGHT_MODEL))
            _initiate_swap_polling()
            _set_compare_state("SWAPPING_M2")
        except Exception as e:
            st.session_state.compare_error = (
                f"Could not initiate swap to {COMPARE_RIGHT_MODEL}: {type(e).__name__}: {e}"
            )
            _set_compare_state("FAILED")
        st.rerun()

    elif compare_state == "SWAPPING_M2":
        s = health.get("status") if health else None
        if s == "ready" and health.get("model_key") == COMPARE_RIGHT_MODEL:
            _set_compare_state("RUNNING_M2")
            st.rerun()
        elif s == "failed":
            st.session_state.compare_error = health.get("swap_error", "swap failed")
            _set_compare_state("FAILED")
            st.rerun()
        # else: wait, sidebar's auto-refresh schedules the rerun.

    elif compare_state == "RUNNING_M2":
        # Show m1 results above so user has context while m2 runs
        m1 = st.session_state.compare_results.get("m1")
        if m1 is not None:
            with st.expander(f"✅ Phase 1 (`{m1['model']}`) — finished, click to view", expanded=False):
                m1_ph = create_streaming_placeholders()
                for i, r in enumerate(m1["results"]):
                    truncated = r.prompt[:100] + ("..." if len(r.prompt) > 100 else "")
                    m1_ph[i]["prompt"].caption(f"Prompt: _{truncated}_")
                    render_final(m1_ph, i, r)

        st.markdown(f"### Phase 2 of 2 — running on `{COMPARE_RIGHT_MODEL}`")
        m2_placeholders = create_streaming_placeholders()
        for i, prompt in enumerate(case.prompts):
            truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
            m2_placeholders[i]["prompt"].caption(f"Prompt: _{truncated}_")
            m2_placeholders[i]["response"].markdown("_(waiting for tokens…)_")
            m2_placeholders[i]["metric"].empty()

        settings = st.session_state.compare_settings

        _m2_batch_start = [time.time()]
        def make_cb_m2(idx: int):
            def cb(r: StreamResult) -> None:
                render_streaming(m2_placeholders, idx, r, _m2_batch_start[0])
            return cb

        with st.spinner(f"Running case {case.key} on {COMPARE_RIGHT_MODEL}..."):
            batch_start, results, snapshots = asyncio.run(fire_batch_async(
                server_url, case,
                settings["max_tokens"], settings["temperature"], settings["top_p"],
                settings["poll_interval_ms"],
                on_token_callbacks=[make_cb_m2(i) for i in range(3)],
            ))
            _m2_batch_start[0] = batch_start

        for i, r in enumerate(results):
            render_final(m2_placeholders, i, r)

        st.session_state.compare_results["m2"] = {
            "model": COMPARE_RIGHT_MODEL,
            "batch_start": batch_start,
            "results": results,
            "snapshots": snapshots,
        }

        _set_compare_state("DONE")
        st.rerun()

    elif compare_state == "DONE":
        # Render the comparison view
        m1 = st.session_state.compare_results["m1"]
        m2 = st.session_state.compare_results["m2"]
        case_key_done = st.session_state.compare_case_key
        case_done: TestCase = ALL_CASES[case_key_done]

        st.subheader(f"⚖️ {case_done.name} — `{m1['model']}` vs `{m2['model']}`")

        # ---- Aggregate metrics table ----
        m1_metrics = _aggregate_metrics(m1["results"], m1["snapshots"], m1["batch_start"])
        m2_metrics = _aggregate_metrics(m2["results"], m2["snapshots"], m2["batch_start"])

        def _fmt(value, fmt="{:.1f}"):
            return fmt.format(value) if isinstance(value, (int, float)) else str(value)

        comparison_rows = [
            ("Total tokens",                  f"{m1_metrics['total_tokens']:,}",       f"{m2_metrics['total_tokens']:,}"),
            ("Batch duration (s)",            f"{m1_metrics['batch_duration']:.2f}",   f"{m2_metrics['batch_duration']:.2f}"),
            ("Aggregate throughput (tok/s)",  f"{m1_metrics['aggregate_throughput']:.1f}", f"{m2_metrics['aggregate_throughput']:.1f}"),
            ("Mean TTFT (ms)",                f"{m1_metrics['mean_ttft']:.0f}",        f"{m2_metrics['mean_ttft']:.0f}"),
            ("Failed requests",               f"{m1_metrics['failures']}/3",            f"{m2_metrics['failures']}/3"),
            ("Retries used",                  f"{m1_metrics['total_retries']}",        f"{m2_metrics['total_retries']}"),
            ("Peak KV cache (%)",             f"{m1_metrics['peak_cache_pct']:.1f}",   f"{m2_metrics['peak_cache_pct']:.1f}"),
        ]
        comparison_df = pd.DataFrame(
            comparison_rows,
            columns=["Metric", m1["model"], m2["model"]],
        )
        st.markdown("**Aggregate metrics**")
        st.table(comparison_df.set_index("Metric"))

        # ---- Overlaid waterfall ----
        st.markdown(
            "**Timing waterfall (overlaid)** — same prompts on both models. "
            f"Light orange/green = `{m1['model']}` prefill/decode; "
            f"dark orange/green = `{m2['model']}`. Each bar pair is the same "
            "Req# across models so you can see relative timing directly."
        )
        st.plotly_chart(
            build_compare_waterfall(
                m1["results"], m1["batch_start"], m1["model"],
                m2["results"], m2["batch_start"], m2["model"],
            ),
            use_container_width=True,
        )

        # ---- Side-by-side scheduler timelines ----
        st.markdown("**Scheduler timelines**")
        tl_cols = st.columns(2)
        with tl_cols[0]:
            st.caption(f"`{m1['model']}`")
            tl1 = build_scheduler_timeline(m1["snapshots"], m1["batch_start"])
            if tl1 is not None:
                st.plotly_chart(tl1, use_container_width=True)
            else:
                st.info("No snapshots captured.")
        with tl_cols[1]:
            st.caption(f"`{m2['model']}`")
            tl2 = build_scheduler_timeline(m2["snapshots"], m2["batch_start"])
            if tl2 is not None:
                st.plotly_chart(tl2, use_container_width=True)
            else:
                st.info("No snapshots captured.")

        # ---- Side-by-side outputs ----
        st.markdown("**Per-request outputs**")
        for i in range(3):
            st.markdown(f"#### Request {i + 1}")
            out_cols = st.columns(2)
            r1 = m1["results"][i] if i < len(m1["results"]) else None
            r2 = m2["results"][i] if i < len(m2["results"]) else None
            with out_cols[0]:
                st.caption(f"`{m1['model']}`")
                if r1 is not None and not r1.error:
                    st.markdown(
                        "<div style='font-size:0.92em; line-height:1.45; "
                        "max-height:280px; overflow-y:auto; padding:8px; "
                        "background:rgba(127,127,127,0.06); border-radius:6px;'>"
                        f"{r1.text}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"⏱ {r1.ttft_ms:.0f} ms TTFT · ⏳ {r1.duration_seconds:.2f} s · "
                        f"📝 {r1.total_tokens} tok · ⚡ {r1.throughput_tokens_per_sec:.1f} tok/s"
                    )
                else:
                    st.error(r1.error if r1 else "no result")
            with out_cols[1]:
                st.caption(f"`{m2['model']}`")
                if r2 is not None and not r2.error:
                    st.markdown(
                        "<div style='font-size:0.92em; line-height:1.45; "
                        "max-height:280px; overflow-y:auto; padding:8px; "
                        "background:rgba(127,127,127,0.06); border-radius:6px;'>"
                        f"{r2.text}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"⏱ {r2.ttft_ms:.0f} ms TTFT · ⏳ {r2.duration_seconds:.2f} s · "
                        f"📝 {r2.total_tokens} tok · ⚡ {r2.throughput_tokens_per_sec:.1f} tok/s"
                    )
                else:
                    st.error(r2.error if r2 else "no result")
            st.divider()

        # ---- Combined CSV exports ----
        st.markdown("---")
        st.subheader("⬇️ Comparison exports")
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname_prefix = f"vllm_compare_case_{case_done.key}_{m1['model']}_vs_{m2['model']}_{ts_str}"

        # Combined per-request: model_key column distinguishes rows.
        combined_per_request = pd.DataFrame(
            build_per_request_rows(m1["results"], case_done.key, m1["batch_start"], model_key=m1["model"])
            + build_per_request_rows(m2["results"], case_done.key, m2["batch_start"], model_key=m2["model"])
        )
        combined_per_token = pd.DataFrame(
            build_per_token_rows(m1["results"], m1["batch_start"], model_key=m1["model"])
            + build_per_token_rows(m2["results"], m2["batch_start"], model_key=m2["model"])
        )
        combined_snapshots = pd.DataFrame(
            build_snapshot_rows(m1["snapshots"], m1["batch_start"], model_key=m1["model"])
            + build_snapshot_rows(m2["snapshots"], m2["batch_start"], model_key=m2["model"])
        )
        comparison_metrics_csv = comparison_df.to_csv(index=False).encode("utf-8")

        dl_cols = st.columns(4)
        with dl_cols[0]:
            st.download_button(
                label="📊 Aggregate metrics",
                data=comparison_metrics_csv,
                file_name=f"{fname_prefix}_aggregate.csv",
                mime="text/csv", use_container_width=True, key="dl_compare_aggregate",
            )
        with dl_cols[1]:
            st.download_button(
                label=f"📋 Per-request ({len(combined_per_request)})",
                data=combined_per_request.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_per_request.csv",
                mime="text/csv", use_container_width=True, key="dl_compare_per_request",
            )
        with dl_cols[2]:
            st.download_button(
                label=f"🪙 Per-token ({len(combined_per_token)})",
                data=combined_per_token.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_per_token.csv",
                mime="text/csv", use_container_width=True,
                disabled=combined_per_token.empty, key="dl_compare_per_token",
            )
        with dl_cols[3]:
            st.download_button(
                label=f"📈 Snapshots ({len(combined_snapshots)})",
                data=combined_snapshots.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_scheduler_snapshots.csv",
                mime="text/csv", use_container_width=True,
                disabled=combined_snapshots.empty, key="dl_compare_snapshots",
            )

        with st.expander("📋 Combined per-request table"):
            disp = combined_per_request.copy()
            if "prompt" in disp.columns:
                disp["prompt"] = disp["prompt"].apply(
                    lambda p: (p[:80] + "…") if isinstance(p, str) and len(p) > 80 else p
                )
            st.dataframe(disp, use_container_width=True, hide_index=True)

    elif compare_state == "FAILED":
        # Header already rendered above; nothing further to do.
        pass

    elif compare_state == "IDLE":
        # Header explained the flow; no further content needed until user starts.
        pass
