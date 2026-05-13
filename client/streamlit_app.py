"""
streamlit_app.py — Dashboard for the distributed vLLM inference demo.

Fires the three prompts of a chosen test case (A/B/C) concurrently against
the server, streams tokens live into three side-by-side panels, then
visualizes the resulting timing data to make continuous batching and
PagedAttention behavior visible.

Polls /metrics every 200 ms during the batch in a concurrent asyncio task
so the scheduler timeline reflects the actual mid-batch peak.

Sidebar includes a live model panel that surfaces the server's current
state (ready / swapping / failed) and a model-swap UI that hits
/admin/load_model and polls for completion.

Provides CSV downloads of three telemetry datasets:
  - per-request summary (includes retry attempts + errors)
  - per-token timeline (long-form: request_idx, token_idx, timestamp_ms, text)
  - scheduler snapshots time-series

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


# ---- Page setup -----------------------------------------------------------

st.set_page_config(
    page_title="vLLM Inference Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Sidebar: server + model panel ----------------------------------------

# Streamlit reruns the whole script on every interaction, so we just
# fetch /health on every render and drive UI off of that. The server is
# the source of truth for model state; no client-side state.machine needed.

with st.sidebar:
    st.header("Server")
    server_url = st.text_input(
        "URL",
        value="http://192.168.88.23:8000",
        help="The CentOS server running server.py.",
    )

    # ---- Live model panel ----
    health: Optional[dict] = None
    health_err: Optional[str] = None
    try:
        health = asyncio.run(get_health(server_url))
    except Exception as e:
        health_err = f"{type(e).__name__}: {e}"

    status_box = st.empty()
    if health_err is not None:
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

    # ---- Model swap UI ----
    # Available iff we successfully reached /health AND the server lists
    # at least one model. Hidden during a swap (would just be re-clicking
    # ourselves into a 409).
    if health is not None and health.get("status") != "swapping":
        available = health.get("available_models", []) or []
        current_model = health.get("model_key")
        if available:
            # Default the selector to whatever's currently loaded. If we're
            # in failed/loading state with no model, default to first option.
            default_idx = available.index(current_model) if current_model in available else 0
            target_model = st.selectbox(
                "Switch model",
                options=available,
                index=default_idx,
                help=(
                    "Pick a model to load on the server. If different "
                    "from currently-loaded, a Swap button appears."
                ),
            )

            if target_model != current_model:
                if st.button(
                    f"🔄 Swap to `{target_model}`",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        asyncio.run(load_model(server_url, target_model))
                        st.rerun()    # pick up the 'swapping' state immediately
                    except httpx.HTTPStatusError as e:
                        st.error(f"Swap request rejected: HTTP {e.response.status_code}")
                    except Exception as e:
                        st.error(f"Could not issue swap: {type(e).__name__}: {e}")

    # ---- Auto-refresh during swap ----
    # Inside the `with st.sidebar` block we schedule a 1-second refresh so
    # the elapsed-seconds counter and final status update without the user
    # touching anything. Outside the swap state, no auto-refresh happens.
    auto_refresh_pending = (
        health is not None and health.get("status") == "swapping"
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

    st.divider()

    # Disable Run if the server isn't ready — protects against firing into
    # a 503 mid-swap. Spell out why so the user isn't confused.
    server_ready = (
        health is not None
        and health.get("status") == "ready"
        and health.get("model_key") is not None
    )
    run_btn = st.button(
        f"▶ Run case {case_key}",
        type="primary",
        use_container_width=True,
        disabled=not server_ready,
        help=None if server_ready else "Server isn't ready — see status panel above.",
    )

# After the sidebar context closes, fire the auto-refresh if we observed a
# swap-in-progress. Sleeping inside `with st.sidebar` would still work but
# keeping it out makes the sleep+rerun explicit.
if auto_refresh_pending:
    time.sleep(1.0)
    st.rerun()


# ---- Header ---------------------------------------------------------------

st.title("⚡ Distributed vLLM Inference Demo")
st.markdown(
    f"**{case.name}** — three prompts fired concurrently, "
    "tokens streamed back via Server-Sent Events."
)

with st.expander("Show the three prompts that will be fired", expanded=False):
    for i, p in enumerate(case.prompts, 1):
        st.markdown(f"**Prompt {i}**")
        st.code(p, language=None)


# ---- Live streaming layout ------------------------------------------------

cols = st.columns(3, gap="medium")
prompt_placeholders: list[st.delta_generator.DeltaGenerator] = []
response_placeholders: list[st.delta_generator.DeltaGenerator] = []
metric_placeholders: list[st.delta_generator.DeltaGenerator] = []

for i, col in enumerate(cols):
    with col:
        st.subheader(f"Request {i + 1}")
        prompt_placeholders.append(st.empty())
        response_placeholders.append(st.empty())
        metric_placeholders.append(st.empty())

st.divider()
analysis_placeholder = st.empty()


def render_streaming(idx: int, result: StreamResult, batch_start: float) -> None:
    elapsed = max(time.time() - batch_start, 1e-6)
    tok_count = len(result.tokens)
    rate = tok_count / elapsed if elapsed > 0 else 0.0

    response_placeholders[idx].markdown(
        "<div style='font-size:0.92em; line-height:1.45; "
        "max-height:340px; overflow-y:auto; padding:8px; "
        "background:rgba(127,127,127,0.06); border-radius:6px;'>"
        f"{result.text}<span style='opacity:0.5'>▌</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    ttft_str = f"`{result.ttft_ms:.0f} ms`" if result.ttft_ms else "`…`"
    metric_placeholders[idx].markdown(
        f"⏱ TTFT: {ttft_str}  \n"
        f"📝 Tokens: `{tok_count}`  \n"
        f"⚡ Live rate: `{rate:.1f} tok/s`"
    )


def render_final(idx: int, result: StreamResult) -> None:
    if result.error:
        response_placeholders[idx].error(f"Request failed: {result.error}")
    else:
        response_placeholders[idx].markdown(
            "<div style='font-size:0.92em; line-height:1.45; "
            "max-height:340px; overflow-y:auto; padding:8px; "
            "background:rgba(127,127,127,0.06); border-radius:6px;'>"
            f"{result.text}"
            "</div>",
            unsafe_allow_html=True,
        )

    ttft = result.ttft_ms or 0
    dur = result.duration_seconds or 0
    rate = result.throughput_tokens_per_sec or 0
    retry_line = ""
    if result.attempts > 1:
        retry_line = f"🔁 Attempts: `{result.attempts}`  \n"

    metric_placeholders[idx].markdown(
        f"⏱ TTFT: `{ttft:.0f} ms`  \n"
        f"⏳ Duration: `{dur:.2f} s`  \n"
        f"📝 Tokens: `{result.total_tokens}`  \n"
        f"⚡ Throughput: `{rate:.1f} tok/s`  \n"
        f"{retry_line}"
        f"🏁 Finish: `{result.finish_reason or '—'}`"
    )


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
        x=xs, y=running, name="Running requests", mode="lines",
        line=dict(shape="hv", color="#4CAF50", width=2.5),
        hovertemplate="t=%{x:.2f}s — %{y} running<extra></extra>",
    ))
    if any(w > 0 for w in waiting):
        fig.add_trace(go.Scatter(
            x=xs, y=waiting, name="Waiting requests", mode="lines",
            line=dict(shape="hv", color="#FF9800", width=1.5, dash="dot"),
            hovertemplate="t=%{x:.2f}s — %{y} waiting<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=cache, name="KV cache usage (%)", mode="lines",
        line=dict(color="#2196F3", width=2), yaxis="y2",
        hovertemplate="t=%{x:.2f}s — %{y:.1f}%<extra></extra>",
    ))
    max_running = max(running) if running else 0
    max_cache = max(cache) if cache else 0
    fig.update_layout(
        xaxis_title="Time since batch start (s)",
        yaxis=dict(title="Requests", side="left",
                   range=[0, max(max_running, 3) + 0.5], tick0=0, dtick=1),
        yaxis2=dict(title="KV cache usage (%)", side="right", overlaying="y",
                    range=[0, max(max_cache * 1.15, 1.0)], ticksuffix="%"),
        height=280, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---- Telemetry table builders --------------------------------------------

def build_per_request_rows(results, case_key, batch_start):
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append({
            "request_idx": i,
            "request_id": r.request_id,
            "case": case_key,
            "model_key": r.model_key or "",
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
        })
    return rows


def build_per_token_rows(results, batch_start):
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
                "token_idx": j,
                "timestamp_ms": round(t_rel_ms, 2),
                "delta_from_prev_token_ms": round(delta_ms, 2),
                "cumulative_token_count": tok.token_count,
                "text": tok.text,
            })
            prev_ts = tok.timestamp
    return rows


def build_snapshot_rows(snapshots, batch_start):
    return [
        {
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


# ---- Run handler ----------------------------------------------------------

if run_btn:
    for i, prompt in enumerate(case.prompts):
        truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
        prompt_placeholders[i].caption(f"Prompt: _{truncated}_")
        response_placeholders[i].markdown("_(waiting for tokens…)_")
        metric_placeholders[i].empty()
    analysis_placeholder.empty()

    async def fire_batch() -> tuple[float, list[StreamResult], list[MetricsSnapshot]]:
        start = time.time()
        metrics_snapshots: list[MetricsSnapshot] = []

        def make_cb(idx: int):
            def cb(r: StreamResult) -> None:
                render_streaming(idx, r, start)
            return cb

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
                        max_tokens=int(max_tokens_override),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        on_token=make_cb(i),
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

    with st.spinner(f"Running case {case.key} — three concurrent streams…"):
        batch_start, results, metrics_snapshots = asyncio.run(fire_batch())

    for i, r in enumerate(results):
        render_final(i, r)

    with analysis_placeholder.container():
        st.subheader("📊 Batch analysis")
        # Surface the model that produced these results — useful when
        # comparing runs across model swaps. Falls back to anything any
        # request reported, in case the active model changed between
        # /health and the request actually being served.
        active_model_keys = sorted({(r.model_key or "?") for r in results})
        active_model_str = ", ".join(active_model_keys) if active_model_keys else "?"
        st.caption(f"Model: `{active_model_str}`")

        total_tokens = sum(r.total_tokens for r in results)
        last_complete = max((r.completed_at or batch_start) for r in results)
        batch_duration = max(last_complete - batch_start, 1e-6)
        aggregate_throughput = total_tokens / batch_duration
        ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
        mean_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0
        failures = sum(1 for r in results if r.error)
        total_retries = sum(max(0, r.attempts - 1) for r in results)

        m = st.columns(6)
        m[0].metric("Total tokens", f"{total_tokens:,}")
        m[1].metric("Batch duration", f"{batch_duration:.2f} s")
        m[2].metric("Aggregate throughput", f"{aggregate_throughput:.1f} tok/s")
        m[3].metric("Mean TTFT", f"{mean_ttft:.0f} ms")
        m[4].metric("Failed requests", f"{failures}/3")
        m[5].metric(
            "Retries used",
            f"{total_retries}",
            help="Sum of retries across all 3 requests. 0 = all succeeded on first attempt.",
        )

        if total_retries > 0 or failures > 0:
            retry_msgs = []
            for i, r in enumerate(results, start=1):
                if r.attempts > 1 or r.error:
                    status = "❌ failed" if r.error else "✅ recovered"
                    retry_msgs.append(
                        f"- **Request {i}**: {status} after {r.attempts} attempt(s)"
                        + (f" — `{r.error}`" if r.error else "")
                    )
            st.info(
                "🔁 **Connection retries occurred during this batch:**\n\n"
                + "\n".join(retry_msgs)
            )

        st.markdown(
            "**Timing waterfall** — orange = prefill (waiting for first token), "
            "green = decode (streaming tokens). Overlapping bars are evidence "
            "of continuous batching."
        )
        st.plotly_chart(build_waterfall(results, batch_start), use_container_width=True)

        st.markdown(
            "**Inter-token latency** — gap between consecutive token arrivals. "
            "Spikes can indicate scheduler events (new request joining the batch, "
            "another request finishing) or network jitter."
        )
        itl_fig = build_itl_chart(results)
        if itl_fig is not None:
            st.plotly_chart(itl_fig, use_container_width=True)
        else:
            st.info("Not enough tokens to compute ITL.")

        st.markdown(
            "**Scheduler timeline** — `/metrics` polled at "
            f"`{poll_interval_ms} ms` intervals during the batch. The green "
            "step plot shows how many requests the scheduler had running at "
            "each moment; the blue line shows KV cache occupancy from "
            "PagedAttention."
        )
        timeline_fig = build_scheduler_timeline(metrics_snapshots, batch_start)
        if timeline_fig is not None:
            st.plotly_chart(timeline_fig, use_container_width=True)

            peak_running = max((s.num_running for s in metrics_snapshots), default=0)
            peak_waiting = max((s.num_waiting for s in metrics_snapshots), default=0)
            peak_cache = max((s.gpu_cache_usage_perc for s in metrics_snapshots), default=0.0)
            peak_blocks_used = max(
                (s.gpu_blocks_used for s in metrics_snapshots if s.gpu_blocks_used is not None),
                default=0,
            )
            blocks_total = next(
                (s.gpu_blocks_total for s in metrics_snapshots if s.gpu_blocks_total is not None),
                None,
            )
            s_cols = st.columns(5)
            s_cols[0].metric("Peak concurrent requests", f"{peak_running}")
            s_cols[1].metric("Peak waiting", f"{peak_waiting}")
            s_cols[2].metric("Peak KV cache usage", f"{peak_cache:.1f}%")
            s_cols[3].metric(
                "Peak KV blocks used",
                f"{peak_blocks_used}" + (f" / {blocks_total}" if blocks_total else ""),
            )
            s_cols[4].metric("Polls captured", f"{len(metrics_snapshots)}")
        else:
            st.warning(
                "No mid-batch /metrics snapshots captured. Check the server "
                "logs and verify `/metrics` returns real `num_running_requests`."
            )

        st.markdown("---")
        st.subheader("⬇️ Telemetry export")

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include model key in the filename so swap-and-rerun comparisons
        # don't collide on disk.
        model_tag = active_model_keys[0] if len(active_model_keys) == 1 else "mixed"
        fname_prefix = f"vllm_demo_case_{case.key}_{model_tag}_{ts_str}"

        per_request_df = pd.DataFrame(build_per_request_rows(results, case.key, batch_start))
        per_token_df = pd.DataFrame(build_per_token_rows(results, batch_start))
        snapshot_df = pd.DataFrame(build_snapshot_rows(metrics_snapshots, batch_start))

        dl_cols = st.columns(3)
        with dl_cols[0]:
            st.download_button(
                label=f"📋 Per-request summary ({len(per_request_df)} rows)",
                data=per_request_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_per_request.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_per_request",
            )
        with dl_cols[1]:
            st.download_button(
                label=f"🪙 Per-token timeline ({len(per_token_df)} rows)",
                data=per_token_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_per_token.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=per_token_df.empty,
                key="dl_per_token",
            )
        with dl_cols[2]:
            st.download_button(
                label=f"📈 Scheduler snapshots ({len(snapshot_df)} rows)",
                data=snapshot_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname_prefix}_scheduler_snapshots.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=snapshot_df.empty,
                key="dl_scheduler",
            )

        with st.expander("📋 Per-request summary"):
            display_df = per_request_df.copy()
            if "prompt" in display_df.columns:
                display_df["prompt"] = display_df["prompt"].apply(
                    lambda p: (p[:80] + "…") if isinstance(p, str) and len(p) > 80 else p
                )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander(f"🪙 Per-token timeline ({len(per_token_df)} tokens)"):
            if per_token_df.empty:
                st.caption("(No tokens received.)")
            else:
                req_options = ["All"] + [
                    f"Request {i}" for i in sorted(per_token_df["request_idx"].unique())
                ]
                pick = st.selectbox(
                    "Filter by request",
                    options=req_options,
                    key="per_token_filter",
                )
                if pick == "All":
                    view_df = per_token_df
                else:
                    idx = int(pick.split()[1])
                    view_df = per_token_df[per_token_df["request_idx"] == idx]
                st.dataframe(view_df, use_container_width=True, hide_index=True, height=420)

        with st.expander(f"📈 Scheduler snapshots ({len(snapshot_df)} polls)"):
            if snapshot_df.empty:
                st.caption("(No snapshots captured.)")
            else:
                st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

else:
    if server_ready:
        st.info(
            "Pick a case in the sidebar and click **Run**. Three prompts "
            "will fire concurrently and stream into the panels above. "
            "Scheduler state is polled during the batch; three CSV "
            "telemetry exports appear afterwards."
        )
    else:
        st.warning(
            "Server isn't ready yet — see the **Model** panel in the sidebar. "
            "If a swap is in progress, the page will refresh automatically "
            "as it completes."
        )