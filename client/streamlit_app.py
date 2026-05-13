"""
streamlit_app.py — Dashboard for the distributed vLLM inference demo.

Fires the three prompts of a chosen test case (A/B/C) concurrently against
the server, streams tokens live into three side-by-side panels, then
visualizes the resulting timing data to make continuous batching and
PagedAttention behavior visible.

Now also polls /metrics every 200 ms during the batch in a concurrent
asyncio task, so the scheduler timeline reflects the actual mid-batch
peak — not the drained post-batch state.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
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


# ---- Sidebar: server + case selection -------------------------------------

with st.sidebar:
    st.header("Server")
    server_url = st.text_input(
        "URL",
        value="http://192.168.88.23:8000",
        help="The CentOS server running server.py.",
    )

    health_box = st.empty()
    if st.button("Check health", use_container_width=True):
        try:
            health = asyncio.run(get_health(server_url))
            health_box.success(
                f"**{health['status']}**\n\n"
                f"Model: `{health.get('model_key')}`\n\n"
                f"Repo: `{health.get('model_repo')}`\n\n"
                f"Uptime: {health.get('uptime_seconds', 0):.0f} s"
            )
        except Exception as e:
            health_box.error(f"Cannot reach server: {e}")

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

    run_btn = st.button(
        f"▶ Run case {case_key}",
        type="primary",
        use_container_width=True,
    )


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
    """Update the live panel for one request as tokens arrive."""
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
    """Replace the live panel content with the finalized result."""
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
    metric_placeholders[idx].markdown(
        f"⏱ TTFT: `{ttft:.0f} ms`  \n"
        f"⏳ Duration: `{dur:.2f} s`  \n"
        f"📝 Tokens: `{result.total_tokens}`  \n"
        f"⚡ Throughput: `{rate:.1f} tok/s`  \n"
        f"🏁 Finish: `{result.finish_reason or '—'}`"
    )


def build_waterfall(results: list[StreamResult], batch_start: float) -> go.Figure:
    """Horizontal bar chart of prefill+decode phases per request."""
    fig = go.Figure()
    for i, r in enumerate(results):
        label = f"Req {i + 1}"
        submitted_rel = r.submitted_at - batch_start
        first_token_rel = (r.first_token_at - batch_start) if r.first_token_at else submitted_rel
        completed_rel = (r.completed_at - batch_start) if r.completed_at else first_token_rel

        prefill_dur = max(first_token_rel - submitted_rel, 0)
        decode_dur = max(completed_rel - first_token_rel, 0)

        fig.add_trace(go.Bar(
            y=[label],
            x=[prefill_dur],
            base=[submitted_rel],
            orientation="h",
            name="Prefill / TTFT",
            marker_color="#FF9F40",
            hovertemplate=f"{label} prefill: {prefill_dur * 1000:.0f} ms<extra></extra>",
            showlegend=(i == 0),
        ))
        fig.add_trace(go.Bar(
            y=[label],
            x=[decode_dur],
            base=[first_token_rel],
            orientation="h",
            name="Decode",
            marker_color="#4CAF50",
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
    """Inter-token latency per request as a line chart."""
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
            x=sub["Token index"],
            y=sub["ITL (ms)"],
            mode="lines",
            name=req_name,
        ))
    fig.update_layout(
        xaxis_title="Token index",
        yaxis_title="Inter-token latency (ms)",
        height=260,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_scheduler_timeline(
    snapshots: list[MetricsSnapshot],
    batch_start: float,
) -> Optional[go.Figure]:
    """Dual-axis time-series of scheduler state during the batch.

    Left axis (step plot): num_running_requests — discrete count, 0..3.
    Right axis (line):     gpu_cache_usage_perc — smooth fraction.

    The step-plot shape on running-requests makes the
    schedule-fill / schedule-drain pattern of continuous batching obvious;
    the cache-usage line shows PagedAttention pressure independently.
    """
    if not snapshots:
        return None

    xs = [(s.timestamp - batch_start) for s in snapshots]
    running = [s.num_running for s in snapshots]
    waiting = [s.num_waiting for s in snapshots]
    cache = [s.gpu_cache_usage_perc for s in snapshots]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=running,
        name="Running requests",
        mode="lines",
        line=dict(shape="hv", color="#4CAF50", width=2.5),
        hovertemplate="t=%{x:.2f}s — %{y} running<extra></extra>",
    ))
    if any(w > 0 for w in waiting):
        fig.add_trace(go.Scatter(
            x=xs, y=waiting,
            name="Waiting requests",
            mode="lines",
            line=dict(shape="hv", color="#FF9800", width=1.5, dash="dot"),
            hovertemplate="t=%{x:.2f}s — %{y} waiting<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=cache,
        name="KV cache usage (%)",
        mode="lines",
        line=dict(color="#2196F3", width=2),
        yaxis="y2",
        hovertemplate="t=%{x:.2f}s — %{y:.1f}%<extra></extra>",
    ))

    max_running = max(running) if running else 0
    max_cache = max(cache) if cache else 0
    fig.update_layout(
        xaxis_title="Time since batch start (s)",
        yaxis=dict(
            title="Requests",
            side="left",
            range=[0, max(max_running, 3) + 0.5],
            tick0=0, dtick=1,
        ),
        yaxis2=dict(
            title="KV cache usage (%)",
            side="right",
            overlaying="y",
            range=[0, max(max_cache * 1.15, 1.0)],
            ticksuffix="%",
        ),
        height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---- Run handler ----------------------------------------------------------

if run_btn:
    # Reset panels
    for i, prompt in enumerate(case.prompts):
        truncated = prompt[:100] + ("..." if len(prompt) > 100 else "")
        prompt_placeholders[i].caption(f"Prompt: _{truncated}_")
        response_placeholders[i].markdown("_(waiting for tokens…)_")
        metric_placeholders[i].empty()
    analysis_placeholder.empty()

    async def fire_batch() -> tuple[float, list[StreamResult], list[MetricsSnapshot]]:
        """Fire the three concurrent streams + a /metrics polling task.

        The polling task is started before the streams and cancelled in a
        finally: block after gather() returns, so it captures the full
        idle → ramp → peak → drain envelope.
        """
        start = time.time()
        metrics_snapshots: list[MetricsSnapshot] = []

        def make_cb(idx: int):
            def cb(r: StreamResult) -> None:
                render_streaming(idx, r, start)
            return cb

        # Start the polling task. It mutates metrics_snapshots in place,
        # so we still have everything captured if it gets cancelled before
        # returning.
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
                        client,
                        server_url,
                        prompt,
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

    # Finalize each panel
    for i, r in enumerate(results):
        render_final(i, r)

    # ---- Aggregate analysis ----
    with analysis_placeholder.container():
        st.subheader("📊 Batch analysis")

        total_tokens = sum(r.total_tokens for r in results)
        last_complete = max((r.completed_at or batch_start) for r in results)
        batch_duration = max(last_complete - batch_start, 1e-6)
        aggregate_throughput = total_tokens / batch_duration
        ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
        mean_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0
        failures = sum(1 for r in results if r.error)

        m = st.columns(5)
        m[0].metric("Total tokens", f"{total_tokens:,}")
        m[1].metric("Batch duration", f"{batch_duration:.2f} s")
        m[2].metric("Aggregate throughput", f"{aggregate_throughput:.1f} tok/s")
        m[3].metric("Mean TTFT", f"{mean_ttft:.0f} ms")
        m[4].metric("Failed requests", f"{failures}/3")

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

        # ---- Live scheduler timeline (NEW) ----
        st.markdown(
            "**Scheduler timeline** — `/metrics` polled at "
            f"`{poll_interval_ms} ms` intervals during the batch. The green "
            "step plot shows how many requests the scheduler had running at "
            "each moment; the blue line shows KV cache occupancy from "
            "PagedAttention. A long flat green segment at 3 followed by "
            "staggered drops is continuous batching working as advertised."
        )
        timeline_fig = build_scheduler_timeline(metrics_snapshots, batch_start)
        if timeline_fig is not None:
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Peak summary — the headline numbers replacing the old
            # post-batch snapshot.
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
                "No mid-batch /metrics snapshots captured. Either the server's "
                "scheduler block returned errors/idle for the whole run, or the "
                "polling task didn't get a chance to tick. Check the server "
                "logs and verify `/metrics` returns real `num_running_requests`."
            )

        with st.expander("Raw per-request data"):
            rows = []
            for i, r in enumerate(results):
                rows.append({
                    "#": i + 1,
                    "Prompt (truncated)": r.prompt[:60] + ("…" if len(r.prompt) > 60 else ""),
                    "TTFT (ms)": f"{r.ttft_ms:.0f}" if r.ttft_ms else "—",
                    "Duration (s)": f"{r.duration_seconds:.2f}" if r.duration_seconds else "—",
                    "Tokens": r.total_tokens,
                    "Tok/s": f"{r.throughput_tokens_per_sec:.1f}" if r.throughput_tokens_per_sec else "—",
                    "Finish": r.finish_reason or "—",
                    "Error": r.error or "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander(f"Raw scheduler snapshots ({len(metrics_snapshots)} polls)"):
            if metrics_snapshots:
                snap_df = pd.DataFrame([
                    {
                        "t (s)": round(s.timestamp - batch_start, 3),
                        "running": s.num_running,
                        "waiting": s.num_waiting,
                        "cache %": round(s.gpu_cache_usage_perc, 1),
                        "blocks used": s.gpu_blocks_used,
                        "blocks total": s.gpu_blocks_total,
                        "step": s.step_counter,
                        "stats age (s)": s.stats_age_seconds,
                    }
                    for s in metrics_snapshots
                ])
                st.dataframe(snap_df, use_container_width=True, hide_index=True)
            else:
                st.caption("(No snapshots captured.)")

else:
    st.info(
        "Pick a case in the sidebar and click **Run**. "
        "Three prompts will be fired concurrently and tokens will stream in "
        "live across the three panels. The scheduler timeline will be polled "
        "during the batch to capture mid-batch peak state."
    )