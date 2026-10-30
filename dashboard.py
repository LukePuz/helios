import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import alerts
import db
import process_manager as pm
from config import SYMBOLS, START_VALUE, IB_PORT
from market_data import get_latest_quotes
from ibkr import IBKRClient, place_market_order

st.set_page_config(page_title="Helios Hedgefund", layout="wide", page_icon="☀️")


@st.cache_resource
def _client() -> IBKRClient:
    return IBKRClient()


client = _client()


# ── Workflow helpers ──────────────────────────────────────────────────────────

_WORKFLOW_NODES: list[tuple[str, str, str]] = [
    ("redis",     "Redis",       "Message Broker"),
    ("db",        "Supabase",    "Database"),
    ("ollama",    "Ollama",      "AI Engine"),
    ("beat",      "Scheduler",   "Celery Beat"),
    ("worker",    "Ingestion",   "Celery Worker"),
    ("catalyst",  "Catalyst",    "Calendar API"),
    ("screener",  "Screener",    "Event Filter"),
    ("analyst",   "Analyst",     "Thesis + Conviction"),
    ("risk",      "Risk Mgr",    "Position Sizing"),
    ("execution", "Execution",   "Telegram → IBKR"),
]


@st.cache_data(ttl=4)
def _get_workflow_status() -> dict[str, bool]:
    svc = pm.get_status()
    ollama_ok, _ = pm.ollama_status()

    # Supabase: fast socket probe (avoids opening a full DB connection)
    supabase_host = os.getenv("SUPABASE_HOST", "")
    db_ok = False
    if supabase_host:
        try:
            with socket.create_connection((supabase_host, 5432), timeout=2):
                db_ok = True
        except Exception:
            db_ok = False

    # IBKR TWS: socket probe on whichever port TRADING_ENV actually resolves
    # to (config.IB_PORT), not a raw os.getenv("IB_PORT", "7497") -- that
    # always fell back to paper's port regardless of TRADING_ENV, since
    # IB_PORT is deliberately absent from .env now (see config.py's
    # "Trading environment" section). Found 2026-08-18 while tracing the
    # ops-queue starvation incident; happened not to be visible only
    # because paper's ibc.service was coincidentally also still running.
    try:
        with socket.create_connection(("127.0.0.1", IB_PORT), timeout=0.5):
            ibkr_ok = True
    except Exception:
        ibkr_ok = False

    telegram_ok = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    pipeline_ok = (svc.get("celery_worker", False) and svc.get("celery_llm_worker", False)
                   and svc.get("celery_screen_worker", False)
                   and svc.get("celery_ingestion_worker", False) and ollama_ok)

    return {
        "redis":     svc.get("redis", False),
        "db":        db_ok,
        "ollama":    ollama_ok,
        "beat":      svc.get("celery_beat", False),
        "worker":    svc.get("celery_worker", False),
        "catalyst":  svc.get("catalyst_service", False),
        "screener":  pipeline_ok,
        "analyst":   pipeline_ok,
        "risk":      pipeline_ok,
        "execution": telegram_ok and ibkr_ok,
    }


def _build_workflow_figure(statuses: dict[str, bool]) -> go.Figure:
    n      = len(_WORKFLOW_NODES)
    R      = 3.4    # node ring radius
    R_lb   = R + 1.2  # label ring radius
    angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    xs     = [R    * math.cos(a) for a in angles]
    ys     = [R    * math.sin(a) for a in angles]
    lxs    = [R_lb * math.cos(a) for a in angles]
    lys    = [R_lb * math.sin(a) for a in angles]

    fig = go.Figure()

    # --- Arrows between consecutive nodes ---
    node_r = 0.43
    for i in range(n):
        j = (i + 1) % n
        xA, yA = xs[i], ys[i]
        xB, yB = xs[j], ys[j]
        dx, dy = xB - xA, yB - yA
        dist   = math.sqrt(dx ** 2 + dy ** 2)
        ux, uy = dx / dist, dy / dist
        fig.add_annotation(
            x=xB - ux * node_r,   y=yB - uy * node_r,
            ax=xA + ux * node_r,  ay=yA + uy * node_r,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3,
            arrowsize=1.2, arrowwidth=1.8, arrowcolor="#4a4a5a",
            text="",
        )

    # --- Node circles ---
    node_colors = [
        "#00cc88" if statuses.get(nid, False) else "#cc2222"
        for nid, _, _ in _WORKFLOW_NODES
    ]
    node_hover = [
        (
            f"<b>{label}</b> — {sub}<br>"
            f"{'✅ Active' if statuses.get(nid) else '🔴 Offline'}"
        )
        for nid, label, sub in _WORKFLOW_NODES
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=54,
            color=node_colors,
            line=dict(width=3, color="rgba(255,255,255,0.18)"),
            opacity=0.92,
        ),
        hovertext=node_hover,
        hoverinfo="text",
    ))

    # --- Labels outside the ring ---
    for i, (nid, label, sub) in enumerate(_WORKFLOW_NODES):
        active    = statuses.get(nid, False)
        dot_color = "#00cc88" if active else "#cc2222"
        status_lbl = "● Active" if active else "● Offline"
        fig.add_annotation(
            x=lxs[i], y=lys[i],
            text=(
                f"<b>{label}</b><br>"
                f"<span style='color:#aaa;font-size:10px'>{sub}</span><br>"
                f"<span style='color:{dot_color};font-size:10px'>{status_lbl}</span>"
            ),
            showarrow=False,
            font=dict(size=11, color="white"),
            align="center",
            bgcolor="rgba(14,17,23,0.90)",
            bordercolor=dot_color,
            borderwidth=1,
            borderpad=5,
        )

    fig.update_layout(
        height=660,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(range=[-6.4, 6.4], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-6.4, 6.4], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        title=dict(text="Live System Workflow", font=dict(size=18, color="#ccc"), x=0.5),
    )
    return fig


# ── Sidebar: System Control ───────────────────────────────────────────────────
with st.sidebar:
    st.title("Helios Hedgefund")
    st.caption("Paper trading only")


    status = pm.get_status()

    def _dot(running: bool) -> str:
        return "🟢" if running else "🔴"

    # Redis
    redis_status = "Running" if status["redis"] else "Not running"
    st.markdown(f"{_dot(status['redis'])} **Redis** — {redis_status}")
    if not status["redis"]:
        st.caption("Run: docker start redis")

    # Ollama
    ollama_model = os.getenv("ANALYST_MODEL", "qwen2.5:72b")
    ollama_ok, ollama_info = pm.ollama_status()
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"{_dot(ollama_ok)} **Ollama** — {ollama_model}")
    with col2.popover("ℹ️"):
        st.markdown(f"**Current model:** `{ollama_model}`\n\nThe AI reasoning engine. Generates investment theses with conviction scores before any trade executes.")
    if ollama_ok:
        st.caption(ollama_info)
    else:
        st.caption("Ollama not running — run 'ollama serve' (or check its systemd service)")

    st.divider()

    # One-click start/stop all services
    col_all1, col_all2 = st.columns(2)
    if col_all1.button("▶ Start All", use_container_width=True, type="primary"):
        for _name in pm.SERVICES:
            pm.start(_name)
        st.session_state["startup_time"] = time.time()
        st.toast("All services started.")
        st.rerun()
    if col_all2.button("⏹ Stop All", use_container_width=True):
        pm.stop_all()
        st.toast("All services stopped.")
        st.rerun()

    _SERVICE_INFO = {
        "celery_worker": "Fast I/O ops tasks only — trailing-stop checks, portfolio snapshots, the drawdown circuit breaker, and stop-order reconciliation. Ingestion polling moved to its own worker below (2026-08-18) after a degraded SEC EDGAR endpoint was found monopolizing this queue for 28+ minutes at a stretch, silently starving these safety-critical checks.",
        "celery_ingestion_worker": "Runs the background ingestion tasks — pulls SEC filings, FRED macro data, arXiv quantum/AI papers, options flow, congressional trades, insider filings, news, fundamentals, and price history. Isolated from the ops worker above so a slow or degraded external data source can never delay the drawdown check or stop monitoring. Each event gets stored in the vector database and queued for the AI pipeline.",
        "celery_llm_worker": "Dedicated worker for the 72B analyst — only receives events that already passed screening (see celery_screen_worker). Drains the priority lane (events about currently-held positions) before the bulk lane (new-idea analyses), and runs one inference at a time so the GPU isn't contended.",
        "celery_screen_worker": "Fast 7B relevance filter, CPU-only, runs several in parallel. Rejects most incoming news/filings/options events cheaply so only genuine survivors reach the expensive 72B worker above.",
        "celery_beat": "The scheduler. Fires each ingestion task on a timer: options flow every 5 min, SEC filings every 10 min, macro data every hour, congressional trades every 6 hours, arXiv papers once daily at 9 AM ET. Also triggers daily regime assessment at 9:30 AM and scanner at 9:35 AM ET.",
        "catalyst_service": "Shared catalyst calendar API for earnings, conferences, indirect read-throughs, and conflict flags.",
    }

    for name, svc in pm.SERVICES.items():
        running = status[name]
        col1, col2, col3 = st.columns([3, 1, 2])
        col1.markdown(f"{_dot(running)} **{svc['label']}**")
        with col2.popover("ℹ️"):
            st.markdown(_SERVICE_INFO[name])
        btn_label = "Stop" if running else "Start"
        btn_type  = "secondary" if running else "primary"
        if col3.button(btn_label, key=f"btn_{name}", type=btn_type):
            if running:
                result = pm.stop(name)
                st.toast(f"Stopped {svc['label']}" if result == "ok" else result)
            else:
                result = pm.start(name)
                st.toast(f"Started {svc['label']}" if result == "ok" else result)
            st.rerun()


    st.divider()

    # Regime badge
    try:
        _regime = db.get_latest_regime()
        if _regime:
            _rname  = _regime.get("regime", "Unknown")
            _rscore = float(_regime.get("score", 0))
            _remoji = (
                "🟢🟢" if "Strong Risk-On" in _rname else
                "🟢"  if "Risk-On"        in _rname else
                "🔴🔴" if "Strong Risk-Off" in _rname else
                "🔴"  if "Risk-Off"       in _rname else
                "🟡"
            )
            st.markdown(f"**Market Regime**")
            st.markdown(f"{_remoji} **{_rname}**")
            st.progress(_rscore / 100, text=f"Score: {_rscore:.0f}/100")
        else:
            st.caption("No regime data yet.")
    except Exception:
        pass

    st.divider()

    if st.button("Refresh Dashboard", use_container_width=True, type="primary"):
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_workflow, tab_activity, tab_portfolio, tab_regime, tab_approvals, tab_outcomes, tab_performance, tab_macro, tab_technicals, tab_signals, tab_trade, tab_logs = st.tabs([
    "Workflow", "Activity", "Portfolio", "Market Regime", "Pending Approvals", "Outcomes", "Performance", "Macro", "Technicals", "Signal Log", "Manual Trade", "Logs",
])


# ── Tab 0: Workflow ───────────────────────────────────────────────────────────
with tab_workflow:
    st.header("System Workflow")

    # During the 45 s after clicking Start All, auto-refresh every 2 s so
    # nodes light up in real time as each service comes online.
    _startup_t = st.session_state.get("startup_time")
    _in_startup = _startup_t is not None and (time.time() - _startup_t) < 45
    if _in_startup:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, key="wf_autorefresh")
        _secs_left = max(0, int(45 - (time.time() - _startup_t)))
        st.info(f"Services starting — refreshing every 2 s ({_secs_left}s remaining).")

    _wf_status = _get_workflow_status()
    st.plotly_chart(_build_workflow_figure(_wf_status), use_container_width=True)

    st.divider()

    # Status table beneath the diagram
    _active_count = sum(1 for v in _wf_status.values() if v)
    _total_count  = len(_WORKFLOW_NODES)
    c1, c2 = st.columns([1, 3])
    c1.metric("Services Online", f"{_active_count} / {_total_count}")

    _rows = [
        {
            "Service": f"{'🟢' if _wf_status.get(nid) else '🔴'} {label}",
            "Role":    sub,
            "Status":  "Active" if _wf_status.get(nid) else "Offline",
        }
        for nid, label, sub in _WORKFLOW_NODES
    ]
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    if c2.button("↺ Refresh Status", key="wf_refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Tab: Activity Feed ───────────────────────────────────────────────────────
with tab_activity:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="activity_autorefresh")

    st.header("Live Activity Feed")
    st.caption("Real-time commentary from every pipeline stage — auto-refreshes every 5 seconds.")

    _SENTIMENT_COLOR = {
        "bullish": "#00cc88",
        "bearish": "#cc4444",
        "alert":   "#ff8800",
        "neutral": "#888888",
    }
    _STAGE_ICONS = {
        "screener":  "🔍",
        "analyst":   "🧠",
        "risk":      "📊",
        "telegram":  "📱",
        "execution": "✅",
        "system":    "⚙️",
        "edgar":     "📄",
        "options":   "📈",
        "macro":     "🌐",
        "insider":   "👤",
    }

    try:
        # Reload db module so new functions are visible even in a stale process
        import importlib, db as _db_mod
        importlib.reload(_db_mod)
        _feed_rows = _db_mod.get_activity_feed(limit=120)
    except Exception as _e:
        _feed_rows = []
        st.error(f"Could not load feed: {_e}")

    if not _feed_rows:
        st.info("No activity yet — the feed will populate as the pipeline processes events.")
    else:
        # Summary metrics strip
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _bullish_n = sum(1 for r in _feed_rows if r["sentiment"] == "bullish")
        _alert_n   = sum(1 for r in _feed_rows if r["sentiment"] == "alert")
        _tickers   = list(dict.fromkeys(r["ticker"] for r in _feed_rows if r["ticker"]))
        _mc1.metric("Events (last 120)", len(_feed_rows))
        _mc2.metric("Bullish signals", _bullish_n)
        _mc3.metric("Alerts", _alert_n)
        _mc4.metric("Tickers seen", len(_tickers))

        st.divider()

        # Filter controls
        _fcol1, _fcol2 = st.columns([2, 2])
        _stage_filter = _fcol1.multiselect(
            "Filter by stage",
            options=["screener", "analyst", "risk", "telegram", "execution", "system"],
            default=[],
            key="feed_stage_filter",
        )
        _ticker_filter = _fcol2.multiselect(
            "Filter by ticker",
            options=_tickers,
            default=[],
            key="feed_ticker_filter",
        )

        _visible = _feed_rows
        if _stage_filter:
            _visible = [r for r in _visible if r["stage"] in _stage_filter]
        if _ticker_filter:
            _visible = [r for r in _visible if r["ticker"] in _ticker_filter]

        # Render each row as a styled card
        for _row in _visible:
            _icon      = _STAGE_ICONS.get(_row["stage"], "•")
            _color     = _SENTIMENT_COLOR.get(_row["sentiment"], "#888888")
            _ts        = _row["ts"]
            _ts_str    = _ts.strftime("%H:%M:%S") if hasattr(_ts, "strftime") else str(_ts)[:19]
            _ticker_lbl = f"**{_row['ticker']}**" if _row["ticker"] else ""
            _stage_lbl  = _row["stage"].upper()

            st.markdown(
                f"""<div style="
                    border-left: 3px solid {_color};
                    padding: 6px 12px;
                    margin-bottom: 4px;
                    background: rgba(255,255,255,0.03);
                    border-radius: 0 4px 4px 0;
                    font-size: 13px;
                ">
                <span style="color:#666;font-size:11px">{_ts_str}</span>
                &nbsp;
                <span style="color:{_color};font-weight:600">{_icon} {_stage_lbl}</span>
                &nbsp;
                <span style="color:#ccc">{_ticker_lbl}</span>
                &nbsp;—&nbsp;
                <span style="color:#eee">{_row['message']}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Tab: Portfolio ────────────────────────────────────────────────────────────
with tab_portfolio:
    st.header("Portfolio Overview")

    try:
        account  = client.get_account()
        pv       = float(account.portfolio_value)
        cash     = float(account.cash)
        bp       = float(account.buying_power)
        pl       = pv - START_VALUE
        pl_pct   = pl / START_VALUE

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value",      f"${pv:,.2f}")
        c2.metric("P&L vs $100k Start",   f"${pl:+,.2f}", f"{pl_pct:+.2%}")
        c3.metric("Cash",                 f"${cash:,.2f}")
        c4.metric("Buying Power",         f"${bp:,.2f}")
    except Exception as e:
        st.error(f"Could not fetch account: {e}")

    st.divider()

    col_pos, col_prices = st.columns(2)

    with col_pos:
        st.subheader("Open Positions")
        try:
            positions = client.get_all_positions()
            if not positions:
                st.info("No open positions.")
            else:
                quotes = get_latest_quotes(SYMBOLS)
                rows = []
                for p in positions:
                    sym     = p.symbol
                    current = quotes.get(sym, {}).get("mid", float(p.current_price or 0))
                    entry   = float(p.avg_entry_price)
                    qty     = int(float(p.qty))
                    pl_pos  = (current - entry) * qty
                    rows.append({
                        "Symbol":  sym,
                        "Qty":     qty,
                        "Entry":   f"${entry:.2f}",
                        "Current": f"${current:.2f}",
                        "P&L $":   f"${pl_pos:+,.2f}",
                        "P&L %":   f"{(current - entry) / entry:+.2%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load positions: {e}")

    with col_prices:
        st.subheader("Live Prices")
        try:
            quotes = get_latest_quotes(SYMBOLS)
            rows = [
                {"Symbol": sym, "Price": f"${q['mid']:.2f}",
                 "Fresh": "✅" if not q.get("is_stale") else "⏳ stale"}
                for sym, q in quotes.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not fetch prices: {e}")

    st.divider()

    st.subheader("Portfolio Value Over Time")
    try:
        history = db.get_portfolio_history(limit=288)
        if not history:
            st.info("No history yet — appears after the first snapshot.")
        else:
            df_hist = pd.DataFrame(history)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], utc=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["total_value"],
                mode="lines", line=dict(color="#00cc88", width=2),
                fill="tozeroy", fillcolor="rgba(0,204,136,0.08)",
            ))
            fig.add_hline(y=START_VALUE, line_dash="dot", line_color="#888",
                          annotation_text=f"Start ${START_VALUE:,.0f}")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(tickprefix="$", tickformat=",.0f"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load history: {e}")

    st.divider()

    col_t, col_s = st.columns(2)

    with col_t:
        st.subheader("Recent Trades")
        try:
            trades = db.get_recent_trades(limit=20)
            if not trades:
                st.info("No trades logged yet.")
            else:
                df_t = pd.DataFrame(trades)
                df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], utc=True).dt.strftime("%m/%d %H:%M")
                df_t["price"] = df_t["price"].apply(
                    lambda x: "unknown" if x is None or x != x else f"${x:.2f}")   # NULL price: M116
                st.dataframe(df_t, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load trades: {e}")

    with col_s:
        st.subheader("Recent Signals")
        try:
            signals = db.get_recent_signals(limit=20)
            if not signals:
                st.info("No signals logged yet.")
            else:
                df_s = pd.DataFrame(signals)
                df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True).dt.strftime("%m/%d %H:%M")
                df_s["confidence"] = df_s["confidence"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_s, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load signals: {e}")


# ── Tab 2: Market Regime ──────────────────────────────────────────────────────
with tab_regime:
    st.header("Market Regime")

    if st.button("Refresh Regime", key="regime_refresh"):
        st.rerun()

    try:
        regime_data = db.get_latest_regime()
        if not regime_data:
            st.info(
                "No regime data yet. Regime is assessed at 9:30 AM ET on market days "
                "or at startup via `python main.py`."
            )
        else:
            rname  = regime_data.get("regime", "Unknown")
            rscore = float(regime_data.get("score") or 0)
            rts    = regime_data.get("timestamp", "")

            remoji = (
                "🟢🟢" if "Strong Risk-On" in rname else
                "🟢"  if "Risk-On"        in rname else
                "🔴🔴" if "Strong Risk-Off" in rname else
                "🔴"  if "Risk-Off"       in rname else
                "🟡"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Regime", f"{remoji} {rname}")
            c2.metric("Score", f"{rscore:.1f} / 100")
            if rts:
                c3.metric("Last Updated", pd.to_datetime(rts, utc=True).strftime("%b %d %I:%M %p UTC"))

            st.progress(rscore / 100)

            # Markov persistence signal (guarded — older DB rows predate this key)
            _markov = (regime_data.get("signals_json") or {}).get("markov") or {}
            _mstate = _markov.get("state")
            _mstick = _markov.get("stickiness")
            _msig   = _markov.get("signal")
            if _mstate is not None and _mstick is not None and _msig is not None:
                st.caption(
                    f"SPY Markov state: **{_mstate}** "
                    f"(stickiness {_mstick:.0%}, directional signal {_msig:+.2f})"
                )

            # Portfolio impact guidance
            impact = regime_data.get("portfolio_impact_json") or {}
            if impact:
                st.divider()
                st.subheader("Portfolio Impact Guidance")
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Equity Exposure", impact.get("equity_exposure", "—"))
                ic2.metric("Cash Target",     impact.get("cash_target", "—"))
                lev_allowed = impact.get("options_leverage_allowed", False)
                ic3.metric("Leverage / Options", "✅ Allowed" if lev_allowed else "⛔ Restricted")

            # Signal breakdown
            signals_json = regime_data.get("signals_json") or {}
            if signals_json:
                st.divider()
                st.subheader("Signal Scores")
                sig_rows = [
                    {"Signal": k.replace("_", " ").title(), "Value": v}
                    for k, v in signals_json.items()
                ]
                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

            reasoning = regime_data.get("reasoning", "")
            if reasoning:
                st.divider()
                with st.expander("Model Reasoning"):
                    st.text(reasoning)

    except Exception as e:
        st.warning(f"Could not load regime data: {e}")


# ── Tab 3: Pending Approvals ───────────────────────────────────────────────────
with tab_approvals:
    st.header("Pending Approvals")
    st.caption("Recommendations waiting for human approval via Telegram inline buttons.")

    col_filter, col_refresh = st.columns([4, 1])
    status_filter = col_filter.selectbox(
        "Status", ["PENDING", "APPROVED", "REJECTED", "ALL"], index=0, key="appr_status"
    )
    if col_refresh.button("Refresh", key="appr_refresh"):
        st.rerun()

    try:
        status_arg = None if status_filter == "ALL" else status_filter
        approvals  = db.get_pending_approvals(status=status_arg)

        if not approvals:
            st.info(f"No {status_filter.lower()} approvals.")
        else:
            rows = []
            for a in approvals:
                stop_raw = a.get("stop_loss_price")
                rows.append({
                    "ID":         a.get("id", ""),
                    "Ticker":     a.get("ticker", ""),
                    "Action":     a.get("action", ""),
                    "Conviction": f"{a.get('conviction', 0)}/10",
                    "Bucket":     a.get("thesis_bucket", ""),
                    "Size %":     f"{float(a.get('position_size_pct') or 0):.1%}",
                    "Stop":       f"${float(stop_raw):.2f}" if stop_raw else "—",
                    "Status":     a.get("status", ""),
                    "Sent":       pd.to_datetime(a.get("timestamp"), utc=True).strftime("%m/%d %H:%M")
                                  if a.get("timestamp") else "",
                })

            df_a = pd.DataFrame(rows)
            st.dataframe(df_a, use_container_width=True, hide_index=True)

            # Detail view
            st.divider()
            st.subheader("Detail")
            sel_id = st.selectbox("View reasoning for ID:", [r["ID"] for r in rows], key="appr_sel")
            for a in approvals:
                if a.get("id") == sel_id:
                    st.text_area("Reasoning", a.get("reasoning", ""), height=150, disabled=True)
                    with st.expander("Entry Strategy"):
                        st.text(a.get("entry_strategy", "—"))
                    break

    except Exception as e:
        st.warning(f"Could not load approvals: {e}")


# ── Tab 4: Outcomes ───────────────────────────────────────────────────────────
with tab_outcomes:
    st.header("Trade Outcomes")
    st.caption("30/60/90-day performance tracking for approved trades. Populates over time.")

    if st.button("Refresh", key="outcomes_refresh"):
        st.rerun()

    try:
        summary = db.get_outcomes_summary()
        all_outcomes = db.get_all_outcomes(limit=200)

        if not all_outcomes:
            st.info("No outcome records yet. Outcomes are created automatically when a trade is approved and executed.")
        else:
            # Summary metrics
            total = sum(b["total"] for b in summary.values())
            wins  = sum(b["wins"]  for b in summary.values())
            overall_wr = wins / total if total else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Trades Tracked", total)
            m2.metric("Overall Win Rate", f"{overall_wr:.0%}" if total else "—")
            m3.metric("Buckets Active", len(summary))

            # By bucket table
            if summary:
                st.divider()
                st.subheader("By Bucket")
                bucket_rows = [
                    {
                        "Bucket":      bucket,
                        "Trades":      d["total"],
                        "Win Rate":    f"{d['win_rate']:.0%}" if d["total"] else "—",
                        "Avg Outcome": f"{d['avg_outcome_pct']:.1%}" if d["avg_outcome_pct"] else "—",
                    }
                    for bucket, d in summary.items()
                ]
                st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)

            # Full log
            st.divider()
            st.subheader("Full Outcome Log")
            outcome_rows = []
            for o in all_outcomes:
                ep = o.get("entry_price")
                outcome_rows.append({
                    "Ticker":    o.get("ticker", ""),
                    "Action":    o.get("action", ""),
                    "Conviction": f"{o.get('conviction', '—')}/10",
                    "Bucket":    o.get("thesis_bucket", ""),
                    "Entry":     f"${float(ep):.2f}" if ep else "—",
                    "Entry Date": pd.to_datetime(o.get("entry_date"), utc=True).strftime("%m/%d/%y") if o.get("entry_date") else "—",
                    "30d":       f"${float(o['review_30d_price']):.2f}" if o.get("review_30d_price") else "—",
                    "60d":       f"${float(o['review_60d_price']):.2f}" if o.get("review_60d_price") else "—",
                    "90d":       f"${float(o['review_90d_price']):.2f}" if o.get("review_90d_price") else "—",
                    "Outcome %": f"{float(o['outcome_pct']):.1%}" if o.get("outcome_pct") else "Pending",
                    "Alpha vs SPY": f"{float(o['alpha_pct']):+.1%}" if o.get("alpha_pct") is not None else "—",
                })
            st.dataframe(pd.DataFrame(outcome_rows), use_container_width=True, hide_index=True)

            # Scatter plot (conviction vs outcome) — fills in over time
            completed = [o for o in all_outcomes if o.get("outcome_pct") is not None and o.get("conviction")]
            if completed:
                st.divider()
                st.subheader("Conviction vs Outcome")
                scatter_df = pd.DataFrame([
                    {
                        "Conviction": o["conviction"],
                        "Outcome %":  float(o["outcome_pct"]) * 100,
                        "Bucket":     o.get("thesis_bucket", ""),
                        "Ticker":     o.get("ticker", ""),
                    }
                    for o in completed
                ])
                fig = go.Figure()
                colors = {"Compounder": "#00cc88", "Accelerator": "#3399ff", "Moonshot": "#ff9900"}
                for bucket in scatter_df["Bucket"].unique():
                    sub = scatter_df[scatter_df["Bucket"] == bucket]
                    fig.add_trace(go.Scatter(
                        x=sub["Conviction"], y=sub["Outcome %"],
                        mode="markers+text", text=sub["Ticker"],
                        textposition="top center", name=bucket,
                        marker=dict(size=10, color=colors.get(bucket, "#888")),
                    ))
                fig.add_hline(y=0, line_dash="dot", line_color="#888")
                fig.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Conviction (1-10)", yaxis_title="Outcome %",
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load outcomes: {e}")


# ── Tab: Performance ──────────────────────────────────────────────────────────
with tab_performance:
    st.header("Performance vs Benchmark")
    st.caption(
        "Before this tab existed, there was no way to answer \"are we actually beating "
        "the market?\" — the only comparison was a flat line at the $100k starting value."
    )

    if st.button("Refresh", key="perf_refresh"):
        st.rerun()

    try:
        history = db.get_portfolio_history(limit=2000)
        if not history:
            st.info("No history yet — appears after the first snapshot.")
        else:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["spy"] = df["benchmark"].apply(lambda b: b.get("SPY") if isinstance(b, dict) else None)
            df["qqq"] = df["benchmark"].apply(lambda b: b.get("QQQ") if isinstance(b, dict) else None)

            has_benchmark = df["spy"].notna().any()

            st.subheader("Equity Curve vs Benchmark")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["total_value"],
                mode="lines", name="Portfolio", line=dict(color="#00cc88", width=2),
            ))
            if has_benchmark:
                base_value = df["total_value"].iloc[0]
                for col, name, color in (("spy", "SPY (rebased)", "#3399ff"),
                                         ("qqq", "QQQ (rebased)", "#ff9900")):
                    sub = df[df[col].notna()]
                    if len(sub) < 2:
                        continue
                    base_price = sub[col].iloc[0]
                    rebased = base_value * (sub[col] / base_price)
                    fig.add_trace(go.Scatter(
                        x=sub["timestamp"], y=rebased, mode="lines",
                        name=name, line=dict(color=color, width=1.5, dash="dot"),
                    ))
            else:
                fig.add_hline(y=START_VALUE, line_dash="dot", line_color="#888",
                              annotation_text=f"Start ${START_VALUE:,.0f}")
                st.caption("Benchmark data not yet available — will appear as snapshots accumulate "
                          "(requires the Phase 5 take_portfolio_snapshot update to have run).")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(tickprefix="$", tickformat=",.0f"))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Drawdown")
            running_max = df["total_value"].cummax()
            drawdown_pct = (df["total_value"] - running_max) / running_max
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=df["timestamp"], y=drawdown_pct * 100, mode="lines",
                line=dict(color="#cc4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(204,68,68,0.15)",
            ))
            fig_dd.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis=dict(ticksuffix="%"), showlegend=False)
            st.plotly_chart(fig_dd, use_container_width=True)
            st.caption(f"Max drawdown observed: {drawdown_pct.min():.1%}")

            # Daily-resampled returns for Sharpe/Sortino/alpha — the raw
            # series is 5-min ticks, which overstates volatility if used
            # directly for an annualized ratio.
            daily = df.set_index("timestamp").resample("1D").last().dropna(subset=["total_value"])
            if len(daily) >= 5:
                daily_returns = daily["total_value"].pct_change().dropna()
                sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
                         if daily_returns.std() > 0 else None)
                downside = daily_returns[daily_returns < 0]
                sortino = (daily_returns.mean() / downside.std() * (252 ** 0.5)
                          if len(downside) > 1 and downside.std() > 0 else None)

                c1, c2, c3 = st.columns(3)
                c1.metric("Annualized Sharpe", f"{sharpe:.2f}" if sharpe is not None else "—")
                c2.metric("Annualized Sortino", f"{sortino:.2f}" if sortino is not None else "—")

                if has_benchmark and daily["spy"].notna().sum() >= 5:
                    spy_returns = daily["spy"].pct_change().dropna()
                    aligned = pd.concat([daily_returns, spy_returns], axis=1, join="inner")
                    aligned.columns = ["portfolio", "spy"]
                    if len(aligned) >= 5 and aligned["spy"].var() > 0:
                        beta = aligned["portfolio"].cov(aligned["spy"]) / aligned["spy"].var()
                        alpha_annualized = (aligned["portfolio"].mean() - beta * aligned["spy"].mean()) * 252
                        c3.metric("Beta vs SPY", f"{beta:.2f}")
                        st.caption(f"Annualized alpha vs SPY (CAPM): {alpha_annualized:+.1%}")
            else:
                st.caption("Need at least 5 days of snapshots to compute Sharpe/Sortino/beta.")
    except Exception as e:
        st.warning(f"Could not load performance chart: {e}")

    st.divider()

    col_attr, col_slip = st.columns(2)
    with col_attr:
        st.subheader("Bucket Attribution")
        st.caption("Approximate — see get_bucket_attribution() docstring for the method.")
        try:
            attribution = db.get_bucket_attribution()
            if not attribution:
                st.info("No closed trades with sizing data yet.")
            else:
                rows = [
                    {
                        "Bucket": a["thesis_bucket"],
                        "Closed Trades": a["closed_trades"],
                        "$ P&L": f"${a['dollar_pnl']:+,.2f}",
                        "$ Deployed": f"${a['dollar_deployed']:,.2f}",
                        "Return on Deployed": (f"{a['dollar_pnl']/a['dollar_deployed']:+.1%}"
                                               if a["dollar_deployed"] else "—"),
                    }
                    for a in attribution
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load attribution: {e}")

    with col_slip:
        st.subheader("Execution Quality")
        try:
            slippage = db.get_slippage_summary()
            st.metric("Orders with recorded slippage", slippage.get("count", 0))
            avg = slippage.get("avg_slippage_pct")
            st.metric("Average slippage (cost)", f"{avg:+.2%}" if avg is not None else "—")
        except Exception as e:
            st.warning(f"Could not load execution stats: {e}")

    st.divider()
    st.subheader("Ingestion Health")
    try:
        health = db.get_ingestion_health()
        if not health:
            st.info("No ingestion heartbeats recorded yet.")
        else:
            rows = []
            for h in health:
                last_ok = h.get("last_success_at")
                rows.append({
                    "Source": h["source"],
                    "Last Success": last_ok.strftime("%m/%d %H:%M") if last_ok else "never",
                    "Last Item Count": h.get("last_item_count"),
                    "Last Error": (h.get("last_error") or "")[:80],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load ingestion health: {e}")


# ── Tab 5: Macro ───────────────────────────────────────────────────────────────
with tab_macro:
    st.header("Macro Environment (FRED)")

    @st.cache_data(ttl=3600)
    def _macro_snapshot() -> dict:
        from ingestion.macro import get_macro_snapshot
        return get_macro_snapshot()

    @st.cache_data(ttl=3600)
    def _macro_summary() -> str:
        from ingestion.macro import get_macro_summary_text
        return get_macro_summary_text()

    if st.button("Refresh Macro Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching data from FRED..."):
        snap    = _macro_snapshot()
        summary = _macro_summary()

    # Key metrics grid
    cols = st.columns(5)
    key_series = ["FEDFUNDS", "VIXCLS", "DGS10", "T10Y2Y", "UNRATE"]
    for i, sid in enumerate(key_series):
        d = snap.get(sid, {})
        val = d.get("value")
        cols[i].metric(d.get("label", sid), f"{val:.2f}" if val is not None else "N/A")

    st.divider()

    # Full table
    rows = []
    for sid, d in snap.items():
        rows.append({"Indicator": d.get("label", sid), "Value": d.get("value"), "Date": d.get("date")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # Summary text (used by the AI model for context)
    with st.expander("Model Context Text (what the AI sees)"):
        st.text(summary)


# ── Tab 3: Technicals ─────────────────────────────────────────────────────────
with tab_technicals:
    st.header("Technical Indicators (Polygon)")

    symbol = st.selectbox("Select Symbol", SYMBOLS, key="tech_symbol")

    @st.cache_data(ttl=300)
    def _snapshot(sym: str) -> dict:
        from ingestion.polygon_stream import get_snapshot
        return get_snapshot(sym)

    @st.cache_data(ttl=300)
    def _rsi(sym: str) -> float | None:
        from ingestion.polygon_stream import get_rsi
        return get_rsi(sym)

    @st.cache_data(ttl=300)
    def _macd(sym: str) -> dict | None:
        from ingestion.polygon_stream import get_macd
        return get_macd(sym)

    if st.button("Refresh", key="tech_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner(f"Fetching {symbol}..."):
        snap_tech = _snapshot(symbol)
        rsi_val   = _rsi(symbol)
        macd_val  = _macd(symbol)

    # Price row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price",      f"${snap_tech.get('price') or 0:.2f}")
    c2.metric("Change",     f"{snap_tech.get('change_pct') or 0:+.2f}%")
    c3.metric("VWAP",       f"${snap_tech.get('vwap') or 0:.2f}")
    c4.metric("Volume",     f"{int(snap_tech.get('volume') or 0):,}")
    c5.metric("Prev Close", f"${snap_tech.get('prev_close') or 0:.2f}")

    st.divider()

    # Indicators row
    c1, c2, c3 = st.columns(3)

    if rsi_val is not None:
        rsi_label = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "Neutral"
        rsi_color = "inverse" if rsi_val > 70 else "normal" if rsi_val < 30 else "off"
        c1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label, delta_color=rsi_color)
    else:
        c1.metric("RSI (14)", "N/A")

    if macd_val:
        c2.metric("MACD", f"{macd_val.get('value') or 0:.4f}", macd_val.get("trend"))
        c3.metric("MACD Histogram", f"{macd_val.get('histogram') or 0:.4f}")
    else:
        c2.metric("MACD", "N/A")
        c3.metric("MACD Histogram", "N/A")

    st.divider()

    # Day range bar
    low  = snap_tech.get("low")
    high = snap_tech.get("high")
    price = snap_tech.get("price")
    if low and high and price:
        st.subheader("Day Range")
        pct_in_range = (price - low) / (high - low) if high != low else 0.5
        st.progress(pct_in_range, text=f"${low:.2f}  ←  ${price:.2f}  →  ${high:.2f}")


# ── Tab 6: Signal Log ────────────────────────────────────────────────────────
with tab_signals:
    st.header("Signal Log")

    col_filter, col_limit = st.columns([3, 1])
    limit = col_limit.number_input("Show", min_value=10, max_value=500, value=100, step=10)

    try:
        signals = db.get_recent_signals(limit=int(limit))
        if not signals:
            st.info("No signals yet.")
        else:
            df_s = pd.DataFrame(signals)
            df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True).dt.strftime("%m/%d %H:%M")
            # Handle both legacy float (0-1) and new conviction integer (1-10)
            df_s["conviction"] = df_s["confidence"].apply(
                lambda x: f"{int(round(x))}/10" if x is not None and x > 1.0 else (f"{x:.0%}" if x is not None else "—")
            )
            df_s = df_s.drop(columns=["confidence"], errors="ignore")

            all_sigs = sorted(df_s["signal"].unique().tolist())
            selected = col_filter.multiselect("Filter signals", all_sigs, default=all_sigs)
            df_s = df_s[df_s["signal"].isin(selected)]

            st.dataframe(df_s, use_container_width=True, hide_index=True, height=600)
    except Exception as e:
        st.warning(f"Could not load signals: {e}")


# ── Tab 6: Logs ───────────────────────────────────────────────────────────────
with tab_logs:
    st.header("Live Logs")

    log_tab_labels = [svc["label"] for svc in pm.SERVICES.values()] + ["Ollama"]
    agent_tabs = st.tabs(log_tab_labels)

    for tab, (name, svc) in zip(agent_tabs, pm.SERVICES.items()):
        with tab:
            running = status[name]
            st.caption(f"Status: {'🟢 Running' if running else '🔴 Stopped'}")
            lines = st.slider("Lines", 20, 300, 80, key=f"logslider_{name}")
            st.code(pm.get_log(name, lines), language=None)

    with agent_tabs[-1]:
        ollama_ok, ollama_info = pm.ollama_status()
        st.caption(f"Status: {'🟢 Running' if ollama_ok else '🔴 Not running'} — {ollama_info}")
        if sys.platform == "win32":
            ollama_log = Path(os.path.expanduser("~")) / "AppData" / "Local" / "Ollama" / "server.log"
        else:
            # launch.sh redirects `ollama serve` here; a systemd-managed
            # install instead logs to `journalctl -u ollama`.
            ollama_log = Path(__file__).parent / "logs" / "ollama.log"
        if ollama_log.exists():
            lines = st.slider("Lines", 20, 300, 80, key="logslider_ollama")
            text = ollama_log.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(text[-lines:]), language=None)
        elif sys.platform == "win32":
            st.info("Ollama log not found. Check system tray to confirm Ollama is running.")
        else:
            st.info(f"Ollama log not found at {ollama_log}. If Ollama runs as a systemd "
                     "service, check `journalctl -u ollama` instead.")


# ── Tab 5: Manual Trade ───────────────────────────────────────────────────────
with tab_trade:
    st.header("Manual Trade")
    st.warning("Paper trading only — executes on IBKR paper account (TWS must be running).")

    col_form, col_confirm = st.columns(2)

    with col_form:
        trade_symbol = st.selectbox("Symbol", SYMBOLS, key="manual_symbol")
        trade_side   = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        trade_qty    = st.number_input("Quantity (shares)", min_value=1, max_value=10000, value=1, step=1)

        try:
            q         = get_latest_quotes([trade_symbol])
            est_price = q.get(trade_symbol, {}).get("mid", 0.0)
            st.metric("Current Price",    f"${est_price:.2f}")
            st.metric("Estimated Value",  f"${est_price * trade_qty:,.2f}")
        except Exception:
            est_price = 0.0
            st.caption("Could not fetch current price.")

    with col_confirm:
        st.subheader("Order Summary")
        side_color = "green" if trade_side == "BUY" else "red"
        st.markdown(f"""
| Field | Value |
|---|---|
| Symbol | **{trade_symbol}** |
| Side | **:{side_color}[{trade_side}]** |
| Quantity | **{trade_qty} shares** |
| Type | Market (DAY) |
| Est. Value | **${est_price * trade_qty:,.2f}** |
        """)

        confirmed = st.checkbox("I confirm this is a paper trade and I want to submit this order.")

        if st.button("Submit Order", type="primary", disabled=not confirmed):
            try:
                # Routed through execution.execute_manual — previously this
                # called place_market_order directly, which skipped stop
                # placement on BUY (the position was invisible to the
                # trailing-stop monitor), didn't clamp SELL to owned shares
                # or cancel existing stops first, and logged the trade at
                # the pre-order estimated price instead of the real fill.
                from execution import execute_manual
                result = execute_manual(trade_symbol, trade_side, trade_qty,
                                        reason="Manual trade from dashboard")
                if result["success"]:
                    st.success(f"Order executed: {result['message']}")
                else:
                    st.error(f"Order failed: {result['message']}")
            except Exception as e:
                st.error(f"Order failed: {e}")
