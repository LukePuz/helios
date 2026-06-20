import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# streamlit-autorefresh is optional — fall back to a manual button if missing
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

import db
from market_data import get_latest_quotes
from alpaca.trading.client import TradingClient

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
SYMBOLS = ["META", "AAPL", "NVDA", "GOOGL", "ORCL"]
START_VALUE = 100_000.0

st.set_page_config(page_title="MANGOS Trading Bot", layout="wide")

if _HAS_AUTOREFRESH:
    st_autorefresh(interval=30_000, key="dashboard_refresh")
else:
    if st.button("Refresh"):
        st.rerun()

# Cache the Alpaca client for the session
@st.cache_resource
def _trading_client() -> TradingClient:
    return TradingClient(API_KEY, API_SECRET, paper=True)

client = _trading_client()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("MANGOS Paper Trading Dashboard")
st.caption("META · AAPL · NVDA · GOOGL · ORCL  |  Paper trading only  |  Refreshes every 30s")

# ---------------------------------------------------------------------------
# Account metrics
# ---------------------------------------------------------------------------
try:
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    pl = portfolio_value - START_VALUE
    pl_pct = pl / START_VALUE

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    c2.metric("P&L vs $100k Start", f"${pl:+,.2f}", f"{pl_pct:+.2%}")
    c3.metric("Cash", f"${cash:,.2f}")
    c4.metric("Buying Power", f"${buying_power:,.2f}")
except Exception as e:
    st.error(f"Could not fetch account: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Positions + Live Prices
# ---------------------------------------------------------------------------
col_pos, col_prices = st.columns(2)

with col_pos:
    st.subheader("Open Positions")
    try:
        positions = client.get_all_positions()
        if not positions:
            st.info("No open positions.")
        else:
            try:
                quotes = get_latest_quotes(SYMBOLS)
            except Exception:
                quotes = {}
            rows = []
            for p in positions:
                sym = p.symbol
                current = quotes.get(sym, {}).get("mid", float(p.current_price or 0))
                entry = float(p.avg_entry_price)
                qty = int(float(p.qty))
                pl_pos = (current - entry) * qty
                pl_pct_pos = (current - entry) / entry
                rows.append({
                    "Symbol": sym,
                    "Qty": qty,
                    "Entry": f"${entry:.2f}",
                    "Current": f"${current:.2f}",
                    "P&L $": f"${pl_pos:+,.2f}",
                    "P&L %": f"{pl_pct_pos:+.2%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load positions: {e}")

with col_prices:
    st.subheader("Live MANGOS Prices")
    try:
        quotes = get_latest_quotes(SYMBOLS)
        price_rows = [
            {
                "Symbol": sym,
                "Bid": f"${q['bid']:.2f}",
                "Ask": f"${q['ask']:.2f}",
                "Mid": f"${q['mid']:.2f}",
            }
            for sym, q in quotes.items()
        ]
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not fetch prices: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Portfolio value chart
# ---------------------------------------------------------------------------
st.subheader("Portfolio Value Over Time")
try:
    history = db.get_portfolio_history(limit=288)
    if not history:
        st.info("No history yet — data appears after the first tick.")
    else:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], utc=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["total_value"],
            mode="lines",
            name="Portfolio",
            line=dict(color="#00cc88", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,204,136,0.08)",
        ))
        fig.add_hline(
            y=START_VALUE,
            line_dash="dot",
            line_color="#888",
            annotation_text=f"Start ${START_VALUE:,.0f}",
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load history: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Trades + Signals
# ---------------------------------------------------------------------------
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
            df_t["price"] = df_t["price"].apply(lambda x: f"${x:.2f}")
            st.dataframe(df_t, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load trades: {e}")

with col_s:
    st.subheader("Signal Log")
    try:
        signals = db.get_recent_signals(limit=20)
        if not signals:
            st.info("No signals logged yet.")
        else:
            df_s = pd.DataFrame(signals)
            df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True).dt.strftime("%m/%d %H:%M")
            df_s["confidence"] = df_s["confidence"].apply(lambda x: f"{x:.2%}")
            # Color-code the signal column
            def _color_signal(val: str) -> str:
                if "BUY" in val and "BLOCK" not in val and "REJECT" not in val:
                    return "color: #00cc88"
                if "SELL" in val and "BLOCK" not in val and "REJECT" not in val:
                    return "color: #ff4b4b"
                return "color: #888"
            st.dataframe(
                df_s.style.applymap(_color_signal, subset=["signal"]),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        st.warning(f"Could not load signals: {e}")
