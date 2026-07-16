"""
Central configuration. All strategy parameters and risk rules live here.
Symbol universe is now driven by knowledge/watchlist.md via knowledge_loader.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Symbol universe — derived from watchlist at import time
# Keep SYMBOLS for any module that still references it directly.
# ---------------------------------------------------------------------------
def _load_symbols() -> list[str]:
    try:
        from knowledge_loader import get_all_tickers
        tickers = get_all_tickers()
        if tickers:
            return tickers
    except Exception:
        pass
    # Fallback if knowledge_loader isn't available yet
    return [
        "META", "AAPL", "NVDA", "GOOGL", "ORCL", "MSFT", "AMZN", "TSLA",
        "AMD", "AVGO", "QCOM", "MU", "ARM", "AMAT", "LRCX", "ASML", "KLAC",
        "PLTR", "CRM", "NOW", "SNOW", "NET", "DDOG",
        "IONQ", "RGTI", "QBTS", "QUBT",
    ]


SYMBOLS: list[str] = _load_symbols()

# ---------------------------------------------------------------------------
# Portfolio baseline — must match this account's actual starting net
# liquidation value, not an arbitrary round number. Used as the P&L=0 line
# on the dashboard and as the fallback high-water-mark seed for the
# drawdown-halt circuit breaker (tasks.check_portfolio_drawdown) the first
# time it ever runs against a given database. IBKR paper accounts seed at
# $1,000,000 by default, not $100k — a mismatch here either makes the
# dashboard show a fake multi-hundred-thousand-dollar "profit," or, worse,
# could false-trigger the drawdown halt on first run if the real balance is
# ever *below* this constant.
START_VALUE = 1_001_433.51
POLL_INTERVAL_SECONDS = 300  # 5 minutes (portfolio snapshots)

# ---------------------------------------------------------------------------
# Risk rules — authoritative source; risk_manager.py reads these
# ---------------------------------------------------------------------------
DRAWDOWN_HALT_PCT = 0.10   # Halt all trading if portfolio drops 10% from start

# Per-bucket position caps (initial entry ceiling)
MAX_POSITION_PCT_COMPOUNDER  = 0.12   # 12% at entry; allow up to 15% via gains
MAX_POSITION_PCT_ACCELERATOR = 0.07   # 7% at entry; allow up to 8% via gains
MAX_POSITION_PCT_MOONSHOT    = 0.02   # 2% hard cap always

# Theme / sector concentration limits
MAX_SINGLE_THEME_PCT   = 0.35   # 35% per sector/thesis
MAX_AI_INFRA_PCT       = 0.60   # 60% total AI infrastructure exposure
MAX_MOONSHOT_TOTAL_PCT = 0.10   # 10% total moonshots

# Cash-reserve cap — a single trade's initial tranche may spend at most this
# fraction of *current cash on hand*, not portfolio value. Because the cap
# shrinks along with cash, spend decays geometrically toward zero instead of
# a burst of same-day auto-executes ever being able to fully drain the
# account (see agents/crew.py auto-execute, no human tap since 2026-07-14).
MAX_TRADE_PCT_OF_CASH_RESERVE = 0.20   # 20% of remaining cash per trade

# Sizing table: (bucket, conviction_band) → (min_pct, max_pct) of portfolio
# conviction_band is the lower bound of the 2-point band
BUCKET_SIZE_TABLE: dict[str, dict[int, tuple[float, float]]] = {
    "Compounder":  {9: (0.08, 0.12), 7: (0.05, 0.08), 5: (0.03, 0.05), 3: (0.02, 0.03)},
    "Accelerator": {9: (0.04, 0.07), 7: (0.03, 0.05), 5: (0.02, 0.03), 3: (0.01, 0.02)},
    "Moonshot":    {9: (0.01, 0.02), 7: (0.005, 0.01), 5: (0.0025, 0.005), 3: (None, None)},
}

# Regime sizing multiplier
REGIME_MULTIPLIER: dict[str, float] = {
    "Strong Risk-On": 1.2,
    "Risk-On":        1.0,
    "Neutral":        0.75,
    "Risk-Off":       0.5,
    "Strong Risk-Off": 0.25,
}

# ATR stop multipliers by bucket
ATR_STOP_MULTIPLIER: dict[str, tuple[float, float]] = {
    "Compounder":  (2.0, 3.0),    # 2-3× ATR
    "Accelerator": (1.5, 2.0),    # 1.5-2× ATR
    "Moonshot":    (None, None),   # defined max loss, not ATR-based
}
MOONSHOT_MAX_LOSS_PCT = 0.35   # 35% of position value for moonshot stop

# Trailing stop activation thresholds
TRAILING_STOP_ACTIVATE: dict[str, float] = {
    "Compounder":  0.30,   # activate after 30% gain
    "Accelerator": 0.20,   # activate after 20% gain
}
TRAILING_STOP_DISTANCE: dict[str, tuple[float, float]] = {
    "Compounder":  (0.15, 0.20),
    "Accelerator": (0.10, 0.15),
}

# ---------------------------------------------------------------------------
# Buy cooldown
# ---------------------------------------------------------------------------
# Minimum gap between auto-executed OPEN/ADD buys on the same ticker, unless
# the new signal's conviction exceeds the prior one. has_active_approval only
# blocks a ticker with a PENDING/PROCESSING row; auto-execute clears that
# within seconds, so independent event sources (EDGAR/options/news/insider)
# could each re-buy the same ticker minutes apart — this is what drove the
# 2026-07-15 10-trade, $0-cash morning.
REBUY_COOLDOWN_HOURS = 4

# ---------------------------------------------------------------------------
# Sell behaviour
# ---------------------------------------------------------------------------
TRIM_FRACTION = 0.33   # analyst TRIM sells this fraction of the held position
# When a monitored stop level is breached and NO protective stop exists at
# the broker, execute a market sell of the full position instead of only
# alerting. Only ever reduces exposure; set False to fall back to alert-only.
AUTO_EXECUTE_PROTECTIVE_SELLS = True

# ---------------------------------------------------------------------------
# Agent pipeline thresholds
# ---------------------------------------------------------------------------
MIN_CONVICTION_TO_QUEUE   = 5      # analyst conviction score (1-10) min to pass to risk
MIN_UPSIDE_DOWNSIDE_RATIO = 2.0    # minimum 2:1 upside/downside required
MIN_COMPOSITE_SCORE_HIGH  = 75     # scanner score threshold for HIGH priority
MIN_COMPOSITE_SCORE_MEDIUM = 50    # scanner score threshold for MEDIUM priority

# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------
REGIME_CACHE_TTL_SECONDS = 1800   # 30 minutes — allows mid-session regime refresh

REGIME_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Strong Risk-On": (75, 100),
    "Risk-On":        (55, 74),
    "Neutral":        (40, 54),
    "Risk-Off":       (20, 39),
    "Strong Risk-Off": (0, 19),
}

# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------
APPROVAL_EXPIRY_HOURS = 6     # pending approvals auto-expire after 6 hours

# ---------------------------------------------------------------------------
# Portfolio safety
# ---------------------------------------------------------------------------
MAX_DRAWDOWN_PCT = 0.15       # alert at 15% drawdown from START_VALUE

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
# When True, new BUY entries use a native IBKR bracket order (entry + stop
# linked by parentId) instead of a market order followed by a separate stop
# placement. This makes the "orphaned stop after a manual exit" class of bug
# structurally impossible for new positions — but it's new code that hasn't
# been exercised against a live TWS session yet, so it ships OFF by default.
# Flip on only after verifying place_bracket_order in paper trading.
import os as _os
USE_BRACKET_ORDERS = _os.getenv("USE_BRACKET_ORDERS", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Ollama model config
# ---------------------------------------------------------------------------
import os
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SCREENER_MODEL   = os.getenv("SCREENER_MODEL", "qwen2.5:7b")
ANALYST_MODEL    = os.getenv("ANALYST_MODEL", "qwen2.5:72b")
CHAT_MODEL       = os.getenv("CHAT_MODEL",    "qwen2.5:3b")   # never use 72b — must not be interrupted

# ---------------------------------------------------------------------------
# Interactive Brokers connection
# ---------------------------------------------------------------------------
IB_HOST      = os.getenv("IB_HOST",      "127.0.0.1")
IB_PORT      = int(os.getenv("IB_PORT",      "7497"))   # 7497=paper, 7496=live
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
