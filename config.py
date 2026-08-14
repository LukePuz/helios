"""
Central configuration. All strategy parameters and risk rules live here.
Symbol universe is now driven by knowledge/watchlist.md via knowledge_loader.
"""
from __future__ import annotations

# .env must be loaded HERE, not assumed already-loaded by whatever imports
# this module first. Every os.getenv() call below (TRADING_ENV in
# particular, now that it's the single source of truth for paper/live)
# silently fell back to its default if config.py happened to be the first
# module imported in the process -- found 2026-08-14 testing the live
# rollout: a standalone `import config` returned TRADING_ENV=paper despite
# .env correctly saying "live", because nothing had loaded .env into
# os.environ yet. ibkr.py/risk_manager.py already do this defensively for
# the same reason; config.py never did despite being what they both defer
# to.
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(dotenv_path=_Path(__file__).parent / ".env")

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
# Trading environment — paper vs. live
# ---------------------------------------------------------------------------
# Single switch that derives every paper/live-dependent constant below
# (broker port, clientId range, portfolio baseline, auto-execution
# permission) so a half-flipped state — e.g. IB_PORT pointed at live while
# START_VALUE is still paper's ~$1M — can't happen from editing one value.
# Added 2026-08-13 ahead of the live rollout; defaults to paper.
import os
TRADING_ENV = os.getenv("TRADING_ENV", "paper").strip().lower()
if TRADING_ENV not in ("paper", "live"):
    raise ValueError(f"TRADING_ENV must be 'paper' or 'live', got {TRADING_ENV!r}")

# Belt-and-suspenders on top of TRADING_ENV: auto-execution
# (agents/crew.py:_route_to_approval) only fires without a human tap when
# this is ALSO explicitly true. Live recommendations instead go through the
# existing Telegram Approve/Reject flow (alerts.send_approval_request) —
# the same human-in-the-loop gate paper trading used before 2026-07-14.
# Two separate, deliberate opt-ins required to reach unattended live
# execution, not one.
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
AUTO_EXECUTE = (TRADING_ENV == "paper") or LIVE_TRADING_ENABLED

if TRADING_ENV == "live":
    _IB_PORT_DEFAULT = "7496"
    _IB_CLIENT_ID_DEFAULT = "40"   # dedicated, distinct from every paper clientId (1, 2, 10-12, 20, 30)
    # Real account balance, set from check_live_account.py's read on
    # 2026-08-13. Used as the P&L=0 dashboard line and the drawdown-halt
    # high-water-mark seed — update via LIVE_START_VALUE in .env if the
    # account is funded further before live trading actually starts; a
    # stale value here makes the halt fire (or fail to fire) on the wrong
    # number.
    START_VALUE = float(os.getenv("LIVE_START_VALUE", "1010.00"))
else:
    _IB_PORT_DEFAULT = "7497"
    _IB_CLIENT_ID_DEFAULT = "1"
    # Portfolio baseline — must match this account's actual starting net
    # liquidation value, not an arbitrary round number. Used as the P&L=0
    # line on the dashboard and as the fallback high-water-mark seed for
    # the drawdown-halt circuit breaker (tasks.check_portfolio_drawdown)
    # the first time it ever runs against a given database. IBKR paper
    # accounts seed at $1,000,000 by default, not $100k — a mismatch here
    # either makes the dashboard show a fake multi-hundred-thousand-dollar
    # "profit," or, worse, could false-trigger the drawdown halt on first
    # run if the real balance is ever *below* this constant.
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

# Conviction-swap — 2026-08-13, added ahead of a live-money switch: when cash
# is too tight to size a high-conviction new idea, check whether an existing
# Accelerator/Moonshot holding has independently fallen weak enough to be
# worth exiting anyway, and if so sell it to free up room. Deliberately
# opportunistic, never forced: SWAP_MIN_CONVICTION_GAP and
# SWAP_MAX_CANDIDATE_CONVICTION must BOTH hold before a position is even
# eligible — a held position at 7+/10 is never touched by this, no matter
# how much higher the new idea scores, because it hasn't independently
# earned an exit. If nothing qualifies, sizing falls through to the existing
# cash-cap/block behavior unchanged. Compounder-bucket holdings (the
# long-term core) are never eligible regardless of conviction.
SWAP_MIN_CONVICTION_GAP        = 2      # new idea must beat the held position by at least this much
SWAP_MAX_CANDIDATE_CONVICTION  = 6      # held position's OWN fresh conviction must be at or below this

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
# Opening-gap confirmation
# ---------------------------------------------------------------------------
# Incident 2026-07-29 (TER): an earnings gap-up doesn't tell you whether the
# move holds or fades -- premarket/opening price discovery runs on a small
# fraction of regular-session volume, and big gaps are routinely given back
# within the first half hour. A BUY (OPEN/ADD) into a gap of at least this
# size from the prior close waits for agents.price_action-style confirmation
# (is the open holding, not already reversing) instead of firing the moment
# the market-hours gate clears -- see execution.execute_approved and
# market_data.check_gap_confirmation.
GAP_CONFIRMATION_THRESHOLD_PCT = 0.05
# If price has already given back this fraction of the gap (open vs prior
# close) the move is judged to be failing -- don't buy into a reversing gap.
GAP_FAIL_RETRACEMENT_PCT = 0.5
# How many times to re-check an unresolved (neither confirmed nor failed)
# gap before giving up -- bounds the retry chain to a fixed window after
# the open rather than deferring indefinitely.
GAP_CONFIRMATION_MAX_RETRIES = 3
GAP_CONFIRMATION_RETRY_MINUTES = 5

# How many times to re-check a BUY that aborted because only a stale price
# was available (IBKR's HMDS intraday-bar service routinely has no data for
# the first few minutes after the open) before giving up -- found
# 2026-07-30/31: this abort path used to be a dead end with no retry at all,
# unlike the two gates above, silently losing every ADD/OPEN signal that
# happened to be in-flight right at market open (5 conviction-7/8 signals
# lost 2026-07-31 alone). Same shape as GAP_CONFIRMATION_*: bounded retries,
# each strictly later than the last, so a persistently-down data feed still
# gives up rather than deferring forever.
STALE_PRICE_MAX_RETRIES = 4
STALE_PRICE_RETRY_MINUTES = 3

# ---------------------------------------------------------------------------
# Sell behaviour
# ---------------------------------------------------------------------------
TRIM_FRACTION = 0.33   # analyst TRIM sells this fraction of the held position
# When a monitored stop level is breached and NO protective stop exists at
# the broker, execute a market sell of the full position instead of only
# alerting. Only ever reduces exposure; set False to fall back to alert-only.
AUTO_EXECUTE_PROTECTIVE_SELLS = True

# reconcile_stop_orders runs every 15 min (tasks.py beat schedule) and used
# to re-alert every unresolved issue (e.g. a position missing a stop) on
# every single cycle — ~96 identical Telegram messages/day for one standing
# condition. Re-alert the same issue at most once per this many hours;
# genuinely new issues or resolved-this-cycle actions always alert.
RECONCILE_ALERT_THROTTLE_HOURS = 4

# ---------------------------------------------------------------------------
# Agent pipeline thresholds
# ---------------------------------------------------------------------------
MIN_CONVICTION_TO_QUEUE   = 5      # analyst conviction score (1-10) min to pass to risk
MIN_UPSIDE_DOWNSIDE_RATIO = 2.0    # minimum 2:1 upside/downside required
MIN_COMPOSITE_SCORE_HIGH  = 75     # scanner score threshold for HIGH priority
MIN_COMPOSITE_SCORE_MEDIUM = 50    # scanner score threshold for MEDIUM priority

# ---------------------------------------------------------------------------
# Account-size tiers — added 2026-08-13 ahead of the live rollout.
# risk_manager.py's size_position is whole-shares-only; BUCKET_SIZE_TABLE's
# percentages, tuned for the ~$1M paper account, produce sub-1-share
# (unexecutable) tranches on most of the watchlist well before the account
# is actually "small" in any other sense. Below $25,000, sizing tightens
# instead of silently failing.
#
# $25,000 is our own risk/diversification threshold, not a regulatory one —
# FINRA's Pattern Day Trader rule (which used to carry this same number as
# its minimum-equity line) was eliminated 2026-06-04, replaced by real-time
# intraday margin monitoring (FINRA Notice 26-10; SR-FINRA-2025-017). IBKR
# specifically now applies the standard Reg T $2,000 minimum for margin/
# short-sale trading instead of the old $25k PDT floor. Below that $2,000
# Reg T line (which covers the live account's current $1,010 balance), the
# account effectively can't use margin regardless of this tiering.
#
# Evaluated live off actual portfolio value via get_active_limits(balance)
# — not a one-time constant — so an account re-tiers automatically as it
# grows or shrinks, no restart or manual edit needed.
# ---------------------------------------------------------------------------
ACCOUNT_TIER_STANDARD_MIN = 25_000.0   # >= this: today's rules, unchanged
ACCOUNT_TIER_REDUCED_MIN  = 5_000.0    # >= this (and < standard): "reduced"
                                        # below this: "micro"

TIER_MAX_CONCURRENT_POSITIONS = {"standard": None, "reduced": 6, "micro": 5}
TIER_MIN_CONVICTION_TO_QUEUE  = {"standard": MIN_CONVICTION_TO_QUEUE, "reduced": 6, "micro": 7}
# Moonshot's normal 2% cap is structurally too small to ever clear the
# whole-share floor at Reduced/Micro balances — raised per-tier instead of
# excluding the bucket outright. Confirmed intentional 2026-08-13 despite
# being a 5-10x jump in single-bet risk: a $202 moonshot at $1,010 (Micro)
# vs. today's $20.
TIER_MAX_POSITION_PCT_MOONSHOT = {"standard": MAX_POSITION_PCT_MOONSHOT, "reduced": 0.10, "micro": 0.20}
# No single share purchase may itself already represent more than this
# fraction of portfolio value — guards against one expensive share (e.g. an
# $800 stock on a $1,010 account) structurally dominating the position.
# None at Standard (never binding there in practice; not worth the extra
# code path at that scale).
TIER_SINGLE_SHARE_CAP_PCT = {"standard": None, "reduced": 0.50, "micro": 0.50}


def get_account_tier(balance: float) -> str:
    if balance >= ACCOUNT_TIER_STANDARD_MIN:
        return "standard"
    if balance >= ACCOUNT_TIER_REDUCED_MIN:
        return "reduced"
    return "micro"


def get_active_limits(balance: float) -> dict:
    """Tier-appropriate sizing limits for the given live portfolio balance.
    Call at the top of every sizing/queueing decision (RiskManager.size_position,
    the analyst-conviction gate in agents/crew.py) rather than reading the
    tier dicts directly, so all callers re-tier consistently."""
    tier = get_account_tier(balance)
    return {
        "tier":                      tier,
        "max_concurrent_positions":  TIER_MAX_CONCURRENT_POSITIONS[tier],
        "min_conviction_to_queue":   TIER_MIN_CONVICTION_TO_QUEUE[tier],
        "max_position_pct_moonshot": TIER_MAX_POSITION_PCT_MOONSHOT[tier],
        "single_share_cap_pct":      TIER_SINGLE_SHARE_CAP_PCT[tier],
    }

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
# Defaults derived from TRADING_ENV above (7497/clientId 1 for paper,
# 7496/clientId 40 for live) — IB_PORT/IB_CLIENT_ID in .env can still
# override explicitly, but the common case (just setting TRADING_ENV) now
# gets the right port and clientId range for free instead of relying on
# .env being edited consistently across multiple separate values.
IB_HOST      = os.getenv("IB_HOST",      "127.0.0.1")
IB_PORT      = int(os.getenv("IB_PORT",      _IB_PORT_DEFAULT))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", _IB_CLIENT_ID_DEFAULT))
