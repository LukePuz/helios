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


# pyarrow's default memory pool (mimalloc in this build) segfaulted when
# pandas allocated from a daemon thread with the agent stack loaded
# (vix-live / earnings-timing, libarrow AllocateResizableBuffer, 2026-11-08);
# the system allocator didn't. Set by environment, not by importing pyarrow
# here: an import would load numpy and start its BLAS threads in every
# process before Celery forks its pool (audit 2026-11-08 15:08 #2).
# process_manager sets it for the workers too; a value already set wins.
import os as _os
_os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
ARROW_MEMORY_POOL = _os.environ["ARROW_DEFAULT_MEMORY_POOL"]

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

# The Streamlit dashboard's bind address. Streamlit's default is every
# interface, which put the Manual Trade tab (live MARKET orders, no login) on
# the home network (audit 2026-11-07 03:08 #4). Loopback by default; set
# HELIOS_DASHBOARD_ADDRESS to this box's Tailscale IP to reach it from the
# tailnet.
DASHBOARD_ADDRESS = os.getenv("HELIOS_DASHBOARD_ADDRESS", "").strip() or "127.0.0.1"

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
# The per-buy drawdown check's baseline (RiskManager.is_halted) when it is
# unusable: "unparseable" (the stored mark is not a number; is_halted raises
# on it, so every caller fails closed), "non_finite" (inf, -inf or nan; the
# check runs on it unchanged: inf blocks as an implausible account value,
# -inf and nan do not block) or "non_positive" (<= 0, stored or the start
# value it falls back to; the check does not block). "shadow": one hwm_guard
# activity row and at most one alert per reason per day (no dollar values,
# sent off the buy path), and in each case the check does exactly what it did
# before. "enforce": also blocks this buy (halt_unknown, not latched) in all
# three. Read case-insensitively; an unknown value logs and runs as shadow.
# Not read from .env.
HWM_GUARD_MODE = "shadow"

# Per-bucket position caps (initial entry ceiling)
MAX_POSITION_PCT_COMPOUNDER  = 0.12   # 12% at entry; allow up to 15% via gains
MAX_POSITION_PCT_ACCELERATOR = 0.07   # 7% at entry; allow up to 8% via gains
MAX_POSITION_PCT_MOONSHOT    = 0.02   # 2% hard cap always

# Theme / sector concentration limits
MAX_SINGLE_THEME_PCT   = 0.35   # 35% per sector/thesis
MAX_AI_INFRA_PCT       = 0.60   # 60% total AI infrastructure exposure
MAX_MOONSHOT_TOTAL_PCT = 0.10   # 10% total moonshots
# A theme or moonshot-total sub-check of risk_manager.check_concentration
# that raises leaves that cap unset, so the buy is sized at full. "shadow"
# (default): unchanged sizing, plus one risk_subcheck_fail activity row
# (would_fail_closed true) and an alert at most hourly per sub-check and
# process. "enforce": that cap becomes 0 (no room), as the outer position
# read error already does. Any other value reads as "shadow". Not read
# from .env.
RISK_SUBCHECK_FAIL_MODE = "shadow"

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
# Tightened from 0.35 to 0.20 (2026-09-11, user request) -- 35% was
# deliberately wide to avoid whipsaw stop-outs on inherently volatile
# names, but on a small account the dollar risk of letting a Moonshot
# position run to a 35% loss before cutting it was judged too large
# relative to the added whipsaw protection. Applies system-wide to every
# Moonshot position (LUNR/QBTS/RGTI at the time of this change), not a
# per-ticker override -- see calculate_stop()'s Moonshot branch.
MOONSHOT_MAX_LOSS_PCT = 0.20

# Fixed dollar-risk position sizing (2026-09-11, user request, per Mark
# Minervini's SEPA methodology): risk_manager.size_position() sizes shares
# purely off each bucket's %-of-account allocation table (BUCKET_SIZE_TABLE)
# -- the dollar amount at risk if the stop hits is whatever falls out of
# that allocation times the stop distance, so two trades in the same
# conviction band get the SAME share count regardless of how close or far
# their stops are. This caps shares so a stop-out never costs more than
# RISK_PCT_PER_TRADE of account equity, taking whichever of {allocation-cap
# shares, risk-cap shares} is smaller -- it only ever tightens sizing on a
# wide-stop trade, never loosens it beyond the existing % caps. 2% is the
# midpoint of Minervini's publicly stated 1.25-2.5% per-trade risk range.
RISK_PCT_PER_TRADE = 0.02

# Record-only volatility shadows on the legacy sizing path (gap hunt
# 2026-11-21: buys with 20-day realised vol above 60% lost more than 10% vs
# SPY within 20 sessions 23% of the time, under 40% 3%; stops sat at a median
# 0.71 of the 20-session 1-sigma move). When on, size_position adds two keys
# to the sizing dict (and so to pending_approvals.risk_json):
#   vol_target_shadow -- the position weight at which weight x rvol20 equals
#     VOL_TARGET_CONTRIBUTION, its ratio to the weight after this order, and
#     the shares this order could buy within it;
#   stop_vol -- the stop distance in ATR multiples and in 20-session sigmas.
# The weights are restated in crew (agents/crew.py, just before the
# approval is written) at the final share count, after the concentration,
# correlation and micro-name cuts: weights_at "final_order" (else
# "size_position"), with shares_at_sizing and order_weight_at_sizing kept.
# A deferred buy re-sized at execution (tasks.retry_deferred_entry) can buy
# fewer shares; like risk_json's own shares_initial, the record keeps the
# approved order. Simulations and dry runs (RiskManager(simulation=True) or
# log_fallback=False: the c6 shadow, the swap check, sizing_dryrun) skip
# it. rvol20 = sample stdev of the last VOL_TARGET_WINDOW_SESSIONS daily log
# returns on completed price_history bars, times sqrt(252); stale_sessions
# counts weekdays since the last cached bar. Nothing reads either key:
# shares, stops and orders are the same with this on or off, and there is
# no live switch. 0.02 = a 5% position in a 40%-vol name.
VOL_TARGET_SIZING_SHADOW_ENABLED = True
VOL_TARGET_CONTRIBUTION = 0.02
VOL_TARGET_WINDOW_SESSIONS = 20

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
# Regime-adaptive profit-taking / staleness (2026-08-29, user request)
# ---------------------------------------------------------------------------
# Everything above this point governs the ORIGINAL, protective-only exit
# path (stop-loss, trailing stop after a large gain) -- it never proactively
# banks a gain or questions whether a flat position still deserves capital.
# This section adds that missing piece, scoped to Accelerator/Moonshot only
# -- Compounder is deliberately excluded, it's the long-term core bucket by
# explicit design (see find_swap_candidate's docstring) and stays
# buy-and-hold on purpose.
#
# Both tables are keyed by agents.market_regime._score_trade_style()'s
# output ("Trending/Expanding" / "Choppy/Distribution"), not the risk-on/
# off `regime` score -- a trending market rewards patience (wide targets,
# long staleness windows); a choppy/distributive one rewards banking gains
# faster and cutting stale capital loose sooner. Concrete motivating
# example: Q2 2026 was a clean uptrend where breakout entries held with a
# wide trailing stop worked; by June/July the market flipped choppy and
# the SAME posture rode gains up and gave them back. Nothing before this
# distinguished the two regimes' correct behavior -- risk_manager.py reads
# agents.market_regime.get_cached_regime()["trade_style"] to pick which
# column applies at decision time.
#
# WIRED INTO THE SCHEDULED TASK LOOP as of 2026-08-29 -- tasks.py's
# check-profit-targets entry (30 min, same cadence as check-stops) calls
# risk_manager.check_profit_targets() automatically, which places real
# SELL orders against live capital. flag_stale_positions() is called from
# tasks.review_positions (the existing daily 10:15am position-review
# task), feeding stale tickers into agents.crew.review_all_held_positions's
# priority queue rather than exiting anything itself.
PROFIT_TARGET_PCT_BY_STYLE: dict[str, dict[str, tuple[float, float]]] = {
    "Trending/Expanding": {
        "Accelerator": (0.20, 0.30),
        "Moonshot":    (0.40, 0.60),
    },
    "Choppy/Distribution": {
        "Accelerator": (0.10, 0.15),
        "Moonshot":    (0.20, 0.30),
    },
}
# Fraction of the position sold when the profit target is first crossed;
# the rest gets its stop moved to breakeven and is otherwise left to run
# until a real exit signal (existing trailing-stop/LLM-review path) --
# same "bank real profit, protect the rest, don't cap the upside" shape
# validated against momentum-scanner's live data this session. Choppy
# regimes bank a bigger share on the theory that a move is less likely to
# have much more room left once it's already up this much in a
# non-trending market.
PARTIAL_SELL_PCT_BY_STYLE: dict[str, float] = {
    "Trending/Expanding": 0.25,
    "Choppy/Distribution": 0.45,
}
# CALENDAR days held (not trading days -- computed straight from
# position_stops.opened_at, no trading-calendar dependency) with less
# than STALE_MOVE_THRESHOLD_PCT move (either direction) before a position
# is bumped to the FRONT of review_all_held_positions's rotation -- not
# auto-exited, just guaranteed a fresh LLM look instead of possibly
# waiting weeks behind other positions in the oldest-reviewed-first queue
# (see that function's docstring for why the queue can be that slow with
# >10 open positions). Values are the ~20/15/10/7 TRADING-day intent
# scaled by ~1.4x (5/7 trading-to-calendar ratio) to land on the same
# real-world window.
STALE_WINDOW_DAYS_BY_STYLE: dict[str, dict[str, int]] = {
    "Trending/Expanding": {"Accelerator": 28, "Moonshot": 21},
    "Choppy/Distribution": {"Accelerator": 14, "Moonshot": 10},
}
STALE_MOVE_THRESHOLD_PCT = 0.05

# ---------------------------------------------------------------------------
# Minimum expected-return RATE (2026-08-30, user request)
# ---------------------------------------------------------------------------
# A trade's expected_return_pct alone doesn't say whether it's actually a
# good use of capital -- a 20% expected move is a great trade if it's
# expected within a month, and a mediocre one if it's not expected for two
# years. This normalizes by TIME: expected_return_pct / weeks_to_catalyst
# must clear this floor. Examples the user gave, all of which clear 1%/week:
#   - 5% expected run-up into an earnings call 3 weeks out: 5/3 = 1.67%/wk
#   - 3% expected move the next session, reading an earnings report right
#     now: with weeks_to_catalyst floored to 1 day (see
#     agents.analyst.compute_weeks_to_catalyst), a huge implied rate --
#     obviously clears the bar, this gate is about NOT missing that a
#     same-day catalyst is a great trade, not about blocking it.
#   - 20% expected move over the next quarter (~13 weeks): 20/13 = 1.54%/wk
#
# Scoped to Accelerator/Moonshot only, matching every other piece of the
# 2026-08-29/30 swing-trading-focus work -- Compounder is the long-term
# core bucket by explicit prior design (see find_swap_candidate's
# docstring) and has no defined catalyst timeline to rate-check against;
# forcing one onto it would fight that bucket's actual purpose, not
# refine it.
MIN_EXPECTED_RETURN_PCT_PER_WEEK = 1.0

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
# crew._get_price's last fallback (yfinance) reaches the analyst brief's
# current price. Off: the old order (after-hours, pre-market, regular) at all
# hours. On: regular first during RTH; pre-market first before noon ET and
# after-hours first after it outside RTH. An analyst-input change, so off by
# default until Luke decides (packet G50).
PRICE_FALLBACK_SESSION_ORDER = os.getenv("PRICE_FALLBACK_SESSION_ORDER", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Sell behaviour
# ---------------------------------------------------------------------------
TRIM_FRACTION = 0.33   # analyst TRIM sells this fraction of the held position

# Trim-or-exit rule (agents/trim_exit.py). A fixed rule evaluated after each
# held-position review, on the verdict the analyst already wrote; it changes
# nothing the analyst reads. Before an earnings print one TRIM or EXIT verdict
# trims and a confirmed EXIT exits; elsewhere a sale needs an earlier sell
# verdict that confirms it. A rule TRIM leaves at least
# TRIM_EXIT_MIN_REMAINING_SHARES under a resized stop. Every value below is a
# placeholder: each one is the owner's decision.
#
# Master switch. Off: the rule is not evaluated; nothing is read, nothing is
# written and routing is as before. On by itself is shadow: one TRIM_EXIT_RULE
# signals row per held review (and per TRIM/EXIT verdict on the event and scan
# paths), no order, routing as before. The rows are written at confidence 0
# with no model: a verdict count outside this repo that keeps its own
# non-verdict list instead of db._NON_ANALYST_SIGNALS (the conviction-share
# script) needs TRIM_EXIT_RULE added before this is turned on.
TRIM_EXIT_ENABLED = os.getenv("TRIM_EXIT_ENABLED", "false").lower() == "true"
# Sell switch. Does nothing without the master switch. Off: shadow. On: in
# the contexts named in TRIM_EXIT_LIVE_CONTEXTS the rule decides whether a
# held review's TRIM/EXIT verdict at TRIM_EXIT_MIN_CONVICTION or above is
# routed, in place of the model's own pass flag (it can add a sale and it can
# withhold one). The sale uses the existing approval and execution path.
# Turning it on changes live trading.
TRIM_EXIT_LIVE = os.getenv("TRIM_EXIT_LIVE", "false").lower() == "true"
# Contexts where the sell switch applies, comma-separated: earnings_pre (the
# print is ahead, within TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE days), earnings_day
# (the print is dated today and is not known to be after the close), general
# (everything else). A context not named is recorded and routes as before.
# Empty: none. Kept as a sorted tuple, not a set: the config hash
# (config_env.env_record) uses repr(), and a set's order differs by process.
TRIM_EXIT_LIVE_CONTEXTS = tuple(sorted({
    s.strip().lower() for s in os.getenv("TRIM_EXIT_LIVE_CONTEXTS", "earnings_pre").split(",")
    if s.strip()}))
# Tickers the rule never routes, comma-separated. Their rows still record what
# the rule would have done. This does not stop the other sell paths (the
# model's own pass flag, the low-conviction rule) or the protective stop.
TRIM_EXIT_EXEMPT_TICKERS = tuple(sorted({
    s.strip().upper() for s in os.getenv("TRIM_EXIT_EXEMPT_TICKERS", "").split(",")
    if s.strip()}))
# The rule reads TRIM/EXIT verdicts at this conviction or above. Below it the
# analyst's low-conviction sell rule applies, unchanged.
TRIM_EXIT_MIN_CONVICTION = int(os.getenv("TRIM_EXIT_MIN_CONVICTION", "5"))
# earnings_pre reaches this many calendar days before the print (0 = only a
# print after today's close).
TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE = int(os.getenv("TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE", "4"))
# Confirmation: the newest analyst verdict on the ticker that is at least
# this many hours older than the evaluation must be a TRIM or EXIT, no older
# than TRIM_EXIT_CONFIRM_MAX_AGE_DAYS and not from before the position's stop
# record was opened. The gap keeps the verdict under evaluation, and others
# from the same session, from confirming it. Values under 1 read as 1, so
# the verdict can never confirm itself.
TRIM_EXIT_CONFIRM_MIN_GAP_HOURS = float(os.getenv("TRIM_EXIT_CONFIRM_MIN_GAP_HOURS", "12"))
TRIM_EXIT_CONFIRM_MAX_AGE_DAYS = int(os.getenv("TRIM_EXIT_CONFIRM_MAX_AGE_DAYS", "7"))
# No rule TRIM within this many days of any recorded SELL on the ticker, so a
# daily earnings review trims once per window (0 = no cooldown). It also holds
# back a one-share TRIM turned into an EXIT (TRIM_EXIT_ONE_SHARE_TRIM=exit),
# so two TRIM verdicts can't sell a 2-share position whole inside the window.
# An EXIT verdict is not held back by it.
TRIM_EXIT_COOLDOWN_DAYS = int(os.getenv("TRIM_EXIT_COOLDOWN_DAYS", "5"))
# No rule sale within this many hours of a decided OPEN/ADD on the ticker
# (0 = no minimum hold).
TRIM_EXIT_MIN_HOLD_HOURS = int(os.getenv("TRIM_EXIT_MIN_HOLD_HOURS", "48"))
# A rule TRIM that cannot leave TRIM_EXIT_MIN_REMAINING_SHARES (one share
# held): "hold" sells nothing; "exit" treats it as an EXIT request, which
# sells only when confirmed and outside TRIM_EXIT_COOLDOWN_DAYS. Any other
# value reads as hold. Execution rechecks it: a one-share EXIT already routed
# is skipped if this is no longer "exit".
TRIM_EXIT_ONE_SHARE_TRIM = os.getenv("TRIM_EXIT_ONE_SHARE_TRIM", "hold").strip().lower()
# Shares a rule TRIM must leave (0 = a one-share TRIM closes the position, as
# the model-flag TRIM does).
TRIM_EXIT_MIN_REMAINING_SHARES = int(os.getenv("TRIM_EXIT_MIN_REMAINING_SHARES", "1"))
# Fraction of the position a rule TRIM sells, whole shares, at least one.
# Defaults to TRIM_FRACTION; a TRIM routed on the model's own flag keeps
# TRIM_FRACTION.
TRIM_EXIT_FRACTION = float(os.getenv("TRIM_EXIT_FRACTION", str(TRIM_FRACTION)))

# No new OPEN/ADD within this many calendar days before earnings (0 = off).
# Off since 2026-10-28 by Luke's decision: "don't forbid trading near earnings,
# the bot should be able to buy and sell based on what it thinks the earnings
# will look like." Earnings dates still reach the analyst (catalyst calendar,
# earnings review); normal caps and stops apply. Also governs the scanner's
# earnings skip/catalyst cap and the setup gates' no-blackout criterion.
EARNINGS_ENTRY_BLACKOUT_DAYS = int(os.getenv("EARNINGS_ENTRY_BLACKOUT_DAYS", "0"))

# When the first tranche buys less than 1 share, buy exactly 1 if that share
# fits the hard per-position cap and the cash-reserve cap (risk_manager.size_position).
ALLOW_ONE_SHARE_ABOVE_ALLOCATION = os.getenv("ALLOW_ONE_SHARE_ABOVE_ALLOCATION", "true").lower() == "true"
# The one share may use up to this fraction of current cash (was the 20%
# per-trade cash cap, which blocked every $400-600 share once most cash was
# invested; Luke 10-29). Normal multi-share sizing keeps the 20% cap.
ONE_SHARE_MAX_PCT_OF_CASH = float(os.getenv("ONE_SHARE_MAX_PCT_OF_CASH", "0.50"))
# Micro tier only: a new OPEN/ADD is cut (or blocked) so the name's held value
# plus the order stays at or under this % of equity. Never sells or trims an
# existing position (risk_manager.RiskManager.micro_name_cap).
MICRO_TIER_NAME_CAP_ENABLED = os.getenv("MICRO_TIER_NAME_CAP_ENABLED", "true").lower() == "true"
MICRO_TIER_NAME_CAP_PCT = float(os.getenv("MICRO_TIER_NAME_CAP_PCT", "30.0"))
# A new OPEN/ADD routes only when the screener marked its news item a new
# fact (agents.crew.new_fact_gate). Sells, HOLD and WATCH are never gated.
# Anything but "true"/"false" keeps the gate on and is alerted
# (crew.new_fact_gate), so a typo can't switch it off.
BUY_REQUIRES_NEW_FACT_RAW = os.getenv("BUY_REQUIRES_NEW_FACT", "true").strip().lower()
BUY_REQUIRES_NEW_FACT = BUY_REQUIRES_NEW_FACT_RAW != "false"
# Which buys: all_buys (every OPEN/ADD; a verdict with no news item can't
# buy), news_only (only buys from a news/event verdict), or
# all_but_held_review_add (all but an ADD from a held-position review).
BUY_NEW_FACT_SCOPE = os.getenv("BUY_NEW_FACT_SCOPE", "news_only").strip().lower()
# Where the new fact may come from: same_run (the verdict's own item), or
# 24h / 3d (also any analysed news item on the ticker within that window).
BUY_NEW_FACT_WINDOW = os.getenv("BUY_NEW_FACT_WINDOW", "24h").strip().lower()
# Quiver Quant (congressional trades, gov contracts): key rejected since
# 10-13; Luke dropped it 10-29 (D2). true turns the poller and the
# scanner lookup back on once a key exists.
QUIVER_ENABLED = os.getenv("QUIVER_ENABLED", "false").lower() == "true"
# Ingestion yield watch (tasks.feed_health, weekdays 18:40 ET): new items per
# source per NYSE session (seen_items) against the median of the source's
# last FEED_HEALTH_BASELINE_SESSIONS non-zero sessions before the run being
# judged, so an outage never becomes its own baseline (the 11-13..11-20
# analyst-ratings freeze: polls "succeeded" with 0 items, and source_yield's
# trailing mean fell under its floor within 2 days). A source alerts at 0 for
# FEED_HEALTH_ZERO_SESSIONS sessions in a row, or under FEED_HEALTH_LOW_RATIO
# of its baseline for FEED_HEALTH_LOW_SESSIONS, once per source per day, with
# one feed_health row per run. Nothing ingested changes. A source needs
# FEED_HEALTH_MIN_BASELINE_SESSIONS non-zero sessions in the look-back to be
# judged. news_title is news's per-item title key (it mirrors news), and
# congressional is skipped while QUIVER_ENABLED is off. Not read from .env.
FEED_HEALTH_ENABLED = True
FEED_HEALTH_BASELINE_SESSIONS = 10
FEED_HEALTH_MIN_BASELINE_SESSIONS = 5
FEED_HEALTH_ZERO_SESSIONS = 2
FEED_HEALTH_LOW_RATIO = 0.2
FEED_HEALTH_LOW_SESSIONS = 3
FEED_HEALTH_LOOKBACK_DAYS = 60
FEED_HEALTH_EXCLUDED_SOURCES = ("news_title",)
# Fundamentals freeze watch (tasks.refresh_fundamentals, record only): after
# each refresh, how many tickers' target_mean_price or forward_eps changed
# from their previous fundamentals_history row (a fundamentals_changes
# row). forward_pe moves with the price every day, so it can't show a freeze
# (all 75 targets and EPS unchanged 10-26..10-29, unflagged). At most
# FUNDAMENTALS_FREEZE_MAX_CHANGED changed on FUNDAMENTALS_FREEZE_REFRESHES
# refreshes in a row: a WARN and one alert a day. Nothing fetched or stored
# changes. Not read from .env.
FUNDAMENTALS_FREEZE_REFRESHES = 2
FUNDAMENTALS_FREEZE_MAX_CHANGED = 0
# Brief sections that change what the analyst sees (professor M37-M39, Q35).
# On from 10-30 (Luke D12, 10-29: needed for plan-following); false turns
# either back off.
BRIEF_PRICE_STRUCTURE = os.getenv("BRIEF_PRICE_STRUCTURE", "true").lower() == "true"
BRIEF_POSITION_CONTEXT = os.getenv("BRIEF_POSITION_CONTEXT", "true").lower() == "true"
# Market-regime fallback reads only overall-market rows (audit 20:53 A2053-1,
# 08:53 A0853-1). On from 10-30 (Luke D3, 10-29); false restores sector-row reads.
REGIME_OVERALL_ONLY = os.getenv("REGIME_OVERALL_ONLY", "true").lower() == "true"
# Regime and scanner score completed daily bars only (A1053-2; Luke D3).
REGIME_COMPLETED_BARS = os.getenv("REGIME_COMPLETED_BARS", "true").lower() == "true"
# Log-only: the regime's bar request spans 250 calendar days, about 173
# completed bars, so its "200DMA" averages fewer than 200. With this on, each
# regime assessment makes one more, wider request (330 calendar days,
# 5-second limit), re-scores trend, the ETF-proxy breadth fallback and sector
# price action on those bars, and records the score and label they imply in
# signals.trend.shadow_200. The live request, score, label, sizing and
# prompts are the same either way; false skips the second request.
REGIME_SHADOW_200_ENABLED = os.getenv("REGIME_SHADOW_200_ENABLED", "true").lower() == "true"
# Stop-out re-entry shadow (record only): an OPEN/ADD that has passed the
# rebuy cooldown, on a ticker whose last exit the bot did not place (a resting
# stop fill or a manual sale) is under 10 days old, gets one
# STOPOUT_REENTRY_SHADOW signals row saying which re-entry cooldowns
# (1, 3, 5 and 10 days) would have blocked it. The row changes nothing, but
# the live entry gate below reads it: false skips the read and so also turns
# off the stop-out re-entry gate. Not read from .env.
STOPOUT_REENTRY_SHADOW_ENABLED = True
# Entry-quality shadow (record only): an OPEN/ADD that has passed the rebuy
# cooldown gets one ENTRY_QUALITY_SHADOW signals row with the triggering
# item's screener new_fact, the last completed close against the 50- and
# 200-day averages and its distance below the 20-day high in ATR14 units (from
# the cached price_history bars, no fetch), and which fixed illustrative trend
# filters would have blocked it. The row changes nothing, but the live entry
# gate below reads its trend check: false skips the read and so also turns off
# the trend gate. Not read from .env.
ENTRY_QUALITY_SHADOW_ENABLED = True
# Story de-duplication and staleness shadow (record only;
# agents/story_dedup.py). Each event analysis records on its verdict's
# provenance.trigger.story_dedup a story key (normalised title plus source
# domain), the earlier event verdict on the same ticker and story (same URL
# or same key) within STORY_DEDUP_WINDOW_HOURS, the item's age at decision
# time, and would_skip_duplicate / would_skip_stale labels at these cut-offs.
# 24 h for both matches BUY_NEW_FACT_WINDOW, the age past which an item
# already stops counting as a new fact. Nothing is skipped and the prompt,
# brief and routing are unchanged. Not read from .env.
STORY_DEDUP_SHADOW_ENABLED = True
STORY_DEDUP_WINDOW_HOURS = 24
STORY_STALE_CUTOFF_HOURS = 24
# Read-through record (record only; agents/crew.py:_read_through_record). When
# an event is analysed, one activity row (stage read_through) lists the
# related symbols the brief names for the event's ticker
# (context_engine._find_related_symbols over SYMBOLS) and which of them the
# book holds, from the positions already read for the brief, once per source
# item and held ticker. No held review is queued; the brief, prompt and
# routing are unchanged. Not read from .env.
READ_THROUGH_RECORD_ENABLED = True
# Target anchoring and fundamentals staleness (record only;
# agents/target_anchor.py). Every analyst verdict's decision.target_anchor:
# the bull target against the street target and blended fair value the brief
# showed (parsed from the prompt sent), and whether the fundamentals' quarter
# values were first seen before the last earnings print. One bounded
# read-only DB read; nothing gates or routes on it. Not read from .env.
TARGET_ANCHOR_RECORD_ENABLED = True
# Held-exit candidates (record only; agents/held_exit.py). Every trim/exit
# rule row (TRIM_EXIT_RULE, one per held review) records what four fixed
# code-side exit rules would say on the last completed close: below both the
# 50- and 200-day averages, below the active plan's structural stop (and,
# beside it, its invalidation price), a time stop (held at least
# HELD_EXIT_TIME_STOP_SESSIONS sessions and not above the entry price), and a
# drawdown of at least HELD_EXIT_DRAWDOWN_ATR x ATR14 from the highest close
# since entry, with the raw numbers. One price_history read and one
# trade_plans read; nothing is sold and trim_exit.decide never reads it.
# Not read from .env.
HELD_EXIT_SHADOW_ENABLED = True
HELD_EXIT_TIME_STOP_SESSIONS = 20
HELD_EXIT_DRAWDOWN_ATR = 3.0
# Live entry gates (agents/crew.py:_entry_gate, from deploy-2026-11-23). An
# OPEN/ADD that has passed the rebuy cooldown is skipped, with one
# ENTRY_GATE_BLOCKED signals row, when (a) the ticker's newest exit the bot
# did not place (a resting stop fill or a sale at the broker;
# db.get_last_broker_exit) is under STOPOUT_REENTRY_GATE_DAYS days old, or
# (b) ENTRY_TREND_GATE_ENABLED and the last completed close is below both its
# 50- and 200-day simple averages on cached bars at most
# ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS calendar days old. The gates use what the
# two shadows above read, so a shadow switched off disables its gate. A read
# that fails, too few bars or stale bars do not block. Higher conviction does
# not override either gate. Sells, stops and sizing are untouched. 0 / False
# turns a gate off. Not read from .env.
STOPOUT_REENTRY_GATE_DAYS = 10
ENTRY_TREND_GATE_ENABLED = True
ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS = 5
# The trend gate applies only while the market itself is in a downtrend: the
# last completed close of ENTRY_TREND_GATE_MARKET_TICKER below its 200-day
# simple average (cached bars, same freshness rule). In June-October 2026,
# with the market above its 200-day, buys below both averages did better than
# buys above them (team gap hunt 2026-11-21, pilot diagnostics), so the gate
# stands down in an up-market. A failed or stale market read does not block.
# Empty string turns the condition off (gate on the stock alone).
ENTRY_TREND_GATE_MARKET_TICKER = "SPY"
# Reclaim shadow for gate-blocked buys (agents/crew.py:_entry_gate_reclaim_record
# and _check_entry_gate_reclaims). Record only: when the entry gate blocks a
# buy, one record per ticker goes to system_state
# entry_gate_reclaim_shadow_<env>; the nightly price refresh then logs, on
# completed bars, the first session every gate that blocked it would pass
# again (stage entry_gate_reclaim_shadow) and the name's move 5/10/20 sessions
# after the block. A record expires ENTRY_GATE_RECLAIM_SHADOW_SESSIONS
# sessions after its block. Nothing here changes a gate outcome, a prompt,
# routing or sizing.
ENTRY_GATE_RECLAIM_SHADOW_ENABLED = True
ENTRY_GATE_RECLAIM_SHADOW_SESSIONS = 20
# Add-to-loser gate (agents/crew.py:_add_loser_check, read by _entry_gate;
# also on a deferred ADD's retry, tasks._deferred_entry_gate), for the written
# rule "never add to a losing position without a new catalyst"
# (knowledge/risk_rules.md). An ADD whose price is more than
# ADD_LOSER_GATE_MIN_PCT percent below the broker's average cost for the
# position would block unless the triggering news item itself has the
# screener's new_fact True. A new-fact item elsewhere in the new-fact gate's
# window does not count. A verdict with no news item (a held review, the daily
# scan, a drop watch) has no new fact, except a drop-watch review while
# DIP_REVIEW_COUNTS_AS_NEW_FACT is on. An unjudged flag (a keyword-fallback
# screen, a missing flag, a stale-entry recheck) never blocks, and neither
# does a failed position or price read or a name not held. OPEN, sells,
# stops and sizing are untouched. What counts as a catalyst for this rule is
# Luke's, so ADD_LOSER_GATE_MODE:
#   "shadow"  (default) the record (with would_block) goes on the ADD
#             approval's risk_json.add_loser_gate, or an add_loser_gate_shadow
#             activity row on a deferred retry; nothing blocks;
#   "enforce" a would_block ADD is skipped with one ENTRY_GATE_BLOCKED row
#             (decision.entry_gate.add_loser), and a deferred one expired;
#   "off"     no read, no field.
# Any other value reads as "shadow". Not read from .env.
ADD_LOSER_GATE_MODE = "shadow"
ADD_LOSER_GATE_MIN_PCT = 0.0
# Per-lot R at an ADD's stop (record only; execution._record_add_lot_r). When
# a BUY's stop is placed over an existing stop record (the whole-position
# re-stop of an ADD, or the stale-record branch that rewrites the record at
# the new shares' stop), one activity row (stage add_lot_r) records, for each
# open lot, R at the placed stop = (entry - placed stop) / (entry - the lot's
# own initial stop), in ratios only, with the largest and the count above
# ADD_LOT_R_THRESHOLD. The stop itself is unchanged. Not read from .env.
ADD_LOT_R_SHADOW_ENABLED = True
ADD_LOT_R_THRESHOLD = 1.2
# Reward-to-risk against the placed stop (record only): every OPEN/ADD that
# reaches the approval record gets risk_json["rr_vs_stop"]
# (agents/crew.py:_rr_vs_stop), the reward to the model's target over the risk
# to the stop sizing placed, the stop distance in % and in ATR, and
# would_block labels at each RR_VS_STOP_LABEL_THRESHOLDS value (r_to_target
# under it). The entry UDR gate reads the model's own ratio, whose downside is
# the model's downside_scenario, not this stop. The thresholds are fixed
# illustrative labels, not gates. Nothing routes or sizes on the field. Not
# read from .env.
RR_VS_STOP_SHADOW_ENABLED = True
RR_VS_STOP_LABEL_THRESHOLDS = (1.5, 2.0)
# Above-market guard on whole-position stops (risk_manager._stop_below_market:
# the ADD merge, the late-fill / queued re-place in
# tasks._reprotect_after_late_sell, the TRIM and manual-sell re-places, and
# the reconcile resize in tasks._fix_overstop). "shadow": one activity row (stage
# stop_above_market_guard, outcome would_*) and an info alert say what the
# guard would place when the level is at or above the price; the stop and
# the record are unchanged. "enforce": the guard places that instead. Not
# read from .env. When the no-loosen floor's resting-stop read fails
# (resting_unread), shadow makes no broker call and judges from the fallback
# stop alone (row resting_read "reread_skipped_shadow"; "floor_read_failed"
# before), so order-path timing is unchanged. Enforce makes one bounded
# re-read (risk_manager._reread_resting_stops, at most
# _GUARD_REREAD_TIMEOUT_S + 1 s, never reconnecting): "reread_ok" or
# "reread_failed". It never re-reads between a stop cancel and its re-place
# ("reread_skipped_window", fallback alone). The TRIM and manual-sell
# re-places and the reconcile resize pass their rows (or []), so they never
# have an unread set.
# Enforce prerequisites, open decisions:
# (1) the late-sell / queued RESTORE, EXIT and PROFIT_TARGET re-place
#     follows the sale's cancel, so it never re-reads and judges from the
#     fallback alone: accept that gap, or allow the re-read there (up to
#     _GUARD_REREAD_TIMEOUT_S + 1 s more with the shares possibly bare);
# (2) a re-read timeout goes through ibkr._bounded_req, which drops the
#     shared session (the open-orders request key), so the cancel and place
#     after it reconnect: give the re-read a timeout path that keeps the
#     session, or accept the drop;
# (3) reread_ok is unreachable after a timeout or connect failure. Each
#     broker-caused failure of the floor read leaves ibkr._ib None or
#     disconnected (the _bounded_req timeout drops it, a get_ib connect or
#     heartbeat failure sets None, so does a mid-request disconnect), and the
#     re-read never reconnects, so it raises ConnectionError at once:
#     reread_failed is then the behaviour from before the re-read. Allow one
#     bounded reconnect (its connect time added to the window), or accept
#     the gap.
STOP_GUARD_MODE = "shadow"

# Recheck contradiction (record only; agents/crew.py:_recheck_contradiction_record).
# Every OPEN/ADD past the rebuy cooldown in _route_to_approval records the most
# recent non-buy verdict (HOLD, WATCH, PASS, TRIM, EXIT, SELL) on the same
# ticker in this env within RECHECK_CONTRADICTION_WINDOW_HOURS: its kind, the
# model's own action, source (stale_entry_recheck, held_review, plan_level,
# event, daily_scan, chat, unknown), age in hours, conviction and whether it
# was a recheck, plus would_block: that verdict was a stale-entry recheck or a
# held review and the current verdict's triggering item is not a new fact. A
# plan-level review (plan_level_review) is classed plan_level, not
# held_review, so it does not count toward would_block: a definition choice
# that is Luke's (the raw trigger path is recorded). One read of signals
# (db.get_recent_verdicts). Written as risk_json.recheck_contradiction, or in
# the decision of the routing row that ended the route before sizing. Nothing
# blocks on it; whether it becomes a gate is Luke's. Not read from .env.
RECHECK_CONTRADICTION_SHADOW_ENABLED = True
RECHECK_CONTRADICTION_WINDOW_HOURS = 4

# Same-day consensus (record only; agents/crew.py:_same_day_consensus_record).
# Every OPEN/ADD past the rebuy cooldown in _route_to_approval records the
# ticker's earlier analyst verdicts today (ET midnight up to the route, this
# env, the current verdict excluded; scripts/fire_test.py verdicts are
# counted, as their analyst rows carry no marker): their count, how
# many were buy-grade (the stored final_pass_to_risk on an OPEN/ADD verdict:
# the analyst's own pass, as decided when each was written) and that share,
# how many the model itself called a passing buy, median conviction and UDR,
# and the distinct stories among them (story_dedup story_key, where present);
# and, from the same read, the count, buy-grade count and share over the
# last 24 h (route - 24 h up to the route), as many buys route before 10:00
# ET when the ET day holds few verdicts.
# One read of signals (db.get_same_day_verdicts, the last 24 h) under
# SAME_DAY_CONSENSUS_READ_TIMEOUT_MS, at most SAME_DAY_CONSENSUS_LIMIT rows; a
# slower or failed read is recorded and the route goes on. Written as
# risk_json.same_day_consensus, or in the decision of the routing row that
# ended the route. Nothing reads it to decide; any gate, threshold or N is
# Luke's (04be). Not read from .env.
SAME_DAY_CONSENSUS_RECORD_ENABLED = True
SAME_DAY_CONSENSUS_READ_TIMEOUT_MS = 500
SAME_DAY_CONSENSUS_LIMIT = 200

# News timeline (record only; agents/news_timeline.py). After each analyst
# verdict on NEWS_TIMELINE_PATHS, a side thread (never the verdict path)
# reads the ticker's news_log headlines and screener rows for the 7 days
# before the verdict and writes an activity row (stage "news_timeline"): the
# newest stories (deduped, about 10) with their screener decision, a title
# tone tag (agents.screener.title_tone) and whether each title reached the
# prompt that was sent, plus counts over the whole window. Nothing reads it
# to decide; showing a timeline in the brief is Luke's call. Not read from .env.
NEWS_TIMELINE_RECORD_ENABLED = True
NEWS_TIMELINE_PATHS = ("event", "daily_scan")

# Scan tier funnel (record only; agents/scan_funnel.py). After each daily
# scan's enqueue, one activity row (stage "scan_tier_funnel") with, per
# watchlist tier, how many names went in, how many each pre-analyst filter
# dropped and how many were sent to the analyst. Counts only. Nothing reads
# it to decide. Off: no row and no counting. Not read from .env.
SCAN_TIER_FUNNEL_RECORD_ENABLED = True

# The whole-position stop after an ADD fill is sized from this process's
# ib.positions() count (ibkr.get_broker_stock_qty): account-wide, synced at
# connect and updated by TWS position messages as of this process's last
# event-loop run. It is not the orders worker's count that refuses an
# oversized SELL (a separate connection). Since 7ba0857 it should equal the
# deduplicated fresh reqPositions read that decides the path, so it is a
# safety net against a duplicated fresh read (11-13: the cache read 1, a
# fresh read with a duplicated row 4, the broker 2; the stop for 4 was
# refused and NVDA had no stop for about 2 min), not an independent third
# source. An unreadable, zero, short or non-finite count keeps the old
# sizing and alerts; a count below the reads or not above the fill alerts.
# One activity row (stage add_stop_qty) per whole-position ADD records the
# counts. False sizes from the reads as before. Not read from .env.
ADD_STOP_QTY_FROM_BROKER = True
# The stop for what a profit-target partial sale leaves is sized from this
# process's ib.positions() count (ibkr.get_broker_stock_qty, read after the
# sale's fill wait and one more event-loop run), not the portfolio cache read
# seconds after the sale, which can still show the shares just sold, so the
# stop is refused. An unreadable, zero, short or non-finite count keeps the
# old sizing and alerts; a count below the expected remainder alerts. One
# activity row (stage pt_stop_qty) per re-place. False sizes from the cache
# as before. Not read from .env.
PT_STOP_QTY_FROM_BROKER = True
# A BUY fill reconcile books late (reconcile_unconfirmed_orders) waits for
# the portfolio cache to show at least the current count before its
# whole-position stop is placed (expect_at_least, as the manual ADD path
# does), so a lagging cache doesn't leave the late shares unstopped. The
# count: a fresh deduplicated reqPositions read, then ib.positions() right
# after it (used when usable, else the fresh read's); a failed fresh read
# means no expectation, as before. One activity row (stage late_fill_expect)
# per late BUY. Deferring changes stop timing (a fresh OPEN's visible shares
# wait up to one reconcile cycle), so it ships in shadow first:
# "shadow": the stop is placed as before, with no expectation; the broker
# reads and the row run after it (record-only, so they never delay the
# stop), with would_defer from the cache count snapshotted before the place
# (as _reprotect_after_late_sell compares it) against the broker count. "enforce": the
# stop waits, and the queued retry keeps the count for 2 tries. "off" (or any
# other value): no reads, as before. Not read from .env.
LATE_FILL_EXPECT_MODE = "shadow"
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
# Reg T line (which covers a micro-tier live balance), the
# account effectively can't use margin regardless of this tiering.
#
# Evaluated live off actual portfolio value via get_active_limits(balance)
# — not a one-time constant — so an account re-tiers automatically as it
# grows or shrinks, no restart or manual edit needed.
# ---------------------------------------------------------------------------
ACCOUNT_TIER_STANDARD_MIN = 25_000.0   # >= this: today's rules, unchanged
ACCOUNT_TIER_REDUCED_MIN  = 5_000.0    # >= this (and < standard): "reduced"
                                        # below this: "micro"

# micro/reduced were 5/6 until 10-30 (Luke 10-29: more options; cash caps still bind).
TIER_MAX_CONCURRENT_POSITIONS = {"standard": None, "reduced": 8, "micro": 8}
TIER_MIN_CONVICTION_TO_QUEUE  = {"standard": MIN_CONVICTION_TO_QUEUE, "reduced": 6, "micro": 7}
# Moonshot's normal 2% cap is structurally too small to ever clear the
# whole-share floor at Reduced/Micro balances — raised per-tier instead of
# excluding the bucket outright. Confirmed intentional 2026-08-13 despite
# being a 5-10x jump in single-bet risk: a 20%-of-account moonshot at the Micro tier
# vs. today's $20.
TIER_MAX_POSITION_PCT_MOONSHOT = {"standard": MAX_POSITION_PCT_MOONSHOT, "reduced": 0.10, "micro": 0.20}
# 2026-08-21: Compounder/Accelerator never got the same tier-scaling
# Moonshot did above — their BUCKET_SIZE_TABLE bands and hard caps stayed
# at the values tuned for the ~$1M paper account at every tier. On the
# micro-tier account, that meant every ADD priced above
# roughly $45-70/share (NVDA, MSFT, GOOG, META, AVGO, ANET, NOW, CRWD,
# MRVL, TSLA, LRCX — dozens/day, real conviction-7/8 analyst output)
# computed a dollar allocation too small to clear the whole-share floor
# and got silently dropped with just a log.info line — no DB row, no
# alert. Only the 3 existing Moonshot positions (RGTI/LUNR/QBTS, all
# under $21/share) could ever size, which is why the account went 4 days
# without a single trade. Scaled up using the same ratio-preserving
# approach as Moonshot's own 2026-08-13 fix (size_position() rescales
# BUCKET_SIZE_TABLE bands by cap-ratio, not just the cap itself).
TIER_MAX_POSITION_PCT_COMPOUNDER  = {"standard": MAX_POSITION_PCT_COMPOUNDER,  "reduced": 0.28, "micro": 0.40}
TIER_MAX_POSITION_PCT_ACCELERATOR = {"standard": MAX_POSITION_PCT_ACCELERATOR, "reduced": 0.20, "micro": 0.35}
# No single share purchase may itself already represent more than this
# fraction of portfolio value — guards against one expensive share (e.g. a
# high-priced stock on a micro-tier account) structurally dominating the position.
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
        "max_position_pct_moonshot":    TIER_MAX_POSITION_PCT_MOONSHOT[tier],
        "max_position_pct_compounder":  TIER_MAX_POSITION_PCT_COMPOUNDER[tier],
        "max_position_pct_accelerator": TIER_MAX_POSITION_PCT_ACCELERATOR[tier],
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
# Daily-loss halt, SHADOW ONLY (audit 2026-11-02 19:27 #10, roadmap 01(c)):
# check_portfolio_drawdown logs would_halt_daily_loss when equity is down
# more than this fraction from the day's first snapshot (ET day). It never
# sets the halt flag. 0.03 is a placeholder, not a chosen value: the
# threshold is Luke's to set.
DAILY_LOSS_SHADOW_PCT = 0.03
# halt_reason prefix of the implausible-read halt: /rebase lifts only this one (professor M145)
IMPLAUSIBLE_HALT_REASON_PREFIX = "Auto: account value under half the HWM"

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

# ibkr.cancel_stop_orders_for with 2+ stops: send every cancel first, then
# poll them all in one loop bounded by the same 5 s (ibkr._cancel_stops_parallel),
# instead of cancel-and-poll one stop at a time ((N-1) x 5 s more unprotected
# time). Same confirmation rule, return value and order matching; when a
# cancelOrder raises, the same cancels are sent (the raise still stops the
# sending). An ib.sleep raise or a disconnect mid-loop can differ (one ib, not
# get_ib per stop). One stop runs today's path either way. Not read from .env.
CANCEL_PARALLEL_ENABLED = False

# ---------------------------------------------------------------------------
# Swing-trading cutover, Phase 4/5 (2026-10-11) -- setup-gate sizing/stops
# ---------------------------------------------------------------------------
# Master switch for the swing-trading redesign's sizing/stop/category
# authority. Everything built across the prior phases (agents/setup_gates.py's
# classify_setup, agents/trade_ladder.py's nightly rungs, agents/valuation.py's
# fair-value ensemble) has run in SHADOW MODE only so far -- logged alongside
# the existing Compounder/Accelerator/Moonshot system (agents/scanner.py's
# setup_gates_detail), never consulted for a real sizing/stop decision. This
# flag is the only thing that flips that: when False (the only state this
# has ever shipped in, and the required default/fallback), risk_manager.
# size_position/calculate_stop/check_trailing_stops/check_profit_targets and
# agents/analyst.py's bucket lookup are byte-for-byte identical to
# pre-cutover behavior -- every new branch gated on this flag is additive and
# reversible, same on/off convention as USE_BRACKET_ORDERS above.
#
# Known from 2+ weeks of live shadow data: the three gates are strict BY
# DESIGN and qualified on ~0% of the 75-ticker watchlist for multiple
# consecutive days in choppy market conditions. Setting this True is a
# deliberate decision to accept "no new entries for a while" in some
# regimes, not a bug to fix first -- see the final report from this phase
# for the full pre-flip checklist (this sizing math has zero live track
# record as of this writing).
SETUP_GATES_ENABLED = os.getenv("SETUP_GATES_ENABLED", "false").lower() == "true"
# Watchlist ETFs (DRAM, QTUM) never have company fundamentals, so the
# missing-fundamentals conviction cap (6) makes them unbuyable wherever the
# floor is 7. True exempts them from that cap. On by default since
# 2026-11-04 (it changes what can be bought; audit A0100-2); the env var
# set to anything but "true" turns it off.
ETF_FUNDAMENTALS_CAP_EXEMPT = os.getenv("ETF_FUNDAMENTALS_CAP_EXEMPT", "true").lower() == "true"

# A held name hitting a level of its trade plan (invalidation, target,
# catalyst passed, horizon end; tasks.watch_plan_levels) queues a review of
# that position instead of only logging PLAN_LEVEL_HIT. Off until Luke turns
# it on: it adds analyst runs and a new trigger (audit A0903-1 step 3).
PLAN_LEVEL_REVIEW_ENABLED = os.getenv("PLAN_LEVEL_REVIEW_ENABLED", "false").lower() == "true"

# Record only: on the same watch_plan_levels tick, one plan_compiled_hit
# activity row the first time the price or calendar crosses one of the plan's
# compiled levels (structural_stop, t1, t2, add_trigger, time_stop), which
# PLAN_LEVEL_HIT does not watch. Queues nothing, changes no PLAN_LEVEL_HIT
# field, and nothing reads the rows.
PLAN_COMPILED_HIT_SHADOW_ENABLED = True
PLAN_COMPILED_HIT_FIRST_TTL_S = 120 * 86400   # how long a level's first-crossed time is kept

# Analyst add-ons (Luke, 2026-11-08 evening). All analyst input, so all off
# until Luke turns them on (live or as an A/B arm); with all three off the
# rendered prompt and brief are byte-identical to before.
# Scenarios: the model adds up to three "if X happens -> action, at what
# level" lines to every verdict (agents.analyst._addon_instructions).
ANALYST_SCENARIOS_ENABLED = os.getenv("ANALYST_SCENARIOS_ENABLED", "false").lower() == "true"
# Earnings plan: on a held name's pre-print earnings review the model also
# fills event_plan (hold through / trim before / exit before, expected move,
# what result breaks the thesis, what it does after the print).
ANALYST_EVENT_PLAN_ENABLED = os.getenv("ANALYST_EVENT_PLAN_ENABLED", "false").lower() == "true"
# Price cache rescale (ingestion/price_history.py:_store_bars). The nightly
# yfinance fetch is split-adjusted but rewrites only the trailing year, so
# after a split the older cached rows stay on the old scale. Before each
# upsert the fetched and stored closes are compared on overlapping dates; a
# median ratio beyond 10% that sits within 2% of a split factor, with at
# least 90% of the dates within 2% of it, is a split basis change. True:
# rescale the stored rows the fetch doesn't overwrite, in the same
# transaction (one price_cache_rescale activity row, applied true). False
# (default, record only): a price_cache_rescale row with applied false and
# the plain upsert. A gap that is no split factor, or not uniform, is never
# rescaled (price_cache_scale_gap row). scripts/rescale_stale_head.py repairs
# by hand. Not read from .env.
PRICE_CACHE_RESCALE_ENABLED = False
# PRICE HISTORY brief section: recent daily closes, weekly closes, distance
# from the 20-day and 52-week highs, today's move and volume against average
# (context_engine._price_history_section).
BRIEF_PRICE_HISTORY = os.getenv("BRIEF_PRICE_HISTORY", "false").lower() == "true"
# Drop watch (tasks.dip_watch): a held or watchlist name that falls hard with
# no headline. Off: shadow rows (DIP_WATCH_SHADOW) only. On: one analyst
# review per ticker per day, through the normal event path and every
# existing gate, so the model can judge a buying chance against a breakdown.
# A name qualifies when today's move is at or below -DIP_REVIEW_MIN_DROP_PCT
# or at or below -DIP_REVIEW_MIN_ATR times ATR(14). The thresholds are
# Luke's; these defaults only set the shadow rows' would_review flag.
DIP_REVIEW_ENABLED = os.getenv("DIP_REVIEW_ENABLED", "false").lower() == "true"
DIP_REVIEW_MIN_DROP_PCT = float(os.getenv("DIP_REVIEW_MIN_DROP_PCT", "5.0"))
DIP_REVIEW_MIN_ATR = float(os.getenv("DIP_REVIEW_MIN_ATR", "2.0"))
# A drop review has no news item, so under BUY_REQUIRES_NEW_FACT its OPEN/ADD
# passes only if some new-fact news on the ticker landed within the window.
# True lets the drop itself count as the new fact. A buy-gate change: Luke's.
DIP_REVIEW_COUNTS_AS_NEW_FACT = os.getenv("DIP_REVIEW_COUNTS_AS_NEW_FACT", "false").lower() == "true"

# A sub-score the model wrote as text ("4/5") or as a decimal string ("4.0")
# reads as an imputed 3 (or the number) instead of failing the verdict as
# ANALYST_FAILED (audit 2026-11-07 09:07 #9). Off until Luke turns it on: a
# reply that never routed could then route, and the imputed 3 reaches
# trade_outcomes and the prompt's sub-score track record.
SUBSCORE_TEXT_IMPUTE_ENABLED = os.getenv("SUBSCORE_TEXT_IMPUTE_ENABLED", "false").lower() == "true"

# An analyst shadow plan whose position has since closed (stop, EXIT,
# profit-take) is closed before the next verdict, so a re-entry starts a new
# plan. Off: the plan stays active and exited_since_plan records the exit.
# Which one the plan metrics use is packet G75, so this stays off until then.
PLAN_CLOSE_ON_EXIT = os.getenv("PLAN_CLOSE_ON_EXIT", "false").lower() == "true"

# Record only (agents/plans._reseed_record): when a NEW_PLAN or SEEDED plan is
# written, one read of the ticker's previous plan (capped at 500 ms, fail-open)
# says whether that plan was closed by an invalidation breach while the same
# position stayed held (reseeded_after_invalidate). Off: no read and no record.
PLAN_RESEED_RECORD_ENABLED = True
# The reseed flag's window in hours from the breach. None: no window (the
# hours are recorded, so any window can be applied later). Choosing one is
# Luke's.
PLAN_RESEED_WINDOW_HOURS = None

# The prompt's YOUR TRACK RECORD block counts closed trades entered on or after
# this date (YYYY-MM-DD). Empty, the default, keeps the current behaviour: all
# closed trades, so 2027 verdicts would see pilot outcomes (professor C35). It
# is analyst input; whether and when to set it is Luke's.
TRACK_RECORD_SINCE = os.getenv("TRACK_RECORD_SINCE", "").strip()

# A deferred entry past tasks.DEFERRED_ENTRY_MAX_AGE_HOURS is expired either
# way; with this on it is only expired, with no fresh analysis queued. Off
# keeps the 2026-10-27 re-analysis. Off until Luke turns it on: it removes
# analyst runs.
STALE_ENTRY_EXPIRE_ONLY = os.getenv("STALE_ENTRY_EXPIRE_ONLY", "false").lower() == "true"

# The rate gate reads a month-only catalyst date ("YYYY-MM") as the 1st of
# that month, so within the month it floors to 1 day and any expected return
# clears the 1%/week bar. On, it uses the 15th, as _catalyst_check's
# provenance already does. Changes which verdicts pass the gate, so it is
# off until Luke decides (audit A1031-0253-1).
CATALYST_MONTH_MIDPOINT_ENABLED = os.getenv("CATALYST_MONTH_MIDPOINT_ENABLED", "false").lower() == "true"

# Log-only: a conviction-6 OPEN/ADD stopped only by a conviction-7 bar (the
# regime gate, the Moonshot OPEN bar, the account-tier floor) writes one
# activity_feed row, stage c6_bar_shadow, with the size it would have got at
# a bar of 6 (agents/c6_shadow.py); tasks._check_c6_bar_outcomes adds its
# forward returns. No live threshold or order changes either way.
C6_BAR_SHADOW_ENABLED = os.getenv("C6_BAR_SHADOW_ENABLED", "true").lower() == "true"

# Record only: a routed OPEN/ADD (pass_to_risk True) gets 2 more analyst draws
# on its stored prompt (byte-exact by prompt_sha), same model and options,
# after routing, one task per draw on llm_p3 (after llm_p2), each waiting
# while the live LLM lanes have work. One activity_feed row, stage
# verdict_vote_shadow, records the draws and their vote (agents/
# verdict_vote.py); nothing reads it back. Off: a draw already running is
# not preempted, so a live analysis landing mid-draw waits up to one call
# (~3 min); about 15-20 extra GPU minutes a day. Not read from .env.
VERDICT_VOTE_SHADOW_ENABLED = False

# Record only: when an ADD writes the broker's average cost as the stop
# record's entry_price (execution._whole_position_stop), one activity_feed
# row, stage entry_basis, compares it with the fill-weighted average of the
# bot's own fills over the open lots and says where the trail arm, profit
# target and breakeven sit measured from those fills (execution.entry_basis).
# Nothing written to position_stops changes. Not read from .env.
ENTRY_BASIS_RECORD_ENABLED = True

# Record only (tasks.book_gap_risk_record, from position_exit_frame after the
# close): one book_gap_risk row per session with each held name's historical
# overnight gaps against today's stop distance, and the book's replayed
# overnight gap loss, all in percent. No threshold, alert or trade change.
GAP_RISK_RECORD_ENABLED = True

# Record only (tasks.book_deployment_record, from position_exit_frame after
# the close, after book_gap_risk): one book_deployment row per session with
# invested and cash as percent of net liquidation, the regime's
# cash_target_pct and the gap to it, the position count and the sector and
# largest-name weights. Percentages only. No threshold, alert, sizing or
# order change; nothing reads it. Not read from .env.
BOOK_DEPLOYMENT_RECORD_ENABLED = True

# Risk-per-trade-of-equity by LIVE setup category (Breakout/Catalyst/
# Continuation) -- the SETUP_GATES_ENABLED equivalent of RISK_PCT_PER_TRADE
# above, which stays exactly as-is and keeps governing Compounder/
# Accelerator/Moonshot sizing when the flag is off. Breakout gets the
# largest allocation: it's the most-corroborated of the three (a confirmed
# Stage 2 trend + a genuine multi-contraction base + a real volume-
# confirmed trigger, vs. Catalyst's two-source corroboration or
# Continuation's single pullback-and-resume read). All three sit inside
# Minervini's own stated 1.25-2.5% per-trade risk range, same source
# RISK_PCT_PER_TRADE's 2% midpoint was drawn from.
CATEGORY_RISK_PCT_PER_TRADE: dict[str, float] = {
    "Breakout":     0.015,
    "Catalyst":     0.010,
    "Continuation": 0.010,
}

# Soft %-of-portfolio ceiling by category -- (min, max) band, SETUP_GATES_
# ENABLED's equivalent of BUCKET_SIZE_TABLE's hard caps. "Soft" is load-
# bearing: risk_manager's new sizing path computes shares from
# risk_dollars / stop_distance FIRST (risk-first sizing, the design this
# whole phase is built around) and this ceiling only ever TRIMS the
# resulting share count -- it never pulls shares UP toward the band the
# way BUCKET_SIZE_TABLE's existing bands drive legacy sizing toward their
# midpoint. Relative ordering mirrors CATEGORY_RISK_PCT_PER_TRADE above.
CATEGORY_SOFT_CEILING_PCT: dict[str, tuple[float, float]] = {
    "Breakout":     (0.06, 0.08),
    "Catalyst":     (0.04, 0.06),
    "Continuation": (0.03, 0.05),
}

# Total open risk across every position -- db.get_portfolio_heat_pct's
# dollar-risk-to-CURRENT-stop definition, so a position already trailed to
# breakeven or better contributes zero heat -- as a % of equity. Scoped to
# the SETUP_GATES_ENABLED path only: the legacy bucket system has no
# portfolio-wide risk budget today, only per-position caps, and retrofitting
# one onto it is out of scope for this phase. risk_manager's new sizing path
# checks this before finalizing share count: if current_heat +
# this_trade's_own_risk would exceed PORTFOLIO_HEAT_CAP_PCT x
# REGIME_MULTIPLIER[regime] (tighter in cautious regimes, same multiplier
# every other sizing step already uses), the new trade is scaled DOWN to
# fit the remaining budget -- mirroring how the existing cash-reserve cap
# partial-sizes a trade instead of hard-blocking it outright.
PORTFOLIO_HEAT_CAP_PCT = 0.07

# Record-only book heat at each buy (agents/crew.py:_book_heat_record),
# written to risk_json as book_heat just before the approval: each held
# position's qty x (snapshot price - its position_stops stop) / portfolio
# value, their sum before and after this order at its final share count,
# n_positions, the largest single risk, and the order's risk in 20-session
# sigmas from the vol record. Reads the newest portfolio snapshot (at most
# 15 minutes old) and position_stops; no broker call. Fractions only.
# Nothing reads it and nothing enforces it on the legacy path; any heat cap
# there is a separate decision.
BOOK_HEAT_SHADOW_ENABLED = True

# Record-only fill watch (tasks.watch_broker_sells, every 2 min in market
# hours): a broker_sell_seen activity row the first time a SELL execution the
# bot did not log as its own order (a resting stop fill or a sale at the
# broker) shows up in today's executions. Reconcile still books the exit
# 15-30 min later; nothing reads these rows, and nothing is placed, cancelled
# or booked. No price or quantity on the row.
BROKER_SELL_WATCH_ENABLED = True

# A BUY that fills after its wait is booked by reconcile (tasks._book_late_fill),
# which decides from the approval's action whether the late shares opened a
# position or added to one. An OPEN on a ticker that was already held is then
# booked as a fresh entry: the stop record is rewritten at this order's level
# and the whole position re-stopped there, which can lower the old shares'
# stop (roadmap 04ah). execute_approved now stores the held count before the
# order on its order row (orders.held_before_qty). "shadow" (default): a
# late_fill_held_before activity row says what the stored count would change;
# the booking is unchanged. "enforce": an OPEN whose stored count is above 0,
# with the old shares still held, is booked as an ADD (the record and its
# level kept). Any other value: today's behaviour, no read, no write.
# Enforce reads before the late shares' stop: the stored count (3 s DB bound),
# the position cache, one executions pass (5 s bound) and the other-BUY count
# (3 s): up to ~11 s more before that stop. Shadow does them after the run.
LATE_FILL_HELD_BEFORE_MODE = "shadow"

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
