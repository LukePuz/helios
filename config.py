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

# ---------------------------------------------------------------------------
# Sell behaviour
# ---------------------------------------------------------------------------
TRIM_FRACTION = 0.33   # analyst TRIM sells this fraction of the held position

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
# Brief sections that change what the analyst sees (professor M37-M39, Q35).
# Off until Luke decides whether the model should see its own past verdicts
# and the gate inputs before the call (2026-10-28).
BRIEF_PRICE_STRUCTURE = os.getenv("BRIEF_PRICE_STRUCTURE", "false").lower() == "true"
BRIEF_POSITION_CONTEXT = os.getenv("BRIEF_POSITION_CONTEXT", "false").lower() == "true"
# Market-regime fallback reads only overall-market rows (audit 20:53 A2053-1,
# 08:53 A0853-1). On from 10-30 (Luke D3, 10-29); false restores sector-row reads.
REGIME_OVERALL_ONLY = os.getenv("REGIME_OVERALL_ONLY", "true").lower() == "true"
# Regime and scanner score completed daily bars only (A1053-2; Luke D3).
REGIME_COMPLETED_BARS = os.getenv("REGIME_COMPLETED_BARS", "true").lower() == "true"
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
