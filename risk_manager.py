"""
Risk Manager — bucket-aware position sizing, ATR-based stops, and drawdown protection.

Implements the full risk framework from knowledge/risk_rules.md:
  - Sizing by bucket (Compounder/Accelerator/Moonshot) × conviction × regime multiplier
  - ATR-based stop calculation
  - Concentration and correlation checks
  - Portfolio-level drawdown halt

Maintains backward-compatible interface for any modules still calling
shares_affordable() and check_sell_allowed().
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

try:
    from config import (
        DRAWDOWN_HALT_PCT,
        BUCKET_SIZE_TABLE,
        REGIME_MULTIPLIER,
        ATR_STOP_MULTIPLIER,
        MOONSHOT_MAX_LOSS_PCT,
        RISK_PCT_PER_TRADE,
        TRAILING_STOP_ACTIVATE,
        TRAILING_STOP_DISTANCE,
        MAX_POSITION_PCT_COMPOUNDER,
        MAX_POSITION_PCT_ACCELERATOR,
        MAX_POSITION_PCT_MOONSHOT,
        MAX_SINGLE_THEME_PCT,
        MAX_AI_INFRA_PCT,
        MAX_MOONSHOT_TOTAL_PCT,
        MAX_TRADE_PCT_OF_CASH_RESERVE,
        SWAP_MIN_CONVICTION_GAP,
        SWAP_MAX_CANDIDATE_CONVICTION,
        get_active_limits,
        TRADING_ENV,
        PROFIT_TARGET_PCT_BY_STYLE,
        PARTIAL_SELL_PCT_BY_STYLE,
        STALE_WINDOW_DAYS_BY_STYLE,
        STALE_MOVE_THRESHOLD_PCT,
        SETUP_GATES_ENABLED,
        CATEGORY_RISK_PCT_PER_TRADE,
        CATEGORY_SOFT_CEILING_PCT,
        PORTFOLIO_HEAT_CAP_PCT,
        ALLOW_ONE_SHARE_ABOVE_ALLOCATION,
    )
except ImportError:
    ALLOW_ONE_SHARE_ABOVE_ALLOCATION = False
    TRADING_ENV                   = "paper"
    DRAWDOWN_HALT_PCT             = 0.10
    BUCKET_SIZE_TABLE             = {
        "Compounder":  {9: (0.08, 0.12), 7: (0.05, 0.08), 5: (0.03, 0.05), 3: (0.02, 0.03)},
        "Accelerator": {9: (0.04, 0.07), 7: (0.03, 0.05), 5: (0.02, 0.03), 3: (0.01, 0.02)},
        "Moonshot":    {9: (0.01, 0.02), 7: (0.005, 0.01), 5: (0.0025, 0.005), 3: (None, None)},
    }
    REGIME_MULTIPLIER             = {"Strong Risk-On": 1.2, "Risk-On": 1.0, "Neutral": 0.75,
                                     "Risk-Off": 0.5, "Strong Risk-Off": 0.25}
    ATR_STOP_MULTIPLIER           = {"Compounder": (2.0, 3.0), "Accelerator": (1.5, 2.0), "Moonshot": (None, None)}
    MOONSHOT_MAX_LOSS_PCT         = 0.20
    RISK_PCT_PER_TRADE            = 0.02
    TRAILING_STOP_ACTIVATE        = {"Compounder": 0.30, "Accelerator": 0.20}
    TRAILING_STOP_DISTANCE        = {"Compounder": (0.15, 0.20), "Accelerator": (0.10, 0.15)}
    MAX_POSITION_PCT_COMPOUNDER   = 0.12
    MAX_POSITION_PCT_ACCELERATOR  = 0.07
    MAX_POSITION_PCT_MOONSHOT     = 0.02
    MAX_SINGLE_THEME_PCT          = 0.35
    MAX_AI_INFRA_PCT              = 0.60
    MAX_MOONSHOT_TOTAL_PCT        = 0.10
    MAX_TRADE_PCT_OF_CASH_RESERVE = 0.20
    SWAP_MIN_CONVICTION_GAP       = 2
    SWAP_MAX_CANDIDATE_CONVICTION = 6
    PROFIT_TARGET_PCT_BY_STYLE    = {
        "Trending/Expanding":  {"Accelerator": (0.20, 0.30), "Moonshot": (0.40, 0.60)},
        "Choppy/Distribution": {"Accelerator": (0.10, 0.15), "Moonshot": (0.20, 0.30)},
    }
    PARTIAL_SELL_PCT_BY_STYLE     = {"Trending/Expanding": 0.25, "Choppy/Distribution": 0.45}
    STALE_WINDOW_DAYS_BY_STYLE    = {
        "Trending/Expanding":  {"Accelerator": 28, "Moonshot": 21},
        "Choppy/Distribution": {"Accelerator": 14, "Moonshot": 10},
    }
    STALE_MOVE_THRESHOLD_PCT      = 0.05
    SETUP_GATES_ENABLED           = False
    CATEGORY_RISK_PCT_PER_TRADE   = {"Breakout": 0.015, "Catalyst": 0.010, "Continuation": 0.010}
    CATEGORY_SOFT_CEILING_PCT     = {"Breakout": (0.06, 0.08), "Catalyst": (0.04, 0.06), "Continuation": (0.03, 0.05)}
    PORTFOLIO_HEAT_CAP_PCT        = 0.07

    def get_active_limits(balance: float) -> dict:
        return {"tier": "standard", "max_concurrent_positions": None,
                "min_conviction_to_queue": 5,
                "max_position_pct_moonshot": MAX_POSITION_PCT_MOONSHOT,
                "max_position_pct_compounder": MAX_POSITION_PCT_COMPOUNDER,
                "max_position_pct_accelerator": MAX_POSITION_PCT_ACCELERATOR,
                "single_share_cap_pct": None}


# ---------------------------------------------------------------------------
# Helpers: Alpaca bar fetch for ATR
# ---------------------------------------------------------------------------

def _fetch_bars_for_atr(ticker: str, days: int = 20) -> list[dict]:
    try:
        from ibkr import get_historical_bars
        bars = get_historical_bars(ticker, duration=f"{days + 5} D", bar_size="1 day")
        return [{"h": b["high"], "l": b["low"], "c": b["close"]} for b in bars]
    except Exception as e:
        log.debug(f"ATR bar fetch failed for {ticker}: {e}")
        return []


def _compute_atr(bars: list[dict]) -> float | None:
    """True Range ATR over available bars."""
    if len(bars) < 2:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs)


# ---------------------------------------------------------------------------
# Conviction band lookup
# ---------------------------------------------------------------------------

def _price_correlation(bars_a: list[dict], bars_b: list[dict]) -> float | None:
    """
    Pearson correlation of daily returns between two {date, close} bar series,
    aligned by date (only dates present in both series are used).
    Returns None if fewer than 10 overlapping return observations exist.
    """
    a_by_date = {b["date"]: b["close"] for b in bars_a}
    b_by_date = {b["date"]: b["close"] for b in bars_b}
    common_dates = sorted(set(a_by_date) & set(b_by_date))
    if len(common_dates) < 15:  # need >= 15 prices for >= 14 daily returns
        return None

    a_prices = [a_by_date[d] for d in common_dates]
    b_prices = [b_by_date[d] for d in common_dates]

    a_returns = [(a_prices[i] - a_prices[i - 1]) / a_prices[i - 1]
                 for i in range(1, len(a_prices)) if a_prices[i - 1]]
    b_returns = [(b_prices[i] - b_prices[i - 1]) / b_prices[i - 1]
                 for i in range(1, len(b_prices)) if b_prices[i - 1]]
    if len(a_returns) < 10 or len(a_returns) != len(b_returns):
        return None

    try:
        import numpy as np
        corr = np.corrcoef(a_returns, b_returns)[0, 1]
        return float(corr) if corr == corr else None  # NaN check
    except Exception as e:
        log.debug(f"Correlation computation failed: {e}")
        return None


def _conviction_band(conviction: int) -> int:
    """Maps conviction (1-10) to the nearest lower band key (9,7,5,3)."""
    if conviction >= 9:
        return 9
    elif conviction >= 7:
        return 7
    elif conviction >= 5:
        return 5
    else:
        return 3


# ---------------------------------------------------------------------------
# RiskManager class
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(self, client: object | None = None, start_value: float | None = None):
        if client is None:
            from ibkr import get_client
            client = get_client()
        self._client = client
        account = self._get_account()
        self.start_value: float = start_value or float(account.portfolio_value)

    def _get_account(self) -> object:
        return self._client.get_account()

    def portfolio_value(self) -> float:
        return float(self._get_account().portfolio_value)

    def buying_power(self) -> float:
        return float(self._get_account().buying_power)

    def cash(self) -> float:
        """Raw cash balance (not leveraged buying power) — the actual reserve
        the per-trade cash cap in size_position() is measured against."""
        return float(self._get_account().cash)

    def _alert_cash_blocked(self, cash_reserve: float) -> None:
        """Cash exhaustion used to fail silently — a buy would just come
        back with 0 shares and crew.py would drop it with a debug log line,
        with nobody told why the system stopped opening new positions.
        Throttled to once per calendar day."""
        try:
            import db
            from datetime import date
            today = date.today().isoformat()
            if db.get_state("cash_blocked_alert_date") == today:
                return
            db.set_state("cash_blocked_alert_date", today)
            import alerts
            alerts._send(
                f"💰 ALL BUYS BLOCKED — cash reserve is ${cash_reserve:,.2f}\n"
                f"The cash-reserve cap ({MAX_TRADE_PCT_OF_CASH_RESERVE:.0%}/trade) can no "
                f"longer size any new position. This is expected once the account "
                f"is close to fully deployed; sells/exits still work normally."
            )
        except Exception as e:
            log.warning(f"_alert_cash_blocked failed: {e}")

    # ------------------------------------------------------------------
    # Drawdown halt
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        """
        Halt if (a) the persisted manual/auto halt flag is set, or
        (b) drawdown from the persisted high-water mark exceeds the limit.
        Baselines live in the system_state table so every process agrees.

        Keys are trading_env-scoped (high_water_mark_paper vs
        high_water_mark_live, etc.) -- found 2026-08-14: a shared
        unscoped "high_water_mark" key held paper's ~$1.09M baseline, and
        reconnecting to the live account (~$2k) would have hit the
        implausible-value guard below on EVERY check, permanently
        (not just on one transient bad read, which is what that guard is
        actually for) -- silently disabling the drawdown halt, the single
        most important safety mechanism in the system, for live trading.
        """
        # Persisted halt flag (set manually via /halt or by the drawdown task)
        try:
            import db
            if db.get_state(f"trading_halted_{TRADING_ENV}", "false") == "true":
                log.warning(f"HALT — trading_halted_{TRADING_ENV} flag is set in system_state")
                return True
            hwm_str = db.get_state(f"high_water_mark_{TRADING_ENV}")
        except Exception as e:
            log.warning(f"Halt state read failed ({e}) — falling back to instance baseline")
            hwm_str = None

        current  = self.portfolio_value()
        baseline = float(hwm_str) if hwm_str else self.start_value
        if baseline <= 0:
            return False

        # Plausibility guard: a transient broker read of $0 (clientId
        # collision, cold-morning connection race) would otherwise compute
        # as a 100% drawdown and latch trading_halted until a manual
        # /resume. Treat an implausible single-step read as "unknown, don't
        # halt" rather than as a real drawdown — this is checked on every
        # buy attempt, not on a timer, so failing open here just means one
        # sizing call proceeds on stale-but-not-dangerous data; failing
        # closed would mean one bad read stops all trading indefinitely.
        if current <= 0 or current < 0.5 * baseline:
            log.warning(f"Implausible portfolio value ${current:,.2f} "
                       f"(baseline ${baseline:,.2f}) — skipping halt check")
            return False

        drawdown = (baseline - current) / baseline
        if drawdown >= DRAWDOWN_HALT_PCT:
            log.warning(
                f"DRAWDOWN HALT — drawdown {drawdown:.1%} exceeds {DRAWDOWN_HALT_PCT:.0%} "
                f"(high-water=${baseline:,.2f}, current=${current:,.2f})"
            )
            try:
                import db
                db.set_state(f"trading_halted_{TRADING_ENV}", "true")
                db.set_state(f"halt_reason_{TRADING_ENV}",
                             f"Auto: drawdown {drawdown:.1%} from HWM ${baseline:,.2f}")
            except Exception:
                pass
            return True

        # Ratchet the high-water mark up as the portfolio grows
        try:
            import db
            if current > baseline:
                db.set_state(f"high_water_mark_{TRADING_ENV}", str(current))
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Bucket-aware position sizing
    # ------------------------------------------------------------------

    def size_position(
        self,
        ticker: str,
        conviction: int,
        bucket: str,
        regime: str,
        price: float,
    ) -> dict:
        """
        Calculate full sizing parameters for a new position.

        Returns a dict with:
            position_size_pct    — target allocation as % of portfolio
            dollar_amount        — target dollar value
            initial_entry_pct   — 50-60% staged entry first tranche
            initial_dollar       — dollar value for first tranche
            shares_initial       — whole shares for first tranche
            stop_loss_price      — computed stop level
            stop_method          — description of stop methodology
            trailing_stop_activate_at_gain_pct
            trailing_stop_distance_pct
            max_loss_if_stopped  — dollars at risk if stop hits from entry
            regime_multiplier    — applied multiplier
            flags                — list of warning strings
        """
        if self.is_halted():
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": ["HALTED — portfolio drawdown limit reached"]}

        pv     = self.portfolio_value()
        limits = get_active_limits(pv)

        # Account-tier conviction floor — added 2026-08-13 ahead of the live
        # rollout. Below $25k the whole-shares-only constraint below makes
        # low-conviction ideas structurally more likely to waste a scarce
        # position slot on an unexecutable or marginal size; Reduced/Micro
        # require a higher bar before sizing even begins. No effect at
        # Standard (min_conviction_to_queue == today's MIN_CONVICTION_TO_QUEUE).
        if conviction < limits["min_conviction_to_queue"]:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Conviction {conviction} below the {limits['tier']}-tier minimum "
                              f"({limits['min_conviction_to_queue']}) — skipping"]}

        # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_GATES_
        # ENABLED and a live setup-gate classification + trade ladder exist
        # for this ticker, risk-first sizing off the gate's structural stop
        # REPLACES the legacy %-of-account allocation-table method below
        # entirely for this call. Returns None (not a sizing dict) when
        # there's no live category/ladder data for this ticker (e.g. a
        # legacy position still carrying an old bucket, or the gates
        # haven't classified it) -- same "insufficient data -> fall back"
        # convention compute_trend_template already uses elsewhere in this
        # codebase, so the pre-cutover method directly below is always the
        # fallback, never bypassed outright.
        if SETUP_GATES_ENABLED:
            new_path = self._size_position_setup_gates(ticker, conviction, bucket, regime, price, pv)
            if new_path is not None:
                return new_path

        band  = _conviction_band(conviction)
        sizes = BUCKET_SIZE_TABLE.get(bucket, BUCKET_SIZE_TABLE["Accelerator"])
        lo, hi = sizes.get(band, (0.01, 0.02))

        if lo is None:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Conviction {conviction} too low for {bucket} — WATCH only"]}

        # BUCKET_SIZE_TABLE's bands are calibrated against each bucket's
        # Standard-tier hard cap — raising the hard cap alone doesn't change
        # anything because these bands, not the cap, are what actually binds
        # at Standard. At Reduced/Micro the tier cap IS the intended scale,
        # so scale the bands themselves by the same ratio, preserving
        # conviction-band shape instead of flattening it. Originally
        # Moonshot-only (2026-08-13 sign-off); extended to Compounder/
        # Accelerator 2026-08-21 after the same silent-drop failure mode
        # (sub-1-share allocations, no alert) turned out to affect them too
        # — see TIER_MAX_POSITION_PCT_COMPOUNDER/ACCELERATOR in config.py.
        standard_cap = {
            "Compounder":  MAX_POSITION_PCT_COMPOUNDER,
            "Accelerator": MAX_POSITION_PCT_ACCELERATOR,
            "Moonshot":    MAX_POSITION_PCT_MOONSHOT,
        }.get(bucket)
        tier_cap = {
            "Compounder":  limits.get("max_position_pct_compounder"),
            "Accelerator": limits.get("max_position_pct_accelerator"),
            "Moonshot":    limits.get("max_position_pct_moonshot"),
        }.get(bucket)
        if limits["tier"] != "standard" and standard_cap and tier_cap:
            tier_scale = tier_cap / standard_cap
            lo, hi = lo * tier_scale, hi * tier_scale

        # Midpoint of range, then apply regime multiplier
        base_pct  = (lo + hi) / 2
        mult      = REGIME_MULTIPLIER.get(regime, 0.75)
        target    = base_pct * mult

        # Enforce hard caps — tier-dependent for all three buckets (see
        # config.TIER_MAX_POSITION_PCT_*): the Standard-tier caps are
        # structurally too small to clear the whole-share floor below at
        # Reduced/Micro balances, so each is raised per-tier instead.
        hard_cap = tier_cap if tier_cap is not None else MAX_POSITION_PCT_ACCELERATOR

        target = min(target, hard_cap)

        # Check if adding would exceed current position cap
        held = self.current_position_value(ticker)
        held_pct = held / pv if pv > 0 else 0

        # Max-concurrent-positions cap — Reduced/Micro tiers only, and only
        # for a genuinely NEW position (held_pct == 0); adding to something
        # already held doesn't consume a new slot. Checked here rather than
        # upstream in crew.py so it applies uniformly to every caller of
        # size_position, not just the auto-execution path.
        if held_pct == 0 and limits["max_concurrent_positions"] is not None:
            try:
                open_count = len(self._client.get_all_positions())
            except Exception as e:
                log.error(f"Could not fetch position count for max-concurrent check: {e}")
                open_count = None
            if open_count is not None and open_count >= limits["max_concurrent_positions"]:
                return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                        "cash_blocked": False,
                        "flags": [f"At the {limits['tier']}-tier max of "
                                  f"{limits['max_concurrent_positions']} concurrent positions "
                                  f"({open_count} held) — skipping new {ticker} position"]}

        remaining = max(0, target - held_pct)
        if remaining < 0.005:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Position already at/near target ({held_pct:.1%} held)"]}

        if not price or price <= 0:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"Invalid price ({price!r}) — cannot size position"]}

        # Single-share cap — Reduced/Micro tiers only (see
        # config.TIER_SINGLE_SHARE_CAP_PCT). One share alone can't already
        # represent more than this fraction of the account; guards against
        # one expensive share structurally dominating a small portfolio in
        # a way the % sizing bands above don't otherwise catch, since those
        # operate on target allocation, not on what a single indivisible
        # share costs.
        share_cap_pct = limits["single_share_cap_pct"]
        if share_cap_pct is not None and price > pv * share_cap_pct:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False,
                    "flags": [f"{ticker} share price ${price:,.2f} exceeds the {limits['tier']}-tier "
                              f"single-share cap of {share_cap_pct:.0%} portfolio "
                              f"(${pv * share_cap_pct:,.2f}) — skipping"]}

        dollar_amount  = pv * remaining
        initial_pct    = 0.55  # 55% initial tranche (midpoint of 50-60%)
        initial_dollar = dollar_amount * initial_pct

        flags = []

        # Cash-reserve cap — this trade's actual spend may not exceed a fixed
        # fraction of *current cash*, not portfolio value. Since the cap
        # shrinks along with cash, remaining cash decays geometrically toward
        # zero instead of hitting it outright after a burst of auto-executes.
        cash_reserve = self.cash()
        cash_cap = max(cash_reserve, 0) * MAX_TRADE_PCT_OF_CASH_RESERVE
        cash_capped = initial_dollar > cash_cap
        if cash_capped:
            initial_dollar = cash_cap
            flags.append(
                f"Capped at {MAX_TRADE_PCT_OF_CASH_RESERVE:.0%} of cash reserve "
                f"(${cash_reserve:,.0f} available)"
            )

        # Whole shares only — IBKR rejects fractional stock orders on most tickers
        shares_initial = int(initial_dollar / price)

        # One share above the allocation (2026-10-27). On a micro account the
        # 55% first tranche is often smaller than one share of a $400-600
        # name, so every pass since Oct 13 sized to 0 and nothing traded.
        # Allow exactly 1 share when it still fits the hard per-position cap
        # and the cash-reserve cap; the single-share cap above and the
        # per-trade risk cap below still apply. Decided by the lead session
        # after Luke delegated the call (Jarvis, bus #36).
        # Held value counts toward the cap: on an ADD, checking the new share
        # alone let the position reach target + one share, up to about twice
        # the hard cap (limitations audit 2026-10-28 09:00, finding 5).
        if (shares_initial < 1 and ALLOW_ONE_SHARE_ABOVE_ALLOCATION
                and held + price <= pv * hard_cap and price <= cash_cap):
            flags.append(
                f"Allocation ${initial_dollar:,.2f} is under 1 share; buying 1 share "
                f"(${price:,.2f} = {price / pv:.1%} of portfolio, within the "
                f"{hard_cap:.0%} position cap)"
            )
            shares_initial = 1
            initial_dollar = price

        if shares_initial < 1:
            cash_blocked = cash_reserve <= 0 or cash_capped
            if cash_blocked:
                # This is buys silently vanishing at crew.py's "0 shares"
                # drop with no one told why — surface it, throttled to once
                # a day so a busy morning doesn't spam the same alert on
                # every subsequent buy attempt.
                self._alert_cash_blocked(cash_reserve)
            # cash_blocked distinguishes "genuinely out of room" from every
            # other reason size_position can return 0 shares (halted,
            # conviction too low for the bucket, already at target, bad
            # price) — 2026-08-13, this is what tells the conviction-swap
            # check in crew.py whether it's even worth looking for a
            # position to rotate out of.
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": cash_blocked,
                    "flags": flags + [f"Allocation ${initial_dollar:,.2f} buys less than 1 share "
                              f"of {ticker} at ${price:,.2f} — skipping"]}

        # Stop loss
        stop_price, stop_method = self.calculate_stop(ticker, price, bucket)

        # Fixed dollar-risk sizing cap (2026-09-11, user request, Minervini
        # SEPA methodology): shares_initial above came purely from the
        # bucket's %-of-account allocation table -- a wide-stop trade and a
        # tight-stop trade in the same conviction band get the SAME share
        # count, with the dollar risk simply whatever falls out of that
        # allocation times the stop distance. This recomputes shares so a
        # stop-out costs at most RISK_PCT_PER_TRADE of equity and takes
        # whichever of the two share counts is SMALLER -- it only ever
        # tightens sizing on a wide-stop trade, never loosens beyond the
        # existing % cap. Applied after the whole-shares/cash-cap checks
        # above (which already guarantee shares_initial >= 1) since the
        # stop distance -- and therefore this cap -- isn't known until
        # calculate_stop() runs.
        if stop_price and stop_price < price:
            risk_dollars = pv * RISK_PCT_PER_TRADE
            risk_based_shares = int(risk_dollars / (price - stop_price))
            if risk_based_shares < shares_initial:
                if risk_based_shares < 1:
                    return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                            "cash_blocked": False,
                            "flags": flags + [
                                f"Stop distance (${price - stop_price:.2f}, {stop_method}) is too wide to "
                                f"buy even 1 share of {ticker} within the {RISK_PCT_PER_TRADE:.1%} "
                                f"per-trade risk cap (${risk_dollars:,.2f}) — skipping"]}
                flags.append(
                    f"Risk-based sizing: ${price - stop_price:.2f} stop distance limits to "
                    f"{risk_based_shares} shares (vs {shares_initial} from the {bucket} allocation "
                    f"cap) to keep max loss at/under {RISK_PCT_PER_TRADE:.1%} of equity"
                )
                shares_initial = risk_based_shares
                initial_dollar = shares_initial * price

        # Trailing stop params
        trail_activate = TRAILING_STOP_ACTIVATE.get(bucket)
        trail_dist_lo, trail_dist_hi = TRAILING_STOP_DISTANCE.get(bucket, (None, None))
        trail_distance = (trail_dist_lo + trail_dist_hi) / 2 if trail_dist_lo else None

        # Max loss at stop
        initial_cost = shares_initial * price
        max_loss = (initial_cost - shares_initial * stop_price) if stop_price else initial_cost * 0.35

        if mult < 1.0:
            flags.append(f"Regime multiplier {mult}x applied ({regime})")
        if target == hard_cap:
            flags.append(f"Size capped at {hard_cap:.0%} bucket maximum")

        return {
            "ticker":              ticker,
            "position_size_pct":   round(remaining, 4),
            "dollar_amount":       round(dollar_amount, 2),
            "initial_entry_pct":   initial_pct,
            "initial_dollar":      round(initial_dollar, 2),
            "shares_initial":      shares_initial,
            # What this order alone buys, as a share of the portfolio. Can
            # exceed position_size_pct when one whole share is above target.
            "order_pct":           round(shares_initial * price / pv, 4) if pv else 0,
            "stop_loss_price":     stop_price,
            "stop_method":         stop_method,
            "trailing_stop_activate_at_gain_pct": trail_activate,
            "trailing_stop_distance_pct": trail_distance,
            "max_loss_if_stopped": round(max_loss, 2),
            "regime_multiplier":   mult,
            "bucket":              bucket,
            "conviction":          conviction,
            "flags":               flags,
        }

    # ------------------------------------------------------------------
    # Swing-trading cutover, Phase 4/5 (2026-10-11): setup-gate sizing path
    # ------------------------------------------------------------------

    def _size_position_setup_gates(
        self, ticker: str, conviction: int, bucket: str, regime: str, price: float, pv: float,
    ) -> dict | None:
        """
        SETUP_GATES_ENABLED sizing path, called from size_position() above.
        Risk-first: get the gate's own structural stop_price FIRST (from
        agents.setup_gates.classify_setup, which reads agents.trade_ladder's
        nightly rungs), THEN derive risk_dollars = equity x
        CATEGORY_RISK_PCT_PER_TRADE[category] x REGIME_MULTIPLIER[regime],
        THEN shares = risk_dollars / (entry - stop) -- the reverse order
        from the legacy path, which sizes off a %-of-account table first and
        only derives dollar risk as a side effect. CATEGORY_SOFT_CEILING_PCT
        is then applied as a pure TRIM (never raises shares), same posture
        as the cash-reserve and portfolio-heat caps below.

        Returns None (never a zero-share dict) when there's no live
        category+ladder match for this ticker -- the caller must fall back
        to the legacy allocation-table method in that case. Every actual
        "can't size this" outcome below (stop too wide, at/above ceiling,
        cash-blocked, heat-blocked) returns a real 0-share sizing dict
        instead, matching the legacy path's own convention so callers
        (agents/crew.py's swap-candidate logic, alerts) see a consistent
        shape either way.
        """
        if bucket not in CATEGORY_RISK_PCT_PER_TRADE:
            return None
        try:
            import db
            from agents.setup_gates import classify_setup
        except Exception as e:
            log.warning(f"[{ticker}] setup-gate sizing unavailable ({e}) — falling back to legacy sizing")
            return None

        ladder = db.get_trade_ladder(ticker)
        if not ladder:
            return None

        try:
            best, _all_gates = classify_setup(ticker)
        except Exception as e:
            log.warning(f"[{ticker}] classify_setup failed ({e}) — falling back to legacy sizing")
            return None

        if not best or best.get("category") != bucket or best.get("stop_price") is None:
            return None

        stop_price = float(best["stop_price"])
        if not price or price <= 0 or stop_price >= price:
            # A structurally invalid stop (at/above entry) can't be risk-sized
            # at all — fall back rather than divide by zero/negative.
            log.warning(f"[{ticker}] setup-gate stop ${stop_price:.2f} is not below "
                        f"price ${price:.2f} — falling back to legacy sizing")
            return None

        risk_pct     = CATEGORY_RISK_PCT_PER_TRADE[bucket]
        mult         = REGIME_MULTIPLIER.get(regime, 0.75)
        risk_dollars = pv * risk_pct * mult
        shares       = int(risk_dollars / (price - stop_price))

        flags: list[str] = []
        if shares < 1:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False, "heat_blocked": False,
                    "flags": [f"Stop distance (${price - stop_price:.2f}, structural/{bucket}) is too "
                              f"wide to buy even 1 share of {ticker} within the {risk_pct:.1%} "
                              f"per-trade risk cap (${risk_dollars:,.2f}) — skipping"]}

        # Soft ceiling — trims, never raises (see config.CATEGORY_SOFT_CEILING_PCT).
        held     = self.current_position_value(ticker)
        _lo, ceiling_hi = CATEGORY_SOFT_CEILING_PCT.get(bucket, (None, None))
        if ceiling_hi is not None:
            ceiling_dollar = max(0.0, pv * ceiling_hi - held)
            ceiling_shares = int(ceiling_dollar / price) if price else 0
            if ceiling_shares < shares:
                flags.append(f"Soft ceiling ({ceiling_hi:.0%} of portfolio) trims {shares} -> {ceiling_shares} shares")
                shares = ceiling_shares
        if shares < 1:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": False, "heat_blocked": False,
                    "flags": flags + [f"{ticker} already at/above the {bucket} soft ceiling — skipping"]}

        # Cash-reserve cap — identical convention to the legacy path.
        cash_reserve  = self.cash()
        cash_cap      = max(cash_reserve, 0) * MAX_TRADE_PCT_OF_CASH_RESERVE
        initial_dollar = shares * price
        cash_capped   = initial_dollar > cash_cap
        if cash_capped:
            shares = int(cash_cap / price) if price else 0
            initial_dollar = shares * price
            flags.append(f"Capped at {MAX_TRADE_PCT_OF_CASH_RESERVE:.0%} of cash reserve "
                         f"(${cash_reserve:,.0f} available)")
        cash_blocked = False
        if shares < 1:
            cash_blocked = cash_reserve <= 0 or cash_capped
            if cash_blocked:
                self._alert_cash_blocked(cash_reserve)
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "cash_blocked": cash_blocked, "heat_blocked": False,
                    "flags": flags + [f"Allocation ${initial_dollar:,.2f} buys less than 1 share "
                              f"of {ticker} at ${price:,.2f} — skipping"]}

        # Portfolio heat cap (config.PORTFOLIO_HEAT_CAP_PCT) — partial-sizes
        # down to fit the remaining budget, the same "scale down, don't
        # hard-block" posture the cash-reserve cap above already uses, not
        # a hard block. See db.get_portfolio_heat_pct's docstring for the
        # exact dollar-risk-to-CURRENT-stop definition.
        heat_blocked = False
        try:
            positions_map = {p.symbol: {"qty": float(p.qty)} for p in self._client.get_all_positions()}
            heat = db.get_portfolio_heat_pct(pv, positions_map)
            current_heat_pct = heat.get("heat_pct", 0.0)
        except Exception as e:
            log.warning(f"[{ticker}] portfolio heat lookup failed ({e}) — treating current heat as 0")
            current_heat_pct = 0.0

        heat_cap = PORTFOLIO_HEAT_CAP_PCT * mult
        new_trade_risk_pct = (shares * (price - stop_price)) / pv if pv else 0.0
        if current_heat_pct + new_trade_risk_pct > heat_cap:
            remaining_budget_dollars = max(0.0, heat_cap - current_heat_pct) * pv
            heat_capped_shares = int(remaining_budget_dollars / (price - stop_price))
            if heat_capped_shares < shares:
                flags.append(
                    f"Portfolio heat cap ({heat_cap:.1%} of equity, {current_heat_pct:.1%} already used) "
                    f"trims {shares} -> {heat_capped_shares} shares"
                )
                shares = heat_capped_shares
                initial_dollar = shares * price
            if shares < 1:
                heat_blocked = True
                return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                        "cash_blocked": False, "heat_blocked": True,
                        "flags": flags + [f"Portfolio heat already at/above the {heat_cap:.1%} cap "
                                  f"({current_heat_pct:.1%} used) — skipping"]}

        max_loss = shares * (price - stop_price)
        if mult < 1.0:
            flags.append(f"Regime multiplier {mult}x applied ({regime})")

        return {
            "ticker":              ticker,
            "position_size_pct":   round(initial_dollar / pv, 4) if pv else 0,
            "dollar_amount":       round(initial_dollar, 2),
            "initial_entry_pct":   1.0,   # risk-sized directly, no staged tranche
            "initial_dollar":      round(initial_dollar, 2),
            "shares_initial":      shares,
            "order_pct":           round(initial_dollar / pv, 4) if pv else 0,
            "stop_loss_price":     stop_price,
            "stop_method":         f"Setup-gate structural stop ({bucket})",
            "trailing_stop_activate_at_gain_pct": None,  # see check_trailing_stops' ladder-ratchet path
            "trailing_stop_distance_pct": None,
            "max_loss_if_stopped": round(max_loss, 2),
            "regime_multiplier":   mult,
            "bucket":              bucket,
            "conviction":          conviction,
            "cash_blocked":        cash_blocked,
            "heat_blocked":        heat_blocked,
            "target_prices":       best.get("target_prices", []),
            "flags":               flags,
        }

    # ------------------------------------------------------------------
    # ATR-based stop calculation
    # ------------------------------------------------------------------

    def get_atr(self, ticker: str, days: int = 14) -> float | None:
        bars = _fetch_bars_for_atr(ticker, days=days)
        return _compute_atr(bars)

    def calculate_stop(
        self, ticker: str, entry_price: float, bucket: str
    ) -> tuple[float | None, str]:
        """
        Returns (stop_price, method_description).
        """
        # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_GATES_
        # ENABLED and a live gate classification with a real stop_price
        # exists for this ticker under this exact category, use it directly
        # instead of the ATR-multiple calc below. Falls back to the
        # existing ATR logic otherwise (bad classify_setup call, no live
        # match, or no stop_price on the match) -- same "insufficient data
        # -> fall back" convention compute_trend_template already uses
        # elsewhere in this codebase.
        if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
            try:
                from agents.setup_gates import classify_setup
                best, _all_gates = classify_setup(ticker)
            except Exception as e:
                log.warning(f"[{ticker}] setup-gate stop lookup failed ({e}) — falling back to ATR stop")
                best = None
            if best and best.get("category") == bucket and best.get("stop_price") is not None:
                return float(best["stop_price"]), f"Setup-gate structural stop ({bucket})"

        if bucket == "Moonshot":
            stop = round(entry_price * (1 - MOONSHOT_MAX_LOSS_PCT), 2)
            return stop, f"Moonshot hard stop: {MOONSHOT_MAX_LOSS_PCT:.0%} max loss from entry"

        atr = self.get_atr(ticker)
        if atr is None or atr == 0:
            fallback_pct = 0.08 if bucket == "Compounder" else 0.05
            stop = round(entry_price * (1 - fallback_pct), 2)
            log.warning(
                f"[{ticker}] ATR unavailable — using flat {fallback_pct:.0%} fallback stop ${stop:.2f}. "
                "Stop accuracy may be poor; verify manually."
            )
            try:
                import alerts as _al
                _al._send(
                    f"⚠️ ATR Fallback Stop — {ticker}\n"
                    f"Could not compute ATR-based stop. Using flat <b>{fallback_pct:.0%}</b> stop "
                    f"(${stop:.2f}). Verify this is appropriate before approving."
                )
            except Exception:
                pass
            return stop, f"Fallback stop ({fallback_pct:.0%} below entry — ATR unavailable)"

        mult_lo, mult_hi = ATR_STOP_MULTIPLIER.get(bucket, (1.5, 2.0))
        mult = (mult_lo + mult_hi) / 2
        stop = round(entry_price - mult * atr, 2)
        return stop, f"{mult:.1f}× ATR ({atr:.2f}) below entry = ${stop:.2f}"

    # ------------------------------------------------------------------
    # Concentration and correlation checks
    # ------------------------------------------------------------------

    def check_concentration(self, ticker: str, proposed_size_pct: float,
                            bucket: str | None = None) -> dict:
        """
        Checks if adding ticker at proposed_size_pct would breach any limit.
        Returns {passed, flags, ai_infrastructure_total, ai_infra_size_cap_pct}.
        FAIL-CLOSED: an errored check counts as failed so a broken data feed
        can never wave a trade through unchecked.

        2026-08-11: this used to be advisory only — a "failed" check just
        appended text to entry_strategy's flags field, which nobody reviews
        in real time since auto-execution has no human-in-the-loop tap
        (2026-07-14 decision). The portfolio sat over its own stated 60% AI
        infra limit for a week-plus (65.3%-65.8% observed) while every new
        ADD to an AI-infra name kept sailing through anyway. ai_infra_total
        exceeding the limit is still only informational for the *other*
        sub-checks below (theme/moonshot/single-position) — but for AI
        infra specifically, the caller now gets ai_infra_size_cap_pct: the
        largest proposed_size_pct for THIS ticker that would bring the
        total back to exactly the limit, so sizing can be capped instead of
        just flagged. Only meaningful when ticker itself is an AI-infra
        name (only then does its own size actually drive the total) — None
        otherwise, since capping an unrelated ticker wouldn't address an
        AI-infra breach caused entirely by other, pre-existing positions.
        """
        try:
            positions = self._client.get_all_positions()
            pv = self.portfolio_value()
        except Exception as e:
            log.error(f"Concentration check ERRORED for {ticker}: {e}")
            return {"passed": False,
                    "flags": [f"RISK CHECK FAILED — could not verify concentration: {e}. "
                              "Verify manually before approving."],
                    "ai_infrastructure_total": 0,
                    "ai_infra_size_cap_pct": None}

        flags: list[str] = []

        # Current position weights
        pos_values: dict[str, float] = {}
        for p in positions:
            pct = float(p.market_value) / pv if pv else 0
            pos_values[p.symbol] = pct

        # Add proposed
        pos_values[ticker] = pos_values.get(ticker, 0) + proposed_size_pct

        # AI infrastructure tickers (approximate)
        ai_infra_keywords = {"NVDA", "AMD", "ARM", "INTC", "MSFT", "AMZN", "GOOG", "GOOGL",
                             "META", "AVGO", "ANET", "CRDO", "MRVL", "ASML", "SNPS", "KLAC",
                             "LRCX", "AMAT", "MU", "CRWV", "ORCL", "CLS", "PLTR", "SNOW",
                             "DDOG", "VRT", "PWR", "GEV", "CEG"}
        ai_infra_total = sum(v for t, v in pos_values.items() if t in ai_infra_keywords)

        ai_infra_size_cap_pct = None
        if ticker in ai_infra_keywords:
            ai_infra_total_without_this = ai_infra_total - proposed_size_pct
            ai_infra_size_cap_pct = max(0.0, MAX_AI_INFRA_PCT - ai_infra_total_without_this)

        if ai_infra_total > MAX_AI_INFRA_PCT:
            flags.append(f"AI infrastructure exposure {ai_infra_total:.1%} exceeds {MAX_AI_INFRA_PCT:.0%} limit")

        if pos_values.get(ticker, 0) > MAX_POSITION_PCT_COMPOUNDER:
            flags.append(f"{ticker} position {pos_values[ticker]:.1%} exceeds Compounder cap {MAX_POSITION_PCT_COMPOUNDER:.0%}")

        # Theme concentration — positions sharing the proposed ticker's sector.
        # theme_size_cap_pct mirrors ai_infra_size_cap_pct's shape below --
        # found 2026-08-18 that this breach, like moonshot's, was flagged in
        # text but never actually capped (only AI-infra got that fix on
        # 2026-08-11). Only meaningful for a ticker that's actually in the
        # breaching sector; None otherwise.
        theme_size_cap_pct = None
        try:
            from knowledge_loader import load_watchlist
            watchlist = load_watchlist()
            sector = watchlist.get(ticker, {}).get("sector", "")
            if sector:
                theme_total = sum(
                    v for t, v in pos_values.items()
                    if watchlist.get(t, {}).get("sector", "") == sector
                )
                theme_total_without_this = theme_total - proposed_size_pct
                theme_size_cap_pct = max(0.0, MAX_SINGLE_THEME_PCT - theme_total_without_this)
                if theme_total > MAX_SINGLE_THEME_PCT:
                    flags.append(f"'{sector}' theme exposure {theme_total:.1%} exceeds "
                                 f"{MAX_SINGLE_THEME_PCT:.0%} limit")
        except Exception as e:
            log.warning(f"Theme concentration sub-check unavailable: {e}")

        # Total moonshot exposure — RGTI incident 2026-08-18: flagged
        # correctly ("13.7% exceeds 10% limit") but nothing capped it,
        # since only ai_infra_size_cap_pct existed at the time. moonshot_
        # size_cap_pct only computed when bucket=="Moonshot", since that's
        # the only case this ticker's own size drives the total.
        moonshot_size_cap_pct = None
        if bucket == "Moonshot":
            try:
                from knowledge_loader import load_watchlist
                watchlist = load_watchlist()
                moonshot_total = sum(
                    v for t, v in pos_values.items()
                    if watchlist.get(t, {}).get("role", "") == "Moonshot"
                )
                moonshot_total_without_this = moonshot_total - proposed_size_pct
                moonshot_size_cap_pct = max(0.0, MAX_MOONSHOT_TOTAL_PCT - moonshot_total_without_this)
                if moonshot_total > MAX_MOONSHOT_TOTAL_PCT:
                    flags.append(f"Total moonshot exposure {moonshot_total:.1%} exceeds "
                                 f"{MAX_MOONSHOT_TOTAL_PCT:.0%} limit")
            except Exception as e:
                log.warning(f"Moonshot concentration sub-check unavailable: {e}")

        return {
            "passed": len(flags) == 0,
            "flags":  flags,
            "ai_infrastructure_total": round(ai_infra_total, 4),
            "ai_infra_size_cap_pct": (round(ai_infra_size_cap_pct, 4)
                                      if ai_infra_size_cap_pct is not None else None),
            "theme_size_cap_pct": (round(theme_size_cap_pct, 4)
                                    if theme_size_cap_pct is not None else None),
            "moonshot_size_cap_pct": (round(moonshot_size_cap_pct, 4)
                                       if moonshot_size_cap_pct is not None else None),
        }

    def check_correlation(self, ticker: str, corr_threshold: float = 0.7,
                          min_history_days: int = 30) -> dict:
        """
        Real return-series correlation against currently held positions, using
        the price_history cache (ingestion/price_history.py, refreshed
        nightly). Previously this only counted positions sharing the same
        watchlist sector STRING — no actual price relationship was ever
        computed, so e.g. two stocks in different "sectors" that move in
        lockstep were invisible to it.

        Falls back to the old sector-heuristic when cached price history isn't
        available yet (e.g. immediately after a fresh install, before the
        first nightly refresh has run) so this never breaks sizing outright.
        """
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            log.error(f"Correlation check ERRORED for {ticker}: {e}")
            return {"shared_catalyst_count": 0,
                    "note": f"RISK CHECK FAILED — correlation unverifiable ({e}); "
                            "size reduced 25% as a precaution",
                    "reduce_by": 0.25}

        held_symbols = [p.symbol for p in positions]
        if not held_symbols:
            return {"shared_catalyst_count": 0, "note": "No open positions", "reduce_by": 0}

        try:
            import db
            bulk = db.get_price_history_bulk([ticker] + held_symbols, days=90)
        except Exception as e:
            log.warning(f"Price history unavailable for correlation check: {e}")
            bulk = {}

        if ticker not in bulk or len(bulk[ticker]) < min_history_days:
            return self._check_correlation_sector_fallback(ticker, positions)

        correlated: list[tuple[str, float]] = []
        for sym in held_symbols:
            if sym not in bulk or len(bulk[sym]) < min_history_days:
                continue
            corr = _price_correlation(bulk[ticker], bulk[sym])
            if corr is not None and corr >= corr_threshold:
                correlated.append((sym, round(corr, 2)))

        reduce_by = 0.25 if len(correlated) >= 2 else 0
        note = (
            f"Return-correlated (r>={corr_threshold}) with {len(correlated)} held "
            f"position(s): {correlated}" if correlated
            else f"No high return-correlation (r>={corr_threshold}) with current holdings"
        )
        return {
            "shared_catalyst_count": len(correlated),
            "note": note,
            "reduce_by": reduce_by,
            "correlations": dict(correlated),
        }

    def _check_correlation_sector_fallback(self, ticker: str, positions: list) -> dict:
        """Sector-string fallback used only when price history isn't cached yet."""
        try:
            from knowledge_loader import load_watchlist
            watchlist = load_watchlist()
        except Exception as e:
            return {"shared_catalyst_count": 0,
                    "note": f"Correlation check skipped: {e}", "reduce_by": 0}

        ticker_sector = watchlist.get(ticker, {}).get("sector", "")
        shared = sum(
            1 for p in positions
            if watchlist.get(p.symbol, {}).get("sector", "") == ticker_sector
        )
        reduce_by = 0.25 if shared >= 2 else 0
        return {
            "shared_catalyst_count": shared,
            "note": (f"[price history not yet cached — sector fallback] "
                     f"{shared} existing positions in {ticker_sector}") if ticker_sector
                    else "Sector unknown",
            "reduce_by": reduce_by,
        }

    def _latest_conviction(self, ticker: str) -> int | None:
        """Freshest available conviction reading for a held position — the
        most recent analyst signal (logged on every review, including
        routine HOLDs, see db.get_latest_signal_for's docstring) if one
        exists, otherwise the conviction the position was last entered/
        added at. Returns None only if neither exists (shouldn't happen
        for a genuinely held position)."""
        import db
        signal = db.get_latest_signal_for(ticker)
        if signal is not None:
            return round(signal["confidence"] * 10)
        stop_rec = db.get_position_stop(ticker)
        if stop_rec and stop_rec.get("conviction") is not None:
            return int(stop_rec["conviction"])
        return None

    def find_swap_candidate(self, new_ticker: str, new_conviction: int) -> dict | None:
        """
        Added 2026-08-13, ahead of a live-money switch: when cash is too
        tight to size a high-conviction new idea, the old behavior was to
        just cap or block it outright, even if the portfolio was sitting on
        a position nobody's excited about anymore. This looks for an
        existing Accelerator/Moonshot holding that has independently fallen
        weak enough to be worth exiting anyway, so the new idea can be
        funded by rotation instead of by force.

        Deliberately opportunistic, never forced — by explicit design
        decision (2026-08-13), not just an implementation detail:
          - Compounder-bucket holdings (the long-term core) are never
            eligible, regardless of conviction gap.
          - A candidate needs BOTH new_conviction - candidate_conviction >=
            SWAP_MIN_CONVICTION_GAP AND the candidate's own fresh conviction
            <= SWAP_MAX_CANDIDATE_CONVICTION. The second condition is the
            one that actually matters: a held position at 7+/10 is never
            touched by this no matter how much higher the new idea scores,
            because it hasn't independently earned an exit — this isn't a
            relative ranking that always finds a "weakest" position to
            sacrifice, it's a check for whether anything already looks like
            it deserves to go.
          - If multiple positions qualify, prefers realizing a LOSS (or the
            smallest gain) over a bigger gain — this is a real, deliberate
            tax consideration, not an afterthought: in a taxable account,
            selling a position that's up triggers a capital gain (almost
            certainly short-term given how this system holds positions,
            taxed at ordinary income rates, not the lower long-term rate).
            Preferring losses here both minimizes that drag and tends to
            align with "this one probably isn't working out anyway."
          - Returns None if nothing qualifies — callers must fall through
            to the existing cash-cap/block behavior unchanged, never invent
            a sale just because cash is tight.

        Returns {"ticker", "conviction", "qty", "market_value",
        "unrealized_pl"} for the single best candidate, or None.
        """
        import db
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            log.error(f"find_swap_candidate: could not fetch positions: {e}")
            return None

        candidates = []
        for p in positions:
            ticker = p.symbol
            if ticker == new_ticker:
                continue
            stop_rec = db.get_position_stop(ticker)
            bucket = (stop_rec or {}).get("thesis_bucket")
            if bucket not in ("Accelerator", "Moonshot"):
                continue
            conviction = self._latest_conviction(ticker)
            if conviction is None:
                continue
            if new_conviction - conviction < SWAP_MIN_CONVICTION_GAP:
                continue
            if conviction > SWAP_MAX_CANDIDATE_CONVICTION:
                continue
            try:
                candidates.append({
                    "ticker":         ticker,
                    "conviction":     conviction,
                    "qty":            abs(float(p.qty)),
                    "market_value":   float(p.market_value),
                    "unrealized_pl":  float(p.unrealized_pl),
                })
            except (TypeError, ValueError) as e:
                log.warning(f"[{ticker}] find_swap_candidate: skipping, bad position data: {e}")

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c["unrealized_pl"], c["conviction"]))
        return candidates[0]

    def shares_after_concentration(self, ticker: str, sizing: dict, bucket: str) -> int:
        """shares_initial after the concentration caps, checked at the larger
        of target and order weight (same rule as crew._route_to_approval)."""
        shares = int(sizing.get("shares_initial", 0) or 0)
        if shares <= 0:
            return 0
        proposed = max(sizing.get("position_size_pct") or 0, sizing.get("order_pct") or 0)
        conc = self.check_concentration(ticker, proposed, bucket=bucket)
        if conc.get("passed", False):
            return shares
        caps = [c for c in (conc.get("ai_infra_size_cap_pct"), conc.get("theme_size_cap_pct"),
                            conc.get("moonshot_size_cap_pct")) if c is not None]
        cap = min(caps) if caps else None
        if cap is None or cap >= proposed:
            return shares
        base = (sizing.get("order_pct") or 0) or proposed
        return min(shares, int(shares * cap / base + 1e-9)) if base > 0 else 0

    def would_size_after_swap(self, ticker: str, conviction: int, bucket: str,
                              regime: str, price: float, swap: dict) -> int:
        """Shares the new buy would get if `swap` were sold first: sizing run
        against the account with the swap's market value added to cash and
        its position removed. crew used to sell first and re-size after, so a
        buy that still sized to 0 left the holding sold and nothing bought
        (limitations audit 2026-10-28 09:00, finding 9). Returns 0 on error."""
        from types import SimpleNamespace
        real = self._client
        try:
            acct = real.get_account()
            freed = float(swap.get("market_value") or 0)
            view_acct = SimpleNamespace(portfolio_value=float(acct.portfolio_value),
                                        cash=float(acct.cash) + freed,
                                        buying_power=float(acct.buying_power) + freed)
            remaining = [p for p in real.get_all_positions() if p.symbol != swap.get("ticker")]
        except Exception as e:
            log.warning(f"[{ticker}] swap pre-check could not read the account: {e}")
            return 0

        class _View:
            def get_account(self):
                return view_acct

            def get_all_positions(self):
                return remaining

            def get_open_position(self, symbol):
                for p in remaining:
                    if p.symbol == symbol:
                        return p
                raise Exception(f"No open position in {symbol}")

        try:
            sim = RiskManager(_View(), start_value=self.start_value)
            sizing = sim.size_position(ticker, conviction, bucket, regime, price)
            # Same concentration cap crew applies after sizing (professor M36):
            # a swap whose buy the caps cut to 0 must not sell the holding.
            return sim.shares_after_concentration(ticker, sizing, bucket)
        except Exception as e:
            log.warning(f"[{ticker}] swap pre-check sizing failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Backward-compatible methods (used by remaining code)
    # ------------------------------------------------------------------

    def current_position_value(self, symbol: str) -> float:
        try:
            position = self._client.get_open_position(symbol)
            return float(position.market_value)
        except Exception:
            return 0.0

    def shares_affordable(self, symbol: str, price: float) -> int:
        """Legacy method — uses flat 2%/20% caps. Use size_position() for new code."""
        if self.is_halted():
            return 0
        pv = self.portfolio_value()
        trade_cap    = pv * 0.02
        position_cap = pv * 0.20
        held         = self.current_position_value(symbol)
        budget       = min(trade_cap, max(0, position_cap - held))
        return round(budget / price, 4) if price else 0.0

    def check_sell_allowed(self, symbol: str, shares: float) -> bool:
        try:
            position = self._client.get_open_position(symbol)
            owned = float(position.qty)
            if shares > owned:
                log.warning(f"Sell rejected — only {owned} shares of {symbol} owned")
                return False
            return True
        except Exception:
            log.warning(f"Sell rejected — no position in {symbol}")
            return False

    # ------------------------------------------------------------------
    # Trailing stop monitor
    # ------------------------------------------------------------------

    def check_trailing_stops(self) -> list[dict]:
        """
        Review all open positions against their logged stops.
        Returns list of {ticker, action, current_price, stop_price, reason}
        for any positions that should be exited or have trailing stop updated.
        """
        try:
            import db
            stops    = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"Trailing stop check failed: {e}")
            return []

        alerts: list[dict] = []
        for stop_record in stops:
            ticker = stop_record["ticker"]
            pos    = positions.get(ticker)
            if not pos:
                continue

            current = float(pos.current_price)
            stop_px = stop_record["stop_price"]
            bucket  = stop_record.get("thesis_bucket", "Accelerator")

            # Check if stop hit
            if current <= stop_px:
                alerts.append({
                    "ticker":        ticker,
                    "action":        "EXIT",
                    "current_price": current,
                    "stop_price":    stop_px,
                    "reason":        f"Stop triggered at ${stop_px:.2f} (current ${current:.2f})",
                })
                continue

            # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_
            # GATES_ENABLED, a position carrying a live setup category uses
            # the ladder-rung ratchet below INSTEAD OF the fixed ATR-
            # multiple trail-activation logic that follows — skip it
            # entirely for this ticker rather than letting both paths touch
            # the same stop in the same run.
            if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
                self._ratchet_ladder_stop(ticker, pos, current, stop_px, alerts)
                continue

            # Check if trailing stop should be activated
            entry    = stop_record.get("entry_price", current)
            gain_pct = (current - entry) / entry if entry else 0
            activate_at = TRAILING_STOP_ACTIVATE.get(bucket)
            trail_lo, trail_hi = TRAILING_STOP_DISTANCE.get(bucket, (None, None))

            if activate_at and gain_pct >= activate_at and trail_lo:
                trail_dist = (trail_lo + trail_hi) / 2
                already_trailing = bool(stop_record.get("trailing_active"))

                if not already_trailing:
                    # ONE-TIME conversion to a native IBKR TRAIL order. After
                    # this, the broker ratchets the stop itself on every tick
                    # — no more cancel/replace round-trips, and no window
                    # where the position is briefly unprotected between
                    # cancelling the old stop and placing the new one.
                    #
                    # Trail distance is ATR-based (N x ATR, same multiplier
                    # table as the initial stop) rather than a flat percent —
                    # a flat % gives every stock in a bucket the same trail
                    # regardless of how it individually trades, the same
                    # one-size-fits-all problem the initial stop already
                    # solved by using ATR instead of a flat percent. Falls
                    # back to the flat-% distance only if ATR is unavailable.
                    atr = self.get_atr(ticker)
                    trail_amount = None
                    trail_pct = None
                    if atr:
                        mult_lo, mult_hi = ATR_STOP_MULTIPLIER.get(bucket, (1.5, 2.0))
                        mult = (mult_lo + mult_hi) / 2
                        trail_amount = round(mult * atr, 2)
                        new_stop = round(current - trail_amount, 2)
                        method_label = f"{mult:.1f}x ATR (${atr:.2f}) = ${trail_amount:.2f}"
                    else:
                        trail_pct = round(trail_dist * 100, 2)
                        new_stop = round(current * (1 - trail_dist), 2)
                        method_label = f"{trail_pct}% (ATR unavailable, flat-% fallback)"

                    # A re-arm after a partial sale or TRIM starts the trail
                    # from today's price; don't convert while that would sit
                    # below the fixed stop already in place (audit 12:53 #3).
                    if stop_px and new_stop < float(stop_px):
                        log.info(f"[{ticker}] Trail not armed yet: ${new_stop:.2f} would be below "
                                 f"the current stop ${float(stop_px):.2f}")
                        continue

                    try:
                        import tasks
                        qty = round(abs(float(pos.qty)), 4)
                        tasks.manage_cancel_stop_orders_for(ticker)
                        if trail_amount is not None:
                            result = tasks.manage_place_trailing_stop(ticker, qty, trail_amount=trail_amount)
                        else:
                            result = tasks.manage_place_trailing_stop(ticker, qty, trail_percent=trail_pct)
                        db.set_stop_broker_order_id(ticker, result["order_id"])
                        db.update_trailing_stop(ticker, new_stop, current)
                        log.info(f"[{ticker}] Converted to native TRAIL stop ({method_label}), "
                                f"orderId={result['order_id']}")
                    except Exception as e:
                        log.error(f"[{ticker}] Native TRAIL conversion failed — falling back to "
                                 f"one-shot cancel/replace: {e}")
                        prev_high = stop_record.get("trailing_high") or entry
                        if current > prev_high and new_stop > stop_px:
                            try:
                                import tasks
                                tasks.manage_cancel_stop_orders_for(ticker)
                                qty = round(abs(float(pos.qty)), 4)
                                result = tasks.manage_stop_place(ticker, qty, new_stop)
                                db.set_stop_broker_order_id(ticker, result["order_id"])
                                db.update_trailing_stop(ticker, new_stop, current)
                                log.info(f"[{ticker}] Broker stop moved to ${new_stop:.2f} (fallback path)")
                            except Exception as e2:
                                log.error(f"[{ticker}] Fallback stop move also failed: {e2}")
                                alerts.append({
                                    "ticker":        ticker,
                                    "action":        "FIX_STOP",
                                    "current_price": current,
                                    "stop_price":    new_stop,
                                    "reason":        f"Trailing stop should be ${new_stop:.2f} but "
                                                     f"broker order update FAILED — fix manually in TWS",
                                })
                else:
                    # Already converted to a native TRAIL order — the broker
                    # is ratcheting it automatically. Just track the new high
                    # in DB for dashboard visibility; no order action needed.
                    prev_high = stop_record.get("trailing_high") or entry
                    if current > prev_high:
                        try:
                            db.update_trailing_stop(ticker, stop_px, current)
                        except Exception as e:
                            log.warning(f"Failed to update trailing high for {ticker}: {e}")

        return alerts

    def _ratchet_ladder_stop(self, ticker: str, pos: object, current: float,
                              stop_px: float, alerts: list[dict]) -> None:
        """
        SETUP_GATES_ENABLED trailing-stop path (2026-10-11): ratchets the
        stop to the highest ladder rung (agents.trade_ladder's nightly
        rungs, confluence_score >= 1 -- i.e. every real rung, single-type
        levels included, not just multi-confluence clusters) strictly
        below the current price, refreshed from whatever the ladder's
        latest nightly build computed -- instead of a fixed ATR-multiple
        trail distance. Ratchet-only: never moves the stop down, same
        guarantee the legacy ATR-trail path above already provides (a new
        candidate rung below the existing stop is simply ignored).
        """
        import db
        ladder = db.get_trade_ladder(ticker)
        if not ladder:
            return
        candidates = [
            r["price"] for r in (ladder.get("rungs") or [])
            if r.get("confluence_score", 0) >= 1 and r["price"] < current
        ]
        if not candidates:
            return
        new_stop = max(candidates)
        if new_stop <= stop_px:
            return  # not an improvement -- never move down

        try:
            import tasks
            qty = round(abs(float(pos.qty)), 4)
            tasks.manage_cancel_stop_orders_for(ticker)
            result = tasks.manage_stop_place(ticker, qty, new_stop)
            db.set_stop_broker_order_id(ticker, result["order_id"])
            db.update_trailing_stop(ticker, new_stop, current)
            log.info(f"[{ticker}] Ladder trailing stop ratcheted ${stop_px:.2f} -> ${new_stop:.2f}")
        except Exception as e:
            log.error(f"[{ticker}] Ladder trailing stop ratchet FAILED: {e}")
            alerts.append({
                "ticker":        ticker,
                "action":        "FIX_STOP",
                "current_price": current,
                "stop_price":    new_stop,
                "reason":        f"Ladder trailing stop should be ${new_stop:.2f} but broker "
                                 f"order update FAILED — fix manually in TWS",
            })

    def _breakeven_without_sale(self, ticker: str, qty: float, new_stop: float) -> bool:
        """Profit target hit on a position too small to sell part of (1-3
        shares x 25-45% rounds to 0). Before 2026-10-28 that just continued,
        and the breakeven stop move lived inside the partial sale, so it never
        happened: a 1-share position could rise past target, round-trip and
        stop out at a loss (limitations audit 07:00, finding 3). Now the stop
        moves to breakeven for the whole position and the target is marked
        taken. Cancel first, then place; if the new stop fails, the old one
        goes back on. Returns True when the target is marked taken."""
        import alerts
        import db
        import tasks

        stop_rec = db.get_position_stop(ticker) or {}
        old_stop = stop_rec.get("stop_price")
        if old_stop is not None and float(old_stop) >= new_stop:
            db.mark_profit_target_taken(ticker, float(old_stop))
            return True
        try:
            tasks.manage_cancel_stop_orders_for(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Breakeven move: could not cancel broker stop: {e}")
            return False
        try:
            result = tasks.manage_stop_place(ticker, round(qty, 4), new_stop)
        except Exception as e:
            log.error(f"[{ticker}] Breakeven stop placement failed: {e}")
            restored = False
            if old_stop:
                try:
                    result = tasks.manage_stop_place(ticker, round(qty, 4), float(old_stop))
                    db.set_stop_broker_order_id(ticker, result["order_id"])
                    restored = True
                except Exception as e2:
                    log.error(f"[{ticker}] Could not restore stop: {e2}")
            if not restored:
                alerts._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\n"
                    f"Breakeven stop move failed and the old stop could not be restored. "
                    f"Place a stop for {round(qty, 4)} shares manually."
                )
            return False
        db.set_stop_broker_order_id(ticker, result["order_id"])
        db.mark_profit_target_taken(ticker, new_stop)
        log.info(f"[{ticker}] Profit target hit on {qty:g} shares (too few to sell part): "
                 f"stop moved to breakeven ${new_stop:.2f}, no sale")
        return True

    def _partial_sell_and_resize_stop(self, ticker: str, qty: float, sell_qty: float,
                                      new_stop: float,
                                      decision_price: float | None = None) -> dict | None:
        """Profit-target partial sale with the broker stop kept in step.

        Before 2026-10-27 both profit-target paths sold part of the position and
        only wrote the breakeven stop to the DB. The broker STP stayed at the old
        price for the ORIGINAL quantity, so it could oversell into a short, and
        the 30-min backstop saw the size mismatch as uncovered. Same pattern as
        execution.py's TRIM path: cancel the stop, sell, then place a stop for
        what's left. If the sale fails, the old stop goes back on.
        Returns the order result, or None if the sale failed."""
        import alerts
        import db
        import ibkr
        import tasks

        stop_rec = db.get_position_stop(ticker) or {}
        old_stop = stop_rec.get("stop_price")
        # Never move the stop down: after a trail conversion the recorded stop
        # can already be above breakeven (limitations audit 12:53, finding 3).
        if old_stop is not None:
            new_stop = max(new_stop, float(old_stop))
        try:
            tasks.manage_cancel_stop_orders_for(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Profit-target: could not cancel broker stop, not selling: {e}")
            return None

        try:
            order_result = ibkr.place_market_order(ticker, "SELL", sell_qty)
        except Exception as e:
            log.error(f"[{ticker}] Profit-target partial sell FAILED: {e}")
            if old_stop:
                try:
                    result = tasks.manage_stop_place(ticker, round(qty, 4), float(old_stop))
                    db.set_stop_broker_order_id(ticker, result["order_id"])
                except Exception as e2:
                    log.error(f"[{ticker}] Could not restore stop after failed sale: {e2}")
                    alerts._send(
                        f"🚨 UNPROTECTED POSITION — {ticker}\n"
                        f"Profit-target sale failed and the stop could not be restored. "
                        f"Place stop @ ${float(old_stop):.2f} for {round(qty, 4)} shares manually."
                    )
            return None

        if float(order_result.get("filled_qty") or 0) <= 0:
            # Not restored automatically: a late fill plus a full-size stop
            # could short. Not marked taken, so the target can fire again.
            log.error(f"[{ticker}] Profit-target sale not confirmed filled: {order_result}")
            # Logged unconfirmed so reconcile_unconfirmed_orders sees a late
            # fill and re-protects the rest (professor C12, 2026-10-28).
            try:
                db.log_order(ticker=ticker, side="SELL", order_type="MARKET", qty=sell_qty,
                             broker_order_id=order_result.get("order_id"),
                             decision_price=decision_price, status="unconfirmed")
            except Exception as e:
                log.error(f"[{ticker}] Could not log the unconfirmed profit-target sale: {e}")
            alerts._send(
                f"⚠️ PROFIT-TARGET SALE UNCONFIRMED — {ticker}\n"
                f"SELL {sell_qty} submitted but no fill confirmed. The broker stop was "
                f"cancelled and NOT re-placed. Check the position in TWS."
            )
            return None

        # Record the sale like every other fill (audit 2026-10-28 10:53 #3):
        # before this, partial sales never reached trades or orders.
        filled = float(order_result.get("filled_qty") or 0)
        fill_price = order_result.get("fill_price") or decision_price or 0.0
        try:
            db.log_trade(symbol=ticker, side="SELL", quantity=filled, price=fill_price,
                         reason="Profit target partial sell")
            local_id = db.log_order(ticker=ticker, side="SELL", order_type="MARKET",
                                    qty=sell_qty, broker_order_id=order_result.get("order_id"),
                                    decision_price=decision_price, status="submitted")
            if local_id and local_id > 0 and order_result.get("fill_price"):
                db.update_order_fill(local_id, order_result["fill_price"], filled, "filled")
        except Exception as e:
            log.warning(f"[{ticker}] Profit-target sale logging failed: {e}")

        try:
            remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
        except Exception:
            remaining = round(qty - float(order_result.get("filled_qty") or sell_qty), 4)
        if remaining > 0:
            try:
                result = tasks.manage_stop_place(ticker, remaining, new_stop)
                db.set_stop_broker_order_id(ticker, result["order_id"])
                log.info(f"[{ticker}] Stop re-placed @ ${new_stop:.2f} for remaining {remaining} shares")
            except Exception as e:
                log.error(f"[{ticker}] Failed to re-place stop after profit-target sale: {e}")
                alerts._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\n"
                    f"Profit-target sale filled but stop re-placement failed. "
                    f"Place stop @ ${new_stop:.2f} for {remaining} shares manually."
                )
        # The stop actually used (may be above breakeven).
        return {**order_result, "new_stop": new_stop}

    def check_profit_targets(self) -> list[dict]:
        """
        Regime-adaptive profit-taking for Accelerator/Moonshot positions
        (2026-08-29, user request) -- the piece missing from the
        protective-only stop/trailing-stop path above: nothing there ever
        proactively banks a gain, it only ever reacts to price falling
        back. See config.py's PROFIT_TARGET_PCT_BY_STYLE docstring for
        the full rationale (a concrete Q2->Q3 regime-shift example where
        the same buy-and-hold posture rode gains up and gave them back).

        For each Accelerator/Moonshot position that hasn't already taken
        its target: reads that TICKER'S OWN sector trade_style (2026-08-29
        -- agents.market_regime.get_trade_style_for_ticker(), falling back
        to the whole-market get_cached_regime()["trade_style"] if the
        ticker isn't a current S&P 500 constituent or its sector hasn't
        been assessed yet; falls back further to the more conservative
        "Choppy/Distribution" table if no regime data is available at
        all, rather than silently skipping every position). A healthcare
        holding is evaluated against healthcare's own regime, not
        whatever the broad market or an unrelated sector happens to be
        doing -- see agents.market_regime.get_trade_style_for_ticker's
        docstring for why that distinction matters (a sector can trend
        while the broad market chops, or vice versa). Computes that
        style+bucket's profit target (midpoint of the configured range,
        same convention check_trailing_stops uses for ATR_STOP_MULTIPLIER
        above), and if unrealized gain has crossed it, sells
        PARTIAL_SELL_PCT_BY_STYLE of the position and moves the
        remainder's stop to breakeven in one atomic DB write
        (db.mark_profit_target_taken) -- so a crash between the sell and
        the stop update can't leave the remainder both unmarked (re-sold
        again next run) and unprotected (no breakeven floor recorded).

        Compounder positions are never touched -- PROFIT_TARGET_PCT_BY_STYLE
        has no Compounder entry at all, so this loop skips them by
        construction, not a special-cased check.

        Returns [{"ticker", "action": "PARTIAL_SELL", "shares_sold",
        "new_stop", "trade_style", "gain_pct", "order_result"}] for every
        position actually acted on this run.
        """
        import db

        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
            market_trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"check_profit_targets: market regime lookup failed, defaulting cautious: {e}")
            market_trade_style = None
        if market_trade_style not in PROFIT_TARGET_PCT_BY_STYLE:
            market_trade_style = "Choppy/Distribution"

        def _trade_style_for(ticker: str) -> str:
            try:
                from agents.market_regime import get_trade_style_for_ticker
                style = get_trade_style_for_ticker(ticker)
            except Exception as e:
                log.debug(f"[{ticker}] sector trade_style lookup failed: {e}")
                style = None
            return style if style in PROFIT_TARGET_PCT_BY_STYLE else market_trade_style

        try:
            stops = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"check_profit_targets: could not fetch positions: {e}")
            return []

        results: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
            if stop_record.get("profit_target_taken"):
                continue

            # Moved up from below the old bucket-table lookup (unchanged
            # computation, just earlier) so the SETUP_GATES_ENABLED branch
            # immediately below can use pos/entry/current/gain_pct too —
            # behaviorally identical for every legacy bucket, which simply
            # falls through to the unchanged code further down.
            pos = positions.get(ticker)
            entry = stop_record.get("entry_price")
            if not pos or not entry:
                continue
            current = float(pos.current_price)
            gain_pct = (current - entry) / entry

            # Swing-trading cutover, Phase 4/5 (2026-10-11): a position
            # carrying a live setup category reads its scale-out schedule
            # from the gate/ladder instead of the flat PROFIT_TARGET_PCT_
            # BY_STYLE table — handled entirely, never falls through to the
            # legacy table-driven logic below.
            if SETUP_GATES_ENABLED and bucket in CATEGORY_RISK_PCT_PER_TRADE:
                trade_style = _trade_style_for(ticker)
                partial_pct = PARTIAL_SELL_PCT_BY_STYLE[trade_style]
                self._check_profit_target_setup_gates(
                    ticker, bucket, entry, current, gain_pct, partial_pct, trade_style, results
                )
                continue

            trade_style = _trade_style_for(ticker)
            targets_by_bucket = PROFIT_TARGET_PCT_BY_STYLE[trade_style]
            if bucket not in targets_by_bucket:
                continue  # Compounder, or any bucket not in this framework
            partial_pct = PARTIAL_SELL_PCT_BY_STYLE[trade_style]

            target_lo, target_hi = targets_by_bucket[bucket]
            target = (target_lo + target_hi) / 2
            if gain_pct < target:
                continue

            qty = abs(float(pos.qty))
            sell_qty = int(qty * partial_pct)  # whole shares; IBKR rejects fractional orders
            if sell_qty <= 0:
                if self._breakeven_without_sale(ticker, qty, entry):
                    results.append({"ticker": ticker, "action": "BREAKEVEN_STOP", "shares_sold": 0,
                                    "new_stop": entry, "trade_style": trade_style,
                                    "gain_pct": gain_pct})
                continue

            new_stop = entry  # breakeven on the remainder
            order_result = self._partial_sell_and_resize_stop(ticker, qty, sell_qty, new_stop,
                                                              decision_price=current)
            if order_result is None:
                continue
            new_stop = order_result.get("new_stop", new_stop)
            db.mark_profit_target_taken(ticker, new_stop)
            log.info(f"[{ticker}] Profit target hit ({gain_pct:.1%} >= {target:.1%}, "
                     f"style={trade_style}): sold {sell_qty} shares, "
                     f"remainder stop -> breakeven ${new_stop:.2f}")
            results.append({
                "ticker": ticker, "action": "PARTIAL_SELL", "shares_sold": sell_qty,
                "new_stop": new_stop, "trade_style": trade_style, "gain_pct": gain_pct,
                "order_result": order_result,
            })

        return results

    def _check_profit_target_setup_gates(
        self, ticker: str, bucket: str, entry: float, current: float, gain_pct: float,
        partial_pct: float, trade_style: str, results: list[dict],
    ) -> bool:
        """
        SETUP_GATES_ENABLED profit-taking path (2026-10-11): the scale-out
        schedule comes from the live setup-gate's own target_prices
        (agents/setup_gates.py's classify_setup, which projects them off
        agents.trade_ladder's rungs at classification time) instead of the
        flat PROFIT_TARGET_PCT_BY_STYLE %-of-gain table. Falls back to the
        nearest ladder rungs above entry if the ticker's live category no
        longer matches this position's stored bucket (e.g. the setup has
        since stopped qualifying) rather than skipping target-taking
        outright.

        Layers the EXISTING valuation-tier override on top UNCONDITIONALLY:
        an "Overextended" fair-value read (agents/valuation.py, via the
        same ladder) forces a partial sell even if price hasn't reached a
        ladder target yet -- same override tasks.py's digest code already
        surfaces for held positions, just now acted on here instead of
        only displayed.

        This method issues the SAME kind of real partial-sell market order
        the pre-existing (pre-cutover) check_profit_targets loop already
        places for Accelerator/Moonshot above -- it is NOT a new class of
        live-order behavior, just a different trigger (ladder target /
        valuation tier instead of the flat %-of-gain table) reached only
        when SETUP_GATES_ENABLED is explicitly set True. Never reached in
        this task's deliverable, which ships with that flag False.

        Returns True if a sell was actually executed this run.
        """
        import db

        ladder = db.get_trade_ladder(ticker)
        valuation_tier = ladder.get("valuation_tier") if ladder else None
        overextended = valuation_tier == "Overextended"

        best = None
        try:
            from agents.setup_gates import classify_setup
            best, _all_gates = classify_setup(ticker)
        except Exception as e:
            log.debug(f"[{ticker}] profit-target classify_setup failed: {e}")

        target_prices = (best.get("target_prices") or []) if (best and best.get("category") == bucket) else []
        if not target_prices and ladder:
            target_prices = sorted(
                r["price"] for r in (ladder.get("rungs") or []) if r["price"] > entry
            )[:3]

        nearest_target = min(target_prices) if target_prices else None
        target_hit = nearest_target is not None and current >= nearest_target

        if not (target_hit or overextended):
            return False

        try:
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"[{ticker}] setup-gate profit-target: could not fetch positions: {e}")
            return False
        pos = positions.get(ticker)
        if not pos:
            return False

        qty = abs(float(pos.qty))
        sell_qty = int(qty * partial_pct)  # whole shares; IBKR rejects fractional orders
        if sell_qty <= 0:
            return self._breakeven_without_sale(ticker, qty, entry)

        new_stop = entry  # breakeven on the remainder, same convention as the legacy path
        order_result = self._partial_sell_and_resize_stop(ticker, qty, sell_qty, new_stop,
                                                          decision_price=current)
        if order_result is None:
            return False
        new_stop = order_result.get("new_stop", new_stop)
        db.mark_profit_target_taken(ticker, new_stop)
        if overextended and target_hit:
            reason = f"ladder target ${nearest_target:.2f} hit + Overextended valuation"
        elif overextended:
            reason = "Overextended valuation override"
        else:
            reason = f"ladder target ${nearest_target:.2f} hit"
        log.info(f"[{ticker}] Setup-gate profit target ({reason}, gain {gain_pct:.1%}): "
                 f"sold {sell_qty} shares, remainder stop -> breakeven ${new_stop:.2f}")
        results.append({
            "ticker": ticker, "action": "PARTIAL_SELL", "shares_sold": sell_qty,
            "new_stop": new_stop, "trade_style": trade_style, "gain_pct": gain_pct,
            "setup_category": bucket, "reason": reason, "order_result": order_result,
        })
        return True

    def flag_stale_positions(self) -> list[dict]:
        """
        Identifies Accelerator/Moonshot positions that have gone nowhere
        (less than STALE_MOVE_THRESHOLD_PCT move, either direction) for
        longer than their style+bucket's stale window -- capital sitting
        idle with no momentum, independent of whether it happens to be up
        or down slightly. See config.py's STALE_WINDOW_DAYS_BY_STYLE
        docstring (calendar days, not trading days -- computed straight
        from position_stops.opened_at with no trading-calendar
        dependency).

        Deliberately does NOT exit anything itself -- returns the flagged
        list for the caller (tasks.review_positions, once wired) to bump
        to the FRONT of review_all_held_positions's oldest-reviewed-first
        rotation, so a stale position gets a fresh LLM look promptly
        instead of possibly waiting weeks behind other holdings in the
        queue. The actual TRIM/EXIT/HOLD call stays the LLM's -- same
        division of labor the rest of this system already uses
        (risk_manager computes hard numbers, the LLM makes the
        qualitative call).

        Returns [{"ticker", "days_held", "pct_move", "trade_style",
        "thesis_bucket"}] for every position past its stale window.
        """
        import db

        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
            market_trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"flag_stale_positions: market regime lookup failed, defaulting cautious: {e}")
            market_trade_style = None
        if market_trade_style not in STALE_WINDOW_DAYS_BY_STYLE:
            market_trade_style = "Choppy/Distribution"

        def _trade_style_for(ticker: str) -> str:
            try:
                from agents.market_regime import get_trade_style_for_ticker
                style = get_trade_style_for_ticker(ticker)
            except Exception as e:
                log.debug(f"[{ticker}] sector trade_style lookup failed: {e}")
                style = None
            return style if style in STALE_WINDOW_DAYS_BY_STYLE else market_trade_style

        try:
            stops = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"flag_stale_positions: could not fetch positions: {e}")
            return []

        now = datetime.now(timezone.utc)
        flagged: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
            trade_style = _trade_style_for(ticker)
            windows = STALE_WINDOW_DAYS_BY_STYLE[trade_style]
            if bucket not in windows:
                continue
            pos = positions.get(ticker)
            opened_at = stop_record.get("opened_at")
            entry = stop_record.get("entry_price")
            if not pos or not opened_at or not entry:
                continue

            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            days_held = (now - opened_at).days
            if days_held < windows[bucket]:
                continue

            pct_move = abs((float(pos.current_price) - entry) / entry)
            if pct_move >= STALE_MOVE_THRESHOLD_PCT:
                continue

            flagged.append({
                "ticker": ticker, "days_held": days_held, "pct_move": pct_move,
                "trade_style": trade_style, "thesis_bucket": bucket,
            })
            log.info(f"[{ticker}] Flagged stale: held {days_held}d (window {windows[bucket]}d), "
                     f"moved only {pct_move:.1%} (threshold {STALE_MOVE_THRESHOLD_PCT:.0%})")

        return flagged

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    rm = RiskManager()

    pv = rm.portfolio_value()
    print(f"Portfolio value: ${pv:,.2f}")
    print(f"Halted: {rm.is_halted()}")
    print()

    sizing = rm.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On", price=130.0)
    print("NVDA sizing (Compounder, conviction=8, Risk-On):")
    for k, v in sizing.items():
        print(f"  {k}: {v}")
