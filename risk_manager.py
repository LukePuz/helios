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
    )
except ImportError:
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
    MOONSHOT_MAX_LOSS_PCT         = 0.35
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
        its target: reads the CURRENT trade_style from
        agents.market_regime.get_cached_regime() (falls back to the more
        conservative "Choppy/Distribution" table if regime data isn't
        available or the packet is stale/missing, rather than silently
        skipping every position), computes that style+bucket's profit
        target (midpoint of the configured range, same convention
        check_trailing_stops uses for ATR_STOP_MULTIPLIER above), and if
        unrealized gain has crossed it, sells PARTIAL_SELL_PCT_BY_STYLE
        of the position and moves the remainder's stop to breakeven in
        one atomic DB write (db.mark_profit_target_taken) -- so a crash
        between the sell and the stop update can't leave the remainder
        both unmarked (re-sold again next run) and unprotected (no
        breakeven floor recorded).

        Compounder positions are never touched -- PROFIT_TARGET_PCT_BY_STYLE
        has no Compounder entry at all, so this loop skips them by
        construction, not a special-cased check.

        NOT wired into tasks.py's beat_schedule as of 2026-08-29 -- see
        config.py's PROFIT_TARGET_PCT_BY_STYLE docstring for why running
        this automatically against live capital is a deliberate, separate
        step held pending explicit confirmation.

        Returns [{"ticker", "action": "PARTIAL_SELL", "shares_sold",
        "new_stop", "trade_style", "gain_pct", "order_result"}] for every
        position actually acted on this run.
        """
        import db

        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
            trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"check_profit_targets: regime lookup failed, defaulting cautious: {e}")
            trade_style = None
        if trade_style not in PROFIT_TARGET_PCT_BY_STYLE:
            trade_style = "Choppy/Distribution"

        try:
            stops = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"check_profit_targets: could not fetch positions: {e}")
            return []

        targets_by_bucket = PROFIT_TARGET_PCT_BY_STYLE[trade_style]
        partial_pct = PARTIAL_SELL_PCT_BY_STYLE[trade_style]
        results: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
            if bucket not in targets_by_bucket or stop_record.get("profit_target_taken"):
                continue
            pos = positions.get(ticker)
            entry = stop_record.get("entry_price")
            if not pos or not entry:
                continue

            current = float(pos.current_price)
            gain_pct = (current - entry) / entry
            target_lo, target_hi = targets_by_bucket[bucket]
            target = (target_lo + target_hi) / 2
            if gain_pct < target:
                continue

            qty = abs(float(pos.qty))
            sell_qty = round(qty * partial_pct, 4)
            if sell_qty <= 0:
                continue

            try:
                import ibkr
                order_result = ibkr.place_market_order(ticker, "SELL", sell_qty)
            except Exception as e:
                log.error(f"[{ticker}] Profit-target partial sell FAILED: {e}")
                continue

            new_stop = entry  # breakeven on the remainder
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
            trade_style = regime_packet.get("trade_style") if regime_packet else None
        except Exception as e:
            log.warning(f"flag_stale_positions: regime lookup failed, defaulting cautious: {e}")
            trade_style = None
        if trade_style not in STALE_WINDOW_DAYS_BY_STYLE:
            trade_style = "Choppy/Distribution"

        try:
            stops = db.get_all_position_stops()
            positions = {p.symbol: p for p in self._client.get_all_positions()}
        except Exception as e:
            log.warning(f"flag_stale_positions: could not fetch positions: {e}")
            return []

        windows = STALE_WINDOW_DAYS_BY_STYLE[trade_style]
        now = datetime.now(timezone.utc)
        flagged: list[dict] = []

        for stop_record in stops:
            ticker = stop_record["ticker"]
            bucket = stop_record.get("thesis_bucket")
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
