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

import requests
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.models import TradeAccount

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
    )
except ImportError:
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


# ---------------------------------------------------------------------------
# Helpers: Alpaca bar fetch for ATR
# ---------------------------------------------------------------------------

def _fetch_bars_for_atr(ticker: str, days: int = 20) -> list[dict]:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key:
        return []

    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days + 5)).strftime("%Y-%m-%d")
    url   = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    hdrs  = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"timeframe": "1Day", "start": start, "end": end, "limit": days + 5, "feed": "iex"}
    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=10)
        resp.raise_for_status()
        return [{"h": b["h"], "l": b["l"], "c": b["c"]} for b in resp.json().get("bars", [])]
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
    def __init__(self, client: TradingClient, start_value: float | None = None):
        self._client = client
        account = self._get_account()
        self.start_value: float = start_value or float(account.portfolio_value)

    def _get_account(self) -> TradeAccount:
        return self._client.get_account()

    def portfolio_value(self) -> float:
        return float(self._get_account().portfolio_value)

    def buying_power(self) -> float:
        return float(self._get_account().buying_power)

    # ------------------------------------------------------------------
    # Drawdown halt
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        current  = self.portfolio_value()
        drawdown = (self.start_value - current) / self.start_value
        if drawdown >= DRAWDOWN_HALT_PCT:
            log.warning(
                f"DRAWDOWN HALT — drawdown {drawdown:.1%} exceeds {DRAWDOWN_HALT_PCT:.0%} "
                f"(start=${self.start_value:,.2f}, current=${current:,.2f})"
            )
            return True
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
                    "flags": ["HALTED — portfolio drawdown limit reached"]}

        pv    = self.portfolio_value()
        band  = _conviction_band(conviction)
        sizes = BUCKET_SIZE_TABLE.get(bucket, BUCKET_SIZE_TABLE["Accelerator"])
        lo, hi = sizes.get(band, (0.01, 0.02))

        if lo is None:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "flags": [f"Conviction {conviction} too low for {bucket} — WATCH only"]}

        # Midpoint of range, then apply regime multiplier
        base_pct  = (lo + hi) / 2
        mult      = REGIME_MULTIPLIER.get(regime, 0.75)
        target    = base_pct * mult

        # Enforce hard caps
        hard_cap = {
            "Compounder":  MAX_POSITION_PCT_COMPOUNDER,
            "Accelerator": MAX_POSITION_PCT_ACCELERATOR,
            "Moonshot":    MAX_POSITION_PCT_MOONSHOT,
        }.get(bucket, MAX_POSITION_PCT_ACCELERATOR)

        target = min(target, hard_cap)

        # Check if adding would exceed current position cap
        held = self.current_position_value(ticker)
        held_pct = held / pv if pv > 0 else 0
        remaining = max(0, target - held_pct)
        if remaining < 0.005:
            return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
                    "flags": [f"Position already at/near target ({held_pct:.1%} held)"]}

        dollar_amount  = pv * remaining
        initial_pct    = 0.55  # 55% initial tranche (midpoint of 50-60%)
        initial_dollar = dollar_amount * initial_pct
        shares_initial = max(1, int(initial_dollar // price))

        # Stop loss
        stop_price, stop_method = self.calculate_stop(ticker, price, bucket)

        # Trailing stop params
        trail_activate = TRAILING_STOP_ACTIVATE.get(bucket)
        trail_dist_lo, trail_dist_hi = TRAILING_STOP_DISTANCE.get(bucket, (None, None))
        trail_distance = (trail_dist_lo + trail_dist_hi) / 2 if trail_dist_lo else None

        # Max loss at stop
        initial_cost = shares_initial * price
        max_loss = (initial_cost - shares_initial * stop_price) if stop_price else initial_cost * 0.35

        flags = []
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
            # Fallback: use % of entry
            fallback_pct = 0.08 if bucket == "Compounder" else 0.05
            stop = round(entry_price * (1 - fallback_pct), 2)
            return stop, f"Fallback stop ({fallback_pct:.0%} below entry — ATR unavailable)"

        mult_lo, mult_hi = ATR_STOP_MULTIPLIER.get(bucket, (1.5, 2.0))
        mult = (mult_lo + mult_hi) / 2
        stop = round(entry_price - mult * atr, 2)
        return stop, f"{mult:.1f}× ATR ({atr:.2f}) below entry = ${stop:.2f}"

    # ------------------------------------------------------------------
    # Concentration and correlation checks
    # ------------------------------------------------------------------

    def check_concentration(self, ticker: str, proposed_size_pct: float) -> dict:
        """
        Checks if adding ticker at proposed_size_pct would breach any limit.
        Returns {passed, flags, ai_infrastructure_total}.
        """
        try:
            positions = self._client.get_all_positions()
            pv = self.portfolio_value()
        except Exception as e:
            return {"passed": True, "flags": [f"Concentration check skipped: {e}"],
                    "ai_infrastructure_total": 0}

        flags: list[str] = []

        # Current position weights
        pos_values: dict[str, float] = {}
        for p in positions:
            pct = float(p.market_value) / pv if pv else 0
            pos_values[p.symbol] = pct

        # Add proposed
        pos_values[ticker] = pos_values.get(ticker, 0) + proposed_size_pct

        # AI infrastructure tickers (approximate)
        ai_infra_keywords = {"NVDA", "AMD", "ARM", "INTC", "MSFT", "AMZN", "GOOG", "META",
                             "AVGO", "ANET", "CRDO", "MRVL", "ASML", "SNPS", "KLAC", "LRCX",
                             "AMAT", "MU", "CRWV", "ORCL", "CLS", "PLTR", "SNOW", "DDOG",
                             "VRT", "PWR", "GEV", "CEG"}
        ai_infra_total = sum(v for t, v in pos_values.items() if t in ai_infra_keywords)

        if ai_infra_total > MAX_AI_INFRA_PCT:
            flags.append(f"AI infrastructure exposure {ai_infra_total:.1%} exceeds {MAX_AI_INFRA_PCT:.0%} limit")

        if pos_values.get(ticker, 0) > MAX_POSITION_PCT_COMPOUNDER:
            flags.append(f"{ticker} position {pos_values[ticker]:.1%} exceeds Compounder cap {MAX_POSITION_PCT_COMPOUNDER:.0%}")

        return {
            "passed": len(flags) == 0,
            "flags":  flags,
            "ai_infrastructure_total": round(ai_infra_total, 4),
        }

    def check_correlation(self, ticker: str) -> dict:
        """
        Simple correlation check: flag if 3+ existing positions share the same
        primary sector as the proposed ticker.
        """
        try:
            from knowledge_loader import load_watchlist
            watchlist = load_watchlist()
            positions = self._client.get_all_positions()
        except Exception as e:
            return {"shared_catalyst_count": 0, "note": f"Correlation check skipped: {e}", "reduce_by": 0}

        ticker_entry  = watchlist.get(ticker, {})
        ticker_sector = ticker_entry.get("sector", "")

        shared = sum(
            1 for p in positions
            if watchlist.get(p.symbol, {}).get("sector", "") == ticker_sector
        )

        reduce_by = 0.25 if shared >= 2 else 0  # reduce 25% if 3+ positions same sector
        return {
            "shared_catalyst_count": shared,
            "note": f"{shared} existing positions in {ticker_sector}" if ticker_sector else "Sector unknown",
            "reduce_by": reduce_by,
        }

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
        return int(budget // price)

    def check_sell_allowed(self, symbol: str, shares: int) -> bool:
        try:
            position = self._client.get_open_position(symbol)
            owned = int(position.qty)
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
                new_stop   = round(current * (1 - trail_dist), 2)
                prev_high  = stop_record.get("trailing_high") or entry

                if current > prev_high:
                    try:
                        db.update_trailing_stop(ticker, new_stop, current)
                        log.info(f"[{ticker}] Trailing stop updated to ${new_stop:.2f} (high=${current:.2f})")
                    except Exception as e:
                        log.warning(f"Failed to update trailing stop for {ticker}: {e}")

        return alerts


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        raise EnvironmentError("ALPACA credentials not found in .env")

    client = TradingClient(api_key, api_secret, paper=True)
    rm     = RiskManager(client)

    pv = rm.portfolio_value()
    print(f"Portfolio value: ${pv:,.2f}")
    print(f"Halted: {rm.is_halted()}")
    print()

    sizing = rm.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On", price=130.0)
    print("NVDA sizing (Compounder, conviction=8, Risk-On):")
    for k, v in sizing.items():
        print(f"  {k}: {v}")
