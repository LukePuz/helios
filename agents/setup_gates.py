"""
Setup gates — deterministic classification into the three swing-trading
categories (Breakout, Catalyst, Continuation), replacing the long-term
conviction buckets (Compounder/Accelerator/Moonshot) per the 2026-09-13
strategy redesign.

SHADOW MODE: classify_setup() is meant to be wired into scanner.py as an
additional logged field alongside the existing composite-score system —
it does NOT yet gate which candidates reach the analyst, and setup_
category does NOT yet replace thesis_bucket as sizing/stop authority.
That cutover lives behind agents/analyst.py's SETUP_GATES_ENABLED flag
(a later phase), which defaults off. Nothing in this module places an
order, sizes a position, or changes a stop.

Each gate splits into two questions:
  - qualifies: is this a structurally valid setup at all (feeds the
    candidate pipeline / "on watch" reporting)
  - triggered: has the actual entry condition fired on the current bar
    (feeds execution timing, once wired)
A ticker can qualify without triggering.
"""
import logging

log = logging.getLogger(__name__)

BREAKOUT_VOLUME_MULTIPLE = 1.5          # breakout day volume vs. 20-day average
CONTINUATION_PULLBACK_ATR_FRACTION = 0.5   # how close to a rising MA rung counts as "testing" it
CONTINUATION_LOOKBACK_BARS = 5

_RISK_OFF_REGIMES = ("Risk-Off", "Strong Risk-Off")


def _resolve_regimes(ticker: str) -> tuple[dict, dict]:
    """Returns (market_packet, sector_packet). Sector falls back to the
    market-wide packet when the ticker isn't a resolvable S&P 500
    constituent (small-caps like RGTI/QBTS/LUNR aren't) -- same "some
    read beats no read" posture agents.market_regime.
    get_trade_style_for_ticker's own docstring already documents."""
    from agents.market_regime import get_cached_regime, get_cached_sector_regime
    market = get_cached_regime()
    sector = market
    try:
        from ingestion.sp500_universe import SECTOR_ETF_MAP, get_sp500_constituents
        gics_sector = get_sp500_constituents().get(ticker, {}).get("gics_sector")
        etf = SECTOR_ETF_MAP.get(gics_sector) if gics_sector else None
        if etf:
            sector = get_cached_sector_regime(etf, regime_type="gics_sector")
    except Exception as e:
        log.debug(f"[{ticker}] sector regime resolution failed, using market-wide: {e}")
    return market, sector


def _gate_result(
    category: str, qualifies: bool, triggered: bool, criteria: dict,
    entry_price: float | None = None, stop_price: float | None = None,
    target_prices: list[float] | None = None, conviction_adjustment: int = 0, reasoning: str = "",
) -> dict:
    return {
        "category": category, "qualifies": qualifies, "triggered": triggered,
        "criteria": criteria, "entry_price": entry_price, "stop_price": stop_price,
        "target_prices": target_prices or [], "conviction_adjustment": conviction_adjustment,
        "reasoning": reasoning,
    }


def check_breakout_gate(ticker: str, ladder: dict | None = None, bars: list[dict] | None = None) -> dict:
    """Qualification: market/sector not in outright risk-off, Trend
    Template confirmed (Stage 2), a genuine base with >=2 contractions
    detected. Trigger: close above the base ceiling on >=1.5x average
    volume. Stop/targets read straight from the ladder's own rungs."""
    import db
    ladder = ladder if ladder is not None else db.get_trade_ladder(ticker)
    if not ladder:
        return _gate_result("Breakout", False, False, {}, reasoning="No trade ladder computed yet")

    market, sector = _resolve_regimes(ticker)
    try:
        no_blackout = not db.is_near_earnings(ticker, days=5)
    except Exception:
        no_blackout = True

    criteria = {
        "market_not_risk_off":        market.get("regime") not in _RISK_OFF_REGIMES,
        "sector_not_strong_risk_off": sector.get("regime") != "Strong Risk-Off",
        "trend_stage_confirmed":      ladder.get("trend_stage") == "stage2_confirmed",
        "base_detected":              bool(ladder.get("base_detected")),
        "base_contractions_ge_2":     (ladder.get("base_contractions") or 0) >= 2,
        "no_earnings_blackout":       no_blackout,
    }
    qualifies = all(criteria.values())

    triggered, entry_price, stop_price, target_prices = False, None, None, []
    if qualifies:
        if bars is None:
            try:
                bars = db.get_price_history(ticker, days=25)
            except Exception:
                bars = []
        ceiling = ladder.get("base_ceiling")
        if bars and len(bars) >= 21 and ceiling:
            today = bars[-1]
            avg_vol = sum(b["volume"] for b in bars[-21:-1]) / 20
            if today["close"] > ceiling and avg_vol and today["volume"] >= BREAKOUT_VOLUME_MULTIPLE * avg_vol:
                triggered = True
                entry_price = today["close"]
                stop_price = ladder.get("base_floor")
                rungs_above = sorted(r["price"] for r in (ladder.get("rungs") or []) if r["price"] > ceiling)
                if rungs_above:
                    target_prices = rungs_above[:3]
                elif stop_price is not None:
                    target_prices = [round(ceiling + (ceiling - stop_price), 2)]  # measured-move projection

    return _gate_result(
        "Breakout", qualifies, triggered, criteria, entry_price, stop_price, target_prices,
        reasoning=f"{sum(criteria.values())}/{len(criteria)} criteria met" + (" — TRIGGERED" if triggered else ""),
    )


def check_catalyst_gate(ticker: str, ladder: dict | None = None, recent_signals: list[dict] | None = None) -> dict:
    """Qualification (looser than Breakout -- a real catalyst can work in
    more environments): market not Strong Risk-Off, a deterministic
    gap+volume signal active (agents.trade_ladder.compute_gap_catalyst,
    already computed nightly), AND a corroborating real event from the
    screener pipeline -- two independent confirmations, same principle as
    the analyst's own multi-source gate, so an unexplained gap (data
    glitch, thin-liquidity spike) can't pass alone."""
    import db
    ladder = ladder if ladder is not None else db.get_trade_ladder(ticker)
    if not ladder:
        return _gate_result("Catalyst", False, False, {}, reasoning="No trade ladder computed yet")

    market, sector = _resolve_regimes(ticker)

    corroborated = False
    try:
        signals = recent_signals if recent_signals is not None else db.get_recent_signals(limit=50)
        corroborated = any(s.get("symbol") == ticker for s in signals)
    except Exception as e:
        log.debug(f"[{ticker}] catalyst corroboration check failed: {e}")

    criteria = {
        "market_not_strong_risk_off": market.get("regime") != "Strong Risk-Off",
        "gap_catalyst_active":        bool(ladder.get("gap_catalyst_active")),
        "corroborating_event":        corroborated,
    }
    qualifies = all(criteria.values())
    # Sector headwind doesn't block a real catalyst, but asks for one
    # more point of conviction before treating it as fully confirmed.
    conviction_adjustment = 1 if (qualifies and sector.get("regime") in _RISK_OFF_REGIMES) else 0

    triggered = qualifies  # gap+volume+corroboration IS the trigger here; gap-hold already checked inside compute_gap_catalyst
    entry_price = ladder.get("gap_day_high") if triggered else None
    stop_price = ladder.get("gap_day_low") if triggered else None
    target_prices = []
    if triggered and entry_price:
        rungs_above = sorted(r["price"] for r in (ladder.get("rungs") or []) if r["price"] > entry_price)
        target_prices = rungs_above[:3]

    return _gate_result(
        "Catalyst", qualifies, triggered, criteria, entry_price, stop_price, target_prices,
        conviction_adjustment=conviction_adjustment,
        reasoning=f"{sum(criteria.values())}/{len(criteria)} criteria met" + (" — TRIGGERED" if triggered else ""),
    )


def check_continuation_gate(ticker: str, ladder: dict | None = None, bars: list[dict] | None = None) -> dict:
    """The most regime-dependent of the three: requires the ticker's OWN
    sector to be Trending/Expanding (not just the broad market), a
    confirmed Stage 2 uptrend, and a genuine pullback to a rising MA rung
    on contracting volume. Trigger: a resumption up-day off that test."""
    import db
    from agents.market_regime import get_trade_style_for_ticker

    ladder = ladder if ladder is not None else db.get_trade_ladder(ticker)
    if not ladder:
        return _gate_result("Continuation", False, False, {}, reasoning="No trade ladder computed yet")

    market, _sector = _resolve_regimes(ticker)
    trade_style = get_trade_style_for_ticker(ticker) or market.get("trade_style")

    if bars is None:
        try:
            bars = db.get_price_history(ticker, days=30)
        except Exception:
            bars = []

    pullback_ok, resumption, tested_rung = False, False, None
    atr = ladder.get("atr")
    if bars and len(bars) >= CONTINUATION_LOOKBACK_BARS + 1 and atr:
        recent = bars[-CONTINUATION_LOOKBACK_BARS:]
        ma_rungs = [r for r in (ladder.get("rungs") or []) if any(t in r["types"] for t in ("ma10", "ma21", "ma50"))]
        band = atr * CONTINUATION_PULLBACK_ATR_FRACTION
        for rung in ma_rungs:
            if any(abs(b["low"] - rung["price"]) <= band for b in recent):
                pullback_ok, tested_rung = True, rung
                break
        if pullback_ok:
            pullback_leg = recent[:-1] or recent
            pullback_vol_avg = sum(b["volume"] for b in pullback_leg) / len(pullback_leg)
            today = bars[-1]
            if today["close"] > today["open"] and today["volume"] >= pullback_vol_avg:
                resumption = True

    criteria = {
        "market_not_risk_off":  market.get("regime") not in _RISK_OFF_REGIMES,
        "sector_trending":      trade_style == "Trending/Expanding",
        "trend_stage_confirmed": ladder.get("trend_stage") == "stage2_confirmed",
        "pullback_to_rising_ma": pullback_ok,
    }
    qualifies = all(criteria.values())
    triggered = qualifies and resumption

    entry_price, stop_price, target_prices = None, None, []
    if triggered:
        entry_price = bars[-1]["close"]
        stop_price = tested_rung["price"] if tested_rung else None
        swing_highs = sorted(
            r["price"] for r in (ladder.get("rungs") or [])
            if "swing_high" in r["types"] and r["price"] > entry_price
        )
        target_prices = swing_highs[:2]

    return _gate_result(
        "Continuation", qualifies, triggered, criteria, entry_price, stop_price, target_prices,
        reasoning=f"{sum(criteria.values())}/{len(criteria)} criteria met" + (" — TRIGGERED" if triggered else ""),
    )


# Catalyst first: a real event has the strongest independent
# justification and can co-occur with either of the other two.
_PRIORITY = ("Catalyst", "Breakout", "Continuation")


def classify_setup(ticker: str) -> tuple[dict | None, dict[str, dict]]:
    """
    Runs all three gates once (sharing one ladder fetch). Returns
    (best_match, all_gates) where best_match is the highest-priority
    TRIGGERED gate, else the highest-priority QUALIFYING gate ("on
    watch"), else None if nothing matches at all. all_gates always has
    all three results for inspection/logging regardless of best_match.
    """
    import db
    ladder = db.get_trade_ladder(ticker)
    gates = {
        "Catalyst":     check_catalyst_gate(ticker, ladder=ladder),
        "Breakout":     check_breakout_gate(ticker, ladder=ladder),
        "Continuation": check_continuation_gate(ticker, ladder=ladder),
    }
    triggered = [gates[c] for c in _PRIORITY if gates[c]["triggered"]]
    if triggered:
        return triggered[0], gates
    qualifying = [gates[c] for c in _PRIORITY if gates[c]["qualifies"]]
    if qualifying:
        return qualifying[0], gates
    return None, gates


LEGACY_HEALTH_ALERT_THROTTLE_HOURS = 24
_LEGACY_HEALTH_STATE_KEY = "legacy_position_health_last_alerted"


def check_legacy_position_health(ticker: str, latest_thesis: dict) -> dict | None:
    """
    2026-09-15, user request: a position matching ZERO of the three live
    setup categories (Breakout/Catalyst/Continuation all qualifies=False)
    AND carrying no genuine near-term catalyst (latest_thesis's
    no_near_term_catalyst) is exactly the profile the swing-trading
    redesign says shouldn't sit unexamined -- found reviewing 4 real held
    positions that all 4 matched this profile simultaneously, invisibly,
    with nothing forcing a decision. Emits a visible activity_feed alert
    (never a silent log line) -- deliberately NOT an auto-trim; that's a
    live-order behavior change, this is a visibility fix so a human
    actually sees it and decides.

    Throttled to once per LEGACY_HEALTH_ALERT_THROTTLE_HOURS per ticker
    (system_state-backed) -- same "don't re-alert a standing condition
    every single cycle" posture as tasks.reconcile_stop_orders'
    RECONCILE_ALERT_THROTTLE_HOURS, since review_all_held_positions can
    revisit the same handful of positions daily.

    Returns the alert dict if one was emitted, None if the position is
    healthy or the alert was throttled.
    """
    import json
    import db
    from datetime import datetime, timezone

    if not latest_thesis.get("no_near_term_catalyst"):
        return None

    _best, gates = classify_setup(ticker)
    if any(g["qualifies"] for g in gates.values()):
        return None  # still matches at least one live setup -- not stale

    try:
        throttle = json.loads(db.get_state(_LEGACY_HEALTH_STATE_KEY) or "{}")
    except Exception:
        throttle = {}
    last_alerted = throttle.get(ticker)
    if last_alerted:
        elapsed_h = (datetime.now(timezone.utc) - datetime.fromisoformat(last_alerted)).total_seconds() / 3600
        if elapsed_h < LEGACY_HEALTH_ALERT_THROTTLE_HOURS:
            return None

    alert = {
        "ticker": ticker,
        "conviction": latest_thesis.get("conviction"),
        "action": latest_thesis.get("action"),
        "reason": "No live setup match (Breakout/Catalyst/Continuation all fail) and no near-term "
                  "company-specific catalyst -- legacy position with nothing currently justifying continued sizing.",
    }
    try:
        from feed import emit
        emit(
            "position_health", ticker,
            f"STALE LEGACY POSITION -- conviction {alert['conviction']}/10, action {alert['action']}, "
            f"no live setup match, no near-term catalyst. Worth an explicit trim/hold decision.",
            "alert",
        )
    except Exception as e:
        log.debug(f"[{ticker}] legacy health alert emit failed: {e}")

    throttle[ticker] = datetime.now(timezone.utc).isoformat()
    try:
        db.set_state(_LEGACY_HEALTH_STATE_KEY, json.dumps(throttle))
    except Exception as e:
        log.debug(f"[{ticker}] legacy health throttle save failed: {e}")

    return alert


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    tickers = sys.argv[1:] or ["NVDA"]
    for t in tickers:
        best, all_gates = classify_setup(t)
        print(f"\n=== {t} ===")
        print(f"  best match: {best['category'] if best else None} "
              f"(triggered={best['triggered'] if best else None})" if best else "  no match")
        for cat, g in all_gates.items():
            print(f"  {cat}: qualifies={g['qualifies']} triggered={g['triggered']} — {g['reasoning']}")
            print(f"    criteria: {g['criteria']}")
