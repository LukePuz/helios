"""
Scanner Agent — deterministic screening of the full watchlist.

Scores every ticker on four dimensions (0-25 each → 0-100 composite)
and routes HIGH/MEDIUM priority candidates to the Analyst Agent.

Follows scanner_agent.md spec exactly.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

try:
    from config import MIN_COMPOSITE_SCORE_HIGH, MIN_COMPOSITE_SCORE_MEDIUM
except ImportError:
    MIN_COMPOSITE_SCORE_HIGH   = 75
    MIN_COMPOSITE_SCORE_MEDIUM = 50

# ---------------------------------------------------------------------------
# Alpaca bar fetcher (reused for momentum scoring)
# ---------------------------------------------------------------------------

def _fetch_bars(ticker: str, days: int = 90) -> list[dict]:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key:
        return []

    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days + 10)).strftime("%Y-%m-%d")

    url     = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params  = {"timeframe": "1Day", "start": start, "end": end, "limit": days + 10, "feed": "iex"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        bars = resp.json().get("bars", [])
        return [{"date": b["t"][:10], "open": b["o"], "high": b["h"],
                 "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars]
    except Exception as e:
        log.debug(f"Bar fetch failed for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Individual scoring functions (each returns 0-25)
# ---------------------------------------------------------------------------

def _score_technical(bars: list[dict], ticker: str = "") -> tuple[int, str]:
    """
    Price position relative to 50DMA and 200DMA, plus real trend structure
    (swing highs/lows -- Dow Theory) and 200DMA slope. Max 25 points.

    2026-07-26: replaced the old "new 20-day high/low in the last 3
    sessions" breakout/breakdown check with agents.price_action's swing
    structure -- a single-bar extreme is not trend structure, and the old
    check couldn't tell "genuine uptrend" from "bounced once inside a
    downtrend". Also added a 200DMA-slope check: price sitting above a
    FALLING 200DMA is a much weaker signal than above a rising one, which
    the old code couldn't distinguish at all.
    """
    if len(bars) < 50:
        return 0, "Insufficient data"

    closes  = [b["close"] for b in bars]
    current = closes[-1]
    ma50    = sum(closes[-50:]) / 50
    ma200   = sum(closes[-200:]) / 200 if len(closes) >= 200 else sum(closes) / len(closes)

    score = 0
    notes_parts = []

    # Price vs MAs
    if current > ma50 and current > ma200:
        score += 10
        notes_parts.append(f"above 50DMA ({ma50:.2f}) and 200DMA ({ma200:.2f})")
    elif current > ma50:
        score += 5
        notes_parts.append(f"above 50DMA ({ma50:.2f}) only")
    else:
        notes_parts.append(f"below 50DMA ({ma50:.2f})")

    try:
        from agents.price_action import compute_price_action
        pa = compute_price_action(ticker=ticker, bars=bars)
    except Exception as e:
        pa = None
        log.debug(f"price_action unavailable for technical scoring: {e}")

    if pa:
        if pa["structure"] == "uptrend":
            score += 10
            notes_parts.append("swing structure: uptrend (higher highs + higher lows)")
        elif pa["structure"] == "downtrend":
            score = max(0, score - 10)
            notes_parts.append("swing structure: downtrend (lower highs + lower lows) — penalized")
        else:
            notes_parts.append(f"swing structure: {pa['structure']}")

        if pa["ma200_direction"] == "rising":
            score += 5
            notes_parts.append("200DMA rising")
        elif pa["ma200_direction"] == "falling":
            score = max(0, score - 5)
            notes_parts.append("200DMA falling — penalized")
    else:
        # Fall back to the old single-bar check only when price_action
        # genuinely can't compute (e.g. too little history) rather than
        # silently dropping this half of the score.
        if len(bars) >= 23:
            recent_high = max(b["high"] for b in bars[-23:-3])
            if any(b["high"] > recent_high for b in bars[-3:]):
                score += 10
                notes_parts.append("recent 20-day breakout (fallback check)")
            recent_low = min(b["low"] for b in bars[-23:-3])
            if any(b["low"] < recent_low for b in bars[-3:]):
                score = max(0, score - 5)
                notes_parts.append("recent breakdown — penalized (fallback check)")

    score = max(0, min(25, score))
    return score, f"Technical: {score}/25 — " + "; ".join(notes_parts)


def _score_momentum(bars: list[dict], ticker: str = "") -> tuple[int, str]:
    """
    Price momentum: 20-day and 60-day % changes.
    Proxy for revenue/earnings momentum (proper fundamental data needs earnings API).
    Max 25 points.
    """
    if len(bars) < 20:
        return 0, "Insufficient data for momentum"

    closes  = [b["close"] for b in bars]
    current = closes[-1]

    mom_20 = (current - closes[-20]) / closes[-20] if closes[-20] else 0
    mom_60 = (current - closes[-60]) / closes[-60] if len(closes) >= 60 and closes[-60] else mom_20

    score = 0
    notes_parts = [f"20d: {mom_20:+.1%}", f"60d: {mom_60:+.1%}"]

    # 20-day momentum
    if mom_20 > 0.15:
        score += 12
    elif mom_20 > 0.05:
        score += 8
    elif mom_20 > 0:
        score += 4
    else:
        score += 0

    # 60-day momentum
    if mom_60 > 0.25:
        score += 10
    elif mom_60 > 0.10:
        score += 6
    elif mom_60 > 0:
        score += 3
    else:
        score += 0

    # Volume confirmation via OBV direction, not a raw surge flag.
    # 2026-07-26: the old check added +3 for ANY 5-day volume surge vs the
    # prior 20-day average, with NO regard for direction -- a volume spike
    # on a down move scored identically to one on an up move, silently
    # rewarding distribution as if it were accumulation. OBV only confirms
    # when volume is actually flowing the same way price is moving.
    try:
        from agents.price_action import compute_price_action
        pa = compute_price_action(ticker=ticker, bars=bars)
    except Exception as e:
        pa = None
        log.debug(f"price_action unavailable for momentum scoring: {e}")

    if pa:
        if pa["volume_confirmation"] == "confirmed":
            score += 3
            notes_parts.append("volume (OBV) confirms uptrend")
        elif pa["volume_confirmation"] == "bearish_divergence":
            score = max(0, score - 3)
            notes_parts.append("volume (OBV) bearish divergence — penalized")
    elif len(bars) >= 25:
        # Fallback only if price_action genuinely can't compute.
        recent_vol  = sum(b["volume"] for b in bars[-5:]) / 5
        prior_avg   = sum(b["volume"] for b in bars[-25:-5]) / 20
        if prior_avg > 0 and recent_vol > prior_avg * 1.3:
            score += 3
            notes_parts.append("volume surge (fallback check, no direction)")

    score = max(0, min(25, score))
    return score, f"Momentum: {score}/25 — " + "; ".join(notes_parts)


def _score_momentum_with_flow(bars: list[dict], ticker: str) -> tuple[int, str]:
    """Wraps _score_momentum and appends Unusual Whales options flow signal."""
    score, notes = _score_momentum(bars, ticker)

    try:
        from ingestion.options_flow import get_flow_summary
        flow = get_flow_summary(ticker)
        if flow.get("unusual"):
            bonus = 8
            score = min(25, score + bonus)
            premium = flow.get("total_premium", 0)
            notes += f" | Unusual options flow (${premium:,.0f} premium)"
        elif flow.get("call_ratio", 0.5) > 0.65:
            bonus = 3
            score = min(25, score + bonus)
            notes += " | Bullish options skew"
    except Exception as e:
        log.debug(f"[{ticker}] Options flow scoring skipped: {e}")

    return score, notes


_EARNINGS_CATALYST_DAYS = 21


def _score_catalyst(watchlist_entry: dict, recent_filings: list[str]) -> tuple[int, str, str]:
    """
    Catalyst proximity scoring. Returns (score, notes, scan_type).
    Max 25 points.
    - earnings date from watchlist notes or filings (best effort)
    - filing freshness
    """
    notes_parts = []
    scan_type   = "CORE"
    score       = 0

    ticker = watchlist_entry.get("ticker", "")
    tier   = watchlist_entry.get("tier", "Watch")
    role   = watchlist_entry.get("role", "Accelerator")

    # Check if there's a recent filing (8-K) for this ticker
    if ticker in recent_filings:
        score     += 15
        scan_type  = "CATALYST"
        notes_parts.append("recent 8-K filing detected")

    # Moonshots get extra catalyst credit for any recent news
    if role == "Moonshot" and ticker in recent_filings:
        scan_type = "TAIL"

    # Government contract awards (Quiver Quant; off with QUIVER_ENABLED, Luke D2)
    try:
        from config import QUIVER_ENABLED
        if not QUIVER_ENABLED:
            raise RuntimeError("Quiver disabled")
        from ingestion.congressional import get_gov_contracts
        contracts = get_gov_contracts(ticker, days_back=14)
        if contracts:
            total_val = sum(c.get("amount", 0) for c in contracts)
            pts = 15 if total_val > 50_000_000 else 10
            score = min(25, score + pts)
            agency = contracts[0].get("agency", "Gov")
            notes_parts.append(f"Gov contract ${total_val/1e6:.0f}M ({agency})")
            scan_type = "CATALYST"
    except Exception as e:
        log.debug(f"[{ticker}] Gov contracts check skipped: {e}")

    # Dark pool significant print (Unusual Whales)
    try:
        from ingestion.options_flow import get_darkpool_activity
        dp = get_darkpool_activity(ticker)
        if dp.get("has_significant_print"):
            score = min(25, score + 8)
            notes_parts.append(f"Dark pool ${dp['largest_print']/1e6:.1f}M print")
    except Exception as e:
        log.debug(f"[{ticker}] Dark pool check skipped: {e}")

    # A dated earnings report soon is a catalyst (Luke D4, 10-29). Same
    # Postgres calendar the bot acts on.
    try:
        import db
        from datetime import date
        nxt = db.get_next_earnings(ticker)
        if nxt is not None:
            days_out = (nxt - date.today()).days
            if 0 <= days_out <= _EARNINGS_CATALYST_DAYS:
                score += 10
                notes_parts.append(f"earnings in {days_out}d ({nxt})")
                scan_type = "CATALYST"
    except Exception as e:
        log.debug(f"[{ticker}] Earnings proximity check skipped: {e}")

    # Tier-based base catalyst score (Core tickers have more visibility)
    if tier == "Core" and score == 0:
        notes_parts.append("Core tier — ongoing catalyst monitoring")
    elif tier == "Speculative" and score > 0:
        scan_type = "TAIL"

    score = min(25, score)
    catalyst_detail = "; ".join(notes_parts) if notes_parts else "No near-term catalyst identified"
    return score, f"Catalyst: {score}/25 — {catalyst_detail}", scan_type


def _score_thesis_alignment(watchlist_entry: dict) -> tuple[int, str]:
    """
    Thesis alignment score based on conviction tier and role.
    Max 25 points.
    """
    tier = watchlist_entry.get("tier", "Watch")
    role = watchlist_entry.get("role", "Accelerator")

    if tier == "Core":
        score = 25
        note  = "Core conviction tier"
    elif tier == "Watch":
        score = 15
        note  = "Watch tier — needs validation"
    else:  # Speculative
        score = 10
        note  = "Speculative/Moonshot"

    # Role modifier
    if role == "Compounder":
        note += " | Compounder bucket"
    elif role == "Moonshot":
        score = min(score, 15)  # cap moonshots — pure thesis plays
        note += " | Moonshot bucket (capped)"

    return score, f"Thesis: {score}/25 — {note}"


# ---------------------------------------------------------------------------
# Regime-based threshold override
# ---------------------------------------------------------------------------

def _regime_threshold(regime: str) -> int:
    """Returns the minimum composite score to pass to analyst for a given regime."""
    return {
        "Strong Risk-On": MIN_COMPOSITE_SCORE_MEDIUM,
        "Risk-On":        MIN_COMPOSITE_SCORE_MEDIUM,
        "Neutral":        65,
        "Risk-Off":       80,   # only very high signals get through
        "Strong Risk-Off": 999, # nothing new; monitor only
    }.get(regime, MIN_COMPOSITE_SCORE_MEDIUM)


def _priority_label(score: int, regime: str) -> str:
    threshold = _regime_threshold(regime)
    if score >= MIN_COMPOSITE_SCORE_HIGH and score >= threshold:
        return "HIGH"
    elif score >= MIN_COMPOSITE_SCORE_MEDIUM and score >= threshold:
        return "MEDIUM"
    elif score >= 25:
        return "LOW"
    else:
        return "PASS"


# ---------------------------------------------------------------------------
# Fetch recent EDGAR filings for all tickers (quick check)
# ---------------------------------------------------------------------------

def _get_recent_filings() -> set[str]:
    """Watchlist tickers with an 8-K the EDGAR poller saw in the last 3 days.
    Until 10-30 this imported names edgar.py doesn't have, so the ImportError
    was swallowed and no scan ever scored an 8-K (audit A1253-1; Luke D4)."""
    try:
        import db
        return db.recent_filing_tickers("8-K", days=3)
    except Exception as e:
        log.warning(f"Recent 8-K lookup failed: {e}")
        return set()


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def run_scan(regime_packet: dict | None = None) -> dict:
    """
    Screen the full watchlist and return a candidate list for the Analyst.

    Args:
        regime_packet: Output from market_regime.assess_regime() or get_cached_regime().
                       If None, fetches from cache.

    Returns:
        Scanner output packet matching scanner_agent.md spec.
    """
    if regime_packet is None:
        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
        except Exception:
            regime_packet = {"regime": "Neutral", "score": 50}

    regime = regime_packet.get("regime", "Neutral")
    # Logged on each candidate row: which regime row this scan used (C25).
    regime_detail = {
        "score": regime_packet.get("score"),
        "source": regime_packet.get("regime_source") or ("fresh" if regime_packet.get("agent") else None),
        "timestamp": str(regime_packet.get("timestamp")) if regime_packet.get("timestamp") else None,
        "last_bar_date": (regime_packet.get("signals") or {}).get("last_bar_date"),
        "threshold": _regime_threshold(regime),
    }
    log.info(f"Running scanner in {regime} regime...")

    # Strong Risk-Off: don't surface new positions
    if regime == "Strong Risk-Off":
        log.info("Strong Risk-Off — scanner suppressed, monitoring only")
        return {
            "agent": "scanner",
            "date":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "regime_context": regime,
            "candidates": [],
            "disqualified": [],
            "scan_notes": "Scanner suppressed: Strong Risk-Off regime. Monitor existing positions only.",
        }

    # Load watchlist
    try:
        from knowledge_loader import load_watchlist
        watchlist = load_watchlist()
    except Exception as e:
        log.error(f"Failed to load watchlist: {e}")
        return {"agent": "scanner", "candidates": [], "disqualified": [], "scan_notes": f"Watchlist load failed: {e}"}

    # Fetch recent filings for catalyst scoring
    recent_filings = _get_recent_filings()
    log.info(f"Recent filings found for: {recent_filings or 'none'}")

    candidates:   list[dict] = []
    disqualified: list[dict] = []
    threshold = _regime_threshold(regime)
    from agents.market_regime import completed_bars

    for ticker, entry in watchlist.items():
        entry_with_ticker = {**entry, "ticker": ticker}

        # Fetch bars
        bars = completed_bars(_fetch_bars(ticker, days=220))
        if not bars:
            disqualified.append({"ticker": ticker, "reason": "No price data available", "composite_score": 0})
            continue

        # Earnings blackout — skip if earnings within the blackout window, cap
        # catalyst if within 2x it. Both off when EARNINGS_ENTRY_BLACKOUT_DAYS
        # is 0 (Luke, 2026-10-28).
        from config import EARNINGS_ENTRY_BLACKOUT_DAYS as _BLACKOUT
        try:
            import db as _db
            if _BLACKOUT and _db.is_near_earnings(ticker, days=_BLACKOUT):
                log.info(f"[{ticker}] Skipped — earnings within {_BLACKOUT} days")
                disqualified.append({"ticker": ticker, "reason": f"Earnings within {_BLACKOUT} days (blackout)", "composite_score": 0})
                continue
        except Exception:
            pass

        # Score each dimension
        tech_score,     tech_notes    = _score_technical(bars, ticker)
        mom_score,      mom_notes     = _score_momentum_with_flow(bars, ticker)
        cat_score,      cat_notes, scan_type = _score_catalyst(entry_with_ticker, recent_filings)

        # Cap catalyst score if earnings within 10 days (timing risk)
        try:
            import db as _db
            if _BLACKOUT and _db.is_near_earnings(ticker, days=2 * _BLACKOUT):
                if cat_score > 15:
                    cat_score = 15
                    cat_notes += " [earnings risk: within 10 days]"
        except Exception:
            pass
        thesis_score,   thesis_notes  = _score_thesis_alignment(entry_with_ticker)

        composite = tech_score + mom_score + cat_score + thesis_score
        priority  = _priority_label(composite, regime)

        candidate = {
            "ticker":               ticker,
            "scan_type":            scan_type,
            "composite_score":      composite,
            "technical_score":      tech_score,
            "momentum_score":       mom_score,
            "catalyst_score":       cat_score,
            "thesis_alignment_score": thesis_score,
            "catalyst_detail":      cat_notes,
            "technical_notes":      tech_notes,
            "momentum_notes":       mom_notes,
            "thesis_notes":         thesis_notes,
            "priority":             priority,
            "pass_to_analyst":      priority in ("HIGH", "MEDIUM"),
            "role":                 entry.get("role"),
            "tier":                 entry.get("tier"),
            "sector":               entry.get("sector"),
        }

        # Shadow-mode setup classification (2026-09-13) -- logged
        # alongside the existing composite-score system for comparison,
        # does NOT gate pass_to_analyst or replace `role` above. See
        # agents/setup_gates.py's module docstring: this is Phase 2 of
        # the swing-trading redesign, running in parallel with the live
        # system so real-world firing rates can be compared before any
        # cutover (agents/analyst.py's SETUP_GATES_ENABLED flag, not yet
        # built) is considered.
        try:
            from agents.setup_gates import classify_setup
            best_setup, all_setup_gates = classify_setup(ticker)
            candidate["setup_category"] = best_setup["category"] if best_setup else None
            candidate["setup_triggered"] = bool(best_setup and best_setup["triggered"])
            candidate["setup_gates_shadow"] = {
                cat: {
                    "qualifies": g["qualifies"], "triggered": g["triggered"],
                    "criteria": g["criteria"], "reasoning": g["reasoning"],
                }
                for cat, g in all_setup_gates.items()
            }
        except Exception as e:
            # WARNING, not debug (2026-09-17, found investigating why
            # scanner_candidates showed 0/75 rows with setup_category
            # populated across two full days despite the persistence fix
            # working correctly in isolation -- this exception handler's
            # debug-level log was hiding a systematic failure completely.
            log.warning(f"[{ticker}] shadow setup classification failed: {e}")
            candidate["setup_category"] = None
            candidate["setup_triggered"] = False
            candidate["setup_gates_shadow"] = {}

        if priority == "PASS":
            disqualified.append({
                "ticker": ticker,
                "reason": f"Score {composite}/100 below threshold ({threshold}) in {regime} regime",
                "composite_score": composite,
            })
        else:
            candidates.append(candidate)

        # Log to DB
        try:
            import db
            db.log_scanner_candidate(
                ticker=ticker,
                scan_type=scan_type,
                composite_score=composite,
                technical_score=tech_score,
                momentum_score=mom_score,
                catalyst_score=cat_score,
                thesis_alignment_score=thesis_score,
                priority=priority,
                catalyst_detail=cat_notes,
                passed_to_analyst=priority in ("HIGH", "MEDIUM"),
                regime_at_scan=regime,
                setup_category=candidate.get("setup_category"),
                setup_triggered=candidate.get("setup_triggered"),
                setup_gates_detail=candidate.get("setup_gates_shadow"),
                regime_detail=regime_detail,
            )
        except Exception as e:
            log.debug(f"Scanner DB log failed for {ticker}: {e}")

    # Sort by composite score descending
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    high_count   = sum(1 for c in candidates if c["priority"] == "HIGH")
    medium_count = sum(1 for c in candidates if c["priority"] == "MEDIUM")

    scan_notes = (
        f"{len(candidates)} candidates surfaced ({high_count} HIGH, {medium_count} MEDIUM). "
        f"{len(disqualified)} disqualified. Regime: {regime}."
    )

    log.info(scan_notes)

    return {
        "agent":          "scanner",
        "date":           datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "regime_context": regime,
        "candidates":     candidates,
        "disqualified":   disqualified,
        "scan_notes":     scan_notes,
    }


def get_high_priority_candidates(regime_packet: dict | None = None) -> list[dict]:
    """Convenience: run scan and return only HIGH priority candidates."""
    result = run_scan(regime_packet)
    return [c for c in result.get("candidates", []) if c["priority"] == "HIGH"]
