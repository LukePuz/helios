"""
Market Regime Agent — scores the macro and market environment daily.

Produces a regime classification (Strong Risk-On → Strong Risk-Off) that
gates all downstream agent activity. All other agents read this before acting.

Scoring follows knowledge/market_regime_rules.md exactly.
"""
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache — holds last assessed regime in memory
# ---------------------------------------------------------------------------
_regime_cache: dict | None = None
_cache_ts: float = 0.0

try:
    from config import REGIME_CACHE_TTL_SECONDS, REGIME_THRESHOLDS
except ImportError:
    REGIME_CACHE_TTL_SECONDS = 3600
    REGIME_THRESHOLDS = {
        "Strong Risk-On": (75, 100),
        "Risk-On":        (55, 74),
        "Neutral":        (40, 54),
        "Risk-Off":       (20, 39),
        "Strong Risk-Off": (0, 19),
    }

# ---------------------------------------------------------------------------
# Signal scoring constants (from market_regime_rules.md)
# ---------------------------------------------------------------------------
_SCORE = {"Very Bullish": 5, "Bullish": 2, "Neutral": 0, "Bearish": -2, "Very Bearish": -5}
_WEIGHTS = {"trend": 1, "volatility": 2, "liquidity": 2, "credit": 2, "breadth": 1, "volume": 1}

# Min/max possible weighted raw score (for normalization)
_MAX_RAW = sum(5 * w for w in _WEIGHTS.values())   # all Very Bullish
_MIN_RAW = sum(-5 * w for w in _WEIGHTS.values())  # all Very Bearish


def _normalize(raw: float) -> float:
    """Map raw weighted score to 0–100."""
    return round((raw - _MIN_RAW) / (_MAX_RAW - _MIN_RAW) * 100, 1)


def _regime_from_score(score: float) -> str:
    for name, (lo, hi) in REGIME_THRESHOLDS.items():
        if lo <= score <= hi:
            return name
    return "Neutral"


# ---------------------------------------------------------------------------
# Individual signal scorers
# ---------------------------------------------------------------------------

def _score_trend(spy_bars: list[dict]) -> tuple[int, str]:
    """
    Uses SPY bar data. Computes 50-day and 200-day SMAs.
    Returns (raw_score, notes).
    """
    if len(spy_bars) < 50:
        return _SCORE["Neutral"], "Insufficient bar data for trend analysis"

    closes = [b["close"] for b in spy_bars]
    current = closes[-1]
    ma50  = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else sum(closes) / len(closes)

    # Golden/Death Cross: compare current 50DMA vs 200DMA
    # Use slope: if 50DMA > 200DMA assume golden cross posture
    if ma50 > ma200 and current > ma50:
        score_label = "Very Bullish"
        notes = f"Golden Cross posture: 50DMA ({ma50:.2f}) > 200DMA ({ma200:.2f}), price ({current:.2f}) above both"
    elif current > ma50 and current > ma200:
        score_label = "Bullish"
        notes = f"Price ({current:.2f}) above both 50DMA ({ma50:.2f}) and 200DMA ({ma200:.2f})"
    elif current > ma50:
        score_label = "Neutral"
        notes = f"Price ({current:.2f}) above 50DMA ({ma50:.2f}) but below 200DMA ({ma200:.2f})"
    elif current > ma200:
        score_label = "Bearish"
        notes = f"Price ({current:.2f}) below 50DMA ({ma50:.2f}), above 200DMA ({ma200:.2f})"
    else:
        # Death Cross posture
        score_label = "Very Bearish"
        notes = f"Death Cross posture: price ({current:.2f}) below 50DMA ({ma50:.2f}) and 200DMA ({ma200:.2f})"

    return _SCORE[score_label], notes


def _score_volatility(
    vix: float | None,
    trend_raw: int | None = None,
    breadth_raw: int | None = None,
) -> tuple[int, str]:
    """
    VIX-based regime signal. Contrarian at extreme fear (>40) — but per
    knowledge/market_regime_rules.md: "Apply only when trend and breadth
    signals are not simultaneously Very Bearish. A VIX > 40 during a Death
    Cross environment may indicate more downside, not a bottom." The
    previous implementation applied Very Bullish unconditionally, ignoring
    that guard entirely.
    """
    if vix is None:
        return _SCORE["Neutral"], "VIX data unavailable"

    if vix > 40:
        both_very_bearish = (trend_raw is not None and breadth_raw is not None
                              and trend_raw <= -5 and breadth_raw <= -5)
        if both_very_bearish:
            return _SCORE["Bearish"], (
                f"VIX={vix:.1f}: extreme fear, but trend AND breadth are both "
                f"Very Bearish (Death Cross posture) — contrarian bullish signal "
                f"suppressed per regime rules, treating as continued stress"
            )
        return _SCORE["Very Bullish"], f"VIX={vix:.1f}: extreme fear / capitulation signal (contrarian bullish)"
    elif vix > 25:
        return _SCORE["Bearish"], f"VIX={vix:.1f}: elevated fear"
    elif vix > 15:
        return _SCORE["Neutral"], f"VIX={vix:.1f}: normal range"
    elif vix > 12:
        return _SCORE["Neutral"], f"VIX={vix:.1f}: mild complacency"
    else:
        return _SCORE["Bearish"], f"VIX={vix:.1f}: very low — market underpricing risk"


def _score_liquidity(macro: dict) -> tuple[int, str]:
    """
    Liquidity: Fed Funds Rate direction + yield curve as credit conditions proxy.
    """
    fedfunds = macro.get("FEDFUNDS", {}).get("value")
    t10y2y   = macro.get("T10Y2Y",   {}).get("value")

    if fedfunds is None:
        return _SCORE["Neutral"], "Fed Funds data unavailable"

    notes_parts = [f"Fed Funds: {fedfunds:.2f}%"]

    # Use yield curve to infer liquidity stress
    if t10y2y is not None:
        notes_parts.append(f"10Y-2Y spread: {t10y2y:.2f}%")
        if t10y2y > 0.5 and fedfunds < 4.0:
            return _SCORE["Bullish"], " | ".join(notes_parts) + " — stable, benign"
        elif t10y2y > 0 and fedfunds < 5.5:
            return _SCORE["Neutral"], " | ".join(notes_parts) + " — mixed signals"
        elif t10y2y < -0.25:
            return _SCORE["Bearish"], " | ".join(notes_parts) + " — inverted yield curve, tightening stress"
        elif t10y2y < -0.5:
            return _SCORE["Very Bearish"], " | ".join(notes_parts) + " — severe inversion, high liquidity stress"
        else:
            return _SCORE["Neutral"], " | ".join(notes_parts)

    # Fallback: just use Fed Funds level
    if fedfunds > 5.0:
        return _SCORE["Bearish"], f"Fed Funds: {fedfunds:.2f}% — high rates constraining liquidity"
    elif fedfunds < 2.0:
        return _SCORE["Bullish"], f"Fed Funds: {fedfunds:.2f}% — accommodative"
    else:
        return _SCORE["Neutral"], f"Fed Funds: {fedfunds:.2f}%"


def _score_credit(macro: dict) -> tuple[int, str]:
    """
    Credit proxy: use yield curve spread (T10Y2Y).
    A severely inverted curve signals credit stress.
    """
    t10y2y = macro.get("T10Y2Y", {}).get("value")
    dgs10  = macro.get("DGS10",  {}).get("value")
    dgs2   = macro.get("DGS2",   {}).get("value")

    if t10y2y is None and dgs10 is not None and dgs2 is not None:
        t10y2y = round(dgs10 - dgs2, 3)

    if t10y2y is None:
        return _SCORE["Neutral"], "Credit spread data unavailable"

    if t10y2y > 1.0:
        return _SCORE["Bullish"], f"Yield spread +{t10y2y:.2f}% — healthy credit conditions"
    elif t10y2y > 0:
        return _SCORE["Neutral"], f"Yield spread +{t10y2y:.2f}% — slightly positive"
    elif t10y2y > -0.5:
        return _SCORE["Bearish"], f"Yield spread {t10y2y:.2f}% — inverted, stress signal"
    else:
        return _SCORE["Very Bearish"], f"Yield spread {t10y2y:.2f}% — deeply inverted, significant credit stress"


def _score_breadth(spy_bars: list[dict], sample_tickers: list[str]) -> tuple[int, str]:
    """
    Market breadth: estimate % of watchlist stocks above their 200DMA.
    Uses the Alpaca/Polygon data for a sample (avoids expensive full-market call).
    """
    if not sample_tickers:
        # Fallback: use SPY price vs 200DMA as a single-name proxy
        if len(spy_bars) >= 200:
            closes = [b["close"] for b in spy_bars]
            ma200 = sum(closes[-200:]) / 200
            current = closes[-1]
            pct = 65 if current > ma200 else 40
            note = f"SPY proxy breadth: {'above' if current > ma200 else 'below'} 200DMA — estimated {pct}%"
            return _SCORE["Bullish"] if pct > 55 else _SCORE["Neutral"], note
        return _SCORE["Neutral"], "Breadth data unavailable"

    above = 0
    total = 0
    for ticker_bars in sample_tickers:
        if len(ticker_bars) >= 200:
            closes = [b["close"] for b in ticker_bars]
            ma200 = sum(closes[-200:]) / 200
            if closes[-1] > ma200:
                above += 1
            total += 1

    if total == 0:
        return _SCORE["Neutral"], "Breadth data unavailable"

    pct = above / total * 100
    if pct > 70:
        return _SCORE["Very Bullish"], f"{pct:.0f}% of sample above 200DMA — strong breadth"
    elif pct > 55:
        return _SCORE["Bullish"], f"{pct:.0f}% of sample above 200DMA — healthy breadth"
    elif pct > 40:
        return _SCORE["Neutral"], f"{pct:.0f}% of sample above 200DMA — neutral breadth"
    elif pct > 25:
        return _SCORE["Bearish"], f"{pct:.0f}% of sample above 200DMA — narrow breadth"
    else:
        return _SCORE["Very Bearish"], f"{pct:.0f}% of sample above 200DMA — very narrow breadth"


def _score_etf_flows(flows: dict) -> tuple[int, str]:
    """
    Scores market breadth from Unusual Whales ETF flow data.
    Bullish = money flowing into risk-on ETFs (SPY, QQQ, SOXX).
    Returns (raw_score, notes) using the same _SCORE scale.
    """
    if not flows:
        return _SCORE["Neutral"], "ETF flow data unavailable — using price proxy"

    bullish = sum(
        1 for f in flows.values()
        if f.get("call_ratio", 0.5) > 0.55
    )
    total = len(flows)
    if total == 0:
        return _SCORE["Neutral"], "No ETF flow data"

    ratio = bullish / total
    tickers = list(flows.keys())
    if ratio >= 0.8:
        return _SCORE["Very Bullish"], f"ETF flows bullish across {bullish}/{total} ETFs {tickers}"
    elif ratio >= 0.6:
        return _SCORE["Bullish"], f"ETF flows bullish in {bullish}/{total} ETFs {tickers}"
    elif ratio >= 0.4:
        return _SCORE["Neutral"], f"ETF flows mixed ({bullish}/{total} bullish) {tickers}"
    elif ratio >= 0.2:
        return _SCORE["Bearish"], f"ETF flows bearish in majority ({bullish}/{total} bullish) {tickers}"
    else:
        return _SCORE["Very Bearish"], f"ETF flows bearish across {total - bullish}/{total} ETFs {tickers}"


def _score_volume(spy_bars: list[dict]) -> tuple[int, str]:
    """
    Volume trend: compare recent volume on up days vs down days.
    Accumulation = more volume on up days.
    """
    if len(spy_bars) < 10:
        return _SCORE["Neutral"], "Insufficient data for volume analysis"

    recent = spy_bars[-20:]  # last 20 sessions
    up_vol   = sum(b["volume"] for b in recent if b["close"] > b["open"])
    down_vol = sum(b["volume"] for b in recent if b["close"] <= b["open"])

    if up_vol + down_vol == 0:
        return _SCORE["Neutral"], "No volume data"

    ratio = up_vol / (up_vol + down_vol)
    if ratio > 0.60:
        return _SCORE["Bullish"], f"Accumulation pattern: {ratio:.0%} of volume on up days"
    elif ratio < 0.40:
        return _SCORE["Bearish"], f"Distribution pattern: {ratio:.0%} of volume on up days"
    else:
        return _SCORE["Neutral"], f"Mixed volume: {ratio:.0%} on up days"


# ---------------------------------------------------------------------------
# Watchlist breadth sample (uses the price_history cache)
# ---------------------------------------------------------------------------

def _fetch_watchlist_bars_for_breadth(sample_size: int = 30) -> list[list[dict]]:
    """
    Returns cached daily-bar series for a sample of watchlist tickers, in the
    {"close": ...} shape _score_breadth expects. This was previously always
    called with an empty list (dead code) because fetching bars live for the
    whole watchlist on every regime check was too slow; the nightly
    price_history cache (ingestion/price_history.py) makes it cheap.
    Returns [] if the cache is empty (e.g. before the first nightly refresh
    has run) so the caller falls back to ETF flows / the SPY proxy.
    """
    try:
        import db
        from knowledge_loader import get_all_tickers

        tickers = get_all_tickers()[:sample_size]
        bulk = db.get_price_history_bulk(tickers, days=210)
        # Require enough history for a real 200DMA — otherwise this ticker
        # can't contribute to breadth and would just look artificially bearish.
        return [bars for bars in bulk.values() if len(bars) >= 200]
    except Exception as e:
        log.debug(f"Watchlist breadth sample unavailable: {e}")
        return []


# ---------------------------------------------------------------------------
# SPY bar fetcher (uses Alpaca via env — already available)
# ---------------------------------------------------------------------------

def _fetch_spy_bars(days: int = 220) -> list[dict]:
    """Fetch daily SPY bars via Alpaca REST. Returns list of {date, open, high, low, close, volume}."""
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        return []

    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days + 30)).strftime("%Y-%m-%d")

    url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params  = {"timeframe": "1Day", "start": start, "end": end, "limit": days + 30, "feed": "iex"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        bars_raw = resp.json().get("bars", [])
        return [
            {
                "date":   b["t"][:10],
                "open":   b["o"],
                "high":   b["h"],
                "low":    b["l"],
                "close":  b["c"],
                "volume": b["v"],
            }
            for b in bars_raw
        ]
    except Exception as e:
        log.warning(f"SPY bar fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------

def assess_regime() -> dict:
    """
    Full macro + market regime scoring.
    Returns a structured regime packet (matches market_regime_agent.md spec).
    Persists to DB and updates in-memory cache.
    """
    global _regime_cache, _cache_ts

    log.info("Assessing market regime...")
    now = datetime.now(timezone.utc)

    # --- Fetch data ---
    spy_bars = _fetch_spy_bars(days=220)

    try:
        from ingestion.macro import get_macro_snapshot
        macro = get_macro_snapshot()
    except Exception as e:
        log.warning(f"Macro fetch failed: {e}")
        macro = {}

    vix = macro.get("VIXCLS", {}).get("value")

    # --- Score each signal ---
    # Trend and breadth are computed before volatility because the VIX>40
    # contrarian-bullish signal must be suppressed when both are Very Bearish
    # (Death Cross posture) — see _score_volatility's docstring.
    trend_raw,      trend_notes      = _score_trend(spy_bars)

    watchlist_bars = _fetch_watchlist_bars_for_breadth()
    if watchlist_bars:
        breadth_raw, breadth_notes = _score_breadth(spy_bars, watchlist_bars)
    else:
        try:
            from ingestion.options_flow import get_etf_flows
            etf_flows = get_etf_flows()
            breadth_raw, breadth_notes = _score_etf_flows(etf_flows)
        except Exception:
            breadth_raw, breadth_notes = _score_breadth(spy_bars, [])

    volatility_raw, volatility_notes = _score_volatility(vix, trend_raw, breadth_raw)
    liquidity_raw,  liquidity_notes  = _score_liquidity(macro)
    credit_raw,     credit_notes     = _score_credit(macro)
    volume_raw,     volume_notes     = _score_volume(spy_bars)

    # --- Apply weights ---
    weighted_raw = (
        trend_raw      * _WEIGHTS["trend"]      +
        volatility_raw * _WEIGHTS["volatility"] +
        liquidity_raw  * _WEIGHTS["liquidity"]  +
        credit_raw     * _WEIGHTS["credit"]     +
        breadth_raw    * _WEIGHTS["breadth"]    +
        volume_raw     * _WEIGHTS["volume"]
    )

    score      = _normalize(weighted_raw)
    regime     = _regime_from_score(score)
    confidence = min(100.0, round(abs(score - 50) * 2, 1))  # higher when far from neutral

    # --- Identify key drivers & risks ---
    scores_map = {
        "trend":      (trend_raw,      trend_notes),
        "volatility": (volatility_raw, volatility_notes),
        "liquidity":  (liquidity_raw,  liquidity_notes),
        "credit":     (credit_raw,     credit_notes),
        "breadth":    (breadth_raw,    breadth_notes),
        "volume":     (volume_raw,     volume_notes),
    }
    key_drivers = [
        notes for name, (raw, notes) in scores_map.items() if raw >= 2
    ]
    key_risks = [
        notes for name, (raw, notes) in scores_map.items() if raw <= -2
    ]

    # --- Portfolio impact guidance ---
    _impact = {
        "Strong Risk-On": {
            "equity_exposure": "Increase",    "accelerator_sizing": "Increase",
            "cash_target_pct": 7,             "options_allowed": True,
            "leverage_allowed": True,
        },
        "Risk-On": {
            "equity_exposure": "Maintain",   "accelerator_sizing": "Increase",
            "cash_target_pct": 12,           "options_allowed": True,
            "leverage_allowed": False,
        },
        "Neutral": {
            "equity_exposure": "Maintain",   "accelerator_sizing": "Reduce",
            "cash_target_pct": 20,           "options_allowed": False,
            "leverage_allowed": False,
        },
        "Risk-Off": {
            "equity_exposure": "Reduce",     "accelerator_sizing": "Reduce",
            "cash_target_pct": 32,           "options_allowed": False,
            "leverage_allowed": False,
        },
        "Strong Risk-Off": {
            "equity_exposure": "Reduce",     "accelerator_sizing": "Exit",
            "cash_target_pct": 45,           "options_allowed": False,
            "leverage_allowed": False,
        },
    }
    portfolio_impact = _impact.get(regime, _impact["Neutral"])

    # --- Build reasoning ---
    signal_lines = "\n".join(
        f"  {name} ({_WEIGHTS[name]}x): score={raw} — {notes}"
        for name, (raw, notes) in scores_map.items()
    )
    reasoning = (
        f"Regime: {regime} | Score: {score}/100 | Confidence: {confidence:.0f}%\n\n"
        f"Signal breakdown:\n{signal_lines}\n\n"
        f"Key drivers: {'; '.join(key_drivers) if key_drivers else 'None'}\n"
        f"Key risks: {'; '.join(key_risks) if key_risks else 'None'}"
    )

    packet = {
        "agent":      "market_regime",
        "date":       now.strftime("%Y-%m-%d"),
        "timestamp":  now.isoformat(),
        "regime":     regime,
        "score":      score,
        "confidence": confidence,
        "signals": {
            "trend":      {"score": trend_raw,      "notes": trend_notes},
            "volatility": {"score": volatility_raw, "vix": vix, "notes": volatility_notes},
            "liquidity":  {"score": liquidity_raw,  "notes": liquidity_notes},
            "credit":     {"score": credit_raw,     "notes": credit_notes},
            "breadth":    {"score": breadth_raw,    "notes": breadth_notes},
            "volume":     {"score": volume_raw,     "notes": volume_notes},
        },
        "portfolio_impact": portfolio_impact,
        "key_drivers": key_drivers,
        "key_risks":   key_risks,
        "reasoning":   reasoning,
    }

    # --- Persist to DB ---
    try:
        import db
        db.log_regime(
            regime, score, confidence,
            packet["signals"],
            portfolio_impact,
            reasoning,
        )
    except Exception as e:
        log.warning(f"Failed to persist regime to DB: {e}")

    # --- Update cache ---
    _regime_cache = packet
    _cache_ts = time.monotonic()

    log.info(f"Regime assessment complete: {regime} (score={score})")
    return packet


def get_cached_regime() -> dict:
    """
    Returns cached regime if fresh (< REGIME_CACHE_TTL_SECONDS old).
    Falls back to DB, then calls assess_regime() if nothing available.
    """
    global _regime_cache, _cache_ts

    # In-memory cache hit
    if _regime_cache and (time.monotonic() - _cache_ts) < REGIME_CACHE_TTL_SECONDS:
        return _regime_cache

    # DB fallback
    try:
        import db
        db_regime = db.get_latest_regime()
        if db_regime:
            # Check if DB entry is recent enough
            ts = db_regime.get("timestamp")
            if ts:
                age = (datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)).total_seconds()
                if age < REGIME_CACHE_TTL_SECONDS:
                    # Reconstruct packet from DB row
                    _regime_cache = {
                        "agent":      "market_regime",
                        "regime":     db_regime["regime"],
                        "score":      db_regime["score"],
                        "confidence": db_regime["confidence"],
                        "signals":    db_regime.get("signals_json") or {},
                        "portfolio_impact": db_regime.get("portfolio_impact_json") or {},
                        "reasoning":  db_regime["reasoning"],
                        "timestamp":  str(ts),
                    }
                    _cache_ts = time.monotonic()
                    return _regime_cache
    except Exception as e:
        log.debug(f"DB regime fallback failed: {e}")

    # Nothing fresh — run fresh assessment
    return assess_regime()


def get_intraday_vix() -> float | None:
    """
    Live VIX via yfinance ^VIX — FRED's VIXCLS is end-of-day and lags by a
    full session, which is too stale to drive an "intraday" reassessment
    trigger (poll_macro was comparing yesterday's close against itself).
    """
    try:
        import yfinance as yf
        price = yf.Ticker("^VIX").fast_info.last_price
        return float(price) if price else None
    except Exception as e:
        log.debug(f"Intraday VIX fetch failed: {e}")
        return None


def should_reassess(new_vix: float) -> bool:
    """
    Returns True if the VIX has moved > 5 points since the last regime assessment.
    Used by poll_macro to trigger intraday regime reassessment on macro shocks.
    """
    if _regime_cache is None:
        return False
    last_vix = (_regime_cache.get("signals", {})
                             .get("volatility", {})
                             .get("vix"))
    if last_vix is None:
        return False
    return abs(new_vix - last_vix) > 5


def get_regime_multiplier(regime: str | None = None) -> float:
    """Convenience: returns the position sizing multiplier for a regime."""
    from config import REGIME_MULTIPLIER
    if regime is None:
        regime = get_cached_regime().get("regime", "Neutral")
    return REGIME_MULTIPLIER.get(regime, 0.75)
