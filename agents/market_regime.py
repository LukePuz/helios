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
_WEIGHTS = {"trend": 1, "volatility": 2, "liquidity": 2, "credit": 2, "breadth": 1, "volume": 1, "markov": 1}

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


# Empirical: a genuinely clean trend clears this; pure chop/whipsaw sits
# well below it. Not weight-tuned against historical data yet -- a
# reasonable starting point, revisit once trade_style has run against
# more real regime transitions (see docstring below for the motivating
# example this was built from).
TRADE_STYLE_ER_THRESHOLD = 0.35


def _score_trade_style(spy_bars: list[dict], lookback: int = 20) -> tuple[str, float, str]:
    """
    Classifies TRADE STYLE -- trending/expanding vs choppy/distribution --
    a dimension orthogonal to the risk-on/off score above. Risk-on/off
    answers "bullish or bearish"; this answers "is there a clean,
    tradeable trend right now, or is price chopping/topping without one."
    Added 2026-08-29 (user request), motivated by a concrete real example:
    Q2 2026 was a clean uptrend that favored breakout/momentum entries
    with wide trailing stops (this system's current default posture); by
    June/July it flipped to a choppy, distributive market where the same
    buy-and-hold posture rode gains up and gave them back. Nothing in the
    regime output distinguished the two -- assess_regime()'s score stayed
    wherever the risk-on/off signals put it, with no signal for "the kind
    of market that rewards momentum entries is over."

    Uses Kaufman's Efficiency Ratio: net move over `lookback` sessions
    divided by the sum of every bar-to-bar move in that window. 1.0 is a
    perfectly smooth trend with zero chop; near 0 is pure noise with no
    net progress regardless of how much the price moved intraday.
    Combined with the 20DMA's own slope (rising vs falling over the prior
    10 sessions), since a HIGH efficiency ratio during a FALLING 20DMA is
    exactly the "topping" case -- a smooth move, just not an upward one
    worth chasing with a momentum entry.

    Returns (label, efficiency_ratio, notes). label is one of
    "Trending/Expanding" or "Choppy/Distribution" -- deliberately binary,
    not a 5-level scale like the risk-on/off score, since this is meant
    to gate a concrete behavior switch (see config.py's
    PROFIT_TARGET_PCT_BY_STYLE / STALE_WINDOW_DAYS_BY_STYLE), not another
    shade of sentiment.
    """
    if len(spy_bars) < lookback + 10:
        return ("Choppy/Distribution", 0.0,
                "Insufficient bar data for trade-style analysis (defaulting cautious)")

    closes = [b["close"] for b in spy_bars]
    window = closes[-lookback:]
    net_move = abs(window[-1] - window[0])
    path_length = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    efficiency_ratio = round(net_move / path_length, 3) if path_length > 0 else 0.0

    sma20_now = sum(closes[-20:]) / 20
    sma20_10d_ago = sum(closes[-30:-10]) / 20
    slope_up = sma20_now > sma20_10d_ago

    if efficiency_ratio >= TRADE_STYLE_ER_THRESHOLD and slope_up:
        label = "Trending/Expanding"
        notes = (f"Efficiency ratio {efficiency_ratio:.2f} (>= {TRADE_STYLE_ER_THRESHOLD}) with a "
                  f"rising 20DMA ({sma20_now:.2f} vs {sma20_10d_ago:.2f} 10d ago) -- clean, "
                  f"tradeable uptrend")
    elif efficiency_ratio >= TRADE_STYLE_ER_THRESHOLD:
        label = "Choppy/Distribution"
        notes = (f"Efficiency ratio {efficiency_ratio:.2f} is trend-like but the 20DMA is falling "
                  f"({sma20_now:.2f} vs {sma20_10d_ago:.2f} 10d ago) -- a smooth move, just not one "
                  f"worth chasing with momentum entries (distribution/breakdown, not expansion)")
    else:
        label = "Choppy/Distribution"
        notes = (f"Efficiency ratio {efficiency_ratio:.2f} (< {TRADE_STYLE_ER_THRESHOLD}) -- no "
                  f"clean net progress, price is chopping/whipsawing over the last {lookback} sessions")

    return label, efficiency_ratio, notes


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


def _score_relative_strength(
    sector_bars: list[dict], benchmark_bars: list[dict],
    medium_lookback: int = 30, short_lookback: int = 5,
) -> tuple[int, str]:
    """
    Sector-vs-benchmark relative strength: the sector_etf/benchmark price
    ratio and its trend. Added 2026-07-26 -- the direct answer to "could we
    have caught the June 2026 tech rotation early": SOXX ran +28.9% vs SPY
    from 2026-05-15 to 2026-06-22, then reversed -9.9% relative in the next
    4 sessions, while SPY itself stayed flat (98-102 range) the entire time
    -- completely invisible to every SPY-only signal above. A rotation
    shows up in relative performance before it's obvious in absolute price,
    which is exactly what this catches: not just "is the sector falling"
    but "is money leaving this sector for others."

    Also flags plain overextension (medium_chg > 20%) as Neutral rather
    than Very Bullish even with no reversal yet -- an extended run being
    scored as straightforwardly bullish is how you miss the setup, not
    just the reversal.

    short_lookback=5 (not 10): backtested against the actual June 2026
    reversal, a 10-day window still spanned back into the tail end of the
    rally itself (e.g. as-of 2026-06-26, day-10 was 2026-06-11 -- before
    the 06-22 peak), netting positive and completely missing the rollover
    on every date checked 06-23 through 06-30. A 5-day window isolates the
    acute reversal and correctly flags it on 06-26. Tradeoff: 5-day is
    noisier -- it also fired one ~3-day false alarm in mid-May during an
    otherwise-intact uptrend -- but since this is one of several weighted
    signals feeding an aggregate score (not a standalone trigger), missing
    a real multi-week rotation is the more expensive failure mode.
    """
    sector_by_date = {b["date"]: b["close"] for b in sector_bars if b.get("close")}
    bench_by_date  = {b["date"]: b["close"] for b in benchmark_bars if b.get("close")}
    common_dates   = sorted(set(sector_by_date) & set(bench_by_date))

    if len(common_dates) < medium_lookback + 1:
        return _SCORE["Neutral"], "Insufficient overlapping history for relative strength"

    ratio = [sector_by_date[d] / bench_by_date[d] for d in common_dates if bench_by_date[d]]
    if len(ratio) < medium_lookback + 1:
        return _SCORE["Neutral"], "Insufficient ratio history for relative strength"

    medium_chg = (ratio[-1] - ratio[-medium_lookback]) / ratio[-medium_lookback] if ratio[-medium_lookback] else 0.0
    short_chg  = (ratio[-1] - ratio[-short_lookback]) / ratio[-short_lookback] if ratio[-short_lookback] else 0.0

    if medium_chg > 0.20:
        if short_chg < -0.03:
            return _SCORE["Very Bearish"], (
                f"Sector extended +{medium_chg:.1%} vs benchmark over {medium_lookback}d, now "
                f"rolling over {short_chg:+.1%} in the last {short_lookback}d — rotation risk"
            )
        return _SCORE["Neutral"], (
            f"Sector extended +{medium_chg:.1%} vs benchmark over {medium_lookback}d — "
            f"historically stretched, elevated reversal risk despite still leading"
        )
    if medium_chg > 0.10:
        if short_chg < -0.02:
            return _SCORE["Bearish"], (
                f"Sector +{medium_chg:.1%} vs benchmark over {medium_lookback}d but rolling over "
                f"{short_chg:+.1%} recently — possible rotation starting"
            )
        return _SCORE["Bullish"], f"Sector leading benchmark +{medium_chg:.1%} over {medium_lookback}d, trend intact"
    if medium_chg > 0:
        label = "Bullish" if short_chg > 0 else "Neutral"
        return _SCORE[label], f"Sector modestly leading benchmark +{medium_chg:.1%} over {medium_lookback}d"
    if short_chg < -0.02:
        return _SCORE["Very Bearish"], (
            f"Sector underperforming benchmark {medium_chg:+.1%} over {medium_lookback}d, "
            f"accelerating {short_chg:+.1%} recently"
        )
    return _SCORE["Bearish"], f"Sector underperforming benchmark {medium_chg:+.1%} over {medium_lookback}d"


def _score_price_action_volume(bars: list[dict], ticker: str = "") -> tuple[int, str]:
    """Reuses agents/price_action.py's OBV/swing-structure analysis (already
    built and wired into the scanner's stock-level scoring) at the sector-
    ETF level -- distinguishes a real, volume-confirmed reversal from an
    ordinary pullback, instead of adding a second parallel implementation."""
    try:
        from agents.price_action import compute_price_action
        pa = compute_price_action(ticker=ticker, bars=bars)
    except Exception as e:
        return _SCORE["Neutral"], f"Price-action analysis unavailable: {e}"
    if not pa:
        return _SCORE["Neutral"], "Insufficient history for price-action analysis"

    # Map price_action's 1-5 scale onto this module's -5..+5 scale.
    score_map = {1: "Very Bearish", 2: "Bearish", 3: "Neutral", 4: "Bullish", 5: "Very Bullish"}
    label = score_map.get(pa["score"], "Neutral")
    return _SCORE[label], pa["summary"]


def _score_markov(packet: dict, label: str = "SPY") -> tuple[int, str]:
    """
    Empirical Markov transition-persistence signal for 20-day return
    states. See agents/markov_regime.py for the transition-matrix estimation
    and knowledge/market_regime_rules.md for the threshold rationale.
    `label` identifies the underlying series in the notes text (SPY, a
    sector ETF ticker, or "RS ratio" for the relative-strength series).
    """
    if not packet.get("available"):
        return _SCORE["Neutral"], packet.get("notes") or "Markov data unavailable"

    state = packet.get("state", "unknown")
    row_samples = packet.get("row_samples", 0)
    if row_samples < 10:
        return _SCORE["Neutral"], (
            f"Markov signal suppressed — only {row_samples} historical samples "
            f"for state '{state}'"
        )

    signal     = packet.get("signal", 0.0)
    stickiness = packet.get("stickiness", 0.0)
    notes = (
        f"{label} Markov state={state}, stickiness={stickiness:.0%}, "
        f"signal={signal:+.2f} (n={row_samples})"
    )

    if signal >= 0.30:
        return _SCORE["Very Bullish"], notes
    elif signal >= 0.10:
        return _SCORE["Bullish"], notes
    elif signal > -0.10:
        return _SCORE["Neutral"], notes
    elif signal > -0.30:
        return _SCORE["Bearish"], notes
    else:
        return _SCORE["Very Bearish"], notes


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

def _fetch_bars(ticker: str = "SPY", days: int = 220) -> list[dict]:
    """Fetch daily bars for any ticker via Alpaca REST. Returns list of
    {date, open, high, low, close, volume}. Generalized 2026-07-26 from a
    SPY-only fetcher so sector ETFs (XLK, SOXX) can reuse the exact same
    trend/volume/relative-strength scoring the overall regime already uses,
    instead of a broad-market-only view of a book that's 100% AI/tech."""
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        return []

    end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days + 30)).strftime("%Y-%m-%d")

    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
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
        log.warning(f"{ticker} bar fetch failed: {e}")
        return []


def _fetch_spy_bars(days: int = 220) -> list[dict]:
    """Thin backward-compatible wrapper -- kept so the one existing call
    site in assess_regime() doesn't need to change."""
    return _fetch_bars("SPY", days)


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
    trade_style, efficiency_ratio, trade_style_notes = _score_trade_style(spy_bars)

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

    try:
        from agents.markov_regime import get_markov_packet
        markov_packet = get_markov_packet()
    except Exception as e:
        log.warning(f"Markov regime fetch failed: {e}")
        markov_packet = {"available": False, "notes": str(e)}
    markov_raw, markov_notes = _score_markov(markov_packet)

    # --- Apply weights ---
    weighted_raw = (
        trend_raw      * _WEIGHTS["trend"]      +
        volatility_raw * _WEIGHTS["volatility"] +
        liquidity_raw  * _WEIGHTS["liquidity"]  +
        credit_raw     * _WEIGHTS["credit"]     +
        breadth_raw    * _WEIGHTS["breadth"]    +
        volume_raw     * _WEIGHTS["volume"]     +
        markov_raw     * _WEIGHTS["markov"]
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
        "markov":     (markov_raw,     markov_notes),
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

    # --- Markov signal detail (only when the packet was actually available) ---
    markov_signal = {"score": markov_raw, "notes": markov_notes}
    if markov_packet.get("available"):
        markov_signal["state"]      = markov_packet.get("state")
        markov_signal["stickiness"] = markov_packet.get("stickiness")
        markov_signal["signal"]     = markov_packet.get("signal")

    # --- Build reasoning ---
    signal_lines = "\n".join(
        f"  {name} ({_WEIGHTS[name]}x): score={raw} — {notes}"
        for name, (raw, notes) in scores_map.items()
    )
    reasoning = (
        f"Regime: {regime} | Score: {score}/100 | Confidence: {confidence:.0f}%\n"
        f"Trade style: {trade_style} (efficiency ratio {efficiency_ratio:.2f}) — {trade_style_notes}\n\n"
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
        # Orthogonal to `regime` above (bullish/bearish) -- answers "is
        # there a clean trend to trade right now." Top-level, not just
        # inside `signals`, since it's meant to be read directly by
        # risk_manager.py to select which profit-target/stale-window
        # table applies (see config.py's *_BY_STYLE dicts), the same way
        # `regime` itself is read directly for the sizing multiplier.
        "trade_style": trade_style,
        "signals": {
            "trend":      {"score": trend_raw,      "notes": trend_notes},
            "trade_style": {"label": trade_style, "efficiency_ratio": efficiency_ratio,
                             "notes": trade_style_notes},
            "volatility": {"score": volatility_raw, "vix": vix, "notes": volatility_notes},
            "liquidity":  {"score": liquidity_raw,  "notes": liquidity_notes},
            "credit":     {"score": credit_raw,     "notes": credit_notes},
            "breadth":    {"score": breadth_raw,    "notes": breadth_notes},
            "volume":     {"score": volume_raw,     "notes": volume_notes},
            "markov":     markov_signal,
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


# ---------------------------------------------------------------------------
# Sector regime — added 2026-07-26. The overall regime above is entirely
# SPY/macro-based; every ticker in this system's watchlist (all 101, across
# all 15 thesis sections) is AI/tech-infrastructure themed, so a broad-
# market-neutral reading tells you very little about this book's actual
# risk. Confirmed against real data: SOXX ran +28.9% vs SPY from 2026-05-15
# to 2026-06-22, then reversed -9.9% relative in the next 4 sessions, while
# SPY stayed flat (98-102 range) the whole time and the overall regime
# stayed pinned at "Neutral" throughout. This is a second, independent
# assessment (own weights, own cache, own DB rows via regime_type='sector')
# -- it does not replace or change assess_regime() above.
# ---------------------------------------------------------------------------

_SECTOR_WEIGHTS = {
    "trend": 1, "volatility": 1, "breadth": 1, "volume": 1,
    "relative_strength": 3, "price_action_volume": 2, "markov": 1,
}
_SECTOR_MAX_RAW = sum(5 * w for w in _SECTOR_WEIGHTS.values())
_SECTOR_MIN_RAW = sum(-5 * w for w in _SECTOR_WEIGHTS.values())

_sector_regime_cache: dict[str, dict] = {}
_sector_cache_ts: dict[str, float] = {}


def _normalize_sector(raw: float) -> float:
    return round((raw - _SECTOR_MIN_RAW) / (_SECTOR_MAX_RAW - _SECTOR_MIN_RAW) * 100, 1)


def assess_sector_regime(
    sector_etf: str = "XLK",
    spy_bars: list[dict] | None = None,
    breadth_tickers: list[str] | None = None,
    regime_type: str = "sector",
) -> dict:
    """
    Same regime classification as assess_regime(), scoped to one sector ETF
    instead of SPY, plus two signals the overall regime doesn't have:
    relative-strength-vs-SPY (rotation detection) and OBV-confirmed
    price-action structure (agents/price_action.py, reused as-is).

    Liquidity/credit aren't sector-specific (macro conditions apply to the
    whole market) so they're deliberately left out here rather than
    duplicating the same macro read under a different label.

    breadth_tickers: explicit ticker list to sample for the breadth signal.
    Defaults to the AI/tech watchlist (the original XLK/SOXX use case,
    where that watchlist already IS the relevant universe). Added
    2026-07-26 for assess_market_sector_map()'s full 11-sector GICS sweep,
    where e.g. Financials' breadth needs real Financials constituents --
    the watchlist sample would otherwise silently repeat the same
    tech-breadth number under every unrelated sector's name.

    regime_type: passed through to db.log_regime, so the thesis-specific
    XLK/SOXX tracking (regime_type="sector", unchanged) and the broad GICS
    sweep (regime_type="gics_sector") don't collide as the same ticker can
    appear in both (XLK is both our thesis's sector ETF and the GICS
    Information Technology ETF).
    """
    from agents.markov_regime import get_markov_packet, get_markov_packet_from_closes

    log.info(f"Assessing {sector_etf} sector regime ({regime_type})...")
    now = datetime.now(timezone.utc)

    sector_bars = _fetch_bars(sector_etf, days=220)
    if spy_bars is None:
        spy_bars = _fetch_bars("SPY", days=220)

    trend_raw, trend_notes = _score_trend(sector_bars)

    try:
        from ingestion.macro import get_macro_snapshot
        macro = get_macro_snapshot()
    except Exception as e:
        log.warning(f"Macro fetch failed for sector regime: {e}")
        macro = {}
    vix = macro.get("VIXCLS", {}).get("value")
    volatility_raw, volatility_notes = _score_volatility(vix, trend_raw, None)

    if breadth_tickers is not None:
        try:
            import db
            bulk = db.get_price_history_bulk(breadth_tickers, days=210)
            breadth_bars = [bars for bars in bulk.values() if len(bars) >= 200]
        except Exception as e:
            log.debug(f"Sector breadth sample fetch failed for {sector_etf}: {e}")
            breadth_bars = []
    else:
        breadth_bars = _fetch_watchlist_bars_for_breadth()
    breadth_raw, breadth_notes = _score_breadth(sector_bars, breadth_bars)

    volume_raw, volume_notes = _score_volume(sector_bars)

    relative_strength_raw, relative_strength_notes = _score_relative_strength(sector_bars, spy_bars)

    price_action_raw, price_action_notes = _score_price_action_volume(sector_bars, sector_etf)

    # Markov on the RS ratio series itself (persistence of the ROTATION,
    # not just of the sector's own absolute price) -- the more direct
    # signal for "is this rotation likely to continue or mean-revert."
    # Needs ~10y of history for a reliable chain, same as get_markov_packet's
    # own SPY fetch -- the 220-day Alpaca bars used for trend/volume/RS above
    # are nowhere near enough, so this fetches its own long history via
    # yfinance rather than reusing sector_bars/spy_bars.
    markov_packet = None
    try:
        import yfinance as yf
        sector_hist = yf.Ticker(sector_etf).history(period="10y")["Close"].dropna()
        spy_hist    = yf.Ticker("SPY").history(period="10y")["Close"].dropna()
        sector_by_date = {d.strftime("%Y-%m-%d"): float(c) for d, c in sector_hist.items()}
        bench_by_date  = {d.strftime("%Y-%m-%d"): float(c) for d, c in spy_hist.items()}
        common_dates   = sorted(set(sector_by_date) & set(bench_by_date))
        ratio_series   = [sector_by_date[d] / bench_by_date[d] for d in common_dates if bench_by_date[d]]
        if len(ratio_series) >= 500:
            markov_packet = get_markov_packet_from_closes(
                ratio_series, label=f"{sector_etf}/SPY ratio", as_of=common_dates[-1],
            )
    except Exception as e:
        log.debug(f"RS-ratio Markov fetch failed for {sector_etf}: {e}")

    if not markov_packet or not markov_packet.get("available"):
        # Not enough aligned history for a reliable ratio-based chain --
        # fall back to the sector ETF's own absolute-price persistence
        # rather than forcing a thin, unreliable sample.
        markov_packet = get_markov_packet(sector_etf)
    markov_raw, markov_notes = _score_markov(markov_packet, label=sector_etf)

    weighted_raw = (
        trend_raw              * _SECTOR_WEIGHTS["trend"] +
        volatility_raw         * _SECTOR_WEIGHTS["volatility"] +
        breadth_raw             * _SECTOR_WEIGHTS["breadth"] +
        volume_raw              * _SECTOR_WEIGHTS["volume"] +
        relative_strength_raw   * _SECTOR_WEIGHTS["relative_strength"] +
        price_action_raw        * _SECTOR_WEIGHTS["price_action_volume"] +
        markov_raw              * _SECTOR_WEIGHTS["markov"]
    )
    score      = _normalize_sector(weighted_raw)
    regime     = _regime_from_score(score)
    confidence = min(100.0, round(abs(score - 50) * 2, 1))

    scores_map = {
        "trend":               (trend_raw, trend_notes),
        "volatility":          (volatility_raw, volatility_notes),
        "breadth":             (breadth_raw, breadth_notes),
        "volume":              (volume_raw, volume_notes),
        "relative_strength":   (relative_strength_raw, relative_strength_notes),
        "price_action_volume": (price_action_raw, price_action_notes),
        "markov":              (markov_raw, markov_notes),
    }
    key_drivers = [notes for _, (raw, notes) in scores_map.items() if raw >= 2]
    key_risks   = [notes for _, (raw, notes) in scores_map.items() if raw <= -2]

    signal_lines = "\n".join(
        f"  {name} ({_SECTOR_WEIGHTS[name]}x): score={raw} — {notes}"
        for name, (raw, notes) in scores_map.items()
    )
    reasoning = (
        f"{sector_etf} Sector Regime: {regime} | Score: {score}/100 | Confidence: {confidence:.0f}%\n\n"
        f"Signal breakdown:\n{signal_lines}\n\n"
        f"Key drivers: {'; '.join(key_drivers) if key_drivers else 'None'}\n"
        f"Key risks: {'; '.join(key_risks) if key_risks else 'None'}"
    )

    packet = {
        "agent": "market_regime_sector",
        "sector_etf": sector_etf,
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat(),
        "regime": regime,
        "score": score,
        "confidence": confidence,
        "signals": {name: {"score": raw, "notes": notes} for name, (raw, notes) in scores_map.items()},
        "key_drivers": key_drivers,
        "key_risks": key_risks,
        "reasoning": reasoning,
    }

    try:
        import db
        db.log_regime(
            regime, score, confidence, packet["signals"], {}, reasoning,
            regime_type=regime_type, sector_etf=sector_etf,
        )
    except Exception as e:
        log.warning(f"Failed to persist sector regime to DB: {e}")

    cache_key = f"{sector_etf}:{regime_type}"
    _sector_regime_cache[cache_key] = packet
    _sector_cache_ts[cache_key] = time.monotonic()

    log.info(f"{sector_etf} sector regime assessment complete: {regime} (score={score})")
    return packet


def get_cached_sector_regime(sector_etf: str = "XLK", regime_type: str = "sector") -> dict:
    """Same TTL-cache/DB-fallback pattern as get_cached_regime(), scoped
    per sector ETF (and per regime_type, since XLK is tracked both as our
    thesis's sector ETF and as the GICS Information Technology ETF)."""
    cache_key = f"{sector_etf}:{regime_type}"
    if cache_key in _sector_regime_cache and \
       (time.monotonic() - _sector_cache_ts.get(cache_key, 0)) < REGIME_CACHE_TTL_SECONDS:
        return _sector_regime_cache[cache_key]
    try:
        return assess_sector_regime(sector_etf, regime_type=regime_type)
    except Exception as e:
        log.error(f"assess_sector_regime({sector_etf}, {regime_type}) failed: {e}")
        return {"available": False, "regime": "Neutral", "score": 50.0, "confidence": 0.0,
                "reasoning": f"Sector regime unavailable: {e}"}


def assess_market_sector_map() -> dict:
    """
    Broad-market cross-sector regime map: assess_sector_regime() run across
    all 11 GICS/SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU,
    XLB, XLC, XLRE), each with a real breadth sample drawn from that
    sector's actual S&P 500 constituents.

    Distinct from the XLK/SOXX calls elsewhere in this module (regime_type
    "sector"): those track our specific AI/tech thesis against the
    watchlist as its own breadth universe. This is the "how's the whole
    market rotating" view -- logged as regime_type="gics_sector" so the two
    never collide even where the same ETF (XLK) appears in both. Pure
    price/volume signals, no LLM involved -- safe to run as a frequent,
    cheap ops-queue task.
    """
    from ingestion.sp500_universe import SECTOR_ETF_MAP, get_tickers_by_gics_sector

    log.info("Assessing full GICS sector map (11 sectors)...")
    spy_bars = _fetch_bars("SPY", days=220)

    results: dict[str, dict] = {}
    for gics_sector, etf in SECTOR_ETF_MAP.items():
        try:
            breadth_sample = get_tickers_by_gics_sector(gics_sector, limit=20)
            packet = assess_sector_regime(
                etf, spy_bars=spy_bars,
                breadth_tickers=breadth_sample, regime_type="gics_sector",
            )
            results[etf] = {**packet, "gics_sector": gics_sector}
        except Exception as e:
            log.warning(f"assess_sector_regime({etf}) failed in sector map: {e}")

    log.info(f"GICS sector map complete: {len(results)}/{len(SECTOR_ETF_MAP)} sectors assessed")
    return results


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
