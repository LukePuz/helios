"""
Price action & volume analysis — deterministic (not LLM-estimated), same
philosophy as agents/chase_risk.py: real technical-analysis math computed
in code, not asked of the model or eyeballed from a single moving-average
comparison.

Replaces two genuinely weak spots found 2026-07-26 while reviewing the
scanner's scoring:
  - _score_technical's "recent 20-day breakout/breakdown" flag was a
    single-bar check, not real trend structure. Swing-point analysis
    (higher-highs/higher-lows vs lower-highs/lower-lows -- classic Dow
    Theory) replaces it.
  - _score_momentum's volume signal was NON-DIRECTIONAL: any 5-day volume
    surge added +3 regardless of whether those were up-days or down-days,
    so a volume spike on a down move was scored as bullish. On-Balance
    Volume (OBV) trend fixes this -- it only confirms a price move when
    volume is actually flowing the same direction.

Used by both agents/scanner.py (wired into composite_score) and
agents/analyst.py (informational context in the prompt, like chase_risk).
"""
import logging

log = logging.getLogger(__name__)

SWING_WINDOW = 3       # bars on each side to qualify as a local swing point
TREND_LOOKBACK = 20    # bars for MA-slope / OBV-trend comparisons
MA_SLOPE_THRESHOLD = 0.01   # 1% over TREND_LOOKBACK counts as rising/falling


def _compute_obv(bars: list[dict]) -> list[float]:
    """Standard On-Balance Volume: cumulative volume, signed by whether
    that day closed up or down from the prior day."""
    obv = [0.0]
    for i in range(1, len(bars)):
        if bars[i]["close"] > bars[i - 1]["close"]:
            obv.append(obv[-1] + bars[i]["volume"])
        elif bars[i]["close"] < bars[i - 1]["close"]:
            obv.append(obv[-1] - bars[i]["volume"])
        else:
            obv.append(obv[-1])
    return obv


def _pct_direction(values: list[float], lookback: int, threshold: float) -> str:
    """rising/falling/flat from simple % change over `lookback` bars --
    appropriate for price-scale series (MAs, price itself) where a
    percentage threshold is meaningful."""
    if len(values) < lookback + 1 or not values[-lookback]:
        return "flat"
    change = (values[-1] - values[-lookback]) / abs(values[-lookback])
    if change > threshold:
        return "rising"
    if change < -threshold:
        return "falling"
    return "flat"


def _obv_direction(obv: list[float], bars: list[dict], lookback: int) -> str:
    """rising/falling/flat for OBV specifically -- OBV has no natural
    price-like scale (it can sit near zero), so a raw-change-vs-average-
    daily-volume epsilon is used instead of a percentage."""
    if len(obv) < lookback + 1:
        return "flat"
    change = obv[-1] - obv[-lookback]
    avg_daily_vol = sum(b["volume"] for b in bars[-lookback:]) / lookback if bars[-lookback:] else 0
    epsilon = avg_daily_vol * 2  # requires the equivalent of ~2 avg days of net volume flow
    if avg_daily_vol == 0:
        return "flat"
    if change > epsilon:
        return "rising"
    if change < -epsilon:
        return "falling"
    return "flat"


def _find_swings(bars: list[dict], window: int = SWING_WINDOW) -> list[tuple]:
    """Local swing highs/lows via a simple fractal: bar i qualifies if its
    high/low is the most extreme within +/-window bars around it."""
    swings = []
    n = len(bars)
    for i in range(window, n - window):
        seg = bars[i - window:i + window + 1]
        if bars[i]["high"] == max(b["high"] for b in seg):
            swings.append((i, bars[i]["high"], "high"))
        elif bars[i]["low"] == min(b["low"] for b in seg):
            swings.append((i, bars[i]["low"], "low"))
    return swings


def _swing_structure(bars: list[dict]) -> tuple[str, str]:
    """Dow-Theory trend structure: sequences of higher-highs+higher-lows
    (uptrend), lower-highs+lower-lows (downtrend), or anything else
    (mixed/consolidating). Returns (structure, note)."""
    swings = _find_swings(bars)
    if len(swings) < 4:
        return "insufficient_data", "not enough swing points to assess structure"

    # Collapse consecutive same-type swings to the more extreme one (a
    # fractal window can flag two nearby highs before a real low forms).
    collapsed = []
    for s in swings:
        if collapsed and collapsed[-1][2] == s[2]:
            if (s[2] == "high" and s[1] > collapsed[-1][1]) or (s[2] == "low" and s[1] < collapsed[-1][1]):
                collapsed[-1] = s
        else:
            collapsed.append(s)

    # Standard Dow Theory definition: compare only the last two highs and
    # last two lows, not three. Requiring three strictly-monotonic points
    # in a row is brittle against ordinary noisy price action -- a single
    # minor pullback breaks it and everything reads "mixed", which is
    # exactly what happened testing this against real data (every one of
    # 5 real tickers came back "mixed" with a 3-point requirement).
    highs = [s[1] for s in collapsed if s[2] == "high"][-2:]
    lows = [s[1] for s in collapsed if s[2] == "low"][-2:]

    if len(highs) < 2 or len(lows) < 2:
        return "insufficient_data", "not enough alternating swing points to assess structure"

    higher_highs = highs[0] < highs[1]
    higher_lows = lows[0] < lows[1]
    lower_highs = highs[0] > highs[1]
    lower_lows = lows[0] > lows[1]

    if higher_highs and higher_lows:
        return "uptrend", f"higher high ({highs[0]:.2f} -> {highs[1]:.2f}) and higher low ({lows[0]:.2f} -> {lows[1]:.2f})"
    if lower_highs and lower_lows:
        return "downtrend", f"lower high ({highs[0]:.2f} -> {highs[1]:.2f}) and lower low ({lows[0]:.2f} -> {lows[1]:.2f})"
    return "mixed", f"highs {highs[0]:.2f}->{highs[1]:.2f}, lows {lows[0]:.2f}->{lows[1]:.2f} -- no consistent higher-highs/higher-lows or lower-highs/lower-lows"


def compute_price_action(ticker: str, bars: list[dict] | None = None) -> dict | None:
    """
    Returns a dict with OBV trend, MA slopes, swing structure, and a
    combined 1-5 score + narrative summary -- or None if there isn't
    enough history yet.

    `bars` can be passed in directly (scanner.py already fetches them for
    other scoring) to avoid a redundant fetch; otherwise fetched fresh via
    trading_bot's own price history cache.
    """
    if bars is None:
        try:
            import db
            bars = db.get_price_history(ticker, days=220)
        except Exception as e:
            log.debug(f"[{ticker}] price_action: history fetch failed: {e}")
            return None

    if len(bars) < 60:
        return None

    closes = [b["close"] for b in bars]
    ma50 = [sum(closes[max(0, i - 49):i + 1]) / len(closes[max(0, i - 49):i + 1]) for i in range(len(closes))]
    ma200_window = min(200, len(closes))
    ma200 = [sum(closes[max(0, i - ma200_window + 1):i + 1]) / len(closes[max(0, i - ma200_window + 1):i + 1]) for i in range(len(closes))]

    ma50_dir = _pct_direction(ma50, TREND_LOOKBACK, MA_SLOPE_THRESHOLD)
    ma200_dir = _pct_direction(ma200, TREND_LOOKBACK, MA_SLOPE_THRESHOLD)
    price_dir = _pct_direction(closes, TREND_LOOKBACK, MA_SLOPE_THRESHOLD)

    obv = _compute_obv(bars)
    obv_dir = _obv_direction(obv, bars, TREND_LOOKBACK)

    structure, structure_note = _swing_structure(bars[-60:])

    # Volume confirmation: does OBV agree with where price is actually
    # heading? This is the direct fix for the old non-directional flag.
    if price_dir == "rising" and obv_dir == "rising":
        volume_confirmation = "confirmed"
    elif price_dir == "rising" and obv_dir == "falling":
        volume_confirmation = "bearish_divergence"
    elif price_dir == "falling" and obv_dir == "falling":
        volume_confirmation = "confirmed_down"
    elif price_dir == "falling" and obv_dir == "rising":
        volume_confirmation = "bullish_divergence"
    else:
        volume_confirmation = "neutral"

    # Combined 1-5 score, same convention as the other sub-scores (5 =
    # most supportive of a long, 1 = least).
    score = 3
    if structure == "uptrend":
        score += 1
    elif structure == "downtrend":
        score -= 1
    if ma200_dir == "rising":
        score += 1
    elif ma200_dir == "falling":
        score -= 1
    if volume_confirmation == "confirmed":
        score += 1
    elif volume_confirmation == "bearish_divergence":
        score -= 1
    elif volume_confirmation == "confirmed_down":
        # A genuinely confirmed downtrend -- distinct from bearish
        # divergence (rally with no volume support), but still bearish
        # for a long-only system and must not go unscored.
        score -= 1
    elif volume_confirmation == "bullish_divergence":
        # Accumulation despite price weakness -- a mild, early-reversal
        # signal, not full confirmation, so it gets half the weight of a
        # confirmed uptrend rather than the same +1.
        score += 0.5
    score = max(1, min(5, round(score)))

    tag = "BULLISH" if score >= 4 else "BEARISH" if score <= 2 else "NEUTRAL"

    summary_parts = []
    if structure == "uptrend":
        summary_parts.append(f"Price structure is a genuine uptrend ({structure_note})")
    elif structure == "downtrend":
        summary_parts.append(f"Price structure is a genuine downtrend ({structure_note})")
    else:
        summary_parts.append(f"Price structure is mixed/consolidating ({structure_note})")
    summary_parts.append(f"the 200-day average is itself {ma200_dir}")
    if volume_confirmation == "confirmed":
        summary_parts.append("volume (OBV) is confirming the uptrend")
    elif volume_confirmation == "bearish_divergence":
        summary_parts.append("volume (OBV) is NOT confirming this rally -- a bearish divergence")
    elif volume_confirmation == "confirmed_down":
        summary_parts.append("volume (OBV) is confirming the downtrend")
    elif volume_confirmation == "bullish_divergence":
        summary_parts.append("volume (OBV) is accumulating despite price weakness -- a possible early reversal signal")
    else:
        summary_parts.append("volume (OBV) is neutral/inconclusive")

    return {
        "score": score,
        "tag": tag,
        "structure": structure,
        "ma50_direction": ma50_dir,
        "ma200_direction": ma200_dir,
        "volume_confirmation": volume_confirmation,
        "summary": "; ".join(summary_parts) + ".",
    }


# ---------------------------------------------------------------------------
# Trend Template (2026-09-11, user request)
# ---------------------------------------------------------------------------
# Mark Minervini's 8-point "Trend Template" for a confirmed Stage 2 uptrend,
# from his publicly documented SEPA criteria (not this codebase's own
# invention) -- a deterministic, ALL-8-OR-NOTHING gate, distinct from
# compute_price_action's above SUBJECTIVE 1-5 score (structure/OBV/MA-slope
# used as context, not a pass/fail filter). Wired into agents/analyst.py as
# a hard gate for Accelerator OPEN/ADD, same "prompt instruction alone isn't
# reliable enough" reasoning as the UDR/multi-source/expected-return-rate
# gates there.

TREND_TEMPLATE_RS_LOOKBACK = 252            # ~1 trading year -- both the 52-week hi/lo window and the RS-proxy return window
TREND_TEMPLATE_MA200_SLOPE_LOOKBACK = 20    # ~1 calendar month of trading days, matches "rising for at least 1 month"
TREND_TEMPLATE_MIN_ABOVE_52W_LOW = 0.25     # price >= 25% above the 52-week low
TREND_TEMPLATE_MAX_BELOW_52W_HIGH = 0.25    # price within 25% of the 52-week high
# The real 8th criterion (IBD Relative Strength ranking >= 70) is a
# PERCENTILE across the entire market -- computing that properly means
# ranking this ticker's trailing return against every other stock's, a
# separate and much larger undertaking (see agents/market_regime.py's
# SECTOR_ETF_MAP-style universe sweeps for the shape that would take).
# This is a proxy, not a replication: trailing-year return vs SPY's same-
# window return, requiring real outperformance -- directionally the same
# idea (market-relative strength, not just "the stock went up"), but never
# confused with the literal IBD percentile in the returned reasoning.
RS_PROXY_MIN_OUTPERFORMANCE = 0.0   # must simply be beating SPY, not just close to it


def compute_trend_template(
    ticker: str, bars: list[dict] | None = None, spy_bars: list[dict] | None = None,
) -> dict | None:
    """
    Returns {"pass": bool, "criteria": {name: bool, ...}, "reasoning": str}
    or None if there isn't enough history yet (mirrors compute_price_action's
    own None-on-insufficient-data convention).

    `bars`/`spy_bars` can be passed in to avoid a redundant fetch; otherwise
    fetched fresh via trading_bot's own price history cache.
    """
    if bars is None:
        try:
            import db
            bars = db.get_price_history(ticker, days=TREND_TEMPLATE_RS_LOOKBACK + TREND_TEMPLATE_MA200_SLOPE_LOOKBACK + 10)
        except Exception as e:
            log.debug(f"[{ticker}] trend_template: history fetch failed: {e}")
            return None
    if spy_bars is None:
        try:
            import db
            spy_bars = db.get_price_history("SPY", days=TREND_TEMPLATE_RS_LOOKBACK + 10)
        except Exception as e:
            log.debug(f"[{ticker}] trend_template: SPY history fetch failed: {e}")
            spy_bars = None

    min_bars_needed = TREND_TEMPLATE_RS_LOOKBACK + TREND_TEMPLATE_MA200_SLOPE_LOOKBACK
    if len(bars) < min_bars_needed:
        return None

    closes = [b["close"] for b in bars]
    price = closes[-1]

    def _sma(window: int) -> list[float]:
        return [sum(closes[max(0, i - window + 1):i + 1]) / len(closes[max(0, i - window + 1):i + 1])
                for i in range(len(closes))]

    ma50, ma150, ma200 = _sma(50), _sma(150), _sma(200)

    window = closes[-TREND_TEMPLATE_RS_LOOKBACK:]
    low_52w, high_52w = min(window), max(window)

    ma200_slope_rising = (
        len(ma200) > TREND_TEMPLATE_MA200_SLOPE_LOOKBACK
        and ma200[-1] > ma200[-TREND_TEMPLATE_MA200_SLOPE_LOOKBACK]
    )

    pct_above_low = (price - low_52w) / low_52w if low_52w else 0.0
    pct_below_high = (high_52w - price) / high_52w if high_52w else 1.0

    rs_proxy_pass = None
    rs_note = "SPY history unavailable -- RS proxy skipped (fails closed, see below)"
    if spy_bars and len(spy_bars) >= TREND_TEMPLATE_RS_LOOKBACK:
        spy_closes = [b["close"] for b in spy_bars][-TREND_TEMPLATE_RS_LOOKBACK:]
        ticker_return = (window[-1] - window[0]) / window[0] if window[0] else 0.0
        spy_return = (spy_closes[-1] - spy_closes[0]) / spy_closes[0] if spy_closes[0] else 0.0
        outperformance = ticker_return - spy_return
        rs_proxy_pass = outperformance > RS_PROXY_MIN_OUTPERFORMANCE
        rs_note = (f"trailing-year return {ticker_return:+.1%} vs SPY {spy_return:+.1%} "
                   f"({outperformance:+.1%} relative -- RS proxy, not the literal IBD percentile)")

    criteria = {
        "price_above_150dma_and_200dma": price > ma150[-1] and price > ma200[-1],
        "150dma_above_200dma":           ma150[-1] > ma200[-1],
        "200dma_rising_1_month":         ma200_slope_rising,
        "50dma_above_150dma_and_200dma": ma50[-1] > ma150[-1] and ma50[-1] > ma200[-1],
        "price_above_50dma":             price > ma50[-1],
        "above_52w_low_by_25pct":        pct_above_low >= TREND_TEMPLATE_MIN_ABOVE_52W_LOW,
        "within_25pct_of_52w_high":      pct_below_high <= TREND_TEMPLATE_MAX_BELOW_52W_HIGH,
        # Fails closed when SPY data is unavailable rather than silently
        # dropping to a 7-point check -- same "fail toward re-checking, not
        # silently skipping" posture used elsewhere in this codebase.
        "relative_strength_proxy":       bool(rs_proxy_pass),
    }
    all_pass = all(criteria.values())

    failed = [k for k, v in criteria.items() if not v]
    reasoning = (
        "All 8 Trend Template criteria met -- confirmed Stage 2 uptrend."
        if all_pass else
        f"Failed {len(failed)}/8 Trend Template criteria: {', '.join(failed)}. {rs_note}"
    )

    return {"pass": all_pass, "criteria": criteria, "reasoning": reasoning}
