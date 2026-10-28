"""
Trade ladder — deterministic (not LLM-estimated), same house style as
agents/chase_risk.py and agents/price_action.py. Computes a per-ticker set
of technical price levels ("rungs": swing points, moving averages, base
floor/ceiling, volume nodes, gap levels) plus a valuation layer (peer-
implied fair value, Street consensus target, valuation tier), merged into
one ordered ladder.

Built 2026-09-13 as part of the swing-trading strategy redesign (see
agents/setup_gates.py for how the three setup categories -- Breakout,
Catalyst, Continuation -- read stops/targets from this instead of the old
flat-%/ATR-multiple-only approach). Runs nightly (tasks.refresh_trade_ladder)
and writes one row per ticker to db.trade_ladder via db.upsert_trade_ladder.

Nothing in this module places an order or changes sizing on its own --
it only computes and persists levels. agents/setup_gates.py and
risk_manager.py are the consumers.
"""
import logging

log = logging.getLogger(__name__)

# How many bars a base-contraction / volume-node scan looks back over.
BASE_LOOKBACK = 60
VOLUME_NODE_LOOKBACK = 90
VOLUME_NODE_TOP_K = 4

# A leg's range must be <= this fraction of the PRIOR leg's range to count
# as a genuine contraction (VCP definition) -- 0.75 is deliberately loose
# (each leg only needs to be meaningfully tighter, not dramatically so);
# real Minervini-style bases often contract more sharply than this, but a
# looser threshold here just means base_contractions undercounts on a
# textbook base rather than missing a looser-but-real one.
CONTRACTION_RATIO = 0.75

# Fixed-dollar-risk-style gap threshold: a gap must be at least this many
# ATRs (scaled to the ticker's own volatility, not a flat %) OR this flat
# floor, whichever is larger -- the floor exists only to filter noise on
# low-ATR/illiquid names where 1x ATR might be under 1%.
GAP_ATR_MULTIPLE = 1.0
GAP_MIN_PCT_FLOOR = 0.03
GAP_MIN_VOLUME_RATIO = 2.0

# Rungs within this many ATRs of each other merge into one (confluence).
RUNG_MERGE_ATR_FRACTION = 0.5
RUNG_BAND_ATR_FRACTION = 0.25

# Valuation tiers now live in agents/valuation.py (single source of
# truth for the 4-method fundamentals ensemble that replaced the original
# 2-anchor peer+Street calc here) -- see compute_fair_value below.


def _sanitize_bars(bars: list[dict]) -> list[dict]:
    """Carries the previous valid bar's OHLC forward over any bar with a
    non-finite (NaN/inf) value -- found 2026-09-13 running this against
    real cached data: NVDA's price_history has one bad close several
    months back (an ingestion-side glitch, not new). SMA-only calcs
    elsewhere in this codebase (price_action.py) are naturally immune
    since each window only looks backward a fixed distance, but EMA is
    cumulative -- once poisoned, EVERY value after it is NaN forever, not
    just the affected window. This is why the pre-existing bug never
    surfaced there but broke every ma10/ma21/52w rung here immediately."""
    import math
    cleaned = []
    last_good: dict | None = None
    for b in bars:
        values = (b.get("open"), b.get("high"), b.get("low"), b.get("close"))
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
            if last_good is None:
                continue  # no prior good bar to carry forward -- drop it
            b = {**b, "open": last_good["open"], "high": last_good["high"],
                 "low": last_good["low"], "close": last_good["close"]}
        cleaned.append(b)
        last_good = b
    return cleaned


def _compute_atr14(bars: list[dict]) -> float | None:
    """Standard 14-day True Range ATR off cached daily bars (db.get_price_
    history's high/low/close shape) -- deliberately NOT risk_manager.
    _compute_atr, which fetches live from IBKR and expects a different bar
    shape; the nightly ladder refresh should work entirely off the cached
    price_history table, no live broker dependency."""
    if len(bars) < 15:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, prev_c = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    window = trs[-14:]
    return sum(window) / len(window) if window else None


def _sma(closes: list[float], period: int) -> list[float]:
    return [sum(closes[max(0, i - period + 1):i + 1]) / len(closes[max(0, i - period + 1):i + 1])
            for i in range(len(closes))]


def _ema(closes: list[float], period: int) -> list[float]:
    if not closes:
        return []
    k = 2 / (period + 1)
    out = [closes[0]]
    for c in closes[1:]:
        out.append(c * k + out[-1] * (1 - k))
    return out


# ---------------------------------------------------------------------------
# Base contraction (VCP) detector
# ---------------------------------------------------------------------------

def compute_base_contraction(bars: list[dict], lookback: int = BASE_LOOKBACK) -> dict | None:
    """
    Returns {"base_detected", "base_contractions", "base_ceiling",
    "base_floor", "tightness_pct", "base_volume_dryup"} or None if there
    isn't enough swing structure yet. Reuses price_action's own swing-
    finding (same fractal window/collapsing logic already validated
    there) rather than reimplementing it.
    """
    if len(bars) < lookback:
        return None
    window = bars[-lookback:]

    from agents.price_action import _find_swings
    swings = _find_swings(window)
    if len(swings) < 4:
        return None

    collapsed = []
    for s in swings:
        if collapsed and collapsed[-1][2] == s[2]:
            if (s[2] == "high" and s[1] > collapsed[-1][1]) or (s[2] == "low" and s[1] < collapsed[-1][1]):
                collapsed[-1] = s
        else:
            collapsed.append(s)

    legs = []
    for i in range(1, len(collapsed)):
        idx0, price0, type0 = collapsed[i - 1]
        idx1, price1, type1 = collapsed[i]
        if type0 == type1:
            continue
        high, low = max(price0, price1), min(price0, price1)
        if high <= 0:
            continue
        legs.append({
            "start_idx": idx0, "end_idx": idx1,
            "range_pct": (high - low) / high, "high": high, "low": low,
        })

    if len(legs) < 2:
        return None

    # Walk backward from the most recent leg, counting how many
    # consecutive legs each contract vs. the one before it.
    contractions = 1
    for i in range(len(legs) - 1, 0, -1):
        prior_range = legs[i - 1]["range_pct"]
        if prior_range > 0 and legs[i]["range_pct"] <= CONTRACTION_RATIO * prior_range:
            contractions += 1
        else:
            break

    base_detected = contractions >= 2
    considered = legs[-contractions:] if contractions >= 2 else legs[-2:]
    base_ceiling = max(l["high"] for l in considered)
    base_floor = min(l["low"] for l in considered)
    tightness_pct = legs[-1]["range_pct"]

    def _avg_vol(leg: dict) -> float:
        seg = window[leg["start_idx"]:leg["end_idx"] + 1]
        return sum(b["volume"] for b in seg) / len(seg) if seg else 0.0

    first_vol, last_vol = _avg_vol(considered[0]), _avg_vol(considered[-1])
    volume_dryup = last_vol < first_vol if first_vol else False

    return {
        "base_detected":     base_detected,
        "base_contractions": contractions,
        "base_ceiling":      round(base_ceiling, 2),
        "base_floor":        round(base_floor, 2),
        "tightness_pct":     round(tightness_pct, 4),
        "base_volume_dryup": volume_dryup,
    }


# ---------------------------------------------------------------------------
# Volume-node levels (lightweight volume profile)
# ---------------------------------------------------------------------------

def compute_volume_nodes(
    bars: list[dict], atr: float | None, lookback: int = VOLUME_NODE_LOOKBACK, top_k: int = VOLUME_NODE_TOP_K,
) -> list[dict]:
    """
    Buckets price into ATR/4-wide bins (tied to the ticker's own
    volatility, not a fixed dollar or % width) and returns the top-K
    local-maxima buckets by volume, as rungs. Each day's volume is
    assigned to the bucket containing (H+L+C)/3 -- a reasonable single-
    point approximation since true intrabar volume distribution isn't
    available from daily bars.
    """
    if not atr or atr <= 0 or len(bars) < 20:
        return []
    window = bars[-lookback:]
    bucket_width = atr / 4
    if bucket_width <= 0:
        return []

    buckets: dict[int, float] = {}
    for b in window:
        typical = (b["high"] + b["low"] + b["close"]) / 3
        key = round(typical / bucket_width)
        buckets[key] = buckets.get(key, 0.0) + b["volume"]

    if not buckets:
        return []

    keys_sorted = sorted(buckets.keys())
    maxima = []
    for i, k in enumerate(keys_sorted):
        v = buckets[k]
        left = buckets.get(keys_sorted[i - 1]) if i > 0 else None
        right = buckets.get(keys_sorted[i + 1]) if i < len(keys_sorted) - 1 else None
        if (left is None or v >= left) and (right is None or v >= right):
            maxima.append((k, v))

    maxima.sort(key=lambda x: x[1], reverse=True)
    top = maxima[:top_k]
    max_vol = max(buckets.values())

    return [
        {"price": round(k * bucket_width, 2), "types": ["volume_node"], "relative_weight": round(v / max_vol, 3)}
        for k, v in top
    ]


# ---------------------------------------------------------------------------
# Gap + volume Catalyst detector
# ---------------------------------------------------------------------------

def compute_gap_catalyst(bars: list[dict], atr: float | None) -> dict:
    """
    Returns {"gap_catalyst_active", "gap_pct", "gap_day_high",
    "gap_day_low", "volume_ratio"}. gap_catalyst_active requires BOTH an
    ATR-scaled gap (not a flat %) AND volume >= 2x the 20-day average --
    two independent confirmations, same principle as the analyst's own
    multi-source corroboration gate, so a single unexplained gap (data
    glitch, thin-liquidity spike) can't read as a real catalyst alone.
    """
    if len(bars) < 21:
        return {"gap_catalyst_active": False}

    today, prior = bars[-1], bars[-2]
    prior_close = prior["close"]
    if not prior_close:
        return {"gap_catalyst_active": False}

    gap_pct = (today["open"] - prior_close) / prior_close
    threshold = max(GAP_ATR_MULTIPLE * (atr / prior_close) if atr else 0.0, GAP_MIN_PCT_FLOOR)

    avg_vol_20 = sum(b["volume"] for b in bars[-21:-1]) / 20
    volume_ratio = today["volume"] / avg_vol_20 if avg_vol_20 else 0.0

    gap_amount = today["open"] - prior_close
    gap_failed = False
    if gap_amount != 0:
        try:
            from config import GAP_FAIL_RETRACEMENT_PCT
        except ImportError:
            GAP_FAIL_RETRACEMENT_PCT = 0.5
        retraced = ((today["open"] - today["close"]) / gap_amount if gap_amount > 0
                    else (today["close"] - today["open"]) / (-gap_amount))
        gap_failed = retraced > GAP_FAIL_RETRACEMENT_PCT

    active = abs(gap_pct) >= threshold and volume_ratio >= GAP_MIN_VOLUME_RATIO and not gap_failed

    return {
        "gap_catalyst_active": bool(active),
        "gap_pct": round(gap_pct, 4),
        "gap_day_high": today["high"],
        "gap_day_low": today["low"],
        "volume_ratio": round(volume_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Fair value + valuation tier
# ---------------------------------------------------------------------------

def compute_fair_value(ticker: str, current_price: float | None = None) -> dict:
    """
    Thin delegation to agents.valuation's 4-method fundamentals ensemble
    (DCF, comps, Rule-of-40 multiple, PEG) -- kept as a function here so
    build_trade_ladder's call site didn't need to change when the old
    2-anchor (peer + Street) calc was replaced 2026-09-13. See
    agents/valuation.py's module docstring for why Street consensus was
    demoted to a reference range instead of a blended anchor.
    """
    from agents.valuation import compute_valuation_ensemble
    return compute_valuation_ensemble(ticker, current_price=current_price)


# ---------------------------------------------------------------------------
# Assembly: rung merge (confluence) + build_trade_ladder
# ---------------------------------------------------------------------------

def _merge_rungs(raw_levels: list[dict], atr: float) -> list[dict]:
    """Sorts all raw (price, types) levels and clusters anything within
    RUNG_MERGE_ATR_FRACTION x ATR of each other into one rung. A volume
    node's price wins as the cluster's anchor when one's present (the
    most "real" kind of level -- actual traded volume, not a derived
    line); otherwise the cluster's prices are averaged."""
    if not raw_levels or not atr or atr <= 0:
        return sorted(raw_levels, key=lambda r: r["price"])

    merge_dist = atr * RUNG_MERGE_ATR_FRACTION
    band = atr * RUNG_BAND_ATR_FRACTION
    ordered = sorted(raw_levels, key=lambda r: r["price"])

    clusters: list[list[dict]] = []
    for level in ordered:
        if clusters and level["price"] - clusters[-1][-1]["price"] <= merge_dist:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    rungs = []
    for cluster in clusters:
        vol_node = next((c for c in cluster if "volume_node" in c["types"]), None)
        anchor = vol_node["price"] if vol_node else sum(c["price"] for c in cluster) / len(cluster)
        types = sorted({t for c in cluster for t in c["types"]})
        rungs.append({
            "price": round(anchor, 2),
            "band_low": round(anchor - band, 2),
            "band_high": round(anchor + band, 2),
            "types": types,
            "confluence_score": len(types),
            "detail": ", ".join(f"{t} (${c['price']:.2f})" for c, t in zip(cluster, types)) if len(cluster) > 1 else types[0],
        })
    return rungs


def build_trade_ladder(ticker: str, bars: list[dict] | None = None) -> dict | None:
    """
    Assembles the full per-ticker ladder: swing highs/lows, MA stack
    (10/21 EMA, 50/150/200 SMA), 52-week high/low, base floor/ceiling,
    volume nodes, gap levels, and the valuation layer -- merged into one
    ordered rungs array. Returns None if there isn't enough price history
    yet (same insufficient-data convention as compute_trend_template).
    """
    if bars is None:
        try:
            import db
            bars = db.get_price_history(ticker, days=260)
        except Exception as e:
            log.debug(f"[{ticker}] trade_ladder: history fetch failed: {e}")
            return None

    if len(bars) < 60:
        return None

    bars = _sanitize_bars(bars)
    if len(bars) < 60:
        return None

    closes = [b["close"] for b in bars]
    current_price = closes[-1]
    atr = _compute_atr14(bars)

    ema10, ema21 = _ema(closes, 10), _ema(closes, 21)
    sma50, sma150, sma200 = _sma(closes, 50), _sma(closes, 150), _sma(closes, 200)

    raw_levels = [
        {"price": ema10[-1], "types": ["ma10"]},
        {"price": ema21[-1], "types": ["ma21"]},
        {"price": sma50[-1], "types": ["ma50"]},
        {"price": sma150[-1], "types": ["ma150"]},
        {"price": sma200[-1], "types": ["ma200"]},
    ]

    try:
        from agents.price_action import _find_swings
        for _idx, price, kind in _find_swings(bars[-60:]):
            raw_levels.append({"price": price, "types": [f"swing_{kind}"]})
    except Exception as e:
        log.debug(f"[{ticker}] swing levels unavailable: {e}")

    lookback_52w = min(252, len(closes))
    window_52w = closes[-lookback_52w:]
    raw_levels.append({"price": min(window_52w), "types": ["52w_low"]})
    raw_levels.append({"price": max(window_52w), "types": ["52w_high"]})

    base = compute_base_contraction(bars)
    if base and base["base_detected"]:
        raw_levels.append({"price": base["base_ceiling"], "types": ["base_ceiling"]})
        raw_levels.append({"price": base["base_floor"], "types": ["base_floor"]})

    volume_nodes = compute_volume_nodes(bars, atr) if atr else []
    raw_levels.extend(volume_nodes)

    gap = compute_gap_catalyst(bars, atr)
    if gap.get("gap_catalyst_active"):
        raw_levels.append({"price": gap["gap_day_high"], "types": ["gap_day_high"]})
        raw_levels.append({"price": gap["gap_day_low"], "types": ["gap_day_low"]})

    rungs = _merge_rungs(raw_levels, atr) if atr else sorted(raw_levels, key=lambda r: r["price"])

    trend_stage = "not_confirmed"
    try:
        from agents.price_action import compute_trend_template
        tt = compute_trend_template(ticker, bars=bars)
        if tt and tt["pass"]:
            trend_stage = "stage2_confirmed"
    except Exception as e:
        log.debug(f"[{ticker}] trend template unavailable for ladder: {e}")

    valuation = compute_fair_value(ticker, current_price=current_price)

    ladder = {
        "current_price": round(current_price, 2),
        "atr": round(atr, 4) if atr else None,
        "trend_stage": trend_stage,
        "base_detected": bool(base and base["base_detected"]),
        "base_ceiling": base.get("base_ceiling") if base else None,
        "base_floor": base.get("base_floor") if base else None,
        "base_contractions": base.get("base_contractions") if base else None,
        "base_volume_dryup": base.get("base_volume_dryup") if base else None,
        "gap_catalyst_active": gap.get("gap_catalyst_active", False),
        "gap_day_high": gap.get("gap_day_high"),
        "gap_day_low": gap.get("gap_day_low"),
        "rungs": rungs,
        **valuation,
    }
    return ladder


def refresh_all_ladders() -> int:
    """Batch driver for the nightly task (tasks.refresh_trade_ladder) --
    computes and upserts the ladder for every watchlist ticker. Returns
    count updated."""
    import db
    from knowledge_loader import get_all_tickers

    updated = 0
    for ticker in get_all_tickers():
        try:
            ladder = build_trade_ladder(ticker)
            if ladder:
                db.upsert_trade_ladder(ticker, ladder)
                updated += 1
        except Exception as e:
            log.warning(f"[{ticker}] trade ladder build failed: {e}")
    return updated


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    import sys
    from pathlib import Path
    # Running this file directly (`python agents/trade_ladder.py`) puts
    # agents/ on sys.path[0], not the repo root -- `import db` etc. would
    # otherwise fail silently inside build_trade_ladder's broad except and
    # print "Insufficient history" with no clue why (a pre-existing
    # limitation shared by ingestion/fundamentals.py's own __main__ block;
    # fixed here rather than left broken like that one).
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    tickers = sys.argv[1:] or ["NVDA"]
    for t in tickers:
        result = build_trade_ladder(t)
        print(f"\n=== {t} ===")
        if result is None:
            print("Insufficient history")
            continue
        for k, v in result.items():
            if k != "rungs":
                print(f"  {k}: {v}")
        print("  rungs:")
        for r in result["rungs"]:
            print(f"    ${r['price']:.2f} [{r['confluence_score']}] {r['types']}")
