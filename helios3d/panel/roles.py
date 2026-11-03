"""The four analyst roles, the PM and the red team: data slices and prompts.

Each analyst reads its own slice through helios3d.features as of the
decision time (the record's as_of_ts), so no role sees rows recorded later.
A prompt is the shared header first (identical for the four analysts of one
decision: ticker, decision time, trigger, rules and the JSON schema), then
the role's data, then the role's instructions last. Calls for one decision
run back to back, so Ollama can reuse the cached prefix.

What a role sees:
- fundamentals: the latest fundamentals_history row and the earnings
  calendar.
- technical_flow: daily bars (h3d.price_history_asof when it has rows, else
  public.price_history, flagged not as-of safe), reduced to trend, relative
  strength against SPY, volume and volatility figures.
- macro_regime: the latest overall regime row and SPY's trend.
- news_sentiment: headlines logged in the last NEWS_DAYS days and the
  earnings calendar.
No role reads portfolio snapshots, positions, the live prompt or the live
verdict: the panel never sees account figures, and it is not anchored on the
live answer.

SCRUBBING (scrub_slice): every slice goes through the decision-record
scrubber's rules before it is rendered, and the stored data_slice is exactly
what was rendered. Two parent-path exemptions, as decision.py exempts
versions.config:
- scalar values under ``fundamentals`` and ``regime.signals`` keep their keys
  (the scrubber's account-word rule would drop fcf_margin, net_profit_margin,
  return_on_equity and net_cash, which are company figures, and the regime's
  liquidity signal, which is market-wide);
- ``headlines[*].title``, ``regime.reasoning`` and the flattened
  ``regime.signals.*`` notes are public text: the key is kept, figures are
  kept, account ids are still redacted.
The macro slice flattens signals_json (nested dicts, agents/market_regime.py)
into bounded dotted keys (flatten_signals).
Every other key and string gets the full scrub. Model outputs and model or
parse error text get account-id redaction only (runner._ids_only): no role
reads an account source, so a figure in a model's text is a company figure.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import date, datetime, timedelta
from typing import Callable

from helios3d.panel import schemas
from helios3d.panel.config import ANALYST_ROLES, PM_ROLE, RED_TEAM_ROLE
from helios3d.record.decision import _TEXT_CAP, _denied, redact_text

# 2: scrub_slice exemptions (fundamentals, regime signals, public text) and
# the flattened regime signals.
PANEL_VERSION = 2
NEWS_DAYS = 14
NEWS_LIMIT = 15
BARS_DAYS = 260
BENCHMARK = "SPY"
TRIGGER_TITLE_CAP = 200
REGIME_REASONING_CAP = 400
# Flattened regime signals: at most this many keys, nested this deep, and
# string values cut to this length.
SIGNALS_MAX_KEYS = 40
SIGNALS_MAX_DEPTH = 3
SIGNALS_TEXT_CAP = 160

# Parents whose scalar values keep their keys (see SCRUBBING above).
_SCALAR_EXEMPT_PARENTS = ("fundamentals", "regime.signals")
# Paths of public text: figures kept, account ids redacted, and the key is
# kept even when the scrubber denies its name ("reasoning"). The regime's
# signal notes are market-wide text from agents/market_regime.py.
_PUBLIC_TEXT_RE = re.compile(r"^(?:headlines\[\d+\]\.title|regime\.reasoning"
                             r"|regime\.signals\.[^\[\]]+)$")

# Sources each role may read (the runner refuses any other).
ROLE_SOURCES = {
    "fundamentals": ("fundamentals", "earnings_calendar"),
    "technical_flow": ("price_bars_asof", "price_bars"),
    "macro_regime": ("regime", "price_bars_asof", "price_bars"),
    "news_sentiment": ("news", "earnings_calendar"),
    # price_context(), shared by the PM and the red team
    "pm": ("price_bars_asof", "price_bars", "earnings_calendar"),
}
FORBIDDEN_SOURCES = ("portfolio_snapshot", "position_stops")

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

ANALYST_HEADER = """You are one analyst on a research panel reviewing a single US stock.
Each analyst sees a different slice of the data. Judge only from the data you are given.
If your data is thin or missing, say so in the rationale and lower your confidence.

Ticker: {ticker}
Decision time: {as_of}
Why it is being reviewed: {trigger}

Reply with JSON only, no prose and no code fence, fields in this order:
{schema}
"""

ROLE_INSTRUCTIONS = {
    "fundamentals": (
        "Your role: FUNDAMENTALS analyst. Judge earnings quality, growth, margins, balance "
        "sheet and valuation (against the company's own history where the data allows). "
        "Ignore price action; another analyst covers it."),
    "technical_flow": (
        "Your role: TECHNICAL/FLOW analyst. Judge trend, relative strength against SPY, "
        "volume and the volatility regime from the figures above. Ignore fundamentals and "
        "news; other analysts cover them."),
    "macro_regime": (
        "Your role: MACRO/REGIME analyst. Judge whether the market regime and the broad "
        "tape favour adding risk in this stock now (risk-on or risk-off, trend of the "
        "market). Ignore company specifics; other analysts cover them."),
    "news_sentiment": (
        "Your role: NEWS/SENTIMENT analyst. Judge the tone and materiality of the recent "
        "headlines and any upcoming earnings date. Headline times are when the system "
        "logged them. Ignore price levels; another analyst covers them."),
}

PM_TEMPLATE = """You are the portfolio manager of a research panel reviewing one US stock.
Four analysts (fundamentals, technical/flow, macro/regime, news/sentiment) each scored it
1-10 (1 strongly bearish, 5 no edge, 10 strongly bullish) with a confidence 0-1.

Ticker: {ticker}
Decision time: {as_of}
Why it is being reviewed: {trigger}

Analyst outputs:
{analysts}

Price context (for the plan's levels):
{price_context}

Next earnings date on record: {earnings}

Write the thesis from the analysts' evidence, say whether the panel is in consensus or
split, then a trade plan (fill it even for a non-buy verdict: it is the plan you would
follow if you did buy), then your verdict and conviction 1-10.
Reply with JSON only, no prose and no code fence, fields in this order:
{schema}
"""

RED_TEAM_TEMPLATE = """You are the red team of a research panel. The portfolio manager wants to buy
the stock below. Your only job is to argue against the buy as strongly as the evidence allows.
Do not restate the thesis; attack it.

Ticker: {ticker}
Decision time: {as_of}

PM thesis: {thesis}
PM verdict: {verdict}, conviction {conviction}/10
PM plan: {plan}

Price context:
{price_context}

Reply with JSON only, no prose and no code fence, fields in this order:
{schema}
"""


def templates_sha() -> str:
    """Hash of every template and the panel version; part of config_sha."""
    blob = json.dumps({"v": PANEL_VERSION, "h": ANALYST_HEADER, "r": ROLE_INSTRUCTIONS,
                       "pm": PM_TEMPLATE, "rt": RED_TEAM_TEMPLATE,
                       "s": [schemas.ANALYST_SCHEMA_TEXT, schemas.PM_SCHEMA_TEXT,
                             schemas.RED_TEAM_SCHEMA_TEXT],
                       "n": [NEWS_DAYS, NEWS_LIMIT, BARS_DAYS, BENCHMARK]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pure figure helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pct(a, b) -> float | None:
    a, b = _f(a), _f(b)
    if a is None or b in (None, 0):
        return None
    return round((a / b - 1) * 100, 2)


def bar_summary(bars: list[dict]) -> dict:
    """Trend, volume and volatility figures from daily bars (oldest first).
    Pure; a figure needing more bars than exist is None."""
    closes = [_f(b.get("close")) for b in bars]
    rows = [(b, c) for b, c in zip(bars, closes) if c is not None]
    if not rows:
        return {"bars": 0}
    closes = [c for _, c in rows]
    vols = [_f(b.get("volume")) for b, _ in rows]
    highs = [_f(b.get("high")) or c for b, c in rows]
    lows = [_f(b.get("low")) or c for b, c in rows]
    last = closes[-1]

    def ret(n):
        return _pct(last, closes[-1 - n]) if len(closes) > n else None

    def sma(n):
        return sum(closes[-n:]) / n if len(closes) >= n else None

    def vs(n):
        s = sma(n)
        return _pct(last, s) if s else None
    daily = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
             if closes[i - 1] > 0 and closes[i] > 0]
    vol20 = (round(statistics.pstdev(daily[-20:]) * math.sqrt(252) * 100, 1)
             if len(daily) >= 20 else None)
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    atr14 = round(sum(trs[-14:]) / 14, 4) if len(trs) >= 14 else None
    v20 = [v for v in vols[-20:] if v is not None]
    v60 = [v for v in vols[-60:] if v is not None]
    vol_ratio = (round((sum(v20) / len(v20)) / (sum(v60) / len(v60)), 2)
                 if len(v20) >= 20 and len(v60) >= 60 and sum(v60) > 0 else None)
    yr = closes[-252:]
    last_bar = rows[-1][0].get("date")
    return {"bars": len(closes), "last_bar_date": str(last_bar) if last_bar is not None else None,
            "last_close": round(last, 4), "ret_5d_pct": ret(5), "ret_20d_pct": ret(20),
            "ret_60d_pct": ret(60), "vs_sma20_pct": vs(20), "vs_sma50_pct": vs(50),
            "vs_sma200_pct": vs(200), "realized_vol_20d_ann_pct": vol20, "atr14": atr14,
            "volume_20d_vs_60d": vol_ratio,
            "from_52w_high_pct": _pct(last, max(yr)) if len(yr) >= 2 else None,
            "from_52w_low_pct": _pct(last, min(yr)) if len(yr) >= 2 else None,
            "range_20d": ([round(min(lows[-20:]), 4), round(max(highs[-20:]), 4)]
                          if len(closes) >= 20 else None)}


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else None
    try:
        from decimal import Decimal
        if isinstance(v, Decimal):
            return _jsonable(float(v))
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _scalar_row(row: dict, skip=("id", "ticker")) -> dict:
    """A row's scalar values only (dicts/lists dropped), JSON-safe."""
    out = {}
    for k, v in row.items():
        if k in skip or isinstance(v, (dict, list, tuple)):
            continue
        out[k] = _jsonable(v)
    return out


def _source_meta(res) -> dict:
    return {"source": res.source, "asof_safe": res.asof_safe, "rows": len(res.rows),
            "refused": res.refused}


def flatten_signals(sig, max_keys: int = SIGNALS_MAX_KEYS, max_depth: int = SIGNALS_MAX_DEPTH,
                    text_cap: int = SIGNALS_TEXT_CAP) -> dict:
    """market_regime_log.signals_json (nested dicts, agents/market_regime.py)
    as one flat dict of dotted keys to scalars: trend.score, volatility.vix,
    liquidity.notes, markov.state, ... Lists become a short joined string.
    Bounded: the first ``max_keys`` keys in sorted order, ``max_depth``
    levels, strings cut to ``text_cap``; ``_truncated`` says how many were
    left out."""
    flat: dict = {}

    def walk(obj, prefix, depth):
        for k in sorted(obj, key=str):
            v = obj[k]
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                if depth < max_depth:
                    walk(v, key, depth + 1)
                continue
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v[:10])
            v = _jsonable(v)
            if isinstance(v, str):
                v = v[:text_cap]
            if v is None or isinstance(v, (str, bool, int, float)):
                flat[key] = v
    if isinstance(sig, dict):
        walk(sig, "", 1)
    keys = list(flat)
    out = {k: flat[k] for k in keys[:max_keys]}
    if len(keys) > max_keys:
        out["_truncated"] = len(keys) - max_keys
    return out


def scrub_slice(obj, path: str = "", dropped: list | None = None, key: str = ""):
    """The decision-record scrub with the panel-input exemptions above. Pure;
    the output is what is rendered and what is stored."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            exempt = ((path in _SCALAR_EXEMPT_PARENTS and not isinstance(v, (dict, list, tuple)))
                      or (isinstance(v, str) and bool(_PUBLIC_TEXT_RE.match(p))))
            if not exempt and _denied(k, v, path):
                if dropped is not None:
                    dropped.append(p)
                continue
            out[str(k)] = scrub_slice(v, p, dropped, str(k))
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub_slice(v, f"{path}[{i}]", dropped, key) for i, v in enumerate(obj)]
    if isinstance(obj, str):
        if len(obj) > _TEXT_CAP and dropped is not None:
            dropped.append(f"{path} (truncated from {len(obj)} chars)")
        cut = obj[:_TEXT_CAP]
        red = redact_text(cut, figures=not _PUBLIC_TEXT_RE.match(path))
        if red != cut and dropped is not None:
            dropped.append(f"{path} (figure redacted)")
        return red
    return obj


# ---------------------------------------------------------------------------
# Data slices
# ---------------------------------------------------------------------------

FeatureGet = Callable[..., object]


def _default_get():
    from helios3d.features import get
    return get


def _guarded(get: FeatureGet, role: str) -> FeatureGet:
    allowed = ROLE_SOURCES.get(role, ())

    def g(source, ticker, as_of, **kw):
        if source in FORBIDDEN_SOURCES or source not in allowed:
            raise PermissionError(f"role {role} may not read {source}")
        return get(source, ticker, as_of, **kw)
    return g


def _bars(get, ticker, as_of, conn):
    """price_bars_asof when it has rows, else price_bars (flagged)."""
    res = get("price_bars_asof", ticker, as_of, days=BARS_DAYS, conn=conn)
    if not res.rows:
        res = get("price_bars", ticker, as_of, days=BARS_DAYS, conn=conn)
    return res


def _earnings(get, ticker, as_of, conn) -> tuple[str | None, dict]:
    res = get("earnings_calendar", ticker, as_of, limit=5, conn=conn)
    d0 = as_of.date()
    upcoming = sorted(r["earnings_date"] for r in res.rows
                      if isinstance(r.get("earnings_date"), date) and r["earnings_date"] >= d0)
    nxt = upcoming[0].isoformat() if upcoming else None
    confirmed = next((bool(r.get("confirmed")) for r in res.rows
                      if upcoming and r.get("earnings_date") == upcoming[0]), None)
    return nxt, {**_source_meta(res), "next_earnings_date": nxt, "confirmed": confirmed}


def build_slice(role: str, ticker: str, as_of: datetime, conn=None,
                get: FeatureGet | None = None) -> dict:
    """The role's data as a JSON-safe dict, scrubbed. ``sources`` lists each
    read with its asof_safe flag and row counts."""
    g = _guarded(get or _default_get(), role)
    data: dict = {}
    sources: list[dict] = []
    if role == "fundamentals":
        res = g("fundamentals", ticker, as_of, limit=1, conn=conn)
        sources.append(_source_meta(res))
        data["fundamentals"] = _scalar_row(res.rows[0]) if res.rows else None
        nxt, meta = _earnings(g, ticker, as_of, conn)
        sources.append(meta)
        data["next_earnings_date"] = nxt
    elif role == "technical_flow":
        res = _bars(g, ticker, as_of, conn)
        sources.append(_source_meta(res))
        data["stock"] = bar_summary(res.rows)
        spy = _bars(g, BENCHMARK, as_of, conn)
        sources.append({**_source_meta(spy), "ticker": BENCHMARK})
        s = bar_summary(spy.rows)
        data["spy"] = {k: s.get(k) for k in ("ret_5d_pct", "ret_20d_pct", "ret_60d_pct")}
        data["relative_strength_vs_spy_pct"] = {
            k: (round(data["stock"][k] - s[k], 2)
                if data["stock"].get(k) is not None and s.get(k) is not None else None)
            for k in ("ret_20d_pct", "ret_60d_pct")}
    elif role == "macro_regime":
        res = g("regime", None, as_of, regime_type="overall", limit=1, conn=conn)
        sources.append(_source_meta(res))
        if res.rows:
            r = res.rows[0]
            sig = r.get("signals_json")
            if isinstance(sig, str):
                try:
                    sig = json.loads(sig)
                except ValueError:
                    sig = {}
            data["regime"] = {
                "label": r.get("regime"), "score": _jsonable(r.get("score")),
                "confidence": _jsonable(r.get("confidence")),
                "as_of": _jsonable(r.get("timestamp")),
                "signals": flatten_signals(sig),
                "reasoning": (str(r.get("reasoning"))[:REGIME_REASONING_CAP]
                              if r.get("reasoning") else None)}
        else:
            data["regime"] = None
        spy = _bars(g, BENCHMARK, as_of, conn)
        sources.append({**_source_meta(spy), "ticker": BENCHMARK})
        s = bar_summary(spy.rows)
        data["spy"] = {k: s.get(k) for k in ("ret_20d_pct", "ret_60d_pct", "vs_sma50_pct",
                                              "vs_sma200_pct", "realized_vol_20d_ann_pct")}
    elif role == "news_sentiment":
        res = g("news", ticker, as_of, limit=NEWS_LIMIT,
                since=as_of - timedelta(days=NEWS_DAYS), conn=conn)
        sources.append(_source_meta(res))
        data["headlines"] = [{"logged_at": _jsonable(r.get("logged_at") or r.get("published_at")),
                              "title": str(r.get("title") or "")[:200]} for r in res.rows]
        nxt, meta = _earnings(g, ticker, as_of, conn)
        sources.append(meta)
        data["next_earnings_date"] = nxt
    else:
        raise ValueError(f"unknown analyst role {role!r}")
    dropped: list = []
    return {"role": role, "data": scrub_slice(_jsonable(data), "", dropped),
            "sources": sources, "dropped": dropped}


def price_context(ticker: str, as_of: datetime, conn=None, get: FeatureGet | None = None) -> dict:
    """Last close, ATR, 20-day range and next earnings date, for the PM's
    plan levels and the red team. Same readers as technical_flow."""
    g = _guarded(get or _default_get(), PM_ROLE)
    res = _bars(g, ticker, as_of, conn)
    s = bar_summary(res.rows)
    nxt, _ = _earnings(g, ticker, as_of, conn)
    return {"last_close": s.get("last_close"), "last_bar_date": s.get("last_bar_date"),
            "atr14": s.get("atr14"), "range_20d": s.get("range_20d"),
            "realized_vol_20d_ann_pct": s.get("realized_vol_20d_ann_pct"),
            "next_earnings_date": nxt, "price_source": res.source, "asof_safe": res.asof_safe}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def trigger_text(trigger_path: str | None, inputs: dict | None) -> str:
    """The record's trigger path and (public) title, account ids redacted."""
    t = (inputs or {}).get("trigger") if isinstance(inputs, dict) else None
    title = t.get("title") if isinstance(t, dict) else None
    parts = [str(trigger_path or "unknown")]
    if isinstance(title, str) and title.strip():
        parts.append(redact_text(title.strip()[:TRIGGER_TITLE_CAP], figures=False))
    return ": ".join(parts)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def shared_header(ticker: str, as_of: datetime, trigger: str) -> str:
    return ANALYST_HEADER.format(ticker=ticker, as_of=as_of.isoformat(timespec="minutes"),
                                 trigger=trigger, schema=schemas.ANALYST_SCHEMA_TEXT)


def render_analyst(role: str, ticker: str, as_of: datetime, trigger: str, slc: dict) -> str:
    """Shared header first, the role's data next, the role's instructions last."""
    return (shared_header(ticker, as_of, trigger)
            + f"\nYour data ({role}):\n{_dump(slc['data'])}\n\n"
            + ROLE_INSTRUCTIONS[role] + "\n")


def render_pm(ticker: str, as_of: datetime, trigger: str, analysts: dict[str, dict | None],
              ctx: dict) -> str:
    lines = []
    for role in ANALYST_ROLES:
        a = analysts.get(role)
        if not a:
            lines.append(f"- {role}: no output (the call failed or did not parse)")
            continue
        lines.append(f"- {role}: score {a['score']}, confidence {a['confidence']:.2f}. "
                     f"{a['rationale']} What would change my mind: {a['what_would_change_my_mind']}")
    pc = {k: ctx.get(k) for k in ("last_close", "last_bar_date", "atr14", "range_20d",
                                  "realized_vol_20d_ann_pct")}
    return PM_TEMPLATE.format(ticker=ticker, as_of=as_of.isoformat(timespec="minutes"),
                              trigger=trigger, analysts="\n".join(lines),
                              price_context=_dump(pc),
                              earnings=ctx.get("next_earnings_date") or "none on record",
                              schema=schemas.PM_SCHEMA_TEXT)


def render_red_team(ticker: str, as_of: datetime, pm: dict, ctx: dict) -> str:
    pc = {k: ctx.get(k) for k in ("last_close", "last_bar_date", "atr14", "range_20d",
                                  "next_earnings_date")}
    return RED_TEAM_TEMPLATE.format(ticker=ticker, as_of=as_of.isoformat(timespec="minutes"),
                                    thesis=pm["thesis"], verdict=pm["verdict"],
                                    conviction=pm["conviction"], plan=_dump(pm.get("plan")),
                                    price_context=_dump(pc), schema=schemas.RED_TEAM_SCHEMA_TEXT)


def agreement_facts(analysts: dict[str, dict | None]) -> dict:
    """Mechanical facts about the analyst scores, stored beside the PM's own
    agreement label. No threshold is applied here: what spread counts as a
    split is not decided in code."""
    scores = {r: a["score"] for r, a in analysts.items() if a}
    conf = [a["confidence"] for a in analysts.values() if a]
    xs = list(scores.values())
    return {"n_scored": len(xs), "n_missing": len(ANALYST_ROLES) - len(xs), "scores": scores,
            "min": min(xs) if xs else None, "max": max(xs) if xs else None,
            "range": (max(xs) - min(xs)) if xs else None,
            "mean": round(statistics.fmean(xs), 3) if xs else None,
            "stdev": round(statistics.pstdev(xs), 3) if len(xs) > 1 else None,
            "n_above_5": sum(1 for x in xs if x > 5), "n_below_5": sum(1 for x in xs if x < 5),
            "mean_confidence": round(statistics.fmean(conf), 3) if conf else None}


__all__ = ["ANALYST_ROLES", "PM_ROLE", "RED_TEAM_ROLE", "build_slice", "price_context",
           "render_analyst", "render_pm", "render_red_team", "agreement_facts", "bar_summary",
           "templates_sha", "prompt_sha", "trigger_text", "shared_header"]
