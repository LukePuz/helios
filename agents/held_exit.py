"""Held-position exit candidates (record only).

The model has written no TRIM/EXIT verdict since 10-15, and the trim/exit
rule only routes sells the model writes, so every closed lot has ended at its
broker stop. This records, on every TRIM_EXIT_RULE row, what four fixed
code-side exit rules would have said, with the raw numbers behind each so
other thresholds can be scored later. Nothing routes on it, and
trim_exit.decide never reads it.

All four are evaluated on the last completed daily close from the
price_history cache (today's unfinished bar dropped, non-finite bars carried
forward or dropped as the entry trend check does):

  below_both_sma           close below both its 50- and 200-day simple
                           averages (the live entry trend gate's test)
  below_plan_structural    close below the active trade plan's structural
                           stop (trade_plans.structural_stop, from the price
                           ladder); None with no plan or no level
  below_plan_invalidation  close below the plan's invalidation_price (the
                           model's downside, or the stop for a held_seed
                           plan); recorded beside (b) because the 11-21
                           audit found those levels unstable
  time_stop                held at least HELD_EXIT_TIME_STOP_SESSIONS
                           completed sessions (the entry's ET date counts as
                           session 1) with the close at or below the entry
                           price; sessions are cached bars, so a gap in the
                           price cache undercounts them
  atr_drawdown             the highest close since the entry date minus the
                           close is at least HELD_EXIT_DRAWDOWN_ATR times ATR14

Each candidate is {"would_exit": True/False/None, "reason": None or why it
is None}. firing lists the candidates that are True; any_would_exit is
bool(firing) (a None candidate is unknown, not a fire).

The parameters (20 sessions, 3 ATR) are illustrative and recorded on the
row. 3 ATR matches the entry-quality shadow's drawdown_gt_3atr label; 20
sessions is about one trading month, twice the 10-day stop-out window.
"""

from __future__ import annotations

import math
from datetime import date, datetime

HELD_EXIT_VERSION = 1
SMA_SHORT, SMA_LONG = 50, 200

CANDIDATES = ("below_both_sma", "below_plan_structural", "below_plan_invalidation",
              "time_stop", "atr_drawdown")


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _d(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _atr14(bars: list[dict]) -> float | None:
    """14-bar simple average true range, as trade_ladder._compute_atr14."""
    if len(bars) < 15:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, lo, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return sum(trs[-14:]) / 14


def _cand(would, reason=None) -> dict:
    return {"would_exit": would, "reason": None if would is not None else reason}


def candidates(bars: list[dict], *, entry_price, entry_date, plan: dict | None,
               plan_read: str = "ok", time_stop_sessions: int = 20,
               drawdown_atr: float = 3.0) -> dict:
    """The record from completed, finite daily bars (oldest first, each with
    date/high/low/close), the lot's entry price and entry date (ET date of
    the first fill), and the active plan (None: no plan) with plan_read the
    plan read's outcome. Pure; the backfill calls it on bars cut at each
    day. Never raises on bad values (they give None labels)."""
    bars = [b for b in (bars or []) if _f(b.get("close")) is not None
            and _f(b.get("high")) is not None and _f(b.get("low")) is not None]
    entry = _f(entry_price)
    if entry is not None and entry <= 0:
        entry = None
    edate = _d(entry_date)
    rec = {"version": HELD_EXIT_VERSION,
           "params": {"time_stop_sessions": time_stop_sessions, "drawdown_atr": drawdown_atr,
                      "sma_short": SMA_SHORT, "sma_long": SMA_LONG},
           "bars": len(bars), "last_bar_date": None, "last_close": None,
           "entry_price": entry, "entry_date": edate.isoformat() if edate else None,
           "sessions_held": None, "since_entry_truncated": None, "close_vs_entry_pct": None,
           "sma50": None, "sma200": None, "pct_vs_sma50": None, "pct_vs_sma200": None,
           "highest_close_since_entry": None, "highest_close_date": None, "atr14": None,
           "drawdown_from_high_pct": None, "drawdown_from_high_atr": None,
           "plan": None, "plan_read": plan_read}
    c = {}
    if not bars:
        for k in CANDIDATES:
            c[k] = _cand(None, "no_cached_bars")
        rec["candidates"], rec["firing"], rec["any_would_exit"] = c, [], False
        return rec

    closes = [float(b["close"]) for b in bars]
    last = closes[-1]
    rec["last_bar_date"], rec["last_close"] = str(bars[-1].get("date"))[:10], round(last, 4)
    if entry is not None:
        rec["close_vs_entry_pct"] = round((last / entry - 1) * 100, 4)

    # (a) both averages
    smas = {}
    for n in (SMA_SHORT, SMA_LONG):
        if len(closes) >= n:
            smas[n] = sum(closes[-n:]) / n
            rec[f"sma{n}"] = round(smas[n], 4)
            rec[f"pct_vs_sma{n}"] = round((last / smas[n] - 1) * 100, 4)
    if len(smas) == 2:
        c["below_both_sma"] = _cand(last < smas[SMA_SHORT] and last < smas[SMA_LONG])
    else:
        c["below_both_sma"] = _cand(None, "insufficient_bars")

    # (b) plan levels
    if plan_read != "ok":
        c["below_plan_structural"] = _cand(None, "plan_read_error")
        c["below_plan_invalidation"] = _cand(None, "plan_read_error")
    elif not isinstance(plan, dict):
        c["below_plan_structural"] = _cand(None, "no_plan")
        c["below_plan_invalidation"] = _cand(None, "no_plan")
    else:
        structural, invalid = _f(plan.get("structural_stop")), _f(plan.get("invalidation_price"))
        rec["plan"] = {"plan_id": plan.get("id"), "source": plan.get("source"),
                       "revision": plan.get("revision"), "levels_source": plan.get("levels_source"),
                       "structural_stop": structural, "invalidation_price": invalid,
                       "created_at": str(plan.get("created_at")) if plan.get("created_at") else None}
        c["below_plan_structural"] = (_cand(last < structural) if structural and structural > 0
                                      else _cand(None, "no_structural_stop"))
        c["below_plan_invalidation"] = (_cand(last < invalid) if invalid and invalid > 0
                                        else _cand(None, "no_invalidation"))

    # (c) time stop and (d) drawdown both need the entry date
    since = [b for b in bars if edate is not None and _d(b.get("date")) is not None
             and _d(b.get("date")) >= edate]
    if edate is None:
        c["time_stop"] = _cand(None, "no_entry_date")
        c["atr_drawdown"] = _cand(None, "no_entry_date")
    elif not since:
        c["time_stop"] = _cand(None, "no_bars_since_entry")
        c["atr_drawdown"] = _cand(None, "no_bars_since_entry")
    else:
        first = _d(bars[0].get("date"))
        # The earliest bar we hold is after the entry date (the read hit its
        # row limit, or the cache starts late), so sessions_held and the high
        # since entry are lower bounds. Judged on the bars themselves, not on
        # the row count, because today's partial bar is dropped before this.
        truncated = first is not None and first > edate
        rec["since_entry_truncated"] = truncated
        rec["sessions_held"] = len(since)
        if entry is None:
            c["time_stop"] = _cand(None, "no_entry_price")
        else:
            c["time_stop"] = _cand(len(since) >= time_stop_sessions and last <= entry)
        hi = max(since, key=lambda b: float(b["close"]))
        high = float(hi["close"])
        rec["highest_close_since_entry"] = round(high, 4)
        rec["highest_close_date"] = str(hi.get("date"))[:10]
        rec["drawdown_from_high_pct"] = round((1 - last / high) * 100, 4) if high > 0 else None
        atr = _atr14(bars)
        if atr is not None and atr > 0:
            rec["atr14"] = round(atr, 4)
            dd = (high - last) / atr
            rec["drawdown_from_high_atr"] = round(dd, 4)
            c["atr_drawdown"] = _cand(dd >= drawdown_atr)
        else:
            c["atr_drawdown"] = _cand(None, "insufficient_bars")

    rec["candidates"] = c
    rec["firing"] = [k for k in CANDIDATES if c[k]["would_exit"] is True]
    rec["any_would_exit"] = bool(rec["firing"])
    return rec
