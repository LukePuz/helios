"""Record-only target anchoring and fundamentals staleness on a verdict.

It runs after the reply, from the decision dict in analyst._analyze_inner, and
goes on signals.decision.target_anchor. Nothing in the prompt, the brief, the
gates or routing reads it. Never raises.

From the prompt text the model was sent (so the values are the ones it saw,
not a later re-read of trade_ladder):
  street_target, fair_value  the brief's "Blended fair value: $X (confidence
                             ...); street consensus target $Y" line
                             (context_engine._price_structure_section). None
                             with a reason in missing when the line, or the
                             street figure ("n/a"), is absent.
  bull_target                valuation.bull_case_target, parsed as
                             agents.plans._dollars parses it.
  target_vs_street_pct       (bull_target / street_target - 1) * 100, 2 dp.
  target_vs_fair_value_pct   (bull_target / fair_value - 1) * 100, 2 dp.
  fv_confidence              the line's "(confidence X)" as printed: "high",
                             "medium", "low" (agents/valuation.py: how far the
                             intrinsic and market family medians differ), or
                             None when the line is absent or prints "n/a".
                             The line carries no family values, so none are
                             recorded (and none are re-read).
  target_equals_fv           |bull_target / fair_value - 1| < TARGET_EQUALS_TOL
  target_equals_street       the same against street_target. None when either
                             side is missing. target_equals_tol stores the
                             tolerance (0.005 = 0.5%).
No band or label is applied beyond that equality test: how close counts as
anchored is Luke's.

From one read-only DB statement (one round trip, statement_timeout 500 ms,
rolled back, fail-open):
  fundamentals_updated_at    fundamentals.updated_at: the row's last refresh.
                             It changes every night, so it can't show stale
                             quarter data by itself.
  fundamentals_as_of         the first fundamentals_history snapshot (at or
                             before `at`) carrying the current values of the
                             quarter-based fields in FUNDAMENTALS_QUARTER_FIELDS.
                             The price-based multiples move daily and are left
                             out. fundamentals_as_of_bound is "history_start"
                             when that run reaches the oldest row read, so the
                             values may be older.
  last_print_date            the newest earnings date before `at`'s ET date,
                             from earnings_calendar_history (rows recorded at
                             or before `at`) and past rows of earnings_calendar.
  print_today                an earnings date on `at`'s ET date (its time of
                             day isn't known here, so it isn't last_print_date).
  fundamentals_predate_last_print
                             True when fundamentals_as_of's ET date is before
                             last_print_date, False when after, None when it is
                             the same day (the refresh can fall on either side
                             of the print) or either side is unknown.

Limits:
  - A data revision between prints (a quarter-based field changing without a
    print) moves fundamentals_as_of later, so predate can undercount.
  - Identical quarter-field values across two quarters would read as stale.
  - fundamentals_history starts 2026-09-13 (bound "history_start" before
    then), and earnings dates are only as good as the calendar's rows.
  - fundamentals_updated_at is the row as read; a replay at a past `at` gets
    today's value, not the one at `at` (the other fields are as of `at`).
  - On the analysis path: one round trip, under statement_timeout plus the
    network and any pool connect.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATEMENT_TIMEOUT_MS = 500
HISTORY_ROWS = 400
FUNDAMENTALS_QUARTER_FIELDS = ("revenue_growth_yoy", "fcf_margin", "net_profit_margin",
                               "return_on_equity")
_PRICE = r"\$([\d,]+(?:\.\d+)?)"
TARGET_EQUALS_TOL = 0.005    # |bull / x - 1| below this counts as equal (0.5%)
_FAIR_LINE_RE = re.compile(r"Blended fair value: " + _PRICE + r" \(confidence ([^)\n]*)\); "
                           r"street consensus target (?:" + _PRICE + r"|n/a)")


def _finite(v):
    try:
        v = float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _pct(a, b):
    return round((a / b - 1) * 100, 2) if a is not None and b else None


def _equals(a, b):
    return abs(a / b - 1) < TARGET_EQUALS_TOL if a is not None and b else None


def brief_targets(prompt: str | None) -> dict:
    """The street target, blended fair value and its confidence in the
    prompt text, with the count of lines found (the first is used). Pure;
    never raises."""
    out = {"fair_value": None, "street_target": None, "fv_confidence": None,
           "fair_value_lines": 0}
    try:
        found = list(_FAIR_LINE_RE.finditer(prompt or ""))
        out["fair_value_lines"] = len(found)
        if found:
            out["fair_value"] = _finite(found[0].group(1))
            conf = found[0].group(2).strip()
            out["fv_confidence"] = conf if conf and conf != "n/a" else None
            out["street_target"] = _finite(found[0].group(3)) if found[0].group(3) else None
    except Exception:
        pass
    return out


def targets_record(parsed: dict | None, prompt: str | None) -> dict:
    """The target fields (pure; no I/O). Never raises."""
    missing = {}
    rec = brief_targets(prompt)
    bull = None
    try:
        from agents.plans import _dollars
        val = (parsed or {}).get("valuation")
        bull = _finite(_dollars(val.get("bull_case_target"))) if isinstance(val, dict) else None
    except Exception as e:
        missing["bull_target"] = f"error: {type(e).__name__}"
    if bull is None:
        missing.setdefault("bull_target", "unparsed")
    if not rec["fair_value_lines"]:
        missing["fair_value"] = missing["street_target"] = "no fair value line in the prompt"
    elif rec["street_target"] is None:
        missing["street_target"] = "n/a in the brief"
    rec.update(bull_target=bull,
               target_vs_street_pct=_pct(bull, rec["street_target"]),
               target_vs_fair_value_pct=_pct(bull, rec["fair_value"]),
               target_equals_fv=_equals(bull, rec["fair_value"]),
               target_equals_street=_equals(bull, rec["street_target"]),
               target_equals_tol=TARGET_EQUALS_TOL,
               missing=missing)
    return rec


def _et_date(t):
    return t.astimezone(ET).date() if isinstance(t, datetime) and t.tzinfo else None


def fundamentals_as_of(rows) -> tuple:
    """(first recorded_at of the newest run of equal quarter values, bound).
    rows: (recorded_at, *FUNDAMENTALS_QUARTER_FIELDS) newest first."""
    if not rows:
        return None, None

    def key(r):
        return tuple(None if v is None else round(float(v), 6) for v in r[1:])
    head, first = key(rows[0]), rows[0][0]
    for r in rows[1:]:
        if key(r) != head:
            return first, "changed"
        first = r[0]
    return first, "history_start"


def predates(as_of, print_date):
    d = _et_date(as_of)
    if d is None or print_date is None:
        return None
    return True if d < print_date else (False if d > print_date else None)


# One statement, one round trip: the settings and a single SELECT of the three
# scalars and the newest HISTORY_ROWS quarter-field snapshots (as a JSON array,
# newest first), so the run logic stays in fundamentals_as_of.
_READ_SQL = (
    "SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = %(ms)s; "
    "SELECT (SELECT updated_at FROM fundamentals WHERE ticker = %(t)s), "
    "(SELECT max(d) FROM ("
    "SELECT earnings_date AS d FROM earnings_calendar_history "
    "WHERE ticker = %(t)s AND earnings_date < %(today)s AND recorded_at <= %(at)s "
    "UNION ALL SELECT earnings_date FROM earnings_calendar "
    "WHERE ticker = %(t)s AND earnings_date < %(today)s) x), "
    "(SELECT count(*) FROM earnings_calendar_history WHERE ticker = %(t)s "
    "AND earnings_date = %(today)s AND recorded_at <= %(at)s), "
    "(SELECT coalesce(json_agg(json_build_array(h.recorded_at, "
    + ", ".join(f"h.{c}" for c in FUNDAMENTALS_QUARTER_FIELDS)
    + ") ORDER BY h.recorded_at DESC), '[]'::json) FROM ("
    "SELECT recorded_at, " + ", ".join(FUNDAMENTALS_QUARTER_FIELDS)
    + " FROM fundamentals_history WHERE ticker = %(t)s AND recorded_at <= %(at)s "
    "ORDER BY recorded_at DESC LIMIT %(n)s) h)")


def _history_rows(value) -> list:
    """The JSON history array as (recorded_at datetime, *values) tuples."""
    out = []
    for item in value or []:
        t = item[0]
        if isinstance(t, str):
            t = datetime.fromisoformat(t.replace("Z", "+00:00"))
        out.append((t, *item[1:]))
    return out


def read_fundamentals_timing(ticker: str, at: datetime | None = None, conn=None) -> dict:
    """The DB-side fields at `at` (default now). One execute (one round trip):
    read only, SET LOCAL statement_timeout STATEMENT_TIMEOUT_MS and one
    SELECT, then rolled back. Fail-open: timing_read is "error: X" and the
    fields are None. Never raises."""
    at = at or datetime.now(timezone.utc)
    out = {"fundamentals_updated_at": None, "fundamentals_as_of": None,
           "fundamentals_as_of_bound": None, "last_print_date": None, "print_today": None,
           "fundamentals_predate_last_print": None, "timing_read": "ok"}
    import db
    own = conn is None
    try:
        conn = conn or db._get_conn()
    except Exception as e:
        out["timing_read"] = f"error: {type(e).__name__}"
        return out
    try:
        with conn.cursor() as cur:
            cur.execute(_READ_SQL, {"ms": int(STATEMENT_TIMEOUT_MS), "t": ticker,
                                    "today": _et_date(at), "at": at, "n": HISTORY_ROWS})
            upd, last, n_today, hist = cur.fetchone() or (None, None, 0, [])
        as_of, bound = fundamentals_as_of(_history_rows(hist))
        out.update(fundamentals_updated_at=upd.isoformat() if isinstance(upd, datetime) else None,
                   fundamentals_as_of=as_of.isoformat() if isinstance(as_of, datetime) else None,
                   fundamentals_as_of_bound=bound,
                   last_print_date=last.isoformat() if last else None,
                   print_today=bool(n_today),
                   fundamentals_predate_last_print=predates(as_of, last))
    except Exception as e:
        out["timing_read"] = f"error: {type(e).__name__}"
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        if own:
            try:
                db._put_conn(conn)
            except Exception:
                pass
    return out


def target_anchor_record(parsed: dict | None, prompt: str | None, ticker: str,
                         at: datetime | None = None) -> dict:
    """signals.decision.target_anchor. Off (config.TARGET_ANCHOR_RECORD_ENABLED
    False): {"enabled": False}. Never raises."""
    try:
        import config
        if not getattr(config, "TARGET_ANCHOR_RECORD_ENABLED", False):
            return {"enabled": False}
        rec = targets_record(parsed, prompt)
        rec.update(read_fundamentals_timing(ticker, at))
        rec["fundamentals_quarter_fields"] = list(FUNDAMENTALS_QUARTER_FIELDS)
        return rec
    except Exception as e:
        return {"error": type(e).__name__}
