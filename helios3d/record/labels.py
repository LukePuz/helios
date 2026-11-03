"""Forward-return labels for analyst verdicts (Helios 3D, Wave 0).

Research tooling only: this module computes and stores returns. It does not
decide what any return means for any hypothesis.

DEFINITIONS (definition subject to Luke's pre-registration):

- Verdict: a row of public.signals that passes the existing verdict filter
  used by db.get_recent_verdicts: ``signal NOT IN db._NON_ANALYST_SIGNALS``,
  and ``trading_env`` equal to config.TRADING_ENV, or NULL (legacy rows from
  before migration 51) whose reasoning does not start with
  db.DEGRADED_SIGNAL_PREFIX. ``action`` is ``signals.signal``; ``conviction``
  is ``round(signals.confidence * 10)``, the 1-10 scale the analyst wrote.
- Prices: yfinance daily closes, ``auto_adjust=False``, column ``Close``
  (split-adjusted, not dividend-adjusted, so returns are price returns, not
  total returns). ``label_source`` records this as LABEL_SOURCE.
- Trading calendar: the completed SPY sessions in the downloaded data. A
  session counts as completed once its close in America/New_York has
  passed: 16:00, or 13:00 on an NYSE early close (market_calendar, the
  calendar market_data.is_market_hours uses; it covers 2026-2027, and days
  outside it are taken to close at 16:00).
- Entry reference (``entry_ref_price``): the ticker's close on the first
  session whose close is at or after ``verdict_at`` (in ET). A verdict at
  10:00 ET uses that day's close; one at 17:00 ET, at 14:00 ET on a half day,
  or on a weekend uses the next session's close. ``entry_close_at`` stores
  the close time used.
- Horizon h (5, 10, 20, 60 trading days): ``fwd_price`` is the ticker's close
  h sessions after the entry session on the SPY calendar.
  ``fwd_return = fwd_price / entry_ref_price - 1`` (close to close).
- ``spy_fwd_return``: the same computation for SPY over the same two
  sessions. ``excess_return = fwd_return - spy_fwd_return`` (simple
  difference, not a beta-adjusted or compounded excess).
- A (verdict, horizon) pair gets no label when the horizon session has not
  completed yet, or when either close for the ticker or SPY is missing or
  not positive (delisted symbol, halted day, bad data). Every such pair
  except ``horizon_not_reached`` (not yet due) is written to h3d.label_skips
  with its reason, so a name that drops out is counted, not lost.
- Provenance per label: ``entry_date``, ``fwd_date``, ``code_sha`` (the
  commit that computed it) and ``price_snapshot_sha``, a hash of the label
  source, the symbols, both dates and the four closes used.
- Re-runs upsert on (signal_id, horizon_days): a stored label is recomputed
  from the current download and ``computed_at`` moves. When its
  ``price_snapshot_sha`` changes (the closes were revised), the old row is
  first copied to h3d.verdict_label_history.

Reads public.signals only, inside READ ONLY transactions. Writes only to
h3d.verdict_labels, h3d.verdict_label_history and h3d.label_skips. Not scheduled in Celery; see
helios3d/deploy/helios3d-label.service for the planned unit.

CLI (dry-run is the default; it reads, computes and prints, and writes
nothing):
    python -m helios3d.record.labels [--dry-run] [--since YYYY-MM-DD]
    python -m helios3d.record.labels --apply [--since YYYY-MM-DD]
"""
import argparse
import bisect
import hashlib
import json
import logging
import math
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HORIZONS: tuple[int, ...] = (5, 10, 20, 60)
BENCHMARK = "SPY"
LABEL_SOURCE = "yfinance:close_split_adj:v1"
SESSION_CLOSE_ET = time(16, 0)
# Calendar days of history fetched before the earliest verdict, so the entry
# session is inside the download even after a long weekend.
_LEAD_DAYS = 7


# ---------------------------------------------------------------------------
# Read verdicts (public.signals, read-only)
# ---------------------------------------------------------------------------

def verdict_query(since: datetime | None):
    """SQL and params for the verdict read; the filter mirrors
    db.get_recent_verdicts without the per-ticker and time-window terms."""
    import db
    from config import TRADING_ENV
    sql = ("SELECT id, symbol, timestamp, signal, confidence FROM signals "
           "WHERE signal NOT IN %s AND symbol IS NOT NULL AND timestamp IS NOT NULL "
           "AND (trading_env=%s OR (trading_env IS NULL AND COALESCE(reasoning, '') NOT LIKE %s)) ")
    params = [db._NON_ANALYST_SIGNALS, TRADING_ENV, db.DEGRADED_SIGNAL_PREFIX + "%"]
    if since is not None:
        sql += "AND timestamp >= %s "
        params.append(since)
    sql += "ORDER BY id"
    return sql, tuple(params)


def fetch_verdicts(conn, since: datetime | None = None) -> list[dict]:
    from helios3d.schema.conn import read_only
    sql, params = verdict_query(since)
    with read_only(conn) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    out = []
    for sid, sym, ts, sig, conf in rows:
        out.append({
            "signal_id": int(sid),
            "ticker": str(sym).strip().upper(),
            "verdict_at": ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
            "action": sig,
            "conviction": round(float(conf) * 10) if conf is not None else None,
        })
    return out


# ---------------------------------------------------------------------------
# Prices (yfinance)
# ---------------------------------------------------------------------------

def yf_symbol(ticker: str) -> str:
    """Helios symbol to Yahoo symbol (class shares: BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def session_close_at(d: date) -> datetime:
    """The session's close in ET: 13:00 on an NYSE early close, else 16:00
    (SESSION_CLOSE_ET)."""
    from market_calendar import NYSE_EARLY_CLOSES, EARLY_CLOSE
    return datetime.combine(d, EARLY_CLOSE if d in NYSE_EARLY_CLOSES else SESSION_CLOSE_ET,
                            tzinfo=ET)


def _to_date(idx) -> date:
    if isinstance(idx, datetime):
        return idx.date()
    if isinstance(idx, date):
        return idx
    return idx.to_pydatetime().date()  # pandas Timestamp


def fetch_closes(tickers: list[str], start: date, end: date) -> dict[str, dict[date, float]]:
    """Daily closes per Helios ticker as {date: close}, from one batched
    yf.download call. Tickers with no data are absent from the result."""
    import pandas as pd
    import yfinance as yf

    ymap = {yf_symbol(t): t for t in tickers}
    df = yf.download(list(ymap), start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     interval="1d", auto_adjust=False, actions=False, progress=False,
                     threads=True, group_by="column")
    if df is None or len(df) == 0:
        return {}
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=next(iter(ymap)))
    out: dict[str, dict[date, float]] = {}
    for ysym in close.columns:
        t = ymap.get(str(ysym))
        if t is None:
            continue
        series = {}
        for idx, v in close[ysym].items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv > 0:
                series[_to_date(idx)] = fv
        if series:
            out[t] = series
    return out


# ---------------------------------------------------------------------------
# Labels (pure)
# ---------------------------------------------------------------------------

def completed_sessions(benchmark: dict[date, float], now: datetime) -> list[date]:
    """Sorted benchmark session dates whose 16:00 ET close is at or before now
    (drops today's in-progress bar during market hours)."""
    return sorted(d for d in benchmark if session_close_at(d) <= now)


def entry_index(calendar: list[date], verdict_at: datetime) -> int | None:
    """Index of the first session whose close is at or after verdict_at."""
    v_et = verdict_at.astimezone(ET)
    closes = [session_close_at(d) for d in calendar]
    i = bisect.bisect_left(closes, v_et)
    return i if i < len(calendar) else None


def price_snapshot_sha(ticker: str, entry_d: date, fwd_d: date,
                       p0: float, p1: float, b0: float, b1: float) -> str:
    """Hash of everything a label's numbers come from: the label source, the
    ticker and benchmark, both session dates and the four closes (to 6
    decimals, so last-bit float noise between downloads doesn't count as a
    revision). A re-run with revised closes gets a different hash."""
    body = json.dumps([LABEL_SOURCE, ticker, BENCHMARK, entry_d.isoformat(), fwd_d.isoformat(),
                       *(f"{float(x):.6f}" for x in (p0, p1, b0, b1))])
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def code_sha() -> str | None:
    """The commit this process loaded (agents.analyst's import-time
    code_sha, the same value activity_feed rows carry); None if unreadable."""
    try:
        from agents.analyst import _CODE_SHA
        return _CODE_SHA.get("code_sha")
    except Exception:
        return None


# Skip reasons that are not yet due rather than failed; not written to h3d.label_skips.
_PENDING_SKIPS = ("horizon_not_reached", "entry_not_reached")


def compute_labels(verdicts: list[dict], closes: dict[str, dict[date, float]],
                   now: datetime, horizons=HORIZONS, skipped: list | None = None,
                   sha: str | None = None) -> tuple[list[dict], dict]:
    """Label rows for every (verdict, horizon) that can be computed, plus a
    count of skips by reason. When `skipped` is a list, one dict per skipped
    pair (signal, horizon, reason, dates known) is appended to it. `sha` is
    the code_sha stored on each row. Pure: no I/O."""
    skips = {"horizon_not_reached": 0, "entry_not_reached": 0, "no_entry_session": 0,
             "missing_ticker_price": 0,
             "missing_benchmark_price": 0}
    bench = closes.get(BENCHMARK) or {}
    calendar = completed_sessions(bench, now)
    rows = []

    def skip(v, h, reason, entry_d=None, fwd_d=None):
        skips[reason] += 1
        if skipped is not None:
            skipped.append({"signal_id": v["signal_id"], "ticker": v["ticker"],
                            "verdict_at": v["verdict_at"], "horizon_days": h, "reason": reason,
                            "entry_date": entry_d, "fwd_date": fwd_d,
                            "label_source": LABEL_SOURCE, "code_sha": sha})
    for v in verdicts:
        i = entry_index(calendar, v["verdict_at"]) if calendar else None
        if i is None:
            # After the last completed close the entry session just hasn't
            # happened yet; only an empty calendar is a real dropout.
            reason = "entry_not_reached" if calendar else "no_entry_session"
            for h in horizons:
                skip(v, h, reason)
            continue
        series = closes.get(v["ticker"]) or {}
        entry_d = calendar[i]
        for h in horizons:
            j = i + h
            if j >= len(calendar):
                skip(v, h, "horizon_not_reached", entry_d)
                continue
            fwd_d = calendar[j]
            p0, p1 = series.get(entry_d), series.get(fwd_d)
            if not p0 or not p1:
                skip(v, h, "missing_ticker_price", entry_d, fwd_d)
                continue
            b0, b1 = bench.get(entry_d), bench.get(fwd_d)
            if not b0 or not b1:
                skip(v, h, "missing_benchmark_price", entry_d, fwd_d)
                continue
            r = p1 / p0 - 1.0
            br = b1 / b0 - 1.0
            rows.append({
                "signal_id": v["signal_id"], "ticker": v["ticker"], "verdict_at": v["verdict_at"],
                "action": v["action"], "conviction": v["conviction"], "horizon_days": h,
                "entry_ref_price": p0, "fwd_price": p1, "fwd_return": r,
                "spy_fwd_return": br, "excess_return": r - br, "label_source": LABEL_SOURCE,
                "entry_date": entry_d, "fwd_date": fwd_d,
                "entry_close_at": session_close_at(entry_d), "code_sha": sha,
                "price_snapshot_sha": price_snapshot_sha(v["ticker"], entry_d, fwd_d,
                                                         p0, p1, b0, b1),
            })
    return rows, skips


# ---------------------------------------------------------------------------
# Write (h3d.verdict_labels only)
# ---------------------------------------------------------------------------

_COLS = ("signal_id", "ticker", "verdict_at", "action", "conviction", "horizon_days",
         "entry_ref_price", "fwd_price", "fwd_return", "spy_fwd_return", "excess_return",
         "label_source", "entry_date", "fwd_date", "entry_close_at", "code_sha",
         "price_snapshot_sha")

# Before the upsert: copy each stored label whose input closes changed (a
# different price_snapshot_sha, or none: a row from before migration 2) to
# the history table, so a recomputation never loses the old value.
ARCHIVE_SQL = (
    "INSERT INTO h3d.verdict_label_history (" + ", ".join(_COLS) + ", computed_at) "
    "SELECT " + ", ".join(f"l.{c}" for c in _COLS) + ", l.computed_at "
    "FROM h3d.verdict_labels l JOIN (VALUES %s) AS n(signal_id, horizon_days, price_snapshot_sha) "
    "ON l.signal_id = n.signal_id AND l.horizon_days = n.horizon_days "
    "WHERE l.price_snapshot_sha IS DISTINCT FROM n.price_snapshot_sha"
)

UPSERT_SQL = (
    "INSERT INTO h3d.verdict_labels (" + ", ".join(_COLS) + ", computed_at) VALUES %s "
    "ON CONFLICT (signal_id, horizon_days) DO UPDATE SET "
    + ", ".join(f"{c}=EXCLUDED.{c}" for c in _COLS if c not in ("signal_id", "horizon_days"))
    + ", computed_at=NOW()"
)


_SKIP_COLS = ("signal_id", "ticker", "verdict_at", "horizon_days", "reason", "entry_date",
              "fwd_date", "label_source", "code_sha")

SKIPS_UPSERT_SQL = (
    "INSERT INTO h3d.label_skips (" + ", ".join(_SKIP_COLS) + ", first_seen_at, last_seen_at) "
    "VALUES %s ON CONFLICT (signal_id, horizon_days) DO UPDATE SET "
    + ", ".join(f"{c}=EXCLUDED.{c}" for c in _SKIP_COLS if c not in ("signal_id", "horizon_days"))
    + ", last_seen_at=NOW()"
)


SKIPS_CLEAR_SQL = (
    "DELETE FROM h3d.label_skips s USING (VALUES %s) AS v(signal_id, horizon_days) "
    "WHERE s.signal_id = v.signal_id AND s.horizon_days = v.horizon_days"
)


def upsert_skips(conn, skipped: list[dict], labelled: list[dict] = (),
                 page_size: int = 500) -> int:
    """Write the pairs that got no label (all reasons but _PENDING_SKIPS) to
    h3d.label_skips, one row per (signal, horizon); first_seen_at stays.
    Pairs in `labelled` got a label this run, so their old skip rows go."""
    import psycopg2.extras
    rows = [s for s in skipped if s["reason"] not in _PENDING_SKIPS]
    if not rows and not labelled:
        return 0
    try:
        with conn.cursor() as cur:
            if labelled:
                psycopg2.extras.execute_values(
                    cur, SKIPS_CLEAR_SQL,
                    [(r["signal_id"], r["horizon_days"]) for r in labelled],
                    template="(%s::integer, %s::integer)", page_size=page_size)
            if rows:
                psycopg2.extras.execute_values(
                    cur, SKIPS_UPSERT_SQL, [tuple(r[c] for c in _SKIP_COLS) for r in rows],
                    template="(" + ", ".join(["%s"] * len(_SKIP_COLS)) + ", NOW(), NOW())",
                    page_size=page_size)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


def upsert_labels(conn, rows: list[dict], page_size: int = 500) -> int:
    """Archive labels whose inputs changed, then upsert, in one transaction."""
    import psycopg2.extras
    if not rows:
        return 0
    values = [tuple(r[c] for c in _COLS) for r in rows]
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, ARCHIVE_SQL,
                [(r["signal_id"], r["horizon_days"], r["price_snapshot_sha"]) for r in rows],
                template="(%s::integer, %s::integer, %s::varchar)", page_size=page_size)
            psycopg2.extras.execute_values(cur, UPSERT_SQL, values,
                                           template="(" + ", ".join(["%s"] * len(_COLS)) + ", NOW())",
                                           page_size=page_size)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def run(apply: bool = False, since: datetime | None = None, horizons=HORIZONS,
        now: datetime | None = None, conn=None) -> dict:
    """Read verdicts, fetch closes, compute labels; upsert only when apply."""
    from helios3d.schema.conn import connection
    if conn is None:
        with connection() as c:
            return run(apply=apply, since=since, horizons=horizons, now=now, conn=c)
    now = now or datetime.now(timezone.utc)
    verdicts = fetch_verdicts(conn, since)
    summary = {"mode": "apply" if apply else "dry-run", "verdicts": len(verdicts),
               "labels": 0, "written": 0, "skips": {}, "migrations_applied": []}
    if not verdicts:
        return summary
    start = min(v["verdict_at"] for v in verdicts).astimezone(ET).date() - timedelta(days=_LEAD_DAYS)
    tickers = sorted({v["ticker"] for v in verdicts} | {BENCHMARK})
    closes = fetch_closes(tickers, start, now.astimezone(ET).date())
    skipped: list[dict] = []
    sha = code_sha()
    rows, skips = compute_labels(verdicts, closes, now, horizons, skipped=skipped, sha=sha)
    summary.update(labels=len(rows), skips=skips, code_sha=sha,
                   tickers=len(tickers), tickers_without_prices=len(set(tickers) - set(closes)))
    if apply:
        from helios3d.schema import migrate
        summary["migrations_applied"] = migrate.apply(conn)
        summary["written"] = upsert_labels(conn, rows)
        summary["skips_written"] = upsert_skips(conn, skipped, labelled=rows)
    else:
        summary["sample"] = [
            {k: r[k] for k in ("signal_id", "ticker", "action", "horizon_days", "fwd_return",
                               "excess_return")} for r in rows[:5]]
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Forward-return labels for analyst verdicts (h3d).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="compute and print; write nothing (default)")
    g.add_argument("--apply", action="store_true", help="upsert into h3d.verdict_labels")
    p.add_argument("--since", type=date.fromisoformat, default=None,
                   help="only verdicts at or after this date (ET midnight)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    since = datetime.combine(args.since, time(0, 0), tzinfo=ET) if args.since else None
    summary = run(apply=bool(args.apply), since=since)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
