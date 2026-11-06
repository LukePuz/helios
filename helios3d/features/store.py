"""Point-in-time readers: ``get(source, ticker, as_of)``.

Each reader returns only rows the system could have known at ``as_of``. It
filters on the source's as-of column in SQL, then checks every returned row
again in Python and drops (and counts in ``refused``) any whose knowledge
time is after ``as_of``. ``as_of`` must be a timezone-aware datetime.

Every result carries ``asof_safe``. The rule used here is mechanical, not a
research judgment: ``asof_safe`` is True only when the source's as-of column
is set when the row is written and the row is not changed afterwards
(append-only), so the filter alone guarantees no look-ahead. Otherwise it is
False, ``reason`` says why, and the rows are still returned. Whether a
flagged source is good enough for a given analysis is not decided here.

==================  ==============================================  =========
source              as-of column (knowledge time)                   asof_safe
==================  ==============================================  =========
price_bars          public.price_history.date, at its session       False
                    close (no recorded_at; rows are upserted)
price_bars_asof     h3d.price_history_asof.captured_at, and the     True
                    bar's session close
fundamentals        public.fundamentals_history.recorded_at          True
regime              public.market_regime_log.timestamp, or the       True
                    spill-replay time for replayed rows
earnings_calendar   public.earnings_calendar_history.recorded_at     True
earnings_consensus  public.earnings_consensus_history.fetched_at    True
news                public.news_log.published_at (the insert time;   True
                    see NEWS_REASON)
portfolio_snapshot  public.portfolio_snapshots.timestamp             True
position_stops      latest of opened_at, updated_at,                 False
                    thesis_updated_at (rows change in place)
==================  ==============================================  =========

Reads public tables (and h3d.price_history_asof) inside READ ONLY
transactions; writes nothing. read_only() rolls back the connection before
and after the read, so passing ``conn=`` discards any open transaction on
that connection: commit first. Portfolio rows carry account values: they must
not go into research exports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

from helios3d.record.labels import ET, session_close_at


@dataclass
class FeatureResult:
    source: str
    ticker: str | None
    as_of: datetime
    rows: list[dict]
    asof_column: str
    asof_safe: bool
    reason: str
    refused: int = 0  # rows the Python check dropped as known after as_of
    caveats: tuple[str, ...] = field(default_factory=tuple)


PRICE_BARS_REASON = (
    "public.price_history has no recorded_at and is upserted (db.upsert_price_bars, "
    "ON CONFLICT DO UPDATE), so a stored bar may have been rewritten after as_of by a "
    "later refresh, a data fix, or a split/dividend re-adjustment (ingestion/price_history.py "
    "uses yfinance Ticker.history() defaults, which adjust past bars). Only the bar date "
    "is filtered: bars whose session has not closed by as_of are refused.")
NEWS_REASON = (
    "news_log.published_at is never set by its writer (db.log_news_item), so it is the "
    "insert time (DEFAULT NOW()): when Helios logged the headline, not when the article "
    "was published. As a knowledge time it cannot leak the future; it includes the "
    "ingest delay.")
POSITION_STOPS_REASON = (
    "public.position_stops holds one current row per open position, changed in place "
    "and deleted when the position closes. Rows changed after as_of are refused (their "
    "values are later than as_of), positions closed since as_of are absent, and some "
    "updates (setup_category/valuation_tier_at_entry) do not move updated_at. The result "
    "is the open positions now that are unchanged since as_of, not the book at as_of.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aware(as_of: datetime) -> datetime:
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    return as_of


def _ts(v) -> datetime | None:
    """A DB timestamp as an aware datetime (naive values are taken as UTC)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


def _query(conn, sql: str, params: tuple) -> list[dict]:
    from helios3d.schema.conn import connection, read_only
    if conn is None:
        with connection() as c:
            return _query(c, sql, params)
    with read_only(conn) as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _refuse_after(rows: list[dict], known_at: Callable[[dict], datetime | None],
                  as_of: datetime) -> tuple[list[dict], int]:
    """Rows whose knowledge time is at or before as_of; a row with no
    knowledge time is refused too."""
    kept = []
    for r in rows:
        k = known_at(r)
        if k is not None and k <= as_of:
            kept.append(r)
    return kept, len(rows) - len(kept)


def _bar_close(r: dict) -> datetime | None:
    d = r.get("date")
    if isinstance(d, datetime):
        d = d.date()
    return session_close_at(d) if isinstance(d, date) else None


def _env(trading_env: str | None) -> str:
    if trading_env:
        return trading_env
    from config import TRADING_ENV
    return TRADING_ENV


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def price_bars(ticker: str, as_of: datetime, days: int = 260, conn=None) -> FeatureResult:
    """Up to ``days`` daily bars from public.price_history whose session closed
    by as_of, oldest first. Values are today's stored values (asof_safe False).
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    days = max(int(days), 0)
    t = ticker.strip().upper()
    rows = _query(conn,
                  "SELECT ticker, date, open, high, low, close, volume FROM public.price_history "
                  "WHERE ticker = %s AND date <= %s ORDER BY date DESC LIMIT %s",
                  (t, as_of.astimezone(ET).date(), days + 1))
    kept, refused = _refuse_after(rows, _bar_close, as_of)
    kept = sorted(kept, key=lambda r: r["date"])[-days:] if days else []
    return FeatureResult("price_bars", t, as_of, kept,
                         "public.price_history.date (session close)", False, PRICE_BARS_REASON,
                         refused)


def latest_capture_per_date(rows: list[dict], as_of: datetime) -> list[dict]:
    """Pure: for each bar date, the capture with the latest captured_at at or
    before as_of (ties: highest id). Captures after as_of, or with no
    captured_at, are ignored. Returned oldest date first."""
    best: dict = {}
    for r in rows:
        c = _ts(r.get("captured_at"))
        if c is None or c > as_of:
            continue
        key = (c, r.get("id") if r.get("id") is not None else -1)
        d = r.get("date")
        if d not in best or key > best[d][0]:
            best[d] = (key, r)
    return [best[d][1] for d in sorted(best)]


def price_bars_asof(ticker: str, as_of: datetime, days: int = 260, conn=None) -> FeatureResult:
    """Daily bars as h3d.price_history_asof had captured them by as_of: for each
    bar date, the latest capture at or before as_of, oldest first. Each row
    has ``captured_before_close`` (the capture ran before that bar's session
    closed, so the bar may be partial). The rule is applied again in Python
    (latest_capture_per_date) to whatever the SQL returns.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    days = max(int(days), 0)
    t = ticker.strip().upper()
    rows = _query(conn,
                  "SELECT DISTINCT ON (date) id, ticker, date, open, high, low, close, volume, captured_at "
                  "FROM h3d.price_history_asof WHERE ticker = %s AND captured_at <= %s AND date <= %s "
                  "ORDER BY date DESC, captured_at DESC, id DESC LIMIT %s",
                  (t, as_of, as_of.astimezone(ET).date(), days + 1))

    def known(r):
        c, b = _ts(r.get("captured_at")), _bar_close(r)
        return max(c, b) if c and b else None
    kept, refused = _refuse_after(rows, known, as_of)
    kept = latest_capture_per_date(kept, as_of)
    for r in kept:
        r["captured_before_close"] = _ts(r["captured_at"]) < _bar_close(r)
    kept = sorted(kept, key=lambda r: r["date"])[-days:] if days else []
    return FeatureResult(
        "price_bars_asof", t, as_of, kept,
        "h3d.price_history_asof.captured_at (and the bar's session close)", True,
        "append-only capture of price_history; a bar is shown as it was at the last capture "
        "at or before as_of", refused,
        ("empty before the first capture run (python -m helios3d.features.capture --apply)",
         "same price basis as price_history at capture time (yfinance history() defaults)"))


def fundamentals(ticker: str, as_of: datetime, limit: int = 1, conn=None) -> FeatureResult:
    """The latest ``limit`` fundamentals_history rows recorded by as_of, newest first.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    t = ticker.strip().upper()
    rows = _query(conn,
                  "SELECT * FROM public.fundamentals_history WHERE ticker = %s AND recorded_at <= %s "
                  "ORDER BY recorded_at DESC, id DESC LIMIT %s", (t, as_of, limit))
    kept, refused = _refuse_after(rows, lambda r: _ts(r.get("recorded_at")), as_of)
    return FeatureResult(
        "fundamentals", t, as_of, kept, "public.fundamentals_history.recorded_at", True,
        "append-only; recorded_at is the insert time (DEFAULT NOW(), never set by the writer)",
        refused,
        ("migration 60 columns (forward_eps, targets, shares_outstanding, net_cash, market_cap) "
         "are NULL before 2026-10-30",))


# Only replays after as_of matter: a row replayed at or before as_of with a
# timestamp at or before as_of was known by as_of. The a.ts bound lets the
# planner use activity_feed_ts_idx when as_of is recent; for an old as_of it
# still reads most of activity_feed.
_REGIME_SQL = (
    "WITH replayed AS ("
    " SELECT (a.extra_json->>'row_id')::bigint AS row_id, MIN(a.ts) AS replayed_at"
    " FROM public.activity_feed a"
    " WHERE a.ts > %s AND a.stage = 'spill_replay' AND a.extra_json->>'kind' = 'regime'"
    " GROUP BY 1) "
    "SELECT r.id, r.timestamp, r.regime, r.score, r.confidence, r.signals_json, "
    "r.portfolio_impact_json, r.reasoning, r.regime_type, r.sector_etf, x.replayed_at "
    "FROM public.market_regime_log r LEFT JOIN replayed x ON x.row_id = r.id "
    "WHERE r.timestamp <= %s AND x.row_id IS NULL ")


def regime(ticker: str | None, as_of: datetime, regime_type: str | None = None,
           sector_etf: str | None = None, limit: int = 1, conn=None) -> FeatureResult:
    """The latest ``limit`` market_regime_log rows known by as_of, newest first.
    ``ticker`` is ignored (the regime is market-wide). ``regime_type='overall'``
    also matches legacy NULL rows, as db.get_latest_regime does; None matches
    every type. A row written by scripts/replay_spill.py keeps its original
    timestamp, so it counts from its replay time (its activity_feed
    spill_replay row) instead.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    sql, params = _REGIME_SQL, [as_of, as_of]
    if regime_type == "overall":
        sql += "AND (r.regime_type = 'overall' OR r.regime_type IS NULL) "
    elif regime_type is not None:
        sql += "AND r.regime_type = %s "
        params.append(regime_type)
    if sector_etf is not None:
        sql += "AND r.sector_etf = %s "
        params.append(sector_etf.strip().upper())
    sql += "ORDER BY r.timestamp DESC, r.id DESC LIMIT %s"
    params.append(limit)
    rows = _query(conn, sql, tuple(params))

    def known(r):
        t, rp = _ts(r.get("timestamp")), _ts(r.get("replayed_at"))
        return max(t, rp) if t and rp else t
    kept, refused = _refuse_after(rows, known, as_of)
    return FeatureResult(
        "regime", None, as_of, kept,
        "public.market_regime_log.timestamp (spill-replayed rows: activity_feed.ts)", True,
        "append-only; timestamp is the insert time except for spill-replayed rows, which "
        "are dated by their replay", refused,
        ("db.get_latest_regime reads the newest row of any type unless REGIME_OVERALL_ONLY; "
         "pass regime_type to choose",))


def earnings_calendar(ticker: str, as_of: datetime, limit: int = 20, conn=None) -> FeatureResult:
    """earnings_calendar_history rows (the dates the refresh believed) recorded
    by as_of, newest first.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    t = ticker.strip().upper()
    rows = _query(conn,
                  "SELECT id, ticker, earnings_date, confirmed, recorded_at "
                  "FROM public.earnings_calendar_history WHERE ticker = %s AND recorded_at <= %s "
                  "ORDER BY recorded_at DESC, id DESC LIMIT %s", (t, as_of, limit))
    kept, refused = _refuse_after(rows, lambda r: _ts(r.get("recorded_at")), as_of)
    return FeatureResult(
        "earnings_calendar", t, as_of, kept, "public.earnings_calendar_history.recorded_at", True,
        "append-only; recorded_at is the insert time (DEFAULT NOW())", refused,
        ("table exists since db migration 62 (2026-10-29); nothing before that",
         "written only by the refresh path that replaces future dates "
         "(upsert_earnings_date(replace_future=True)); no row means unknown, not no earnings"))


def earnings_consensus(ticker: str, as_of: datetime, limit: int = 1, conn=None) -> FeatureResult:
    """The latest ``limit`` earnings_consensus_history rows (consensus EPS and
    revenue estimates, EPS surprise history) fetched by as_of, newest first.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    t = ticker.strip().upper()
    rows = _query(conn,
                  "SELECT * FROM public.earnings_consensus_history WHERE ticker = %s AND fetched_at <= %s "
                  "ORDER BY fetched_at DESC, id DESC LIMIT %s", (t, as_of, limit))
    kept, refused = _refuse_after(rows, lambda r: _ts(r.get("fetched_at")), as_of)
    return FeatureResult(
        "earnings_consensus", t, as_of, kept, "public.earnings_consensus_history.fetched_at", True,
        "append-only; fetched_at is the insert time (DEFAULT NOW(), never set by the writer)",
        refused,
        ("table exists since db migration 79 (2026-11-03); nothing before that",
         "status/parts say which parts yfinance returned; ETFs are not fetched (status no_data)",
         "eps_history.surprisePercent is Yahoo's raw value, unit not converted"))


def news(ticker: str, as_of: datetime, limit: int = 50, since: datetime | None = None,
         conn=None) -> FeatureResult:
    """news_log headlines logged by as_of (and at or after ``since`` when
    given), newest first. Each row also has ``logged_at`` (= published_at).
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    t = ticker.strip().upper()
    sql = ("SELECT id, ticker, title, url, image_url, published_at FROM public.news_log "
           "WHERE ticker = %s AND published_at <= %s ")
    params = [t, as_of]
    if since is not None:
        sql += "AND published_at >= %s "
        params.append(_aware(since))
    sql += "ORDER BY published_at DESC, id DESC LIMIT %s"
    params.append(limit)
    rows = _query(conn, sql, tuple(params))
    kept, refused = _refuse_after(rows, lambda r: _ts(r.get("published_at")), as_of)
    for r in kept:
        r["logged_at"] = r["published_at"]
    return FeatureResult("news", t, as_of, kept, "public.news_log.published_at (insert time)",
                         True, NEWS_REASON, refused,
                         ("not the article publish time; a publish-time window needs another source",))


def portfolio_snapshot(ticker: str | None, as_of: datetime, trading_env: str | None = None,
                       limit: int = 1, conn=None) -> FeatureResult:
    """The latest ``limit`` portfolio_snapshots rows (whole book; ``ticker`` is
    ignored) taken by as_of for the trading env (default config.TRADING_ENV).
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    limit = max(int(limit), 0)
    rows = _query(conn,
                  "SELECT id, timestamp, trading_env, total_value, cash, positions_json, benchmark_json "
                  "FROM public.portfolio_snapshots WHERE trading_env = %s AND timestamp <= %s "
                  "ORDER BY timestamp DESC, id DESC LIMIT %s", (_env(trading_env), as_of, limit))
    kept, refused = _refuse_after(rows, lambda r: _ts(r.get("timestamp")), as_of)
    return FeatureResult(
        "portfolio_snapshot", None, as_of, kept, "public.portfolio_snapshots.timestamp", True,
        "append-only; timestamp is the insert time (DEFAULT NOW())", refused,
        ("about every 5 minutes; the newest snapshot can be up to one interval old",
         "account values: keep out of research exports"))


def _stop_known_at(r: dict) -> datetime | None:
    ts = [_ts(r.get(c)) for c in ("opened_at", "updated_at", "thesis_updated_at")]
    ts = [t for t in ts if t is not None]
    return max(ts) if _ts(r.get("opened_at")) else None


def position_stops(ticker: str | None, as_of: datetime, trading_env: str | None = None,
                   conn=None) -> FeatureResult:
    """position_stops rows opened by as_of and unchanged since (asof_safe False;
    see POSITION_STOPS_REASON). ``ticker`` None returns every position.
    A caller-supplied ``conn`` has its open transaction rolled back (read_only())."""
    as_of = _aware(as_of)
    sql = "SELECT * FROM public.position_stops WHERE trading_env = %s AND opened_at <= %s "
    params = [_env(trading_env), as_of]
    t = None
    if ticker:
        t = ticker.strip().upper()
        sql += "AND ticker = %s "
        params.append(t)
    sql += "ORDER BY ticker"
    rows = _query(conn, sql, tuple(params))
    kept, refused = _refuse_after(rows, _stop_known_at, as_of)
    return FeatureResult("position_stops", t, as_of, kept,
                         "public.position_stops: latest of opened_at, updated_at, thesis_updated_at",
                         False, POSITION_STOPS_REASON, refused)


SOURCES: dict[str, Callable[..., FeatureResult]] = {
    "price_bars": price_bars,
    "price_bars_asof": price_bars_asof,
    "fundamentals": fundamentals,
    "regime": regime,
    "earnings_calendar": earnings_calendar,
    "earnings_consensus": earnings_consensus,
    "news": news,
    "portfolio_snapshot": portfolio_snapshot,
    "position_stops": position_stops,
}


def get(source: str, ticker: str | None, as_of: datetime, **kwargs) -> FeatureResult:
    """One source's rows for ``ticker`` known at ``as_of``; extra keyword
    arguments go to that source's reader (limit, days, conn, ...)."""
    try:
        fn = SOURCES[source]
    except KeyError:
        raise ValueError(f"unknown feature source {source!r}; one of {sorted(SOURCES)}") from None
    return fn(ticker, as_of, **kwargs)
