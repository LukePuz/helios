"""Append-only capture of public.price_history into h3d.price_history_asof.

public.price_history is upserted (ON CONFLICT DO UPDATE) with no recorded_at,
so a bar's stored values can change after the fact. Each capture run copies
every (ticker, date) bar that is new or whose open/high/low/close/volume
differs from its latest capture, stamped with captured_at =
statement_timestamp() of the INSERT (one value per run, taken after the run
holds the capture lock). Runs are serialized by a transaction-level advisory
lock (CAPTURE_LOCK_KEY), so two concurrent runs cannot both append the same
bars: the second waits, then sees the first's rows. Unchanged bars are not copied again, so a
row's captured_at is the first run that saw that value. The reader
helios3d.features.store.price_bars_asof takes, per date, the latest capture
at or before as_of. A value that changed and changed back between two runs
is not seen.

Reads public.price_history; writes only h3d.price_history_asof. Not
scheduled in Celery or beat. To run it (from the repo root, with the venv):

    venv/bin/python -m helios3d.features.capture             # dry run: counts only
    venv/bin/python -m helios3d.features.capture --apply     # apply h3d migrations, then capture
    venv/bin/python -m helios3d.features.capture --apply --ticker NVDA --ticker SPY

A useful time is after the nightly price_history refresh has finished. When
it is scheduled, a systemd timer or cron line running the --apply command is
enough; the run is idempotent (a second run with no price changes inserts 0
rows).
"""
import argparse
import logging
import sys

# pg_advisory_xact_lock key for capture runs: the bigint whose bytes spell
# "h3d_capt". Released automatically at commit or rollback.
CAPTURE_LOCK_KEY = 0x6833645F63617074
LOCK_SQL = "SELECT pg_advisory_xact_lock(%s)"
ISOLATION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"

_LATEST = (
    "latest AS (SELECT DISTINCT ON (ticker, date) ticker, date, open, high, low, close, volume "
    "FROM h3d.price_history_asof {where} ORDER BY ticker, date, captured_at DESC, id DESC)")
_CHANGED = (
    "FROM public.price_history p LEFT JOIN latest l ON l.ticker = p.ticker AND l.date = p.date "
    "WHERE (l.ticker IS NULL OR (p.open, p.high, p.low, p.close, p.volume) "
    "IS DISTINCT FROM (l.open, l.high, l.low, l.close, l.volume)) {and_p}")


def _filters(tickers):
    if not tickers:
        return "", "", ()
    return "WHERE ticker = ANY(%s)", "AND p.ticker = ANY(%s)", (list(tickers),)


def capture_sql(tickers=None, sha: str | None = None) -> tuple[str, tuple]:
    """The INSERT ... SELECT that appends new or changed bars."""
    w, a, tp = _filters(tickers)
    sql = ("WITH " + _LATEST.format(where=w) + " "
           "INSERT INTO h3d.price_history_asof "
           "(ticker, date, open, high, low, close, volume, captured_at, code_sha) "
           "SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume, statement_timestamp(), %s "
           + _CHANGED.format(and_p=a))
    # Param order follows placeholder order: latest's filter, code_sha, p's filter.
    return sql, (*tp, sha, *tp)


def count_sql(tickers=None) -> tuple[str, tuple]:
    """Rows a capture would append: (total, new, changed)."""
    w, a, tp = _filters(tickers)
    sql = ("WITH " + _LATEST.format(where=w) + " "
           "SELECT COUNT(*), COUNT(*) FILTER (WHERE l.ticker IS NULL), "
           "COUNT(*) FILTER (WHERE l.ticker IS NOT NULL) " + _CHANGED.format(and_p=a))
    return sql, (*tp, *tp)


def _count_without_table(tickers=None) -> tuple[str, tuple]:
    sql = "SELECT COUNT(*) FROM public.price_history"
    if tickers:
        return sql + " WHERE ticker = ANY(%s)", (list(tickers),)
    return sql, ()


def _table_exists(cur) -> bool:
    cur.execute("SELECT to_regclass('h3d.price_history_asof')")
    return cur.fetchone()[0] is not None


def plan(conn, tickers=None) -> dict:
    """Dry run: what a capture would append, read inside a READ ONLY transaction."""
    from helios3d.schema.conn import read_only
    with read_only(conn) as cur:
        if not _table_exists(cur):
            sql, params = _count_without_table(tickers)
            cur.execute(sql, params)
            n = int(cur.fetchone()[0])
            return {"table_exists": False, "would_append": n, "new": n, "changed": 0}
        sql, params = count_sql(tickers)
        cur.execute(sql, params)
        total, new, changed = (int(x) for x in cur.fetchone())
    return {"table_exists": True, "would_append": total, "new": new, "changed": changed}


def capture(conn, tickers=None, sha: str | None = None) -> int:
    """Apply h3d migrations, then append new or changed bars in one
    transaction. Returns the number of rows appended. Raises ValueError on an
    autocommit connection, before anything is written."""
    if getattr(conn, "autocommit", False) is True:
        # Each statement would commit on its own: the advisory lock would be
        # released before the INSERT and the isolation level would not apply.
        raise ValueError("capture() needs a transactional connection (autocommit is on)")
    from helios3d.schema import migrate
    migrate.apply(conn)
    sql, params = capture_sql(tickers, sha)
    conn.rollback()  # start a fresh transaction so the isolation level can be set
    try:
        with conn.cursor() as cur:
            # READ COMMITTED: the INSERT takes its snapshot after the lock is
            # held, so it sees a concurrent run's committed rows.
            cur.execute(ISOLATION_SQL)
            cur.execute(LOCK_SQL, (CAPTURE_LOCK_KEY,))
            cur.execute(sql, params)
            n = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return int(n or 0)


def run(apply: bool = False, tickers=None, conn=None) -> dict:
    from helios3d.schema.conn import connection
    if conn is None:
        with connection() as c:
            return run(apply=apply, tickers=tickers, conn=c)
    tickers = sorted({t.strip().upper() for t in tickers}) if tickers else None
    summary = {"mode": "apply" if apply else "dry-run", "tickers": tickers or "all"}
    if apply:
        from helios3d.record.labels import code_sha
        summary["appended"] = capture(conn, tickers, sha=code_sha())
    else:
        summary.update(plan(conn, tickers))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Append-only capture of price_history into h3d.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="count what would be appended (default)")
    g.add_argument("--apply", action="store_true", help="append to h3d.price_history_asof")
    p.add_argument("--ticker", action="append", default=None, help="limit to a ticker (repeatable)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run(apply=bool(args.apply), tickers=args.ticker)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
