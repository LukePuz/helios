"""position_stops.entry_price_source records how entry_price was set
(professor m590, migration 85). Record only: a failed source write never
fails the stop record."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import db
import tasks


@contextmanager
def _conn(fail_with=None):
    cur = MagicMock()
    calls = []

    def ex(sql, params=None):
        calls.append((sql, params))
        if fail_with and "entry_price_source" in sql and "INSERT" in sql:
            raise fail_with
    cur.execute.side_effect = ex
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        yield calls, conn


def test_source_is_written_in_the_upsert_itself():
    """Professor 59d: the source and entry_price commit together."""
    with _conn() as (calls, conn), patch("db.TRADING_ENV", "live"):
        assert db.upsert_position_stop("NVDA", 100.0, 90.0, "atr", "Compounder", 7,
                                       entry_price_source="broker_avg") is True
    sql, params = next(c for c in calls if "INSERT INTO position_stops" in c[0])
    assert "entry_price_source = EXCLUDED.entry_price_source" in sql
    assert params[-1] == "broker_avg" and params[1] == "live"
    assert not any(c[0].startswith("UPDATE") for c in calls)
    conn.commit.assert_called_once()


def test_no_source_given_clears_it():
    with _conn() as (calls, _):
        db.upsert_position_stop("NVDA", 100.0, 90.0, "atr", "Compounder", 7)
    assert next(c for c in calls if "INSERT INTO position_stops" in c[0])[1][-1] is None


def test_before_migration_85_the_record_is_written_without_it():
    import psycopg2
    with _conn(fail_with=psycopg2.errors.UndefinedColumn("no column")) as (calls, conn):
        assert db.upsert_position_stop("NVDA", 100.0, 90.0, "atr", "Compounder", 7,
                                       entry_price_source="fill") is True
    sqls = [c[0] for c in calls]
    assert "ROLLBACK TO SAVEPOINT stop_entry_source" in sqls
    retry = [c for c in calls if "INSERT INTO position_stops" in c[0]][-1]
    assert "entry_price_source" not in retry[0] and len(retry[1]) == 8
    conn.commit.assert_called_once()


def test_any_other_failure_fails_the_whole_write():
    with _conn(fail_with=RuntimeError("timeout")) as (calls, conn):
        assert db.upsert_position_stop("NVDA", 100.0, 90.0, "atr", "Compounder", 7,
                                       entry_price_source="fill") is False
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def _snapshot(sources):
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=1000.0, cash=500.0)
    client.get_all_positions.return_value = [
        MagicMock(symbol="NVDA", qty=1, market_value=100.0, avg_entry_price=112.5, unrealized_pl=-12.5)]
    src = (patch("db.get_entry_price_sources", side_effect=sources) if isinstance(sources, Exception)
           else patch("db.get_entry_price_sources", return_value=sources))
    with patch("ibkr.get_client", return_value=client), src, \
         patch("db.get_all_position_stops", return_value=[{"ticker": "NVDA", "entry_price": 100.0}]), \
         patch("db.log_snapshot") as snap, patch("yfinance.Ticker", side_effect=RuntimeError("offline")):
        tasks.take_portfolio_snapshot.run()
    return snap.call_args.kwargs["positions"]


def test_snapshot_records_the_source():
    assert _snapshot({"NVDA": "fill"})["NVDA"]["record_entry_source"] == "fill"


def test_snapshot_source_is_none_when_unreadable():
    pos = _snapshot(RuntimeError("column does not exist"))
    assert pos["NVDA"]["record_entry_source"] is None and pos["NVDA"]["record_entry_price"] == 100.0
