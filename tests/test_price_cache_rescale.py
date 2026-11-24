"""Stale-head rescale on the nightly price refresh (ingestion/price_history.py:
_scale_check / _store_bars, db.rescale_and_upsert_price_bars): a fetch on a
new split basis rescales the cached rows the fetch doesn't overwrite, in one
locked transaction with the upsert; dividend-sized gaps do nothing; a gap
that is no split factor, or not uniform across the shared dates, is only
reported; a write that fails or raises skips the upsert. Synthetic data only."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

import config
import db
from ingestion import price_history as ph

START = date(2025, 7, 1)


def _days(n, start=START):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _bar(d, c, v=1_000_000):
    return {"date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": v}


DATES = _days(330)
STORED = [_bar(d, 800.0 + i) for i, d in enumerate(DATES[:320])]   # old basis, through DATES[319]


def _fetched(k, n=251):
    """The trailing n dates on a basis k x the stored one, plus one new day."""
    window = DATES[320 - n + 1:321]
    return [_bar(d, (800.0 + DATES.index(d)) * k) for d in window]


@pytest.fixture
def fake_db(monkeypatch):
    monkeypatch.setattr(config, "PRICE_CACHE_RESCALE_ENABLED", True)
    m = MagicMock()
    m.get_price_history.side_effect = lambda t, days=260, strict=False: STORED[-days:]
    m.upsert_price_bars.side_effect = lambda t, bars: len(bars)
    m.rescale_and_upsert_price_bars.side_effect = lambda t, k, bars, volume_k=None, check=None: {
        "rows": 70, "first": DATES[0], "last": DATES[69], "upserted": len(bars)}
    for name in ("get_price_history", "upsert_price_bars", "rescale_and_upsert_price_bars", "log_activity"):
        monkeypatch.setattr(db, name, getattr(m, name))
    return m


def _stages(m):
    return [c.args[0] for c in m.log_activity.call_args_list]


def test_production_default_is_record_only():
    assert config.PRICE_CACHE_RESCALE_ENABLED is False


class TestScaleCheck:
    @pytest.mark.parametrize("k,label", [(1 / 8, "1/8"), (0.5, "1/2"), (2.0, "2"), (10.0, "10"),
                                         (2 / 3, "2/3"), (1 / 3, "1/3"), (1 / 15, "1/15"),
                                         (1 / 20, "1/20"), (50.0, "50"), (3 / 4, "3/4"),
                                         (4 / 5, "4/5"), (5 / 4, "5/4"), (4 / 3, "4/3")])
    def test_split_factors(self, fake_db, k, label):
        chk = ph._scale_check("SNDK", _fetched(k))
        assert chk["action"] == "rescale" and chk["factor"] == label
        assert abs(chk["k"] - k) < 1e-6 and abs(chk["exact"] - k) < 1e-12
        assert chk["overlap"] == 250 and chk["uniform_share"] == 1.0
        assert chk["first_fetched"] == DATES[70] and chk["last_fetched"] == DATES[320]
        assert chk["check_date"] == DATES[70] and chk["check_close"] == 870.0

    def test_dividend_sized_gap_is_nothing(self, fake_db):
        chk = ph._scale_check("XOM", _fetched(0.995))
        assert chk["action"] == "none" and chk["factor"] is None

    def test_unexplained_gap(self, fake_db):
        chk = ph._scale_check("XYZ", _fetched(1.4))
        assert chk["action"] == "unexplained" and chk["factor"] is None

    def test_near_but_outside_factor_tolerance(self, fake_db):
        assert ph._scale_check("XYZ", _fetched(0.5 * 1.03))["action"] == "unexplained"

    @pytest.mark.parametrize("adjusted,action", [(0.60, "unexplained"), (0.85, "unexplained"),
                                                 (0.95, "rescale")])
    def test_partly_adjusted_series_is_not_rescaled(self, fake_db, adjusted, action):
        bars = _fetched(1 / 8)
        cut = int(len(bars) * (1 - adjusted))          # the oldest dates still on the stored basis
        for b in bars[:cut]:
            for f in ("open", "high", "low", "close"):
                b[f] *= 8
        chk = ph._scale_check("SNDK", bars)
        assert chk["factor"] == "1/8" and chk["action"] == action
        assert abs(chk["uniform_share"] - (250 - cut) / 250) < 1e-3

    def test_new_ticker_too_little_overlap(self, fake_db):
        fake_db.get_price_history.side_effect = lambda *a, **k: []
        assert ph._scale_check("NEW", _fetched(0.5))["action"] == "too_little_overlap"

    def test_minimum_overlap_is_half_the_fetch(self, fake_db):
        # 251 fetched bars need ceil(125.5) = 126 shared dates
        fake_db.get_price_history.side_effect = lambda t, days=260, strict=False: STORED[-125:]
        assert ph._scale_check("X", _fetched(0.5))["action"] == "too_little_overlap"
        fake_db.get_price_history.side_effect = lambda t, days=260, strict=False: STORED[-126:]
        assert ph._scale_check("X", _fetched(0.5))["action"] == "rescale"

    def test_minimum_overlap_floor_of_twenty(self, fake_db):
        assert ph._scale_check("X", _fetched(0.5, n=20))["action"] == "too_little_overlap"   # 19 shared
        assert ph._scale_check("X", _fetched(0.5, n=21))["action"] == "rescale"              # 20 shared


class TestStoreBars:
    def test_eight_for_one_rescales_in_one_write(self, fake_db):
        bars = _fetched(1 / 8)
        assert ph._store_bars("SNDK", bars) == len(bars)
        call = fake_db.rescale_and_upsert_price_bars.call_args
        t, k, b = call.args
        assert (t, b) == ("SNDK", bars) and abs(k - 0.125) < 1e-6
        assert call.kwargs["volume_k"] == 0.125                       # the exact factor
        assert call.kwargs["check"] == (DATES[70], 870.0)
        fake_db.upsert_price_bars.assert_not_called()
        assert _stages(fake_db) == ["price_cache_rescale"]
        extra = fake_db.log_activity.call_args.kwargs["extra"]
        assert extra["applied"] is True and extra["rows"] == 70 and extra["factor"] == "1/8"
        assert extra["rows_first"] == DATES[0] and extra["rows_last"] == DATES[69]
        assert "check_close" not in extra and extra["check_date"] == DATES[70]

    def test_two_for_one(self, fake_db):
        ph._store_bars("APH", _fetched(0.5))
        call = fake_db.rescale_and_upsert_price_bars.call_args
        assert abs(call.args[1] - 0.5) < 1e-6 and call.kwargs["volume_k"] == 0.5
        fake_db.upsert_price_bars.assert_not_called()

    def test_dividend_gap_plain_upsert_no_row(self, fake_db):
        bars = _fetched(0.995)
        assert ph._store_bars("XOM", bars) == len(bars)
        fake_db.upsert_price_bars.assert_called_once_with("XOM", bars)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()
        assert _stages(fake_db) == []

    def test_unexplained_gap_reported_not_rescaled(self, fake_db):
        bars = _fetched(1.4)
        assert ph._store_bars("XYZ", bars) == len(bars)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()
        fake_db.upsert_price_bars.assert_called_once_with("XYZ", bars)
        assert _stages(fake_db) == ["price_cache_scale_gap"]
        assert "unexplained scale gap" in fake_db.log_activity.call_args.args[2]

    def test_non_uniform_gap_reported_not_rescaled(self, fake_db):
        bars = _fetched(1 / 8)
        for b in bars[:100]:
            b["close"] *= 8
        ph._store_bars("SNDK", bars)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()
        fake_db.upsert_price_bars.assert_called_once_with("SNDK", bars)
        assert _stages(fake_db) == ["price_cache_scale_gap"]

    def test_check_error_plain_upsert(self, fake_db):
        fake_db.get_price_history.side_effect = RuntimeError("pool exhausted")
        bars = _fetched(1 / 8)
        assert ph._store_bars("SNDK", bars) == len(bars)
        fake_db.upsert_price_bars.assert_called_once_with("SNDK", bars)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()

    def test_failed_rescale_write_skips_the_upsert(self, fake_db):
        fake_db.rescale_and_upsert_price_bars.side_effect = lambda *a, **k: None
        assert ph._store_bars("SNDK", _fetched(1 / 8)) == 0
        fake_db.upsert_price_bars.assert_not_called()
        assert _stages(fake_db) == ["price_cache_rescale_failed"]

    def test_raising_rescale_write_skips_the_upsert(self, fake_db):
        def boom(*a, **k):
            raise psycopg2.InterfaceError("connection already closed")
        fake_db.rescale_and_upsert_price_bars.side_effect = boom
        assert ph._store_bars("SNDK", _fetched(1 / 8)) == 0
        fake_db.upsert_price_bars.assert_not_called()
        assert _stages(fake_db) == ["price_cache_rescale_failed"]

    def test_switched_off_records_only(self, fake_db, monkeypatch):
        monkeypatch.setattr(config, "PRICE_CACHE_RESCALE_ENABLED", False)
        bars = _fetched(1 / 8)
        assert ph._store_bars("SNDK", bars) == len(bars)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()
        fake_db.upsert_price_bars.assert_called_once_with("SNDK", bars)
        assert _stages(fake_db) == ["price_cache_rescale"]
        extra = fake_db.log_activity.call_args.kwargs["extra"]
        assert extra["applied"] is False and extra["factor"] == "1/8" and extra["uniform_share"] == 1.0

    def test_activity_row_failure_does_not_raise(self, fake_db):
        fake_db.log_activity.side_effect = RuntimeError("down")
        assert ph._store_bars("XYZ", _fetched(1.4)) == 251

    def test_second_run_after_a_rescale_does_nothing(self, fake_db):
        # after the rescale the stored window sits on the fetched basis
        rescaled = [_bar(b["date"], b["close"] / 8 * 8) for b in STORED]
        fetched = _fetched(1.0)
        fake_db.get_price_history.side_effect = lambda t, days=260, strict=False: rescaled[-days:]
        ph._store_bars("SNDK", fetched)
        fake_db.rescale_and_upsert_price_bars.assert_not_called()
        assert _stages(fake_db) == []


def test_nightly_refresh_goes_through_store_bars():
    with patch("knowledge_loader.get_all_tickers", return_value=["SNDK"]), \
         patch("ingestion.price_history._fetch_bars", return_value=[_bar(DATES[0], 1.0)]), \
         patch("ingestion.price_history._store_bars", return_value=1) as sb, \
         patch("ingestion.price_history.time.sleep"):
        assert ph.refresh_all_price_history() == 3      # SNDK, SPY, QQQ
    assert [c.args[0] for c in sb.call_args_list] == ["SNDK", "SPY", "QQQ"]


def test_sp500_refresh_goes_through_store_bars():
    with patch("ingestion.sp500_universe.get_sp500_tickers", return_value=["APH"]), \
         patch("knowledge_loader.get_all_tickers", return_value=[]), \
         patch("db.get_newest_bar_dates", return_value={}), \
         patch("ingestion.price_history._fetch_bars", return_value=[_bar("2026-11-20", 1.0)]), \
         patch("ingestion.price_history._store_bars", return_value=1) as sb, \
         patch("ingestion.price_history.time.sleep"):
        assert ph.refresh_sp500_price_history(today=date(2026, 11, 21)) == 1
    assert sb.call_args.args[0] == "APH"


class TestDbRescale:
    BARS = [_bar("2026-11-20", 10.0), _bar("2026-11-18", 11.0), _bar("2026-11-19", 12.0)]

    def _conn(self, dates=(), check_close=870.0):
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [(d,) for d in dates]
        cur.fetchone.return_value = (check_close,) if check_close is not None else None
        return conn, cur

    def _run(self, conn, **kw):
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            return db.rescale_and_upsert_price_bars("SNDK", 0.1277, self.BARS, **kw)

    def test_one_locked_transaction(self):
        conn, cur = self._conn([date(2025, 7, 2), date(2025, 7, 1)])
        out = self._run(conn, volume_k=0.125, check=("2025-11-21", 870.0))
        assert out == {"rows": 2, "first": "2025-07-01", "last": "2025-07-02", "upserted": 3}
        calls = [c.args for c in cur.execute.call_args_list]
        assert calls[0] == ("SELECT pg_advisory_xact_lock(hashtext('ph:' || %s))", ("SNDK",))
        assert calls[1][0].startswith("SELECT close FROM price_history") and calls[1][1] == ("SNDK", "2025-11-21")
        sql, params = calls[2]
        assert sql.startswith("UPDATE price_history SET open=open*%s")
        assert "date < %s OR (date <= %s AND NOT (date = ANY(%s::date[])))" in sql
        assert params[:4] == (0.1277,) * 4 and params[4] == 0.125          # volume by the exact factor
        assert params[5:] == ("SNDK", "2026-11-18", "2026-11-20", ["2026-11-18", "2026-11-19", "2026-11-20"])
        assert len(calls) == 6                                             # lock, check, update, 3 upserts
        conn.commit.assert_called_once()

    def test_volume_defaults_to_k(self):
        conn, cur = self._conn()
        self._run(conn)
        update = next(c.args for c in cur.execute.call_args_list if c.args[0].startswith("UPDATE"))
        assert update[1][4] == 0.1277

    @pytest.mark.parametrize("now", [871.0, None])
    def test_changed_or_missing_check_close_aborts(self, now):
        conn, cur = self._conn(check_close=now)
        assert self._run(conn, volume_k=0.125, check=("2025-11-21", 870.0)) is None
        assert not any(c.args[0].startswith(("UPDATE", "INSERT")) for c in cur.execute.call_args_list)
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()

    def test_failure_rolls_back_everything(self):
        conn, cur = self._conn()
        cur.execute.side_effect = [None, None, RuntimeError("insert failed")]
        assert self._run(conn) is None
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()

    def test_rollback_that_raises_still_returns_none(self):
        conn, cur = self._conn()
        cur.execute.side_effect = [None, psycopg2.OperationalError("server closed the connection")]
        conn.rollback.side_effect = psycopg2.InterfaceError("connection already closed")
        assert self._run(conn) is None
        conn.commit.assert_not_called()

    def test_bad_inputs(self):
        assert db.rescale_and_upsert_price_bars("SNDK", 0.0, [_bar("2026-11-20", 1.0)]) is None
        assert db.rescale_and_upsert_price_bars("SNDK", True, [_bar("2026-11-20", 1.0)]) is None
        assert db.rescale_and_upsert_price_bars("SNDK", 0.5, []) is None

    def test_plain_upsert_unchanged(self):
        conn, cur = self._conn()
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            assert db.upsert_price_bars("NVDA", [_bar("2026-11-20", 1.0)]) == 1
        assert cur.execute.call_args.args[0].strip().startswith("INSERT INTO price_history")
        conn.commit.assert_called_once()
