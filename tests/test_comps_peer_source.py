"""Comps records which peer pool its median came from (comps_peer_source:
"industry", "watchlist_fallback", or None with no comps value) and how many
values it was taken over (comps_peer_count), on the trade_ladder row
(migration 83). Log-only: the comps value, blend and tier are unchanged."""
import logging
from unittest.mock import MagicMock, patch

import psycopg2.errors
import pytest

import db
from agents import valuation

FUND = {"ticker": "AAA", "price_to_sales": 10.0, "market_cap": 1000.0, "shares_outstanding": 10.0}
REVENUE_PER_SHARE = (1000.0 / 10.0) / 10.0

WATCHLIST = {"AAA": {"sector": "Tech"}, "B": {"sector": "Tech"},
             "C": {"sector": "Tech"}, "D": {"sector": "Tech"}}
ALL_ROWS = [
    {"ticker": "AAA", "industry": "Semis", "price_to_sales": 10.0},
    {"ticker": "B", "industry": "Semis", "price_to_sales": 4.0},
    {"ticker": "C", "industry": "Semis", "price_to_sales": 5.0},
    {"ticker": "D", "industry": "Semis", "price_to_sales": 6.0},
]
SP500 = [{"ticker": "X", "price_to_sales": 7.0}, {"ticker": "Y", "price_to_sales": 8.0}]


def _data(industry="Semis", all_rows=ALL_ROWS, sp500=SP500, watchlist=WATCHLIST):
    """Patches the peer-median functions' data sources."""
    fund = dict(FUND, industry=industry)
    return [
        patch("db.get_fundamentals", return_value=fund),
        patch("db.get_fundamentals_all", return_value=all_rows),
        patch("db.get_sp500_fundamentals_by_industry", return_value=sp500),
        patch("knowledge_loader.load_watchlist", return_value=watchlist),
    ]


def _comps(meta=None, **kw):
    ps = _data(**kw)
    for p in ps:
        p.start()
    try:
        return valuation.compute_comps_fair_value("AAA", dict(FUND, industry=kw.get("industry", "Semis")),
                                                  **({} if meta is None else {"meta": meta}))
    finally:
        for p in ps:
            p.stop()


def test_industry_path_value_unchanged_and_source_recorded():
    # broad pool: B, C, D (watchlist, AAA excluded) + X, Y -> [4,5,6,7,8], median 6
    meta = {}
    assert _comps() == 6.0 * (1000.0 / 10.0) / 10.0
    assert _comps(meta) == _comps() == 60.0
    assert meta == {"peer_source": "industry", "peer_count": 5}


def test_watchlist_fallback_value_unchanged_and_source_recorded():
    # no industry -> sector peers AAA, B, C, D -> [4,5,6,10], median 5.5
    meta = {}
    assert _comps(industry=None) == 5.5 * (1000.0 / 10.0) / 10.0
    assert _comps(meta, industry=None) == _comps(industry=None) == 55.0
    assert meta == {"peer_source": "watchlist_fallback", "peer_count": 4}


def test_broad_pool_too_thin_falls_back():
    meta = {}
    assert _comps(meta, sp500=[]) == 55.0          # broad pool 3 < 5 peers
    assert meta == {"peer_source": "watchlist_fallback", "peer_count": 4}


def test_no_comps_value_records_nothing():
    meta = {}
    assert _comps(meta, industry=None, watchlist={"AAA": {"sector": "Tech"}}) is None
    assert meta == {}


def test_peer_median_return_values_unchanged_by_out():
    from ingestion.fundamentals import get_industry_peer_median, get_sector_peer_median
    ps = _data()
    for p in ps:
        p.start()
    try:
        out_i, out_s = {}, {}
        assert get_industry_peer_median("AAA", "price_to_sales") == \
            get_industry_peer_median("AAA", "price_to_sales", out=out_i) == 6.0
        assert get_sector_peer_median("AAA", "price_to_sales") == \
            get_sector_peer_median("AAA", "price_to_sales", out=out_s) == 5.5
        assert out_i == {"peer_count": 5} and out_s == {"peer_count": 4}
    finally:
        for p in ps:
            p.stop()


def _ensemble(comps=None, **kw):
    """compute_valuation_ensemble with fixed intrinsic methods; comps either
    the real function on patched data or a stub returning a fixed value."""
    ps = _data(**kw) + [
        patch.object(valuation, "compute_dcf_fair_value", lambda t, f: 80.0),
        patch.object(valuation, "compute_residual_income_fair_value", lambda t, f: None),
        patch.object(valuation, "compute_justified_multiple_fair_value", lambda t, f: None),
        patch.object(valuation, "compute_rule40_fair_value", lambda t, f: None),
        patch.object(valuation, "compute_peg_fair_value", lambda t, f: None),
        patch.object(valuation, "compute_reverse_dcf", lambda t, f, p: None),
    ]
    if comps is not None:
        ps.append(patch.object(valuation, "compute_comps_fair_value", comps))
    for p in ps:
        p.start()
    try:
        return valuation.compute_valuation_ensemble("AAA", current_price=50.0)
    finally:
        for p in ps:
            p.stop()


@pytest.mark.parametrize("industry,value,source,count", [
    ("Semis", 60.0, "industry", 5),
    (None, 55.0, "watchlist_fallback", 4),
])
def test_ensemble_records_source_and_numbers_unchanged(industry, value, source, count):
    r = _ensemble(industry=industry)
    assert (r["comps_peer_source"], r["comps_peer_count"]) == (source, count)
    stub = _ensemble(comps=lambda t, f, meta=None: value, industry=industry)
    assert stub["comps_peer_source"] is None
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("comps_peer_")}
    assert strip(r) == strip(stub)
    assert r["comps_fair_value"] == value
    assert r["fair_value_blended"] == round((80.0 + value) / 2, 2)
    assert r["valuation_tier"] is not None


def test_ensemble_source_none_without_comps():
    r = _ensemble(industry=None, watchlist={"AAA": {"sector": "Tech"}})
    assert r["comps_fair_value"] is None
    assert r["comps_peer_source"] is None and r["comps_peer_count"] is None


# --- storage (migration 83) ---------------------------------------------------

def _conn(fail_column=None):
    cur = MagicMock()
    calls = []

    def execute(sql, params=None):
        calls.append((sql, params))
        if fail_column and fail_column in sql:
            raise psycopg2.errors.UndefinedColumn(
                f'column "{fail_column}" of relation "trade_ladder" does not exist')
    cur.execute.side_effect = execute
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, calls


@pytest.fixture(autouse=True)
def _fresh_warnings():
    db._missing_column_warned.clear()
    yield
    db._missing_column_warned.clear()


LADDER = {"current_price": 50.0, "comps_fair_value": 60.0, "fair_value_blended": 70.0,
          "rungs": [], "valuation_methods_used": ["comps", "dcf"],
          "comps_peer_source": "industry", "comps_peer_count": 5}


def test_upsert_writes_peer_columns():
    conn, calls = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.upsert_trade_ladder("AAA", LADDER)
    assert len(calls) == 1
    sql, params = calls[0]
    assert "comps_peer_source = EXCLUDED.comps_peer_source" in sql
    assert "comps_peer_count = EXCLUDED.comps_peer_count" in sql
    assert sql.count("%s") == len(params) == 1 + len(db._LADDER_COLS) + 2
    assert params[0] == "AAA" and params[-2:] == ("industry", 5)
    conn.commit.assert_called_once()


def test_upsert_with_migration_pending_writes_the_rest(caplog):
    conn, calls = _conn("comps_peer_source")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         caplog.at_level(logging.WARNING):
        db.upsert_trade_ladder("AAA", LADDER)
        db.upsert_trade_ladder("AAA", LADDER)
    retried = [c for c in calls if "comps_peer" not in c[0]]
    assert len(retried) == 2
    sql, params = retried[0]
    assert sql.count("%s") == len(params) == 1 + len(db._LADDER_COLS)
    assert "justified_multiple_fair_value = EXCLUDED.justified_multiple_fair_value" in sql
    assert conn.commit.call_count == 2
    assert sum("comps_peer_source is missing" in r.getMessage() for r in caplog.records) == 1


def test_upsert_other_errors_still_fail(caplog):
    conn, calls = _conn("comps_fair_value")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         caplog.at_level(logging.ERROR):
        db.upsert_trade_ladder("AAA", LADDER)
    assert len(calls) == 1
    conn.commit.assert_not_called()
    assert any("upsert_trade_ladder failed" in r.getMessage() for r in caplog.records)


def test_read_with_migration_pending():
    cur = MagicMock()
    cur.fetchone.return_value = {"ticker": "AAA", "comps_fair_value": 60, "fair_value_blended": 70}
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        row = db.get_trade_ladder("AAA")
    assert row["comps_fair_value"] == 60.0
    assert row.get("comps_peer_source") is None and row.get("comps_peer_count") is None


def test_brief_text_ignores_peer_fields():
    import context_engine
    base = {"current_price": 50.0, "atr": 1.5, "fair_value_blended": 70.0,
            "valuation_confidence": "low", "street_consensus_target": 65.0, "rungs": []}
    texts = []
    for extra in ({}, {"comps_peer_source": "watchlist_fallback", "comps_peer_count": 4}):
        with patch("db.get_trade_ladder", return_value=dict(base, **extra)), \
             patch("config.BRIEF_PRICE_STRUCTURE", True), \
             patch("agents.price_action.compute_price_action", return_value=None), \
             patch("agents.price_action.compute_trend_template", return_value=None), \
             patch("agents.chase_risk.compute_chase_risk", return_value=None):
            texts.append(context_engine._price_structure_section("AAA", price=50.0))
    assert "Blended fair value: $70.00" in texts[0] and texts[0] == texts[1]
