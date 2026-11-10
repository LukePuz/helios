"""Audit 11-06 23:24 #8: a per-position track record is recorded beside the
per-tranche block the prompt shows. Record-only; no live DB."""
from unittest.mock import MagicMock, patch

import db
from agents import analyst


def test_by_position_query_groups_tranches_by_exit_and_scores_the_entry_leg():
    cur = MagicMock()
    cur.fetchall.return_value = [(7, 2, 1, 0.0, 1.5, 5)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.get_conviction_calibration_by_position(since="2026-11-01")
    sql = cur.execute.call_args.args[0]
    assert "DISTINCT ON (ticker, pos_key)" in sql and "entry_date ASC" in sql
    assert "COALESCE(exit_date::text, 'row:' || id::text)" in sql
    # m607: the since bound is applied after DISTINCT ON, to the position's first leg
    assert sql.index("DISTINCT ON") < sql.index("WHERE entry_date >=")
    assert rows == [{"conviction": 7, "positions": 2, "wins": 1, "win_rate": 0.5,
                     "avg_outcome_pct": 0.0, "avg_alpha_pct": 1.5, "legs": 5}]   # 0.0 kept


def test_record_is_cached_and_never_raises():
    analyst._BY_POSITION_CACHE.clear()
    with patch("db.get_conviction_calibration_by_position", return_value=[]) as q:
        first = analyst._track_record_by_position()
        assert first["rows"] == [] and first["computed_at"]
        assert analyst._track_record_by_position() is first
    assert q.call_count == 1
    analyst._BY_POSITION_CACHE.clear()
    with patch("db.get_conviction_calibration_by_position", side_effect=RuntimeError("pool")):
        assert analyst._track_record_by_position() == {"error": "RuntimeError"}
    assert "v" not in analyst._BY_POSITION_CACHE      # errors aren't cached (m607)
    analyst._BY_POSITION_CACHE.clear()


def test_by_position_stays_off_routing_rows():
    from agents import crew
    prov = {"model": "m", "track_record": {"since": None, "shown": True, "by_position": {"rows": []}}}
    fields = crew._route_row_fields({"provenance": prov}, "ROUTE_NO_PRICE")
    tr = fields["provenance"]["track_record"]
    assert "by_position" not in tr and tr["shown"] is True
    assert "by_position" in prov["track_record"]          # the verdict's own copy untouched
