"""TRACK_RECORD_SINCE bounds the prompt's track record by entry date
(professor C35). Default empty: unchanged, no bound."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import db
from agents import analyst


@contextmanager
def _cur():
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        yield cur


def test_default_has_no_date_bound():
    with _cur() as cur:
        db.get_conviction_calibration()
        db.get_subscore_calibration("catalyst_quality")
    for call in cur.execute.call_args_list:
        assert "entry_date" not in call.args[0] and len(call.args[1]) == 1


def test_since_bounds_both_queries():
    with _cur() as cur:
        db.get_conviction_calibration(since="2027-01-01")
        db.get_subscore_calibration("catalyst_quality", since="2027-01-01")
    for call in cur.execute.call_args_list:
        assert "AND entry_date >= (%s::timestamp AT TIME ZONE 'America/New_York')" in call.args[0] and call.args[1][-1] == "2027-01-01"


def test_setting_bounds_the_prompt_queries_by_default():
    with patch("config.TRACK_RECORD_SINCE", "2027-01-01", create=True):
        assert analyst._track_record_since() == "2027-01-01"
        with _cur() as cur:
            analyst._build_track_record_text()
        assert cur.execute.call_args_list
        for call in cur.execute.call_args_list:
            assert "AND entry_date >= (%s::timestamp AT TIME ZONE 'America/New_York')" in call.args[0]
    for bad in ("", "2027", "soon", "2027-13-01", "2027-02-30"):
        with patch("config.TRACK_RECORD_SINCE", bad, create=True):
            assert analyst._track_record_since() is None


def test_malformed_value_warns_and_is_unbounded(caplog):
    with patch("config.TRACK_RECORD_SINCE", "2027-1-1", create=True):
        with caplog.at_level("WARNING"):
            assert db.track_record_since() is None
    assert "not a YYYY-MM-DD date" in caplog.text


def test_text_sha_hashes_only_the_block_the_model_saw():
    p1 = "x\n=== YOUR TRACK RECORD ===\nConviction 7: 3 trades\n\n=== NEXT ===\nother"
    p2 = "y\n=== YOUR TRACK RECORD ===\nConviction 7: 3 trades\n\n=== NEXT ===\ndifferent"
    p3 = "x\n=== YOUR TRACK RECORD ===\nConviction 7: 4 trades\n\n=== NEXT ===\nother"
    a, b, c = (analyst._track_record_text_sha(p) for p in (p1, p2, p3))
    assert a == b and a != c and len(a) == 12
    assert analyst._track_record_text_sha("no block") is None


def test_the_block_is_followed_by_a_header_so_text_sha_stops_there():
    """text_sha ends the block at the next '\\n===' line; pin that the
    template puts a === header right after it (professor 62g)."""
    import inspect
    src = inspect.getsource(analyst._build_prompt)
    after = src.split("{track_record_section}", 1)[1].lstrip("\n")
    assert after.startswith("=== ")
