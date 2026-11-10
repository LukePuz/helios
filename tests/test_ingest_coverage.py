"""One ingest_coverage activity row per poll that cut or aged out unseen
items (professor m603); none when every count is zero. Log-only."""
from unittest.mock import patch

import tasks


def test_writes_only_when_something_was_cut():
    with patch("db.log_activity") as act:
        tasks._log_ingest_coverage("news", {"cap_hits": {}, "aged_out_n": 0, "aged_out": {}})
        act.assert_not_called()
        tasks._log_ingest_coverage("edgar", {"aged_out_unseen": 2, "aged_out": ["NVDA 8-K x", "MSFT 8-K y"]})
    stage, ticker = act.call_args.args[:2]
    assert stage == "ingest_coverage" and ticker == ""
    assert act.call_args.kwargs["extra"] == {"source": "edgar", "aged_out_unseen": 2,
                                             "aged_out": ["NVDA 8-K x", "MSFT 8-K y"]}


def test_never_raises():
    with patch("db.log_activity", side_effect=RuntimeError("db")):
        tasks._log_ingest_coverage("news", {"aged_out_n": 1})


def test_repeats_are_skipped_until_the_set_changes_and_cap_hits_always_write():
    tasks._INGEST_COVERAGE_LAST.clear()
    extra = {"cap_hits": {}, "aged_out_n": 3, "aged_out": {"google_news:NVDA": 3}}
    with patch("db.log_activity") as act:
        tasks._log_ingest_coverage("news", extra, change_key=frozenset({"google_news:NVDA"}))
        tasks._log_ingest_coverage("news", extra, change_key=frozenset({"google_news:NVDA"}))
        assert act.call_count == 1
        tasks._log_ingest_coverage("news", extra, change_key=frozenset({"google_news:NVDA", "yfinance:MSFT"}))
        assert act.call_count == 2
        tasks._log_ingest_coverage("news", {**extra, "cap_hits": {"google_news": ["NVDA"]}},
                                   change_key=frozenset({"google_news:NVDA", "yfinance:MSFT"}), always=True)
        assert act.call_count == 3


def test_a_backoff_skipped_edgar_pass_writes_no_row():
    from ingestion import edgar
    edgar._AGED_OUT_THIS_PASS[:] = [("acc1", "NVDA 8-K old")]   # left from the previous pass
    tasks._INGEST_COVERAGE_LAST.clear()

    def skipped(on_filing=None):
        edgar.LAST_PASS.clear()
        edgar.LAST_PASS.update({"backoff_skip": True, "poll_failed": False})
        return []
    with patch("ingestion.edgar.poll_once", side_effect=skipped), \
         patch("tasks._run_singleton", side_effect=lambda name, fn, **kw: fn()), \
         patch("db.log_activity") as act:
        tasks.poll_edgar.run()
    assert not any(c.args and c.args[0] == "ingest_coverage" for c in act.call_args_list)
    edgar._AGED_OUT_THIS_PASS.clear()


def test_a_clean_poll_resets_so_the_same_set_is_written_again():
    tasks._INGEST_COVERAGE_LAST.clear()
    key = frozenset({"acc1"})
    with patch("db.log_activity") as act:
        tasks._log_ingest_coverage("edgar", {"aged_out_unseen": 1, "aged_out": ["x"]}, change_key=key)
        tasks._log_ingest_coverage("edgar", {"aged_out_unseen": 0, "aged_out": []}, change_key=frozenset())
        tasks._log_ingest_coverage("edgar", {"aged_out_unseen": 1, "aged_out": ["x"]}, change_key=key)
    assert act.call_count == 2
