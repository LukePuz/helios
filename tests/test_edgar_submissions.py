"""EDGAR polls data.sec.gov submissions JSON, one request per ticker, instead
of browse-edgar atom feeds per ticker per form (300 requests a pass, read
timeouts by the hundred, one pass 1,030 s on a 10 min schedule)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingestion import edgar

_NOW = datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _fixture(now=_NOW):
    """A trimmed submissions JSON (columns as SEC sends them, newest first)."""
    rows = [
        ("4",       "0001127602-26-041234", now - timedelta(hours=1),  "xslF345X05/wk-form4.xml"),
        ("8-K",     "0001326801-26-000110", now - timedelta(hours=2),  "meta-20261103.htm"),
        ("SC 13G",  "0000315066-26-001111", now - timedelta(hours=3),  "sc13g.htm"),
        ("8-K/A",   "0001326801-26-000108", now - timedelta(hours=5),  "meta-8ka.htm"),
        ("10-Q",    "0001326801-26-000105", now - timedelta(hours=30), "meta-10q.htm"),
        ("424B5",   "0001193125-26-222222", now - timedelta(hours=40), "prosupp.htm"),
        ("S-8",     "0001326801-26-000100", now - timedelta(hours=41), "s8.htm"),
        ("10-K",    "0001326801-26-000010", now - timedelta(days=200), "meta-10k.htm"),
    ]
    return {"cik": "1326801", "name": "Meta Platforms, Inc.", "filings": {"recent": {
        "form": [r[0] for r in rows],
        "accessionNumber": [r[1] for r in rows],
        "acceptanceDateTime": [_iso(r[2]) for r in rows],
        "filingDate": [r[2].strftime("%Y-%m-%d") for r in rows],
        "primaryDocument": [r[3] for r in rows],
    }, "files": []}}


def _resp(data=None, status=200, headers=None):
    return MagicMock(status_code=status, headers=headers or {},
                     json=MagicMock(return_value=data))


def test_parse_keeps_wanted_forms_in_window_in_the_old_order():
    rows = edgar._parse_submissions(_fixture(), now=_NOW)
    # By FILING_FORMS order (8-K, 10-Q, 10-K, 4), newest first within a form.
    assert [(r["form"], r["actual_form"]) for r in rows] == [
        ("8-K", "8-K"), ("8-K", "8-K/A"), ("10-Q", "10-Q"), ("4", "4"), ("4", "424B5")]
    assert rows[0]["accepted_at"] == _NOW - timedelta(hours=2)
    assert rows[0]["primary_document"] == "meta-20261103.htm"


def test_form_filter_is_the_atom_prefix_match():
    """browse-edgar's type= was a prefix filter: type=4 returned 424B5 and 425
    (both in the logs as sec_form4), type=8-K returned 8-K/A. Unchanged."""
    assert edgar._wanted_form("8-K") == "8-K"
    assert edgar._wanted_form("8-K/A") == "8-K"
    assert edgar._wanted_form("10-Q") == "10-Q"
    assert edgar._wanted_form("10-K/A") == "10-K"
    assert edgar._wanted_form("4/A") == "4"
    assert edgar._wanted_form("424B5") == "4"
    assert edgar._wanted_form("425") == "4"
    for other in ("SC 13G", "S-8", "DEF 14A", "3", "144", ""):
        assert edgar._wanted_form(other) is None


def test_dedup_key_and_url_match_the_atom_feed():
    """seen_items holds the atom <id> for every filing stored before the
    switch (logged: urn:tag:sec.gov,2008:accession-number=0001848475-26-000005),
    so the same filing must produce the same key and stay seen."""
    acc = "0001848475-26-000005"
    assert edgar._seen_key(acc) == "urn:tag:sec.gov,2008:accession-number=0001848475-26-000005"
    assert edgar._index_url("0001326801", "0001326801-26-000110") == (
        "https://www.sec.gov/Archives/edgar/data/1326801/000132680126000110/"
        "0001326801-26-000110-index.htm")
    assert edgar._title("8-K") == "8-K  - Current report"
    assert edgar._title("8-K/A") == "8-K/A [Amend]  - Current report"
    assert edgar._title("4") == "4  - Statement of changes in beneficial ownership of securities"


def test_already_stored_filing_is_not_reemitted_after_the_switch():
    atom_ids = {"urn:tag:sec.gov,2008:accession-number=0001326801-26-000110"}
    with patch("ingestion.edgar.requests.get", return_value=_resp(_fixture(datetime.now(timezone.utc)))), \
         patch("ingestion.edgar._fetch_filing_text", return_value="body"), \
         patch.object(edgar, "_SEEN_THIS_PASS", set(atom_ids)):
        filings = edgar._get_recent_filings("META", "0001326801")
    keys = [f["_seen_key"] for f in filings]
    assert "urn:tag:sec.gov,2008:accession-number=0001326801-26-000110" not in keys
    assert len(keys) == 4


def test_filing_dict_has_the_same_fields_as_before():
    now = datetime.now(timezone.utc)
    with patch("ingestion.edgar.requests.get", return_value=_resp(_fixture(now))), \
         patch("ingestion.edgar._fetch_filing_text", return_value="body") as text, \
         patch.object(edgar, "_SEEN_THIS_PASS", set()):
        filings = edgar._get_recent_filings("META", "0001326801")
    f8k, f4 = filings[0], filings[3]
    assert set(f8k) == {"ticker", "cik", "form", "actual_form", "source", "type", "title",
                        "url", "timestamp", "text", "_seen_key"}
    # The filing's own form beside the matched prefix (audit 2026-11-07 03:08 #1).
    assert [(f["form"], f["actual_form"]) for f in filings] == [
        ("8-K", "8-K"), ("8-K", "8-K/A"), ("10-Q", "10-Q"), ("4", "4"), ("4", "424B5")]
    assert (f8k["form"], f8k["source"], f8k["type"]) == ("8-K", "sec_filing", "8-K")
    assert (f4["form"], f4["source"], f4["type"]) == ("4", "sec_form4", "Form 4")
    assert f8k["cik"] == "0001326801" and f8k["text"] == "body"
    # Eastern time with an offset, like the atom <updated>.
    ts = datetime.fromisoformat(f8k["timestamp"])
    assert ts.utcoffset() in (timedelta(hours=-4), timedelta(hours=-5))
    assert abs(ts - (now - timedelta(hours=2))) < timedelta(seconds=1)
    text.assert_any_call(f8k["url"], "8-K")


def test_one_request_per_ticker_with_the_user_agent_and_a_timeout():
    with patch("ingestion.edgar.requests.get", return_value=_resp(_fixture())) as get:
        assert edgar._get_submissions("META", "0001326801") is not None
        edgar._get_submissions("ASML", "0937556")
    url, kw = get.call_args_list[0].args[0], get.call_args_list[0].kwargs
    assert url == "https://data.sec.gov/submissions/CIK0001326801.json"
    assert get.call_args_list[1].args[0] == "https://data.sec.gov/submissions/CIK0000937556.json"
    assert kw["headers"] is edgar.HEADERS and "User-Agent" in kw["headers"]
    assert kw["timeout"] == edgar._REQUEST_TIMEOUT


def test_retry_with_backoff_is_bounded():
    with patch("ingestion.edgar.requests.get", side_effect=requests.ReadTimeout("slow")) as get, \
         patch("ingestion.edgar.time.sleep") as sleep:
        assert edgar._get_submissions("META", "1") is None
    assert get.call_count == edgar._MAX_ATTEMPTS == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0]


@pytest.fixture(autouse=True)
def _fresh_tz_and_state():
    """No timezone decision carried between tests; system_state in a dict."""
    state = {}

    def get_strict(k, d=None):
        return state.get(k, d)

    with patch.object(edgar, "_TZ_DECISION", None), patch.object(edgar, "_TZ_LOGGED", False), \
         patch.dict(edgar._TZ_POOL, {"rows": 0, "morning": 0, "late": 0}), \
         patch("db.get_state", side_effect=get_strict), \
         patch("db.get_state_strict", side_effect=get_strict), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)), \
         patch("tasks._reconcile_alert_due", return_value=True):
        yield state


def test_429_stops_at_once_with_retry_after():
    with patch("ingestion.edgar.requests.get",
               return_value=_resp(status=429, headers={"Retry-After": "120"})) as get, \
         patch("ingestion.edgar.time.sleep") as sleep:
        with pytest.raises(edgar._RateLimited) as exc:
            edgar._get_submissions("META", "1")
    assert exc.value.retry_after == 120.0
    assert get.call_count == 1 and not sleep.called


def test_5xx_is_retried_then_succeeds():
    seq = [_resp(status=503), _resp(status=502), _resp({"ok": 1})]
    with patch("ingestion.edgar.requests.get", side_effect=seq) as get, \
         patch("ingestion.edgar.time.sleep") as sleep:
        assert edgar._get_submissions("META", "1") == {"ok": 1}
    assert get.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0]


def test_404_is_not_retried():
    with patch("ingestion.edgar.requests.get", return_value=_resp(status=404)) as get, \
         patch("ingestion.edgar.time.sleep") as sleep:
        assert edgar._get_submissions("META", "1") is None
    assert get.call_count == 1 and not sleep.called


def test_one_ticker_failing_does_not_stop_the_others():
    def get(url, **kw):
        if "CIK0000000002" in url:
            raise requests.ConnectionError("reset")
        if "CIK0000000003" in url:
            return _resp({"filings": {"recent": {"form": ["8-K"], "accessionNumber": [None]}}})
        return _resp(_fixture(datetime.now(timezone.utc)))

    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1", "BBB": "2", "CCC": "3", "DDD": "4"}), \
         patch("ingestion.edgar.requests.get", side_effect=get), \
         patch("ingestion.edgar.time.sleep"), \
         patch("ingestion.edgar._fetch_filing_text", return_value=""), \
         patch("ingestion.edgar._note_pass_health") as health, \
         patch("db.seen_ids", return_value=set()):
        filings = edgar.poll_once()
    assert {f["ticker"] for f in filings} == {"AAA", "DDD"}
    assert edgar.LAST_PASS["ok"] == 3 and edgar.LAST_PASS["failed"] == 1
    assert edgar.LAST_PASS["poll_failed"] is False
    assert health.call_args.args[0] is False


def _run_pass(tickers, results, **extra):
    """poll_once over tickers, _get_recent_filings answering from results
    ({ticker: list | None | Exception}); returns the tickers fetched."""
    fetched = []

    def recent(t, c):
        fetched.append(t)
        r = results.get(t, [])
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(edgar, "TICKER_TO_CIK", tickers), \
         patch("ingestion.edgar._get_recent_filings", side_effect=recent), \
         patch("ingestion.edgar.time.sleep"), \
         patch("ingestion.edgar._note_pass_health") as health, \
         patch("db.seen_ids", return_value=set()):
        edgar.poll_once()
    return fetched, health


def test_pass_stops_after_five_consecutive_failures_anywhere():
    tickers = {f"T{i:02d}": str(i) for i in range(20)}
    # 3 good, then failures from T03 on: stops after T03..T07.
    results = {f"T{i:02d}": None for i in range(3, 20)}
    fetched, health = _run_pass(tickers, results)
    assert fetched == [f"T{i:02d}" for i in range(8)]
    assert "5 tickers in a row failed" in edgar.LAST_PASS["aborted"]
    assert edgar.LAST_PASS["skipped"] == 12
    # 3 of 20 ok is under half: a failed pass.
    assert edgar.LAST_PASS["poll_failed"] is True and health.call_args.args[0] is True


def test_consecutive_abort_with_half_or_more_ok_is_not_a_failed_pass():
    tickers = {f"T{i:02d}": str(i) for i in range(20)}
    results = {f"T{i:02d}": None for i in range(10, 20)}
    fetched, health = _run_pass(tickers, results)
    assert len(fetched) == 15 and edgar.LAST_PASS["ok"] == 10
    assert edgar.LAST_PASS["poll_failed"] is False


def test_scattered_failures_do_not_abort():
    tickers = {f"T{i:02d}": str(i) for i in range(20)}
    results = {f"T{i:02d}": None for i in range(0, 20, 2)}      # every other one fails
    fetched, _ = _run_pass(tickers, results)
    assert len(fetched) == 20 and not edgar.LAST_PASS["aborted"]


def test_start_rotates_to_where_the_last_pass_stopped(_fresh_tz_and_state):
    state = _fresh_tz_and_state
    tickers = {f"T{i}": str(i) for i in range(6)}
    clock = iter([0, 0, 0, 0, 0, 1000] + [2000] * 50)   # budget runs out before the 5th ticker
    with patch("ingestion.edgar.time.monotonic", side_effect=lambda: next(clock)):
        fetched, _ = _run_pass(tickers, {})
    assert fetched == ["T0", "T1", "T2", "T3"]
    assert edgar.LAST_PASS["poll_failed"] is False          # out of time with successes
    key = [k for k in state if k.startswith("edgar_next_start_")][0]
    assert state[key] == "4"
    fetched, _ = _run_pass(tickers, {})
    assert fetched == ["T4", "T5", "T0", "T1", "T2", "T3"]
    assert state[key] == "4"


def test_429_stops_the_pass_and_sets_a_backoff(_fresh_tz_and_state):
    state = _fresh_tz_and_state
    tickers = {f"T{i}": str(i) for i in range(10)}
    before = datetime.now(timezone.utc)
    fetched, health = _run_pass(tickers, {"T2": edgar._RateLimited(30.0)})
    assert fetched == ["T0", "T1", "T2"]
    assert "429" in edgar.LAST_PASS["aborted"]
    key = [k for k in state if k.startswith("edgar_backoff_until_")][0]
    until = datetime.fromisoformat(state[key])
    assert timedelta(seconds=59) < until - before < timedelta(seconds=65)   # max(30, 60)
    # The next pass skips without fetching and isn't counted as failed.
    fetched, health = _run_pass(tickers, {})
    assert fetched == [] and edgar.LAST_PASS["backoff_skip"] is True
    assert not health.called


def test_429_backoff_uses_a_longer_retry_after(_fresh_tz_and_state):
    state = _fresh_tz_and_state
    before = datetime.now(timezone.utc)
    _run_pass({"A": "1"}, {"A": edgar._RateLimited(600.0)})
    until = datetime.fromisoformat(next(v for k, v in state.items() if k.startswith("edgar_backoff_until_")))
    assert timedelta(seconds=599) < until - before < timedelta(seconds=605)


def _backoff_after(state, retry_after):
    before = datetime.now(timezone.utc)
    state.clear()
    _run_pass({"A": "1"}, {"A": edgar._RateLimited(retry_after)})
    until = datetime.fromisoformat(next(v for k, v in state.items() if k.startswith("edgar_backoff_until_")))
    return until - before


def test_429_backoff_is_capped_at_an_hour(_fresh_tz_and_state):
    assert timedelta(seconds=3599) < _backoff_after(_fresh_tz_and_state, 10 ** 9) < timedelta(seconds=3605)


def test_429_backoff_with_infinite_or_nan_retry_after_uses_the_minimum(_fresh_tz_and_state):
    for value in (float("inf"), float("nan")):
        assert timedelta(seconds=59) < _backoff_after(_fresh_tz_and_state, value) < timedelta(seconds=65)


def test_non_finite_or_unparseable_retry_after_header_is_none():
    for header in ("inf", "nan", "Wed, 21 Oct 2026 07:28:00 GMT", None):
        with patch("ingestion.edgar.requests.get",
                   return_value=_resp(status=429, headers={"Retry-After": header})):
            with pytest.raises(edgar._RateLimited) as exc:
                edgar._get_submissions("META", "1")
        assert exc.value.retry_after is None, header


def test_poll_resumes_after_the_backoff(_fresh_tz_and_state):
    state = _fresh_tz_and_state
    state["edgar_backoff_until_paper"] = state["edgar_backoff_until_live"] = \
        (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    fetched, health = _run_pass({"A": "1"}, {})
    assert fetched == ["A"] and health.called


def test_sleep_between_tickers_keeps_under_ten_per_second():
    with patch.object(edgar, "TICKER_TO_CIK", {"A": "1", "B": "2", "C": "3"}), \
         patch("ingestion.edgar._get_recent_filings", return_value=[]), \
         patch("ingestion.edgar.time.sleep") as sleep, \
         patch("ingestion.edgar._note_pass_health"), \
         patch("db.seen_ids", return_value=set()):
        edgar.poll_once()
    assert [c.args[0] for c in sleep.call_args_list] == [edgar._REQUEST_GAP_S] * 2
    assert edgar._REQUEST_GAP_S >= 0.1


def test_stale_alert_once_per_spell_recovery_after_two_good_passes():
    with patch("alerts._send") as send:
        for _ in range(edgar._STALE_AFTER_FAILED_PASSES + 4):
            edgar._note_pass_health(True, "0 of 75 tickers ok")
        assert send.call_count == 1 and "EDGAR stale" in send.call_args.args[0]
        edgar._note_pass_health(False)
        assert send.call_count == 1                       # one good pass: not yet
        edgar._note_pass_health(False)
        assert send.call_count == 2 and "EDGAR recovered after 7 failed polls" in send.call_args.args[0]
        edgar._note_pass_health(False)
        assert send.call_count == 2


def test_a_failure_between_good_passes_keeps_the_spell_open():
    with patch("alerts._send") as send:
        for _ in range(edgar._STALE_AFTER_FAILED_PASSES):
            edgar._note_pass_health(True)
        edgar._note_pass_health(False)
        edgar._note_pass_health(True)                     # spell continues, no second alert
        edgar._note_pass_health(False)
        assert send.call_count == 1
        edgar._note_pass_health(False)
        assert send.call_count == 2 and "recovered" in send.call_args.args[0]


def test_stale_alert_is_throttled_across_spells():
    due = iter([True, False])
    with patch("alerts._send") as send, \
         patch("tasks._reconcile_alert_due", side_effect=lambda k: next(due)) as gate:
        for spell in range(2):
            for _ in range(edgar._STALE_AFTER_FAILED_PASSES):
                edgar._note_pass_health(True)
            edgar._note_pass_health(False)
            edgar._note_pass_health(False)
    stale = [c for c in send.call_args_list if "EDGAR stale" in c.args[0]]
    assert len(stale) == 1
    assert [c.args[0] for c in gate.call_args_list] == ["edgar_stale", "edgar_stale"]


def test_short_failure_spell_sends_nothing():
    with patch("alerts._send") as send:
        for _ in range(edgar._STALE_AFTER_FAILED_PASSES - 1):
            edgar._note_pass_health(True)
        edgar._note_pass_health(False)
        edgar._note_pass_health(False)
    assert not send.called


def test_edgar_recovered_is_a_low_alert_and_stale_is_critical():
    import alert_policy
    assert alert_policy.classify("✅ EDGAR recovered after 4 failed polls.") == (
        alert_policy.LOW, "edgar_recovered")
    assert alert_policy.classify("⚠️ EDGAR stale: 3 polls failed (0 of 75 tickers ok).")[0] == \
        alert_policy.CRITICAL


def test_pass_health_never_raises():
    with patch("db.get_state_strict", side_effect=RuntimeError("db down")):
        edgar._note_pass_health(True)


def test_failed_pass_records_a_heartbeat_failure_and_backoff_skip_records_nothing():
    import tasks
    passes = iter([
        {"tickers": 75, "ok": 0, "failed": 5, "poll_failed": True, "aborted": "5 tickers in a row failed"},
        {"backoff_skip": True, "poll_failed": False},
    ])

    def fake_poll(on_filing=None):
        edgar.LAST_PASS.clear()
        edgar.LAST_PASS.update(next(passes))
        return []

    with patch("ingestion.edgar.poll_once", side_effect=fake_poll), \
         patch("tasks._heartbeat_ok") as ok, patch("tasks._heartbeat_fail") as fail:
        tasks.poll_edgar.run()
        assert not ok.called
        assert fail.call_args.args[0] == "edgar" and "5 tickers in a row failed" in fail.call_args.args[1]
        fail.reset_mock()
        tasks.poll_edgar.run()
        assert not ok.called and not fail.called
    edgar.LAST_PASS.clear()


def _rows(rows, desc=None):
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "accessionNumber": [r[1] for r in rows],
        "acceptanceDateTime": [r[2] if isinstance(r[2], str) else _iso(r[2]) for r in rows],
        "filingDate": [(r[2] if isinstance(r[2], str) else _iso(r[2]))[:10] for r in rows],
        "primaryDocument": ["d.htm" for _ in rows],
        "primaryDocDescription": desc or ["" for _ in rows],
    }}}


def test_only_the_five_newest_per_form_are_emitted_like_the_atom_count():
    """The atom query asked for count=5 per form: 7 recent Form 4s gave 5."""
    # Deliberately out of order: the cap follows acceptanceDateTime.
    hours = [3, 1, 7, 2, 6, 4, 5]
    data = _rows([("4", f"0000000001-26-00000{h}", _NOW - timedelta(hours=h)) for h in hours]
                 + [("8-K", "0000000002-26-000001", _NOW - timedelta(hours=8))])
    rows = edgar._parse_submissions(data, now=_NOW)
    assert [r["accession"] for r in rows if r["form"] == "4"] == [
        f"0000000001-26-00000{h}" for h in (1, 2, 3, 4, 5)]
    assert [r["form"] for r in rows] == ["8-K"] + ["4"] * 5      # the cap is per form


def test_seven_recent_form_4s_emit_only_the_newest_five_filings():
    now = datetime.now(timezone.utc)
    data = _rows([("4", f"0000000001-26-00000{h}", now - timedelta(hours=h)) for h in range(1, 8)])
    with patch("ingestion.edgar.requests.get", return_value=_resp(data)), \
         patch("ingestion.edgar._fetch_filing_text", return_value=""), \
         patch.object(edgar, "_SEEN_THIS_PASS", set()):
        filings = edgar._get_recent_filings("META", "0001326801")
    assert [f["_seen_key"] for f in filings] == [
        edgar._seen_key(f"0000000001-26-00000{h}") for h in range(1, 6)]


def test_cap_counts_424b_and_425_under_form_4_as_the_atom_prefix_did():
    data = _rows([("424B5", "0000000003-26-000001", _NOW - timedelta(hours=1))]
                 + [("4", f"0000000001-26-00000{h}", _NOW - timedelta(hours=h)) for h in (2, 3, 4, 5, 6)])
    rows = edgar._parse_submissions(data, now=_NOW)
    assert [r["actual_form"] for r in rows] == ["424B5", "4", "4", "4", "4"]


def test_cap_comes_before_the_age_filter():
    """Five newer filings push a sixth out, as count=5 did, whatever their age."""
    data = _rows([("8-K", f"0000000001-26-00000{d}", _NOW - timedelta(days=d)) for d in (3, 4, 5, 6, 7)]
                 + [("8-K", "0000000001-26-000009", _NOW - timedelta(days=9))])
    assert edgar._parse_submissions(data, now=_NOW) == []
    later = edgar._parse_submissions(data, now=_NOW - timedelta(days=10))
    assert len(later) == 5 and "0000000001-26-000009" not in [r["accession"] for r in later]


# ── acceptanceDateTime timezone ─────────────────────────────────────────────

def _tz_fixture(hours, form="4", start=0):
    """len(hours) rows of form, the first (newest) on 11-03, the rest on 11-02/11-01."""
    rows = []
    for n, h in enumerate(hours):
        day = 3 if n == 0 else (2 if n % 2 else 1)
        rows.append((form, f"0000000001-26-{start + n:06d}", f"2026-11-{day:02d}T{h:02d}:05:11.000Z"))
    return _rows(rows)


def _recent(data):
    return data["filings"]["recent"]


_LATE_NOW = datetime(2026, 11, 3, 23, tzinfo=timezone.utc)


def test_eastern_style_data_with_pre_market_rows_reads_as_eastern():
    # 12 rows at raw 06-09 (pre-market, ET), none at 22-05.
    data = _tz_fixture([16] + [7, 8] * 6 + [10, 12, 14, 15, 17, 19, 21] * 2)
    tz, counts = edgar._acceptance_tz(_recent(data))
    assert tz is edgar._ET and edgar._TZ_DECISION is edgar._ET
    assert counts == {"rows": 27, "morning": 12, "late": 0}
    rows = edgar._parse_submissions(data, now=_LATE_NOW, tz=tz)
    newest = [r for r in rows if r["accession"] == "0000000001-26-000000"][0]
    # 16:05:11 ET on 11-03 (EST, -05:00) is 21:05:11 UTC.
    assert newest["accepted_at"] == datetime(2026, 11, 3, 21, 5, 11, tzinfo=timezone.utc)


def test_utc_style_data_with_evening_rows_reads_as_utc():
    # Raw 22-05 = 17:00-01:00 ET: after-close 8-Ks and Form 4s; none at raw 06-09.
    data = _tz_fixture([21] + [22, 23, 0, 1] * 3 + [12, 14, 16, 18, 20])
    tz, counts = edgar._acceptance_tz(_recent(data))
    assert tz == timezone.utc and edgar._TZ_DECISION == timezone.utc
    assert counts == {"rows": 18, "morning": 0, "late": 12}
    rows = edgar._parse_submissions(data, now=_LATE_NOW, tz=tz)
    newest = [r for r in rows if r["accession"] == "0000000001-26-000000"][0]
    assert newest["accepted_at"] == datetime(2026, 11, 3, 21, 5, 11, tzinfo=timezone.utc)


def test_small_business_hours_filer_leaves_it_undecided():
    data = _tz_fixture([16] + [10, 11, 12, 13, 14, 15, 16] * 3)
    tz, counts = edgar._acceptance_tz(_recent(data))
    assert tz == timezone.utc and edgar._TZ_DECISION is None
    assert counts == {"rows": 22, "morning": 0, "late": 0}
    rows = edgar._parse_submissions(data, now=_LATE_NOW)
    newest = [r for r in rows if r["accession"] == "0000000001-26-000000"][0]
    assert newest["accepted_at"] == datetime(2026, 11, 3, 16, 5, 11, tzinfo=timezone.utc)


def test_one_odd_row_does_not_flip_a_clear_eastern_decision():
    # 120 rows: 12 pre-market, one at raw 23 (1 of 120 is under 1%), the rest midday.
    data = _tz_fixture([16] + [7] * 12 + [23] + [10, 11, 12, 13, 14, 15, 16] * 15 + [12])
    tz, counts = edgar._acceptance_tz(_recent(data))
    assert counts == {"rows": 120, "morning": 12, "late": 1}
    assert tz is edgar._ET


def test_too_many_late_rows_for_eastern_stays_undecided():
    data = _tz_fixture([16] + [7] * 12 + [23, 22] + [12] * 50)   # 2 of 65 late is over 1%
    tz, _ = edgar._acceptance_tz(_recent(data))
    assert tz == timezone.utc and edgar._TZ_DECISION is None


def test_pooling_across_two_responses_reaches_a_decision():
    first = _tz_fixture([16] + [7] * 6 + [12] * 10)
    tz, counts = edgar._acceptance_tz(_recent(first))
    assert edgar._TZ_DECISION is None and counts["morning"] == 6
    second = _tz_fixture([15] + [8] * 5 + [13] * 10, start=100)
    tz, counts = edgar._acceptance_tz(_recent(second))
    assert tz is edgar._ET and counts == {"rows": 33, "morning": 11, "late": 0}


def test_only_wanted_form_rows_are_pooled():
    data = _tz_fixture([7] * 20, form="S-8")
    tz, counts = edgar._acceptance_tz(_recent(data))
    assert counts == {"rows": 0, "morning": 0, "late": 0} and edgar._TZ_DECISION is None


def test_the_decision_is_cached_for_the_process():
    edgar._acceptance_tz(_recent(_tz_fixture([16] + [7, 8] * 6)))
    assert edgar._TZ_DECISION is edgar._ET
    tz, _ = edgar._acceptance_tz(_recent(_tz_fixture([21] + [22, 23, 1] * 20)))
    assert tz is edgar._ET
    rows = edgar._parse_submissions(_tz_fixture([16]), now=_LATE_NOW)     # default: the decision
    assert rows[0]["accepted_at"] == datetime(2026, 11, 3, 21, 5, 11, tzinfo=timezone.utc)


def test_decision_is_logged_once_with_its_counts(caplog):
    undecided = _tz_fixture([16] + [12] * 10)
    deciding = _tz_fixture([16] + [7, 8] * 6, start=100)
    responses = iter([undecided, deciding, deciding])
    with patch("ingestion.edgar.requests.get", side_effect=lambda *a, **k: _resp(next(responses))), \
         patch("ingestion.edgar._fetch_filing_text", return_value=""), \
         patch.object(edgar, "_SEEN_THIS_PASS", set()), \
         caplog.at_level("INFO", logger="ingestion.edgar"):
        for _ in range(3):
            edgar._get_recent_filings("META", "0001326801")
        assert edgar._TZ_LOGGED is True
    lines = [r.getMessage() for r in caplog.records if "acceptanceDateTime check" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "read as ET (24 rows pooled, 12 at raw 06-09, 0 at raw 22-05)" in line
    assert "0000000001-26-000100 raw='2026-11-03T16:05:11.000Z'" in line
    assert "parsed_et=2026-11-03T16:05:11-05:00" in line


# ── Titles ──────────────────────────────────────────────────────────────────

def test_title_falls_back_to_primary_doc_description():
    assert edgar._title("424B7", "Prospectus supplement") == "424B7  - Prospectus supplement"
    assert edgar._title("424B7/A", "Amended supplement") == "424B7/A [Amend]  - Amended supplement"
    assert edgar._title("8-K", "ignored when known") == "8-K  - Current report"
    now = datetime.now(timezone.utc)
    data = _rows([("424B7", "0000000001-26-000001", now - timedelta(hours=1))],
                 desc=["Prospectus supplement"])
    with patch("ingestion.edgar.requests.get", return_value=_resp(data)), \
         patch("ingestion.edgar._fetch_filing_text", return_value=""), \
         patch.object(edgar, "_SEEN_THIS_PASS", set()):
        filings = edgar._get_recent_filings("META", "0001326801")
    assert filings[0]["title"] == "424B7  - Prospectus supplement"


def test_unseen_filings_just_past_the_age_cutoff_are_counted_not_dispatched():
    """A filing accepted Friday evening is ~58 h old at Monday's first poll:
    the 48 h filter drops it; it is now counted and logged (audit 11-06 17:34 #4)."""
    now = datetime.now(timezone.utc)
    data = _fixture(now)
    rec = data["filings"]["recent"]
    rec["form"].insert(0, "8-K")
    rec["accessionNumber"].insert(0, "0001326801-26-000099")
    rec["acceptanceDateTime"].insert(0, _iso(now - timedelta(hours=58)))
    rec["filingDate"].insert(0, (now - timedelta(hours=58)).strftime("%Y-%m-%d"))
    rec["primaryDocument"].insert(0, "meta-fri.htm")
    edgar._AGED_OUT_THIS_PASS.clear()
    with patch("ingestion.edgar._is_seen", return_value=False):
        edgar._note_aged_out("META", data, timezone.utc)
    assert len(edgar._AGED_OUT_THIS_PASS) == 1
    acc, label = edgar._AGED_OUT_THIS_PASS[0]
    assert acc == "0001326801-26-000099" and label.startswith("META 8-K ")
    edgar._AGED_OUT_THIS_PASS.clear()
    with patch("ingestion.edgar._is_seen", return_value=True):
        edgar._note_aged_out("META", data, timezone.utc)
    assert edgar._AGED_OUT_THIS_PASS == []
    with patch("ingestion.edgar._is_seen", return_value=False):
        assert all(r["accession"] != "0001326801-26-000099"
                   for r in edgar._parse_submissions(data, tz=timezone.utc))



def test_an_aged_out_filing_is_warned_once_but_counted_every_pass(caplog):
    item = ("0001326801-26-000099", "META 8-K 2026-11-06T19:30-05:00")
    edgar._AGED_OUT_WARNED.discard(item[0])

    def recent(ticker, cik):
        edgar._AGED_OUT_THIS_PASS.append(item)
        return []
    counts = []
    with patch("ingestion.edgar.TICKER_TO_CIK", {"META": "1"}), \
         patch("ingestion.edgar._get_recent_filings", side_effect=recent), \
         patch("ingestion.edgar._get_state", return_value=None), patch("ingestion.edgar._set_state"), \
         patch("ingestion.edgar._backoff_until", return_value=None), patch("ingestion.edgar._note_pass_health"), \
         patch("db.seen_ids", return_value=set()), caplog.at_level("WARNING"):
        for _ in range(2):
            edgar.poll_once()
            counts.append(edgar.LAST_PASS["aged_out_unseen"])
    assert counts == [1, 1]
    assert caplog.text.count("META 8-K 2026-11-06T19:30") == 1
