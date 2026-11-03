"""insider.py finds Form 4 filings in data.sec.gov submissions JSON through
edgar.py's helpers, instead of one browse-edgar atom request per ticker
(type=4, count=10). Same filings, same seen keys, same payload."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingestion import edgar, insider


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _data(rows):
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "accessionNumber": [r[1] for r in rows],
        "acceptanceDateTime": [_iso(r[2]) for r in rows],
        "filingDate": [r[2].strftime("%Y-%m-%d") for r in rows],
        "primaryDocument": ["d.xml" for _ in rows],
    }}}


def _resp(data=None, status=200, headers=None):
    return MagicMock(status_code=status, headers=headers or {},
                     json=MagicMock(return_value=data))


_BUY = {"owner_name": "JANE DOE", "is_officer": True, "is_director": False,
        "officer_title": "CFO", "total_shares_bought": 1000.0, "total_value_bought": 50_000.0}


@pytest.fixture(autouse=True)
def _state():
    state = {}
    with patch.object(edgar, "_TZ_DECISION", None), patch.object(edgar, "_TZ_LOGGED", False), \
         patch.dict(edgar._TZ_POOL, {"rows": 0, "morning": 0, "late": 0}), \
         patch("db.get_state", side_effect=lambda k, d=None: state.get(k, d)), \
         patch("db.set_state", side_effect=lambda k, v: state.__setitem__(k, v)):
        yield state


def _search(data, seen=(), parsed=_BUY):
    """insider._search_form4 on one response; returns (results, docs looked up, marked seen)."""
    looked_up, marked = [], []
    with patch("ingestion.edgar.requests.get", return_value=_resp(data)) as get, \
         patch("ingestion.insider._is_seen", side_effect=lambda k: k in seen), \
         patch("ingestion.insider._mark_seen", side_effect=marked.append), \
         patch("ingestion.insider._find_primary_doc_url",
               side_effect=lambda u: looked_up.append(u) or u + "#xml"), \
         patch("ingestion.insider._parse_form4_transactions", return_value=parsed):
        results = insider._search_form4("META", "0001326801")
    assert get.call_count == 1
    assert get.call_args.args[0] == "https://data.sec.gov/submissions/CIK0001326801.json"
    return results, looked_up, marked


def test_ten_newest_form4_prefix_filings_any_age_like_count_10():
    now = datetime.now(timezone.utc)
    rows = [("4", f"0000000001-26-{d:06d}", now - timedelta(days=d)) for d in range(1, 13)]
    rows += [("8-K", "0000000002-26-000001", now), ("424B5", "0000000003-26-000001", now - timedelta(hours=1)),
             ("4/A", "0000000003-26-000002", now - timedelta(hours=2))]
    results, looked_up, _ = _search(_data(rows))
    accs = [u.split("/")[-1].replace("-index.htm", "") for u in looked_up]
    # 424B5 and 4/A count under type=4 (prefix), the 8-K doesn't; days 9+ fall past the cap.
    assert accs == ["0000000003-26-000001", "0000000003-26-000002"] + [
        f"0000000001-26-{d:06d}" for d in range(1, 9)]
    assert len(results) == 10
    # No age window: an 8-day-old Form 4 still counts, as in the atom query.
    assert "0000000001-26-000008" in accs


def test_seen_key_url_and_payload_match_the_atom_entry():
    acc = "0001127602-26-041234"
    accepted = datetime(2026, 11, 3, 21, 5, 11, tzinfo=timezone.utc)
    results, _, _ = _search(_data([("4", acc, accepted)]))
    r = results[0]
    assert r["_seen_key"] == "urn:tag:sec.gov,2008:accession-number=0001127602-26-041234"
    assert r["url"] == ("https://www.sec.gov/Archives/edgar/data/1326801/"
                        "000112760226041234/0001127602-26-041234-index.htm")
    assert set(r) == {"source", "ticker", "form", "title", "filing_date", "url", "text", "_seen_key"}
    assert (r["source"], r["form"], r["ticker"]) == ("sec_form4", "4", "META")
    assert r["filing_date"] == "2026-11-03T16:05:11-05:00"       # ET with offset, like <updated>
    assert "JANE DOE (CFO) bought 1,000 shares ($50,000)" in r["title"]


def test_already_seen_filings_are_skipped_and_non_buys_marked_seen():
    now = datetime.now(timezone.utc)
    rows = [("4", "0000000001-26-000001", now), ("4", "0000000001-26-000002", now - timedelta(hours=1))]
    seen = {edgar._seen_key("0000000001-26-000001")}
    results, looked_up, marked = _search(_data(rows), seen=seen, parsed=None)
    assert results == [] and len(looked_up) == 1
    assert marked == [edgar._seen_key("0000000001-26-000002")]


def test_failed_fetch_returns_none():
    with patch("ingestion.edgar.requests.get", return_value=_resp(status=404)):
        assert insider._search_form4("META", "1") is None


def _poll(tickers, results):
    fetched = []

    def search(t, c):
        fetched.append(t)
        r = results.get(t, [])
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(insider, "TICKER_TO_CIK", tickers), \
         patch("ingestion.insider._search_form4", side_effect=search), \
         patch("ingestion.insider.time.sleep"):
        out = insider.poll_once()
    return fetched, out


def test_poll_stops_after_five_consecutive_failures():
    tickers = {f"T{i:02d}": str(i) for i in range(20)}
    fetched, _ = _poll(tickers, {f"T{i:02d}": None for i in range(2, 20)})
    assert fetched == [f"T{i:02d}" for i in range(7)]


def test_429_sets_the_shared_backoff_and_the_next_poll_skips(_state):
    tickers = {f"T{i}": str(i) for i in range(5)}
    before = datetime.now(timezone.utc)
    fetched, _ = _poll(tickers, {"T1": edgar._RateLimited(10 ** 9)})
    assert fetched == ["T0", "T1"]
    key = [k for k in _state if k.startswith("edgar_backoff_until_")]
    assert len(key) == 1
    until = datetime.fromisoformat(_state[key[0]])
    assert timedelta(seconds=3599) < until - before < timedelta(seconds=3605)   # edgar's cap
    fetched, out = _poll(tickers, {})
    assert fetched == [] and out == []


def test_edgar_backoff_also_pauses_insider(_state):
    _state["edgar_backoff_until_paper"] = _state["edgar_backoff_until_live"] = \
        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    fetched, _ = _poll({"A": "1"}, {})
    assert fetched == []


def test_one_failure_does_not_stop_the_pass_and_results_are_collected():
    fetched, out = _poll({"A": "1", "B": "2", "C": "3"},
                         {"A": [{"x": 1}], "B": RuntimeError("bad"), "C": [{"x": 3}]})
    assert fetched == ["A", "B", "C"] and out == [{"x": 1}, {"x": 3}]


# ── Transient per-filing fetch failures ─────────────────────────────────────

_XML = """<?xml version="1.0"?><ownershipDocument><nonDerivativeTable>
<nonDerivativeTransaction><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""


def _doc(text="", status=200):
    r = MagicMock(status_code=status, text=text, content=text.encode())
    if status >= 400:
        err = requests.HTTPError(f"HTTP {status}")
        err.response = MagicMock(status_code=status)
        r.raise_for_status.side_effect = err
    return r


def _search_with_docs(accepted, index, xml):
    """_search_form4 over one Form 4 accepted at `accepted`, the index and
    XML fetches answered by index/xml (a response or an exception)."""
    if accepted is None:      # no acceptanceDateTime and no filingDate
        data = {"filings": {"recent": {"form": ["4"], "accessionNumber": ["0000000001-26-000001"],
                                       "acceptanceDateTime": [""], "filingDate": [""]}}}
    else:
        data = _data([("4", "0000000001-26-000001", accepted)])
    marked = []

    def get(url, **kw):
        if "data.sec.gov" in url:
            return _resp(data)
        r = index if url.endswith("-index.htm") else xml
        if isinstance(r, Exception):
            raise r
        return r

    with patch("requests.get", side_effect=get), \
         patch("ingestion.insider._is_seen", return_value=False), \
         patch("ingestion.insider._mark_seen", side_effect=marked.append):
        results = insider._search_form4("META", "0001326801")
    return results, marked


_INDEX = _doc('<a href="/Archives/edgar/data/1/000000000126000001/form4.xml">x</a>')


@pytest.mark.parametrize("xml", [
    pytest.param(requests.ReadTimeout("slow"), id="timeout"),
    pytest.param(requests.ConnectionError("reset"), id="connection"),
    pytest.param(requests.exceptions.ChunkedEncodingError("cut"), id="chunked"),
    pytest.param(_doc(status=503), id="5xx")])
def test_transient_xml_failure_leaves_a_recent_filing_unseen(xml):
    results, marked = _search_with_docs(datetime.now(timezone.utc) - timedelta(hours=2), _INDEX, xml)
    assert results == [] and marked == []


def _doc_429(retry_after):
    r = _doc(status=429)
    r.raise_for_status.side_effect.response = MagicMock(status_code=429,
                                                        headers={"Retry-After": retry_after})
    return r


@pytest.mark.parametrize("which", ["index", "xml"])
def test_429_on_a_filing_fetch_raises_rate_limited_and_leaves_it_unseen(which):
    index, xml = (_doc_429("120"), None) if which == "index" else (_INDEX, _doc_429("120"))
    with pytest.raises(edgar._RateLimited) as exc:
        _search_with_docs(datetime.now(timezone.utc) - timedelta(hours=2), index, xml)
    assert exc.value.retry_after == 120.0


def test_429_on_a_filing_fetch_starts_the_shared_backoff_and_stops_the_pass(_state):
    data = _data([("4", "0000000001-26-000001", datetime.now(timezone.utc))])
    marked, fetched = [], []

    def get(url, **kw):
        fetched.append(url)
        if "data.sec.gov" in url:
            return _resp(data)
        return _INDEX if url.endswith("-index.htm") else _doc_429("600")

    before = datetime.now(timezone.utc)
    with patch.object(insider, "TICKER_TO_CIK", {"A": "1", "B": "2"}), \
         patch("requests.get", side_effect=get), \
         patch("ingestion.insider.time.sleep"), \
         patch("ingestion.insider._is_seen", return_value=False), \
         patch("ingestion.insider._mark_seen", side_effect=marked.append):
        assert insider.poll_once() == []
    assert marked == []
    assert not any("CIK0000000002" in u for u in fetched)          # B never polled
    assert "429" in insider.LAST_PASS["aborted"]
    until = datetime.fromisoformat(next(v for k, v in _state.items()
                                        if k.startswith("edgar_backoff_until_")))
    assert timedelta(seconds=599) < until - before < timedelta(seconds=605)


def test_transient_failure_without_an_acceptance_time_marks_it_seen():
    results, marked = _search_with_docs(None, _INDEX, requests.ReadTimeout("slow"))
    assert results == [] and marked == [edgar._seen_key("0000000001-26-000001")]


def test_transient_index_failure_leaves_a_recent_filing_unseen():
    results, marked = _search_with_docs(datetime.now(timezone.utc) - timedelta(hours=2),
                                        requests.ReadTimeout("slow"), None)
    assert results == [] and marked == []


def test_transient_failure_on_a_filing_past_the_retry_age_marks_it_seen():
    old = datetime.now(timezone.utc) - timedelta(hours=insider._TRANSIENT_RETRY_MAX_AGE_H + 1)
    results, marked = _search_with_docs(old, _INDEX, requests.ReadTimeout("slow"))
    assert results == [] and marked == [edgar._seen_key("0000000001-26-000001")]


@pytest.mark.parametrize("index,xml", [
    pytest.param(_INDEX, _doc(_XML), id="non-buy"),
    pytest.param(_INDEX, _doc("<not xml"), id="bad-xml"),
    pytest.param(_INDEX, _doc(status=404), id="xml-404"),
    pytest.param(_doc(status=404), None, id="index-404"),
    pytest.param(_doc("<html>no xml links</html>"), None, id="no-xml-link")])
def test_real_non_buys_and_unreadable_filings_are_marked_seen(index, xml):
    results, marked = _search_with_docs(datetime.now(timezone.utc) - timedelta(hours=2), index, xml)
    assert results == [] and marked == [edgar._seen_key("0000000001-26-000001")]


def test_submissions_failure_logs_as_insider(caplog):
    with patch("ingestion.edgar.requests.get", return_value=_resp(status=404)), \
         caplog.at_level("WARNING", logger="ingestion.edgar"):
        assert insider._search_form4("META", "1") is None
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Insider Form 4 fetch failed for META") for m in msgs)
    assert not any("EDGAR fetch failed" in m for m in msgs)


# ── Pass status, budget, rotation, heartbeat ────────────────────────────────

def test_last_pass_counts_and_no_success_is_a_failed_pass():
    _poll({"A": "1", "B": "2"}, {"A": None, "B": None})
    assert insider.LAST_PASS["ok"] == 0 and insider.LAST_PASS["failed"] == 2
    assert insider.LAST_PASS["poll_failed"] is True
    _poll({"A": "1", "B": "2"}, {"A": None})
    assert insider.LAST_PASS["ok"] == 1 and insider.LAST_PASS["poll_failed"] is False


def test_backoff_skip_is_flagged_and_not_failed(_state):
    _state["edgar_backoff_until_paper"] = _state["edgar_backoff_until_live"] = \
        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    _poll({"A": "1"}, {})
    assert insider.LAST_PASS["backoff_skip"] is True and insider.LAST_PASS["poll_failed"] is False


def test_budget_stops_the_pass_and_the_next_starts_where_it_stopped(_state):
    tickers = {f"T{i}": str(i) for i in range(6)}
    clock = iter([0, 0, 0, 0, 0, 1000] + [2000] * 50)    # out of time before the 5th ticker
    with patch("ingestion.insider.time.monotonic", side_effect=lambda: next(clock)):
        fetched, _ = _poll(tickers, {})
    assert fetched == ["T0", "T1", "T2", "T3"]
    assert "pass ran past" in insider.LAST_PASS["aborted"]
    key = [k for k in _state if k.startswith("insider_next_start_")][0]
    assert _state[key] == "4"
    fetched, _ = _poll(tickers, {})
    assert fetched == ["T4", "T5", "T0", "T1", "T2", "T3"]


def test_poll_insider_heartbeat_follows_last_pass():
    import tasks
    passes = iter([
        {"tickers": 75, "ok": 0, "failed": 5, "poll_failed": True, "aborted": "5 tickers in a row failed"},
        {"backoff_skip": True, "poll_failed": False},
        {"tickers": 75, "ok": 75, "failed": 0, "poll_failed": False},
    ])

    def fake_poll():
        insider.LAST_PASS.clear()
        insider.LAST_PASS.update(next(passes))
        return []

    with patch("ingestion.insider.poll_once", side_effect=fake_poll), \
         patch("tasks._heartbeat_ok") as ok, patch("tasks._heartbeat_fail") as fail:
        tasks.poll_insider.run()
        assert not ok.called
        assert fail.call_args.args[0] == "insider" and "0 of 75 tickers ok" in fail.call_args.args[1]
        fail.reset_mock()
        tasks.poll_insider.run()
        assert not ok.called and not fail.called
        tasks.poll_insider.run()
        assert ok.call_args.args == ("insider", 0) and not fail.called
    insider.LAST_PASS.clear()
