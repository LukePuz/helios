"""One seen-ids query per EDGAR pass instead of one per feed entry (a pass
took 11-18 minutes at ~230 ms per DB round trip)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from ingestion import edgar


def _submissions(*rows):
    """A submissions JSON whose filings.recent holds rows of (form, accession, accepted)."""
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "accessionNumber": [r[1] for r in rows],
        "acceptanceDateTime": [r[2] for r in rows],
        "filingDate": [r[2][:10] for r in rows],
        "primaryDocument": ["doc.htm" for _ in rows],
    }}}


def _resp(data):
    return MagicMock(status_code=200, json=MagicMock(return_value=data))


def _recent():
    # Dated from now: entries past the age limit are skipped before the seen check.
    return (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_poll_checks_seen_once_per_pass():
    data = _submissions(("8-K", "0000000001-26-000001", _recent()),
                        ("4", "0000000001-26-000002", _recent()))
    seen = {edgar._seen_key("0000000001-26-000001"), edgar._seen_key("0000000001-26-000002")}
    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1", "BBB": "2"}), \
         patch("ingestion.edgar.requests.get", return_value=_resp(data)), \
         patch("ingestion.edgar.time.sleep"), \
         patch("db.seen_ids", return_value=seen) as seen_ids, \
         patch("db.is_seen") as is_seen:
        assert edgar.poll_once() == []
    assert seen_ids.call_count == 1 and not is_seen.called
    assert edgar._SEEN_THIS_PASS is None          # cleared after the pass


def test_poll_falls_back_to_per_item_checks_when_the_batch_fails():
    data = _submissions(("8-K", "0000000001-26-000001", _recent()))
    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1"}), \
         patch("ingestion.edgar.requests.get", return_value=_resp(data)), \
         patch("ingestion.edgar.time.sleep"), \
         patch("db.seen_ids", return_value=None), \
         patch("db.is_seen", return_value=True) as is_seen:
        edgar.poll_once()
    assert is_seen.called


def test_filing_in_two_feeds_is_dispatched_once_per_pass():
    """A filing dispatched (and marked seen by the dispatch) earlier in the
    pass must not dispatch again from another company's submissions."""
    data = _submissions(("4", "0000000009-26-000001", _recent()))
    sent = []

    def dispatch(filing):
        sent.append(filing.pop("_seen_key", None))

    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1", "BBB": "2"}), \
         patch("ingestion.edgar.requests.get", return_value=_resp(data)), \
         patch("ingestion.edgar.time.sleep"), patch("ingestion.edgar._fetch_filing_text", return_value=""), \
         patch("db.seen_ids", return_value=set()):
        edgar.poll_once(on_filing=dispatch)
    assert sent == [edgar._seen_key("0000000009-26-000001")]
