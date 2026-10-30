"""One seen-ids query per EDGAR pass instead of one per feed entry (a pass
took 11-18 minutes at ~230 ms per DB round trip)."""
from unittest.mock import MagicMock, patch

from ingestion import edgar

_ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>seen-1</id><title>old</title><updated>2026-10-30T09:00:00Z</updated></entry>
<entry><id>seen-2</id><title>old</title><updated>2026-10-30T09:00:00Z</updated></entry>
</feed>"""


def test_poll_checks_seen_once_per_pass():
    resp = MagicMock(text=_ATOM)
    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1", "BBB": "2"}), \
         patch("ingestion.edgar.requests.get", return_value=resp), \
         patch("ingestion.edgar.time.sleep"), \
         patch("db.seen_ids", return_value={"seen-1", "seen-2"}) as seen_ids, \
         patch("db.is_seen") as is_seen:
        assert edgar.poll_once() == []
    assert seen_ids.call_count == 1 and not is_seen.called
    assert edgar._SEEN_THIS_PASS is None          # cleared after the pass


def test_poll_falls_back_to_per_item_checks_when_the_batch_fails():
    resp = MagicMock(text=_ATOM)
    with patch.object(edgar, "TICKER_TO_CIK", {"AAA": "1"}), \
         patch("ingestion.edgar.requests.get", return_value=resp), \
         patch("ingestion.edgar.time.sleep"), \
         patch("db.seen_ids", return_value=None), \
         patch("db.is_seen", return_value=True) as is_seen:
        edgar.poll_once()
    assert is_seen.called
