"""One Telegram per feed per ET day after consecutive all-ticker failures in
market hours (audit 2026-11-09 16:53 #10). Alert only."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ingestion import news

MKT = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)      # 10:00 ET Tuesday


def _redis(count, alert_set=True):
    r = MagicMock()
    r.incr.return_value = count
    r.set.return_value = alert_set
    return r


def _run(r, down=True, market=True):
    with patch("stop_lock._redis", return_value=r), \
         patch("market_data.is_market_hours", return_value=market), \
         patch("alerts._send") as send:
        sent = news._track_feed_outage("yfinance", down, "NVDA: YFRateLimitError", now_utc=MKT)
    return sent, send


def test_third_failed_poll_in_market_hours_alerts_once():
    r = _redis(3)
    sent, send = _run(r)
    assert sent is True
    assert "NEWS FEED DOWN" in send.call_args[0][0] and "yfinance" in send.call_args[0][0]
    assert r.set.call_args[0][0] == "helios:news_feed_down_alert:yfinance:2026-11-10"
    assert r.set.call_args.kwargs == {"nx": True, "ex": 86400}


def test_below_threshold_or_after_hours_does_not_alert():
    assert _run(_redis(2))[0] is False
    sent, send = _run(_redis(5), market=False)
    assert sent is False and not send.called


def test_already_alerted_today_does_not_repeat():
    sent, send = _run(_redis(7, alert_set=False))
    assert sent is False and not send.called


def test_healthy_poll_resets_the_count():
    r = _redis(0)
    sent, send = _run(r, down=False)
    assert sent is False and not send.called
    r.delete.assert_called_once_with("helios:news_feed_down:yfinance")
    r.incr.assert_not_called()


def test_redis_failure_never_raises():
    with patch("stop_lock._redis", side_effect=RuntimeError("down")):
        assert news._track_feed_outage("yfinance", True, None, now_utc=MKT) is False


def test_feed_stats_marks_all_failed_feed_down():
    news._FEED_STATS.clear()
    news._note_feed("yfinance", "NVDA", error=RuntimeError("Yahoo down"))
    news._note_feed("google", "NVDA", raw=3, returned=1)
    with patch("ingestion.news._track_feed_outage") as track:
        news._log_feed_stats()
    calls = {c.args[0]: c.args[1] for c in track.call_args_list}
    assert calls == {"google": False, "yfinance": True}
    news._FEED_STATS.clear()
