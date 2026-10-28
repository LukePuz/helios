"""Professor run 5 fixes: M31 detection, M34 traceback redaction, C3 config stamp."""
import logging

import log_redaction
from agents import market_regime


def test_missing_input_patterns():
    pat = market_regime._MISSING_INPUT
    for note in ("VIX data unavailable", "Insufficient bar data for trend", "No volume data",
                 "No ETF flow data", "history fetch returned no data"):
        assert pat.search(note), note
    assert not pat.search("SPY above 200-day MA, golden cross")


def test_traceback_is_redacted(caplog):
    log_redaction.install()
    with caplog.at_level(logging.ERROR):
        try:
            raise RuntimeError("GET /bot123456789:AAexampleFAKEtoken0000000000000000/getUpdates")
        except RuntimeError:
            logging.getLogger("t").exception("poll failed")
    assert "AAexample" not in caplog.text
    formatted = logging.Formatter().format(caplog.records[-1])
    assert "AAexample" not in formatted and "bot[REDACTED]" in formatted


def test_treatment_config_is_stamped():
    from agents import analyst
    cfg = analyst._treatment_config()
    assert cfg["EARNINGS_ENTRY_BLACKOUT_DAYS"] == 0 and "ANALYST_MODEL" in cfg


def test_unparsed_watchlist_rows_are_reported(caplog):
    import knowledge_loader
    knowledge_loader.load_watchlist.cache_clear()
    with caplog.at_level(logging.WARNING):
        knowledge_loader.load_watchlist()
    assert "INTC" in caplog.text and "NLST" in caplog.text
