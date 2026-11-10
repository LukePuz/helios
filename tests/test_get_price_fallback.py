"""crew._get_price's yfinance fallback: during regular hours the regular
price comes first, since postMarketPrice is the previous evening's print;
outside them the extended-hours price comes first (Luke D6)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import crew

INFO = {"postMarketPrice": 101.0, "preMarketPrice": 102.0, "regularMarketPrice": 100.0}


def _price(rth: bool, info=INFO, hour: int = 18, on: bool = True):
    from datetime import datetime as real_dt

    class _DT(real_dt):
        @classmethod
        def now(cls, tz=None):
            return real_dt(2026, 11, 4, hour, 0, tzinfo=tz)

    fake = SimpleNamespace(info=dict(info), fast_info=SimpleNamespace(last_price=99.0))
    with patch("market_data.get_session_quote", return_value=None), \
         patch("ingestion.polygon_stream.get_snapshot", return_value={}), \
         patch("market_data.get_latest_quotes", return_value={}), \
         patch("market_data.is_market_hours", return_value=rth), \
         patch("yfinance.Ticker", return_value=fake), \
         patch("datetime.datetime", _DT), \
         patch("config.PRICE_FALLBACK_SESSION_ORDER", on):
        return crew._get_price("AAA", None)


def test_regular_hours_prefers_the_regular_price():
    assert _price(True) == 100.0


def test_outside_regular_hours_prefers_extended_hours():
    assert _price(False) == 101.0


def test_falls_back_in_order_when_fields_are_missing():
    assert _price(True, {"postMarketPrice": 101.0}) == 101.0
    assert _price(False, {"regularMarketPrice": 100.0}) == 100.0
    assert _price(True, {}) == 99.0


def test_before_the_open_prefers_todays_pre_market_print():
    assert _price(False, hour=7) == 102.0
    assert _price(False, {"postMarketPrice": 101.0}, hour=7) == 101.0


def test_switch_defaults_off():
    """Default off (analyst input, packet G50). Read from config.py's source,
    so an env that turns the switch on doesn't fail this test."""
    import config
    src = Path(config.__file__).read_text()
    assert ('PRICE_FALLBACK_SESSION_ORDER = os.getenv("PRICE_FALLBACK_SESSION_ORDER", '
            '"false").lower() == "true"') in src


def test_switch_off_keeps_the_old_order_at_all_hours():
    """Off: after-hours, pre-market, regular, as before 81e11c5."""
    assert _price(True, on=False) == 101.0
    assert _price(False, hour=7, on=False) == 101.0
    assert _price(True, {"regularMarketPrice": 100.0}, on=False) == 100.0


def test_switch_is_named_in_the_treatment_config():
    from agents.analyst import _treatment_config
    for value in (False, True):
        with patch("config.PRICE_FALLBACK_SESSION_ORDER", value):
            assert _treatment_config()["PRICE_FALLBACK_SESSION_ORDER"] is value
