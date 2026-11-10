"""_stock's qualify is bounded by RequestTimeout, keeps the session on a
timeout, and is cached per ticker (audits 2026-11-08 15:08 #8, 17:17 #6)."""
from unittest.mock import MagicMock

import pytest

import ibkr


def _ib(conid=0):
    ib = MagicMock()
    ib.RequestTimeout = 0
    seen = []

    def qualify(c):
        seen.append(ib.RequestTimeout)
        c.conId = conid
    ib.qualifyContracts.side_effect = qualify
    return ib, seen


def test_qualify_runs_under_the_request_timeout():
    ib, seen = _ib()
    c = ibkr._stock(ib, "NVDA")
    assert c.symbol == "NVDA"
    assert seen == [ibkr._REQ_TIMEOUT_S] and ib.RequestTimeout == 0


def test_qualify_timeout_raises_and_keeps_the_session(monkeypatch):
    ib = MagicMock()
    ib.RequestTimeout = 0
    ib.qualifyContracts.side_effect = TimeoutError()
    sentinel = object()
    monkeypatch.setattr(ibkr, "_ib", sentinel)
    with pytest.raises(TimeoutError):
        ibkr._stock(ib, "NVDA")
    ib.disconnect.assert_not_called()
    assert ibkr._ib is sentinel
    assert "NVDA" not in ibkr._QUALIFIED


def test_qualified_contract_is_cached_as_a_copy():
    ib, seen = _ib(conid=4815747)
    a = ibkr._stock(ib, "NVDA")
    b = ibkr._stock(ib, "NVDA")
    assert len(seen) == 1                       # second call made no request
    assert a.conId == b.conId == 4815747 and a is not b
    b.exchange = "XX"
    assert ibkr._stock(ib, "NVDA").exchange == "SMART"


def test_unqualified_result_is_not_cached():
    ib, seen = _ib(conid=0)
    ibkr._stock(ib, "ZZZZ")
    ibkr._stock(ib, "ZZZZ")
    assert len(seen) == 2
