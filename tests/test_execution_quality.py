"""scripts/execution_quality.py: implementation shortfall in basis points
(positive = worse for us) on fixture rows, missing fields counted and never
imputed, and the script only reads."""
import importlib.util
import inspect
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "execution_quality.py")
_spec = importlib.util.spec_from_file_location("execution_quality", _PATH)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

NY = eq.NY


def _et(day, h, m=0, s=0):
    return datetime(2026, 11, day, h, m, s, tzinfo=NY)


def _order(side="BUY", fill=101.0, dec=100.0, sub=100.0, brief=None, submitted=None, lag_s=1.5,
           approval_gap_min=0.05, intent="OPEN", order_type="MARKET", source="receipt"):
    submitted = submitted or _et(18, 11, 0)
    return {"side": side, "order_type": order_type, "intent": intent, "decision_price": dec,
            "submit_price": sub, "fill_price": fill, "brief_price": brief,
            "submitted_at": submitted,
            "filled_at": submitted + timedelta(seconds=lag_s) if lag_s is not None else None,
            "filled_at_source": source,
            "approval_at": submitted - timedelta(minutes=approval_gap_min)
            if approval_gap_min is not None else None}


def test_basis_points_are_signed_worse_positive():
    assert eq.bp_buy(101, 100) == pytest.approx(100.0)          # paid more
    assert eq.bp_buy(99, 100) == pytest.approx(-100.0)
    assert eq.bp_sell(99, 100) == pytest.approx(100.0)          # sold for less
    assert eq.bp_sell(101, 100) == pytest.approx(-100.0)
    for bad in (None, 0, -1, "x", float("nan")):
        assert eq.bp_buy(101, bad) is None and eq.bp_sell(bad, 100) is None


def test_open_window_in_et():
    assert eq.open_window(_et(18, 9, 30)) == "open_15"
    assert eq.open_window(_et(18, 9, 44, 59)) == "open_15"
    assert eq.open_window(_et(18, 9, 45)) == "later_rth"
    assert eq.open_window(_et(18, 15, 59)) == "later_rth"
    assert eq.open_window(_et(18, 16, 0)) == "outside_rth"
    assert eq.open_window(_et(18, 9, 29)) == "outside_rth"
    assert eq.open_window(_et(21, 10)) == "outside_rth"                     # Saturday
    assert eq.open_window(_et(26, 10)) == "outside_rth"                     # Thanksgiving
    assert eq.open_window(datetime(2026, 11, 18, 14, 35)) == "open_15"     # naive = UTC
    assert eq.open_window(None) is None


def test_lag_bucket():
    assert [eq.lag_bucket(x) for x in (0.5, 5, 5.1, 30, 31, None)] == \
        ["<=5s", "<=5s", "5-30s", "5-30s", ">30s", None]


def test_entry_row_measures_and_path():
    r = eq.entry_row(_order(fill=100.5, dec=100.0, sub=100.0, brief=101.0))
    assert r["bp_decision"] == pytest.approx(50.0) and r["bp_submit"] == pytest.approx(50.0)
    assert r["bp_brief"] == pytest.approx((100.5 / 101 - 1) * 1e4)
    assert r["decision_is_submit"] is True and r["path"] == "immediate"
    assert r["window"] == "later_rth" and r["lag"] == "<=5s" and r["lag_s"] == pytest.approx(1.5)
    assert eq.entry_row(_order(approval_gap_min=11))["path"] == "deferred"
    assert eq.entry_row(_order(approval_gap_min=None))["path"] is None
    assert eq.entry_row(_order(sub=99.0))["decision_is_submit"] is False


def test_a_lag_over_an_hour_is_listed_not_summarised():
    r = eq.entry_row(_order(lag_s=5 * 3600 + 1.2, source=None))
    assert r["lag_suspect"] is True and r["lag_s"] is None and r["lag"] == "suspect"
    assert eq.entry_row(_order(lag_s=None))["lag"] is None


def test_stop_row_sign_and_kind():
    r = eq.stop_row({"slippage_pct": -1.903, "slippage_vs_broker_trigger_pct": -1.903,
                     "at_open": True, "outside_rth": False})
    assert r["bp_stop"] == pytest.approx(190.3) and r["kind"] == "gap_through"
    r = eq.stop_row({"slippage_pct": 0.001, "at_open": False, "outside_rth": False})
    assert r["bp_stop"] == pytest.approx(-0.1) and r["bp_trigger"] is None and r["kind"] == "normal"
    assert eq.stop_row({"slippage_pct": -0.5, "outside_rth": True})["kind"] == "gap_through"
    assert eq.stop_row({"slippage_pct": "x"})["bp_stop"] is None
    assert eq.stop_row(None)["kind"] is None


def test_broker_sells_unmatched():
    t = _et(19, 16)
    rows = [{"ticker": "NVDA", "ts": t}]
    sells = [{"ticker": "NVDA", "ts": t + timedelta(minutes=30)},
             {"ticker": "NVDA", "ts": t + timedelta(hours=3)},       # too far
             {"ticker": "MSFT", "ts": t}]                             # other ticker
    assert eq.broker_sells_unmatched(sells, rows) == 2


def _fixture():
    orders = [_order(fill=100.5, brief=100.0),                                 # +50 bp
              _order(fill=99.0, submitted=_et(18, 9, 31)),                     # -100, open_15
              _order(fill=100.2, approval_gap_min=600, lag_s=12),              # deferred
              _order(fill=100.1, lag_s=5 * 3600 + 1, source=None),             # suspect lag
              _order(fill=100.0, dec=None, sub=None),                          # missing prices
              _order(side="SELL", intent="TRIM", fill=99.5, dec=100.0, sub=100.0)]
    stops = [{"slippage_pct": -1.903, "slippage_vs_broker_trigger_pct": -1.903, "at_open": True,
              "outside_rth": False},
             {"slippage_pct": 0.001, "slippage_vs_broker_trigger_pct": 0.001, "at_open": False,
              "outside_rth": False},
             {"slippage_pct": None, "at_open": None, "outside_rth": None}]
    t = _et(19, 16)
    return orders, stops, [{"ticker": "NVDA", "ts": t}, {"ticker": "MSFT", "ts": t}], \
        [{"ticker": "NVDA", "ts": t}]


def test_build_readout():
    orders, stops, sells, meta = _fixture()
    text = eq.build(orders, stops, sells, meta, "2026-10-28", "live", "abc1234")
    assert text.startswith("# Execution quality (live) since 2026-10-28")
    assert "Filled BUYs: 5." in text and "Filled SELLs: 1." in text
    assert "| vs decision price | n 4, median +15.0, range -100.0 to +50.0 |" in text
    assert "| vs price when the brief was built | n 1, median +50.0, range +50.0 to +50.0 |" in text
    assert "| lag over 3600 s (listed, not summarised) | 1 |" in text
    assert "| decision price equal to the quote at submit | 4 of 5 |" in text
    assert "| open_15 | n 1, median -100.0, range -100.0 to -100.0 |" in text
    assert "| deferred | n 1, median +20.0, range +20.0 to +20.0 |" in text
    assert "| suspect | n 1," in text
    assert "| TRIM | n 1, median +50.0, range +50.0 to +50.0 | n 1, median +50.0" in text
    assert "Broker sells with no stop_slippage row within 2 h: 1 of 2." in text
    assert "| gap_through | n 1, median +190.3" in text and "| normal | n 1, median -0.1" in text
    assert "| missing | n 0 | n 0 |" in text
    assert ("orders: brief_price 5, decision_price 1, filled_at_source 1, submit_price 1 (of 6)."
            in text)
    assert ("stop_slippage: at_open 1, outside_rth 1, slippage_pct 1, "
            "slippage_vs_broker_trigger_pct 1 (of 3).") in text
    assert "$" not in text


def test_build_with_no_rows():
    text = eq.build([], [], [], [], "2026-10-28", "paper", "x")
    assert "Filled BUYs: 0." in text and "| vs decision price | n 0 |" in text
    assert "orders: none (of 0)." in text


def test_main_writes_the_file(tmp_path, capsys):
    out = tmp_path / "eq.md"
    rows = ([_order()], [{"ts": _et(19, 16), "ticker": "NVDA", "extra": {"slippage_pct": -0.2}}], [],
            [])
    with patch.object(eq, "_fetch", return_value=rows) as fetch:
        assert eq.main(["--since", "2026-11-01", "--env", "paper", "--out", str(out)]) == 0
    assert fetch.call_args.args[1] == "paper"
    assert out.read_text().startswith("# Execution quality (paper) since 2026-11-01")
    assert "Rows: 1." in capsys.readouterr().out


def test_reads_only():
    src = inspect.getsource(eq)
    for word in ("INSERT", "UPDATE ", "DELETE FROM", "commit(", "set_state(", "log_activity(",
                 "send_task(", "apply_async(", ".delay(", "alerts", "telegram", "import ibkr",
                 "requests."):
        assert word not in src, word
    fetch = inspect.getsource(eq._fetch)
    assert "SET TRANSACTION READ ONLY" in fetch and "statement_timeout" in fetch
    assert "conn.rollback()" in fetch


def test_partial_rows_are_reported_apart_and_change_no_stop_count():
    orders, stops, sells, meta = _fixture()
    before = eq.build(orders, stops, sells, meta, "2026-10-28", "live", "abc1234")
    partial = [{"ts": _et(19, 15), "ticker": "MSFT",
                "extra": {"slippage_pct": -3.0, "slippage_vs_broker_trigger_pct": -1.0,
                          "at_open": True, "outside_rth": False}}]
    text = eq.build(orders, stops, sells, meta, "2026-10-28", "live", "abc1234",
                    partial_rows=partial)
    # Everything before the new section is unchanged: no row pooled, no count moved.
    head = before.split("## Partial stop fills")[0]
    assert text.startswith(head)
    assert "Broker sells with no stop_slippage row within 2 h: 1 of 2." in text
    sec = text.split("## Partial stop fills (stop_slippage_partial rows; order's day VWAP at booking)"
                     )[1].split("## Missing")[0]
    assert "Rows: 1." in sec and "a partial row within 2 h: 1." in sec
    assert "| all | n 1, median +300.0, range +300.0 to +300.0 | n 1, median +100.0" in sec
    empty = eq.build(orders, stops, sells, meta, "2026-10-28", "live", "abc1234")
    assert "Rows: 0." in empty.split("## Partial stop fills")[1]
    assert "stage = 'stop_slippage_partial'" in inspect.getsource(eq._fetch)
