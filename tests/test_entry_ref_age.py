"""The price references an OPEN/ADD BUY is sent on (record only): the sizing
price's bar time (risk_json.arrival), the pre-submit quote's bar time, their
ages at submit and the bid/ask snapshot place_market_order already took, in
one activity row (stage entry_ref_age) written after everything else. The
order itself is unchanged and nothing is read before the submit."""
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import db
import execution
import market_data
from agents import crew

SUB = datetime(2026, 11, 20, 15, 0, 30, tzinfo=timezone.utc)
SIZING_BAR = datetime(2026, 11, 20, 14, 58, tzinfo=timezone.utc)
SUBMIT_BAR = datetime(2026, 11, 20, 14, 59, tzinfo=timezone.utc)
QUOTE = {"md_type": 3, "md_type_label": "delayed", "bid": 99.9, "ask": 100.1, "last": 100.0,
         "quote_time": "2026-11-20T15:00:29+00:00", "quote_age_s": 1.0}


def _arrival(**over):
    a = {"sizing_price": 100.0, "sizing_at": (SIZING_BAR + timedelta(seconds=40)).isoformat(),
         "sizing_price_source": "latest_quote", "sizing_bar_time": SIZING_BAR.isoformat(),
         "sizing_price_age_s": 40.0, "sizing_price_is_stale": False}
    a.update(over)
    return a


def _pq(**over):
    q = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True,
         "price_source": "1min", "bar_time": SUBMIT_BAR.replace(tzinfo=None),
         "bar_time_utc": SUBMIT_BAR}
    q.update(over)
    return q


# ── pure pieces ──────────────────────────────────────────────────────────────

def test_ref_at_submit_ages():
    r = execution._entry_ref_at_submit({"arrival": _arrival()}, _pq(), SUB)
    assert r["sizing_bar_time"] == SIZING_BAR.isoformat()
    assert r["submit_bar_time"] == SUBMIT_BAR.isoformat()
    assert r["ref_age_s"] == 150.0 and r["submit_ref_age_s"] == 90.0
    assert r["sizing_to_submit_s"] == 110.0
    assert r["sizing_price_source"] == "latest_quote" and r["submit_price_source"] == "1min"
    assert r["submit_quote_is_stale"] is False and "error" not in r


def test_ref_at_submit_unknowns_are_none():
    r = execution._entry_ref_at_submit({}, None, SUB)
    assert r["submitted_at"] == SUB.isoformat()
    assert all(r[k] is None for k in ("sizing_bar_time", "submit_bar_time", "ref_age_s",
                                      "submit_ref_age_s", "sizing_to_submit_s"))
    # A naive bar time has no known zone: unknown, not guessed.
    r = execution._entry_ref_at_submit({"arrival": _arrival(sizing_bar_time="2026-11-20T09:58:00")},
                                       _pq(bar_time_utc=None), SUB)
    assert r["sizing_bar_time"] is None and r["ref_age_s"] is None and r["submit_ref_age_s"] is None
    assert execution._entry_ref_at_submit({"arrival": "junk"}, "junk", SUB)["ref_age_s"] is None


def test_ref_at_submit_does_no_io():
    """Only dict reads, datetime parsing and rounding: nothing that could
    reach the broker, the DB or the network before placeOrder."""
    import ast
    import textwrap
    allowed = {"get", "isoformat", "fromisoformat", "astimezone", "replace", "total_seconds",
               "round", "isinstance", "bool", "type", "_iso_utc"}
    for fn in (execution._entry_ref_at_submit, execution._iso_utc):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        called = {c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id
                  for c in ast.walk(tree) if isinstance(c, ast.Call)}
        assert called <= allowed, called - allowed
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & {"db", "ibkr", "market_data", "alerts", "time", "tasks"}


def test_priced_quote_carries_the_utc_bar_time():
    bars = [{"time": "2026-11-20 09:59:00-05:00", "close": 100.0}]
    with patch("market_data.get_historical_bars", return_value=bars), \
         patch("market_data._is_market_hours", return_value=True):
        q = market_data.get_priced_quote("ZZZ")
    assert q["bar_time_utc"] == SUBMIT_BAR and q["bar_time"] == datetime(2026, 11, 20, 9, 59)
    with patch("market_data.get_historical_bars", return_value=[{"time": "20261120", "close": 1.0}]), \
         patch("market_data._is_market_hours", return_value=True):
        assert market_data.get_priced_quote("ZZZ")["bar_time_utc"] is None
    with patch("market_data.get_priced_quote", return_value=_pq()):
        assert market_data.get_latest_quotes(["ZZZ"])["ZZZ"]["bar_time_utc"] == SUBMIT_BAR


def test_get_price_info_gives_the_utc_bar_time():
    info = {}
    with patch("market_data.get_session_quote", return_value=None), \
         patch("ingestion.polygon_stream.get_snapshot", return_value={}), \
         patch("market_data.get_latest_quotes",
               return_value={"ZZZ": {"mid": 100.0, "is_stale": False,
                                     "bar_time": SUBMIT_BAR.replace(tzinfo=None),
                                     "bar_time_utc": SUBMIT_BAR}}):
        assert crew._get_price("ZZZ", None, info=info) == 100.0
    assert info["source"] == "latest_quote" and info["as_of_utc"] == SUBMIT_BAR
    info = {}
    et = SUBMIT_BAR.astimezone(market_data._ET)
    with patch("market_data.get_session_quote", return_value={"price": 101.0, "bar_time": et}):
        assert crew._get_price("ZZZ", None, info=info) == 101.0
    assert info["source"] == "session_quote" and info["as_of_utc"] == SUBMIT_BAR
    assert info["as_of"] is et   # the brief's as_of is unchanged


def test_arrival_record_sizing_provenance():
    r = crew._arrival_record({}, 100.0, {"source": "latest_quote", "is_stale": False,
                                         "as_of_utc": datetime.now(timezone.utc) - timedelta(seconds=30)})
    assert r["sizing_price_source"] == "latest_quote" and r["sizing_price_is_stale"] is False
    assert 29.0 <= r["sizing_price_age_s"] <= 32.0
    assert datetime.fromisoformat(r["sizing_bar_time"]).tzinfo is not None
    r = crew._arrival_record({}, 100.0)
    assert r["sizing_price_source"] is None and r["sizing_bar_time"] is None
    assert r["sizing_price_age_s"] is None and "sizing_info_error" not in r
    r = crew._arrival_record({}, 100.0, {"source": "yfinance_info", "as_of": None})
    assert r["sizing_price_source"] == "yfinance_info" and r["sizing_bar_time"] is None


def test_routing_passes_the_sizing_price_info():
    src = inspect.getsource(crew)
    assert "price = _get_price(ticker, trading_client, info=sizing_price_info)" in src
    assert src.count("price_info=sizing_price_info)") == 2


# ── execute_approved ─────────────────────────────────────────────────────────

def _execute(action="OPEN", place=None, arrival=True, pq=None, extra_patches=()):
    approval = {"id": 171, "ticker": "NVDA", "action": action, "conviction": 7,
                "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
                "risk_json": {"shares_initial": 1, "price_at_sizing": 100.0,
                              "stop_loss_price": 90.0,
                              **({"arrival": _arrival()} if arrival else {})}}
    order = []
    db_fns = {n: MagicMock(return_value=None) for n, f in vars(db).items()
              if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == "db"}
    db_fns["log_activity_queued"] = MagicMock(side_effect=lambda *a, **k: order.append(a[0]))
    place = place or {"return_value": {"filled_qty": 1, "fill_price": 100.0, "order_id": 1,
                                       "perm_id": 77, "status": "Filled", "quote": dict(QUOTE)}}
    with patch.multiple("db", **db_fns), \
         patch("ibkr.get_ib"), patch("ibkr.get_client") as client, \
         patch("ibkr.place_market_order", **place) as placed, \
         patch("ibkr.get_latest_price", return_value=100.0), \
         patch("market_data.get_priced_quote", return_value=pq if pq is not None else _pq()), \
         patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}), \
         patch("market_data.is_market_hours", return_value=True), \
         patch("execution._whole_position_stop", return_value=(95.0, None)), \
         patch("execution._now_utc", return_value=SUB), \
         patch("tasks.manage_stop_place",
               side_effect=lambda *a, **k: order.append("stop") or {"order_id": 9}), \
         patch("alerts._send"), \
         patch("alerts.send_trade_alert", side_effect=lambda **k: order.append("alert")):
        client.return_value.get_open_position.side_effect = Exception("No open position")
        client.return_value.get_all_positions.return_value = []
        from contextlib import ExitStack
        with ExitStack() as st:
            for p in extra_patches:
                st.enter_context(p)
            ok = execution.execute_approved(approval)
    rows = [c.args for c in db_fns["log_activity_queued"].call_args_list
            if c.args[0] == "entry_ref_age"]
    return ok, rows, order, placed


def test_filled_open_writes_one_row_after_the_stop_and_alert():
    ok, rows, order, placed = _execute()
    assert ok is True and len(rows) == 1
    assert order.index("entry_ref_age") > order.index("stop")
    assert order.index("entry_ref_age") > order.index("alert")
    x = rows[0][4]
    assert x["ref_age_s"] == 150.0 and x["submit_ref_age_s"] == 90.0
    assert (x["bid"], x["ask"], x["md_type_label"], x["quote_age_s"]) == (99.9, 100.1, "delayed", 1.0)
    assert x["approval_id"] == 171 and x["action"] == "OPEN" and x["order_type"] == "MARKET"
    assert x["broker_order_id"] == 1 and x["perm_id"] == 77 and x["filled"] is True
    assert x["outcome_reason"] is None and x["submit_error"] is None
    assert x["submit_attempted"] is True
    assert placed.call_args.args == ("NVDA", "BUY", 1)


def test_order_is_the_same_with_or_without_the_record():
    _, _, _, with_arr = _execute()
    _, _, _, without = _execute(arrival=False)
    assert with_arr.call_args.args == without.call_args.args
    with patch("execution._record_entry_ref_age", side_effect=RuntimeError("boom")):
        ok, _, _, raised = _execute()    # a raise in the record keeps the return value
    assert ok is True and raised.call_args.args == with_arr.call_args.args


def test_writer_never_raises_and_old_approvals_record_none():
    _, rows, _, _ = _execute(arrival=False)
    x = rows[0][4]
    assert x["sizing_bar_time"] is None and x["ref_age_s"] is None
    assert x["submit_ref_age_s"] == 90.0
    assert execution._record_entry_ref_age("NVDA", None, "OPEN", {"submitted_at": None}) is False
    assert execution._record_entry_ref_age("NVDA", {"id": 1}, "OPEN", None) is False


def test_unconfirmed_and_raised_submits_are_recorded():
    ok, rows, _, _ = _execute(place={"return_value": {"filled_qty": 0, "fill_price": None,
                                                      "order_id": 4, "status": "Submitted",
                                                      "quote": dict(QUOTE)}})
    assert ok is False and len(rows) == 1
    x = rows[0][4]
    assert x["filled"] is False and x["outcome_reason"].startswith("unconfirmed")
    ok, rows, _, _ = _execute(place={"side_effect": TimeoutError("lost")})
    assert ok is False and len(rows) == 1
    x = rows[0][4]
    assert x["submit_error"] == "TimeoutError" and x["bid"] is None and x["filled"] is None
    assert x["outcome_reason"] == "failed: exception during execution"


def test_add_is_recorded_and_aborted_entries_are_not():
    _, rows, _, _ = _execute(action="ADD")
    assert len(rows) == 1 and rows[0][4]["action"] == "ADD"
    # Stale quote: the entry is deferred before any submit, so no row.
    with patch("execution._stale_buy_outcome", return_value=False), patch("time.sleep"):
        _, rows, _, placed = _execute(pq=_pq(is_stale=True))
    assert rows == [] and not placed.called


def test_bracket_fill_is_unknown_not_false():
    ref = {"order_type": "BRACKET", "order_result": {"order_id": 5, "filled_qty": None}}
    with patch("db.log_activity_queued") as act:
        assert execution._record_entry_ref_age("NVDA", {"id": 1}, "OPEN", ref)
    x = act.call_args.args[4]
    assert x["filled"] is None and x["order_type"] == "BRACKET" and x["broker_order_id"] == 5


def test_the_row_is_queued_never_written_inside_the_lock():
    src = inspect.getsource(execution._record_entry_ref_age)
    assert "db.log_activity_queued(" in src and "db.log_activity(" not in src


def test_a_raising_record_helper_still_sends_the_order():
    ok, rows, _, placed = _execute(extra_patches=[
        patch("execution._entry_ref_at_submit", side_effect=RuntimeError("boom"))])
    assert ok is True and placed.call_args.args == ("NVDA", "BUY", 1) and rows == []
    ok, _, _, placed2 = _execute(extra_patches=[
        patch("execution._iso_utc", side_effect=RuntimeError("boom"))])
    assert ok is True and placed2.call_args.args == ("NVDA", "BUY", 1)


def test_a_raise_before_the_place_call_is_marked_not_attempted():
    ok, rows, _, placed = _execute(extra_patches=[
        patch("execution._early_order_row", side_effect=RuntimeError("row"))])
    assert ok is False and not placed.called and len(rows) == 1
    x = rows[0][4]
    assert x["submit_attempted"] is False and x["order_type"] is None and x["filled"] is None
