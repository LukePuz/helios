"""book_gap_risk (record only): each held name's last 250 overnight gaps
against today's stop distance and the book's replayed overnight gap loss,
in percent, from DB reads after the close. No threshold or trade change."""
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import tasks
from config import TRADING_ENV

PV = 200_000.0


def _bars(gaps, start=date(2025, 1, 1), close0=100.0):
    """Daily bars whose overnight gaps are exactly `gaps` (open_t/close_t-1 - 1);
    each bar closes at its open."""
    out = [{"date": start.isoformat(), "open": close0, "close": close0}]
    c = close0
    for i, g in enumerate(gaps, 1):
        o = c * (1 + g)
        out.append({"date": (start + timedelta(days=i)).isoformat(), "open": o, "close": o})
        c = o
    return out


def _pos(weight, px=100.0):
    mv = weight * PV
    return {"qty": mv / px, "market_value": mv, "avg_entry_price": px}


def test_percentile_is_linear_between_order_statistics():
    vals = sorted([0.0, 1.0, 2.0, 3.0, 4.0])
    assert tasks._pctile(vals, 0.0) == 0.0 and tasks._pctile(vals, 1.0) == 4.0
    assert tasks._pctile(vals, 0.5) == 2.0 and tasks._pctile(vals, 0.05) == pytest.approx(0.2)
    assert tasks._pctile([], 0.5) is None


def test_name_fields_and_window():
    gaps = [0.0] * 260 + [-0.12, 0.01]          # 262 gaps: only the last 250 count
    bars = {"ANET": _bars(gaps)}
    close = bars["ANET"][-1]["close"]
    out = tasks._book_gap_risk({"ANET": _pos(0.10, px=close)}, PV, {"ANET": close * 0.92}, bars)
    assert out["read"] == "ok" and out["n_positions"] == 1
    p = out["positions"][0]
    assert p["weight_pct"] == 10.0 and p["stop_below_close_pct"] == pytest.approx(8.0)
    assert p["n_gaps"] == 250 and p["first_gap_date"] == bars["ANET"][13]["date"]
    assert p["last_gap_date"] == bars["ANET"][-1]["date"]
    assert p["gap_worst_pct"] == pytest.approx(-12.0) and p["nights_past_stop"] == 1
    assert p["gap_worst_date"] == bars["ANET"][-2]["date"]
    assert out["invested_pct"] == 10.0
    # One name: the book's worst night is its weight x its worst gap.
    assert out["book_gap_worst_pct_nav"] == pytest.approx(-1.2)
    assert out["n_common_nights"] == 250 and out["n_common_nights_any_past_stop"] == 1
    assert out["loss_at_stops_pct_nav"] == pytest.approx(0.8)
    assert out["loss_at_stops_close_basis_pct_nav"] == pytest.approx(0.8)


def test_book_replay_uses_common_nights_and_weights():
    a = _bars([-0.10, 0.0, 0.02])
    b = _bars([-0.02, -0.05])                     # one fewer night
    out = tasks._book_gap_risk({"A": _pos(0.10), "B": _pos(0.05)}, PV,
                               {"A": 1.0, "B": 1.0}, {"A": a, "B": b})
    assert out["n_common_nights"] == 2
    assert out["common_first_date"] == a[1]["date"] and out["common_last_date"] == a[2]["date"]
    # night 1: 0.10 x -10% + 0.05 x -2% = -1.1% of NAV; night 2: 0 + 0.05 x -5% = -0.25%
    assert out["book_gap_worst_pct_nav"] == pytest.approx(-1.1)
    assert out["book_gap_worst_date"] == a[1]["date"]
    assert out["replay_names"] == ["A", "B"] and out["replay_excluded"] == []


def test_missing_inputs_are_none_with_a_reason():
    out = tasks._book_gap_risk({"A": _pos(0.1), "B": _pos(0.1), "C": _pos(0.1),
                                "S": {"qty": -1, "market_value": 100.0},
                                "U": {"qty": 1, "market_value": None}}, PV,
                               {"B": 90.0, "C": 90.0},
                               {"A": _bars([0.0, -0.01]), "B": RuntimeError("db"), "C": _bars([0.01])})
    by = {p["ticker"]: p for p in out["positions"]}
    assert by["A"]["stop_below_close_pct"] is None and by["A"]["nights_past_stop"] is None
    assert by["A"]["reason"] == "no position_stops record" and out["n_no_stop"] == 1
    assert out["unprotected_weight_pct"] == 10.0
    assert by["B"]["reason"] == "bars unreadable: RuntimeError" and by["B"]["n_gaps"] == 0
    assert by["B"]["stop_below_close_pct"] is None and out["loss_at_stops_close_basis_pct_nav"] is None
    assert by["C"]["reason"].startswith("short window: 1 of 250")
    assert out["replay_excluded"] == ["B"] and out["n_short"] == 1 and out["n_unpriced"] == 1
    assert tasks._book_gap_risk({}, 0, {}, {})["read"] == "no_portfolio_value"
    assert tasks._book_gap_risk(None, "x", None, None)["read"] == "no_portfolio_value"


def test_percent_only_no_values_or_share_counts():
    out = tasks._book_gap_risk({"A": _pos(0.1, px=123.0)}, PV, {"A": 110.0}, {"A": _bars([0.0])})
    text = json.dumps(out)
    for v in (PV, 0.1 * PV, 0.1 * PV / 123.0, 123.0, 110.0):
        assert str(v) not in text and f"{v:g}" not in text.replace('"', " ")
    assert not {"qty", "market_value", "total_value", "cash", "stop_price"} & set(
        k for p in out["positions"] for k in p) - {"stop_below_close_pct"}


def test_agrees_with_book_heat_on_the_same_snapshot():
    """book_heat's sum_open_risk_pct (a fraction) x 100 is loss_at_stops_pct_nav:
    the same snapshot price and stops. They differ only by units, timing (buy
    vs after the close) and the close-basis variant."""
    from agents import crew
    positions = {"ANET": _pos(0.10, px=150.0), "NVDA": _pos(0.076, px=180.0),
                 "IONQ": _pos(0.013, px=40.0)}
    stops = {"ANET": 137.7, "NVDA": 162.7, "IONQ": 45.0}        # IONQ's stop above price
    inp = {"positions": positions, "pv": PV, "age_min": 1.0,
           "stops": [{"ticker": t, "stop_price": s} for t, s in stops.items()]}
    with patch("agents.crew._book_heat_inputs", return_value=inp):
        heat = crew._book_heat_record("X", {})
    mine = tasks._book_gap_risk(positions, PV, stops, {t: _bars([0.0]) for t in positions})
    assert heat["read"] == "ok"
    assert mine["loss_at_stops_pct_nav"] == pytest.approx(heat["sum_open_risk_pct"] * 100, abs=1e-3)


# ── the writer ───────────────────────────────────────────────────────────────

def _write(inp, bars=None, done=frozenset(), enabled=True, monkeypatch=None, age_min=5):
    if inp is not None and "ts" not in inp:
        inp = {**inp, "ts": datetime.now(timezone.utc) - timedelta(minutes=age_min)}
    if inp is not None:
        inp = {**inp, "bars": bars or {}}
    with patch("config.GAP_RISK_RECORD_ENABLED", enabled), \
         patch("tasks._gap_risk_inputs", return_value=inp) as reads, \
         patch("db.get_price_history", side_effect=AssertionError("bars come from the one read")), \
         patch("db.get_activity_shadow_keys", return_value=set(done)), \
         patch("db.log_activity_direct", return_value=True) as act:
        ok = tasks.book_gap_risk_record("2026-11-20")
    return ok, act, reads


def test_writer_row_carries_window_definitions_and_dates():
    b = _bars([0.0, -0.03])
    b[-1]["date"] = "2026-11-20"
    ok, act, _ = _write({"positions": {"ANET": _pos(0.1)}, "pv": PV, "stops": [("ANET", 92.0)]},
                        {"ANET": b})
    assert ok
    stage, ticker, msg, _, x = act.call_args.args
    assert stage == "book_gap_risk" and ticker == "" and "%" in msg and "$" not in msg
    assert x["read"] == "ok" and x["session_date"] == "2026-11-20" and x["window_gaps"] == 250
    assert x["definitions"]["gap"].startswith("open on day t / close on day t-1 - 1")
    assert x["close_dates"] == ["2026-11-20"] and x["close_is_today"] is True
    assert x["shadow_key"] == f"{TRADING_ENV}:2026-11-20" and x["env"] == TRADING_ENV
    assert x["positions"][0]["first_gap_date"] == b[1]["date"]
    assert x["snapshot_age_min"] == pytest.approx(5, abs=0.5)


def test_writer_dedupes_disables_and_reports_unreadable_inputs():
    ok, act, reads = _write({"positions": {}, "pv": PV, "stops": []},
                            done={f"{TRADING_ENV}:2026-11-20"})
    assert ok is False and not act.called and not reads.called
    ok, act, reads = _write({"positions": {}, "pv": PV, "stops": []}, enabled=False)
    assert ok is False and not act.called and not reads.called
    ok, act, _ = _write(None)
    assert act.call_args.args[4]["read"] == "no_snapshot"
    with patch("config.GAP_RISK_RECORD_ENABLED", True), \
         patch("tasks._gap_risk_inputs", side_effect=RuntimeError("down")), \
         patch("db.get_activity_shadow_keys", return_value=set()), \
         patch("db.log_activity_direct", return_value=True) as act:
        assert tasks.book_gap_risk_record("2026-11-20")
    x = act.call_args.args[4]
    assert x["read"] == "error: RuntimeError" and x["close_is_today"] is None
    ok, act, _ = _write({"positions": {"A": _pos(0.1)}, "pv": PV, "stops": []}, age_min=90)
    assert act.call_args.args[4]["read"] == "stale_snapshot"
    with patch("db.get_activity_shadow_keys", side_effect=RuntimeError("x")):
        assert tasks.book_gap_risk_record("2026-11-20") is False      # never raises


def test_writer_reads_only_the_db():
    src = inspect.getsource(tasks.book_gap_risk_record) + inspect.getsource(tasks._gap_risk_inputs)
    for word in ("ibkr", "_get_client", "alerts", "apply_async", "requests", "yfinance",
                 "INSERT", "UPDATE "):
        assert word not in src, word


def test_inputs_are_three_bounded_statements_in_one_read():
    from unittest.mock import MagicMock
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    snap = ({"ANET": {"qty": 3, "market_value": 1.0}, "OLD": {"qty": 0, "market_value": 0}},
            PV, datetime(2026, 11, 20, 22, 30, tzinfo=timezone.utc))
    cur.fetchone.return_value = snap
    cur.fetchall.side_effect = [[("ANET", 92.0)],
                                [("ANET", date(2026, 11, 19), 100.0, 101.0),
                                 ("ANET", date(2026, 11, 20), 99.0, 98.0)]]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        inp = tasks._gap_risk_inputs()
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert sqls[0] == "SET LOCAL statement_timeout = 3000" and len(sqls) == 4
    assert "ROW_NUMBER()" in sqls[3] and cur.execute.call_args_list[3].args[1] == (["ANET"], 251)
    assert inp["bars"]["ANET"][1] == {"date": "2026-11-20", "open": 99.0, "close": 98.0}
    conn.rollback.assert_called_once()
    put.assert_called_once()
    cur.fetchone.return_value = None
    cur.fetchall.side_effect = None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert tasks._gap_risk_inputs() is None


def test_a_failing_gap_record_leaves_the_exit_frame_rows_alone():
    stops = [{"ticker": "NVDA", "entry_price": 100.0, "stop_price": 92.0,
              "thesis_bucket": "Compounder", "trailing_active": False,
              "profit_target_taken": False, "opened_at": "2026-10-01 14:00:00"}]
    order = []
    with patch("db.get_all_position_stops", return_value=stops), \
         patch("db.get_price_history", return_value=[{"date": "2026-11-20", "close": 101.0}]), \
         patch("tasks._get_client") as client, \
         patch("db.log_activity_direct", side_effect=lambda *a: order.append(a[0]) or True), \
         patch("tasks.book_gap_risk_record",
               side_effect=lambda d: order.append("gap") or (_ for _ in ()).throw(RuntimeError("x"))), \
         patch("tasks.book_deployment_record", side_effect=lambda d: order.append("deployment")):
        client.return_value.get_all_positions.return_value = []
        tasks.position_exit_frame()                      # no raise
    assert order == ["position_exit_frame", "gap", "deployment"]


def test_position_exit_frame_writes_it_after_its_rows_and_on_its_early_return():
    calls = []
    with patch("db.get_all_position_stops", return_value=[]), \
         patch("tasks._get_client") as client, \
         patch("tasks.book_gap_risk_record", side_effect=lambda d: calls.append(d)):
        client.return_value.get_all_positions.return_value = []
        tasks.position_exit_frame()
    assert len(calls) == 1
    with patch("db.get_all_position_stops", side_effect=RuntimeError("x")), \
         patch("tasks.book_gap_risk_record", side_effect=lambda d: calls.append(d)):
        tasks.position_exit_frame()
    assert len(calls) == 2


# ── the readout ──────────────────────────────────────────────────────────────

def test_readout_table_names_and_definitions():
    from scripts import gap_risk_readout as gr
    pos = tasks._book_gap_risk({"ANET": _pos(0.1)}, PV, {"ANET": 81.0}, {"ANET": _bars([0.0, -0.1])})
    row = {**pos, "session_date": "2026-11-20", "close_is_today": True,
           "definitions": tasks._GAP_RISK_DEFINITIONS, "shadow_key": "live:2026-11-20", "env": "live"}
    text = gr.readout([row, dict(row)], date(2026, 11, 24), "live", "abc")
    assert "Rows: 1." in text                          # a re-run's duplicate shown once
    assert "| 2026-11-20 | ok | 1 | 10 |" in text and "## Names on 2026-11-20" in text
    assert "| ANET | 10 | 10 |" in text and "**gap:** open on day t" in text
    assert "$" not in text
    assert "No book_gap_risk rows yet." in gr.readout([], date(2026, 11, 24), "live", "x")


def test_readout_reads_only():
    from scripts import gap_risk_readout as gr
    src = inspect.getsource(gr)
    for word in ("INSERT", "UPDATE ", "DELETE FROM", "commit(", "log_activity", "apply_async",
                 "import ibkr", "alerts", "requests."):
        assert word not in src, word
    f = inspect.getsource(gr._fetch)
    assert "SET TRANSACTION READ ONLY" in f and "statement_timeout" in f and "rollback()" in f


def test_suspect_gaps_are_flagged_and_the_replay_is_given_with_and_without_them():
    assert tasks.GAP_SUSPECT_ABS_PCT == 40.0
    # A's third night is an 8-for-1 stored step (-87.5%); B's +41% is past the flag too.
    a = _bars([-0.05, 0.01, -0.875, 0.0, -0.02])
    b = _bars([0.0, 0.41, -0.03, 0.01, -0.40])     # -40% exactly is not flagged
    out = tasks._book_gap_risk({"A": _pos(0.10), "B": _pos(0.10)}, PV,
                               {"A": 1.0, "B": 1.0}, {"A": a, "B": b})
    pa, pb = out["positions"]
    assert pa["suspect_gaps"] == [{"date": a[3]["date"], "gap_pct": -87.5}]
    assert pb["suspect_gaps"] == [{"date": b[2]["date"], "gap_pct": 41.0}]
    assert out["n_suspect_gaps"] == 2 and out["suspect_names"] == ["A", "B"]
    assert out["suspect_nights"] == [a[2]["date"], a[3]["date"]]
    # Flagged, not dropped: the name's own stats and the full replay keep them.
    assert pa["n_gaps"] == 5 and pa["gap_worst_pct"] == pytest.approx(-87.5)
    assert out["n_common_nights"] == 5
    assert out["book_gap_worst_pct_nav"] == pytest.approx(0.1 * -87.5 + 0.1 * -3.0)
    assert out["book_gap_worst_date"] == a[3]["date"]
    # Without those two nights: nights 1, 4 and 5.
    assert out["n_common_nights_ex_suspect"] == 3
    assert out["book_gap_worst_pct_nav_ex_suspect"] == pytest.approx(0.1 * -2.0 + 0.1 * -40.0)
    assert out["book_gap_worst_date_ex_suspect"] == a[5]["date"]
    assert out["n_common_nights_any_past_stop_ex_suspect"] is not None


def test_no_suspect_gap_gives_the_same_replay_both_ways():
    a = _bars([-0.10, 0.0, 0.02])
    out = tasks._book_gap_risk({"A": _pos(0.10)}, PV, {"A": 1.0}, {"A": a})
    assert out["n_suspect_gaps"] == 0 and out["suspect_names"] == [] and out["suspect_nights"] == []
    assert out["positions"][0]["suspect_gaps"] == []
    for k in ("book_gap_p5_pct_nav", "book_gap_p1_pct_nav", "book_gap_worst_pct_nav",
              "book_gap_worst_date", "n_common_nights_any_past_stop"):
        assert out[k + "_ex_suspect"] == out[k], k
    assert out["n_common_nights_ex_suspect"] == out["n_common_nights"] == 3


def test_a_suspect_night_outside_the_common_nights_is_listed_on_the_name_only():
    a = _bars([0.50, -0.01, 0.0])                   # A's first night: B has no gap then
    b = _bars([-0.02, -0.05])
    b = [dict(x, date=a[i + 1]["date"]) for i, x in enumerate(b)]
    out = tasks._book_gap_risk({"A": _pos(0.1), "B": _pos(0.1)}, PV, {"A": 1.0, "B": 1.0},
                               {"A": a, "B": b})
    assert out["positions"][0]["suspect_gaps"] == [{"date": a[1]["date"], "gap_pct": 50.0}]
    assert out["suspect_nights"] == []
    assert out["n_common_nights_ex_suspect"] == out["n_common_nights"] == 2
    # Every night is suspect: the ex-suspect replay is empty, not an error.
    one = tasks._book_gap_risk({"A": _pos(0.1)}, PV, {"A": 1.0}, {"A": _bars([0.6])})
    assert one["read"] == "ok" and one["n_common_nights_ex_suspect"] == 0
    assert one["book_gap_worst_pct_nav_ex_suspect"] is None
    assert "suspect_gaps" in tasks._GAP_RISK_DEFINITIONS and "_ex_suspect" in tasks._GAP_RISK_DEFINITIONS


def test_readout_shows_suspect_gaps_and_the_replay_without_them():
    from scripts import gap_risk_readout as gr
    pos = tasks._book_gap_risk({"SNDK": _pos(0.1)}, PV, {"SNDK": 1.0},
                               {"SNDK": _bars([0.0, -0.875, 0.01])})
    row = {**pos, "session_date": "2026-11-24", "close_is_today": True,
           "definitions": tasks._GAP_RISK_DEFINITIONS, "shadow_key": "live:2026-11-24", "env": "live"}
    text = gr.readout([row], date(2026, 11, 24), "live", "abc")
    assert "suspect nights" in text and "worst without them" in text and "suspect gaps %" in text
    assert f"{pos['positions'][0]['suspect_gaps'][0]['date']} -87.5" in text
    assert "**suspect_gaps:**" in text and "$" not in text
    clean = tasks._book_gap_risk({"A": _pos(0.1)}, PV, {"A": 1.0}, {"A": _bars([0.0, -0.1])})
    assert "| none |" in gr.readout([{**clean, "session_date": "d", "env": "live"}],
                                    date(2026, 11, 24), "live", "x")
