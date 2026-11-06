"""helios3d.replay harness: diffs, oracle steps, routing, sizing, the
read-only DB source and byte-identical reports. Fakes only; no live DB."""
import json
import random
import re
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from helios3d.replay import __main__ as cli
from helios3d.replay import gates as G
from helios3d.replay import harness as H

DAY = date(2026, 10, 29)
T0 = datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc)
CATALYST = (T0 + timedelta(days=14)).date().isoformat()


def _mo(action="WATCH", conviction=5, udr=3.0, pass_to_risk=False, ret=25.0, sources=2, **extra):
    mo = {"action": action, "conviction": conviction, "upside_downside_ratio": udr,
          "pass_to_risk": pass_to_risk, "expected_return_pct": ret,
          "catalyst": {"date": CATALYST},
          "sources": [{"type": "news", "detail": str(i)} for i in range(sources)]}
    mo.update(extra)
    return mo


def _sizing_cfg():
    import config
    return {k: H._jsonable(getattr(config, k)) for k in H.SIZING_CONFIG_KEYS}


_SIZING_CFG = _sizing_cfg()


def _prov(role_cfg=None, regime="Risk-On", held=True, fund="ok", prompt_sha="p1"):
    prov = {"config": {"SETUP_GATES_ENABLED": False, "MIN_EXPECTED_RETURN_PCT_PER_WEEK": 1.0,
                       "CATALYST_MONTH_MIDPOINT_ENABLED": False,
                       "ETF_FUNDAMENTALS_CAP_EXEMPT": False, **_SIZING_CFG, **(role_cfg or {})},
            "prompt_sha": prompt_sha, "fundamentals_read": fund,
            "brief_exposure": {"held_at_gate": held}}
    if regime is not None:
        prov["regime_used"] = {"label": regime, "source": "overall", "timestamp": None}
    if fund is None:
        del prov["fundamentals_read"]
    if held == "absent":
        del prov["brief_exposure"]
    return prov


def _decision(mo, inp_overrides=None, **prov_kw):
    """A recorded decision consistent with today's gates."""
    prov = _prov(**prov_kw)
    rec_inp = {"held": prov.get("brief_exposure", {}).get("held_at_gate"), "bucket": "Compounder",
               "regime": prov.get("regime_used", {}).get("label"), "fund_read": "ok",
               "as_of": T0, "min_return_per_week": 1.0, "month_midpoint": False,
               "etf_exempt": False, "trend_template_pass": True,
               "prev_verdict": None, "is_etf": False}
    rec_inp.update(inp_overrides or {})
    out = G.replay_gates(mo, rec_inp)
    dec = {k: out[k] for k in ("model_action", "model_conviction", "model_pass_to_risk",
                               "final_action", "final_conviction", "final_pass_to_risk", "gates")}
    dec["model_output"] = mo
    return dec, prov, out


class FakeSource:
    def __init__(self):
        self.recs, self.sigs, self.appr, self.route_rows, self.vids = [], {}, {}, [], {}
        self.prev, self.port = {}, {}
        self.calls = []

    def add(self, sid, ticker, mo, decision=None, prov=None, role="Compounder", approval=None,
            ts=None, signal=None, env="paper", code_sha="abc"):
        if decision is None:
            decision, prov, _ = _decision(mo)
        self.recs.append({"signal_id": sid, "decision_id": f"d{sid}", "as_of_ts": ts or T0,
                          "ticker": ticker, "trading_env": env,
                          "action": signal or decision.get("final_action"),
                          "conviction": 5, "code_sha": code_sha, "trigger": "daily_scan",
                          "approval_id": approval})
        self.sigs[sid] = {"signal": signal or decision.get("final_action"), "confidence": 0.5,
                          "decision": decision, "provenance": prov, "prompt_role": role}
        self.vids.setdefault(ticker, []).append(sid)

    def records(self, lo, hi):
        self.calls.append("records")
        return [r for r in self.recs if lo <= r["as_of_ts"] < hi], "fake"

    def signals(self, ids):
        return {i: self.sigs[i] for i in ids if i in self.sigs}

    def approvals(self, ids):
        return {i: self.appr[i] for i in ids if i in self.appr}

    def routes(self, verdicts, hi):
        return H.assign_routes(verdicts, self.route_rows,
                               {t: sorted(v) for t, v in self.vids.items()})

    def prev_verdict(self, rec):
        self.calls.append(("prev", rec["signal_id"]))
        return self.prev.get(rec["signal_id"])

    def portfolio(self, rec):
        return self.port.get(rec["signal_id"])


def _entry(report, sid):
    return next(e for e in report["decisions"] if e["signal_id"] == sid)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_consistent_records_match():
    src = FakeSource()
    src.add(1, "AAA", _mo("WATCH", 5))
    src.add(2, "BBB", _mo("OPEN", 6, pass_to_risk=True))      # Compounder c6 Risk-On: passes
    src.route_rows.append({"id": 3, "ticker": "BBB", "kind": "ROUTE_ZERO_SHARES",
                           "trading_env": "paper", "prompt_sha": "p1"})
    rep = H.run(DAY, DAY, src)
    assert rep["summary"]["status"] == {"match": 2}
    e = _entry(rep, 2)
    assert e["route"] == {"recorded": "ROUTE_ZERO_SHARES", "method": "prompt_sha",
                          "rows": ["ROUTE_ZERO_SHARES"]}
    assert e["fully_replayed"] is True           # Compounder: no Trend Template step
    assert {s["step"]: s["status"] for s in e["steps"]}["route_branch"] == G.NOT_REPLAYABLE


def test_trend_template_is_oracle_for_accelerator_buys():
    src = FakeSource()
    mo = _mo("OPEN", 8, pass_to_risk=True)
    dec, prov, _ = _decision(mo, {"bucket": "Accelerator", "trend_template_pass": False})
    assert any(G.gate_kind(g) == "trend_template" for g in dec["gates"])
    src.add(1, "AAA", mo, dec, prov, role="Accelerator")
    e = _entry(H.run(DAY, DAY, src), 1)
    assert e["status"] == "match" and e["fully_replayed"] is False
    tt = next(s for s in e["steps"] if s["step"] == "trend_template")
    assert tt["status"] == G.ORACLE and tt["fired"] is True


@pytest.mark.parametrize("record, tt_pass, fired", [
    ({"pass": False, "criteria": {}, "reasoning": "x"}, False, True),
    ({"pass": True, "criteria": {}, "reasoning": "x"}, True, False),
    ({"status": "insufficient_history"}, None, False),
    ({"status": "read_error"}, None, False),
])
def test_trend_template_is_replayed_from_the_recorded_check(record, tt_pass, fired):
    src = FakeSource()
    mo = _mo("OPEN", 8, pass_to_risk=True)
    dec, prov, _ = _decision(mo, {"bucket": "Accelerator", "trend_template_pass": tt_pass})
    dec["trend_template"] = record
    src.add(1, "AAA", mo, dec, prov, role="Accelerator", approval=None if fired else 9)
    e = _entry(H.run(DAY, DAY, src), 1)
    tt = next(s for s in e["steps"] if s["step"] == "trend_template")
    assert tt["status"] == G.REPLAYED and tt["fired"] is fired
    assert e["status"] == "match"
    assert e["input_sources"]["trend_template_pass"].startswith("decision.trend_template")


def test_trend_template_not_run_record_stays_oracle():
    sig = {"provenance": _prov(), "prompt_role": "Accelerator",
           "decision": {"trend_template": {"status": "not_run"}}}
    inp, src = H.gate_inputs({"as_of_ts": T0}, sig, None)
    assert "trend_template_pass" not in inp and src["trend_template_pass"] == "not recorded"


def test_regime_gate_replayed_and_diff_reported():
    src = FakeSource()
    mo = _mo("OPEN", 6, pass_to_risk=True)
    dec, prov, _ = _decision(mo, regime="Neutral")
    assert dec["final_action"] == "WATCH"
    src.add(1, "AAA", mo, dec, prov)
    # A record that says the OPEN went through although the regime gate applies.
    bad = dict(dec, final_action="OPEN", final_pass_to_risk=True, gates=[])
    src.add(2, "BBB", mo, bad, _prov(regime="Neutral"), approval=9)
    rep = H.run(DAY, DAY, src)
    assert _entry(rep, 1)["status"] == "match"
    e = _entry(rep, 2)
    assert e["status"] == "differ"
    assert e["diff"]["final_action"] == ["OPEN", "WATCH"]
    assert e["diff"]["routed"] == [True, False]
    assert e["diff"]["gate_kinds"] == [[], ["regime"]]
    assert rep["summary"]["differ_fields"]["final_action"] == 1


def test_not_replayable_reasons():
    src = FakeSource()
    src.add(1, "AAA", {}, decision={"final_action": "WATCH"}, prov={}, signal="WATCH")
    src.add(2, "BBB", {}, decision={"failure": {"kind": "timeout"}}, prov={}, signal="ANALYST_FAILED")
    src.add(3, "CCC", {}, decision={"final_action": "WATCH", "model_output": "text",
                                    "gates": []}, prov={}, signal="WATCH")
    src.add(4, "DDD", {}, decision={"final_action": "WATCH", "model_output": {"conviction": "high"},
                                    "gates": []}, prov={}, signal="WATCH")
    rep = H.run(DAY, DAY, src)
    reasons = rep["summary"]["not_replayable_reasons"]
    assert rep["summary"]["status"] == {"not_replayable": 4}
    assert reasons == {
        "no decision.model_output / final fields on the signals row": 1,
        "no verdict: ANALYST_FAILED": 1,
        "decision.model_output is not a JSON object": 1,
        "stored model output fails the live parse (ValueError)": 1}


def test_dated_catalyst_without_expected_return_downgrades():
    src = FakeSource()
    mo = _mo("OPEN", 8, pass_to_risk=True, ret=None)
    dec, prov, out = _decision(mo, {"bucket": "Accelerator"})
    assert out["gates"] == ["return rate (no expected_return_pct): OPEN -> WATCH"]
    src.add(1, "AAA", mo, dec, prov, role="Accelerator")
    e = _entry(H.run(DAY, DAY, src), 1)
    assert e["status"] == "match" and e["gate_text_equal"]


# ---------------------------------------------------------------------------
# Inputs and oracle steps
# ---------------------------------------------------------------------------

def test_missing_inputs_become_oracle_steps():
    src = FakeSource()
    mo = _mo("OPEN", 6, pass_to_risk=True)
    dec, _, _ = _decision(mo, regime="Neutral")
    prov = _prov(regime=None, held="absent", fund=None)
    src.add(1, "AAA", mo, dec, prov, role=None)
    e = _entry(H.run(DAY, DAY, src), 1)
    st = {s["step"]: s["status"] for s in e["steps"]}
    assert e["status"] == "match" and not e["fully_replayed"]
    assert st["regime"] == G.ORACLE and st["moonshot"] == G.ORACLE
    assert e["input_sources"]["bucket"] == "unknown"


def test_setup_gates_on_leaves_bucket_unknown_and_approval_bucket_used_otherwise():
    rec = {"as_of_ts": T0}
    inp, src = H.gate_inputs(rec, {"provenance": _prov(role_cfg={"SETUP_GATES_ENABLED": True}),
                                   "prompt_role": "Accelerator"}, {"bucket": "Moonshot"})
    assert "bucket" not in inp and src["bucket"].startswith("unknown: SETUP_GATES_ENABLED")
    inp, src = H.gate_inputs(rec, {"provenance": _prov(), "prompt_role": None}, {"bucket": "Moonshot"})
    assert inp["bucket"] == "Moonshot" and src["bucket"] == "pending_approvals.thesis_bucket"


def test_unrecorded_config_is_not_taken_from_today():
    inp, src = H.gate_inputs({"as_of_ts": T0}, {"provenance": {"config": {}},
                                                "prompt_role": "Accelerator"}, None)
    for k in ("min_return_per_week", "month_midpoint", "etf_exempt", "bucket"):
        assert k not in inp
    assert src["min_return_per_week"] == "not recorded"
    assert src["bucket"] == "unknown: SETUP_GATES_ENABLED not recorded"
    assert "held" not in inp and "regime" not in inp and "fund_read" not in inp



def test_the_malformed_catalyst_policy_is_passed_through_when_recorded():
    for policy in ("watch", "watch_rate_scoped"):
        prov = _prov(role_cfg={"malformed_catalyst_policy": policy})
        inp, src = H.gate_inputs({"as_of_ts": T0}, {"provenance": prov, "prompt_role": None}, None)
        assert inp["malformed_catalyst_policy"] == policy
        assert src["malformed_catalyst_policy"] == "provenance.config"
    inp, src = H.gate_inputs({"as_of_ts": T0}, {"provenance": _prov(), "prompt_role": None}, None)
    assert "malformed_catalyst_policy" not in inp
    assert src["malformed_catalyst_policy"] == "not recorded"


@pytest.mark.parametrize("policy, action, role, want_gates", [
    ("watch_rate_scoped", "OPEN", "Accelerator",
     ["malformed catalyst (expected_return_pct): OPEN -> WATCH"]),
    ("watch_rate_scoped", "EXIT", "Accelerator", []),
    ("watch_rate_scoped", "OPEN", "Compounder", []),
    ("watch", "EXIT", "Compounder", ["malformed catalyst (expected_return_pct): EXIT -> WATCH"]),
])
def test_null_return_rows_replay_under_their_recorded_policy(policy, action, role, want_gates):
    src = FakeSource()
    mo = _mo(action, 8, pass_to_risk=True, ret=None)
    dec, prov, out = _decision(mo, {"bucket": role, "malformed_catalyst_policy": policy},
                               role_cfg={"malformed_catalyst_policy": policy})
    assert out["gates"] == want_gates
    src.add(1, "AAA", mo, dec, prov, role=role)
    if out["final_pass_to_risk"]:
        src.route_rows.append({"id": 3, "ticker": "AAA", "kind": "ROUTE_ZERO_SHARES",
                               "trading_env": "paper", "prompt_sha": "p1"})
    e = _entry(H.run(DAY, DAY, src), 1)
    assert e["status"] == "match" and e["gate_text_equal"]
    st = {s["step"]: s["status"] for s in e["steps"]}
    assert G.ORACLE not in (st["malformed_catalyst"], st["malformed_return"])

def test_unrecorded_rate_config_makes_the_rate_step_oracle():
    src = FakeSource()
    mo = _mo("OPEN", 8, pass_to_risk=True, ret=1.0)
    dec, prov, out = _decision(mo, {"bucket": "Moonshot"})
    assert [G.gate_kind(g) for g in out["gates"]] == ["return_rate"]
    del prov["config"]["MIN_EXPECTED_RETURN_PCT_PER_WEEK"]
    src.add(1, "AAA", mo, dec, prov, role="Moonshot")
    e = _entry(H.run(DAY, DAY, src), 1)
    rr = next(s for s in e["steps"] if s["step"] == "return_rate")
    assert e["status"] == "match" and rr["status"] == G.ORACLE and rr["fired"]
    assert "not recorded" in rr["reason"]


def test_unrecorded_etf_exemption_is_oracle():
    mo = _mo("OPEN", 9, pass_to_risk=True)
    base = {"held": True, "bucket": "Compounder", "regime": "Risk-On", "fund_read": "missing",
            "as_of": T0, "min_return_per_week": 1.0, "month_midpoint": False, "is_etf": True}
    rec = G.replay_gates(mo, dict(base, etf_exempt=True))
    got = G.replay_gates(mo, base, rec["gates"])
    assert got["gates"] == rec["gates"] and got["final_conviction"] == 9
    fs = next(s for s in got["steps"] if s["step"] == "fundamentals")
    assert fs["status"] == G.ORACLE and "ETF_FUNDAMENTALS_CAP_EXEMPT" in fs["reason"]


def test_sizing_needs_recorded_config():
    src = _sized_source()
    del src.sigs[1]["provenance"]["config"]["ONE_SHARE_MAX_PCT_OF_CASH"]
    sz = _entry(H.run(DAY, DAY, src, sizing=True), 1)["sizing"]
    assert sz["reason"] == "config not recorded: ONE_SHARE_MAX_PCT_OF_CASH"


def test_low_conviction_sell_reads_previous_verdict_only_when_needed():
    src = FakeSource()
    mo = _mo("TRIM", 3, pass_to_risk=False)
    prev = {"signal": "HOLD", "confidence": 0.3, "timestamp": T0 - timedelta(days=1)}
    dec, prov, out = _decision(mo, {"prev_verdict": prev})
    assert out["final_pass_to_risk"] is True
    src.add(1, "AAA", mo, dec, prov)
    src.add(2, "BBB", _mo("WATCH", 5))
    src.prev[1] = prev
    src.route_rows.append({"id": 5, "ticker": "AAA", "kind": "SELL_NO_POSITION",
                           "trading_env": "paper", "prompt_sha": None})
    rep = H.run(DAY, DAY, src)
    assert [c for c in src.calls if c != "records"] == [("prev", 1)]
    e = _entry(rep, 1)
    assert e["status"] == "match" and e["fully_replayed"]
    assert e["route"]["method"] == "id_window" and e["route"]["recorded"] == "SELL_NO_POSITION"


def test_previous_verdict_unreadable_falls_back_to_oracle():
    src = FakeSource()
    mo = _mo("TRIM", 3)
    prev = {"signal": "HOLD", "confidence": 0.3, "timestamp": T0 - timedelta(days=1)}
    dec, prov, _ = _decision(mo, {"prev_verdict": prev})
    src.add(1, "AAA", mo, dec, prov)
    src.route_rows.append({"id": 5, "ticker": "AAA", "kind": "SELL_NO_POSITION",
                           "trading_env": "paper", "prompt_sha": None})

    def boom(rec):
        raise RuntimeError("db down")
    src.prev_verdict = boom
    e = _entry(H.run(DAY, DAY, src), 1)
    assert e["status"] == "match"
    assert e["input_sources"]["prev_verdict"] == "unreadable (RuntimeError)"
    assert next(s for s in e["steps"] if s["step"] == "low_conviction_sell")["status"] == G.ORACLE


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_assign_routes_windows_and_prompt_sha():
    verdicts = [{"signal_id": 10, "ticker": "AAA", "trading_env": "paper", "prompt_sha": "a",
                 "approval_id": None},
                {"signal_id": 20, "ticker": "AAA", "trading_env": "paper", "prompt_sha": "b",
                 "approval_id": 7},
                {"signal_id": 30, "ticker": "BBB", "trading_env": "paper", "prompt_sha": "c",
                 "approval_id": None}]
    rows = [{"id": 11, "ticker": "AAA", "kind": "CATALYST_GATE_ERROR", "trading_env": "paper",
             "prompt_sha": None},
            {"id": 12, "ticker": "AAA", "kind": "EARNINGS_BLOCKED", "trading_env": "paper",
             "prompt_sha": "a"},
            {"id": 13, "ticker": "AAA", "kind": "ROUTE_NO_PRICE", "trading_env": "paper",
             "prompt_sha": "zz"},          # names another verdict: dropped
            {"id": 14, "ticker": "AAA", "kind": "COOLDOWN_SKIPPED", "trading_env": "live",
             "prompt_sha": None},          # other env: dropped
            {"id": 25, "ticker": "AAA", "kind": "SHADOW_SWAP", "trading_env": "paper",
             "prompt_sha": None}]
    got = H.assign_routes(verdicts, rows, {"AAA": [10, 20], "BBB": [30]})
    assert got[10]["outcome"] == "EARNINGS_BLOCKED" and got[10]["method"] == "prompt_sha"
    assert got[10]["row_ids"] == [11, 12] and got[10]["next_verdict_id"] == 20
    assert got[20]["outcome"] == "approval" and got[20]["rows"] == ["SHADOW_SWAP"]
    assert got[30] == {"outcome": "none", "method": None, "rows": [], "row_ids": [],
                       "next_verdict_id": None}


def test_pass_without_routing_record_differs():
    src = FakeSource()
    src.add(1, "AAA", _mo("OPEN", 8, pass_to_risk=True))
    e = _entry(H.run(DAY, DAY, src), 1)
    assert e["replayed"]["routed"] is True and e["recorded"]["routed"] is False
    assert e["status"] == "differ" and list(e["diff"]) == ["routed"]


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def _sized_source():
    src = FakeSource()
    src.add(1, "AAA", _mo("OPEN", 8, pass_to_risk=True), approval=7)
    src.appr[7] = {"ticker": "AAA", "action": "OPEN", "conviction": 8, "bucket": "Compounder",
                   "trading_env": "paper", "shares_initial": None, "price_at_sizing": 50.0,
                   "stop_loss_price": 45.0, "size_capped": False}
    src.port[1] = {"pv": 100_000.0, "cash": 50_000.0, "positions": {"ZZZ": {"market_value": 1.0}}}
    return src


def test_sizing_replayed_through_limits():
    from helios3d.portfolio import limits
    src = _sized_source()
    want = limits.size_legacy(conviction=8, bucket="Compounder", regime="Risk-On", price=50.0,
                              pv=100_000.0, cash=50_000.0, held_value=0.0, open_count=1,
                              stop_price=45.0)
    src.appr[7]["shares_initial"] = want["shares_initial"]
    rep = H.run(DAY, DAY, src, sizing=True)
    sz = _entry(rep, 1)["sizing"]
    assert sz["status"] == G.REPLAYED and sz["match"] is True
    assert sz["replayed_shares"] == want["shares_initial"] > 0
    assert rep["summary"]["sizing"] == {"replayed_match": 1}
    # No account values in the report.
    text = H.report_bytes(rep).decode()
    assert "100000" not in text and "50000" not in text


def test_sizing_mismatch_and_disabled_and_drift():
    src = _sized_source()
    src.appr[7]["shares_initial"] = 1
    sz = _entry(H.run(DAY, DAY, src, sizing=True), 1)["sizing"]
    assert sz["status"] == G.REPLAYED and sz["match"] is False and "concentration" in sz["reason"]
    sz = _entry(H.run(DAY, DAY, src, sizing=False), 1)["sizing"]
    assert sz["status"] == G.NOT_REPLAYABLE and "--sizing" in sz["reason"]
    src.sigs[1]["provenance"]["config"]["MAX_TRADE_PCT_OF_CASH_RESERVE"] = -1
    sz = _entry(H.run(DAY, DAY, src, sizing=True), 1)["sizing"]
    assert sz["reason"] == "config changed since the decision: MAX_TRADE_PCT_OF_CASH_RESERVE"


def test_sizing_not_replayable_for_route_rows():
    src = FakeSource()
    src.add(1, "AAA", _mo("OPEN", 8, pass_to_risk=True))
    src.route_rows.append({"id": 2, "ticker": "AAA", "kind": "ROUTE_ZERO_SHARES",
                           "trading_env": "paper", "prompt_sha": "p1"})
    sz = _entry(H.run(DAY, DAY, src, sizing=True), 1)["sizing"]
    assert sz["status"] == G.NOT_REPLAYABLE and "ROUTE_ZERO_SHARES" in sz["reason"]


# ---------------------------------------------------------------------------
# Determinism, range, CLI
# ---------------------------------------------------------------------------

def _mixed_source(seed):
    src = FakeSource()
    rng = random.Random(7)          # the same rows for every seed
    items = []
    for sid in range(1, 41):
        mo = _mo(rng.choice(["OPEN", "WATCH", "HOLD", "TRIM", "ADD"]), rng.choice([3, 5, 6, 8]),
                 udr=rng.choice([1.0, 3.0]), pass_to_risk=rng.choice([True, False]))
        items.append((sid, mo))
    for sid, mo in items:
        src.add(sid, f"T{sid % 7}", mo, ts=T0 + timedelta(minutes=sid))
    random.Random(seed).shuffle(src.recs)   # only the order depends on the seed
    return src


def test_report_is_byte_identical_across_runs_and_input_order():
    a = H.report_bytes(H.run(DAY, DAY, _mixed_source(1)))
    b = H.report_bytes(H.run(DAY, DAY, _mixed_source(1)))
    c = H.report_bytes(H.run(DAY, DAY, _mixed_source(2)))   # same rows, another order
    assert a == b == c
    assert json.loads(a)["summary"]["decisions"] == 40


def test_range_is_et_days_inclusive():
    lo, hi = H.range_bounds(date(2026, 11, 2), date(2026, 11, 6))
    assert lo == datetime(2026, 11, 2, 5, 0, tzinfo=timezone.utc)
    assert hi == datetime(2026, 11, 7, 5, 0, tzinfo=timezone.utc)
    src = FakeSource()
    src.add(1, "AAA", _mo(), ts=datetime(2026, 10, 30, 3, 59, tzinfo=timezone.utc))  # 10-29 ET
    src.add(2, "AAA", _mo(), ts=datetime(2026, 10, 30, 4, 0, tzinfo=timezone.utc))   # 10-30 ET
    assert [e["signal_id"] for e in H.run(DAY, DAY, src)["decisions"]] == [1]
    with pytest.raises(ValueError):
        H.range_bounds(date(2026, 11, 3), date(2026, 11, 2))


def test_cli_writes_report_and_twice_gives_same_bytes(tmp_path, capsys):
    src = _mixed_source(3)
    out = tmp_path / "r.json"

    class _Conn:
        pass

    from contextlib import contextmanager

    @contextmanager
    def fake_connection():
        yield _Conn()
    with patch("helios3d.schema.conn.connection", fake_connection), \
            patch.object(H, "DbSource", lambda conn: src):
        assert cli.main(["--start", "2026-10-29", "--end", "2026-10-29", "--out", str(out)]) == 0
        first = out.read_bytes()
        assert cli.main(["--start", "2026-10-29", "--end", "2026-10-29", "--out", str(out)]) == 0
    assert out.read_bytes() == first
    printed = json.loads(capsys.readouterr().out.split("\n}\n")[0] + "\n}")
    assert printed["decisions"] == 40 and printed["out"] == str(out)
    with pytest.raises(SystemExit):
        cli.main(["--start", "2026-10-29", "--end", "2026-10-28"])
    with pytest.raises(SystemExit):
        cli.main(["--start", "2026-10-29", "--end", "2026-10-29", "--llm", "live"])


def test_default_out_is_under_repo_logs():
    p = H.default_out(date(2026, 11, 2), date(2026, 11, 6))
    assert p == H.REPO / "logs" / "helios3d" / "replay_2026-11-02_2026-11-06.json"


# ---------------------------------------------------------------------------
# DbSource: read-only SQL
# ---------------------------------------------------------------------------

class _RecCursor:
    def __init__(self, conn):
        self.conn, self._rows = conn, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        assert not re.match(r"(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE)\b", s, re.I), s
        if s == "SET TRANSACTION READ ONLY":
            self.conn.ro = True
            self._rows = []
            return
        assert self.conn.ro, f"query outside READ ONLY: {s[:60]}"
        self.conn.queries.append(s)
        self._rows = self.conn.answer(s, params)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _RecConn:
    def __init__(self, has_h3d=True):
        self.ro, self.queries, self.has_h3d = False, [], has_h3d

    def cursor(self):
        return _RecCursor(self)

    def rollback(self):
        self.ro = False

    def commit(self):
        raise AssertionError("replay must not commit")

    def answer(self, s, params):
        if s.startswith("SELECT to_regclass('h3d.decision_record')"):
            return [("h3d.decision_record" if self.has_h3d else None,)]
        if s.startswith("SELECT signal_id, decision_id"):
            return [(5, "d5", T0, "aaa ", "paper", "WATCH", 5, "abc", "daily_scan", None)]
        if s.startswith("SELECT id, signal, confidence, decision"):
            dec, prov, _ = _decision(_mo())
            return [(5, "WATCH", 0.5, dec, prov, "Compounder")]
        return []


def test_db_source_reads_only_inside_read_only_transactions():
    conn = _RecConn()
    rep = H.run(DAY, DAY, H.DbSource(conn))
    assert rep["record_source"] == "h3d.decision_record"
    assert _entry(rep, 5)["status"] == "match" and _entry(rep, 5)["ticker"] == "AAA"
    assert any("substring(prompt from E'\\nPortfolio Role: (Compounder|Accelerator|Moonshot)\\n')" in q for q in conn.queries)
    # The prompt is read only through that substring.
    for q in conn.queries:
        q2 = q.replace("prompt_sha", "")
        if "prompt" in q2:
            assert "substring(prompt from" in q2 and q2.count("prompt") == 1, q


def test_db_source_builds_records_when_h3d_table_absent():
    conn = _RecConn(has_h3d=False)
    rep = H.run(DAY, DAY, H.DbSource(conn))
    assert rep["record_source"].startswith("built in memory")
    assert rep["decisions"] == []
    assert any("FROM public.signals" in q for q in conn.queries)


def test_package_imports_no_live_analyst():
    import ast
    import pathlib
    for p in pathlib.Path(H.__file__).parent.glob("*.py"):
        tree = ast.parse(p.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not [n for n in names if n.startswith("agents") or n in ("risk_manager", "tasks")], p
