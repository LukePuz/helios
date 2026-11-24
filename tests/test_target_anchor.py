"""Record-only target anchoring and fundamentals staleness
(agents/target_anchor.py) on signals.decision.target_anchor: the bull target
against the street target and fair value the brief showed (parsed from the
prompt sent), and the fundamentals' quarter-values time against the last
print. Nothing in the prompt, brief, gates or routing reads it."""
import importlib.util
import inspect
import json
import os
from contextlib import ExitStack
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import analyst
from agents import target_anchor as ta

_BV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "batch_verify.py")
_spec = importlib.util.spec_from_file_location("batch_verify_target_anchor", _BV_PATH)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)

UTC = timezone.utc
LADDER = {"fair_value_blended": 150.0, "valuation_confidence": "medium",
          "street_consensus_target": 200.0, "current_price": 180.0, "rungs": []}


def _section(ladder):
    """The brief's own price-structure section for a ladder row."""
    import context_engine
    with patch.object(config, "BRIEF_PRICE_STRUCTURE", True), \
         patch("agents.price_action.compute_price_action", return_value=None), \
         patch("agents.price_action.compute_trend_template", return_value=None), \
         patch("agents.chase_risk.compute_chase_risk", return_value=None), \
         patch("db.get_trade_ladder", return_value=ladder):
        return context_engine._price_structure_section("NVDA", 180.0)


# ── the brief's values, parsed from the text ───────────────────────────────

def test_parses_the_line_the_brief_writes():
    text = _section(LADDER)
    assert "Blended fair value" in text
    got = ta.brief_targets("header\n" + text)
    assert got == {"fair_value": 150.0, "street_target": 200.0, "fv_confidence": "medium",
                   "fair_value_lines": 1}


@pytest.mark.parametrize("conf,want", [("high", "high"), ("medium", "medium"), ("low", "low"),
                                       (None, None)])
def test_fv_confidence_is_the_lines_own_word(conf, want):
    text = _section(dict(LADDER, valuation_confidence=conf))
    assert ("confidence n/a" in text) is (conf is None)
    assert ta.brief_targets(text)["fv_confidence"] == want


def test_no_family_values_are_recorded_from_the_line():
    rec = ta.targets_record({"valuation": {"bull_case_target": "$150"}}, _section(LADDER))
    assert not [k for k in rec if "intrinsic" in k or "market_family" in k or "family" in k]


def test_target_equals_fv_and_street_within_the_tolerance():
    assert ta.TARGET_EQUALS_TOL == 0.005
    rec = ta.targets_record({"valuation": {"bull_case_target": "$150.70"}}, _section(LADDER))
    assert rec["target_equals_fv"] is True and rec["target_equals_street"] is False
    assert rec["target_equals_tol"] == 0.005
    rec = ta.targets_record({"valuation": {"bull_case_target": "$150.76"}}, _section(LADDER))
    assert rec["target_equals_fv"] is False                      # 0.507%: just past it
    rec = ta.targets_record({"valuation": {"bull_case_target": "$199.10"}}, _section(LADDER))
    assert rec["target_equals_fv"] is False and rec["target_equals_street"] is True
    rec = ta.targets_record({"valuation": {"bull_case_target": "$149.30"}}, _section(LADDER))
    assert rec["target_equals_fv"] is True                       # below fair value counts too


def test_equality_is_none_when_a_side_is_missing():
    na = ta.targets_record({"valuation": {"bull_case_target": "$150"}},
                           _section(dict(LADDER, street_consensus_target=None)))
    assert na["target_equals_fv"] is True and na["target_equals_street"] is None
    none = ta.targets_record({"valuation": {"bull_case_target": "$150"}}, "no section")
    assert none["target_equals_fv"] is None and none["fv_confidence"] is None
    unparsed = ta.targets_record({"valuation": {"bull_case_target": "Unknown"}}, _section(LADDER))
    assert unparsed["target_equals_fv"] is None and unparsed["target_equals_street"] is None
    assert unparsed["fv_confidence"] == "medium"


def test_street_na_and_no_line_carry_a_reason():
    na = ta.targets_record({"valuation": {"bull_case_target": "$200"}},
                           _section(dict(LADDER, street_consensus_target=None)))
    assert na["fair_value"] == 150.0 and na["street_target"] is None
    assert na["missing"]["street_target"] == "n/a in the brief"
    assert na["target_vs_fair_value_pct"] == 33.33 and na["target_vs_street_pct"] is None
    none = ta.targets_record({"valuation": {"bull_case_target": "$200"}}, "no section")
    assert none["missing"]["fair_value"] == "no fair value line in the prompt"
    assert none["target_vs_street_pct"] is None and none["target_vs_fair_value_pct"] is None


def test_the_percentages_from_the_bull_target():
    rec = ta.targets_record({"valuation": {"bull_case_target": "$204 (12 months)"}},
                            _section(LADDER))
    assert rec["bull_target"] == 204.0
    assert rec["target_vs_street_pct"] == 2.0 and rec["target_vs_fair_value_pct"] == 36.0
    assert rec["missing"] == {}


@pytest.mark.parametrize("parsed", [None, {}, {"valuation": "x"},
                                    {"valuation": {"bull_case_target": "Unknown"}}])
def test_an_unparsed_bull_target_is_none_with_a_reason(parsed):
    rec = ta.targets_record(parsed, _section(LADDER))
    assert rec["bull_target"] is None and rec["missing"]["bull_target"] == "unparsed"
    assert rec["street_target"] == 200.0 and rec["target_vs_street_pct"] is None


def test_the_first_line_is_used_and_the_count_kept():
    two = _section(LADDER) + "\n" + _section(dict(LADDER, fair_value_blended=99.0))
    got = ta.brief_targets(two)
    assert got["fair_value"] == 150.0 and got["fair_value_lines"] == 2


# ── fundamentals timing ─────────────────────────────────────────────────────

def _h(day, hour, *vals):
    return (datetime(2026, 11, day, hour, tzinfo=UTC), *vals)


def test_as_of_is_the_first_snapshot_of_the_current_quarter_values():
    rows = [_h(20, 22, 1.032, 0.27, 0.62, 1.17), _h(19, 22, 1.032, 0.27, 0.62, 1.17),
            _h(18, 22, 1.059, 0.14, 0.64, 1.17), _h(17, 22, 1.059, 0.14, 0.64, 1.17)]
    assert ta.fundamentals_as_of(rows) == (rows[1][0], "changed")
    assert ta.fundamentals_as_of(rows[2:]) == (rows[3][0], "history_start")
    assert ta.fundamentals_as_of([]) == (None, None)


def test_predates_compares_et_dates_and_same_day_is_none():
    print_day = date(2026, 11, 17)
    assert ta.predates(datetime(2026, 11, 10, 22, tzinfo=UTC), print_day) is True
    assert ta.predates(datetime(2026, 11, 19, 22, tzinfo=UTC), print_day) is False
    assert ta.predates(datetime(2026, 11, 17, 22, tzinfo=UTC), print_day) is None
    # 11-18 02:00 UTC is 11-17 21:00 ET: the print day.
    assert ta.predates(datetime(2026, 11, 18, 2, tzinfo=UTC), print_day) is None
    assert ta.predates(None, print_day) is None and ta.predates(datetime.now(UTC), None) is None


class _Cur:
    def __init__(self, results, fail_at=None):
        self.sql, self.results, self.fail_at, self._last = [], list(results), fail_at, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sql.append((sql, params))
        if self.fail_at is not None and self.fail_at in sql:
            raise RuntimeError("statement timeout")
        if "SELECT" in sql:
            self._last = self.results.pop(0)

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last


def _conn(cur):
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


AT = datetime(2026, 11, 18, 15, 0, tzinfo=UTC)          # 10:00 ET on 11-18
HIST = [_h(17, 22, 1.059, 0.14, 0.64, 1.17), _h(10, 22, 1.059, 0.14, 0.64, 1.17),
        _h(9, 22, 1.0, 0.1, 0.6, 1.1)]
# As psycopg2 decodes the json_agg column: ISO strings and numbers.
HIST_JSON = [[r[0].isoformat(), *r[1:]] for r in HIST]


def test_the_db_read_is_bounded_read_only_and_as_of_at():
    cur = _Cur([(datetime(2026, 11, 17, 22, tzinfo=UTC), date(2026, 11, 17), 0, HIST_JSON)])
    conn = _conn(cur)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        out = ta.read_fundamentals_timing("NVDA", AT)
    assert len(cur.sql) == 1                             # one execute, one round trip
    sql, params = cur.sql[0]
    assert sql.startswith("SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = %(ms)s; "
                          "SELECT ")
    assert sql.count("SELECT (SELECT") == 1 and sql.count(";") == 2
    assert all(k not in sql.upper() for k in ("INSERT", "UPDATE ", "DELETE"))
    assert params == {"ms": 500, "t": "NVDA", "today": date(2026, 11, 18), "at": AT,
                      "n": ta.HISTORY_ROWS}
    assert ta.STATEMENT_TIMEOUT_MS == 500
    conn.rollback.assert_called_once()
    put.assert_called_once_with(conn)
    assert out["timing_read"] == "ok"
    assert out["fundamentals_as_of"] == HIST[1][0].isoformat()
    assert out["fundamentals_as_of_bound"] == "changed"
    assert out["last_print_date"] == "2026-11-17" and out["print_today"] is False
    assert out["fundamentals_predate_last_print"] is True
    assert out["fundamentals_updated_at"] == "2026-11-17T22:00:00+00:00"


def test_a_failed_statement_fails_open():
    cur = _Cur([(None, None, 0, HIST_JSON)], fail_at="statement_timeout")
    conn = _conn(cur)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put:
        out = ta.read_fundamentals_timing("NVDA", AT)
    assert out["timing_read"] == "error: RuntimeError"
    assert out["fundamentals_predate_last_print"] is None and out["fundamentals_as_of"] is None
    conn.rollback.assert_called_once() and put.assert_called_once_with(conn)


def test_no_connection_fails_open():
    with patch("db._get_conn", side_effect=RuntimeError("pool")):
        out = ta.read_fundamentals_timing("NVDA", AT)
    assert out["timing_read"] == "error: RuntimeError" and out["last_print_date"] is None


def test_no_rows_reads_as_unknown_not_error():
    cur = _Cur([(None, None, 1, [])])
    with patch("db._get_conn", return_value=_conn(cur)), patch("db._put_conn"):
        out = ta.read_fundamentals_timing("NEW", AT)
    assert out["timing_read"] == "ok" and out["fundamentals_as_of"] is None
    assert out["print_today"] is True and out["fundamentals_predate_last_print"] is None


# ── the record and the switch ───────────────────────────────────────────────

@pytest.mark.switches(TARGET_ANCHOR_RECORD_ENABLED=False)
def test_switch_off_reads_nothing():
    with patch("db._get_conn") as get:
        assert ta.target_anchor_record({}, "x", "NVDA") == {"enabled": False}
    assert not get.called


@pytest.mark.switches(TARGET_ANCHOR_RECORD_ENABLED=True)
def test_the_record_never_raises():
    with patch("agents.target_anchor.targets_record", side_effect=RuntimeError("x")):
        assert ta.target_anchor_record({}, "x", "NVDA") == {"error": "RuntimeError"}
    rec = ta.target_anchor_record({"valuation": {"bull_case_target": "$204"}},
                                  _section(LADDER), "NVDA")
    for k in bv.TARGET_ANCHOR_REQUIRED:
        assert k in rec, k
    assert rec["fundamentals_quarter_fields"] == list(ta.FUNDAMENTALS_QUARTER_FIELDS)


# ── analyst integration ─────────────────────────────────────────────────────

def _analyze(record=None, ladder=LADDER):
    parsed = {
        "action": "HOLD", "conviction": 6, "thesis_strength": 3, "catalyst_quality": 3,
        "valuation_risk_reward": 3, "data_completeness": 3, "thesis_bucket": "Compounder",
        "thesis_summary": "t", "expectation_gap": {}, "catalyst": {"event": "e", "date": "2026-11-20"},
        "valuation": {"current_multiple": "30x", "peer_range": "20x to 40x",
                      "bull_case_target": "$204", "downside_scenario": "$160"},
        "expected_return_pct": 13.0, "key_risks": [], "pass_to_risk": False,
        "upside_downside_ratio": 1.2, "reasoning": "r",
        "sources": [{"type": "news", "detail": "a"}, {"type": "x", "detail": "b"}],
    }
    log_signal = MagicMock()
    ollama = MagicMock(return_value=json.dumps(parsed))
    patches = [
        patch("agents.analyst._call_ollama", ollama),
        patch("knowledge_loader.load_thesis", return_value=""),
        patch("knowledge_loader.load_philosophy", return_value=""),
        patch("knowledge_loader.get_watchlist_entry", return_value={"role": "Compounder"}),
        patch("agents.chase_risk.compute_chase_risk", return_value=None),
        patch("agents.price_action.compute_price_action", return_value=None),
        patch("db.get_fundamentals", return_value={"market_cap": 1}),
        patch("db.get_latest_signal_for", return_value=None),
        patch("db.get_conviction_calibration", return_value=[]),
        patch("db.get_trade_ladder", return_value=None),
        patch("db.log_signal", log_signal),
        patch("feed.emit"),
        patch.object(config, "TARGET_ANCHOR_RECORD_ENABLED", True),
    ]
    if record is not None:
        patches.append(patch("agents.target_anchor.target_anchor_record", record))
    brief = "Current Price: $180.00\n" + _section(ladder)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = analyst.analyze({"ticker": "NVDA", "role": "Compounder", "tier": "Core",
                                  "sector": "Semis", "held": True},
                                 situation_brief=brief,
                                 regime_packet={"regime": "Risk-On", "score": 70, "reasoning": ""})
    return result, log_signal.call_args.kwargs["decision"], ollama.call_args


def _without_clock(d):
    if isinstance(d, dict):
        return {k: _without_clock(v) for k, v in d.items() if not k.endswith("_at")}
    return d


def test_on_the_verdicts_decision_from_the_prompt_sent():
    spy = MagicMock(wraps=ta.target_anchor_record)
    result, decision, sent = _analyze(record=spy)
    assert spy.call_count == 1
    assert spy.call_args.args[1] == sent.args[0]        # the exact prompt _call_ollama got
    rec = decision["target_anchor"]
    # db.get_trade_ladder is patched to None for the call, so 200 / 150 can
    # only come from the brief text in the prompt.
    assert rec["street_target"] == 200.0 and rec["fair_value"] == 150.0
    assert rec["target_vs_street_pct"] == 2.0 and rec["target_vs_fair_value_pct"] == 36.0


def test_a_failing_record_changes_nothing_else():
    result, decision, sent = _analyze()
    off, decision_off, sent_off = _analyze(record=MagicMock(side_effect=RuntimeError("x")))
    assert decision_off["target_anchor"] == {"error": "RuntimeError"}
    assert sent == sent_off
    for r in (result, off):
        r.pop("timestamp")
    assert _without_clock(result) == _without_clock(off)
    assert "target_anchor" not in result
    skip = {"target_anchor", "catalyst_check"}
    assert {k: v for k, v in decision.items() if k not in skip} == \
           {k: v for k, v in decision_off.items() if k not in skip}


def test_not_in_the_prompt_or_brief_path():
    for f in (analyst._build_prompt, analyst._prompt_template_full_text):
        assert "target_anchor" not in inspect.getsource(f)
    import context_engine
    import knowledge_loader
    for m in (context_engine, knowledge_loader):
        assert "target_anchor" not in inspect.getsource(m)
    assert "agents.target_anchor" not in analyst._BRIEF_MODULES


# ── batch_verify ────────────────────────────────────────────────────────────

def _v(i, **over):
    rec = {k: None for k in bv.TARGET_ANCHOR_REQUIRED}
    rec.update({"timing_read": "ok", **over})
    return {"id": i, "ticker": "NVDA", "catalyst_check": {}, "target_anchor": rec}


def test_batch_verify_ok_warn_fail():
    assert bv.check_target_anchor([_v(1), _v(2, street_target=200.0)])["status"] == "OK"
    for k in ("fv_confidence", "target_equals_fv", "target_equals_street", "target_equals_tol"):
        assert k in bv.TARGET_ANCHOR_REQUIRED
        short = _v(3)
        short["target_anchor"].pop(k)
        assert bv.check_target_anchor([short])["status"] == "FAIL"
    chk = bv.check_target_anchor([_v(1, fv_confidence="low", target_equals_fv=True)])
    assert chk["reads"]["fv_confidence"]["low"] == 1
    assert bv.check_target_anchor([])["status"] == "WARN"
    assert bv.check_target_anchor([_v(1), {"id": 2, "ticker": "X", "target_anchor": None}])["status"] == "FAIL"
    errs = [_v(i, timing_read="error: RuntimeError") for i in range(3)] + [_v(9)]
    assert bv.check_target_anchor(errs)["status"] == "FAIL"
    off = bv.check_target_anchor([{"id": 1, "ticker": "X", "target_anchor": None}], enabled=False)
    assert off["status"] == "OK"


def test_batch_verify_reads_it_from_the_shared_verdict_fetch():
    assert "check_target_anchor(verdict_rows, target_anchor_on)" in inspect.getsource(bv.run_checks)
    assert "target_anchor_on=target_anchor_enabled()" in inspect.getsource(bv.main)
    fetch = inspect.getsource(bv._fetch_verdicts)
    assert "decision->'target_anchor'" in fetch and "SET TRANSACTION READ ONLY" in fetch
