"""Record-only fixes (branch add-loser-record):
- N1: a shadow-mode add-to-loser record (agents/crew.py:_add_loser_check)
  rides on the route rows' decision as add_loser_gate (the approval's
  risk_json key), so a gate-blocked ADD's ENTRY_GATE_BLOCKED row carries it;
  the gate outcome and every other field are unchanged.
- N6: the held-exit time stop record (agents/held_exit.py, trim_exit.
  _held_exit_fields) names its cost basis and records a first-fill cost
  beside the blended one (None on the live path: no read there).
Synthetic data only: no DB, broker or Redis."""
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import crew, held_exit, trim_exit


@pytest.fixture(autouse=True)
def _gates(monkeypatch):
    monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "shadow")
    monkeypatch.setattr(config, "ADD_LOSER_GATE_MIN_PCT", 0.0)
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "STOPOUT_REENTRY_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", False)
    monkeypatch.setattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", False)


def _result(action="ADD"):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": "Accelerator",
            "reasoning": "r", "pass_to_risk": True,
            "provenance": {"prompt_sha": "abc", "model": "m",
                           "trigger": {"path": "event", "source": "news", "title": "t",
                                       "new_fact": False}}}


def _client():
    c = MagicMock()
    c.get_all_positions.return_value = [
        SimpleNamespace(symbol="NVDA", avg_entry_price=100.0, current_price=72.5, qty=1)]
    return c


_EXIT = {"exit_at": "2026-11-22T20:00:00+00:00", "hours_since": 30.0, "lots": 1,
         "lot_outcome_pct": [-20.0], "lot_alpha_pct": [-15.0], "lot_convictions": [7],
         "entries_since_exit": 0}


def _route(action="ADD", exit_=_EXIT, check=None):
    """One routed verdict. exit_: a recent stop-out (the stop-out gate blocks)
    or None (no gate blocks; sizing returns 0 shares). check: patch
    _add_loser_check's return (None = remove the record altogether)."""
    ctx = [patch("db.has_active_approval", return_value=False),
           patch("db.get_last_executed_approval", return_value=None),
           patch("db.get_last_broker_exit", return_value=exit_),
           patch("agents.crew._get_price", return_value=72.5),
           patch("agents.crew.new_fact_gate", return_value=None),
           patch("risk_manager.RiskManager")]
    if check is not None:
        ctx.append(patch("agents.crew._add_loser_check", return_value=check()))
    with patch("db.log_signal") as ls, patch.object(crew, "_entry_gate", wraps=crew._entry_gate) as eg:
        for c in ctx:
            c.start()
        try:
            from risk_manager import RiskManager
            RiskManager.return_value.size_position.return_value = {
                "shares_initial": 0, "flags": ["too small"], "position_size_pct": 0}
            crew._route_to_approval(_result(action), {"regime": "Neutral"}, _client())
            sized = RiskManager.return_value.size_position.call_count
        finally:
            for c in reversed(ctx):
                c.stop()
    return sized, ls.call_args_list, eg


def _rows(calls, kind):
    return [c for c in calls if c[0][1] == kind]


def _strip(calls, key="add_loser_gate"):
    out = []
    for c in calls:
        kw = json.loads(json.dumps(c.kwargs, default=str))
        (kw.get("decision") or {}).pop(key, None)
        out.append([json.loads(json.dumps(c.args[1:], default=str)), kw])
    return out


# ── N1: the shadow record on the route rows ─────────────────────────────────

class TestShadowRecordOnRouteRows:
    def test_a_gate_blocked_shadow_add_carries_the_record(self):
        sized, calls, eg = _route()
        (row,) = _rows(calls, "ENTRY_GATE_BLOCKED")
        d = row[1]["decision"]
        rec = d["add_loser_gate"]
        assert rec["mode"] == "shadow" and rec["would_block"] is True and rec["blocked"] is False
        assert rec["pct_vs_avg_cost"] == -27.5
        assert "add_loser" not in d["entry_gate"]          # never a gate half in shadow
        assert eg.call_args.kwargs["add_loser"] is None     # the gate gets no record
        assert sized == 0

    def test_the_gate_outcome_and_every_other_field_are_unchanged(self):
        sized, calls, eg = _route()
        ref_sized, ref_calls, ref_eg = _route(check=lambda: None)
        assert sized == ref_sized == 0
        assert _strip(calls) == _strip(ref_calls)
        kw, ref_kw = dict(eg.call_args.kwargs), dict(ref_eg.call_args.kwargs)
        rx, ref_rx = kw.pop("row_extra"), ref_kw.pop("row_extra")
        assert "add_loser_gate" in rx and "add_loser_gate" not in ref_rx
        assert {k: v for k, v in rx.items() if k != "add_loser_gate"} == ref_rx
        assert kw == ref_kw and kw["add_loser"] is None          # the gate's own inputs: equal
        assert eg.call_args.args == ref_eg.call_args.args
        assert "add_loser_gate" not in _rows(ref_calls, "ENTRY_GATE_BLOCKED")[0][1]["decision"]

    def test_an_unblocked_shadow_add_carries_it_on_the_later_route_row(self):
        sized, calls, _ = _route(exit_=None)
        ref_sized, ref_calls, _ = _route(exit_=None, check=lambda: None)
        assert sized == ref_sized == 1
        (row,) = _rows(calls, "ROUTE_ZERO_SHARES")
        assert row[1]["decision"]["add_loser_gate"]["would_block"] is True
        assert not _rows(calls, "ENTRY_GATE_BLOCKED")
        assert _strip(calls) == _strip(ref_calls)

    def test_enforce_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "enforce")
        sized, calls, eg = _route()
        (row,) = _rows(calls, "ENTRY_GATE_BLOCKED")
        d = row[1]["decision"]
        assert "add_loser_gate" not in d
        assert d["entry_gate"]["add_loser"]["blocked"] is True
        assert eg.call_args.kwargs["add_loser"]["mode"] == "enforce"
        assert all("add_loser_gate" not in (c.kwargs.get("decision") or {}) for c in calls)

    def test_off_writes_no_record(self, monkeypatch):
        monkeypatch.setattr(config, "ADD_LOSER_GATE_MODE", "off")
        _, calls, _ = _route()
        assert all("add_loser_gate" not in (c.kwargs.get("decision") or {}) for c in calls)

    def test_a_non_add_is_unchanged(self):
        with patch("agents.crew._add_loser_check", wraps=crew._add_loser_check) as chk:
            sized, calls, _ = _route(action="OPEN")
        chk.assert_not_called()
        assert all("add_loser_gate" not in (c.kwargs.get("decision") or {}) for c in calls)
        assert _rows(calls, "ENTRY_GATE_BLOCKED")


# ── N6: the time stop's cost basis ──────────────────────────────────────────

LAST = date(2026, 11, 20)


def _bars(closes):
    n = len(closes)
    return [{"date": (LAST - timedelta(days=n - 1 - i)).isoformat(), "high": c + 1.0,
             "low": c - 1.0, "close": float(c)} for i, c in enumerate(closes)]


class TestHeldExitBasis:
    def _run(self, **kw):
        return held_exit.candidates(_bars([100.0 - i * 0.5 for i in range(40)]), entry_price=100.0,
                                    entry_date=LAST - timedelta(days=30), plan=None, **kw)

    def test_defaults_record_no_basis_and_no_first_fill(self):
        rec = self._run()
        assert rec["entry_price_basis"] is None and rec["entry_date_basis"] == "first_fill"
        assert rec["first_fill_price"] is None and rec["first_fill_price_source"] is None
        assert rec["first_fill_price_reason"] == "not_given"
        assert rec["close_vs_first_fill_pct"] is None
        assert held_exit.HELD_EXIT_VERSION == 1

    def test_the_decision_fields_do_not_move(self):
        base = self._run()
        rec = self._run(entry_basis="broker_avg_cost", first_fill_price=120.0,
                        first_fill_source="fills_table")
        for k in ("candidates", "firing", "any_would_exit", "sessions_held", "close_vs_entry_pct",
                  "entry_price"):
            assert rec[k] == base[k]
        assert rec["first_fill_price"] == 120.0 and rec["first_fill_price_source"] == "fills_table"
        assert rec["first_fill_price_reason"] is None
        assert rec["close_vs_first_fill_pct"] == round((80.5 / 120.0 - 1) * 100, 4)

    @pytest.mark.parametrize("bad", [0, -3, "x", float("nan")])
    def test_an_unusable_first_fill_is_none(self, bad):
        rec = self._run(first_fill_price=bad, first_fill_source="s", first_fill_reason="r")
        assert rec["first_fill_price"] is None and rec["first_fill_price_source"] is None
        assert rec["first_fill_price_reason"] == "r"

    def test_no_bars_still_records_the_basis(self):
        rec = held_exit.candidates([], entry_price=100.0, entry_date=LAST, plan=None,
                                   entry_basis="broker_avg_cost", first_fill_reason="no_read_on_path")
        assert rec["entry_price_basis"] == "broker_avg_cost"
        assert rec["first_fill_price_reason"] == "no_read_on_path"

    def test_the_live_path_names_the_blended_basis_and_reads_nothing_new(self, monkeypatch):
        monkeypatch.setattr(config, "HELD_EXIT_SHADOW_ENABLED", True)
        now = datetime(2026, 11, 23, 15, 30, tzinfo=timezone.utc)
        raw = [{**b, "open": b["close"], "volume": 1} for b in _bars([100.0] * 40)]
        reads = {}
        with patch("db.get_price_history", return_value=raw) as ph, \
             patch("db.get_active_plan", return_value=None) as gp:
            rec = trim_exit._held_exit_fields("NVDA", now, 100.0, (now - timedelta(days=30)).isoformat(),
                                              reads)
        ph.assert_called_once(); gp.assert_called_once()
        assert rec["entry_price_basis"] == "broker_avg_cost"
        assert rec["first_fill_price"] is None and rec["first_fill_price_source"] is None
        assert rec["first_fill_price_reason"] == "no_read_on_path"
        import ast
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(trim_exit._held_exit_fields)))
        db_calls = sorted(n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute)
                          and isinstance(n.func.value, ast.Name) and n.func.value.id == "db")
        assert db_calls == ["get_active_plan", "get_price_history"]   # no new read


def test_the_readout_names_the_basis():
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "shadow_outcomes.py")
    spec = importlib.util.spec_from_file_location("shadow_outcomes_basis_t", p)
    so = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(so)
    cands = {"candidates": {c: {"would_exit": False} for c in held_exit.CANDIDATES}}
    rows = [("NVDA", datetime(2026, 11, 20, 15, tzinfo=timezone.utc),
             {**cands, "entry_price_basis": "broker_avg_cost", "first_fill_price": None}),
            ("AMD", datetime(2026, 11, 20, 15, tzinfo=timezone.utc), cands)]
    with patch.object(so, "_forward", return_value=""):
        text = so.section_held_exit(rows, {}, date(2026, 11, 20), "live", "x")
    assert "Rows by entry_price_basis: broker_avg_cost 1, not recorded 1" in text
    assert "rows with a first-fill cost: 0" in text and "research choice (Luke)" in text


def test_template_shas_unchanged():
    from agents import analyst
    assert analyst._template_shas() == {
        "addon_template_sha": "ea7b89255103", "brief_template_full_sha": "a994de527dcd",
        "brief_template_sha": "2d7efec589d0", "prompt_db_queries_sha": "1ce9c6ccff32",
        "prompt_template_full_sha": "c553694b23b4", "prompt_template_sha": "6b93fe2037b7"}
