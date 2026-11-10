"""Shadow readout counts key paths and true booleans, never values (A1031-1453-4)."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "shadow_readout", os.path.join(os.path.dirname(__file__), "..", "scripts", "shadow_readout.py"))
shadow_readout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shadow_readout)


def test_counts_paths_and_true_booleans_without_values():
    rows = [
        ("HOLD", {"gate_flags": {"rubric_gate_conflict": True, "verdicts_today": 3}}, None),
        ("HOLD", {"gate_flags": {"rubric_gate_conflict": False}}, {"price": 123.45}),
        ("SELL_SKIPPED_CLOSED", None, None),
    ]
    text = shadow_readout.readout(rows, 7, "live", "abc123")
    assert "| HOLD | 2 |" in text and "| SELL_SKIPPED_CLOSED | 1 |" in text
    assert "| provenance.gate_flags.rubric_gate_conflict | 2 | 1 |" in text
    assert "| provenance.gate_flags.verdicts_today | 1 |  |" in text
    assert "123.45" not in text and "Rows: 3" in text


def test_depth_is_capped():
    deep = {"a": {"b": {"c": {"d": True}}}}
    text = shadow_readout.readout([("X", deep, None)], 7, "live", "h")
    assert "provenance.a.b" in text and "provenance.a.b.c.d" not in text


def test_routing_rows_do_not_recount_the_verdict_provenance():
    prov = {"model": "qwen2.5:72b", "llm_call": {"total_duration": 1}, "would_revise": True}
    rows = [("OPEN", prov, None),
            ("ROUTE_APPROVAL_PENDING", dict(prov), {"route_exit": "ROUTE_APPROVAL_PENDING"}),
            ("COOLDOWN_SKIPPED", dict(prov), {"route_exit": "COOLDOWN_SKIPPED"})]
    text = shadow_readout.readout(rows, 7, "live", "h")
    assert "| provenance.would_revise | 1 | 1 |" in text
    assert "| provenance.llm_call.total_duration | 1 |  |" in text
    assert "| decision.route_exit | 2 |  |" in text
    assert "| ROUTE_APPROVAL_PENDING | 1 |" in text and "Rows: 3" in text


def test_own_provenance_of_non_verdict_rows_is_still_counted():
    from agents import crew
    verdict = {"model": "qwen2.5:72b", "would_revise": True}
    copy = crew._route_row_fields({"provenance": verdict, "action": "OPEN", "conviction": 7},
                                  "ROUTE_NO_PRICE")
    rows = [("OPEN", verdict, None),
            ("ROUTE_NO_PRICE", copy["provenance"], copy["decision"]),
            ("SCREENED_IN", {"model": "qwen2.5:7b", "opinion_shadow": {"would_pass": True}}, {}),
            ("SCREENED_OUT", {"would_pass_held_negative": True}, None),
            ("PLAN_LEVEL_HIT", {"plan_id": 1, "review_queued": False}, None),
            ("EVENT_LOST", {"code_sha": "abc"}, {"reason": "expired"})]
    text = shadow_readout.readout(rows, 7, "live", "h")
    assert "| provenance.would_revise | 1 | 1 |" in text           # the ROUTE_* copy is not counted
    assert "| provenance.model | 2 |  |" in text
    assert "| provenance.opinion_shadow.would_pass | 1 | 1 |" in text
    assert "| provenance.would_pass_held_negative | 1 | 1 |" in text
    assert "| provenance.plan_id | 1 |  |" in text and "| provenance.code_sha | 1 |  |" in text
    assert "| decision.route_exit | 1 |  |" in text and "| ROUTE_NO_PRICE | 1 |" in text
