"""The thesis drift diff is recorded after each held review's thesis update
(audit 11-06 01:32 #2): log-only, never raises."""
from unittest.mock import patch

from agents import crew

ORIGINAL = {"as_of": "2026-10-01T14:00:00+00:00", "action": "OPEN", "conviction": 7,
            "catalyst": {"event": "earnings", "date": "2026-10-20"}, "catalyst_status": "pending",
            "weeks_to_catalyst": 3, "no_near_term_catalyst": False, "key_risks": ["capex"]}
RESULT = {"action": "HOLD", "conviction": 5, "catalyst": {"event": "none", "date": None},
          "weeks_to_catalyst": None, "key_risks": ["capex", "china"], "reasoning": "r"}


def _capture(thesis_row, opens=(), stop=None):
    with patch("db.update_latest_thesis"), patch("db.compute_no_near_term_catalyst", return_value=True), \
         patch("db.get_thesis", return_value=thesis_row), \
         patch("db.get_open_outcomes", return_value=list(opens)), \
         patch("db.get_position_stop", return_value=stop), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("agents.setup_gates.check_legacy_position_health"):
        crew._capture_latest_thesis("NVDA", RESULT)
    return act


def test_drift_is_recorded_against_the_entry_thesis():
    act = _capture({"original": ORIGINAL, "latest": None})
    act.assert_called_once()
    stage, ticker, msg, sentiment, extra = act.call_args.args
    assert (stage, ticker, sentiment) == ("thesis_drift", "NVDA", "neutral")
    assert extra["conviction_delta"] == -2 and extra["material_change"] is True
    assert extra["new_risks"] == ["china"] and extra["catalyst_degraded"] is True
    assert extra["catalyst_date_passed"] is True and extra["catalyst_status"] == "pending"
    assert extra["original_action"] == "OPEN" and extra["latest_action"] == "HOLD"
    assert "-2" in msg
    import db
    assert extra["env"] == db.TRADING_ENV
    assert {"code_sha", "env_sha", "lib_versions"} <= set(extra)


def test_no_entry_thesis_writes_no_row():
    assert not _capture({"original": None, "latest": {}}).called
    assert not _capture(None).called


def test_a_drift_failure_never_blocks_the_capture(caplog):
    with patch("db.update_latest_thesis") as upd, patch("db.compute_no_near_term_catalyst", return_value=True), \
         patch("db.get_thesis", side_effect=RuntimeError("db down")), \
         patch("agents.setup_gates.check_legacy_position_health") as health, caplog.at_level("WARNING"):
        crew._capture_latest_thesis("NVDA", RESULT)
    upd.assert_called_once() and health.assert_called_once()
    assert health.called and "thesis drift record failed" in caplog.text


def test_a_float_conviction_still_writes_the_row():
    with patch("db.update_latest_thesis"), patch("db.compute_no_near_term_catalyst", return_value=True), \
         patch("db.get_thesis", return_value={"original": {**ORIGINAL, "conviction": 7.0}}), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("agents.setup_gates.check_legacy_position_health"):
        crew._capture_latest_thesis("NVDA", {**RESULT, "conviction": 5})
    assert act.call_args.args[2] == "Thesis drift: conviction -2.0"


def test_entry_thesis_from_an_earlier_position_is_flagged():
    """m587: the newest open outcome entered after the entry thesis was
    written means the thesis is the old position's."""
    from datetime import datetime, timezone
    new_entry = datetime(2026, 11, 5, 16, 8, tzinfo=timezone.utc)
    act = _capture({"original": ORIGINAL}, opens=[{"ticker": "NVDA", "entry_date": new_entry, "action": "OPEN"},
                                                  {"ticker": "AMD", "entry_date": new_entry, "action": "OPEN"}],
                   stop={"opened_at": datetime(2026, 9, 1, tzinfo=timezone.utc)})
    extra = act.call_args.args[4]
    assert extra["original_predates_position"] is True
    assert extra["position_entry_at"] == new_entry.isoformat()
    assert extra["position_entry_basis"] == "open_leg"
    assert extra["position_opened_at"].startswith("2026-09-01")


def test_position_check_unknown_without_an_open_outcome():
    extra = _capture({"original": ORIGINAL}).call_args.args[4]
    assert extra["original_predates_position"] is None and extra["position_entry_at"] is None


def test_later_add_legs_do_not_move_the_position_entry():
    """58h: an ADD opens its own outcome row; the reference stays the OPEN."""
    from datetime import datetime, timezone
    opened = datetime(2026, 10, 1, 14, 3, tzinfo=timezone.utc)      # 3 min after ORIGINAL.as_of
    add = datetime(2026, 10, 20, tzinfo=timezone.utc)
    extra = _capture({"original": ORIGINAL}, opens=[{"ticker": "NVDA", "entry_date": opened, "action": "OPEN"},
                                                    {"ticker": "NVDA", "entry_date": add, "action": "ADD"}]
                     ).call_args.args[4]
    assert extra["position_entry_at"] == opened.isoformat() and extra["original_predates_position"] is False
    only_adds = _capture({"original": ORIGINAL}, opens=[{"ticker": "NVDA", "entry_date": add, "action": "ADD"},
                                                        {"ticker": "NVDA", "entry_date": opened, "action": "ADD"}]
                         ).call_args.args[4]
    assert only_adds["position_entry_basis"] == "min_leg" and only_adds["position_entry_at"] == opened.isoformat()


def test_non_dict_provenance_keeps_the_row():
    with patch("db.update_latest_thesis"), patch("db.compute_no_near_term_catalyst", return_value=True), \
         patch("db.get_thesis", return_value={"original": ORIGINAL}), \
         patch("db.get_open_outcomes", return_value=[]), patch("db.get_position_stop", return_value=None), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("agents.setup_gates.check_legacy_position_health"):
        crew._capture_latest_thesis("NVDA", {**RESULT, "provenance": "not-a-dict"})
    assert act.called and act.call_args.args[4]["review_path"] is None


def test_drift_row_carries_calibration_fields():
    """Audit 11-06 03:34 #b: entry/latest/previous conviction, risk counts,
    catalyst precision, so a threshold can be set from data."""
    previous = {"as_of": "2026-11-01T14:00:00+00:00", "conviction": 6}
    act = _capture({"original": ORIGINAL, "latest": previous})
    extra = act.call_args.args[4]
    assert (extra["entry_conviction"], extra["latest_conviction"], extra["previous_conviction"]) == (7, 5, 6)
    assert extra["conviction_delta_vs_previous"] == -1
    assert extra["previous_as_of"] == "2026-11-01T14:00:00+00:00"
    assert (extra["original_risk_count"], extra["latest_risk_count"], extra["new_risk_count"]) == (1, 2, 1)
    assert (extra["original_catalyst_precision"], extra["latest_catalyst_precision"]) == ("day", None)


def test_first_review_has_no_previous_delta():
    extra = _capture({"original": ORIGINAL, "latest": None}).call_args.args[4]
    assert extra["previous_conviction"] is None and extra["conviction_delta_vs_previous"] is None


def test_catalyst_precision():
    assert [crew._catalyst_precision(c) for c in (
        {"date": "2026-11-20"}, {"date": "2026-11"}, {"date": "ongoing"}, {"date": "Q1 2027"},
        {"date": None}, None)] == ["day", "month", "ongoing", "other", None, None]


def test_previous_read_status_and_predating():
    from datetime import datetime, timezone
    previous = {"as_of": "2026-09-01T14:00:00+00:00", "conviction": 6}
    opens = [{"ticker": "NVDA", "action": "OPEN",
              "entry_date": datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)}]
    extra = _capture({"original": ORIGINAL, "latest": previous}, opens=opens).call_args.args[4]
    assert extra["previous_read"] == "ok" and extra["previous_predates_position"] is True
    extra = _capture({"original": ORIGINAL, "latest": None}).call_args.args[4]
    assert extra["previous_read"] == "none" and extra["previous_predates_position"] is None


def test_unreadable_previous_is_marked_error():
    calls = []

    def get_thesis(ticker, strict=False):
        calls.append(strict)
        if strict:
            raise RuntimeError("pool")
        return {"original": ORIGINAL, "latest": None}
    with patch("db.update_latest_thesis"), patch("db.compute_no_near_term_catalyst", return_value=True), \
         patch("db.get_thesis", side_effect=get_thesis), patch("db.get_open_outcomes", return_value=[]), \
         patch("db.get_position_stop", return_value=None), \
         patch("db.log_activity_direct", return_value=True) as act, \
         patch("agents.setup_gates.check_legacy_position_health"):
        crew._capture_latest_thesis("NVDA", RESULT)
    extra = act.call_args.args[4]
    assert extra["previous_read"] == "error" and extra["previous_conviction"] is None


def test_first_review_previous_is_the_entry_snapshot():
    extra = _capture({"original": ORIGINAL, "latest": dict(ORIGINAL)}).call_args.args[4]
    assert extra["previous_is_entry"] is True and extra["conviction_delta_vs_previous"] == -2
    extra = _capture({"original": ORIGINAL, "latest": {"as_of": "2026-11-01T14:00:00+00:00",
                                                       "conviction": 6}}).call_args.args[4]
    assert extra["previous_is_entry"] is False
