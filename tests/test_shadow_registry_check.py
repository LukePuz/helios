"""scripts/shadow_registry_check.py on fake rows: counts since the load date
and in the last sessions, the newest row, the switch as loaded, the silent,
alarm and no-trigger flags by expect_rows, and the read-only transaction.
No DB."""
import importlib.util
import json
import os
import types
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "shadow_registry_check.py")
_spec = importlib.util.spec_from_file_location("shadow_registry_check", _PATH)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

NY = ZoneInfo("America/New_York")
LOAD = datetime(2026, 11, 23, 19, 0, tzinfo=NY)
# Fri 2026-12-04 12:00 ET: the last 5 sessions are Mon 11-30 .. Fri 12-04.
NOW = datetime(2026, 12, 4, 17, 0, tzinfo=timezone.utc)


def _et(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=NY).astimezone(timezone.utc)


def _cfg(**kw):
    return types.SimpleNamespace(**kw)


ALL_ON = {st["switch"]: True for st in src.STAGES if st["switch"]}
BY_STAGE = {st["stage"]: st for st in src.STAGES}


# ── sessions and dates ──────────────────────────────────────────────────────

def test_window_is_the_last_five_sessions_including_today():
    sessions, start = src.session_window(NOW)
    assert sessions == [date(2026, 11, 30), date(2026, 12, 1), date(2026, 12, 2),
                        date(2026, 12, 3), date(2026, 12, 4)]
    assert start == datetime(2026, 11, 28, 0, 0, tzinfo=NY).astimezone(timezone.utc)


def test_window_skips_the_thanksgiving_holiday_and_weekend():
    sat = datetime(2026, 11, 28, 15, 0, tzinfo=timezone.utc)
    sessions, _ = src.session_window(sat)
    assert date(2026, 11, 26) not in sessions and sessions[-1] == date(2026, 11, 27)
    assert len(sessions) == 5


def test_parse_load():
    assert src.parse_load(None) == datetime(2026, 11, 23, 0, 0, tzinfo=NY)
    assert src.parse_load("2026-11-23 19:23") == datetime(2026, 11, 23, 19, 23, tzinfo=NY)


# ── switches ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,records,shown", [
    (True, True, "True"), (False, False, "False"), ("shadow", True, "shadow"),
    ("enforce", True, "enforce"), ("off", False, "off"), ("OFF", False, "OFF")])
def test_switch_state(value, records, shown):
    assert src.switch_state(_cfg(X=value), "X") == {"value": shown, "records": records}


def test_switch_absent_from_this_clone_is_not_recording():
    assert src.switch_state(_cfg(), "X") == {"value": "absent", "records": False}


def test_no_switch_records():
    assert src.switch_state(_cfg(), None) == {"value": "no switch", "records": True}


def test_no_switch_with_its_writer_absent_is_not_recording():
    assert src.switch_state(_cfg(), None, False) == {"value": "absent", "records": False}


def test_writer_present(tmp_path):
    (tmp_path / "tasks.py").write_text('db.log_activity_direct("fundamentals_changes", "")\n')
    assert src.writer_present(str(tmp_path), ("tasks.py", '"fundamentals_changes"')) is True
    assert src.writer_present(str(tmp_path), ("tasks.py", '"feed_drops"')) is False
    assert src.writer_present(str(tmp_path), ("missing.py", '"x"')) is False
    assert src.writer_present(str(tmp_path), None) is True


def test_every_named_writer_is_in_this_clone():
    root = os.path.dirname(os.path.dirname(_PATH))
    for st in src.STAGES:
        if st["switch"] is None:
            assert st.get("writer"), st["stage"]
            assert src.writer_present(root, st["writer"]), st["stage"]
        else:
            assert "writer" not in st, st["stage"]


def test_a_clone_without_the_writer_is_not_silent(tmp_path):
    _, start = src.session_window(NOW)
    lines = src.build({"fundamentals_changes": _NONE}, _cfg(**ALL_ON), LOAD, start,
                      str(tmp_path))
    ln = next(x for x in lines if x["stage"] == "fundamentals_changes")
    assert ln["switch_value"] == "absent"
    assert ln["records"] is False
    assert ln["silent"] is False
    text = src.render_text(lines, _meta())
    assert "no switch, writer absent (not recording)" in text


# ── counting ────────────────────────────────────────────────────────────────

def _one(stage_name, per_source, cfg=None, trigger=None):
    _, start = src.session_window(NOW)
    results = {stage_name: per_source}
    if trigger is not None:
        results[src.TRIGGERS_KEY] = {stage_name: trigger}
    lines = src.build(results, cfg or _cfg(**ALL_ON), LOAD, start)
    return next(ln for ln in lines if ln["stage"] == stage_name)


_NONE = [{"times": [], "newest": None}]


def test_counts_since_load_and_in_the_last_sessions():
    times = [_et(2026, 11, 23, 18), _et(2026, 11, 24), _et(2026, 11, 27),
             _et(2026, 11, 30), _et(2026, 12, 4, 10)]
    ln = _one("add_lot_r", [{"times": times, "newest": times[-1]}])
    assert ln["rows_since_load"] == 4            # the 18:00 row predates the 19:00 load
    assert ln["rows_last_sessions"] == 2         # 11-30 and 12-04
    assert ln["newest"] == "2026-12-04T10:00-05:00"
    assert ln["silent"] is False and ln["records"] is True


def test_weekend_rows_before_the_first_session_count_toward_it():
    sat = _et(2026, 11, 28)
    ln = _one("add_lot_r", [{"times": [sat], "newest": sat}])
    assert ln["rows_last_sessions"] == 1


def test_always_stage_switch_on_and_no_row_in_the_sessions_is_silent():
    assert BY_STAGE["story_dedup"]["expect_rows"] == "always"
    old = _et(2026, 11, 24)
    ln = _one("story_dedup", [{"times": [old], "newest": old}])
    assert ln["rows_since_load"] == 1 and ln["rows_last_sessions"] == 0
    assert ln["silent"] is True
    assert ln["alarm"] is False
    assert ln["no_trigger_yet"] is False


def test_always_stage_with_no_switch_is_silent_without_rows():
    assert BY_STAGE["fundamentals_changes"]["switch"] is None
    ln = _one("fundamentals_changes", _NONE)
    assert ln["switch_value"] == "no switch"
    assert ln["records"] is True
    assert ln["silent"] is True


# ── conditional and alarm stages ────────────────────────────────────────────

def test_conditional_stage_with_no_trigger_rows_is_not_silent():
    assert BY_STAGE["book_heat"]["expect_rows"] == "conditional"
    ln = _one("book_heat", _NONE, trigger=[])
    assert ln["silent"] is False
    assert ln["no_trigger_yet"] is True
    assert ln["trigger_rows"] == 0


def test_conditional_stage_whose_trigger_fired_without_its_rows_is_silent():
    ln = _one("book_heat", _NONE, trigger=[_et(2026, 12, 2)])
    assert ln["trigger_rows"] == 1
    assert ln["silent"] is True
    assert ln["no_trigger_yet"] is False


def test_conditional_trigger_before_the_span_does_not_count():
    # The span is max(load, first session): 11-24's approval predates 11-28 00:00 ET.
    ln = _one("book_heat", _NONE, trigger=[_et(2026, 11, 24)])
    assert ln["trigger_rows"] == 0
    assert ln["silent"] is False
    assert ln["no_trigger_yet"] is True


def test_conditional_stage_with_its_own_row_after_the_trigger_is_not_silent():
    t = _et(2026, 12, 2, 13)
    ln = _one("book_heat", [{"times": [t], "newest": t}], trigger=[_et(2026, 12, 2)])
    assert ln["silent"] is False
    assert ln["no_trigger_yet"] is False


def test_conditional_stage_switched_off_is_not_silent_when_triggered():
    cfg = _cfg(**{**ALL_ON, "BOOK_HEAT_SHADOW_ENABLED": False})
    ln = _one("book_heat", _NONE, cfg, trigger=[_et(2026, 12, 2)])
    assert ln["silent"] is False


def test_conditional_stage_without_a_named_trigger_is_never_silent():
    assert "trigger" not in BY_STAGE["earnings_event_record"]
    ln = _one("earnings_event_record", _NONE, trigger=[_et(2026, 12, 2)])
    assert ln["trigger_rows"] is None           # an unnamed trigger is not read
    assert ln["silent"] is False
    assert ln["no_trigger_yet"] is True


def test_alarm_stage_with_no_rows_is_quiet():
    assert BY_STAGE["hwm_guard"]["expect_rows"] == "alarm"
    ln = _one("hwm_guard", _NONE)
    assert ln["silent"] is False
    assert ln["alarm"] is False
    assert ln["no_trigger_yet"] is False


def test_alarm_stage_with_a_row_in_the_sessions_is_flagged():
    t = _et(2026, 12, 3)
    ln = _one("risk_subcheck_fail", [{"times": [t], "newest": t}])
    assert ln["alarm"] is True
    assert ln["silent"] is False


def test_alarm_stage_row_before_the_sessions_is_not_flagged():
    t = _et(2026, 11, 24)
    ln = _one("hwm_guard", [{"times": [t], "newest": t}])
    assert ln["rows_since_load"] == 1
    assert ln["alarm"] is False


def test_switch_off_with_no_rows_is_not_silent():
    cfg = _cfg(**{**ALL_ON, "VERDICT_VOTE_SHADOW_ENABLED": False})
    ln = _one("verdict_vote_shadow", [{"times": [], "newest": None}], cfg)
    assert ln["records"] is False and ln["silent"] is False and ln["newest"] is None


def test_absent_switch_with_no_rows_is_not_silent():
    cfg = _cfg(**{k: v for k, v in ALL_ON.items() if k != "STOP_GUARD_MODE"})
    ln = _one("stop_above_market_guard", [{"times": [], "newest": None}], cfg)
    assert ln["switch_value"] == "absent" and ln["silent"] is False


def test_two_sources_are_summed_and_the_newest_is_the_later():
    a, b = _et(2026, 12, 1), _et(2026, 12, 3)
    ln = _one("recheck_contradiction", [{"times": [a], "newest": a},
                                        {"times": [b, b], "newest": b}])
    assert ln["rows_since_load"] == 3 and ln["rows_last_sessions"] == 3
    assert ln["newest"].startswith("2026-12-03")


def test_naive_times_are_utc():
    naive = datetime(2026, 12, 1, 17, 0)        # 12:00 ET
    ln = _one("add_lot_r", [{"times": [naive], "newest": naive}])
    assert ln["rows_last_sessions"] == 1 and ln["newest"] == "2026-12-01T12:00-05:00"


def test_a_failed_read_is_shown_not_counted():
    _, start = src.session_window(NOW)
    lines = src.build({"add_lot_r": RuntimeError("timeout")}, _cfg(**ALL_ON), LOAD, start)
    by = {ln["stage"]: ln for ln in lines}
    assert by["add_lot_r"]["read_error"] == "RuntimeError" and by["add_lot_r"]["silent"] is None
    assert by["book_heat"]["read_error"] == "not read"
    assert by["book_heat"]["alarm"] is None
    assert len(lines) == len(src.STAGES)


def test_every_batch_card_is_listed_once():
    names = [st["stage"] for st in src.STAGES]
    assert len(names) == len(set(names)) == 39
    assert {st["card"] for st in src.STAGES} >= {"held-exit candidates", "book heat",
                                                 "above-market stop guard",
                                                 "self-consistency vote on routed entries",
                                                 "book overnight gap risk",
                                                 "quote a market order was sent on",
                                                 "same-day consensus on OPEN/ADD routing"}


def test_every_switch_is_in_config():
    import config
    for st in src.STAGES:
        if st["switch"] is not None:
            assert hasattr(config, st["switch"]), st["switch"]


def test_every_stage_has_a_known_expect_and_readable_sources():
    for st in src.STAGES:
        assert st["expect_rows"] in src.EXPECT, st["stage"]
        for table, _ in st["sources"]:
            assert table in src._SQL_BASE, st["stage"]
        if "trigger" in st:
            assert st["expect_rows"] == "conditional", st["stage"]
            assert st["trigger"][0] in src._SQL_BASE, st["stage"]


def test_alarm_stages_match_batch_verify():
    path = os.path.join(os.path.dirname(_PATH), "batch_verify.py")
    spec = importlib.util.spec_from_file_location("batch_verify_for_registry", path)
    bv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bv)
    alarm = {st["stage"] for st in src.STAGES if st["expect_rows"] == "alarm"}
    assert alarm == set(bv.ALARM_STAGES)


# ── output ──────────────────────────────────────────────────────────────────

def _meta():
    sessions, start = src.session_window(NOW)
    return {"env": "paper", "clone": "abc1234", "generated": "2026-12-04T12:00-05:00",
            "load": LOAD.isoformat(timespec="minutes"),
            "sessions": [d.isoformat() for d in sessions],
            "window_start": start.astimezone(NY).isoformat(timespec="minutes")}


def test_text_marks_silent_and_not_recording():
    _, start = src.session_window(NOW)
    cfg = _cfg(**{**ALL_ON, "VERDICT_VOTE_SHADOW_ENABLED": False})
    results = {st["stage"]: [{"times": [], "newest": None}] for st in src.STAGES}
    lines = src.build(results, cfg, LOAD, start)
    text = src.render_text(lines, _meta())
    assert "VERDICT_VOTE_SHADOW_ENABLED=False (not recording)" in text
    always = [st["stage"] for st in src.STAGES if st["expect_rows"] == "always"]
    assert len(always) == 9
    assert f"{len(always)} silent, 0 alarm of 39 stages." in text
    assert "no trigger yet" in text
    assert "no switch" in text
    assert "$" not in text


def test_text_marks_alarm_and_a_triggered_silent_stage():
    _, start = src.session_window(NOW)
    t = _et(2026, 12, 3)
    results = {st["stage"]: [{"times": [t], "newest": t}] for st in src.STAGES}
    results["hwm_guard"] = [{"times": [t], "newest": t}]
    results["book_heat"] = _NONE
    results[src.TRIGGERS_KEY] = {"book_heat": [t]}
    lines = src.build(results, _cfg(**ALL_ON), LOAD, start)
    text = src.render_text(lines, _meta())
    by = {ln["stage"]: ln for ln in lines}
    assert by["book_heat"]["silent"] is True
    assert by["hwm_guard"]["alarm"] is True
    assert by["risk_subcheck_fail"]["alarm"] is True
    assert "1 silent, 2 alarm of 39 stages." in text


def test_json_is_serialisable():
    _, start = src.session_window(NOW)
    t = _et(2026, 12, 2)
    lines = src.build({"add_lot_r": [{"times": [t], "newest": t}]}, _cfg(**ALL_ON), LOAD, start)
    json.dumps({"meta": _meta(), "stages": lines})


# ── the read ────────────────────────────────────────────────────────────────

class _Cur:
    def __init__(self, conn):
        self.conn = conn
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.conn.sql.append(sql)
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("statement timeout")
        self._last = sql

    def fetchall(self):
        return [(t,) for t in self.conn.rows.get(self._key(), [])]

    def fetchone(self):
        rows = self.conn.rows.get(self._key(), [])
        return (max(rows) if rows else None,)

    def _key(self):
        for k in self.conn.rows:
            if k in self._last:
                return k
        return None


class _Conn:
    def __init__(self, rows=None, fail_on=None):
        self.rows, self.fail_on, self.sql, self.rolled_back = rows or {}, fail_on, [], False

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled_back = True

    def commit(self):
        raise AssertionError("never commits")


def test_fetch_is_read_only_and_rolled_back():
    t = _et(2026, 12, 2)
    conn = _Conn(rows={"stage = 'add_lot_r'": [t]})
    out = src.fetch(conn, "paper", LOAD)
    assert conn.sql[0] == "SET TRANSACTION READ ONLY" and "statement_timeout" in conn.sql[1]
    assert conn.rolled_back
    assert out["add_lot_r"] == [{"times": [t], "newest": t}]
    assert all(s.lstrip().upper().startswith(("SELECT", "SET")) for s in conn.sql)
    assert len(out) == len(src.STAGES) + 1          # plus the triggers


def test_fetch_reads_each_trigger_once_and_hands_it_to_every_stage():
    t = _et(2026, 12, 2)
    conn = _Conn(rows={"action IN ('OPEN', 'ADD')": [t]})
    out = src.fetch(conn, "paper", LOAD)
    users = [st["stage"] for st in src.STAGES if st.get("trigger") == src._BUY_APPROVALS]
    assert len(users) >= 2
    for name in users:
        assert out[src.TRIGGERS_KEY][name] == [t]
    assert sum(1 for s in conn.sql if "action IN ('OPEN', 'ADD')" in s) == 1
    assert set(out[src.TRIGGERS_KEY]) == {st["stage"] for st in src.STAGES if st.get("trigger")}


def test_fetch_env_filters():
    conn = _Conn()
    src.fetch(conn, "paper", LOAD)
    sig = [s for s in conn.sql if "FROM signals" in s]
    apr = [s for s in conn.sql if "FROM pending_approvals" in s]
    act = [s for s in conn.sql if "FROM activity_feed" in s]
    ords = [s for s in conn.sql if "FROM orders" in s]
    trd = [s for s in conn.sql if "FROM trades" in s]
    assert ords and all("trading_env = %(env)s" in s for s in ords)
    assert trd and all("trading_env = %(env)s" in s for s in trd)
    assert sig and all("trading_env = %(env)s" in s for s in sig)
    assert apr and all("is_test" in s and "trading_env = %(env)s" in s for s in apr)
    assert act and all("extra_json->>'env'" in s for s in act)


def test_a_failed_statement_stops_the_reads_and_still_rolls_back():
    conn = _Conn(fail_on="risk_json ? 'rr_vs_stop'")
    out = src.fetch(conn, "paper", LOAD)
    assert isinstance(out["rr_vs_stop"], RuntimeError)
    assert "book_heat" not in out and conn.rolled_back


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert src.main([]) == 2
    assert "6543" in capsys.readouterr().err
