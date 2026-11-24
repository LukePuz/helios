"""scripts/entry_funnel.py: per-session counts of OPEN/ADD verdicts through the
gates to fills on fixture rows, the --check line for the monitor cron, and
that the script only reads."""
import importlib.util
import inspect
import os
from datetime import date, datetime, timezone
from unittest.mock import patch

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "entry_funnel.py")
_spec = importlib.util.spec_from_file_location("entry_funnel", _PATH)
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

NY = ef.NY
MON, TUE, WED, THU, FRI = (date(2026, 11, d) for d in (16, 17, 18, 19, 20))
SESSIONS = [MON, TUE, WED, THU, FRI]


def _ts(d, hour=15, minute=0):
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=NY)


def _route(d, kind, action="OPEN", hour=15, **dec):
    return {"ts": _ts(d, hour), "kind": kind, "decision": {"verdict_action": action, **dec}}


def _gate(d, *halves, action="OPEN", path=None, approval_id=None):
    g = {h: {"blocked": True} for h in halves}
    if path:
        g.update(path=path, approval_id=approval_id)
    return _route(d, "ENTRY_GATE_BLOCKED", action, entry_gate=g)


def _appr(i, d, action="OPEN", status="APPROVED", is_test=False, al=None, rc=None, hour=15):
    return {"id": i, "ts": _ts(d, hour), "action": action, "status": status, "is_test": is_test,
            "add_loser_gate": al, "recheck_contradiction": rc}


# ------------------------------------------------------------ sessions

def test_session_of_rolls_weekends_and_holidays_forward_in_et():
    assert ef.session_of(_ts(FRI, 23)) == FRI
    assert ef.session_of(datetime(2026, 11, 21, 3, tzinfo=timezone.utc)) == FRI   # Fri 22:00 ET
    assert ef.session_of(_ts(date(2026, 11, 21))) == date(2026, 11, 23)          # Sat -> Mon
    assert ef.session_of(_ts(date(2026, 11, 26))) == date(2026, 11, 27)          # Thanksgiving
    assert ef.session_of(datetime(2026, 11, 17, 14)) == TUE                       # naive = UTC
    assert ef.session_of(None) is None


def test_sessions_until_and_default_until():
    assert ef.sessions_until(FRI, 5) == SESSIONS
    assert ef.sessions_until(date(2026, 11, 22), 2) == [THU, FRI]                # Sun -> Fri
    assert ef.sessions_until(date(2026, 11, 27), 3) == [date(2026, 11, 24), date(2026, 11, 25),
                                                         date(2026, 11, 27)]
    assert ef.default_until(_ts(FRI, 16, 5)) == FRI
    assert ef.default_until(_ts(FRI, 15, 59)) == THU                              # still trading
    assert ef.default_until(_ts(MON, 9)) == date(2026, 11, 13)
    assert ef.default_until(_ts(date(2026, 11, 22), 12)) == FRI


def test_window_covers_the_weekend_before_the_first_session():
    start, end = ef.window([MON, TUE])
    assert start == datetime(2026, 11, 14, 0, tzinfo=NY).astimezone(timezone.utc)   # Sat
    assert end == datetime(2026, 11, 18, 0, tzinfo=NY).astimezone(timezone.utc)


# ------------------------------------------------------------ funnel

def _fixture():
    verdicts = ([{"ts": _ts(MON), "action": "OPEN"}] * 3 + [{"ts": _ts(MON), "action": "ADD"}]
                + [{"ts": _ts(TUE), "action": "OPEN"}] * 4
                + [{"ts": _ts(TUE), "action": "HOLD"}])                  # not a buy verdict
    routes = [
        _gate(MON, "trend"), _gate(MON, "stopout", "trend"),
        _route(MON, "ROUTE_ZERO_SHARES", "ADD"),
        _route(MON, "STOPOUT_REENTRY_SHADOW"),                           # a shadow, not a stop
        _route(MON, "COOLDOWN_SKIPPED", "SELL"),                         # not a buy route
        _route(TUE, "ROUTE_MICRO_NAME_CAP", recheck_contradiction={"would_block": True}),
        _gate(TUE, action="ADD"),                                        # no half says blocked
        _gate(TUE, "stopout", path="deferred_retry", approval_id=2),     # expires approval 2
        _gate(TUE, "stopout", path="deferred_retry", approval_id=999),   # approval outside
    ]
    approvals = [
        _appr(1, MON, al={"would_block": True}),
        _appr(2, MON, status="EXPIRED"),
        _appr(3, TUE, action="ADD", status="PENDING", rc={"would_block": True}),
        _appr(4, TUE, status="FAILED"),
        _appr(5, TUE, is_test=True),                                     # excluded
        _appr(6, TUE, action="TRIM"),                                    # not a buy
    ]
    orders = [{"approval_id": 1, "filled_qty": 3.0}, {"approval_id": 4, "filled_qty": 0},
              {"approval_id": 4, "filled_qty": None}, {"approval_id": 5, "filled_qty": 1.0}]
    activity = [{"ts": _ts(WED), "extra": {"approval_id": 3, "add_loser_gate": {"would_block": True}}},
                {"ts": _ts(WED), "extra": {"approval_id": 77, "add_loser_gate": {"would_block": True}}},
                {"ts": _ts(WED), "extra": {"approval_id": 3, "add_loser_gate": {"would_block": False}}}]
    return verdicts, routes, approvals, orders, activity


def test_build_funnel_counts_each_stage():
    f = ef.build_funnel(SESSIONS, *_fixture())
    mon, tue, wed = f[MON], f[TUE], f[WED]
    assert dict(mon["verdicts"]) == {"OPEN": 3, "ADD": 1}
    assert mon["stopped_rows"] == 3
    assert dict(mon["stopped"]) == {"entry_gate:trend": 2, "entry_gate:stopout": 1,
                                    "ROUTE_ZERO_SHARES": 1}
    assert dict(mon["shadow_rows"]) == {"STOPOUT_REENTRY_SHADOW": 1}
    assert mon["routed"] == 2 and mon["retry_blocked"] == 1 and mon["passed"] == 1
    assert dict(mon["retry_gate"]) == {"stopout": 1}                     # on the approval's day
    assert mon["approved"] == 1 and mon["ordered"] == 1 and mon["filled"] == 1
    assert mon["add_loser_would_block"] == 1

    assert dict(tue["verdicts"]) == {"OPEN": 4}
    assert dict(tue["stopped"]) == {"ROUTE_MICRO_NAME_CAP": 1, "entry_gate:unknown": 1}
    assert tue["routed"] == 2 and tue["passed"] == 2                     # PENDING counts
    assert tue["retry_blocked"] == 1                                     # approval 999, row day
    assert dict(tue["status"]) == {"PENDING": 1, "FAILED": 1}
    assert tue["ordered"] == 1 and tue["filled"] == 0
    assert tue["recheck_would_block"] == 2                               # route row + approval
    assert wed["add_loser_would_block"] == 1                             # approval 77 not this env
    assert sum(f[s]["routed"] for s in SESSIONS) == 4


def test_rows_outside_the_sessions_are_ignored():
    rows = ([{"ts": _ts(date(2026, 11, 23)), "action": "OPEN"}], [], [], [], [])
    f = ef.build_funnel([THU, FRI], *rows)
    assert all(sum(f[s]["verdicts"].values()) == 0 for s in (THU, FRI))


def test_readout_has_counts_and_percentages_only():
    f = ef.build_funnel(SESSIONS, *_fixture())
    text = ef.readout(f, SESSIONS, "live", "abc1234")
    assert text.startswith("# Entry funnel (live), sessions 2026-11-16..2026-11-20")
    assert "| 2026-11-16 | 4 (3/1) | 3 | 2 | 1 | 1 | 25% | 1 | 1 | 1 |" in text
    assert "| all | 8 (7/1) | 5 | 4 | 2 | 3 | 38% | 1 | 2 | 1 |" in text
    assert "| entry_gate:trend | 2 | 0 | 0 | 0 | 0 | 2 | 25% |" in text
    assert "| retry:stopout | 1 | 1 | 0 | 0 | 0 | 2 | 25% |" in text
    assert "| add_loser would_block | 1 | 0 | 1 | 0 | 0 | 2 |" in text
    assert "| STOPOUT_REENTRY_SHADOW rows | 1 | 0 | 0 | 0 | 0 | 1 |" in text
    assert "$" not in text


def test_readout_with_no_rows():
    f = ef.build_funnel([FRI], [], [], [], [], [])
    text = ef.readout(f, [FRI], "live", "x")
    assert "| 2026-11-20 | 0 (0/0) | 0 | 0 | 0 | 0 | n/a |" in text and "No stop rows." in text


# ------------------------------------------------------------ --check

def test_check_fires_on_zero_passed_with_enough_verdicts():
    verdicts = [{"ts": _ts(THU), "action": "OPEN"}] * 2 + [{"ts": _ts(FRI), "action": "ADD"}]
    routes = [_gate(THU, "trend"), _gate(THU, "trend"), _gate(FRI, "stopout", action="ADD")]
    approvals = [_appr(1, WED)]                                          # earlier session passed
    f = ef.build_funnel(SESSIONS, verdicts, routes, approvals, [], [])
    line = ef.funnel_check(f, SESSIONS)
    assert line == ("ENTRY FUNNEL: 0 passed buys in the last 2 sessions (2026-11-19..2026-11-20) "
                    "on 3 OPEN/ADD verdicts; stops: entry_gate:trend 2, entry_gate:stopout 1")
    assert "\n" not in line


def test_check_quiet_with_a_passed_buy_or_few_verdicts():
    verdicts = [{"ts": _ts(FRI), "action": "OPEN"}] * 5
    f = ef.build_funnel(SESSIONS, verdicts, [], [_appr(1, FRI)], [], [])
    assert ef.funnel_check(f, SESSIONS) is None
    f = ef.build_funnel(SESSIONS, verdicts[:2], [], [], [], [])
    assert ef.funnel_check(f, SESSIONS) is None
    # a retry-blocked approval is not a passed buy
    f = ef.build_funnel(SESSIONS, verdicts, [_gate(FRI, "stopout", path="deferred_retry",
                                                   approval_id=1)], [_appr(1, FRI)], [], [])
    assert "retry:stopout 1" in ef.funnel_check(f, SESSIONS)


def test_main_check_exit_codes(capsys):
    rows = ([{"ts": _ts(FRI), "action": "OPEN"}] * 3, [_gate(FRI, "trend")] * 3, [], [], [])
    with patch.object(ef, "_fetch", return_value=rows) as fetch:
        assert ef.main(["--check", "--until", "2026-11-20"]) == 1
    assert fetch.call_args.args == (ef.sessions_until(FRI, 5), "live")
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.startswith("ENTRY FUNNEL: 0 passed buys")
    with patch.object(ef, "_fetch", return_value=([], [], [], [], [])):
        assert ef.main(["--check", "--until", "2026-11-20"]) == 0
    assert capsys.readouterr().out == ""
    with patch.object(ef, "_fetch", side_effect=RuntimeError("pool")):
        assert ef.main(["--check", "--until", "2026-11-20"]) == 2
    assert capsys.readouterr().out == "ENTRY FUNNEL: check failed, the read raised RuntimeError\n"


def test_main_readout_writes_out_file(tmp_path, capsys):
    out = tmp_path / "f.md"
    with patch.object(ef, "_fetch", return_value=([], [], [], [], [])):
        assert ef.main(["--until", "2026-11-20", "--sessions", "3", "--out", str(out)]) == 0
    assert out.read_text().startswith("# Entry funnel (live), sessions 2026-11-18..2026-11-20")


# ------------------------------------------------------------ read-only

def test_reads_only():
    src = inspect.getsource(ef)
    for word in ("INSERT", "UPDATE ", "DELETE FROM", "commit(", "set_state(", "log_activity",
                 "send_task(", "apply_async(", ".delay(", "alerts", "telegram", "import ibkr",
                 "requests."):
        assert word not in src, word
    fetch = inspect.getsource(ef._fetch)
    assert "SET TRANSACTION READ ONLY" in fetch and "statement_timeout" in fetch
    assert "conn.rollback()" in fetch
    assert fetch.count("trading_env = %s") == 4        # signals x2, approvals, orders
