"""weekly_readouts: one file of the forward readouts, each section in its own
try, newest 12 files kept, one activity row; reads only, sends nothing."""
import inspect
import os
from datetime import date, datetime, timedelta, timezone

import pytest

import db
import tasks
import weekly_readouts as wr

NY = wr.NY
NOW = datetime(2026, 11, 27, 17, 37, tzinfo=NY)                   # Fri 17:37 ET
SINCE = date(2026, 10, 2)                                         # 8 weeks before


def _sessions(start=date(2026, 9, 28), n=50):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DATES = _sessions()


def _bars(drift):
    return [(d, 100.0 * (1 + drift) ** i) for i, d in enumerate(DATES)]


def _ts(d, hour=14):
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


@pytest.fixture
def fixtures(monkeypatch):
    fr = wr._load("forward_readout")
    vs = wr._load("vol_shadow_readout")
    fl = wr._load("fill_lag_readout")
    calls = []
    bars = {"SPY": _bars(0.001), "AMD": _bars(0.004), "MSFT": _bars(-0.002)}

    def fr_fetch(kinds, since, env, by_conviction=False):
        calls.append(("forward", tuple(kinds), since, env, by_conviction))
        if by_conviction:
            rows = [("AMD", "OPEN", _ts(DATES[5]), {"final_conviction": 7, "confidence": 0.7,
                                                   "logprobs_expected": 7.2}),
                    ("MSFT", "ADD", _ts(DATES[6]), {"final_conviction": 8, "confidence": 0.8,
                                                   "logprobs_expected": None})]
        else:
            rows = [("AMD", "OPEN", _ts(DATES[5]), {}),
                    ("MSFT", "ENTRY_GATE_BLOCKED", _ts(DATES[6]),
                     {"entry_gate": {"trend": {"blocked": True}}}),
                    ("AMD", "TRIM_EXIT_RULE", _ts(DATES[7]), {"held_below_both_sma": True}),
                    ("MSFT", "TRIM_EXIT_RULE", _ts(DATES[7]), {"held_below_both_sma": None}),
                    ("MSFT", "STOPOUT_REENTRY_SHADOW", _ts(DATES[8]), {})]
            rows = [r for r in rows if r[1] in kinds]
        return rows, bars

    def vs_fetch(env):
        calls.append(("vol", env))
        hl = {t: [{"date": str(d), "high": c * 1.01, "low": c * 0.99, "close": c} for d, c in b]
              for t, b in bars.items()}
        fills = [{"symbol": "AMD", "qty": 3.0, "price": 120.0, "ts": _ts(DATES[30]), "env": "live",
                  "approval_id": None, "buys_before_qty": 0.0, "sells_before_qty": 0.0},
                 {"symbol": "AMD", "qty": 3.0, "price": 101.0, "ts": _ts(date(2026, 9, 1)),
                  "env": "live", "approval_id": None, "buys_before_qty": 0.0,
                  "sells_before_qty": 0.0}]                                     # before the window
        return fills, hl, {"live": [(_ts(DATES[0]), 100000.0)]}, []

    def fl_fetch(since, env):
        calls.append(("fill_lag", since, env))
        return ([("AMD", _ts(DATES[20], 15), _ts(DATES[20], 15))],
                [("AMD", _ts(DATES[20], 16))], [], [("AMD", _ts(DATES[20], 15) + timedelta(minutes=5),
                                                   "HOLD")], [])

    monkeypatch.setattr(fr, "_fetch", fr_fetch)
    monkeypatch.setattr(vs, "_fetch", vs_fetch)
    def pl_fetch(since, env):
        calls.append(("plan_levels", since, env))
        hl = {t: [{"date": d, "high": c * 1.01, "low": c * 0.99, "close": c} for d, c in b]
              for t, b in bars.items()}
        plan = {"id": 1, "ticker": "AMD", "status": "active", "source": "analyst",
                "created_at": _ts(DATES[5], 13), "closed_at": None, "entry_price": 100.0,
                "structural_stop": 90.0, "t1": 104.0, "t2": 130.0, "add_trigger": 105.0,
                "horizon_end": DATES[40], "levels_source": "ladder"}
        return [plan], [], hl

    pl = wr._load("plan_level_readout")
    monkeypatch.setattr(fl, "_fetch", fl_fetch)
    monkeypatch.setattr(pl, "_fetch", pl_fetch)

    def ef_fetch(sessions, env):
        calls.append(("entry_funnel", tuple(sessions), env))
        s = sessions[-1]
        ts = datetime(s.year, s.month, s.day, 15, tzinfo=timezone.utc)
        return ([{"ts": ts, "action": "OPEN"}],
                [{"ts": ts, "kind": "ENTRY_GATE_BLOCKED",
                  "decision": {"verdict_action": "OPEN", "entry_gate": {"trend": {"blocked": True}}}}],
                [], [], [])
    monkeypatch.setattr(wr._load("entry_funnel"), "_fetch", ef_fetch)

    def so_fetch(since, env):
        calls.append(("shadow_outcomes", since, env))
        return {**_EMPTY_SO, "bars": bars,
                "dips": [("AMD", _ts(DATES[20], 15), True)]}
    monkeypatch.setattr(wr._load("shadow_outcomes"), "_fetch", so_fetch)
    return {"calls": calls, "fl": fl, "mp": monkeypatch}


_EMPTY_SO = {"trim": [], "events": [], "approvals": [], "routing": [], "loser": [], "guard": [],
             "scans": [], "dips": [], "bars": {}}


def test_build_runs_every_section_over_eight_weeks(fixtures):
    text, statuses = wr.build(NOW, "live", "abc1234")
    assert statuses == {k: "ok" for k, _, _ in wr.SECTIONS}
    fwd = [c for c in fixtures["calls"] if c[0] == "forward"]
    assert {c[2] for c in fwd} == {SINCE} and {c[3] for c in fwd} == {"live"}
    assert ("forward", ("ENTRY_GATE_BLOCKED", "OPEN", "ADD"), SINCE, "live", False) in fwd
    assert ("forward", ("OPEN", "ADD"), SINCE, "live", True) in fwd
    assert ("forward", wr.SHADOW_KINDS, SINCE, "live", False) in fwd
    assert ("vol", "all") in fixtures["calls"] and ("fill_lag", SINCE, "live") in fixtures["calls"]
    assert ("plan_levels", SINCE, "live") in fixtures["calls"]
    assert "### Compiled plan levels readout" in text and "t1 crossed (in force, intraday)" in text
    for _, title, _ in wr.SECTIONS:
        assert f"\n## {title}\n" in text
    assert text.startswith("# Weekly readouts (live), 2026-11-27")
    assert "### Forward-return readout" in text                            # script headings demoted
    assert "TRIM_EXIT_RULE (held_below_both_sma true)" in text
    assert "TRIM_EXIT_RULE (held_below_both_sma unknown)" in text
    assert "Fills: 1;" in text                                              # the Sept fill is out
    assert "### Entry funnel (live)" in text and "| entry_gate:trend |" in text
    assert [c[0] for c in fixtures["calls"]].count("entry_funnel") == 1
    assert ("shadow_outcomes", SINCE, "live") in fixtures["calls"]
    assert "### Drop watch by would_review" in text and "DIP_WATCH_SHADOW would_review true" in text
    assert "section failed" not in text


def test_one_failing_section_does_not_stop_the_rest(fixtures):
    fixtures["mp"].setattr(fixtures["fl"], "_fetch",
                           lambda *a, **k: (_ for _ in ()).throw(TimeoutError("db")))
    text, statuses = wr.build(NOW, "live", "abc1234")
    assert statuses["fill_lag"] == "failed: TimeoutError"
    assert all(v == "ok" for k, v in statuses.items() if k != "fill_lag")
    assert "## Fill lag on broker exits\n\nsection failed: TimeoutError" in text
    assert "fill_lag failed: TimeoutError" in text.splitlines()[2]


def test_run_writes_one_file_prunes_and_logs_one_row(fixtures, tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr(db, "log_activity",
                        lambda stage, ticker, msg, sent="neutral", extra=None: rows.append(
                            (stage, ticker, msg, sent, extra)))
    for i in range(14):
        (tmp_path / f"weekly-2026-0{1 + i // 9}-{10 + i % 9:02d}.md").write_text("old")
    (tmp_path / "notes.md").write_text("keep me")
    out = wr.run(now=NOW, out_dir=str(tmp_path))
    assert out["path"] == str(tmp_path / "weekly-2026-11-27.md")
    names = sorted(os.listdir(tmp_path))
    weekly = [n for n in names if n.startswith("weekly-")]
    assert len(weekly) == 12 and weekly[-1] == "weekly-2026-11-27.md"
    assert "notes.md" in names and not any(n.endswith(".tmp") for n in names)
    assert len(out["removed"]) == 3
    assert len(rows) == 1
    stage, ticker, msg, sent, extra = rows[0]
    assert stage == "weekly_readouts" and sent == "neutral" and "8 of 8 sections ok" in msg
    assert extra["sections"] == out["sections"] and extra["weeks"] == 8
    assert extra["path"].endswith("weekly-2026-11-27.md")


def test_run_marks_a_failed_section_on_the_row(fixtures, tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr(db, "log_activity", lambda *a, **k: rows.append((a, k)))
    fixtures["mp"].setattr(fixtures["fl"], "_fetch", lambda *a, **k: (_ for _ in ()).throw(KeyError()))
    out = wr.run(now=NOW, out_dir=str(tmp_path))
    (stage, _, msg, sent), kw = rows[0]
    assert sent == "alert" and "failed: fill_lag" in msg
    assert kw["extra"]["sections"]["fill_lag"] == "failed: KeyError"
    assert os.path.exists(out["path"])


def test_run_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert wr.run(now=NOW, out_dir=str(tmp_path)) is None
    monkeypatch.setattr(wr, "build", lambda *a, **k: ("x", {"gate": "ok"}))
    monkeypatch.setattr(db, "log_activity", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert wr.run(now=NOW, out_dir=str(tmp_path))["sections"] == {"gate": "ok"}


def test_prune_keeps_only_newest_by_name(tmp_path):
    for d in ("2026-01-02", "2026-03-04", "2025-12-31"):
        (tmp_path / f"weekly-{d}.md").write_text("x")
    (tmp_path / "weekly-notes.md").write_text("x")
    assert wr.prune(str(tmp_path), keep=2) == ["weekly-2025-12-31.md"]
    assert sorted(os.listdir(tmp_path)) == ["weekly-2026-01-02.md", "weekly-2026-03-04.md",
                                            "weekly-notes.md"]


def test_shadow_label():
    assert wr.shadow_label("ENTRY_QUALITY_SHADOW", {"held_below_both_sma": True}) == "ENTRY_QUALITY_SHADOW"
    assert wr.shadow_label("TRIM_EXIT_RULE", {"held_below_both_sma": False}).endswith("false)")
    assert wr.shadow_label("TRIM_EXIT_RULE", None).endswith("unknown)")


def test_reads_only_and_sends_nothing():
    src = inspect.getsource(wr)
    for word in ("telegram", "send_message", "send_alert", "notify", "smtp", "requests.",
                 "send_task(", "apply_async(", ".delay(", "INSERT", "UPDATE ", "DELETE FROM",
                 "commit(", "set_state("):
        assert word not in src, word
    sections = "".join(inspect.getsource(fn) for _, _, fn in wr.SECTIONS)
    assert "subprocess" not in sections and "_fetch(" in sections
    for name in ("forward_readout", "vol_shadow_readout", "fill_lag_readout", "plan_level_readout",
                 "entry_funnel", "shadow_outcomes"):
        fetch = inspect.getsource(wr._load(name)._fetch)
        assert "SET TRANSACTION READ ONLY" in fetch and "rollback()" in fetch


def test_beat_runs_friday_after_the_price_refresh_on_ops():
    entry = tasks.app.conf.beat_schedule["weekly-readouts"]
    assert entry["task"] == "tasks.weekly_readouts"
    sched = entry["schedule"]
    assert sched.day_of_week == {5} and sched.hour == {17} and sched.minute == {37}
    assert entry["options"]["queue"] == "ops" and entry["options"]["expires"] == 3600
    refresh = tasks.app.conf.beat_schedule["refresh-price-history"]["schedule"]
    assert refresh.hour == {16} and refresh.minute == {45}


def test_task_runs_as_a_singleton(monkeypatch):
    seen = []
    monkeypatch.setattr(tasks, "_run_singleton", lambda name, fn, max_hold_s=1800: seen.append(name) or fn())
    monkeypatch.setattr(wr, "run", lambda: seen.append("ran"))
    tasks.weekly_readouts.run()
    assert seen == ["weekly_readouts", "ran"]


def test_empty_window_still_writes_every_section(monkeypatch):
    fr, vs, fl = (wr._load(n) for n in ("forward_readout", "vol_shadow_readout", "fill_lag_readout"))
    monkeypatch.setattr(fr, "_fetch", lambda *a, **k: ([], {}))
    monkeypatch.setattr(vs, "_fetch", lambda env: ([], {}, {}, []))
    monkeypatch.setattr(fl, "_fetch", lambda since, env: ([], [], [], [], []))
    monkeypatch.setattr(wr._load("plan_level_readout"), "_fetch", lambda since, env: ([], [], {}))
    monkeypatch.setattr(wr._load("entry_funnel"), "_fetch", lambda sessions, env: ([], [], [], [], []))
    monkeypatch.setattr(wr._load("shadow_outcomes"), "_fetch", lambda since, env: dict(_EMPTY_SO))
    text, statuses = wr.build(NOW, "live", "abc1234")
    assert statuses == {k: "ok" for k, _, _ in wr.SECTIONS}, statuses
