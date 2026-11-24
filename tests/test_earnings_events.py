"""Per-event earnings record (ingestion/earnings_events.py): pre rows when a
date is first seen, post rows when its actual appears, estimated two-date
windows, date revisions (a new row, never an overwrite), missing values as
None with a reason, the task wiring after the 17:30 fetch, and that nothing
the analyst or the order path reads names it. Fake source data only."""
import pathlib
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from ingestion import earnings_events as EE

ROOT = pathlib.Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
ENV = "live"


def _at(d: str, hour=17, minute=35):
    return datetime.combine(date.fromisoformat(d), datetime.min.time(), ET).replace(
        hour=hour, minute=minute)


def _hist(q, actual, est=None, diff=None, surprise=None):
    return {"quarter": q, "epsActual": actual, "epsEstimate": est, "epsDifference": diff,
            "surprisePercent": surprise}


HIST_JUL = [_hist("2026-07-31", 1.05, 1.01, 0.04, 0.0396), _hist("2026-04-30", 0.96, 0.93)]
HIST_OCT = [_hist("2026-10-31", 1.30, 1.25, 0.05, 0.04)] + HIST_JUL


def _rec(dates, history=HIST_JUL, eps_avg=1.25, rev_avg=54.9e9, q_end="2026-10-31",
         parts=None):
    """A consensus snapshot as fetch_consensus returns it (next date: the
    first future date; two dates: an estimated window)."""
    future = sorted(dates)
    return {
        "source": "yfinance", "status": "ok",
        "next_earnings_date": date.fromisoformat(future[0]) if future else None,
        "next_earnings_is_estimate": len(future) > 1 if future else None,
        "calendar_event": {"dates": future, "eps_avg": eps_avg, "eps_low": 1.1, "eps_high": 1.4,
                           "rev_avg": rev_avg, "rev_low": 52e9, "rev_high": 57e9},
        "eps_estimate": {"0q": {"avg": 1.25, "numberOfAnalysts": 31, "end_date": q_end}},
        "revenue_estimate": {"0q": {"avg": 54.9e9, "numberOfAnalysts": 29, "end_date": q_end}},
        "eps_history": history,
        "parts": parts or {"calendar": "ok", "eps_estimate": "ok", "revenue_estimate": "ok",
                           "eps_history": "ok"},
    }


def _run(rec, prior=(), written=None, on="2026-11-10"):
    """build_rows for NVDA on the ET day `on`, at 17:35."""
    written = set(written or ())
    for p in prior:
        written.add(p["shadow_key"])
    counts = {}
    rows = EE.build_rows(ENV, "NVDA", rec, list(prior), written, _at(on), on, counts)
    return rows, counts


# ---- pre rows ----

def test_a_first_seen_date_writes_one_pre_row():
    (row,), _ = _run(_rec(["2026-11-19"]))
    assert row["kind"] == "pre" and row["shadow_key"] == "live:NVDA:2026-11-19:pre"
    assert row["announce_date"] == "2026-11-19" and row["announce_date_is_estimate"] is False
    assert row["window_dates"] == ["2026-11-19"]
    assert row["consensus_eps"] == 1.25 and row["consensus_revenue"] == 54.9e9
    assert row["estimate_quarter_end"] == "2026-10-31"
    assert row["last_reported_quarter"] == "2026-07-31"
    assert row["time_of_day"] is None and row["missing"]["time_of_day"] == EE.TIME_OF_DAY_REASON
    assert row["revision"] is None and row["source"] == "yfinance"
    assert row["as_of"].startswith("2026-11-10T22:35")                  # UTC
    assert row["env"] == ENV and row["ticker"] == "NVDA"


def test_a_date_already_written_writes_nothing():
    (first,), _ = _run(_rec(["2026-11-19"]))
    rows, _ = _run(_rec(["2026-11-19"]), prior=[first], on="2026-11-11")
    assert rows == []


def test_an_estimated_two_date_window_is_one_row_on_its_first_date():
    """The source gives two dates (an estimated window): one pre row keyed on
    the first, flagged as an estimate, with both dates kept."""
    (row,), _ = _run(_rec(["2026-11-24", "2026-11-18"]))
    assert row["announce_date"] == "2026-11-18" and row["announce_date_is_estimate"] is True
    assert row["window_dates"] == ["2026-11-18", "2026-11-24"]
    assert row["shadow_key"] == "live:NVDA:2026-11-18:pre"


def test_a_revised_date_is_a_new_row_and_the_old_one_stays():
    (first,), _ = _run(_rec(["2026-11-18", "2026-11-24"]), on="2026-11-03")
    rows, _ = _run(_rec(["2026-11-20"]), prior=[first], on="2026-11-10")    # window firmed up
    (row,) = rows
    assert row["shadow_key"] == "live:NVDA:2026-11-20:pre"
    assert row["revision"] == {"previous_announce_date": "2026-11-18",
                               "previous_shadow_key": "live:NVDA:2026-11-18:pre",
                               "previous_is_estimate": True,
                               "previous_as_of": first["as_of"]}
    assert first["announce_date"] == "2026-11-18" and first["revision"] is None   # untouched


def test_a_revision_back_to_a_date_already_written_writes_nothing_and_is_counted():
    (a,), _ = _run(_rec(["2026-11-18"]), on="2026-11-03")
    (b,), _ = _run(_rec(["2026-11-20"]), prior=[a], on="2026-11-05")
    rows, counts = _run(_rec(["2026-11-18"]), prior=[a, b], on="2026-11-07")
    assert rows == [] and counts["revision_to_seen_date"] == 1


def test_the_next_quarter_date_the_same_evening_is_a_new_event_not_a_revision():
    """NVDA reports after the close on 11-19; at 17:35 the source already
    shows the next quarter's date, before its history has the actual."""
    (pre,), _ = _run(_rec(["2026-11-19"]), on="2026-11-10")
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_JUL, q_end="2027-01-31"), prior=[pre],
                   on="2026-11-19")
    (nxt,) = rows
    assert nxt["kind"] == "pre" and nxt["announce_date"] == "2027-02-25"
    assert nxt["revision"] is None
    # The next day the actual appears: the 11-19 event gets its post, the
    # February one (still ahead) does not.
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT, q_end="2027-01-31"),
                   prior=[pre, nxt], on="2026-11-20")
    (post,) = rows
    assert post["kind"] == "post" and post["announce_date"] == "2026-11-19"
    assert post["quarter"] == "2026-10-31"


# ---- post rows ----

def test_the_actual_appearing_writes_a_post_row_with_the_source_surprise():
    (pre,), _ = _run(_rec(["2026-11-19"]), on="2026-11-10")
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT, q_end="2027-01-31"), prior=[pre],
                   on="2026-11-20")
    post, nxt = rows
    assert post["shadow_key"] == "live:NVDA:2026-11-19:post"
    assert post["pre_shadow_key"] == pre["shadow_key"]
    assert post["quarter"] == "2026-10-31" and post["matched_by"] == "newest_after_last_reported"
    assert post["eps_actual"] == 1.30 and post["eps_estimate_source"] == 1.25
    assert post["eps_difference_source"] == 0.05 and post["surprise_percent_source"] == 0.04
    assert "not converted" in post["surprise_percent_units"]
    assert post["revenue_actual"] is None and post["revenue_surprise"] is None
    assert post["missing"]["revenue_actual"] == EE.REVENUE_ACTUAL_REASON
    assert nxt["kind"] == "pre" and nxt["last_reported_quarter"] == "2026-10-31"
    # Written once.
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT, q_end="2027-01-31"),
                   prior=[pre, post, nxt], on="2026-11-21")
    assert rows == []


def test_no_post_before_the_date_or_without_a_newer_quarter():
    (pre,), _ = _run(_rec(["2026-11-19"]), on="2026-11-10")
    rows, _ = _run(_rec(["2026-11-19"], history=HIST_OCT), prior=[pre], on="2026-11-18")
    assert rows == []                                                   # date not reached
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_JUL), prior=[pre], on="2026-11-20")
    assert [r["kind"] for r in rows] == ["pre"]                         # no actual yet


def test_a_revised_event_posts_once_on_its_latest_date():
    (a,), _ = _run(_rec(["2026-11-18", "2026-11-24"]), on="2026-11-03")
    (b,), _ = _run(_rec(["2026-11-19"]), prior=[a], on="2026-11-05")
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT), prior=[a, b], on="2026-11-20")
    posts = [r for r in rows if r["kind"] == "post"]
    assert len(posts) == 1 and posts[0]["announce_date"] == "2026-11-19"
    assert posts[0]["pre_announce_dates"] == ["2026-11-18", "2026-11-19"]


def test_a_quarter_already_claimed_is_not_claimed_again():
    """The next event's date passes before the source adds its actual: the
    newest quarter belongs to the earlier post, so no post yet."""
    (pre,), _ = _run(_rec(["2026-11-19"]), on="2026-11-10")
    post, nxt = _run(_rec(["2027-02-25"], history=HIST_OCT, q_end="2027-01-31"), prior=[pre],
                     on="2026-11-20")[0]
    rows, _ = _run(_rec(["2027-05-27"], history=HIST_OCT, q_end="2027-04-30"),
                   prior=[pre, post, nxt], on="2027-02-25")
    assert [r["kind"] for r in rows] == ["pre"] and rows[0]["revision"] is None


def test_without_a_last_reported_quarter_the_estimate_quarter_is_matched():
    (pre,), _ = _run(_rec(["2026-11-19"], history=None), on="2026-11-10")
    assert pre["last_reported_quarter"] is None and "last_reported_quarter" in pre["missing"]
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT), prior=[pre], on="2026-11-20")
    post = rows[0]
    assert post["kind"] == "post" and post["matched_by"] == "estimate_quarter_end"
    # With neither known, no post (nothing is guessed).
    (bare,), _ = _run(_rec(["2026-11-19"], history=None, q_end=None), on="2026-11-10")
    rows, _ = _run(_rec(["2027-02-25"], history=HIST_OCT), prior=[bare], on="2026-11-20")
    assert all(r["kind"] == "pre" for r in rows)


# ---- missing values ----

def test_missing_values_are_none_with_a_reason_never_filled_in():
    rec = _rec(["2026-11-19"], eps_avg=None, rev_avg=float("nan"))
    (row,), _ = _run(rec)
    assert row["consensus_eps"] is None and row["consensus_revenue"] is None
    assert row["trend_eps_avg"] == 1.25          # recorded separately, not used to fill
    assert "consensus_eps" in row["missing"] and "consensus_revenue" in row["missing"]
    rec = _rec(["2026-11-19"], parts={"calendar": "ok", "eps_estimate": "error:KeyError",
                                       "revenue_estimate": "ok", "eps_history": "ok"})
    rec["eps_estimate"] = None
    (row,), _ = _run(rec)
    assert row["trend_eps_avg"] is None
    assert "error:KeyError" in row["missing"]["trend_eps_avg"]


def test_no_date_writes_no_pre_row():
    rec = _rec([])
    rows, counts = _run(rec)
    assert rows == [] and counts["no_date"] == 1


# ---- record_events: reads, writes, budget, env ----

def _record(recs, prior=(), keys=None, write_ok=True, budget_s=90.0, clock=None):
    got = []

    def write(row):
        got.append(row)
        if isinstance(write_ok, Exception):
            raise write_ok
        return write_ok
    keys = {p["shadow_key"] for p in prior} if keys is None else keys
    kw = {} if clock is None else {"clock": clock}
    counts = EE.record_events(recs, ENV, read_prior=lambda: list(prior),
                              read_keys=lambda: set(keys), write=write,
                              now=_at("2026-11-10").astimezone(timezone.utc), budget_s=budget_s,
                              **kw)
    return counts, got


def test_record_events_writes_and_counts():
    counts, got = _record({"NVDA": _rec(["2026-11-19"]), "AMD": _rec([])})
    assert [r["ticker"] for r in got] == ["NVDA"]
    assert counts["pre"] == 1 and counts["no_date"] == 1 and counts["post"] == 0


def test_unreadable_earlier_rows_write_nothing():
    for prior, keys in ((None, set()), ((), None)):
        got = []
        counts = EE.record_events({"NVDA": _rec(["2026-11-19"])}, ENV,
                                  read_prior=lambda: prior, read_keys=lambda: keys,
                                  write=lambda r: got.append(r) or True)
        assert counts["read_failed"] is True and got == []


def test_a_failed_write_is_counted_and_never_raises():
    counts, _ = _record({"NVDA": _rec(["2026-11-19"])}, write_ok=False)
    assert counts["write_failed"] == 1 and counts["pre"] == 0
    counts, _ = _record({"NVDA": _rec(["2026-11-19"])}, write_ok=RuntimeError("db"))
    assert counts["write_failed"] == 1


def test_the_budget_stops_the_writes():
    ticks = iter([0, 0, 100, 100, 100])
    counts, got = _record({"NVDA": _rec(["2026-11-19"]), "AMD": _rec(["2026-11-04"])},
                          budget_s=60, clock=lambda: next(ticks, 100))
    assert len(got) == 1 and counts["not_written_budget"] == 1


def test_another_envs_rows_are_not_this_envs_events():
    (paper,), _ = _run(_rec(["2026-11-18"]))
    paper = dict(paper, env="paper", shadow_key="paper:NVDA:2026-11-18:pre")
    counts, got = _record({"NVDA": _rec(["2026-11-20"])}, prior=[paper])
    (row,) = got
    assert row["revision"] is None and row["shadow_key"].startswith("live:")


def test_empty_snapshots_read_nothing():
    def boom():
        raise AssertionError("read")
    assert EE.record_events({}, ENV, read_prior=boom, read_keys=boom)["pre"] == 0


# ---- the source snapshot and the task ----

def test_the_fetch_keeps_the_calendar_event_fields_without_storing_them(monkeypatch):
    from test_earnings_consensus import _full_ticker, TODAY
    from ingestion import earnings_consensus as source
    t = _full_ticker()
    t.calendar = {"Earnings Date": [date(2026, 11, 18), date(2026, 11, 24)],
                  "Earnings Average": 1.25, "Earnings Low": 1.1, "Earnings High": 1.4,
                  "Revenue Average": 54.9e9, "Revenue Low": float("nan"), "Revenue High": 57e9}
    rec = source.fetch_consensus("NVDA", yf_ticker=t, today=TODAY)
    ce = rec["calendar_event"]
    assert ce["dates"] == ["2026-11-18", "2026-11-24"] and ce["eps_avg"] == 1.25
    assert ce["rev_low"] is None                                        # NaN cleaned
    row = EE.pre_row(ENV, "NVDA", rec, _at("2026-11-03"))
    assert row["announce_date"] == "2026-11-18" and row["announce_date_is_estimate"] is True
    assert row["estimate_quarter_end"] == "2026-12-31"
    assert row["last_reported_quarter"] == "2026-09-30"
    import db
    seen = {}

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, args=None):
            seen["sql"], seen["args"] = sql, args
    from unittest.mock import MagicMock
    conn = MagicMock()
    conn.cursor.return_value = Cur()
    monkeypatch.setattr(db, "_connect_with_retry", lambda *a, **k: conn)
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    assert db.log_earnings_consensus("NVDA", rec)
    assert "calendar_event" not in seen["sql"]
    assert not any("Earnings" in str(a) for a in seen["args"])


def test_the_run_hands_each_fetched_snapshot_to_on_record(monkeypatch):
    from ingestion import earnings_consensus as source
    import knowledge_loader
    monkeypatch.setattr(knowledge_loader, "get_etf_tickers", lambda: {"QTUM"})
    seen = {}
    source.refresh_all_earnings_consensus(
        tickers=["NVDA", "QTUM", "AMD"], fetch=lambda t: _rec(["2026-11-19"]),
        write=lambda t, r: True, pause_s=0, on_record=seen.__setitem__)
    assert set(seen) == {"NVDA", "AMD"}                                 # not the ETF row
    # A failing callback does not stop the run.
    counts = source.refresh_all_earnings_consensus(
        tickers=["NVDA", "AMD"], fetch=lambda t: _rec(["2026-11-19"]),
        write=lambda t, r: True, pause_s=0,
        on_record=lambda t, r: (_ for _ in ()).throw(RuntimeError("x")))
    assert counts["written"] == 2


def test_the_task_records_events_from_the_same_snapshots(monkeypatch):
    import tasks
    from ingestion import earnings_consensus as source
    got = {}

    def run(**kw):
        kw["on_record"]("NVDA", _rec(["2026-11-19"]))
        return {"written": 1}
    monkeypatch.setattr(source, "refresh_all_earnings_consensus", run)
    monkeypatch.setattr(tasks, "_heartbeat_ok", lambda s, c: None)
    monkeypatch.setattr(EE, "record_events",
                        lambda recs, env, **kw: got.update(recs=recs, env=env, **kw) or {})
    tasks.refresh_earnings_consensus.run()
    assert set(got["recs"]) == {"NVDA"} and got["budget_s"] == tasks._EARNINGS_EVENTS_BUDGET_S
    assert got["budget_s"] < tasks._EARNINGS_CONSENSUS_DEADLINE_MARGIN_S
    # A failed fetch run records no events.
    got.clear()

    def fail(**kw):
        raise RuntimeError("yahoo")
    monkeypatch.setattr(source, "refresh_all_earnings_consensus", fail)
    monkeypatch.setattr(tasks, "_heartbeat_fail", lambda s, e: None)
    tasks.refresh_earnings_consensus.run()
    assert got == {}
    # A raising record step does not fail the task.
    monkeypatch.setattr(source, "refresh_all_earnings_consensus", run)
    monkeypatch.setattr(EE, "record_events", lambda *a, **k: (_ for _ in ()).throw(KeyError()))
    tasks.refresh_earnings_consensus.run()


_ALLOWED = {"tasks.py", "ingestion/earnings_events.py", "ingestion/earnings_consensus.py",
            "scripts/batch_verify.py", "scripts/shadow_registry_check.py"}   # readouts
_SKIP = ("tests", "venv", ".venv", ".git", ".claude")


def test_nothing_the_analyst_or_the_order_path_reads_names_it():
    named = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in _SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        if "earnings_event_record" in text or "earnings_events" in text:
            named.add(rel.as_posix())
    assert named == _ALLOWED, named


# ---- batch_verify ----

def _bv():
    import importlib.util
    path = ROOT / "scripts" / "batch_verify.py"
    spec = importlib.util.spec_from_file_location("batch_verify_ee", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _act(i, extra):
    return {"id": i, "stage": "earnings_event_record", "ticker": extra.get("ticker"),
            "extra": extra}


def test_batch_verify_stage():
    bv = _bv()
    assert bv.check_earnings_events([], ENV)["status"] == "WARN"          # conditional
    (pre,), _ = _run(_rec(["2026-11-19"]), on="2026-11-10")
    post, nxt = _run(_rec(["2027-02-25"], history=HIST_OCT, q_end="2027-01-31"), prior=[pre],
                     on="2026-11-20")[0]
    rev, = _run(_rec(["2027-02-26"], history=HIST_OCT, q_end="2027-01-31"),
                prior=[pre, post, nxt], on="2026-11-24")[0]
    rows = [_act(1, pre), _act(2, post), _act(3, nxt), _act(4, rev),
            _act(5, dict(pre, env="paper"))]
    c = bv.check_earnings_events(rows, ENV)
    assert c["status"] == "OK" and c["rows"] == 4
    assert c["reads"]["kind"] == {"pre": 3, "post": 1}
    assert c["reads"]["revision"] == {"first_seen": 2, "revised": 1}
    assert c["reads"]["matched_by"] == {"newest_after_last_reported": 1}
    bad = {k: v for k, v in pre.items() if k != "announce_date"}
    assert bv.check_earnings_events([_act(6, bad)], ENV)["status"] == "FAIL"
    unexplained = dict(pre, consensus_eps=None, missing={})
    c = bv.check_earnings_events([_act(7, unexplained)], ENV)
    assert c["status"] == "FAIL" and "consensus_eps" in c["extra_fail"][0]
    assert bv.check_earnings_events([_act(8, dict(pre, kind="other"))], ENV)["status"] == "FAIL"
    assert "$" not in bv.readout([c], bv.parse_since(None, _at("2026-11-24")), ENV, "x", _at(
        "2026-11-24"), [])


def test_batch_verify_reads_the_stage():
    import inspect
    bv = _bv()
    assert "EARNINGS_STAGE" in inspect.getsource(bv._fetch)
    assert bv.EARNINGS_STAGE == EE.STAGE
