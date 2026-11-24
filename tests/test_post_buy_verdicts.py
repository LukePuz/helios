"""scripts/post_buy_verdicts.py on fake rows: the window after each fill, the
entry bar read from this clone, per-buy and pooled counts, the output, and
the read-only transaction. No DB."""
import importlib.util
import json
import os
import types
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "post_buy_verdicts.py")
_spec = importlib.util.spec_from_file_location("post_buy_verdicts", _PATH)
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)
EF = pb._entry_funnel()
NY = ZoneInfo("America/New_York")

BAR = {"conviction": 7, "udr": 2.0, "analyst_udr_floor": 2.0,
       "sources": {"conviction": "c", "udr": "u", "analyst_udr_floor": "f"}}


def _et(m, d, h, mi=0):
    return datetime(2026, m, d, h, mi, tzinfo=NY).astimezone(timezone.utc)


def _v(ticker, ts, kind="HOLD", c=7, udr=2.5, held=False, path="event", source="news", vid=None):
    return {"id": vid, "ticker": ticker, "ts": ts, "kind": kind,
            "confidence": None if c is None else c / 10, "udr": udr, "held_review": held,
            "path": path, "source": source}


def _buy(ticker="NVDA", fill=None, action="OPEN", aid=1):
    return {"approval_id": aid, "ticker": ticker, "action": action, "filled_at": fill}


# ── the bar ─────────────────────────────────────────────────────────────────

def test_bar_is_read_from_this_clone():
    import config
    with open(os.path.join(os.path.dirname(_PATH), "..", "agents", "analyst.py")) as f:
        src = f.read()
    bar = pb.entry_bar(config, src)
    assert bar["conviction"] is not None and bar["udr"] == float(config.MIN_UPSIDE_DOWNSIDE_RATIO)
    assert bar["analyst_udr_floor"] is not None


def test_bar_follows_the_source_and_config_not_a_literal():
    src = ('    regime_rule_fired = (action == "OPEN" and regime_str in _ENTRY_RULE_REGIMES '
           'and conviction < 8)\n_ENTRY_UDR_FLOOR = 1.5   # x\n')
    bar = pb.entry_bar(types.SimpleNamespace(MIN_UPSIDE_DOWNSIDE_RATIO=2.5), src)
    assert (bar["conviction"], bar["udr"], bar["analyst_udr_floor"]) == (8, 2.5, 1.5)


def test_bar_unknown_when_the_rule_line_is_gone():
    bar = pb.entry_bar(types.SimpleNamespace(MIN_UPSIDE_DOWNSIDE_RATIO=2.0), "nothing here")
    assert bar["conviction"] is None
    assert pb.meets_bar(9, 3.0, bar) is None


@pytest.mark.parametrize("c,udr,want", [(7, 2.0, True), (6, 3.0, False), (8, 1.9, False),
                                        (None, 2.5, None), (7, None, None)])
def test_meets_bar(c, udr, want):
    assert pb.meets_bar(c, udr, BAR) is want


# ── the window ──────────────────────────────────────────────────────────────

def test_window_is_the_fill_session_and_the_next_n():
    sessions, end = pb.buy_window(EF, _et(11, 20, 10, 28), 5)
    assert sessions == [date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 24),
                        date(2026, 11, 25), date(2026, 11, 27), date(2026, 11, 30)]
    assert end == datetime(2026, 12, 1, 0, 0, tzinfo=NY).astimezone(timezone.utc)
    s0, end0 = pb.buy_window(EF, _et(11, 20, 10, 28), 0)
    assert s0 == [date(2026, 11, 20)] and end0 == datetime(2026, 11, 21, tzinfo=NY).astimezone(timezone.utc)


def test_verdicts_before_the_fill_or_after_the_window_are_left_out():
    fill = _et(11, 20, 10, 28)
    vs = [_v("NVDA", _et(11, 20, 10, 20), kind="OPEN"),          # the buy's own verdict, before
          _v("NVDA", fill),                                       # at the fill: not later
          _v("NVDA", _et(11, 20, 10, 47), c=6, udr=1.6),
          _v("NVDA", _et(11, 23, 9, 0)),
          _v("NVDA", _et(11, 24, 9, 0)),                          # past --sessions 1
          _v("AMD", _et(11, 20, 11, 0))]                          # another ticker
    res = pb.build([_buy(fill=fill)], vs, BAR, 1, EF)
    rows = res["buys"][0]["verdicts"]
    assert [r["minutes_after_fill"] for r in rows] == [19.0, round((_et(11, 23, 9) - fill).total_seconds() / 60, 1)]
    assert [r["session_offset"] for r in rows] == [0, 1]


def test_weekend_verdict_counts_toward_the_next_session():
    fill = _et(11, 20, 15, 0)                                     # Friday
    res = pb.build([_buy(fill=fill)], [_v("NVDA", _et(11, 21, 12))], BAR, 1, EF)
    assert res["buys"][0]["verdicts"][0]["session_offset"] == 1


# ── counts ──────────────────────────────────────────────────────────────────

def _two_buys():
    f1, f2 = _et(11, 20, 10, 28), _et(11, 18, 10, 0)
    vs = [_v("NVDA", _et(11, 20, 10, 47), c=6, udr=1.6, held=False),
          _v("NVDA", _et(11, 20, 11, 30), c=7, udr=3.0),
          _v("NVDA", _et(11, 20, 14, 0), c=7, udr=None),
          _v("NVDA", _et(11, 23, 10, 0), c=5, udr=0.5, held=True, path="scheduled_review",
             source="portfolio_review"),
          _v("IONQ", _et(11, 18, 10, 42), c=7, udr=2.0)]
    return [_buy("NVDA", f1, aid=11), _buy("IONQ", f2, "ADD", aid=12)], vs


def test_per_buy_counts():
    buys, vs = _two_buys()
    res = pb.build(buys, vs, BAR, 5, EF)
    ionq, nvda = res["buys"]                                     # in fill order
    assert ionq["ticker"] == "IONQ" and nvda["ticker"] == "NVDA"
    c = nvda["counts"]
    assert (c["verdicts"], c["verdicts_same_session"]) == (4, 3)
    assert (c["meets_bar"], c["below_bar"], c["bar_unknown"]) == (1, 2, 1)
    assert c["same_session_below_bar"] == 1 and c["held_review_verdicts"] == 1
    assert c["first_below_bar_minutes"] == 19.0
    assert c["by_session"][0] == {"session": "2026-11-20", "verdicts": 3, "below_bar": 1}
    assert c["by_session"][1] == {"session": "2026-11-23", "verdicts": 1, "below_bar": 1}
    held = nvda["verdicts"][-1]
    assert held["held_review"] is True and held["trigger_path"] == "scheduled_review"
    assert ionq["counts"]["below_bar"] == 0 and ionq["counts"]["first_below_bar_minutes"] is None


def test_pooled_counts():
    buys, vs = _two_buys()
    p = pb.build(buys, vs, BAR, 5, EF)["pooled"]
    assert p == {"buys": 2, "buys_skipped_no_fill_time": 0, "buys_fill_time_flagged": 0,
                 "verdicts": 5,
                 "verdicts_same_session": 4, "meets_bar": 2, "below_bar": 2, "bar_unknown": 1,
                 "same_session_below_bar": 1, "held_review_verdicts": 1,
                 "buys_with_same_session_below_bar": 1, "distinct_verdicts": 5,
                 "distinct_below_bar": 2}


def test_a_buy_without_a_fill_time_is_listed_as_skipped():
    res = pb.build([_buy(fill=None, aid=3)], [_v("NVDA", _et(11, 20, 11))], BAR, 5, EF)
    assert res["buys"][0]["skipped"] and res["buys"][0]["verdicts"] == []
    assert res["pooled"]["buys"] == 0 and res["pooled"]["buys_skipped_no_fill_time"] == 1


def test_two_buys_of_one_ticker_each_get_their_own_window():
    f1, f2 = _et(11, 20, 10, 0), _et(11, 20, 13, 0)
    vs = [_v("NVDA", _et(11, 20, 11, 0)), _v("NVDA", _et(11, 20, 14, 0))]
    vs = [{**v, "id": i} for i, v in enumerate(vs)]
    res = pb.build([_buy(fill=f1, aid=1), _buy(fill=f2, action="ADD", aid=2)], vs, BAR, 0, EF)
    assert [b["counts"]["verdicts"] for b in res["buys"]] == [2, 1]
    assert res["pooled"]["verdicts"] == 3 and res["pooled"]["distinct_verdicts"] == 2


def test_conviction_is_the_final_confidence_times_ten_and_bad_udr_is_unknown():
    fill = _et(11, 20, 10)
    vs = [_v("NVDA", _et(11, 20, 11), c=None), {**_v("NVDA", _et(11, 20, 12)), "udr": "n/a"},
          {**_v("NVDA", _et(11, 20, 13)), "udr": float("nan")}]
    rows = pb.build([_buy(fill=fill)], vs, BAR, 0, EF)["buys"][0]["verdicts"]
    assert rows[0]["conviction"] is None and rows[0]["meets_bar"] is None
    assert rows[1]["udr"] is None and rows[2]["udr"] is None


# ── flagged fill times (the pre-11-06 5 h zone error) ──────────────────────

def _msft(**kw):
    """MSFT 11-03: submitted 20:07:19Z, last row write 20:07:21Z, filled_at
    recorded 01:07:21Z the next day (5 h late)."""
    sub_at = datetime(2026, 11, 3, 20, 7, 19, tzinfo=timezone.utc)
    return {"approval_id": 166, "ticker": "MSFT", "action": "OPEN",
            "filled_at": datetime(2026, 11, 4, 1, 7, 21, tzinfo=timezone.utc),
            "submitted_at": sub_at, "updated_at": sub_at + __import__("datetime").timedelta(seconds=2),
            **kw}


def test_a_fill_after_the_rows_last_write_is_flagged_and_used_as_recorded_by_default():
    t, src, flag = pb.fill_time(_msft())
    assert (t.hour, src, flag) == (1, "filled_at", "filled_at_after_last_write")


def test_submitted_mode_uses_submitted_at_for_flagged_buys_only():
    t, src, flag = pb.fill_time(_msft(), "submitted")
    assert (t.hour, t.minute, src) == (20, 7, "submitted_at") and flag
    ok = {**_msft(), "filled_at": datetime(2026, 11, 3, 20, 7, 21, tzinfo=timezone.utc)}
    assert pb.fill_time(ok, "submitted")[1:] == ("filled_at", None)


def test_flag_changes_which_verdicts_fall_after_the_fill():
    v = [_v("MSFT", datetime(2026, 11, 3, 20, 53, tzinfo=timezone.utc), c=6, udr=1.2)]
    rec = pb.build([_msft()], v, BAR, 0, EF)
    assert rec["buys"][0]["counts"]["verdicts"] == 0 and rec["pooled"]["buys_fill_time_flagged"] == 1
    sub = pb.build([_msft()], v, BAR, 0, EF, suspect_fill="submitted")
    b = sub["buys"][0]
    assert b["counts"]["verdicts"] == 1 and b["counts"]["first_below_bar_minutes"] == 45.7
    assert b["fill_time_source"] == "submitted_at"
    text = pb.render_text(sub, {**_meta(), "suspect_fill": "submitted"})
    assert "FLAG filled_at_after_last_write" in text


def test_fetch_reads_verdicts_from_the_earlier_candidate_fill_time():
    m = _msft()
    conn = _Conn(buys=[(166, "MSFT", "OPEN", m["filled_at"], m["submitted_at"], m["updated_at"])])
    pb.fetch(conn, "live", pb.parse_since(None), 0, (), EF)
    (_, args), = [(s, a) for s, a in conn.sql if "FROM signals" in s]
    assert args[2] == m["submitted_at"]


# ── output ──────────────────────────────────────────────────────────────────

def _meta(bar=BAR):
    return {"env": "live", "clone": "abc1234", "generated": "2026-11-21T21:00-05:00",
            "since": "the first buy", "n_sessions": 5, "bar": bar, "bar_note": "descriptive only"}


def test_text_says_descriptive_and_shows_no_dollars():
    buys, vs = _two_buys()
    text = pb.render_text(pb.build(buys, vs, BAR, 5, EF), _meta())
    assert "DESCRIPTIVE ONLY" in text and "$" not in text
    assert "## NVDA OPEN (approval 11), filled 2026-11-20T10:28-05:00 ET" in text
    assert "buys with at least one same-session verdict below the bar: 1 of 2" in text


def test_text_flags_a_udr_floor_mismatch_and_an_unknown_bar():
    bar = {**BAR, "analyst_udr_floor": 1.5}
    assert "_ENTRY_UDR_FLOOR is 1.5" in pb.render_text(pb.build([], [], bar, 5, EF), _meta(bar))
    bar = {**BAR, "conviction": None}
    assert "bar is unknown" in pb.render_text(pb.build([], [], bar, 5, EF), _meta(bar))


def test_json_is_serialisable():
    buys, vs = _two_buys()
    json.dumps({"meta": _meta(), **pb.build(buys, vs, BAR, 5, EF)}, default=str)


# ── the read ────────────────────────────────────────────────────────────────

class _Cur:
    def __init__(self, conn):
        self.conn, self._res = conn, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.conn.sql.append((sql, args))
        if "FROM orders" in sql:
            self._res = self.conn.buys
        elif "FROM signals" in sql:
            self._res = [r for r in self.conn.verdicts if r[1] == args[1]]
        else:
            self._res = []

    def fetchall(self):
        return list(self._res)


class _Conn:
    def __init__(self, buys=(), verdicts=()):
        self.buys, self.verdicts, self.sql, self.rolled_back = list(buys), list(verdicts), [], False

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled_back = True

    def commit(self):
        raise AssertionError("never commits")


def test_fetch_is_read_only_filters_env_and_tests_and_rolls_back():
    fill = _et(11, 20, 10, 28)
    conn = _Conn(buys=[(11, "NVDA", "OPEN", fill, fill, fill), (12, "AMD", "ADD", None, None, None)],
                 verdicts=[(5, "NVDA", _et(11, 20, 10, 47), "HOLD", 0.6, "1.6", False, "event", "news")])
    buys, verdicts = pb.fetch(conn, "live", pb.parse_since(None), 5, ("PLAN_LEVEL_HIT",), EF)
    sqls = [s for s, _ in conn.sql]
    assert sqls[0] == "SET TRANSACTION READ ONLY" and "statement_timeout" in sqls[1]
    assert conn.rolled_back
    assert all(s.lstrip().upper().startswith(("SELECT", "SET")) for s in sqls)
    order_sql, order_args = conn.sql[2]
    assert "is_test" in order_sql and "o.status = 'filled'" in order_sql and order_args[0] == "live"
    sig = [(s, a) for s, a in conn.sql if "FROM signals" in s]
    assert len(sig) == 1                                        # AMD has no fill time: no read
    assert sig[0][1][0] == "live" and sig[0][1][4] == ["PLAN_LEVEL_HIT"]
    assert verdicts[0]["udr"] == "1.6" and verdicts[0]["held_review"] is False
    assert [b["ticker"] for b in buys] == ["NVDA", "AMD"]


def test_main_refuses_another_port(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_PORT", "5432")
    assert pb.main([]) == 2
    assert "6543" in capsys.readouterr().err
