import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import verdict_spread as vs  # noqa: E402


def _row(sym, hour_utc, conv, udr, action="HOLD", day=4):
    return (sym, datetime(2026, 11, day, hour_utc, tzinfo=timezone.utc),
            {"model_conviction": conv, "udr_check": {"model": udr}, "final_action": action})


def test_spread_groups_by_et_day_and_flags_a_straddle():
    rows = [_row("NVDA", 14, 6, 1.0), _row("NVDA", 16, 7, 3.6, "OPEN"), _row("NVDA", 18, 5, 2.4),
            _row("AMD", 14, 6, 2.5), _row("AMD", 15, 6, 3.0), _row("AMD", 16, 6, 2.1),
            _row("PLTR", 14, 6, 1.0), _row("PLTR", 15, 6, 3.0)]   # 2 verdicts: below the minimum
    days = vs.spread(rows, 2.0, 3)
    assert [(d["ticker"], d["n"], d["crosses_floor"]) for d in days] == [("AMD", 3, False), ("NVDA", 3, True)]
    nvda = days[1]
    assert nvda["conviction"] == (5, 7) and nvda["udr"] == (1.0, 3.6) and nvda["actions"] == ["HOLD", "OPEN"]
    assert nvda["actions_differ"] is True and nvda["entry_action"] is True
    assert days[0]["actions_differ"] is False and days[0]["entry_action"] is False


def test_et_day_boundary():
    # 03:00 UTC on 11-05 is 22:00 ET on 11-04
    rows = [_row("X", 3, 6, 1.0, day=5), _row("X", 14, 6, 3.0), _row("X", 15, 6, 3.0)]
    assert vs.spread(rows, 2.0, 3)[0]["day"] == "2026-11-04"


def test_rows_without_udr_check_or_numbers_are_skipped():
    rows = [_row("X", 14, 6, 1.0), _row("X", 15, None, None), ("X", datetime(2026, 11, 4, 16, tzinfo=timezone.utc), {})]
    d = vs.spread(rows, 2.0, 2)[0]
    assert d["n"] == 2 and d["udr"] == (1.0, 1.0) and d["crosses_floor"] is False


def test_render_has_counts_only_and_the_summary():
    days = vs.spread([_row("NVDA", 14, 6, 1.0), _row("NVDA", 16, 7, 3.6), _row("NVDA", 18, 5, 2.4)], 2.0, 3)
    text = vs.render(days, 2.0, 21, 3, "live", "abc1234")
    assert "straddles the 2 entry floor on 1; of those, the final action differs within the day on 0" in text
    assert "| 2026-11-04 | NVDA | 3 | 5-7 | 1.00-3.60 | yes |  |  | HOLD |" in text
    assert "$" not in text


T0 = datetime(2026, 11, 16, 14, 0, tzinfo=timezone.utc)


def _verdict(sym, minute, path="event", new_fact=True, action="HOLD", prior=None, enqueued=True):
    """(symbol, timestamp, decision, trigger): a verdict whose event joined
    the queue `minute` minutes after T0 and was written three minutes later."""
    at = T0 + timedelta(minutes=minute)
    trigger = {"path": path, "new_fact": new_fact, "title": "HEADLINE TEXT"}
    if enqueued:
        trigger["llm_enqueued_at"] = at.isoformat()
    return (sym, at + timedelta(minutes=3),
            {"final_action": action, "model_conviction": 6, "udr_check": {"model": 2.5},
             "prior_verdicts_72h": prior, "model_output": {"reasoning": "VERDICT TEXT"}}, trigger)


def test_calls_left_counts_one_call_per_ticker_window():
    rows = [_verdict("NVDA", m) for m in (0, 10, 16, 31, 70)] + [_verdict("AMD", m) for m in (5, 50)]
    rows += [_verdict("NVDA", 20, path="daily_scan", enqueued=False),   # not an event: unchanged
             _verdict("NVDA", 21, enqueued=False)]                       # event with no enqueue time
    out = vs.calls_left(rows)
    assert out["verdicts"] == 9 and out["replayed"] == 7
    # 15: NVDA {0,10} {16} {31} {70}, AMD {5} {50}; 30: NVDA {0,10,16} {31} {70}, AMD {5} {50};
    # 60: NVDA {0,10,16,31} {70}, AMD {5,50}
    assert out["left"] == {15: 6, 30: 5, 60: 3}


def test_calls_left_windows_are_per_ticker_and_open_at_the_first_event():
    assert vs.calls_left([_verdict("NVDA", 0), _verdict("AMD", 0)], (30,))["left"] == {30: 2}
    # 14 and 16 straddle a quarter-hour mark but sit in the window opened at 14
    assert vs.calls_left([_verdict("NVDA", 14), _verdict("NVDA", 16)], (15,))["left"] == {15: 1}
    # an event exactly one window after the first opens the next window
    assert vs.calls_left([_verdict("NVDA", 0), _verdict("NVDA", 15)], (15,))["left"] == {15: 2}
    assert vs.calls_left([_verdict("NVDA", 0), _verdict("NVDA", 14)], (15,))["left"] == {15: 1}


def test_calls_left_opens_the_next_window_at_the_first_event_outside_the_last():
    # {0} {20, 31}: the second window opens at 20 and runs to 35. Fixed
    # 15-minute slots counted from the first event would give {0} {20} {31}.
    assert vs.calls_left([_verdict("NVDA", m) for m in (0, 20, 31)], (15,))["left"] == {15: 2}
    # {0} {40, 65}: 30-minute slots from the first event would split 40 from 65
    assert vs.calls_left([_verdict("NVDA", m) for m in (0, 40, 65)], (30,))["left"] == {30: 2}


def test_calls_left_does_not_depend_on_row_order_or_enqueue_time_format():
    rows = [_verdict("NVDA", m) for m in (31, 0, 70, 16, 10)]
    assert vs.calls_left(rows)["left"] == {15: 4, 30: 3, 60: 2}
    naive = ("NVDA", T0, {}, {"path": "event", "llm_enqueued_at": "2026-11-16T14:05:00"})   # read as UTC
    aware = ("NVDA", T0, {}, {"path": "event", "llm_enqueued_at": "2026-11-16T09:10:00-05:00"})
    assert vs.calls_left([naive, aware], (15, 30))["left"] == {15: 1, 30: 1}


def test_calls_left_leaves_rows_without_a_readable_enqueue_time_out_of_the_replay():
    rows = [("X", T0, {}, None), ("X", T0, {}, {"path": "event", "llm_enqueued_at": None}),
            ("X", T0, {}, {"path": "event", "llm_enqueued_at": "not a time"}),
            ("X", T0, {}, {"path": "scheduled_review", "llm_enqueued_at": T0.isoformat()})]
    assert vs.calls_left(rows) == {"verdicts": 4, "replayed": 0, "left": {15: 0, 30: 0, 60: 0}}
    assert vs.calls_left([]) == {"verdicts": 0, "replayed": 0, "left": {15: 0, 30: 0, 60: 0}}


def test_new_fact_counts_cover_event_triggers_only():
    rows = [_verdict("NVDA", 0, new_fact=True), _verdict("NVDA", 1, new_fact=False),
            _verdict("NVDA", 2, new_fact=False), _verdict("AMD", 3, new_fact=None),
            _verdict("AMD", 4, new_fact="false"),                            # not a boolean: not judged
            _verdict("AMD", 5, path="daily_scan", new_fact=False),           # not an event
            ("AMD", T0, {}, None)]
    assert vs.new_fact_counts(rows) == {"event": 5, "true": 1, "false": 2, "not_judged": 2,
                                        "false_share": 0.4}
    assert vs.new_fact_counts([])["false_share"] is None


def test_new_fact_counts_put_an_event_with_no_screen_in_not_judged():
    # a drop-watch review: path "event", queued with no screen, so no new_fact and no enqueue time
    dip = ("NVDA", T0, {}, {"path": "event", "source": "dip_watch", "new_fact": None,
                            "llm_enqueued_at": None})
    assert vs.new_fact_counts([dip, _verdict("NVDA", 1, new_fact=False)]) == {
        "event": 2, "true": 0, "false": 1, "not_judged": 1, "false_share": 0.5}


def test_buys_are_split_by_the_previous_verdict():
    hold = {"t": "2026-11-16 13:50", "signal": "HOLD", "c": 7, "udr": 4.2}
    add = {"t": "2026-11-16 13:40", "signal": "ADD", "c": 7, "udr": 4.2}
    rows = [_verdict("NVDA", 0, action="ADD", prior=[hold, add]),    # newest first: previous is HOLD
            _verdict("NVDA", 1, action="ADD", prior=[add, hold]),    # previous is a buy
            _verdict("MSFT", 2, action="OPEN", prior=[{"signal": "WATCH"}]),
            _verdict("MSFT", 3, action="OPEN", prior=[]),            # none in 72 h
            _verdict("AMD", 4, action="OPEN", prior=None),           # not recorded
            _verdict("AMD", 5, action="HOLD", prior=[hold]),         # not a buy: not counted
            ("AMD", T0, None, None)]
    assert vs.buys_after_non_buy(rows) == {"buys": 5, "prior_buy": 1, "prior_not_buy": 2, "no_prior": 1,
                                           "not_recorded": 1, "prior_not_buy_share": 0.4}
    assert vs.buys_after_non_buy([_verdict("AMD", 0)]) == {
        "buys": 0, "prior_buy": 0, "prior_not_buy": 0, "no_prior": 0, "not_recorded": 0,
        "prior_not_buy_share": None}


def test_the_previous_verdict_is_the_newest_entry_of_the_72h_list():
    hold = {"t": "2026-11-16 13:50", "signal": "HOLD", "c": 7, "udr": 4.2}
    add = {"t": "2026-11-16 13:40", "signal": "ADD", "c": 7, "udr": 4.2}
    opened = {"t": "2026-11-16 13:30", "signal": "OPEN", "c": 7, "udr": 4.2}

    def split(prior):
        out = vs.buys_after_non_buy([_verdict("NVDA", 0, action="ADD", prior=prior)])
        return out["prior_buy"], out["prior_not_buy"]

    # the list is newest first: only its first entry decides, whatever the older ones were
    assert split([hold, add]) == (0, 1)
    assert split([add, hold]) == (1, 0)
    assert split([hold, add, opened]) == (0, 1)
    assert split([add, hold, hold]) == (1, 0)
    # a previous OPEN is a buy as much as a previous ADD
    assert split([opened]) == (1, 0)
    assert split([opened, hold]) == (1, 0)
    assert split([hold, opened]) == (0, 1)


def test_render_replay_prints_counts_and_shares_only():
    rows = [_verdict("NVDA", m, new_fact=m > 0) for m in (0, 10, 16, 31)]
    rows += [_verdict("NVDA", 40, path="daily_scan", enqueued=False),
             _verdict("MSFT", 5, action="OPEN", prior=[{"signal": "HOLD"}])]
    text = vs.render_replay(vs.calls_left(rows), vs.new_fact_counts(rows), vs.buys_after_non_buy(rows))
    assert "Verdicts: 6; triggered by a queued event and replayed: 5; counted unchanged" in text
    assert "| actual | 5 | 100% | 6 |" in text
    assert "| 15 min | 4 | 80% | 5 |" in text and "| 30 min | 3 | 60% | 4 |" in text
    assert "| 60 min | 2 | 40% | 3 |" in text
    assert "| true | 4 | 80% |" in text and "| false | 1 | 20% |" in text and "| not judged | 0 | 0% |" in text
    # the section says what "not judged" holds and what the shares are of
    assert "of the 5 event-triggered verdicts. Shares are of all of them." in text
    assert "queued with no screen and no news item" in text and "drop-watch review" in text
    assert "Verdicts with final action OPEN or ADD: 1." in text
    assert "| a buy | 0 | 0% |" in text and "| not a buy | 1 | 100% |" in text
    for leak in ("NVDA", "MSFT", "HEADLINE", "VERDICT TEXT", "$"):
        assert leak not in text


def test_render_replay_with_no_rows():
    text = vs.render_replay(vs.calls_left([]), vs.new_fact_counts([]), vs.buys_after_non_buy([]))
    assert "| actual | 0 | n/a | 0 |" in text and "| 15 min | 0 | n/a | 0 |" in text
    assert "| false | 0 | n/a |" in text and "| not a buy | 0 | n/a |" in text


def test_spread_reads_rows_that_carry_a_trigger():
    rows = [_verdict("NVDA", m) for m in (0, 10, 20)]
    assert [(d["ticker"], d["n"]) for d in vs.spread(rows, 2.0, 3)] == [("NVDA", 3)]


def test_main_reads_the_trigger_read_only_and_appends_the_aggregates(capsys):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = ([_verdict("NVDA", m, new_fact=False) for m in (0, 10, 40)]
                                 + [_verdict("NVDA", 50, action="ADD", prior=[{"signal": "HOLD"}])])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn") as put, \
            patch.object(sys, "argv", ["verdict_spread.py", "--days", "7"]):
        vs.main()
    statements = [c.args[0] for c in cur.execute.call_args_list]
    assert statements[0] == "SET TRANSACTION READ ONLY" and len(statements) == 2
    assert statements[1].startswith("SELECT symbol, timestamp, decision, provenance->'trigger' FROM signals ")
    put.assert_called_once_with(conn)
    conn.commit.assert_not_called()
    text = capsys.readouterr().out
    assert "| 2026-11-16 | NVDA | 4 |" in text                     # the per-day table is still there
    assert "| actual | 4 | 100% | 4 |" in text and "| 30 min | 2 | 50% | 2 |" in text
    assert "| false | 3 | 75% |" in text and "| not a buy | 1 | 100% |" in text
    assert "HEADLINE" not in text and "VERDICT TEXT" not in text
