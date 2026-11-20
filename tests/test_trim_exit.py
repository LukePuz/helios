"""The trim/exit rule (agents/trim_exit.py): default off, shadow first, a
pure decide(), one TRIM_EXIT_RULE row per evaluation, routing on a copy only
when the rule owns the verdict, and the recheck execution runs before it
touches the stop. No live DB, broker or Redis."""
import copy
import json
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from agents import trim_exit

NOW = datetime(2026, 11, 16, 15, 30, tzinfo=timezone.utc)      # 10:30 ET, Monday
PRINT = date(2026, 11, 18)                                       # two days ahead
OPENED = NOW - timedelta(days=20)
PRE_EVENT = {"ticker": "NVDA", "type": "earnings review", "earnings_phase": "pre",
             "earnings_timing": {"time_of_day": "AMC"}}

READS_OK = ("market", "stop", "implied_move", "verdicts", "last_buy", "last_sell",
            "pending_sell", "active_approval")

FLAGS = {
    "TRIM_EXIT_ENABLED": True, "TRIM_EXIT_LIVE": True,
    "TRIM_EXIT_LIVE_CONTEXTS": ("earnings_pre",), "TRIM_EXIT_EXEMPT_TICKERS": (),
    "TRIM_EXIT_MIN_CONVICTION": 5, "TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE": 4,
    "TRIM_EXIT_CONFIRM_MIN_GAP_HOURS": 12.0, "TRIM_EXIT_CONFIRM_MAX_AGE_DAYS": 7,
    "TRIM_EXIT_COOLDOWN_DAYS": 5, "TRIM_EXIT_MIN_HOLD_HOURS": 48,
    "TRIM_EXIT_ONE_SHARE_TRIM": "hold", "TRIM_EXIT_MIN_REMAINING_SHARES": 1,
    "TRIM_EXIT_FRACTION": 0.33,
}


def _cfg(**over) -> dict:
    """A flag snapshot as trim_exit._cfg() returns it."""
    with _flags(**over):
        return trim_exit._cfg()


@contextmanager
def _flags(**over):
    with ExitStack() as st:
        for k, v in {**FLAGS, **over}.items():
            st.enter_context(patch.object(config, k, v, create=True))
        yield


def _verdict(action: str, hours_ago: float, conviction: int = 6) -> dict:
    return {"timestamp": (NOW - timedelta(hours=hours_ago)).isoformat(), "action": action,
            "conviction": conviction}


def _inputs(**over) -> dict:
    """In-domain earnings_pre TRIM with every read ok and nothing blocking."""
    d = {"ticker": "NVDA", "path": "earnings_review", "observe": False,
         "evaluated_at": NOW.isoformat(), "action": "TRIM", "conviction": 6,
         "verdict_signal_id": 101, "earnings_phase": "pre", "days_to_print": 2,
         "owned": 10.0, "stop_record": True, "opened_at": OPENED.isoformat(),
         "verdicts": [_verdict("TRIM", 0.01)],
         "last_buy": None, "last_sell": None, "pending_sell": False, "active_approval": False,
         "market_open": True, "reads": {k: "ok" for k in READS_OK}}
    d.update(over)
    return d


def _result(action="EXIT", conviction=6, pass_to_risk=False, ticker="NVDA", signal_id=101) -> dict:
    return {"ticker": ticker, "action": action, "conviction": conviction,
            "pass_to_risk": pass_to_risk, "thesis_bucket": "Accelerator", "reasoning": "r",
            "provenance": {"signal_id": signal_id, "model": "m"}}


POSITIONS = {"NVDA": {"qty": 10.0, "entry_price": 100.0, "current_price": 110.0}}


@contextmanager
def _db(verdicts=None, stop="default", log_id=55, **raise_on):
    """Every read the rule makes, patched; raise_on={name: exc} makes that
    patched function raise. Yields the log_signal mock."""
    stop = {"stop_price": 99.0, "opened_at": OPENED} if stop == "default" else stop
    rows = [{"timestamp": NOW - timedelta(minutes=1), "signal": "EXIT", "conviction": 6}]
    vals = {
        "db.get_position_stop": stop,
        "db.get_recent_verdicts": rows if verdicts is None else verdicts,
        "db.get_last_executed_approval": None,
        "db.sell_trades_since": None,
        "db.get_unconfirmed_orders": [],
        "db.awaiting_fill_sell_tickers": set(),
        "db.trade_pending_sell_tickers": set(),
        "db.has_active_approval": False,
        "db.get_next_earnings_row": (PRINT, True),
        "db.get_state": None,
        "market_data.is_market_hours": True,
        "market_data.market_session": "rth",
    }
    with ExitStack() as st:
        for target, v in vals.items():
            exc = raise_on.get(target.split(".")[-1])
            st.enter_context(patch(target, side_effect=exc) if exc else patch(target, return_value=v))
        log_signal = st.enter_context(patch("db.log_signal", return_value=log_id))
        st.enter_context(patch.object(trim_exit, "_utcnow", return_value=NOW))
        yield log_signal


def _row(log_signal) -> dict:
    """The decision of the one TRIM_EXIT_RULE row written, through JSON as
    the database stores it."""
    calls = [c for c in log_signal.call_args_list if c.args[1] == "TRIM_EXIT_RULE"]
    assert len(calls) == 1, calls
    return json.loads(json.dumps(calls[0].kwargs["decision"], default=str))


# ── switches ─────────────────────────────────────────────────────────────────

def test_master_off_returns_same_object_no_db():
    result = _result(action="EXIT", pass_to_risk=True)
    before = copy.deepcopy(result)
    with _flags(TRIM_EXIT_ENABLED=False, TRIM_EXIT_LIVE=True), \
         patch("db._get_conn") as get_conn, patch("db._get_pool") as get_pool, \
         patch("db._writer_conn") as writer, patch("db.log_signal") as log_signal, \
         patch("db.get_state") as get_state, patch("db.get_position_stop") as get_stop, \
         patch("market_data.is_market_hours") as hours, \
         patch.object(trim_exit, "_gather") as gather:
        for observe in (False, True):
            out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT,
                                  earnings_date=PRINT, positions=POSITIONS, observe=observe)
            assert out is result
    assert result == before
    for m in (get_conn, get_pool, writer, log_signal, get_state, get_stop, hours, gather):
        m.assert_not_called()


def test_config_defaults_are_off_and_shadow(monkeypatch):
    """The built defaults: master and sell switch off, earnings_pre the only
    live context, nothing exempt. The sets are sorted tuples, so their repr
    (and env_sha) is the same in every process. config.py loads the
    checkout's .env on every load, and load_dotenv fills unset variables, so
    it is stubbed here: a .env that sets a flag can't stand in for its
    default."""
    import importlib
    import dotenv
    try:
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        for k in FLAGS:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(config)
        assert config.TRIM_EXIT_ENABLED is False and config.TRIM_EXIT_LIVE is False
        assert config.TRIM_EXIT_LIVE_CONTEXTS == ("earnings_pre",)
        assert config.TRIM_EXIT_EXEMPT_TICKERS == ()
        assert config.TRIM_EXIT_FRACTION == config.TRIM_FRACTION
        assert config.TRIM_EXIT_ONE_SHARE_TRIM == "hold"
        assert (config.TRIM_EXIT_MIN_CONVICTION, config.TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE,
                config.TRIM_EXIT_CONFIRM_MIN_GAP_HOURS, config.TRIM_EXIT_CONFIRM_MAX_AGE_DAYS,
                config.TRIM_EXIT_COOLDOWN_DAYS, config.TRIM_EXIT_MIN_HOLD_HOURS,
                config.TRIM_EXIT_MIN_REMAINING_SHARES) == (5, 4, 12.0, 7, 5, 48, 1)
        monkeypatch.setenv("TRIM_EXIT_LIVE_CONTEXTS", " General, earnings_pre ,,")
        monkeypatch.setenv("TRIM_EXIT_EXEMPT_TICKERS", "nvda, amd")
        importlib.reload(config)
        assert config.TRIM_EXIT_LIVE_CONTEXTS == ("earnings_pre", "general")
        assert config.TRIM_EXIT_EXEMPT_TICKERS == ("AMD", "NVDA")
    finally:
        # The environment and load_dotenv as they were, then config as the
        # rest of the run loaded it.
        monkeypatch.undo()
        importlib.reload(config)


# ── decide(): the table ──────────────────────────────────────────────────────

@pytest.mark.parametrize("phase,days,context", [("pre", 2, "earnings_pre"), ("day", 0, "earnings_day"),
                                                (None, None, "general"), ("post", -1, "general")])
@pytest.mark.parametrize("verdict", ["TRIM", "EXIT"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_decide_table_each_row(phase, days, context, verdict, confirmed):
    cfg = _cfg(TRIM_EXIT_LIVE_CONTEXTS=("earnings_pre", "earnings_day", "general"))
    verdicts = [_verdict(verdict, 0.01)] + ([_verdict("TRIM", 20)] if confirmed else [])
    out = trim_exit.decide(_inputs(action=verdict, earnings_phase=phase, days_to_print=days,
                                   verdicts=verdicts), cfg)
    assert out["context"] == context and out["confirmed"] is confirmed
    if context == "earnings_pre":
        want = verdict if confirmed else "TRIM"
    else:      # earnings_day decides as general does
        want = verdict if confirmed else None
    assert out["would_action"] == want
    assert out["routed"] is (want is not None) and out["owns_pass_flag"] is True
    if want is None:
        assert out["reason"] == "wait_confirm"
    assert out["would_qty"] == (None if want is None else 10.0 if want == "EXIT" else 3.0)


def test_earnings_pre_needs_the_print_inside_the_window():
    cfg = _cfg()
    assert trim_exit.decide(_inputs(days_to_print=4), cfg)["context"] == "earnings_pre"
    assert trim_exit.decide(_inputs(days_to_print=5), cfg)["context"] == "general"
    assert trim_exit.decide(_inputs(days_to_print=0), cfg)["context"] == "earnings_pre"   # AMC today
    assert trim_exit.decide(_inputs(earnings_phase="post", days_to_print=-1), cfg)["context"] == "general"


def test_below_min_conviction_is_out_of_domain():
    out = trim_exit.decide(_inputs(conviction=4), _cfg())
    assert out["in_domain"] is False and out["owns_pass_flag"] is False and out["routed"] is False
    assert out["reason"] == "below_min_conviction" and out["would_action"] is None


def test_hold_verdict_is_recorded_not_owned():
    out = trim_exit.decide(_inputs(action="HOLD"), _cfg())
    assert out["is_sell"] is False and out["reason"] == "not_sell_verdict"
    assert out["owns_pass_flag"] is False and out["routed"] is False


# ── confirmation ─────────────────────────────────────────────────────────────

def test_confirm_ignores_current_verdict():
    cfg = _cfg()
    # The verdict under evaluation and one an hour older: both inside the gap.
    out = trim_exit.decide(_inputs(action="EXIT", verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 1)]), cfg)
    assert out["confirmed"] is False and out["confirm_reason"] == "no_earlier_verdict"
    assert out["would_action"] == "TRIM"        # earnings_pre, unconfirmed EXIT
    out = trim_exit.decide(_inputs(action="EXIT", verdicts=[_verdict("EXIT", 0.01), _verdict("TRIM", 13)]), cfg)
    assert out["confirmed"] is True and out["would_action"] == "EXIT"
    # The newest verdict past the gap decides: a HOLD there is not a confirmation.
    out = trim_exit.decide(_inputs(action="EXIT", verdicts=[_verdict("EXIT", 0.01), _verdict("HOLD", 13),
                                                             _verdict("EXIT", 30)]), cfg)
    assert out["confirmed"] is False and out["confirm_reason"] == "earlier_not_sell"


def test_confirm_before_opened_at():
    out = trim_exit.decide(_inputs(action="EXIT", opened_at=(NOW - timedelta(hours=10)).isoformat(),
                                   verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 24)]), _cfg())
    assert out["confirmed"] is False and out["confirm_reason"] == "before_opened_at"
    out = trim_exit.decide(_inputs(action="EXIT", opened_at=None,
                                   verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 24)]), _cfg())
    assert out["confirmed"] is False and out["confirm_reason"] == "opened_at_unknown"


def test_confirm_too_old():
    out = trim_exit.decide(_inputs(action="EXIT", verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 8 * 24)]),
                           _cfg())
    assert out["confirmed"] is False and out["confirm_reason"] == "too_old"


def test_confirm_read_error_unconfirmed():
    out = trim_exit.decide(_inputs(action="EXIT", verdicts=None,
                                   reads={**{k: "ok" for k in READS_OK}, "verdicts": "error:RuntimeError"}),
                           _cfg())
    assert out["confirmed"] is False and out["confirm_reason"] == "verdicts_unreadable"
    assert out["would_action"] == "TRIM"
    # Through the reads: the history read raising leaves the EXIT unconfirmed.
    with _flags(), _db(get_recent_verdicts=RuntimeError("pool")) as log_signal:
        trim_exit.apply(_result(), path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                        positions=POSITIONS)
    row = _row(log_signal)
    assert row["reads"]["verdicts"] == "error:RuntimeError" and row["confirmed"] is False


# ── blocks ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("over,cfg_over,block", [
    ({}, {"TRIM_EXIT_LIVE": False}, "shadow"),
    ({"observe": True}, {}, "observe"),
    ({}, {"TRIM_EXIT_EXEMPT_TICKERS": ("NVDA",)}, "exempt"),
    ({"earnings_phase": None, "days_to_print": None}, {}, "context_not_live"),
    ({"owned": 0.0}, {}, "no_position"),
    ({"last_sell": {"n": 1}}, {}, "cooldown"),
    ({"last_buy": {"conviction": 6}}, {}, "min_hold"),
    ({"stop_record": False}, {}, "no_stop_record"),
    ({"pending_sell": True}, {}, "pending_sell"),
    ({"active_approval": True}, {}, "active_approval"),
    ({"market_open": False}, {}, "market_closed"),
    ({"verdict_signal_id": None}, {}, "verdict_not_logged"),
])
def test_each_block_alone(over, cfg_over, block):
    out = trim_exit.decide(_inputs(**over), _cfg(**cfg_over))
    assert out["blocked_by"] == [block]
    assert out["routed"] is False
    assert out["owns_pass_flag"] is (block not in trim_exit.STAND_DOWN)


def test_cooldown_holds_back_a_trim_only():
    cfg = _cfg()
    out = trim_exit.decide(_inputs(action="EXIT", last_sell={"n": 1},
                                   verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 20)]), cfg)
    assert out["would_action"] == "EXIT" and out["blocked_by"] == [] and out["routed"] is True
    out = trim_exit.decide(_inputs(last_sell={"n": 1}), _cfg(TRIM_EXIT_COOLDOWN_DAYS=0))
    assert out["blocked_by"] == [] and out["routed"] is True


def test_cooldown_holds_back_a_one_share_exit():
    """TRIM_EXIT_ONE_SHARE_TRIM=exit: day 1 trims 2 shares to 1; day 2's
    TRIM verdict, confirmed by day 1's, becomes an EXIT of the last share.
    It came from a TRIM, so the cooldown holds it back like the TRIM."""
    exit_cfg = _cfg(TRIM_EXIT_ONE_SHARE_TRIM="exit")
    day2 = _inputs(owned=1.0, last_sell={"n": 1}, verdicts=[_verdict("TRIM", 0.01), _verdict("TRIM", 24)])
    out = trim_exit.decide(day2, exit_cfg)
    assert out["would_action"] == "EXIT" and out["reason"] == "trim_one_share_exit"
    assert out["blocked_by"] == ["cooldown"] and out["owns_pass_flag"] is True and out["routed"] is False
    # An unreadable sell history holds it back too.
    out = trim_exit.decide({**day2, "last_sell": None,
                            "reads": {**day2["reads"], "last_sell": "error:RuntimeError"}}, exit_cfg)
    assert out["blocked_by"] == ["cooldown"]
    # No sell in the window, or no cooldown: it routes.
    assert trim_exit.decide({**day2, "last_sell": None}, exit_cfg)["routed"] is True
    assert trim_exit.decide(day2, _cfg(TRIM_EXIT_ONE_SHARE_TRIM="exit", TRIM_EXIT_COOLDOWN_DAYS=0))["routed"] is True
    # A confirmed EXIT verdict on the same share is not held back.
    out = trim_exit.decide({**day2, "action": "EXIT"}, exit_cfg)
    assert out["reason"] == "confirmed" and out["blocked_by"] == [] and out["routed"] is True


def test_confirm_gap_has_a_floor():
    """A gap of 0 (or less) does not let the verdict under evaluation confirm
    itself: the floor is CONFIRM_MIN_GAP_FLOOR_HOURS."""
    general = dict(earnings_phase=None, days_to_print=None, action="EXIT")
    for gap in (0.0, -5.0):
        cfg = _cfg(TRIM_EXIT_CONFIRM_MIN_GAP_HOURS=gap, TRIM_EXIT_LIVE_CONTEXTS=("general",))
        out = trim_exit.decide(_inputs(**general, verdicts=[_verdict("EXIT", 0.01)]), cfg)
        assert out["confirmed"] is False and out["confirm_reason"] == "no_earlier_verdict"
        assert out["reason"] == "wait_confirm" and out["routed"] is False
        # A verdict older than the floor still confirms.
        out = trim_exit.decide(_inputs(**general, verdicts=[_verdict("EXIT", 0.01),
                                                            _verdict("EXIT", trim_exit.CONFIRM_MIN_GAP_FLOOR_HOURS + 0.5)]),
                               cfg)
        assert out["confirmed"] is True and out["routed"] is True


def test_exit_verdict_without_stop_record_stands_down():
    """An EXIT verdict on a position with no stop record: the rule can't
    confirm it (no opened_at) or trim, so it stands down and the model's
    own pass flag routes it as before. A TRIM verdict there still sells
    nothing, and an unreadable stop record still fails closed."""
    for phase, days, contexts in (("pre", 2, ("earnings_pre",)), (None, None, ("general",))):
        cfg = _cfg(TRIM_EXIT_LIVE_CONTEXTS=contexts)
        base = _inputs(action="EXIT", earnings_phase=phase, days_to_print=days, stop_record=False,
                       opened_at=None, verdicts=[_verdict("EXIT", 0.01), _verdict("EXIT", 20)])
        out = trim_exit.decide(base, cfg)
        assert out["confirm_reason"] == "opened_at_unknown"
        assert out["blocked_by"] == ["exit_no_stop_record"]
        assert out["owns_pass_flag"] is False and out["routed"] is False
        out = trim_exit.decide({**base, "action": "TRIM"}, cfg)
        assert out["blocked_by"] == ["no_stop_record"] and out["owns_pass_flag"] is True
        out = trim_exit.decide({**base, "stop_record": None,
                                "reads": {**base["reads"], "stop": "error:RuntimeError"}}, cfg)
        assert out["blocked_by"] == ["no_stop_record"] and out["owns_pass_flag"] is True
    # Through apply(): the model's own EXIT comes back untouched, so it routes.
    for model_pass in (True, False):
        result = _result(action="EXIT", conviction=7, pass_to_risk=model_pass)
        with _flags(), _db(stop=None) as log_signal:
            out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                                  positions=POSITIONS)
        assert out is result
        row = _row(log_signal)
        assert row["stop_record"] is False and row["blocked_by"] == ["exit_no_stop_record"]
        assert trim_exit.decide(row, row["flags"]) == {k: row[k] for k in trim_exit.OUTPUT_KEYS}
    # The stop record unreadable: the rule owns it and withholds the sale.
    result = _result(action="EXIT", conviction=7, pass_to_risk=True)
    with _flags(), _db(get_position_stop=RuntimeError("pool")) as log_signal:
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
    assert out is not result and out["pass_to_risk"] is False
    assert _row(log_signal)["blocked_by"] == ["no_stop_record"]


@pytest.mark.parametrize("fn,block", [
    ("get_position_stop", "no_stop_record"),
    ("get_last_executed_approval", "min_hold"),
    ("sell_trades_since", "cooldown"),
    ("get_unconfirmed_orders", "pending_sell"),
    ("awaiting_fill_sell_tickers", "pending_sell"),
    ("trade_pending_sell_tickers", "pending_sell"),
    ("has_active_approval", "active_approval"),
    ("is_market_hours", "market_closed"),
])
def test_each_read_error_fails_closed(fn, block):
    result = _result(action="TRIM", pass_to_risk=True)
    with _flags(), _db(**{fn: RuntimeError("down")}) as log_signal:
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
    row = _row(log_signal)
    assert block in row["blocked_by"] and row["routed"] is False
    assert any(v.startswith("error:") for v in row["reads"].values())
    # The rule owns this verdict, so the model's own pass flag is withheld.
    assert out is not result and out["pass_to_risk"] is False and "trim_exit" not in out


def test_pending_sell_reads_todays_unconfirmed_sell_only():
    old = {"ticker": "NVDA", "side": "SELL", "created_at": NOW - timedelta(days=1)}
    today = {"ticker": "NVDA", "side": "SELL", "created_at": NOW - timedelta(minutes=5)}
    with patch("db.awaiting_fill_sell_tickers", return_value=set()), \
         patch("db.trade_pending_sell_tickers", return_value=set()):
        with patch("db.get_unconfirmed_orders", return_value=[old]):
            assert trim_exit._pending_sell("NVDA", NOW) == {"pending": False, "sources": []}
        with patch("db.get_unconfirmed_orders", return_value=[today]):
            assert trim_exit._pending_sell("NVDA", NOW)["sources"] == ["unconfirmed"]
    with patch("db.get_unconfirmed_orders", return_value=[]), \
         patch("db.awaiting_fill_sell_tickers", return_value={"NVDA"}), \
         patch("db.trade_pending_sell_tickers", return_value={"NVDA"}):
        assert trim_exit._pending_sell("NVDA", NOW)["sources"] == ["awaiting_fill", "trade_pending"]


# ── quantities ───────────────────────────────────────────────────────────────

def test_trim_qty_half_1_2_3_6_7_10():
    cfg = _cfg()
    assert [trim_exit.trim_qty(n, cfg) for n in (0.5, 1, 1.5, 2, 3, 6, 7, 10)] == [0, 0, 0, 1, 1, 1, 2, 3]
    assert [trim_exit.trim_qty(n, cfg) for n in (0, -3, None, "x", float("nan"))] == [0] * 5
    # Remaining 0: a one-share TRIM closes the position, as the model-flag TRIM does.
    assert trim_exit.trim_qty(1, _cfg(TRIM_EXIT_MIN_REMAINING_SHARES=0)) == 1
    # Fraction 0.5: 2 of 4, 5 of 10.
    half = _cfg(TRIM_EXIT_FRACTION=0.5)
    assert [trim_exit.trim_qty(n, half) for n in (2, 4, 10)] == [1, 2, 5]
    with _flags():
        assert trim_exit.trim_qty(7) == 2          # reads config when no snapshot is given


def test_one_share_trim_hold_and_exit():
    confirmed = [_verdict("TRIM", 0.01), _verdict("TRIM", 20)]
    hold = trim_exit.decide(_inputs(owned=1.0, verdicts=confirmed), _cfg())
    assert hold["would_action"] is None and hold["reason"] == "trim_one_share" and hold["routed"] is False
    exit_cfg = _cfg(TRIM_EXIT_ONE_SHARE_TRIM="exit")
    out = trim_exit.decide(_inputs(owned=1.0, verdicts=confirmed), exit_cfg)
    assert out["would_action"] == "EXIT" and out["would_qty"] == 1.0 and out["routed"] is True
    assert out["reason"] == "trim_one_share_exit" and out["remaining_qty"] == 0
    # Unconfirmed, "exit" still sells nothing: one share leaves only by a confirmed EXIT.
    out = trim_exit.decide(_inputs(owned=1.0), exit_cfg)
    assert out["would_action"] is None and out["reason"] == "trim_one_share"
    out = trim_exit.decide(_inputs(owned=1.0, action="EXIT"), exit_cfg)     # unconfirmed EXIT -> TRIM
    assert out["would_action"] is None
    # Any value other than "exit" reads as hold.
    out = trim_exit.decide(_inputs(owned=1.0, verdicts=confirmed), _cfg(TRIM_EXIT_ONE_SHARE_TRIM="tighten"))
    assert out["would_action"] is None and out["reason"] == "trim_one_share"


def test_one_share_exit_from_a_stale_snapshot_is_not_routed():
    """TRIM_EXIT_ONE_SHARE_TRIM=exit: the review's position snapshot held 1
    share, so a confirmed TRIM became an EXIT. 4 shares are held by routing
    (bought since), so a partial sale exists: the EXIT is not routed, and
    nothing is sold."""
    from agents import crew
    rows = [{"timestamp": NOW - timedelta(minutes=1), "signal": "TRIM", "conviction": 6},
            {"timestamp": NOW - timedelta(hours=20), "signal": "TRIM", "conviction": 6}]
    snapshot = {"NVDA": {"qty": 1.0, "entry_price": 100.0, "current_price": 110.0}}
    with _flags(TRIM_EXIT_ONE_SHARE_TRIM="exit"), _db(verdicts=rows):
        routed = trim_exit.apply(_result(action="TRIM"), path="earnings_review", event=PRE_EVENT,
                                 earnings_date=PRINT, positions=snapshot)
    assert routed["action"] == "EXIT" and routed["pass_to_risk"] is True
    assert routed["trim_exit"]["reason"] == "trim_one_share_exit"
    client = MagicMock()
    client.get_all_positions.return_value = [MagicMock(
        symbol="NVDA", qty=4.0, avg_entry_price=100.0, current_price=110.0, market_value=440.0)]
    with _flags(TRIM_EXIT_ONE_SHARE_TRIM="exit"), \
         patch("db.has_active_approval", return_value=False), \
         patch("db.log_pending_approval") as lpa, patch("db.log_signal") as log_signal, \
         patch("execution.execute_approved") as execute, \
         patch("agents.crew._get_price", return_value=110.0):
        crew._route_to_approval(routed, {"regime": "Neutral"}, client)
    lpa.assert_not_called()
    execute.assert_not_called()
    assert log_signal.call_args.args[1] == "ROUTE_ZERO_SHARES"
    assert log_signal.call_args.kwargs["decision"]["verdict_action"] == "TRIM"


def test_legacy_qty_recorded_beside_the_rules():
    out = trim_exit.decide(_inputs(owned=2.0), _cfg())
    assert out["legacy_qty"] == 1 and out["would_qty"] == 1.0 and out["remaining_qty"] == 1.0
    out = trim_exit.decide(_inputs(owned=1.0), _cfg())
    assert out["legacy_qty"] == 1 and out["would_qty"] is None


# ── apply(): shadow, live, observe ───────────────────────────────────────────

def test_shadow_writes_row_result_unchanged_no_route():
    from agents import crew
    for model_pass in (False, True):
        result = _result(action="EXIT", pass_to_risk=model_pass)
        before = copy.deepcopy(result)
        with _flags(TRIM_EXIT_LIVE=False), _db() as log_signal, \
             patch("agents.crew._earnings_timing", return_value={"time_of_day": "AMC"}), \
             patch("agents.crew._earnings_review_event",
                   side_effect=lambda t, d: {**PRE_EVENT, "earnings_phase": "pre"}), \
             patch("agents.crew._analyze_held_position", return_value=result), \
             patch("agents.crew._capture_latest_thesis"), \
             patch("agents.crew._route_guarded") as route, \
             patch("agents.crew._c6_bar_unrouted"):
            out = crew.review_held_position("NVDA", None, POSITIONS, {}, earnings_date=PRINT)
        assert out is result and result == before
        if model_pass:
            route.assert_called_once()
            assert route.call_args.args[0] is result      # the model's own pass, as before
        else:
            route.assert_not_called()
        row = _row(log_signal)
        call = [c for c in log_signal.call_args_list if c.args[1] == "TRIM_EXIT_RULE"][0]
        assert call.args[0] == "NVDA" and call.args[2] == 0.0
        assert call.kwargs["provenance"]["verdict_signal_id"] == 101
        assert call.kwargs["provenance"]["path"] == "earnings_review"
        assert "code_sha" in call.kwargs["provenance"]
        assert row["mode"] == "shadow" and row["blocked_by"][0] == "shadow"
        assert row["would_action"] == "TRIM" and row["routed"] is False
        assert row["options_shadow_key"] == f"{config.TRADING_ENV}:NVDA:2026-11-16"
        assert row["flags"]["TRIM_EXIT_LIVE"] is False and row["rule_version"] == trim_exit.RULE_VERSION


def test_hold_review_writes_a_row_for_the_denominator():
    result = _result(action="HOLD", pass_to_risk=False)
    with _flags(), _db() as log_signal, \
         patch("db.get_recent_verdicts") as verdicts:
        out = trim_exit.apply(result, path="scheduled_review", positions=POSITIONS)
    assert out is result
    row = _row(log_signal)
    assert row["reason"] == "not_sell_verdict" and row["context"] == "general"
    verdicts.assert_not_called()      # sell-only reads are skipped


def test_live_routes_copy_original_unmutated():
    result = _result(action="EXIT", pass_to_risk=False)
    before = copy.deepcopy(result)
    with _flags(), _db() as log_signal:
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
    assert result == before
    assert out is not result and out["pass_to_risk"] is True and out["action"] == "TRIM"
    assert out["provenance"] is result["provenance"]          # a shallow copy, nothing rewritten
    tag = out["trim_exit"]
    assert tag["row_id"] == 55 and tag["verdict_action"] == "EXIT" and tag["action"] == "TRIM"
    assert tag["context"] == "earnings_pre" and tag["confirmed"] is False and tag["would_qty"] == 3.0
    row = _row(log_signal)
    assert row["mode"] == "live" and row["routed"] is True and row["blocked_by"] == []


def test_held_review_routes_the_copy_and_returns_the_original():
    from agents import crew
    result = _result(action="EXIT", pass_to_risk=False)
    before = copy.deepcopy(result)
    with _flags(), _db(), \
         patch("agents.crew._earnings_timing", return_value={"time_of_day": "AMC"}), \
         patch("agents.crew._earnings_review_event", side_effect=lambda t, d: dict(PRE_EVENT)), \
         patch("agents.crew._analyze_held_position", return_value=result), \
         patch("agents.crew._capture_latest_thesis"), \
         patch("agents.crew._route_guarded") as route, \
         patch("agents.crew._c6_bar_unrouted") as unrouted:
        out = crew.review_held_position("NVDA", None, POSITIONS, {}, earnings_date=PRINT)
    assert out is result and result == before
    routed = route.call_args.args[0]
    assert routed is not result and routed["action"] == "TRIM" and routed["pass_to_risk"] is True
    assert routed["trim_exit"]["row_id"] == 55
    unrouted.assert_not_called()
    # Plan-level review: same hook, its own path; general context is not live.
    passed = _result(action="EXIT", pass_to_risk=True)
    with _flags(), _db() as log_signal, \
         patch("ibkr.IBKRClient"), \
         patch("agents.crew._get_positions", return_value=POSITIONS), \
         patch("agents.market_regime.get_cached_regime", return_value={}), \
         patch("agents.crew._analyze_held_position", return_value=passed), \
         patch("agents.crew._capture_latest_thesis"), patch("db.set_state"), \
         patch("agents.crew._route_guarded") as route:
        assert crew.review_plan_level("NVDA", "invalidation", 95.0, 96.0) is passed
    assert route.call_args.args[0] is passed
    row = _row(log_signal)
    assert row["path"] == "plan_level_review" and row["context"] == "general"


def test_plan_level_review_routes_the_rules_copy():
    """With general live, a plan-level review routes on the rule's copy: a
    model-passed, unconfirmed EXIT is withheld, and a confirmed TRIM the
    model did not pass is routed with the tag. The original comes back."""
    from agents import crew

    def run(result, verdicts):
        with _flags(TRIM_EXIT_LIVE_CONTEXTS=("general",)), _db(verdicts=verdicts), \
             patch("ibkr.IBKRClient"), \
             patch("agents.crew._get_positions", return_value=POSITIONS), \
             patch("agents.market_regime.get_cached_regime", return_value={}), \
             patch("agents.crew._analyze_held_position", return_value=result), \
             patch("agents.crew._capture_latest_thesis"), patch("db.set_state"), \
             patch("agents.crew._route_guarded") as route, \
             patch("agents.crew._c6_bar_unrouted") as unrouted:
            assert crew.review_plan_level("NVDA", "invalidation", 95.0, 96.0) is result
        return route, unrouted

    passed = _result(action="EXIT", pass_to_risk=True)
    before = copy.deepcopy(passed)
    route, unrouted = run(passed, None)          # one EXIT verdict: unconfirmed
    route.assert_not_called()
    withheld = unrouted.call_args.args[0]
    assert withheld is not passed and withheld["pass_to_risk"] is False and passed == before
    trim = _result(action="TRIM", pass_to_risk=False)
    rows = [{"timestamp": NOW - timedelta(minutes=1), "signal": "TRIM", "conviction": 6},
            {"timestamp": NOW - timedelta(hours=20), "signal": "TRIM", "conviction": 6}]
    route, unrouted = run(trim, rows)
    unrouted.assert_not_called()
    sent = route.call_args.args[0]
    assert sent is not trim and sent["pass_to_risk"] is True and sent["action"] == "TRIM"
    assert sent["trim_exit"]["context"] == "general" and sent["trim_exit"]["confirmed"] is True
    assert trim["pass_to_risk"] is False and "trim_exit" not in trim


def test_live_unwritten_row_routes_nothing():
    result = _result(action="TRIM", pass_to_risk=True)
    with _flags(), _db(log_id=None):
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
    assert out["pass_to_risk"] is False and "trim_exit" not in out


def test_live_context_rule_owns_pass_flag_both_ways():
    live_all = ("earnings_pre", "general")
    # Adds a sale the model did not pass.
    with _flags(), _db():
        out = trim_exit.apply(_result(action="TRIM", pass_to_risk=False), path="earnings_review",
                              event=PRE_EVENT, earnings_date=PRINT, positions=POSITIONS)
    assert out["pass_to_risk"] is True and out["action"] == "TRIM"
    # Withholds one the model passed: general context, unconfirmed.
    result = _result(action="EXIT", pass_to_risk=True)
    with _flags(TRIM_EXIT_LIVE_CONTEXTS=live_all), _db() as log_signal:
        out = trim_exit.apply(result, path="scheduled_review", positions=POSITIONS)
    assert out is not result and out["pass_to_risk"] is False and result["pass_to_risk"] is True
    assert _row(log_signal)["reason"] == "wait_confirm"
    # Withholds one the model passed when a block applies.
    with _flags(), _db() as log_signal, patch("db.has_active_approval", return_value=True):
        out = trim_exit.apply(_result(action="EXIT", pass_to_risk=True), path="earnings_review",
                              event=PRE_EVENT, earnings_date=PRINT, positions=POSITIONS)
    assert out["pass_to_risk"] is False and _row(log_signal)["blocked_by"] == ["active_approval"]


def test_non_live_context_keeps_legacy_routing():
    for model_pass in (True, False):
        result = _result(action="EXIT", pass_to_risk=model_pass)
        with _flags(), _db() as log_signal:      # live contexts: earnings_pre only
            out = trim_exit.apply(result, path="scheduled_review", positions=POSITIONS)
        assert out is result and out["pass_to_risk"] is model_pass
        row = _row(log_signal)
        assert row["context"] == "general" and "context_not_live" in row["blocked_by"]
        assert row["owns_pass_flag"] is False


def test_earnings_day_is_its_own_context():
    event = {**PRE_EVENT, "earnings_phase": "day", "earnings_timing": {"time_of_day": "BMO"}}
    for contexts, owned in ((("earnings_pre",), False), (("earnings_day",), True)):
        result = _result(action="TRIM", pass_to_risk=True)
        with _flags(TRIM_EXIT_LIVE_CONTEXTS=contexts), _db() as log_signal:
            out = trim_exit.apply(result, path="earnings_review", event=event,
                                  earnings_date=NOW.date(), positions=POSITIONS)
        row = _row(log_signal)
        assert row["context"] == "earnings_day" and row["days_to_print"] == 0
        assert row["earnings_time_of_day"] == "BMO" and row["owns_pass_flag"] is owned
        # Like general: one unconfirmed verdict sells nothing.
        assert row["would_action"] is None and row["reason"] == "wait_confirm"
        assert (out is result) is (not owned)
        assert out["pass_to_risk"] is (not owned)


def test_decide_recomputes_from_logged_row():
    cases = [
        (_result(action="EXIT"), {}, {}),
        (_result(action="TRIM"), {"TRIM_EXIT_LIVE": False}, {}),
        (_result(action="EXIT"), {}, {"get_last_executed_approval": RuntimeError("x")}),
    ]
    decided = []
    for result, flag_over, raises in cases:
        with _flags(**flag_over), _db(**raises) as log_signal, \
             patch("db.get_state", return_value=json.dumps(
                 {"NVDA": {"et_date": "2026-11-16", "implied_move_pct": 6.0, "expiry": "2026-11-20"}})):
            trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                            positions=POSITIONS)
        row = _row(log_signal)
        decided.append(row)
        again = trim_exit.decide(row, row["flags"])
        assert again == {k: row[k] for k in trim_exit.OUTPUT_KEYS}
        # The record-only fields are not inputs: changed or absent, the
        # decision is the same.
        assert row["implied_move_last_pct"] == 6.0 and row["verdict_pass_to_risk"] is False
        assert "model_pass_to_risk" not in row
        assert trim_exit.decide({**row, "implied_move_last_pct": 99.0, "gap_vs_stop": 99.0},
                                row["flags"]) == again
        record_only = ("implied_move_last_pct", "implied_move_last_date", "implied_move_last_dte",
                       "gap_vs_stop", "verdict_pass_to_risk", "earnings_calendar_date",
                       "earnings_calendar_confirmed", "options_shadow_key", "unknown_live_contexts",
                       "implied_move_event_pct", "implied_move_event_expiry", "implied_move_event_dte",
                       "implied_move_event_earnings_date", "implied_move_event_date",
                       "gap_vs_stop_event", "gap_vs_stop_basis")
        assert trim_exit.decide({**row, "implied_move_event_pct": 99.0, "gap_vs_stop_event": 99.0,
                                 "gap_vs_stop_basis": "event_spanning"}, row["flags"]) == again
        assert trim_exit.decide({**row, **{k: None for k in record_only}}, row["flags"]) == again
        assert trim_exit.decide({k: v for k, v in row.items() if k not in record_only},
                                row["flags"]) == again
    # What the cases decided: the first routes an unconfirmed earnings TRIM.
    assert [c["routed"] for c in decided] == [True, False, False]
    assert decided[0]["blocked_by"] == [] and decided[0]["would_action"] == "TRIM"
    assert decided[1]["blocked_by"] == ["shadow"]
    assert decided[2]["blocked_by"] == ["min_hold"] and decided[2]["owns_pass_flag"] is True


def test_implied_move_recorded_as_the_gap_shadow_reads_it():
    book = {"NVDA": {"et_date": "2026-11-13", "implied_move_pct": 4.5, "expiry": "2026-11-14"}}
    with _flags(), _db() as log_signal, patch("db.get_state", return_value=json.dumps(book)) as gs:
        trim_exit.apply(_result(action="HOLD"), path="scheduled_review", positions=POSITIONS)
    row = _row(log_signal)
    assert gs.call_args.args[0] == f"options_implied_move_{config.TRADING_ENV}"
    assert (row["implied_move_last_pct"], row["implied_move_last_date"], row["implied_move_last_dte"]) \
        == (4.5, "2026-11-13", 1)
    # stop 99 under a 110 price: 10% away.
    assert row["stop_distance_pct"] == 10.0 and row["gap_vs_stop"] == 0.45
    # A read that can't be parsed records None and raises nothing.
    with _flags(), _db() as log_signal, patch("db.get_state", return_value="{not json"):
        out = trim_exit.apply(_result(action="HOLD"), path="scheduled_review", positions=POSITIONS)
    row = _row(log_signal)
    assert out["action"] == "HOLD" and row["implied_move_last_pct"] is None and row["gap_vs_stop"] is None
    assert row["reads"]["implied_move"].startswith("error:")


def _event_state(event=8.25, ev_date="2026-11-16", expiry="2026-11-20", nearest=2.88,
                 et_date="2026-11-16"):
    rec = {"et_date": et_date, "implied_move_pct": nearest, "expiry": "2026-11-16"}
    if event is not None:
        rec["event_spanning"] = {"implied_move_pct": event, "expiry": expiry, "dte": 4,
                                 "earnings_date": "2026-11-18", "et_date": ev_date}
    return json.dumps({"NVDA": rec})


def _event_row(state, result=None, **apply_kw):
    kw = {"path": "earnings_review", "event": PRE_EVENT, "earnings_date": PRINT}
    kw.update(apply_kw)
    with _flags(TRIM_EXIT_LIVE=False), _db() as log_signal, patch("db.get_state", return_value=state):
        trim_exit.apply(result or _result(action="HOLD"), positions=POSITIONS, **kw)
    return _row(log_signal)


def test_gap_vs_stop_event_when_the_print_is_on_or_before_the_event_expiry():
    row = _event_row(_event_state())
    # stop 99 under a 110 price: 10% away. gap_vs_stop keeps the nearest move.
    assert row["stop_distance_pct"] == 10.0 and row["gap_vs_stop"] == 0.288
    assert row["implied_move_event_pct"] == 8.25 and row["implied_move_event_expiry"] == "2026-11-20"
    assert row["implied_move_event_dte"] == 4 and row["implied_move_event_earnings_date"] == "2026-11-18"
    assert row["implied_move_event_date"] == "2026-11-16"
    assert row["gap_vs_stop_event"] == 0.825 and row["gap_vs_stop_basis"] == "event_spanning"
    on_expiry = _event_row(_event_state(expiry=PRINT.isoformat()))
    assert on_expiry["gap_vs_stop_basis"] == "event_spanning"


def test_gap_vs_stop_basis_is_nearest_after_the_event_expiry_or_without_a_move():
    after = _event_row(_event_state(expiry="2026-11-17"))         # print 11-18 is after it
    assert after["gap_vs_stop_basis"] == "nearest" and after["gap_vs_stop_event"] == 0.825
    assert after["gap_vs_stop"] == 0.288
    missing = _event_row(_event_state(event=None))
    assert missing["gap_vs_stop_basis"] == "nearest" and missing["gap_vs_stop_event"] is None
    assert missing["implied_move_event_pct"] is None and missing["gap_vs_stop"] == 0.288
    no_print = _event_row(_event_state(), path="scheduled_review", event=None, earnings_date=None)
    assert no_print["gap_vs_stop_basis"] == "nearest" and no_print["gap_vs_stop_event"] == 0.825
    # An event move from a later date than the evaluation is not read.
    future = _event_row(_event_state(ev_date="2026-11-17", et_date="2026-11-17"))
    assert future["implied_move_event_pct"] is None and future["gap_vs_stop_basis"] == "nearest"
    # A pre-open read of an earlier date's entry keeps its event move.
    earlier = _event_row(_event_state(ev_date="2026-11-13", et_date="2026-11-13"))
    assert earlier["implied_move_event_date"] == "2026-11-13"
    assert earlier["gap_vs_stop_basis"] == "event_spanning"
    # An old-format entry (no event key) and an unreadable state.
    old = _event_row(json.dumps({"NVDA": {"et_date": "2026-11-16", "implied_move_pct": 2.88}}))
    assert old["gap_vs_stop_basis"] == "nearest" and old["gap_vs_stop"] == 0.288
    broken = _event_row("{not json")
    assert broken["gap_vs_stop_basis"] == "nearest" and broken["gap_vs_stop_event"] is None
    assert broken["reads"]["implied_move"].startswith("error:")


def test_event_move_fields_change_no_decision():
    result = _result(action="TRIM")
    rows = [_event_row(state, result=result) for state in
            (_event_state(), _event_state(event=None), _event_state(expiry="2026-11-17"))]
    outs = [{k: r[k] for k in trim_exit.OUTPUT_KEYS} for r in rows]
    assert outs[0] == outs[1] == outs[2]
    assert trim_exit._gap_vs_stop_event(None, PRINT) == \
        {"gap_vs_stop_event": None, "gap_vs_stop_basis": "error"}
    assert trim_exit._gap_vs_stop_event({"implied_move_event_pct": object()}, PRINT) == \
        {"gap_vs_stop_event": None, "gap_vs_stop_basis": "nearest"}


def test_exempt_recorded_not_routed():
    result = _result(action="TRIM", pass_to_risk=False)
    with _flags(TRIM_EXIT_EXEMPT_TICKERS=("NVDA",)), _db() as log_signal:
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
    assert out is result
    row = _row(log_signal)
    assert row["blocked_by"] == ["exempt"] and row["would_action"] == "TRIM" and row["routed"] is False
    # The model's own pass still routes as before (exemption covers the rule only).
    passed = _result(action="TRIM", pass_to_risk=True)
    with _flags(TRIM_EXIT_EXEMPT_TICKERS=("NVDA",)), _db():
        assert trim_exit.apply(passed, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                               positions=POSITIONS) is passed


def test_apply_exception_live_fails_closed():
    result = _result(action="EXIT", pass_to_risk=True)
    before = copy.deepcopy(result)
    with _flags(), _db(), patch.object(trim_exit, "_gather", side_effect=RuntimeError("boom")):
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT,
                              positions=POSITIONS)
        assert out is not result and out["pass_to_risk"] is False and result == before
        # Known not to own it: the original comes back.
        assert trim_exit.apply(result, path="scheduled_review", positions=POSITIONS) is result
        assert trim_exit.apply(result, path="event", observe=True) is result
        assert trim_exit.apply(_result(action="EXIT", conviction=3, pass_to_risk=True),
                               path="earnings_review", event=PRE_EVENT, earnings_date=PRINT)["pass_to_risk"]
    with _flags(TRIM_EXIT_LIVE=False), _db(), patch.object(trim_exit, "_gather", side_effect=RuntimeError):
        assert trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT) is result
    with _flags(), _db(), patch.object(trim_exit, "decide", side_effect=KeyError("x")):
        out = trim_exit.apply(result, path="earnings_review", event=PRE_EVENT, earnings_date=PRINT)
        assert out["pass_to_risk"] is False
    # Nothing escapes, whatever comes in.
    with _flags(), _db():
        assert trim_exit.apply(None, path="x") is None
        assert trim_exit.apply("junk", path="x") == "junk"
    with _flags(), _db(), patch.object(trim_exit, "_log_row", side_effect=RuntimeError("db")):
        out = trim_exit.apply(_result(action="EXIT"), path="earnings_review", event=PRE_EVENT,
                              earnings_date=PRINT, positions=POSITIONS)
    assert out["pass_to_risk"] is False and "trim_exit" not in out     # no row, no sale


def test_observe_site_never_changes_routing():
    from agents import crew
    live_all = ("earnings_pre", "earnings_day", "general")
    for model_pass in (True, False):
        result = _result(action="EXIT", pass_to_risk=model_pass,
                         signal_id=101) | {"held_review": True}
        hold = _result(action="HOLD", ticker="AMD")
        client = MagicMock()
        with _flags(TRIM_EXIT_LIVE_CONTEXTS=live_all), _db() as log_signal, \
             patch("ibkr.IBKRClient", return_value=client), \
             patch("agents.crew._get_positions", return_value=POSITIONS), \
             patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
             patch("agents.analyst.analyze_batch", return_value=[result, hold]), \
             patch("agents.scan_guard.log_lost_candidates"), \
             patch("agents.crew._route_guarded") as route, \
             patch("agents.crew._c6_bar_unrouted") as unrouted:
            crew.process_candidates([{"ticker": "NVDA", "priority": "HIGH"},
                                     {"ticker": "AMD", "priority": "HIGH"}])
        if model_pass:
            assert route.call_args.args[0] is result
        else:
            route.assert_not_called()
            assert unrouted.call_args_list[0].args[0] is result
        row = _row(log_signal)                    # one row: the EXIT, none for the HOLD
        assert row["mode"] == "observe" and row["path"] == "daily_scan"
        assert "observe" in row["blocked_by"] and row["owns_pass_flag"] is False
        # The event path, the same way.
        event = {"ticker": "NVDA", "source": "news", "title": "t", "text": "x"}
        with _flags(TRIM_EXIT_LIVE_CONTEXTS=live_all), _db() as log_signal, \
             patch("ibkr.IBKRClient"), \
             patch("agents.crew._get_positions", return_value=POSITIONS), \
             patch("agents.crew._get_price", return_value=110.0), \
             patch("context_engine.assemble_brief", return_value="BRIEF"), \
             patch("agents.analyst.analyze", return_value=result), \
             patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
             patch("knowledge_loader.get_watchlist_entry", return_value={}), \
             patch("agents.crew._route_guarded") as route, \
             patch("agents.crew._c6_bar_unrouted") as unrouted:
            assert crew.deep_analyze(dict(event), {"relevant": True}) is result
        if model_pass:
            assert route.call_args.args[0] is result
        else:
            route.assert_not_called()
            assert unrouted.call_args.args[0] is result
        row = _row(log_signal)
        assert row["mode"] == "observe" and row["path"] == "event" and row["routed"] is False
    # Called directly: the original back, every time.
    with _flags(TRIM_EXIT_LIVE_CONTEXTS=live_all), _db() as log_signal:
        r = _result(action="TRIM", pass_to_risk=False)
        assert trim_exit.apply(r, path="event", event={"title": "x"}, positions=POSITIONS,
                               observe=True) is r
        h = _result(action="HOLD")
        assert trim_exit.apply(h, path="event", positions=POSITIONS, observe=True) is h
    assert len([c for c in log_signal.call_args_list if c.args[1] == "TRIM_EXIT_RULE"]) == 1


# ── execution recheck ────────────────────────────────────────────────────────

TAG = {"row_id": 55, "context": "earnings_pre", "verdict_action": "TRIM", "action": "TRIM",
       "reason": "earnings_trim"}


def test_execution_block_live_off_exempt_trim_would_close():
    block = trim_exit.execution_block
    with _flags(TRIM_EXIT_LIVE=False):
        assert block("NVDA", "EXIT", 10, 10, TAG) == "switched off since routing"
    with _flags(TRIM_EXIT_ENABLED=False):
        assert block("NVDA", "TRIM", 1, 10, TAG) == "switched off since routing"
    with _flags(TRIM_EXIT_EXEMPT_TICKERS=("NVDA",)):
        assert block("nvda", "TRIM", 1, 10, TAG) == "ticker exempt since routing"
    with _flags():
        assert block("NVDA", "TRIM", 2, 2, TAG) == "trim would leave too few shares"
        assert block("NVDA", "TRIM", 1, 1, TAG) == "trim would leave too few shares"
        assert block("NVDA", "TRIM", None, 2, TAG) == "quantity unreadable"
        assert block("NVDA", "TRIM", 1, 2, TAG) is None
        assert block("NVDA", "EXIT", 2, 2, {**TAG, "action": "EXIT", "reason": "confirmed"}) is None
    with _flags(TRIM_EXIT_MIN_REMAINING_SHARES=0):
        assert block("NVDA", "TRIM", 1, 1, TAG) is None


def test_execution_block_context_no_longer_live():
    """Narrowing TRIM_EXIT_LIVE_CONTEXTS stops a rule sale already queued; a
    tag with no context fails the check."""
    with _flags(TRIM_EXIT_LIVE_CONTEXTS=()):
        assert trim_exit.execution_block("NVDA", "TRIM", 1, 10, TAG) == "context not live"
    with _flags(TRIM_EXIT_LIVE_CONTEXTS=("general",)):
        assert trim_exit.execution_block("NVDA", "EXIT", 10, 10, TAG) == "context not live"
    with _flags():
        for tag in ({"row_id": 55}, None, "x"):
            assert trim_exit.execution_block("NVDA", "TRIM", 1, 10, tag) == "context not live"
        assert trim_exit.execution_block("NVDA", "TRIM", 1, 10, TAG) is None


def test_execution_block_one_share_exit_needs_one_share():
    """A one-share EXIT (TRIM_EXIT_ONE_SHARE_TRIM=exit) is skipped when the
    position now allows a partial sale: shares bought since the review."""
    tag = {**TAG, "action": "EXIT", "reason": "trim_one_share_exit"}
    with _flags(TRIM_EXIT_ONE_SHARE_TRIM="exit"):
        assert trim_exit.execution_block("NVDA", "EXIT", 1, 1, tag) is None
        assert (trim_exit.execution_block("NVDA", "EXIT", 4, 4, tag)
                == "one-share exit, but a partial sale exists now")
        assert trim_exit.execution_block("NVDA", "EXIT", 4, None, tag) == "quantity unreadable"
        # A confirmed EXIT verdict sells everything held, whatever the count.
        assert trim_exit.execution_block("NVDA", "EXIT", 4, 4, {**tag, "reason": "confirmed"}) is None
    # TRIM_EXIT_ONE_SHARE_TRIM set back to hold after routing: the queued
    # one-share EXIT is skipped; a confirmed EXIT verdict is not.
    for setting in ("hold", "tighten"):
        with _flags(TRIM_EXIT_ONE_SHARE_TRIM=setting):
            assert (trim_exit.execution_block("NVDA", "EXIT", 1, 1, tag)
                    == "one-share exit no longer enabled")
            assert trim_exit.execution_block("NVDA", "EXIT", 1, 1, {**tag, "reason": "confirmed"}) is None


def test_unknown_live_context_is_recorded():
    """A name in TRIM_EXIT_LIVE_CONTEXTS that is no context is never live,
    and the row says so."""
    with _flags(TRIM_EXIT_LIVE_CONTEXTS=("earnings-pre",)), _db() as log_signal:
        out = trim_exit.apply(_result(action="TRIM", pass_to_risk=False), path="earnings_review",
                              event=PRE_EVENT, earnings_date=PRINT, positions=POSITIONS)
    row = _row(log_signal)
    assert out["pass_to_risk"] is False and "context_not_live" in row["blocked_by"]
    assert row["unknown_live_contexts"] == ["earnings-pre"]
    with _flags(), _db() as log_signal:
        trim_exit.apply(_result(action="TRIM"), path="earnings_review", event=PRE_EVENT,
                        earnings_date=PRINT, positions=POSITIONS)
    assert _row(log_signal)["unknown_live_contexts"] == []


def test_rule_kind_is_kept_out_of_verdict_reads():
    import db
    assert "TRIM_EXIT_RULE" in db._NON_ANALYST_SIGNALS and "TRIM_EXIT_RULE" in db._RECORD_ONLY_SIGNALS
    assert trim_exit.KIND == "TRIM_EXIT_RULE"
