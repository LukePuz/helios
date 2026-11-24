"""agents/crew.py:_read_through_record: one queued read_through row naming the
held names an analysed event bears on, from the brief's own related-symbol
mapping; deduped per (source item, held ticker); never raises; and
scripts/batch_verify.py's conditional, WARN-only read_through stage."""
import importlib.util
import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

import config
import context_engine
from agents import crew

_BV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "batch_verify.py")
_spec = importlib.util.spec_from_file_location("batch_verify_read_through", _BV_PATH)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)

SYMBOLS = ["NVDA", "AMD", "AVGO", "MU", "AMAT"]
EVENT = {"ticker": "NVDA", "source": "news", "title": "Nvidia China export curbs widen",
         "url": "https://example.com/a", "text": "x"}
BRIEF = ("=== EVENT ===\nNVDA ...\n\n=== RELATED SYMBOLS IN WATCHLIST ===\n"
         "NVDA may have indirect effects on: AMD, AVGO, MU, AMAT\n"
         "Consider second-order impact on these positions.")
RESULT = {"action": "HOLD", "conviction": 5, "provenance": {"signal_id": 4242}}


def _pos(*held, zero=()):
    out = {t: {"qty": 1.0, "entry_price": 1.0, "current_price": 1.0} for t in held}
    out.update({t: {"qty": 0.0, "entry_price": 1.0, "current_price": 1.0} for t in zero})
    return out


def _run(positions, event=EVENT, brief=BRIEF, brief_error=None, result=RESULT):
    with patch.object(config, "SYMBOLS", SYMBOLS), \
         patch("db.log_activity_queued") as queued:
        extra = crew._read_through_record(event, event["ticker"], positions,
                                          "2026-11-24T15:00:00+00:00", brief, brief_error, result)
    return extra, queued


@pytest.fixture
def on():
    with patch.object(config, "READ_THROUGH_RECORD_ENABLED", True, create=True):
        yield


def test_a_held_related_name_gets_one_queued_row(on):
    extra, queued = _run(_pos("AMD", "NVDA", zero=("MU",)))
    queued.assert_called_once()
    stage, ticker, msg, sentiment, row = queued.call_args[0]
    assert (stage, ticker, sentiment) == ("read_through", "NVDA", "neutral")
    assert row is extra and "AMD" in msg
    assert row["event_ticker"] == "NVDA" and row["signal_id"] == 4242
    assert row["related"] == [{"ticker": "AMD", "held": True}, {"ticker": "AVGO", "held": False},
                              {"ticker": "AMAT", "held": False}, {"ticker": "MU", "held": False}]
    assert row["held_related"] == ["AMD"] and row["held_related_deduped"] == []
    assert row["item"]["item_id"] == "https://example.com/a"
    assert row["held_as_of"] == "2026-11-24T15:00:00+00:00"
    assert row["section_in_brief"] is True and row["section_within_char_cap"] is True
    assert row["dedupe_read"] == "ok" and row["action"] == "HOLD" and row["conviction"] == 5
    assert row["env"] == config.TRADING_ENV
    for k in bv.READ_THROUGH_REQUIRED:
        assert k in row, k


def test_the_mapping_is_the_briefs_own_function_called_as_the_brief_calls_it(on):
    real = context_engine._find_related_symbols
    with patch("context_engine._find_related_symbols", side_effect=real) as spy:
        extra, _ = _run(_pos("AMD"))
    spy.assert_called_once_with("NVDA", SYMBOLS)
    assert [r["ticker"] for r in extra["related"]] == real("NVDA", SYMBOLS)
    # No forked copy of the mapping in crew.py.
    assert '"AVGO", "AMAT"' not in inspect.getsource(crew)


def test_the_heading_read_is_the_one_the_brief_writes():
    src = inspect.getsource(context_engine)
    assert src.count(crew._RELATED_SECTION_HEADING) == 2   # event and scan briefs


def test_deduped_per_source_item_and_held_ticker(on):
    first, q1 = _run(_pos("AMD"))
    again, q2 = _run(_pos("AMD"))
    assert first is not None and again is None and not q2.called
    later, q3 = _run(_pos("AMD", "AVGO"))
    assert later["held_related"] == ["AVGO"] and later["held_related_deduped"] == ["AMD"]
    other = dict(EVENT, url="https://example.com/b")
    new_item, _ = _run(_pos("AMD"), event=other)
    assert new_item["held_related"] == ["AMD"]


def test_redis_unreadable_writes_the_row_undeduped(on):
    with patch("stop_lock._redis", side_effect=ConnectionError("down")):
        extra, queued = _run(_pos("AMD"))
    queued.assert_called_once()
    assert extra["dedupe_read"] == "error: ConnectionError" and extra["held_related"] == ["AMD"]


def test_an_item_without_an_id_is_not_deduped(on):
    bare = {"ticker": "NVDA", "source": "news", "text": "x"}
    a, _ = _run(_pos("AMD"), event=bare)
    b, _ = _run(_pos("AMD"), event=bare)
    assert a["dedupe_read"] == b["dedupe_read"] == "no_item_id"


@pytest.mark.parametrize("positions,kw", [
    ({}, {}),                                               # nothing held
    (_pos("NVDA", "QCOM"), {}),                             # held, but not related
    (_pos(zero=("AMD",)), {}),                              # held count 0
    (_pos("AMD"), {"brief_error": "RuntimeError"}),         # the book was not read
    (None, {}),
])
def test_no_row_without_a_held_related_name(on, positions, kw):
    extra, queued = _run(positions, **kw)
    assert extra is None and not queued.called


def test_no_row_for_a_ticker_with_no_related_symbols(on):
    ev = dict(EVENT, ticker="AVGO")
    extra, queued = _run(_pos("NVDA", "AMD"), event=ev)
    assert extra is None and not queued.called


@pytest.mark.switches(READ_THROUGH_RECORD_ENABLED=False)
def test_switch_off_writes_nothing():
    extra, queued = _run(_pos("AMD"))
    assert extra is None and not queued.called


def test_the_section_past_the_char_cap_is_recorded(on):
    from agents import analyst
    long_brief = "x" * (analyst._BRIEF_MAX_CHARS + 10) + "\n" + BRIEF
    extra, _ = _run(_pos("AMD"), brief=long_brief)
    assert extra["section_in_brief"] is True and extra["section_within_char_cap"] is False
    extra, _ = _run(_pos("AMD"), event=dict(EVENT, url="u2"), brief="no section")
    assert extra["section_in_brief"] is False


def test_never_raises(on):
    with patch("context_engine._find_related_symbols", side_effect=RuntimeError("boom")):
        assert _run(_pos("AMD"))[0] is None
    with patch.object(config, "SYMBOLS", SYMBOLS), \
         patch("db.log_activity_queued", side_effect=RuntimeError("queue")):
        assert crew._read_through_record(EVENT, "NVDA", _pos("AMD"), None, BRIEF, None,
                                         RESULT) is None
    assert _run(_pos("AMD"), event=dict(EVENT, url="u3"), result=None)[0]["signal_id"] is None


def test_a_soft_time_limit_is_re_raised(on):
    exc = pytest.importorskip("celery.exceptions").SoftTimeLimitExceeded
    with patch("context_engine._find_related_symbols", side_effect=exc()):
        with pytest.raises(exc):
            _run(_pos("AMD"))


# ── deep_analyze integration ────────────────────────────────────────────────

def _deep(positions):
    client = MagicMock()
    seen = {}

    def fake_analyze(candidate, situation_brief="", regime_packet=None, trigger=None):
        seen["brief"] = situation_brief
        return dict(RESULT, ticker="NVDA", pass_to_risk=False)
    with patch.object(config, "SYMBOLS", SYMBOLS), \
         patch("ibkr.IBKRClient", return_value=client), \
         patch("context_engine.assemble_brief", return_value=BRIEF), \
         patch("agents.crew._get_positions", return_value=positions) as get_pos, \
         patch("agents.crew._get_price", return_value=100.0), \
         patch("agents.analyst.analyze", side_effect=fake_analyze) as analyze, \
         patch("agents.market_regime.get_cached_regime", return_value={"regime": "Neutral"}), \
         patch("knowledge_loader.get_watchlist_entry", return_value={}), \
         patch("agents.crew._c6_bar_unrouted"), \
         patch("agents.crew.review_held_position") as review, \
         patch("db.log_activity_queued") as queued:
        result = crew.deep_analyze(dict(EVENT), {"relevant": True})
    return result, queued, get_pos, analyze, review, client, seen


def test_deep_analyze_queues_the_row_with_the_verdicts_signal_id(on):
    result, queued, get_pos, analyze, review, client, seen = _deep(_pos("AMD", "MU"))
    rows = [c for c in queued.call_args_list if c[0][0] == "read_through"]
    assert len(rows) == 1
    row = rows[0][0][4]
    assert row["signal_id"] == 4242 and row["held_related"] == ["AMD", "MU"]
    assert seen["brief"] == BRIEF                    # the brief is unchanged
    assert analyze.call_count == 1 and result["action"] == "HOLD"
    assert get_pos.call_count == 1                   # the one book read the brief used
    assert not client.get_all_positions.called       # no new broker call
    assert not review.called                         # no held review is queued


def test_deep_analyze_survives_a_failing_record(on):
    with patch("context_engine._find_related_symbols", side_effect=RuntimeError("boom")):
        result, queued, *_ = _deep(_pos("AMD"))
    assert result["action"] == "HOLD"
    assert not [c for c in queued.call_args_list if c[0][0] == "read_through"]


# ── batch_verify ────────────────────────────────────────────────────────────

def _bv_row(i, **over):
    extra = {k: None for k in bv.READ_THROUGH_REQUIRED}
    extra.update({"env": "live", "event_ticker": "NVDA", "dedupe_read": "ok",
                  "section_within_char_cap": True, **over})
    return {"id": i, "stage": "read_through", "ticker": "NVDA", "extra": extra}


def test_batch_verify_no_row_is_warn():
    chk = bv.check_read_through([], "live")
    assert chk["status"] == "WARN" and "conditional" in chk["why"]


def test_batch_verify_good_rows_ok_and_other_env_ignored():
    rows = [_bv_row(1), _bv_row(2), _bv_row(3, env="paper", dedupe_read="error: X")]
    chk = bv.check_read_through(rows, "live")
    assert chk["status"] == "OK" and chk["rows"] == 2


@pytest.mark.parametrize("row", [
    {"id": 9, "stage": "read_through", "ticker": "NVDA", "extra": {"env": "live"}},
    _bv_row(9, dedupe_read="error: ConnectionError"),
    _bv_row(9, section_within_char_cap=False),
])
def test_batch_verify_problems_are_warn_never_fail(row):
    chk = bv.check_read_through([_bv_row(1), row], "live")
    assert chk["status"] == "WARN"


def test_batch_verify_fetches_and_runs_the_stage():
    src = inspect.getsource(bv._fetch)
    assert "READ_THROUGH_STAGE" in src
    assert "check_read_through(activity, env)" in inspect.getsource(bv.run_checks)
