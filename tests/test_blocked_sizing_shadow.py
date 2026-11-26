"""Two record-only shadows on blocked or repeated work.

1. would_be_sizing (agents/crew.py:_would_be_sizing_record) on the route's
   ENTRY_GATE_BLOCKED row: the stop calculate_stop would set on cached bars,
   R against it, the volatility weight and the book's open risk, marked
   hypothetical. It never changes the gate's outcome or its row's gate
   fields, and carries no price, dollar or share count.
2. story_dedup.event_cluster (agents/story_dedup.py): title-token Jaccard
   against the earlier event verdicts the de-dup shadow already reads.

Synthetic data only: no DB, broker or Redis."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
import risk_manager
from agents import crew, story_dedup as sd

_REAL_WOULD_BE_STOP = crew._would_be_stop
NOW = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _gates_on(monkeypatch):
    monkeypatch.setattr(config, "STOPOUT_REENTRY_GATE_DAYS", 10)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_ENABLED", True)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS", 5)
    monkeypatch.setattr(config, "ENTRY_TREND_GATE_MARKET_TICKER", "")
    monkeypatch.setattr(config, "BLOCKED_SIZING_SHADOW_ENABLED", True)
    monkeypatch.setattr(config, "RR_VS_STOP_LABEL_THRESHOLDS", (1.5, 2.0))
    monkeypatch.setattr(config, "VOL_TARGET_CONTRIBUTION", 0.02)
    monkeypatch.setattr(config, "VOL_TARGET_WINDOW_SESSIONS", 20)


def _result(action="OPEN", bucket="Accelerator", target="$130.00", udr=2.0):
    return {"ticker": "NVDA", "action": action, "conviction": 7, "thesis_bucket": bucket,
            "reasoning": "r", "pass_to_risk": True, "upside_downside_ratio": udr,
            "valuation": {"bull_case_target": target, "downside_scenario": "$85.00"},
            "provenance": {"prompt_sha": "abc", "model": "m"}}


def _trend(last_close=100.0, below=True):
    return {"read": "ok", "bar_age_days": 1, "below_sma50": below, "below_sma200": below,
            "pct_vs_sma50": -30.1 if below else 4.0, "pct_vs_sma200": -28.2 if below else 9.0,
            "last_close": last_close, "last_bar_date": "2026-11-20"}


def _stop(hours=50.0):
    return {"read": "ok", "hours_since": hours, "exit_at": "2026-11-21T13:00:00+00:00"}


def _vol_bars(n=21, step=0.01):
    # Alternating +/- step log moves: a known, non-zero realised volatility.
    import math
    closes, c = [], 100.0
    for i in range(n):
        closes.append(c)
        c = c * math.exp(step if i % 2 == 0 else -step)
    # Dated back from yesterday (the real date): the record's stale_sessions
    # counts sessions since the last bar, so fixed dates made it grow with the
    # real clock, and on 2027-02-25 it read 90, the banned stop value.
    from datetime import date, timedelta
    end = date.today() - timedelta(days=1)
    return [{"date": (end - timedelta(days=n - 1 - i)).isoformat(), "close": x}
            for i, x in enumerate(closes)]


def _heat_inputs():
    return {"positions": {"MSFT": {"qty": 1.0, "market_value": 500.0}},
            "pv": 5000.0, "age_min": 3.0, "stops": [{"ticker": "MSFT", "stop_price": 450.0}]}


@pytest.fixture
def reads(monkeypatch):
    """The three reads would_be_sizing makes, faked: a 90.0 stop by an ATR
    multiple, 21 cached closes, one held position with a stop."""
    monkeypatch.setattr(crew, "_would_be_stop",
                        lambda t, p, b: (90.0, "2.0× ATR (5.00) below entry = $90.00", 5.0, 2.0))
    monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda ticker, window: _vol_bars())
    monkeypatch.setattr(crew, "_book_heat_inputs", _heat_inputs)


def _blocked_rows(ls):
    return [c for c in ls.call_args_list if c[0][1] == "ENTRY_GATE_BLOCKED"]


def _gate(result=None, trend=None, stop_rec=None, path=None):
    with patch("db.log_signal") as ls:
        out = crew._entry_gate("NVDA", result or _result(), stop_rec, _trend() if trend is None else trend,
                               path=path)
    return out, _blocked_rows(ls)


def _decision(row):
    return row[1]["decision"]


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


# ── would_be_sizing on the block row ─────────────────────────────────────────

class TestWouldBeSizingOnTheRow:
    def test_block_row_carries_the_hypothetical_record(self, reads):
        out, rows = _gate()
        assert out is True
        assert len(rows) == 1
        wbs = _decision(rows[0])["would_be_sizing"]
        assert wbs["hypothetical"] is True
        assert wbs["price_basis"] == "last_completed_close"
        assert wbs["price_bar_date"] == "2026-11-20"

    def test_stop_and_r_values(self, reads):
        wbs = _decision(_gate()[1][0])["would_be_sizing"]
        assert wbs["stop"]["kind"] == "atr_multiple"
        assert wbs["stop"]["stop_frac"] == 0.1
        assert wbs["stop"]["atr_multiple"] == 2.0
        assert wbs["rr"]["r_to_target"] == 3.0          # (130 - 100) / (100 - 90)
        assert wbs["rr"]["target_source"] == "bull_case_target"
        assert wbs["rr"]["stop_atr"] == 2.0              # (100 - 90) / 5
        assert wbs["rr"]["would_block"] == {"r_lt_1.5": False, "r_lt_2": False}
        assert wbs["rr"]["model_udr"] == 2.0
        assert wbs["rr"]["target_frac"] == 0.3            # (130 - 100) / 100
        assert wbs["rr"]["floor_vs_stop_frac"] == -0.05   # (85 - 90) / 100
        assert "stop_pct" not in wbs["rr"]
        assert "target_pct" not in wbs["rr"]

    def test_vol_weight_and_book_heat(self, reads):
        wbs = _decision(_gate()[1][0])["would_be_sizing"]
        vt = wbs["vol_target"]
        assert vt["read"] == "ok"
        assert vt["reason"] is None
        assert vt["vol_weight"] == round(0.02 / vt["rvol20"], 4)
        bh = wbs["book_heat"]
        assert bh["read"] == "ok"
        assert bh["sum_open_risk_frac"] == 0.01           # 1 x (500 - 450) / 5000
        assert bh["n_positions"] == 1
        assert bh["n_with_stop"] == 1
        assert bh["reason"] is None
        for gone in ("order_risk_pct", "positions", "max_single_risk_pct", "max_single_ticker",
                     "unprotected_value_pct", "sum_open_risk_pct"):
            assert gone not in bh
        assert wbs["stop_vol"]["stop_frac"] == 0.1
        assert wbs["stop_vol"]["sigma20_frac"] is not None
        assert wbs["order_risk_at_vol_weight_frac"] == round(vt["vol_weight"] * 0.1, 4)
        assert wbs["sum_open_risk_after_at_vol_weight_frac"] == round(
            0.01 + wbs["order_risk_at_vol_weight_frac"], 4)

    def test_one_unit_convention(self, reads):
        """Shares of price or of the book are fractions named *_frac; no
        field is named *_pct."""
        wbs = _decision(_gate()[1][0])["would_be_sizing"]
        keys = [k for _, k, _ in _walk(wbs)]
        assert [k for k in keys if k.endswith("_pct")] == []
        assert {"stop_frac", "target_frac", "floor_vs_stop_frac", "sigma20_frac",
                "sum_open_risk_frac", "order_risk_at_vol_weight_frac"} <= set(keys)
        assert wbs["units"].startswith("fractions")

    def test_no_price_dollar_or_share_field(self, reads):
        wbs = _decision(_gate()[1][0])["would_be_sizing"]
        banned = {"price", "stop_loss_price", "target", "atr", "model_floor", "stop_method",
                  "shares", "shares_initial", "vol_weight_shares", "shares_at_sizing",
                  "dollar_amount", "pv", "total_value"}
        # not_computed's keys name what was left out (a label, not a value).
        keys = {k for _, k, _ in _walk({k: v for k, v in wbs.items() if k != "not_computed"})}
        assert keys & banned == set()
        values = [v for _, _, v in _walk(wbs) if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for dollars in (100.0, 90.0, 130.0, 85.0, 5.0, 5000.0, 500.0, 450.0):
            assert dollars not in values
        assert not any(isinstance(v, str) and "$" in v for _, _, v in _walk(wbs))

    def test_not_computed_says_why(self, reads):
        nc = _decision(_gate()[1][0])["would_be_sizing"]["not_computed"]
        assert set(nc) == {"allocation_size", "shares", "add_lot_r"}

    def test_gate_fields_identical_with_the_shadow_on_or_off(self, reads, monkeypatch):
        on = _decision(_gate(stop_rec=_stop())[1][0])
        monkeypatch.setattr(config, "BLOCKED_SIZING_SHADOW_ENABLED", False)
        off = _decision(_gate(stop_rec=_stop())[1][0])
        assert "would_be_sizing" not in off
        assert {k: v for k, v in on.items() if k != "would_be_sizing"} == off

    def test_switch_off_writes_no_record_and_makes_no_read(self, monkeypatch):
        monkeypatch.setattr(config, "BLOCKED_SIZING_SHADOW_ENABLED", False)
        stop = MagicMock()
        monkeypatch.setattr(crew, "_would_be_stop", stop)
        out, rows = _gate()
        assert out is True
        assert "would_be_sizing" not in _decision(rows[0])
        stop.assert_not_called()

    def test_a_raising_record_still_blocks_with_one_row(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_sizing_record",
                            MagicMock(side_effect=RuntimeError("boom")))
        out, rows = _gate()
        assert out is True
        assert len(rows) == 1
        assert "would_be_sizing" not in _decision(rows[0])
        assert _decision(rows[0])["entry_gate"]["trend"]["blocked"] is True

    def test_a_passing_buy_computes_nothing(self, monkeypatch):
        rec = MagicMock()
        monkeypatch.setattr(crew, "_would_be_sizing_record", rec)
        out, rows = _gate(trend=_trend(below=False))
        assert out is False
        assert rows == []
        rec.assert_not_called()

    def test_deferred_retry_block_gets_no_record(self, monkeypatch):
        rec = MagicMock()
        monkeypatch.setattr(crew, "_would_be_sizing_record", rec)
        out, rows = _gate(stop_rec=_stop(), path="deferred_retry")
        assert out is True
        assert len(rows) == 1
        rec.assert_not_called()
        assert "would_be_sizing" not in _decision(rows[0])


# ── the record on its own ────────────────────────────────────────────────────

class TestWouldBeSizingRecord:
    def test_no_last_close_records_why_and_reads_nothing(self, monkeypatch):
        stop = MagicMock()
        monkeypatch.setattr(crew, "_would_be_stop", stop)
        for t in (None, {}, _trend(last_close=None), _trend(last_close=float("nan")),
                  _trend(last_close=0)):
            rec = crew._would_be_sizing_record("NVDA", _result(), t)
            assert rec["hypothetical"] is True
            assert rec["price_basis"] is None
            assert rec["reason"].startswith("no price in hand")
        stop.assert_not_called()

    def test_a_failing_stop_leaves_the_rest(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", MagicMock(side_effect=RuntimeError("stop")))
        monkeypatch.setattr(risk_manager, "_vol_shadow_bars", lambda ticker, window: _vol_bars())
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["stop"]["reason"] == "error: RuntimeError"
        assert rec["stop"]["stop_frac"] is None
        assert rec["rr"]["r_to_target"] is None
        assert rec["rr"]["reason"] == "no_stop"
        assert rec["vol_target"]["read"] == "ok"
        assert rec["order_risk_at_vol_weight_frac"] is None

    def test_a_failed_bar_read_is_said_not_a_fallback(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop",
                            MagicMock(side_effect=crew._BarsReadFailed("QueryCanceled")))
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["stop"]["reason"] == "bars read failed: QueryCanceled"
        assert rec["stop"]["kind"] is None

    def test_vol_switch_off_skips_the_read(self, monkeypatch, reads):
        monkeypatch.setattr(config, "VOL_TARGET_SIZING_SHADOW_ENABLED", False)
        bars = MagicMock(return_value=_vol_bars())
        monkeypatch.setattr(risk_manager, "_vol_shadow_bars", bars)
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["vol_target"] == {"read": "switch off"}
        assert rec["stop_vol"] == {"read": "switch off"}
        bars.assert_not_called()
        assert rec["order_risk_at_vol_weight_frac"] is None
        assert rec["book_heat"]["read"] == "ok"

    def test_heat_switch_off_skips_the_read(self, monkeypatch, reads):
        monkeypatch.setattr(config, "BOOK_HEAT_SHADOW_ENABLED", False)
        inputs = MagicMock(return_value=_heat_inputs())
        monkeypatch.setattr(crew, "_book_heat_inputs", inputs)
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["book_heat"] == {"read": "switch off"}
        inputs.assert_not_called()
        assert rec["sum_open_risk_after_at_vol_weight_frac"] is None
        assert rec["vol_target"]["read"] == "ok"

    def test_a_stop_at_or_above_price_is_no_stop(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", lambda t, p, b: (100.0, "Fallback stop (5%)", None, None))
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["stop"]["reason"] == "no stop below price"
        assert rec["stop"]["stop_frac"] is None

    def test_a_spent_budget_skips_the_vol_and_heat_reads(self, monkeypatch, reads):
        monkeypatch.setattr(crew, "_WOULD_BE_BUDGET_S", 0.0)
        bars = MagicMock(return_value=_vol_bars())
        inputs = MagicMock(return_value=_heat_inputs())
        monkeypatch.setattr(risk_manager, "_vol_shadow_bars", bars)
        monkeypatch.setattr(crew, "_book_heat_inputs", inputs)
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["stop"]["stop_frac"] == 0.1             # the stop runs first
        assert rec["vol_target"] == {"read": "skipped: time budget spent"}
        assert rec["stop_vol"] == {"read": "skipped: time budget spent"}
        assert rec["book_heat"] == {"read": "skipped: time budget spent"}
        bars.assert_not_called()
        inputs.assert_not_called()
        assert rec["budget_s"] == 0.0
        assert rec["order_risk_at_vol_weight_frac"] is None

    def test_a_spent_budget_leaves_the_gate_unchanged(self, monkeypatch, reads):
        on = _decision(_gate(stop_rec=_stop())[1][0])
        monkeypatch.setattr(crew, "_WOULD_BE_BUDGET_S", 0.0)
        out, rows = _gate(stop_rec=_stop())
        assert out is True
        assert len(rows) == 1
        spent = _decision(rows[0])
        assert spent["entry_gate"] == on["entry_gate"]
        assert spent["would_be_sizing"]["book_heat"] == {"read": "skipped: time budget spent"}

    def test_the_budget_is_bounded(self):
        assert 0 < crew._WOULD_BE_BUDGET_S <= 5.0
        assert crew._WOULD_BE_READ_TIMEOUT_MS <= 3000

    def test_off_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "BLOCKED_SIZING_SHADOW_ENABLED", False)
        assert crew._would_be_sizing_record("NVDA", _result(), _trend()) is None

    def test_never_raises(self, monkeypatch):
        monkeypatch.setattr(crew, "_rr_vs_stop", MagicMock(side_effect=RuntimeError("x")))
        monkeypatch.setattr(crew, "_book_heat_record", MagicMock(side_effect=RuntimeError("y")))
        monkeypatch.setattr(risk_manager, "_vol_shadow_record", MagicMock(side_effect=RuntimeError("z")))
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["rr"] == {"error": "RuntimeError"}
        assert rec["book_heat"] == {"read": "error: RuntimeError"}
        assert rec["vol_target"] == {"read": "error: RuntimeError"}
        assert crew._would_be_sizing_record("NVDA", "not a dict", "nor this")["reason"].startswith(
            "no price in hand")


class TestWouldBeStop:
    """_would_be_stop runs the real calculate_stop on a simulation
    RiskManager that has no broker."""

    def _bars(self, n=20):
        return [{"h": 101.0, "l": 99.0, "c": 100.0} for _ in range(n)]

    def test_atr_stop_from_cached_bars(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", _REAL_WOULD_BE_STOP)
        cached = MagicMock(return_value=self._bars())
        live = MagicMock(side_effect=AssertionError("no broker bars"))
        monkeypatch.setattr(risk_manager, "_cached_bars_for_atr", cached)
        monkeypatch.setattr(risk_manager, "_fetch_bars_for_atr", live)
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
        stop, method, atr, mult = crew._would_be_stop("NVDA", 100.0, "Accelerator")
        lo, hi = risk_manager.ATR_STOP_MULTIPLIER["Accelerator"]
        assert atr == 2.0
        assert mult == (lo + hi) / 2
        assert stop == round(100.0 - mult * 2.0, 2)
        assert crew._stop_kind(method) == "atr_multiple"
        cached.assert_called_once()
        live.assert_not_called()

    def test_moonshot_hard_stop(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", _REAL_WOULD_BE_STOP)
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
        stop, method, atr, mult = crew._would_be_stop("RGTI", 100.0, "Moonshot")
        assert stop == round(100.0 * (1 - risk_manager.MOONSHOT_MAX_LOSS_PCT), 2)
        assert crew._stop_kind(method) == "moonshot_hard"

    def test_no_atr_falls_back_without_alert_row_or_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(crew, "_would_be_stop", _REAL_WOULD_BE_STOP)
        monkeypatch.setattr(risk_manager, "_cached_bars_for_atr", lambda t, days=20: [])
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
        queued = MagicMock()
        monkeypatch.setattr(risk_manager, "_queue_atr_fallback_log", queued)
        filters_before = list(risk_manager.log.filters)
        with patch("alerts._send") as send, caplog.at_level("WARNING", logger="risk_manager"):
            stop, method, atr, mult = crew._would_be_stop("NVDA", 100.0, "Accelerator")
        assert crew._stop_kind(method) == "atr_fallback"
        send.assert_not_called()
        queued.assert_not_called()
        assert not any("ATR unavailable" in r.getMessage() for r in caplog.records)
        assert risk_manager.log.filters == filters_before

    def test_the_real_path_still_warns(self, monkeypatch, caplog):
        """Only the would-be call is quiet: calculate_stop on its own (the
        sizing path) still logs the fallback warning."""
        monkeypatch.setattr(risk_manager, "_cached_bars_for_atr", lambda t, days=20: [])
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
        monkeypatch.setattr(risk_manager, "_queue_atr_fallback_log", MagicMock())
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm.simulation, rm.last_stop_atr, rm.last_stop_mult = True, None, None
        with caplog.at_level("WARNING", logger="risk_manager"):
            rm.calculate_stop("NVDA", 100.0, "Accelerator", log_fallback=False)
        assert any("ATR unavailable" in r.getMessage() for r in caplog.records)

    def test_the_quiet_filter_only_drops_its_own_thread(self):
        import logging
        import threading
        q = crew._quiet_atr_fallback_warning()
        mine = logging.LogRecord("risk_manager", logging.WARNING, "", 0, "[X] ATR unavailable — x", (), None)
        other = logging.LogRecord("risk_manager", logging.WARNING, "", 0, "[X] ATR unavailable — x", (), None)
        other.thread = threading.get_ident() + 1
        plain = logging.LogRecord("risk_manager", logging.WARNING, "", 0, "something else", (), None)
        assert q.filter(mine) is False
        assert q.filter(other) is True
        assert q.filter(plain) is True

    def test_a_failed_bar_read_raises_instead_of_a_fallback_stop(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", _REAL_WOULD_BE_STOP)
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
        import db

        def failing(ticker, days=260, strict=False):
            bound = db._PRICE_HISTORY_BOUND.get()
            assert bound is not None and bound["timeout_ms"] == crew._WOULD_BE_READ_TIMEOUT_MS
            bound["errors"].append("QueryCanceled")
            return []
        monkeypatch.setattr(db, "get_price_history", failing)
        with pytest.raises(crew._BarsReadFailed, match="QueryCanceled"):
            crew._would_be_stop("NVDA", 100.0, "Accelerator")
        assert db._PRICE_HISTORY_BOUND.get() is None

    def test_setup_gates_on_skip_the_stop_before_any_read(self, monkeypatch):
        monkeypatch.setattr(crew, "_would_be_stop", _REAL_WOULD_BE_STOP)
        monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", True)
        cached = MagicMock(return_value=[])
        monkeypatch.setattr(risk_manager, "_cached_bars_for_atr", cached)
        import agents.setup_gates as sg
        classify = MagicMock(side_effect=AssertionError("no ladder read"))
        monkeypatch.setattr(sg, "classify_setup", classify)
        with pytest.raises(crew._StopNotEvaluated, match="setup gates on"):
            crew._would_be_stop("NVDA", 100.0, "Accelerator")
        classify.assert_not_called()
        cached.assert_not_called()
        rec = crew._would_be_sizing_record("NVDA", _result(), _trend())
        assert rec["stop"]["reason"].startswith("not evaluated: setup gates on")
        assert rec["stop"]["stop_frac"] is None
        classify.assert_not_called()

    def test_the_client_raises_an_attribute_error_on_any_use(self):
        c = crew._NoBrokerClient()
        with pytest.raises(AttributeError, match="no broker read"):
            c.get_account()
        assert hasattr(c, "get_all_positions") is False
        assert getattr(c, "anything", "default") == "default"


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, args=None):
        self.conn.sql.append((sql, args))
        if self.conn.fail and sql.startswith("SELECT"):
            raise RuntimeError("canceled")

    def fetchall(self):
        return [("2026-11-20", 1.0, 2.0, 0.5, 1.5, 100)]


class _FakeConn:
    def __init__(self, fail=False):
        self.sql, self.fail, self.rolled_back = [], fail, 0

    def cursor(self):
        return _FakeCursor(self)

    def rollback(self):
        self.rolled_back += 1


class TestPriceHistoryBound:
    """db.price_history_bound: a statement timeout and an error note for
    get_price_history inside the block only."""

    def _patch(self, monkeypatch, conn):
        import db
        monkeypatch.setattr(db, "_get_conn", lambda: conn)
        monkeypatch.setattr(db, "_put_conn", lambda c: None)
        return db

    def test_outside_the_block_nothing_changes(self, monkeypatch):
        conn = _FakeConn()
        db = self._patch(monkeypatch, conn)
        assert len(db.get_price_history("NVDA", days=5)) == 1
        assert [q for q, _ in conn.sql if "statement_timeout" in q] == []
        assert conn.rolled_back == 0

    def test_inside_the_block_the_read_is_bounded_and_rolled_back(self, monkeypatch):
        conn = _FakeConn()
        db = self._patch(monkeypatch, conn)
        with db.price_history_bound(3000) as bound:
            assert len(db.get_price_history("NVDA", days=5)) == 1
        assert conn.sql[0] == ("SET LOCAL statement_timeout = %s", (3000,))
        assert conn.rolled_back == 1
        assert bound["errors"] == []
        assert db._PRICE_HISTORY_BOUND.get() is None

    def test_a_failed_read_is_noted_strict_or_not(self, monkeypatch):
        conn = _FakeConn(fail=True)
        db = self._patch(monkeypatch, conn)
        with db.price_history_bound(3000) as bound:
            assert db.get_price_history("NVDA", days=5) == []
            with pytest.raises(RuntimeError):
                db.get_price_history("NVDA", days=5, strict=True)
        assert bound["errors"] == ["RuntimeError", "RuntimeError"]
        assert conn.rolled_back == 2

    def test_the_bound_is_reset_after_an_exception(self):
        import db
        with pytest.raises(ValueError):
            with db.price_history_bound(1000):
                raise ValueError("x")
        assert db._PRICE_HISTORY_BOUND.get() is None


@pytest.mark.parametrize("method,kind", [
    ("Setup-gate structural stop (Compounder)", "setup_gate"),
    ("Moonshot hard stop: 20% max loss from entry", "moonshot_hard"),
    ("Fallback stop (5% below entry — ATR unavailable)", "atr_fallback"),
    ("1.8× ATR (2.00) below entry = $96.40", "atr_multiple"),
    ("something else", "other"),
    (None, None), ("", None),
])
def test_stop_kind(method, kind):
    assert crew._stop_kind(method) == kind


# ── event_cluster ────────────────────────────────────────────────────────────

def _prior(sid, hours_ago, title, url=None, source="news"):
    return {"signal_id": sid, "timestamp": NOW - timedelta(hours=hours_ago),
            "title": title, "url": url, "source": source}


def _trig(title, url=None, source="news"):
    return {"path": "event", "source": source, "title": title, "url": url}


class TestTitleTokens:
    def test_function_words_ticker_and_one_letter_words_dropped(self):
        toks = sd.title_tokens("NVDA: Nvidia's CEO Jensen Huang Returns as the Board Reverses", ticker="NVDA")
        assert toks == frozenset({"nvidia", "ceo", "jensen", "huang", "returns", "board", "reverses"})

    def test_google_news_publisher_suffix_dropped(self):
        url = "https://news.google.com/rss/articles/abc"
        assert "reuters" not in sd.title_tokens("Huang back as Nvidia CEO - Reuters", url)
        assert "reuters" in sd.title_tokens("Huang back as Nvidia CEO - Reuters", "https://x.com/a")

    def test_empty(self):
        assert sd.title_tokens(None) == frozenset()
        assert sd.title_tokens("!!!") == frozenset()

    def test_scores(self):
        a, b = frozenset({"x", "y", "z", "w"}), frozenset({"x", "y"})
        assert sd.token_jaccard(a, b) == 0.5
        assert sd.token_overlap(a, b) == 1.0
        assert sd.token_jaccard(a, frozenset()) is None
        assert sd.token_overlap(frozenset(), b) is None


class TestEventCluster:
    def test_restatement_clusters_with_the_first_and_best_verdict(self):
        prior = [_prior(1, 5.0, "Nvidia board reinstates Jensen Huang as CEO"),
                 _prior(2, 1.0, "Nvidia reinstates Jensen Huang as CEO after board vote"),
                 _prior(3, 2.0, "Microsoft Azure outage hits cloud customers")]
        ec = sd.event_cluster(_trig("Board reinstates Jensen Huang as Nvidia CEO"), prior, NOW, 24.0, 0.5,
                              "NVDA")
        assert ec["would_cluster"] is True
        assert ec["compared"] == 3
        assert ec["best_signal_id"] == 1
        assert ec["best_score"] == 1.0
        assert ec["n_at_or_above"] == 2
        assert ec["first_signal_id"] == 1
        assert ec["first_hours_before"] == 5.0
        assert ec["threshold"] == 0.5
        assert ec["method"] == "title_token_jaccard"

    def test_different_event_does_not_cluster(self):
        ec = sd.event_cluster(_trig("Nvidia unveils new Rubin GPU at GTC"),
                              [_prior(1, 2.0, "Nvidia board reinstates Jensen Huang as CEO")],
                              NOW, 24.0, 0.5, "NVDA")
        assert ec["would_cluster"] is False
        assert ec["best_score"] < 0.5
        assert ec["n_at_or_above"] == 0
        assert ec["first_signal_id"] is None

    def test_overlap_recorded_beside_jaccard(self):
        ec = sd.event_cluster(_trig("Huang returns as Nvidia CEO"),
                              [_prior(7, 1.0, "Huang returns as Nvidia CEO after board scandal probe finally ends")],
                              NOW, 24.0, 0.5, "NVDA")
        assert ec["best_overlap"] == 1.0
        assert ec["best_overlap_signal_id"] == 7
        assert ec["best_score"] < 0.5
        assert ec["would_cluster"] is False

    def test_window_and_future_rows_ignored(self):
        same = "Nvidia board reinstates Jensen Huang as CEO"
        prior = [_prior(1, 30.0, same), _prior(2, -1.0, same), _prior(3, 0.0, same),
                 {"signal_id": 4, "timestamp": "bad", "title": same}, _prior(5, 1.0, None)]
        ec = sd.event_cluster(_trig(same), prior, NOW, 24.0, 0.5, "NVDA")
        assert ec["compared"] == 0
        assert ec["would_cluster"] is False
        assert ec["best_score"] is None

    def test_no_title_gives_none(self):
        ec = sd.event_cluster(_trig(None), [_prior(1, 1.0, "Nvidia CEO")], NOW, 24.0, 0.5, "NVDA")
        assert ec["would_cluster"] is None
        assert ec["tokens"] == 0


class TestClassifyAndRecord:
    def test_classify_without_threshold_has_no_cluster_key_and_is_version_1(self):
        rec = sd.classify(_trig("Nvidia CEO returns"), [], NOW, 24.0, 24.0)
        assert "event_cluster" not in rec
        assert rec["version"] == sd.STORY_DEDUP_VERSION_NO_CLUSTER == 1

    def test_error_fallback_is_version_1(self, monkeypatch):
        monkeypatch.setattr(config, "STORY_DEDUP_SHADOW_ENABLED", True)
        monkeypatch.setattr(config, "STORY_DEDUP_WINDOW_HOURS", "not a number")
        rec = sd.story_dedup_record("NVDA", _trig("x"))
        assert rec == {"version": 1, "read": "error: ValueError"}

    def test_classify_with_threshold(self):
        rec = sd.classify(_trig("Board reinstates Jensen Huang as Nvidia CEO"),
                          [_prior(1, 2.0, "Nvidia board reinstates Jensen Huang as CEO", url="https://a.com/1")],
                          NOW, 24.0, 24.0, cluster_threshold=0.5, ticker="NVDA")
        assert rec["event_cluster"]["would_cluster"] is True
        assert rec["duplicate_of"] is None   # a different title and URL: not a duplicate
        assert rec["version"] == 2

    def test_recheck_is_labelled_false(self):
        same = "Nvidia board reinstates Jensen Huang as CEO"
        rec = sd.classify(_trig(same, source="stale_entry_recheck"), [_prior(1, 2.0, same)],
                          NOW, 24.0, 24.0, cluster_threshold=0.5, ticker="NVDA")
        assert rec["event_cluster"]["would_cluster"] is False
        assert rec["event_cluster"]["best_score"] == 1.0

    @pytest.fixture
    def shadow_on(self, monkeypatch):
        monkeypatch.setattr(config, "STORY_DEDUP_SHADOW_ENABLED", True)
        monkeypatch.setattr(config, "STORY_DEDUP_WINDOW_HOURS", 24)
        monkeypatch.setattr(config, "STORY_STALE_CUTOFF_HOURS", 24)
        monkeypatch.setattr(config, "STORY_EVENT_CLUSTER_JACCARD", 0.5)

    def test_record_uses_the_config_threshold_and_ticker(self, shadow_on, monkeypatch):
        monkeypatch.setattr(config, "STORY_EVENT_CLUSTER_JACCARD", 0.9)
        prior = [{"signal_id": 1, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1),
                  "title": "NVDA Nvidia board reinstates Jensen Huang", "url": None, "source": "news"}]
        with patch("db.recent_event_verdicts", return_value=prior):
            rec = sd.story_dedup_record("NVDA", _trig("Nvidia board reinstates Jensen Huang NVDA"))
        ec = rec["event_cluster"]
        assert ec["threshold"] == 0.9
        assert ec["best_score"] == 1.0                   # the ticker token is dropped on both sides
        assert ec["would_cluster"] is True

    def test_read_failure_gives_none_labels(self, shadow_on):
        with patch("db.recent_event_verdicts", side_effect=RuntimeError("db down")):
            rec = sd.story_dedup_record("NVDA", _trig("Nvidia CEO returns"))
        ec = rec["event_cluster"]
        assert rec["read"] == "error: RuntimeError"
        assert ec["would_cluster"] is None
        assert ec["compared"] is None
        assert ec["n_at_or_above"] is None
