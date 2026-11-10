"""The orders row is written as soon as the broker has a market order, before
place_market_order's fill wait, and the outcome of the wait updates that row.
A worker killed during the wait used to leave a live broker order with no
orders row; reconcile now adopts the leftover 'submitted' row and books a
late fill once."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import db
import execution
import ibkr
import tasks

AWAIT = db.AWAITING_FILL_NOTE


class _Killed(BaseException):
    """Stands in for the worker being stopped mid-wait (SIGTERM, OOM)."""


class FakeOrders:
    """An in-memory orders + trades table behind the db.* order functions,
    with the same guards as the SQL."""

    def __init__(self, fail_early_write=False, early_write_lands_anyway=False):
        self.rows: dict[int, dict] = {}
        self.trades: list[dict] = []
        self.fail_early_write = fail_early_write
        self.early_write_lands_anyway = early_write_lands_anyway   # committed, client saw an error
        self.events: list[str] = []          # write order: "order:<status>", "trade"
        self.mocks: dict = {}                # mocks of the last _run_exit / _reconcile
        self.stops: dict = {}
        self.outcomes: dict = {}
        self._next = 1

    def log_order(self, ticker, side, order_type, qty, broker_order_id=None, limit_price=None,
                  stop_price=None, decision_price=None, approval_id=None, parent_order_id=None,
                  status="submitted", intent=None, perm_id=None, submitted_at=None,
                  submit_price=None, reconcile_note=None):
        if self.fail_early_write and reconcile_note == AWAIT and not self.early_write_lands_anyway:
            return -1
        rid, self._next = self._next, self._next + 1
        self.rows[rid] = {"id": rid, "ticker": ticker, "side": side, "qty": float(qty),
                          "order_type": order_type, "broker_order_id": broker_order_id,
                          "perm_id": perm_id, "decision_price": decision_price,
                          "approval_id": approval_id, "intent": intent, "stop_price": stop_price,
                          "status": status, "reconcile_note": reconcile_note,
                          "filled_qty": None, "fill_price": None, "submitted_at": submitted_at,
                          "submit_price": submit_price, "trading_env": db.TRADING_ENV,
                          "created_at": datetime.now(timezone.utc)}
        if self.early_write_lands_anyway and reconcile_note == AWAIT:
            return -1
        return rid

    def finish_submitted_order(self, order_id, status, perm_id=None):
        r = self.rows.get(order_id)
        if not r or r["status"] != "submitted":
            return False
        r["status"] = status
        r["perm_id"] = r["perm_id"] or perm_id
        if r["reconcile_note"] == AWAIT:
            r["reconcile_note"] = None
        return True

    def update_order_fill(self, order_id, fill_price, filled_qty, status,
                          expect_filled_qty=None, filled_at=None, filled_at_source=None):
        r = self.rows[order_id]
        if expect_filled_qty is not None and (
                r["status"] not in ("unconfirmed", "partial")
                or abs((r["filled_qty"] or 0) - expect_filled_qty) >= 1e-4):
            return False
        r.update(fill_price=fill_price, filled_qty=filled_qty, status=status)
        self.events.append(f"order:{status}")
        if r["reconcile_note"] == AWAIT:
            r["reconcile_note"] = None
        return True

    def mark_submitted_order_fill(self, order_id, fill_price, filled_qty, status, filled_at=None,
                                  filled_at_source=None):
        """Same as the SQL: only while 'submitted'; rows updated."""
        r = self.rows[order_id]
        if r["status"] != "submitted":
            return 0
        if not self.update_order_fill(order_id, fill_price, filled_qty, status, filled_at=filled_at):
            return None
        r["reconcile_note"] = db.TRADE_PENDING_NOTE      # same write as the fill
        return 1

    def claim_trade_pending(self, order_id, note, expect_note=db.TRADE_PENDING_NOTE):
        r = self.rows.get(order_id)
        if not r or r["reconcile_note"] != expect_note:
            return 0
        r["reconcile_note"] = note
        self.events.append(f"note:{note}")
        return 1

    def claim_and_log_trade(self, order_id, note, symbol, side, quantity, price, reason,
                            price_source=None, expect_note=db.TRADE_PENDING_NOTE,
                            order_row_id=None):
        """The one transaction: claim (note must still be expect_note), then
        the trade row; all or nothing."""
        r = self.rows.get(order_id)
        if not r or r["reconcile_note"] != expect_note:
            return False
        self.log_trade(symbol=symbol, side=side, quantity=quantity, price=price, reason=reason,
                       price_source=price_source, order_row_id=order_row_id)  # raises: nothing changed
        r["reconcile_note"] = note
        self.events.append(f"note:{note}")
        return True

    def get_stale_trade_pending_orders(self, min_age_minutes=10):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
        return [dict(r) for r in self.rows.values()
                if r["status"] in ("filled", "partial")
                and r["reconcile_note"] in db.TRADE_PENDING_NOTES and r["created_at"] < cutoff]

    def order_reconcile_note(self, order_id):
        r = self.rows.get(order_id)
        return r["reconcile_note"] if r else None

    # position_stops and trade_outcomes, as far as these paths use them
    def upsert_position_stop(self, ticker, entry_price, stop_price, stop_method, bucket,
                             conviction, trail_distance_pct=None, entry_price_source=None):
        self.events.append("stop_record")
        self.stops[ticker] = {"entry_price": float(entry_price), "stop_price": float(stop_price),
                              "entry_price_source": entry_price_source,
                              "stop_method": stop_method, "thesis_bucket": bucket,
                              "conviction": conviction, "trailing_active": False,
                              "profit_target_taken": False,
                              "updated_at": datetime.now(timezone.utc)}

    def get_position_stop(self, ticker, strict=False):
        rec = self.stops.get(ticker)
        return dict(rec, ticker=ticker) if rec else None

    def position_stop_written(self, ticker):
        rec = self.stops.get(ticker)
        return ({k: rec[k] for k in ("updated_at", "entry_price", "stop_method",
                                     "thesis_bucket", "conviction")} if rec else None)

    def log_trade_outcome(self, approval_id, **kw):
        if approval_id not in self.outcomes:          # the real one dedupes per approval
            self.outcomes[approval_id] = kw
        return True

    def set_order_reconcile_note(self, order_id, note):
        r = self.rows.get(order_id)
        if r and r["reconcile_note"] in (None, AWAIT):
            r["reconcile_note"] = note
            return True
        return False

    def set_order_perm_id(self, order_id, perm_id):
        r = self.rows[order_id]
        if r["perm_id"] is None:
            r["perm_id"] = perm_id
            return True
        return False

    def log_trade(self, **kw):
        self.events.append("trade")
        self.trades.append({**kw, "timestamp": datetime.now(timezone.utc)})
        return True

    def trade_logged_near(self, ticker, side, around, intent=None, order_id=None):
        """db.trade_logged_near's rule, in Python (the SQL itself is checked
        in test_trade_logged_near_sql_matches_the_rule)."""
        end = around + timedelta(minutes=db.TRADE_MATCH_CAP_MINUTES)
        if order_id is not None:
            later = [r["created_at"] for r in self.rows.values() if r["ticker"] == ticker
                     and r["side"] == side and r.get("trading_env") == db.TRADING_ENV
                     and r["id"] > order_id]
            if later:
                end = min(end, min(later))
        start = around - timedelta(seconds=db.TRADE_MATCH_BEFORE_S)

        def execution_reason(reason):
            if (reason.startswith(("Backstop", "Profit target", "Reconciled exit"))
                    or "Late-fill reconciled" in reason):
                return False
            if intent and not intent.upper().startswith("MANUAL"):
                return reason.startswith(intent + " ") and " (conviction=" in reason
            return not ("(conviction=" in reason and "/10)" in reason)
        return float(sum(t["quantity"] for t in self.trades
                         if t["symbol"] == ticker and t["side"] == side
                         and start <= t["timestamp"] < end
                         and execution_reason(t.get("reason") or "")))

    def find_awaiting_order_row(self, ticker, side, broker_order_id):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
        hits = [r["id"] for r in self.rows.values() if r["ticker"] == ticker and r["side"] == side
                and r["broker_order_id"] == broker_order_id and r["status"] == "submitted"
                and r["reconcile_note"] == AWAIT and r.get("trading_env") == db.TRADING_ENV
                and r["created_at"] > cutoff]
        return max(hits) if hits else None

    def awaiting_fill_sell_tickers(self):
        return {r["ticker"] for r in self.rows.values() if r["side"] == "SELL"
                and r["status"] == "submitted" and r["reconcile_note"] == AWAIT}

    def trade_pending_sell_tickers(self):
        return {r["ticker"] for r in self.rows.values() if r["side"] == "SELL"
                and r["status"] in ("filled", "partial")
                and r["reconcile_note"] in db.TRADE_PENDING_NOTES}

    def adopt_orphaned_submitted_orders(self, min_age_minutes=10):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
        out = []
        for r in self.rows.values():
            if r["status"] == "submitted" and r["reconcile_note"] == AWAIT and r["created_at"] < cutoff:
                r["status"], r["reconcile_note"] = "unconfirmed", db.ORPHANED_FILL_WAIT_NOTE
                out.append({k: r[k] for k in ("id", "ticker", "side", "broker_order_id")})
        return out

    def get_unconfirmed_orders(self, max_age_hours=24):
        return [{**r, "filled_qty": r["filled_qty"] or 0.0} for r in self.rows.values()
                if r["status"] in ("unconfirmed", "partial") and r["broker_order_id"] is not None]

    def patches(self):
        names = ("log_order", "finish_submitted_order", "update_order_fill",
                 "set_order_reconcile_note", "set_order_perm_id", "log_trade",
                 "adopt_orphaned_submitted_orders", "get_unconfirmed_orders",
                 "trade_logged_near", "mark_submitted_order_fill",
                 "awaiting_fill_sell_tickers", "trade_pending_sell_tickers",
                 "find_awaiting_order_row",
                 "claim_trade_pending", "claim_and_log_trade", "order_reconcile_note",
                 "upsert_position_stop", "get_position_stop", "position_stop_written",
                 "log_trade_outcome",
                 "get_stale_trade_pending_orders")
        return [patch(f"db.{n}", getattr(self, n)) for n in names]


def _approval(action="EXIT"):
    return {"id": 1, "ticker": "NVDA", "action": action, "conviction": 7,
            "thesis_bucket": "Accelerator", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 999, "price_at_sizing": 100.0,
                          "stop_loss_price": 90.0}}


def _broker(result=None, raise_after_submit=None, perm_at_submit=None):
    """A place_market_order stand-in: runs on_submitted the way ibkr does
    (order id known, perm id usually not yet), then 'waits'."""
    calls = []

    def place(ticker, side, qty, fill_timeout=30.0, on_submitted=None):
        calls.append((ticker, side, qty))
        if on_submitted is not None:
            on_submitted(501, perm_at_submit)
        if raise_after_submit is not None:
            raise raise_after_submit
        return {"order_id": 501, "perm_id": 9001, "ticker": ticker, "side": side, "qty": qty,
                **(result or {})}
    return place, calls


def _run_exit(fake, place, restore_effect=None):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty=42.0)
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client", return_value=client),
                  patch("ibkr.cancel_stop_orders_for", return_value=0),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=None),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("config.USE_BRACKET_ORDERS", False),
                  patch("db.get_position_stop", return_value=None),
                  patch("db.delete_position_stop"), patch("db.close_trade_outcome"),
                  patch("db.log_trade_outcome"), patch("alerts.send_trade_alert"),
                  patch("alerts._send"), patch("tasks._save_untracked_order", return_value=True),
                  patch("execution._restore_stop_after_failed_sell", side_effect=restore_effect)):
            fake.mocks[p.attribute] = st.enter_context(p)
        return execution.execute_approved(_approval("EXIT"))


def _reconcile(fake, fill):
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("tasks._retry_untracked_orders"), patch("tasks._retry_pending_reprotects"),
                  patch("tasks._retry_pending_buy_outcomes"), patch("tasks._backfill_perm_ids"),
                  patch("tasks._resolve_if_cancelled"),
                  patch("ibkr.get_fill_for_order", return_value=fill),
                  patch("db.get_approval_by_id", return_value=None),
                  patch("tasks._reprotect_after_late_sell", return_value=True),
                  patch("alerts._send")):
            fake.mocks[p.attribute] = st.enter_context(p)
        tasks._reconcile_unconfirmed_orders_locked(MagicMock())


def test_kill_during_the_wait_leaves_exactly_one_submitted_row():
    fake = FakeOrders()
    place, _ = _broker(raise_after_submit=_Killed())
    with pytest.raises(_Killed):
        _run_exit(fake, place)
    assert len(fake.rows) == 1
    row = fake.rows[1]
    assert row["status"] == "submitted" and row["reconcile_note"] == AWAIT
    assert row["broker_order_id"] == 501 and row["side"] == "SELL" and row["qty"] == 42.0
    assert row["intent"] == "EXIT" and row["approval_id"] == 1 and row["decision_price"] == 100.0
    assert row["submitted_at"] is not None
    assert fake.trades == []


def test_a_normal_fill_updates_the_one_row():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert len(fake.rows) == 1
    row = fake.rows[1]
    assert row["status"] == "filled" and row["filled_qty"] == 42.0 and row["fill_price"] == 101.5
    assert row["reconcile_note"] is None
    assert row["perm_id"] == 9001                        # assigned during the wait, backfilled
    assert len(fake.trades) == 1


def test_a_part_fill_leaves_the_one_row_partial():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 20.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert len(fake.rows) == 1 and fake.rows[1]["status"] == "partial"
    assert fake.rows[1]["reconcile_note"] is None


def test_an_unfilled_order_updates_the_one_row_to_unconfirmed():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert len(fake.rows) == 1
    row = fake.rows[1]
    assert row["status"] == "unconfirmed" and row["reconcile_note"] is None
    assert row["perm_id"] == 9001
    assert fake.trades == []


def test_a_dead_sell_marks_the_one_row_cancelled():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Inactive"})
    _run_exit(fake, place)
    assert len(fake.rows) == 1 and fake.rows[1]["status"] == "cancelled"
    assert fake.rows[1]["reconcile_note"] is None


@pytest.mark.parametrize("result,status", [
    ({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"}, "filled"),
    ({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"}, "unconfirmed"),
])
def test_a_failed_early_write_falls_back_to_the_late_insert(result, status):
    fake = FakeOrders(fail_early_write=True)
    place, _ = _broker(result)
    _run_exit(fake, place)
    assert len(fake.rows) == 1
    row = fake.rows[1]
    assert row["status"] == status and row["reconcile_note"] is None
    assert row["broker_order_id"] == 501 and row["perm_id"] == 9001


def test_an_early_write_that_raises_falls_back_too():
    fake = FakeOrders()
    real = fake.log_order

    def log_order(**kw):
        if kw.get("reconcile_note") == AWAIT:
            raise RuntimeError("pool exhausted")
        return real(**kw)
    fake.log_order = log_order
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert [r["status"] for r in fake.rows.values()] == ["unconfirmed"]


def test_a_failed_unconfirmed_update_inserts_nothing_and_leaves_the_row_for_adoption():
    """A second 'unconfirmed' row next to the adoptable early one would be
    booked twice by reconcile."""
    fake = FakeOrders()
    fake.finish_submitted_order = lambda *a, **k: False
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert len(fake.rows) == 1
    assert fake.rows[1]["status"] == "submitted" and fake.rows[1]["reconcile_note"] == AWAIT


def test_a_failed_fill_write_replaces_the_awaiting_note():
    """Filled and booked, but the fill write failed: the note must stop the
    row from being adopted (and the fill booked again)."""
    fake = FakeOrders()
    fake.mark_submitted_order_fill = lambda *a, **k: None
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert fake.rows[1]["reconcile_note"] == "fill_write_failed"
    fake.rows[1]["created_at"] -= timedelta(hours=1)
    assert fake.adopt_orphaned_submitted_orders() == []


def test_reconcile_adopts_an_orphaned_row_and_books_its_late_fill_once():
    fake = FakeOrders()
    place, _ = _broker(raise_after_submit=_Killed())
    with pytest.raises(_Killed):
        _run_exit(fake, place)
    fill = {"filled_qty": 42.0, "fill_price": 99.8, "fill_time": None}

    _reconcile(fake, fill)                                     # too young: left alone
    assert fake.rows[1]["status"] == "submitted" and fake.trades == []

    fake.rows[1]["created_at"] -= timedelta(minutes=11)
    _reconcile(fake, fill)
    row = fake.rows[1]
    assert row["status"] == "filled" and row["filled_qty"] == 42.0
    assert row["reconcile_note"] == db.ORPHANED_FILL_WAIT_NOTE
    assert len(fake.trades) == 1 and fake.trades[0]["quantity"] == 42.0

    _reconcile(fake, fill)                                     # next cycle: nothing new
    assert len(fake.trades) == 1 and len(fake.rows) == 1


def test_reconcile_never_adopts_a_finished_row():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    fake.rows[1]["created_at"] -= timedelta(hours=1)
    _reconcile(fake, {"filled_qty": 42.0, "fill_price": 101.5, "fill_time": None})
    assert len(fake.trades) == 1 and fake.rows[1]["status"] == "filled"


def test_manual_buy_fill_updates_the_one_row():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"),
                  patch("tasks.manage_stop_place", return_value={"order_id": 9}),
                  patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
                m.return_value.calculate_stop.return_value = (92.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert len(fake.rows) == 1
    row = fake.rows[1]
    assert row["status"] == "filled" and row["submit_price"] == 100.0
    assert row["intent"] in ("MANUAL_OPEN", "MANUAL_ADD", "MANUAL_BUY")
    assert len(fake.trades) == 1


def test_manual_kill_during_the_wait_leaves_one_submitted_row():
    fake = FakeOrders()
    place, _ = _broker(raise_after_submit=_Killed())
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place), patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        with pytest.raises(_Killed):
            execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert [(r["status"], r["reconcile_note"]) for r in fake.rows.values()] == [("submitted", AWAIT)]


# --- the db and ibkr pieces -------------------------------------------------

def _conn(rowcount=1, fetchall=()):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = rowcount
    cur.fetchone.return_value = (11,)
    cur.fetchall.return_value = list(fetchall)
    return conn, cur


def test_log_order_writes_the_note_in_the_same_insert():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=7,
                            reconcile_note=AWAIT) == 11
    sql, params = cur.execute.call_args.args
    assert "submit_price, reconcile_note)" in sql and params[-1] == AWAIT
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=7)
    assert "reconcile_note" not in cur.execute.call_args.args[0]     # other callers unchanged


def test_finish_submitted_order_only_moves_a_submitted_row():
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.finish_submitted_order(5, "unconfirmed", perm_id=77) is True
    sql, params = cur.execute.call_args.args
    assert "WHERE id=%s AND status='submitted'" in sql
    assert "reconcile_note=NULLIF(reconcile_note, %s)" in sql and "perm_id=COALESCE(perm_id, %s)" in sql
    assert params == ("unconfirmed", 77, AWAIT, 5)
    conn, cur = _conn(rowcount=0)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.finish_submitted_order(5, "unconfirmed") is False
    assert db.finish_submitted_order(-1, "unconfirmed") is False


def test_adoption_sql_is_guarded_on_status_note_env_and_age():
    conn, cur = _conn(fetchall=[(3, "NVDA", "SELL", 501)])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.adopt_orphaned_submitted_orders(10)
    sql, params = cur.execute.call_args.args
    assert "SET status='unconfirmed'" in sql and "WHERE status='submitted' AND reconcile_note=%s" in sql
    assert "trading_env=%s" in sql and "make_interval(mins => %s)" in sql
    assert params == (db.ORPHANED_FILL_WAIT_NOTE, AWAIT, db.TRADING_ENV, 10)
    assert out == [{"id": 3, "ticker": "NVDA", "side": "SELL", "broker_order_id": 501}]
    bad = MagicMock()
    bad.cursor.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=bad), patch("db._put_conn"):
        assert db.adopt_orphaned_submitted_orders() is None


def test_update_order_fill_clears_the_awaiting_note():
    conn, cur = _conn()
    cur.fetchone.return_value = (None, "SELL")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.update_order_fill(4, 10.0, 3.0, "filled")
    sql, params = cur.execute.call_args.args
    assert "reconcile_note=NULLIF(reconcile_note, %s)" in sql and AWAIT in params


def test_reconcile_survives_a_failed_adoption():
    fake = FakeOrders()
    fake.adopt_orphaned_submitted_orders = MagicMock(side_effect=RuntimeError("down"))
    _reconcile(fake, None)          # no raise


class _Order:
    orderId = 501
    permId = 0
    tif = None


def _fake_ib(events):
    trade = MagicMock()
    trade.order = _Order()
    trade.isDone.return_value = True
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 3.0
    trade.orderStatus.avgFillPrice = 10.0
    trade.fills = []
    ib = MagicMock()
    ib.placeOrder.side_effect = lambda c, o: events.append("place") or trade
    ib.waitOnUpdate.side_effect = lambda timeout=1: events.append("wait")
    trade.isDone.side_effect = lambda: events.append("isDone") or True
    return ib


def test_on_submitted_runs_after_place_and_before_the_wait():
    events = []
    ib = _fake_ib(events)
    with patch("ibkr.get_ib", return_value=ib), patch("ibkr._stock", return_value=MagicMock()), \
         patch("ibkr.ibi.MarketOrder", return_value=_Order()):
        out = ibkr.place_market_order("NVDA", "BUY", 3, fill_timeout=1.0,
                                      on_submitted=lambda oid, perm: events.append(("cb", oid, perm)))
    assert events[:3] == ["place", ("cb", 501, None), "isDone"]
    assert out["filled_qty"] == 3.0


def test_a_failing_on_submitted_does_not_stop_the_order():
    events = []
    ib = _fake_ib(events)

    def boom(oid, perm):
        raise RuntimeError("db down")
    with patch("ibkr.get_ib", return_value=ib), patch("ibkr._stock", return_value=MagicMock()), \
         patch("ibkr.ibi.MarketOrder", return_value=_Order()):
        out = ibkr.place_market_order("NVDA", "BUY", 3, fill_timeout=1.0, on_submitted=boom)
    assert out["order_id"] == 501 and out["fill_price"] == 10.0


# --- review fixes -----------------------------------------------------------

_REAL_FINISH = db.finish_submitted_order
_REAL_MARK_FILL = db.mark_submitted_order_fill


def test_dead_sell_restores_the_stop_even_when_the_pool_raises():
    """F1: the row update can't skip the stop restore, and it runs after it."""
    import psycopg2.pool
    fake = FakeOrders()
    fake.finish_submitted_order = _REAL_FINISH
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Inactive"})
    with patch("db._get_conn", side_effect=psycopg2.pool.PoolError("exhausted")):
        _run_exit(fake, place)
    fake.mocks["_restore_stop_after_failed_sell"].assert_called_once_with("NVDA", floor=None)
    assert fake.rows[1]["status"] == "submitted" and fake.rows[1]["reconcile_note"] == AWAIT


def test_dead_sell_restores_the_stop_before_touching_the_row():
    order = []
    fake = FakeOrders()
    fake.finish_submitted_order = lambda *a, **k: order.append("row") or True
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Cancelled"})
    _run_exit(fake, place, restore_effect=lambda t, **k: order.append("stop"))
    assert order == ["stop", "row"]


def _manual_dead_sell(fake, restore_effect=None):
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Inactive"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        client = st.enter_context(patch("ibkr.get_client"))
        client.return_value.get_open_position.return_value = MagicMock(qty=5)
        for p in (patch("ibkr.get_ib"), patch("ibkr.place_market_order", side_effect=place),
                  patch("tasks.manage_cancel_stop_orders_for", return_value=0),
                  patch("execution._stops_still_resting", return_value=0.0),
                  patch("db.get_state", return_value=None),
                  patch("market_data.is_market_hours", return_value=True), patch("alerts._send")):
            st.enter_context(p)
        restore = st.enter_context(patch("execution._restore_stop_after_failed_sell",
                                         side_effect=restore_effect))
        out = execution.execute_manual("NVDA", "SELL", 5, "test")
    return out, restore


def test_manual_dead_sell_restores_the_stop_when_the_row_update_raises():
    fake = FakeOrders()
    fake.finish_submitted_order = MagicMock(side_effect=RuntimeError("pool"))
    out, restore = _manual_dead_sell(fake)
    restore.assert_called_once_with("NVDA", floor=None)
    assert out["success"] is False and "restored" in out["message"]


def test_manual_dead_sell_restores_the_stop_before_touching_the_row():
    order = []
    fake = FakeOrders()
    fake.finish_submitted_order = lambda *a, **k: order.append("row") or True
    _manual_dead_sell(fake, restore_effect=lambda t, **k: order.append("stop"))
    assert order == ["stop", "row"]


def test_unconfirmed_alert_fires_when_the_row_update_raises():
    """F2: a raising row update must not skip the FILL UNCONFIRMED alert."""
    fake = FakeOrders()
    fake.finish_submitted_order = MagicMock(side_effect=RuntimeError("pool"))
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    sent = " ".join(str(c.args[0]) for c in fake.mocks["_send"].call_args_list)
    assert "FILL UNCONFIRMED" in sent
    assert len(fake.rows) == 1 and fake.rows[1]["reconcile_note"] == AWAIT   # adopted later


def test_manual_unconfirmed_returns_its_message_when_the_row_update_raises():
    fake = FakeOrders()
    fake.finish_submitted_order = MagicMock(side_effect=RuntimeError("pool"))
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place), patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        out = execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert out["success"] is False and "no fill confirmed" in out["message"]


def test_the_order_row_is_marked_before_the_trade_row():
    """F3a: once the trade row exists, the order row can't still be adoptable."""
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert fake.events[:2] == ["order:filled", "trade"]


def test_manual_order_row_is_marked_before_the_trade_row():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("db.upsert_position_stop"), patch("db.set_stop_broker_order_id"),
                  patch("tasks.manage_stop_place", return_value={"order_id": 9}),
                  patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
                m.return_value.calculate_stop.return_value = (92.0, "ATR")
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert fake.events[:2] == ["order:filled", "trade"]


def test_a_raising_fill_write_still_books_the_trade():
    import psycopg2.pool
    fake = FakeOrders()
    fake.mark_submitted_order_fill = _REAL_MARK_FILL
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    with patch("db._get_conn", side_effect=psycopg2.pool.PoolError("exhausted")):
        _run_exit(fake, place)
    assert len(fake.trades) == 1


def _booked_but_row_lost(fake):
    """The execution path booked the trade, but every write to the order row
    after the wait was lost: the row still looks orphaned."""
    real_mark, real_note = fake.mark_submitted_order_fill, fake.set_order_reconcile_note
    fake.mark_submitted_order_fill = lambda *a, **k: None
    fake.set_order_reconcile_note = lambda *a, **k: False
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    fake.mark_submitted_order_fill, fake.set_order_reconcile_note = real_mark, real_note
    assert fake.rows[1]["status"] == "submitted" and fake.rows[1]["reconcile_note"] == AWAIT
    assert len(fake.trades) == 1
    fake.rows[1]["created_at"] -= timedelta(minutes=11)
    for t in fake.trades:
        t["timestamp"] -= timedelta(minutes=11)


def test_an_adopted_row_with_a_booked_trade_gets_no_second_trade_row():
    """F3b."""
    fake = FakeOrders()
    _booked_but_row_lost(fake)
    fill = {"filled_qty": 42.0, "fill_price": 101.5, "fill_time": None}
    _reconcile(fake, fill)
    row = fake.rows[1]
    assert row["status"] == "filled" and row["filled_qty"] == 42.0
    assert len(fake.trades) == 1
    fake.mocks["_reprotect_after_late_sell"].assert_not_called()
    sent = [str(c.args[0]) for c in fake.mocks["_send"].call_args_list]
    assert len(sent) == 1 and "ORDER ROW REPAIRED" in sent[0]
    _reconcile(fake, fill)                                     # next cycle: nothing
    assert len(fake.trades) == 1 and fake.mocks["_send"].call_count == 0


def test_an_adopted_row_waits_when_the_booked_trade_check_fails():
    fake = FakeOrders()
    _booked_but_row_lost(fake)
    fake.trade_logged_near = MagicMock(side_effect=RuntimeError("db down"))
    _reconcile(fake, {"filled_qty": 42.0, "fill_price": 101.5, "fill_time": None})
    assert fake.rows[1]["status"] == "unconfirmed" and len(fake.trades) == 1


def test_a_late_fill_trade_row_does_not_count_as_booked():
    fake = FakeOrders()
    fake.trades.append({"symbol": "NVDA", "side": "SELL", "quantity": 42.0,
                        "reason": "EXIT Late-fill reconciled (order placed unconfirmed, ...)",
                        "timestamp": datetime.now(timezone.utc)})
    assert fake.trade_logged_near("NVDA", "SELL", datetime.now(timezone.utc)) == 0.0


def test_trade_logged_near_sql_and_errors():
    conn, cur = _conn()
    at = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.trade_logged_near("NVDA", "SELL", at) == 11.0
    sql, params = cur.execute.call_args.args
    assert "COALESCE(SUM(quantity), 0)" in sql and "LIMIT" not in sql
    assert "FROM trades WHERE symbol=%s AND side=%s AND trading_env=%s" in sql
    assert params[:7] == ("NVDA", "SELL", db.TRADING_ENV, at, 5, at, 20)
    cur.fetchone.return_value = None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.trade_logged_near("NVDA", "SELL", at) == 0.0
    cur.execute.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        with pytest.raises(RuntimeError):
            db.trade_logged_near("NVDA", "SELL", at)


def test_order_row_writers_return_false_when_the_pool_raises():
    import psycopg2.pool
    with patch("db._get_conn", side_effect=psycopg2.pool.PoolError("exhausted")), \
         patch("db._put_conn") as put:
        assert db.finish_submitted_order(5, "cancelled") is False
        assert db.update_order_fill(5, 10.0, 3.0, "filled") is False
        assert db.adopt_orphaned_submitted_orders() is None
    put.assert_not_called()


def test_unconfirmed_orders_carry_the_reconcile_note():
    conn, cur = _conn(fetchall=[(1, "NVDA", "SELL", 42, 501, None, 1, "EXIT", None,
                                 datetime.now(timezone.utc), None, None, None,
                                 db.ORPHANED_FILL_WAIT_NOTE)])
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.get_unconfirmed_orders()
    assert "reconcile_note" in cur.execute.call_args.args[0]
    assert rows[0]["reconcile_note"] == db.ORPHANED_FILL_WAIT_NOTE


# --- quantity-aware repair, pending fill-wait SELLs, guarded late mark -------

def _booked_part_then_row_lost(fake, booked_qty):
    """Like _booked_but_row_lost, but the execution path saw (and booked) a
    fill of booked_qty within its wait."""
    real_mark, real_note = fake.mark_submitted_order_fill, fake.set_order_reconcile_note
    fake.mark_submitted_order_fill = lambda *a, **k: None
    fake.set_order_reconcile_note = lambda *a, **k: False
    place, _ = _broker({"fill_price": 101.5, "filled_qty": booked_qty, "status": "Submitted"})
    with patch("tasks._queue_reprotect"):
        _run_exit(fake, place)
    fake.mark_submitted_order_fill, fake.set_order_reconcile_note = real_mark, real_note
    assert fake.rows[1]["status"] == "submitted" and fake.rows[1]["reconcile_note"] == AWAIT
    assert [t["quantity"] for t in fake.trades] == [booked_qty]
    fake.rows[1]["created_at"] -= timedelta(minutes=11)
    for t in fake.trades:
        t["timestamp"] -= timedelta(minutes=11)


def test_repair_alert_states_the_quantities():
    fake = FakeOrders()
    _booked_but_row_lost(fake)
    _reconcile(fake, {"filled_qty": 42.0, "fill_price": 101.5, "fill_time": None})
    text = fake.mocks["_send"].call_args.args[0]
    assert "ORDER ROW REPAIRED" in text and "filled 42" in text and "42 was already recorded" in text


def test_an_adopted_row_with_part_booked_books_only_the_difference():
    fake = FakeOrders()
    _booked_part_then_row_lost(fake, 20.0)
    fill = {"filled_qty": 42.0, "fill_price": 101.0, "fill_time": None}
    with patch("tasks._reconcile_alert_due", return_value=True):
        _reconcile(fake, fill)
    row = fake.rows[1]
    assert row["status"] == "filled" and row["filled_qty"] == 42.0
    assert [t["quantity"] for t in fake.trades] == [20.0, 22.0]
    assert "Late-fill reconciled" in fake.trades[1]["reason"]
    fake.mocks["_reprotect_after_late_sell"].assert_called_once()     # usual late-fill stop work
    sent = " ".join(str(c.args[0]) for c in fake.mocks["_send"].call_args_list)
    assert "ORDER ROW REPAIRED" in sent and "20 was already recorded" in sent and "other 22" in sent
    _reconcile(fake, fill)                                     # next cycle: nothing new
    assert [t["quantity"] for t in fake.trades] == [20.0, 22.0]


def test_an_adopted_row_with_more_booked_than_filled_is_left_alone():
    fake = FakeOrders()
    _booked_but_row_lost(fake)                                 # 42 booked
    fill = {"filled_qty": 30.0, "fill_price": 101.5, "fill_time": None}
    with patch("tasks._reconcile_alert_due", return_value=True):
        _reconcile(fake, fill)
    assert fake.rows[1]["status"] == "unconfirmed" and fake.rows[1]["filled_qty"] is None
    assert len(fake.trades) == 1
    fake.mocks["_reprotect_after_late_sell"].assert_not_called()
    text = fake.mocks["_send"].call_args.args[0]
    assert "filled 30" in text and "42 shares" in text and "Nothing was marked" in text
    with patch("tasks._reconcile_alert_due", return_value=False):   # throttled after that
        _reconcile(fake, fill)
    fake.mocks["_send"].assert_not_called()
    assert len(fake.trades) == 1


def test_fake_and_db_list_fill_wait_sells():
    fake = FakeOrders()
    place, _ = _broker(raise_after_submit=_Killed())
    with pytest.raises(_Killed):
        _run_exit(fake, place)
    assert fake.awaiting_fill_sell_tickers() == {"NVDA"}
    conn, cur = _conn(fetchall=[("NVDA", True), ("AMD", False)])   # AMD: over a day old
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.awaiting_fill_sell_tickers() == {"NVDA"}
    sql, params = cur.execute.call_args.args
    assert "side='SELL' AND status='submitted'" in sql and "reconcile_note=%s" in sql
    assert "interval '1 day'" in sql
    assert params == (AWAIT, db.TRADING_ENV)
    cur.execute.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        with pytest.raises(RuntimeError):
            db.awaiting_fill_sell_tickers()


def _adopted_during_the_stall(fake, result):
    """place_market_order whose caller stalls past the adoption delay: by
    the time the wait returns, reconcile has adopted the row."""
    inner, _ = _broker(result)

    def place(*a, **k):
        out = inner(*a, **k)
        fake.rows[1]["status"], fake.rows[1]["reconcile_note"] = "unconfirmed", db.ORPHANED_FILL_WAIT_NOTE
        return out
    return place


def test_a_fill_on_an_already_adopted_row_is_left_to_reconcile():
    fake = FakeOrders()
    place = _adopted_during_the_stall(fake, {"fill_price": 101.5, "filled_qty": 42.0,
                                             "status": "Filled"})
    _run_exit(fake, place)
    assert fake.trades == []
    assert fake.rows[1]["status"] == "unconfirmed"            # reconcile owns it
    assert "left to reconcile" in (execution.last_outcome_reason() or "")
    text = fake.mocks["_send"].call_args.args[0]
    assert "FILL LEFT TO RECONCILE" in text
    fake.mocks["delete_position_stop"].assert_not_called()


def test_manual_fill_on_an_already_adopted_row_is_left_to_reconcile():
    fake = FakeOrders()
    place = _adopted_during_the_stall(fake, {"fill_price": 100.0, "filled_qty": 3.0,
                                             "status": "Filled"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("db.upsert_position_stop")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
                m.return_value.calculate_stop.return_value = (92.0, "ATR")
        stop = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
        send = st.enter_context(patch("alerts._send"))
        out = execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert fake.trades == [] and out["success"] is False and "reconcile" in out["message"]
    stop.assert_not_called()
    assert "FILL LEFT TO RECONCILE" in send.call_args.args[0]


def test_mark_submitted_order_fill_sql_and_results():
    conn, cur = _conn(rowcount=0)
    cur.fetchone.return_value = (None, "SELL")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.mark_submitted_order_fill(4, 10.0, 3.0, "filled") == 0
    sql = cur.execute.call_args.args[0]
    assert sql.endswith("WHERE id=%s AND status='submitted'")
    conn, cur = _conn(rowcount=1)
    cur.fetchone.return_value = (None, "SELL")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.mark_submitted_order_fill(4, 10.0, 3.0, "filled") == 1
        assert db.update_order_fill(4, 10.0, 3.0, "filled") is True
    assert "status='submitted'" not in cur.execute.call_args.args[0]   # reconcile's update unguarded
    import psycopg2.pool
    with patch("db._get_conn", side_effect=psycopg2.pool.PoolError("x")):
        assert db.mark_submitted_order_fill(4, 10.0, 3.0, "filled") is None


# --- M211: the booked-trade check is tied to the execution path's own row ----

def test_trade_logged_near_sql_matches_the_rule():
    at = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.trade_logged_near("NVDA", "SELL", at, intent="TRIM")
    sql, params = cur.execute.call_args.args
    assert params[:7] == ("NVDA", "SELL", db.TRADING_ENV, at, 5, at, 20)
    assert params[7:11] == ("Backstop%", "Profit target%", "Reconciled exit%", "%Late-fill reconciled%")
    assert sql.count("NOT LIKE %s") == 4 and sql.rstrip().endswith("AND COALESCE(reason, '') LIKE %s")
    assert params[11] == "TRIM % (conviction=%"
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.trade_logged_near("ZZZ", "BUY", at, intent="MANUAL_OPEN")
    sql, params = cur.execute.call_args.args
    assert sql.count("NOT LIKE %s") == 5 and params[-1] == "%(conviction=%/10)%"


def _like(pattern, text):
    """Postgres LIKE (% and _, backslash escapes) for checking the patterns."""
    import re
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            out += re.escape(pattern[i + 1]); i += 2; continue
        out += ".*" if c == "%" else "." if c == "_" else re.escape(c)
        i += 1
    return re.fullmatch(out, text, re.S) is not None


@pytest.mark.parametrize("reason,intent,counts", [
    ("TRIM approved via Telegram (conviction=7/10)", "TRIM", True),
    ("TRIM executed (trigger=x) (conviction=7/10)", "TRIM", True),
    ("EXIT auto-executed (conviction=7/10)", "TRIM", False),          # another approval order
    ("Backstop stop-breach sell (stop $90.00)", "TRIM", False),
    ("Profit target partial sell", "TRIM", False),
    ("Profit target hit (12.0%, ...)", "TRIM", False),
    ("Reconciled exit — position closed at the broker with no ...", "TRIM", False),
    ("TRIM Late-fill reconciled (order placed unconfirmed, ...)", "TRIM", False),
    ("Manual trade from dashboard", "MANUAL_OPEN", True),
    ("!urgent chat command", "MANUAL_BUY", True),
    ("Swap — freeing capital for X (conviction 8/10); ...", "MANUAL_SELL", True),
    ("OPEN auto-executed (conviction=8/10)", "MANUAL_ADD", False),     # an approval order
    ("Backstop stop-breach sell (stop $90.00) [retry]", "MANUAL_SELL", False),
])
def test_the_sql_patterns_select_only_the_execution_paths_rows(reason, intent, counts):
    at = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.trade_logged_near("NVDA", "SELL", at, intent=intent)
    sql, params = cur.execute.call_args.args
    likes = params[7:]
    n_not = sql.count("NOT LIKE %s")
    matched = all(not _like(p, reason) for p in likes[:n_not]) and \
        all(_like(p, reason) for p in likes[n_not:])
    assert matched is counts


def test_an_orphaned_trim_is_booked_despite_a_backstop_sale_in_the_window():
    """Review M211: the backstop's 22 used to count as the TRIM's 20 already
    booked (22 > 20), so the TRIM was never booked."""
    fake = FakeOrders()
    place, _ = _broker(raise_after_submit=_Killed())
    with pytest.raises(_Killed):
        _run_exit(fake, place)
    fake.rows[1].update(qty=20.0, intent="TRIM")
    fake.trades.append({"symbol": "NVDA", "side": "SELL", "quantity": 22.0,
                        "reason": "Backstop stop-breach sell (stop $90.00)",
                        "timestamp": datetime.now(timezone.utc) + timedelta(seconds=40)})
    fake.rows[1]["created_at"] -= timedelta(minutes=11)
    for t in fake.trades:
        t["timestamp"] -= timedelta(minutes=11)
    _reconcile(fake, {"filled_qty": 20.0, "fill_price": 99.0, "fill_time": None})
    assert fake.rows[1]["status"] == "filled"
    assert [t["quantity"] for t in fake.trades] == [22.0, 20.0]
    assert "Late-fill reconciled" in fake.trades[1]["reason"]


def _trade(fake, seconds_after, qty=5.0, reason="TRIM auto-executed (conviction=6/10)",
           around=None):
    fake.trades.append({"symbol": "NVDA", "side": "SELL", "quantity": qty, "reason": reason,
                        "timestamp": (around or fake.rows[1]["created_at"])
                        + timedelta(seconds=seconds_after)})


def _one_order(fake, **kw):
    return fake.log_order(**{"ticker": "NVDA", "side": "SELL", "order_type": "MARKET", "qty": 5,
                             "broker_order_id": 700, "intent": "TRIM", "status": "filled", **kw})


@pytest.mark.parametrize("seconds_after", [-4, 35, 300, 590, 1190])
def test_a_slow_execution_trade_row_is_still_found(seconds_after):
    """The execution path's trade row has no fixed bound (a priceless fill
    waits on get_latest_price, the DB calls have no timeouts): anything up
    to the 20-min cap counts while no later same-side order exists."""
    fake = FakeOrders()
    oid = _one_order(fake)
    _trade(fake, seconds_after)
    assert fake.trade_logged_near("NVDA", "SELL", fake.rows[oid]["created_at"], intent="TRIM",
                                  order_id=oid) == 5.0


@pytest.mark.parametrize("seconds_after", [-6, 1201])
def test_a_trade_row_outside_the_window_is_not_counted(seconds_after):
    fake = FakeOrders()
    oid = _one_order(fake)
    _trade(fake, seconds_after)
    assert fake.trade_logged_near("NVDA", "SELL", fake.rows[oid]["created_at"], intent="TRIM",
                                  order_id=oid) == 0.0


def test_the_next_same_side_orders_trade_is_not_counted():
    fake = FakeOrders()
    first = _one_order(fake)
    second = _one_order(fake, broker_order_id=701)
    fake.rows[second]["created_at"] = fake.rows[first]["created_at"] + timedelta(seconds=120)
    _trade(fake, 150, qty=7.0)                                 # the second order's row
    assert fake.trade_logged_near("NVDA", "SELL", fake.rows[first]["created_at"], intent="TRIM",
                                  order_id=first) == 0.0
    assert fake.trade_logged_near("NVDA", "SELL", fake.rows[second]["created_at"], intent="TRIM",
                                  order_id=second) == 7.0
    other = _one_order(fake, side="BUY", broker_order_id=702)    # another side doesn't cut it
    fake.rows[other]["created_at"] = fake.rows[first]["created_at"] + timedelta(seconds=10)
    _trade(fake, 60, qty=3.0)
    assert fake.trade_logged_near("NVDA", "SELL", fake.rows[first]["created_at"], intent="TRIM",
                                  order_id=first) == 3.0


def test_the_window_sql_stops_at_the_next_same_side_order_and_the_cap():
    at = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    conn, cur = _conn()
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.trade_logged_near("NVDA", "SELL", at, intent="TRIM", order_id=41)
    sql, params = cur.execute.call_args.args
    assert "timestamp >= %s - make_interval(secs => %s)" in sql
    assert ("timestamp < LEAST(COALESCE((SELECT MIN(o.created_at) FROM orders o WHERE "
            "o.ticker=%s AND o.side=%s AND o.trading_env=%s AND o.id > %s), "
            "%s + make_interval(mins => %s)), %s + make_interval(mins => %s))") in sql
    assert params[:13] == ("NVDA", "SELL", db.TRADING_ENV, at, 5, "NVDA", "SELL",
                           db.TRADING_ENV, 41, at, 20, at, 20)
    assert "BETWEEN" not in sql


def test_the_window_cap_stays_above_the_adoption_and_recovery_ages():
    import inspect
    adopt = inspect.signature(db.adopt_orphaned_submitted_orders).parameters["min_age_minutes"]
    recover = inspect.signature(tasks._recover_trade_pending_orders).parameters["min_age_minutes"]
    stale = inspect.signature(db.get_stale_trade_pending_orders).parameters["min_age_minutes"]
    assert db.TRADE_MATCH_CAP_MINUTES > max(adopt.default, recover.default, stale.default)
    src = inspect.getsource(tasks._reconcile_unconfirmed_orders_locked)
    assert "_adopt_orphaned_submitted_orders()" in src and "_recover_trade_pending_orders()" in src
    assert "db.adopt_orphaned_submitted_orders()" in inspect.getsource(tasks._adopt_orphaned_submitted_orders)

# --- M212: an early write that landed although it reported a failure ---------

@pytest.mark.parametrize("result,status", [
    ({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"}, "filled"),
    ({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"}, "unconfirmed"),
    ({"fill_price": None, "filled_qty": 0.0, "status": "Inactive"}, "cancelled"),
])
def test_a_landed_early_write_is_reused_not_duplicated(result, status):
    fake = FakeOrders(fail_early_write=True, early_write_lands_anyway=True)
    place, _ = _broker(result)
    _run_exit(fake, place)
    assert len(fake.rows) == 1 and fake.rows[1]["status"] == status
    assert fake.rows[1]["reconcile_note"] is None


def test_manual_landed_early_write_is_reused():
    fake = FakeOrders(fail_early_write=True, early_write_lands_anyway=True)
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place), patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    assert [r["status"] for r in fake.rows.values()] == ["unconfirmed"]


def test_a_failed_row_check_falls_back_to_the_insert_and_alerts():
    fake = FakeOrders(fail_early_write=True, early_write_lands_anyway=True)
    fake.find_awaiting_order_row = MagicMock(side_effect=RuntimeError("db down"))
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert sorted(r["status"] for r in fake.rows.values()) == ["submitted", "unconfirmed"]
    sent = " ".join(str(c.args[0]) for c in fake.mocks["_send"].call_args_list)
    assert "ORDER ROW CHECK FAILED" in sent


def test_no_lookup_when_the_early_write_was_never_tried_or_worked():
    fake = FakeOrders()
    fake.find_awaiting_order_row = MagicMock(return_value=None)
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    fake.find_awaiting_order_row.assert_not_called()
    fake = FakeOrders(fail_early_write=True)                       # nothing landed
    fake.find_awaiting_order_row = MagicMock(return_value=None)
    _run_exit(fake, place)
    fake.find_awaiting_order_row.assert_called_once_with("NVDA", "SELL", 501)
    assert len(fake.rows) == 1


def test_find_awaiting_order_row_sql():
    conn, cur = _conn()
    cur.fetchone.return_value = (17,)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.find_awaiting_order_row("NVDA", "SELL", 501) == 17
    sql, params = cur.execute.call_args.args
    assert "trading_env=%s AND ticker=%s AND side=%s AND broker_order_id=%s" in sql
    assert "status='submitted' AND reconcile_note=%s" in sql and "INTERVAL '10 minutes'" in sql
    assert params == (db.TRADING_ENV, "NVDA", "SELL", 501, AWAIT)
    cur.fetchone.return_value = None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.find_awaiting_order_row("NVDA", "SELL", 501) is None
    cur.execute.side_effect = RuntimeError("down")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        with pytest.raises(RuntimeError):
            db.find_awaiting_order_row("NVDA", "SELL", 501)


# --- M213: a worker stopped between the fill mark and the trade row ----------

def _age(fake, minutes=11):
    for r in fake.rows.values():
        r["created_at"] -= timedelta(minutes=minutes)
    for t in fake.trades:
        t["timestamp"] -= timedelta(minutes=minutes)


def _killed_before_trade_row(fake, result):
    """The fill is marked, then the worker dies before the trade row commits."""
    real = fake.log_trade

    def dies(**kw):
        raise _Killed()
    fake.log_trade = dies
    place, _ = _broker(result)
    with pytest.raises(_Killed):
        _run_exit(fake, place)
    fake.log_trade = real


def test_the_trade_pending_note_is_set_with_the_fill_and_cleared_after_the_trade_row():
    fake = FakeOrders()
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert fake.events == ["order:filled", "trade", "note:None"]
    assert fake.rows[1]["reconcile_note"] is None


def test_a_failed_trade_row_transaction_rolls_back_without_a_spill_and_reconcile_books_once():
    """F1: an insert or commit error rolls the claim back too, so the note
    stays trade_pending (nothing spilled) and reconcile books it later; the
    execution path still does its stop work now."""
    fake = FakeOrders()
    fake.claim_and_log_trade = lambda *a, **k: None          # rolled back
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    with patch("db._spill") as spill:
        _run_exit(fake, place)
    spill.assert_not_called()
    assert fake.trades == [] and fake.rows[1]["reconcile_note"] == db.TRADE_PENDING_STOP_DONE_NOTE
    fake.mocks["delete_position_stop"].assert_called_once_with("NVDA")   # stop work went ahead
    del fake.claim_and_log_trade                              # back to the real one
    _age(fake)
    _reconcile(fake, None)
    _reconcile(fake, None)
    assert [t["quantity"] for t in fake.trades] == [42.0]
    assert fake.rows[1]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    fake.mocks["_reprotect_after_late_sell"].assert_not_called()       # stop work not redone


def test_claim_and_log_trade_is_one_transaction():
    conn, cur = _conn(rowcount=1)
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"), \
         patch("db._spill") as spill, patch("db._record_write_failure") as rec:
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r", "fill") is True
        upd, ins = [c.args for c in cur.execute.call_args_list]
        assert upd[0].startswith("UPDATE orders SET reconcile_note=%s") and \
            "WHERE id=%s AND reconcile_note=%s" in upd[0]
        assert upd[1] == (None, 4, db.TRADE_PENDING_NOTE)
        assert ins[0].startswith("INSERT INTO trades (symbol, trading_env, side, quantity, price, "
                                 "reason, price_source)")
        assert ins[1] == ("NVDA", db.TRADING_ENV, "SELL", 3, 10.0, "r", "fill")
        conn.commit.assert_called_once()

        conn, cur = _conn(rowcount=0)                         # someone else claimed it
        with patch("db._get_conn", return_value=conn):
            assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r") is False
        assert cur.execute.call_count == 1                    # no INSERT
        conn.rollback.assert_called_once(); conn.commit.assert_not_called()

        conn, cur = _conn(rowcount=1)
        cur.execute.side_effect = [None, RuntimeError("insert failed")]
        with patch("db._get_conn", return_value=conn):
            assert db.claim_and_log_trade(4, "x", "NVDA", "SELL", 3, 10.0, "r") is None
        conn.rollback.assert_called(); conn.commit.assert_not_called()

        conn, cur = _conn(rowcount=1)
        conn.commit.side_effect = RuntimeError("commit failed")
        with patch("db._get_conn", return_value=conn):
            assert db.claim_and_log_trade(4, "x", "NVDA", "SELL", 3, float("nan"), "r") is None
        assert cur.execute.call_args_list[1].args[1][4] is None          # log_trade's price rule
        conn.rollback.assert_called()
    spill.assert_not_called(); rec.assert_not_called()

def test_a_kill_before_the_trade_row_is_booked_once_by_reconcile():
    fake = FakeOrders()
    _killed_before_trade_row(fake, {"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    row = fake.rows[1]
    assert row["status"] == "filled" and row["reconcile_note"] == db.TRADE_PENDING_NOTE
    assert fake.trades == []

    _reconcile(fake, None)                                    # too young: left for the worker
    assert fake.trades == [] and row["reconcile_note"] == db.TRADE_PENDING_NOTE

    _age(fake)
    _reconcile(fake, None)
    assert [t["quantity"] for t in fake.trades] == [42.0]
    assert fake.trades[0]["price"] == 101.5 and "Late-fill reconciled" in fake.trades[0]["reason"]
    assert row["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE and row["status"] == "filled"
    # the SELL's skipped stop work: re-protect the rest / close when flat
    fake.mocks["_reprotect_after_late_sell"].assert_called_once()
    assert fake.mocks["_reprotect_after_late_sell"].call_args.args[0] == "NVDA"
    sent = [str(c.args[0]) for c in fake.mocks["_send"].call_args_list]
    assert len(sent) == 1 and "TRADE ROW RECOVERED" in sent[0]

    _reconcile(fake, None)                                    # once only
    assert len(fake.trades) == 1


def _pending_row(fake, qty=42.0, price=101.5, side="SELL", intent="EXIT"):
    rid = fake.log_order(ticker="NVDA", side=side, order_type="MARKET", qty=qty, broker_order_id=501,
                         intent=intent, approval_id=1, status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, price, qty, "filled")
    return rid


def test_a_trade_pending_row_whose_trade_row_matches_only_has_the_note_cleared():
    fake = FakeOrders()
    rid = _pending_row(fake)
    _trade(fake, 30, qty=42.0, reason="EXIT auto-executed (conviction=7/10)")
    _age(fake)
    _reconcile(fake, None)
    assert len(fake.trades) == 1 and fake.rows[rid]["reconcile_note"] is None
    fake.mocks["_reprotect_after_late_sell"].assert_not_called()
    sent = [str(c.args[0]) for c in fake.mocks["_send"].call_args_list]
    assert len(sent) == 1 and "TRADE ROW CHECK" in sent[0]


@pytest.mark.parametrize("booked", [20.0, 50.0])
def test_a_trade_pending_row_with_a_different_booked_quantity_is_left_and_alerted(booked):
    fake = FakeOrders()
    rid = _pending_row(fake)
    _trade(fake, 30, qty=booked, reason="EXIT auto-executed (conviction=7/10)")
    _age(fake)
    with patch("tasks._reconcile_alert_due", return_value=True):
        _reconcile(fake, None)
    assert len(fake.trades) == 1 and fake.rows[rid]["reconcile_note"] == db.TRADE_PENDING_NOTE
    fake.mocks["_reprotect_after_late_sell"].assert_not_called()
    text = fake.mocks["_send"].call_args.args[0]
    assert "TRADE ROW MISMATCH" in text and f"{booked:g} shares" in text and "filled 42" in text
    with patch("tasks._reconcile_alert_due", return_value=False):
        _reconcile(fake, None)
    fake.mocks["_send"].assert_not_called()


def test_the_race_reconcile_claims_between_the_mark_and_the_trade_row():
    """F1: reconcile's recovery runs after the mark and before the execution
    path's trade-row transaction: exactly one trade row, and the stop and
    outcome work done once (by reconcile; the execution path stops)."""
    fake = FakeOrders()
    real_mark = fake.mark_submitted_order_fill
    reprotect = MagicMock(return_value=True)

    def mark_then_reconcile_runs(*a, **k):
        n = real_mark(*a, **k)
        _age(fake)                                       # the worker stalls past the delay
        with patch("tasks._reprotect_after_late_sell", reprotect), \
             patch("db.get_approval_by_id", return_value=None):
            tasks._recover_trade_pending_orders()
        return n
    fake.mark_submitted_order_fill = mark_then_reconcile_runs
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert [t["quantity"] for t in fake.trades] == [42.0]
    assert "Late-fill reconciled" in fake.trades[0]["reason"]
    reprotect.assert_called_once()                       # reconcile's stop work, once
    fake.mocks["delete_position_stop"].assert_not_called()      # none by the execution path
    fake.mocks["close_trade_outcome"].assert_not_called()
    assert "left to reconcile" in (execution.last_outcome_reason() or "")


@pytest.mark.parametrize("problem", ["no_qty", "check_raises", "no_price", "rolled_back"])
def test_a_stuck_trade_pending_row_alerts_after_30_min(problem):
    """F4: a row reconcile can't resolve alerts (throttled) once 30 min old."""
    fake = FakeOrders()
    rid = _pending_row(fake, price=None if problem == "no_price" else 101.5)
    if problem == "no_qty":
        fake.rows[rid]["filled_qty"] = 0.0
    if problem == "check_raises":
        fake.trade_logged_near = MagicMock(side_effect=RuntimeError("db down"))
    if problem == "rolled_back":
        fake.claim_and_log_trade = lambda *a, **k: None
    _age(fake, 20)
    with patch("tasks._reconcile_alert_due", return_value=True), \
         patch("ibkr.get_latest_price", return_value=None):
        _reconcile(fake, None)
    fake.mocks["_send"].assert_not_called()                       # 20 min: not yet
    _age(fake, 11)
    with patch("tasks._reconcile_alert_due", return_value=True) as due, \
         patch("ibkr.get_latest_price", return_value=None):
        _reconcile(fake, None)
    assert fake.trades == []
    text = fake.mocks["_send"].call_args.args[0]
    assert "TRADE ROW PENDING" in text
    assert any(c.args[0] == f"trade_pending_stuck_{rid}" for c in due.call_args_list)
    with patch("tasks._reconcile_alert_due", return_value=False), \
         patch("ibkr.get_latest_price", return_value=None):
        _reconcile(fake, None)
    fake.mocks["_send"].assert_not_called()                       # throttled


def test_left_to_reconcile_is_recorded_approved_not_failed():
    """F5."""
    execution._outcome(f"{execution._LEFT_TO_RECONCILE} (trade row recovered)")
    assert execution.approval_outcome(False) == (
        "APPROVED", "skipped: fill left to reconcile (trade row recovered)")
    execution._outcome("skipped: sell Inactive unfilled, stop restored")
    assert execution.approval_outcome(False) == ("FAILED", "skipped: sell Inactive unfilled, stop restored")
    assert execution.approval_outcome(True) == ("APPROVED", None)


def test_a_part_filled_trade_pending_row_books_its_part_then_the_rest_once():
    fake = FakeOrders()
    with patch("tasks._queue_reprotect"):
        _killed_before_trade_row(fake, {"fill_price": 101.5, "filled_qty": 20.0,
                                        "status": "Submitted"})
    assert fake.rows[1]["status"] == "partial"
    fill = {"filled_qty": 42.0, "fill_price": 101.0, "fill_time": None}
    _reconcile(fake, fill)                                    # young: the main loop waits too
    assert fake.trades == []
    _age(fake)
    _reconcile(fake, fill)
    assert [t["quantity"] for t in fake.trades] == [20.0, 22.0]
    assert fake.rows[1]["status"] == "filled" and fake.rows[1]["filled_qty"] == 42.0
    _reconcile(fake, fill)
    assert len(fake.trades) == 2


def test_a_stalled_worker_writes_no_trade_row_once_reconcile_recovered_it():
    fake = FakeOrders()
    real_mark = fake.mark_submitted_order_fill

    def mark_then_stall(*a, **k):
        n = real_mark(*a, **k)
        # ... the worker stalls past the delay; reconcile claims and books it
        fake.rows[1]["reconcile_note"] = db.TRADE_BOOKED_BY_RECONCILE_NOTE
        return n
    fake.mark_submitted_order_fill = mark_then_stall
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert fake.trades == []
    assert "left to reconcile" in (execution.last_outcome_reason() or "")
    assert "FILL LEFT TO RECONCILE" in fake.mocks["_send"].call_args.args[0]
    fake.mocks["delete_position_stop"].assert_not_called()


def test_a_trade_pending_buy_gets_the_outcome_and_one_whole_position_stop():
    """The work execution skipped after the trade row: outcome insert, stop
    record, and the stop placed by _reprotect_after_late_sell (sized to
    what resting stops leave, so no second stop)."""
    fake = FakeOrders()
    fake.log_order(ticker="NVDA", side="BUY", order_type="MARKET", qty=10, broker_order_id=601,
                   approval_id=7, intent="OPEN", status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(1, 50.0, 10.0, "filled")
    _age(fake)
    approval = {"id": 7, "action": "OPEN", "stop_loss_price": 45.0, "status": "EXECUTED",
                "thesis_bucket": "Accelerator", "conviction": 7,
                "risk_json": {"stop_loss_price": 45.0, "price_at_sizing": 50.0}}
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        st.enter_context(patch("db.get_approval_by_id", return_value=approval))
        st.enter_context(patch("db.get_position_stop", return_value=None))
        upsert = st.enter_context(patch("db.upsert_position_stop"))
        outcome = st.enter_context(patch("tasks._apply_buy_outcome"))
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=True))
        st.enter_context(patch("alerts._send"))
        tasks._recover_trade_pending_orders()
        tasks._recover_trade_pending_orders()                 # a second run does nothing
    assert [t["quantity"] for t in fake.trades] == [10.0]
    outcome.assert_called_once()
    assert outcome.call_args.args[1]["kind"] == "insert" and outcome.call_args.args[1]["entry_price"] == 50.0
    upsert.assert_called_once()
    reprotect.assert_called_once_with("NVDA", {}, None, from_queue=True)


def test_a_priceless_trade_pending_row_waits_for_a_price():
    fake = FakeOrders()
    fake.log_order(ticker="NVDA", side="SELL", order_type="MARKET", qty=5, broker_order_id=602,
                   intent="TRIM", status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(1, None, 5.0, "filled")
    _age(fake)
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        st.enter_context(patch("ibkr.get_fill_for_order", return_value=None))
        st.enter_context(patch("ibkr.get_latest_price", return_value=None))
        tasks._recover_trade_pending_orders()
    assert fake.trades == [] and fake.rows[1]["reconcile_note"] == db.TRADE_PENDING_NOTE


def test_trade_pending_sql():
    conn, cur = _conn(rowcount=1)
    cur.fetchone.return_value = (None, "SELL")
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.mark_submitted_order_fill(4, 10.0, 3.0, "filled") == 1
    sql, params = cur.execute.call_args.args
    assert ", reconcile_note=%s," in sql and sql.endswith("WHERE id=%s AND status='submitted'")
    assert params[-2] == db.TRADE_PENDING_NOTE
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.update_order_fill(4, 10.0, 3.0, "filled") is True
    assert "NULLIF(reconcile_note, %s)" in cur.execute.call_args.args[0]   # reconcile's own write

    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.claim_trade_pending(4, None) == 1
    sql, params = cur.execute.call_args.args
    assert "WHERE id=%s AND reconcile_note=%s" in sql and params == (None, 4, db.TRADE_PENDING_NOTE)

    now = datetime.now(timezone.utc)
    cur.fetchall.return_value = [(4, "NVDA", "SELL", 5, 602, None, None, "TRIM", None, now,
                                  5, 10.0, None, None, "filled", db.TRADE_PENDING_NOTE)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        rows = db.get_stale_trade_pending_orders(10)
    sql, params = cur.execute.call_args.args
    assert "status IN ('filled', 'partial') AND reconcile_note IN (%s, %s) AND trading_env=%s" in sql
    assert params == (db.TRADE_PENDING_NOTE, db.TRADE_PENDING_STOP_DONE_NOTE, db.TRADING_ENV, 10)
    assert rows[0]["filled_qty"] == 5.0 and rows[0]["intent"] == "TRIM"
    assert rows[0]["reconcile_note"] == db.TRADE_PENDING_NOTE

    import psycopg2.pool
    with patch("db._get_conn", side_effect=psycopg2.pool.PoolError("x")):
        assert db.claim_trade_pending(4, None) is None
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r") is None


def test_log_trade_reports_success():
    conn, cur = _conn()
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x") is True
    cur.execute.side_effect = RuntimeError("down")
    with patch("db._writer_conn", return_value=conn), patch("db._put_conn"), \
         patch("db._record_write_failure"):
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x") is False
    with patch("db._writer_conn", return_value=None):
        assert db.log_trade("NVDA", "SELL", 3, 10.0, "x") is False


# --- m279: the migration-75 fallback keeps reconcile_note --------------------

def test_mig75_fallback_keeps_the_note():
    import psycopg2.errors
    conn, cur = _conn()
    cur.execute.side_effect = [psycopg2.errors.UndefinedColumn("submitted_at"), None]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=7,
                            reconcile_note=AWAIT) == 11
    sql, params = cur.execute.call_args.args
    assert "submitted_at" not in sql and "reconcile_note" in sql and params[-1] == AWAIT


def test_mig75_and_76_fallback_drops_the_note_last():
    import psycopg2.errors
    conn, cur = _conn()
    cur.execute.side_effect = [psycopg2.errors.UndefinedColumn("submitted_at"),
                               psycopg2.errors.UndefinedColumn("reconcile_note"), None]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=7,
                            reconcile_note=AWAIT) == 11
    sql, params = cur.execute.call_args.args
    assert "reconcile_note" not in sql and len(params) == 14


def test_mig75_fallback_without_a_note_is_unchanged():
    import psycopg2.errors
    conn, cur = _conn()
    cur.execute.side_effect = [psycopg2.errors.UndefinedColumn("submitted_at"), None]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.log_order("NVDA", "SELL", "MARKET", 3, broker_order_id=7) == 11
    assert cur.execute.call_count == 2 and "reconcile_note" not in cur.execute.call_args.args[0]


@pytest.mark.parametrize("other", ["another_env", "old", "other_ticker", "other_side", "not_awaiting"])
def test_an_unrelated_row_with_the_same_broker_id_is_never_reused(other):
    """broker_order_id repeats across accounts and sessions (id 320: a paper
    ORCL BUY and a live PLTR BUY): the fallback must insert its own row and
    leave the other one untouched."""
    fake = FakeOrders()
    rid = fake.log_order(ticker="NVDA", side="SELL", order_type="MARKET", qty=42,
                         broker_order_id=501, status="submitted", reconcile_note=AWAIT)
    fake.fail_early_write = True                     # this order's own early write fails
    r = fake.rows[rid]
    if other == "another_env":
        r["trading_env"] = "paper" if db.TRADING_ENV != "paper" else "live"
    elif other == "old":
        r["created_at"] -= timedelta(minutes=11)
    elif other == "other_ticker":
        r["ticker"] = "ORCL"
    elif other == "other_side":
        r["side"] = "BUY"
    else:
        r["status"], r["reconcile_note"] = "filled", None
    before = dict(r)
    place, _ = _broker({"fill_price": None, "filled_qty": 0.0, "status": "Submitted"})
    _run_exit(fake, place)
    assert len(fake.rows) == 2
    assert fake.rows[rid] == before                              # untouched
    assert fake.rows[2]["status"] == "unconfirmed"               # its own row


def test_the_row_check_sql_scopes_env_ticker_side_state_and_ten_minutes():
    conn, cur = _conn()
    cur.fetchone.return_value = None
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.find_awaiting_order_row("NVDA", "SELL", 320) is None
    sql, params = cur.execute.call_args.args
    for clause in ("trading_env=%s", "ticker=%s", "side=%s", "broker_order_id=%s",
                   "status='submitted'", "reconcile_note=%s",
                   "created_at > NOW() - INTERVAL '10 minutes'"):
        assert clause in sql
    assert params == (db.TRADING_ENV, "NVDA", "SELL", 320, AWAIT)


# --- error-path BUYs: recovery must not rewrite the stop record ---------------

def _fail_trade_row_once(fake):
    """The execution path's trade-row transaction fails (rolled back)."""
    real = fake.claim_and_log_trade
    calls = {"n": 0}

    def claim(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(*a, **k)
    fake.claim_and_log_trade = claim


def _fail_stop_done_note(fake):
    real = fake.claim_trade_pending

    def claim(order_id, note, expect_note=db.TRADE_PENDING_NOTE):
        if note == db.TRADE_PENDING_STOP_DONE_NOTE:
            return None                                  # the note write fails
        return real(order_id, note, expect_note=expect_note)
    fake.claim_trade_pending = claim


def _run_manual_open(fake):
    place, _ = _broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        client = st.enter_context(patch("ibkr.get_client"))
        client.return_value.get_open_position.side_effect = RuntimeError("No open position")
        for p in (patch("risk_manager.RiskManager"),
                  patch("market_data.get_priced_quote", return_value={"price": 100.0}),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("db.set_stop_broker_order_id"), patch("alerts._send")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
                m.return_value.calculate_stop.return_value = (92.0, "ATR")
        stop = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
        execution.execute_manual("ZZZ", "BUY", 3, "test")
    return stop


def _open_approval():
    return {"id": 1, "ticker": "ZZZ", "action": "OPEN", "conviction": 7,
            "thesis_bucket": "Moonshot", "stop_loss_price": 90.0,
            "risk_json": {"shares_initial": 3, "price_at_sizing": 100.0, "stop_loss_price": 90.0,
                          "stop_method": "2.0x ATR"}}


def _run_approval_open(fake):
    place, _ = _broker({"fill_price": 100.0, "filled_qty": 3.0, "status": "Filled"})
    quote = {"price": 100.0, "is_stale": False, "age_minutes": 1.0, "market_open": True}
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        for p in (patch("ibkr.get_ib"), patch("ibkr.get_client"), patch("risk_manager.RiskManager"),
                  patch("ibkr.place_market_order", side_effect=place),
                  patch("ibkr.get_latest_price", return_value=100.0),
                  patch("market_data.get_priced_quote", return_value=quote),
                  patch("market_data.is_market_hours", return_value=True),
                  patch("market_data.check_gap_confirmation", return_value={"status": "no_gap"}),
                  patch("config.USE_BRACKET_ORDERS", False), patch("time.sleep"),
                  patch("db.set_stop_broker_order_id"), patch("alerts._send"),
                  patch("alerts.send_trade_alert"),
                  patch("execution._capture_original_thesis_if_new")):
            m = st.enter_context(p)
            if p.attribute == "RiskManager":
                m.return_value.is_halted.return_value = False
        stop = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 9}))
        assert execution.execute_approved(_open_approval()) is True
    return stop


def _recover(fake, approval=None):
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        st.enter_context(patch("db.get_approval_by_id", return_value=approval))
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=True))
        stop = st.enter_context(patch("tasks.manage_stop_place", return_value={"order_id": 10}))
        send = st.enter_context(patch("alerts._send"))
        st.enter_context(patch("tasks._held_before_fill", return_value=False))
        tasks._recover_trade_pending_orders()
        tasks._recover_trade_pending_orders()            # a second run does nothing
    return reprotect, stop, send


def _trailed(fake, ticker):
    """What can happen to the record in the 10 min before recovery."""
    fake.stops[ticker].update(stop_price=97.5, trailing_active=True, profit_target_taken=True)
    return dict(fake.stops[ticker])


@pytest.mark.parametrize("note_write_fails", [False, True])
@pytest.mark.parametrize("kind", ["manual", "approval"])
def test_an_error_path_buy_keeps_its_stop_record_through_recovery(kind, note_write_fails):
    fake = FakeOrders()
    _fail_trade_row_once(fake)
    if note_write_fails:
        _fail_stop_done_note(fake)
    ticker = "ZZZ"
    exec_stop = _run_manual_open(fake) if kind == "manual" else _run_approval_open(fake)
    assert fake.trades == []
    assert exec_stop.call_count == 1                              # the execution path's stop
    expected_note = db.TRADE_PENDING_NOTE if note_write_fails else db.TRADE_PENDING_STOP_DONE_NOTE
    assert fake.rows[1]["reconcile_note"] == expected_note
    record = _trailed(fake, ticker)
    outcomes_before = dict(fake.outcomes)

    _age(fake)
    reprotect, rec_stop, send = _recover(fake, _open_approval() if kind == "approval" else None)

    assert [t["quantity"] for t in fake.trades] == [3.0]           # exactly one trade row
    assert fake.rows[1]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    assert fake.stops[ticker] == record                            # record untouched
    assert fake.outcomes == outcomes_before                        # one outcome at most
    if kind == "approval":
        assert list(fake.outcomes) == [1]
    rec_stop.assert_not_called()                                   # no extra stop
    sent = " ".join(str(c.args[0]) for c in send.call_args_list)
    if note_write_fails:
        reprotect.assert_called_once_with(ticker, {}, None, from_queue=True)   # cover-only
        assert "STOP RECORD KEPT" in sent
    else:
        reprotect.assert_not_called()
        assert "TRADE ROW RECOVERED" in sent


def test_a_leftover_record_of_a_closed_position_is_still_replaced():
    """The kill case (no execution stop work): a record from an earlier
    position, older than the order row, is not this order's."""
    fake = FakeOrders()
    fake.upsert_position_stop("ZZZ", 60.0, 55.0, "old", "Accelerator", 5)
    fake.stops["ZZZ"]["updated_at"] -= timedelta(days=3)
    rid = fake.log_order(ticker="ZZZ", side="BUY", order_type="MARKET", qty=3, broker_order_id=801,
                         intent="OPEN", approval_id=1, status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, 100.0, 3.0, "filled")
    _age(fake)
    fake.stops["ZZZ"]["updated_at"] -= timedelta(days=3)
    _recover(fake, _open_approval())
    assert fake.stops["ZZZ"]["entry_price"] == 100.0 and fake.stops["ZZZ"]["stop_method"] == "2.0x ATR"


def _bumped_leftover(fake, ticker="ZZZ", entry=100.0, method="2.0x ATR", bucket="Moonshot",
                     conviction=7, intent="OPEN"):
    """A closed position's record, then this order's row, then a write that
    bumps the leftover's updated_at (e.g. a re-stop setting its broker id)."""
    fake.upsert_position_stop(ticker, entry, 55.0, method, bucket, conviction)
    fake.stops[ticker]["updated_at"] -= timedelta(days=3)
    rid = fake.log_order(ticker=ticker, side="BUY", order_type="MARKET", qty=3, broker_order_id=802,
                         intent=intent, approval_id=1 if intent == "OPEN" else None,
                         status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, 100.0, 3.0, "filled")
    _age(fake)
    fake.stops[ticker]["updated_at"] = datetime.now(timezone.utc)
    return rid


@pytest.mark.parametrize("field,value", [
    ("entry", 100.4),             # within 0.5% of the fill, not to the cent
    ("entry", 60.0),
    ("method", "1.5x ATR"),
    ("bucket", "Accelerator"),
    ("conviction", 5),
])
def test_a_bumped_leftover_unlike_this_orders_record_is_replaced(field, value):
    fake = FakeOrders()
    _bumped_leftover(fake, **{field: value})
    _recover(fake, _open_approval())
    rec = fake.stops["ZZZ"]
    assert rec["entry_price"] == 100.0 and rec["stop_price"] != 55.0
    assert (rec["stop_method"], rec["thesis_bucket"], rec["conviction"]) == ("2.0x ATR", "Moonshot", 7)


@pytest.mark.parametrize("conviction,kept", [(7, True), (5, False)])
def test_a_manual_buy_record_needs_the_manual_conviction(conviction, kept):
    fake = FakeOrders()
    _bumped_leftover(fake, method="ATR", bucket="Accelerator", conviction=conviction,
                     intent="MANUAL_OPEN")
    before = dict(fake.stops["ZZZ"])
    # A last quote above the flat fallback stop, so it does not lower it.
    with patch("market_data.get_priced_quote", return_value={"price": 101.0}) as quote:
        _recover(fake, None)
    assert (fake.stops["ZZZ"] == before) is kept
    if not kept:
        quote.assert_called_once_with("ZZZ")       # the new record was checked against it
        assert fake.stops["ZZZ"]["entry_price"] == 100.0 and fake.stops["ZZZ"]["stop_price"] == 92.0
    else:
        quote.assert_not_called()


def test_a_record_read_error_keeps_only_an_existing_record():
    fake = FakeOrders()
    fake.position_stop_written = MagicMock(side_effect=RuntimeError("db down"))
    now = datetime.now(timezone.utc)
    with ExitStack() as st:
        for p in fake.patches():
            st.enter_context(p)
        assert tasks._stop_record_is_this_orders("ZZZ", now, 100.0) is False       # none: write it
        fake.upsert_position_stop("ZZZ", 100.0, 92.0, "ATR", "Accelerator", 7)
        assert tasks._stop_record_is_this_orders("ZZZ", now, 100.0) is True        # keep it
        assert tasks._stop_record_is_this_orders("ZZZ", None, 100.0) is False


def test_a_read_error_with_no_record_still_writes_the_open_record():
    fake = FakeOrders()
    fake.position_stop_written = MagicMock(side_effect=RuntimeError("db down"))
    rid = fake.log_order(ticker="ZZZ", side="BUY", order_type="MARKET", qty=3, broker_order_id=803,
                         intent="OPEN", approval_id=1, status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, 100.0, 3.0, "filled")
    _age(fake)
    _recover(fake, _open_approval())
    assert fake.stops["ZZZ"]["entry_price"] == 100.0 and fake.stops["ZZZ"]["stop_method"] == "2.0x ATR"


def test_an_add_recovered_through_the_guard_keeps_the_record_and_only_covers():
    """An approval ADD whose execution rewrote the record at the blended
    entry (so it is not "this order's" by entry), then failed its trade row:
    the position was held before, so the record stays and only missing
    cover is placed."""
    fake = FakeOrders()
    fake.upsert_position_stop("ZZZ", 95.0, 88.0, "2.0x ATR", "Moonshot", 7)    # blended, ADD
    rid = fake.log_order(ticker="ZZZ", side="BUY", order_type="MARKET", qty=3, broker_order_id=804,
                         intent="ADD", approval_id=1, status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, 100.0, 3.0, "filled")
    record = _trailed(fake, "ZZZ")
    _age(fake)
    approval = dict(_open_approval(), action="ADD")
    with patch("tasks._held_before_fill", return_value=True) as held:
        with ExitStack() as st:
            for p in fake.patches():
                st.enter_context(p)
            st.enter_context(patch("db.get_approval_by_id", return_value=approval))
            reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=True))
            stop = st.enter_context(patch("tasks.manage_stop_place"))
            st.enter_context(patch("alerts._send"))
            tasks._recover_trade_pending_orders()
            tasks._recover_trade_pending_orders()
    assert [t["quantity"] for t in fake.trades] == [3.0]
    assert fake.stops["ZZZ"] == record
    held.assert_called_once_with("ZZZ", 3.0, "ADD", seen={})
    reprotect.assert_called_once_with("ZZZ", {}, None, from_queue=True)
    stop.assert_not_called()


def test_a_trim_recovered_via_stop_done_books_once_without_stop_work():
    fake = FakeOrders()
    rid = _pending_row(fake, qty=10.0, intent="TRIM")
    fake.rows[rid]["reconcile_note"] = db.TRADE_PENDING_STOP_DONE_NOTE
    _age(fake)
    reprotect, stop, send = _recover(fake, dict(_approval("TRIM")))
    assert [(t["side"], t["quantity"]) for t in fake.trades] == [("SELL", 10.0)]
    assert fake.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    reprotect.assert_not_called()
    stop.assert_not_called()
    assert "TRADE ROW RECOVERED" in " ".join(str(c.args[0]) for c in send.call_args_list)


def _claim_lands_but_errors(fake):
    real = fake.claim_and_log_trade

    def claim(*a, **k):
        real(*a, **k)
        return None                                   # committed, then reported an error
    fake.claim_and_log_trade = claim


@pytest.mark.parametrize("stop_done", [False, True])
def test_a_recovery_claim_that_landed_despite_an_error_finishes_the_booking(stop_done):
    fake = FakeOrders()
    rid = _pending_row(fake)
    if stop_done:
        fake.rows[rid]["reconcile_note"] = db.TRADE_PENDING_STOP_DONE_NOTE
    _claim_lands_but_errors(fake)
    _age(fake)
    reprotect, _, send = _recover(fake, dict(_approval("EXIT")))
    assert [t["quantity"] for t in fake.trades] == [42.0]
    assert fake.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    sent = " ".join(str(c.args[0]) for c in send.call_args_list)
    assert "TRADE ROW PENDING" not in sent
    if stop_done:
        reprotect.assert_not_called()
        assert "TRADE ROW RECOVERED" in sent
    else:
        reprotect.assert_called_once()


def test_a_recovery_claim_that_did_not_land_stays_pending():
    fake = FakeOrders()
    rid = _pending_row(fake)
    fake.claim_and_log_trade = lambda *a, **k: None
    _age(fake, 31)
    reprotect, _, send = _recover(fake, dict(_approval("EXIT")))
    assert fake.trades == [] and fake.rows[rid]["reconcile_note"] == db.TRADE_PENDING_NOTE
    reprotect.assert_not_called()


def test_trade_pending_sell_tickers_sql():
    conn, cur = _conn()
    cur.fetchall.return_value = [("NVDA",)]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        assert db.trade_pending_sell_tickers() == {"NVDA"}
    sql, params = cur.execute.call_args.args
    assert "side='SELL'" in sql and "status IN ('filled', 'partial')" in sql
    assert "reconcile_note IN (%s, %s)" in sql and "interval '1 day'" in sql
    assert params == (*db.TRADE_PENDING_NOTES, db.TRADING_ENV)


def test_a_mark_that_landed_despite_an_error_goes_through_the_claim():
    fake = FakeOrders()
    real = fake.mark_submitted_order_fill

    def lands_but_errors(*a, **k):
        real(*a, **k)
        return None
    fake.mark_submitted_order_fill = lands_but_errors
    place, _ = _broker({"fill_price": 101.5, "filled_qty": 42.0, "status": "Filled"})
    _run_exit(fake, place)
    assert len(fake.trades) == 1 and fake.rows[1]["reconcile_note"] is None   # claimed + cleared
    _age(fake)
    _reconcile(fake, None)
    assert len(fake.trades) == 1


def test_the_stuck_alert_fires_when_the_throttle_check_raises():
    fake = FakeOrders()
    rid = _pending_row(fake)
    fake.rows[rid]["filled_qty"] = 0.0
    _age(fake, 31)
    with patch("tasks._reconcile_alert_due", side_effect=RuntimeError("db down")):
        _reconcile(fake, None)
    assert "TRADE ROW PENDING" in fake.mocks["_send"].call_args.args[0]


def test_claim_and_log_trade_retries_a_transient_connect_error_without_spilling():
    import psycopg2
    conn, cur = _conn(rowcount=1)
    with patch("db._get_conn", side_effect=[psycopg2.OperationalError("could not connect"), conn]), \
         patch("db._is_transient_connect_error", return_value=True), patch("db._time.sleep"), \
         patch("db._put_conn"), patch("db._spill") as spill:
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r",
                                      expect_note=db.TRADE_PENDING_STOP_DONE_NOTE) is True
    assert cur.execute.call_args_list[0].args[1] == (None, 4, db.TRADE_PENDING_STOP_DONE_NOTE)
    with patch("db._get_conn", side_effect=RuntimeError("down")), patch("db._spill") as spill2, \
         patch("db._is_transient_connect_error", return_value=False):
        assert db.claim_and_log_trade(4, None, "NVDA", "SELL", 3, 10.0, "r") is None
    spill.assert_not_called(); spill2.assert_not_called()


def test_stale_trade_pending_sql_includes_the_stop_done_note():
    conn, cur = _conn()
    cur.fetchall.return_value = []
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.get_stale_trade_pending_orders(10)
    sql, params = cur.execute.call_args.args
    assert "reconcile_note IN (%s, %s)" in sql and ", reconcile_note, filled_at_source " in sql
    assert params == (db.TRADE_PENDING_NOTE, db.TRADE_PENDING_STOP_DONE_NOTE, db.TRADING_ENV, 10)


def test_telegram_shows_left_to_reconcile_as_approved():
    import inspect
    import telegram_bot
    src = inspect.getsource(telegram_bot)
    assert '"✅ APPROVED (filled; recorded by reconcile)" if status == "APPROVED"' in src


# --- recovery: approval read and stop-work failures (M214, M215) -------------

def _pending_open_buy(fake):
    rid = fake.log_order(ticker="ZZZ", side="BUY", order_type="MARKET", qty=3, broker_order_id=805,
                         intent="OPEN", approval_id=1, status="submitted", reconcile_note=AWAIT)
    fake.mark_submitted_order_fill(rid, 100.0, 3.0, "filled")
    _age(fake)
    return rid


def test_an_unread_approval_before_the_claim_leaves_the_row_for_the_next_run():
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    _, _, send = _recover(fake, None)                       # get_approval_by_id: None (DB error)
    assert fake.trades == [] and fake.rows[rid]["reconcile_note"] == db.TRADE_PENDING_NOTE
    assert "ZZZ" not in fake.stops
    send.assert_not_called()                                # under 30 min: the stuck alert waits
    reprotect, _, _ = _recover(fake, _open_approval())      # next run reads it
    assert [t["quantity"] for t in fake.trades] == [3.0]
    assert fake.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    assert fake.stops["ZZZ"]["stop_method"] == "2.0x ATR" and list(fake.outcomes) == [1]
    reprotect.assert_called_once_with("ZZZ", {}, None, from_queue=True)


def test_an_unread_approval_past_30_minutes_alerts_stuck():
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    _age(fake, 31)
    with patch("tasks._reconcile_alert_due", return_value=True):
        _, _, send = _recover(fake, None)
    assert fake.trades == [] and fake.rows[rid]["reconcile_note"] == db.TRADE_PENDING_NOTE
    assert "TRADE ROW PENDING" in send.call_args.args[0]


def _late_fill_patches(st, fake, approval):
    for p in fake.patches():
        st.enter_context(p)
    read = st.enter_context(patch("db.get_approval_by_id", return_value=approval))
    queue = st.enter_context(patch("tasks._queue_reprotect"))
    manual = st.enter_context(patch("tasks.open_manual_action"))
    send = st.enter_context(patch("alerts._send"))
    st.enter_context(patch("tasks._held_before_fill", return_value=False))
    return read, queue, manual, send


def test_an_unread_approval_with_no_stop_record_asks_for_a_manual_stop():
    """_book_late_fill reading None for an order with an approval (its own
    read, as reconcile's late-fill path does): no record or outcome can be
    written, and with no record the queued retry can place nothing, so the
    alert says so and a manual action is opened."""
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    order = dict(fake.rows[rid])
    with ExitStack() as st:
        _, queue, manual, send = _late_fill_patches(st, fake, None)
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell"))
        tasks._book_late_fill(order, 3.0, 100.0, 100.0, 0.0, trade_logged=True,
                              keep_record_since=order["created_at"])
    queue.assert_called_once_with("ZZZ", {"intent": "BUY"}, None)
    reprotect.assert_not_called()
    assert "ZZZ" not in fake.stops and fake.outcomes == {}
    manual.assert_called_once()
    assert manual.call_args.args[0] == "ZZZ" and manual.call_args.kwargs == {"clear_on_flat": False}
    failed = [str(c.args[0]) for c in send.call_args_list if "STOP WORK FAILED" in str(c.args[0])]
    assert len(failed) == 1 and "No stop record exists" in failed[0]
    assert "retry is queued" not in failed[0]


def test_an_unread_approval_with_a_stop_record_says_the_retry_is_queued():
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    fake.upsert_position_stop("ZZZ", 100.0, 92.0, "2.0x ATR", "Moonshot", 7)
    order = dict(fake.rows[rid])
    with ExitStack() as st:
        _, queue, manual, send = _late_fill_patches(st, fake, None)
        st.enter_context(patch("tasks._reprotect_after_late_sell"))
        tasks._book_late_fill(order, 3.0, 100.0, 100.0, 0.0, trade_logged=True,
                              keep_record_since=order["created_at"])
    queue.assert_called_once()
    manual.assert_not_called()
    failed = [str(c.args[0]) for c in send.call_args_list if "STOP WORK FAILED" in str(c.args[0])]
    assert len(failed) == 1 and "A stop retry is queued" in failed[0]


def test_the_late_fill_path_still_reads_the_approval_itself():
    """Without approval=, _book_late_fill reads it (reconcile's late fills)."""
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    order = dict(fake.rows[rid])
    with ExitStack() as st:
        read, queue, _, _ = _late_fill_patches(st, fake, _open_approval())
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=True))
        tasks._book_late_fill(order, 3.0, 100.0, 100.0, 0.0, trade_logged=True)
    read.assert_called_once_with(1)
    assert fake.stops["ZZZ"]["stop_method"] == "2.0x ATR" and list(fake.outcomes) == [1]
    reprotect.assert_called_once_with("ZZZ", {}, None, from_queue=True)
    queue.assert_not_called()


def test_an_outcome_error_does_not_skip_the_buy_stop():
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    order = dict(fake.rows[rid])
    with ExitStack() as st:
        _, queue, _, send = _late_fill_patches(st, fake, _open_approval())
        st.enter_context(patch("tasks._apply_buy_outcome", side_effect=RuntimeError("db down")))
        reprotect = st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=True))
        tasks._book_late_fill(order, 3.0, 100.0, 100.0, 0.0, trade_logged=True)
    assert fake.stops["ZZZ"]["entry_price"] == 100.0
    reprotect.assert_called_once_with("ZZZ", {}, None, from_queue=True)
    queue.assert_not_called()
    assert not any("STOP WORK FAILED" in str(c.args[0]) for c in send.call_args_list)


def _recover_with(fake, approval, *extra):
    with ExitStack() as st:
        _, queue, manual, send = _late_fill_patches(st, fake, approval)
        for p in extra:
            st.enter_context(p)
        tasks._recover_trade_pending_orders()
        tasks._recover_trade_pending_orders()            # never re-claimed or re-booked
    return queue, manual, send


def _failed_alerts(send):
    return [str(c.args[0]) for c in send.call_args_list if "STOP WORK FAILED" in str(c.args[0])]


def test_a_sell_stop_work_error_after_the_claim_alerts_now_and_queues_the_stop():
    fake = FakeOrders()
    rid = _pending_row(fake)
    _age(fake)
    queue, manual, send = _recover_with(
        fake, dict(_approval("EXIT")),
        patch("tasks._reprotect_after_late_sell", side_effect=RuntimeError("broker down")))
    assert [t["quantity"] for t in fake.trades] == [42.0]
    assert fake.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    queue.assert_called_once()
    assert queue.call_args.args[0] == "NVDA" and queue.call_args.args[2] == 101.5
    manual.assert_not_called()
    sent = [str(c.args[0]) for c in send.call_args_list]
    assert len(sent) == 1 and "STOP WORK FAILED" in sent[0] and "A stop retry is queued" in sent[0]


def test_a_buy_whose_queue_write_fails_after_the_claim_says_it_was_not_queued():
    """The BUY's stop placement fails and its own queue write raises: the
    stop work did not finish, so the except path reports it, worded from
    the second queue attempt failing too."""
    fake2 = FakeOrders()
    rid = _pending_open_buy(fake2)
    with ExitStack() as st:
        _late_fill_patches(st, fake2, _open_approval())
        st.enter_context(patch("tasks._reprotect_after_late_sell", return_value=False))
        raising = st.enter_context(patch("tasks._queue_reprotect",
                                         side_effect=RuntimeError("redis down")))
        send = st.enter_context(patch("alerts._send"))
        tasks._recover_trade_pending_orders()
        tasks._recover_trade_pending_orders()
    assert [t["quantity"] for t in fake2.trades] == [3.0]
    assert fake2.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    assert raising.call_count == 2                       # the branch's, then the report's
    failed = _failed_alerts(send)
    assert len(failed) == 1 and "could NOT be queued" in failed[0]


def test_a_buy_that_queued_its_stop_is_not_reported_again():
    """Placement failed and the branch queued the retry itself: no second
    queue and no STOP WORK FAILED alert."""
    fake = FakeOrders()
    _pending_open_buy(fake)
    queue, _, send = _recover_with(
        fake, _open_approval(), patch("tasks._reprotect_after_late_sell", return_value=False))
    queue.assert_called_once_with("ZZZ", {"intent": "BUY"}, None)
    assert _failed_alerts(send) == []


def test_an_error_after_the_stop_work_finished_does_not_queue_or_alert_it():
    fake = FakeOrders()
    rid = _pending_open_buy(fake)
    approval = dict(_open_approval(), status="FAILED")
    queue, manual, send = _recover_with(
        fake, approval,
        patch("tasks._reprotect_after_late_sell", return_value=True),
        patch("db.update_approval_status", side_effect=RuntimeError("db down")))
    assert [t["quantity"] for t in fake.trades] == [3.0]
    assert fake.rows[rid]["reconcile_note"] == db.TRADE_BOOKED_BY_RECONCILE_NOTE
    queue.assert_not_called()
    manual.assert_not_called()
    assert _failed_alerts(send) == []
