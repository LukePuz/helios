"""
Order execution — the one place that actually talks to the broker for a
BUY/SELL, shared by every trade-initiating path in the system:
  - agents/crew.py's auto-execute pipeline (analyst recommendation -> risk
    sizing -> straight to the broker, no human tap, per 2026-07-14 decision)
  - Telegram callback approvals (Approve button, when a human does review one)
  - !urgent chat commands (telegram_bot._execute_chat_order)
  - the dashboard's manual trade form

Before this module existed, each of those four had its own copy (or a near
copy) of stop-placement / phantom-fill / owned-quantity-clamp logic. The
dashboard's manual BUY in particular skipped stop placement entirely (the
position was invisible to the trailing-stop monitor) and its manual SELL
neither cancelled existing stops nor clamped to owned shares. One execution
path now backs all four call sites, so a fix here fixes it everywhere.
"""
import logging

from stop_lock import locked_by_ticker
import threading

log = logging.getLogger(__name__)


def _now_utc():
    """Timezone-aware now, for the orders.submitted_at record."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _capture_original_thesis_if_new(ticker: str, approval: dict) -> None:
    """Writes original_thesis once, at genuine first entry -- never
    overwritten by a later ADD to the same position (db.get_thesis
    returning something means it's already set). Fixes the gap found
    2026-09-13: a HOLD verdict's full analyst reasoning was previously
    lost after a scheduled review, only a 176-char activity_feed line
    survived; this is the entry-side half of that fix (the review-side
    half is agents/crew.py's review_held_position). Failures are logged,
    never raised -- losing a thesis snapshot must never block a real
    fill from completing."""
    try:
        import db
        from datetime import datetime, timezone
        if db.get_thesis(ticker):
            return
        analyst_json = approval.get("analyst_json") or {}
        thesis = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "setup_category": None,  # not yet classified live -- setup_gates.py runs in shadow mode
            "action": approval.get("action"),
            "conviction": approval.get("conviction"),
            "thesis_summary": analyst_json.get("thesis_summary"),
            "gate_criteria": None,
            "catalyst": analyst_json.get("catalyst"),
            "catalyst_status": "pending",
            "expected_return_pct": analyst_json.get("expected_return_pct"),
            "expected_downside_pct": analyst_json.get("expected_downside_pct"),
            "weeks_to_catalyst": analyst_json.get("weeks_to_catalyst"),
            "no_near_term_catalyst": db.compute_no_near_term_catalyst(
                analyst_json.get("catalyst"), analyst_json.get("weeks_to_catalyst")),
            **db.catalyst_horizon_any(analyst_json.get("catalyst")),
            "valuation_tier": None,
            "key_risks": analyst_json.get("key_risks", []),
            "reasoning": analyst_json.get("reasoning"),
        }
        db.set_original_thesis(ticker, thesis)
    except Exception as e:
        # WARNING, not debug -- same reasoning as agents/crew.py's
        # _capture_latest_thesis (2026-09-17): losing this snapshot is
        # real data loss for the drift-tracking system, not something
        # that should require a debug-level grep to discover after the fact.
        log.warning(f"[{ticker}] original thesis capture failed (non-fatal to the fill, but the snapshot is lost): {e}")


# ---------------------------------------------------------------------------
# Approval-driven execution (auto-execute pipeline + Telegram Approve button)
# ---------------------------------------------------------------------------

_POSITION_MISMATCH_ALERT = (
    "🚨 POSITION MISMATCH — {ticker}\n"
    "Sell filled; {expected} shares should remain but the broker shows none. "
    "No stop placed (it could open a short); the stop record is kept. Check TWS.")


def _unworked_remainder(ticker: str, remaining: float, local_working: float = 0.0,
                        floor: float | None = None) -> float:
    """Shares left after a part-filled sell that its still-working rest won't
    sell, so the stop placed now isn't over-sized once it fills (professor
    62i, as _reprotect_after_late_sell sizes it). An unreadable open-order
    read subtracts the rest this sell knows it left working (local_working:
    ordered minus filled): stopping all of them would leave the account short
    once both fill, the M129 case (professor M321). It also queues a RESTORE,
    which re-sizes from a fresh read. If the working rest is later cancelled
    (a DAY expiry), its shares are left unstopped until the stop coverage
    check sees them."""
    import tasks
    try:
        import ibkr
        selling = tasks._working_mkt_sell_qty(ibkr.get_open_orders(), ticker)
    except Exception as e:
        selling = max(float(local_working or 0.0), 0.0)
        log.warning(f"[{ticker}] open-order read failed after a part fill ({e}); "
                    f"stopping {max(remaining - selling, 0.0):g} of {remaining:g} "
                    f"({selling:g} still working by this sell), restore queued")
        try:
            tasks._queue_reprotect(ticker, {"intent": "RESTORE", "floor": floor}, None)
        except Exception as e2:
            log.error(f"[{ticker}] Could not queue the stop restore after the part fill: {e2}")
    return round(max(remaining - selling, 0.0), 4)


def _alert_no_stop_record(ticker: str, what: str, remaining: float, seed_px: float | None) -> None:
    """A sell left shares with no stop record: the stop was cancelled before
    the sell, so nothing protects them. Alert and open a manual action now
    (professor 62i); the highest stop read before the cancel (M228) is
    offered as the price. With that price known, a RESTORE is also queued
    carrying it as seed_stop (m606): the retry places it only against a
    fresh price above it, and no stop record is invented (the record's entry,
    bucket and conviction are unknown here)."""
    import tasks
    px = f" @ ${float(seed_px):.2f} (the stop that rested before the sell)" if seed_px else ""
    queued = False
    if seed_px:
        try:
            tasks._queue_reprotect(ticker, {"intent": "RESTORE", "seed_stop": float(seed_px)}, None)
            queued = True
        except Exception as e:
            log.error(f"[{ticker}] Could not queue the seeded stop restore: {e}")
    try:
        import alerts as alert_module
        alert_module._send(f"🚨 UNPROTECTED POSITION — {ticker}\n{what}: {remaining:g} shares left and "
                           f"no stop on record. Place one in TWS{px}."
                           + (" A stop at that price is queued and retried each reconcile cycle; "
                              "check it's in TWS." if queued else ""))
    except Exception:
        pass
    tasks.open_manual_action(ticker, f"place a stop{px} for {remaining:g} shares after {what.lower()}")


def _remaining_after_sell(ticker: str, owned: float, filled_qty: float) -> float | None:
    """Shares left after a sell, for re-placing the stop. Before this, any
    error reading the position counted as 0 shares left, which deleted the
    stop record while the broker stop was already cancelled: the remaining
    shares sat unprotected with no record (professor M22, 2026-10-28).
    Now the expected remainder (owned - filled) is the fallback, and a
    broker read can only lower it, so a lagging portfolio read can't size
    a stop above what is held. Returns None when the broker shows no
    position but shares were expected: a SELL stop for shares that aren't
    there could open a short, so the caller alerts instead of placing one."""
    import ibkr
    expected = max(round(owned - filled_qty, 4), 0.0)
    try:
        positions = ibkr.get_client().get_all_positions()
    except Exception as e:
        log.warning(f"[{ticker}] Position read after sell failed ({e}); "
                    f"using expected remainder {expected}")
        return expected
    held = next((p for p in positions if p.symbol == ticker), None)
    if held is None:
        if expected > 0:
            log.error(f"[{ticker}] Broker shows no position but {expected} shares "
                      f"expected after the sell; not placing a stop")
            return None
        return 0.0
    return min(round(abs(float(held.qty)), 4), expected)


def _finish_dead_sell_row(ticker: str, order_row_id: int, order_result: dict) -> None:
    """A SELL the broker reported dead with nothing filled: its row written
    before the fill wait becomes 'cancelled' (no late fill can come). If that
    write fails the row is adopted later as 'unconfirmed', and reconcile marks
    it cancelled from the broker's completed orders. Never raises: callers
    run it after the stop restore, and nothing here may skip what follows."""
    try:
        import db
        if db.finish_submitted_order(order_row_id, "cancelled", perm_id=order_result.get("perm_id")):
            return
    except Exception as e:
        log.error(f"[{ticker}] dead SELL's order row {order_row_id} update raised: {e}")
    log.error(f"[{ticker}] dead SELL's order row {order_row_id} not set cancelled; "
              f"reconcile adopts it")


def _early_order_row(**row):
    """An on_submitted callback for ibkr.place_market_order, and the state it
    fills. The callback writes the orders row (status 'submitted',
    AWAITING_FILL_NOTE) as soon as the broker has the order, before the fill
    wait, so a worker killed during the wait leaves a row that reconcile
    adopts (db.adopt_orphaned_submitted_orders). state["id"] is the row id,
    or None when the write failed: the caller then writes the row after the
    wait as before. Never raises."""
    import db
    state = {"id": None, "perm_id": None, "attempted": False}

    def on_submitted(order_id, perm_id):
        state["perm_id"] = perm_id
        state["attempted"] = True
        try:
            oid = db.log_order(**row, broker_order_id=order_id, perm_id=perm_id,
                               status="submitted", reconcile_note=db.AWAITING_FILL_NOTE)
        except Exception as e:
            log.error(f"[{row.get('ticker')}] order row before the fill wait not written: {e}")
            oid = -1
        if (oid or -1) > 0:
            state["id"] = oid
        else:
            log.error(f"[{row.get('ticker')}] order row before the fill wait not written; "
                      f"writing it after the wait instead")
    return on_submitted, state


def _recover_early_row(ticker: str, side: str, early: dict, order_result: dict) -> None:
    """The early write reported a failure: its INSERT may still have
    committed (an error after the commit). Look for that row before anything
    falls back to inserting a second one for the same broker order, and use
    it as the early row if found. If the lookup fails, the fallback insert
    goes ahead as before, with an alert that the order may have two rows.
    Never raises."""
    if early.get("id") or not early.get("attempted") or order_result.get("order_id") is None:
        return
    import db
    try:
        found = db.find_awaiting_order_row(ticker, side, order_result["order_id"])
    except Exception as e:
        log.error(f"[{ticker}] could not check for an order row written before the wait: {e}")
        try:
            import alerts as alert_module
            alert_module._send(
                f"⚠️ ORDER ROW CHECK FAILED — <b>{ticker}</b>\n"
                f"The order row for {side} order {order_result.get('order_id')} reported a failed "
                f"write before the fill wait, and the check for it failed too, so a new row is "
                f"written. If the first write did land, the order has two rows: check the orders "
                f"table for broker order {order_result.get('order_id')}.")
        except Exception as e2:
            log.error(f"[{ticker}] order-row-check alert failed: {e2}")
        return
    if found:
        log.warning(f"[{ticker}] the order row before the wait was written after all (id {found}); "
                    f"using it instead of a second row")
        early["id"] = found


def _mark_early_row_filled(ticker: str, early: dict, order_result: dict,
                           filled_qty: float, qty: float) -> bool:
    """Record a fill seen within the wait on the row written before it:
    'partial' when part filled (reconcile books the rest, professor M53),
    else 'filled'; a fill without a price is recorded with none. Runs BEFORE
    the trade row is written: once the row is marked it can't be adopted
    from its fill wait and booked a second time. A failed write gets the
    fill_write_failed note in place of AWAITING_FILL_NOTE, and the reconcile
    adoption checks the trades table too. Also fills in the perm id TWS
    assigned during the wait.

    The mark applies only while the row is still 'submitted'. If it updates
    no row, reconcile has already adopted the order from its fill wait (this
    worker stalled past the adoption delay) and may have booked the fill:
    returns False, after a critical alert, and the caller must not write the
    trade row or touch stops. True otherwise, including when the write fails
    (the caller goes on as before). Never raises."""
    import db
    order_row_id = early["id"]
    try:
        n = db.mark_submitted_order_fill(order_row_id, order_result.get("fill_price"), filled_qty,
                                         "partial" if filled_qty < float(qty) - 1e-6 else "filled",
                                         filled_at=order_result.get("fill_time"),
                                         filled_at_source=order_result.get("fill_time_source"))
    except Exception as e:
        log.error(f"[{ticker}] fill write for order row {order_row_id} raised: {e}")
        n = None
    if n == 0:
        log.critical(f"[{ticker}] order row {order_row_id} was already adopted by reconcile; "
                     f"this fill is left to reconcile (no trade row, no stop work here)")
        try:
            import alerts as alert_module
            alert_module._send(
                f"🚨 FILL LEFT TO RECONCILE — <b>{ticker}</b>\n"
                f"Order {order_result.get('order_id')} filled {filled_qty:g}, but its order row "
                f"was already taken over by reconcile (this worker stalled after the fill wait), "
                f"so reconcile books the fill and handles the stop. Nothing was recorded here. "
                f"Check the trade rows and the stop against TWS.")
        except Exception as e:
            log.error(f"[{ticker}] fill-left-to-reconcile alert failed: {e}")
        return False
    if n:
        early["trade_pending"] = True   # the mark set TRADE_PENDING_NOTE
    if n is None:
        # The write may have committed although it reported an error: then
        # the row is trade_pending and the trade row must go through the
        # claim, or reconcile could book it again.
        try:
            if db.order_reconcile_note(order_row_id) == db.TRADE_PENDING_NOTE:
                log.warning(f"[{ticker}] fill write for order row {order_row_id} reported an "
                            f"error but landed")
                early["trade_pending"] = True
                n = 1
        except Exception as e:
            log.error(f"[{ticker}] could not re-read order row {order_row_id}: {e}")
    if n is None:
        log.warning(f"[{ticker}] fill write failed for order row {order_row_id}")
        try:
            db.set_order_reconcile_note(order_row_id, "fill_write_failed")
        except Exception as e:
            log.error(f"[{ticker}] fill_write_failed note for order row {order_row_id} raised: {e}")
    perm = order_result.get("perm_id")
    if perm and not early.get("perm_id"):
        try:
            db.set_order_perm_id(order_row_id, perm)
        except Exception as e:
            log.warning(f"perm id backfill failed for order row {order_row_id}: {e}")
    return True


# The _outcome reason prefix for a fill whose recording was left to reconcile:
# the order filled, so callers record the approval APPROVED, not FAILED.
_LEFT_TO_RECONCILE = "skipped: fill left to reconcile"


def approval_outcome(success: bool) -> tuple[str, str | None]:
    """(status, reason) for an approval after execute_approved: APPROVED on
    success, and also when the fill was left to reconcile (it filled; only its
    recording moved), else FAILED with the outcome reason."""
    if success:
        return "APPROVED", None
    reason = last_outcome_reason()
    if reason and reason.startswith(_LEFT_TO_RECONCILE):
        return "APPROVED", reason
    return "FAILED", reason


def _write_trade_row(ticker: str, early: dict, order_result: dict, filled_qty: float,
                     **trade) -> bool:
    """Write the execution path's trade row. For a row marked trade_pending,
    the note is cleared and the trade row inserted in one transaction
    (db.claim_and_log_trade), so it and reconcile's recovery can't both book
    it. Returns False when reconcile has already claimed the row (this worker
    stalled past the recovery delay): the caller then writes nothing more and
    does no stop work, and a critical alert goes out. True otherwise. If the
    transaction fails, nothing is written or spilled, the note stays and
    reconcile books the trade after its delay; the caller still does the stop
    work now (reconcile's later stop work sizes to what resting stops leave,
    and the outcome row is written once per approval). Never raises."""
    import db
    if not early.get("trade_pending"):
        # The row written before the fill wait, when there is one; without
        # it the orders row is written after the trade row and this is NULL.
        db.log_trade(**trade, order_row_id=early.get("id") or None)
        return True
    try:
        r = db.claim_and_log_trade(early["id"], None, **trade, order_row_id=early["id"])
    except Exception as e:
        log.error(f"[{ticker}] trade row transaction for order row {early['id']} raised: {e}")
        r = None
    if r is True:
        return True
    if r is None:
        log.error(f"[{ticker}] trade row for order row {early['id']} not written (rolled back); "
                  f"it stays trade_pending and reconcile books it")
        early["trade_row_failed"] = True   # _stop_work_done marks the row when the rest is done
        return True
    log.critical(f"[{ticker}] order row {early['id']} was already recovered by reconcile; "
                 f"no trade row or stop work here")
    try:
        import alerts as alert_module
        alert_module._send(
            f"🚨 FILL LEFT TO RECONCILE — <b>{ticker}</b>\n"
            f"Order {order_result.get('order_id')} filled {filled_qty:g}; this worker stalled "
            f"before recording the trade and reconcile has already done it (with the stop "
            f"work). Nothing was recorded here. Check the trade rows and the stop against TWS.")
    except Exception as e:
        log.error(f"[{ticker}] fill-left-to-reconcile alert failed: {e}")
    return False


def _stop_work_done(ticker: str, early: dict) -> None:
    """The trade-row transaction failed but the stop and outcome work after it
    is done: mark the row trade_pending_stop_done (only while it is still
    trade_pending), so reconcile books only the trade row and doesn't redo
    the stop record, stop or outcome. If this write fails the row stays
    trade_pending, and reconcile's recovery keeps a stop record written after
    the order. Never raises."""
    if not early.get("trade_row_failed"):
        return
    import db
    try:
        n = db.claim_trade_pending(early["id"], db.TRADE_PENDING_STOP_DONE_NOTE)
    except Exception as e:
        log.error(f"[{ticker}] stop-done note for order row {early['id']} raised: {e}")
        n = None
    if n != 1:
        log.error(f"[{ticker}] order row {early['id']} not marked stop-done (result {n}); "
                  f"reconcile keeps the stop record written for it")


def _log_unconfirmed_order(order_row_id: int | None = None, **row) -> bool:
    """Write the 'unconfirmed' orders row that reconcile_unconfirmed_orders
    uses to find a late fill. With order_row_id (the row written before the
    fill wait) that row is set 'unconfirmed' instead and nothing is
    inserted; if that update fails the row keeps AWAITING_FILL_NOTE and
    reconcile adopts it as 'unconfirmed' (a second row would be booked
    twice). db.log_order returns -1 on failure without
    raising, and that used to be ignored: a late fill then left a live
    position with no stop and no trade row, invisible to reconcile (audit
    2026-11-02 #7). On a failed write the row goes into the untracked-order
    store (Redis and system_state), which reconcile re-writes from every
    cycle (that is the retry), and one ORDER UNTRACKED alert goes out.
    Ownership: db.log_order spills every failed write itself, so this only
    spills when log_order raised instead of returning -1. This alert is the
    one alert for the event: db's generic hourly "DB WRITE FAILED" for
    log_order is held off for this call. Returns True if the row was written."""
    import time
    import db
    import tasks
    import alerts as alert_module
    ticker = row.get("ticker")
    if order_row_id:
        try:
            if db.finish_submitted_order(order_row_id, "unconfirmed", perm_id=row.get("perm_id")):
                return True
        except Exception as e:
            log.error(f"[{ticker}] order row {order_row_id} unconfirmed update raised: {e}")
        log.error(f"[{ticker}] order row {order_row_id} not set unconfirmed; it stays 'submitted' "
                  f"({db.AWAITING_FILL_NOTE}) and reconcile adopts it")
        return False
    throttle = getattr(db, "_LAST_WRITE_ALERT", None)
    prev = throttle.get("log_order") if isinstance(throttle, dict) else None
    if isinstance(throttle, dict):
        throttle["log_order"] = time.time()   # this call's alert is ORDER UNTRACKED below
    try:
        if (db.log_order(**row) or -1) > 0:
            if isinstance(throttle, dict):
                if prev is None:
                    throttle.pop("log_order", None)
                else:
                    throttle["log_order"] = prev
            return True
        log.error(f"[{ticker}] unconfirmed order row not written (log_order returned -1; spilled by db)")
    except Exception as e:
        log.error(f"[{ticker}] unconfirmed order row not written: {e}")
        try:
            db._spill("log_order", row, str(e))
        except Exception as e2:
            log.error(f"[{ticker}] unconfirmed order spill failed: {e2}")
    try:
        saved = tasks._save_untracked_order(row)
    except Exception as e:
        log.error(f"[{ticker}] could not save the untracked order for reconcile: {e}")
        saved = False
    try:
        alert_module._send(
            f"🚨 ORDER UNTRACKED — {ticker}\n"
            f"{row.get('side')} {row.get('qty')} (order id {row.get('broker_order_id')}, "
            f"perm id {row.get('perm_id')}) has no fill confirmed and its order row could not "
            f"be written. "
            + ("Reconcile retries the row every cycle; " if saved else
               "It could NOT be queued for reconcile either; ")
            + "check TWS for a late fill and protect any shares by hand."
        )
    except Exception:
        pass
    try:
        tasks.open_manual_action(ticker, f"check order {row.get('broker_order_id')} for a late "
                                         f"fill (order row not written)")
    except Exception as e:
        log.error(f"[{ticker}] could not open the manual action for the untracked order: {e}")
    return False


def _schedule_held_sell_retry(approval: dict, trigger: str) -> None:
    """One retry ~5 min after a TRIM/EXIT held on unreadable stops (m127);
    never from the retry itself."""
    if trigger == "held_sell_retry" or approval.get("id") is None:
        return
    try:
        from tasks import retry_held_sell, HELD_SELL_RETRY_DELAY_S
        retry_held_sell.apply_async(args=[approval["id"]], countdown=HELD_SELL_RETRY_DELAY_S, queue="ops")
    except Exception as e:
        log.error(f"[{approval.get('ticker')}] could not schedule the held-sell retry: {e}")


def _stops_still_resting(ticker: str, prices_out: list | None = None) -> float | None:
    """Total quantity of SELL stops still working at the broker for ticker,
    or None if it can't be read. A sell must not go out on top of a resting
    stop: after an EXIT that stop would short the account (limitations audit
    22:53, A2253-1). A failed read is tried once more (a timed-out read drops
    the session, so the second one reconnects).

    prices_out (optional list): filled with the stop prices of a successful
    read, so a sell path's pre-cancel check also gives _floor_before_cancel
    its floor without a second broker read."""
    import tasks
    for attempt in (1, 2):
        try:
            stops = tasks._resting_sell_stops(ticker)
            total = sum(float(o["qty"]) for o in stops)
        except Exception as e:
            log.warning(f"[{ticker}] could not re-read resting stops (try {attempt}/2): {e}")
            continue
        if prices_out is not None:
            prices_out[:] = [o.get("stop_price") for o in stops]
        return total
    return None


# A stop being cancelled. tasks._NOT_WORKING counts it as working (the
# cancel can still be rejected), which is right for "is it safe to sell /
# place on top?", but not for a restore: cancel_stop_orders_for returns with
# a cancel still pending after its poll, and counting that stop as cover
# would leave the position bare once the cancel lands.
_CANCELLING_STATUSES = frozenset({"PendingCancel", "ApiPendingCancel"})


def _restore_cover(ticker: str, cancelling_out: list | None = None) -> float | None:
    """Shares a restore can count as already covered: working SELL stops
    other than ones being cancelled, or None if the read fails (tried
    twice). A stop still being cancelled gets one second look after ~5 s
    (same as reconcile and _fix_overstop, tasks.py): a slow cancel usually
    lands by then. If one is still cancelling and the broker then rejects
    the cancel, the restore leaves the position over-stopped; the caller
    queues a coverage re-check for that (_queue_coverage_recheck).
    Unprotected is worse than over-stopped, so a pending cancel never skips
    the restore.

    cancelling_out (optional list): filled with the shares still being
    cancelled after the second look ([0.0] when none)."""
    import tasks
    looked_again = False
    for attempt in (1, 2):
        try:
            stops = tasks._resting_sell_stops(ticker)
            if not looked_again and any(o.get("status") in _CANCELLING_STATUSES for o in stops):
                looked_again = True
                import time
                time.sleep(5)   # a slow cancel; one more look before sizing the restore
                stops = tasks._resting_sell_stops(ticker)
            cancelling = sum(float(o["qty"]) for o in stops
                             if o.get("status") in _CANCELLING_STATUSES)
            covered = sum(float(o["qty"]) for o in stops
                          if o.get("status") not in _CANCELLING_STATUSES)
        except Exception as e:
            log.warning(f"[{ticker}] could not read resting stops for the restore "
                        f"(try {attempt}/2): {e}")
            continue
        if cancelling:
            log.warning(f"[{ticker}] {cancelling:g} shares of stops are still being cancelled; "
                        f"not counted as cover for the restore")
        if cancelling_out is not None:
            cancelling_out[:] = [cancelling]
        return covered
    return None


def _queue_coverage_recheck(ticker: str, cancelling: list) -> None:
    """A restore went on while another stop was still being cancelled
    (_restore_cover's cancelling_out). If the broker rejects that cancel the
    stop goes back to working and both rest; a fast drop could fill both and
    open a short. Queue tasks.recheck_stop_coverage (_fix_overstop) for ~60 s
    out so that is trimmed then, not at the next reconcile (up to 15 min).
    Never raises: reconcile still catches it if the queue is down."""
    if not cancelling or not cancelling[0]:
        return
    try:
        import tasks
        if tasks._queue_stop_coverage_recheck(ticker):
            log.warning(f"[{ticker}] restore placed while {cancelling[0]:g} shares of stops were "
                        f"still being cancelled; stop coverage re-check queued in 60 s")
    except Exception as e:
        log.error(f"[{ticker}] could not queue the stop coverage re-check ({e}); "
                  f"reconcile checks the coverage")


def _alert_stop_unknown(ticker: str, qty: float, stop_price: float) -> None:
    """A stop placement got no reply: it may or may not be resting. Say so
    instead of retrying (professor C24); reconcile_stop_orders confirms over
    two cycles and fixes the coverage either way."""
    import alerts as alert_module
    log.error(f"[{ticker}] Stop placement result unknown ({qty:g} @ ${stop_price:.2f}); "
              f"not retrying, reconcile will fix coverage")
    try:
        alert_module._send(
            f"⚠️ STOP STATUS UNKNOWN — {ticker}\n"
            f"A stop for {qty:g} shares @ ${stop_price:.2f} got no reply from the broker. "
            f"Not retried (it may be resting); the stop check fixes coverage within ~30 min. "
            f"Check TWS if this repeats."
        )
    except Exception:
        pass


def _alert_stop_refused(ticker: str, qty: float, err: Exception) -> None:
    """The broker layer refused the stop: it shows fewer shares long than the
    stop is for, so it would open or add to a short (professor run 11)."""
    import alerts as alert_module
    log.error(f"[{ticker}] Stop for {qty:g} refused: {err}")
    try:
        alert_module._send(
            f"⚠️ STOP REFUSED — {ticker}\n{err}. Usually the broker's position hasn't caught "
            f"up with the fill yet; the stop is queued and retried each reconcile cycle. If "
            f"this repeats, check the position and stops in TWS."
        )
    except Exception:
        pass


def _queue_stop_for_retry(ticker: str, qty: float, stop_price: float, alert: bool = True) -> None:
    """A stop request never reached the orders worker (busy or stuck):
    nothing was sent. Hand it to the reconcile re-protect queue, which
    places the recorded stop for the held shares every cycle until done,
    instead of blocking this worker on retries (audit 04:53, A0453-3)."""
    import alerts as alert_module
    import tasks
    log.error(f"[{ticker}] Stop request not picked up by the orders worker; queued for the "
              f"reconcile retry ({qty:g} @ ${stop_price:.2f})")
    try:
        tasks._queue_reprotect(ticker, {"intent": "BUY"}, None)
    except Exception as e:
        log.error(f"[{ticker}] Could not queue the stop retry: {e}")
    if not alert:
        return  # the caller already sent its own alert (professor M81)
    try:
        alert_module._send(
            f"⚠️ STOP DELAYED — {ticker}\n"
            f"The orders worker didn't pick up the stop for {qty:g} shares @ ${stop_price:.2f} "
            f"(nothing was sent). It is queued and retried every reconcile cycle; check the "
            f"orders worker if this repeats."
        )
    except Exception:
        pass


def _write_stop_record(ticker: str, **fields) -> bool:
    """db.upsert_position_stop for the paths that write the record and then
    place the stop. A DB failure here (pool exhausted, DB down) used to raise
    before the stop was placed and leave the fill unprotected. Now it never
    raises: False means the record is missing, the caller places the stop
    anyway and then calls _alert_stop_record_missing. The placement is never
    delayed by a retry here."""
    import db
    try:
        ok = db.upsert_position_stop(ticker=ticker, **fields)
    except Exception as e:
        log.error(f"[{ticker}] stop record write failed ({type(e).__name__}: {e}); "
                  f"placing the stop anyway")
        return False
    if ok is False:   # upsert_position_stop logged and swallowed the error
        log.error(f"[{ticker}] stop record write failed; placing the stop anyway")
        return False
    return True


def _set_stop_order_id(ticker: str, order_id) -> None:
    """db.set_stop_broker_order_id right after a stop was placed. It runs
    inside the placement retry loops, where a DB error raised here read as a
    failed placement and put a second stop on. Never raises."""
    import db
    try:
        db.set_stop_broker_order_id(ticker, order_id)
    except Exception as e:
        log.error(f"[{ticker}] stop {order_id} is placed but its order id could not be "
                  f"recorded: {type(e).__name__}: {e}")


def _record_placed_stop(ticker: str, what: str, order_id, writes) -> bool:
    """The DB writes that follow a stop the broker has ACCEPTED. Once the
    stop is placed nothing may place another for the ticker, so a write
    failure (pool exhausted: db._get_conn raises outside the writers' own
    try) must not reach a caller's fallback or restore path, which would
    stack a second full-size stop (professor M227).

    writes: [(label, zero-arg callable)], each tried on its own. Never
    raises. A failure is logged and alerted once, and the record is left as
    it is for the next stop check / reconcile to work from. True when all
    writes went through."""
    failed = []
    for label, fn in writes:
        try:
            fn()
        except Exception as e:
            failed.append(f"{label}: {type(e).__name__}: {e}")
    if not failed:
        return True
    detail = "; ".join(failed)
    log.error(f"[{ticker}] {what}: stop {order_id} is placed at the broker but the record "
              f"write failed ({detail}); nothing else placed, record left for reconcile")
    try:
        import alerts as alert_module
        alert_module._send(
            f"⚠️ STOP RECORD NOT UPDATED — {ticker}\n"
            f"{what}: the new stop (order {order_id}) is resting at the broker, but its DB "
            f"record could not be updated ({detail}). No other stop was placed. The record "
            f"may still show the old stop until the next stop check or reconcile; check TWS "
            f"if this repeats."
        )
    except Exception:
        pass
    return False


def _restore_sized_stop(ticker: str, held: float, stop_price: float, what: str,
                        market_price: float | None = None) -> tuple[str, dict | None]:
    """Put a stop back after a failed replacement, for the shares no working
    SELL stop covers. The failed call may still have placed its stop (an
    error after ib.placeOrder that isn't OrdersCallUnknown, a lost worker),
    so a full-size restore without this read could stack two (professor
    M227 follow-up). Same rule as _restore_stop_after_failed_sell.

    Returns ("placed", result), ("covered", None) when nothing is uncovered,
    ("unread", None) when coverage could not be read: then nothing is
    placed, the restore is queued for the reconcile retry and an alert sent,
    or ("sign_flip", None) when the cache reads long and a fresh read short:
    no SELL stop is placed (it would add to a short), the sign-flip alert is
    sent (throttled with reconcile's) and the restore is queued, its retry
    checking the sign again. A placement error raises to the caller."""
    import alerts as alert_module
    import tasks
    # Size to what is held now: the caller's quantity was read before its
    # cancel and failed placement, and a fill since then changes it. A failed
    # or flat read keeps the caller's (the portfolio cache can lag a fill,
    # and the broker layer refuses a stop larger than the position).
    try:
        import ibkr
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        if pos is not None and float(pos.qty) > 0:
            fresh = round(float(pos.qty), 4)
            try:   # the cache can lag a sale: never above a fresh read that shows the ticker
                direct = float(ibkr.get_fresh_stock_positions().get(ticker, 0.0))
                if direct > 0:
                    fresh = round(min(fresh, direct), 4)
                elif direct < 0:
                    # Same rule as reconcile's _smaller_read: reads that
                    # disagree on the sign get no SELL stop. The stop was
                    # already cancelled, so alert and queue the restore (the
                    # queued retry re-checks the sign before placing).
                    log.error(f"[{ticker}] {what}: position reads disagree on sign (cache "
                              f"{fresh:g}, fresh {direct:g}); no stop restored")
                    tasks._sign_flip_alert(ticker, fresh, direct)
                    try:
                        tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
                    except Exception as e:
                        log.error(f"[{ticker}] Could not queue the stop restore: {e}")
                    return "sign_flip", None
                else:
                    # Left at the cache: under-cover is worse than a stop the
                    # broker layer refuses for being over the position.
                    log.warning(f"[{ticker}] {what}: the fresh position read shows no {ticker} "
                                f"(cache {fresh:g}); restore sized from the cache")
            except Exception as e:
                log.warning(f"[{ticker}] {what}: fresh position read failed ({e}); using the cache")
            if abs(fresh - float(held)) > 1e-6:
                log.warning(f"[{ticker}] {what}: {fresh:g} shares held now (was {float(held):g}); "
                            f"restore sized to {fresh:g}")
            held = fresh
    except Exception as e:
        log.warning(f"[{ticker}] {what}: position re-read failed ({e}); restore sized to "
                    f"{float(held):g}")
    cancelling = []
    covered = _restore_cover(ticker, cancelling)
    if covered is None:
        log.error(f"[{ticker}] {what}: stop restore skipped, resting stops could not be read")
        queued = False
        try:
            tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
            queued = True
        except Exception as e:
            log.error(f"[{ticker}] Could not queue the stop restore: {e}")
        try:
            alert_module._send(
                f"⚠️ CHECK STOPS — {ticker}\n"
                f"{what} failed and the resting stops could not be read, so no stop was "
                f"restored" + (" yet; it is queued and retried every reconcile cycle" if queued
                               else "") + f". Check TWS; restore level ${float(stop_price):.2f}."
            )
        except Exception:
            pass
        return "unread", None
    qty = round(float(held) - covered, 4)
    if qty <= 0:
        log.warning(f"[{ticker}] {what}: {covered:g} shares already covered by working stops; "
                    f"no restore placed")
        return "covered", None
    result = tasks.manage_stop_place(ticker, qty, float(stop_price), market_price=market_price)
    log.info(f"[{ticker}] {what}: stop restored @ ${float(stop_price):.2f} for {qty:g} uncovered shares")
    _queue_coverage_recheck(ticker, cancelling)
    return "placed", result


def _raise_record_writes(ticker: str, placed_px: float, record_px: float, stop_rec: dict) -> list:
    """The record write for a stop placed above its record (raised to a
    resting TRAIL's trigger): [] when it isn't higher."""
    import db
    if float(placed_px) <= float(record_px) + 1e-9:
        return []
    return [("record stop", lambda: db.update_trailing_stop(
        ticker, float(placed_px), (stop_rec or {}).get("trailing_high"), native_trail=False,
        strict=True))]   # strict: _record_placed_stop reports a failed write


def _floor_before_cancel(ticker: str, prices: list, what: str) -> float | None:
    """Highest working SELL stop from the read the sell path made just before
    its cancel (_stops_still_resting's prices_out), so the stop put back
    afterwards (restore or re-place) is never below a native TRAIL's trigger
    (professor M228). No extra broker read. None when there is no stop, no
    prices were read, or a price was unreadable (logged and alerted,
    throttled): the caller then uses the record, as before."""
    if not prices:
        return None
    if any(p is None for p in prices):
        log.warning(f"[{ticker}] {what}: a resting stop has no readable price; the stop put "
                    f"back after the sell uses the record, without the no-loosen floor")
        try:
            import tasks
            if tasks._reconcile_alert_due(f"stop_no_loosen_floor_price_{ticker}"):
                import alerts as alert_module
                alert_module._send(
                    f"⚠️ STOP FLOOR UNREAD — <b>{ticker}</b>\n{what}: a resting stop has no "
                    f"readable price, so a stop put back after the sell may sit below a resting "
                    f"TRAIL it replaced. Check the stop in TWS.")
        except Exception as e:
            log.warning(f"[{ticker}] no-loosen alert failed: {e}")
        return None
    return max(float(p) for p in prices)


_MANUAL_ACTION_CLEARS = ("This reminder clears once resting stops cover the position, not "
                         "when the record is back, so the record has to be fixed by hand.")


def _stop_outcome_text(outcome: str) -> tuple[str, str]:
    """(what happened to the stop, what to do) for the record alerts.
    outcome: "placed", "unknown" (no reply, may be resting), or
    "queued"/"failed" (not placed)."""
    if outcome == "placed":
        return "was placed at the broker", ""
    if outcome == "unknown":
        return ("got no reply from the broker (it may or may not be resting)",
                " If it isn't resting in TWS, place it in TWS now.")
    return ("was NOT placed",
            " It was handed to the retry queue, but the queue can't place a stop "
            "without the record: place it in TWS now.")


def _alert_stop_record_missing(ticker: str, qty: float, stop_price: float, outcome: str,
                               stale_stop: float | None = None) -> None:
    """One alert, and a manual action, when the stop went to the broker (or
    was tried) without its position_stops row.

    The stop check leaves a stop alone while its position is held (it
    cancels only stops with no position), but nothing trails it, and neither
    the reprotect queue nor the stop check places a stop without a record.
    The record comes back through the startup adoption (main._adopt_position,
    record-only), which skips a ticker that still has a record: after an ADD
    (stale_stop = the old record's stop) the old record stays as it is.

    The manual action clears once resting stops cover the position or it is
    flat (tasks._review_manual_actions); there is no "record exists" check."""
    import alerts as alert_module
    import tasks
    how, todo = _stop_outcome_text(outcome)
    if stale_stop is not None:
        repair = (f" A stale record remains at the old stop ${stale_stop:.2f} (the position "
                  f"before this add), and the next Helios start will NOT fix it: adoption "
                  f"skips a ticker that has a record. Fix the record by hand.")
    else:
        repair = " The next Helios start adopts the record from the resting stop."
    log.error(f"[{ticker}] stop @ ${stop_price:.2f} for {qty:g} shares {how}, but its DB "
              f"record is missing")
    try:
        alert_module._send(
            f"⚠️ STOP RECORD MISSING — {ticker}\n"
            f"The protective stop for {qty:g} shares @ ${stop_price:.2f} {how}, but its DB "
            f"record (position_stops) could not be written.{todo} The stop check won't cancel "
            f"a stop while the position is held, but nothing trails or re-places it until the "
            f"record is back.{repair} {_MANUAL_ACTION_CLEARS}"
        )
    except Exception:
        pass
    try:
        tasks.open_manual_action(ticker, f"the stop record @ ${stop_price:.2f} is missing; "
                                         f"check the stop in TWS and restore the record by hand",
                                 clear_on_flat=True)
    except Exception as e:
        log.error(f"[{ticker}] could not open the manual action: {e}")


def _alert_prior_stop_unread(ticker: str, qty: float, stop_price: float, outcome: str,
                             prior) -> None:
    """A BUY that may have added to a position whose prior stop record could
    not be read. The record write was skipped (an overwrite would hand the
    trail and ladder code the new shares' level for the whole position), the
    old stop was left resting, and only the new shares got a stop.

    prior is a fresh read made after placement: the kept record (dict) or
    "unknown" (that read failed too; then the alert covers both "kept" and
    "missing"). With no record the caller writes one instead of calling
    this. Opens a manual action."""
    import alerts as alert_module
    import tasks
    how, todo = _stop_outcome_text(outcome)
    lines = [f"The prior stop record could not be read when this BUY filled, so the old stop "
             f"was left in place and only the {qty:g} new shares got their own stop "
             f"@ ${stop_price:.2f}, which {how}.{todo}"]
    lower = (" A later trail or ratchet works from that record and cancels every stop "
             "for one whole-position stop, so it could LOWER the new shares' stop")
    if isinstance(prior, dict):
        old = prior.get("stop_price")
        old_txt = f"${float(old):.2f}" if old is not None else "its old level"
        line = f"The existing record was KEPT, not overwritten: it still holds the old stop {old_txt}."
        if old is None or float(stop_price) > float(old):
            line += lower + f" below ${stop_price:.2f}."
        lines.append(line)
    else:
        lines.append(f"Whether a record exists is unknown (the read failed again). If one "
                     f"exists, it was KEPT, not overwritten; if the new stop is above its old "
                     f"stop, a later trail or ratchet from it could LOWER the new shares' stop "
                     f"below ${stop_price:.2f}. If none exists, STOP RECORD MISSING: nothing "
                     f"trails or re-places these stops.")
    lines.append(f"Check every stop on this position in TWS and fix the record by hand. "
                 f"{_MANUAL_ACTION_CLEARS}")
    log.error(f"[{ticker}] prior stop record unreadable on a possible add; record write "
              f"skipped, {qty:g} new shares stopped @ ${stop_price:.2f} ({outcome})")
    try:
        alert_module._send(f"⚠️ CHECK STOPS — {ticker}\n" + " ".join(lines))
    except Exception:
        pass
    try:
        tasks.open_manual_action(ticker, f"check the stops and fix the stop record by hand "
                                         f"after an add whose prior record could not be read",
                                 clear_on_flat=True)
    except Exception as e:
        log.error(f"[{ticker}] could not open the manual action: {e}")


# Order states after which nothing can fill (ib_async OrderStatus).
_DEAD_ORDER_STATUSES = ("Cancelled", "ApiCancelled", "Inactive")

# How each execute_approved trigger reads in trades.reason (professor M134).
_TRIGGER_LABELS = {
    "auto": "auto-executed",
    "telegram": "approved via Telegram",
    "auto_retry": "executed on a deferred retry",
    "held_sell_retry": "executed on a held-sell retry",
}


def _restore_stop_after_failed_sell(ticker: str, floor: float | None = None) -> None:
    """A TRIM/EXIT cancels the broker stop before its market SELL. If the
    order then fails to go in, put the stop back for what's held (limitations
    audit 2026-10-27, finding 8). Before this the position was left with no
    broker stop until someone noticed.

    floor: the highest working SELL stop read before the cancel
    (_floor_before_cancel). The stop goes back at the higher of it and the
    record, so a native TRAIL's trigger isn't given back (professor M228)."""
    import alerts as alert_module
    import db
    import ibkr
    import tasks
    stop_rec = db.get_position_stop(ticker)
    if not stop_rec or not stop_rec.get("stop_price"):
        return
    try:
        pos = ibkr.get_client().get_open_position(ticker)
        held = round(abs(float(pos.qty)), 4)
        # Only the part not already covered by a stop that never got
        # cancelled, or the two together would oversell (A2253-1). A stop
        # still PendingCancel doesn't count (_restore_cover).
        cancelling = []
        covered = _restore_cover(ticker, cancelling)
        if covered is None:
            # Unknown coverage: a full stop on top of one that is still
            # working would oversell, so place nothing and say so (A0053-2).
            # Queued: the reprotect retry waits while coverage is unreadable
            # and sizes to what other stops leave once it can read (m116), so
            # the stop comes back in ~5 min rather than at reconcile section 3
            # (professor M131).
            log.error(f"[{ticker}] Stop restore skipped: resting stops could not be read")
            queued = False
            try:
                tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
                queued = True
            except Exception as e2:
                log.error(f"[{ticker}] Could not queue the stop restore: {e2}")
            try:
                alert_module._send(
                    f"⚠️ CHECK STOPS — {ticker}\n"
                    f"A sell did not go through and the resting stops could not be read, "
                    f"so no stop was restored"
                    + (" yet; it is queued and retried every reconcile cycle" if queued else "")
                    + f". Check TWS; recorded stop ${float(stop_rec['stop_price']):.2f}."
                )
            except Exception:
                pass
            return
        qty = round(held - covered, 4)
        if qty <= 0:
            return
        import risk_manager
        record_px = float(stop_rec["stop_price"])
        # The position read's last price: a floor at/above it would fill at once.
        px = risk_manager._floored_level(ticker, record_px, floor, getattr(pos, "current_price", None),
                                         "Stop restore after a failed sell")
        result = tasks.manage_stop_place(ticker, qty, px)
        _queue_coverage_recheck(ticker, cancelling)
        # Placed: a record write failure must not read as a failed restore (M227).
        _record_placed_stop(ticker, "Stop restore after a failed sell", result.get("order_id"),
                            [("order id", lambda: db.set_stop_broker_order_id(ticker, result["order_id"]))]
                            + _raise_record_writes(ticker, px, record_px, stop_rec))
        log.info(f"[{ticker}] Sell failed; stop restored @ ${px:.2f} for {qty} shares")
    except Exception as e:
        log.error(f"[{ticker}] Sell failed and the stop could not be restored: {e}")
        # Hand it to the reconcile retry queue, which re-reads coverage under
        # the lock and places the recorded stop: a lapsed lock (LockLost) or a
        # busy worker shouldn't leave recovery to a manual step (A0653-5).
        queued = False
        try:
            tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
            queued = True
        except Exception as e2:
            log.error(f"[{ticker}] Could not queue the stop restore: {e2}")
        try:
            alert_module._send(
                f"🚨 UNPROTECTED POSITION — {ticker}\n"
                f"A sell failed after its stop was cancelled, and the stop could not be "
                f"restored. "
                + ("It is queued and retried every reconcile cycle; " if queued else "")
                + f"place stop @ ${float(stop_rec['stop_price']):.2f} manually if it isn't back "
                f"within ~15 min."
            )
        except Exception:
            pass
        tasks.open_manual_action(ticker, f"restore the stop @ ${float(stop_rec['stop_price']):.2f}")


def _record_stale_entry(ticker: str, prior: dict, filled_qty: float, broker_qty: float | None,
                        path: str = "execution", decision_basis: str | None = None) -> None:
    """Activity row (stage stale_record_at_entry) for a BUY filled while a stop
    record from a position the broker no longer holds still exists. path
    names the caller: "execution" (broker_qty from the position read),
    "reconcile" or "reconcile_manual" (a late fill; broker_qty None, the
    approval's action or a position read decided), "bracket" or "manual".
    decision_basis says what called the position fresh: "post_fill_read",
    "approval_action", "position_read" or "pre_order_read". Record only;
    never raises."""
    try:
        import db
        versions = {}
        try:
            from agents.analyst import shadow_versions
            versions = shadow_versions()
        except Exception:
            pass
        opened = prior.get("opened_at")
        # The old stop's broker order id and entry, so each row can be
        # checked later: an old order filled or cancelled means a real exit,
        # one still working means a lagged position read (professor m586).
        # get_position_stop doesn't select the order id; this read is rare.
        # Strict, so a null id from a failed read is told apart from a
        # record with no order id (professor run 58).
        try:
            # Its own guard: a pool error here (raised before the read's own
            # try) leaves the id null instead of losing the row (SEF 133c).
            prior_order = next((r.get("broker_order_id") for r in db.get_all_position_stops(strict=True)
                                if r.get("ticker") == ticker), None)
            order_read = "ok"
        except Exception:
            prior_order, order_read = None, "error"
        log.warning(f"[{ticker}] BUY filled with a stale stop record (opened {opened}): "
                    f"the new position keeps that record's opened_at and entry thesis")
        db.log_activity_direct(
            "stale_record_at_entry", ticker,
            f"BUY filled while a stop record from an unbooked exit still exists (opened {opened})",
            "neutral",
            {"prior_opened_at": opened.isoformat() if hasattr(opened, "isoformat") else opened,
             "prior_stop_method": prior.get("stop_method"),
             "prior_bucket": prior.get("thesis_bucket"),
             "prior_entry_price": prior.get("entry_price"), "prior_broker_order_id": prior_order,
             "prior_order_read": order_read, "decision_basis": decision_basis,
             "filled_qty": filled_qty, "broker_qty": broker_qty, "path": path, "env": db.TRADING_ENV,
             "code_sha": versions.get("code_sha"), "env_sha": versions.get("env_sha")})
    except Exception as e:
        log.warning(f"[{ticker}] stale-record entry row not written: {e}")


def _note_stale_record(ticker: str, filled_qty: float, action: str | None, path: str,
                       decision_basis: str | None = None) -> None:
    """The stale_record_at_entry check for the BUY paths that write the stop
    record without _whole_position_stop: bracket and manual (professor m586).
    Call before the record is written over. Fresh vs added is decided as on
    the late-fill path (tasks._held_before_fill: the action when it is ADD
    or OPEN, else a position read). Record only; never raises."""
    try:
        import db
        import tasks
        try:
            prior = db.get_position_stop(ticker, strict=True)
        except Exception as e:
            log.warning(f"[{ticker}] stale-record check skipped, stop record unreadable: {e}")
            return
        if not prior:
            return
        seen: dict = {}
        if tasks._held_before_fill(ticker, filled_qty, action, seen=seen):
            return
        basis = decision_basis or ("approval_action" if action in ("ADD", "OPEN") else "position_read")
        _record_stale_entry(ticker, prior, filled_qty, seen.get("broker_qty"), path=path,
                            decision_basis=basis)
    except Exception as e:
        log.warning(f"[{ticker}] stale-record check failed: {e}")


def _whole_position_stop(ticker: str, new_stop: float, filled_qty: float,
                         status: dict | None = None, market_price: float | None = None):
    """Stop handling when a BUY adds to a position that already has a stop.

    Before 2026-10-27 an ADD overwrote entry_price with the add's fill price
    and placed a broker stop for only the added shares, leaving the first
    tranche on an older stop the DB no longer knew about. Now: one stop for
    the whole position, at the higher of the old and new stop (never
    loosened), with the broker's average cost as the entry price.

    Returns (stop_price, (avg_cost, total_qty)) for an add, or
    (new_stop, None) for a fresh position or when the broker lookup fails
    (then the old behaviour applies: a stop for the filled shares only).

    status (optional dict) is filled in for the caller's alerts:
    "prior_stop" = the existing record's stop price on an add, and
    "prior_unreadable" = True whenever the record read failed (pool exhausted,
    DB down, or a query error: the read is strict). Then only the filled
    shares are stopped, at new_stop, and the old stop is not cancelled: the
    earlier shares keep their resting stop, and no whole-position level is
    guessed without the old one. The caller writes the record only once a
    read after placement shows none exists."""
    import db
    import ibkr
    import tasks
    try:
        prior = db.get_position_stop(ticker, strict=True)   # a query error is "unknown", not "none"
    except Exception as e:
        log.warning(f"[{ticker}] prior stop record unreadable ({type(e).__name__}: {e}); "
                    f"stopping only the {filled_qty:g} filled shares and leaving any old stop")
        # Always "unknown", never read from the positions: the portfolio
        # cache lags a fill (tasks.py, QBTS note) and can show a real add as
        # flat or as exactly this fill.
        if status is not None:
            status["prior_unreadable"] = True
        return new_stop, None
    if not prior or not prior.get("stop_price"):
        return new_stop, None
    try:
        pos = ibkr.get_client().get_open_position(ticker)
        total_qty = round(abs(float(pos.qty)), 4)
        avg_cost = float(pos.avg_entry_price)
    except Exception as e:
        log.warning(f"[{ticker}] ADD: position lookup failed, stopping only the added shares: {e}")
        return new_stop, None
    if total_qty <= filled_qty:
        # Nothing was held before this fill, yet a stop record exists: the
        # old position's exit isn't booked yet, so this entry keeps its
        # opened_at and entry thesis (audit 11-06 03:34 #5). Log-only.
        _record_stale_entry(ticker, prior, filled_qty, total_qty, decision_basis="post_fill_read")
        return new_stop, None
    stop = max(new_stop, float(prior["stop_price"]))
    # A native TRAIL's trigger can sit above the record: the whole-position
    # stop never goes below the highest resting stop it cancels. Read before
    # the cancel; a failed read keeps the level (professor M228).
    # market_price is the fill's price from the caller; an implausible one
    # (None, NaN, <= 0, > 1e9) counts as missing (risk_manager._plausible_price).
    import risk_manager
    stop, _ = risk_manager._floor_at_resting_stop(ticker, stop, "ADD whole-position stop",
                                                  market_price=market_price)
    try:
        tasks.manage_cancel_stop_orders_for(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] ADD: could not cancel the old stop, stopping only the added shares: {e}")
        return new_stop, None
    log.info(f"[{ticker}] ADD: one stop for all {total_qty} shares @ ${stop:.2f} "
             f"(old ${float(prior['stop_price']):.2f}, new ${new_stop:.2f}); avg cost ${avg_cost:.2f}")
    if status is not None:
        status["prior_stop"] = float(prior["stop_price"])
    return stop, (avg_cost, total_qty)


def log_buy_outcome(approval: dict, ticker: str, action: str, entry_price: float,
                    entry_date=None, entry_benchmark_price: float | None = None) -> bool:
    """The trade_outcomes row for a filled BUY, with the analyst's scores from
    the approval. Called at the fill and by reconcile for a BUY that filled
    after the wait window (professor M123). False if the write failed."""
    import db
    analyst_json = approval.get("analyst_json") or {}
    try:
        planned_rps, fill_rps = _risk_per_share(approval, entry_price)
    except Exception as e:                 # log-only: never block the outcome row
        # warning, not debug: a silent failure leaves both columns NULL
        # with nothing in the log to say why.
        log.warning(f"[{ticker}] risk-per-share record failed: {e}")
        planned_rps, fill_rps = None, None
    return db.log_trade_outcome(
        approval_id=approval["id"],
        ticker=ticker,
        action=action,
        conviction=approval.get("conviction", 5),
        bucket=approval.get("thesis_bucket", ""),
        entry_price=entry_price,
        thesis_strength=analyst_json.get("thesis_strength"),
        catalyst_quality=analyst_json.get("catalyst_quality"),
        valuation_risk_reward=analyst_json.get("valuation_risk_reward"),
        data_completeness=analyst_json.get("data_completeness"),
        chase_risk=analyst_json.get("chase_risk"),
        chase_risk_pct_from_kijun=analyst_json.get("chase_risk_pct_from_kijun"),
        price_action=analyst_json.get("price_action"),
        price_action_structure=analyst_json.get("price_action_structure"),
        # Catalyst/expected-return tracking (2026-08-30, user request) --
        # promotes what used to only live in this same analyst_json blob
        # into first-class trade_outcomes columns so it can actually be
        # compared against what happens later, not just referenced at
        # the moment of the original call.
        catalyst_event=(analyst_json.get("catalyst") or {}).get("event"),
        catalyst_date=(analyst_json.get("catalyst") or {}).get("date"),
        catalyst_is_priced_in=(analyst_json.get("catalyst") or {}).get("is_priced_in"),
        expected_return_pct=analyst_json.get("expected_return_pct"),
        expected_downside_pct=analyst_json.get("expected_downside_pct"),
        thesis_summary=analyst_json.get("thesis_summary"),
        sources=analyst_json.get("sources"),
        source_count=analyst_json.get("source_count"),
        weeks_to_catalyst=analyst_json.get("weeks_to_catalyst"),
        expected_return_per_week=analyst_json.get("expected_return_per_week"),
        entry_date=entry_date,
        entry_benchmark_price=entry_benchmark_price,
        planned_risk_per_share=planned_rps,
        fill_risk_per_share=fill_rps,
        imputed_fields=analyst_json.get("imputed_fields"),
    )


# Why the last execute_approved call in this thread returned False, for the
# approval's status_reason. Every False used to be recorded as a bare FAILED,
# whether it was a scheduled retry, a deliberate skip or a real failure
# (professor M-6, 2026-10-28). The status stays FAILED (retry_deferred_entry
# keys on it); the reason says which.
_OUTCOME = threading.local()


def _outcome(reason: str) -> bool:
    _OUTCOME.reason = reason
    return False


def last_outcome_reason() -> str | None:
    return getattr(_OUTCOME, "reason", None)


def outcome_deferred(reason: str | None) -> bool:
    """True when execute_approved returned False because it scheduled a
    retry_deferred_entry re-run (market closed, stale price, gap check) and
    already sent its own deferral alert. The approval stays FAILED (the retry
    claims it FAILED -> PROCESSING), but callers must not report it as an
    execution failure. A retry that couldn't be scheduled is a "failed:"
    reason instead, so it still alerts."""
    return bool(reason) and reason.startswith("deferred:")


_BUY_RESERVE_TTL_S = 300   # a sent buy's cost counts against cash at most this long
_SIZING_DRIFT_TOLERANCE = 1.08   # same 8% the drift guard allows


def _buy_reservations() -> list[dict]:
    """Buys approved in the last few minutes: {ticker, notional, cash_before, ts}.
    IBKR's cash figure doesn't subtract unfilled orders and refreshes on a timer."""
    import stop_lock
    r = stop_lock._redis()
    out = []
    for key in r.scan_iter("helios:buy_reserve:*", count=1000):
        val = r.get(key)
        if not val:
            continue
        try:
            notional, cash_before, ts = (float(x) for x in val.decode().split("|"))
        except ValueError:
            log.warning(f"Ignoring malformed buy reservation {key!r}")
            continue
        key = key.decode() if isinstance(key, bytes) else key
        out.append({"ticker": key.split(":")[2], "notional": notional,
                    "cash_before": cash_before, "ts": ts})
    return out


def _unreflected(reserved: list[dict], cash_now: float) -> float:
    """Reserved spend not yet visible in IBKR's cash. Once cash has dropped
    by a buy's cost since it was reserved, that buy has filled and is in the
    figure already (third pre-pull review #1: otherwise counted twice)."""
    if not reserved:
        return 0.0
    drop = max(r["cash_before"] for r in reserved) - cash_now
    pending = 0.0
    for r in sorted(reserved, key=lambda r: r["ts"]):
        if drop >= 0.9 * r["notional"]:
            drop -= r["notional"]
        else:
            pending += r["notional"]
    return pending


_RESERVATION = threading.local()   # this thread's last reservation key, by ticker


def _reserve_buy(ticker: str, notional: float, cash_before: float) -> None:
    """One key per order (review 4 #5: a per-ticker key let a second buy of
    the same name overwrite the first one's reservation)."""
    import time
    import stop_lock
    now = time.time()
    key = f"helios:buy_reserve:{ticker}:{time.time_ns()}"
    stop_lock._redis().set(key, f"{notional}|{cash_before}|{now}", ex=_BUY_RESERVE_TTL_S)
    if not hasattr(_RESERVATION, "keys"):
        _RESERVATION.keys = {}
    _RESERVATION.keys[ticker] = key


def _release_buy(ticker: str) -> None:
    """The buy definitely didn't go out: free its cash and slot now."""
    key = getattr(_RESERVATION, "keys", {}).pop(ticker, None)
    if key is None:
        return
    try:
        import stop_lock
        stop_lock._redis().delete(key)
    except Exception as e:
        log.warning(f"[{ticker}] buy reservation not released (expires in {_BUY_RESERVE_TTL_S}s): {e}")


def _buy_room_at_execution(ticker: str, action: str, shares: int, price: float | None,
                           sized_at: float | None = None) -> tuple[int, str | None]:
    """Re-check cash and the position cap when a buy is actually sent.

    Sizing checks both, but a buy sized before the open runs later with its
    stored share count, and buys deferred to the same open run in parallel
    against the same cash (pre-pull reviews 10-29). One check at a time
    (a Redis lock), each subtracting buys approved in the last 5 minutes that
    IBKR's cash doesn't show yet, under the same per-trade cash limits as
    sizing (with the drift guard's 8% tolerance). Returns (shares to buy,
    None), or (0, reason) to skip. Fails closed, including without Redis."""
    if not price:
        return 0, "no price to check cash against at execution"
    try:
        import ibkr
        import stop_lock
        from config import (ALLOW_ONE_SHARE_ABOVE_ALLOCATION, MAX_TRADE_PCT_OF_CASH_RESERVE,
                            ONE_SHARE_MAX_PCT_OF_CASH, get_active_limits)
        from risk_manager import RiskManager
        stop_lock._redis().ping()   # ticker_lock runs unlocked without Redis; this must not
        with stop_lock.ticker_lock("__BUY_ROOM__", wait=60):
            rm = RiskManager(ibkr.get_client())
            # All reservations count toward cash, the own ticker's too (a
            # second buy of it within 5 minutes); the cap ignores the own
            # ticker since buying more of a name takes no new slot.
            reserved = _buy_reservations()
            limits = get_active_limits(rm.portfolio_value())
            cap = limits.get("max_concurrent_positions")
            if action == "OPEN" and cap is not None:
                held = {p.symbol for p in rm._client.get_all_positions()}
                taken = held | {r["ticker"] for r in reserved if r["ticker"] != ticker}
                if ticker not in taken and len(taken) >= cap:
                    return 0, f"position cap reached at execution ({len(taken)} of {cap} held or being bought)"
            cash_now = rm.cash()
            cash = max(cash_now - _unreflected(reserved, cash_now), 0)
            # The caps are exact against today's cash. The drift tolerance
            # applies to the price only: a buy sized at sized_at may still
            # count its shares at that price if the price has moved up by no
            # more than 8% since (professor M98: it used to widen the caps).
            cap_price = (sized_at if sized_at and price <= sized_at * _SIZING_DRIFT_TOLERANCE
                         else price)
            cap_price = min(cap_price, price)
            affordable = int(cash * MAX_TRADE_PCT_OF_CASH_RESERVE / cap_price)
            if (affordable < 1 and ALLOW_ONE_SHARE_ABOVE_ALLOCATION
                    and cap_price <= cash * ONE_SHARE_MAX_PCT_OF_CASH):
                affordable = 1
            if affordable < 1:
                return 0, "not enough cash at execution under the per-trade cash limits"
            if shares > affordable:
                log.warning(f"[{ticker}] {action} cut from {shares} to {affordable} shares: "
                            f"cash changed since sizing")
                shares = affordable
            _reserve_buy(ticker, shares * price, cash_now)
            return shares, None
    except Exception as e:
        return 0, f"cash/position check failed at execution ({str(e)[:100]})"


def _manual_buy_room(ticker: str, qty: float, price: float | None) -> str | None:
    """Cash check for a manual BUY (dashboard, !urgent). The size is the
    human's choice, so no per-trade cap; but it may not spend more than the
    cash left after buys still in flight, and it reserves its cost so
    automated buys sent meanwhile see it (professor M99). Returns a reason
    to block, or None. Fails closed, including without Redis."""
    if not price:
        return "no price to check cash against"
    try:
        import ibkr
        import stop_lock
        from risk_manager import RiskManager
        stop_lock._redis().ping()
        with stop_lock.ticker_lock("__BUY_ROOM__", wait=60):
            rm = RiskManager(ibkr.get_client())
            cash_now = rm.cash()
            cash = max(cash_now - _unreflected(_buy_reservations(), cash_now), 0)
            if qty * price > cash:
                return (f"not enough cash: {qty:g} x ${price:,.2f} is more than the cash left "
                        f"after buys in flight")
            _reserve_buy(ticker, qty * price, cash_now)
            return None
    except Exception as e:
        return f"cash check failed ({str(e)[:100]})"


_STOP_MIN_GAP_BELOW_FILL = 0.01
_MANUAL_FALLBACK_STOP_PCT = 0.08   # the flat fallback's widest band in calculate_stop


def _stop_anchored_to_fill(ticker: str, stop: float, sized_at: float | None,
                           fill: float | None) -> float:
    """The stop was set from the sizing price. If the fill came in lower (a
    pre-market price that faded by the open, review 5 #1) and left the stop
    at, above or within 1% of the fill, move it to the planned distance below
    the fill. A stop still safely below the fill keeps its level, so a
    structural stop at support isn't moved (28323dd review #2/#3)."""
    anchored = _anchored_stop(stop, sized_at, fill)
    if anchored != stop:
        log.warning(f"[{ticker}] Stop moved from ${stop:.2f} to ${anchored:.2f}: filled at "
                    f"${fill:.2f}, below the ${sized_at:.2f} sizing price")
    return anchored


def _anchored_stop(stop: float, sized_at: float | None, fill: float | None) -> float:
    """_stop_anchored_to_fill's rule, without the log line."""
    if not (sized_at and fill and 0 < stop < sized_at and fill < sized_at):
        return stop
    if stop < fill * (1 - _STOP_MIN_GAP_BELOW_FILL):
        return stop
    return round(min(stop, fill * stop / sized_at), 2)


def _risk_per_share(approval: dict, fill: float | None) -> tuple[float | None, float | None]:
    """(as sized, at the fill) per-share risk to the stop for a BUY, log-only
    (audit A1031-1053-1). The fill side uses the stop placed for the new
    shares (anchored to a lower fill), before an ADD merges stops."""
    risk_json = approval.get("risk_json") or {}
    try:
        stop = float(approval.get("stop_loss_price") or risk_json.get("stop_loss_price") or 0)
        sized_at = float(risk_json.get("price_at_sizing") or 0) or None
    except (TypeError, ValueError):
        return None, None
    if stop <= 0:
        return None, None
    planned = round(sized_at - stop, 4) if sized_at and sized_at > stop else None
    at_fill = round(float(fill) - _anchored_stop(stop, sized_at, float(fill)), 4) if fill else None
    return planned, at_fill


def _log_skipped_sell(ticker: str, action: str, shares, approval: dict) -> None:
    """Log-only (audit A0903-4): a TRIM/EXIT decided while the market is
    closed is dropped, where a buy is deferred to the open. Record the
    decision and the price then, so the cost of not deferring sells can be
    measured against the next open before anything changes. Never raises."""
    try:
        import db
        import ibkr
        try:
            price = ibkr.get_latest_price(ticker)
        except Exception:
            price = None
        db.log_signal(ticker, "SELL_SKIPPED_CLOSED", 0.0,
                      f"{action} {shares} not sent: market closed; would defer to the open",
                      provenance={"approval_id": approval.get("id"), "action": action,
                                  "shares": shares, "conviction": approval.get("conviction"),
                                  "price_at_skip": price})
    except Exception as e:
        log.warning(f"[{ticker}] could not log the skipped sell: {e}")


def _send_wait_alert(text: str, attempt: int, ticker: str) -> None:
    """A WAITING notice goes out on the first check of a wait only; later
    checks of the same wait are logged (audit 2026-11-07 19:07 #7: one ANET
    order sent 3 WAITING notices, then ORDER ABORTED). The final result
    (executed, aborted, skipped) is still sent on its own. Never raises."""
    if attempt > 1:
        log.info(f"[{ticker}] waiting notice not re-sent (check {attempt}): {text.splitlines()[0]}")
        return
    try:
        import alerts as alert_module
        alert_module._send(text + "\nLater re-checks of this wait send no notice; the result is sent when it's known.")
    except Exception:
        pass


# The whole execution holds the ticker's stop lock: a SELL cancels the stop,
# sells and re-places it; a BUY places the first stop (stop_lock).
@locked_by_ticker(lambda approval, *a, **k: approval["ticker"],
                  on_busy=lambda: _outcome("skipped: another stop change held the ticker"))
def execute_approved(
    approval: dict, trigger: str = "telegram",
    gap_retry_attempt: int = 1, price_retry_attempt: int = 1,
) -> bool:
    """
    Submit the approved order to IBKR via TWS, log the trade and stop.

    trigger names how this fired, for the trades.reason audit trail
    (_TRIGGER_LABELS): "auto" for the crew.py pipeline (no human ever saw
    this), "telegram" for an actual Approve-button tap, "auto_retry" for
    tasks.retry_deferred_entry and "held_sell_retry" for tasks.retry_held_sell.
    The one behavior difference: a held_sell_retry that is held again is not
    re-scheduled (_schedule_held_sell_retry).

    gap_retry_attempt counts opening-gap-confirmation re-checks (see the
    gap-confirmation block below and tasks.retry_deferred_entry).
    price_retry_attempt counts stale-price re-checks (see the price-
    freshness block below) -- a separate counter, since a BUY can hit either
    gate independently of the other. Both are distinct from the
    market-hours-closed retry, which always starts fresh attempt counts of 1.

    Returns True on success; on False, last_outcome_reason() says why.
    """
    _OUTCOME.reason = None
    ticker    = approval["ticker"]
    action    = approval["action"]
    risk_json = approval.get("risk_json") or {}

    # Pre-flight: verify IBKR is reachable before touching the market
    try:
        import ibkr as _ibkr_check
        _ibkr_check.get_ib()  # retries 3× internally
    except Exception as conn_err:
        log.error(f"[{ticker}] IBKR not connected — cannot execute order: {conn_err}")
        try:
            import alerts as _al
            _al._send(
                f"⚠️ IBKR CONNECTION FAILED\n"
                f"Could not execute approved order for <b>{ticker}</b>.\n"
                f"Error: {conn_err}\n"
                f"Manual execution required."
            )
        except Exception:
            pass
        return _outcome("failed: IBKR unreachable")

    try:
        import ibkr
        import tasks
        import db
        import alerts as alert_module

        shares = risk_json.get("shares_initial", 0)
        if shares <= 0:
            log.error(f"[{ticker}] No shares_initial in risk_json — cannot execute")
            return _outcome("failed: no shares_initial in risk_json")

        # Drawdown halt — block new BUY entries if portfolio is in drawdown
        if action in ("OPEN", "ADD"):
            try:
                from risk_manager import RiskManager
                rm = RiskManager(ibkr.get_client())
                if rm.is_halted():
                    if getattr(rm, "halt_unknown", False) is True:
                        log.error(f"[{ticker}] Order blocked — halt state or account value unreadable")
                        alert_module._send(
                            f"🛑 ORDER BLOCKED — halt check unreadable\n"
                            f"<b>{ticker}</b> {action} was approved but not sent: the halt state or "
                            f"the account value couldn't be read safely. No halt is set; "
                            f"nothing to reset. Sells and stops are unaffected."
                        )
                        return _outcome("blocked: halt check unreadable")
                    log.error(f"[{ticker}] Order blocked — portfolio drawdown halt active")
                    alert_module._send(
                        f"🛑 ORDER BLOCKED — Drawdown Halt\n"
                        f"<b>{ticker}</b> {action} order was approved but cannot execute.\n"
                        f"Portfolio drawdown limit is active. No new BUY entries allowed.\n"
                        f"Review and reset the halt flag when drawdown is resolved."
                    )
                    return _outcome("blocked: drawdown halt active")
            except Exception as halt_err:
                # Fail closed: an unchecked halt must not let a buy through
                # (A1031-1053-2). It ends FAILED; only a manual retry re-runs it.
                log.warning(f"[{ticker}] Drawdown halt check failed: {halt_err} — blocking")
                try:
                    alert_module._send(
                        f"🛑 ORDER BLOCKED — halt check failed\n"
                        f"<b>{ticker}</b> {action} was approved but not sent: the drawdown halt "
                        f"check errored. No halt is set. Sells and stops are unaffected."
                    )
                except Exception:
                    pass
                return _outcome("blocked: halt check failed")

        if action in ("OPEN", "ADD"):
            side = "BUY"
            # New-entry orders only, gated on market hours -- found
            # 2026-07-26: an auto-executed ADD submitted a DAY market BUY
            # while the market was closed (Sunday evening). IBKR doesn't
            # reject that, it queues the order for the next session
            # (warning 399), so it sat live at the broker while our own
            # 30s fill-wait declared the trade FAILED with nothing
            # recorded -- an orphaned, unprotected order that would have
            # tried to fill at next open with no stop and no trade log
            # entry. A stale, hours-old recommendation shouldn't silently
            # execute at whatever price next open brings anyway; the
            # analyst should re-run fresh when the market's actually open.
            # Risk-reducing SELLs (TRIM/EXIT) are deliberately NOT gated
            # here -- ibkr.place_market_order's own market-closed check is
            # the backstop for those, since blocking a protective sell
            # outright would go against the "never abort a risk-reducing
            # order" principle elsewhere in this file.
            try:
                from market_data import is_market_hours, next_market_open
                if not is_market_hours():
                    eta = next_market_open()
                    log.error(f"[{ticker}] {action} blocked — market is closed, "
                              f"not submitting a new entry order; retry scheduled for "
                              f"{eta.isoformat()}")
                    scheduled = False
                    try:
                        from tasks import retry_deferred_entry
                        retry_deferred_entry.apply_async(
                            args=[approval["id"]], eta=eta, queue="ops",
                        )
                        scheduled = True
                    except Exception as e:
                        log.error(f"[{ticker}] Could not schedule market-open retry "
                                  f"for approval #{approval.get('id')}: {e}")
                    from zoneinfo import ZoneInfo
                    eta_et = eta.astimezone(ZoneInfo("America/New_York"))
                    alert_module._send(
                        f"⏸️ ORDER DEFERRED — Market Closed\n"
                        f"<b>{ticker}</b> {action} was approved but the market is closed.\n"
                        f"Not submitting now — a stale recommendation shouldn't fill blind. "
                        f"Will retry automatically at the next open "
                        f"({eta_et.strftime('%a %I:%M %p ET')}) with a fresh price/drift check."
                    )
                    if not scheduled:
                        return _outcome("failed: market closed, market-open retry not scheduled")
                    return _outcome("deferred: market closed, retry scheduled for the open")
            except Exception as e:
                log.warning(f"[{ticker}] Market-hours check failed, proceeding: {e}")
        elif action in ("TRIM", "EXIT"):
            side = "SELL"
            # Clamp to what we actually hold — a TRIM sized larger than the
            # position must never flip the account short.
            try:
                qty   = ibkr.get_client().get_open_position(ticker).qty
                owned = round(abs(float(qty)), 4)
            except Exception:
                owned = 0.0
            if owned <= 0:
                log.error(f"[{ticker}] {action} requested but no position found")
                return _outcome("failed: no position to sell")
            shares = owned if action == "EXIT" else min(shares, owned)
            # Outside regular hours ibkr.place_market_order cancels the DAY
            # order anyway, but the stop was already gone and was never put
            # back, so the position sat unprotected into the open (limitations
            # audit 20:53, A2053-3). Don't touch the stop or send the order:
            # same outcome for the sale, stop kept.
            from market_data import is_market_hours
            if not is_market_hours():
                log.warning(f"[{ticker}] {action} not sent: market closed; stop left in place")
                try:
                    alert_module._send(
                        f"⏸️ {action} NOT SENT — <b>{ticker}</b>\n"
                        f"The market is closed, so the sell order would be cancelled unfilled. "
                        f"The protective stop is unchanged; the next analysis during market "
                        f"hours can repeat the {action}."
                    )
                except Exception:
                    pass
                _log_skipped_sell(ticker, action, shares, approval)
                return _outcome("skipped: market closed, sell not sent (stop kept)")
            # NOTE: broker stops are cancelled AFTER the drift check below —
            # cancelling first meant a drift abort left the position both
            # unsold and unprotected.
        else:
            log.info(f"[{ticker}] Action {action} does not require order submission")
            return True

        # Price freshness check — reject if market moved >8% since sizing.
        # Uses the staleness-aware quote so a last-RTH price hours after the
        # close can't masquerade as "no drift".
        price_at_sizing = risk_json.get("price_at_sizing")
        current_price_pre = None
        try:
            from market_data import get_priced_quote
            pq = get_priced_quote(ticker)
            if pq and pq["is_stale"] and side == "BUY":
                # Incident 2026-07-29 (TER): right at the open, IBKR's
                # intraday 1-min bars aren't populated yet ("HMDS query
                # returned no data"), so get_priced_quote falls back to a
                # daily-bar close that can be many hours old. The staleness
                # was correctly detected and logged/alerted below, but the
                # drift check still used that stale price for the
                # comparison -- against a price_at_sizing computed from the
                # same pre-gap close, so it looked like ~0% drift and six
                # queued market-open retries all bought straight into a
                # ~13% earnings gap with the guard never firing. Give the
                # intraday feed a few seconds to catch up (it usually does)
                # before treating the price as unusable for a BUY.
                import time as _time
                for _ in range(3):
                    _time.sleep(3)
                    # A None on a re-check keeps the last quote: replacing it
                    # with None skipped the stale abort and the drift check.
                    pq = get_priced_quote(ticker) or pq
                    if pq and not pq["is_stale"]:
                        break
            if pq is None:
                # No 1-min or daily bar at all: use the latest price, as on
                # an exception below, so the drift check still runs.
                log.warning(f"[{ticker}] No priced quote for the {side}; drift check uses the latest price")
                current_price_pre = ibkr.get_latest_price(ticker)
            if pq:
                current_price_pre = pq["price"]
                if pq["is_stale"]:
                    log.warning(
                        f"[{ticker}] Price is STALE (age={pq['age_minutes']}min, "
                        f"market_open={pq['market_open']}, "
                        f"source={pq.get('price_source')}) — drift check unreliable"
                    )
                    if side == "BUY":
                        # A stale price can't rule out a large gap since
                        # sizing -- proceeding would compare a stale price
                        # to itself and silently defeat the drift guard
                        # (exactly what let TER's six retries chase a 13%
                        # gap). Abort rather than buy blind; sells are
                        # unaffected -- a risk-reducing order already
                        # proceeds through drift by design (see below).
                        #
                        # Incident 2026-07-30/31: this used to just abort
                        # for good here, with no retry at all -- unlike the
                        # market-hours-closed gate and the gap-confirmation
                        # gate below, both of which reschedule themselves.
                        # IBKR's HMDS intraday-bar service reliably has no
                        # data for the first few minutes after the open, so
                        # this fired (and permanently killed) 5 conviction
                        # 7-8 approvals on 2026-07-31 alone, every one of
                        # them a data feed catching up, not a real problem.
                        # Now retries on the same bounded, ever-later-eta
                        # shape as GAP_CONFIRMATION_* instead of giving up
                        # on the first miss.
                        from config import STALE_PRICE_MAX_RETRIES, STALE_PRICE_RETRY_MINUTES
                        try:   # log-only: would a Yahoo quote have been fresh? (off the order path)
                            from datetime import datetime as _dt, timezone as _tz
                            from tasks import stale_quote_fallback_shadow
                            # retry=False: a slow broker must not hold the order path (m545)
                            stale_quote_fallback_shadow.apply_async(
                                args=[ticker, approval.get("id"), price_retry_attempt,
                                      pq.get("age_minutes"), _dt.now(_tz.utc).isoformat()],
                                queue="ops", expires=600, retry=False)
                        except Exception as e:
                            log.debug(f"[{ticker}] quote fallback shadow not queued: {e}")
                        if price_retry_attempt < STALE_PRICE_MAX_RETRIES:
                            from datetime import datetime, timedelta, timezone
                            eta = datetime.now(timezone.utc) + timedelta(minutes=STALE_PRICE_RETRY_MINUTES)
                            log.warning(
                                f"[{ticker}] Could not get a fresh price "
                                f"(attempt {price_retry_attempt}/{STALE_PRICE_MAX_RETRIES}) — "
                                f"re-checking at {eta.isoformat()} rather than risk an unguarded gap"
                            )
                            scheduled = False
                            try:
                                from tasks import retry_deferred_entry
                                retry_deferred_entry.apply_async(
                                    args=[approval["id"], gap_retry_attempt, price_retry_attempt + 1],
                                    eta=eta, queue="ops",
                                )
                                scheduled = True
                            except Exception as e:
                                log.error(f"[{ticker}] Could not schedule stale-price re-check: {e}")
                            _send_wait_alert(
                                f"⏸️ WAITING FOR FRESH PRICE — {ticker}\n"
                                f"Only a stale price (${current_price_pre:.2f}) was available "
                                f"(check {price_retry_attempt}/{STALE_PRICE_MAX_RETRIES}) — IBKR's "
                                f"intraday feed usually catches up within a few minutes of the open. "
                                f"Will re-check in {STALE_PRICE_RETRY_MINUTES} min.",
                                price_retry_attempt, ticker)
                            if not scheduled:
                                return _outcome("failed: stale price, re-check not scheduled")
                            return _outcome("deferred: stale price, re-check scheduled")
                        log.error(
                            f"[{ticker}] Could not get a fresh price after "
                            f"{STALE_PRICE_MAX_RETRIES} checks — giving up rather than "
                            f"risk an unguarded gap"
                        )
                        try:
                            alert_module._send(
                                f"⚠️ ORDER ABORTED — Price Unavailable\n"
                                f"<b>{ticker}</b> {action}: still only a stale price "
                                f"(${current_price_pre:.2f}) after {STALE_PRICE_MAX_RETRIES} "
                                f"checks over ~{STALE_PRICE_MAX_RETRIES * STALE_PRICE_RETRY_MINUTES} min "
                                f"— can't rule out a large gap since this was sized. "
                                f"Re-run analysis or check TWS."
                            )
                        except Exception:
                            pass
                        return _outcome("skipped: price still stale after retries")
                    try:
                        alert_module._send(
                            f"⚠️ STALE PRICE — {ticker}\n"
                            f"Latest price ${current_price_pre:.2f} is from a stale bar "
                            f"(market open: {pq['market_open']}). "
                            f"Order will proceed but verify the price in TWS."
                        )
                    except Exception:
                        pass
        except Exception:
            current_price_pre = ibkr.get_latest_price(ticker)
        if price_at_sizing and current_price_pre:
            drift = abs(current_price_pre - price_at_sizing) / price_at_sizing
            if drift > 0.08:
                if side == "BUY":
                    log.error(
                        f"[{ticker}] Price drift {drift:.1%} since sizing "
                        f"(was ${price_at_sizing:.2f}, now ${current_price_pre:.2f}) — aborting order"
                    )
                    try:
                        alert_module._send(
                            f"⚠️ ORDER ABORTED — Price Drift\n"
                            f"<b>{ticker}</b> moved {drift:.1%} since recommendation was sized.\n"
                            f"Sized at: <b>${price_at_sizing:.2f}</b> | Now: <b>${current_price_pre:.2f}</b>\n"
                            f"Re-run analysis to get fresh sizing before approving."
                        )
                    except Exception:
                        pass
                    return _outcome("skipped: price drifted since sizing")
                # A risk-reducing SELL must never be blocked by drift — a big
                # move is often exactly why we're exiting. Warn and proceed.
                log.warning(
                    f"[{ticker}] Price drift {drift:.1%} since sizing on SELL "
                    f"(was ${price_at_sizing:.2f}, now ${current_price_pre:.2f}) — proceeding anyway"
                )

        # Opening-gap confirmation — added 2026-07-29 after TER: a stock
        # that gaps hard on an overnight catalyst can give the entire move
        # back within the first half hour once real regular-session volume
        # shows up (premarket/opening price discovery runs on a fraction of
        # normal volume). This is a separate check from the drift guard
        # above -- drift compares against price_at_sizing and can pass
        # cleanly even for a real gap if the analysis itself was done after
        # the gap already happened. Only gates BUYs; a risk-reducing SELL
        # is never held up waiting for confirmation.
        if side == "BUY":
            try:
                from market_data import check_gap_confirmation
                from config import GAP_CONFIRMATION_MAX_RETRIES, GAP_CONFIRMATION_RETRY_MINUTES
                gap = check_gap_confirmation(ticker)
                if gap["status"] == "unreadable":
                    # Same path as a raised check: retry, then skip (M148).
                    raise RuntimeError("gap check could not read the bars")
                if gap["status"] == "failed":
                    log.error(
                        f"[{ticker}] Opening gap already reversing "
                        f"({gap['retracement_pct']:.0%} given back of a {gap['gap_pct']:.1%} gap) "
                        f"— aborting {action} rather than chase a fading move"
                    )
                    alert_module._send(
                        f"⚠️ ORDER ABORTED — Gap Reversing\n"
                        f"<b>{ticker}</b> gapped {gap['gap_pct']:+.1%} from prior close "
                        f"(${gap['prior_close']:.2f} → ${gap['open_price']:.2f}) but has already "
                        f"given back {gap['retracement_pct']:.0%} of that move "
                        f"(now ${gap['current_price']:.2f}). Skipping — this looks like a fade, "
                        f"not a move that's holding."
                    )
                    return _outcome("skipped: opening gap fading")
                if gap["status"] == "pending":
                    if gap_retry_attempt >= GAP_CONFIRMATION_MAX_RETRIES:
                        log.warning(
                            f"[{ticker}] Opening gap ({gap['gap_pct']:+.1%}) never confirmed "
                            f"within {GAP_CONFIRMATION_MAX_RETRIES} checks — giving up"
                        )
                        alert_module._send(
                            f"⚠️ ORDER SKIPPED — Gap Unconfirmed\n"
                            f"<b>{ticker}</b> gapped {gap['gap_pct']:+.1%} but price action never "
                            f"clearly confirmed or failed within the check window. Skipping this "
                            f"entry rather than guess — re-run analysis if the thesis still holds."
                        )
                        return _outcome("skipped: opening gap unconfirmed after max checks")
                    from datetime import datetime, timedelta, timezone
                    eta = datetime.now(timezone.utc) + timedelta(minutes=GAP_CONFIRMATION_RETRY_MINUTES)
                    log.info(
                        f"[{ticker}] Opening gap ({gap['gap_pct']:+.1%}) not yet confirmed "
                        f"(attempt {gap_retry_attempt}/{GAP_CONFIRMATION_MAX_RETRIES}) — "
                        f"re-checking at {eta.isoformat()}"
                    )
                    scheduled = False
                    try:
                        from tasks import retry_deferred_entry
                        retry_deferred_entry.apply_async(
                            args=[approval["id"], gap_retry_attempt + 1], eta=eta, queue="ops",
                        )
                        scheduled = True
                    except Exception as e:
                        log.error(f"[{ticker}] Could not schedule gap re-check: {e}")
                    _send_wait_alert(
                        f"⏸️ WAITING FOR CONFIRMATION — {ticker}\n"
                        f"Gapped {gap['gap_pct']:+.1%} from prior close — waiting to see if the "
                        f"move holds before entering (check {gap_retry_attempt}/{GAP_CONFIRMATION_MAX_RETRIES}). "
                        f"Will re-check in {GAP_CONFIRMATION_RETRY_MINUTES} min.",
                        gap_retry_attempt, ticker)
                    if not scheduled:
                        return _outcome("failed: gap re-check not scheduled")
                    return _outcome("deferred: waiting for gap confirmation, re-check scheduled")
                # "confirmed" or "no_gap" -- proceed normally.
            except Exception as e:
                # Fails closed like the stale-price check above (limitations
                # audit 2026-10-28 14:53, #8): retry later, then skip.
                from config import GAP_CONFIRMATION_MAX_RETRIES, GAP_CONFIRMATION_RETRY_MINUTES
                if gap_retry_attempt < GAP_CONFIRMATION_MAX_RETRIES:
                    from datetime import datetime, timedelta, timezone
                    eta = datetime.now(timezone.utc) + timedelta(minutes=GAP_CONFIRMATION_RETRY_MINUTES)
                    log.warning(f"[{ticker}] Gap-confirmation check failed ({e}); "
                                f"re-checking at {eta.isoformat()}")
                    try:
                        from tasks import retry_deferred_entry
                        retry_deferred_entry.apply_async(
                            args=[approval["id"], gap_retry_attempt + 1], eta=eta, queue="ops",
                        )
                    except Exception as e2:
                        log.error(f"[{ticker}] Could not schedule gap re-check: {e2}")
                        return _outcome("failed: gap re-check not scheduled")
                    _send_wait_alert(
                        f"⏸️ WAITING FOR CONFIRMATION — {ticker}\n"
                        f"The opening-gap check could not run ({str(e)[:120]}); re-checking in "
                        f"{GAP_CONFIRMATION_RETRY_MINUTES} min "
                        f"(check {gap_retry_attempt}/{GAP_CONFIRMATION_MAX_RETRIES}).",
                        gap_retry_attempt, ticker)
                    return _outcome("deferred: gap re-check scheduled")
                log.error(f"[{ticker}] Gap-confirmation check failed again ({e}); skipping {action}")
                alert_module._send(
                    f"⚠️ ORDER SKIPPED — Gap Check Failed\n"
                    f"<b>{ticker}</b>: the opening-gap check could not run ({str(e)[:120]}). "
                    f"Skipping this entry rather than buy unchecked."
                )
                return _outcome("skipped: gap check could not run")

        if side == "BUY":
            shares, room_block = _buy_room_at_execution(
                ticker, action, shares, current_price_pre or price_at_sizing, price_at_sizing)
            if room_block:
                log.error(f"[{ticker}] {action} not sent: {room_block}")
                try:
                    alert_module._send(f"⏸️ {action} NOT SENT — <b>{ticker}</b>\n{room_block}.")
                except Exception:
                    pass
                return _outcome(f"skipped: {room_block}")

        pre_cancel_prices: list = []   # filled by the read below (professor M228)
        if side == "SELL" and _stops_still_resting(ticker, prices_out=pre_cancel_prices) is None:
            # Unreadable before the cancel: hold the sell with the stop still
            # in place, rather than cancel it and then find out (professor M131).
            log.error(f"[{ticker}] {action} not sent: resting stops could not be read; stop left in place")
            try:
                alert_module._send(
                    f"⚠️ {action} NOT SENT — <b>{ticker}</b>\n"
                    f"The broker's resting stops could not be read, so the sell was held and the "
                    f"stop left in place. Check TWS."
                )
            except Exception:
                pass
            _schedule_held_sell_retry(approval, trigger)
            return _outcome("skipped: resting stops unreadable, sell not sent (stop left in place)")

        # Highest resting stop, read before the cancel: a restore or the TRIM
        # re-place never goes below a native TRAIL's trigger (professor M228).
        sell_floor = _floor_before_cancel(ticker, pre_cancel_prices, f"{action} stop") \
            if side == "SELL" else None
        if side == "SELL":
            # Cancel the existing broker stop only now that the sell is
            # definitely going ahead, so the fill can't race the stop and
            # leave an order that would short the account.
            try:
                cancelled = tasks.manage_cancel_stop_orders_for(ticker)
                if cancelled:
                    log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before {action}")
            except Exception as e:
                log.warning(f"[{ticker}] Could not cancel existing stops before {action}: {e}")
            # The cancel above sees only this client's stops; a resting stop
            # another client placed shows only in this read, so an unreadable
            # broker is treated as a stop still resting.
            resting = _stops_still_resting(ticker)
            if resting or resting is None:
                log.error(f"[{ticker}] {action} not sent: " + (
                    "resting stops could not be read" if resting is None
                    else f"stop(s) for {resting} shares still resting"))
                try:
                    alert_module._send(
                        f"⚠️ {action} NOT SENT — <b>{ticker}</b>\n" + (
                            "The broker's resting stops could not be read" if resting is None
                            else "An existing stop could not be cancelled")
                        + ", and selling on top of a stop could short the account. "
                          "Check the stop orders in TWS."
                    )
                except Exception:
                    pass
                # The cancel may have taken some stops before one stuck;
                # put back what is now uncovered (A0053-2).
                _restore_stop_after_failed_sell(ticker, floor=sell_floor)
                if resting is None:
                    _schedule_held_sell_retry(approval, trigger)
                return _outcome("skipped: resting stops unreadable, sell not sent" if resting is None
                                else "skipped: stop still resting, sell not sent")

        stop_price_for_entry = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")

        try:
            from config import USE_BRACKET_ORDERS
        except Exception:
            USE_BRACKET_ORDERS = False

        bracket_used = False
        early = {"id": None, "perm_id": None}   # the row written before the fill wait
        submitted_at = _now_utc()   # record-keeping only (orders.submitted_at)
        if side == "BUY" and USE_BRACKET_ORDERS and stop_price_for_entry:
            # Native bracket: entry + stop linked by parentId — TWS enforces
            # the pairing atomically, so this can't produce an orphaned stop.
            try:
                bracket_result = ibkr.place_bracket_order(
                    ticker, side, shares, stop_loss_price=float(stop_price_for_entry),
                )
                bracket_used = True
                order_result = {"order_id": bracket_result["parent_order_id"], "fill_price": None, "filled_qty": None}
                log.info(f"[{ticker}] Bracket order submitted: {side} {shares} shares "
                        f"(parent={bracket_result['parent_order_id']}, stop={bracket_result['stop_order_id']})")
            except Exception as e:
                log.warning(f"[{ticker}] Bracket order failed, falling back to market+separate stop: {e}")
                bracket_used = False

        if not bracket_used:
            # Submit market order — result includes the confirmed fill price
            try:
                import stop_lock
                stop_lock.assert_held(ticker)  # professor M69
            except Exception:
                if side == "SELL":
                    _restore_stop_after_failed_sell(ticker, floor=sell_floor)
                else:
                    _release_buy(ticker)   # nothing was sent
                raise
            on_submitted, early = _early_order_row(
                ticker=ticker, side=side, order_type="MARKET", qty=shares,
                decision_price=price_at_sizing, approval_id=approval["id"], intent=action,
                submitted_at=submitted_at, submit_price=current_price_pre,
            )
            try:
                order_result = ibkr.place_market_order(ticker, side, shares,
                                                       on_submitted=on_submitted)
            except Exception:
                # A buy's reservation stays until it expires: the order may
                # have been sent before the error and can still fill (review 4 #6).
                if side == "SELL":
                    _restore_stop_after_failed_sell(ticker, floor=sell_floor)
                raise
            log.info(f"[{ticker}] Order submitted: {side} {shares} shares (id={order_result['order_id']})")
            _recover_early_row(ticker, side, early, order_result)

        if not bracket_used:
            filled_qty = float(order_result.get("filled_qty") or 0)
            if filled_qty <= 0 and side == "BUY" and \
                    order_result.get("status") in _DEAD_ORDER_STATUSES:
                _release_buy(ticker)   # rejected/cancelled: no late fill can come
            if filled_qty <= 0 and side == "SELL" and \
                    order_result.get("status") in _DEAD_ORDER_STATUSES:
                # Definitely not working at the broker, so no late fill can
                # come: put the stop back now (A2053-3).
                log.error(f"[{ticker}] SELL came back {order_result.get('status')} unfilled; "
                          f"restoring the stop")
                _restore_stop_after_failed_sell(ticker, floor=sell_floor)
                if early["id"]:
                    _finish_dead_sell_row(ticker, early["id"], order_result)
                return _outcome(f"skipped: sell {order_result.get('status')} unfilled, stop restored")
            if filled_qty <= 0:
                # Nothing confirmably filled within the wait window. Never
                # fabricate a fill — a phantom trade row plus a stop for
                # shares that may not exist silently diverges DB state from
                # the broker.
                log.error(f"[{ticker}] No fill confirmed within wait window "
                          f"(id={order_result.get('order_id')}) — nothing recorded")
                _log_unconfirmed_order(
                    order_row_id=early["id"],
                    ticker=ticker, side=side, order_type="MARKET",
                    qty=shares, broker_order_id=order_result.get("order_id"),
                    perm_id=order_result.get("perm_id"),
                    decision_price=price_at_sizing, approval_id=approval["id"],
                    status="unconfirmed", intent=action,
                    submitted_at=submitted_at, submit_price=current_price_pre,
                )
                try:
                    stop_note = ""
                    if side == "SELL":
                        # Not restored automatically: the sell may still fill
                        # late, and a full-size stop would then short.
                        stop_note = ("\nThe broker stop was cancelled before the sell and NOT "
                                     "re-placed. Check the position and re-place it if needed.")
                    alert_module._send(
                        f"⚠️ FILL UNCONFIRMED — <b>{ticker}</b>\n"
                        f"{side} {shares} shares submitted (order id "
                        f"{order_result.get('order_id')}) but no fill confirmed within "
                        f"the wait window. Nothing recorded — reconcile in TWS.{stop_note}"
                    )
                except Exception:
                    pass
                return _outcome("unconfirmed: no fill within the wait window (reconcile checks later)")
        else:
            # Bracket path returns no fill info by design — TWS enforces the
            # entry+stop pairing atomically, so assume the requested size.
            filled_qty = shares

        if early["id"]:
            # The order row first, then the trade row (see _mark_early_row_filled).
            if not _mark_early_row_filled(ticker, early, order_result, filled_qty, shares):
                return _outcome(f"{_LEFT_TO_RECONCILE} (order row already adopted)")

        # Real fill price; fall back to latest quote only if fill unconfirmed
        current_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
        price_source = "fill" if order_result.get("fill_price") else ("quote" if current_price else None)

        # Log trade — reason reflects how this actually fired (see trigger
        # docstring above); it used to always say "approved via Telegram"
        # even for fully autonomous crew.py executions no human ever saw.
        # Every trigger named: the retries used to read "approved via Telegram"
        # too (professor M134). A deferred retry may come from either path.
        trigger_label = _TRIGGER_LABELS.get(trigger, f"executed (trigger={trigger})")
        if not _write_trade_row(
                ticker, early, order_result, filled_qty,
                symbol=ticker,
                side=side,
                quantity=filled_qty,
                price=current_price,
                reason=f"{action} {trigger_label} (conviction={approval['conviction']}/10)",
                price_source=price_source):
            return _outcome(f"{_LEFT_TO_RECONCILE} (trade row recovered)")

        # Order-lifecycle + slippage logging (decision price = price used at sizing time)
        local_order_id = early["id"]   # already marked filled above
        try:
            if not early["id"]:
                local_order_id = db.log_order(
                    ticker=ticker, side=side,
                    order_type="BRACKET" if bracket_used else "MARKET",
                    qty=shares, broker_order_id=order_result.get("order_id"),
                    perm_id=order_result.get("perm_id"),
                    decision_price=price_at_sizing, approval_id=approval["id"],
                    status="submitted", intent=action,
                    submitted_at=submitted_at, submit_price=current_price_pre,
                )
            if not early["id"] and local_order_id > 0 and (
                    order_result.get("fill_price")
                    or (filled_qty > 0 and not bracket_used)):   # bracket qty is assumed
                # Part filled within the wait: the rest may still be working,
                # so leave it 'partial' for reconcile_unconfirmed_orders to
                # pick up the remainder (professor M53). A fill confirmed
                # without a price is still a fill: record it with no price
                # rather than leaving the row 'submitted' (professor run 17).
                partial = not bracket_used and filled_qty < float(shares) - 1e-6
                if not db.update_order_fill(local_order_id, order_result.get("fill_price"),
                                            filled_qty, "partial" if partial else "filled",
                                            filled_at=order_result.get("fill_time"),
                                            filled_at_source=order_result.get("fill_time_source")):
                    # The row stays 'submitted'; say why (a note, not a status).
                    log.warning(f"[{ticker}] fill write failed for order row {local_order_id}")
                    db.set_order_reconcile_note(local_order_id, "fill_write_failed")
            elif local_order_id > 0 and bracket_used:
                db.set_order_reconcile_note(local_order_id, "bracket_fill_assumed")
        except Exception as e:
            log.debug(f"[{ticker}] Order/slippage logging failed (non-critical): {e}")

        # Stop lifecycle on SELL: EXIT removes the stop record entirely;
        # TRIM re-places the stop for the remaining shares.
        if side == "SELL":
            if action == "EXIT":
                if filled_qty < float(shares) - 1e-6:
                    # Part filled: the order stays 'partial' and reconcile
                    # closes the outcome from the cumulative fill once the
                    # rest fills (professor M121), not at this part's price.
                    # The stop record stays until the position is flat: the
                    # unsold shares were left with no stop and no record when
                    # the remainder was cancelled (audit 2026-11-02 #2). The
                    # queued re-protect puts the recorded stop on whatever a
                    # working sell won't sell, and closes the exit when flat.
                    log.info(f"[{ticker}] EXIT part-filled ({filled_qty:g} of {shares}); outcome "
                             f"left open for reconcile, stop record kept until flat")
                    queued = False
                    try:
                        tasks._queue_reprotect(
                            ticker, {"intent": "EXIT",
                                     "id": local_order_id if (local_order_id or 0) > 0 else None},
                            current_price or None)
                        queued = True
                    except Exception as e:
                        log.error(f"[{ticker}] Could not queue the stop for the EXIT remainder: {e}")
                    try:
                        alert_module._send(
                            f"⚠️ EXIT PART-FILLED — {ticker}\n"
                            f"Sold {filled_qty:g} of {shares}. The stop record is kept; "
                            + ("the unsold shares get the recorded stop on the next reconcile "
                               "cycle unless the rest of the sell is still working."
                               if queued else
                               "the stop retry could NOT be queued. Check TWS and protect the "
                               "unsold shares.")
                        )
                    except Exception:
                        pass
                else:
                    db.delete_position_stop(ticker)
                    log.info(f"[{ticker}] Position closed — stop record deleted")
                    try:
                        db.close_trade_outcome(ticker, exit_price=current_price, exit_source="model_exit",
                                              exit_price_source=price_source,
                                              exit_fill_at=order_result.get("fill_time"),
                                              exit_fill_at_source=order_result.get("fill_time_source"))
                    except Exception as e:
                        log.warning(f"[{ticker}] close_trade_outcome failed: {e}")
            else:  # TRIM — re-place stop at same price for remaining quantity
                remaining = _remaining_after_sell(ticker, owned, filled_qty)
                try:
                    stop_rec = db.get_position_stop(ticker, strict=True)
                    rec_read = True
                except Exception as e:
                    log.error(f"[{ticker}] Stop record unreadable after TRIM: {e}")
                    stop_rec, rec_read = None, False
                part_filled = filled_qty < float(shares) - 1e-6
                # Part filled: the rest may still be working, so the stop goes
                # on what it won't sell, placed now (professor 62i).
                stop_qty = (_unworked_remainder(ticker, remaining, float(shares) - filled_qty, sell_floor)
                            if part_filled and stop_rec and remaining else remaining)
                if remaining is None:
                    alert_module._send(_POSITION_MISMATCH_ALERT.format(
                        ticker=ticker, expected=round(owned - filled_qty, 4)))
                elif remaining > 0 and rec_read and not stop_rec:
                    # No record: the stop was cancelled before the sell and
                    # the retry can't price one (audit 11-06 21:36 #4).
                    log.warning(f"[{ticker}] TRIM left {remaining:g} shares with no stop on record")
                    _alert_no_stop_record(ticker, "A TRIM", remaining, sell_floor)
                elif remaining > 0 and not rec_read:
                    log.warning(f"[{ticker}] TRIM: stop record unreadable; stop for the remaining shares queued")
                    queued = False
                    try:
                        tasks._queue_reprotect(
                            ticker, {"intent": "RESTORE",
                                     "id": local_order_id if (local_order_id or 0) > 0 else None,
                                     "floor": sell_floor},
                            current_price or None)
                        queued = True
                    except Exception as e:
                        log.error(f"[{ticker}] Could not queue the stop after the TRIM: {e}")
                    try:
                        alert_module._send(
                            f"⚠️ TRIM STOP RECORD UNREADABLE — {ticker}\n"
                            f"{remaining:g} shares left after the sell. "
                            + ("The stop is retried on the next reconcile cycle." if queued else
                               "The stop retry could NOT be queued. Check TWS and protect the shares.")
                        )
                    except Exception:
                        pass
                elif remaining > 0 and stop_qty <= 0:
                    log.info(f"[{ticker}] TRIM part-filled; the working sell covers the "
                             f"remaining {remaining:g} shares, no stop placed")
                elif remaining > 0 and stop_rec:
                    try:
                        import risk_manager
                        replace_px = risk_manager._floored_level(
                            ticker, float(stop_rec["stop_price"]), sell_floor, current_price,
                            "Stop re-place after TRIM")
                        new_stop = tasks.manage_stop_place(ticker, stop_qty, replace_px,
                                                           market_price=current_price)
                        _record_placed_stop(   # placed: a write failure is not "unprotected" (M227)
                            ticker, "Stop re-place after TRIM", new_stop.get("order_id"),
                            [("order id", lambda: db.set_stop_broker_order_id(ticker, new_stop["order_id"]))]
                            + _raise_record_writes(ticker, replace_px, stop_rec["stop_price"], stop_rec))
                        log.info(f"[{ticker}] Stop re-placed for {stop_qty:g} of the remaining {remaining:g} shares")
                    except Exception as e:
                        log.error(f"[{ticker}] Failed to re-place stop after TRIM: {e}")
                        queued = False
                        try:
                            tasks._queue_reprotect(ticker, {"intent": "RESTORE", "floor": sell_floor},
                                                   None)  # m39
                            queued = True
                        except Exception as e2:
                            log.error(f"[{ticker}] Could not queue the stop restore: {e2}")
                        alert_module._send(
                            f"🚨 UNPROTECTED POSITION — {ticker}\n"
                            f"TRIM filled but stop re-placement failed. "
                            + ("It is queued and retried every reconcile cycle; " if queued else "")
                            + f"place stop @ ${stop_rec['stop_price']:.2f} for {remaining} shares "
                            f"manually if it isn't back within ~15 min."
                        )
                        tasks.open_manual_action(ticker, f"re-place the stop @ "
                                                         f"${float(stop_rec['stop_price']):.2f} after a TRIM")
                elif remaining <= 0:
                    # A TRIM that sold the last share closes the position:
                    # close its outcome row too, or it stays open and a later
                    # EXIT closes it at that later price (professor C10).
                    db.delete_position_stop(ticker)
                    try:
                        db.close_trade_outcome(ticker, exit_price=current_price, exit_source="model_trim_last",
                                              exit_price_source=price_source,
                                              exit_fill_at=order_result.get("fill_time"),
                                              exit_fill_at_source=order_result.get("fill_time_source"))
                    except Exception as e:
                        log.warning(f"[{ticker}] close_trade_outcome failed: {e}")

        # Log stop level and place stop order with IBKR.
        # Skipped when a bracket order already placed the stop atomically —
        # re-placing it here would create a second, redundant stop.
        stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
        if stop_price and side == "BUY" and bracket_used:
            _note_stale_record(ticker, filled_qty, action, path="bracket")
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
                entry_price_source=price_source,
            )
            db.set_stop_broker_order_id(ticker, bracket_result["stop_order_id"])
            _capture_original_thesis_if_new(ticker, approval)
            log.info(f"[{ticker}] Bracket stop recorded (orderId={bracket_result['stop_order_id']})")
        elif stop_price and side == "BUY":
            stop_price = _stop_anchored_to_fill(ticker, float(stop_price), price_at_sizing, current_price)
            entry_for_stops, stop_qty = current_price, filled_qty
            prior_info: dict = {}
            stop_price, add_to_existing = _whole_position_stop(ticker, float(stop_price), filled_qty,
                                                               status=prior_info,
                                                               market_price=current_price)
            entry_source = price_source
            if add_to_existing:
                entry_for_stops, stop_qty = add_to_existing
                entry_source = "broker_avg"
            # Prior record unreadable on a possible ADD: keep whatever record
            # exists. The new shares' level written over it would let a later
            # trail or ladder ratchet re-stop the whole position from it, below
            # the old shares' stop.
            prior_unread = bool(prior_info.get("prior_unreadable"))
            record_ok = prior_unread or _write_stop_record(
                ticker,
                entry_price=entry_for_stops,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
                entry_price_source=entry_source,
            )
            _capture_original_thesis_if_new(ticker, approval)
            stop_placed = False
            stop_outcome = "failed"   # for the record-missing alert
            for stop_attempt in range(1, 4):
                try:
                    stop_result = tasks.manage_stop_place(ticker, stop_qty, float(stop_price),
                                                         market_price=current_price)
                    _set_stop_order_id(ticker, stop_result["order_id"])
                    stop_placed, stop_outcome = True, "placed"
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] Stop order attempt {stop_attempt}/3 failed: {e}")
                    if type(e).__name__ == "SellExceedsPosition":
                        # The broker shows fewer shares than the stop is for.
                        # Right after a fill that is usually position lag, so
                        # hand it to the retry queue (re-reads every cycle; a
                        # real short just fails again) rather than drop it or
                        # retry inline (professor run 11 follow-up).
                        _alert_stop_refused(ticker, stop_qty, e)
                        _queue_stop_for_retry(ticker, stop_qty, float(stop_price), alert=False)
                        stop_placed, stop_outcome = True, "queued"
                        break
                    if isinstance(e, tasks.OrdersCallUnknown):
                        # The stop may be resting already; a retry could stack
                        # a second one. Reconcile fixes coverage either way
                        # within two cycles (professor C24).
                        _alert_stop_unknown(ticker, stop_qty, float(stop_price))
                        stop_placed, stop_outcome = True, "unknown"  # not "failed": alerted above
                        break
                    if isinstance(e, tasks.OrdersCallExpired) or type(e).__name__ == "LockLost":
                        # The orders worker is busy or stuck, or our ticker lock
                        # lapsed (another process may be changing stops);
                        # retrying here only repeats that. The queue re-reads
                        # coverage under a fresh lock (A0453-3, m39).
                        _queue_stop_for_retry(ticker, stop_qty, float(stop_price))
                        stop_placed, stop_outcome = True, "queued"  # handed to the retry queue
                        break
                    if stop_attempt < 3:
                        import time as _time
                        _time.sleep(2 * stop_attempt)
            if prior_unread:
                try:   # one read, after placement: a record, none, or unknown
                    prior_now = db.get_position_stop(ticker, strict=True)
                except Exception:
                    prior_now = "unknown"
                if prior_now is None:
                    # No record: a fresh entry after all (or an add with
                    # none): the healthy path's write, now after placement.
                    record_ok = _write_stop_record(
                        ticker,
                        entry_price=entry_for_stops,
                        stop_price=float(stop_price),
                        stop_method=risk_json.get("stop_method", ""),
                        bucket=approval.get("thesis_bucket", "Accelerator"),
                        conviction=approval.get("conviction", 5),
                        trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
                        entry_price_source=entry_source,
                    )
                    if not record_ok:
                        _alert_stop_record_missing(ticker, stop_qty, float(stop_price),
                                                   stop_outcome)
                else:
                    _alert_prior_stop_unread(ticker, stop_qty, float(stop_price), stop_outcome,
                                             prior_now)
            elif not record_ok:
                _alert_stop_record_missing(
                    ticker, stop_qty, float(stop_price), stop_outcome,
                    stale_stop=prior_info.get("prior_stop") if add_to_existing else None)
            if not stop_placed:
                log.error(f"[{ticker}] CRITICAL — stop order failed after 3 attempts. Position is UNPROTECTED.")
                queued = False
                try:
                    tasks._queue_reprotect(ticker, {"intent": "BUY"}, None)  # m39
                    queued = True
                except Exception as e:
                    log.error(f"[{ticker}] Could not queue the stop retry: {e}")
                try:
                    alert_module._send(
                        f"🚨 UNPROTECTED POSITION — ACTION REQUIRED\n"
                        f"<b>{ticker}</b>: entry order was filled but stop order failed after 3 attempts.\n"
                        f"Stop target: <b>${float(stop_price):.2f}</b> for {stop_qty} shares.\n"
                        + ("It is queued and retried every reconcile cycle, but " if queued else "")
                        + "place the stop manually in TWS now."
                    )
                    tasks.open_manual_action(ticker, f"place a stop @ ${float(stop_price):.2f}")
                except Exception:
                    pass

        # Log outcome record for feedback loop — sub-scores live in
        # analyst_json (the full analyst.analyze() result), not as their own
        # pending_approvals columns, since only conviction/thesis_bucket are
        # promoted to top-level columns there.
        analyst_json = approval.get("analyst_json") or {}
        # BUYs only (professor C5, 2026-10-27). A SELL used to add an EXIT
        # outcome row right after close_trade_outcome closed the BUY's row,
        # and nothing ever closed it.
        if side == "BUY":
            # A failed write is queued for reconcile to retry (professor M125).
            try:
                from tasks import _apply_buy_outcome
                _apply_buy_outcome(approval["id"], {"kind": "insert", "ticker": ticker, "action": action,
                                                    "entry_price": current_price}, approval)
            except Exception as e:
                log.error(f"[{ticker}] outcome row write/queue failed: {e}")
        _stop_work_done(ticker, early)

        # Send trade alert — full rationale + all conviction scores + bucket,
        # not just the bare fill details, per explicit user request.
        alert_module.send_trade_alert(
            symbol=ticker,
            side=side,
            qty=filled_qty,
            price=current_price,
            reason=f"{action} {trigger_label}",
            conviction=approval.get("conviction"),
            thesis_bucket=approval.get("thesis_bucket"),
            reasoning=approval.get("reasoning") or analyst_json.get("reasoning"),
            thesis_strength=analyst_json.get("thesis_strength"),
            catalyst_quality=analyst_json.get("catalyst_quality"),
            valuation_risk_reward=analyst_json.get("valuation_risk_reward"),
            data_completeness=analyst_json.get("data_completeness"),
            chase_risk=analyst_json.get("chase_risk"),
            price_action=analyst_json.get("price_action"),
        )

        return True

    except Exception as e:
        log.error(f"[{ticker}] Order execution failed: {e}")
        return _outcome("failed: exception during execution")


def _stop_record_or_none(ticker: str) -> dict | None:
    """The position's stop record, or None if it can't be read (a manual
    ADD then falls back to the normal BUY stop path rather than failing
    after the fill)."""
    import db
    try:
        return db.get_position_stop(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] could not read the stop record: {e}")
        return None


# ---------------------------------------------------------------------------
# Manual / !urgent execution (dashboard form, chat !urgent command)
# ---------------------------------------------------------------------------

@locked_by_ticker(lambda ticker, *a, **k: ticker, on_busy={
    "success": False, "filled_qty": 0.0, "fill_price": 0.0, "stop_price": None,
    "message": "Another stop change is in progress for this ticker; try again shortly."})
def execute_manual(
    ticker: str,
    side: str,
    qty: float,
    reason: str,
    bucket: str = "Accelerator",
    conviction: int = 7,
) -> dict:
    """
    Executes a manually-initiated BUY or SELL (dashboard form, !urgent chat
    command) with the same safety properties as the approval path:
    phantom-fill-safe, SELL is clamped to owned shares and cancels existing
    stops first, BUY gets an ATR-based stop placed and recorded in
    position_stops (previously the dashboard's manual BUY skipped this
    entirely, leaving the position invisible to the trailing-stop monitor).

    Returns {"success": bool, "filled_qty": float, "fill_price": float,
    "message": str, "stop_price": float | None}.
    """
    import ibkr
    import tasks
    import db
    import alerts as alert_module
    from risk_manager import RiskManager

    try:
        rm = RiskManager(ibkr.get_client())
    except Exception as e:
        return {"success": False, "filled_qty": 0, "fill_price": None,
                "stop_price": None, "message": f"RiskManager init failed: {e}"}

    if side == "BUY" and rm.is_halted():
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": "Blocked — portfolio drawdown halt is active. No new buys."}

    sell_floor = None   # set before the SELL's stop cancel (professor M228)
    if side == "SELL":
        try:
            owned = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
        except Exception:
            owned = 0.0
        if owned <= 0:
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"No position in {ticker} — nothing to sell."}
        qty = min(qty, owned)
        from market_data import is_market_hours
        if not is_market_hours():
            # Would be cancelled unfilled after the stop was removed (A2053-3).
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"Market is closed: SELL {ticker} not sent, stop left in place. "
                               f"Send it again during market hours."}
        pre_cancel_prices: list = []   # filled by the read below (professor M228)
        if _stops_still_resting(ticker, prices_out=pre_cancel_prices) is None:
            # Unreadable before the cancel: hold with the stop in place (M131).
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"SELL {ticker} not sent: the broker's resting stops could not be "
                               f"read, so the stop was left in place. Check TWS."}
        # Highest resting stop, read before the cancel (professor M228).
        sell_floor = _floor_before_cancel(ticker, pre_cancel_prices, "Manual SELL stop")
        # Cancel existing stops first so the fill can't race the stop and
        # leave an order that would short the account.
        try:
            cancelled = tasks.manage_cancel_stop_orders_for(ticker)
            if cancelled:
                log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before manual SELL")
        except Exception as e:
            log.warning(f"[{ticker}] Could not cancel existing stops before manual SELL: {e}")
        resting = _stops_still_resting(ticker)
        if resting or resting is None:
            _restore_stop_after_failed_sell(ticker, floor=sell_floor)  # A0053-2
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"SELL {ticker} not sent: " + (
                        "the broker's resting stops could not be read" if resting is None
                        else "an existing stop could not be cancelled")
                        + " (selling on top of a stop could short). Check TWS."}

    order_intent = f"MANUAL_{side}"
    held_before = 0.0
    quote_price = None   # the BUY room check's quote; orders.submit_price
    if side == "BUY":
        # Read before sending, so a late fill knows whether it opened or added
        # (a read right after a fill can be stale; A0253-4).
        try:
            held_before = abs(float(ibkr.get_client().get_open_position(ticker).qty))
            order_intent = "MANUAL_ADD" if held_before > 0 else "MANUAL_OPEN"
        except Exception as e:
            order_intent = "MANUAL_OPEN" if str(e).startswith("No open position") else "MANUAL_BUY"
        try:
            import market_data
            q = market_data.get_priced_quote(ticker)
            quote_price = float(q["price"]) if q and q.get("price") else None
        except Exception:
            quote_price = None
        block = _manual_buy_room(ticker, qty, quote_price)
        if block:
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"BUY {ticker} not sent: {block}."}

    try:
        import stop_lock
        stop_lock.assert_held(ticker)  # professor M69
    except Exception as e:
        if side == "SELL":
            _restore_stop_after_failed_sell(ticker, floor=sell_floor)
        else:
            _release_buy(ticker)   # nothing was sent
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submission failed: {e}"}
    submitted_at = _now_utc()   # record-keeping only (orders.submitted_at)
    on_submitted, early = _early_order_row(
        ticker=ticker, side=side, order_type="MARKET", qty=qty, intent=order_intent,
        submitted_at=submitted_at, submit_price=quote_price,
    )
    try:
        order_result = ibkr.place_market_order(ticker, side, qty, on_submitted=on_submitted)
    except Exception as e:
        if side == "SELL":
            # Same as execute_approved: the stop was cancelled above, so put
            # it back (limitations audit 2026-10-28, finding 9).
            _restore_stop_after_failed_sell(ticker, floor=sell_floor)
        # A BUY keeps its reservation: the order may have gone out.
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submission failed: {e}"}
    _recover_early_row(ticker, side, early, order_result)

    filled_qty = float(order_result.get("filled_qty") or 0)
    if filled_qty <= 0 and side == "BUY" and order_result.get("status") in _DEAD_ORDER_STATUSES:
        _release_buy(ticker)   # rejected/cancelled: no late fill can come
    if filled_qty <= 0 and side == "SELL" and order_result.get("status") in _DEAD_ORDER_STATUSES:
        _restore_stop_after_failed_sell(ticker, floor=sell_floor)
        if early["id"]:
            _finish_dead_sell_row(ticker, early["id"], order_result)
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"SELL {ticker} came back {order_result.get('status')} unfilled; "
                           f"the stop was restored."}
    if filled_qty <= 0:
        # No confirmed fill — never fabricate one; a phantom trade row plus
        # a stop for shares that may not exist would diverge DB state from
        # the broker.
        log.error(f"[{ticker}] No fill confirmed for manual {side} {qty} "
                  f"(id={order_result.get('order_id')}) — no trade recorded")
        # An orders row lets reconcile_unconfirmed_orders book a late fill
        # and protect it; without one a late manual fill stayed invisible
        # (professor M108). No decision price: slippage stats stay
        # approval-path only.
        _log_unconfirmed_order(order_row_id=early["id"],
                               ticker=ticker, side=side, order_type="MARKET", qty=qty,
                               broker_order_id=order_result.get("order_id"),
                               perm_id=order_result.get("perm_id"),
                               status="unconfirmed", intent=order_intent,
                               submitted_at=submitted_at, submit_price=quote_price)
        stop_note = (" The broker stop was cancelled before the sell and NOT re-placed "
                     "(a late fill plus a full-size stop could short); check it."
                     if side == "SELL" else "")
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submitted (id {order_result.get('order_id')}) but no fill "
                           f"confirmed within the wait window. No trade recorded; a late fill "
                           f"is reconciled automatically — check TWS.{stop_note}"}

    if early["id"]:
        # The order row first, then the trade row (see _mark_early_row_filled).
        if not _mark_early_row_filled(ticker, early, order_result, filled_qty, qty):
            return {"success": False, "filled_qty": filled_qty,
                    "fill_price": order_result.get("fill_price"), "stop_price": None,
                    "message": f"{side} {ticker} filled {filled_qty:g}, but reconcile had already "
                               f"taken over its order row, so reconcile records the fill and the "
                               f"stop. Nothing was recorded here; check TWS."}
    fill_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
    if not _write_trade_row(ticker, early, order_result, filled_qty,
                            symbol=ticker, side=side, quantity=filled_qty, price=fill_price,
                            reason=reason,
                            price_source="fill" if order_result.get("fill_price") else
                            ("quote" if fill_price else None)):
        return {"success": False, "filled_qty": filled_qty,
                "fill_price": order_result.get("fill_price"), "stop_price": None,
                "message": f"{side} {ticker} filled {filled_qty:g}, but reconcile had already "
                           f"recorded the fill and the stop. Nothing was recorded here; check TWS."}
    local_order_id = None   # the stop retry below reads it even if the order row failed
    try:
        local_order_id = early["id"] or db.log_order(
            ticker=ticker, side=side, order_type="MARKET", qty=qty,
            broker_order_id=order_result.get("order_id"), perm_id=order_result.get("perm_id"),
            status="submitted", intent=order_intent,
            submitted_at=submitted_at, submit_price=quote_price)
        if not early["id"] and local_order_id > 0 and filled_qty > 0:
            # A part fill stays 'partial' so reconcile books the rest (M53);
            # a fill confirmed without a price is recorded with none.
            if not db.update_order_fill(local_order_id, order_result.get("fill_price"), filled_qty,
                                        "partial" if filled_qty < float(qty) - 1e-6 else "filled",
                                        filled_at=order_result.get("fill_time"),
                                        filled_at_source=order_result.get("fill_time_source")):
                log.warning(f"[{ticker}] fill write failed for order row {local_order_id}")
                db.set_order_reconcile_note(local_order_id, "fill_write_failed")
    except Exception as e:
        log.debug(f"[{ticker}] Manual order logging failed (non-critical): {e}")

    stop_price = None
    if side == "SELL":
        # Keep DB stop records in sync with what remains at the broker
        remaining = _remaining_after_sell(ticker, owned, filled_qty)
        if remaining is None:
            import alerts
            alerts._send(_POSITION_MISMATCH_ALERT.format(
                ticker=ticker, expected=round(owned - filled_qty, 4)))
        elif remaining <= 0:
            db.delete_position_stop(ticker)
            try:
                db.close_trade_outcome(ticker, exit_price=fill_price, exit_source="manual_sell",
                                       exit_price_source="fill" if order_result.get("fill_price") else "quote",
                                       exit_fill_at=order_result.get("fill_time"),
                                       exit_fill_at_source=order_result.get("fill_time_source"))
            except Exception:
                pass  # no open outcome row is expected for positions not opened via approval
        else:
            try:
                stop_rec = db.get_position_stop(ticker, strict=True)
                rec_read = True
            except Exception as e:
                log.error(f"[{ticker}] Stop record unreadable after a manual sell: {e}")
                stop_rec, rec_read = None, False
            part_filled = filled_qty < float(qty) - 1e-6
            stop_qty = (_unworked_remainder(ticker, remaining, float(qty) - filled_qty, sell_floor)
                        if part_filled and stop_rec else remaining)
            if rec_read and not stop_rec:
                # No record (audit 11-06 21:36 #4): nothing protects the rest
                # and the retry can't price a stop.
                log.warning(f"[{ticker}] Manual sell left {remaining:g} shares with no stop on record")
                _alert_no_stop_record(ticker, "A manual sell", remaining, sell_floor)
            elif not rec_read:
                log.warning(f"[{ticker}] Manual sell: stop record unreadable; stop for the remaining shares queued")
                queued = False
                try:
                    tasks._queue_reprotect(
                        ticker, {"intent": "RESTORE",
                                 "id": local_order_id if (local_order_id or 0) > 0 else None,
                                 "floor": sell_floor},
                        fill_price or None)
                    queued = True
                except Exception as e:
                    log.error(f"[{ticker}] Could not queue the stop after the manual sell: {e}")
                try:
                    alert_module._send(
                        f"⚠️ MANUAL SELL STOP RECORD UNREADABLE — {ticker}\n"
                        f"{remaining:g} shares left. "
                        + ("The stop is retried on the next reconcile cycle." if queued else
                           "The stop retry could NOT be queued. Check TWS and protect the shares."))
                except Exception:
                    pass
            elif stop_qty <= 0:
                log.info(f"[{ticker}] Manual sell part-filled; the working sell covers the "
                         f"remaining {remaining:g} shares, no stop placed")
            elif stop_rec:
                try:
                    import risk_manager
                    replace_px = risk_manager._floored_level(
                        ticker, float(stop_rec["stop_price"]), sell_floor, fill_price,
                        "Stop re-place after a manual sell")
                    new_stop = tasks.manage_stop_place(ticker, stop_qty, replace_px,
                                                       market_price=fill_price)
                    _record_placed_stop(   # placed: a write failure is not "unprotected" (M227)
                        ticker, "Stop re-place after a manual sell", new_stop.get("order_id"),
                        [("order id", lambda: db.set_stop_broker_order_id(ticker, new_stop["order_id"]))]
                        + _raise_record_writes(ticker, replace_px, stop_rec["stop_price"], stop_rec))
                except Exception as e:
                    log.error(f"[{ticker}] Failed to re-place stop after manual sell: {e}")
                    # Queue and track it like the other failed re-places (M87).
                    try:
                        tasks._queue_reprotect(ticker, {"intent": "RESTORE", "floor": sell_floor}, None)
                    except Exception as e2:
                        log.error(f"[{ticker}] Could not queue the stop restore: {e2}")
                    try:
                        alert_module._send(
                            f"🚨 UNPROTECTED POSITION — {ticker}\nA manual sell filled but the stop "
                            f"for the remaining {remaining:g} shares failed. It is queued and retried "
                            f"every reconcile cycle; place stop @ ${float(stop_rec['stop_price']):.2f} "
                            f"manually if it isn't back within ~15 min.")
                    except Exception:
                        pass
                    tasks.open_manual_action(ticker, f"re-place the stop @ "
                                                     f"${float(stop_rec['stop_price']):.2f} after a manual sell")
    elif order_intent == "MANUAL_ADD" and (add_rec := _stop_record_or_none(ticker)):
        # Added to a held position: keep its stop record (a fresh ATR stop
        # from this fill could loosen it) and protect the whole position at
        # that level, not just the new shares (professor M115).
        stop_price = float(add_rec["stop_price"])
        placed = None
        try:
            placed = tasks._reprotect_after_late_sell(ticker, {}, None, from_queue=True,
                                                      expect_at_least=held_before + filled_qty)
        except Exception as e:
            log.error(f"[{ticker}] Re-protecting after a manual ADD failed: {e}")
        if placed is not True:
            try:
                tasks._queue_reprotect(ticker, {"intent": "BUY"}, None)
            except Exception as e:
                log.error(f"[{ticker}] Could not queue the stop retry: {e}")
            # The old stop may already be cancelled, or not cover the new
            # shares: treat the position as unprotected until a retry lands.
            try:
                alert_module._send(
                    f"🚨 UNPROTECTED POSITION — {ticker}\nManual ADD filled, but the stop for the "
                    f"whole position (@ ${stop_price:.2f}) isn't confirmed. It is queued and retried "
                    f"every reconcile cycle; place it manually if it isn't back within ~15 min.")
            except Exception:
                pass
            tasks.open_manual_action(ticker, f"confirm the stop @ ${stop_price:.2f} covers the whole "
                                             f"position after a manual ADD")
    else:  # BUY — place an ATR stop and make the position visible to the monitor
        # The read before the order decides here; the read after a fill can
        # lag. MANUAL_ADD lands here when the ADD branch found no record.
        # MANUAL_BUY: the read before the order failed (professor 58j).
        pre_order = {"MANUAL_OPEN": "OPEN", "MANUAL_ADD": "ADD"}.get(order_intent)
        _note_stale_record(ticker, filled_qty, pre_order, path="manual",
                           decision_basis="pre_order_read" if pre_order else None)
        try:
            stop_price, stop_method = rm.calculate_stop(ticker, fill_price, bucket,
                                                        log_caller="post_fill")
        except Exception as e:
            log.warning(f"[{ticker}] calculate_stop failed: {e}")
            stop_price, stop_method = None, ""
        import math
        try:
            fill_ok = float(fill_price) > 0 and math.isfinite(float(fill_price))
        except (TypeError, ValueError):
            fill_ok = False
        if not stop_price and fill_ok:
            # Never leave a manual buy without a stop (professor M107): fall
            # back to a flat stop below the fill and say so, never above the
            # last quote (a stale fill price could otherwise put it there).
            stop_price = round(float(fill_price) * (1 - _MANUAL_FALLBACK_STOP_PCT), 2)
            if quote_price and stop_price >= quote_price:
                stop_price = round(quote_price * (1 - _MANUAL_FALLBACK_STOP_PCT), 2)
            stop_method = f"fallback flat {_MANUAL_FALLBACK_STOP_PCT:.0%} (ATR stop failed)"
            log.error(f"[{ticker}] ATR stop unavailable for a manual BUY; using {stop_method} @ ${stop_price:.2f}")
            try:
                alert_module._send(f"⚠️ {ticker}: manual BUY filled but the ATR stop couldn't be computed; "
                                   f"placing a fallback stop @ ${stop_price:.2f}. Review it.")
            except Exception:
                pass
        if not stop_price:
            log.error(f"[{ticker}] Manual BUY filled with no fill price and no stop")
            try:
                alert_module._send(f"🚨 UNPROTECTED POSITION — {ticker}\nManual BUY filled, but no stop "
                                   f"could be computed (no fill price). Place a stop manually now.")
            except Exception:
                pass
            tasks.open_manual_action(ticker, "place a stop after a manual BUY (no stop could be computed)")
        if stop_price:
            record_ok = _write_stop_record(
                ticker, entry_price=fill_price,
                stop_price=float(stop_price), stop_method=stop_method or "manual ATR stop",
                bucket=bucket, conviction=conviction, trail_distance_pct=None,
                entry_price_source="fill" if order_result.get("fill_price") else "quote",
            )
            stop_placed = False
            stop_outcome = "failed"   # for the record-missing alert
            for attempt in range(1, 4):
                try:
                    stop_result = tasks.manage_stop_place(ticker, filled_qty, float(stop_price),
                                                         market_price=fill_price)
                    _set_stop_order_id(ticker, stop_result["order_id"])
                    stop_placed, stop_outcome = True, "placed"
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] Stop attempt {attempt}/3: {e}")
                    if type(e).__name__ == "SellExceedsPosition":
                        _alert_stop_refused(ticker, filled_qty, e)
                        _queue_stop_for_retry(ticker, filled_qty, float(stop_price), alert=False)
                        stop_placed, stop_outcome = True, "queued"
                        break
                    if isinstance(e, tasks.OrdersCallUnknown):
                        _alert_stop_unknown(ticker, filled_qty, float(stop_price))  # C24
                        stop_placed, stop_outcome = True, "unknown"
                        break
                    if isinstance(e, tasks.OrdersCallExpired):
                        _queue_stop_for_retry(ticker, filled_qty, float(stop_price))  # A0453-3
                        stop_placed, stop_outcome = True, "queued"
                        break
                    if attempt < 3:
                        import time as _time
                        _time.sleep(2 * attempt)
            if not record_ok:
                _alert_stop_record_missing(ticker, filled_qty, float(stop_price), stop_outcome)
            if not stop_placed:
                try:
                    tasks._queue_reprotect(ticker, {"intent": "BUY"}, None)  # M87
                except Exception as e:
                    log.error(f"[{ticker}] Could not queue the stop retry: {e}")
                try:
                    alert_module._send(
                        f"🚨 UNPROTECTED POSITION — {ticker}\n"
                        f"Manual order filled but stop failed. It is queued and retried every "
                        f"reconcile cycle, but place stop @ ${stop_price:.2f} manually now."
                    )
                except Exception:
                    pass
                tasks.open_manual_action(ticker, f"place a stop @ ${float(stop_price):.2f} after a manual BUY")

    _stop_work_done(ticker, early)
    return {
        "success": True,
        "filled_qty": filled_qty,
        "fill_price": fill_price,
        "stop_price": float(stop_price) if stop_price else None,
        "message": f"{side} {filled_qty:g} {ticker} @ ~${fill_price:.2f}"
                   + (f", stop @ ${stop_price:.2f}" if stop_price else ""),
    }
