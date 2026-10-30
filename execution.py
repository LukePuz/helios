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


def _stops_still_resting(ticker: str) -> float | None:
    """Total quantity of SELL stops still working at the broker for ticker,
    or None if it can't be read. A sell must not go out on top of a resting
    stop: after an EXIT that stop would short the account (limitations audit
    22:53, A2253-1)."""
    import tasks
    try:
        return sum(float(o["qty"]) for o in tasks._resting_sell_stops(ticker))
    except Exception as e:
        log.warning(f"[{ticker}] could not re-read resting stops: {e}")
        return None


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


# Order states after which nothing can fill (ib_async OrderStatus).
_DEAD_ORDER_STATUSES = ("Cancelled", "ApiCancelled", "Inactive")


def _restore_stop_after_failed_sell(ticker: str) -> None:
    """A TRIM/EXIT cancels the broker stop before its market SELL. If the
    order then fails to go in, put the stop back for what's held (limitations
    audit 2026-10-27, finding 8). Before this the position was left with no
    broker stop until someone noticed."""
    import alerts as alert_module
    import db
    import ibkr
    import tasks
    stop_rec = db.get_position_stop(ticker)
    if not stop_rec or not stop_rec.get("stop_price"):
        return
    try:
        held = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
        # Only the part not already covered by a stop that never got
        # cancelled, or the two together would oversell (A2253-1).
        covered = _stops_still_resting(ticker)
        if covered is None:
            # Unknown coverage: a full stop on top of one that is still
            # working would oversell, so place nothing and say so (A0053-2).
            log.error(f"[{ticker}] Stop restore skipped: resting stops could not be read")
            try:
                alert_module._send(
                    f"⚠️ CHECK STOPS — {ticker}\n"
                    f"A sell did not go through and the resting stops could not be read, "
                    f"so no stop was restored. Check TWS; recorded stop "
                    f"${float(stop_rec['stop_price']):.2f}."
                )
            except Exception:
                pass
            return
        qty = round(held - covered, 4)
        if qty <= 0:
            return
        result = tasks.manage_stop_place(ticker, qty, float(stop_rec["stop_price"]))
        db.set_stop_broker_order_id(ticker, result["order_id"])
        log.info(f"[{ticker}] Sell failed; stop restored @ ${float(stop_rec['stop_price']):.2f} for {qty} shares")
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


def _whole_position_stop(ticker: str, new_stop: float, filled_qty: float):
    """Stop handling when a BUY adds to a position that already has a stop.

    Before 2026-10-27 an ADD overwrote entry_price with the add's fill price
    and placed a broker stop for only the added shares, leaving the first
    tranche on an older stop the DB no longer knew about. Now: one stop for
    the whole position, at the higher of the old and new stop (never
    loosened), with the broker's average cost as the entry price.

    Returns (stop_price, (avg_cost, total_qty)) for an add, or
    (new_stop, None) for a fresh position or when the broker lookup fails
    (then the old behaviour applies: a stop for the filled shares only)."""
    import db
    import ibkr
    import tasks
    prior = db.get_position_stop(ticker)
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
        return new_stop, None  # nothing was held before this fill
    stop = max(new_stop, float(prior["stop_price"]))
    try:
        tasks.manage_cancel_stop_orders_for(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] ADD: could not cancel the old stop, stopping only the added shares: {e}")
        return new_stop, None
    log.info(f"[{ticker}] ADD: one stop for all {total_qty} shares @ ${stop:.2f} "
             f"(old ${float(prior['stop_price']):.2f}, new ${new_stop:.2f}); avg cost ${avg_cost:.2f}")
    return stop, (avg_cost, total_qty)


def log_buy_outcome(approval: dict, ticker: str, action: str, entry_price: float) -> None:
    """The trade_outcomes row for a filled BUY, with the analyst's scores from
    the approval. Called at the fill and by reconcile for a BUY that filled
    after the wait window (professor M123)."""
    import db
    analyst_json = approval.get("analyst_json") or {}
    db.log_trade_outcome(
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
    if not (sized_at and fill and 0 < stop < sized_at and fill < sized_at):
        return stop
    if stop < fill * (1 - _STOP_MIN_GAP_BELOW_FILL):
        return stop
    anchored = round(min(stop, fill * stop / sized_at), 2)
    log.warning(f"[{ticker}] Stop moved from ${stop:.2f} to ${anchored:.2f}: filled at "
                f"${fill:.2f}, below the ${sized_at:.2f} sizing price")
    return anchored


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

    trigger distinguishes how this fired, purely for the trades.reason
    audit trail: "auto" for the crew.py pipeline (no human ever saw this),
    "telegram" for an actual Approve-button tap. Everything else about
    execution is identical regardless of trigger.

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
                    log.error(f"[{ticker}] Order blocked — portfolio drawdown halt active")
                    alert_module._send(
                        f"🛑 ORDER BLOCKED — Drawdown Halt\n"
                        f"<b>{ticker}</b> {action} order was approved but cannot execute.\n"
                        f"Portfolio drawdown limit is active. No new BUY entries allowed.\n"
                        f"Review and reset the halt flag when drawdown is resolved."
                    )
                    return _outcome("blocked: drawdown halt active")
            except Exception as halt_err:
                log.warning(f"[{ticker}] Drawdown halt check failed: {halt_err}")

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
                    try:
                        from tasks import retry_deferred_entry
                        retry_deferred_entry.apply_async(
                            args=[approval["id"]], eta=eta, queue="ops",
                        )
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
                    pq = get_priced_quote(ticker)
                    if pq and not pq["is_stale"]:
                        break
            if pq:
                current_price_pre = pq["price"]
                if pq["is_stale"]:
                    log.warning(
                        f"[{ticker}] Price is STALE (age={pq['age_minutes']}min, "
                        f"market_open={pq['market_open']}) — drift check unreliable"
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
                        if price_retry_attempt < STALE_PRICE_MAX_RETRIES:
                            from datetime import datetime, timedelta, timezone
                            eta = datetime.now(timezone.utc) + timedelta(minutes=STALE_PRICE_RETRY_MINUTES)
                            log.warning(
                                f"[{ticker}] Could not get a fresh price "
                                f"(attempt {price_retry_attempt}/{STALE_PRICE_MAX_RETRIES}) — "
                                f"re-checking at {eta.isoformat()} rather than risk an unguarded gap"
                            )
                            try:
                                from tasks import retry_deferred_entry
                                retry_deferred_entry.apply_async(
                                    args=[approval["id"], gap_retry_attempt, price_retry_attempt + 1],
                                    eta=eta, queue="ops",
                                )
                            except Exception as e:
                                log.error(f"[{ticker}] Could not schedule stale-price re-check: {e}")
                            try:
                                alert_module._send(
                                    f"⏸️ WAITING FOR FRESH PRICE — {ticker}\n"
                                    f"Only a stale price (${current_price_pre:.2f}) was available "
                                    f"(check {price_retry_attempt}/{STALE_PRICE_MAX_RETRIES}) — IBKR's "
                                    f"intraday feed usually catches up within a few minutes of the open. "
                                    f"Will re-check in {STALE_PRICE_RETRY_MINUTES} min."
                                )
                            except Exception:
                                pass
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
                    try:
                        from tasks import retry_deferred_entry
                        retry_deferred_entry.apply_async(
                            args=[approval["id"], gap_retry_attempt + 1], eta=eta, queue="ops",
                        )
                    except Exception as e:
                        log.error(f"[{ticker}] Could not schedule gap re-check: {e}")
                    alert_module._send(
                        f"⏸️ WAITING FOR CONFIRMATION — {ticker}\n"
                        f"Gapped {gap['gap_pct']:+.1%} from prior close — waiting to see if the "
                        f"move holds before entering (check {gap_retry_attempt}/{GAP_CONFIRMATION_MAX_RETRIES}). "
                        f"Will re-check in {GAP_CONFIRMATION_RETRY_MINUTES} min."
                    )
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

        if side == "SELL":
            # Cancel the existing broker stop only now that the sell is
            # definitely going ahead, so the fill can't race the stop and
            # leave an order that would short the account.
            cancel_ok = True
            try:
                cancelled = tasks.manage_cancel_stop_orders_for(ticker)
                if cancelled:
                    log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before {action}")
            except Exception as e:
                cancel_ok = False
                log.warning(f"[{ticker}] Could not cancel existing stops before {action}: {e}")
            resting = _stops_still_resting(ticker)
            if resting or (resting is None and not cancel_ok):
                log.error(f"[{ticker}] {action} not sent: stop(s) for {resting} shares still resting")
                try:
                    alert_module._send(
                        f"⚠️ {action} NOT SENT — <b>{ticker}</b>\n"
                        f"An existing stop could not be cancelled, and selling on top of it "
                        f"could short the account. Check the stop orders in TWS."
                    )
                except Exception:
                    pass
                # The cancel may have taken some stops before one stuck;
                # put back what is now uncovered (A0053-2).
                _restore_stop_after_failed_sell(ticker)
                return _outcome("skipped: stop still resting, sell not sent")

        stop_price_for_entry = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")

        try:
            from config import USE_BRACKET_ORDERS
        except Exception:
            USE_BRACKET_ORDERS = False

        bracket_used = False
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
                    _restore_stop_after_failed_sell(ticker)
                else:
                    _release_buy(ticker)   # nothing was sent
                raise
            try:
                order_result = ibkr.place_market_order(ticker, side, shares)
            except Exception:
                # A buy's reservation stays until it expires: the order may
                # have been sent before the error and can still fill (review 4 #6).
                if side == "SELL":
                    _restore_stop_after_failed_sell(ticker)
                raise
            log.info(f"[{ticker}] Order submitted: {side} {shares} shares (id={order_result['order_id']})")

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
                _restore_stop_after_failed_sell(ticker)
                return _outcome(f"skipped: sell {order_result.get('status')} unfilled, stop restored")
            if filled_qty <= 0:
                # Nothing confirmably filled within the wait window. Never
                # fabricate a fill — a phantom trade row plus a stop for
                # shares that may not exist silently diverges DB state from
                # the broker.
                log.error(f"[{ticker}] No fill confirmed within wait window "
                          f"(id={order_result.get('order_id')}) — nothing recorded")
                try:
                    db.log_order(
                        ticker=ticker, side=side, order_type="MARKET",
                        qty=shares, broker_order_id=order_result.get("order_id"),
                        decision_price=price_at_sizing, approval_id=approval["id"],
                        status="unconfirmed", intent=action,
                    )
                except Exception:
                    pass
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

        # Real fill price; fall back to latest quote only if fill unconfirmed
        current_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
        price_source = "fill" if order_result.get("fill_price") else ("quote" if current_price else None)

        # Log trade — reason reflects how this actually fired (see trigger
        # docstring above); it used to always say "approved via Telegram"
        # even for fully autonomous crew.py executions no human ever saw.
        trigger_label = "auto-executed" if trigger == "auto" else "approved via Telegram"
        db.log_trade(
            symbol=ticker,
            side=side,
            quantity=filled_qty,
            price=current_price,
            reason=f"{action} {trigger_label} (conviction={approval['conviction']}/10)",
            price_source=price_source,
        )

        # Order-lifecycle + slippage logging (decision price = price used at sizing time)
        try:
            local_order_id = db.log_order(
                ticker=ticker, side=side,
                order_type="BRACKET" if bracket_used else "MARKET",
                qty=shares, broker_order_id=order_result.get("order_id"),
                decision_price=price_at_sizing, approval_id=approval["id"],
                status="submitted", intent=action,
            )
            if local_order_id > 0 and (order_result.get("fill_price")
                                       or (filled_qty > 0 and not bracket_used)):   # bracket qty is assumed
                # Part filled within the wait: the rest may still be working,
                # so leave it 'partial' for reconcile_unconfirmed_orders to
                # pick up the remainder (professor M53). A fill confirmed
                # without a price is still a fill: record it with no price
                # rather than leaving the row 'submitted' (professor run 17).
                partial = not bracket_used and filled_qty < float(shares) - 1e-6
                db.update_order_fill(local_order_id, order_result.get("fill_price"),
                                     filled_qty, "partial" if partial else "filled")
        except Exception as e:
            log.debug(f"[{ticker}] Order/slippage logging failed (non-critical): {e}")

        # Stop lifecycle on SELL: EXIT removes the stop record entirely;
        # TRIM re-places the stop for the remaining shares.
        if side == "SELL":
            if action == "EXIT":
                db.delete_position_stop(ticker)
                log.info(f"[{ticker}] Position closed — stop record deleted")
                if filled_qty < float(shares) - 1e-6:
                    # Part filled: the order stays 'partial' and reconcile
                    # closes the outcome from the cumulative fill once the
                    # rest fills (professor M121), not at this part's price.
                    log.info(f"[{ticker}] EXIT part-filled ({filled_qty:g} of {shares}); outcome "
                             f"left open for reconcile")
                else:
                    try:
                        db.close_trade_outcome(ticker, exit_price=current_price, exit_source="model_exit",
                                              exit_price_source=price_source)
                    except Exception as e:
                        log.warning(f"[{ticker}] close_trade_outcome failed: {e}")
            else:  # TRIM — re-place stop at same price for remaining quantity
                remaining = _remaining_after_sell(ticker, owned, filled_qty)
                stop_rec = db.get_position_stop(ticker)
                if remaining is None:
                    alert_module._send(_POSITION_MISMATCH_ALERT.format(
                        ticker=ticker, expected=round(owned - filled_qty, 4)))
                elif remaining > 0 and stop_rec:
                    try:
                        new_stop = tasks.manage_stop_place(ticker, remaining, stop_rec["stop_price"])
                        db.set_stop_broker_order_id(ticker, new_stop["order_id"])
                        log.info(f"[{ticker}] Stop re-placed for remaining {remaining} shares")
                    except Exception as e:
                        log.error(f"[{ticker}] Failed to re-place stop after TRIM: {e}")
                        queued = False
                        try:
                            tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)  # m39
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
                                              exit_price_source=price_source)
                    except Exception as e:
                        log.warning(f"[{ticker}] close_trade_outcome failed: {e}")

        # Log stop level and place stop order with IBKR.
        # Skipped when a bracket order already placed the stop atomically —
        # re-placing it here would create a second, redundant stop.
        stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
        if stop_price and side == "BUY" and bracket_used:
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )
            db.set_stop_broker_order_id(ticker, bracket_result["stop_order_id"])
            _capture_original_thesis_if_new(ticker, approval)
            log.info(f"[{ticker}] Bracket stop recorded (orderId={bracket_result['stop_order_id']})")
        elif stop_price and side == "BUY":
            stop_price = _stop_anchored_to_fill(ticker, float(stop_price), price_at_sizing, current_price)
            entry_for_stops, stop_qty = current_price, filled_qty
            stop_price, add_to_existing = _whole_position_stop(ticker, float(stop_price), filled_qty)
            if add_to_existing:
                entry_for_stops, stop_qty = add_to_existing
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=entry_for_stops,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )
            _capture_original_thesis_if_new(ticker, approval)
            stop_placed = False
            for stop_attempt in range(1, 4):
                try:
                    stop_result = tasks.manage_stop_place(ticker, stop_qty, float(stop_price))
                    db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                    stop_placed = True
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
                        stop_placed = True
                        break
                    if isinstance(e, tasks.OrdersCallUnknown):
                        # The stop may be resting already; a retry could stack
                        # a second one. Reconcile fixes coverage either way
                        # within two cycles (professor C24).
                        _alert_stop_unknown(ticker, stop_qty, float(stop_price))
                        stop_placed = True  # not "failed": unknown, alerted above
                        break
                    if isinstance(e, tasks.OrdersCallExpired) or type(e).__name__ == "LockLost":
                        # The orders worker is busy or stuck, or our ticker lock
                        # lapsed (another process may be changing stops);
                        # retrying here only repeats that. The queue re-reads
                        # coverage under a fresh lock (A0453-3, m39).
                        _queue_stop_for_retry(ticker, stop_qty, float(stop_price))
                        stop_placed = True  # handed to the retry queue, alerted
                        break
                    if stop_attempt < 3:
                        import time as _time
                        _time.sleep(2 * stop_attempt)
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
            log_buy_outcome(approval, ticker, action, current_price)

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
        # Cancel existing stops first so the fill can't race the stop and
        # leave an order that would short the account.
        cancel_ok = True
        try:
            cancelled = tasks.manage_cancel_stop_orders_for(ticker)
            if cancelled:
                log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before manual SELL")
        except Exception as e:
            cancel_ok = False
            log.warning(f"[{ticker}] Could not cancel existing stops before manual SELL: {e}")
        resting = _stops_still_resting(ticker)
        if resting or (resting is None and not cancel_ok):
            _restore_stop_after_failed_sell(ticker)  # A0053-2
            return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                    "message": f"SELL {ticker} not sent: an existing stop could not be "
                               f"cancelled (selling on top of it could short). Check TWS."}

    order_intent = f"MANUAL_{side}"
    held_before = 0.0
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
            _restore_stop_after_failed_sell(ticker)
        else:
            _release_buy(ticker)   # nothing was sent
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submission failed: {e}"}
    try:
        order_result = ibkr.place_market_order(ticker, side, qty)
    except Exception as e:
        if side == "SELL":
            # Same as execute_approved: the stop was cancelled above, so put
            # it back (limitations audit 2026-10-28, finding 9).
            _restore_stop_after_failed_sell(ticker)
        # A BUY keeps its reservation: the order may have gone out.
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submission failed: {e}"}

    filled_qty = float(order_result.get("filled_qty") or 0)
    if filled_qty <= 0 and side == "BUY" and order_result.get("status") in _DEAD_ORDER_STATUSES:
        _release_buy(ticker)   # rejected/cancelled: no late fill can come
    if filled_qty <= 0 and side == "SELL" and order_result.get("status") in _DEAD_ORDER_STATUSES:
        _restore_stop_after_failed_sell(ticker)
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
        try:
            db.log_order(ticker=ticker, side=side, order_type="MARKET", qty=qty,
                         broker_order_id=order_result.get("order_id"),
                         status="unconfirmed", intent=order_intent)
        except Exception as e:
            log.warning(f"[{ticker}] Could not log the unconfirmed manual order: {e}")
        stop_note = (" The broker stop was cancelled before the sell and NOT re-placed "
                     "(a late fill plus a full-size stop could short); check it."
                     if side == "SELL" else "")
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submitted (id {order_result.get('order_id')}) but no fill "
                           f"confirmed within the wait window. No trade recorded; a late fill "
                           f"is reconciled automatically — check TWS.{stop_note}"}

    fill_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
    db.log_trade(symbol=ticker, side=side, quantity=filled_qty, price=fill_price, reason=reason,
                 price_source="fill" if order_result.get("fill_price") else ("quote" if fill_price else None))
    try:
        local_order_id = db.log_order(ticker=ticker, side=side, order_type="MARKET", qty=qty,
                                      broker_order_id=order_result.get("order_id"),
                                      status="submitted", intent=order_intent)
        if local_order_id > 0 and filled_qty > 0:
            # A part fill stays 'partial' so reconcile books the rest (M53);
            # a fill confirmed without a price is recorded with none.
            db.update_order_fill(local_order_id, order_result.get("fill_price"), filled_qty,
                                 "partial" if filled_qty < float(qty) - 1e-6 else "filled")
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
                                       exit_price_source="fill" if order_result.get("fill_price") else "quote")
            except Exception:
                pass  # no open outcome row is expected for positions not opened via approval
        else:
            stop_rec = db.get_position_stop(ticker)
            if stop_rec:
                try:
                    new_stop = tasks.manage_stop_place(ticker, remaining, stop_rec["stop_price"])
                    db.set_stop_broker_order_id(ticker, new_stop["order_id"])
                except Exception as e:
                    log.error(f"[{ticker}] Failed to re-place stop after manual sell: {e}")
                    # Queue and track it like the other failed re-places (M87).
                    try:
                        tasks._queue_reprotect(ticker, {"intent": "RESTORE"}, None)
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
        try:
            stop_price, stop_method = rm.calculate_stop(ticker, fill_price, bucket)
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
            db.upsert_position_stop(
                ticker=ticker, entry_price=fill_price,
                stop_price=float(stop_price), stop_method=stop_method or "manual ATR stop",
                bucket=bucket, conviction=conviction, trail_distance_pct=None,
            )
            stop_placed = False
            for attempt in range(1, 4):
                try:
                    stop_result = tasks.manage_stop_place(ticker, filled_qty, float(stop_price))
                    db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                    stop_placed = True
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] Stop attempt {attempt}/3: {e}")
                    if type(e).__name__ == "SellExceedsPosition":
                        _alert_stop_refused(ticker, filled_qty, e)
                        _queue_stop_for_retry(ticker, filled_qty, float(stop_price), alert=False)
                        stop_placed = True
                        break
                    if isinstance(e, tasks.OrdersCallUnknown):
                        _alert_stop_unknown(ticker, filled_qty, float(stop_price))  # C24
                        stop_placed = True
                        break
                    if isinstance(e, tasks.OrdersCallExpired):
                        _queue_stop_for_retry(ticker, filled_qty, float(stop_price))  # A0453-3
                        stop_placed = True
                        break
                    if attempt < 3:
                        import time as _time
                        _time.sleep(2 * attempt)
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

    return {
        "success": True,
        "filled_qty": filled_qty,
        "fill_price": fill_price,
        "stop_price": float(stop_price) if stop_price else None,
        "message": f"{side} {filled_qty:g} {ticker} @ ~${fill_price:.2f}"
                   + (f", stop @ ${stop_price:.2f}" if stop_price else ""),
    }
