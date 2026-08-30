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

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval-driven execution (auto-execute pipeline + Telegram Approve button)
# ---------------------------------------------------------------------------

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

    Returns True on success.
    """
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
        return False

    try:
        import ibkr
        import tasks
        import db
        import alerts as alert_module

        shares = risk_json.get("shares_initial", 0)
        if shares <= 0:
            log.error(f"[{ticker}] No shares_initial in risk_json — cannot execute")
            return False

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
                    return False
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
                    return False
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
                return False
            shares = owned if action == "EXIT" else min(shares, owned)
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
                            return False
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
                        return False
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
                    return False
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
                    return False
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
                        return False
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
                    return False
                # "confirmed" or "no_gap" -- proceed normally.
            except Exception as e:
                log.warning(f"[{ticker}] Gap-confirmation check failed, proceeding: {e}")

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
            order_result = ibkr.place_market_order(ticker, side, shares)
            log.info(f"[{ticker}] Order submitted: {side} {shares} shares (id={order_result['order_id']})")

        if not bracket_used:
            filled_qty = float(order_result.get("filled_qty") or 0)
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
                        status="unconfirmed",
                    )
                except Exception:
                    pass
                try:
                    alert_module._send(
                        f"⚠️ FILL UNCONFIRMED — <b>{ticker}</b>\n"
                        f"{side} {shares} shares submitted (order id "
                        f"{order_result.get('order_id')}) but no fill confirmed within "
                        f"the wait window. Nothing recorded — reconcile in TWS."
                    )
                except Exception:
                    pass
                return False
        else:
            # Bracket path returns no fill info by design — TWS enforces the
            # entry+stop pairing atomically, so assume the requested size.
            filled_qty = shares

        # Real fill price; fall back to latest quote only if fill unconfirmed
        current_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0

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
        )

        # Order-lifecycle + slippage logging (decision price = price used at sizing time)
        try:
            local_order_id = db.log_order(
                ticker=ticker, side=side,
                order_type="BRACKET" if bracket_used else "MARKET",
                qty=shares, broker_order_id=order_result.get("order_id"),
                decision_price=price_at_sizing, approval_id=approval["id"],
                status="submitted",
            )
            if local_order_id > 0 and order_result.get("fill_price"):
                db.update_order_fill(local_order_id, order_result["fill_price"],
                                     filled_qty, "filled")
        except Exception as e:
            log.debug(f"[{ticker}] Order/slippage logging failed (non-critical): {e}")

        # Stop lifecycle on SELL: EXIT removes the stop record entirely;
        # TRIM re-places the stop for the remaining shares.
        if side == "SELL":
            if action == "EXIT":
                db.delete_position_stop(ticker)
                log.info(f"[{ticker}] Position closed — stop record deleted")
                # Close the feedback-loop outcome record with the real fill price
                try:
                    db.close_trade_outcome(ticker, exit_price=current_price)
                except Exception as e:
                    log.warning(f"[{ticker}] close_trade_outcome failed: {e}")
            else:  # TRIM — re-place stop at same price for remaining quantity
                try:
                    remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
                except Exception:
                    remaining = 0
                stop_rec = db.get_position_stop(ticker)
                if remaining > 0 and stop_rec:
                    try:
                        new_stop = tasks.manage_stop_place(ticker, remaining, stop_rec["stop_price"])
                        db.set_stop_broker_order_id(ticker, new_stop["order_id"])
                        log.info(f"[{ticker}] Stop re-placed for remaining {remaining} shares")
                    except Exception as e:
                        log.error(f"[{ticker}] Failed to re-place stop after TRIM: {e}")
                        alert_module._send(
                            f"🚨 UNPROTECTED POSITION — {ticker}\n"
                            f"TRIM filled but stop re-placement failed. "
                            f"Place stop @ ${stop_rec['stop_price']:.2f} for {remaining} shares manually."
                        )
                elif remaining <= 0:
                    db.delete_position_stop(ticker)

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
            log.info(f"[{ticker}] Bracket stop recorded (orderId={bracket_result['stop_order_id']})")
        elif stop_price and side == "BUY":
            db.upsert_position_stop(
                ticker=ticker,
                entry_price=current_price,
                stop_price=float(stop_price),
                stop_method=risk_json.get("stop_method", ""),
                bucket=approval.get("thesis_bucket", "Accelerator"),
                conviction=approval.get("conviction", 5),
                trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
            )
            stop_placed = False
            for stop_attempt in range(1, 4):
                try:
                    stop_result = tasks.manage_stop_place(ticker, filled_qty, float(stop_price))
                    db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                    stop_placed = True
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] Stop order attempt {stop_attempt}/3 failed: {e}")
                    if stop_attempt < 3:
                        import time as _time
                        _time.sleep(2 * stop_attempt)
            if not stop_placed:
                log.error(f"[{ticker}] CRITICAL — stop order failed after 3 attempts. Position is UNPROTECTED.")
                try:
                    alert_module._send(
                        f"🚨 UNPROTECTED POSITION — ACTION REQUIRED\n"
                        f"<b>{ticker}</b>: entry order was filled but stop order failed after 3 attempts.\n"
                        f"Stop target: <b>${float(stop_price):.2f}</b> for {filled_qty} shares.\n"
                        f"Place stop manually in TWS immediately."
                    )
                except Exception:
                    pass

        # Log outcome record for feedback loop — sub-scores live in
        # analyst_json (the full analyst.analyze() result), not as their own
        # pending_approvals columns, since only conviction/thesis_bucket are
        # promoted to top-level columns there.
        analyst_json = approval.get("analyst_json") or {}
        db.log_trade_outcome(
            approval_id=approval["id"],
            ticker=ticker,
            action=action,
            conviction=approval.get("conviction", 5),
            bucket=approval.get("thesis_bucket", ""),
            entry_price=current_price,
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
        return False


# ---------------------------------------------------------------------------
# Manual / !urgent execution (dashboard form, chat !urgent command)
# ---------------------------------------------------------------------------

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
        # Cancel existing stops first so the fill can't race the stop and
        # leave an order that would short the account.
        try:
            cancelled = tasks.manage_cancel_stop_orders_for(ticker)
            if cancelled:
                log.info(f"[{ticker}] Cancelled {cancelled} stop order(s) before manual SELL")
        except Exception as e:
            log.warning(f"[{ticker}] Could not cancel existing stops before manual SELL: {e}")

    try:
        order_result = ibkr.place_market_order(ticker, side, qty)
    except Exception as e:
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submission failed: {e}"}

    filled_qty = float(order_result.get("filled_qty") or 0)
    if filled_qty <= 0:
        # No confirmed fill — never fabricate one; a phantom trade row plus
        # a stop for shares that may not exist would diverge DB state from
        # the broker.
        log.error(f"[{ticker}] No fill confirmed for manual {side} {qty} "
                  f"(id={order_result.get('order_id')}) — nothing recorded")
        return {"success": False, "filled_qty": 0, "fill_price": None, "stop_price": None,
                "message": f"Order submitted (id {order_result.get('order_id')}) but no fill "
                           f"confirmed within the wait window. Nothing recorded — check TWS."}

    fill_price = order_result.get("fill_price") or ibkr.get_latest_price(ticker) or 0.0
    db.log_trade(symbol=ticker, side=side, quantity=filled_qty, price=fill_price, reason=reason)

    stop_price = None
    if side == "SELL":
        # Keep DB stop records in sync with what remains at the broker
        try:
            remaining = round(abs(float(ibkr.get_client().get_open_position(ticker).qty)), 4)
        except Exception:
            remaining = 0
        if remaining <= 0:
            db.delete_position_stop(ticker)
            try:
                db.close_trade_outcome(ticker, exit_price=fill_price)
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
    else:  # BUY — place an ATR stop and make the position visible to the monitor
        try:
            stop_price, stop_method = rm.calculate_stop(ticker, fill_price, bucket)
        except Exception as e:
            log.warning(f"[{ticker}] calculate_stop failed: {e}")
            stop_price, stop_method = None, ""
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
                    if attempt < 3:
                        import time as _time
                        _time.sleep(2 * attempt)
            if not stop_placed:
                try:
                    alert_module._send(
                        f"🚨 UNPROTECTED POSITION — {ticker}\n"
                        f"Manual order filled but stop failed. Place stop @ ${stop_price:.2f} manually."
                    )
                except Exception:
                    pass

    return {
        "success": True,
        "filled_qty": filled_qty,
        "fill_price": fill_price,
        "stop_price": float(stop_price) if stop_price else None,
        "message": f"{side} {filled_qty:g} {ticker} @ ~${fill_price:.2f}"
                   + (f", stop @ ${stop_price:.2f}" if stop_price else ""),
    }
