"""
Interactive Brokers connection module via ib_async (the maintained fork of
ib_insync — the original is unmaintained and its asyncio internals break on
Python 3.13+ with "Timeout should be used inside a task").
Replaces Alpaca for order execution, account data, and market data.

Requirements:
  pip install ib_async

TWS must be running with API enabled:
  - Paper trading port: 7497  (IB_PORT=7497)
  - Live trading port:  7496  (IB_PORT=7496)

Set in .env:
  TRADING_ENV=paper          # or "live" — see config.py; derives IB_PORT,
                              # IB_CLIENT_ID and START_VALUE defaults below
  IB_HOST=127.0.0.1
  IB_PORT=7497                # optional override; normally derived from TRADING_ENV
  IB_CLIENT_ID=1               # optional override; normally derived from TRADING_ENV
"""
import logging
import os
from pathlib import Path
from typing import Any

# NOTE: do not re-add nest_asyncio.apply() here — its loop patching is broken
# on Python 3.13+ and makes every connect fail with "Timeout should be used
# inside a task". ib_async manages its own per-thread event loops.
import ib_async as ibi
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

# Sourced from config.py (not read directly from the environment here) so
# this can never drift from config.TRADING_ENV's derived paper/live
# defaults — see config.py's "Trading environment" section.
from config import IB_HOST, IB_PORT, IB_CLIENT_ID as _IB_CLIENT_ID_BASE

_ib: ibi.IB | None = None
_client_id: int | None = None   # resolved lazily on first connect
_last_heartbeat: float = 0.0   # time.monotonic() of last successful heartbeat

_HEARTBEAT_INTERVAL_S = 30
_HEARTBEAT_TIMEOUT_S  = 8


def _resolve_client_id() -> int:
    """A Celery worker started with --pool=prefork -c N forks N child
    processes that all inherit the same IB_CLIENT_ID from the environment —
    TWS allows only one connection per clientId, so without this every fork
    but one gets rejected ("clientId already in use"), silently breaking
    whichever ops task (stop checks, drawdown, snapshots) lands on a loser
    fork. Offset by the prefork worker index when running inside one."""
    try:
        from billiard import current_process
        idx = getattr(current_process(), "index", None)
        if idx is not None:
            return _IB_CLIENT_ID_BASE + int(idx)
    except Exception:
        pass
    return _IB_CLIENT_ID_BASE


def _get_client_id() -> int:
    global _client_id
    if _client_id is None:
        _client_id = _resolve_client_id()
    return _client_id


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_ib(max_attempts: int = 3) -> ibi.IB:
    global _ib, _last_heartbeat, _client_id
    import asyncio
    import time
    for attempt in range(1, max_attempts + 1):
        try:
            if _ib is None:
                _ib = ibi.IB()
            if not _ib.isConnected():
                cid = _get_client_id()
                try:
                    _ib.connect(IB_HOST, IB_PORT, clientId=cid,
                                readonly=False, timeout=10)
                except Exception as connect_err:
                    # Error 326: another process already holds this clientId
                    # (belt-and-braces on top of _resolve_client_id — covers
                    # non-Celery processes or an unexpected id collision).
                    # Fall back to a pid-derived id and retry once.
                    if "already in use" in str(connect_err) or "326" in str(connect_err):
                        fallback = 100 + (os.getpid() % 800)
                        log.warning(f"clientId {cid} already in use — retrying with {fallback}")
                        _client_id = fallback
                        _ib.connect(IB_HOST, IB_PORT, clientId=fallback,
                                    readonly=False, timeout=10)
                        cid = fallback
                    else:
                        raise
                _last_heartbeat = 0.0
                log.info(f"Connected to IBKR TWS at {IB_HOST}:{IB_PORT} (clientId={cid})")
            # Heartbeat — round-trips to TWS to catch stale connections that
            # report isConnected()=True but are no longer communicating.
            # Two constraints discovered the hard way:
            #   - TWS silently ignores a repeated reqCurrentTime arriving
            #     within the same second, and ib_async waits for the reply
            #     with NO timeout — so back-to-back get_ib() calls deadlock.
            #     Hence the 30s throttle.
            #   - The asyncio.wait_for bound turns a genuinely dead session
            #     into an exception that triggers the reconnect below,
            #     instead of an infinite hang.
            if time.monotonic() - _last_heartbeat > _HEARTBEAT_INTERVAL_S:
                ibi.util.run(asyncio.wait_for(
                    _ib.reqCurrentTimeAsync(), timeout=_HEARTBEAT_TIMEOUT_S))
                _last_heartbeat = time.monotonic()
            return _ib
        except Exception as e:
            log.warning(f"IBKR connect attempt {attempt}/{max_attempts} failed: {e}")
            try:
                if _ib is not None:
                    _ib.disconnect()
            except Exception:
                pass
            _ib = None  # force fresh IB() on next attempt
            _last_heartbeat = 0.0
            if attempt < max_attempts:
                time.sleep(3 * attempt)  # 3s, 6s backoff
    raise ConnectionError(
        f"Could not connect to IBKR TWS at {IB_HOST}:{IB_PORT} after {max_attempts} attempts. "
        "Ensure TWS or IB Gateway is running with API connections enabled."
    )


def disconnect() -> None:
    global _ib
    if _ib and _ib.isConnected():
        _ib.disconnect()
        _ib = None
        log.info("Disconnected from IBKR TWS.")


def _stock(ib: ibi.IB, ticker: str) -> ibi.Stock:
    # Must be qualified (conId populated) before use — ib_async hashes
    # Contract objects internally (market data/order tracking dicts), which
    # raises "can't be hashed because no 'conId' value exists" on a bare
    # Stock(). ib_insync tolerated this; ib_async does not.
    contract = ibi.Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    return contract


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_latest_price(ticker: str) -> float | None:
    """
    Returns the latest price via historical minute bars.
    Uses zero market data lines — no streaming subscription opened.
    """
    try:
        # Try last minute bar first (most recent trade during/after market hours)
        bars = get_historical_bars(ticker, duration="1 D", bar_size="1 min")
        if bars:
            return float(bars[-1]["close"])
        # Fall back to daily bar (last close)
        bars = get_historical_bars(ticker, duration="2 D", bar_size="1 day")
        if bars:
            return float(bars[-1]["close"])
    except Exception as e:
        log.warning(f"get_latest_price({ticker}) failed: {e}")
    return None


def get_historical_bars(
    ticker: str,
    duration: str = "20 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> list[dict]:
    """
    Returns OHLCV bars from IBKR historical data.
    duration examples: '10 D', '1 M', '3 M'
    bar_size examples: '1 day', '1 hour', '5 mins'
    """
    try:
        ib = get_ib()
        contract = _stock(ib, ticker)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )
        return [
            {
                "time":   str(b.date),
                "open":   float(b.open),
                "high":   float(b.high),
                "low":    float(b.low),
                "close":  float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    except Exception as e:
        log.warning(f"get_historical_bars({ticker}) failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

class SellExceedsPosition(RuntimeError):
    """A SELL for more shares than the broker shows long (including a short
    or a flat position) would open or add to a short. Not retryable."""


def _long_qty(positions, ticker: str) -> float:
    # Stock only: a short option on the same symbol must not read as a short
    # stock position and block the stop (professor m47).
    return sum(float(p.position) for p in positions
               if p.contract.symbol == ticker and getattr(p.contract, "secType", "STK") == "STK")


def _refuse_sell_on_short(ib, ticker: str, qty: float | None = None) -> None:
    """Every SELL here (stop, trail, market) assumes a long position at least
    as large as the order; otherwise it opens or adds to a short. Callers
    size from abs(qty) in many places, so the check sits where all of them
    meet (professor M63 residual, run 11 follow-up). The position cache can
    lag right after a fill, so a shortfall is confirmed with a fresh
    reqPositions() before refusing."""
    def short_of(positions):
        held = _long_qty(positions, ticker)
        return held < 0 or (qty is not None and float(qty) > held + 1e-6), held
    bad, held = short_of(ib.positions())
    if bad:
        try:
            bad, held = short_of(ib.reqPositions())
        except Exception:
            pass
    if bad:
        raise SellExceedsPosition(f"{ticker}: SELL {qty if qty is not None else ''} refused; "
                                  f"broker shows {held:g} shares")


def place_market_order(ticker: str, side: str, qty: float, fill_timeout: float = 30.0) -> dict:
    """
    Place a market order and wait for the fill. side = 'BUY' or 'SELL'.
    Returns dict with order_id, status, fill_price (None if unfilled at timeout),
    and filled_qty.

    Incident 2026-07-26/27: a "DAY" market order submitted while the market
    is closed doesn't get rejected -- IBKR accepts it and queues it for the
    next session (API warning 399, "will not be placed at the exchange
    until [next open]"), so it sits PreSubmitted indefinitely. The 30s
    fill-wait below then reports it as unfilled, and callers (execution.
    execute_approved) marked the trade FAILED without ever cancelling the
    now-orphaned order -- it was still live at the broker, disconnected
    from any DB record, and would have silently attempted to fill at the
    next open with no stop-loss and no trade log entry. If the wait window
    ends with the order still pending AND the market is closed, cancel it
    here so no caller can ever leave one dangling, regardless of whether
    that caller itself checks market hours first.
    """
    ib = get_ib()
    if side == "SELL":
        _refuse_sell_on_short(ib, ticker, qty)
    contract = _stock(ib, ticker)

    # Subscribe to delayed data briefly to satisfy TWS precautionary check
    ib.reqMarketDataType(3)
    td = ib.reqMktData(contract, "", False, False)
    ib.sleep(1)

    order = ibi.MarketOrder(side.upper(), qty)
    order.tif = "DAY"  # market orders must not persist across sessions
    trade = ib.placeOrder(contract, order)

    # Wait for the real fill instead of guessing the price afterwards.
    # Incident 2026-07-28: this used to count loop iterations ("waited += 1"
    # per call) as a proxy for elapsed seconds, on the assumption each
    # ib.waitOnUpdate(timeout=1) call blocks for ~1s. At 9:30:02 ET market
    # open, a burst of already-queued order/market-data events (many other
    # tickers' stop orders re-broadcasting status) made waitOnUpdate return
    # near-instantly on every call, so the loop burned through all 30
    # "seconds" in about 1 real second and gave up — logging "no fill" for
    # a CRWD BUY that filled for real 12 seconds later, live and untracked
    # at the broker with no stop, no trade row. Use actual wall-clock time
    # so a fast-draining event queue can't fake out the wait window.
    import time as _time
    start = _time.monotonic()
    while not trade.isDone() and (_time.monotonic() - start) < fill_timeout:
        ib.waitOnUpdate(timeout=1)

    ib.cancelMktData(contract)

    fill_price = float(trade.orderStatus.avgFillPrice or 0) or None
    filled_qty = float(trade.orderStatus.filled or 0)
    status     = trade.orderStatus.status

    if fill_price:
        log.info(f"Order FILLED: {side.upper()} {filled_qty} {ticker} @ ${fill_price:.2f} "
                 f"(orderId={trade.order.orderId})")
    else:
        log.warning(f"Order not confirmed filled after {fill_timeout:.0f}s: "
                    f"{side.upper()} {qty} {ticker} status={status} "
                    f"(orderId={trade.order.orderId}) — fill will need reconciliation")

        # Still pending and the market's closed -- this will never fill
        # within any reasonable wait window (it's queued for next session,
        # not slow). Cancel it rather than leaving it live and untracked;
        # see this function's docstring for the incident this prevents.
        if status in ("PreSubmitted", "PendingSubmit", "Submitted"):
            try:
                from market_data import is_market_hours
                if not is_market_hours():
                    ib.cancelOrder(trade.order)
                    ib.sleep(1)
                    log.warning(
                        f"Cancelled {side.upper()} {qty} {ticker} (orderId={trade.order.orderId}) "
                        f"— market is closed, order would have queued for next session untracked"
                    )
                    status = "Cancelled"
            except Exception as e:
                log.error(f"Failed to cancel stale {side.upper()} {ticker} order "
                          f"(orderId={trade.order.orderId}) after market-closed check: {e}")

    return {
        "order_id":   trade.order.orderId,
        "ticker":     ticker,
        "side":       side.upper(),
        "qty":        qty,
        "filled_qty": filled_qty,
        "fill_price": fill_price,
        "status":     status,
    }


def _compute_marketable_limit(ib: ibi.IB, contract: ibi.Stock, side: str,
                              offset_pct: float = 0.001) -> float:
    """
    Snapshots bid/ask and returns a limit price aggressive enough to fill
    like a market order but bounded — ask+offset for BUY, bid-offset for
    SELL. Falls back to last trade price if bid/ask aren't available
    (illiquid name / data permissions).
    """
    ib.reqMarketDataType(3)
    md = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    bid  = md.bid  if md.bid  and md.bid  > 0 else None
    ask  = md.ask  if md.ask  and md.ask  > 0 else None
    last = md.last if md.last and md.last > 0 else None
    ib.cancelMktData(contract)

    if side.upper() == "BUY":
        base = ask or last
    else:
        base = bid or last
    if base is None:
        raise ValueError(f"No price data available to compute a marketable limit for {contract.symbol}")

    return base * (1 + offset_pct) if side.upper() == "BUY" else base * (1 - offset_pct)


def place_limit_order(
    ticker: str,
    side: str,
    qty: float,
    limit_price: float | None = None,
    tif: str = "DAY",
    fill_timeout: float = 30.0,
) -> dict:
    """
    Place a limit order. If limit_price is None, computes a marketable limit
    (aggressive enough to fill promptly, capped so a fast-moving market can't
    fill at an arbitrarily bad price the way a market order can).
    Unfilled-at-timeout orders are NOT cancelled here — the caller decides
    whether to cancel-and-retry or leave it working (DAY orders expire with
    the session; the reconciliation task catches anything left dangling).
    """
    ib = get_ib()
    contract = _stock(ib, ticker)

    if limit_price is None:
        limit_price = _compute_marketable_limit(ib, contract, side)
    limit_price = round(limit_price, 2)

    order = ibi.LimitOrder(side.upper(), qty, limit_price)
    order.tif = tif
    trade = ib.placeOrder(contract, order)

    waited = 0.0
    while not trade.isDone() and waited < fill_timeout:
        ib.waitOnUpdate(timeout=1)
        waited += 1

    fill_price = float(trade.orderStatus.avgFillPrice or 0) or None
    filled_qty = float(trade.orderStatus.filled or 0)
    status     = trade.orderStatus.status

    if fill_price:
        log.info(f"Limit order FILLED: {side.upper()} {filled_qty} {ticker} @ ${fill_price:.2f} "
                 f"(limit=${limit_price:.2f}, orderId={trade.order.orderId})")
    else:
        log.warning(f"Limit order not filled after {fill_timeout:.0f}s: "
                    f"{side.upper()} {qty} {ticker} @ ${limit_price:.2f} status={status} "
                    f"(orderId={trade.order.orderId})")

    return {
        "order_id":    trade.order.orderId,
        "ticker":      ticker,
        "side":        side.upper(),
        "qty":         qty,
        "limit_price": limit_price,
        "filled_qty":  filled_qty,
        "fill_price":  fill_price,
        "status":      status,
    }


def place_bracket_order(
    ticker: str,
    side: str,
    qty: float,
    stop_loss_price: float,
    entry_type: str = "LIMIT",
    limit_price: float | None = None,
    take_profit_price: float | None = None,
    tif: str = "DAY",
) -> dict:
    """
    Native parent/child bracket: entry order + stop-loss child (+ optional
    take-profit child). TWS enforces the entry/stop pairing atomically via
    parentId linkage — if the entry never fills, the stop never activates;
    if one exit child fills, TWS cancels its sibling. This makes the whole
    class of "orphaned stop after a manual exit" bugs structurally
    impossible for NEW positions opened this way (existing positions still
    rely on the Phase 0 manual lifecycle + reconciliation as a backstop).

    Only the LAST order in the chain is submitted with transmit=True — until
    then TWS holds the whole bracket un-transmitted, so partial submission
    isn't possible.
    """
    if not stop_loss_price:
        raise ValueError("place_bracket_order requires a stop_loss_price")

    ib = get_ib()
    contract = _stock(ib, ticker)
    action = side.upper()
    reverse_action = "SELL" if action == "BUY" else "BUY"

    if entry_type == "MARKET":
        parent = ibi.MarketOrder(action, qty)
    else:
        if limit_price is None:
            limit_price = _compute_marketable_limit(ib, contract, action)
        parent = ibi.LimitOrder(action, qty, round(limit_price, 2))
    parent.orderId  = ib.client.getReqId()
    parent.tif      = tif
    parent.transmit = False

    stop = ibi.StopOrder(reverse_action, qty, round(stop_loss_price, 2))
    stop.orderId  = ib.client.getReqId()
    stop.parentId = parent.orderId
    stop.tif      = "GTC"
    stop.transmit = take_profit_price is None  # last child transmits the whole bracket

    children = [stop]
    tp = None
    if take_profit_price:
        tp = ibi.LimitOrder(reverse_action, qty, round(take_profit_price, 2))
        tp.orderId  = ib.client.getReqId()
        tp.parentId = parent.orderId
        tp.tif      = "GTC"
        tp.transmit = True
        children.append(tp)

    for o in [parent] + children:
        ib.placeOrder(contract, o)
    ib.sleep(1)

    log.info(
        f"Bracket order placed: {action} {qty} {ticker} "
        f"(parent={parent.orderId}, stop={stop.orderId}"
        + (f", take_profit={tp.orderId}" if tp else "") + ")"
    )

    return {
        "parent_order_id":      parent.orderId,
        "stop_order_id":        stop.orderId,
        "take_profit_order_id": tp.orderId if tp else None,
        "limit_price":          limit_price if entry_type != "MARKET" else None,
    }


def place_trailing_stop_order(
    ticker: str, qty: float,
    trail_percent: float | None = None, trail_amount: float | None = None,
) -> dict:
    """
    Native IBKR TRAIL order — the broker ratchets the stop price itself as
    the position gains, so no cancel/replace round-trip (and the gap that
    creates between "old stop cancelled" and "new stop live") is needed once
    a position's trailing stop has activated.

    Exactly one of trail_percent / trail_amount must be given. trail_amount
    (a dollar distance, e.g. N x ATR) is preferred when available — it
    adapts to each stock's own volatility the same way the initial ATR
    stop already does, instead of every stock in a bucket trailing the same
    flat percent regardless of how it individually trades.
    """
    if (trail_percent is None) == (trail_amount is None):
        raise ValueError("place_trailing_stop_order: give exactly one of trail_percent/trail_amount")

    ib = get_ib()
    _refuse_sell_on_short(ib, ticker, qty)
    contract = _stock(ib, ticker)
    order_kwargs = {"trailingPercent": trail_percent} if trail_percent is not None else {"auxPrice": trail_amount}
    order = ibi.Order(
        action="SELL", totalQuantity=qty, orderType="TRAIL", tif="GTC", **order_kwargs,
    )
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    label = f"{trail_percent}%" if trail_percent is not None else f"${trail_amount:.2f}"
    log.info(f"Native trailing stop placed: SELL {qty} {ticker} trailing {label} "
             f"(orderId={trade.order.orderId})")
    return {"order_id": trade.order.orderId, "trail_percent": trail_percent, "trail_amount": trail_amount}


def place_stop_order(ticker: str, qty: float, stop_price: float) -> dict:
    """Place a GTC sell stop-loss order. Returns order_id for lifecycle tracking."""
    ib = get_ib()
    _refuse_sell_on_short(ib, ticker, qty)
    contract = _stock(ib, ticker)
    order    = ibi.StopOrder("SELL", qty, stop_price, tif="GTC")
    trade    = ib.placeOrder(contract, order)
    ib.sleep(1)
    log.info(f"Stop order placed: SELL {qty} {ticker} @ ${stop_price:.2f} "
             f"(orderId={trade.order.orderId})")
    return {"order_id": trade.order.orderId, "stop_price": stop_price}


def cancel_order(order_id: int, _poll_seconds: int = 5) -> bool:
    """Cancel an open order by orderId. Returns True only if the broker
    actually confirms the cancel (order status reaches a genuinely
    terminal Cancelled/ApiCancelled state) — a client that doesn't own the
    order (wrong clientId) can send cancelOrder() without IBKR raising
    anything, and the order just stays open. Trusting a clean call here
    caused reconcile_stop_orders to report the same orphaned stop
    "cancelled" every 15-minute cycle forever without it ever actually
    going away.

    Incident 2026-08-04 (MRVL): accepting "PendingCancel" itself as
    confirmed (as this used to) is not enough -- it's a genuinely
    transient in-flight state, not a terminal one. A cross-clientId
    cancel request live-tested that day showed status flip to
    PendingCancel within a second (exactly what the old check trusted),
    then silently revert to PreSubmitted three minutes later once IBKR
    finished rejecting it -- by then the caller had already placed a
    replacement stop on top, believing the old one gone. Polling
    synchronously for the full multi-minute window a rejection can take
    isn't practical here (this runs inline in time-sensitive execution
    paths), so this now polls for up to _poll_seconds for the status to
    reach an actually-terminal state instead of trusting the first read.
    A cancel still unresolved after the poll window returns False (not
    confirmed), even if it later lands; callers treat that stop as still
    resting (A0253-1). A real same-clientId cancel usually resolves within a
    second or two. Treat "True" as "very likely", not "certain", for a stop
    this client didn't place.
    """
    ib = get_ib()
    for trade in ib.openTrades():
        if trade.order.orderId == order_id:
            ib.cancelOrder(trade.order)
            for _ in range(_poll_seconds):
                ib.sleep(1)
                status = trade.orderStatus.status
                if status in ("Cancelled", "ApiCancelled"):
                    log.info(f"Cancel confirmed for orderId={order_id} "
                             f"({trade.order.action} {trade.order.totalQuantity} "
                             f"{trade.contract.symbol}) status={status}")
                    return True
            log.warning(f"Cancel sent for orderId={order_id} but status is still "
                        f"'{status}' after {_poll_seconds}s — broker likely ignored "
                        f"it (owned by a different clientId) or hasn't resolved yet; "
                        f"needs manual cancel/verification in TWS")
            return False
    log.warning(f"cancel_order: orderId={order_id} not found among open orders")
    return False


def _price_or_none(v) -> float | None:
    """ib_async decodes an empty price field to UNSET_DOUBLE (~1.8e308), not
    0; treat that, 0 and None as "no price" (audit 08:53, A0853-4)."""
    try:
        f = float(v or 0)
    except (TypeError, ValueError):
        return None
    return f if 0 < f < 1e300 else None


def get_open_orders() -> list[dict]:
    """
    Returns all open orders as dicts:
    {order_id, ticker, action, order_type, qty, stop_price, status}
    """
    ib = get_ib()
    # Build from the snapshot reqAllOpenOrders() returns, not openTrades():
    # this process's cache is never pruned for orders another client
    # (the orders worker) cancels, so a cancelled stop would still read as
    # resting here (audit 00:53 A0053-1).
    trades = ib.reqAllOpenOrders()
    ib.sleep(1)
    results = []
    seen = set()
    for trade in trades:
        o = trade.order
        key = (getattr(o, "permId", 0) or None, getattr(o, "clientId", None), o.orderId)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "order_id":   o.orderId,
            "ticker":     trade.contract.symbol,
            "action":     o.action,
            "order_type": o.orderType,
            "qty":        float(o.totalQuantity),
            # For a TRAIL, auxPrice is the trailing distance, not a price; its
            # current trigger is trailStopPrice (audit 06:53, A0653-4).
            "stop_price": _price_or_none(getattr(o, "trailStopPrice", None)
                                         if o.orderType == "TRAIL"
                                         else getattr(o, "auxPrice", None)),
            "status":     trade.orderStatus.status,
            # Unfilled part; totalQuantity alone overstates a part-filled
            # stop's coverage (professor M64).
            "remaining":  float(getattr(trade.orderStatus, "remaining", 0) or 0) or None,
        })
    return results


def cancel_stop_orders_for(ticker: str) -> int:
    """Cancel all open SELL stop orders for a ticker. Returns the count
    the broker actually CONFIRMED cancelled, not just requested.

    Incident 2026-08-03: this used to fire-and-forget ib.cancelOrder() and
    count every matching order as cancelled regardless of whether IBKR
    actually accepted it -- the same silent-failure shape cancel_order()
    was hardened against on 2026-07-27 (SNDK), where a client that doesn't
    own an order (a different clientId placed it) can call cancelOrder()
    without IBKR raising anything, and the order just stays open. Every
    caller here (the SELL-side stop cleanup, the trailing-stop conversion)
    treats a nonzero return as "the old stop is gone, safe to place a new
    one" -- so a silent failure meant a fresh stop landed ON TOP of a
    still-live old one instead of replacing it. Found via the 2026-08-03
    EOD review: ORCL, SNDK, and VRT each ended that day carrying a stale
    duplicate stop this way when the ops worker's clientId didn't match
    whichever session had originally placed the old order. Over-coverage,
    not a naked position that time -- but that was luck, not a guarantee.
    Now reuses cancel_order()'s verification per order and logs a clear
    warning for anything that didn't actually go away, so a caller placing
    a replacement stop right after at least has the fact on record even
    though it can't force a cross-clientId cancel to succeed.
    """
    ib = get_ib()
    matches = [
        trade.order for trade in ib.openTrades()
        if trade.contract.symbol == ticker
        and trade.order.action == "SELL"
        and trade.order.orderType in ("STP", "STP LMT", "TRAIL")
    ]
    confirmed = 0
    stuck = []
    for order in matches:
        if cancel_order(order.orderId):
            confirmed += 1
        else:
            stuck.append(order.orderId)
    if stuck:
        log.warning(
            f"[{ticker}] {len(stuck)} stop order(s) did NOT actually cancel "
            f"(orderIds={stuck}) — likely owned by a different clientId. Any "
            f"new stop placed now will overlap with these, not replace them. "
            f"Cancel manually in TWS."
        )
    return confirmed


def get_fill_for_order(broker_order_id: int, ticker: str | None = None,
                       side: str | None = None, placed_at=None) -> dict | None:
    """Looks up actual executions for a specific order by id, aggregating
    partials into one (filled_qty, avg_price). Returns None if nothing has
    executed for that order (yet, or ever).

    Added 2026-07-29 alongside tasks.reconcile_unconfirmed_orders: a market
    order can legitimately fill *after* place_market_order's fill-wait gives
    up (the CRWD incident -- filled 12s past the window, order recorded
    "unconfirmed", position ended up live and untracked with no stop). This
    lets a periodic reconciliation pass find such late fills using IBKR's
    own execution record, rather than trusting whatever place_market_order
    observed in its own limited wait window.

    2026-08-04: confirmed (live, against the real broker, clientId 999) that
    reqExecutions() only ever returns the CURRENT TRADING DAY's executions,
    across all clientIds -- and that passing ExecutionFilter(time=<30 days
    ago>) makes zero difference: identical result sets with and without it.
    This is a hard TWS/IB API limitation, not something this call can work
    around -- do not "fix" this by adding an ExecutionFilter time bound,
    that was tried and verified not to work. CRM, CRWD, and ORCL order 40
    (2026-07-26) are confirmed real broker fills with no DB trade record
    that this function cannot recover once the fill has aged past the
    current session -- closing that gap needs either IBKR's separate Flex
    Query historical-reports API or manual reconciliation against a broker
    statement, not reqExecutions.

    orderIds are per clientId and reqExecutions returns every client's
    executions, so an id alone can match an unrelated order (professor M-4,
    2026-10-28). Callers pass the order's ticker, side (BUY/SELL) and
    placement time; an execution must agree on each one given.
    """
    ib = get_ib()
    execs = ib.reqExecutions()
    want_side = {"BUY": "BOT", "SELL": "SLD"}.get(side) if side else None
    matches = [
        e for e in execs
        if e.execution.orderId == broker_order_id
        and (ticker is None or e.contract.symbol == ticker)
        and (want_side is None or e.execution.side == want_side)
        and (placed_at is None or _exec_time(e) is None or _exec_time(e) >= placed_at)
    ]
    if not matches:
        return None
    total_qty = sum(float(e.execution.shares) for e in matches)
    if total_qty <= 0:
        return None
    avg_price = sum(float(e.execution.shares) * float(e.execution.avgPrice) for e in matches) / total_qty
    return {"filled_qty": total_qty, "fill_price": avg_price}


def _exec_time(record):
    """Execution time as an aware datetime, or None if it can't be read."""
    from datetime import timezone
    t = getattr(record.execution, "time", None)
    if t is None or not hasattr(t, "tzinfo"):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def get_recent_sell_fill(ticker: str) -> dict | None:
    """Looks up today's SELL execution(s) for a ticker by symbol rather than
    a specific orderId, aggregating same-day partials into one (filled_qty,
    fill_price). Returns None if there's no SELL execution for this ticker
    today.

    Added 2026-08-09 alongside reconcile_stop_orders' stale-DB-stop cleanup:
    that cleanup used to just delete a position_stops row once its ticker
    was no longer held, with no attempt to find out *why* it disappeared.
    DDOG (2026-08-06) went from 84 shares to 0 with no bot-initiated sell on
    record and no confirmed fill on its OWN tracked resting stop order
    (which sat orphaned, uncancellable -- a different clientId owned it --
    for the rest of that day) -- the position just vanished from every DB
    table with no trade or trade_outcome record, and the cleanup logged it
    as routine "deleted stale DB stop record" noise.

    Searching by symbol (matching get_fill_for_order's already-verified-
    working reqExecutions() call, just filtered client-side on
    contract.symbol + execution.side instead of orderId -- see that
    function's docstring for why an ExecutionFilter isn't used here either)
    catches an exit through ANY sell order for this ticker -- the tracked
    stop, a backstop sell that somehow wasn't logged, or a manual sale
    placed directly in TWS -- not just the one order this system happened
    to be watching. Same same-trading-day cap as get_fill_for_order
    (confirmed live 2026-08-04): this only finds an exit that happened
    earlier today. Anything older needs a Flex Query or a manual
    broker-statement pull, same as that function.
    """
    ib = get_ib()
    execs = ib.reqExecutions()
    sells = [e for e in execs if e.contract.symbol == ticker and e.execution.side == "SLD"]
    if not sells:
        return None
    total_qty = sum(float(e.execution.shares) for e in sells)
    if total_qty <= 0:
        return None
    avg_price = sum(float(e.execution.shares) * float(e.execution.avgPrice) for e in sells) / total_qty
    return {"filled_qty": total_qty, "fill_price": avg_price}


# ---------------------------------------------------------------------------
# Alpaca-compatible adapter (used by RiskManager and tasks)
# ---------------------------------------------------------------------------

class _IBKRAccount:
    """Mirrors Alpaca's TradeAccount interface."""
    def __init__(self, portfolio_value: float, buying_power: float, cash: float):
        self.portfolio_value = portfolio_value
        self.buying_power    = buying_power
        self.cash            = cash


class _IBKRPosition:
    """Mirrors Alpaca's Position interface."""
    def __init__(
        self,
        symbol: str,
        qty: float,
        market_value: float,
        avg_entry_price: float,
        current_price: float,
        unrealized_pl: float,
    ):
        self.symbol          = symbol
        self.qty             = qty
        self.market_value    = market_value
        self.avg_entry_price = avg_entry_price
        self.current_price   = current_price
        self.unrealized_pl   = unrealized_pl


class IBKRClient:
    """
    Drop-in adapter for Alpaca's TradingClient.
    Exposes get_account(), get_all_positions(), get_open_position()
    so RiskManager and tasks work without further changes.
    """

    def get_account(self) -> _IBKRAccount:
        ib   = get_ib()
        vals = {}
        for v in ib.accountValues():
            if v.currency == "USD":
                vals[v.tag] = v.value
        # CashBalance silently stopped being sent by IBKR for this account at
        # some point (found 2026-07-23: it came back missing entirely, not
        # just zero) — vals.get(..., 0) doesn't distinguish "tag absent" from
        # "genuinely zero cash", so this defaulted to $0 with no error while
        # the account actually held $847K, silently blocking every new BUY's
        # risk sizing (20% of a $0 reserve is $0) for an unknown stretch of
        # time. TotalCashValue is IBKR's more universally-sent equivalent —
        # preferred now, with CashBalance kept as a fallback in case some
        # account mode sends that one instead.
        cash_str = vals.get("TotalCashValue", vals.get("CashBalance", 0))
        return _IBKRAccount(
            portfolio_value = float(vals.get("NetLiquidation", 0)),
            buying_power    = float(vals.get("BuyingPower",    0)),
            cash            = float(cash_str),
        )

    def get_all_positions(self) -> list[_IBKRPosition]:
        ib = get_ib()
        return [
            _IBKRPosition(
                symbol          = item.contract.symbol,
                qty             = float(item.position),
                market_value    = float(item.marketValue),
                avg_entry_price = float(item.averageCost),
                current_price   = float(item.marketPrice),
                unrealized_pl   = float(item.unrealizedPNL),
            )
            for item in ib.portfolio()
            if item.contract.secType == "STK" and float(item.position) != 0
        ]

    def get_open_position(self, symbol: str) -> _IBKRPosition:
        """Raises Exception if no position exists — matches Alpaca behaviour."""
        for pos in self.get_all_positions():
            if pos.symbol == symbol:
                return pos
        raise Exception(f"No open position in {symbol}")


# Module-level singleton — reuse across tasks
_client: IBKRClient | None = None


def get_client() -> IBKRClient:
    global _client
    if _client is None:
        _client = IBKRClient()
    return _client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    acct   = client.get_account()
    print(f"Portfolio value: ${acct.portfolio_value:,.2f}")
    print(f"Cash:            ${acct.cash:,.2f}")
    print(f"Buying power:    ${acct.buying_power:,.2f}")
    print()
    positions = client.get_all_positions()
    if positions:
        print(f"{'Symbol':<8} {'Qty':>6} {'Mkt Value':>12} {'Unreal P&L':>12}")
        for p in positions:
            print(f"{p.symbol:<8} {p.qty:>6.0f} ${p.market_value:>11,.2f} ${p.unrealized_pl:>11,.2f}")
    else:
        print("No open positions.")
