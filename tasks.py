"""
Celery async task definitions.
Each ingestion source runs as a periodic Celery beat task,
firing events into the agent pipeline without blocking the main thread.

Requires Redis running locally (via Docker):
  docker run -d -p 6379:6379 redis:alpine

Start worker:   celery -A tasks worker --loglevel=info --pool=solo
Start scheduler: celery -A tasks beat --loglevel=info
"""

import json
import logging
import os
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

import log_redaction
import stop_lock
from stop_lock import locked_by_ticker, ticker_lock

log_redaction.install()  # bot token / account id out of every worker log
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery("helios", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer        = "json",
    accept_content         = ["json"],
    result_serializer      = "json",
    timezone               = "America/New_York",
    enable_utc             = True,
    task_track_started     = True,
    worker_prefetch_multiplier = 1,  # one task at a time — model inference is heavy
    # Two-lane queue architecture. Everything defaults to "ops" (fast I/O:
    # pollers, snapshots, stop checks); LLM analysis is dispatched explicitly
    # to the llm_p* queues and consumed by a dedicated single-concurrency
    # worker so a 3-minute 72B call can never starve stop monitoring again.
    task_default_queue     = "ops",
    # "sorted" makes a worker consuming multiple queues drain them in name
    # order — llm_p0 (events touching currently-held tickers) strictly before
    # llm_p1 (everything else). The names are chosen to sort that way.
    #
    # visibility_timeout: incident 2026-07-30 -- tasks.retry_deferred_entry is
    # scheduled with an eta up to several days out (a Friday-evening deferral
    # retries at Monday's open). Celery's Redis transport tracks a delivered-
    # but-not-yet-executed eta task as "unacked" the whole time it's waiting
    # in the worker's internal timer, regardless of task_acks_late -- if
    # visibility_timeout (default 3600s/1h) elapses before the eta arrives,
    # Redis assumes the message was lost and redelivers it. With the 1h
    # default, an evening deferral got redelivered roughly once an hour
    # overnight, so by market open several duplicate copies of the same
    # approval fired back-to-back (5+ redundant "ORDER ABORTED" alerts for
    # one ticker, and a real risk of two copies executing concurrently).
    # Set well past the longest eta this app schedules (next_market_open()
    # across a weekend), with margin for a long weekend/holiday.
    broker_transport_options = {
        "queue_order_strategy": "sorted",
        "visibility_timeout": 6 * 24 * 3600,
    },
)

# LLM analysis queues — see task_default_queue comment above.
LLM_QUEUE_HELD     = "llm_p0"   # events about tickers we already hold — front of the line
LLM_QUEUE_BULK     = "llm_p1"   # scanner candidates and non-held-ticker events
# Chat-requested "analyze TICKER for me" research (see agents.crew.research_
# ticker) — always last: a curiosity question must never delay analysis of
# anything actually held or an organic news-driven idea. Name sorts after
# llm_p1 under queue_order_strategy=sorted above, same worker/concurrency.
LLM_QUEUE_RESEARCH = "llm_p2"

# External-data pollers (EDGAR, options flow, macro, arXiv, congressional,
# insider, analyst ratings, news, fundamentals, price history) get their own
# queue/worker, separate from "ops" — added 2026-08-18 after poll_edgar was
# found taking 28-31 minutes per run (SEC's servers degraded, "Read timed
# out" on nearly every request) while sharing ops's 3 workers with
# check_portfolio_drawdown, take_portfolio_snapshot, reconcile_stop_orders,
# and auto_expire_approvals. With prefetch=1, one worker stuck 30 minutes on
# a slow EDGAR poll is a third of ops's capacity gone; several of these
# tasks piling up at once backed the whole queue up past every other
# scheduled task's `expires`, which Celery just silently drops rather than
# running late -- 752 "Discarding revoked task" events in a single day,
# spanning task types with no ingestion involvement at all (check_ollama_
# gpu_health, check_portfolio_drawdown, auto_expire_approvals among them).
# The visible symptom was hourly repeat Telegram alerts from
# check_ingestion_health ("options_flow: last success 48min ago, expected
# <15min") that never resolved, because poll_options_flow itself kept
# getting starved the same way. Same fix shape as the llm_p*/screen split
# below: an unreliable external data source must never be able to starve
# safety-critical or fast local tasks just by sharing a worker pool.
INGESTION_QUEUE = "ingestion"

# Screening (7B, CPU-forced) gets its own queue/worker, separate from the
# llm_p* queues (72B, GPU-bound) — added 2026-07-16 after a busy news day
# backed llm_p1 up past 700 items. Before this, screen+analyze ran as one
# task on the single-concurrency llm worker, so a quick reject still had
# to wait behind however many ~170s analyses were already ahead of it in
# queue — the screening *verdict* was as backed up as the analysis itself.
# Now only genuine survivors (tasks.screen_event -> tasks.deep_analyze_event)
# ever reach llm_p0/llm_p1; screening runs at its own concurrency and never
# waits on the 72B stage.
SCREEN_QUEUE = "screen"
SCREEN_EVENT_EXPIRES_SECONDS = 2 * 3600

# An event analyzed hours late is analyzing a stale market — discard instead.
LLM_EVENT_EXPIRES_SECONDS = 4 * 3600

# ── Order lifecycle (the ONE clientId that ever touches a stop order) ──────────
# 2026-08-11: MRVL/SNDK/VRT/ORCL/DDOG were all the same root cause. A stop got
# PLACED under whichever process handled the entry (celery_llm_worker,
# clientId=20, or main.py's Telegram path, clientId=1), but every LATER touch
# — trailing-stop conversion, orphan cleanup, a manual sell's pre-cancel — ran
# from celery_worker's ops pool (clientId 10/11/12, whichever of its 3 forks
# happened to pick up the task). IBKR silently ignores a cancel/modify from a
# clientId that didn't place the order, so that mismatch is guaranteed, not
# intermittent, for every stop this system has ever placed. Confirmed live
# that clientId=0's documented "master client" status does not actually grant
# cancel rights over another client's order in this account/TWS config either
# — the cancel *looked* to succeed locally but a neutral third connection
# showed the order still resting at the broker afterward, the same
# looks-cancelled-isn't shape ibkr.cancel_order was hardened against for the
# MRVL incident.
#
# Fix: exactly one process (celery_orders_worker, IB_CLIENT_ID=30, -c 1 so
# that base is never fork-offset) is allowed to place or cancel a stop.
# Everything else calls the three functions below instead of ibkr.py's
# placement/cancel functions directly. Each checks whether it's ALREADY
# running inside that one process (cheap: just the env var process_manager.py
# sets for it) — if so, call the broker directly; if not, dispatch a task to
# the "orders" queue and wait for the result. The direct-call branch also
# means celery_orders_worker's own tasks never call back into themselves.
from celery.signals import beat_init, task_failure, task_revoked, worker_init


def _boot_service_name(sender) -> str:
    """'celery:<node name before @>' for a worker (ops, llm, orders...), so
    the host name isn't recorded."""
    hostname = getattr(sender, "hostname", None) or ""
    return f"celery:{hostname.split('@')[0] or 'worker'}"


@worker_init.connect
def _pin_code_stamp(sender=None, **_):
    """Import the analyst (which reads the commit stamp) in the worker's
    parent at boot, before the pool forks, so the stamp and the analyst code
    come from the files present at start, not whatever is on disk at the
    first analysis (professor M75 a). Then write the boot row (M181)."""
    try:
        import agents.analyst  # noqa: F401
    except Exception as e:
        log.warning(f"could not pre-import agents.analyst at worker start: {e}")
        return
    try:
        agents.analyst.record_boot(_boot_service_name(sender))
    except Exception as e:
        log.warning(f"boot row not written: {e}")


@beat_init.connect
def _record_beat_boot(**_):
    """The beat scheduler's boot row (M181)."""
    try:
        import agents.analyst
        agents.analyst.record_boot("celery:beat")
    except Exception as e:
        log.warning(f"beat boot row not written: {e}")


ORDERS_QUEUE = "orders"
_ORDERS_WORKER_CLIENT_ID = "30"


def _is_orders_worker() -> bool:
    return os.getenv("IB_CLIENT_ID") == _ORDERS_WORKER_CLIENT_ID


# Every broker task takes the caller's deadline (epoch seconds) and refuses
# to act once it has passed, so a request the caller has given up on can't
# act afterwards. The caller then waits on the task's real result instead
# of guessing from open orders (professor C21/C22, 2026-10-29: a slow cancel
# judged "failed" from a 5 s open-orders check still ran later and left the
# position naked; matching on qty/price could adopt a stuck old stop).
class OrdersCallExpired(Exception):
    """The orders worker refused the request: it arrived after its deadline,
    so nothing was sent to the broker."""


class OrdersCallUnknown(Exception):
    """No result from the orders worker even after the extra wait: the broker
    call may or may not have happened."""


_EXPIRED = {"_expired": True}


def _past(deadline) -> bool:
    import time
    return deadline is not None and time.time() > deadline


_STARTED_KEY = "helios:orders:started:{}"


def _mark_started(task) -> bool:
    """Record that the orders worker picked this request up. Written before
    the deadline check, so a missing marker after the deadline means the
    worker never touched it and will refuse it when it does (A0253-9).
    Returns False if it can't be written; the task then refuses, since a
    lost reply would otherwise read as "never started" (professor M68)."""
    try:
        import stop_lock
        stop_lock._redis().set(_STARTED_KEY.format(task.request.id), "1", ex=900)
        return True
    except Exception as e:
        log.error(f"{task.name}: could not write started marker, refusing: {e}")
        return False


def _was_started(task_id: str) -> bool | None:
    """True/False from the started marker; None if Redis can't be read."""
    try:
        import stop_lock
        return bool(stop_lock._redis().exists(_STARTED_KEY.format(task_id)))
    except Exception:
        return None


@app.task(name="tasks._broker_place_stop_order", bind=True, max_retries=1)
def _broker_place_stop_order(self, ticker: str, qty: float, stop_price: float,
                             deadline: float | None = None,
                             market_price: float | None = None) -> dict:
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    if market_price is None:
        return ibkr.place_stop_order(ticker, qty, stop_price)
    return ibkr.place_stop_order(ticker, qty, stop_price, market_price=market_price)


@app.task(name="tasks._broker_cancel_stop_orders_for", bind=True, max_retries=1)
def _broker_cancel_stop_orders_for(self, ticker: str, deadline: float | None = None):
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    return ibkr.cancel_stop_orders_for(ticker)


@app.task(name="tasks._broker_cancel_order", bind=True, max_retries=1)
def _broker_cancel_order(self, order_id: int, deadline: float | None = None,
                         symbol: str | None = None, perm_id: int | None = None):
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    match = {k: v for k, v in (("symbol", symbol), ("perm_id", perm_id)) if v}
    return ibkr.cancel_order(order_id, **match)


@app.task(name="tasks._broker_place_trailing_stop_order", bind=True, max_retries=1)
def _broker_place_trailing_stop_order(self, ticker: str, qty: float,
                                       trail_amount: float | None = None,
                                       trail_percent: float | None = None,
                                       deadline: float | None = None) -> dict:
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    return ibkr.place_trailing_stop_order(ticker, qty, trail_amount=trail_amount,
                                           trail_percent=trail_percent)


# How long past its own timeout a caller keeps waiting for a broker task it
# has already handed off. The task can't start after the deadline, so this
# only covers one that started in time and is still running (a slow cancel:
# 5 s polls per order plus reconnect backoff).
_ORDERS_EXTRA_WAIT_S = 120


def _orders_call(task, args, kwargs=None, timeout: int = 30):
    """Run a broker task on the orders worker and return its real result.
    Raises OrdersCallExpired if it was refused for arriving late (nothing
    sent), OrdersCallUnknown if no result came even after the extra wait."""
    import time
    from celery.exceptions import TaskRevokedError
    from celery.exceptions import TimeoutError as CeleryTimeout
    from celery.result import allow_join_result
    deadline = time.time() + timeout
    with allow_join_result():
        res = task.apply_async(args=args, kwargs={**(kwargs or {}), "deadline": deadline},
                               queue=ORDERS_QUEUE, expires=timeout)
        try:
            try:
                out = res.get(timeout=timeout)
            except CeleryTimeout:
                log.warning(f"{task.name}: no reply in {timeout}s; waiting for the result")
                out = res.get(timeout=_ORDERS_EXTRA_WAIT_S)
        except TaskRevokedError as e:
            raise OrdersCallExpired(f"{task.name} expired before it ran") from e
        except CeleryTimeout as e:
            # Never picked up (the worker was busy) is not "maybe sent": it
            # will be refused when the worker reaches it (A0253-9).
            if _was_started(res.id) is False and getattr(res, "state", None) == "PENDING":
                raise OrdersCallExpired(f"{task.name} never started (orders worker busy); "
                                        f"nothing sent") from e
            raise OrdersCallUnknown(f"{task.name}: no result after "
                                    f"{timeout + _ORDERS_EXTRA_WAIT_S}s") from e
    if out == _EXPIRED:
        raise OrdersCallExpired(f"{task.name} reached the worker after its deadline")
    return out


# Order states that are no longer protecting anything (ib_async OrderStatus).
# PendingCancel is NOT one of them: IBKR can reject the cancel and the stop
# reverts to working minutes later (MRVL 08-04, ibkr.cancel_order), so it
# still counts as resting for every "safe to sell / place?" check (audit
# 02:53, A0253-1).
_NOT_WORKING = {"Cancelled", "ApiCancelled", "Inactive", "Filled"}


def _resting_sell_stops(ticker: str, types=("STP", "STP LMT", "TRAIL")) -> list[dict]:
    """Working SELL stops for ticker, with qty = the unfilled remainder
    (professor M64: totalQuantity counted part-filled, cancelling and
    inactive stops as coverage)."""
    import ibkr
    return [o for o in _working_sell_stops(ibkr.get_open_orders(), types) if o["ticker"] == ticker]


def _working_mkt_sell_qty(open_orders: list[dict], ticker: str) -> float:
    """Unfilled shares of working market SELLs for ticker. Those shares are
    already being sold, so a stop sized for them as well would short once
    both fill (professor M129; same rule as the backstop, 44b889a)."""
    return sum(float(o["qty"]) for o in _working_sell_stops(open_orders, ("MKT",)) if o["ticker"] == ticker)


def _working_sell_stops(open_orders: list[dict], types=("STP", "STP LMT", "TRAIL")) -> list[dict]:
    """The one coverage definition: SELL stops not in a dead state, with
    qty = the unfilled remainder. Shared by _resting_sell_stops and
    reconcile_stop_orders sections 2b and 3 (audit 02:53, A0253-6)."""
    out = []
    for o in open_orders:
        if (o["action"] != "SELL" or o["order_type"] not in types
                or o.get("status") in _NOT_WORKING):
            continue
        rem = o.get("remaining")
        out.append({**o, "qty": float(rem) if rem not in (None, 0) else float(o["qty"])})
    return out


def manage_stop_place(ticker: str, qty: float, stop_price: float, timeout: int = 30,
                      market_price: float | None = None) -> dict:
    """Place a protective stop — see the module-level comment above for why
    this exists instead of calling ibkr.place_stop_order directly. Every
    caller anywhere in the codebase should go through this, including code
    that happens to already be running in the orders worker (that branch is
    just a direct call, essentially free) — the point is there being exactly
    one way to do this, so a future direct ibkr.place_stop_order call can't
    quietly reintroduce the bug.

    A fixed STP replaces any native TRAIL the position had (callers cancel
    stops first), so trailing_active is cleared: check_trailing_stops then
    re-arms the trail on its next run if the gain still qualifies. Before
    2026-10-28 a partial sale, TRIM or failed-sell restore left the flag TRUE
    and the trail never came back (limitations audit 12:53, finding 3).
    Callers that convert to a trail set it again via update_trailing_stop.
    market_price: the last price where the caller has it, so the tick
    round-up in ibkr.place_stop_order can't reach the market."""
    stop_lock.assert_held(ticker)  # professor M69
    mp = {"market_price": market_price} if market_price else {}
    if _is_orders_worker():
        import ibkr
        result = ibkr.place_stop_order(ticker, qty, stop_price, **mp)
    else:
        result = _orders_call(_broker_place_stop_order, [ticker, qty, stop_price], mp or None,
                              timeout=timeout)
    try:
        import db
        db.clear_trailing_active(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] could not clear trailing_active after a fixed stop: {e}")
    return result


def manage_cancel_stop_orders_for(ticker: str, timeout: int = 30) -> int:
    """Cancel all resting stop orders for a ticker — see manage_stop_place."""
    stop_lock.assert_held(ticker)  # professor M69
    if _is_orders_worker():
        import ibkr
        return ibkr.cancel_stop_orders_for(ticker)
    return _orders_call(_broker_cancel_stop_orders_for, [ticker], timeout=timeout)


def manage_place_trailing_stop(ticker: str, qty: float, trail_amount: float | None = None,
                               trail_percent: float | None = None, timeout: int = 30) -> dict:
    """Place a native TRAIL stop (the trailing-stop conversion path) — see
    manage_stop_place. Exactly one of trail_amount/trail_percent, matching
    ibkr.place_trailing_stop_order's own contract."""
    stop_lock.assert_held(ticker)  # professor M69
    if _is_orders_worker():
        import ibkr
        return ibkr.place_trailing_stop_order(ticker, qty, trail_amount=trail_amount,
                                               trail_percent=trail_percent)
    return _orders_call(_broker_place_trailing_stop_order, [ticker, qty],
                        {"trail_amount": trail_amount, "trail_percent": trail_percent},
                        timeout=timeout)


def manage_cancel_order(order_id: int, timeout: int = 30, symbol: str | None = None,
                        perm_id: int | None = None) -> bool:
    """Cancel a specific order by id — see manage_stop_place. Also used for
    the stale-market-BUY-order cleanup at startup (main.py), not just stops:
    the same ownership rule applies to any order type, not only STP.
    symbol / perm_id: when given, ibkr.cancel_order refuses an order with
    this id on another ticker or with another permId (orderIds repeat
    across clientIds). Sent as keywords only when set, so a message without
    them still matches the worker task."""
    match = {k: v for k, v in (("symbol", symbol), ("perm_id", perm_id)) if v}
    if _is_orders_worker():
        import ibkr
        return ibkr.cancel_order(order_id, **match)
    return _orders_call(_broker_cancel_order, [order_id], match or None, timeout=timeout)


# ── Periodic schedule ─────────────────────────────────────────────────────────
# Every entry carries "expires": a task not picked up within that window is
# discarded by the worker instead of executed late. Interval tasks expire
# after one interval (running a 4-hour-old portfolio snapshot after fresher
# ones is pure waste — this is what let the queue back up 296 deep on
# 2026-07-15). Daily/weekly tasks get generous expiries sized to the
# machine's sleep window (stack down 19:00→05:00 weekdays, all weekend), so
# beat's catch-up send on wake still executes rather than being discarded.
app.conf.beat_schedule = {
    # Market regime — daily at 9:30 AM ET (market open). LLM-heavy → llm
    # lane, priority: it gates the day's sizing multiplier.
    "assess-regime": {
        "task":     "tasks.assess_regime_daily",
        "schedule": crontab(hour=9, minute=30, day_of_week="mon-fri"),
        "options":  {"queue": LLM_QUEUE_HELD, "expires": 6 * 3600},
    },
    # Daily scanner — 9:35 AM ET (5 min after regime). The scan itself is
    # fast (ops); it enqueues analyze_candidates onto the llm lane.
    "daily-scan": {
        "task":     "tasks.run_daily_scan",
        "schedule": crontab(hour=9, minute=35, day_of_week="mon-fri"),
        "options":  {"expires": 4 * 3600},
    },
    # Full GICS sector map — 9:45 AM ET, after the overall regime/scan.
    # Pure price/volume signals across all 11 sector ETFs, no LLM involved
    # -- cheap enough for ops queue. Gives the "how's the whole market
    # rotating" view the AI/tech-only regime above can't (see
    # assess_market_sector_map's docstring).
    "assess-gics-sector-map": {
        "task":     "tasks.assess_gics_sector_map",
        "schedule": crontab(hour=9, minute=45, day_of_week="mon-fri"),
        "options":  {"expires": 4 * 3600},
    },
    # Watchlist-candidate discovery — weekly, Monday 9:00 AM ET. Scheduled
    # after refresh-sp500-history's Monday 05:00 catch-up (see that entry's
    # comment) so this reads freshly-populated bars, not week-old ones from
    # before the catch-up ran -- now a 4h buffer instead of 15min since the
    # wake moved from 08:45 to 05:00.
    "candidate-scan": {
        "task":     "tasks.run_candidate_scan",
        "schedule": crontab(hour=9, minute=0, day_of_week="mon"),
        "options":  {"expires": 6 * 3600},
    },
    # Position review — reassess already-held positions for TRIM/EXIT/HOLD,
    # independent of whether a new event happens to fire on one. Added
    # 2026-07-25: this was the actual gap behind the system never selling —
    # _route_to_approval's TRIM/EXIT handling already worked correctly,
    # nothing ever triggered it on a schedule (see agents/crew.py's
    # review_all_held_positions docstring). 10:15 AM, after the daily scan,
    # so new-candidate analysis isn't waiting behind it.
    "position-review": {
        "task":     "tasks.review_positions",
        "schedule": crontab(hour=10, minute=15, day_of_week="mon-fri"),
        "options":  {"queue": LLM_QUEUE_HELD, "expires": 6 * 3600},
    },
    # Check trailing stops — every 30 min during market hours
    "check-stops": {
        "task":     "tasks.check_trailing_stops",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # Regime-adaptive profit-taking — every 30 min, same cadence as
    # check-stops above (2026-08-29, user request; see config.py's
    # PROFIT_TARGET_PCT_BY_STYLE and risk_manager.check_profit_targets).
    # This is the step that makes that feature actually place live SELL
    # orders on a schedule, not just be present-but-dormant code.
    "check-profit-targets": {
        "task":     "tasks.check_profit_targets",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # SEC filings — every 10 minutes during market hours
    "poll-edgar": {
        "task":     "tasks.poll_edgar",
        "schedule": 600,  # 10 minutes
        "options":  {"queue": INGESTION_QUEUE, "expires": 600},
    },
    # Options flow — every 5 minutes during market hours
    "poll-options": {
        "task":     "tasks.poll_options_flow",
        "schedule": 300,
        "options":  {"queue": INGESTION_QUEUE, "expires": 300},
    },
    # Macro data snapshot — every hour
    "poll-macro": {
        "task":     "tasks.poll_macro",
        "schedule": 3600,
        "options":  {"queue": INGESTION_QUEUE, "expires": 3600},
    },
    # arXiv papers — once per day at 9 AM ET
    "poll-arxiv": {
        "task":     "tasks.poll_arxiv",
        "schedule": crontab(hour=9, minute=0),
        "options":  {"queue": INGESTION_QUEUE, "expires": 24 * 3600},
    },
    # Congressional trades — every 6 hours
    "poll-congressional": {
        "task":     "tasks.poll_congressional",
        "schedule": 21600,
        "options":  {"queue": INGESTION_QUEUE, "expires": 21600},
    },
    # Insider trades — every 30 minutes
    "poll-insider": {
        "task":     "tasks.poll_insider",
        "schedule": 1800,
        "options":  {"queue": INGESTION_QUEUE, "expires": 1800},
    },
    # Analyst rating changes (yfinance, free) — every 30 minutes, same
    # cadence as insider (both are "check for something new," not
    # continuous streams).
    "poll-analyst-ratings": {
        "task":     "tasks.poll_analyst_ratings",
        "schedule": 1800,
        "options":  {"queue": INGESTION_QUEUE, "expires": 1800},
    },
    # Portfolio snapshot — every 5 minutes during market hours
    "portfolio-snapshot": {
        "task":     "tasks.take_portfolio_snapshot",
        "schedule": 300,
        "options":  {"expires": 300},
    },
    # Shadow plan level watcher, log-only (audit A2114-4) — every 15 minutes
    "watch-plan-levels": {
        "task":     "tasks.watch_plan_levels",
        "schedule": 900,
        "options":  {"queue": INGESTION_QUEUE, "expires": 900},
    },
    # Flag open outcome rows with no position (professor M111) — weekdays
    # 15:45, while the 5-minute portfolio snapshot is fresh
    "flag-orphan-outcomes": {
        "task":     "tasks.flag_orphan_outcomes",
        "schedule": crontab(hour=15, minute=45, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 3600},
    },
    # Auto-expire stale approval requests — every 30 minutes
    "expire-approvals": {
        "task":     "tasks.auto_expire_approvals",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # Earnings calendar refresh — Monday 8:00 AM ET. Used to fire before
    # the old 08:45 wake and need beat's catch-up; now runs live, 3h after
    # the 05:00 wake.
    "refresh-earnings": {
        "task":     "tasks.refresh_earnings_calendar",
        # Daily: weekly left moved dates stale for days in earnings season (A1453-2).
        "schedule": crontab(hour=8, minute=0, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 24 * 3600},
    },
    # Macro calendar (FOMC/CPI/jobs/election) — weekly is plenty since the
    # underlying schedule is fixed months out; idempotent upsert either way.
    "refresh-macro-calendar": {
        "task":     "tasks.refresh_macro_calendar",
        "schedule": crontab(hour=8, minute=5, day_of_week="mon"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 24 * 3600},
    },
    # Sector earnings-cluster detection — after refresh-earnings (8:00) so
    # this week's per-ticker dates are already fresh.
    "refresh-sector-calendar": {
        "task":     "tasks.refresh_sector_calendar",
        "schedule": crontab(hour=8, minute=15, day_of_week="mon"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 24 * 3600},
    },
    # Catalyst conflicts — weekdays 8:10 AM ET (pre-wake; catch-up on wake)
    "alert-catalyst-conflicts": {
        "task":     "tasks.alert_catalyst_conflicts",
        "schedule": crontab(hour=8, minute=10, day_of_week="mon-fri"),
        "options":  {"expires": 6 * 3600},
    },
    # Portfolio drawdown circuit breaker — every 30 minutes
    "check-drawdown": {
        "task":     "tasks.check_portfolio_drawdown",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # Stop-order reconciliation — every 15 minutes (orphan detection/cleanup)
    "reconcile-stops": {
        "task":     "tasks.reconcile_stop_orders",
        "schedule": 900,
        "options":  {"expires": 900},
    },
    # Late-fill reconciliation — every 5 minutes. Catches orders that filled
    # after place_market_order's wait window gave up (see
    # tasks.reconcile_unconfirmed_orders docstring for the 2026-07-28 CRWD
    # incident this closes); short interval since an unprotected live
    # position is the failure mode.
    #
    # 2026-08-11: expires used to be 280 -- tighter than this task's own
    # 300s schedule, unlike every other periodic safety task here (they all
    # use expires == their own schedule interval, e.g. reconcile-stops
    # below is 900/900, check-stops is 1800/1800). The idea was "a delayed
    # run shouldn't stack behind another," but Celery's expires just DROPS
    # an expired task outright rather than collapsing duplicates -- found
    # this was actually discarding a large fraction of runs (~24% on a
    # sampled trading day, ~60% over a longer window) whenever the shared
    # ops queue fell even slightly behind, not just in rare pileups. A NET
    # order sat unreconciled for over a day because of this -- every single
    # scheduled check for it that day was either dropped by this expiry or
    # hit a real IBKR connectivity gap, and the two together meant zero
    # actual reconciliation attempts landed. Widened well past the 5-minute
    # schedule: a late run of this specific task is cheap and idempotent
    # (it only acts on orders still genuinely unconfirmed), so "eventually
    # correct" beats "silently skipped" here.
    "reconcile-unconfirmed-orders": {
        "task":     "tasks.reconcile_unconfirmed_orders",
        "schedule": 300,
        "options":  {"expires": 3600},
    },
    # Low-frequency overnight-dead-with-no-traffic safety net — every 30
    # minutes, not 10. The 72B server only handles one request at a time
    # (-np 1), and this probe occupies that slot for up to 300s each run;
    # at 10-minute intervals it was itself a meaningful source of the
    # contention it was meant to catch (found 2026-07-21). The primary
    # detection path is now agents.analyst._stub_recommendation, which
    # reacts to real production timeouts instead of adding synthetic load.
    "check-ollama-gpu": {
        "task":     "tasks.check_ollama_gpu_health",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # ChromaDB collection integrity — hourly. The 2026-07-17 write-lock fix
    # (vector_store._write_lock) stopped concurrent-write races from
    # corrupting a collection, but a process that crashes/gets killed
    # mid-write can still leave one corrupted regardless of locking — and
    # a corrupted collection segfaults the whole process on a bare
    # col.count(), which is exactly what happened again on 2026-07-21 to
    # the same "news" collection, three days after the lock fix. This
    # checks each collection in an isolated subprocess (so a crash there
    # can't take down a real worker) and auto-rebuilds anything corrupted.
    "check-vector-store": {
        "task":     "tasks.check_vector_store_health",
        "schedule": 3600,
        "options":  {"expires": 3600},
    },
    # 30/60/90-day outcome price reviews — weekdays 4:30 PM ET
    "review-outcomes": {
        "task":     "tasks.review_trade_outcomes",
        "schedule": crontab(hour=16, minute=30, day_of_week="mon-fri"),
        "options":  {"expires": 24 * 3600},
    },
    # Vector store pruning — Sunday 3:00 AM. Stack is down all weekend;
    # actually runs via catch-up Monday 05:00 (~26h later). Kept there:
    # moving it changes which memories the analyst's brief can retrieve on
    # Mondays (analyst input; professor m164), so that's not ours to move.
    "prune-vector-store": {
        "task":     "tasks.prune_vector_store",
        "schedule": crontab(hour=3, minute=0, day_of_week="sun"),
        "options":  {"expires": 36 * 3600},
    },
    # seen_items dedup table (not analyst input) — Friday evening.
    "prune-seen-items": {
        "task":     "tasks.prune_seen_items",
        "schedule": crontab(hour=17, minute=50, day_of_week="fri"),
        "options":  {"expires": 72 * 3600},
    },
    # News (yfinance + Google News RSS) — every 15 minutes
    # IPO watch (Luke, 2026-10-28): is a pending company listed yet?
    "ipo-watch": {
        "task":     "tasks.check_ipo_watch",
        "schedule": crontab(hour=8, minute=20, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 6 * 3600},
    },
    "poll-news": {
        "task":     "tasks.poll_news",
        "schedule": 900,
        "options":  {"queue": INGESTION_QUEUE, "expires": 900},
    },
    # Fundamentals refresh — 5:00 PM ET. Must run inside the awake window
    # (currently 05:00-19:00): the old 2:00 AM slot predated even the
    # original 08:45 wake, so it never fired at all — the fundamentals
    # table stayed empty forever, which silently capped every analyst
    # conviction at 6 via the ungrounded-valuation guardrail.
    "refresh-fundamentals": {
        "task":     "tasks.refresh_fundamentals",
        "schedule": crontab(hour=17, minute=0, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 2 * 3600},  # must finish before the 19:00 stack-down
    },
    # Price history cache refresh — 4:45 PM ET (before fundamentals, after
    # close; same awake-window constraint as above).
    "refresh-price-history": {
        "task":     "tasks.refresh_price_history",
        "schedule": crontab(hour=16, minute=45, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 2 * 3600},
    },
    # Trade ladder refresh — 5:15 PM ET, after BOTH refresh-price-history
    # (16:45) and refresh-fundamentals (17:00) so it reads same-day-fresh
    # bars and multiples. Ops queue, no LLM involved -- must finish before
    # the 19:00 stack-down like its two inputs above.
    "refresh-trade-ladder": {
        "task":     "tasks.refresh_trade_ladder",
        "schedule": crontab(hour=17, minute=15, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 2 * 3600},
    },
    # S&P 500 universe price history — weekly, Saturday 9:00 AM. The stack
    # is down all weekend, so it actually runs via the Monday 05:00 catch-up
    # (~44h later), which is fine —
    # feeds GICS-sector breadth + candidate_scanner, neither of which needs
    # day-fresh data (see refresh_sp500_history's docstring).
    "refresh-sp500-history": {
        "task":     "tasks.refresh_sp500_history",
        "schedule": crontab(hour=9, minute=0, day_of_week="sat"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 48 * 3600},
    },
    # Weekday catch-up for the above: only tickers more than 7 days stale,
    # so it's a no-op after a completed weekly run (audit A1031-0053-12).
    "refresh-sp500-history-catchup": {
        "task":     "tasks.refresh_sp500_history_catchup",
        "schedule": crontab(hour=5, minute=15, day_of_week="mon-fri"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 3600},
    },
    # Broadened Comps peer universe — Saturday 9:30 AM, after
    # refresh-sp500-history so this reads the same-week price data.
    # Same down-all-weekend caveat as its sibling above (Monday 05:00
    # catch-up).
    "refresh-sp500-fundamentals": {
        "task":     "tasks.refresh_sp500_fundamentals",
        "schedule": crontab(hour=9, minute=30, day_of_week="sat"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 48 * 3600},
    },
    # Ingestion health check — hourly
    "check-ingestion-health": {
        "task":     "tasks.check_ingestion_health",
        "schedule": 3600,
        "options":  {"expires": 3600},
    },
    # Low-severity alert repeats held back by alert_policy, sent as one
    # message — hourly at :50 while the machine is awake. Critical alerts
    # never wait for this.
    "flush-alert-digest": {
        "task":     "tasks.flush_alert_digest",
        "schedule": crontab(minute=50, hour="6-18"),
        "options":  {"expires": 1800},
    },
    # Daily digest — weekdays 4:15 PM ET (after market close). A digest
    # delivered the next morning is noise — short expiry.
    "daily-digest": {
        "task":     "tasks.send_daily_digest",
        "schedule": crontab(hour=16, minute=15, day_of_week="mon-fri"),
        "options":  {"expires": 3 * 3600},
    },
    # Weekly self-review — Friday 18:00, after the week's last close and
    # before the 19:00 stop (~1 min run). It was Sunday 18:00, which ran at
    # Monday 05:00 against pre-market analyses (A1031-1253-7). LLM-heavy →
    # llm bulk lane; 72h expiry so a missed Friday runs at Monday's start.
    "weekly-self-review": {
        "task":     "tasks.weekly_self_review",
        "schedule": crontab(hour=18, minute=0, day_of_week="fri"),
        "options":  {"queue": LLM_QUEUE_BULK, "expires": 72 * 3600},
    },
}

from config import QUIVER_ENABLED as _QUIVER_ENABLED
if not _QUIVER_ENABLED:   # Luke D2, 10-29
    app.conf.beat_schedule.pop("poll-congressional", None)


def _get_client():
    """Lazy-loads IBKR client adapter."""
    from ibkr import get_client
    return get_client()


# ── LLM analysis dispatch ─────────────────────────────────────────────────────
# The 72B analyst pipeline used to run inline inside the pollers, so a burst
# of filings/news pinned both workers on multi-minute inference while stop
# checks and snapshots queued behind them (296-deep backlog on 2026-07-15).
# Pollers now enqueue analysis onto dedicated llm_p* queues and return in
# seconds; a separate single-concurrency worker drains the LLM lanes.

def _held_tickers() -> set[str]:
    """Tickers currently held, from the latest portfolio snapshot (refreshed
    every 5 min by take_portfolio_snapshot — fresh enough for routing)."""
    try:
        import db
        from config import TRADING_ENV
        conn = db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT positions_json FROM portfolio_snapshots "
                    "WHERE trading_env=%s ORDER BY timestamp DESC LIMIT 1",
                    (TRADING_ENV,),
                )
                row = cur.fetchone()
        finally:
            db._put_conn(conn)
        return {t.upper() for t in (row[0] or {})} if row else set()
    except Exception as e:
        log.warning(f"_held_tickers failed ({e}) — routing to bulk queue")
        return set()


def _fresh_snapshot_positions(max_age_min: float = 15) -> dict:
    """positions_json of the latest portfolio snapshot if it is fresh, else
    {}. No IBKR call: the ingestion worker must not open broker connections."""
    return _snapshot_positions_or_none(max_age_min) or {}


def _snapshot_positions_or_none(max_age_min: float = 15) -> dict | None:
    """Like _fresh_snapshot_positions, but None when there is no fresh
    snapshot, so an empty portfolio ({}) can be told apart from unknown."""
    import db
    from datetime import datetime, timezone
    from config import TRADING_ENV
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT positions_json, timestamp FROM portfolio_snapshots "
                "WHERE trading_env=%s ORDER BY timestamp DESC LIMIT 1",
                (TRADING_ENV,),
            )
            row = cur.fetchone()
    finally:
        db._put_conn(conn)
    if not row or row[0] is None or not row[1]:
        return None
    ts = row[1] if row[1].tzinfo else row[1].replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - ts).total_seconds() > max_age_min * 60:
        return None
    return row[0]


def _broker_held(ticker: str, max_age_min: float = 15) -> tuple[bool | None, str, dict]:
    """Whether the broker holds ticker, from the fresh portfolio snapshot:
    (held, reason, position). held is None when unknown; reason is
    "snapshot", "empty" (no positions at all), "no_snapshot" or "error"
    (audit A1031-0453-2: "held" used to mean "has a stop row")."""
    try:
        positions = _snapshot_positions_or_none(max_age_min)
    except Exception:
        return None, "error", {}
    if positions is None:
        return None, "no_snapshot", {}
    if not positions:
        return False, "empty", {}
    p = positions.get(ticker.upper()) or positions.get(ticker) or {}
    try:
        return (isinstance(p, dict) and float(p.get("qty") or 0) > 0), "snapshot", \
            (p if isinstance(p, dict) else {})
    except (TypeError, ValueError):
        return None, "error", {}


def _held_qty_tickers(max_age_min: float = 15) -> set[str]:
    """Tickers with a nonzero quantity in the fresh snapshot, whatever their
    market value (IBKR reports 0 right after a fill; review of 9a2f13c)."""
    out = set()
    for t, p in _fresh_snapshot_positions(max_age_min).items():
        try:
            if float(p.get("qty") or 0) != 0:
                out.add(t.upper())
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _held_prices(max_age_min: float = 15) -> dict[str, float]:
    """Held tickers' prices (market value / qty) from the latest portfolio
    snapshot, if it is fresh. No IBKR call: the ingestion worker that runs
    the plan watcher must not open broker connections."""
    import math
    out = {}
    for t, p in _fresh_snapshot_positions(max_age_min).items():
        try:
            qty, mv = float(p.get("qty") or 0), float(p.get("market_value") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if qty > 0 and mv > 0 and math.isfinite(mv / qty):
            out[t.upper()] = mv / qty
    return out


def _dispatch_event(event: dict, seen_source: str | None = None) -> None:
    """Queue an ingestion event for screening (see SCREEN_QUEUE comment
    above) — the fast, cheap first stage. Only events that pass go on to
    the expensive 72B analysis (tasks.screen_event dispatches those
    itself, routed to llm_p0/llm_p1 by held-ticker status).

    If the event carries "_seen_key" (single key, marked under seen_source
    or event["source"]) and/or "_seen_keys" (list of (source, key) pairs —
    for pollers with more than one dedup dimension, e.g. news's per-source
    id plus a cross-source title hash), they're marked seen here — AFTER
    the enqueue succeeds — so a Celery retry on a later failure doesn't
    return an already-seen, silently dropped item.
    """
    event = dict(event)  # don't mutate the caller's dict
    seen_key  = event.pop("_seen_key", None)
    seen_keys = event.pop("_seen_keys", None) or []
    if seen_key:
        seen_keys = [(seen_source or event.get("source", "unknown"), seen_key)] + list(seen_keys)
    screen_event.apply_async(
        args=[event], queue=SCREEN_QUEUE, expires=SCREEN_EVENT_EXPIRES_SECONDS,
    )
    if seen_keys:
        try:
            import db
            for src, key in seen_keys:
                if src == "edgar":   # ticker/form feed the scanner's 8-K points (D4)
                    db.mark_seen(src, key, ticker=event.get("ticker"), form=event.get("form"))
                else:
                    db.mark_seen(src, key)
        except Exception as e:
            log.warning(f"mark_seen failed for {seen_keys}: {e}")


# Screen and analysis tasks whose loss gets an EVENT_LOST row: one that
# expired in the queue, was revoked, or failed its last retry left no record.
_EVENT_TASKS = ("tasks.screen_event", "tasks.deep_analyze_event")


def _log_event_lost(task_name: str, reason: str, args, error_class: str | None = None) -> None:
    """Log-only EVENT_LOST row: the task, why, and the event's identifiers
    from the task args. Never raises."""
    try:
        import db
        from agents.crew import event_item_fields
        args = list(args or [])
        event = args[0] if args and isinstance(args[0], dict) else {}
        ticker = str(event.get("ticker") or "UNKNOWN").upper()
        decision = {"task": task_name, "reason": reason, **event_item_fields(event)}
        if error_class:
            decision["error_class"] = error_class
        log.error(f"[{ticker}] {task_name} lost the event ({reason}): {decision.get('title')}")
        db.log_signal(ticker, "EVENT_LOST", 0.0, f"{task_name}: {reason}", decision=decision)
    except Exception as e:
        log.warning(f"EVENT_LOST log failed for {task_name}: {e}")


def _event_task_failed(self, exc, task_id, args, kwargs, einfo):
    """on_failure of the screen and analysis tasks: the final failure, after
    any retries."""
    retries = getattr(self.request, "retries", 0) or 0
    reason = "max_retries" if self.max_retries and retries >= self.max_retries else type(exc).__name__
    _log_event_lost(self.name, reason, args, type(exc).__name__)


@task_failure.connect
def _on_event_task_killed(sender=None, task_id=None, exception=None, args=None, **_):
    """A screen or analysis task killed by the hard time limit or lost with
    its worker process. Neither reaches the task's on_failure: the worker's
    parent sends only this signal. Any other exception is left to
    on_failure, which sees it in the child, so no task id gets two rows."""
    from billiard.exceptions import TimeLimitExceeded, WorkerLostError
    if getattr(sender, "name", None) not in _EVENT_TASKS:
        return
    if isinstance(exception, TimeLimitExceeded):
        reason = "time_limit"
    elif isinstance(exception, WorkerLostError):
        reason = "worker_lost"
    else:
        return
    _log_event_lost(sender.name, reason, args, type(exception).__name__)


@task_revoked.connect
def _on_event_task_revoked(sender=None, request=None, terminated=None, signum=None,
                           expired=None, **_):
    """A screen or analysis task that expired in the queue or was revoked."""
    name = getattr(sender, "name", None) or getattr(request, "task_name", None)
    if name not in _EVENT_TASKS:
        return
    _log_event_lost(name, "expired" if expired else "revoked", getattr(request, "args", None))


@app.task(name="tasks.screen_event", bind=True, max_retries=1, on_failure=_event_task_failed)
def screen_event(self, event: dict):
    """Fast 7B relevance screen (own queue/worker, CPU-only, higher
    concurrency than the 72B stage) — only events that pass are
    dispatched onward to deep_analyze_event. This is what keeps a busy
    news day's rejects from queueing behind slow analyses."""
    try:
        from agents.crew import screen_event_only
        result = screen_event_only(event)
        if not result.get("relevant"):
            return
        ticker = str(event.get("ticker") or "").upper()
        queue = LLM_QUEUE_HELD if ticker and ticker in _held_tickers() else LLM_QUEUE_BULK
        deep_analyze_event.apply_async(
            args=[event, result], queue=queue, expires=LLM_EVENT_EXPIRES_SECONDS,
        )
    except Exception as e:
        log.error(f"screen_event failed for {event.get('ticker')}: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.deep_analyze_event", bind=True, max_retries=1, on_failure=_event_task_failed)
def deep_analyze_event(self, event: dict, screening: dict | None = None):
    """Full 72B analyst pipeline (LLM lane) for an event that already
    passed screening — see tasks.screen_event."""
    try:
        from agents.crew import deep_analyze
        deep_analyze(event, screening)
    except Exception as e:
        log.error(f"deep_analyze_event failed for {event.get('ticker')}: {e}")
        raise self.retry(exc=e, countdown=120)


def _format_research_alert(ticker: str, result: dict, requested_by: str) -> str:
    gap  = result.get("expectation_gap") or {}
    cat  = result.get("catalyst") or {}
    val  = result.get("valuation") or {}
    risks = result.get("key_risks") or []

    lines = [
        f"🔬 <b>Research: {ticker}</b> (requested by {requested_by or 'chat'})",
        f"<b>{result.get('action', '?')}</b> · Conviction {result.get('conviction', '?')}/10 "
        f"· {result.get('thesis_bucket', '?')}",
        f"Thesis {result.get('thesis_strength', '?')}/5 · Catalyst {result.get('catalyst_quality', '?')}/5 · "
        f"Valuation {result.get('valuation_risk_reward', '?')}/5 · Data {result.get('data_completeness', '?')}/5",
        "",
        result.get("thesis_summary", "") or result.get("reasoning", ""),
    ]
    if gap:
        lines += [
            "",
            f"<b>Bull case:</b> {gap.get('bull_case', 'n/a')}",
            f"<b>Priced in:</b> {gap.get('current_price_implies', 'n/a')}",
            f"<b>Primary risk:</b> {gap.get('primary_risk', 'n/a')}",
        ]
    if cat:
        lines.append(
            f"<b>Catalyst:</b> {cat.get('event', 'n/a')} ({cat.get('date', 'n/a')}) — "
            f"{cat.get('expected_upside_if_positive', '?')} / "
            f"{cat.get('expected_downside_if_negative', '?')}"
        )
    if val:
        lines.append(
            f"<b>Valuation:</b> {val.get('current_multiple', 'n/a')} vs peers "
            f"{val.get('peer_range', 'n/a')} · target {val.get('bull_case_target', 'n/a')} / "
            f"downside {val.get('downside_scenario', 'n/a')}"
        )
    if result.get("upside_downside_ratio") is not None:
        lines.append(f"<b>Upside/downside ratio:</b> {result['upside_downside_ratio']}")
    if risks:
        lines.append("<b>Key risks:</b> " + "; ".join(risks))
    lines.append("")
    lines.append(result.get("reasoning", ""))
    lines.append("")
    note = ("would have qualified for auto-execution as a live signal — "
            "this was research only, nothing was traded"
            if result.get("pass_to_risk") else
            "research only — not executed")
    lines.append(f"⚠️ {note}.")
    return "\n".join(l for l in lines if l is not None)


@app.task(name="tasks.research_ticker_on_demand", bind=True, max_retries=1)
def research_ticker_on_demand(self, ticker: str, requested_by: str = ""):
    """Chat-triggered (see chat_assistant._tool_request_deep_analysis), read-
    only run of the full analyst pipeline for one ticker — real brief, real
    72B analysis, real conviction score, delivered back to Telegram. Runs on
    llm_p2, the lowest-priority analysis queue (see LLM_QUEUE_RESEARCH), so
    it never delays anything touching a held position or an organic
    news-driven idea. Never routes to risk sizing or execution."""
    try:
        from agents.crew import research_ticker
        import alerts
        result = research_ticker(ticker)
        alerts._send(_format_research_alert(ticker, result, requested_by))
    except Exception as e:
        log.error(f"research_ticker_on_demand failed for {ticker}: {e}")
        try:
            import alerts
            alerts._send(f"⚠️ Deep analysis of {ticker} failed: {e}")
        except Exception:
            pass


@app.task(name="tasks.analyze_candidates", bind=True, max_retries=1)
def analyze_candidates(self, candidates: list, max_per_run: int = 5):
    """Run scanner candidates through the analyst pipeline (LLM lane)."""
    try:
        from agents.crew import process_candidates
        process_candidates(candidates, max_per_run=max_per_run)
    except Exception as e:
        log.error(f"analyze_candidates failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.retry_deferred_entry", bind=True, max_retries=0)
def retry_deferred_entry(self, approval_id: int, gap_retry_attempt: int = 1, price_retry_attempt: int = 1,
                         recheck_attempt: int = 1):
    """
    Retry for a new-entry (OPEN/ADD) approval that execution.execute_approved
    deferred, for any of three reasons: the market was closed at analysis
    time (scheduled via market_data.next_market_open() -- see
    execute_approved's market-hours gate, added 2026-07-28 alongside the
    5:00 AM ET wake), an opening gap hasn't yet been confirmed as holding vs.
    fading (scheduled via config.GAP_CONFIRMATION_RETRY_MINUTES -- see
    execute_approved's gap-confirmation block, added 2026-07-29), or only a
    stale price was available (scheduled via config.STALE_PRICE_RETRY_MINUTES
    -- see execute_approved's price-freshness block, added 2026-07-31).
    gap_retry_attempt and price_retry_attempt thread through the latter two
    cases independently, each counting against its own config.*_MAX_RETRIES
    so every retry chain is bounded to a fixed window after the open rather
    than deferring indefinitely. recheck_attempt counts the re-size retries
    (an unreadable re-size check) against DEFERRED_RECHECK_MAX_ATTEMPTS.

    Approvals older than DEFERRED_ENTRY_MAX_AGE_HOURS are expired and sent
    back through a fresh 72B analysis instead (2026-10-27), or only expired
    when config.STALE_ENTRY_EXPIRE_ONLY is on. Younger ones do
    NOT re-run the analysis — they reuse execute_approved's own price-drift, gap-confirmation, and price-
    freshness guards as the safety check against firing a now-stale
    decision. If the market is somehow still closed when this fires
    (holiday, clock skew), or the gap/price still hasn't resolved,
    execute_approved's gates defer it again with a fresh retry —
    self-correcting, not a tight loop, since each retry target is strictly
    later than the last and real time keeps moving forward (and both the
    gap-confirmation and stale-price paths have their own hard retry cap).
    """
    import db
    from execution import execute_approved, last_outcome_reason

    approval = db.get_approval_by_id(approval_id)
    if not approval:
        log.warning(f"retry_deferred_entry: approval #{approval_id} not found")
        return
    if approval.get("status") != "FAILED":
        # Already handled some other way (a fresher event re-triggered
        # analysis, a human acted on it, etc.) -- don't double-execute.
        log.info(f"retry_deferred_entry: approval #{approval_id} status is "
                 f"'{approval.get('status')}', not FAILED — skipping retry")
        return
    if db.SWEPT_REASON_MARK in (approval.get("status_reason") or ""):
        # Swept out of PROCESSING: its order may already be at the broker.
        log.warning(f"retry_deferred_entry: approval #{approval_id} was swept out of PROCESSING "
                    f"— not retried; check the broker")
        return

    ticker = approval.get("ticker", "?")

    # Stale decisions get re-analysed instead of executed (2026-10-27,
    # limitations audit finding 13). A Friday-evening approval used to fill at
    # Monday's open on 60-hour-old reasoning, guarded only by price drift.
    age_h = _approval_age_hours(approval)
    if age_h is not None and age_h > DEFERRED_ENTRY_MAX_AGE_HOURS:
        # Conditional: of several duplicate deliveries only the one that
        # moves the row FAILED -> EXPIRED queues the re-analysis.
        from config import STALE_ENTRY_EXPIRE_ONLY
        if not db.expire_failed_entry(approval_id,
                                      f"deferred entry {age_h:.0f}h old "
                                      f"(> {DEFERRED_ENTRY_MAX_AGE_HOURS}h), "
                                      f"{'not re-analysed' if STALE_ENTRY_EXPIRE_ONLY else 're-analysed'}"):
            log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} already expired "
                     f"or claimed — no re-analysis")
            return
        if STALE_ENTRY_EXPIRE_ONLY:
            log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} is {age_h:.0f}h old "
                     f"(> {DEFERRED_ENTRY_MAX_AGE_HOURS}h) — expired, no re-analysis")
            return
        log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} is {age_h:.0f}h old "
                 f"(> {DEFERRED_ENTRY_MAX_AGE_HOURS}h) — expired, queued a fresh analysis")
        deep_analyze_event.apply_async(args=[{
            "ticker": ticker, "source": "stale_entry_recheck", "type": "re-analysis",
            "title": f"Re-check of a {approval.get('action', 'BUY')} approved {age_h:.0f}h ago "
                     f"that never executed",
            "text": "",
        }, None], queue=LLM_QUEUE_BULK, expires=LLM_EVENT_EXPIRES_SECONDS)
        return

    # A deferred buy executes the decision as it stood at approval time, so
    # drop it when newer information contradicts it (audit 10-28 10:53 #2).
    reason = _deferred_buy_superseded(approval)
    if reason:
        db.update_approval_status(approval_id, "EXPIRED", reason=f"superseded: {reason}")
        log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} dropped: {reason}")
        return

    # Re-size against the account as it is now: positions, cash and caps may
    # have changed since approval (audit 10-28 10:53 #2 residual).
    shares_now = _deferred_buy_shares_now(approval)
    resized = None
    if shares_now == _RESIZE_UNREADABLE:
        # The concentration check failed closed (0 room) because positions
        # couldn't be read: not a real 0, so the approval stays FAILED and is
        # retried, not expired. Not executed either. The age cap above bounds
        # the retries, and so does recheck_attempt: an approval whose age
        # can't be read, or that ran out of attempts, is expired instead.
        why = ("its age can't be read" if age_h is None
               else f"{recheck_attempt} re-size attempts" if recheck_attempt >= DEFERRED_RECHECK_MAX_ATTEMPTS
               else None)
        if why:
            if db.expire_failed_entry(approval_id, f"re-size check unreadable and {why}; expired, "
                                                   f"not executed"):
                log.warning(f"retry_deferred_entry: [{ticker}] approval #{approval_id} re-size check "
                            f"unreadable and {why}; expired")
            else:
                log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} already expired "
                         f"or claimed")
            return
        log.warning(f"retry_deferred_entry: [{ticker}] approval #{approval_id} re-size check "
                    f"unreadable; not executed, retrying in {DEFERRED_RECHECK_RETRY_MINUTES} min "
                    f"(attempt {recheck_attempt}/{DEFERRED_RECHECK_MAX_ATTEMPTS})")
        try:
            retry_deferred_entry.apply_async(args=[approval_id, gap_retry_attempt, price_retry_attempt,
                                                   recheck_attempt + 1],
                                             countdown=DEFERRED_RECHECK_RETRY_MINUTES * 60, queue="ops")
        except Exception as e:
            log.error(f"retry_deferred_entry: [{ticker}] could not schedule the re-check retry: {e}")
        return
    if shares_now is not None:
        approved = int((approval.get("risk_json") or {}).get("shares_initial") or 0)
        if shares_now <= 0:
            db.update_approval_status(approval_id, "EXPIRED",
                                      reason="re-check at execution: sizes to 0 shares now")
            log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} dropped: "
                     f"sizes to 0 shares at execution")
            return
        if shares_now < approved:
            log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} reduced "
                     f"{approved} -> {shares_now} shares at execution")
            approval = {**approval, "risk_json": {**approval["risk_json"], "shares_initial": shares_now}}
            # risk_json keeps the approved size; record why fewer were bought.
            resized = f"reduced {approved} -> {shares_now} shares at execution"

    # Claim it first: the retry used to execute while the row sat FAILED, so a
    # duplicate delivery of this task could execute it twice. A crash after
    # this leaves it PROCESSING for the sweep, which never retries it.
    if not db.claim_deferred_entry_for_retry(approval_id):
        log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} not claimed "
                 f"(already taken, swept or unreadable) — skipping")
        return
    success = execute_approved(
        approval, trigger="auto_retry",
        gap_retry_attempt=gap_retry_attempt, price_retry_attempt=price_retry_attempt,
    )
    if success:
        db.update_approval_status(approval_id, "APPROVED", reason=resized)
    else:
        # A fill left to reconcile filled: APPROVED with that reason, not FAILED.
        from execution import approval_outcome
        status, reason = approval_outcome(False)
        db.update_approval_status(approval_id, status, reason=reason)
    log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} "
             f"{'executed' if success else 'still failed'} on retry "
             f"(gap_attempt={gap_retry_attempt}, price_attempt={price_retry_attempt})")


HELD_SELL_RETRY_DELAY_S = 300


@app.task(name="tasks.retry_held_sell", bind=True, max_retries=0)
def retry_held_sell(self, approval_id: int):
    """One re-execution of a model TRIM/EXIT that was held because the
    broker's resting stops couldn't be read (m127). Not a new decision: it
    runs only during market hours, while the position is still held, and
    when no newer approval or verdict exists for the ticker. The attempt and
    its outcome are appended to status_reason."""
    import db
    import ibkr
    from datetime import datetime, timezone
    from execution import execute_approved, last_outcome_reason
    from market_data import is_market_hours

    approval = db.get_approval_by_id(approval_id)
    if not approval or approval.get("status") != "FAILED":
        log.info(f"retry_held_sell: approval #{approval_id} is not FAILED; nothing to retry")
        return
    ticker = approval["ticker"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def note(held_reason: str, outcome: str, status: str = "FAILED") -> None:
        # Appended, so the original hold (before or after the cancel) stays readable.
        db.update_approval_status(approval_id, status, reason=f"{held_reason} | retried at {stamp}: {outcome}")

    def skip(why: str) -> None:
        # The retry runs once, so a skip is final: say so on the row (m129).
        held_reason = db.claim_failed_sell_for_retry(approval_id)
        if held_reason:
            note(held_reason, f"not retried: {why}")
        log.info(f"retry_held_sell: [{ticker}] #{approval_id} not retried, {why}")

    if not is_market_hours():
        skip("market closed")
        return
    try:
        # All positions, not get_open_position: that raises when the stop
        # filled meanwhile, which is the "nothing held" case.
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        held = abs(float(pos.qty)) if pos else 0.0
    except Exception as e:
        log.warning(f"retry_held_sell: [{ticker}] #{approval_id} position read failed ({e})")
        skip("position read failed")
        return
    if held <= 0:
        held_reason = db.claim_failed_sell_for_retry(approval_id)
        if held_reason:
            note(held_reason, "not sent, no position held")
        return
    newer = db.has_newer_ticker_activity(ticker, approval_id, approval.get("timestamp"))
    if newer is None:
        skip("newer-activity check failed")
        return
    if newer:
        held_reason = db.claim_failed_sell_for_retry(approval_id)
        if held_reason:
            note(held_reason, "not sent, superseded by a newer approval or verdict")
        log.info(f"retry_held_sell: [{ticker}] #{approval_id} not retried (superseded)")
        return
    held_reason = db.claim_failed_sell_for_retry(approval_id)
    if not held_reason:
        log.info(f"retry_held_sell: [{ticker}] #{approval_id} already retried or changed; skipping")
        return
    success, outcome = False, "error before an outcome"
    try:
        success = execute_approved(approval, trigger="held_sell_retry")
        outcome = "executed" if success else last_outcome_reason()
    finally:
        # Always leave the claimed row with a final status (professor M135);
        # a fill left to reconcile filled, so it is APPROVED.
        from execution import approval_outcome
        note(held_reason, outcome, approval_outcome(success)[0])
    log.info(f"retry_held_sell: [{ticker}] #{approval_id} {'executed' if success else 'still failed'} on retry")


DEFERRED_ENTRY_MAX_AGE_HOURS = 16
# _deferred_buy_shares_now's answer when the re-size couldn't be done (positions
# or the halt check unreadable, or the re-size raised): the buy is neither sized
# nor expired, and is not executed at the approved size; it waits for a retry.
_RESIZE_UNREADABLE = "concentration_unreadable"
DEFERRED_RECHECK_RETRY_MINUTES = 5
DEFERRED_RECHECK_MAX_ATTEMPTS = 200   # about 16 h at 5 min


def _deferred_buy_shares_now(approval: dict) -> int | None:
    """Shares a deferred OPEN/ADD would get if sized now (sizing + concentration
    caps on the live account, at the price it was approved at). None when it
    isn't a buy or has no sizing price (then the approved size stands);
    _RESIZE_UNREADABLE when the re-check itself fails, so the buy waits for a
    retry instead of going at the approved size (roadmap 00 b)."""
    if approval.get("action") not in ("OPEN", "ADD"):
        return None
    risk_json = approval.get("risk_json") or {}
    price = risk_json.get("price_at_sizing")
    if not price:
        return None
    try:
        import ibkr
        from agents.market_regime import get_cached_regime
        from risk_manager import RiskManager
        regime = (get_cached_regime() or {}).get("regime", "Neutral")
        bucket = approval.get("thesis_bucket") or "Accelerator"
        rm = RiskManager(ibkr.get_client())
        sizing = rm.size_position(approval["ticker"], int(approval.get("conviction") or 5),
                                  bucket, regime, float(price))
        if sizing.get("halt_unknown") is True:
            # One unreadable check mustn't expire the buy for good, nor let
            # it go at the approved size: keep it for a retry.
            log.warning(f"retry_deferred_entry: [{approval.get('ticker')}] halt check "
                        f"unreadable; not executed, kept for a retry")
            return _RESIZE_UNREADABLE
        shares = rm.shares_after_concentration(approval["ticker"], sizing, bucket,
                                               name_cap=True, price=float(price))
        if getattr(rm, "last_concentration_errored", False) is True:
            return _RESIZE_UNREADABLE
        name_cap = getattr(rm, "last_name_cap", None)
        if isinstance(name_cap, dict) and name_cap.get("outcome") == "errored":
            return _RESIZE_UNREADABLE
        return shares
    except Exception as e:
        log.warning(f"retry_deferred_entry: [{approval.get('ticker')}] re-size failed; "
                    f"not executed, kept for a retry: {e}")
        return _RESIZE_UNREADABLE


def _deferred_buy_superseded(approval: dict) -> str | None:
    """Why a deferred OPEN/ADD should not execute, or None if it still stands:
    a later approval for the same ticker exists, or a later analyst verdict
    on the ticker is not a buy (WATCH, HOLD, TRIM, EXIT, ...)."""
    from datetime import timezone
    import db
    if approval.get("action") not in ("OPEN", "ADD"):
        return None
    ticker = approval.get("ticker")
    if db.has_newer_buy_approval(ticker, approval["id"]):
        return "a newer buy approval for the same ticker exists"
    ts = approval.get("timestamp")
    latest = db.get_latest_signal_for(ticker)
    if ts and latest and latest.get("timestamp") and latest.get("signal") not in ("OPEN", "ADD"):
        lts = latest["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if lts.tzinfo is None:
            lts = lts.replace(tzinfo=timezone.utc)
        if lts > ts:
            return (f"newer {latest['signal']} verdict (conviction "
                    f"{latest.get('confidence', 0):g}) at {lts:%Y-%m-%d %H:%M} UTC")
    return None


def _approval_age_hours(approval: dict) -> float | None:
    from datetime import datetime, timezone
    ts = approval.get("timestamp")
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


_DONE_UNFILLED = {"Cancelled", "ApiCancelled", "Inactive"}


def _backfill_perm_ids(pending: list[dict]) -> None:
    """A row logged before TWS assigned its permId has none (professor M127).
    Fill it from the open orders while the order still works, only when
    exactly one open market order has that orderId, symbol, side and
    quantity, and only for today's rows: our market orders are DAY orders,
    and orderId alone is per client, so an old row could match another
    client's stop (m102, review of 0a80ce7). Never raises."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    et = ZoneInfo("America/New_York")
    today = datetime.now(et).date()
    missing = [o for o in pending if not o.get("perm_id") and o.get("created_at") is not None
               and o["created_at"].astimezone(et).date() == today]
    if not missing:
        return
    import db
    import ibkr
    try:
        open_orders = ibkr.get_open_orders()
    except Exception as e:
        log.warning(f"reconcile: open-order read for the permId backfill failed: {e}")
        return
    for order in missing:
        hits = [o for o in open_orders if o.get("order_id") == order["broker_order_id"]
                and o.get("ticker") == order["ticker"] and (o.get("action") or "").upper() == order["side"]
                and o.get("order_type") == "MKT" and order.get("qty") is not None
                and abs(float(o.get("qty") or 0) - float(order["qty"])) < 1e-6 and o.get("perm_id")]
        if len(hits) == 1:
            try:
                if db.set_order_perm_id(order["id"], hits[0]["perm_id"]):
                    order["perm_id"] = hits[0]["perm_id"]
            except Exception as e:
                log.warning(f"[{order['ticker']}] reconcile: permId backfill failed: {e}")


def _resolve_if_cancelled(order: dict, completed: dict) -> None:
    """An unconfirmed/partial order with no new fill used to stay open forever
    when the broker had cancelled or rejected it, so the stale-stop branch
    waited on it until midnight and the stop fill was then missed as no_fill
    (professor M127). Marks the row cancelled only when today's completed
    orders show it Cancelled/ApiCancelled/Inactive (matched on permId, symbol
    and side) with exactly the recorded fill. Not found, no permId, or any
    error: unknown, so it keeps waiting. A larger broker fill than recorded is
    a late fill the execution lookup missed: alerted, not marked."""
    import db
    import ibkr
    ticker, perm = order["ticker"], order.get("perm_id")
    if not perm:
        return
    if "rows" not in completed:
        try:
            completed["rows"] = ibkr.get_completed_orders()
        except Exception as e:
            log.warning(f"reconcile: completed-order read failed: {e}")
            completed["rows"] = None
    rows = completed["rows"]
    if not rows:
        return
    hit = next((c for c in rows if c["perm_id"] == perm and c["ticker"] == ticker
                and c["action"] == order["side"]), None)
    if not hit or hit["status"] not in _DONE_UNFILLED or hit["filled_qty"] is None:
        return
    recorded = order.get("filled_qty") or 0.0
    if hit["filled_qty"] > recorded + 1e-4:
        log.error(f"[{ticker}] reconcile: order {order['broker_order_id']} is {hit['status']} with "
                  f"{hit['filled_qty']:g} filled at the broker but {recorded:g} recorded, and no execution "
                  f"was found; left open")
        try:
            if _reconcile_alert_due(f"fill_mismatch_{order['id']}"):
                import alerts
                alerts._send(f"⚠️ FILL NOT RECORDED — <b>{ticker}</b>\nThe broker shows "
                             f"{hit['filled_qty']:g} shares filled on a {order['side']} it reports "
                             f"{hit['status']}, but the bot recorded {recorded:g} and found no execution. "
                             f"Check the trade in TWS.")
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: fill-mismatch alert failed: {e}")
        return
    if hit["filled_qty"] < recorded - 1e-4:
        log.warning(f"[{ticker}] reconcile: order {order['broker_order_id']} shows less filled at the broker "
                    f"({hit['filled_qty']:g}) than recorded ({recorded:g}); left open")
        return
    if not db.mark_order_cancelled(order["id"], recorded):
        return
    log.warning(f"[{ticker}] reconcile: order {order['broker_order_id']} ({order['side']} "
                f"{order.get('intent')}) is {hit['status']} at the broker with {recorded:g} filled; "
                f"marked cancelled, the rest never sold")
    if order.get("side") == "SELL":
        # The stops were cancelled before this sell, and its unsold shares
        # (all of them if nothing filled) had no stop coming back: the queued
        # re-protect places the recorded stop on what is still held and drops
        # out if the position is flat (audit 2026-11-02 #2).
        try:
            _queue_reprotect(ticker, {"intent": "RESTORE"}, None)
        except Exception as e:
            log.error(f"[{ticker}] reconcile: could not queue the stop after a cancelled sell: {e}")
            try:
                import alerts
                alerts._send(f"🚨 CHECK STOPS — <b>{ticker}</b>\nA SELL was cancelled at the broker "
                             f"and the stop retry could not be queued. Check TWS and protect the "
                             f"unsold shares.")
            except Exception:
                pass
    # A cancelled part-filled exit leaves its outcome to the stale-stop and
    # sweep paths: a flat position means another sale finished it, and the
    # exit price of that mixed close is Luke's open question (review of
    # 0a80ce7).


@app.task(name="tasks.reconcile_unconfirmed_orders", bind=True, max_retries=2)
def reconcile_unconfirmed_orders(self):
    """
    Catches late fills that land after ibkr.place_market_order's fill-wait
    window already gave up. Incident 2026-07-28: a CRWD BUY (144 shares,
    order 764) filled 12s after the 30s wait closed -- see
    place_market_order's wall-clock-timing fix docstring for the bug that
    caused the false "no fill" in the first place. The order was correctly
    logged status='unconfirmed' (never fabricate a fill we didn't see), but
    nothing ever went back to check whether it filled anyway, so a real
    ~$26K position sat live at the broker all day with no stop and no trade
    record -- invisible to every other safety system, since they all read
    the DB, not the broker directly.

    Runs against every 'unconfirmed' order from the last 30 days, checking
    IBKR's own execution record (ibkr.get_fill_for_order) rather than
    trusting the original wait-window snapshot. A match here is a
    confirmed broker execution, not a guess -- recording it doesn't fall
    under the "never fabricate a fill" rule that governs the unconfirmed
    path itself.

    Incident 2026-07-30: this originally looked back only 24h, on the
    assumption a stale snapshot would get caught quickly. In practice 8
    orders across 7 tickers (SNOW, AMD x2, TSLA, CRM, AMKR, ORCL, CRWD)
    aged past that window unreconciled over the course of a week -- for
    AMD and ORCL specifically, the un-reconciled fill left real shares at
    the broker with no DB record and no stop, invisible to every other
    safety check the same way the original CRWD incident was. 30 days is
    deliberately generous rather than tuned to "how long this should
    normally take" -- an order stuck unconfirmed this long is itself a
    symptom of a bug elsewhere, and the check is cheap (one indexed query)
    however far back it looks.
    """
    import db
    import ibkr
    import alerts
    from datetime import timedelta

    # One run at a time: lock waits and broker calls can now make a run
    # outlast the 300 s schedule, and two runs would book the same fill
    # (audit 22:53, A2253-5). The key expires on its own if a run dies.
    # Renewed while the run lasts; without Redis the run is skipped rather
    # than run unguarded (audit 00:53, A0053-6).
    _run_singleton("reconcile_unconfirmed_orders",
                   lambda: _reconcile_unconfirmed_orders_locked(self))


_UNTRACKED_STATE_KEY = "untracked_orders"
_UNTRACKED_REDIS_PREFIX = "helios:untracked_order:"


def _untracked_key(row: dict) -> str:
    return f"{row.get('ticker')}:{row.get('side')}:{row.get('perm_id') or ''}:{row.get('broker_order_id')}"


def _save_untracked_order(row: dict) -> bool:
    """Keep an 'unconfirmed' orders row that could not be written, so the
    next reconcile run can write it and then find a late fill (audit
    2026-11-02 #7). Stored in Redis and in system_state: a DB outage that
    lost the row can lose the state write too. True if either store took it."""
    import db
    rec = {k: row.get(k) for k in ("ticker", "side", "order_type", "qty", "broker_order_id",
                                   "perm_id", "decision_price", "approval_id", "status",
                                   "intent", "stop_price", "submitted_at", "submit_price")}
    rec = json.loads(json.dumps(rec, default=str))   # submitted_at is a datetime: ISO text
    key = _untracked_key(rec)
    saved = False
    try:
        stop_lock._redis().set(_UNTRACKED_REDIS_PREFIX + key, json.dumps(rec, default=str),
                               ex=30 * 86400)
        saved = True
    except Exception as e:
        log.error(f"[{rec['ticker']}] untracked order not saved to Redis: {e}")
    try:
        def add(pending):
            pending[key] = rec
            return pending
        _reprotect_queue_edit(add, key=_UNTRACKED_STATE_KEY)
        saved = saved or key in json.loads(db.get_state(_UNTRACKED_STATE_KEY) or "{}")
    except Exception as e:
        log.error(f"[{rec['ticker']}] untracked order not saved to system_state: {e}")
    return saved


def _retry_untracked_orders() -> None:
    """Write each saved untracked order's row (once: an existing row for that
    broker order counts as written), then drop it from both stores. Runs at
    the start of every reconcile_unconfirmed_orders, so the row is in the
    order list the same run reads."""
    import db
    found: dict[str, dict] = {}
    r = None
    try:
        r = stop_lock._redis()
        for k in r.scan_iter(_UNTRACKED_REDIS_PREFIX + "*"):
            k = k.decode() if isinstance(k, bytes) else k
            v = r.get(k)
            if v:
                found[k[len(_UNTRACKED_REDIS_PREFIX):]] = json.loads(v)
    except Exception as e:
        log.warning(f"reconcile: untracked-order read from Redis failed: {e}")
    try:
        found.update(json.loads(db.get_state(_UNTRACKED_STATE_KEY) or "{}"))
    except Exception as e:
        log.warning(f"reconcile: untracked-order read from system_state failed: {e}")
    done = []
    for key, rec in found.items():
        if not rec.get("perm_id") and rec.get("broker_order_id") is None:
            # Nothing to match a broker order or a late fill on, and no way to
            # tell a written row apart: writing it would add one every cycle.
            # Alert once and drop it.
            log.error(f"[{rec.get('ticker')}] reconcile: untracked order has no perm id or order id; "
                      f"dropped, needs a manual check")
            try:
                import alerts
                alerts._send(f"🚨 ORDER UNTRACKED — <b>{rec.get('ticker')}</b>\nA {rec.get('side')} "
                             f"{rec.get('qty')} order with no order id or perm id could not be logged, "
                             f"so reconcile cannot follow it. Check TWS for a fill and protect any "
                             f"shares by hand.")
            except Exception as e:
                log.error(f"reconcile: untracked-order alert failed: {e}")
            done.append(key)
            continue
        try:
            exists = db.order_row_exists(rec["ticker"], rec["side"], rec.get("broker_order_id"),
                                         rec.get("perm_id"))
            if not exists and (db.log_order(**rec) or -1) <= 0:
                log.error(f"[{rec['ticker']}] reconcile: untracked order row still not written")
                continue
        except Exception as e:
            log.error(f"[{rec.get('ticker')}] reconcile: untracked order row retry failed: {e}")
            continue
        log.warning(f"[{rec['ticker']}] reconcile: untracked order {rec.get('broker_order_id')} "
                    f"now has its orders row")
        done.append(key)
    if not done:
        return
    for key in done:
        try:
            if r is not None:
                r.delete(_UNTRACKED_REDIS_PREFIX + key)
        except Exception as e:
            log.warning(f"reconcile: could not drop untracked order {key} from Redis: {e}")
    try:
        def drop(pending):
            for key in done:
                pending.pop(key, None)
            return pending
        _reprotect_queue_edit(drop, key=_UNTRACKED_STATE_KEY)
    except Exception as e:
        log.warning(f"reconcile: could not drop untracked orders from system_state: {e}")


def _adopt_orphaned_submitted_orders() -> None:
    """Hand rows written before a fill wait whose worker never came back
    (killed during the wait) to this run as 'unconfirmed', so a late fill is
    booked once by the loop below. Never raises."""
    import db
    try:
        adopted = db.adopt_orphaned_submitted_orders()
    except Exception as e:
        log.warning(f"reconcile: orphaned-order adoption failed: {e}")
        return
    for o in adopted or []:
        log.warning(f"[{o['ticker']}] reconcile: order {o['broker_order_id']} ({o['side']}) was still "
                    f"in its fill wait with no outcome recorded (worker stopped?); now unconfirmed")


def _orphan_fill_already_booked(order: dict, fill: dict, total_qty: float, total_avg: float,
                                complete: bool) -> tuple[bool, float]:
    """For an order adopted from its fill wait, how much of its fill the
    execution path already booked (its order-row update was lost, or the
    worker died after the trade row): db.trade_logged_near, the summed
    quantity of the execution path's own trades rows for this order (same
    symbol, side and env, its reason, from 5 s before the order row up to the
    next same-side order row for the ticker, capped at 20 min). Returns
    (skip, booked):
    - nothing booked: (False, 0.0), book the fill as usual;
    - booked == the broker's fill: the order is marked filled with no trade
      row, outcome or stop work, one alert; (True, 0.0);
    - booked < the broker's fill: the order is marked 'partial' at the booked
      quantity and (False, booked) returned, so the caller books only the
      difference through the usual late-fill code; one alert;
    - booked > the broker's fill: nothing is marked, an alert (throttled);
      (True, 0.0).
    Also (True, 0.0) when the check or the mark fails (wait a cycle rather
    than book twice). Never raises."""
    import db
    ticker, side, oid = order["ticker"], order["side"], order["broker_order_id"]
    try:
        booked = float(db.trade_logged_near(ticker, side, order["created_at"],
                                            intent=order.get("intent"),
                                            order_id=order["id"]) or 0.0)
    except Exception as e:
        log.warning(f"[{ticker}] reconcile: booked-trade check failed for orphaned order "
                    f"{oid}; retrying next cycle: {e}")
        return True, 0.0
    if booked <= 1e-6:
        return False, 0.0

    def _alert(text: str) -> None:
        try:
            import alerts
            alerts._send(f"⚠️ ORDER ROW REPAIRED — <b>{ticker}</b>\n{text}")
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: order-row repair alert failed: {e}")

    if booked > total_qty + 1e-4:
        log.error(f"[{ticker}] reconcile: orphaned order {oid} filled {total_qty:g} at the broker "
                  f"but {booked:g} is already booked near it; nothing marked")
        try:
            due = _reconcile_alert_due(f"orphan_overbooked_{order['id']}")
        except Exception:
            due = True
        if due:
            _alert(f"The {side} order {oid} filled {total_qty:g} at the broker, but {booked:g} "
                   f"shares of {side} trades are recorded around it. Nothing was marked or "
                   f"booked; check the trade rows against TWS.")
        return True, 0.0

    if abs(booked - total_qty) <= 1e-4:
        if db.update_order_fill(order["id"], fill_price=total_avg, filled_qty=total_qty,
                                status="filled" if complete else "partial", expect_filled_qty=0.0,
                                filled_at=fill.get("fill_time")) is False:
            log.error(f"[{ticker}] reconcile: could not mark orphaned order {order['id']} filled")
            return True, 0.0
        log.warning(f"[{ticker}] reconcile: orphaned order {oid} already has trade rows for "
                    f"{booked:g}; marked {total_qty:g} filled without booking it again")
        _alert(f"The {side} order {oid} filled {total_qty:g} @ ${total_avg:.2f} at the broker; "
               f"{booked:g} was already recorded in trades but the order row was not. The row "
               f"is now marked {total_qty:g} filled and nothing else was booked. Check the stop "
               f"against TWS.")
        return True, 0.0

    if db.update_order_fill(order["id"], fill_price=total_avg, filled_qty=booked, status="partial",
                            expect_filled_qty=0.0, filled_at=fill.get("fill_time")) is False:
        log.error(f"[{ticker}] reconcile: could not mark orphaned order {order['id']} "
                  f"partial at {booked:g}")
        return True, 0.0
    rest = round(total_qty - booked, 4)
    log.warning(f"[{ticker}] reconcile: orphaned order {oid} has {booked:g} of {total_qty:g} "
                f"already booked; booking the other {rest:g}")
    _alert(f"The {side} order {oid} filled {total_qty:g} @ ${total_avg:.2f} at the broker; "
           f"{booked:g} was already recorded in trades but the order row was not. The row is "
           f"marked {booked:g} and only the other {rest:g} is booked as a late fill.")
    return False, booked


def _reconcile_unconfirmed_orders_locked(task):
    import db
    import ibkr
    import alerts
    from datetime import timedelta

    # Queued retries first: they don't need the order list, and a failed
    # load must not hold back a stop (review of 1492b13).
    _retry_untracked_orders()
    _retry_pending_reprotects()
    _retry_pending_buy_outcomes()
    _adopt_orphaned_submitted_orders()
    _recover_trade_pending_orders()

    try:
        pending = db.get_unconfirmed_orders(max_age_hours=24 * 30)
    except Exception as e:
        log.error(f"reconcile_unconfirmed_orders: could not load unconfirmed orders: {e}")
        raise task.retry(exc=e, countdown=120)

    _backfill_perm_ids(pending)
    completed: dict = {}   # today's completed orders, read once per run (M127)

    for order in pending:
        ticker = order["ticker"]
        if order.get("reconcile_note") in db.TRADE_PENDING_NOTES:
            # A part fill whose trade row isn't confirmed yet: its recorded
            # part is booked (or found) by _recover_trade_pending_orders
            # first, and only then the rest here.
            continue
        try:
            # A row written after the fill wait (the fallback when the
            # early write failed) can postdate a partial: allow a margin.
            placed_at = order.get("created_at")
            if placed_at is not None:
                placed_at -= timedelta(minutes=5)
            fill = ibkr.get_fill_for_order(order["broker_order_id"], ticker=ticker,
                                           side=order["side"], placed_at=placed_at)
        except Exception as e:
            log.warning(f"[{ticker}] reconcile_unconfirmed_orders: fill lookup failed for "
                        f"order {order['broker_order_id']}: {e}")
            continue
        if not fill:
            _resolve_if_cancelled(order, completed)   # done only if the broker says so
            continue

        # A 'partial' order already has some shares recorded: log only the
        # new ones, at their own average price (professor M-4).
        total_qty, total_avg = fill["filled_qty"], fill["fill_price"]
        prev_qty = order.get("filled_qty") or 0.0
        fill_qty = round(total_qty - prev_qty, 4)
        if fill_qty <= 0:
            _resolve_if_cancelled(order, completed)
            continue
        fill_price = ((total_qty * total_avg - prev_qty * (order.get("fill_price") or total_avg))
                      / fill_qty) if prev_qty else total_avg
        complete = order.get("qty") is None or total_qty >= float(order["qty"]) - 1e-6
        log.warning(f"[{ticker}] Late fill found for order {order['broker_order_id']}: "
                    f"{order['side']} {fill_qty} @ ${fill_price:.2f} — was recorded "
                    f"unconfirmed, reconciling now"
                    + ("" if complete else f" ({total_qty:g} of {order['qty']:g} filled)"))

        if order.get("reconcile_note") == db.ORPHANED_FILL_WAIT_NOTE and not prev_qty:
            skip, booked = _orphan_fill_already_booked(order, fill, total_qty, total_avg, complete)
            if skip:
                continue
            if booked:
                # The row now records the booked part: book only the rest,
                # as for any partial (expect_filled_qty below = booked).
                prev_qty = booked
                order = {**order, "filled_qty": booked, "fill_price": total_avg}
                fill_qty = round(total_qty - prev_qty, 4)
                fill_price = total_avg

        intent = order.get("intent")
        # Mark the order first: if that write fails, stop here, or the next
        # cycle would log the same trade again and re-place a BUY stop
        # (professor M51). A failed log_trade after this is spilled.
        if db.update_order_fill(order["id"], fill_price=total_avg, filled_qty=total_qty,
                                status="filled" if complete else "partial",
                                expect_filled_qty=prev_qty, filled_at=fill.get("fill_time")) is False:
            log.error(f"[{ticker}] reconcile: could not mark order {order['id']} filled; "
                      f"trade not logged this cycle")
            continue
        _book_late_fill(order, fill_qty, fill_price, total_avg, prev_qty,
                        fill_time=fill.get("fill_time"))


def _stop_record_is_this_orders(ticker: str, since, fill_price: float,
                                approval: dict | None = None) -> bool:
    """For a trade_pending BUY being recovered: True if the ticker's stop
    record was written by this order's own execution path (its stop work ran
    though its trade row failed), so it must be kept: rewriting it would
    reset a stop that may since have trailed or taken its profit target, and
    for a manual BUY swap the ATR stop for the flat fallback.

    It is this order's only if every field the execution path writes once
    and nothing later changes matches what it would have written:
    - updated_at not before the order row (a leftover of an earlier, closed
      position was written before it);
    - entry_price equal to the fill, to the cent;
    - for an approval order, stop_method, thesis_bucket and conviction as
      the approval gives them (execute_approved's upsert);
    - for a manual BUY, a stop_method set and execute_manual's conviction
      (its default; the dashboard and !urgent, the only manual BUY callers,
      both use it). Its bucket comes from the caller and its method from
      calculate_stop, which the approval path shares, so a manual record is
      told apart by fewer fields than an approval one.
    A trail, a profit target or a re-placement change stop_price,
    trailing_active, profit_target_taken and updated_at, never these, so a
    record that has trailed since is still recognised (and kept). A closed
    position's leftover whose updated_at a later write bumped (a re-stop sets
    its broker order id) would need the same entry to the cent and the same
    method, bucket and conviction to be taken for this one.

    Anything else returns False and the caller writes the record as before.
    That keeps the established rule that a re-entry never inherits an old
    position's level (audit 00:53, A0053-5): a "tighter of old and new" upsert
    would keep a leftover level, which can sit at or above the market. On a
    read error: keep only if a record exists (db.get_position_stop), so a
    fresh OPEN still gets its record. False when since is None (not a
    recovery). Never raises."""
    if since is None:
        return False
    import db
    from datetime import timedelta
    try:
        rec = db.position_stop_written(ticker)
    except Exception as e:
        try:
            exists = db.get_position_stop(ticker) is not None
        except Exception:
            exists = False   # the caller's own read decides, as before this check
        log.error(f"[{ticker}] reconcile: stop record read failed ({e}); "
                  + ("keeping the existing record" if exists else "no record, writing it"))
        return exists
    if not rec:
        return False
    updated_at, entry = rec.get("updated_at"), rec.get("entry_price")
    if updated_at is None or updated_at < since - timedelta(seconds=5):
        return False
    if not entry or not fill_price or abs(entry - float(fill_price)) >= 0.005:
        return False
    if approval is not None:
        risk_json = approval.get("risk_json") or {}
        if (rec.get("stop_method") != risk_json.get("stop_method", "")
                or rec.get("thesis_bucket") != approval.get("thesis_bucket", "Accelerator")
                or rec.get("conviction") != approval.get("conviction", 5)):
            return False
    else:
        import inspect
        import execution
        manual_conviction = inspect.signature(execution.execute_manual).parameters["conviction"].default
        if not rec.get("stop_method") or rec.get("conviction") != manual_conviction:
            return False
    log.warning(f"[{ticker}] reconcile: the stop record was written for this order "
                f"(entry ${entry:.2f}); kept, not rewritten")
    try:
        import alerts
        alerts._send(f"ℹ️ STOP RECORD KEPT — <b>{ticker}</b>\nThe recovered BUY's stop record was "
                     f"already written by its execution, so it was left as is (only missing "
                     f"cover is placed).")
    except Exception as e:
        log.warning(f"[{ticker}] reconcile: stop-record-kept alert failed: {e}")
    return True


def _late_fill_reason(intent: str | None, why: str) -> str:
    return f"{intent + ' ' if intent else ''}Late-fill reconciled ({why})"


_APPROVAL_NOT_READ = object()   # _book_late_fill: read the approval itself


def _book_late_fill(order: dict, fill_qty: float, fill_price: float, total_avg: float,
                    prev_qty: float, fill_time=None,
                    why: str = "order placed unconfirmed, filled after the wait window",
                    headline: str = "🔁 LATE FILL RECONCILED",
                    what: str = ("this order was marked unconfirmed but had actually filled "
                                 "at the broker."),
                    trade_logged: bool = False, keep_record_since=None,
                    approval=_APPROVAL_NOT_READ, progress: dict | None = None) -> None:
    """Book a fill the execution path did not record, once the order row is
    already marked: the trade row, then what the execution path would have
    done after it (a BUY's outcome row and whole-position stop, a SELL's
    re-protect of the rest or close when flat, via _reprotect_after_late_sell,
    which sizes to what resting stops leave, so no second stop), then one
    alert. Used by reconcile_unconfirmed_orders for late fills and for
    filled rows left trade_pending. trade_logged: the caller already wrote
    the trade row (in the same transaction as its claim). keep_record_since:
    the order row's created_at; a BUY stop record written for this order
    since then is kept, not rewritten (_stop_record_is_this_orders).
    approval: the order's approval, when the caller already read it.
    progress: if given, progress["stop_work_done"] is set once the stop step
    has finished (placed, queued or reported), so a caller catching a later
    error knows not to redo it."""
    import db
    import alerts
    ticker = order["ticker"]
    intent = order.get("intent")
    fill = {"fill_time": fill_time}
    if not trade_logged:
        db.log_trade(
            symbol=ticker, side=order["side"], quantity=fill_qty, price=fill_price,
            reason=_late_fill_reason(intent, why),
            price_source="fill", order_row_id=order.get("id"),
        )

    stop_placed: bool | None = False
    progress = progress if progress is not None else {}
    if approval is _APPROVAL_NOT_READ:
        approval = db.get_approval_by_id(order["approval_id"]) if order["approval_id"] else None
    if order["side"] == "BUY" and approval:
        # No part of this order was recorded at the fill, so neither was
        # its outcome row (professor M123); a part fill's row was logged
        # at the part price and is re-averaged over the whole order (M121).
        # The order is already marked filled, so a failed write is queued,
        # not lost (professor M125).
        # An error here must not skip the stop record and placement below.
        try:
            _apply_buy_outcome(approval["id"], {
                "kind": "reaverage" if prev_qty else "insert", "ticker": ticker,
                "action": approval.get("action") or intent or "OPEN", "entry_price": total_avg,
            }, approval)
        except Exception as e:
            log.critical(f"[{ticker}] reconcile: outcome row for approval {approval['id']} "
                         f"not written: {e}")
    if order["side"] == "BUY" and approval:
        risk_json = approval.get("risk_json") or {}
        stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
        if stop_price:
            # One stop for the whole position at the recorded level (the
            # existing one for an ADD, never loosened), the record written
            # before placing, and a retry if placement fails or the ticker
            # is busy. A stop for only the late shares used to sit beside
            # the first tranche's, and upsert could loosen the recorded
            # stop (audit 22:53, A2253-3).
            try:
                if _stop_record_is_this_orders(ticker, keep_record_since, total_avg, approval):
                    pass   # the execution path's own record: leave it, cover-only below
                elif not db.get_position_stop(ticker) or not _held_before_fill(
                        ticker, fill_qty, approval.get("action")):
                    # No record, or the record is a closed position's
                    # leftover (kept up to 2 reconcile cycles): a re-entry
                    # must not inherit that old level (audit 00:53 A0053-5).
                    from execution import _stop_anchored_to_fill   # 28323dd review #1
                    stop_price = _stop_anchored_to_fill(
                        ticker, float(stop_price), risk_json.get("price_at_sizing"), total_avg)
                    db.upsert_position_stop(
                        ticker=ticker, entry_price=total_avg, stop_price=float(stop_price),
                        stop_method=risk_json.get("stop_method", ""),
                        bucket=approval.get("thesis_bucket", "Accelerator"),
                        conviction=approval.get("conviction", 5),
                        trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
                    )
                stop_placed = _reprotect_after_late_sell(ticker, {}, None, from_queue=True)
            except Exception as e:
                log.error(f"[{ticker}] reconcile_unconfirmed_orders: stop placement failed: {e}")
                stop_placed = False
            if stop_placed is not True:
                # None here is a flat read right after a BUY fill (cache
                # lag), not "nothing held": keep retrying (professor M60).
                _queue_reprotect(ticker, {"intent": "BUY"}, None)
        progress["stop_work_done"] = True
        if approval.get("status") == "FAILED":
            db.update_approval_status(order["approval_id"], "APPROVED")
    elif order["side"] == "BUY" and order["approval_id"]:
        # Its approval could not be read (get_approval_by_id returns None on
        # a DB error too): no stop level, so no record or outcome is written
        # here. The fill is booked; retry the stop from the queue and say so.
        log.critical(f"[{ticker}] reconcile: BUY fill booked but approval "
                     f"{order['approval_id']} could not be read; stop and outcome not written")
        stop_placed = False
        _report_stop_work_failed(
            ticker, "BUY", {"intent": "BUY"}, None,
            f"The BUY fill was booked, but its approval {order['approval_id']} could not be "
            f"read, so no stop record or outcome row was written here.")
        progress["stop_work_done"] = True
    elif order["side"] == "BUY" and (intent or "").startswith("MANUAL"):
        # A manual BUY has no approval carrying a stop level: keep the
        # position's stop if this added to it, else a flat fallback stop
        # below the fill, never above the last quote (professor M108).
        # Whether it added was read before sending; shares already
        # booked on a partial mean the position is this order's.
        import math
        action = ("ADD" if intent == "MANUAL_ADD" or prev_qty > 0 else
                  "OPEN" if intent == "MANUAL_OPEN" else None)
        try:
            if _stop_record_is_this_orders(ticker, keep_record_since, total_avg):
                pass   # the execution path's own record: leave it, cover-only below
            elif not db.get_position_stop(ticker) or not _held_before_fill(ticker, fill_qty, action):
                from execution import _MANUAL_FALLBACK_STOP_PCT as pct
                if not (math.isfinite(total_avg) and total_avg > 0):
                    raise ValueError(f"unusable fill price {total_avg}")
                stop_price = round(total_avg * (1 - pct), 2)
                try:
                    import market_data
                    q = market_data.get_priced_quote(ticker)
                    quote = float(q["price"]) if q and q.get("price") else None
                except Exception:
                    quote = None
                if quote and stop_price >= quote:
                    stop_price = round(quote * (1 - pct), 2)
                db.upsert_position_stop(
                    ticker=ticker, entry_price=total_avg,
                    stop_price=stop_price,
                    stop_method=f"manual late fill, flat {pct:.0%}",
                    bucket="Accelerator", conviction=5, trail_distance_pct=None,
                )
            stop_placed = _reprotect_after_late_sell(ticker, {}, None, from_queue=True)
        except Exception as e:
            log.error(f"[{ticker}] reconcile_unconfirmed_orders: manual BUY stop failed: {e}")
            stop_placed = False
        if stop_placed is not True:
            _queue_reprotect(ticker, {"intent": "BUY"}, None)
        progress["stop_work_done"] = True
    elif order["side"] == "SELL":
        # A late SELL fill (partial sale or TRIM whose stop was cancelled
        # before the sale) leaves the rest without a stop. Re-protect it at
        # the recorded stop (professor C12, 2026-10-28).
        stop_placed = _reprotect_after_late_sell(ticker, order, fill_price, exit_avg=total_avg,
                                                 exit_fill_at=fill.get("fill_time"))
        if stop_placed is False:
            # Busy ticker or a failed placement: the order is already
            # marked filled, so remember to retry (professor M50).
            _queue_reprotect(ticker, order, fill_price)
        progress["stop_work_done"] = True

    alert_lines = [
        f"{headline} — <b>{ticker}</b>",
        f"{order['side']} {fill_qty:g} shares @ ${fill_price:.2f} — {what}",
    ]
    if order["side"] == "BUY" or stop_placed is not None:
        alert_lines.append(
            "Protective stop placed." if stop_placed else
            "⚠️ Could not place a protective stop automatically — check TWS."
        )
    try:
        alerts._send("\n".join(alert_lines))
    except Exception:
        pass


def _report_stop_work_failed(ticker: str, side: str, queue_order: dict,
                             fill_price: float | None, what: str) -> None:
    """The stop work after a booked fill did not finish: queue the reprotect
    and send one critical alert, worded from what actually happened. A
    queued BUY reprotect places nothing without a stop record (it re-stops
    from the record), so a BUY with no record (or an unreadable one) is
    said so and opened as a manual action instead. Never raises."""
    import db
    try:
        _queue_reprotect(ticker, queue_order, fill_price)
        queued = True
    except Exception as e:
        log.error(f"[{ticker}] reconcile: could not queue the reprotect: {e}")
        queued = False
    has_record = True
    if side == "BUY":
        try:
            has_record = db.get_position_stop(ticker) is not None
        except Exception as e:
            log.error(f"[{ticker}] reconcile: stop record read failed: {e}")
            has_record = False
    if not has_record:
        tail = ("No stop record exists, so the automatic retry cannot place a stop — "
                "place it in TWS now.")
        try:
            open_manual_action(ticker, "place a stop after a booked BUY (no stop record)",
                               clear_on_flat=False)
        except Exception as e:
            log.error(f"[{ticker}] reconcile: could not open the manual action: {e}")
    elif queued:
        tail = "A stop retry is queued; check the stop against TWS now."
    else:
        tail = "The stop retry could NOT be queued; place or check the stop in TWS now."
    log.critical(f"[{ticker}] reconcile: stop work failed: {what} {tail}")
    try:
        import alerts
        alerts._send(f"🚨 STOP WORK FAILED — <b>{ticker}</b>\n{what} {tail}")
    except Exception as e:
        log.warning(f"[{ticker}] reconcile: stop-work-failed alert failed: {e}")


_TRADE_PENDING_STUCK_MINUTES = 30
_TRADE_PENDING_WHY = "filled within the wait, the worker stopped before the trade row"


def _trade_pending_stuck(order: dict, why: str) -> None:
    """A trade_pending row this run could not resolve: once it is older than
    _TRADE_PENDING_STUCK_MINUTES, one critical alert per throttle window, so
    it is never stuck silently. Never raises."""
    from datetime import datetime, timedelta, timezone
    ticker = order["ticker"]
    log.warning(f"[{ticker}] reconcile: trade_pending order {order['broker_order_id']} "
                f"not resolved this run: {why}")
    try:
        created = order.get("created_at")
        if created is None or datetime.now(timezone.utc) - created < \
                timedelta(minutes=_TRADE_PENDING_STUCK_MINUTES):
            return
        try:
            due = _reconcile_alert_due(f"trade_pending_stuck_{order['id']}")
        except Exception:
            due = True   # unknown: alert rather than stay silent
        if not due:
            return
        import alerts
        alerts._send(f"🚨 TRADE ROW PENDING — <b>{ticker}</b>\nThe {order['side']} order "
                     f"{order['broker_order_id']} was marked filled "
                     f"({float(order.get('filled_qty') or 0):g}) over "
                     f"{_TRADE_PENDING_STUCK_MINUTES} min ago but its trade row is still not "
                     f"confirmed: {why}. Check the trade rows and the stop against TWS.")
    except Exception as e:
        log.warning(f"[{ticker}] reconcile: trade-pending-stuck alert failed: {e}")


def _recover_trade_pending_orders(min_age_minutes: int = 10) -> None:
    """Filled/partial rows whose trade row was never written: the execution
    worker marked the fill (TRADE_PENDING_NOTE in the same write) and then
    stopped before its trade-row transaction cleared it. After
    min_age_minutes, compare the execution path's own trade rows
    (db.trade_logged_near) with the recorded fill:
    - none: claim the row and write the trade row in one transaction
      (db.claim_and_log_trade, note TRADE_BOOKED_BY_RECONCILE_NOTE), then do
      the stop and outcome work with _book_late_fill;
    - the same quantity: clear the note (the trade is there);
    - anything else: critical alert, row left as is.
    The claim is conditional on the note, so each row is handled once, and it
    waits on the execution path's own transaction for the same row. A row it
    can't resolve alerts once it is 30 min old (_trade_pending_stuck).
    Never raises."""
    import db
    import ibkr
    try:
        rows = db.get_stale_trade_pending_orders(min_age_minutes)
    except Exception as e:
        log.warning(f"reconcile: could not load trade_pending orders: {e}")
        return
    for order in rows:
        try:
            _recover_trade_pending_order(order, db, ibkr)
        except Exception as e:
            log.error(f"[{order.get('ticker')}] reconcile: trade_pending recovery crashed: {e}")
            _trade_pending_stuck(order, f"recovery error ({e})")


def _recover_trade_pending_order(order: dict, db, ibkr) -> None:
    ticker, oid = order["ticker"], order["broker_order_id"]
    qty = float(order.get("filled_qty") or 0)
    if qty <= 0:
        _trade_pending_stuck(order, "no filled quantity recorded")
        return
    try:
        booked = float(db.trade_logged_near(ticker, order["side"], order["created_at"],
                                            intent=order.get("intent"),
                                            order_id=order["id"]) or 0.0)
    except Exception as e:
        _trade_pending_stuck(order, f"the booked-trade check failed ({e})")
        return
    note = order.get("reconcile_note") or db.TRADE_PENDING_NOTE
    if abs(booked - qty) <= 1e-4:
        if db.claim_trade_pending(order["id"], None, expect_note=note) != 1:
            return
        log.critical(f"[{ticker}] reconcile: order {oid} was left trade_pending but its trade "
                     f"row ({booked:g}) exists; note cleared")
        try:
            import alerts
            alerts._send(f"🚨 TRADE ROW CHECK — <b>{ticker}</b>\nThe {order['side']} order "
                         f"{oid} filled {qty:g}; the worker stopped before it confirmed the "
                         f"trade row, but {booked:g} is recorded, so nothing was booked. "
                         f"Check the stop against TWS.")
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: trade-row-check alert failed: {e}")
        return
    if booked > 1e-6:
        log.critical(f"[{ticker}] reconcile: trade_pending order {oid} filled {qty:g} but "
                     f"{booked:g} is booked near it; left as is")
        try:
            due = _reconcile_alert_due(f"trade_pending_mismatch_{order['id']}")
        except Exception:
            due = True
        if due:
            try:
                import alerts
                alerts._send(f"🚨 TRADE ROW MISMATCH — <b>{ticker}</b>\nThe {order['side']} order "
                             f"{oid} is recorded as filled {qty:g}, but {booked:g} shares of its "
                             f"trades are recorded. Nothing was booked or cleared; check the "
                             f"trade rows against TWS.")
            except Exception as e:
                log.warning(f"[{ticker}] reconcile: trade-row-mismatch alert failed: {e}")
        return
    price, source = order.get("fill_price"), "fill"
    if not price:
        try:
            fill = ibkr.get_fill_for_order(oid, ticker=ticker, side=order["side"],
                                           placed_at=order.get("created_at"))
            price = (fill or {}).get("fill_price")
            if not price:
                price, source = ibkr.get_latest_price(ticker), "quote"
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: no price for trade_pending order {oid}: {e}")
            price = None
    if not price:
        _trade_pending_stuck(order, "no fill price yet")
        return
    stop_done = note == db.TRADE_PENDING_STOP_DONE_NOTE
    approval = _APPROVAL_NOT_READ
    if order.get("approval_id") and order["side"] == "BUY" and not stop_done:
        # The BUY's stop and outcome work needs its approval: read it before
        # claiming, so a failed read leaves the row for the next run (and the
        # 30-min alert) instead of booking it with no stop work.
        approval = db.get_approval_by_id(order["approval_id"])
        if approval is None:
            _trade_pending_stuck(order, f"approval {order['approval_id']} could not be read")
            return
    reason = _late_fill_reason(order.get("intent"),
                               "filled within the wait, the trade row failed" if stop_done
                               else _TRADE_PENDING_WHY)
    r = db.claim_and_log_trade(order["id"], db.TRADE_BOOKED_BY_RECONCILE_NOTE, ticker,
                               order["side"], qty, float(price), reason, source,
                               expect_note=note, order_row_id=order["id"])
    if r is False:
        return   # the worker (or another run) got there first
    if r is None:
        # The commit may have landed although it reported an error.
        try:
            landed = db.order_reconcile_note(order["id"]) == db.TRADE_BOOKED_BY_RECONCILE_NOTE
        except Exception as e:
            log.error(f"[{ticker}] reconcile: could not re-read order row {order['id']}: {e}")
            landed = False
        if not landed:
            _trade_pending_stuck(order, "the trade row could not be written (rolled back)")
            return
        log.warning(f"[{ticker}] reconcile: trade row for order {oid} reported an error but landed")
    if stop_done:
        # The execution path did the stop and outcome work itself; only its
        # trade row was missing.
        log.critical(f"[{ticker}] reconcile: order {oid} filled {qty:g}; its trade row failed "
                     f"at execution and is booked now (stop work was already done)")
        try:
            import alerts
            alerts._send(f"🚨 TRADE ROW RECOVERED — <b>{ticker}</b>\n{order['side']} {qty:g} "
                         f"shares @ ${float(price):.2f}: the trade row failed at execution and "
                         f"is booked now. The stop and outcome were already handled then.")
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: trade-row-recovered alert failed: {e}")
        return
    # Committed: the trade row and the claim. A crash before the stop work
    # below leaves it to reconcile_stop_orders (a held position with no
    # working stop is re-stopped from its record on the second cycle, or
    # alerted NO STOP / UNPROTECTED POSITION when there is no record).
    log.critical(f"[{ticker}] reconcile: order {oid} filled {qty:g} but the worker stopped "
                 f"before its trade row; booked it now")
    progress: dict = {}
    try:
        _book_late_fill(order, qty, float(price), float(price), 0.0,
                        fill_time=order.get("filled_at"), why=_TRADE_PENDING_WHY,
                        headline="🚨 TRADE ROW RECOVERED",
                        what=("the order row was marked filled but the worker stopped before "
                              "recording the trade; booked now, with the stop work it skipped."),
                        trade_logged=True, keep_record_since=order.get("created_at"),
                        approval=approval, progress=progress)
    except Exception as e:
        # The trade row is booked and the row claimed, so it is never
        # re-selected. Never re-claim or re-book.
        if progress.get("stop_work_done"):
            # The stop step finished (placed or queued); a later step failed.
            log.error(f"[{ticker}] reconcile: order {oid} booked and its stop handled, "
                      f"then a later step failed: {e}")
            return
        # Retry the stop from the queue and alert now (the stuck alert
        # waits 30 min).
        if order["side"] == "BUY":
            queue_order, queue_price = {"intent": "BUY"}, None
        else:
            queue_order, queue_price = order, float(price)
        _report_stop_work_failed(
            ticker, order["side"], queue_order, queue_price,
            f"The {order['side']} order {oid} ({qty:g} @ ${float(price):.2f}) is booked, but "
            f"the stop work after it failed ({e}).")


_REPROTECT_STATE_KEY = "reprotect_pending"


def _reprotect_queue_edit(fn, key: str = _REPROTECT_STATE_KEY) -> None:
    """Apply fn(pending) -> pending to the queue under a short Redis lock.
    Writers now include the LLM worker and the Telegram bot as well as the
    reconcile run, so an unlocked read-modify-write could drop an entry
    added in between (audit 06:53, A0653-2). A busy lock is waited out past
    its 30 s TTL, so a crashed holder can't force an unlocked edit (m37).
    Without Redis the edit still runs: losing the queue entry is worse than
    a rare race."""
    import db
    lock = None
    try:
        lock = stop_lock._redis().lock(f"helios:{key}", timeout=30, blocking_timeout=35)
        if not lock.acquire():
            lock = None
            log.error(f"{key} queue lock still busy after 35 s; editing without it")
    except Exception as e:
        lock = None
        log.warning(f"{key} queue lock unavailable ({e}); editing without it")
    try:
        try:
            pending = json.loads(db.get_state(key) or "{}")
        except Exception:
            pending = {}
        db.set_state(key, json.dumps(fn(pending)))
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _queue_reprotect(ticker: str, order: dict, fill_price: float | None) -> None:
    import uuid
    # The nonce makes a re-queue with the same fields a different entry, so a
    # retry pass running meanwhile can't pop it as the one it saw (m36).
    entry = {"intent": order.get("intent"), "stop_price": order.get("stop_price"),
             "fill_price": fill_price, "order_id": order.get("id"), "nonce": uuid.uuid4().hex}

    def add(pending):
        old = pending.get(ticker) or {}
        if old.get("intent") == "PROFIT_TARGET" and entry["intent"] != "PROFIT_TARGET":
            # Keep the target sale's breakeven-or-higher stop and its
            # target-taken step; a later RESTORE/BUY must not replace them
            # with the lower stop on record (professor M80).
            pending[ticker] = {**{k: v for k, v in old.items() if k != "tries"},
                               "nonce": entry["nonce"]}
            if (entry["intent"] or "").upper() in ("EXIT", "BACKSTOP"):
                # The flat close needs the exit's own order (professor m111).
                pending[ticker]["exit_intent"] = entry["intent"]
                pending[ticker]["exit_order_id"] = entry["order_id"]
                pending[ticker].pop("had_buy", None)   # the exit explains a flat read
            elif entry["intent"] == "BUY":
                # New shares: an old exit's fill must never close what reads
                # flat next (review of 245c9af). The entry keeps the target's
                # stop (M80) but gets the BUY's flat-read retry and alert.
                pending[ticker].pop("exit_intent", None)
                pending[ticker].pop("exit_order_id", None)
                pending[ticker]["had_buy"] = True
        else:
            pending[ticker] = entry
            if entry["intent"] in ("RESTORE", "PROFIT_TARGET") and \
                    (old.get("had_buy") or old.get("intent") == "BUY"):
                # A target sale or a RESTORE never empties the position, so it
                # doesn't explain a flat read after the BUY (reviews of b6b6a77,
                # 153335d). A TRIM or manual sell can, so it doesn't carry it.
                pending[ticker]["had_buy"] = True
            if (entry["intent"] or "").upper() not in ("EXIT", "BACKSTOP", "BUY"):
                # A RESTORE replacing a queued exit keeps that exit's order for
                # the flat close (review of 03efab7, as m111). Never a BUY: its
                # new shares make an old exit's fill the wrong close.
                if (old.get("intent") or "").upper() in ("EXIT", "BACKSTOP"):
                    pending[ticker].update(exit_intent=old["intent"], exit_order_id=old.get("order_id"))
                elif old.get("exit_intent"):
                    pending[ticker].update(exit_intent=old["exit_intent"],
                                           exit_order_id=old.get("exit_order_id"))
        return pending
    _reprotect_queue_edit(add)


_MANUAL_ACTIONS_KEY = "manual_actions"
_MANUAL_REALERT_S = 3600
_MANUAL_FLAT_ONLY_READS = 24  # cycles (~6 h) before a clear_on_flat=False item clears on flat
# A flat ticker's SELL stop left working while reconcile's orphan cleanup is
# on hold: the item clears once the ticker has no working SELL stop.
_HOLD_FLAT_STOP_KIND = "hold_flat_stop"


def _manual_actions_edit(fn) -> None:
    import db
    lock = None
    try:
        lock = stop_lock._redis().lock("helios:manual_actions", timeout=30, blocking_timeout=35)
        if not lock.acquire():
            lock = None
    except Exception:
        lock = None
    try:
        try:
            items = json.loads(db.get_state(_MANUAL_ACTIONS_KEY) or "{}")
        except Exception:
            items = {}
        db.set_state(_MANUAL_ACTIONS_KEY, json.dumps(fn(items)))
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def open_manual_action(ticker: str, what: str, clear_on_flat: bool = True,
                       kind: str | None = None) -> None:
    """Record that an alert asked Luke to fix a stop by hand. The stop check
    clears it once the position is covered or flat, and re-alerts hourly
    until then; one Telegram send used to be the only trace (audit 14:53,
    A1453-4)."""
    import time

    def add(items):
        now = time.time()
        rec = items.get(ticker) or {"first_seen": now}
        # clear_on_flat=False when the problem itself is a position that
        # reads flat: then flat is no evidence it's closed (professor C28a).
        items[ticker] = {**rec, "what": what, "last_alert": now, "clear_on_flat": clear_on_flat}
        # A kind changes how the item clears; an item re-opened without one
        # clears the usual way again.
        if kind:
            items[ticker]["kind"] = kind
        else:
            items[ticker].pop("kind", None)
        return items
    try:
        _manual_actions_edit(add)
    except Exception as e:
        log.warning(f"[{ticker}] could not record the manual action: {e}")


def _review_manual_actions() -> None:
    """Each stop-check cycle: clear manual-action items whose position is now
    covered by resting stops (or gone), re-alert the rest hourly."""
    import time
    import db
    import execution
    import ibkr
    try:
        items = json.loads(db.get_state(_MANUAL_ACTIONS_KEY) or "{}")
    except Exception:
        return
    if not items:
        return
    try:
        positions = ibkr.get_client().get_all_positions()
    except Exception as e:
        log.warning(f"manual-action review: position read failed ({e})")
        return
    # An empty read right after a reconnect looks flat for everything, so it
    # counts only after 4 in a row (~1 h); a real last position closing reads
    # empty too and must clear eventually (professor C28, C28b).
    empty = not positions
    held = {p.symbol: abs(float(p.qty)) for p in positions}
    stopped = None   # tickers with a working SELL stop; read only for hold_flat_stop items
    if any(rec.get("kind") == _HOLD_FLAT_STOP_KIND for rec in items.values()):
        try:
            open_ords = ibkr.get_open_orders()
            # An empty read can be a reconnect gap, not "no stops": then these
            # items wait for a later read (or the usual ~6 h flat rule).
            if open_ords:
                stopped = {o["ticker"] for o in _working_sell_stops(open_ords)}
        except Exception as e:
            log.warning(f"manual-action review: open-orders read failed ({e})")
    now = time.time()
    cleared, realert, flat_once = [], [], []
    for ticker, rec in items.items():
        qty = held.get(ticker, 0.0)
        if (rec.get("kind") == _HOLD_FLAT_STOP_KIND and stopped is not None
                and ticker not in stopped):
            cleared.append(ticker)   # the stop it was about is gone
            continue
        covered = execution._stops_still_resting(ticker) if qty > 0 else None
        need = 3 if empty else 1  # earlier flat reads needed, besides this one
        if qty > 0 and covered is not None and covered >= qty:
            cleared.append(ticker)
        elif qty <= 0 and int(rec.get("flat_reads", 0)) >= (
                need if rec.get("clear_on_flat", True) else _MANUAL_FLAT_ONLY_READS):
            # A "reads flat" item still clears after ~6 h flat, so shares sold
            # by hand don't re-alert forever.
            cleared.append(ticker)  # flat on 2 cycles in a row, or 4 if the reads are empty (C28)
        else:
            if qty <= 0:
                flat_once.append(ticker)
            if now - float(rec.get("last_alert", 0)) >= _MANUAL_REALERT_S:
                realert.append(ticker)
    for ticker in realert:
        rec = items[ticker]
        mins = int((now - float(rec.get("first_seen", now))) / 60)
        try:
            import alerts
            alerts._send(f"🔁 STILL OPEN — <b>{ticker}</b>\nAsked {mins} min ago and still not "
                         f"covered: {rec.get('what', 'place a stop')}. Check TWS.")
        except Exception:
            pass

    def apply(current):
        for t in cleared:
            if t in current:
                log.info(f"[{t}] manual action cleared: position covered or closed")
                current.pop(t)
        for t in realert:
            if t in current:
                current[t]["last_alert"] = now
        for t in current:
            if t in flat_once:
                current[t]["flat_reads"] = int(current[t].get("flat_reads", 0)) + 1
            elif t not in cleared:
                current[t]["flat_reads"] = 0
        return current
    _manual_actions_edit(apply)


# Queued re-protects are retried by reconcile_unconfirmed_orders, every 5 min
# (the comments here used to assume 15, professor m112). Kept at 4 and 2: a
# position left without a stop should reach Luke in ~20 min, not ~1 h.
_REPROTECT_MAX_TRIES = 4  # ~20 min of failed placements
_BUY_FLAT_MAX_TRIES = 2   # ~10 min for a new BUY's position to show up
_EXIT_WAIT_MAX = 48       # 5-min reconcile runs a queued flat EXIT waits on its remainder (~4 h)


def _close_queued_flat_exit(ticker: str, rec: dict) -> bool:
    """A queued re-protect of a part-filled EXIT/BACKSTOP found the position
    flat. The stale-stop branch can't close it: the sale may have deleted the
    stop record, and when it hasn't (a part-filled backstop keeps it), that
    branch leaves this order's own fills out and would read no_fill (professor
    M124; review of f464bd7). Returns True to keep waiting.
    - Order row filled: close at its cumulative average fill, then drop any
      stop record.
    - Remainder still working at the broker, or a lookup failed: wait.
    - Remainder no longer working (something else sold the rest): the exit
      price for that mix is an open question for Luke, so leave it as before.
    close_trade_outcome only closes open rows not flagged orphaned."""
    import db
    import ibkr
    oid = rec.get("order_id")
    if not oid:
        return False   # queued before this field existed
    try:
        row = db.get_order_fill_state(oid)
    except Exception as e:
        log.warning(f"[{ticker}] reconcile: queued flat EXIT lookup failed ({e}); waiting")
        return True
    if not row:
        return False
    if row["status"] == "filled" and row.get("fill_price"):
        # Only this exit's own position: a record or an open outcome newer than
        # the order belongs to a position opened since, and closing it here
        # would book the old exit's price (review of 6f48241).
        try:
            placed = row.get("created_at")
            # strict: a failed read must wait, not pass as "nothing newer" (m114).
            rec_stop = db.get_position_stop(ticker, strict=True) or {}
            newer = placed is None or any(
                t is not None and t > placed
                for t in [rec_stop.get("opened_at")] + [o.get("entry_date") for o in db.get_open_outcomes(strict=True)
                                                         if o.get("ticker") == ticker])
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: queued flat exit position check failed ({e}); waiting")
            return True
        if newer:
            log.warning(f"[{ticker}] reconcile: a position newer than the queued {rec.get('intent')} order "
                        f"exists; its flat close is left to the stale-stop branch")
            return False
        try:
            db.close_trade_outcome(ticker, exit_price=row["fill_price"],
                                   exit_source=f"late_{(rec.get('intent') or 'sell').lower()}",
                                   exit_price_source="fill", exit_fill_at=row.get("filled_at"))
            db.delete_position_stop(ticker)
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: close after a queued flat exit failed: {e}; waiting")
            return True
        return False
    if row["status"] in ("partial", "unconfirmed"):
        try:
            working = any(o.get("order_id") == row.get("broker_order_id")
                          and o.get("status") not in _NOT_WORKING for o in ibkr.get_open_orders())
        except Exception as e:
            log.warning(f"[{ticker}] reconcile: open-order read failed ({e}); queued flat EXIT waiting")
            return True
        if working:
            # Flat while our sell still works: if it fills, it opens a short
            # (review of 11e2f26). Luke decides; one throttled alert.
            try:
                if _reconcile_alert_due(f"flat_exit_working_{ticker}"):
                    import alerts
                    alerts._send(f"🚨 SELL STILL WORKING ON A FLAT POSITION — <b>{ticker}</b>\nThe "
                                 f"broker shows no shares, but the rest of our {rec.get('intent')} order "
                                 f"is still working. If it fills it opens a SHORT. Check TWS and cancel "
                                 f"it if the shares are gone.")
            except Exception as e:
                log.warning(f"[{ticker}] reconcile: flat-with-working-sell alert failed: {e}")
            return True
        # Either something else sold the rest (exit price for that mix is
        # undecided), or the remainder filled since this run started and the
        # order loop below closes it.
        log.info(f"[{ticker}] reconcile: flat after a part-filled {rec.get('intent')} whose remainder "
                 f"is not working; leaving the close to the order loop or as an open mixed close")
        return False
    if row["status"] == "filled":
        log.warning(f"[{ticker}] reconcile: queued flat {rec.get('intent')} order {oid} is filled "
                    f"with no fill price; outcome not closed")
    return False


_OUTCOME_STATE_KEY = "buy_outcome_pending"
_OUTCOME_MAX_TRIES = 24   # reconcile_unconfirmed_orders runs, 5 min apart (~2 h)


def _write_buy_outcome(approval_id: int, rec: dict, approval: dict | None = None) -> bool:
    import db
    try:
        if rec["kind"] == "reaverage":
            updated = db.update_outcome_entry_price(approval_id, rec["entry_price"])
            if updated is None:
                return False          # write failed: retry
            if updated:
                return True
            # No open row: the part fill's own row was never written, so insert
            # it at the whole order's average (deduped per approval; review of
            # fed009b). If the row exists but is closed, it keeps its entry.
            log.warning(f"[{rec['ticker']}] outcome re-average for approval {approval_id}: no open "
                        f"row; inserting if missing (a closed row keeps its part-fill entry price)")
        from execution import log_buy_outcome
        approval = approval or db.get_approval_by_id(approval_id)
        return bool(approval) and bool(log_buy_outcome(
            approval, rec["ticker"], rec["action"], rec["entry_price"],
            entry_date=rec.get("entry_date"), entry_benchmark_price=rec.get("entry_benchmark_price")))
    except Exception as e:
        log.error(f"[{rec.get('ticker')}] outcome {rec.get('kind')} for approval "
                  f"{approval_id} failed: {e}")
        return False


def _apply_buy_outcome(approval_id: int, rec: dict, approval: dict | None = None) -> None:
    """Write a BUY's outcome row (or re-average it); queue it for the next
    reconcile runs if that fails (professor M125). Called at the fill and by
    reconcile, so queue edits take the queue lock."""
    if _write_buy_outcome(approval_id, rec, approval):
        return
    log.error(f"[{rec['ticker']}] BUY outcome {rec['kind']} failed; queued for retry")
    # A row written by a retry keeps this attempt's time and SPY price, not
    # the retry's (SEF, 10-30): alpha would otherwise mix times.
    from datetime import datetime, timezone
    stamp = {"entry_date": datetime.now(timezone.utc).isoformat()}
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").fast_info.last_price
        if spy:
            stamp["entry_benchmark_price"] = float(spy)
    except Exception:
        pass

    def add(pending):
        # A later re-average carries the newer price; the first attempt's
        # stamp is kept, so either kind ends in a row timed at the fill.
        old = pending.get(str(approval_id)) or {}
        first = {k: old[k] for k in ("entry_date", "entry_benchmark_price") if k in old}
        pending[str(approval_id)] = {**rec, **stamp, **first, "tries": 0}
        return pending
    try:
        _reprotect_queue_edit(add, key=_OUTCOME_STATE_KEY)
    except Exception as e:
        log.error(f"[{rec['ticker']}] could not queue the outcome retry: {e}")


def _retry_pending_buy_outcomes() -> None:
    import db
    try:
        pending = json.loads(db.get_state(_OUTCOME_STATE_KEY) or "{}")
    except Exception:
        return
    if not pending:
        return
    outcome = {}  # approval id -> (entry seen, entry to keep or None)
    for aid, rec in pending.items():
        if _write_buy_outcome(int(aid), rec):
            log.info(f"[{rec['ticker']}] reconcile: queued outcome {rec['kind']} for approval {aid} done")
            outcome[aid] = (rec, None)
            continue
        tries = int(rec.get("tries", 0)) + 1
        if tries < _OUTCOME_MAX_TRIES:
            outcome[aid] = (rec, {**rec, "tries": tries})
            continue
        log.error(f"[{rec['ticker']}] reconcile: outcome {rec['kind']} for approval {aid} gave up "
                  f"after {tries} tries")
        try:
            import alerts
            alerts._send(f"⚠️ OUTCOME NOT RECORDED — <b>{rec['ticker']}</b>\nA filled BUY "
                         f"(approval {aid}) could not be written to trade_outcomes for ~2 h. "
                         f"Trading is unaffected; the research record needs a manual fix.")
        except Exception:
            pass
        outcome[aid] = (rec, None)

    def apply(current):
        for aid, (seen, keep) in outcome.items():
            if current.get(aid) != seen:
                continue  # re-queued while we worked: the newer entry stays
            if keep is None:
                current.pop(aid, None)
            else:
                current[aid] = keep
        return current
    try:
        _reprotect_queue_edit(apply, key=_OUTCOME_STATE_KEY)
    except Exception as e:
        log.error(f"reconcile: could not save the outcome retry queue: {e}")


def _retry_pending_reprotects() -> None:
    """Re-protect after late SELL fills whose first attempt failed or found
    the ticker busy (professor M50). Runs every reconcile cycle until each
    one places a stop or finds nothing held. Works from a snapshot, then
    updates only the tickers it processed, and only if their entry wasn't
    replaced meanwhile (A0653-2)."""
    import db
    try:
        pending = json.loads(db.get_state(_REPROTECT_STATE_KEY) or "{}")
    except Exception:
        pending = {}
    if not pending:
        return
    outcome = {}  # ticker -> (entry seen, entry to keep or None)
    for ticker, rec in pending.items():
        # Take the ticker lock here so a busy ticker can be told apart from a
        # failed placement: busy is not a try (pre-pull review #5, 10-29).
        try:
            with stop_lock.ticker_lock(ticker):
                fn = getattr(_reprotect_after_late_sell, "__wrapped__", _reprotect_after_late_sell)
                result = fn(ticker, rec, rec.get("fill_price"), from_queue=True)
        except stop_lock.TickerBusy:
            log.info(f"[{ticker}] reconcile: queued re-protect skipped, ticker busy; not counted")
            continue
        if result == "wait":
            # The EXIT's remainder is still working: keep the entry, untried,
            # for up to ~4 h (professor M124).
            waits = int(rec.get("waits", 0)) + 1
            if waits < _EXIT_WAIT_MAX:
                outcome[ticker] = (rec, {**rec, "waits": waits})
            else:
                log.warning(f"[{ticker}] reconcile: queued flat EXIT still waiting after {waits} cycles; dropped")
                outcome[ticker] = (rec, None)
            continue
        tries = int(rec.get("tries", 0)) + 1
        # A BUY that still reads flat is retried for 2 runs (~10 min; they
        # are 5 min apart, professor m55/m112), then handed to Luke like a
        # failed retry (professor M60).
        flat_buy = result is None and (rec.get("intent") == "BUY" or rec.get("had_buy"))
        if flat_buy and tries >= _BUY_FLAT_MAX_TRIES:
            log.error(f"[{ticker}] reconcile: queued BUY stop still reads flat after {tries} tries")
            try:
                import alerts
                alerts._send(f"🚨 STOP NOT PLACED — <b>{ticker}</b>\nAfter a BUY the broker still "
                             f"shows no position (~10 min), so no stop was placed. Check TWS: if the "
                             f"shares are there, place the stop by hand.")
            except Exception:
                pass
            open_manual_action(ticker, "the stop after a BUY was never placed", clear_on_flat=False)
            outcome[ticker] = (rec, None)
        elif result is False and tries >= _REPROTECT_MAX_TRIES:
            # ~20 min of failures: stop retrying (each try cancels and
            # re-places) and hand it to Luke (professor M79, m112).
            log.error(f"[{ticker}] reconcile: queued re-protect failed {tries} times; giving up")
            sign_flip = ticker in _REPROTECT_SIGN_FLIP
            try:
                import alerts
                if sign_flip:
                    # A SELL stop on a short adds to it: don't say "place one".
                    alerts._send(f"🚨 STOP NOT RESTORED — <b>{ticker}</b>\nThe automatic stop "
                                 f"retry stopped after {tries} tries (~20 min): a fresh read shows "
                                 f"the position short while the portfolio cache shows it long. "
                                 f"Check in TWS whether it is long or short before placing "
                                 f"anything; a SELL stop on a short adds to it.")
                else:
                    alerts._send(f"🚨 STOP NOT RESTORED — <b>{ticker}</b>\nThe automatic stop "
                                 f"retry failed {tries} times (~20 min) and has stopped. Check the "
                                 f"position and place a stop in TWS; the stop check keeps flagging "
                                 f"it until covered.")
            except Exception:
                pass
            open_manual_action(ticker, "check whether the position is long or short (the reads "
                                       "disagree) before placing a stop" if sign_flip
                               else "the automatic stop retry gave up")
            outcome[ticker] = (rec, None)
        elif result is False or flat_buy:
            outcome[ticker] = (rec, {**rec, "tries": tries})
        else:
            log.info(f"[{ticker}] reconcile: queued re-protect done ({result})")
            outcome[ticker] = (rec, None)

    def apply(current):
        for ticker, (seen, keep) in outcome.items():
            if current.get(ticker) != seen:
                continue  # re-queued while we worked: the newer entry stays
            if keep is None:
                current.pop(ticker, None)
            else:
                current[ticker] = keep
        return current
    _reprotect_queue_edit(apply)


def _held_before_fill(ticker: str, fill_qty: float, action: str | None = None) -> bool:
    """True if this late BUY added to a position (keep its stop record),
    False if it opened one (an existing record is a closed position's
    leftover). The approval's action decides; the broker cache can read
    flat or count the new shares right after a fill (audit 02:53, A0253-4).
    Without an action, falls back to the position read; a failed read keeps
    the record."""
    if action in ("ADD", "OPEN"):
        return action == "ADD"
    import ibkr
    try:
        positions = ibkr.get_client().get_all_positions()
    except Exception as e:
        log.warning(f"[{ticker}] could not read position to check the stop record: {e}")
        return True
    pos = next((p for p in positions if p.symbol == ticker), None)
    held = float(pos.qty) if pos else 0.0
    return round(held - float(fill_qty), 4) > 0


# Tickers whose last _reprotect_after_late_sell try was refused because a
# fresh read showed a short, for the queue's give-up alert (same process).
_REPROTECT_SIGN_FLIP: set[str] = set()


def _sign_flip_alert(ticker: str, cache_qty: float, fresh_qty: float) -> None:
    """The cache reads long and a fresh read short: no SELL stop goes on (it
    would add to a short). One 🚨, throttled on the same key as reconcile's
    own sign-flip alert (sign_flip_<T>), so the two don't double up."""
    log.error(f"[{ticker}] position reads disagree on sign (cache {cache_qty:g}, fresh "
              f"{fresh_qty:g}); no SELL stop added")
    try:
        due = _reconcile_alert_due(f"sign_flip_{ticker}")
    except Exception as e:
        log.warning(f"[{ticker}] sign-flip alert throttle unreadable ({e}); sending")
        due = True
    if due:
        try:
            import alerts
            alerts._send(f"🚨 STOP CHECK — <b>{ticker}</b>\nThe position reads disagree "
                         f"on sign: the portfolio cache shows {cache_qty:g} shares, a "
                         f"fresh read shows {fresh_qty:g}. No SELL stop added and "
                         f"nothing cancelled. Check the position in TWS now.")
        except Exception as e:
            log.error(f"[{ticker}] could not send the sign-flip alert: {e}")


def _fresh_read_short(ticker: str, cache_qty: float) -> bool:
    """True (and the sign-flip alert) when the cache reads ticker long and a
    fresh reqPositions() read shows it short. A failed read or one without
    the ticker is False: the caller goes on as before."""
    import ibkr
    if cache_qty <= 0:
        return False
    try:
        fresh = float(ibkr.get_fresh_stock_positions().get(ticker) or 0.0)
    except Exception as e:
        log.warning(f"[{ticker}] fresh position read for the sign check failed ({e})")
        return False
    if fresh < 0:
        _sign_flip_alert(ticker, cache_qty, fresh)
        return True
    return False


@locked_by_ticker(lambda ticker, *a, **k: ticker, on_busy=False)
def _reprotect_after_late_sell(ticker: str, order: dict | None = None,
                               fill_price: float | None = None,
                               from_queue: bool = False,
                               expect_at_least: float | None = None,
                               exit_avg: float | None = None,
                               exit_fill_at=None) -> bool | None | str:
    """After a late SELL fill, finish what the selling path would have done
    (professor M-3, 2026-10-28):
    - nothing left: delete the stop record and close the outcome at the fill
      price, so stale-stop cleanup doesn't log the exit a second time;
    - a profit-target sale: mark the target taken (no second sale) and protect
      the rest at the stop that sale computed (breakeven or higher);
    - otherwise: cancel resting stops and protect the rest at the stop on record.
    Returns True if a stop was placed, False if it should have been but failed,
    None if nothing is held, and (queued retries only) "wait" while a flat
    EXIT's remainder is still working."""
    import db
    import ibkr
    order = order or {}
    _REPROTECT_SIGN_FLIP.discard(ticker)
    try:
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        remaining = round(abs(float(pos.qty)), 4) if pos else 0.0
    except Exception as e:
        log.error(f"[{ticker}] reconcile: could not read the position after a late sell: {e}")
        return False
    if remaining <= 0 and not from_queue:
        # One cached read can show a false flat (07-29 KLAC): confirm once.
        import time
        time.sleep(3)
        try:
            pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
            remaining = round(abs(float(pos.qty)), 4) if pos else 0.0
        except Exception:
            return False
    if pos is not None and float(pos.qty) < 0:
        # A SELL stop on a short adds to it (professor M63 residual): alert
        # and drop the retry rather than place one.
        log.error(f"[{ticker}] reconcile: broker shows a SHORT ({float(pos.qty):g}); no stop placed")
        try:
            import alerts
            alerts._send(f"🚨 SHORT POSITION — <b>{ticker}</b>\n{float(pos.qty):g} shares at the "
                         f"broker. No stop was placed. Check TWS now.")
        except Exception:
            pass
        return None
    if from_queue and pos is not None and _fresh_read_short(ticker, float(pos.qty)):
        # A queued restore after a sign-flip refusal (execution
        # _restore_sized_stop): keep retrying until the reads agree.
        _REPROTECT_SIGN_FLIP.add(ticker)
        return False
    if expect_at_least and 0 < remaining < float(expect_at_least) - 1e-6:
        # The read lags a fill it should include (a manual ADD): resting stops
        # sized for the old position would look like full cover (review of
        # 2f95331). Not confirmed, so the caller queues a retry.
        log.warning(f"[{ticker}] reconcile: broker shows {remaining:g} shares, expected at least "
                    f"{float(expect_at_least):g}; stop not confirmed yet")
        return False
    if remaining <= 0 and from_queue:
        # A queued retry carries an old fill price, and the rest may have
        # exited since by its stop: leave the close to reconcile_stop_orders'
        # stale-stop branch, which logs the real exit (A2253-2). A part-filled
        # EXIT/BACKSTOP deleted that record, so close from its order (M124).
        exit_rec = (order if (order.get("intent") or "").upper() in ("EXIT", "BACKSTOP") else
                    {"intent": order["exit_intent"], "order_id": order.get("exit_order_id")}
                    if order.get("exit_intent") else None)
        if exit_rec and _close_queued_flat_exit(ticker, exit_rec):
            return "wait"
        return None
    if remaining <= 0:
        had_record = bool(db.get_position_stop(ticker))
        if had_record:
            db.delete_position_stop(ticker)
        # Flat: close from the order's cumulative average fill. An EXIT also
        # closes without a stop record: a part-filled EXIT deleted it and
        # left the outcome for this close (professor M121). Rows already
        # closed or flagged orphaned are skipped by close_trade_outcome.
        exit_price = exit_avg or fill_price
        if exit_price and (had_record or (order.get("intent") or "").upper() in ("EXIT", "BACKSTOP")):
            try:
                db.close_trade_outcome(ticker, exit_price=exit_price,
                                       exit_source=f"late_{(order.get('intent') or 'sell').lower()}",
                                       exit_price_source="fill",   # broker average fill
                                       exit_fill_at=exit_fill_at)
            except Exception as e:
                log.warning(f"[{ticker}] reconcile: close_trade_outcome failed: {e}")
        return None
    stop_price = (db.get_position_stop(ticker) or {}).get("stop_price")
    if order.get("intent") == "PROFIT_TARGET" and order.get("stop_price"):
        stop_price = order["stop_price"]
        db.mark_profit_target_taken(ticker, float(stop_price))
    if not stop_price:
        log.error(f"[{ticker}] reconcile: {remaining} shares left after a late sell, no stop on record")
        return False
    open_orders = None
    try:
        import ibkr
        open_orders = ibkr.get_open_orders()
        selling = _working_mkt_sell_qty(open_orders, ticker)
    except Exception as e:
        if from_queue:
            # A queued retry waits rather than stop blind: it counts a try,
            # and the last one alerts and opens a manual action (m116).
            log.warning(f"[{ticker}] reconcile: open-order read failed ({e}); queued stop waits")
            return False
        # Unprotected is worse than over-stopped: section 2b fixes the latter.
        log.warning(f"[{ticker}] reconcile: open-order read failed ({e}); stopping all {remaining} shares")
        selling = 0.0
    if selling > 0:
        # The rest of the sale is still working: protect only what it won't
        # sell (professor M129).
        remaining = round(remaining - selling, 4)
        if remaining <= 0:
            log.info(f"[{ticker}] reconcile: a working SELL covers the remaining shares; no stop placed")
            return True
    if from_queue:
        # Sized from the all-clients view (m116): a stop the cancel can't
        # remove (another client's, or one placed in TWS) stays as cover, and
        # ours goes on only the shares it leaves. position_stops.stop_price
        # stays ours.
        import execution
        covered = execution._stops_still_resting(ticker)
        if covered is None:
            log.warning(f"[{ticker}] reconcile: resting stops could not be read; queued stop waits")
            return False
        if order.get("intent") != "PROFIT_TARGET" and covered >= remaining:
            log.info(f"[{ticker}] reconcile: {remaining} shares already covered by resting stops "
                     f"({covered:g}); nothing cancelled")
            return True
        if covered > 0:
            # Cancel ours, then see what the cancel could not remove.
            try:
                manage_cancel_stop_orders_for(ticker)
            except Exception as e:
                log.warning(f"[{ticker}] reconcile: queued stop could not cancel stops ({e}); waits")
                return False
            left = None
            for attempt in (1, 2):   # a timed-out read drops the session; the second reconnects
                try:
                    left = _resting_sell_stops(ticker)
                    if attempt == 1 and any(o.get("status") == "PendingCancel" for o in left):
                        import time
                        time.sleep(5)   # a slow cancel; same second look as _fix_overstop
                        continue
                    break
                except Exception as e:
                    log.warning(f"[{ticker}] reconcile: re-read after the cancel failed (try {attempt}/2): {e}")
            if left is None:
                # Ours is gone and only cover went down: unprotected is worse
                # than over-stopped, so stop all of it; _fix_overstop trims.
                log.warning(f"[{ticker}] reconcile: stops unreadable after the cancel; stopping all {remaining}")
                left = []
        else:
            left = []
        others = round(sum(o["qty"] for o in left), 4)
        need = round(remaining - others, 4)
        if others > 0:
            prices = ", ".join(f"${float(o['stop_price']):.2f}" if o.get("stop_price") else "?" for o in left)
            log.warning(f"[{ticker}] reconcile: {others:g} shares stay covered by stop(s) the cancel did not "
                        f"remove ({prices}); ours @ ${float(stop_price):.2f} goes on {max(need, 0):g}")
        if need <= 0:
            return True
        try:
            result = manage_stop_place(ticker, need, float(stop_price))
            db.set_stop_broker_order_id(ticker, result["order_id"])
            log.info(f"[{ticker}] reconcile: stop @ ${float(stop_price):.2f} placed for {need} shares")
            return True
        except Exception as e:
            log.error(f"[{ticker}] reconcile: stop placement after a late sell failed: {e}")
            return False
    if order.get("intent") != "PROFIT_TARGET":
        # Already covered (e.g. a stop Luke placed after a "place manually"
        # alert, or an earlier try that did land): cancelling it to re-place
        # ours would leave a gap each cycle (professor M79). A profit-target
        # entry still re-places, since it raises the stop.
        import execution
        covered = execution._stops_still_resting(ticker)
        if covered is not None and covered >= remaining:
            log.info(f"[{ticker}] reconcile: {remaining} shares already covered by resting stops "
                     f"({covered:g}); nothing cancelled")
            return True
    try:
        manage_cancel_stop_orders_for(ticker)
        result = manage_stop_place(ticker, remaining, float(stop_price))
        db.set_stop_broker_order_id(ticker, result["order_id"])
        log.info(f"[{ticker}] reconcile: stop @ ${float(stop_price):.2f} placed for {remaining} shares")
        return True
    except Exception as e:
        log.error(f"[{ticker}] reconcile: stop placement after a late sell failed: {e}")
        return False


# ── New tasks: regime + scanner + stop monitor ────────────────────────────────

@app.task(name="tasks.assess_regime_daily", bind=True, max_retries=2)
def assess_regime_daily(self):
    """Daily market regime assessment at 9:30 AM ET."""
    try:
        from agents.market_regime import assess_regime
        regime = assess_regime()
        log.info(f"Regime assessed: {regime['regime']} (score={regime['score']})")
    except Exception as e:
        log.error(f"assess_regime_daily failed: {e}")
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.assess_gics_sector_map", bind=True, max_retries=2)
def assess_gics_sector_map(self):
    """Daily full-market GICS sector regime sweep at 9:45 AM ET — see
    agents.market_regime.assess_market_sector_map's docstring."""
    try:
        from agents.market_regime import assess_market_sector_map
        results = assess_market_sector_map()
        summary = ", ".join(f"{etf}={p['score']}" for etf, p in results.items())
        log.info(f"GICS sector map assessed: {summary}")
    except Exception as e:
        log.error(f"assess_gics_sector_map failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.run_candidate_scan", bind=True, max_retries=1)
def run_candidate_scan(self):
    """Weekly new-ticker/watchlist-candidate discovery sweep — see
    agents.candidate_scanner's docstring. Output is a shortlist for human
    review only; nothing here touches the actual watchlist or places any
    order."""
    try:
        from agents.candidate_scanner import run_candidate_scan as scan
        result = scan()
        log.info(result["scan_notes"])
        _heartbeat_ok("candidate_scan", len(result["candidates"]))

        top = result["candidates"][:5]
        if top:
            import alerts
            lines = [f"Watchlist candidate scan: {result['scan_notes']}",
                     f"Price bars: median {result.get('median_bar_age')} weekdays old, "
                     f"{(result.get('stale_ticker_share') or 0):.0%} of tickers stale "
                     f"(newest {result.get('newest_bar') or 'unknown'})"
                     + (" — STALE, movement scores are old" if result.get("stale_bars") else ""), ""]
            for c in top:
                lines.append(
                    f"• {c['ticker']} ({c['company']}) — composite {c['composite_score']}, "
                    f"{c['gics_sector']}"
                )
            alerts._send("\n".join(lines))
    except Exception as e:
        log.error(f"run_candidate_scan failed: {e}")
        _heartbeat_fail("candidate_scan", str(e))
        raise self.retry(exc=e, countdown=600)


@app.task(name="tasks.run_daily_scan", bind=True, max_retries=2)
def run_daily_scan(self):
    """Daily scanner run at 9:35 AM ET — surfaces candidates for analyst."""
    try:
        from agents.market_regime import get_cached_regime
        from agents.scanner import run_scan

        regime    = get_cached_regime()
        scan_result = run_scan(regime)
        candidates  = scan_result.get("candidates", [])
        high_medium = [c for c in candidates if c["priority"] in ("HIGH", "MEDIUM")]

        log.info(f"Daily scan: {len(high_medium)} candidates to analyze")
        try:
            log.info(_scan_summary(regime, candidates))
        except Exception as e:
            log.debug(f"scan summary failed: {e}")
        if high_medium:
            analyze_candidates.apply_async(
                args=[high_medium], kwargs={"max_per_run": 5},
                queue=LLM_QUEUE_BULK, expires=LLM_EVENT_EXPIRES_SECONDS,
            )
    except Exception as e:
        log.error(f"run_daily_scan failed: {e}")
        raise self.retry(exc=e, countdown=300)


_US_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "NYS", "ASE", "NAS", "NYSE", "NASDAQ"}


def _find_us_listing(company: str) -> dict | None:
    """A US-listed common stock whose name starts with `company`, or None.
    The yfinance search also returns ETFs, tokenized stocks and funds named
    after the company (checked for Anthropic on 2026-10-28), so only
    quoteType EQUITY on a major US exchange counts."""
    import yfinance as yf
    for q in yf.Search(company, max_results=10).quotes:
        name = (q.get("longname") or q.get("shortname") or "").strip()
        if (q.get("quoteType") == "EQUITY" and q.get("exchange") in _US_EXCHANGES
                and name.lower().startswith(company.lower())):
            return {"symbol": q.get("symbol"), "exchange": q.get("exchange"), "name": name}
    return None


@app.task(name="tasks.check_ipo_watch", bind=True, max_retries=1)
def check_ipo_watch(self):
    """Daily: has any IPO-watch company (knowledge/watchlist.md) listed? One
    Telegram alert per company when it has; the lead then activates it on
    the watchlist and logs that (Luke, 2026-10-28: Anthropic)."""
    import db
    from knowledge_loader import load_ipo_watch
    for entry in load_ipo_watch():
        company = entry["company"]
        key = f"ipo_watch_listed_{company.lower()}"
        if db.get_state(key):
            continue
        try:
            listing = _find_us_listing(company)
        except Exception as e:
            log.warning(f"check_ipo_watch: search for {company} failed: {e}")
            continue
        if not listing:
            log.info(f"check_ipo_watch: {company} not listed yet")
            continue
        db.set_state(key, listing["symbol"])
        log.warning(f"check_ipo_watch: {company} appears listed as {listing['symbol']} "
                    f"({listing['exchange']}, {listing['name']})")
        import alerts
        alerts._send(f"📈 <b>IPO watch</b>: {company} appears listed as "
                     f"<b>{listing['symbol']}</b> ({listing['exchange']}). Not tradeable "
                     f"until it is activated on the watchlist.")


def _scan_summary(regime: dict, candidates: list[dict]) -> str:
    """One line per daily scan: regime, threshold, counts by priority, top
    scores, and regime inputs that were unavailable. On 10-15 the regime moved
    55 -> 53, the MEDIUM threshold went 50 -> 65 and candidates fell from 20-33
    a day to 1-2 without anything saying why (audit 2026-10-28 05:00, #4)."""
    from collections import Counter
    from agents.scanner import _regime_threshold
    name = regime.get("regime", "Neutral")
    counts = Counter(c.get("priority") for c in candidates)
    top = sorted((c.get("composite_score") or 0 for c in candidates), reverse=True)[:3]
    missing = regime.get("missing_inputs") or (regime.get("signals") or {}).get("missing_inputs") or []
    return (f"Scan summary: regime {name} score {regime.get('score')} -> threshold "
            f"{_regime_threshold(name)}; HIGH {counts.get('HIGH', 0)} MEDIUM {counts.get('MEDIUM', 0)} "
            f"LOW {counts.get('LOW', 0)} PASS {counts.get('PASS', 0)}; top scores {top}"
            + (f"; regime inputs unavailable: {', '.join(missing)}" if missing else "")
            # Which row and which bars (A0853-1, A1053-2).
            + f"; regime source {regime.get('regime_source') or 'fresh'} @ {regime.get('timestamp')}"
            + f", last bar {(regime.get('signals') or {}).get('last_bar_date')}")


def _log_exit_routes() -> None:
    """One activity row per review run: for each held position, which
    automatic exit routes its bucket has (exempt_by_bucket, audit
    A1031-1453 N3), and how many have none beyond the stop and a model
    TRIM/EXIT. Log-only; never raises."""
    try:
        import db
        from risk_manager import exit_routes_for_bucket
        held = {}
        for rec in db.get_all_position_stops():
            routes = exit_routes_for_bucket(rec.get("thesis_bucket"))
            held[rec["ticker"]] = {"bucket": rec.get("thesis_bucket"),
                                   "exempt": sorted(k for k, on in routes.items() if not on)}
        no_route = sorted(t for t, h in held.items()
                          if not {"profit_target", "stale_flag", "swap_eligible"} - set(h["exempt"]))
        db.log_activity("exit_routes", "",
                        f"{len(no_route)} of {len(held)} held have no target, stale or swap route",
                        "neutral", {"held": held, "no_auto_exit_route": no_route})
    except Exception as e:
        log.warning(f"exit_routes log failed: {e}")


@app.task(name="tasks.review_positions", bind=True, max_retries=2)
def review_positions(self):
    """Scheduled reassessment of already-held positions for TRIM/EXIT/HOLD
    — see agents/crew.py's review_all_held_positions docstring for why this
    exists (nothing else ever asked this question on a schedule).

    2026-08-29 (user request): first asks risk_manager.flag_stale_positions()
    which Accelerator/Moonshot holdings have gone nowhere for longer than
    their regime-appropriate window, and passes those as priority_tickers
    so a stale position gets a fresh LLM look THIS run instead of possibly
    waiting weeks behind other holdings in the normal rotation. Flagging
    failure doesn't block the review itself running -- falls through to
    the unprioritized rotation, same as if nothing were stale."""
    try:
        from agents.crew import review_all_held_positions
        stale_tickers: list[str] = []
        try:
            from risk_manager import RiskManager
            rm = RiskManager(_get_client())
            stale_tickers = [f["ticker"] for f in rm.flag_stale_positions()]
        except Exception as e:
            log.warning(f"review_positions: stale-position flagging failed, "
                        f"proceeding without priority: {e}")
        _log_exit_routes()
        review_all_held_positions(max_per_run=10, priority_tickers=stale_tickers)
    except Exception as e:
        log.error(f"review_positions failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.check_profit_targets", bind=True, max_retries=2)
def check_profit_targets(self):
    """Scheduled regime-adaptive profit-taking for Accelerator/Moonshot
    positions (2026-08-29, user request) -- see risk_manager.
    check_profit_targets's docstring and config.py's PROFIT_TARGET_PCT_BY_STYLE
    for the full rationale. Unlike check_trailing_stops above, the actual
    SELL order is placed inside risk_manager.check_profit_targets() itself
    (there's no "detect vs execute" split to preserve here — a profit
    target crossing isn't racing a live broker stop the way a stop-loss
    trigger is), so this wrapper's job is just invocation, alerting, and
    Celery retry/error handling."""
    failed = will_retry = False
    try:
        import alerts
        from risk_manager import RiskManager

        client = _get_client()
        rm = RiskManager(client)
        sold = rm.check_profit_targets()
        failed = rm.monitor_read_failed is True
        for result in sold:
            if result.get("action") == "BREAKEVEN_STOP":
                # No shares sold (the partial rounds to 0); only the stop moved.
                alerts._send(
                    f"🛡️ BREAKEVEN STOP — {result['ticker']}\n"
                    f"Profit target hit ({result['gain_pct']:.1%}, style={result['trade_style']}) "
                    f"but the partial sale rounds to 0 shares. Stop moved to breakeven "
                    f"${result['new_stop']:.2f}; nothing sold."
                )
                continue
            # Price is the fill, not the new stop (audit 2026-10-28 10:53 #3).
            fill = (result.get("order_result") or {}).get("fill_price")
            alerts.send_trade_alert(
                symbol=result["ticker"], side="SELL",
                qty=int(result["shares_sold"]), price=fill or result["new_stop"],
                reason=(f"Profit target hit ({result['gain_pct']:.1%}, "
                        f"style={result['trade_style']}) — partial sell, "
                        f"remainder stop moved to breakeven ${result['new_stop']:.2f}"
                        + ("" if fill else " (fill price unavailable; price shown is the stop)")),
            )
    except Exception as e:
        failed = True
        # Only a run with no retry left counts as a failed check (M146).
        will_retry = self.request.retries < self.max_retries
        log.error(f"check_profit_targets failed: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        if not (failed and will_retry):
            _monitor_health("profit_targets", failed)


@app.task(name="tasks.backfill_catalyst_tracking", bind=True, max_retries=1)
def backfill_catalyst_tracking(self, max_per_run: int = 10):
    """On-demand, NOT scheduled (2026-08-30, user request): one-time
    gap-closer for positions held before catalyst/expected-return
    tracking existed -- see agents.crew.backfill_catalyst_tracking's
    docstring. Deliberately not added to beat_schedule; going-forward
    captures already happen automatically at real trade execution
    (execution.py), so there's no ongoing need to run this on a
    schedule once the current backlog is cleared. Naturally resumable:
    call again (or let a retry fire) to continue past whatever
    max_per_run covered this run."""
    try:
        from agents.crew import backfill_catalyst_tracking as _backfill
        results = _backfill(max_per_run=max_per_run)
        backfilled = sum(1 for r in results if r["action"] == "BACKFILLED")
        log.info(f"Catalyst backfill: {backfilled}/{len(results)} tickers backfilled this run")
        return results
    except Exception as e:
        log.error(f"backfill_catalyst_tracking failed: {e}")
        raise self.retry(exc=e, countdown=120)


def _schedule_backstop_retry(ticker: str, alert: dict) -> None:
    """Queue a retry_backstop_sell for a backstop attempt that placed an
    order but got no fill (most commonly: submitted after the close, so
    IBKR queued it for next open per warning 399 instead of filling within
    the 30s wait). Retries at next_market_open() if the market's closed,
    else in 5 minutes (covers a transient broker/connectivity hiccup during
    market hours)."""
    try:
        from datetime import datetime, timedelta, timezone
        from market_data import is_market_hours, next_market_open
        eta = next_market_open() if not is_market_hours() else (
            datetime.now(timezone.utc) + timedelta(minutes=5))
        retry_backstop_sell.apply_async(args=[ticker, alert], eta=eta, queue="ops")
        log.warning(f"[{ticker}] Backstop sell retry scheduled for {eta.isoformat()}")
    except Exception as e:
        log.error(f"[{ticker}] Could not schedule backstop-sell retry: {e}")


def _split_ratio(ticker: str, pos) -> float | None:
    """The split ratio (e.g. 4.0, or 0.1 for 1:10) if the recorded entry and
    the broker's average cost differ by a near-whole factor of 2 or more. The
    broker adjusts its average cost on a split; the stop record doesn't. An
    ADD's blending never moves them that far apart. None if no split is
    suspected or it can't be read."""
    import db
    try:
        entry = float((db.get_position_stop(ticker) or {}).get("entry_price") or 0)
        avg = float(getattr(pos, "avg_entry_price", 0) or 0)
    except Exception:
        return None
    if entry <= 0 or avg <= 0:
        return None
    r = entry / avg
    for c in (r, 1 / r):
        if c >= 1.9 and abs(c - round(c)) / c < 0.04:
            return round(r, 4)
    return None


_BACKSTOP_RETRY_GUARD = "helios:backstop_retry_sent:{ticker}:{day}"


def _claim_backstop_retry_send(ticker: str) -> bool:
    """Once per ticker per trading session, a backstop RETRY may go on to
    cancel stops and sell: several retry_backstop_sell tasks can be queued
    for the same open (one per after-hours check), and each used to sell
    (review of 5f13c44). Redis SET NX, expiring at the 16:00 ET close.
    False if already claimed or Redis can't be reached (unsure: skip)."""
    from datetime import datetime, time as dtime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    close = datetime.combine(now.date(), dtime(16, 0), tzinfo=et)
    ttl = max(60, int((close - now).total_seconds()))
    key = _BACKSTOP_RETRY_GUARD.format(ticker=ticker, day=now.date().isoformat())
    try:
        return bool(stop_lock._redis().set(key, now.isoformat(), ex=ttl, nx=True))
    except Exception as e:
        log.error(f"[{ticker}] backstop retry guard unavailable ({e}); not selling")
        return False


def _backstop_retry_skipped(ticker: str, why: str) -> None:
    log.error(f"[{ticker}] Backstop retry skipped: {why}")
    try:
        import alerts
        alerts._send(f"⚠️ BACKSTOP RETRY SKIPPED — <b>{ticker}</b>\n{why}. Nothing was cancelled or "
                     f"sold. Check the position and its stop in TWS.")
    except Exception:
        pass


@locked_by_ticker(lambda ticker, *a, **k: ticker, on_busy=False)
def _execute_backstop_sell(ticker: str, alert: dict, _is_retry: bool = False) -> bool | None:
    """Market-sell the full position when its stop level is breached and no
    protective stop exists at the broker. This monitor is the last line of
    defence for exactly that case; the sell only ever reduces exposure.

    Incident 2026-07-29 (AMD): this used to just log an error and give up
    when the market-sell got no fill (both attempts that evening were
    submitted after the close and IBKR queued them for next open rather
    than filling within the 30s wait). Nothing ever revisited it, so 50
    shares -- the exact position this function exists to protect -- sat
    fully naked for ~30 hours until a manual review caught it the next
    day. A zero-fill result now schedules retry_backstop_sell instead of
    dropping the problem.
    """
    stops_cancelled = False
    try:
        import ibkr
        import db

        # get_open_position raises both for "not held" and for a failed
        # read; only a successful read showing no position may drop the
        # stop record (professor M61, the M22 pattern).
        try:
            positions = ibkr.get_client().get_all_positions()
        except Exception as e:
            log.error(f"[{ticker}] Backstop sell: position read failed ({e}); retrying later")
            if not _is_retry:
                _schedule_backstop_retry(ticker, alert)
            else:
                _backstop_retry_skipped(ticker, f"the position could not be read ({e})")
            return False
        pos = next((p for p in positions if p.symbol == ticker), None)
        qty = float(pos.qty) if pos else 0.0
        if qty < 0:
            log.error(f"[{ticker}] Backstop sell: broker shows a SHORT position ({qty:g}); "
                      f"not selling")
            try:
                import alerts
                alerts._send(f"🚨 SHORT POSITION — <b>{ticker}</b>\n{qty:g} shares at the broker. "
                             f"The backstop did not sell. Check TWS now.")
            except Exception:
                pass
            return False
        owned = round(qty, 4)
        if owned <= 0:
            # Stale stop record for a position that no longer exists
            log.info(f"[{ticker}] Backstop: no shares held; stale stop record dropped, nothing to sell")
            db.delete_position_stop(ticker)
            return None  # nothing to do, not a failed sale

        ratio = _split_ratio(ticker, pos)
        if ratio:
            # After a split the price sits far below the recorded stop and
            # the broker stop no longer matches the shares, so this path
            # would sell the whole position at market (audit 14:53, A1453-1).
            log.error(f"[{ticker}] Backstop sell blocked: recorded entry vs broker average cost "
                      f"suggests a {ratio:g}:1 split; stop levels need rescaling")
            try:
                already = ticker in json.loads(db.get_state(_MANUAL_ACTIONS_KEY) or "{}")
            except Exception:
                already = False
            if already:
                return False  # tracked; the hourly STILL OPEN re-alert covers it
            open_manual_action(ticker, f"check a possible {ratio:g}:1 split and rescale the stop "
                                       f"record", clear_on_flat=False)
            try:
                import alerts
                alerts._send(f"⚠️ POSSIBLE SPLIT — <b>{ticker}</b>\nThe recorded entry is about "
                             f"{ratio:g}× the broker's average cost, so the stop check thinks the "
                             f"stop was hit. The backstop sell was blocked. Check the position and "
                             f"stops in TWS; the stop record needs rescaling.")
            except Exception:
                pass
            return False

        # Everything that can abort the sale is checked before any stop is
        # cancelled (audit 2026-11-02 #11). Outside regular hours the market
        # SELL is cancelled after its fill wait (ibkr.place_market_order), so
        # cancelling first left the position with no stop overnight.
        from market_data import is_market_hours
        if not is_market_hours():
            log.warning(f"[{ticker}] Backstop: outside regular hours; stops left in place, "
                        f"sale retried at the open")
            _schedule_backstop_retry(ticker, alert)
            return False
        # A SELL of ours still working that covers every share: nothing to
        # sell, so nothing to cancel.
        working = sum(float(o["qty"]) for o in _resting_sell_stops(ticker, types=("MKT",)))
        if owned - working <= 1e-6:
            log.warning(f"[{ticker}] Backstop: a working SELL for {working:g} shares covers all "
                        f"{owned:g}; nothing cancelled or sold")
            return False
        if _is_retry:
            # A retry runs later than the breach that queued it: re-check
            # that the shares are still unprotected, and sell at most once
            # per session (review of 5f13c44).
            if working > 0:
                _backstop_retry_skipped(ticker, f"a SELL for {working:g} shares is still working")
                return False
            try:
                covered = sum(float(o["qty"]) for o in _resting_sell_stops(ticker))
            except Exception as e:
                _backstop_retry_skipped(ticker, f"the resting stops could not be read ({e})")
                return False
            if covered > 0 and covered >= owned - max(0.01, owned * 0.02):
                log.info(f"[{ticker}] Backstop retry: a stop now covers {covered:g} of {owned:g} "
                         f"shares; not selling")
                return False
            if not _claim_backstop_retry_send(ticker):
                _backstop_retry_skipped(ticker, "a backstop retry already ran this session "
                                                "(or its guard could not be checked)")
                return False

        stops_cancelled = True  # set before the call: a raise inside it may follow a cancel
        manage_cancel_stop_orders_for(ticker)  # belt-and-braces before selling
        # A stop that wouldn't cancel still protects its shares, and selling
        # those too would short once both fill (same rule as
        # execute_approved, A2253-1): sell only the uncovered part.
        resting = sum(float(o["qty"]) for o in _resting_sell_stops(ticker))
        if resting > 0:
            log.error(f"[{ticker}] Backstop: stops for {resting:g} of {owned:g} shares still "
                      f"resting after the cancel; selling only the rest")
            try:
                import alerts
                alerts._send(f"⚠️ BACKSTOP — <b>{ticker}</b>\nA stop for {resting:g} of "
                             f"{owned:g} shares would not cancel; only the uncovered shares "
                             f"are sold. Check TWS.")
            except Exception:
                pass
            owned = round(owned - resting, 4)
            if owned <= 0:
                return False
        # A SELL of ours still working (an earlier backstop's or an EXIT's
        # remainder) will sell its shares; selling them again would short
        # once both fill (review of 16009ad). Same remainder rule as stops.
        # Market orders only: the bot places no limit sells, and a manual limit
        # above the market (TWS) would not sell into a stop breach. (Read
        # before the cancel above.)
        if working > 0:
            log.warning(f"[{ticker}] Backstop: a SELL for {working:g} shares is still working; "
                        f"selling only what it doesn't cover")
            try:
                import alerts
                alerts._send(f"⚠️ BACKSTOP — <b>{ticker}</b>\nA market SELL for {working:g} of "
                             f"{owned:g} shares is still working, so only the rest is sold. "
                             f"Check TWS if it doesn't fill.")
            except Exception:
                pass
            owned = round(owned - working, 4)
            if owned <= 0:
                return False
        stop_lock.assert_held(ticker)  # professor M69
        result = ibkr.place_market_order(ticker, "SELL", owned)
        filled = float(result.get("filled_qty") or 0)
        if filled <= 0:
            log.error(f"[{ticker}] Backstop sell submitted but no fill confirmed "
                      f"(id={result.get('order_id')}) — nothing recorded")
            _schedule_backstop_retry(ticker, alert)
            return False

        fill_price = result.get("fill_price") or alert["current_price"]
        # An orders row like every other sale: a part fill stays 'partial' so
        # reconcile_unconfirmed_orders closes the outcome from the cumulative
        # fill once flat, and the stale-stop lookup leaves these fills out
        # (professor, review of c13548c). Written before the trade row so the
        # trade carries its orders.id (professor m301).
        local_id = None
        try:
            local_id = db.log_order(ticker=ticker, side="SELL", order_type="MARKET", qty=owned,
                                    broker_order_id=result.get("order_id"),
                                    perm_id=result.get("perm_id"),
                                    decision_price=alert.get("current_price"), status="submitted",
                                    intent="BACKSTOP")
            if local_id and local_id > 0:
                db.update_order_fill(local_id, result.get("fill_price"), filled,
                                     "filled" if filled >= owned - 1e-6 else "partial",
                                     filled_at=result.get("fill_time"))
        except Exception as e:
            log.warning(f"[{ticker}] Backstop sale order logging failed: {e}")
        db.log_trade(symbol=ticker, side="SELL", quantity=filled, price=fill_price,
                     price_source="fill" if result.get("fill_price") else "quote",
                     reason=f"Backstop stop-breach sell (stop ${alert['stop_price']:.2f})"
                            + (" [retry]" if _is_retry else ""),
                     order_row_id=local_id if isinstance(local_id, int) and local_id > 0 else None)
        # Shares under a stuck stop, or under our working SELL, are still
        # held: keep the record (so a cancelled order's shares can be
        # backstopped next cycle) and leave the outcome for whichever sale
        # finishes the position (professor M128).
        if resting <= 0 and working <= 0:
            if filled < owned - 1e-6:
                # Part filled: this part's price isn't the exit (professor
                # M121). The rest is still working at the broker; the order
                # is 'partial', so reconcile closes the row from the
                # cumulative fill once flat. The record stays: if the broker
                # cancels the rest, the held shares are backstopped or
                # re-protected next cycle, and while it works the working-sell
                # subtraction stops a second sale (professor, M128 class).
                log.warning(f"[{ticker}] Backstop sell part-filled ({filled:g} of {owned:g}); "
                            f"outcome and stop record kept")
            else:
                db.delete_position_stop(ticker)
                try:
                    db.close_trade_outcome(ticker, exit_price=fill_price, exit_source="backstop_sell",
                                           exit_price_source="fill" if result.get("fill_price") else "quote",
                                           exit_fill_at=result.get("fill_time"))
                except Exception as e:
                    log.error(f"[{ticker}] close_trade_outcome failed after the backstop sell: {e}")  # m98
        log.warning(f"[{ticker}] BACKSTOP SELL executed: {filled} shares @ ~${fill_price:.2f}")
        return True
    except Exception as e:
        log.error(f"[{ticker}] Backstop sell FAILED: {e}")
        if not _is_retry:
            _schedule_backstop_retry(ticker, alert)
        elif stops_cancelled:
            # No further retry follows, and the stops may already be gone.
            log.critical(f"[{ticker}] Backstop retry FAILED after its stops were cancelled ({e}); "
                         f"the position may have no stop")
            try:
                import alerts
                alerts._send(f"🚨 BACKSTOP RETRY FAILED — <b>{ticker}</b>\nThe retry failed ({e}) "
                             f"after its stops were cancelled, so the position may have no stop "
                             f"and was not sold. Check the position and its stop in TWS now.")
            except Exception as e2:
                log.error(f"[{ticker}] could not send the backstop retry alert: {e2}")
        else:
            _backstop_retry_skipped(ticker, f"the retry failed before any stop was cancelled ({e})")
        return False


@app.task(name="tasks.retry_backstop_sell", bind=True, max_retries=0)
def retry_backstop_sell(self, ticker: str, alert: dict):
    """Retry for a backstop stop-breach sell that _execute_backstop_sell
    couldn't complete on the first attempt -- see that function's docstring
    for the AMD incident this closes the loop on. It does not re-check the
    price against the stop, but _execute_backstop_sell re-checks that the
    shares are still held, uncovered by a stop and not already being sold,
    and lets only one retry per ticker per session sell (review of 5f13c44)."""
    success = _execute_backstop_sell(ticker, alert, _is_retry=True)
    if success:
        log.info(f"[{ticker}] Backstop sell succeeded on retry")
    elif success is None:
        log.info(f"[{ticker}] Backstop sell retry: nothing held, nothing to sell")
    else:
        log.error(f"[{ticker}] Backstop sell retry did not sell; the reason is logged above")


@app.task(name="tasks.check_trailing_stops", bind=True, max_retries=2)
def check_trailing_stops(self):
    """Monitor open positions against their stop levels. Alert on triggers;
    if the stop level is breached and NO protective stop exists at the
    broker, execute a market sell (AUTO_EXECUTE_PROTECTIVE_SELLS)."""
    failed = will_retry = False
    try:
        from risk_manager import RiskManager
        import alerts

        client = _get_client()
        rm = RiskManager(client)
        triggered = rm.check_trailing_stops()
        failed = rm.monitor_read_failed is True
        if not triggered:
            return

        try:
            from config import AUTO_EXECUTE_PROTECTIVE_SELLS
        except ImportError:
            AUTO_EXECUTE_PROTECTIVE_SELLS = True

        # A live broker stop will fire on its own — never race it with a
        # market sell. Only backstop tickers with no SELL stop at the broker.
        #
        # Found 2026-07-28 (AMKR): this ops-worker connection's local order
        # cache can go stale in exactly one dangerous direction -- IBKR
        # generally only pushes fill/status updates for an order to the
        # clientId that placed it, not to other clients that merely queried
        # it via reqAllOpenOrders(). A stop placed by the execution worker
        # (a different clientId) that later fills is invisible to this
        # connection's cache, which keeps reporting it "PreSubmitted" with
        # filled=0 indefinitely -- so a plain ticker-membership check kept
        # concluding "AMKR still has a live stop" for hours after the real
        # stop had already filled and reduced the position, and the backstop
        # never fired while the remainder sat completely unprotected.
        #
        # Fix: track the covered QUANTITY per ticker, not just presence, and
        # cross-check it against the actual current position size (which
        # updates correctly and promptly via the account's live position
        # feed). A stop genuinely still open for the current position covers
        # it almost exactly; a stop claiming to cover materially more or
        # fewer shares than are actually held cannot be a live, valid stop
        # for what's held right now, and gets treated as unreliable --
        # firing the backstop rather than trusting a number that doesn't
        # match reality.
        import ibkr
        # Cover is working stops plus working market SELLs: a stop sized net
        # of a sale still in progress (M129) is valid, not stale (professor M132).
        try:
            open_orders = ibkr.get_open_orders()
            open_stop_qty: dict[str, float] = {}
            for o in _working_sell_stops(open_orders):
                open_stop_qty[o["ticker"]] = open_stop_qty.get(o["ticker"], 0.0) + o["qty"]
            for t in {o["ticker"] for o in open_orders}:
                selling = _working_mkt_sell_qty(open_orders, t)
                if selling:
                    open_stop_qty[t] = open_stop_qty.get(t, 0.0) + selling
        except Exception as e:
            log.warning(f"Could not fetch open orders for backstop check: {e}")
            open_stop_qty = None  # unknown — don't sell blind

        for alert in triggered:
            ticker = alert["ticker"]
            log.warning(f"STOP TRIGGERED: {ticker} — {alert['reason']}")
            backstopped = False
            if (alert.get("action") == "EXIT"
                    and AUTO_EXECUTE_PROTECTIVE_SELLS
                    and open_stop_qty is not None):
                try:
                    held_qty = abs(float(client.get_open_position(ticker).qty))
                except Exception:
                    held_qty = 0.0
                stop_qty = open_stop_qty.get(ticker, 0.0)
                covered = held_qty > 0 and abs(stop_qty - held_qty) < max(0.01, held_qty * 0.02)
                if held_qty > 0 and not covered:
                    if stop_qty:
                        log.warning(f"[{ticker}] Working stops and sells cover {stop_qty} shares but "
                                    f"{held_qty} are actually held — treating as stale/unreliable")
                    backstopped = _execute_backstop_sell(ticker, alert)
            alerts.send_stop_triggered(
                ticker=ticker,
                current_price=alert["current_price"],
                stop_price=alert["stop_price"],
                reason=alert["reason"] + (" — BACKSTOP SELL EXECUTED" if backstopped else ""),
            )
    except Exception as e:
        failed = True
        # Only a run with no retry left counts as a failed check (M146).
        will_retry = self.request.retries < self.max_retries
        log.error(f"check_trailing_stops failed: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        if not (failed and will_retry):
            _monitor_health("trailing_stops", failed)


# ── Existing tasks (updated call signatures) ──────────────────────────────────

MONITOR_FAILS_BEFORE_ALERT = 3


def _monitor_health(name: str, failed: bool) -> None:
    """Consecutive-failure count for a mechanical monitor (audit A1031-1053-8):
    one alert when it reaches MONITOR_FAILS_BEFORE_ALERT, one when it
    recovers. On 10-19 three monitors failed 99 times each with no alert.
    A spell that alerted gets activity_feed rows (stage "monitor_blind") at
    its start and end, so blind time is measurable later (SEF ask)."""
    try:
        from datetime import datetime, timezone
        import alerts
        import db
        from config import TRADING_ENV
        key = f"monitor_fail_count_{name}_{TRADING_ENV}"
        since_key = f"monitor_fail_since_{name}_{TRADING_ENV}"
        # Strict: a failed read must not restart the count (SEF, after M146).
        count = int(db.get_state_strict(key, "0") or 0)
        now = datetime.now(timezone.utc).isoformat()
        spell = {"check": name, "env": TRADING_ENV}
        if failed:
            count += 1
            db.set_state(key, str(count))
            if count == 1:
                db.set_state(since_key, now)
            if count == MONITOR_FAILS_BEFORE_ALERT:
                alerts._send(
                    f"⚠️ MONITOR BLIND — {name} failed {count} checks in a row.\n"
                    f"Resting broker stops still protect positions, but this check isn't "
                    f"running. See the celery logs; a recovery message follows when it works.")
                db.log_activity("monitor_blind", "", f"{name} blind ({count} failed checks)",
                                "alert", {**spell, "event": "start",
                                             "first_fail_at": db.get_state(since_key) or None,
                                             "alerted_at": now})
        elif count:
            db.set_state(key, "0")
            if count >= MONITOR_FAILS_BEFORE_ALERT:
                alerts._send(f"✅ {name} monitor recovered after {count} failed checks.")
                db.log_activity("monitor_blind", "", f"{name} recovered after {count} failed checks",
                                "neutral", {**spell, "event": "end",
                                             "first_fail_at": db.get_state(since_key) or None,
                                             "recovered_at": now, "failed_attempts": count})
    except Exception as e:
        log.warning(f"_monitor_health({name}) failed: {e}")


def _heartbeat_ok(source: str, count: int) -> None:
    try:
        import db
        db.record_ingestion_success(source, count)
    except Exception:
        pass


def _heartbeat_fail(source: str, error: str) -> None:
    # Redacted before it is stored: the error text can carry a request URL
    # with a credential in it, and ingestion_health is shown on the
    # dashboard (professor m139). Log redaction only covers log records.
    try:
        import db
        import log_redaction
        db.record_ingestion_error(source, log_redaction.redact(str(error)))
    except Exception:
        pass


@app.task(name="tasks.poll_edgar", bind=True, max_retries=3)
def poll_edgar(self):
    """Polls SEC EDGAR for new filings and routes through agent pipeline.
    One run at a time: a pass takes ~11 min on a 10 min schedule, and
    overlapping runs doubled the request rate against SEC's limit (A1053-4)."""
    try:
        from ingestion.edgar import poll_once, LAST_PASS

        def run():
            filings = poll_once(on_filing=lambda f: _dispatch_event(f, seen_source="edgar"))
            log.info(f"EDGAR: {len(filings)} new filing(s)")
            # A failed pass (ingestion.edgar._note_pass_health) is a failure,
            # not a quiet success, so ingestion_health and the staleness check
            # see it. A pass skipped during a 429 backoff records neither.
            if LAST_PASS.get("backoff_skip"):
                pass
            elif LAST_PASS.get("poll_failed"):
                _heartbeat_fail("edgar", f"{LAST_PASS.get('ok', 0)} of {LAST_PASS.get('tickers', 0)} "
                                         f"tickers ok ({LAST_PASS.get('aborted') or 'full pass'})")
            else:
                _heartbeat_ok("edgar", len(filings))
        _run_singleton("poll_edgar", run, max_hold_s=1200)
    except Exception as e:
        log.error(f"poll_edgar failed: {e}")
        _heartbeat_fail("edgar", str(e))
        raise self.retry(exc=e, countdown=60)


_IMPLIED_MOVE_KEY = "options_implied_move_{env}"
_IMPLIED_MOVE_KEEP_DAYS = 7
_IMPLIED_MOVE_LOG_AFTER = (9, 45)      # ET: the daily row is the first computation from here


def _log_implied_move_shadow(moves: dict, now=None) -> int:
    """Log-only: the nearest-expiry ATM straddle implied move per ticker,
    computed by ingestion.options_free from the chain poll_options_flow
    already downloads (no extra network call). Overwrites one system_state
    map (latest per ticker) and writes one activity_feed row per ticker
    per ET date (stage options_shadow), the first computation at or after
    09:45 ET. Held tickers also get stop_distance_pct and
    implied_move_over_stop from position_stops (read only). Nothing reads
    these rows back into a decision. Never raises; returns rows written."""
    try:
        if not moves:
            return 0
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        import db
        from config import TRADING_ENV
        from agents.analyst import shadow_versions
        et_now = now or datetime.now(ZoneInfo("America/New_York"))
        et_date = et_now.date().isoformat()
        key = _IMPLIED_MOVE_KEY.format(env=TRADING_ENV)
        try:
            book = json.loads(db.get_state_strict(key, "") or "{}")
        except ValueError:
            book = {}
        if not isinstance(book, dict):
            book = {}
        keep_from = (et_now.date() - timedelta(days=_IMPLIED_MOVE_KEEP_DAYS)).isoformat()
        book = {t: v for t, v in book.items()
                if isinstance(v, dict) and str(v.get("et_date") or "") >= keep_from}
        logged = 0
        due = [t for t in sorted(moves) if (book.get(t) or {}).get("logged_et_date") != et_date]
        if due and (et_now.hour, et_now.minute) >= _IMPLIED_MOVE_LOG_AFTER:
            # Durable dedup (survives a lost state write); None (read failed):
            # skip logging this cycle rather than risk a duplicate.
            done = db.get_activity_shadow_keys("options_shadow")
            if done is not None:
                held = _held_tickers()
                stops = ({s["ticker"].upper(): s for s in db.get_all_position_stops()}
                         if held & set(due) else {})
                versions = shadow_versions()
                for t in due:
                    try:
                        sk = f"{TRADING_ENV}:{t}:{et_date}"
                        if sk not in done:
                            m = moves[t]
                            extra = {"ticker": t, **m, "et_date": et_date,
                                     "computed_at": et_now.isoformat(),
                                     "held": t.upper() in held, "shadow_key": sk, **versions}
                            if extra["held"]:
                                stop = (stops.get(t.upper()) or {}).get("stop_price")
                                u = m.get("underlying")
                                dist = (round((u - stop) / u * 100, 4)
                                        if stop and stop > 0 and u and u > 0 else None)
                                extra["stop_distance_pct"] = dist
                                extra["implied_move_over_stop"] = (
                                    round(m["implied_move_pct"] / dist, 4)
                                    if dist is not None and dist > 0
                                    and m.get("implied_move_pct") is not None else None)
                            if not db.log_activity(
                                    "options_shadow", t,
                                    f"implied move {t}: {m['implied_move_pct']:.2f}% to {m.get('expiry')} "
                                    f"({m.get('dte')} DTE, {m.get('price_source')})", "neutral", extra):
                                continue
                            logged += 1
                        book.setdefault(t, {})["logged_et_date"] = et_date
                    except Exception as ex:
                        log.debug(f"[{t}] implied move shadow row failed: {ex}")
        for t, m in moves.items():
            prev = book.get(t) or {}
            book[t] = {**m, "et_date": et_date, "computed_at": et_now.isoformat(),
                       "logged_et_date": prev.get("logged_et_date")}
        db.set_state(key, json.dumps(book, default=str))
        return logged
    except Exception as e:
        log.debug(f"implied move shadow failed: {e}")
        return 0


@app.task(name="tasks.poll_options_flow", bind=True, max_retries=3)
def poll_options_flow(self):
    """Checks for unusual options activity across all symbols (Unusual Whales,
    falls back to the free yfinance-based options_free module when no key is set)."""
    try:
        from market_data import _is_market_hours
        if not _is_market_hours():
            # Option chains are stale outside RTH (last-quoted OI/volume,
            # not live) — this poller ran every 5 min around the clock
            # during the awake-but-closed windows (05:00-09:30, 16:00-19:00)
            # for no signal value.
            log.debug("poll_options_flow: market closed, skipping")
            return

        import os
        implied = None
        if os.getenv("UNUSUAL_WHALES_API_KEY", "").startswith("your_"):
            from ingestion.options_free import get_all_flow_summaries
            log.info("UNUSUAL_WHALES_API_KEY not configured — using free yfinance options proxy")
            implied = {}       # log-only implied-move shadow, from the same chains
        else:
            from ingestion.options_flow import get_all_flow_summaries

        import db
        from datetime import date

        summaries = (get_all_flow_summaries(implied_sink=implied) if implied is not None
                     else get_all_flow_summaries())
        if implied:
            _log_implied_move_shadow(implied)
        unusual_count = 0
        for ticker, flow in summaries.items():
            if flow.get("unusual"):
                # "Unusual" (OI concentration) is persistent state, not a
                # one-off event — without this dedup the same ticker was
                # dispatched to the full 72B pipeline every 5 minutes,
                # indefinitely. One analysis per ticker per day is the right
                # cadence; this was the single biggest LLM-cost bug found.
                seen_key = f"{ticker}_{date.today().isoformat()}"
                if db.is_seen("options_flow_dispatch", seen_key):
                    continue
                unusual_count += 1
                event = {
                    "source": "options",
                    "ticker": ticker,
                    "title":  f"Unusual options activity: {ticker}",
                    "text":   flow["summary"],
                }
                _dispatch_event(event)
                db.mark_seen("options_flow_dispatch", seen_key)
        _heartbeat_ok("options_flow", unusual_count)
    except Exception as e:
        log.error(f"poll_options_flow failed: {e}")
        _heartbeat_fail("options_flow", str(e))
        raise self.retry(exc=e, countdown=60)


def _log_would_reassess(live_vix: float) -> None:
    """Log-only (audit A1031-1453-2): the intraday VIX-shock reassessment is
    dead code (should_reassess reads a process cache the ingestion worker
    never fills). Record when it would have fired: live ^VIX more than 5
    points (should_reassess's threshold) from the newest overall regime row.
    One activity_feed row per stored regime row. Market data only."""
    try:
        import math
        import db
        from config import TRADING_ENV
        if not math.isfinite(live_vix):
            return
        row = db.get_latest_regime()
        if not row or not str(row.get("regime_source", "overall")).startswith("overall"):
            return
        stored = ((row.get("signals_json") or {}).get("volatility") or {}).get("vix")
        if stored is None or abs(live_vix - float(stored)) <= 5:
            return
        row_at = str(row.get("timestamp"))
        key = f"would_reassess_logged_{TRADING_ENV}"
        if db.get_state(key, "") == row_at:
            return
        written = db.log_activity("regime_shadow", "", f"would reassess: live VIX {live_vix:.1f} vs "
                                  f"{float(stored):.1f} at the last regime row", "alert",
                                  {"would_reassess": True, "live_vix": round(live_vix, 2),
                                   "stored_vix": round(float(stored), 2),
                                   "delta": round(live_vix - float(stored), 2),
                                   "regime": row.get("regime"), "regime_row_at": row_at})
        # The key only after the row is written: set first, a failed write
        # (log_activity swallows errors) marked the row logged and lost it.
        if written:
            db.set_state(key, row_at)
    except Exception as e:
        log.debug(f"would_reassess log failed: {e}")


@app.task(name="tasks.poll_macro", bind=True, max_retries=3)
def poll_macro(self):
    """Fetches macro snapshot, stores in vector memory, and triggers intraday regime reassessment if VIX spikes."""
    try:
        from ingestion.macro import get_macro_summary_text, get_macro_snapshot
        import vector_store
        from datetime import datetime, timezone

        snap    = get_macro_snapshot()
        summary = get_macro_summary_text()
        vector_store.store(
            collection="macro",
            doc_id=f"macro_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}",
            text=summary,
            metadata={"type": "macro_snapshot"},
        )
        # Cache for context_engine's brief builder — it was calling
        # get_macro_summary_text() itself on every single brief (10 FRED
        # HTTP requests per brief; a 5-candidate scan run made 50 FRED
        # calls). This runs hourly already; the brief just needs to read it.
        import db
        db.set_state("macro_summary_text", summary)
        db.set_state("macro_summary_at", datetime.now(timezone.utc).isoformat())
        log.info("Macro snapshot stored.")

        # Reassess regime if macro conditions changed significantly.
        # Uses live ^VIX (yfinance), not FRED's VIXCLS — FRED is end-of-day
        # and lags a full session, which defeats the point of an "intraday"
        # reassessment trigger (it would just compare yesterday to itself).
        try:
            from agents.market_regime import should_reassess, assess_regime, get_intraday_vix
            vix = get_intraday_vix()
            if vix:
                _log_would_reassess(vix)
            if vix and should_reassess(vix):
                log.info(f"Intraday VIX shift detected ({vix:.1f}) — reassessing market regime")
                assess_regime()
        except Exception as e:
            log.debug(f"Intraday regime check skipped: {e}")

        _heartbeat_ok("macro", 1)
    except Exception as e:
        log.error(f"poll_macro failed: {e}")
        _heartbeat_fail("macro", str(e))
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.poll_arxiv", bind=True, max_retries=3)
def poll_arxiv(self):
    """Fetches new quantum/AI papers from arXiv and stores in vector memory.
    Also checks each paper for non-watchlist S&P 500 company mentions --
    see arxiv_watcher.link_paper_to_tickers's docstring for why."""
    try:
        from ingestion.arxiv_watcher import poll_once, link_paper_to_tickers
        from ingestion.sp500_universe import get_sp500_constituents
        import vector_store
        import db

        papers = poll_once()
        log.info(f"arXiv: {len(papers)} new paper(s)")
        constituents = get_sp500_constituents()
        for paper in papers:
            vector_store.store_arxiv(paper)
            for ticker, company in link_paper_to_tickers(paper):
                gics_sector = constituents.get(ticker, {}).get("gics_sector")
                db.log_watchlist_candidate(
                    ticker=ticker, gics_sector=gics_sector,
                    movement_score=0, thesis_match_score=15, composite_score=15,
                    matched_thesis=None,
                    notes=f"arXiv mention: \"{paper['title'][:100]}\" ({paper['url']})",
                )
                log.info(f"arXiv candidate signal: {company} ({ticker}) mentioned in {paper['title'][:60]}")
        _heartbeat_ok("arxiv", len(papers))
    except Exception as e:
        log.error(f"poll_arxiv failed: {e}")
        _heartbeat_fail("arxiv", str(e))
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.poll_congressional", bind=True, max_retries=3)
def poll_congressional(self):
    """Polls congressional trading data and processes high-conviction trades."""
    try:
        from ingestion.congressional import poll_once

        trades = poll_once()
        log.info(f"Congressional: {len(trades)} new trade(s)")
        for trade in trades:
            seen_key = trade.pop("_seen_key", None)
            # Only send large purchases through the full pipeline
            if trade.get("side", "").upper() == "PURCHASE":
                if seen_key:
                    trade["_seen_key"] = seen_key
                _dispatch_event(trade, seen_source="congressional")
            else:
                import vector_store
                vector_store.store(
                    collection="congressional",
                    doc_id=f"cong_{trade.get('ticker')}_{trade.get('date')}_{trade.get('representative','')}",
                    text=trade["text"],
                    metadata={"ticker": trade["ticker"], "side": trade.get("side", "")},
                )
                if seen_key:
                    import db
                    db.mark_seen("congressional", seen_key)
        _heartbeat_ok("congressional", len(trades))
    except Exception as e:
        log.error(f"poll_congressional failed: {e}")
        _heartbeat_fail("congressional", str(e))
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.poll_insider", bind=True, max_retries=3)
def poll_insider(self):
    """
    Polls Form 4 insider filings. insider.poll_once() now only returns
    filings containing an actual open-market purchase (transaction code "P"),
    so every result here is a genuine buy signal and goes through the full
    analyst pipeline like EDGAR filings and congressional purchases.
    """
    try:
        from ingestion.insider import poll_once, LAST_PASS

        filings = poll_once()
        log.info(f"Insider: {len(filings)} new open-market buy filing(s)")
        for filing in filings:
            # insider.py's own is_seen/mark_seen use source="insider", but
            # each filing's payload carries source="sec_form4" (what the LLM
            # sees) — must pass the dedup source explicitly or _dispatch_event
            # would default to the wrong one and the is_seen check would
            # never match what gets marked.
            _dispatch_event(filing, seen_source="insider")
        # As poll_edgar: a pass where no ticker succeeded is a failure, and
        # one skipped during the SEC 429 backoff records neither.
        if LAST_PASS.get("backoff_skip"):
            pass
        elif LAST_PASS.get("poll_failed"):
            _heartbeat_fail("insider", f"{LAST_PASS.get('ok', 0)} of {LAST_PASS.get('tickers', 0)} "
                                       f"tickers ok ({LAST_PASS.get('aborted') or 'full pass'})")
        else:
            _heartbeat_ok("insider", len(filings))
    except Exception as e:
        log.error(f"poll_insider failed: {e}")
        _heartbeat_fail("insider", str(e))
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.poll_analyst_ratings", bind=True, max_retries=3)
def poll_analyst_ratings(self):
    """Polls free analyst upgrade/downgrade + price-target data (yfinance)
    for the watchlist -- see ingestion/analyst_ratings.py's docstring."""
    try:
        from ingestion.analyst_ratings import poll_once

        items = poll_once()
        log.info(f"Analyst ratings: {len(items)} new item(s)")
        for item in items:
            _dispatch_event(item)
        _heartbeat_ok("analyst_ratings", len(items))
    except Exception as e:
        log.error(f"poll_analyst_ratings failed: {e}")
        _heartbeat_fail("analyst_ratings", str(e))
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.poll_news", bind=True, max_retries=3)
def poll_news(self):
    """Polls free news sources (yfinance + Google News RSS) for the watchlist."""
    try:
        from ingestion.news import poll_once
        from ingestion.company_photos import get_fallback_image
        from knowledge_loader import load_watchlist
        import db

        items = poll_once()
        log.info(f"News: {len(items)} new item(s)")
        watchlist = load_watchlist()
        for item in items:
            # Logged for the public research site's News section regardless
            # of the screener's trade-relevance verdict below -- a reader
            # wants every headline Yahoo/Google published, not just the
            # subset judged worth a 72B trading analysis. Every headline
            # gets a photo: the article's own thumbnail if yfinance
            # provided one, otherwise a company-related fallback (Google
            # News RSS items never carry one at all).
            ticker = item.get("ticker", "")
            image_url = item.get("image_url")
            if not image_url:
                company = watchlist.get(ticker, {}).get("company", ticker)
                try:
                    image_url = get_fallback_image(ticker, company)
                except Exception as e:
                    log.debug(f"[{ticker}] fallback image lookup failed: {e}")
            try:
                db.log_news_item(ticker, item.get("title", ""), item.get("url", ""), image_url)
            except Exception as e:
                log.debug(f"log_news_item failed: {e}")
            _dispatch_event(item)
        _heartbeat_ok("news", len(items))
    except Exception as e:
        log.error(f"poll_news failed: {e}")
        _heartbeat_fail("news", str(e))
        raise self.retry(exc=e, countdown=120)


_PENDING_FILL_MARKER_TTL_S = 200 * 86400   # longer than any plan horizon (180 d)
# Calendar days before analyst's prune that an expired entry whose bars
# never reach its expiry is logged anyway (window_complete false). The prune
# can first drop an entry on expires + _PENDING_PRUNE_GRACE_DAYS + 1 (15),
# so the fallback is eligible from expires + 7. The checker runs only on
# weekdays, after a successful price refresh, so that leaves at least 4
# weekday runs before the prune.
_PENDING_EXPIRED_SLACK_DAYS = 7


def _check_pending_entries(today=None) -> int:
    """Log-only (audit 2026-11-02 19:27 #2, roadmap 01(c)): for each pending
    shadow entry the analyst recorded (agents.analyst._pending_entry_shadow,
    system_state pending_entries_shadow_<env>), check the daily bars
    refresh_price_history just stored, from the session after the verdict
    to the entry's expiry. The first bar that meets a level (low at or
    under an at_or_below level, high at or over an at_or_above one, the
    close at or over add_trigger) writes one activity_feed row, stage
    entry_shadow, would_fill. Each row also keeps the first touch date,
    the first close date and in_zone_at_verdict per level, so either fill
    test can be counted later. An entry past
    its expiry writes one expired row: would_fill true when any level
    filled, plus bars_in_window, last_bar_date and expired_no_data (no
    bars at all, so missing data rather than a miss) and window_complete.
    The expired row waits until the cache holds a bar on or after the
    expiry, so a late last bar can't be logged as a miss; an entry whose
    bars never get there is logged with window_complete false before its
    prune, as is an entry whose expires can't be parsed. Known
    disagreement: a fill row can still arrive after such an incomplete
    row, whose filled_kinds is then empty; window_complete false marks the
    expired row as not final on fills. Reads the DB cache only: no
    broker or network call. Nothing is queued or traded. Never raises."""
    try:
        import json
        from datetime import date, datetime, timedelta
        from zoneinfo import ZoneInfo
        import db
        import stop_lock
        from config import TRADING_ENV
        from agents.analyst import _PENDING_ENTRIES_KEY, _PENDING_PRUNE_GRACE_DAYS, shadow_versions
        book = json.loads(db.get_state(_PENDING_ENTRIES_KEY.format(env=TRADING_ENV), "") or "{}")
        if not isinstance(book, dict) or not book:
            return 0
        today = (today or datetime.now(ZoneInfo("America/New_York")).date()).isoformat()
        bars = db.get_price_history_bulk(sorted(book), days=260)
        r = stop_lock._redis()
        # Rows already in activity_feed, by shadow_key: survives a Redis
        # flush (professor m267). None (read failed): the Redis markers alone.
        done = db.get_activity_shadow_keys("entry_shadow")
        logged = 0
        for ticker, e in book.items():
            try:
                if e.get("expires") in (None, "", "None"):
                    continue           # no expiry: never recorded since the review fix; skip old ones
                start, expires = str(e["et_date"]), str(e["expires"])
                ident = f"{TRADING_ENV}:{ticker}:{e.get('signal_id') or e.get('at')}"
                window = [b for b in bars.get(ticker, []) if start < b["date"] <= min(expires, today)]
                base = {"verdict_signal_id": e.get("signal_id"), "verdict_et_date": start,
                        "verdict_price": e.get("price"), "trigger": e.get("trigger"),
                        "model_action": e.get("model_action"), "conviction": e.get("conviction"),
                        "expires": expires, **shadow_versions()}
                # Verdicts after this one that recorded no entry of their own
                # (professor m269); a fill row keeps those on or before its date.
                later = [v for v in (e.get("later_verdicts") or []) if isinstance(v, dict)]
                filled = []
                outcomes = {}
                for kind, lv in (e.get("levels") or {}).items():
                    level, below = float(lv["level"]), lv.get("side") == "at_or_below"
                    # The bar field a level fills on: the level's own, else
                    # the intraday low/high, but a close for add_trigger, as
                    # compile_plan defines it (professor M208).
                    field = lv.get("price_field") or (
                        "close" if kind == "add_trigger" else ("low" if below else "high"))
                    touch_f = "low" if below else "high"

                    def first(f):
                        return next((b for b in window
                                     if (b[f] <= level if below else b[f] >= level)), None)
                    touch, close_hit = first(touch_f), first("close")
                    in_zone = lv.get("in_zone_at_verdict")
                    if in_zone is None and e.get("price") is not None:
                        in_zone = (float(e["price"]) <= level if below
                                   else float(e["price"]) >= level)
                    # Both fill tests are kept, so either can be counted later.
                    outcomes[kind] = {"level": level, "side": lv.get("side"), "price_field": field,
                                      "in_zone_at_verdict": in_zone,
                                      "first_touch_date": touch["date"] if touch else None,
                                      "first_close_date": close_hit["date"] if close_hit else None}
                    hit = close_hit if field == "close" else first(field)
                    if hit is None:
                        continue
                    filled.append(kind)
                    marker = f"helios:pending_fill:{ident}:{kind}"
                    if done is not None and f"{ident}:{kind}" in done:
                        continue
                    if not r.set(marker, "1", nx=True, ex=_PENDING_FILL_MARKER_TTL_S):
                        continue
                    if field == "close":
                        est = hit["close"]
                    else:
                        est = min(level, hit["open"]) if below else max(level, hit["open"])
                    ok = db.log_activity(
                        "entry_shadow", ticker,
                        f"would fill: {kind} ${level:.2f} {'closed past' if field == 'close' else 'touched'} "
                        f"{hit['date']} (verdict {start}, {e.get('trigger')})", "neutral",
                        {**base, "would_fill": True, "kind": kind, "side": lv.get("side"),
                         "level": level, "fill_date": hit["date"], "fill_price_est": round(est, 4),
                         "bars_to_fill": window.index(hit) + 1, "price_field": field,
                         "in_zone_at_verdict": in_zone,
                         "fill_bar": {k: hit.get(k) for k in ("open", "high", "low", "close")},
                         "first_touch_date": outcomes[kind]["first_touch_date"],
                         "first_close_date": outcomes[kind]["first_close_date"],
                         "shadow_key": f"{ident}:{kind}",
                         "later_verdicts": [v for v in later
                                            if str(v.get("et_date") or "") <= hit["date"]]})
                    if ok:
                        logged += 1
                    else:
                        r.delete(marker)       # a failed write retries on the next run
                # The window is complete once the cache holds a bar dated on
                # or after the expiry (professor M216): a bar after it covers
                # an expiry on a weekend or holiday. Until then the row stays
                # pending and the next run re-reads the bars. An entry whose
                # bars never arrive still gets one row, window_complete
                # false, before analyst prunes it from the book.
                try:
                    exp_day = date.fromisoformat(expires)
                except ValueError:
                    exp_day = None     # unreadable: no date to wait for, log it now
                complete = exp_day is not None and any(
                    b["date"] >= expires for b in bars.get(ticker, []))
                if exp_day is None:
                    final = True
                elif today > expires and not complete:
                    last_chance = (exp_day + timedelta(
                        days=_PENDING_PRUNE_GRACE_DAYS - _PENDING_EXPIRED_SLACK_DAYS)).isoformat()
                    final = today >= last_chance
                else:
                    final = today > expires
                if final:
                    marker = f"helios:pending_fill:{ident}:expired"
                    if (done is None or f"{ident}:expired" not in done) and \
                            r.set(marker, "1", nx=True, ex=_PENDING_FILL_MARKER_TTL_S):
                        # would_fill says whether any level filled in the
                        # window; an empty window is missing data, not a
                        # miss (professor M207).
                        no_data = not window
                        if db.log_activity(
                                "entry_shadow", ticker, f"pending entry expired {expires} "
                                f"(filled: {', '.join(filled) or 'none'}"
                                f"{'; no bars in the window' if no_data else ''})", "neutral",
                                {**base, "would_fill": bool(filled), "expired": True,
                                 "filled_kinds": filled, "kinds": sorted(e.get("levels") or {}),
                                 "bars_in_window": len(window),
                                 "last_bar_date": window[-1]["date"] if window else None,
                                 "expired_no_data": no_data, "window_complete": complete,
                                 "level_outcomes": outcomes,
                                 "shadow_key": f"{ident}:expired",
                                 "later_verdicts": later}):
                            logged += 1
                        else:
                            r.delete(marker)
            except Exception as ex:
                log.debug(f"[{ticker}] pending entry check failed: {ex}")
        return logged
    except Exception as e:
        log.debug(f"pending entries check failed: {e}")
        return 0


def _check_tranche2_shadow(today=None) -> int:
    """Log-only: size_position buys a 55% first tranche (initial_entry_pct)
    and nothing ever schedules the rest. For each held position (fresh
    portfolio snapshot) whose newest open trade_outcomes row points at an
    approval whose risk_json.plan_compiled has an add_trigger (compile_plan:
    a close above t1's band), check the latest cached daily close. A close
    at or over add_trigger writes one activity_feed row, stage
    tranche2_shadow, with the raw facts: the close, by how much it cleared
    the level, held shares, the recorded first-tranche fraction and what
    the rest of the recorded full size would buy at the entry price. A
    position sized in one go (initial_entry_pct 1.0) is skipped. At most
    one row per approval (shadow_key plus a Redis NX marker; the DB keys
    alone when Redis is down). The first failed write stops the run. Whether
    a cross counts as confirmation is not decided here. Reads the snapshot,
    the approval and the price cache only: no broker call. Nothing is
    queued, sized or traded. Never raises."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import db
        import stop_lock
        from config import TRADING_ENV
        from agents.analyst import shadow_versions
        positions = _snapshot_positions_or_none(15)
        if not positions:
            if positions is None:
                log.warning("tranche2 shadow: no fresh portfolio snapshot; skipped")
            return 0
        held = {}
        for t, p in positions.items():
            try:
                qty = float((p or {}).get("qty") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            if qty > 0:
                held[t.upper()] = qty
        if not held:
            return 0
        # Newest open outcome per held ticker: the plan currently in force.
        newest = {}
        for o in db.get_open_outcomes():
            t = (o.get("ticker") or "").upper()
            if t in held and o.get("approval_id") and (
                    t not in newest or str(o.get("entry_date")) >= str(newest[t].get("entry_date"))):
                newest[t] = o
        if not newest:
            return 0
        today = (today or datetime.now(ZoneInfo("America/New_York")).date()).isoformat()
        bars = db.get_price_history_bulk(sorted(newest), days=5)
        done = db.get_activity_shadow_keys("tranche2_shadow")
        try:
            r = stop_lock._redis()
        except Exception:
            r = None              # r.set below then fails over to the DB keys

        def num(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        logged = 0
        for ticker, o in newest.items():
            try:
                approval = db.get_approval_by_id(int(o["approval_id"])) or {}
                risk = approval.get("risk_json") or {}
                plan = risk.get("plan_compiled") if isinstance(risk, dict) else None
                if not isinstance(plan, dict) or not plan.get("add_trigger"):
                    continue
                add_trigger = float(plan["add_trigger"])
                series = bars.get(ticker) or []
                if not series:
                    continue
                bar = series[-1]
                close = float(bar["close"])
                if close < add_trigger:
                    continue
                ts = approval.get("timestamp")
                if hasattr(ts, "astimezone") and ts.tzinfo is not None and \
                        bar["date"] < ts.astimezone(ZoneInfo("America/New_York")).date().isoformat():
                    continue           # a bar older than the plan: the cache hasn't caught up
                aid = int(approval.get("id") or o["approval_id"])
                key = f"{TRADING_ENV}:{ticker}:a{aid}:tranche2"
                if done is not None and key in done:
                    continue
                init_pct = num(risk.get("initial_entry_pct"))
                rest = round(1 - init_pct, 4) if init_pct is not None else None
                if rest is not None and rest <= 0:
                    continue           # sized in one go (initial_entry_pct 1.0): no second tranche
                full_dollar = num(risk.get("dollar_amount"))
                init_dollar = num(risk.get("initial_dollar"))
                shares_init = num(risk.get("shares_initial"))
                entry = num(o.get("entry_price"))
                # Plain arithmetic at the recorded entry price, whole shares:
                # the recorded full size less what the first tranche spent
                # (initial_dollar, after any cash or risk cap), else the full
                # size in shares less the shares bought.
                t2_shares = None
                if full_dollar and entry and entry > 0:
                    if init_dollar is not None:
                        t2_shares = int((full_dollar - init_dollar) / entry)
                    elif shares_init is not None:
                        t2_shares = int(full_dollar / entry) - int(shares_init)
                marker = f"helios:tranche2_shadow:{key}"
                try:
                    fresh = r.set(marker, "1", nx=True, ex=_PENDING_FILL_MARKER_TTL_S)
                except Exception:
                    fresh, marker = done is not None, None   # Redis down: the DB keys alone
                if not fresh:
                    continue
                prov = (approval.get("analyst_json") or {}).get("provenance")
                sid = prov.get("signal_id") if isinstance(prov, dict) else None
                ok = db.log_activity(
                    "tranche2_shadow", ticker,
                    f"close ${close:.2f} {bar['date']} at or over add_trigger ${add_trigger:.2f} "
                    f"(approval {aid}); second tranche not scheduled", "neutral",
                    {"ticker": ticker, "env": TRADING_ENV, "crossed": True,
                     "add_trigger": add_trigger, "close": close, "close_date": bar["date"],
                     "close_minus_trigger": round(close - add_trigger, 4),
                     "close_over_trigger_pct": round((close - add_trigger) / add_trigger * 100, 4),
                     "bar_is_today": bar["date"] == today,
                     "held_shares": held[ticker],
                     "shares_initial": risk.get("shares_initial"),
                     "initial_entry_pct": init_pct,
                     "full_size_dollar": full_dollar,
                     "initial_dollar": init_dollar,
                     "entry_price": entry,
                     "tranche2_fraction": rest,
                     "tranche2_shares_est": t2_shares,
                     "approval_id": aid, "approval_action": approval.get("action"),
                     "approval_at": ts.isoformat() if hasattr(ts, "isoformat") else ts,
                     "verdict_signal_id": sid,
                     "plan_source": plan.get("source"),
                     "plan_t1": plan.get("t1"),
                     "plan_time_stop": plan.get("time_stop"),
                     "shadow_key": key, **shadow_versions()})
                if ok:
                    logged += 1
                    continue
                if marker:
                    try:
                        r.delete(marker)      # a failed write retries on the next run
                    except Exception:
                        pass
                # The DB is likely down: stop here rather than spill one
                # write per remaining ticker.
                log.warning(f"tranche2 shadow: write failed at {ticker}; stopped after {logged} rows")
                break
            except Exception as ex:
                log.warning(f"[{ticker}] tranche2 shadow failed: {ex}")
        return logged
    except Exception as e:
        log.warning(f"tranche2 shadow failed: {e}")
        return 0


_C6_OUTCOME_HORIZONS = (5, 10, 20)   # trading days
_C6_OUTCOME_EXPIRE_DAYS = 40         # sessions after which an incomplete verdict gets its status
_C6_OUTCOME_LOOKBACK_DAYS = 75       # calendar days of verdict rows re-read each run
_C6_OUTCOME_MARKER_TTL_S = 120 * 86400   # past the lookback plus the 40-session wait


def _c6_trading_days_after(et_date: str, today: str) -> int:
    """NYSE sessions after et_date up to and including today."""
    from datetime import date, timedelta
    from market_calendar import NYSE_HOLIDAYS
    d, end, n = date.fromisoformat(et_date) + timedelta(days=1), date.fromisoformat(today), 0
    while d <= end:
        n += d.weekday() < 5 and d not in NYSE_HOLIDAYS
        d += timedelta(days=1)
    return n


def _check_c6_bar_outcomes(today=None) -> int:
    """Log-only forward outcomes for the c6 bar shadow (agents/c6_shadow.py).
    For each c6_bar_shadow verdict row of the last _C6_OUTCOME_LOOKBACK_DAYS
    days with an entry_ref_price, and each horizon N in
    _C6_OUTCOME_HORIZONS: once the cache holds N daily bars dated after the
    verdict's ET date, one row, stage c6_bar_shadow, kind outcome, with the
    Nth bar's close and its return from entry_ref_price (and from the
    verdict day's close when that bar is cached). Closes are adjusted; see
    the c6_shadow docstring. Each verdict also gets one kind outcome_status
    row, so attrition is countable: complete (every horizon logged), no_ref
    (no entry_ref_price), or, _C6_OUTCOME_EXPIRE_DAYS sessions after the
    verdict with horizons missing, missing_bars (some bars cached after the
    verdict, too few) or expired (none). One row per key (shadow_key
    <verdict key>:r<N> / :status, plus a Redis NX marker). The first failed
    write stops the run. Reads the DB cache only. Never raises."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import db
        import stop_lock
        from agents.c6_shadow import MARKER_PREFIX, STAGE
        from agents.analyst import shadow_versions
        rows = db.get_activity_rows(STAGE, "verdict", days=_C6_OUTCOME_LOOKBACK_DAYS)
        rows = [v for v in (rows or []) if v.get("shadow_key") and v.get("as_of") and v.get("ticker")]
        if not rows:
            return 0
        today = (today or datetime.now(ZoneInfo("America/New_York")).date()).isoformat()
        done = db.get_activity_shadow_keys(STAGE)
        try:
            r = stop_lock._redis()
        except Exception:
            r = None
        bars = db.get_price_history_bulk(sorted({v["ticker"] for v in rows}), days=90)
        logged = 0

        def write(key, ticker, msg, extra):
            """None when written now, True when already there, False when
            the write failed."""
            if done is not None and key in done:
                return True
            marker = MARKER_PREFIX + key
            try:
                fresh = r.set(marker, "1", nx=True, ex=_C6_OUTCOME_MARKER_TTL_S)
            except Exception:
                fresh, marker = done is not None, None   # Redis down: the DB keys alone
            if not fresh:
                return True
            if db.log_activity(STAGE, ticker, msg, "neutral",
                               {**extra, "shadow_key": key, **shadow_versions()}):
                return None
            if marker:
                try:
                    r.delete(marker)      # a failed write retries on the next run
                except Exception:
                    pass
            return False

        for v in rows:
            ticker = v["ticker"]
            try:
                et_date = datetime.fromisoformat(v["as_of"]).astimezone(
                    ZoneInfo("America/New_York")).date().isoformat()
                base = {"ticker": ticker, "env": v.get("env"), "signal_id": v.get("signal_id"),
                        "verdict_shadow_key": v["shadow_key"], "verdict_et_date": et_date,
                        "would_queue": v.get("would_queue"), "c7_gates": v.get("c7_gates"),
                        "ref_price_source": v.get("ref_price_source")}
                try:
                    ref = float(v.get("entry_ref_price") or 0)
                except (TypeError, ValueError):
                    ref = 0.0
                writes = []
                have = []
                status = None
                after = []
                if ref <= 0:
                    status = "no_ref"
                else:
                    series = bars.get(ticker) or []
                    after = [b for b in series if b["date"] > et_date]
                    day0 = next((b for b in series if b["date"] == et_date), None)
                    c0 = float(day0["close"]) if day0 else None
                    for n in _C6_OUTCOME_HORIZONS:
                        if len(after) < n:
                            break
                        bar = after[n - 1]
                        close = float(bar["close"])
                        writes.append((f"{v['shadow_key']}:r{n}",
                                       f"c6 {v.get('action')} {n}-day return "
                                       f"{(close / ref - 1) * 100:+.2f}% (verdict {et_date})",
                                       {**base, "kind": "outcome", "horizon_days": n,
                                        "entry_ref_price": ref, "close": close,
                                        "close_date": bar["date"],
                                        "return_pct": round((close / ref - 1) * 100, 4),
                                        "verdict_day_close": c0,
                                        "return_from_verdict_close_pct": (
                                            round((close / c0 - 1) * 100, 4) if c0 else None)}))
                        have.append(n)
                    if len(have) == len(_C6_OUTCOME_HORIZONS):
                        status = "complete"
                    elif _c6_trading_days_after(et_date, today) >= _C6_OUTCOME_EXPIRE_DAYS:
                        status = "missing_bars" if after else "expired"
                if status:
                    writes.append((f"{v['shadow_key']}:status",
                                   f"c6 {v.get('action')} outcome: {status} (verdict {et_date})",
                                   {**base, "kind": "outcome_status", "status": status,
                                    "horizons_logged": have,
                                    "bars_after_verdict": len(after) if ref > 0 else None}))
                for key, msg, extra in writes:
                    res = write(key, ticker, msg, extra)
                    if res is False:
                        log.warning(f"c6 bar outcomes: write failed at {ticker}; "
                                    f"stopped after {logged} rows")
                        return logged
                    logged += res is None
            except Exception as ex:
                log.warning(f"[{ticker}] c6 bar outcome failed: {ex}")
        return logged
    except Exception as e:
        log.warning(f"c6 bar outcomes failed: {e}")
        return 0


@app.task(name="tasks.c6_bar_shadow", ignore_result=True, soft_time_limit=20, time_limit=30)
def c6_bar_shadow(payload: dict) -> None:
    """Log-only c6 bar shadow row (agents/c6_shadow.py), enqueued on ops by
    crew's routing so its broker, DB and cache reads never hold up the next
    candidate. Raises only SoftTimeLimitExceeded."""
    from agents.c6_shadow import run
    run(payload)


# Unheld tickers enter the plan_cross_section snapshot when their latest
# analyst verdict is this many days old or newer (timestamp > now - N days).
_CROSS_SECTION_VERDICT_DAYS = 5
# A held ticker's latest verdict is looked up over this many days (a bound
# on the read; an older verdict gives null components).
_CROSS_SECTION_HELD_VERDICT_DAYS = 365
_CROSS_SECTION_MARKER_TTL_S = 10 * 86400
_CROSS_SECTION_SNAPSHOT_MAX_AGE_MIN = 60   # portfolio snapshot fresh enough to say "held"


def _xs_num(v):
    """A finite float, else None (no bools)."""
    import math
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _xs_dist(level, price):
    """(level - price) / price, None when either is missing."""
    level, price = _xs_num(level), _xs_num(price)
    return round((level - price) / price, 6) if level is not None and price else None


def _plan_cross_section_components(v: dict | None, snapshot_day) -> dict:
    """The compile_plan components a ticker's latest verdict already stored:
    structural_stop / t1 / t2 / compile_source from signals.decision.udr_structural
    (compile_plan run at verdict time), time_stop from decision.pending_entry
    .expires when that was recorded, else compile_plan re-run with no ladder
    on the stored model_output, verdict price and verdict ET date (time_stop
    does not read the ladder, so that part is a pure function of stored
    inputs; the levels do read it and are never recomputed). No LLM call.
    plan_source: stored / stored+time_stop_recomputed / time_stop_stored /
    time_stop_recomputed / none. Missing inputs give None, never an exception."""
    from zoneinfo import ZoneInfo
    out = {"plan_source": "none", "structural_stop": None, "t1": None, "t2": None,
           "time_stop": None, "time_stop_source": None, "compile_source": None,
           "verdict_price": None, "verdict_signal_id": None, "verdict_at": None,
           "verdict_et_date": None, "verdict_age_days": None, "action": None,
           "model_action": None, "conviction": None}
    if not isinstance(v, dict):
        return out
    et = ZoneInfo("America/New_York")
    ts = v.get("timestamp")
    vday = None
    try:
        vday = ts.astimezone(et).date() if ts is not None else None
    except Exception:
        vday = None
    out.update(verdict_signal_id=v.get("signal_id"), action=v.get("signal"),
               model_action=v.get("model_action"), conviction=v.get("conviction"),
               verdict_at=ts.isoformat() if hasattr(ts, "isoformat") else None,
               verdict_et_date=vday.isoformat() if vday else None,
               verdict_age_days=(snapshot_day - vday).days if vday else None)
    udr = v.get("udr_structural") if isinstance(v.get("udr_structural"), dict) else {}
    pend = v.get("pending_entry") if isinstance(v.get("pending_entry"), dict) else {}
    plan = v.get("plan") if isinstance(v.get("plan"), dict) else {}
    out["verdict_price"] = next((p for p in (_xs_num(udr.get("price")), _xs_num(pend.get("price")),
                                             _xs_num(plan.get("price"))) if p), None)
    stored_levels = any(k in udr for k in ("structural_stop", "t1", "t2"))
    if stored_levels:
        out.update(structural_stop=_xs_num(udr.get("structural_stop")), t1=_xs_num(udr.get("t1")),
                   t2=_xs_num(udr.get("t2")), compile_source=udr.get("plan_source"))
    expires = pend.get("expires")
    if expires not in (None, "", "None"):
        out.update(time_stop=str(expires), time_stop_source="stored")
    else:
        parsed = v.get("model_output")
        if isinstance(parsed, dict) and out["verdict_price"] and vday:
            try:
                from agents.plans import compile_plan
                ts_ = compile_plan(parsed, out["verdict_price"], None, None, vday).get("time_stop")
                if ts_ not in (None, "", "None"):
                    out.update(time_stop=str(ts_), time_stop_source="recomputed")
            except Exception:
                pass
    if stored_levels:
        out["plan_source"] = ("stored+time_stop_recomputed"
                              if out["time_stop_source"] == "recomputed" else "stored")
    elif out["time_stop_source"] == "recomputed":
        out["plan_source"] = "time_stop_recomputed"
    elif out["time_stop_source"] == "stored":
        out["plan_source"] = "time_stop_stored"
    return out


def _log_plan_cross_section(now=None) -> int:
    """Log-only (audit 2026-11-03 01:29 #5): each ticker is judged alone,
    so nothing records the held and candidate names side by side. Once per
    trading day this writes one activity_feed row (stage
    plan_cross_section) per ticker for every held position and every
    unheld ticker with an analyst verdict in the last
    _CROSS_SECTION_VERDICT_DAYS days: the latest verdict's stored
    compile_plan components (_plan_cross_section_components), the cached
    close (price_history, else the snapshot's mark for a held name), shares
    if held, and plain distances ((level - price) / price). Raw components
    only: no ranking, score, threshold or rotate decision. Dedup by
    shadow_key {env}:{snapshot_date}:{ticker} in activity_feed, with a Redis
    marker when that read fails. DB and Redis reads only; no broker,
    network or LLM call. Nothing is queued or traded. Never raises."""
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        import db
        import stop_lock
        from config import TRADING_ENV
        from market_calendar import NYSE_HOLIDAYS
        from agents.analyst import shadow_versions
        now = now or datetime.now(timezone.utc)
        day = now.astimezone(ZoneInfo("America/New_York")).date()
        if day.weekday() >= 5 or day in NYSE_HOLIDAYS:
            return 0
        snapshot_date = day.isoformat()

        try:
            positions = _snapshot_positions_or_none(_CROSS_SECTION_SNAPSHOT_MAX_AGE_MIN)
        except Exception:
            positions = None
        broker = {}
        for t, p in (positions or {}).items():
            qty = _xs_num(p.get("qty")) if isinstance(p, dict) else None
            if qty:
                broker[str(t).upper()] = p
        try:
            stop_rows = {str(s["ticker"]).upper() for s in db.get_all_position_stops(strict=True)}
        except Exception:
            stop_rows = None
        if positions is None and stop_rows is None:
            # Neither held source is known: rows would say held=False for
            # every name, so skip the day.
            log.warning(f"plan cross-section {snapshot_date} skipped: no portfolio snapshot "
                        "and the position_stops read failed")
            return 0
        held = set(broker) | (stop_rows or set())

        recent = db.get_latest_verdicts(days=_CROSS_SECTION_VERDICT_DAYS)
        held_v = (db.get_latest_verdicts(tickers=sorted(held), days=_CROSS_SECTION_HELD_VERDICT_DAYS)
                  if held else {})
        if recent is None or held_v is None:
            log.warning(f"plan cross-section {snapshot_date} skipped: the verdict read failed")
            return 0                  # no partial snapshot
        verdicts = {**recent, **held_v}
        tickers = sorted(held | set(recent))
        if not tickers:
            return 0
        bars = db.get_price_history_bulk(tickers, days=1) or {}
        done = db.get_activity_shadow_keys("plan_cross_section")
        try:
            r = stop_lock._redis()
        except Exception:
            r = None              # r.set below then fails over to the DB keys
        versions = shadow_versions()
        logged = 0
        for ticker in tickers:
            try:
                key = f"{TRADING_ENV}:{snapshot_date}:{ticker}"
                if done is not None and key in done:
                    continue
                marker = f"helios:plan_cross_section:{key}"
                try:
                    fresh = r.set(marker, "1", nx=True, ex=_CROSS_SECTION_MARKER_TTL_S)
                except Exception:
                    fresh, marker = done is not None, None   # Redis down: the DB keys alone
                if not fresh:
                    continue
                pos = broker.get(ticker)
                last = (bars.get(ticker) or [None])[-1]
                price, price_src, price_date = None, None, None
                if last and _xs_num(last.get("close")):
                    price, price_src = _xs_num(last["close"]), "price_history_close"
                    price_date = str(last.get("date"))
                elif pos and _xs_num(pos.get("market_value")) and _xs_num(pos.get("qty")):
                    price = round(_xs_num(pos["market_value"]) / _xs_num(pos["qty"]), 4)
                    price_src = "snapshot_mark"
                comp = _plan_cross_section_components(verdicts.get(ticker), day)
                held_broker = (ticker in broker) if positions is not None else None
                held_stop_row = (ticker in stop_rows) if stop_rows is not None else None
                ts = comp["time_stop"]
                try:
                    days_to_ts = ((datetime.strptime(ts, "%Y-%m-%d").date() - day).days
                                  if ts else None)
                except ValueError:
                    days_to_ts = None
                extra = {
                    "snapshot_date": snapshot_date, "shadow_key": key, "env": TRADING_ENV,
                    **versions,
                    "held": ticker in held, "held_broker": held_broker,
                    "held_stop_row": held_stop_row,
                    "shares": _xs_num(pos.get("qty")) if pos else None,
                    "in_recent_window": ticker in recent,
                    "verdict_window_days": _CROSS_SECTION_VERDICT_DAYS,
                    "price": price, "price_source": price_src, "price_date": price_date,
                    **comp,
                    "dist_structural_stop": _xs_dist(comp["structural_stop"], price),
                    "dist_t1": _xs_dist(comp["t1"], price),
                    "dist_t2": _xs_dist(comp["t2"], price),
                    "change_since_verdict": _xs_dist(price, comp["verdict_price"]),
                    "days_to_time_stop": days_to_ts,
                }
                ok = db.log_activity(
                    "plan_cross_section", ticker,
                    f"cross-section {snapshot_date}: {'held' if extra['held'] else 'candidate'}, "
                    f"latest verdict {comp['action'] or 'none'} ({comp['verdict_et_date'] or 'n/a'})",
                    "neutral", extra)
                if ok:
                    logged += 1
                    continue
                if marker:
                    try:
                        r.delete(marker)      # a failed write retries on the next run
                    except Exception:
                        pass
                # The DB is likely down: stop here rather than spill one
                # write per remaining ticker.
                log.warning(f"plan cross-section {snapshot_date}: write failed at {ticker}; "
                            f"stopped after {logged} rows")
                break
            except Exception as ex:
                log.warning(f"[{ticker}] plan cross-section row failed: {ex}")
        return logged
    except Exception as e:
        log.warning(f"plan cross-section snapshot failed: {e}")
        return 0


@app.task(name="tasks.refresh_price_history", bind=True, max_retries=2)
def refresh_price_history(self):
    """Nightly refresh of the price_history cache (watchlist + SPY/QQQ)."""
    try:
        from ingestion.price_history import refresh_all_price_history
        count = refresh_all_price_history()
        log.info(f"Price history refreshed: {count} tickers updated")
        _heartbeat_ok("price_history", count)
        try:
            _check_pending_entries()   # log-only shadow on the bars just stored
        except Exception as e:
            log.warning(f"pending entries hook failed: {e}")
        try:
            _log_plan_cross_section()   # log-only snapshot; never fails this task
        except Exception as e:
            log.warning(f"plan cross-section hook failed: {e}")
        try:
            _check_tranche2_shadow()   # log-only; never fails this task
        except Exception as e:
            log.warning(f"tranche2 shadow hook failed: {e}")
        try:
            _check_c6_bar_outcomes()   # log-only; never fails this task
        except Exception as e:
            log.warning(f"c6 bar outcomes hook failed: {e}")
    except Exception as e:
        log.error(f"refresh_price_history failed: {e}")
        _heartbeat_fail("price_history", str(e))
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.refresh_trade_ladder", bind=True, max_retries=2)
def refresh_trade_ladder(self):
    """Nightly per-ticker trade ladder (2026-09-13, swing-trading redesign)
    -- technical rungs (swings/MAs/base/volume-nodes/gaps) + valuation
    layer (fair value, tier). Scheduled after refresh-price-history and
    refresh-fundamentals so it reads same-day-fresh bars and multiples,
    not yesterday's. Write-only: this stage doesn't gate or size anything
    itself, see agents/setup_gates.py (SETUP_GATES_ENABLED) for the
    consumer side."""
    try:
        from agents.trade_ladder import refresh_all_ladders
        count = refresh_all_ladders()
        log.info(f"Trade ladder refreshed: {count} tickers updated")
        _heartbeat_ok("trade_ladder", count)
    except Exception as e:
        log.error(f"refresh_trade_ladder failed: {e}")
        _heartbeat_fail("trade_ladder", str(e))
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.refresh_sp500_history", bind=True, max_retries=2)
def refresh_sp500_history(self):
    """Weekly refresh of the price_history cache for the full S&P 500
    universe (minus the watchlist, already refreshed nightly) -- feeds
    market_regime's GICS-sector breadth and candidate_scanner's movement
    screen. Weekly, not nightly: ~400 extra yfinance calls is fine at this
    cadence but wasteful daily for signals that don't need day-fresh data
    (see ingestion/price_history.py's refresh_sp500_price_history docstring)."""
    try:
        from ingestion.price_history import refresh_sp500_price_history

        def run():
            count = refresh_sp500_price_history()
            log.info(f"S&P 500 price history refreshed: {count} tickers updated")
            # 0 means no ticker's newest bar advanced (a current cache is
            # skipped, so this is Yahoo serving stale bars or failing): leave
            # the last heartbeat to age into the staleness alert.
            if count:
                _heartbeat_ok("sp500_history", count)
            else:
                log.warning("S&P 500 price history: no ticker advanced; heartbeat not recorded")
        # One run at a time with the weekday catch-up (ingestion runs 4 wide).
        _run_singleton("sp500_history", run, max_hold_s=_SP500_REFRESH_HOLD_S)
    except Exception as e:
        log.error(f"refresh_sp500_history failed: {e}")
        _heartbeat_fail("sp500_history", str(e))
        raise self.retry(exc=e, countdown=600)


_SP500_REFRESH_HOLD_S = 4 * 3600   # a full pass takes ~2.6 h


@app.task(name="tasks.refresh_sp500_history_catchup", bind=True, max_retries=0)
def refresh_sp500_history_catchup(self):
    """Weekday 05:15: refetch only S&P 500 tickers more than 7 days stale.
    The weekly run has no other catch-up if it dies partway or runs on a
    Saturday the stack happens to be up (10-24: killed at the 19:00 stop,
    stale until the next week). A no-op when the weekly run finished."""
    try:
        from ingestion.price_history import refresh_sp500_price_history

        def run():
            count = refresh_sp500_price_history(max_age_days=7)
            log.info(f"S&P 500 price history catch-up: {count} tickers updated")
            # Only when it did something: a no-op day mustn't keep the
            # weekly run's health check green if that run stopped happening.
            if count:
                _heartbeat_ok("sp500_history", count)
        _run_singleton("sp500_history", run, max_hold_s=_SP500_REFRESH_HOLD_S)
    except Exception as e:
        log.error(f"refresh_sp500_history_catchup failed: {e}")
        _heartbeat_fail("sp500_history", str(e))


@app.task(name="tasks.refresh_sp500_fundamentals", bind=True, max_retries=2)
def refresh_sp500_fundamentals(self):
    """Weekly refresh of the sp500_fundamentals table (2026-09-13,
    broadened Comps) -- ~500 sequential yfinance calls, same cadence
    reasoning as refresh_sp500_history above: peer multiples are a slow-
    moving background signal, not worth the extra ingestion cost nightly.
    Feeds agents.valuation.compute_comps_fair_value's broadened
    industry-based peer search via ingestion.fundamentals.
    get_industry_peer_median."""
    try:
        from ingestion.fundamentals import refresh_sp500_fundamentals as _refresh
        count = _refresh()
        log.info(f"S&P 500 fundamentals refreshed: {count} tickers updated")
        _heartbeat_ok("sp500_fundamentals", count)
    except Exception as e:
        log.error(f"refresh_sp500_fundamentals failed: {e}")
        _heartbeat_fail("sp500_fundamentals", str(e))
        raise self.retry(exc=e, countdown=600)


@app.task(name="tasks.refresh_fundamentals", bind=True, max_retries=2)
def refresh_fundamentals(self):
    """Nightly refresh of fundamental metrics for the watchlist (free, via yfinance)."""
    try:
        from ingestion.fundamentals import refresh_all_fundamentals
        count = refresh_all_fundamentals()
        log.info(f"Fundamentals refreshed: {count} tickers updated")
        _heartbeat_ok("fundamentals", count)
    except Exception as e:
        log.error(f"refresh_fundamentals failed: {e}")
        _heartbeat_fail("fundamentals", str(e))
        raise self.retry(exc=e, countdown=300)


def _get_stale_ingestion_sources() -> list[str]:
    """Shared by check_ingestion_health and send_daily_digest."""
    import db
    from datetime import datetime, timezone

    # Expected max gap per source, generously above its beat schedule interval
    # (sp500_history: 8 days, as a weekly run ends ~2.6 h after it starts and a
    # later finish than the week before must not read as stale).
    expected_minutes = {
        "edgar": 20, "options_flow": 15, "macro": 90, "arxiv": 1560,
        "congressional": 400, "insider": 60, "news": 30, "fundamentals": 1560,
        "price_history": 1560, "sp500_history": 11520, "analyst_ratings": 60,
        # Weekly at Mon 09:00, the same minute as an hourly check: a week
        # plus a day, like sp500_history, so a run still in progress isn't stale.
        "candidate_scan": 11520,
    }

    # Calendar awareness (audit 06:53, A0653-6): options_flow only polls in
    # regular hours, and the weekday-only daily jobs don't run at weekends,
    # so off-hours and Monday checks reported them stale every week.
    from zoneinfo import ZoneInfo
    et = datetime.now(ZoneInfo("America/New_York"))
    try:
        # Holidays and early closes too (A0853-7); 15 min after the open so
        # the first poll has had its chance.
        import market_data
        rth = (market_data.is_market_hours(datetime.now(timezone.utc))
               and (et.hour, et.minute) >= (9, 45))
    except Exception:
        rth = et.weekday() < 5 and (9, 45) <= (et.hour, et.minute) < (16, 0)
    rth_only = {"options_flow"}
    weekday_daily = {"fundamentals", "price_history", "arxiv"}

    stale = []
    now = datetime.now(timezone.utc)
    health = {h["source"]: h for h in db.get_ingestion_health()}
    if not _QUIVER_ENABLED:
        expected_minutes.pop("congressional", None)
    for source, max_gap in expected_minutes.items():
        if source in rth_only and not rth:
            continue
        if source in weekday_daily and et.weekday() == 0:
            max_gap += 2 * 1440  # Friday's run is the last before Monday's
        rec = health.get(source)
        if not rec or not rec.get("last_success_at"):
            stale.append(f"{source}: never succeeded")
            continue
        age_min = (now - rec["last_success_at"]).total_seconds() / 60
        if age_min > max_gap:
            stale.append(f"{source}: last success {age_min:.0f}min ago (expected &lt;{max_gap}min)")
    return stale


import time as _time_mod
_PROCESS_START = _time_mod.time()   # this worker's start (prefork children inherit it)
_INGESTION_GRACE_S = 1800


@app.task(name="tasks.check_ingestion_health", bind=True, max_retries=1)
def check_ingestion_health(self):
    """Alerts when an ingestion source hasn't succeeded within its expected
    cadence. Same standing-condition-spam bug as reconcile_stop_orders had
    (found together 2026-07-27): this ran hourly with no dedup, so a source
    that stayed stale for a full trading day re-sent the identical alert
    every single hour. Throttled per the specific set of stale sources, so
    a genuinely new/different issue still alerts promptly."""
    try:
        import alerts
        import time

        # Pollers can't have run while the stack was down, so every source
        # reads stale on the first checks after a start: an alert every
        # morning that teaches the reader to ignore it (audit 04:53, A0453-7).
        if time.time() - _PROCESS_START < _INGESTION_GRACE_S:
            log.info("check_ingestion_health: within the post-start grace period, skipped")
            return
        stale = _get_stale_ingestion_sources()
        if stale:
            log.warning("Ingestion health issues: " + " | ".join(stale))
            issue_key = "ingestion_" + "_".join(sorted(s.split(":")[0] for s in stale))
            if _reconcile_alert_due(issue_key):
                alerts._send("⚠️ <b>Ingestion Health Check</b>\n" + "\n".join(f"• {s}" for s in stale))
    except Exception as e:
        log.error(f"check_ingestion_health failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.flush_alert_digest")
def flush_alert_digest():
    """One Telegram message listing the low-severity alerts that
    alert_policy held back as repeats since the last flush."""
    try:
        import alerts
        alerts.flush_digest()
    except Exception as e:
        log.error(f"flush_alert_digest failed: {e}")


@app.task(name="tasks.send_daily_digest", bind=True, max_retries=2)
def send_daily_digest(self):
    """
    Post-close daily digest — day P&L vs SPY, open positions with stop
    distances, tomorrow's catalysts, pending approvals, ingestion health,
    and halt status. alerts.send_daily_summary() existed but nothing ever
    scheduled it; this is its replacement on the beat schedule.
    """
    try:
        import db
        import alerts
        from datetime import date, timedelta, datetime, timezone

        client  = _get_client()
        account = client.get_account()
        pv      = float(account.portfolio_value)

        # Day P&L: compare against the first snapshot recorded today. Both
        # sides are ET calendar dates (the snapshot's own date was UTC, so
        # a 19:00-24:00 EST snapshot counted as the next day's first).
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        history = db.get_portfolio_history(limit=500)
        today_et = datetime.now(timezone.utc).astimezone(_et).date()
        todays = [h for h in history
                  if (h["timestamp"] if h["timestamp"].tzinfo else h["timestamp"].replace(tzinfo=timezone.utc))
                  .astimezone(_et).date() == today_et]
        start_of_day_value = todays[0]["total_value"] if todays else pv
        day_pl = pv - start_of_day_value
        day_pl_pct = day_pl / start_of_day_value if start_of_day_value else 0.0

        spy_pl_pct = None
        try:
            import yfinance as yf
            spy_hist = yf.Ticker("SPY").history(period="2d")
            if len(spy_hist) >= 2:
                prev_close, last_close = spy_hist["Close"].iloc[-2], spy_hist["Close"].iloc[-1]
                spy_pl_pct = float((last_close - prev_close) / prev_close)
        except Exception as e:
            log.debug(f"SPY comparison unavailable for digest: {e}")

        positions = client.get_all_positions()
        stops = {s["ticker"]: s for s in db.get_all_position_stops()}
        positions_text = []
        for p in positions:
            stop_rec = stops.get(p.symbol)
            dist = ""
            current = float(p.current_price or 0)
            if stop_rec and current:
                d = (current - stop_rec["stop_price"]) / current
                dist = f", stop {d:+.1%} away"

            # 2026-09-17, user request: entry price, days held, %gain, and
            # current trade thesis alongside what was already here (share
            # count, P&L, stop distance) -- makes the digest answer "why
            # am I holding this and how has it done" without a separate
            # lookup, using the thesis-tracking system already in place
            # (db.get_thesis) rather than duplicating that data.
            entry_line = ""
            thesis_line = ""
            if stop_rec:
                entry_price = stop_rec.get("entry_price")
                opened_at = stop_rec.get("opened_at")
                if entry_price and current:
                    gain_pct = (current - entry_price) / entry_price
                    days_held = (datetime.now(timezone.utc) - opened_at).days if opened_at else None
                    days_str = f", {days_held}d held" if days_held is not None else ""
                    entry_line = f", entry ${entry_price:,.2f}{days_str}, {gain_pct:+.1%}"

                try:
                    thesis = db.get_thesis(p.symbol)
                    if thesis:
                        latest = thesis.get("latest") or thesis.get("original") or {}
                        summary = (latest.get("thesis_summary") or "").strip()
                        if summary:
                            # Keep short -- this runs per position in an
                            # already-long digest, not the place for the
                            # full multi-sentence reasoning.
                            if len(summary) > 140:
                                summary = summary[:137] + "..."
                            thesis_line = f"\n    💭 {summary}"
                except Exception as e:
                    log.debug(f"[{p.symbol}] thesis lookup failed for digest: {e}")

            positions_text.append(
                f"{p.symbol}: {p.qty:.0f}sh{entry_line}, P&amp;L ${float(p.unrealized_pl):+,.0f}{dist}{thesis_line}"
            )

        catalysts_text = []
        try:
            from catalyst_service.client import query_events
            tomorrow = (today_et + timedelta(days=1)).isoformat()
            events = query_events(start=tomorrow, end=tomorrow)
            catalysts_text = [f"{e.get('ticker')}: {e.get('event_type')}" for e in events]
        except Exception as e:
            log.debug(f"Catalyst lookup unavailable for digest: {e}")

        pending_count = len(db.get_pending_approvals())
        from config import TRADING_ENV as _TRADING_ENV
        halted = db.get_state(f"trading_halted_{_TRADING_ENV}", "false") == "true"
        ingestion_issues = _get_stale_ingestion_sources()

        atr_fallback_tickers = None
        try:
            from risk_manager import atr_fallback_tickers_today
            atr_fallback_tickers = atr_fallback_tickers_today()
        except Exception as e:
            log.debug(f"ATR fallback tickers unavailable for digest: {e}")

        alerts.send_daily_digest(
            portfolio_value=pv, day_pl=day_pl, day_pl_pct=day_pl_pct,
            spy_pl_pct=spy_pl_pct, positions_text=positions_text,
            catalysts_text=catalysts_text, pending_count=pending_count,
            halted=halted, ingestion_issues=ingestion_issues,
            atr_fallback_tickers=atr_fallback_tickers,
        )
    except Exception as e:
        log.error(f"send_daily_digest failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.flag_orphan_outcomes")
def flag_orphan_outcomes():
    """Open trade_outcomes rows whose ticker isn't held get orphaned_at set:
    stale-stop cleanup deletes the stop record but leaves the outcome open
    when no exit fill is found (professor M111; 108 of 113 pilot rows open).
    Held tickers come from the fresh portfolio snapshot, not IBKR. An empty
    snapshot is skipped: stale and flat look the same."""
    import db
    held = _held_qty_tickers(15)
    if not held:
        log.warning("flag_orphan_outcomes: no fresh held positions in the snapshot; skipped")
        return None
    newly, total = db.flag_orphan_outcomes(held)
    log.info(f"flag_orphan_outcomes: {newly} newly flagged, {total} open outcome rows flagged orphaned")
    return total


@app.task(name="tasks.review_plan_level_hit")
def review_plan_level_hit(ticker: str, kind: str, level: str, price: float | None = None):
    """A held name hit its plan level; review it now (PLAN_LEVEL_REVIEW_ENABLED,
    default off). See agents.crew.review_plan_level."""
    try:
        from agents.crew import review_plan_level
        result = review_plan_level(ticker, kind, level, price)
        return (result or {}).get("action")
    except Exception as e:
        log.error(f"[{ticker}] review_plan_level_hit failed: {e}")
        return None


@app.task(name="tasks.watch_plan_levels")
def watch_plan_levels():
    """Log-only (audit A2114-4/-7): during market hours, check every active
    shadow plan's levels and dates against the current price and write one
    PLAN_LEVEL_HIT row per plan and kind per day: what a level-triggered
    review would have re-analysed. Nothing is queued or traded. Prices come
    from the latest portfolio snapshot (held names, under 15 min old; no
    IBKR call from this worker); unheld plans get the date checks only."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import db
        import market_data
        import stop_lock
        from agents.plans import level_hits
        if not market_data.is_market_hours():
            return 0
        today = datetime.now(ZoneInfo("America/New_York")).date()
        r = stop_lock._redis()
        held = _held_prices()
        plans_now = db.get_active_plans()
        if not held and plans_now:
            # Indistinguishable from "no level crossed" otherwise (SEF, 10-30).
            log.warning(f"watch_plan_levels: no fresh held prices (snapshot stale or missing, or "
                        f"nothing held); price checks skipped for {len(plans_now)} plan(s), "
                        f"date checks only")
        logged = 0
        for plan in plans_now:
            price = held.get(plan["ticker"])
            for kind, level in level_hits(plan, price, today):
                if not r.set(f"helios:plan_hit:{plan['id']}:{kind}", "1", nx=True, ex=86400):
                    continue
                from config import PLAN_LEVEL_REVIEW_ENABLED
                # At most one queued review per ticker per ET day, whatever
                # the plan or level (review of 2be84b5: two kinds in one tick
                # could stack sells).
                queued = bool(PLAN_LEVEL_REVIEW_ENABLED and plan["ticker"] in held and r.set(
                    f"helios:plan_review:{plan['ticker']}:{today.isoformat()}", "1", nx=True, ex=86400))
                db.log_signal(plan["ticker"], "PLAN_LEVEL_HIT", 0.0,
                              f"plan {plan['id']}: {kind} ({level}) at ${price or 0:.2f}; "
                              + ("review queued" if queued else "would re-review"),
                              provenance={"plan_id": plan["id"], "kind": kind, "level": level,
                                          "price": price, "plan_source": plan.get("source"),
                                          "review_queued": queued})
                if queued:
                    review_plan_level_hit.apply_async(
                        args=[plan["ticker"], kind, str(level), price],
                        queue=LLM_QUEUE_HELD, expires=6 * 3600)
                logged += 1
        return logged
    except Exception as e:
        log.error(f"watch_plan_levels failed: {e}")
        return 0


@app.task(name="tasks.take_portfolio_snapshot", bind=True, max_retries=3)
def take_portfolio_snapshot(self):
    """
    Takes a portfolio snapshot and logs to Supabase for the dashboard.
    Also snapshots SPY/QQQ (via yfinance — free, no extra IBKR market data
    lines) alongside it, so the equity curve can be compared to a real
    benchmark instead of only a flat starting-dollar-value line.
    """
    try:
        import db
        client    = _get_client()
        account   = client.get_account()
        positions = client.get_all_positions()

        benchmark = {}
        try:
            import yfinance as yf
            for sym in ("SPY", "QQQ"):
                price = yf.Ticker(sym).fast_info.last_price
                if price:
                    benchmark[sym] = float(price)
        except Exception as e:
            log.debug(f"Benchmark snapshot fetch failed (non-critical): {e}")

        db.log_snapshot(
            total_value=float(account.portfolio_value),
            cash=float(account.cash),
            positions={
                p.symbol: {
                    "qty":              float(p.qty),
                    "market_value":     float(p.market_value),
                    "avg_entry_price":  float(p.avg_entry_price),
                    "unrealized_pl":    float(p.unrealized_pl),
                }
                for p in positions
            },
            benchmark=benchmark or None,
        )
    except Exception as e:
        log.error(f"take_portfolio_snapshot failed: {e}")
        raise self.retry(exc=e, countdown=60)


# ── New tasks: safety, earnings, outcomes, pruning ────────────────────────────

@app.task(name="tasks.auto_expire_approvals", bind=True, max_retries=2)
def auto_expire_approvals(self):
    """Auto-expire pending approvals that are past their expiry time."""
    try:
        import db
        import alerts
        count = db.expire_stale_approvals()
        if count:
            log.info(f"Auto-expired {count} stale approval(s)")
            alerts._send(f"⏰ {count} recommendation(s) expired without a decision.")
        stuck = db.sweep_stuck_processing()
        if stuck:
            # The order may have gone out before the status write was lost.
            names = ", ".join(f"#{i} {a} {t}" for i, t, a in stuck)
            log.error(f"Swept {len(stuck)} approval(s) stuck in PROCESSING: {names}")
            alerts._send(f"⚠️ STUCK APPROVAL(S) CLEARED\n{names} sat in PROCESSING over "
                         f"{db.STUCK_PROCESSING_MINUTES} min and are now FAILED. Check TWS for a fill "
                         f"on each: one may have executed. They won't be retried automatically "
                         f"(a scheduled retry is skipped); re-approve if still wanted.")
    except Exception as e:
        log.error(f"auto_expire_approvals failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.refresh_earnings_calendar", bind=True, max_retries=2)
def refresh_earnings_calendar(self):
    """Refresh earnings dates for all 75 watchlist tickers via yfinance."""
    try:
        from ingestion.earnings_calendar import refresh_all_earnings
        count = refresh_all_earnings()
        log.info(f"Earnings calendar refreshed: {count} tickers updated")
    except Exception as e:
        log.error(f"refresh_earnings_calendar failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.refresh_macro_calendar", bind=True, max_retries=2)
def refresh_macro_calendar(self):
    """Upsert the FOMC/CPI/jobs-report/election calendar (2026-08-30, user
    request) -- see ingestion/macro_calendar.py's module docstring for why
    this is a maintained table rather than a live feed."""
    try:
        from ingestion.macro_calendar import refresh_macro_calendar as _refresh
        count = _refresh()
        log.info(f"Macro calendar refreshed: {count} events upserted")
    except Exception as e:
        log.error(f"refresh_macro_calendar failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.refresh_sector_calendar", bind=True, max_retries=2)
def refresh_sector_calendar(self):
    """Derive sector-level earnings-cluster events from the already-
    refreshed per-ticker earnings calendar (2026-08-30, user request).
    Scheduled after refresh-earnings so it sees this week's fresh dates."""
    try:
        from ingestion.sector_calendar import refresh_sector_calendar as _refresh
        count = _refresh()
        log.info(f"Sector calendar refreshed: {count} cluster events upserted")
    except Exception as e:
        log.error(f"refresh_sector_calendar failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.alert_catalyst_conflicts", bind=True, max_retries=2)
def alert_catalyst_conflicts(self):
    """Ping Telegram for unresolved catalyst conflicts inside the next 10 days."""
    try:
        from datetime import date, timedelta
        import alerts
        from catalyst_service.storage import query_events

        today = date.today().isoformat()
        end = (date.today() + timedelta(days=10)).isoformat()
        conflicts = query_events(start=today, end=end, confidence="conflicting")
        for event in conflicts:
            alerts.send_catalyst_conflict(
                ticker=event["ticker"],
                event_type=event["event_type"],
                window_start=event.get("window_start"),
                window_end=event.get("window_end"),
                note=event.get("thesis_note"),
            )
        if conflicts:
            log.warning(f"Catalyst conflicts alerted: {len(conflicts)}")
    except Exception as e:
        log.error(f"alert_catalyst_conflicts failed: {e}")
        raise self.retry(exc=e, countdown=300)


IMPLAUSIBLE_READS_BEFORE_HALT = 3


def _log_would_halt_daily_loss(pv: float, now=None) -> None:
    """Log-only daily-loss halt (audit 2026-11-02 19:27 #10, roadmap 01(c)):
    Helios has no daily loss limit. On each plausible equity read in
    check_portfolio_drawdown, compare it with the first portfolio snapshot
    of the ET day with a positive value (a $0 cold-start read is skipped and
    counted, professor m266) and write one activity_feed row (stage risk_shadow,
    would_halt_daily_loss) the first time in the day the loss is past
    config.DAILY_LOSS_SHADOW_PCT (a placeholder; the threshold is Luke's).
    Never sets the halt flag. Never raises."""
    try:
        import math
        import db
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from config import DAILY_LOSS_SHADOW_PCT, TRADING_ENV
        from agents.analyst import shadow_versions
        if not math.isfinite(pv) or pv <= 0:
            return
        now = now or datetime.now(timezone.utc)
        et_now = now.astimezone(ZoneInfo("America/New_York"))
        day_start = et_now.replace(hour=0, minute=0, second=0, microsecond=0)
        first = db.get_first_snapshot_since(day_start)
        base = float(first["total_value"]) if first else 0.0
        if not math.isfinite(base) or base <= 0:
            return
        pnl_pct = (pv - base) / base
        if pnl_pct > -DAILY_LOSS_SHADOW_PCT:
            return
        key = f"would_halt_daily_loss_logged_{TRADING_ENV}"
        today = et_now.date().isoformat()
        if db.get_state(key, "") == today:
            return
        written = db.log_activity(
            "risk_shadow", "", f"would halt (daily loss): equity {pnl_pct:+.2%} vs the day's "
            f"first snapshot, shadow threshold -{DAILY_LOSS_SHADOW_PCT:.1%}", "neutral",
            {"would_halt_daily_loss": True, "day_pnl_pct": round(pnl_pct * 100, 3),
             "threshold_pct": round(DAILY_LOSS_SHADOW_PCT * 100, 3),
             "baseline_at": str(first.get("timestamp")), "et_date": today,
             "baseline_skipped_nonpositive": first.get("skipped_nonpositive"),
             **shadow_versions()})
        if written:      # as _log_would_reassess: a failed write retries next cycle
            db.set_state(key, today)
    except Exception as e:
        log.debug(f"would_halt_daily_loss log failed: {e}")


@app.task(name="tasks.check_portfolio_drawdown", bind=True, max_retries=2)
def check_portfolio_drawdown(self):
    """
    Circuit breaker: sets the persisted trading_halted flag when drawdown
    from the high-water mark exceeds MAX_DRAWDOWN_PCT, and maintains the HWM.
    """
    failed = will_retry = False
    try:
        import db
        import alerts
        from config import START_VALUE, MAX_DRAWDOWN_PCT, TRADING_ENV

        # Keys scoped by TRADING_ENV (high_water_mark_paper vs
        # high_water_mark_live, etc.) -- found 2026-08-14: an unscoped
        # "high_water_mark" held paper's ~$1.09M baseline, and reconnecting
        # live (~$2k) would hit the implausible-value guard below on EVERY
        # check, not just a transient bad read (what that guard is actually
        # for) -- silently disabling this exact circuit breaker for live.
        hwm_key    = f"high_water_mark_{TRADING_ENV}"
        halted_key = f"trading_halted_{TRADING_ENV}"
        reason_key = f"halt_reason_{TRADING_ENV}"
        breach_key = f"drawdown_breach_count_{TRADING_ENV}"
        implausible_key = f"drawdown_implausible_count_{TRADING_ENV}"
        rebase_key = f"hwm_rebase_candidate_{TRADING_ENV}"

        client = _get_client()
        pv     = float(client.get_account().portfolio_value)

        # Maintain the persisted high-water mark (seed from START_VALUE,
        # which is itself TRADING_ENV-derived — see config.py)
        hwm = float(db.get_state_strict(hwm_key) or START_VALUE)   # M142: no default on a failed read

        # Plausibility guard: a transient broker read of $0 (e.g. an IBKR
        # clientId collision or cold-morning connection race) previously
        # computed as a 100% drawdown and latched the halt flag until a
        # manual /resume — reject readings that can't be a real single-step
        # move instead of trusting them.
        if pv <= 0 or pv < 0.5 * hwm:
            log.warning(f"Implausible portfolio value ${pv:,.2f} (HWM ${hwm:,.2f}) — "
                       f"skipping drawdown check, not updating HWM")
            db.set_state(breach_key, "0")
            # A persistent "implausible" read is either a stale baseline (from
            # 2026-10-11 the live HWM was the $1M paper default and the breaker
            # was off) or a real collapse. It used to rebase the HWM to the
            # current value after 3 reads, so a real collapse would have
            # re-armed the breaker below it instead of halting (audit
            # A1031-0853-4). Now it fails closed: halt buys, alert, and keep
            # the value as a candidate that /rebase applies after a human
            # looks.
            implausible = int(db.get_state_strict(implausible_key, "0") or 0) + 1 if pv > 0 else 0
            db.set_state(implausible_key, str(implausible))
            if implausible >= IMPLAUSIBLE_READS_BEFORE_HALT:
                db.set_state(rebase_key, str(pv))
                # Plain read: a failed one returns "false" and sets the halt (fails closed).
                if db.get_state(halted_key, "false") != "true":
                    from config import IMPLAUSIBLE_HALT_REASON_PREFIX
                    not_saved = db.set_state_verified([
                        (halted_key, "true"),
                        (reason_key, f"{IMPLAUSIBLE_HALT_REASON_PREFIX} on {implausible} checks; "
                                     f"/rebase after checking")])
                    # Logged and alerted either way, marked when the flag couldn't be
                    # confirmed saved (M149): a retry would see it set and stay silent.
                    halt_saved = halted_key not in not_saved
                    db.log_halt_event("auto_halt", "drawdown_task", "implausible reads",
                                      implausible_reads=implausible, verified=halt_saved)
                    log.critical(f"Drawdown check: {implausible} consecutive implausible reads "
                                 f"(value under half of {hwm_key}) — trading HALTED")
                    try:
                        alerts._send(
                            f"🛑 Trading HALTED ({TRADING_ENV}): account value read under half the "
                            f"high-water mark on {implausible} checks in a row.\n"
                            f"Either the baseline is stale or the account really fell. Check TWS. "
                            f"If the value is right and the baseline is stale, send /rebase to "
                            f"make it the new high-water mark and resume. /resume clears the flag "
                            f"without touching the baseline, but each buy still re-checks and is "
                            f"blocked while the value reads under half of it, and the halt returns "
                            f"after 3 more low reads (about 90 min)."
                            + ("" if halt_saved else "\n⚠️ The halt flag could not be confirmed "
                               "saved; check /status and send /halt if it isn't set.")
                        )
                    except Exception:
                        pass
            return
        db.set_state(implausible_key, "0")
        if db.get_state(rebase_key, ""):
            db.set_state(rebase_key, "")   # a plausible read: no reset pending
        _log_would_halt_daily_loss(pv)     # log-only shadow; never halts

        if pv > hwm:
            db.set_state(hwm_key, str(pv))
            hwm = pv

        drawdown = (hwm - pv) / hwm if hwm > 0 else 0

        if drawdown >= MAX_DRAWDOWN_PCT:
            # Require 2 consecutive breaching reads (this task runs every 30
            # min) before halting — a single noisy tick shouldn't be enough,
            # only a sustained drawdown.
            breach_count = int(db.get_state_strict(breach_key, "0") or 0) + 1
            db.set_state(breach_key, str(breach_count))
            if breach_count < 2:
                log.warning(f"Drawdown {drawdown:.1%} breach #1 — waiting for confirmation "
                           f"before halting")
                return
            already_halted = db.get_state(halted_key, "false") == "true"
            not_saved = db.set_state_verified([
                (halted_key, "true"),
                (reason_key, f"Auto: drawdown {drawdown:.1%} from HWM ${hwm:,.2f}")])
            halt_saved = halted_key not in not_saved
            log.critical(f"DRAWDOWN LIMIT HIT: {drawdown:.1%} — trading HALTED"
                         + ("" if halt_saved else " (flag not confirmed saved)"))
            if not already_halted:
                db.log_halt_event("auto_halt", "drawdown_task", "drawdown",
                                  drawdown_pct=round(drawdown, 4), verified=halt_saved)
                alerts.send_halt_alert(pv, hwm)
                if not halt_saved:
                    alerts._send("⚠️ The drawdown halt flag could not be confirmed saved; check "
                                 "/status and send /halt if it isn't set.")
        else:
            db.set_state(breach_key, "0")
            log.debug(f"Drawdown: {drawdown:.1%} (limit: {MAX_DRAWDOWN_PCT:.0%})")
    except Exception as e:
        failed = True
        # Only a run with no retry left counts as a failed check (M146).
        will_retry = self.request.retries < self.max_retries
        log.error(f"check_portfolio_drawdown failed: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        if not (failed and will_retry):
            _monitor_health("portfolio_drawdown", failed)


# Stops that refused a cancel from _fix_overstop (a cross-client stop shows
# PendingCancel for ~3 min, then reverts: MRVL 08-04, ibkr.cancel_order).
# In system_state, not memory: reconcile runs in other processes. Kept for
# _CANCEL_REFUSED_TTL_S after the last write.
_CANCEL_REFUSED_KEY = "stop_cancel_refused_{}"
_CANCEL_REFUSED_TTL_S = 24 * 3600


def _cancel_refused_ids(ticker: str) -> set:
    """Identities (risk_manager._stop_identity) of the ticker's stops that
    refused an earlier cancel; empty when none, expired or unreadable."""
    import db
    from datetime import datetime, timezone
    try:
        raw = db.get_state(_CANCEL_REFUSED_KEY.format(ticker))
        if not raw:
            return set()
        rec = json.loads(raw)
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(rec["at"])).total_seconds()
        if age > _CANCEL_REFUSED_TTL_S:
            return set()
        return {(s["kind"], s["id"]) for s in rec.get("stops", []) if s.get("kind") == "perm"}
    except Exception as e:
        log.warning(f"[{ticker}] refused-cancel record unreadable ({e}); treated as none")
        return set()


def _remember_cancel_refused(ticker: str, refused: list, present: list) -> None:
    """Add stops that did not cancel to the ticker's refused-cancel record,
    keeping earlier entries still among the open orders (present). Only
    stops with a permId: an orderId isn't stable across clientIds and
    sessions, so one without is not recorded and the next fix treats it as
    before (cancelled with the rest; cover only while working). Never
    raises: without the record the next fix cancels everything again, as
    before."""
    import db
    import risk_manager
    from datetime import datetime, timezone
    try:
        keep = _cancel_refused_ids(ticker) & {risk_manager._stop_identity(o) for o in present}
        stops = [{"kind": k, "id": i} for k, i in sorted(keep, key=str)]
        for o in refused:
            kind, oid = risk_manager._stop_identity(o)
            if kind == "perm" and (kind, oid) not in keep:
                keep.add((kind, oid))
                stops.append({"kind": kind, "id": oid})
        db.set_state(_CANCEL_REFUSED_KEY.format(ticker), json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(), "stops": stops}))
    except Exception as e:
        log.warning(f"[{ticker}] could not record stops that refused a cancel: {e}")


# Phrases in _fix_overstop lines about a cancel that has not gone through;
# they repeat while a stop refuses, so their alerts are throttled per ticker.
# "won't cancel and exceed" (stops over the position: a short if they fill)
# is not among them and is never throttled.
_CANCEL_REFUSED_PHRASES = ("refused a cancel",)
_CANCEL_NOTE_PHRASES = ("still being cancelled",) + _CANCEL_REFUSED_PHRASES


def _stop_fix_alert_due(ticker: str, msg: str) -> bool:
    """Whether to send a _fix_overstop line as an alert. A line reporting a
    placed stop is always sent, and so is a 🚨 about stops that won't cancel
    and exceed the position. A 🚨 about recorded stops that refused a cancel
    and cover the position has its own throttle key (stop_cancel_refused_<T>),
    so no informational line can hold it back; other lines about a cancel
    still in flight or settled around a refused stop share
    stop_cancelling_<T>. Everything else is sent. An unreadable throttle
    sends."""
    if msg.startswith("🔧") and "placed a stop" in msg:
        return True
    if msg.startswith("🚨"):
        if not any(p in msg for p in _CANCEL_REFUSED_PHRASES):
            return True
        key = f"stop_cancel_refused_{ticker}"
    elif any(p in msg for p in _CANCEL_NOTE_PHRASES):
        key = f"stop_cancelling_{ticker}"
    else:
        return True
    try:
        return _reconcile_alert_due(key)
    except Exception as e:
        log.warning(f"[{ticker}] cancel alert throttle unreadable ({e}); sending")
        return True


def _fix_refused_over_position(ticker: str, mine: list, refused, held: float,
                               stop_price: float, last: float, recheck_try: int = 1) -> str:
    """_fix_overstop when recorded stops that refused a cancel are, while
    working, bigger than the position (a partial sale since, say): if they
    fill, the account goes short. Their cancel is retried once per fix; our
    stops stay unless that cancel lands, so the position is never left to
    the refused stops alone. The alert is never throttled."""
    import db
    import execution
    cancelling_now = execution._CANCELLING_STATUSES
    over = [o for o in mine if refused(o) and o.get("status") not in cancelling_now]
    over_qty = sum(float(o["qty"]) for o in over)
    for o in over:
        try:
            manage_cancel_order(int(o["order_id"]), symbol=ticker, perm_id=o.get("perm_id"))
        except Exception as e:
            log.warning(f"[{ticker}] cancel retry of refused stop {o['order_id']} failed: {e}")
    stuck = _resting_sell_stops(ticker)
    for _ in range(_CANCEL_SETTLE_LOOKS):
        if not any(refused(o) and o.get("status") in cancelling_now for o in stuck):
            break
        import time
        time.sleep(5)
        stuck = _resting_sell_stops(ticker)
    if any(refused(o) for o in stuck):
        ids = ", ".join(str(o["order_id"]) for o in over)
        return (f"🚨 {ticker}: stop order(s) {ids} for {over_qty:g} shares won't cancel and "
                f"exceed the {held:g} held (a short if they fill); cancel retried, our stop "
                f"left in place — cancel them in TWS now")
    ours = sum(float(o["qty"]) for o in stuck if o.get("status") not in cancelling_now)
    need = round(held - ours, 4)
    if need > 0:
        result = manage_stop_place(ticker, need, float(stop_price), market_price=last)
        db.set_stop_broker_order_id(ticker, result["order_id"])
        return (f"🔧 {ticker}: the refused stop(s) over the {held:g} held cancelled on retry; "
                f"placed a stop for {need:g} @ ${float(stop_price):.2f}")
    if ours > held + 1e-6:
        # Ours alone are now over the position: the re-check trims them.
        queued = _queue_stop_coverage_recheck(ticker, recheck_try)
        return (f"⚠️ {ticker}: the refused stop(s) over the {held:g} held cancelled on retry; "
                f"our stops for {ours:g} shares are over-stopped against the {held:g} held"
                + (", re-check queued" if queued else ", reconcile re-checks"))
    return (f"🔧 {ticker}: the refused stop(s) over the {held:g} held cancelled on retry; "
            f"our stops cover {ours:g}")


@locked_by_ticker(lambda ticker, *a, **k: ticker,
                  on_busy=lambda: "over-stop fix skipped: ticker busy (next cycle)")
def _fix_overstop(ticker: str, fresh_qty: float | None = None, allow_cancel: bool = True,
                  max_qty: float | None = None, recheck_try: int = 1,
                  no_cancel_reason: str | None = None) -> str:
    """Re-read under the lock and make resting SELL stops cover exactly the
    position at the recorded stop. Over-covered: cancel, re-read what is
    still resting (a cross-client stop may refuse to cancel, A2253-6) and
    place stops only for what is left uncovered. Under-covered (A2253-4),
    including no stop at all (A0053-2): add a stop for the uncovered shares.
    Never places a stop at or above the last price, which would sell at
    once. Returns a line for log/alert.

    fresh_qty: the signed quantity a fresh positions read showed for a ticker
    the portfolio cache reads flat (reconcile's `disagree`). It stands in for
    the position only while the cache still shows none, and then this never
    cancels: it only adds a stop for uncovered shares. allow_cancel=False
    (reconcile while the position reads disagree) likewise never cancels or
    shrinks: an over-cover is only reported.

    max_qty: the smaller of the two position reads while they disagree.
    Stops are sized against min(position qty, max_qty), so they never exceed
    the smaller read; resting stops already at or above it get a report and
    nothing is added or cancelled.

    A stop still being cancelled (execution._CANCELLING_STATUSES) is never
    counted as cover when sizing the stop placed: once the cancel lands it
    would leave the position bare. If one is still cancelling, a
    recheck_stop_coverage is queued (recheck_try: which one this is) for the
    over-stop it leaves if the broker rejects the cancel.

    Stops that did not cancel are recorded (_remember_cancel_refused). A
    later fix doesn't cancel them again: it cancels only the other stops, by
    id, and sizes ours to the shares the refused stops leave (none, with a
    🚨, when they cover the position alone), so coverage settles instead of
    every fix cancelling our stop and re-sending the doomed cancel."""
    import db
    import execution
    import ibkr
    try:
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        from_fresh = pos is None and fresh_qty is not None
        signed = float(pos.qty) if pos else (float(fresh_qty) if from_fresh else 0.0)
        held = round(abs(signed), 4)
        open_orders = ibkr.get_open_orders()
        mine = [o for o in _working_sell_stops(open_orders) if o["ticker"] == ticker]
        resting = sum(float(o["qty"]) for o in mine)
        cancelling_qty = sum(float(o["qty"]) for o in mine
                             if o.get("status") in execution._CANCELLING_STATUSES)
        # Never size stops past the smaller position read.
        capped = max_qty is not None and float(max_qty) < held - 1e-6
        if capped:
            held = round(max(float(max_qty), 0.0), 4)
        # Shares a working market SELL is already selling need no stop
        # (professor M129).
        if signed > 0:
            held = max(round(held - _working_mkt_sell_qty(open_orders, ticker), 4), 0.0)
    except Exception as e:
        return f"⚠️ {ticker}: stop-coverage re-check failed ({e}); check TWS"
    if signed < 0:
        # A SELL stop on a short adds to it (professor M63).
        return f"🚨 {ticker}: broker shows a SHORT of {signed:g}; no stop placed — check TWS now"
    if capped and resting >= held - 1e-6:
        how = "exceed" if resting > held + 1e-6 else "already match"
        return (f"🚨 {ticker}: resting stops for {resting:g} shares {how} the smaller position "
                f"read ({held:g}) while the reads disagree; nothing added or cancelled — "
                f"check the position and its stops in TWS")
    if abs(resting - held) <= 1e-6:
        return f"{ticker}: stop coverage fine on re-check ({resting:g} vs {held:g})"
    stop_price = (db.get_position_stop(ticker) or {}).get("stop_price")
    if not stop_price or held <= 0:
        return (f"🚨 {ticker}: resting stops for {resting:g} shares but {held:g} held and no "
                f"recorded stop to rebuild from — fix the stops in TWS")
    try:
        last = float(ibkr.get_latest_price(ticker) or 0)
    except Exception:
        last = 0.0
    if not last or float(stop_price) >= last:
        return (f"🚨 {ticker}: stops cover {resting:g} of {held:g} shares; recorded stop "
                f"${float(stop_price):.2f} is not below the last price "
                f"({'unknown' if not last else f'${last:.2f}'}), so nothing was placed — "
                f"set the stop in TWS")
    label = (f"stops covered {resting:g} shares but {held:g} held" if resting > held
             else f"stops covered only {resting:g} of {held:g} shares")
    if no_cancel_reason and resting > held:
        # Not a disagreement: the cross-check couldn't be made. Reconcile
        # checks it with its own reads; no alert from here.
        return (f"{ticker}: {label}; {no_cancel_reason}, so nothing cancelled — "
                f"reconcile checks it")
    if (from_fresh or not allow_cancel or max_qty is not None) and resting > held:
        return (f"🚨 {ticker}: {label} while the position reads disagree; "
                f"nothing cancelled — check the stops in TWS")
    try:
        if resting > held:
            import risk_manager
            cancelling_now = execution._CANCELLING_STATUSES
            refused_ids = _cancel_refused_ids(ticker)

            def refused(o):
                return risk_manager._stop_identity(o) in refused_ids
            # Refused stops count as cover only while working: one still
            # showing the cancel may yet land.
            refused_cover = round(sum(float(o["qty"]) for o in mine
                                      if refused(o) and o.get("status") not in cancelling_now), 4)
            if refused_ids and refused_cover > held + 1e-6:
                return _fix_refused_over_position(ticker, mine, refused, held, stop_price, last,
                                                  recheck_try)
            target = round(max(held - refused_cover, 0.0), 4)
            others = [o for o in mine if not refused(o) and o.get("status") not in cancelling_now]
            sent = False
            if not any(refused(o) for o in mine):
                manage_cancel_stop_orders_for(ticker)
                sent = True
            elif sum(float(o["qty"]) for o in others) > target + 1e-6:
                # Only the stops that didn't refuse: the refused ones stay.
                for o in others:
                    manage_cancel_order(int(o["order_id"]), symbol=ticker, perm_id=o.get("perm_id"))
                sent = True
            stuck = _resting_sell_stops(ticker) if sent else list(mine)
            # A slow cancel often lands within seconds; look again (up to
            # _CANCEL_SETTLE_LOOKS x 5 s) before sizing (professor note on
            # A0253-1).
            for _ in range(_CANCEL_SETTLE_LOOKS if sent else 0):
                if not any(o.get("status") in cancelling_now for o in stuck if not refused(o)):
                    break
                import time
                time.sleep(5)
                stuck = _resting_sell_stops(ticker)
            rest = [o for o in stuck if not refused(o)]
            working = [o for o in rest if o.get("status") not in cancelling_now]
            stuck_qty = sum(float(o["qty"]) for o in working)
            cancelling_qty = round(sum(float(o["qty"]) for o in rest) - stuck_qty, 4)
            if sent and (working or cancelling_qty > 0):
                _remember_cancel_refused(ticker, rest, stuck)
            if refused_ids and refused_cover >= held - 1e-6:
                ids = ", ".join(str(o["order_id"]) for o in stuck if refused(o))
                return (f"🚨 {ticker}: stop order(s) {ids} for {refused_cover:g} shares refused a "
                        f"cancel (won't cancel) and cover the {held:g} held; nothing placed — "
                        f"cancel them in TWS if they aren't wanted")
            if stuck_qty > target + 1e-6:
                ids = ", ".join(str(o["order_id"]) for o in working)
                return (f"🚨 {ticker}: stop order(s) {ids} for {stuck_qty:g} shares won't cancel "
                        f"and exceed the {target:g} held — cancel them in TWS")
            need = round(target - stuck_qty, 4)
            if not sent and need <= 0:
                still = round(sum(float(o["qty"]) for o in mine if refused(o)) - refused_cover, 4)
                return (f"{ticker}: stop coverage settled around stop(s) that refused a cancel "
                        f"({refused_cover:g} working, {still:g} still showing the cancel, "
                        f"{stuck_qty:g} ours vs {held:g} held); nothing cancelled")
        else:
            need = round(held - (resting - cancelling_qty), 4)
        if need > 0:
            result = manage_stop_place(ticker, need, float(stop_price), market_price=last)
            db.set_stop_broker_order_id(ticker, result["order_id"])
    except Exception as e:
        return f"🚨 {ticker}: {label}; fix failed ({e}) — check TWS now"
    pending = ""
    if cancelling_qty > 0:
        queued = _queue_stop_coverage_recheck(ticker, recheck_try)
        pending = (f"; {cancelling_qty:g} shares of stops still being cancelled"
                   + (", re-check queued" if queued else ", reconcile re-checks"))
    if need <= 0:
        return (f"⚠️ {ticker}: {label}; stop(s) that did not cancel cover the {held:g} held, "
                f"nothing placed{pending} — check the stops in TWS")
    return f"🔧 {ticker}: {label}; placed a stop for {need:g} @ ${float(stop_price):.2f}{pending}"


def _fix_overstop_safe(ticker: str, **kwargs) -> str:
    """_fix_overstop for the reconcile loops: one ticker's failure must not
    skip the remaining tickers and section 3 (audit 02:53, A0253-7)."""
    try:
        return _fix_overstop(ticker, **kwargs)
    except Exception as e:
        log.error(f"[{ticker}] stop coverage fix crashed: {e}")
        return f"🚨 {ticker}: stop coverage fix failed ({e}) — check TWS"


def _recheck_position_reads(ticker: str) -> dict | None:
    """_fix_overstop arguments for recheck_stop_coverage, from the same two
    reads reconcile compares: the portfolio cache _fix_overstop sizes from
    and a fresh reqPositions() read. If they disagree (an ADD the cache
    hasn't caught up with), the fix only adds and never cancels, sized to
    the smaller read, as in reconcile. A fresh read that fails or doesn't
    show the ticker (reconcile treats an empty read as a possible
    reconnect gap) is no cross-check: add-only, unsized, and an over-cover
    is left to reconcile without an alert. A non-empty fresh read without
    the ticker may be a real flat (zero positions are left out of it):
    None, and the re-check does nothing. Reads that disagree on the sign
    get max_qty 0, as reconcile's _smaller_read: no SELL stop is added."""
    import ibkr
    try:
        fresh_all = ibkr.get_fresh_stock_positions()
    except Exception as e:
        log.warning(f"[{ticker}] stop coverage re-check: fresh position read failed ({e}); "
                    f"no cancels this time")
        return {"allow_cancel": False, "no_cancel_reason": "the fresh position read failed"}
    if fresh_all and not fresh_all.get(ticker):
        log.warning(f"[{ticker}] stop coverage re-check: the fresh position read shows other "
                    f"positions but not {ticker} (flat?); nothing done, reconcile checks it")
        return None
    if not fresh_all.get(ticker):
        log.warning(f"[{ticker}] stop coverage re-check: the fresh position read is empty "
                    f"(a reconnect gap?); no cancels this time")
        return {"allow_cancel": False,
                "no_cancel_reason": f"the fresh position read doesn't show {ticker}"}
    fresh = float(fresh_all[ticker])
    try:
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        cache = float(pos.qty) if pos else 0.0
    except Exception as e:
        log.warning(f"[{ticker}] stop coverage re-check: position read failed ({e}); "
                    f"no cancels this time")
        return {"allow_cancel": False, "no_cancel_reason": "the position read failed"}
    if abs(fresh - cache) <= 1e-6:
        return {}
    log.warning(f"[{ticker}] stop coverage re-check: position reads disagree (cache {cache:g}, "
                f"fresh {fresh:g}); adding only, sized to the smaller")
    if cache * fresh < 0:
        log.error(f"[{ticker}] stop coverage re-check: position reads disagree on sign (cache "
                  f"{cache:g}, fresh {fresh:g}); no SELL stop added")
        smaller = 0.0
    else:
        smaller = abs(fresh) if not cache else min(abs(fresh), abs(cache))
    kw = {"allow_cancel": False, "max_qty": smaller}
    if not cache:
        kw["fresh_qty"] = fresh
    return kw


# Re-checks after a restore placed over a stop still being cancelled: one
# per minute while that cancel is pending, then reconcile takes over.
_COVERAGE_RECHECK_TRIES = 6   # last look ~360 s: past a ~180 s cross-client revert
# 5 s looks _fix_overstop gives its own cancel before sizing the stop.
_CANCEL_SETTLE_LOOKS = 3


def _queue_stop_coverage_recheck(ticker: str, attempt: int = 1) -> bool:
    """Queue recheck_stop_coverage in 60 s, unless attempt is past
    _COVERAGE_RECHECK_TRIES. Never raises: on a failure the position is left
    to reconcile. True when queued."""
    if attempt > _COVERAGE_RECHECK_TRIES:
        log.warning(f"[{ticker}] stop coverage re-check: {attempt - 1} tries done; "
                    f"reconcile checks the coverage")
        return False
    try:
        recheck_stop_coverage.apply_async(args=[ticker], kwargs={"attempt": attempt}, countdown=60)
        return True
    except Exception as e:
        log.error(f"[{ticker}] could not queue the stop coverage re-check ({e}); "
                  f"reconcile checks the coverage")
        return False


@app.task(name="tasks.recheck_stop_coverage", max_retries=0)
def recheck_stop_coverage(ticker: str, attempt: int = 1):
    """Queued ~60 s after a stop restore that went on while another stop
    for ticker was still being cancelled (execution._queue_coverage_recheck).
    If that cancel was rejected, both stops rest and _fix_overstop trims
    them back to the position. While a cancel is still pending it waits
    another minute instead: _fix_overstop would count the cancelling stop
    as cover and could leave the position bare once it lands. After
    _COVERAGE_RECHECK_TRIES the 15-min reconcile handles it.

    The pending check and the fix run under one ticker lock, so no other
    stop change gets in between (the lock is re-entrant: _fix_overstop's
    own nests inside it)."""
    import execution
    try:
        with ticker_lock(ticker):
            try:
                pending = any(o.get("status") in execution._CANCELLING_STATUSES
                              for o in _resting_sell_stops(ticker))
            except Exception as e:
                log.warning(f"[{ticker}] stop coverage re-check: stops unreadable ({e}); "
                            f"reconcile checks it")
                return
            if pending:
                if _queue_stop_coverage_recheck(ticker, attempt + 1):
                    log.info(f"[{ticker}] stop coverage re-check: a cancel is still pending; "
                             f"again in 60 s")
                return
            reads = _recheck_position_reads(ticker)
            if reads is None:
                # Maybe flat, maybe a reconnect gap that clears within a
                # minute: look again (the chain stays capped).
                _queue_stop_coverage_recheck(ticker, attempt + 1)
                return
            msg = _fix_overstop_safe(ticker, recheck_try=attempt + 1, **reads)
    except stop_lock.TickerBusy:
        if _queue_stop_coverage_recheck(ticker, attempt + 1):
            log.info(f"[{ticker}] stop coverage re-check: ticker busy; again in 60 s")
        return
    log.info(f"[{ticker}] stop coverage re-check after a restore: {msg}")
    if msg.startswith(("🔧", "🚨", "⚠️")) and _stop_fix_alert_due(ticker, msg):
        try:
            import alerts
            alerts._send(f"STOP RECONCILE — {msg}")
        except Exception:
            pass


def _reconcile_alert_due(issue_key: str) -> bool:
    """Gate re-alerting the same unresolved reconciliation issue. A standing
    condition (e.g. one position missing a stop) used to re-fire every
    15-minute reconcile cycle — ~96 identical Telegram messages/day. Only
    the first sighting of an issue_key, or one after the throttle window
    has elapsed, is "due"; touching the state also resets the window."""
    import db
    from datetime import datetime, timezone
    from config import RECONCILE_ALERT_THROTTLE_HOURS
    state_key = f"reconcile_alert_{issue_key}"
    last = db.get_state(state_key)
    now = datetime.now(timezone.utc)
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < \
                    RECONCILE_ALERT_THROTTLE_HOURS * 3600:
                return False
        except ValueError:
            pass
    db.set_state(state_key, now.isoformat())
    return True


# Reconcile's multi-cycle confirmations (the orphan and stale-record markers,
# the empty-read counter) count a cycle only this long after the last one, so
# the task's own Celery retries (+2/+4 min) can't stand in for later cycles.
_RECONCILE_MIN_CYCLE_S = 600


def _reconcile_now():
    """The clock reconcile's cycle markers use (a seam for tests)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _pending_marker_ready(key: str) -> bool:
    """Two-cycle marker: True once the marker was armed at least
    _RECONCILE_MIN_CYCLE_S ago. Otherwise arms it (when missing or
    unreadable; a young marker keeps its time) and returns False."""
    import db
    from datetime import datetime
    now = _reconcile_now()
    raw = db.get_state(key)
    if raw:
        try:
            return (now - datetime.fromisoformat(raw)).total_seconds() >= _RECONCILE_MIN_CYCLE_S
        except (TypeError, ValueError):
            pass
    db.set_state(key, now.isoformat())
    return False


def _reconcile_cycle_alert_due(issue_key: str) -> bool:
    """Once-per-reconcile-cycle gate for an alert that must go out on every
    cycle while its condition holds: True unless the same key fired less
    than _RECONCILE_MIN_CYCLE_S ago (the task's own Celery retries). An
    unreadable or unsavable marker counts as due, so the alert still goes."""
    import db
    from datetime import datetime
    state_key = f"reconcile_cycle_alert_{issue_key}"
    now = _reconcile_now()
    try:
        raw = db.get_state(state_key)
        if raw and (now - datetime.fromisoformat(raw)).total_seconds() < _RECONCILE_MIN_CYCLE_S:
            return False
    except Exception:
        pass
    try:
        db.set_state(state_key, now.isoformat())
    except Exception:
        pass
    return True


# An empty fresh positions read while stop records exist confirms a ticker
# flat only on this many consecutive reconcile cycles with the cache empty too.
_EMPTY_READ_CONFIRM_CYCLES = 3
_EMPTY_READ_KEY = "flat_empty_reads_{ticker}"   # "<count>|<iso time of the last count>"
_EMPTY_READ_GAP_S = 2400                         # a longer gap restarts the count
_empty_read_store_alert_at = 0.0                 # in-process: the store itself may be down


def _count_empty_flat_reads(tickers) -> tuple[set[str], list[str], list[str]] | None:
    """Count one more cycle with both position reads empty for each ticker.
    Returns (tickers still waiting, tickers confirmed flat this cycle, tickers
    seen for the first time), or None when the counter store can't be read
    or written; then nothing is confirmed, with an alert at most hourly."""
    import db
    import time as _time
    from datetime import datetime
    global _empty_read_store_alert_at
    now = _reconcile_now()
    waiting: set[str] = set()
    confirmed: list[str] = []
    first: list[str] = []
    try:
        for ticker in sorted(tickers):
            key = _EMPTY_READ_KEY.format(ticker=ticker)
            raw = db.get_state_strict(key, "") or ""
            n, age = 0, None
            if raw:
                count, _, at = raw.partition("|")
                try:
                    age = (now - datetime.fromisoformat(at)).total_seconds()
                    if age <= _EMPTY_READ_GAP_S:
                        n = int(count)
                except (TypeError, ValueError):
                    n, age = 0, None
            if n and age is not None and age < _RECONCILE_MIN_CYCLE_S:
                counted = False   # too soon after the last count (a retry): row left as is
            else:
                n += 1
                counted = True
                if db.set_state_verified([(key, f"{n}|{now.isoformat()}")]):
                    raise RuntimeError(f"{key} did not save")
            if counted and n == 1:
                first.append(ticker)
            if n < _EMPTY_READ_CONFIRM_CYCLES:
                waiting.add(ticker)
            elif counted and n == _EMPTY_READ_CONFIRM_CYCLES:
                confirmed.append(ticker)
    except Exception as e:
        log.error(f"reconcile_stop_orders: empty-read counter unavailable ({e}); nothing is "
                  f"confirmed flat this cycle")
        if _time.monotonic() - _empty_read_store_alert_at >= 3600 or not _empty_read_store_alert_at:
            _empty_read_store_alert_at = _time.monotonic()
            try:
                import alerts
                alerts._send(f"⚠️ STOP CHECK: the broker shows no positions but stop records exist "
                             f"for {', '.join(sorted(tickers))}, and the confirmation counter can't "
                             f"be read or saved ({e}). Nothing is cancelled or deleted until it can.")
            except Exception as e2:
                log.error(f"reconcile_stop_orders: could not send the counter alert: {e2}")
        return None
    return waiting, confirmed, first


def _clear_empty_flat_reads(tickers) -> None:
    import db
    for ticker in tickers:
        try:
            db.set_state(_EMPTY_READ_KEY.format(ticker=ticker), "")
        except Exception as e:
            log.warning(f"[{ticker}] could not reset the empty-read counter: {e}")


_MISSING_HELD_KEY = "reconcile_fresh_missing_held_since"
_MISSING_HELD_ALERT_S = 1800   # ~2 cycles of the block before it goes to Telegram


def _note_missing_held(missing: list[str], blocked, flat_stopped=(), empty_fresh: bool = False) -> None:
    """Track how long the confirming read has kept leaving out held tickers.
    Past _MISSING_HELD_ALERT_S it alerts (throttled): the cache may show a
    position the broker no longer has, and while it does the other tickers'
    orphan stops can't be cleaned up (one could open a short). Clears when
    the read is complete again. While the hold is on, each flat_stopped
    ticker (flat in both reads, still with a working SELL stop) alerts every
    cycle and gets a manual action (professor M221). empty_fresh: the fresh
    read came back empty (a probable reconnect gap), so "flat" rests on the
    cache alone; the alert then says to confirm in TWS before cancelling
    (professor m304). Never raises."""
    import db
    from datetime import datetime
    try:
        if not missing:
            if db.get_state(_MISSING_HELD_KEY, ""):
                db.set_state(_MISSING_HELD_KEY, "")
            return
        now = _reconcile_now()
        raw = db.get_state(_MISSING_HELD_KEY, "") or ""
        since = last = None
        try:
            if raw:
                a, _, b = raw.partition("|")
                since = datetime.fromisoformat(a)
                last = datetime.fromisoformat(b) if b else since
        except ValueError:
            since = last = None
        # A gap since the last incomplete read (cycles with nothing looking
        # flat skip the check and can't clear the marker) starts a new episode.
        if since is None or (now - last).total_seconds() > 2 * _RECONCILE_MIN_CYCLE_S + 300:
            since = now
        db.set_state(_MISSING_HELD_KEY, f"{since.isoformat()}|{now.isoformat()}")
        _alert_held_flat_stops(flat_stopped, empty_fresh=empty_fresh)
        if (now - since).total_seconds() >= _MISSING_HELD_ALERT_S \
                and _reconcile_alert_due("fresh_read_missing_held"):
            import alerts
            alerts._send(f"🚨 STOP CHECK: the portfolio cache shows {', '.join(missing)} but fresh "
                         f"position reads have left it out since {since:%H:%M}. Orphan-stop "
                         f"cleanup for {', '.join(sorted(blocked))} is on hold until the reads "
                         f"agree; an orphan SELL stop on a flat ticker could open a short. "
                         f"Check positions and open orders in TWS.")
    except Exception as e:
        log.warning(f"reconcile_stop_orders: could not track the missing-held read: {e}")


def _touch_missing_held() -> None:
    """A cycle whose confirming read failed: an open missing-held episode
    stays open (its last-seen time moves to now, its start is kept), so a
    failed read between two incomplete ones can't restart it and hold back
    the 30-min alert (professor M222). Only a live episode is refreshed:
    with no marker, an unreadable one, or one whose last-seen time is past
    the episode gap, nothing changes. It never alerts: with the read failed,
    "flat" would rest on the cache alone. Never raises."""
    import db
    from datetime import datetime
    try:
        raw = db.get_state(_MISSING_HELD_KEY, "") or ""
        if not raw:
            return
        a, _, b = raw.partition("|")
        since = datetime.fromisoformat(a)
        last = datetime.fromisoformat(b) if b else since
        now = _reconcile_now()
        if (now - last).total_seconds() > 2 * _RECONCILE_MIN_CYCLE_S + 300:
            return   # stale: the next incomplete read starts a new episode
        db.set_state(_MISSING_HELD_KEY, f"{a}|{now.isoformat()}")
    except Exception as e:
        log.warning(f"reconcile_stop_orders: could not refresh the missing-held marker: {e}")


def _alert_held_flat_stops(tickers, empty_fresh: bool = False) -> None:
    """While the missing-held hold blocks orphan cancels: each ticker that
    reads flat but still has a working SELL stop alerts once per cycle and
    gets a manual action, since that stop could open a short. Nothing is
    cancelled here. empty_fresh: the fresh read came back empty, so only the
    cache (which can lag a held position) says flat; the text then asks for
    a TWS check before any cancel instead of saying both reads agree
    (professor m304; wording only, same cadence). Never raises."""
    import db
    for t in sorted(set(tickers or ())):
        try:
            if _reconcile_cycle_alert_due(f"hold_flat_stop_{t}"):
                import alerts
                if empty_fresh:
                    alerts._send(f"🚨 STOP CHECK — <b>{t}</b>\nReconcile is on hold: the fresh "
                                 f"positions read came back empty, which is usually a reconnect "
                                 f"gap, so {t} reads flat only in the portfolio cache, which can "
                                 f"lag. A SELL stop for {t} is still working. Do not cancel it on "
                                 f"this alert alone: first confirm in TWS that {t} has no "
                                 f"position. Only if it has none, cancel that stop, because it "
                                 f"could open a short. This repeats every stop check while the "
                                 f"hold lasts.")
                else:
                    alerts._send(f"🚨 STOP CHECK — <b>{t}</b>\nReconcile is on hold (position "
                                 f"reads disagree). {t} reads flat in both reads but a SELL stop is "
                                 f"still working. Check TWS: if {t} is really closed, cancel that "
                                 f"stop, because it could open a short. If {t} is still held, "
                                 f"ignore this. This repeats every stop check while the hold lasts.")
        except Exception as e:
            log.error(f"[{t}] could not send the held-flat-stop alert: {e}")
        try:
            try:
                already = t in json.loads(db.get_state(_MANUAL_ACTIONS_KEY) or "{}")
            except Exception:
                already = False
            if not already:   # re-opening would reset the hourly STILL OPEN clock
                text = ("reconcile is on hold (the fresh positions read came back empty, likely "
                        "a reconnect gap) and it reads flat only in the portfolio cache, with a "
                        "SELL stop still working; confirm in TWS that it has no position before "
                        "cancelling that stop (it could open a short), and leave the stop if it "
                        "is still held" if empty_fresh else
                        "reconcile is on hold (position reads disagree) and it "
                        "reads flat in both reads but a SELL stop is still "
                        "working; if it is really closed, cancel that stop (it "
                        "could open a short), if it is still held, ignore this")
                open_manual_action(t, text, clear_on_flat=False, kind=_HOLD_FLAT_STOP_KIND)
        except Exception as e:
            log.error(f"[{t}] could not open the held-flat-stop manual action: {e}")


_ACCOUNT_FLAT_HEALTHY_S = 1800        # no connectivity error or reconnect for this long
_empty_read_gate_alert_at = 0.0       # in-process hourly throttle for the gate alert


def _account_flat_gate() -> tuple[bool, str]:
    """The server's account-level view must agree before an empty-read flat
    is confirmed: stock value 0 (GrossPositionValue, else StockMarketValue)
    and a connection with no 1100/1101/2110 error or reconnect in the last
    30 min. (False, why) when either disagrees or can't be checked."""
    import ibkr
    try:
        value = float(ibkr.get_account_stock_value())
    except Exception as e:
        return False, f"the account-level position value could not be read ({e})"
    if abs(value) > 0.005:
        # No dollar amount: this text goes to the log and to Telegram.
        return False, "the account-level view shows non-zero account stock value"
    try:
        healthy = ibkr.connectivity_error_free_for(_ACCOUNT_FLAT_HEALTHY_S) is True
    except Exception as e:
        return False, f"the connection health could not be checked ({e})"
    if not healthy:
        return False, "the broker connection had a connectivity error or reconnect in the last 30 min"
    return True, ""


def _log_stop_slippage(ticker: str, stop_price, fill: dict | None, source: str,
                       stop_order_id=None, prev_close=None, trailing_active=None,
                       stop_method=None) -> bool:
    """Log-only record of how far a protective SELL stop filled from its stop
    price. After a gap a stop fills at market, below the stop, while the risk
    cap assumes the stop price is the fill. Writes one activity_feed row
    (stage stop_slippage) per stop fill, deduplicated on shadow_key
    ({env}:{ticker}:{stop order id}, else the fill time). No row when the
    record's stop price or the fill price is missing or not positive.

    stop_price, fill_price, shares and slippage_pct keep their original
    meaning on every row: the DB record's stop, and the volume-weighted price
    of ALL the symbol's unrecorded same-day SELLs (which can include a manual
    or backstop sale). Which reference a study uses is not chosen here.

    The filled order (professor M229), in added fields only:
    fill["by_order"] splits those executions by order. When exactly one of
    them is the record's stop order, or a SELL stop reconcile last saw
    resting for this ticker (_STOP_TRIGGER_CACHE_KEY), its own fill is
    matched_fill_price / matched_shares, and fill_match says how it was
    matched: stop_order_perm_id (a cached order whose orderId and permId
    agree) or stop_order_id (orderId only: one side has no permId). No match
    leaves those null, with fill_match symbol_vwap_unrecorded_sells,
    symbol_vwap_ambiguous_orders (more than one order matched) or
    symbol_vwap_perm_id_mismatch (the orderId was seen resting under a
    different permId: orderIds repeat across clientIds, so it is not ours).
    When the matched order was seen resting, broker_trigger_price is its
    last-seen trigger (trailStopPrice for a TRAIL, auxPrice otherwise),
    broker_trigger_seen_at says how old that reading is (reconcile samples
    it once a cycle, so a TRAIL may have ratcheted since), and
    slippage_vs_broker_trigger_pct compares matched_fill_price with it.
    stop_price_source is always db_record (it describes stop_price).

    prev_close: the caller's value, else the newest price_history close dated
    before the fill's ET date (prev_close_date); prev_close_null_reason says
    why it is null. Nothing is sent or changed. Never raises; returns whether
    a row was written."""
    try:
        import math
        from datetime import datetime, time as _time
        from zoneinfo import ZoneInfo
        import db
        from config import TRADING_ENV
        from agents.analyst import shadow_versions
        if not fill:
            return False

        def _pos(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) and f > 0 else None

        record_stop = _pos(stop_price)
        vwap = _pos(fill.get("fill_price"))
        if record_stop is None or vwap is None:
            return False
        _ny = ZoneInfo("America/New_York")
        fill_time = fill.get("fill_time")
        outside_rth = at_open = None
        if isinstance(fill_time, datetime) and fill_time.tzinfo is not None:
            et = fill_time.astimezone(_ny)
            t = et.time()
            outside_rth = et.weekday() >= 5 or not (_time(9, 30) <= t < _time(16, 0))
            at_open = et.weekday() < 5 and _time(9, 30) <= t < _time(9, 35)
        else:
            fill_time = None
        ident = stop_order_id if stop_order_id else (fill_time.isoformat() if fill_time else None)
        if ident is None:
            ident = datetime.now(_ny).date().isoformat()
        key = f"{TRADING_ENV}:{ticker}:{ident}"
        done = db.get_activity_shadow_keys("stop_slippage")
        if done is not None and key in done:
            return False
        shares_total = fill.get("filled_qty")
        order_ids = fill_is_stop = None
        try:
            if isinstance(fill.get("order_ids"), (list, tuple)):
                order_ids = sorted(int(i) for i in fill["order_ids"])
                if stop_order_id:
                    fill_is_stop = order_ids == [int(stop_order_id)]
        except (TypeError, ValueError):
            order_ids = fill_is_stop = None

        # Which order filled, and the trigger it was last seen resting at.
        fill_match = "symbol_vwap_unrecorded_sells"
        group = entry = None
        try:
            groups = [g for g in (fill.get("by_order") or []) if isinstance(g, dict)
                      and g.get("order_id") is not None and _pos(g.get("price"))
                      and _pos(g.get("qty"))]
            if groups:
                cached = [e for e in (_read_stop_trigger_cache() or {}).values()
                          if isinstance(e, dict) and e.get("ticker") == ticker
                          and e.get("order_id") is not None]

                def _perm_clash(g, e):
                    gp, ep = g.get("perm_id"), e.get("perm_id")
                    return (int(g["order_id"]) == int(e["order_id"]) and gp is not None
                            and ep is not None and int(gp) != int(ep))

                def _same(g, e):
                    return int(g["order_id"]) == int(e["order_id"]) and not _perm_clash(g, e)
                clashed = {id(g) for g in groups for e in cached if _perm_clash(g, e)}
                pairs = [(g, e) for g in groups for e in cached if _same(g, e)]
                if stop_order_id:
                    own = [p for p in pairs if int(p[0]["order_id"]) == int(stop_order_id)]
                    pairs = own or pairs
                    if not pairs:   # the record's order filled but was never seen resting
                        pairs = [(g, None) for g in groups
                                 if int(g["order_id"]) == int(stop_order_id)
                                 and id(g) not in clashed]
                if len(pairs) == 1:
                    group, entry = pairs[0]
                    both_perm = (entry is not None and group.get("perm_id") is not None
                                 and entry.get("perm_id") is not None)
                    fill_match = "stop_order_perm_id" if both_perm else "stop_order_id"
                elif pairs:
                    fill_match = "symbol_vwap_ambiguous_orders"
                elif clashed:
                    fill_match = "symbol_vwap_perm_id_mismatch"
        except (TypeError, ValueError, KeyError):
            group = entry = None
            fill_match = "symbol_vwap_unrecorded_sells"
        matched_price = _pos(group["price"]) if group else None
        matched_shares = float(group["qty"]) if group else None
        trigger = _pos(entry.get("trigger")) if entry else None
        trig_slip = (round((matched_price - trigger) / trigger * 100, 4)
                     if trigger is not None and matched_price is not None else None)

        pc = _pos(prev_close)
        pc_date = pc_reason = None
        pc_source = "caller" if pc is not None else None
        if pc is None:
            try:
                fill_day = (fill_time.astimezone(_ny) if fill_time else datetime.now(_ny)).date()
                bars = [b for b in (db.get_price_history(ticker, days=5) or [])
                        if str(b.get("date")) < fill_day.isoformat() and _pos(b.get("close"))]
                if bars:
                    pc, pc_date, pc_source = (float(bars[-1]["close"]), str(bars[-1]["date"]),
                                              "price_history")
                else:
                    pc_reason = "no_price_history_close_before_fill_date"
            except Exception:
                pc, pc_reason = None, "price_history_read_failed"

        stop, price, shares = record_stop, vwap, shares_total
        slip = (price - stop) / stop * 100
        return bool(db.log_activity(
            "stop_slippage", ticker,
            f"stop fill {ticker}: stop ${stop:.2f}, filled ${price:.2f} ({slip:+.2f}%)",
            "neutral",
            {"ticker": ticker, "stop_price": stop, "fill_price": price,
             "slippage_pct": round(slip, 4),
             "shares": float(shares) if shares is not None else None,
             "fill_time": fill_time.isoformat() if fill_time else None,
             "outside_rth": outside_rth, "at_open": at_open,
             "prev_close": pc, "source": source,
             "fill_match": fill_match,
             "fill_order_ids": order_ids, "fill_is_stop_order": fill_is_stop,
             "stop_order_id": stop_order_id, "env": TRADING_ENV,
             "stop_price_source": "db_record",
             "matched_order_id": int(group["order_id"]) if group else None,
             "matched_perm_id": group.get("perm_id") if group else None,
             "matched_fill_price": matched_price, "matched_shares": matched_shares,
             "broker_trigger_price": trigger,
             "broker_trigger_seen_at": entry.get("seen_at") if trigger is not None else None,
             "broker_order_type": entry.get("order_type") if entry else None,
             "slippage_vs_broker_trigger_pct": trig_slip,
             "trailing_active": bool(trailing_active) if trailing_active is not None else None,
             "stop_method": stop_method,
             "prev_close_date": pc_date, "prev_close_source": pc_source,
             "prev_close_null_reason": pc_reason,
             "shadow_key": key, **shadow_versions()}))
    except Exception as e:
        log.debug(f"[{ticker}] stop slippage log failed: {e}")
        return False


_STOP_TRIGGER_CACHE_KEY = "stop_trigger_cache_{env}"
_STOP_TRIGGER_CACHE_MAX_AGE_S = 3 * 86400
_STOP_TRIGGER_CACHE_MAX = 500


def _read_stop_trigger_cache() -> dict | None:
    """The last-seen triggers _cache_stop_triggers keeps, or None if they
    can't be read. Log-only (the stop_slippage row)."""
    try:
        import db
        from config import TRADING_ENV
        raw = db.get_state(_STOP_TRIGGER_CACHE_KEY.format(env=TRADING_ENV), "") or ""
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cache_stop_triggers(working: list[dict], now=None) -> None:
    """Log-only (professor M229): remember each working SELL stop's trigger
    as this reconcile cycle saw it, keyed by orderId and permId, so the
    stop_slippage row for a later fill can use the order's own trigger
    instead of the DB record, which sits below a native TRAIL's. Entries
    unseen for 3 days are dropped. A failed or unparseable read writes
    nothing, so a DB blip can't wipe the cache. Nothing is sent or changed
    at the broker, and no stop decision reads it. Never raises."""
    try:
        import db
        from datetime import datetime, timezone
        from config import TRADING_ENV
        now = now or datetime.now(timezone.utc)
        try:
            raw = db.get_state_strict(_STOP_TRIGGER_CACHE_KEY.format(env=TRADING_ENV), "") or ""
            cache = json.loads(raw) if raw else {}
        except Exception as e:
            log.debug(f"stop trigger cache read failed; not rewritten this cycle: {e}")
            return
        if not isinstance(cache, dict):
            log.debug("stop trigger cache unreadable; not rewritten this cycle")
            return
        for o in working or ():
            if o.get("order_id") is None or not o.get("ticker"):
                continue
            perm = o.get("perm_id")
            cache[f"{o['order_id']}|{perm or ''}"] = {
                "ticker": o["ticker"], "order_id": int(o["order_id"]),
                "perm_id": int(perm) if perm else None, "order_type": o.get("order_type"),
                "trigger": o.get("stop_price"), "seen_at": now.isoformat()}
        keep = {}
        for k, e in cache.items():
            try:
                age = (now - datetime.fromisoformat(e["seen_at"])).total_seconds()
            except Exception:
                continue
            if age <= _STOP_TRIGGER_CACHE_MAX_AGE_S:
                keep[k] = e
        if len(keep) > _STOP_TRIGGER_CACHE_MAX:
            keep = dict(sorted(keep.items(), key=lambda kv: kv[1]["seen_at"])[-_STOP_TRIGGER_CACHE_MAX:])
        db.set_state(_STOP_TRIGGER_CACHE_KEY.format(env=TRADING_ENV), json.dumps(keep, sort_keys=True))
    except Exception as e:
        log.debug(f"stop trigger cache update failed: {e}")


@app.task(name="tasks.reconcile_stop_orders", bind=True, max_retries=2)
def reconcile_stop_orders(self):
    """One run at a time (professor M71): a run can outlast the 900 s
    schedule, and two runs could double-book an exit or both fix one
    ticker's coverage. See _reconcile_stop_orders_body for what it does."""
    def run():
        _reconcile_stop_orders_body(self)
        try:
            _review_manual_actions()
        except Exception as e:
            log.warning(f"manual-action review failed: {e}")
    _run_singleton("reconcile_stop_orders", run)


def _run_singleton(name: str, fn, max_hold_s: float = 1800) -> None:
    """Run fn under a renewed Redis run lock; skip if another run holds it
    or Redis is unavailable (audit 00:53 A0053-6, professor M71)."""
    import threading
    import stop_lock
    try:
        run_lock = stop_lock._redis().lock(f"helios:task:{name}", timeout=stop_lock.LOCK_TTL_S,
                                           blocking_timeout=0, thread_local=False)
        if not run_lock.acquire():
            log.info(f"{name}: previous run still going, skipping")
            return
    except Exception as e:
        log.error(f"{name}: run lock unavailable ({e}); skipping this run")
        return
    done = threading.Event()
    threading.Thread(target=stop_lock._keep_renewed, args=(run_lock, name, done, max_hold_s),
                     daemon=True).start()
    try:
        fn()
    finally:
        done.set()
        try:
            run_lock.release()
        except Exception:
            pass


def _reconcile_stop_orders_body(self):
    """
    Safety net: diff broker-side sell-stop orders against actual positions.
    - Cancels orphaned stops (stop exists, position doesn't) → prevents
      accidental short positions.
    - Deletes stale DB stop records with no matching position.
    - Flags positions that have NO protective stop at the broker.
    Every corrective action and new/unresolved-and-due issue is logged as a
    WARNING (see _reconcile_alert_due for the per-issue throttle) but no
    longer sent to Telegram — 2026-07-28, user request, after the
    unthrottled-repeat bug that caused this was already fixed the day
    before. That silence originally covered the "no protective stop" case
    too (the same category of gap behind the AMKR incident that same day)
    — as of 2026-07-30 it's escalated to Telegram again, on its own
    sustained-duration + throttle rule below, after AMD sat fully naked
    for ~30 hours with this firing to the log every 15 minutes the whole
    time and nobody reading it. The routine orphan-cancel/stale-record
    actions stay log-only. Check logs/celery_worker.log for "Stop
    reconciliation" WARNING lines, or the dashboard, for everything else.

    Incident 2026-07-29 (KLAC): client.get_all_positions() reads
    ib_async's local portfolio cache (ib.portfolio()), populated
    asynchronously from TWS's updatePortfolio stream rather than a
    synchronous snapshot -- right around market open that cache can
    briefly lag a real position by a cycle or two. This task ran at 09:32
    ET, read a cache that didn't yet include KLAC, and cancelled its
    legitimate 56-share stop as "orphaned" -- deleting real stop coverage
    based on a momentarily-stale read. The orphan-cancel branch now
    requires the SAME ticker to look positionless on two consecutive
    15-minute cycles (a system_state-backed marker, since Celery workers
    are stateless across runs) before actually cancelling anything. A
    transient cache lag self-corrects within one cycle; a genuinely closed
    position is still gone 15 minutes later.
    """
    try:
        import db
        import ibkr
        from datetime import datetime, timezone

        client    = ibkr.get_client()
        positions = client.get_all_positions()
        held      = {p.symbol: round(abs(float(p.qty)), 4) for p in positions}
        cache_qty = {p.symbol: round(float(p.qty), 4) for p in positions}   # signed
        longs     = {p.symbol for p in positions if float(p.qty) > 0}
        open_ords = ibkr.get_open_orders()
        # Section 1 (orphans) cancels any stop order; coverage (2b, 3) counts
        # only working stops by their remainder, as _fix_overstop does.
        stops     = [o for o in open_ords
                     if o["action"] == "SELL" and o["order_type"] in ("STP", "STP LMT", "TRAIL")]
        working   = _working_sell_stops(open_ords)
        _cache_stop_triggers(working)   # log-only: the stop_slippage row's trigger (M229)
        all_actions: list[str] = []   # every cycle, for the debug log
        new_actions: list[str] = []   # unthrottled — logged as warning + alerted

        # Sections 1 and 2 cancel stops and delete records for tickers this
        # read doesn't show. The portfolio cache can read empty or short
        # after a reconnect, and two such cycles in a row used to cancel
        # every stop and delete every record (audit 2026-11-02 #6). Before
        # either acts, a fresh reqPositions() read must agree those tickers
        # are gone. It fails: sections 1 and 2 wait this cycle. It shows a
        # ticker held: that ticker is left out of sections 1 and 2 and an
        # alert goes out. Sections 2b and 3 always run for every held ticker:
        # skipping them left naked positions unfixed and unalerted.
        records = db.get_all_position_stops()
        looks_flat = ({o["ticker"] for o in stops} | {r["ticker"] for r in records}) - set(held)
        # Flat per the cache but still with a working SELL stop: alerted every
        # cycle while the missing-held hold blocks their cleanup (M221).
        working_flat = sorted(looks_flat & {o["ticker"] for o in working})
        flat_confirmed = True
        disagree: list[str] = []
        wait_flat: set[str] = set()   # empty-read flat not yet confirmed: markers arm, nothing acts
        fresh = None
        if looks_flat:
            try:
                fresh = ibkr.get_fresh_stock_positions()
            except Exception as e:
                log.warning(f"reconcile_stop_orders: confirming position read failed ({e}); "
                            f"orphan/stale-record checks wait a cycle for {sorted(looks_flat)}")
                fresh = None
                flat_confirmed = False
                _touch_missing_held()
            if fresh is not None and not fresh and records:
                # An empty fresh read while stop records exist looks like the
                # same reconnect gap as the cache, not a confirmed flat. Only
                # 3 consecutive cycles with both reads empty confirm it, so a
                # truly flat account's orphan stops (which could open a short)
                # still get cleaned up.
                if held:
                    log.warning(f"reconcile_stop_orders: confirming position read is empty while "
                                f"stop records exist; orphan/stale-record checks wait a cycle for "
                                f"{sorted(looks_flat)}")
                    flat_confirmed = False
                    _note_missing_held(sorted(t for t, q in held.items() if q), looks_flat,
                                       working_flat, empty_fresh=True)
                else:
                    counted = _count_empty_flat_reads(looks_flat)
                    if counted is None:
                        flat_confirmed = False
                    else:
                        wait_flat, _, first_seen = counted
                        ready = sorted(set(looks_flat) - wait_flat)
                        confirmed_now: list[str] = []
                        if ready:
                            # Both reads come from the same session; the
                            # server's account-level view must agree too.
                            gate_ok, why = _account_flat_gate()
                            if gate_ok:
                                confirmed_now = [t for t in ready
                                                 if _reconcile_alert_due(f"empty_read_flat_{t}")]
                            else:
                                wait_flat |= set(ready)
                                log.error(f"reconcile_stop_orders: {ready} empty on 3+ cycles of both "
                                          f"reads, but {why}; still waiting")
                                import time as _time
                                global _empty_read_gate_alert_at
                                if (not _empty_read_gate_alert_at
                                        or _time.monotonic() - _empty_read_gate_alert_at >= 3600):
                                    _empty_read_gate_alert_at = _time.monotonic()
                                    try:
                                        import alerts
                                        alerts._send(f"⚠️ STOP CHECK: no position in {', '.join(ready)} "
                                                     f"on 3+ cycles of both position reads, but {why}. "
                                                     f"Nothing cancelled or deleted; still waiting.")
                                    except Exception as e:
                                        log.error(f"reconcile_stop_orders: could not send the "
                                                  f"account-check alert: {e}")
                        log.warning(f"reconcile_stop_orders: both position reads empty while stop "
                                    f"records exist; waiting: {sorted(wait_flat)}, confirmed flat: "
                                    f"{sorted(set(looks_flat) - wait_flat)}")
                        for tickers_, text in (
                                (first_seen, "broker shows no position but stop records exist for "
                                             "{t}; confirming over 3 cycles before any stop is "
                                             "cancelled or record deleted"),
                                (confirmed_now, "no position in {t} on 3 consecutive cycles of both "
                                                "position reads, and the account-level view agrees; "
                                                "treating as flat, cleaning up its orphan stops and "
                                                "stop records now")):
                            if not tickers_:
                                continue
                            try:
                                import alerts
                                alerts._send(f"🚨 STOP CHECK: " + text.format(t=", ".join(tickers_)))
                            except Exception as e:
                                log.error(f"reconcile_stop_orders: could not send the empty-read "
                                          f"alert: {e}")
            elif fresh:
                _clear_empty_flat_reads(looks_flat)   # a non-empty read resets the count
            if fresh is not None and flat_confirmed:
                # A fresh read that leaves out a ticker the cache holds is
                # shown incomplete, so its "no position" for the others can't
                # confirm them flat either: orphan cancels and record deletes
                # wait this cycle (professor M218).
                missing_held = sorted(t for t, q in held.items() if q and t not in fresh)
                if missing_held:
                    flat_confirmed = False
                    log.warning(f"reconcile_stop_orders: the confirming position read leaves out "
                                f"held {missing_held}; orphan/stale-record checks wait a cycle "
                                f"for {sorted(looks_flat)}")
                _note_missing_held(missing_held, looks_flat,
                                   [t for t in working_flat if t not in fresh])
            disagree = sorted(t for t in looks_flat if fresh and t in fresh)
            if disagree:
                msg = (f"position reads disagree: the portfolio cache shows no position in "
                       f"{', '.join(disagree)} but a fresh positions read does; their stops and "
                       f"records are left alone this cycle")
                log.error(f"reconcile_stop_orders: {msg}")
                if _reconcile_alert_due("position_reads_disagree"):
                    try:
                        import alerts
                        alerts._send(f"⚠️ STOP CHECK PAUSED\nBroker {msg}. If it persists, check "
                                     f"the IB gateway connection.")
                    except Exception as e:
                        log.error(f"reconcile_stop_orders: could not send the disagreement alert: {e}")
        skip_flat = set(disagree)   # held per the fresh read: never cancelled or deleted
        stopped_now = {o["ticker"] for o in working}
        reread_box: dict = {}   # one extra fresh read per cycle, shared by every ticker

        def _fresh_reread() -> dict | None:
            """A second fresh positions read, taken at most once this cycle,
            for a held ticker the first fresh read left out (professor M210).
            None when it fails."""
            if "read" not in reread_box:
                try:
                    reread_box["read"] = ibkr.get_fresh_stock_positions()
                except Exception as e:
                    log.warning(f"reconcile_stop_orders: second fresh positions read failed ({e})")
                    reread_box["read"] = None
            return reread_box["read"]

        def _smaller_read(t: str) -> float:
            """Smaller of the two position reads (absolute), for sizing stops
            while they disagree; the fresh read alone when the cache is flat,
            and 0 when the two reads disagree on its sign (no SELL stop is
            added). When the fresh read leaves out a ticker the cache holds,
            one more fresh read is taken: if it shows the ticker, the smaller
            of it and the cache is the cap; if not, the cap stays 0 (never an
            oversell) and, while the position is naked, the alert goes out on
            every cycle and a manual action is opened (professor M210)."""
            f_signed = round(float((fresh or {}).get(t) or 0.0), 4)
            if t in held and t not in (fresh or {}):
                again = _fresh_reread()
                if again and round(float(again.get(t) or 0.0), 4) != 0:
                    f_signed = round(float(again[t]), 4)
                    log.warning(f"[{t}] first fresh positions read left it out; a second one "
                                f"shows {f_signed:g} shares (cache {held[t]:g})")
            c_signed = cache_qty.get(t, 0.0)
            if c_signed * f_signed < 0:
                log.error(f"[{t}] position reads disagree on sign (cache {c_signed:g}, fresh "
                          f"{f_signed:g}); no SELL stop added")
                if _reconcile_alert_due(f"sign_flip_{t}"):
                    try:
                        import alerts
                        alerts._send(f"🚨 STOP CHECK — <b>{t}</b>\nThe position reads disagree "
                                     f"on sign: the portfolio cache shows {c_signed:g} shares, a "
                                     f"fresh read shows {f_signed:g}. No SELL stop added and "
                                     f"nothing cancelled. Check the position in TWS now.")
                    except Exception as e:
                        log.error(f"[{t}] could not send the sign-flip alert: {e}")
                return 0.0
            if t in held and f_signed == 0:
                naked = t not in stopped_now
                log.error(f"[{t}] the portfolio cache shows {held[t]:g} shares but two fresh "
                          f"positions reads show none; no SELL stop added"
                          + (" and the position has no stop" if naked else ""))
                # A naked position alerts every cycle (no throttle); one that
                # still has stops keeps the usual throttle.
                due = (_reconcile_cycle_alert_due(f"fresh_missing_{t}") if naked
                       else _reconcile_alert_due(f"fresh_missing_{t}"))
                if due:
                    try:
                        import alerts
                        alerts._send(f"🚨 STOP CHECK — <b>{t}</b>\nThe portfolio cache shows "
                                     f"{held[t]:g} shares but a fresh positions read shows none. "
                                     f"No SELL stop added and nothing cancelled."
                                     + (" It has NO protective stop at the broker; this alert "
                                        "repeats every stop check until it is resolved."
                                        if naked else "")
                                     + " Check the position in TWS now.")
                    except Exception as e:
                        log.error(f"[{t}] could not send the fresh-missing alert: {e}")
                if naked:
                    try:
                        already = t in json.loads(db.get_state(_MANUAL_ACTIONS_KEY) or "{}")
                    except Exception:
                        already = False
                    if not already:   # re-opening would reset the hourly STILL OPEN clock
                        open_manual_action(t, f"the portfolio cache shows {held[t]:g} shares but "
                                              f"fresh position reads show none, and no stop is "
                                              f"placed; check the position and place a stop",
                                           clear_on_flat=False)
                return 0.0
            f = abs(f_signed)
            return f if t not in held else min(held[t], f)

        # Clear any pending-orphan/pending-stale-stop marker for tickers
        # confirmed held this cycle -- a marker set during a transient
        # cache lag shouldn't linger and cause an action on some unrelated
        # future cycle.
        for ticker in set(held) | skip_flat:
            db.set_state(f"orphan_pending_{ticker}", "")
            db.set_state(f"stale_stop_pending_{ticker}", "")
        _clear_empty_flat_reads(set(held) | skip_flat)

        # 1. Orphaned broker stops — stop order with no position on TWO
        # consecutive cycles → cancel. A single missing-this-cycle reading
        # just arms the marker and waits.
        for o in stops:
            if o["ticker"] not in held and flat_confirmed and o["ticker"] not in skip_flat:
                pending_key = f"orphan_pending_{o['ticker']}"
                if not _pending_marker_ready(pending_key):
                    log.info(f"[{o['ticker']}] Stop order {o['order_id']} looks orphaned "
                             f"(no position in this cycle's snapshot) — waiting for the next "
                             f"cycle to confirm before cancelling")
                    continue
                if o["ticker"] in wait_flat:
                    continue   # empty reads not yet confirmed over 3 cycles
                if manage_cancel_order(o["order_id"], symbol=o["ticker"], perm_id=o.get("perm_id")):
                    msg = (f"Cancelled ORPHANED stop: SELL {o['qty']} {o['ticker']} "
                           f"@ ${o['stop_price'] or 0:.2f} (no position held on 2 consecutive "
                           f"checks — would have shorted)")
                    all_actions.append(msg)
                    db.set_state(pending_key, "")
                    # Found 2026-07-27: this was deliberately left unthrottled
                    # on the assumption a successful cancel can't recur for
                    # the same order -- SNDK proved that wrong (a cross-
                    # clientId order whose status transiently reads
                    # "cancelled" in cancel_order's own check without the
                    # cancel actually sticking at the broker), firing the
                    # identical message ~33x in one trading day. Same
                    # order_id-keyed throttle as the other two branches below.
                    if _reconcile_alert_due(f"orphan_cancelled_{o['order_id']}"):
                        new_actions.append(msg)
                else:
                    msg = (f"Could NOT cancel orphaned stop: SELL {o['qty']} {o['ticker']} "
                           f"orderId={o['order_id']} — likely owned by a different clientId, "
                           f"needs manual cancel in TWS")
                    all_actions.append(msg)
                    if _reconcile_alert_due(f"cancel_fail_{o['order_id']}"):
                        new_actions.append(msg)

        # 2. Stale DB stop records with no position → reconcile the exit
        # against today's broker execution history, THEN delete. Requires
        # the SAME ticker to look positionless on TWO consecutive cycles
        # before acting, same as branch 1's orphan-cancel above.
        #
        # Incident 2026-08-06 (DDOG): this used to just delete the record
        # and move on -- an 84-share position disappeared (its own tracked
        # stop order never filled; it sat orphaned/uncancellable the rest
        # of the day) with no bot-initiated sell anywhere in the logs, and
        # this branch logged it as routine "deleted stale DB stop record"
        # noise. No trade or trade_outcome row was ever written -- the
        # exit, and whatever was gained or lost on it, vanished from every
        # DB table with nothing pointing back at it.
        #
        # Now searches by symbol (ibkr.get_recent_sell_fill) for ANY SELL
        # execution today, not just the specific stop order this system
        # was tracking -- catches an exit via the tracked stop, an
        # unlogged backstop sell, or a manual sale placed directly in TWS.
        # A match gets written as a real trade and closes out the
        # trade_outcome row properly. Same same-trading-day cap as
        # get_fill_for_order (confirmed 2026-08-04): only catches an exit
        # from earlier today. If nothing is found, this no longer silently
        # deletes and forgets -- it escalates to Telegram immediately, same
        # urgency as an unprotected position, since "we don't know what
        # happened to a real position" is at least as serious as "this one
        # doesn't have a stop yet."
        #
        # Incident 2026-08-17 (QBTS, live's first trade): client.get_all_
        # positions() read its cache moments after a fresh fill, before
        # the new position had synced -- the exact same cache-lag shape as
        # the 2026-07-29 KLAC incident that motivated branch 1's two-cycle
        # requirement above, just hitting this branch instead. This branch
        # had no such protection, so it declared QBTS's position
        # "disappeared with no matching SELL execution" on the very first
        # read and deleted the (accurate) DB stop record -- the real
        # broker-side stop order was untouched throughout (branch 1's own
        # two-cycle guard protected it correctly), but local tracking
        # (trailing-stop management, entry/bucket/conviction metadata) was
        # gone until manually restored. Same pending-marker pattern as
        # branch 1 closes this.
        # Tickers with one of our SELLs placed today still unconfirmed or
        # part-filled: reconcile_unconfirmed_orders owns that exit (review of
        # cc72fe3). Today only: the execution lookup below reaches only the
        # current day, so an older order must not hold the cleanup back.
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        _today_et = datetime.now(_et).date()
        try:
            pending_sell_tickers = {
                o["ticker"] for o in db.get_unconfirmed_orders()
                if o["side"] == "SELL" and o.get("created_at") is not None
                and o["created_at"].astimezone(_et).date() == _today_et
            }
            # A SELL still in its fill wait, or orphaned there and not yet
            # adopted, is pending too: its fill is not recorded yet.
            pending_sell_tickers |= db.awaiting_fill_sell_tickers()
            # A filled SELL whose trade row reconcile hasn't booked yet.
            pending_sell_tickers |= db.trade_pending_sell_tickers()
        except Exception as e:
            # Unknown is not "none pending": skip stale records this cycle (M126).
            log.warning(f"reconcile_stop_orders: unconfirmed-SELL check failed; stale records wait a cycle: {e}")
            pending_sell_tickers = None
        stop_tickers = {o["ticker"] for o in stops}
        for rec in records:
            if rec["ticker"] not in held and flat_confirmed and rec["ticker"] not in skip_flat:
                ticker = rec["ticker"]
                pending_key = f"stale_stop_pending_{ticker}"
                if not _pending_marker_ready(pending_key):
                    log.info(f"[{ticker}] Position stop record looks stale (no matching "
                             f"position in this cycle's snapshot) — waiting for the next "
                             f"cycle to confirm before reconciling")
                    continue
                if ticker in wait_flat:
                    continue   # empty reads not yet confirmed over 3 cycles
                if ticker in stop_tickers:
                    # Its stop was at the broker this cycle (orphan-cancelled
                    # above, or still resting): the record goes a cycle later,
                    # so it is still there to rebuild from if the position
                    # reappears.
                    log.info(f"[{ticker}] Stale stop record kept: a stop order was at the broker "
                             f"this cycle; deleting no sooner than next cycle")
                    continue
                if pending_sell_tickers is None or ticker in pending_sell_tickers:
                    # Leaving that order's fills out below would orphan the
                    # outcome, so wait for the late-fill reconcile.
                    log.info(f"[{ticker}] Stale stop record: " + (
                        "unconfirmed SELLs unknown (read failed); waiting" if pending_sell_tickers is None
                        else "an unconfirmed SELL order is still being reconciled; waiting"))
                    continue
                try:
                    recorded = db.recorded_sell_order_ids(ticker)   # professor M122
                except Exception:
                    recorded = None
                if recorded is None:
                    # Unknown which sells are already logged: counting them all
                    # would blend them into the exit again. Retry next cycle.
                    log.warning(f"[{ticker}] Stale stop record: recorded-sell lookup failed; retrying "
                                f"next cycle")
                    continue
                try:
                    fill = ibkr.get_recent_sell_fill(ticker, *recorded)
                except Exception as e:
                    # Unknown is not "no SELL": keep the record and the marker and
                    # look again next cycle, or a real exit is labelled no_fill
                    # and its record deleted.
                    log.warning(f"[{ticker}] Exit-fill lookup failed while reconciling "
                                f"stale stop record; retrying next cycle: {e}")
                    continue
                db.set_state(pending_key, "")

                if fill:
                    ok = db.log_trade(
                        symbol=ticker, side="SELL", quantity=fill["filled_qty"],
                        price=fill["fill_price"],
                        reason="Reconciled exit — position closed at the broker with no "
                               "bot-initiated sell on record (resting stop fill or manual "
                               "close); matched to today's broker execution history during "
                               "stale-stop cleanup",
                        price_source="fill",
                    )
                    try:
                        db.close_trade_outcome(ticker, exit_price=fill["fill_price"], exit_source="broker_exit",
                                               exit_price_source="fill", exit_fill_at=fill.get("fill_time"))
                    except Exception as e:
                        log.error(f"[{ticker}] close_trade_outcome failed after reconciling "
                                  f"exit: {e}")
                    try:   # log-only: stop price vs fill, once the trade row is written
                        if ok:
                            _log_stop_slippage(ticker, rec.get("stop_price"), fill,
                                               "reconcile_stale_stop", rec.get("broker_order_id"),
                                               trailing_active=rec.get("trailing_active"),
                                               stop_method=rec.get("stop_method"))
                    except Exception:
                        pass
                    msg = (f"Reconciled untracked exit for {ticker}: SELL "
                           f"{fill['filled_qty']:g} @ ${fill['fill_price']:.2f}")
                    all_actions.append(msg)
                    new_actions.append(msg)
                else:
                    msg = (f"{ticker} position disappeared with NO matching SELL execution "
                           f"found in today's broker history — trade/outcome NOT recorded, "
                           f"needs manual reconciliation against a broker statement")
                    all_actions.append(msg)
                    new_actions.append(msg)
                    log.error(f"[{ticker}] {msg}")
                    try:
                        db.flag_orphan_outcomes_for(ticker, "no_fill")   # professor M120
                    except Exception as e:
                        log.error(f"[{ticker}] could not flag the open outcome orphaned: {e}")
                    if _reconcile_alert_due(f"unreconciled_exit_{ticker}"):
                        try:
                            import alerts
                            alerts._send(
                                f"🚨 UNRECONCILED EXIT — <b>{ticker}</b>\n"
                                f"Position closed at the broker but no matching SELL execution "
                                f"was found in today's history. Trade and outcome NOT recorded. "
                                f"Check TWS/account statement to reconstruct manually."
                            )
                        except Exception as e:
                            log.error(f"[{ticker}] Could not send unreconciled-exit alert: {e}")

                db.delete_position_stop(ticker)

        # 2b. Resting SELL stops for more shares than held (professor M54,
        # 2026-10-28): the one check that catches every stacked-stop race
        # (a late stop after a lost reply, two paths interleaving). If both
        # trigger they oversell into a short. Two consecutive cycles, like
        # the orphan check, then fix under the ticker lock.
        stop_qty: dict[str, float] = {}
        for o in working:
            stop_qty[o["ticker"]] = stop_qty.get(o["ticker"], 0.0) + float(o["qty"])
        # A partly covered position (some stop, but for fewer shares than
        # held) is fixed the same way (audit 22:53, A2253-4); no stop at all
        # is section 3's alert.
        # Tickers the reads disagree on (the cache reads them flat, a fresh
        # read shows them held) are checked at the fresh-read quantity.
        overstop_check = dict(held)
        for ticker in disagree:
            overstop_check.setdefault(ticker, round(abs(float(fresh[ticker])), 4))
        # While the reads disagree, work out each ticker's smaller read once,
        # up front: its sign-flip / fresh-missing alerts then go out on the
        # first cycle, even for a ticker whose stops match or that is naked.
        smaller = {t: _smaller_read(t) for t in overstop_check} if disagree else {}
        for ticker, qty in overstop_check.items():
            pending_key = f"overstop_pending_{ticker}"
            covered = stop_qty.get(ticker, 0.0)
            in_disagree = ticker in disagree and ticker not in held
            if (ticker in longs) or (in_disagree and float(fresh[ticker]) > 0):
                # a short's SELL adds to it; _fix_overstop alerts on that
                qty = max(round(qty - _working_mkt_sell_qty(open_ords, ticker), 4), 0.0)   # professor M129
            if covered <= 0 or abs(covered - qty) <= 1e-6:
                db.set_state(pending_key, "")
                continue
            if in_disagree and covered > qty:
                # The reads disagree on this ticker, so neither quantity is
                # sure: no over-cover cancel or shrink (roadmap 00 e).
                db.set_state(pending_key, "")
                log.error(f"[{ticker}] resting stops cover {covered:g} vs {qty:g} per the fresh "
                          f"read, but the position reads disagree; over-cover fix skipped")
                if _reconcile_alert_due(f"overstop_disagree_{ticker}"):
                    try:
                        import alerts
                        alerts._send(f"🚨 STOP CHECK — <b>{ticker}</b>\nResting stops cover "
                                     f"{covered:g} shares vs {qty:g} held per a fresh read, but the "
                                     f"position reads disagree, so nothing was cancelled. Check the "
                                     f"position and its stops in TWS.")
                    except Exception as e:
                        log.error(f"[{ticker}] could not send the over-cover alert: {e}")
                continue
            # While any ticker's reads disagree, a held ticker whose fresh-read
            # quantity differs from the cache's is not sure either: no
            # over-cover cancel or shrink for it this cycle. Compared signed,
            # so a sign flip (cache +5, fresh -5) is a difference too.
            fresh_mismatch = bool(disagree) and ticker in held and (
                fresh.get(ticker) is None
                or abs(round(float(fresh.get(ticker)), 4) - cache_qty[ticker]) > 1e-6)
            if fresh_mismatch and covered > qty:
                db.set_state(pending_key, "")
                log.error(f"[{ticker}] resting stops cover {covered:g} vs held {qty:g}, but the "
                          f"fresh read shows {fresh.get(ticker)} vs {held[ticker]:g} in the cache "
                          f"while the position reads disagree; over-cover fix skipped")
                if _reconcile_alert_due(f"overstop_qty_mismatch_{ticker}"):
                    try:
                        import alerts
                        alerts._send(f"🚨 STOP CHECK — <b>{ticker}</b>\nResting stops cover "
                                     f"{covered:g} shares vs {qty:g} held, but a fresh positions "
                                     f"read shows a different quantity while the position reads "
                                     f"disagree, so nothing was cancelled. Check the position and "
                                     f"its stops in TWS.")
                    except Exception as e:
                        log.error(f"[{ticker}] could not send the over-cover alert: {e}")
                continue
            # Under-cover while the reads differ: stops never go past the
            # smaller read. Already at or past it: report, add nothing.
            cap = smaller.get(ticker) if (in_disagree or fresh_mismatch) else None
            if cap is not None and covered >= cap - 1e-6 and fresh_mismatch:
                db.set_state(pending_key, "")
                log.error(f"[{ticker}] resting stops cover {covered:g}, at or above the smaller "
                          f"position read ({cap:g}) while the reads disagree; no stop added")
                if _reconcile_alert_due(f"overstop_min_read_{ticker}"):
                    try:
                        import alerts
                        alerts._send(f"🚨 STOP CHECK — <b>{ticker}</b>\nResting stops cover "
                                     f"{covered:g} shares, which "
                                     f"{'exceeds' if covered > cap + 1e-6 else 'matches'} the "
                                     f"smaller position read ({cap:g}) while the reads disagree. "
                                     f"No stop added and nothing cancelled. Check the position "
                                     f"and its stops in TWS.")
                    except Exception as e:
                        log.error(f"[{ticker}] could not send the smaller-read alert: {e}")
                continue
            if not _pending_marker_ready(pending_key):
                log.warning(f"[{ticker}] resting stops cover {covered:g} vs held {qty:g}; "
                            f"confirming next cycle")
                continue
            db.set_state(pending_key, "")
            if in_disagree:
                msg = _fix_overstop_safe(ticker, fresh_qty=float(fresh[ticker]), allow_cancel=False,
                                         max_qty=cap)
            elif fresh_mismatch:
                msg = _fix_overstop_safe(ticker, allow_cancel=False, max_qty=cap)   # add only
            else:
                msg = _fix_overstop_safe(ticker)
            all_actions.append(msg)
            new_actions.append(msg)
            try:
                import alerts
                if _stop_fix_alert_due(ticker, msg):
                    alerts._send(f"STOP RECONCILE — {msg}")
            except Exception:
                pass

        # 3. Positions with no protective stop at the broker → alert, and
        # escalate to Telegram once it's been genuinely sustained.
        #
        # Incident 2026-07-29/30 (AMD): this case was silenced to log-only
        # on 2026-07-28 along with the routine orphan-cancel/stale-record
        # noise, but it's a different kind of event -- it's the one thing
        # standing between a real gap and someone actually finding out.
        # AMD sat fully naked for ~30 hours while this fired to the log
        # every 15 minutes (49x) and nobody saw it. Full silence traded too
        # much for the noise fix it was solving.
        #
        # Compromise: still no Telegram ping for a stop that's merely
        # mid-transition (a TRIM/replace can leave a position briefly
        # unstopped for one cycle in the ordinary course of things) -- only
        # once the SAME ticker has been continuously unstopped across 3
        # consecutive cycles (~45 min) does it escalate, and then on the
        # normal RECONCILE_ALERT_THROTTLE_HOURS cadence after that so it
        # doesn't spam every 15 minutes for a known, ongoing issue either.
        stopped_tickers = {o["ticker"] for o in working}
        for ticker in stopped_tickers:
            db.set_state(f"no_stop_since_{ticker}", "")  # has a stop again -- reset
        # A ticker the cache reads flat but a fresh read shows held (disagree)
        # is checked too, at the fresh-read quantity: a naked one gets the
        # alert and the auto-stop like any other (roadmap 00 d).
        naked_check = dict(held)
        for ticker in disagree:
            naked_check.setdefault(ticker, round(abs(float(fresh[ticker])), 4))
        for ticker, qty in naked_check.items():
            if ticker not in stopped_tickers:
                msg = f"⚠️ {ticker} ({qty} shares) has NO protective stop at the broker"
                all_actions.append(msg)
                if _reconcile_alert_due(f"no_stop_{ticker}"):
                    new_actions.append(msg)

                since_key = f"no_stop_since_{ticker}"
                since = db.get_state(since_key)
                # A marker left by a position that closed while naked isn't
                # "the previous cycle": restart the count if this ticker
                # wasn't seen naked within two cycles (audit 02:53, A0253-8).
                seen_key = f"no_stop_seen_{ticker}"
                seen = db.get_state(seen_key)
                now_iso = datetime.now(timezone.utc).isoformat()
                db.set_state(seen_key, now_iso)
                try:
                    stale = not seen or (datetime.now(timezone.utc)
                                         - datetime.fromisoformat(seen)).total_seconds() > 2400
                except ValueError:
                    stale = True
                if since and stale:
                    since = None
                if not since:
                    db.set_state(since_key, datetime.now(timezone.utc).isoformat())
                else:
                    # Naked on two consecutive cycles: put a stop on at the
                    # recorded level instead of only alerting (A0053-2).
                    if ticker in disagree and ticker not in held:
                        fix = _fix_overstop_safe(ticker, fresh_qty=float(fresh[ticker]),
                                                 allow_cancel=False, max_qty=smaller.get(ticker, 0.0))
                    elif disagree:
                        fix = _fix_overstop_safe(ticker, allow_cancel=False,
                                                 max_qty=smaller.get(ticker, 0.0))
                    elif _fresh_read_short(ticker, float(held.get(ticker, 0.0))):
                        # The cache reads long, a fresh read short: a SELL
                        # stop would add to a short (alert sent above).
                        fix = (f"🚨 {ticker}: position reads disagree on sign; no SELL stop "
                               f"placed — check the position in TWS")
                    else:
                        fix = _fix_overstop_safe(ticker)
                    all_actions.append(fix)
                    new_actions.append(fix)
                    if fix.startswith("🔧"):
                        try:
                            import alerts
                            if _stop_fix_alert_due(ticker, fix):
                                alerts._send(f"STOP RECONCILE — {fix}")
                        except Exception:
                            pass
                        continue
                    elapsed_min = (datetime.now(timezone.utc)
                                   - datetime.fromisoformat(since)).total_seconds() / 60
                    if elapsed_min >= 45 and _reconcile_alert_due(f"no_stop_telegram_{ticker}"):
                        try:
                            import alerts
                            alerts._send(
                                f"🚨 UNPROTECTED POSITION — <b>{ticker}</b>\n"
                                f"{qty:g} shares with no protective stop at the broker for "
                                f"{elapsed_min / 60:.1f}+ hours. Place a stop manually in TWS "
                                f"or check why one wasn't placed."
                            )
                        except Exception as e:
                            log.error(f"[{ticker}] Could not send unprotected-position alert: {e}")

        if all_actions:
            log.info("Stop reconciliation status: " + " | ".join(all_actions))
        else:
            log.info("Stop reconciliation: broker and DB in sync.")

        if new_actions:
            # Logged only, not sent to Telegram — see this task's docstring.
            log.warning("Stop reconciliation — new/unthrottled: " + " | ".join(new_actions))
    except Exception as e:
        log.error(f"reconcile_stop_orders failed: {e}")
        raise self.retry(exc=e, countdown=120)


_OLLAMA_GPU_CHECK_TIMEOUT_S = 300   # see check_ollama_gpu_health docstring
_OLLAMA_RESTART_THROTTLE_S  = 1800  # don't restart more than once per 30 min


def restart_ollama_and_alert(reason: str) -> bool:
    """
    Shared by check_ollama_gpu_health (periodic, low-frequency safety net)
    and agents.analyst._stub_recommendation (reacts to real production
    failures — see its own comment for why that path matters more). Both
    share the same throttle key so they can never double-restart each
    other. Returns True if a restart was actually performed.
    """
    import db
    from datetime import datetime as _dt, timezone as _tz
    last = db.get_state("ollama_restart_at")
    now = _dt.now(_tz.utc)
    due = True
    if last:
        try:
            due = (now - _dt.fromisoformat(last)).total_seconds() >= _OLLAMA_RESTART_THROTTLE_S
        except ValueError:
            pass
    if not due:
        log.warning(f"restart_ollama_and_alert: restarted recently, not restarting again yet ({reason})")
        return False
    db.set_state("ollama_restart_at", now.isoformat())

    import subprocess
    result = subprocess.run(
        ["sudo", "-n", "systemctl", "restart", "ollama.service"],
        capture_output=True, text=True, timeout=30,
    )
    import alerts
    if result.returncode == 0:
        alerts._send(
            f"🔧 <b>Ollama auto-recovery</b>\n{reason} — restarted ollama.service "
            f"automatically. Next analysis may take a minute longer while it reloads."
        )
    else:
        alerts._send(
            f"⚠️ <b>Ollama auto-recovery FAILED</b>\n{reason}, and the automatic restart "
            f"failed (rc={result.returncode}): {result.stderr.strip()[:300]}\n"
            f"Needs a manual `sudo systemctl restart ollama.service`."
        )
    return result.returncode == 0


@app.task(name="tasks.check_ollama_gpu_health", bind=True, max_retries=0)
def check_ollama_gpu_health(self):
    """
    Low-frequency safety net for the case where Ollama is stuck with ZERO
    real traffic to reveal it (e.g. overnight) — the primary detection path
    is now agents.analyst._stub_recommendation, which reacts to real
    production timeouts instead of manufacturing synthetic load. This task
    used to run every 10 minutes and turned out to be part of the problem:
    each probe occupies the 72B server's single concurrency slot (-np 1)
    for up to 300s, directly competing with real analyst calls. Found
    2026-07-21: real calls kept failing all day while this reported "OK"
    every time (nothing was ever stuck for hours) — the probe itself was
    adding load to an already-contended resource. Dropped to a 30-minute
    interval (see beat schedule) to cut that footprint roughly 3x while
    still catching a fully-dead-overnight scenario.

    Timeout is 300s, not a tight bound — this call can legitimately queue
    behind a real in-flight analysis (the 72B server only serves one
    request at a time), and real analyses take ~170-195s when healthy. 300s
    gives room for that without much risk of false-positive restarts, while
    still being well under the 400s+ hangs actually observed when stuck.
    """
    try:
        import time as _time
        import requests
        from config import OLLAMA_HOST, ANALYST_MODEL
        t0 = _time.monotonic()
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": ANALYST_MODEL,
                    "prompt": "Reply with the single word OK.",
                    "stream": False,
                    "options": {"num_predict": 5},
                },
                timeout=_OLLAMA_GPU_CHECK_TIMEOUT_S,
            )
            resp.raise_for_status()
            healthy = bool(resp.json().get("response"))
        except Exception as e:
            log.warning(f"check_ollama_gpu_health: probe failed: {e}")
            healthy = False
        elapsed = _time.monotonic() - t0

        if healthy:
            log.info(f"check_ollama_gpu_health: OK ({elapsed:.1f}s)")
            return

        log.error(f"check_ollama_gpu_health: {ANALYST_MODEL} unresponsive after "
                  f"{elapsed:.1f}s — likely stuck CPU-only again")
        restart_ollama_and_alert(f"{ANALYST_MODEL} unresponsive after {elapsed:.0f}s (periodic check)")
    except Exception as e:
        log.error(f"check_ollama_gpu_health failed: {e}")


_VECTOR_STORE_CHECK_SCRIPT = """
import sys
import vector_store as vs
for name in vs.COLLECTIONS:
    print(f"CHECKING:{name}", flush=True)
    col = vs._get_collection(name)
    n = col.count()
    print(f"OK:{name}:{n}", flush=True)
print("ALL_OK", flush=True)
"""


def _run_vector_store_check() -> tuple[set[str], set[str], "subprocess.CompletedProcess"]:
    """Runs _VECTOR_STORE_CHECK_SCRIPT in a fresh subprocess. Returns
    (collections that started but never reported OK, those that did, the
    finished process)."""
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, "-c", _VECTOR_STORE_CHECK_SCRIPT],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    checking = {l.split(":", 1)[1] for l in lines if l.startswith("CHECKING:")}
    ok = {l.split(":", 2)[1] for l in lines if l.startswith("OK:")}
    return checking - ok, ok, result


def _check_exit_note(result) -> str:
    """'exit -11 (segfault)' or 'exit 1: <last stderr line>' for the log."""
    rc = result.returncode
    if rc == -11:
        return "exit -11 (segfault)"
    tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
    return f"exit {rc}: {tail[0][:300]}"


# Two 120 s checks plus a rebuild (about 2.5 s per 4,000 documents) fit well inside.
_VECTOR_STORE_HEALTH_SOFT_LIMIT_S = 600


# A pending rebuild snapshot older than this gets a (non-digest) alert, once a day.
_VECTOR_STORE_SNAPSHOT_ALERT_HOURS = 24


def _log_vector_store_rebuild(report: dict) -> bool:
    """One activity_feed row (stage vector_store_rebuild) per rebuild the
    health check ran, skipped or saw fail, and per prune skipped for a
    pending snapshot: collection, docs_before (sqlite records read),
    docs_after (count() afterwards), method (rebuild, snapshot_resume,
    skipped_read_failed, skipped_zero_docs, already_resumed, resume_gave_up,
    resume_failed, rebuild_failed, prune_skipped_snapshot_pending),
    started_at, finished_at, resume_failures, snapshot_age_hours, and error
    when there is one. Never raises; returns whether a row was written."""
    try:
        import db
        extra = {k: report.get(k) for k in ("collection", "docs_before", "docs_after", "method",
                                             "started_at", "finished_at", "docs_found", "restored",
                                             "docs_without_text", "resume_failures",
                                             "snapshot_age_hours", "resume_mode", "upserted",
                                             "landed", "remaining", "dead_lettered")}
        if report.get("error"):
            extra["error"] = report["error"]
        method = str(report.get("method", ""))
        if method.endswith("_failed") and not method.startswith("skipped"):
            message = f"'{report.get('collection')}' {method.replace('_', ' ')}: {report.get('error')}"
        elif method == "prune_skipped_snapshot_pending":
            message = (f"'{report.get('collection')}' prune skipped: rebuild snapshot pending for "
                       f"{report.get('snapshot_age_hours')} h")
        elif method.startswith(("skipped", "already", "resume_gave_up")):
            message = f"'{report.get('collection')}' rebuild skipped ({method}): nothing deleted"
        else:
            message = (f"'{report.get('collection')}' rebuilt ({method}): "
                       f"{report.get('restored')}/{report.get('docs_found')} documents restored")
        bad = method != "already_resumed" and (
            method not in ("rebuild", "snapshot_resume") or report.get("error"))
        return bool(db.log_activity("vector_store_rebuild", "", message,
                                    "negative" if bad else "neutral", extra))
    except Exception as e:
        log.warning(f"check_vector_store_health: activity row for the "
                    f"'{report.get('collection')}' rebuild failed: {e}")
        return False


def _alert_set_aside(vs, name: str, n: int) -> None:
    """Critical alert for records a rebuild set aside, once a day per
    collection. Set-asides inside the throttle window are added up in
    system_state (vector_store_dead_letter_unalerted_<name>) and the next
    alert gives the total since the last one. With n=0 (the hourly flush in
    _alert_unfinished_rebuilds) it sends only a held total, once the
    throttle allows. Never raises."""
    try:
        import db
        key = f"vector_store_dead_letter_unalerted_{name}"
        try:
            total = int(db.get_state(key) or 0) + n
        except Exception:
            total = n
        if not total:
            return
        if not _vector_store_alert_due(f"dead_letter_{name}"):
            db.set_state(key, str(total))
            return
        import alerts
        this_run = f"{n} record(s) set aside in this run, {total}" if n else f"{total} record(s) set aside"
        alerts._send(
            f"⚠️ <b>Vector store records set aside</b>\n"
            f"'{name}': {this_run} since the last alert. "
            f"Each failed to upsert on {vs._MAX_RESUME_FAILURES} resumes in a row. Their text, "
            f"metadata and error are kept in .chromadb/.rebuild-{name}.deadletter.jsonl; restore "
            f"them manually. Until then they are not in the analyst's retrieval.")
        db.set_state(key, "0")
    except Exception as e:
        log.warning(f"check_vector_store_health: set-aside alert for '{name}' failed: {e}")


def _rebuild_and_log(vs, name: str, resume_only: bool = False) -> str | None:
    """Runs vs.rebuild_collection(name), writes its activity row, and
    returns the alert line (None when another run had already resumed it,
    or when retries have stopped: _alert_unfinished_rebuilds covers that).
    A rebuild that raises gets a resume_failed / rebuild_failed row, then
    the exception goes on."""
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc).isoformat()
    try:
        report = vs.rebuild_collection(name, resume_only=resume_only)
    except BaseException as e:
        _log_vector_store_rebuild({
            "collection": name, "method": "resume_failed" if resume_only else "rebuild_failed",
            "error": f"{type(e).__name__}: {e}"[:500], "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "resume_failures": vs.rebuild_state(name).get("failures")})
        raise
    _log_vector_store_rebuild(report)
    if report.get("method") == "skipped_count_failed" and _vector_store_alert_due(f"count_failed_{name}"):
        # Without the count, the three-failure bound can't hold: say so at once.
        import alerts
        alerts._send(
            f"⚠️ <b>Vector store rebuild blocked</b>\n"
            f"'{name}': the resume attempt could not be recorded ({report.get('error')}), "
            f"so it was not resumed. The snapshot is kept and prune stays paused for '{name}'. "
            f"Check free disk space under .chromadb.")
    if report.get("dead_lettered"):
        _alert_set_aside(vs, name, int(report["dead_lettered"]))
    if report.get("method") in ("already_resumed", "resume_gave_up", "skipped_count_failed"):
        return None
    if str(report.get("method", "")).startswith("skipped"):
        return f"{name} NOT rebuilt ({report['method']}: {report.get('error')}; nothing deleted)"
    line = f"{name} ({report.get('restored')}/{report.get('docs_found')} docs restored"
    if report.get("method") == "snapshot_resume":
        line += ", resumed from the snapshot of an unfinished rebuild"
    return line + (f"; {report['error']})" if report.get("error") else ")")


def _vector_store_alert_due(key: str) -> bool:
    """Once a day per key (system_state vector_store_alert_<key>). A state
    read or write failure counts as due: an extra alert beats none."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    try:
        import db
        last = db.get_state(f"vector_store_alert_{key}")
        if last and (now - datetime.fromisoformat(last)).total_seconds() < 24 * 3600:
            return False
        db.set_state(f"vector_store_alert_{key}", now.isoformat())
    except Exception as e:
        log.warning(f"check_vector_store_health: alert throttle state failed ({e}); alerting")
    return True


def _alert_unfinished_rebuilds(vs) -> list[str]:
    """For each pending rebuild snapshot older than
    _VECTOR_STORE_SNAPSHOT_ALERT_HOURS, or whose resumes have stopped after
    vs._MAX_RESUME_FAILURES failures, one alert a day. Its title matches no
    low-severity type in alert_policy, so it is sent at once, not digested:
    while a snapshot is pending, prune skips that collection and old
    documents stay in retrieval. Returns the collections alerted."""
    import alerts
    # Set-asides held back by the daily throttle go out once it allows,
    # without waiting for the next set-aside.
    for name in vs.COLLECTIONS:
        if vs._deadletter_path(name).exists():
            _alert_set_aside(vs, name, 0)
    sent = []
    for name in vs.pending_rebuilds():
        try:
            age = vs.snapshot_age_hours(name) or 0.0
            state = vs.rebuild_state(name)
            failures = int(state.get("failures") or 0)
            gave_up = failures >= vs._MAX_RESUME_FAILURES
            if not gave_up and age <= _VECTOR_STORE_SNAPSHOT_ALERT_HOURS:
                continue
            if not _vector_store_alert_due(f"{'gave_up' if gave_up else 'snapshot_old'}_{name}"):
                continue
            why = (f"resumes stopped after {failures} failures in a row" if gave_up
                   else f"{failures} failed resume(s) so far")
            alerts._send(
                f"⚠️ <b>Vector store rebuild unfinished</b>\n"
                f"'{name}': snapshot pending for {age:.0f} h; {why}.\n"
                f"Last error: {state.get('last_error') or 'none recorded'}\n"
                f"Prune is paused for '{name}' until it finishes, so documents past their age "
                f"limit stay in the analyst's retrieval. The snapshot is kept"
                + (f"; after fixing the cause, delete .chromadb/.rebuild-{name}.state.json to "
                   f"let the hourly health check retry." if gave_up else ".")
            )
            sent.append(name)
        except Exception as e:
            log.error(f"check_vector_store_health: unfinished-rebuild alert for '{name}' failed: {e}")
    return sent


@app.task(name="tasks.check_vector_store_health", bind=True, max_retries=0,
          soft_time_limit=_VECTOR_STORE_HEALTH_SOFT_LIMIT_S)
def check_vector_store_health(self):
    """
    A corrupted ChromaDB collection segfaults the whole process on a bare
    col.count() — happened to the "news" collection on 2026-07-17 (fixed
    with a rebuild + a cross-process write lock in vector_store.store) and
    again on 2026-07-21, three days later, despite that lock — a process
    that crashes/gets killed mid-write can still corrupt the on-disk index
    regardless of locking against concurrent writers. Checking this
    in-process isn't safe (the crash would take down this worker too), so
    each check runs in an isolated subprocess: if it dies, only that
    subprocess dies, and the last "CHECKING:<name>" line with no matching
    "OK:<name>" line tells us exactly which collection to rebuild.

    A failed collection is checked once more with the write lock held, so no
    write is in flight, and only one that fails twice is rebuilt. The rebuild
    (vector_store.rebuild_collection) puts the collection's documents back
    from sqlite; it used to delete them all. The subprocess's exit code and
    last stderr line are logged: before, every failure was reported as a
    segfault, including a plain InternalError from count().

    A rebuild that was killed partway leaves a snapshot of the collection's
    records (vector_store.pending_rebuilds); each run finishes those first.
    Every rebuild, and every one skipped because the records could not be
    read, writes an activity_feed row (stage vector_store_rebuild). A rebuild
    state sidecar left without its snapshot is removed first, so it can't
    keep prune off that collection.

    soft_time_limit: a rebuild cut off by it raises out of rebuild_collection
    with its snapshot kept, and the next run resumes it.
    """
    from celery.exceptions import SoftTimeLimitExceeded
    try:
        import vector_store as vs
        try:
            vs.clear_orphan_rebuild_states()
        except SoftTimeLimitExceeded:
            raise
        except Exception as e:
            log.error(f"check_vector_store_health: clearing orphan rebuild states failed: {e}")
        # Before the resumes too: a resume that kills this worker (a segfault
        # in count()) must not keep the alert from ever being sent. The
        # once-a-day throttle stops the second call repeating it.
        _alert_unfinished_rebuilds(vs)
        resumed = []
        for name in vs.pending_rebuilds():
            try:
                log.error(f"check_vector_store_health: '{name}' has an unfinished rebuild; resuming it")
                line = _rebuild_and_log(vs, name, resume_only=True)
                if line:
                    resumed.append(line)
            except SoftTimeLimitExceeded:
                raise
            except Exception as e:
                log.error(f"check_vector_store_health: failed to resume the '{name}' rebuild: {e}")
        if resumed:
            import alerts
            alerts._send(
                f"🔧 <b>Vector store auto-recovery</b>\n"
                f"Unfinished rebuild(s) resumed: {', '.join(resumed)}"
            )
        _alert_unfinished_rebuilds(vs)

        corrupted, ok, result = _run_vector_store_check()

        if not corrupted:
            if result.returncode != 0:
                log.warning(f"check_vector_store_health: subprocess exited {result.returncode} "
                            f"but every collection reported OK first — {result.stderr[-500:]}")
            else:
                log.info(f"check_vector_store_health: OK ({len(ok)} collections)")
            return

        first_note = _check_exit_note(result)
        log.warning(f"check_vector_store_health: {sorted(corrupted)} failed count in a fresh "
                    f"process ({first_note}); checking again with writes held")
        with vs._write_lock():
            corrupted_again, _, result2 = _run_vector_store_check()
        corrupted &= corrupted_again
        if not corrupted:
            log.warning("check_vector_store_health: passed on the second check; not rebuilding")
            return

        rebuilt = []
        for name in sorted(corrupted):
            try:
                log.error(f"check_vector_store_health: '{name}' failed count twice "
                          f"({_check_exit_note(result2)}) — rebuilding from its stored documents")
                rebuilt.append(_rebuild_and_log(vs, name))
            except SoftTimeLimitExceeded:
                raise
            except Exception as e:
                log.error(f"check_vector_store_health: failed to rebuild '{name}': {e}")

        if rebuilt:
            import alerts
            alerts._send(
                f"🔧 <b>Vector store auto-recovery</b>\n"
                f"Collection(s) rebuilt: {', '.join(rebuilt)}\n"
                f"Documents were re-embedded from the store's own sqlite records (semantic-memory "
                f"context, not trading state). Nothing else was affected."
            )
    except SoftTimeLimitExceeded:
        log.error(f"check_vector_store_health: stopped at the {_VECTOR_STORE_HEALTH_SOFT_LIMIT_S} s "
                  f"soft time limit; any unfinished rebuild keeps its snapshot and resumes next run")
    except Exception as e:
        log.error(f"check_vector_store_health failed: {e}")


def _log_outcome_checkpoint(o: dict, days: int, days_held: int, price: float) -> None:
    """One activity_feed row per checkpoint written, so when and from what
    each 30/60/90-day price was taken can be read back (trade_outcomes keeps
    only the price). Never raises."""
    try:
        import db
        from datetime import datetime, timezone
        db.log_activity("outcome_checkpoint", o.get("ticker") or "",
                        f"{days}d checkpoint written (held {days_held}d)", "neutral",
                        {"outcome_id": o.get("id"), "approval_id": o.get("approval_id"),
                         "env": o.get("trading_env"), "checkpoint_days": days,
                         "days_held": days_held, "price": round(float(price), 4),
                         "source": "yfinance fast_info.last_price",
                         "entry_date": str(o.get("entry_date")),
                         "written_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        log.warning(f"[{o.get('ticker')}] checkpoint activity row failed: {e}")


@app.task(name="tasks.review_trade_outcomes", bind=True, max_retries=2)
def review_trade_outcomes(self):
    """
    Record 30/60/90-day prices for open trade outcomes.
    Uses >= threshold (not an exact-day match) so a missed run — scheduler
    downtime, a holiday, the box being off — doesn't permanently skip a
    checkpoint; each checkpoint is only ever written once (guarded by the
    review_Xd_price column already being NULL).

    Covers the open rows of both envs (paper rows went unreviewed after the
    switch to live). Price checkpoints only: the price is market data
    (yfinance), the write goes to that row by id, and nothing here reads the
    account or the broker or touches stops or positions.

    Then notes orders still 'submitted' hours after they were sent
    (db.mark_stale_submitted_orders: reconcile_note only, never status).
    """
    try:
        import db
        import yfinance as yf
        from datetime import date

        # days_held keeps its original basis (local date minus entry_date's
        # UTC date): the checkpoint day is part of how outcomes are measured.
        outcomes = db.get_open_outcomes(all_envs=True)
        prices: dict[str, float | None] = {}   # one fetch per ticker per run
        for o in outcomes:
            if not o.get("entry_date"):
                continue
            days_held = (date.today() - o["entry_date"].date()).days

            pending_checkpoints = [
                days for days, col in (
                    (30, "review_30d_price"),
                    (60, "review_60d_price"),
                    (90, "review_90d_price"),
                )
                if days_held >= days and o.get(col) is None
            ]
            if not pending_checkpoints:
                continue

            if o["ticker"] not in prices:
                try:
                    prices[o["ticker"]] = yf.Ticker(o["ticker"]).fast_info.last_price
                except Exception as e:
                    log.warning(f"[{o['ticker']}] price fetch failed: {e}")
                    prices[o["ticker"]] = None
            price = prices[o["ticker"]]
            if not price:
                continue

            for days in pending_checkpoints:
                written = db.update_outcome_price(o["approval_id"], days, price,
                                                  outcome_id=o.get("id"))
                if not written:
                    log.warning(f"[{o['ticker']}] {days}d checkpoint not written "
                                f"(outcome {o.get('id')}, approval {o['approval_id']}): "
                                f"{'write failed' if written is None else 'no open row with it unset'}")
                    continue
                log.info(f"[{o['ticker']}] {days}d checkpoint (held {days_held}d): ${price:.2f}")
                _log_outcome_checkpoint(o, days, days_held, price)

        # Record-keeping: note orders still 'submitted' hours after they were
        # sent (reconcile_note only; status and the broker are not touched).
        try:
            noted = db.mark_stale_submitted_orders()
            if noted:
                log.warning(f"review_trade_outcomes: {noted} order row(s) still 'submitted' "
                            f"noted as {db.STALE_SUBMITTED_NOTE}")
        except Exception as e:
            log.warning(f"review_trade_outcomes: stale submitted-order note failed: {e}")
    except Exception as e:
        log.error(f"review_trade_outcomes failed: {e}")
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.prune_vector_store", bind=True, max_retries=1)
def prune_vector_store(self):
    """Age-based pruning of ChromaDB collections to keep context fresh."""
    try:
        import vector_store
        # (collection, max_age_days)
        for collection, max_age in vector_store.PRUNE_MAX_AGE_DAYS.items():
            try:
                age = vector_store.snapshot_age_hours(collection)
                if age is not None:
                    # prune_old_documents would skip it too (under the lock);
                    # this leaves a row so the pause is visible.
                    log.warning(f"prune_vector_store: '{collection}' has a rebuild snapshot pending "
                                f"for {age:.1f} h; not pruned")
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc).isoformat()
                    _log_vector_store_rebuild({
                        "collection": collection, "method": "prune_skipped_snapshot_pending",
                        "snapshot_age_hours": round(age, 1), "started_at": now, "finished_at": now})
                    continue
                n = vector_store.prune_old_documents(collection, max_age_days=max_age)
                if n:
                    log.info(f"Pruned {n} docs from '{collection}' (>{max_age}d old)")
            except Exception as e:
                log.warning(f"Could not prune '{collection}': {e}")
        # arxiv is intentionally not pruned (papers are timeless reference material)
    except Exception as e:
        log.error(f"prune_vector_store failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.prune_seen_items", bind=True, max_retries=1)
def prune_seen_items(self):
    """seen_items has no TTL otherwise — a row per distinct news/filing/
    insider/options item ever ingested (75 tickers x multiple sources,
    indefinitely) piles up on the Supabase free tier with nothing ever
    reclaiming it."""
    try:
        import db
        n_general = db.prune_seen_items(days=90)
        n_options = db.prune_seen_items(days=7, source="options_flow_dispatch")
        n_chat = db.prune_chat_history(days=30)
        log.info(f"Pruned seen_items: {n_general} general (>90d), {n_options} options dispatch (>7d), "
                 f"{n_chat} chat_history rows (>30d)")
    except Exception as e:
        log.error(f"prune_seen_items failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.weekly_self_review", bind=True, max_retries=1)
def weekly_self_review(self):
    """
    Weekly performance self-review: closed-trade outcomes + alpha, conviction
    calibration, bucket attribution, slippage, ingestion health — synthesized
    into a narrative by the analyst model, sent to Telegram and stored in
    ChromaDB. This is the closest a solo operator gets to an institutional
    performance-review cadence; the human reads it and adjusts
    knowledge/*.md or config.py accordingly.
    """
    try:
        import db
        import alerts
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        all_outcomes = db.get_all_outcomes(limit=500)
        week_closed = [
            o for o in all_outcomes
            if o.get("exit_date") and o["exit_date"] >= cutoff
        ]
        wins = sum(1 for o in week_closed if o.get("was_correct"))
        alphas = [o["alpha_pct"] for o in week_closed if o.get("alpha_pct") is not None]
        avg_alpha = sum(alphas) / len(alphas) if alphas else None

        calibration = db.get_conviction_calibration()
        attribution = db.get_bucket_attribution()
        slippage = db.get_slippage_summary()
        ingestion_issues = _get_stale_ingestion_sources()

        context_lines = [
            f"=== WEEK OF {datetime.now(timezone.utc).strftime('%Y-%m-%d')} ===",
            f"Closed trades this week: {len(week_closed)}, wins: {wins}"
            + (f" ({wins/len(week_closed):.0%} win rate)" if week_closed else ""),
        ]
        if avg_alpha is not None:
            context_lines.append(f"Average alpha vs SPY on closed trades: {avg_alpha:+.1%}")
        for o in week_closed[:15]:
            context_lines.append(
                f"  {o['ticker']}: {o['action']} conviction={o['conviction']}, "
                f"outcome={o.get('outcome_pct')}, alpha={o.get('alpha_pct')}"
            )

        context_lines.append("\n=== CONVICTION CALIBRATION (all-time) ===")
        for c in calibration:
            context_lines.append(
                f"  Conviction {c['conviction']}/10: {c['total']} trades, "
                f"{c['win_rate']:.0%} win rate, avg alpha {c.get('avg_alpha_pct')}"
            )

        context_lines.append("\n=== BUCKET ATTRIBUTION (all-time) ===")
        for a in attribution:
            context_lines.append(
                f"  {a['thesis_bucket']}: {a['closed_trades']} trades, "
                f"${a['dollar_pnl']:+,.2f} P&L on ${a['dollar_deployed']:,.2f} deployed"
            )

        context_lines.append(f"\n=== EXECUTION ===\nSlippage: {slippage}")
        if ingestion_issues:
            context_lines.append(f"\n=== DATA HEALTH ISSUES ===\n" + "\n".join(ingestion_issues))

        context = "\n".join(context_lines)

        prompt = (
            "You are reviewing this trading system's performance for the past week. "
            "Write a concise, honest self-review (under 250 words) covering: "
            "1) What worked and what didn't this week, 2) whether conviction calibration "
            "looks accurate or overconfident, 3) whether any bucket is not paying for its risk, "
            "4) one concrete, specific suggestion for the human operator to consider adjusting "
            "(e.g. a knowledge file, a config threshold, or a data source). "
            "Be specific and numeric where the data supports it. Do not pad with generic advice.\n\n"
            f"{context}"
        )

        summary = None
        try:
            import requests
            from config import ANALYST_MODEL, OLLAMA_HOST
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": ANALYST_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.3, "num_predict": 500, "num_ctx": 8192}},
                # Same 400s as analyst.py: 72B calls take ~180-195s, and 180
                # timed out the 2026-10-11 review.
                timeout=400,
            )
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()
        except Exception as e:
            log.warning(f"Weekly self-review LLM call failed, sending raw stats instead: {e}")

        if not summary:
            summary = context[:1500]

        alerts._send(f"🗓️ <b>Weekly Self-Review</b>\n\n{summary}")

        try:
            import vector_store
            vector_store.store(
                collection="self_review",
                doc_id=f"self_review_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                text=summary,
                metadata={"type": "weekly_self_review"},
            )
        except Exception as e:
            log.warning(f"Could not store weekly self-review in vector memory: {e}")

    except Exception as e:
        log.error(f"weekly_self_review failed: {e}")
        raise self.retry(exc=e, countdown=600)
