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
from celery.signals import worker_init


@worker_init.connect
def _pin_code_stamp(**_):
    """Import the analyst (which reads the commit stamp) in the worker's
    parent at boot, before the pool forks, so the stamp and the analyst code
    come from the files present at start, not whatever is on disk at the
    first analysis (professor M75 a)."""
    try:
        import agents.analyst  # noqa: F401
    except Exception as e:
        log.warning(f"could not pre-import agents.analyst at worker start: {e}")


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
                             deadline: float | None = None) -> dict:
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    return ibkr.place_stop_order(ticker, qty, stop_price)


@app.task(name="tasks._broker_cancel_stop_orders_for", bind=True, max_retries=1)
def _broker_cancel_stop_orders_for(self, ticker: str, deadline: float | None = None):
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    return ibkr.cancel_stop_orders_for(ticker)


@app.task(name="tasks._broker_cancel_order", bind=True, max_retries=1)
def _broker_cancel_order(self, order_id: int, deadline: float | None = None):
    if not _mark_started(self) or _past(deadline):
        return _EXPIRED
    import ibkr
    return ibkr.cancel_order(order_id)


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


def manage_stop_place(ticker: str, qty: float, stop_price: float, timeout: int = 30) -> dict:
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
    Callers that convert to a trail set it again via update_trailing_stop."""
    stop_lock.assert_held(ticker)  # professor M69
    if _is_orders_worker():
        import ibkr
        result = ibkr.place_stop_order(ticker, qty, stop_price)
    else:
        result = _orders_call(_broker_place_stop_order, [ticker, qty, stop_price],
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


def manage_cancel_order(order_id: int, timeout: int = 30) -> bool:
    """Cancel a specific order by id — see manage_stop_place. Also used for
    the stale-market-BUY-order cleanup at startup (main.py), not just stops:
    the same ownership rule applies to any order type, not only STP."""
    if _is_orders_worker():
        import ibkr
        return ibkr.cancel_order(order_id)
    return _orders_call(_broker_cancel_order, [order_id], timeout=timeout)


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
    # actually runs via catch-up Monday 05:00 (~26h later).
    "prune-vector-store": {
        "task":     "tasks.prune_vector_store",
        "schedule": crontab(hour=3, minute=0, day_of_week="sun"),
        "options":  {"expires": 36 * 3600},
    },
    # seen_items dedup table — same slot as vector store pruning, same
    # down-all-weekend caveat (runs via Monday 05:00 catch-up).
    "prune-seen-items": {
        "task":     "tasks.prune_seen_items",
        "schedule": crontab(hour=3, minute=15, day_of_week="sun"),
        "options":  {"expires": 36 * 3600},
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
    # S&P 500 universe price history — weekly, Saturday 9:00 AM. Same
    # down-all-weekend caveat as prune-vector-store/prune-seen-items above:
    # actually runs via Monday 05:00 catch-up (~44h later), which is fine —
    # feeds GICS-sector breadth + candidate_scanner, neither of which needs
    # day-fresh data (see refresh_sp500_history's docstring).
    "refresh-sp500-history": {
        "task":     "tasks.refresh_sp500_history",
        "schedule": crontab(hour=9, minute=0, day_of_week="sat"),
        "options":  {"queue": INGESTION_QUEUE, "expires": 48 * 3600},
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
    # Daily digest — weekdays 4:15 PM ET (after market close). A digest
    # delivered the next morning is noise — short expiry.
    "daily-digest": {
        "task":     "tasks.send_daily_digest",
        "schedule": crontab(hour=16, minute=15, day_of_week="mon-fri"),
        "options":  {"expires": 3 * 3600},
    },
    # Weekly self-review — Sunday 6:00 PM ET (stack is down then; catch-up
    # runs it Monday 05:00). LLM-heavy → llm bulk lane.
    "weekly-self-review": {
        "task":     "tasks.weekly_self_review",
        "schedule": crontab(hour=18, minute=0, day_of_week="sun"),
        "options":  {"queue": LLM_QUEUE_BULK, "expires": 24 * 3600},
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
    if not row or not row[0] or not row[1]:
        return {}
    ts = row[1] if row[1].tzinfo else row[1].replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - ts).total_seconds() > max_age_min * 60:
        return {}
    return row[0]


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


@app.task(name="tasks.screen_event", bind=True, max_retries=1)
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


@app.task(name="tasks.deep_analyze_event", bind=True, max_retries=1)
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
def retry_deferred_entry(self, approval_id: int, gap_retry_attempt: int = 1, price_retry_attempt: int = 1):
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
    than deferring indefinitely.

    Approvals older than DEFERRED_ENTRY_MAX_AGE_HOURS are expired and sent
    back through a fresh 72B analysis instead (2026-10-27). Younger ones do
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

    ticker = approval.get("ticker", "?")

    # Stale decisions get re-analysed instead of executed (2026-10-27,
    # limitations audit finding 13). A Friday-evening approval used to fill at
    # Monday's open on 60-hour-old reasoning, guarded only by price drift.
    age_h = _approval_age_hours(approval)
    if age_h is not None and age_h > DEFERRED_ENTRY_MAX_AGE_HOURS:
        db.update_approval_status(approval_id, "EXPIRED",
                                  reason=f"deferred entry {age_h:.0f}h old "
                                         f"(> {DEFERRED_ENTRY_MAX_AGE_HOURS}h), re-analysed")
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

    success = execute_approved(
        approval, trigger="auto_retry",
        gap_retry_attempt=gap_retry_attempt, price_retry_attempt=price_retry_attempt,
    )
    if success:
        db.update_approval_status(approval_id, "APPROVED", reason=resized)
    else:
        db.update_approval_status(approval_id, "FAILED", reason=last_outcome_reason())
    log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} "
             f"{'executed' if success else 'still failed'} on retry "
             f"(gap_attempt={gap_retry_attempt}, price_attempt={price_retry_attempt})")


DEFERRED_ENTRY_MAX_AGE_HOURS = 16


def _deferred_buy_shares_now(approval: dict) -> int | None:
    """Shares a deferred OPEN/ADD would get if sized now (sizing + concentration
    caps on the live account, at the price it was approved at). None when it
    isn't a buy or the re-check itself fails (then the approved size stands)."""
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
        return rm.shares_after_concentration(approval["ticker"], sizing, bucket)
    except Exception as e:
        log.warning(f"retry_deferred_entry: [{approval.get('ticker')}] re-size failed, "
                    f"keeping the approved size: {e}")
        return None


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


def _reconcile_unconfirmed_orders_locked(task):
    import db
    import ibkr
    import alerts
    from datetime import timedelta

    try:
        pending = db.get_unconfirmed_orders(max_age_hours=24 * 30)
    except Exception as e:
        log.error(f"reconcile_unconfirmed_orders: could not load unconfirmed orders: {e}")
        raise task.retry(exc=e, countdown=120)

    _retry_pending_reprotects()

    for order in pending:
        ticker = order["ticker"]
        try:
            # The row is written after the fill wait, so a partial can
            # execute before created_at: allow a margin.
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
            continue  # still genuinely unfilled (or cancelled/rejected) -- leave alone

        # A 'partial' order already has some shares recorded: log only the
        # new ones, at their own average price (professor M-4).
        total_qty, total_avg = fill["filled_qty"], fill["fill_price"]
        prev_qty = order.get("filled_qty") or 0.0
        fill_qty = round(total_qty - prev_qty, 4)
        if fill_qty <= 0:
            continue
        fill_price = ((total_qty * total_avg - prev_qty * (order.get("fill_price") or total_avg))
                      / fill_qty) if prev_qty else total_avg
        complete = order.get("qty") is None or total_qty >= float(order["qty"]) - 1e-6
        log.warning(f"[{ticker}] Late fill found for order {order['broker_order_id']}: "
                    f"{order['side']} {fill_qty} @ ${fill_price:.2f} — was recorded "
                    f"unconfirmed, reconciling now"
                    + ("" if complete else f" ({total_qty:g} of {order['qty']:g} filled)"))

        intent = order.get("intent")
        # Mark the order first: if that write fails, stop here, or the next
        # cycle would log the same trade again and re-place a BUY stop
        # (professor M51). A failed log_trade after this is spilled.
        if db.update_order_fill(order["id"], fill_price=total_avg, filled_qty=total_qty,
                                status="filled" if complete else "partial",
                                expect_filled_qty=prev_qty) is False:
            log.error(f"[{ticker}] reconcile: could not mark order {order['id']} filled; "
                      f"trade not logged this cycle")
            continue
        db.log_trade(
            symbol=ticker, side=order["side"], quantity=fill_qty, price=fill_price,
            reason=(f"{intent + ' ' if intent else ''}Late-fill reconciled (order placed "
                    f"unconfirmed, filled after the wait window)"),
            price_source="fill",
        )

        stop_placed: bool | None = False
        approval = db.get_approval_by_id(order["approval_id"]) if order["approval_id"] else None
        if order["side"] == "BUY" and approval and not prev_qty:
            # Nothing of this order was recorded at the fill, so neither was
            # its outcome row (professor M123). A part fill already has one.
            try:
                from execution import log_buy_outcome
                log_buy_outcome(approval, ticker, approval.get("action") or intent or "OPEN", total_avg)
            except Exception as e:
                log.error(f"[{ticker}] reconcile: could not log the late BUY's outcome row: {e}")
        elif order["side"] == "BUY" and approval and prev_qty:
            # The row was logged at the part fill's price: re-average it over
            # the whole order (professor M121, entry side).
            try:
                db.update_outcome_entry_price(approval["id"], total_avg)
            except Exception as e:
                log.error(f"[{ticker}] reconcile: could not re-average the outcome entry: {e}")
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
                    if not db.get_position_stop(ticker) or not _held_before_fill(
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
            if approval.get("status") == "FAILED":
                db.update_approval_status(order["approval_id"], "APPROVED")
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
                if not db.get_position_stop(ticker) or not _held_before_fill(ticker, fill_qty, action):
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
        elif order["side"] == "SELL":
            # A late SELL fill (partial sale or TRIM whose stop was cancelled
            # before the sale) leaves the rest without a stop. Re-protect it at
            # the recorded stop (professor C12, 2026-10-28).
            stop_placed = _reprotect_after_late_sell(ticker, order, fill_price, exit_avg=total_avg)
            if stop_placed is False:
                # Busy ticker or a failed placement: the order is already
                # marked filled, so remember to retry (professor M50).
                _queue_reprotect(ticker, order, fill_price)

        alert_lines = [
            f"🔁 LATE FILL RECONCILED — <b>{ticker}</b>",
            f"{order['side']} {fill_qty:g} shares @ ${fill_price:.2f} — this order was marked "
            f"unconfirmed but had actually filled at the broker.",
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


_REPROTECT_STATE_KEY = "reprotect_pending"


def _reprotect_queue_edit(fn) -> None:
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
        lock = stop_lock._redis().lock("helios:reprotect_queue", timeout=30, blocking_timeout=35)
        if not lock.acquire():
            lock = None
            log.error("reprotect queue lock still busy after 35 s; editing without it")
    except Exception as e:
        lock = None
        log.warning(f"reprotect queue lock unavailable ({e}); editing without it")
    try:
        try:
            pending = json.loads(db.get_state(_REPROTECT_STATE_KEY) or "{}")
        except Exception:
            pending = {}
        db.set_state(_REPROTECT_STATE_KEY, json.dumps(fn(pending)))
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
             "fill_price": fill_price, "nonce": uuid.uuid4().hex}

    def add(pending):
        old = pending.get(ticker) or {}
        if old.get("intent") == "PROFIT_TARGET" and entry["intent"] != "PROFIT_TARGET":
            # Keep the target sale's breakeven-or-higher stop and its
            # target-taken step; a later RESTORE/BUY must not replace them
            # with the lower stop on record (professor M80).
            pending[ticker] = {**{k: v for k, v in old.items() if k != "tries"},
                               "nonce": entry["nonce"]}
        else:
            pending[ticker] = entry
        return pending
    _reprotect_queue_edit(add)


_MANUAL_ACTIONS_KEY = "manual_actions"
_MANUAL_REALERT_S = 3600
_MANUAL_FLAT_ONLY_READS = 24  # cycles (~6 h) before a clear_on_flat=False item clears on flat


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


def open_manual_action(ticker: str, what: str, clear_on_flat: bool = True) -> None:
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
    now = time.time()
    cleared, realert, flat_once = [], [], []
    for ticker, rec in items.items():
        qty = held.get(ticker, 0.0)
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


_REPROTECT_MAX_TRIES = 4  # reconcile_stop_orders cycles, 15 min apart (~1 h)
_BUY_FLAT_MAX_TRIES = 2   # ~30 min for a new BUY's position to show up


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
        tries = int(rec.get("tries", 0)) + 1
        # A BUY that still reads flat is retried for 2 cycles (~30 min; they
        # are 15 min apart, so the old 6 was ~90 min, professor m55), then
        # handed to Luke like a failed retry (professor M60).
        flat_buy = result is None and rec.get("intent") == "BUY"
        if flat_buy and tries >= _BUY_FLAT_MAX_TRIES:
            log.error(f"[{ticker}] reconcile: queued BUY stop still reads flat after {tries} tries")
            try:
                import alerts
                alerts._send(f"🚨 STOP NOT PLACED — <b>{ticker}</b>\nAfter a BUY the broker still "
                             f"shows no position (~30 min), so no stop was placed. Check TWS: if the "
                             f"shares are there, place the stop by hand.")
            except Exception:
                pass
            open_manual_action(ticker, "the stop after a BUY was never placed", clear_on_flat=False)
            outcome[ticker] = (rec, None)
        elif result is False and tries >= _REPROTECT_MAX_TRIES:
            # ~1 h of failures: stop retrying (each try cancels and re-places)
            # and hand it to Luke (professor M79).
            log.error(f"[{ticker}] reconcile: queued re-protect failed {tries} times; giving up")
            try:
                import alerts
                alerts._send(f"🚨 STOP NOT RESTORED — <b>{ticker}</b>\nThe automatic stop retry "
                             f"failed {tries} times (~1 h) and has stopped. Check the position and "
                             f"place a stop in TWS; the stop check keeps flagging it until covered.")
            except Exception:
                pass
            open_manual_action(ticker, "the automatic stop retry gave up")
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


@locked_by_ticker(lambda ticker, *a, **k: ticker, on_busy=False)
def _reprotect_after_late_sell(ticker: str, order: dict | None = None,
                               fill_price: float | None = None,
                               from_queue: bool = False,
                               expect_at_least: float | None = None,
                               exit_avg: float | None = None) -> bool | None:
    """After a late SELL fill, finish what the selling path would have done
    (professor M-3, 2026-10-28):
    - nothing left: delete the stop record and close the outcome at the fill
      price, so stale-stop cleanup doesn't log the exit a second time;
    - a profit-target sale: mark the target taken (no second sale) and protect
      the rest at the stop that sale computed (breakeven or higher);
    - otherwise: cancel resting stops and protect the rest at the stop on record.
    Returns True if a stop was placed, False if it should have been but failed,
    None if nothing is held."""
    import db
    import ibkr
    order = order or {}
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
        # stale-stop branch, which logs the real exit (A2253-2).
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
                                       exit_price_source="fill")   # broker average fill
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
            lines = [f"Watchlist candidate scan: {result['scan_notes']}", ""]
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
    try:
        import alerts
        from risk_manager import RiskManager

        client = _get_client()
        rm = RiskManager(client)
        sold = rm.check_profit_targets()
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
        log.error(f"check_profit_targets failed: {e}")
        raise self.retry(exc=e, countdown=60)


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


@locked_by_ticker(lambda ticker, *a, **k: ticker, on_busy=False)
def _execute_backstop_sell(ticker: str, alert: dict, _is_retry: bool = False) -> bool:
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
            db.delete_position_stop(ticker)
            return False

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
        # above the market (TWS) would not sell into a stop breach.
        working = sum(float(o["qty"]) for o in _resting_sell_stops(ticker, types=("MKT",)))
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
        db.log_trade(symbol=ticker, side="SELL", quantity=filled, price=fill_price,
                     price_source="fill" if result.get("fill_price") else "quote",
                     reason=f"Backstop stop-breach sell (stop ${alert['stop_price']:.2f})"
                            + (" [retry]" if _is_retry else ""))
        # An orders row like every other sale: a part fill stays 'partial' so
        # reconcile_unconfirmed_orders closes the outcome from the cumulative
        # fill once flat, and the stale-stop lookup leaves these fills out
        # (professor, review of c13548c).
        try:
            local_id = db.log_order(ticker=ticker, side="SELL", order_type="MARKET", qty=owned,
                                    broker_order_id=result.get("order_id"),
                                    decision_price=alert.get("current_price"), status="submitted",
                                    intent="BACKSTOP")
            if local_id and local_id > 0:
                db.update_order_fill(local_id, result.get("fill_price"), filled,
                                     "filled" if filled >= owned - 1e-6 else "partial")
        except Exception as e:
            log.warning(f"[{ticker}] Backstop sale order logging failed: {e}")
        if resting <= 0:  # shares under a stuck stop are still held
            db.delete_position_stop(ticker)
            if filled < owned - 1e-6:
                # Part filled: this part's price isn't the exit (professor
                # M121). The rest is still working at the broker; the order
                # is 'partial', so reconcile closes the row from the
                # cumulative fill once flat.
                log.warning(f"[{ticker}] Backstop sell part-filled ({filled:g} of {owned:g}); "
                            f"outcome left open")
            else:
                try:
                    db.close_trade_outcome(ticker, exit_price=fill_price, exit_source="backstop_sell",
                                           exit_price_source="fill" if result.get("fill_price") else "quote")
                except Exception as e:
                    log.error(f"[{ticker}] close_trade_outcome failed after the backstop sell: {e}")  # m98
        log.warning(f"[{ticker}] BACKSTOP SELL executed: {filled} shares @ ~${fill_price:.2f}")
        return True
    except Exception as e:
        log.error(f"[{ticker}] Backstop sell FAILED: {e}")
        if not _is_retry:
            _schedule_backstop_retry(ticker, alert)
        return False


@app.task(name="tasks.retry_backstop_sell", bind=True, max_retries=0)
def retry_backstop_sell(self, ticker: str, alert: dict):
    """Retry for a backstop stop-breach sell that _execute_backstop_sell
    couldn't complete on the first attempt -- see that function's docstring
    for the AMD incident this closes the loop on. Deliberately does not
    re-check whether the stop condition still holds: if this fired, the
    position already has zero broker-side protection, and the only open
    question is whether the sell can go through now."""
    success = _execute_backstop_sell(ticker, alert, _is_retry=True)
    if success:
        log.info(f"[{ticker}] Backstop sell succeeded on retry")
    else:
        log.error(f"[{ticker}] Backstop sell retry ALSO failed — still unprotected")


@app.task(name="tasks.check_trailing_stops", bind=True, max_retries=2)
def check_trailing_stops(self):
    """Monitor open positions against their stop levels. Alert on triggers;
    if the stop level is breached and NO protective stop exists at the
    broker, execute a market sell (AUTO_EXECUTE_PROTECTIVE_SELLS)."""
    try:
        from risk_manager import RiskManager
        import alerts

        client = _get_client()
        rm = RiskManager(client)
        triggered = rm.check_trailing_stops()
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
        try:
            open_stop_qty: dict[str, float] = {}
            for o in ibkr.get_open_orders():
                if o["action"] == "SELL" and o["order_type"] in ("STP", "STP LMT", "TRAIL"):
                    open_stop_qty[o["ticker"]] = open_stop_qty.get(o["ticker"], 0.0) + o["qty"]
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
                        log.warning(f"[{ticker}] Cached stop covers {stop_qty} shares but "
                                    f"{held_qty} are actually held — treating as stale/unreliable")
                    backstopped = _execute_backstop_sell(ticker, alert)
            alerts.send_stop_triggered(
                ticker=ticker,
                current_price=alert["current_price"],
                stop_price=alert["stop_price"],
                reason=alert["reason"] + (" — BACKSTOP SELL EXECUTED" if backstopped else ""),
            )
    except Exception as e:
        log.error(f"check_trailing_stops failed: {e}")
        raise self.retry(exc=e, countdown=60)


# ── Existing tasks (updated call signatures) ──────────────────────────────────

def _heartbeat_ok(source: str, count: int) -> None:
    try:
        import db
        db.record_ingestion_success(source, count)
    except Exception:
        pass


def _heartbeat_fail(source: str, error: str) -> None:
    try:
        import db
        db.record_ingestion_error(source, error)
    except Exception:
        pass


@app.task(name="tasks.poll_edgar", bind=True, max_retries=3)
def poll_edgar(self):
    """Polls SEC EDGAR for new filings and routes through agent pipeline.
    One run at a time: a pass takes ~11 min on a 10 min schedule, and
    overlapping runs doubled the request rate against SEC's limit (A1053-4)."""
    try:
        from ingestion.edgar import poll_once

        def run():
            filings = poll_once(on_filing=lambda f: _dispatch_event(f, seen_source="edgar"))
            log.info(f"EDGAR: {len(filings)} new filing(s)")
            _heartbeat_ok("edgar", len(filings))
        _run_singleton("poll_edgar", run, max_hold_s=1200)
    except Exception as e:
        log.error(f"poll_edgar failed: {e}")
        _heartbeat_fail("edgar", str(e))
        raise self.retry(exc=e, countdown=60)


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
        if os.getenv("UNUSUAL_WHALES_API_KEY", "").startswith("your_"):
            from ingestion.options_free import get_all_flow_summaries
            log.info("UNUSUAL_WHALES_API_KEY not configured — using free yfinance options proxy")
        else:
            from ingestion.options_flow import get_all_flow_summaries

        import db
        from datetime import date

        summaries = get_all_flow_summaries()
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
        from ingestion.insider import poll_once

        filings = poll_once()
        log.info(f"Insider: {len(filings)} new open-market buy filing(s)")
        for filing in filings:
            # insider.py's own is_seen/mark_seen use source="insider", but
            # each filing's payload carries source="sec_form4" (what the LLM
            # sees) — must pass the dedup source explicitly or _dispatch_event
            # would default to the wrong one and the is_seen check would
            # never match what gets marked.
            _dispatch_event(filing, seen_source="insider")
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


@app.task(name="tasks.refresh_price_history", bind=True, max_retries=2)
def refresh_price_history(self):
    """Nightly refresh of the price_history cache (watchlist + SPY/QQQ)."""
    try:
        from ingestion.price_history import refresh_all_price_history
        count = refresh_all_price_history()
        log.info(f"Price history refreshed: {count} tickers updated")
        _heartbeat_ok("price_history", count)
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
        count = refresh_sp500_price_history()
        log.info(f"S&P 500 price history refreshed: {count} tickers updated")
        _heartbeat_ok("sp500_history", count)
    except Exception as e:
        log.error(f"refresh_sp500_history failed: {e}")
        _heartbeat_fail("sp500_history", str(e))
        raise self.retry(exc=e, countdown=600)


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
    expected_minutes = {
        "edgar": 20, "options_flow": 15, "macro": 90, "arxiv": 1560,
        "congressional": 400, "insider": 60, "news": 30, "fundamentals": 1560,
        "price_history": 1560, "sp500_history": 10080, "analyst_ratings": 60,
        "candidate_scan": 10080,
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

        # Day P&L: compare against the first snapshot recorded today
        history = db.get_portfolio_history(limit=500)
        today_str = date.today().isoformat()
        todays = [h for h in history if h["timestamp"].date().isoformat() == today_str]
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
            tomorrow = (date.today() + timedelta(days=1)).isoformat()
            events = query_events(start=tomorrow, end=tomorrow)
            catalysts_text = [f"{e.get('ticker')}: {e.get('event_type')}" for e in events]
        except Exception as e:
            log.debug(f"Catalyst lookup unavailable for digest: {e}")

        pending_count = len(db.get_pending_approvals())
        from config import TRADING_ENV as _TRADING_ENV
        halted = db.get_state(f"trading_halted_{_TRADING_ENV}", "false") == "true"
        ingestion_issues = _get_stale_ingestion_sources()

        alerts.send_daily_digest(
            portfolio_value=pv, day_pl=day_pl, day_pl_pct=day_pl_pct,
            spy_pl_pct=spy_pl_pct, positions_text=positions_text,
            catalysts_text=catalysts_text, pending_count=pending_count,
            halted=halted, ingestion_issues=ingestion_issues,
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


IMPLAUSIBLE_READS_BEFORE_HWM_RESET = 3


@app.task(name="tasks.check_portfolio_drawdown", bind=True, max_retries=2)
def check_portfolio_drawdown(self):
    """
    Circuit breaker: sets the persisted trading_halted flag when drawdown
    from the high-water mark exceeds MAX_DRAWDOWN_PCT, and maintains the HWM.
    """
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

        client = _get_client()
        pv     = float(client.get_account().portfolio_value)

        # Maintain the persisted high-water mark (seed from START_VALUE,
        # which is itself TRADING_ENV-derived — see config.py)
        hwm = float(db.get_state(hwm_key) or START_VALUE)

        # Plausibility guard: a transient broker read of $0 (e.g. an IBKR
        # clientId collision or cold-morning connection race) previously
        # computed as a 100% drawdown and latched the halt flag until a
        # manual /resume — reject readings that can't be a real single-step
        # move instead of trusting them.
        if pv <= 0 or pv < 0.5 * hwm:
            log.warning(f"Implausible portfolio value ${pv:,.2f} (HWM ${hwm:,.2f}) — "
                       f"skipping drawdown check, not updating HWM")
            db.set_state(breach_key, "0")
            # A persistent "implausible" read is a stale baseline, not noise:
            # the halt fires at MAX_DRAWDOWN_PCT long before a real 50% loss.
            # From 2026-10-11 the live HWM was the paper default ($1M), so this
            # branch skipped every check and the circuit breaker was off
            # (limitations audit 2026-10-28). After 3 consecutive reads (90
            # min), reset the HWM to the current value and say so.
            implausible = int(db.get_state(implausible_key, "0") or 0) + 1 if pv > 0 else 0
            db.set_state(implausible_key, str(implausible))
            if implausible >= IMPLAUSIBLE_READS_BEFORE_HWM_RESET:
                db.set_state(hwm_key, str(pv))
                db.set_state(implausible_key, "0")
                log.critical(f"HWM {hwm_key} reset from ${hwm:,.2f} to ${pv:,.2f} after "
                             f"{implausible} consecutive implausible reads")
                try:
                    alerts._send(
                        f"⚠️ Drawdown baseline reset ({TRADING_ENV})\n"
                        f"The high-water mark was more than twice the account value on "
                        f"{implausible} checks in a row, so the drawdown halt was being skipped. "
                        f"Baseline reset to the current value; the halt is active again."
                    )
                except Exception:
                    pass
            return
        db.set_state(implausible_key, "0")

        if pv > hwm:
            db.set_state(hwm_key, str(pv))
            hwm = pv

        drawdown = (hwm - pv) / hwm if hwm > 0 else 0

        if drawdown >= MAX_DRAWDOWN_PCT:
            # Require 2 consecutive breaching reads (this task runs every 30
            # min) before halting — a single noisy tick shouldn't be enough,
            # only a sustained drawdown.
            breach_count = int(db.get_state(breach_key, "0") or 0) + 1
            db.set_state(breach_key, str(breach_count))
            if breach_count < 2:
                log.warning(f"Drawdown {drawdown:.1%} breach #1 — waiting for confirmation "
                           f"before halting")
                return
            already_halted = db.get_state(halted_key, "false") == "true"
            db.set_state(halted_key, "true")
            db.set_state(reason_key,
                         f"Auto: drawdown {drawdown:.1%} from HWM ${hwm:,.2f}")
            log.critical(f"DRAWDOWN LIMIT HIT: {drawdown:.1%} — trading HALTED")
            if not already_halted:
                alerts.send_halt_alert(pv, hwm)
        else:
            db.set_state(breach_key, "0")
            log.debug(f"Drawdown: {drawdown:.1%} (limit: {MAX_DRAWDOWN_PCT:.0%})")
    except Exception as e:
        log.error(f"check_portfolio_drawdown failed: {e}")
        raise self.retry(exc=e, countdown=60)


@locked_by_ticker(lambda ticker: ticker,
                  on_busy=lambda: "over-stop fix skipped: ticker busy (next cycle)")
def _fix_overstop(ticker: str) -> str:
    """Re-read under the lock and make resting SELL stops cover exactly the
    position at the recorded stop. Over-covered: cancel, re-read what is
    still resting (a cross-client stop may refuse to cancel, A2253-6) and
    place stops only for what is left uncovered. Under-covered (A2253-4),
    including no stop at all (A0053-2): add a stop for the uncovered shares.
    Never places a stop at or above the last price, which would sell at
    once. Returns a line for log/alert."""
    import db
    import ibkr
    try:
        pos = next((p for p in ibkr.get_client().get_all_positions() if p.symbol == ticker), None)
        signed = float(pos.qty) if pos else 0.0
        held = round(abs(signed), 4)
        resting = sum(float(o["qty"]) for o in _resting_sell_stops(ticker))
    except Exception as e:
        return f"⚠️ {ticker}: stop-coverage re-check failed ({e}); check TWS"
    if signed < 0:
        # A SELL stop on a short adds to it (professor M63).
        return f"🚨 {ticker}: broker shows a SHORT of {signed:g}; no stop placed — check TWS now"
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
    try:
        if resting > held:
            manage_cancel_stop_orders_for(ticker)
            stuck = _resting_sell_stops(ticker)
            if any(o.get("status") == "PendingCancel" for o in stuck):
                # A slow cancel often lands within seconds; one more look
                # before treating it as stuck (professor note on A0253-1).
                import time
                time.sleep(5)
                stuck = _resting_sell_stops(ticker)
            stuck_qty = sum(float(o["qty"]) for o in stuck)
            if stuck_qty > held + 1e-6:
                ids = ", ".join(str(o["order_id"]) for o in stuck)
                return (f"🚨 {ticker}: stop order(s) {ids} for {stuck_qty:g} shares won't cancel "
                        f"and exceed the {held:g} held — cancel them in TWS")
            need = round(held - stuck_qty, 4)
        else:
            need = round(held - resting, 4)
        if need > 0:
            result = manage_stop_place(ticker, need, float(stop_price))
            db.set_stop_broker_order_id(ticker, result["order_id"])
    except Exception as e:
        return f"🚨 {ticker}: {label}; fix failed ({e}) — check TWS now"
    return f"🔧 {ticker}: {label}; placed a stop for {need:g} @ ${float(stop_price):.2f}"


def _fix_overstop_safe(ticker: str) -> str:
    """_fix_overstop for the reconcile loops: one ticker's failure must not
    skip the remaining tickers and section 3 (audit 02:53, A0253-7)."""
    try:
        return _fix_overstop(ticker)
    except Exception as e:
        log.error(f"[{ticker}] stop coverage fix crashed: {e}")
        return f"🚨 {ticker}: stop coverage fix failed ({e}) — check TWS"


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
        held      = {p.symbol: round(abs(float(p.qty)), 4)
                     for p in client.get_all_positions()}
        open_ords = ibkr.get_open_orders()
        # Section 1 (orphans) cancels any stop order; coverage (2b, 3) counts
        # only working stops by their remainder, as _fix_overstop does.
        stops     = [o for o in open_ords
                     if o["action"] == "SELL" and o["order_type"] in ("STP", "STP LMT", "TRAIL")]
        working   = _working_sell_stops(open_ords)
        all_actions: list[str] = []   # every cycle, for the debug log
        new_actions: list[str] = []   # unthrottled — logged as warning + alerted

        # Clear any pending-orphan/pending-stale-stop marker for tickers
        # confirmed held this cycle -- a marker set during a transient
        # cache lag shouldn't linger and cause an action on some unrelated
        # future cycle.
        for ticker in held:
            db.set_state(f"orphan_pending_{ticker}", "")
            db.set_state(f"stale_stop_pending_{ticker}", "")

        # 1. Orphaned broker stops — stop order with no position on TWO
        # consecutive cycles → cancel. A single missing-this-cycle reading
        # just arms the marker and waits.
        for o in stops:
            if o["ticker"] not in held:
                pending_key = f"orphan_pending_{o['ticker']}"
                if not db.get_state(pending_key):
                    db.set_state(pending_key, datetime.now(timezone.utc).isoformat())
                    log.info(f"[{o['ticker']}] Stop order {o['order_id']} looks orphaned "
                             f"(no position in this cycle's snapshot) — waiting for the next "
                             f"cycle to confirm before cancelling")
                    continue
                if manage_cancel_order(o["order_id"]):
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
        except Exception as e:
            log.warning(f"reconcile_stop_orders: unconfirmed-SELL check failed, not waiting on any: {e}")
            pending_sell_tickers = set()
        for rec in db.get_all_position_stops():
            if rec["ticker"] not in held:
                ticker = rec["ticker"]
                pending_key = f"stale_stop_pending_{ticker}"
                if not db.get_state(pending_key):
                    db.set_state(pending_key, datetime.now(timezone.utc).isoformat())
                    log.info(f"[{ticker}] Position stop record looks stale (no matching "
                             f"position in this cycle's snapshot) — waiting for the next "
                             f"cycle to confirm before reconciling")
                    continue
                if ticker in pending_sell_tickers:
                    # Leaving that order's fills out below would orphan the
                    # outcome, so wait for the late-fill reconcile.
                    log.info(f"[{ticker}] Stale stop record: an unconfirmed SELL order is still being "
                             f"reconciled; waiting")
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
                db.set_state(pending_key, "")
                fill = None
                try:
                    fill = ibkr.get_recent_sell_fill(ticker, recorded)
                except Exception as e:
                    log.warning(f"[{ticker}] Exit-fill lookup failed while reconciling "
                                f"stale stop record: {e}")

                if fill:
                    db.log_trade(
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
                                               exit_price_source="fill")
                    except Exception as e:
                        log.error(f"[{ticker}] close_trade_outcome failed after reconciling "
                                  f"exit: {e}")
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
        for ticker, qty in held.items():
            pending_key = f"overstop_pending_{ticker}"
            covered = stop_qty.get(ticker, 0.0)
            if covered <= 0 or abs(covered - qty) <= 1e-6:
                db.set_state(pending_key, "")
                continue
            if not db.get_state(pending_key):
                db.set_state(pending_key, datetime.now(timezone.utc).isoformat())
                log.warning(f"[{ticker}] resting stops cover {covered:g} vs held {qty:g}; "
                            f"confirming next cycle")
                continue
            db.set_state(pending_key, "")
            msg = _fix_overstop_safe(ticker)
            all_actions.append(msg)
            new_actions.append(msg)
            try:
                import alerts
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
        for ticker, qty in held.items():
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
                    fix = _fix_overstop_safe(ticker)
                    all_actions.append(fix)
                    new_actions.append(fix)
                    if fix.startswith("🔧"):
                        try:
                            import alerts
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


@app.task(name="tasks.check_vector_store_health", bind=True, max_retries=0)
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
    """
    try:
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
        corrupted = checking - ok

        if not corrupted:
            if result.returncode != 0:
                log.warning(f"check_vector_store_health: subprocess exited {result.returncode} "
                            f"but every collection reported OK first — {result.stderr[-500:]}")
            else:
                log.info(f"check_vector_store_health: OK ({len(ok)} collections)")
            return

        import vector_store as vs
        client = vs._get_client()
        rebuilt = []
        for name in corrupted:
            try:
                log.error(f"check_vector_store_health: '{name}' is corrupted (segfaulted on "
                          f"count) — rebuilding")
                try:
                    client.delete_collection(name)
                except Exception:
                    pass  # may already be gone/unreadable — get_or_create below still works
                client.get_or_create_collection(name=name, metadata={"description": vs.COLLECTIONS.get(name, name)})
                rebuilt.append(name)
            except Exception as e:
                log.error(f"check_vector_store_health: failed to rebuild '{name}': {e}")

        if rebuilt:
            import alerts
            alerts._send(
                f"🔧 <b>Vector store auto-recovery</b>\n"
                f"Corrupted collection(s) rebuilt: {', '.join(rebuilt)}\n"
                f"Their accumulated history is gone (this is semantic-memory context, not "
                f"trading state) — they'll refill from ongoing ingestion. Nothing else was affected."
            )
    except Exception as e:
        log.error(f"check_vector_store_health failed: {e}")


@app.task(name="tasks.review_trade_outcomes", bind=True, max_retries=2)
def review_trade_outcomes(self):
    """
    Record 30/60/90-day prices for open trade outcomes.
    Uses >= threshold (not an exact-day match) so a missed run — scheduler
    downtime, a holiday, the box being off — doesn't permanently skip a
    checkpoint; each checkpoint is only ever written once (guarded by the
    review_Xd_price column already being NULL).
    """
    try:
        import db
        import yfinance as yf
        from datetime import date

        outcomes = db.get_open_outcomes()
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

            try:
                price = yf.Ticker(o["ticker"]).fast_info.last_price
            except Exception as e:
                log.warning(f"[{o['ticker']}] price fetch failed: {e}")
                continue
            if not price:
                continue

            for days in pending_checkpoints:
                db.update_outcome_price(o["approval_id"], days, price)
                log.info(f"[{o['ticker']}] {days}d checkpoint (held {days_held}d): ${price:.2f}")
    except Exception as e:
        log.error(f"review_trade_outcomes failed: {e}")
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.prune_vector_store", bind=True, max_retries=1)
def prune_vector_store(self):
    """Age-based pruning of ChromaDB collections to keep context fresh."""
    try:
        import vector_store
        # (collection, max_age_days)
        pruning_rules = [
            ("news",          90),
            ("options",       60),
            ("congressional", 180),
            ("filings",       90),
            ("insider",       365),
            ("macro",         180),
        ]
        for collection, max_age in pruning_rules:
            try:
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
