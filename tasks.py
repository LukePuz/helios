"""
Celery async task definitions.
Each ingestion source runs as a periodic Celery beat task,
firing events into the agent pipeline without blocking the main thread.

Requires Redis running locally (via Docker):
  docker run -d -p 6379:6379 redis:alpine

Start worker:   celery -A tasks worker --loglevel=info --pool=solo
Start scheduler: celery -A tasks beat --loglevel=info
"""

import logging
import os
from pathlib import Path

from celery import Celery
from celery.schedules import crontab
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
ORDERS_QUEUE = "orders"
_ORDERS_WORKER_CLIENT_ID = "30"


def _is_orders_worker() -> bool:
    return os.getenv("IB_CLIENT_ID") == _ORDERS_WORKER_CLIENT_ID


@app.task(name="tasks._broker_place_stop_order", bind=True, max_retries=1)
def _broker_place_stop_order(self, ticker: str, qty: float, stop_price: float) -> dict:
    import ibkr
    return ibkr.place_stop_order(ticker, qty, stop_price)


@app.task(name="tasks._broker_cancel_stop_orders_for", bind=True, max_retries=1)
def _broker_cancel_stop_orders_for(self, ticker: str) -> int:
    import ibkr
    return ibkr.cancel_stop_orders_for(ticker)


@app.task(name="tasks._broker_cancel_order", bind=True, max_retries=1)
def _broker_cancel_order(self, order_id: int) -> bool:
    import ibkr
    return ibkr.cancel_order(order_id)


def manage_stop_place(ticker: str, qty: float, stop_price: float, timeout: int = 30) -> dict:
    """Place a protective stop — see the module-level comment above for why
    this exists instead of calling ibkr.place_stop_order directly. Every
    caller anywhere in the codebase should go through this, including code
    that happens to already be running in the orders worker (that branch is
    just a direct call, essentially free) — the point is there being exactly
    one way to do this, so a future direct ibkr.place_stop_order call can't
    quietly reintroduce the bug."""
    if _is_orders_worker():
        import ibkr
        return ibkr.place_stop_order(ticker, qty, stop_price)
    from celery.result import allow_join_result
    with allow_join_result():
        return _broker_place_stop_order.apply_async(
            args=[ticker, qty, stop_price], queue=ORDERS_QUEUE,
        ).get(timeout=timeout)


def manage_cancel_stop_orders_for(ticker: str, timeout: int = 30) -> int:
    """Cancel all resting stop orders for a ticker — see manage_stop_place."""
    if _is_orders_worker():
        import ibkr
        return ibkr.cancel_stop_orders_for(ticker)
    from celery.result import allow_join_result
    with allow_join_result():
        return _broker_cancel_stop_orders_for.apply_async(
            args=[ticker], queue=ORDERS_QUEUE,
        ).get(timeout=timeout)


@app.task(name="tasks._broker_place_trailing_stop_order", bind=True, max_retries=1)
def _broker_place_trailing_stop_order(self, ticker: str, qty: float,
                                       trail_amount: float | None = None,
                                       trail_percent: float | None = None) -> dict:
    import ibkr
    return ibkr.place_trailing_stop_order(ticker, qty, trail_amount=trail_amount,
                                           trail_percent=trail_percent)


def manage_place_trailing_stop(ticker: str, qty: float, trail_amount: float | None = None,
                               trail_percent: float | None = None, timeout: int = 30) -> dict:
    """Place a native TRAIL stop (the trailing-stop conversion path) — see
    manage_stop_place. Exactly one of trail_amount/trail_percent, matching
    ibkr.place_trailing_stop_order's own contract."""
    if _is_orders_worker():
        import ibkr
        return ibkr.place_trailing_stop_order(ticker, qty, trail_amount=trail_amount,
                                               trail_percent=trail_percent)
    from celery.result import allow_join_result
    with allow_join_result():
        return _broker_place_trailing_stop_order.apply_async(
            args=[ticker, qty], kwargs={"trail_amount": trail_amount, "trail_percent": trail_percent},
            queue=ORDERS_QUEUE,
        ).get(timeout=timeout)


def manage_cancel_order(order_id: int, timeout: int = 30) -> bool:
    """Cancel a specific order by id — see manage_stop_place. Also used for
    the stale-market-BUY-order cleanup at startup (main.py), not just stops:
    the same ownership rule applies to any order type, not only STP."""
    if _is_orders_worker():
        import ibkr
        return ibkr.cancel_order(order_id)
    from celery.result import allow_join_result
    with allow_join_result():
        return _broker_cancel_order.apply_async(
            args=[order_id], queue=ORDERS_QUEUE,
        ).get(timeout=timeout)


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
    # SEC filings — every 10 minutes during market hours
    "poll-edgar": {
        "task":     "tasks.poll_edgar",
        "schedule": 600,  # 10 minutes
        "options":  {"expires": 600},
    },
    # Options flow — every 5 minutes during market hours
    "poll-options": {
        "task":     "tasks.poll_options_flow",
        "schedule": 300,
        "options":  {"expires": 300},
    },
    # Macro data snapshot — every hour
    "poll-macro": {
        "task":     "tasks.poll_macro",
        "schedule": 3600,
        "options":  {"expires": 3600},
    },
    # arXiv papers — once per day at 9 AM ET
    "poll-arxiv": {
        "task":     "tasks.poll_arxiv",
        "schedule": crontab(hour=9, minute=0),
        "options":  {"expires": 24 * 3600},
    },
    # Congressional trades — every 6 hours
    "poll-congressional": {
        "task":     "tasks.poll_congressional",
        "schedule": 21600,
        "options":  {"expires": 21600},
    },
    # Insider trades — every 30 minutes
    "poll-insider": {
        "task":     "tasks.poll_insider",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # Analyst rating changes (yfinance, free) — every 30 minutes, same
    # cadence as insider (both are "check for something new," not
    # continuous streams).
    "poll-analyst-ratings": {
        "task":     "tasks.poll_analyst_ratings",
        "schedule": 1800,
        "options":  {"expires": 1800},
    },
    # Portfolio snapshot — every 5 minutes during market hours
    "portfolio-snapshot": {
        "task":     "tasks.take_portfolio_snapshot",
        "schedule": 300,
        "options":  {"expires": 300},
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
        "schedule": crontab(hour=8, minute=0, day_of_week="mon"),
        "options":  {"expires": 24 * 3600},
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
    "poll-news": {
        "task":     "tasks.poll_news",
        "schedule": 900,
        "options":  {"expires": 900},
    },
    # Fundamentals refresh — 5:00 PM ET. Must run inside the awake window
    # (currently 05:00-19:00): the old 2:00 AM slot predated even the
    # original 08:45 wake, so it never fired at all — the fundamentals
    # table stayed empty forever, which silently capped every analyst
    # conviction at 6 via the ungrounded-valuation guardrail.
    "refresh-fundamentals": {
        "task":     "tasks.refresh_fundamentals",
        "schedule": crontab(hour=17, minute=0, day_of_week="mon-fri"),
        "options":  {"expires": 2 * 3600},  # must finish before the 19:00 stack-down
    },
    # Price history cache refresh — 4:45 PM ET (before fundamentals, after
    # close; same awake-window constraint as above).
    "refresh-price-history": {
        "task":     "tasks.refresh_price_history",
        "schedule": crontab(hour=16, minute=45, day_of_week="mon-fri"),
        "options":  {"expires": 2 * 3600},
    },
    # S&P 500 universe price history — weekly, Saturday 9:00 AM. Same
    # down-all-weekend caveat as prune-vector-store/prune-seen-items above:
    # actually runs via Monday 05:00 catch-up (~44h later), which is fine —
    # feeds GICS-sector breadth + candidate_scanner, neither of which needs
    # day-fresh data (see refresh_sp500_history's docstring).
    "refresh-sp500-history": {
        "task":     "tasks.refresh_sp500_history",
        "schedule": crontab(hour=9, minute=0, day_of_week="sat"),
        "options":  {"expires": 48 * 3600},
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
        conn = db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT positions_json FROM portfolio_snapshots "
                    "ORDER BY timestamp DESC LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            db._put_conn(conn)
        return {t.upper() for t in (row[0] or {})} if row else set()
    except Exception as e:
        log.warning(f"_held_tickers failed ({e}) — routing to bulk queue")
        return set()


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

    Deliberately does NOT re-run the 72B analysis — reuses
    execute_approved's own price-drift, gap-confirmation, and price-
    freshness guards as the safety check against firing a now-stale
    decision. If the market is somehow still closed when this fires
    (holiday, clock skew), or the gap/price still hasn't resolved,
    execute_approved's gates defer it again with a fresh retry —
    self-correcting, not a tight loop, since each retry target is strictly
    later than the last and real time keeps moving forward (and both the
    gap-confirmation and stale-price paths have their own hard retry cap).
    """
    import db
    from execution import execute_approved

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
    success = execute_approved(
        approval, trigger="auto_retry",
        gap_retry_attempt=gap_retry_attempt, price_retry_attempt=price_retry_attempt,
    )
    db.update_approval_status(approval_id, "APPROVED" if success else "FAILED")
    log.info(f"retry_deferred_entry: [{ticker}] approval #{approval_id} "
             f"{'executed' if success else 'still failed'} on retry "
             f"(gap_attempt={gap_retry_attempt}, price_attempt={price_retry_attempt})")


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

    try:
        pending = db.get_unconfirmed_orders(max_age_hours=24 * 30)
    except Exception as e:
        log.error(f"reconcile_unconfirmed_orders: could not load unconfirmed orders: {e}")
        raise self.retry(exc=e, countdown=120)

    for order in pending:
        ticker = order["ticker"]
        try:
            fill = ibkr.get_fill_for_order(order["broker_order_id"])
        except Exception as e:
            log.warning(f"[{ticker}] reconcile_unconfirmed_orders: fill lookup failed for "
                        f"order {order['broker_order_id']}: {e}")
            continue
        if not fill:
            continue  # still genuinely unfilled (or cancelled/rejected) -- leave alone

        fill_qty, fill_price = fill["filled_qty"], fill["fill_price"]
        log.warning(f"[{ticker}] Late fill found for order {order['broker_order_id']}: "
                    f"{order['side']} {fill_qty} @ ${fill_price:.2f} — was recorded "
                    f"unconfirmed, reconciling now")

        db.log_trade(
            symbol=ticker, side=order["side"], quantity=fill_qty, price=fill_price,
            reason="Late-fill reconciled (order placed unconfirmed, filled after the wait window)",
        )
        db.update_order_fill(order["id"], fill_price=fill_price, filled_qty=fill_qty, status="filled")

        stop_placed = False
        approval = db.get_approval_by_id(order["approval_id"]) if order["approval_id"] else None
        if order["side"] == "BUY" and approval:
            risk_json = approval.get("risk_json") or {}
            stop_price = approval.get("stop_loss_price") or risk_json.get("stop_loss_price")
            if stop_price:
                try:
                    stop_result = manage_stop_place(ticker, fill_qty, float(stop_price))
                    db.upsert_position_stop(
                        ticker=ticker, entry_price=fill_price, stop_price=float(stop_price),
                        stop_method=risk_json.get("stop_method", ""),
                        bucket=approval.get("thesis_bucket", "Accelerator"),
                        conviction=approval.get("conviction", 5),
                        trail_distance_pct=risk_json.get("trailing_stop_distance_pct"),
                    )
                    db.set_stop_broker_order_id(ticker, stop_result["order_id"])
                    stop_placed = True
                except Exception as e:
                    log.error(f"[{ticker}] reconcile_unconfirmed_orders: stop placement failed: {e}")
            if approval.get("status") == "FAILED":
                db.update_approval_status(order["approval_id"], "APPROVED")

        alert_lines = [
            f"🔁 LATE FILL RECONCILED — <b>{ticker}</b>",
            f"{order['side']} {fill_qty:g} shares @ ${fill_price:.2f} — this order was marked "
            f"unconfirmed but had actually filled at the broker.",
        ]
        if order["side"] == "BUY":
            alert_lines.append(
                "Protective stop placed." if stop_placed else
                "⚠️ Could not place a protective stop automatically — check TWS."
            )
        try:
            alerts._send("\n".join(alert_lines))
        except Exception:
            pass


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
        if high_medium:
            analyze_candidates.apply_async(
                args=[high_medium], kwargs={"max_per_run": 5},
                queue=LLM_QUEUE_BULK, expires=LLM_EVENT_EXPIRES_SECONDS,
            )
    except Exception as e:
        log.error(f"run_daily_scan failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.review_positions", bind=True, max_retries=2)
def review_positions(self):
    """Scheduled reassessment of already-held positions for TRIM/EXIT/HOLD
    — see agents/crew.py's review_all_held_positions docstring for why this
    exists (nothing else ever asked this question on a schedule)."""
    try:
        from agents.crew import review_all_held_positions
        review_all_held_positions(max_per_run=10)
    except Exception as e:
        log.error(f"review_positions failed: {e}")
        raise self.retry(exc=e, countdown=300)


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

        try:
            pos   = ibkr.get_client().get_open_position(ticker)
            owned = round(abs(float(pos.qty)), 4)
        except Exception:
            owned = 0.0
        if owned <= 0:
            # Stale stop record for a position that no longer exists
            db.delete_position_stop(ticker)
            return False

        manage_cancel_stop_orders_for(ticker)  # belt-and-braces before selling
        result = ibkr.place_market_order(ticker, "SELL", owned)
        filled = float(result.get("filled_qty") or 0)
        if filled <= 0:
            log.error(f"[{ticker}] Backstop sell submitted but no fill confirmed "
                      f"(id={result.get('order_id')}) — nothing recorded")
            _schedule_backstop_retry(ticker, alert)
            return False

        fill_price = result.get("fill_price") or alert["current_price"]
        db.log_trade(symbol=ticker, side="SELL", quantity=filled, price=fill_price,
                     reason=f"Backstop stop-breach sell (stop ${alert['stop_price']:.2f})"
                            + (" [retry]" if _is_retry else ""))
        db.delete_position_stop(ticker)
        try:
            db.close_trade_outcome(ticker, exit_price=fill_price)
        except Exception:
            pass
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
    """Polls SEC EDGAR for new filings and routes through agent pipeline."""
    try:
        from ingestion.edgar import poll_once

        filings = poll_once()
        log.info(f"EDGAR: {len(filings)} new filing(s)")
        for filing in filings:
            _dispatch_event(filing, seen_source="edgar")
        _heartbeat_ok("edgar", len(filings))
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

    stale = []
    now = datetime.now(timezone.utc)
    health = {h["source"]: h for h in db.get_ingestion_health()}
    for source, max_gap in expected_minutes.items():
        rec = health.get(source)
        if not rec or not rec.get("last_success_at"):
            stale.append(f"{source}: never succeeded")
            continue
        age_min = (now - rec["last_success_at"]).total_seconds() / 60
        if age_min > max_gap:
            stale.append(f"{source}: last success {age_min:.0f}min ago (expected &lt;{max_gap}min)")
    return stale


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
        from datetime import date, timedelta

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
            positions_text.append(
                f"{p.symbol}: {p.qty:.0f}sh, P&amp;L ${float(p.unrealized_pl):+,.0f}{dist}"
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
            return

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
        stops     = [o for o in open_ords
                     if o["action"] == "SELL" and o["order_type"] in ("STP", "STP LMT", "TRAIL")]
        all_actions: list[str] = []   # every cycle, for the debug log
        new_actions: list[str] = []   # unthrottled — logged as warning + alerted

        # Clear any pending-orphan marker for tickers confirmed held this
        # cycle -- a marker set during a transient cache lag shouldn't
        # linger and cause a cancel on some unrelated future cycle.
        for ticker in held:
            db.set_state(f"orphan_pending_{ticker}", "")

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
        # against today's broker execution history, THEN delete.
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
        for rec in db.get_all_position_stops():
            if rec["ticker"] not in held:
                ticker = rec["ticker"]
                fill = None
                try:
                    fill = ibkr.get_recent_sell_fill(ticker)
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
                    )
                    try:
                        db.close_trade_outcome(ticker, exit_price=fill["fill_price"])
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
        stopped_tickers = {o["ticker"] for o in stops}
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
                if not since:
                    db.set_state(since_key, datetime.now(timezone.utc).isoformat())
                else:
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
                timeout=180,
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
