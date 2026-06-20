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

app = Celery("trading_bot", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer        = "json",
    accept_content         = ["json"],
    result_serializer      = "json",
    timezone               = "America/New_York",
    enable_utc             = True,
    task_track_started     = True,
    worker_prefetch_multiplier = 1,  # one task at a time — model inference is heavy
)

# ── Periodic schedule ─────────────────────────────────────────────────────────
app.conf.beat_schedule = {
    # Market regime — daily at 9:30 AM ET (market open)
    "assess-regime": {
        "task":     "tasks.assess_regime_daily",
        "schedule": crontab(hour=9, minute=30, day_of_week="mon-fri"),
    },
    # Daily scanner — 9:35 AM ET (5 min after regime)
    "daily-scan": {
        "task":     "tasks.run_daily_scan",
        "schedule": crontab(hour=9, minute=35, day_of_week="mon-fri"),
    },
    # Check trailing stops — every 30 min during market hours
    "check-stops": {
        "task":     "tasks.check_trailing_stops",
        "schedule": 1800,
    },
    # SEC filings — every 10 minutes during market hours
    "poll-edgar": {
        "task":     "tasks.poll_edgar",
        "schedule": 600,  # 10 minutes
    },
    # Options flow — every 5 minutes during market hours
    "poll-options": {
        "task":     "tasks.poll_options_flow",
        "schedule": 300,
    },
    # Macro data snapshot — every hour
    "poll-macro": {
        "task":     "tasks.poll_macro",
        "schedule": 3600,
    },
    # arXiv papers — once per day at 9 AM ET
    "poll-arxiv": {
        "task":     "tasks.poll_arxiv",
        "schedule": crontab(hour=9, minute=0),
    },
    # Congressional trades — every 6 hours
    "poll-congressional": {
        "task":     "tasks.poll_congressional",
        "schedule": 21600,
    },
    # Insider trades — every 30 minutes
    "poll-insider": {
        "task":     "tasks.poll_insider",
        "schedule": 1800,
    },
    # Portfolio snapshot — every 5 minutes during market hours
    "portfolio-snapshot": {
        "task":     "tasks.take_portfolio_snapshot",
        "schedule": 300,
    },
    # Auto-expire stale approval requests — every 30 minutes
    "expire-approvals": {
        "task":     "tasks.auto_expire_approvals",
        "schedule": 1800,
    },
    # Earnings calendar refresh — Monday 8:00 AM ET
    "refresh-earnings": {
        "task":     "tasks.refresh_earnings_calendar",
        "schedule": crontab(hour=8, minute=0, day_of_week="mon"),
    },
    # Portfolio drawdown circuit breaker — every 30 minutes
    "check-drawdown": {
        "task":     "tasks.check_portfolio_drawdown",
        "schedule": 1800,
    },
    # 30/60/90-day outcome price reviews — weekdays 4:30 PM ET
    "review-outcomes": {
        "task":     "tasks.review_trade_outcomes",
        "schedule": crontab(hour=16, minute=30, day_of_week="mon-fri"),
    },
    # Vector store pruning — Sunday 3:00 AM
    "prune-vector-store": {
        "task":     "tasks.prune_vector_store",
        "schedule": crontab(hour=3, minute=0, day_of_week="sun"),
    },
}


def _get_client():
    """Lazy-loads Alpaca client."""
    import os
    from alpaca.trading.client import TradingClient
    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    return TradingClient(api_key, api_secret, paper=True)


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


@app.task(name="tasks.run_daily_scan", bind=True, max_retries=2)
def run_daily_scan(self):
    """Daily scanner run at 9:35 AM ET — surfaces candidates for analyst."""
    try:
        from agents.market_regime import get_cached_regime
        from agents.scanner import run_scan
        from agents.crew import process_candidates

        regime    = get_cached_regime()
        scan_result = run_scan(regime)
        candidates  = scan_result.get("candidates", [])
        high_medium = [c for c in candidates if c["priority"] in ("HIGH", "MEDIUM")]

        log.info(f"Daily scan: {len(high_medium)} candidates to analyze")
        if high_medium:
            process_candidates(high_medium, max_per_run=5)
    except Exception as e:
        log.error(f"run_daily_scan failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.check_trailing_stops", bind=True, max_retries=2)
def check_trailing_stops(self):
    """Monitor open positions against their stop levels. Alert on triggers."""
    try:
        from risk_manager import RiskManager
        import alerts

        client = _get_client()
        rm = RiskManager(client)
        triggered = rm.check_trailing_stops()
        for alert in triggered:
            log.warning(f"STOP TRIGGERED: {alert['ticker']} — {alert['reason']}")
            alerts.send_stop_triggered(
                ticker=alert["ticker"],
                current_price=alert["current_price"],
                stop_price=alert["stop_price"],
                reason=alert["reason"],
            )
    except Exception as e:
        log.error(f"check_trailing_stops failed: {e}")
        raise self.retry(exc=e, countdown=60)


# ── Existing tasks (updated call signatures) ──────────────────────────────────

@app.task(name="tasks.poll_edgar", bind=True, max_retries=3)
def poll_edgar(self):
    """Polls SEC EDGAR for new filings and routes through agent pipeline."""
    try:
        from ingestion.edgar import poll_once
        from agents.crew import process_event

        filings = poll_once()
        log.info(f"EDGAR: {len(filings)} new filing(s)")
        for filing in filings:
            process_event(filing)
    except Exception as e:
        log.error(f"poll_edgar failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.poll_options_flow", bind=True, max_retries=3)
def poll_options_flow(self):
    """Checks for unusual options activity across all symbols."""
    try:
        from ingestion.options_flow import get_all_flow_summaries
        from agents.crew import process_event

        summaries = get_all_flow_summaries()
        for ticker, flow in summaries.items():
            if flow.get("unusual"):
                event = {
                    "source": "options",
                    "ticker": ticker,
                    "title":  f"Unusual options activity: {ticker}",
                    "text":   flow["summary"],
                }
                process_event(event)
    except Exception as e:
        log.error(f"poll_options_flow failed: {e}")
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
        log.info("Macro snapshot stored.")

        # Reassess regime if macro conditions changed significantly
        try:
            from agents.market_regime import should_reassess, assess_regime
            vix = snap.get("VIXCLS", {}).get("value")
            if vix and should_reassess(float(vix)):
                log.info(f"Intraday VIX shift detected ({vix:.1f}) — reassessing market regime")
                assess_regime()
        except Exception as e:
            log.debug(f"Intraday regime check skipped: {e}")

    except Exception as e:
        log.error(f"poll_macro failed: {e}")
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.poll_arxiv", bind=True, max_retries=3)
def poll_arxiv(self):
    """Fetches new quantum/AI papers from arXiv and stores in vector memory."""
    try:
        from ingestion.arxiv_watcher import poll_once
        import vector_store

        papers = poll_once()
        log.info(f"arXiv: {len(papers)} new paper(s)")
        for paper in papers:
            vector_store.store_arxiv(paper)
    except Exception as e:
        log.error(f"poll_arxiv failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.poll_congressional", bind=True, max_retries=3)
def poll_congressional(self):
    """Polls congressional trading data and processes high-conviction trades."""
    try:
        from ingestion.congressional import poll_once
        from agents.crew import process_event

        trades = poll_once()
        log.info(f"Congressional: {len(trades)} new trade(s)")
        for trade in trades:
            # Only send large purchases through the full pipeline
            if trade.get("side", "").upper() == "PURCHASE":
                process_event(trade)
            else:
                import vector_store
                vector_store.store(
                    collection="congressional",
                    doc_id=f"cong_{trade.get('ticker')}_{trade.get('date')}_{trade.get('representative','')}",
                    text=trade["text"],
                    metadata={"ticker": trade["ticker"], "side": trade.get("side", "")},
                )
    except Exception as e:
        log.error(f"poll_congressional failed: {e}")
        raise self.retry(exc=e, countdown=300)


@app.task(name="tasks.poll_insider", bind=True, max_retries=3)
def poll_insider(self):
    """Polls Form 4 insider trades. Buys by C-suite are processed as full events."""
    try:
        from ingestion.insider import poll_once
        import vector_store

        filings = poll_once()
        log.info(f"Insider: {len(filings)} new filing(s)")
        for filing in filings:
            vector_store.store(
                collection="insider",
                doc_id=filing.get("url", filing.get("title", "")),
                text=filing["text"],
                metadata={"ticker": filing["ticker"]},
            )
    except Exception as e:
        log.error(f"poll_insider failed: {e}")
        raise self.retry(exc=e, countdown=120)


@app.task(name="tasks.take_portfolio_snapshot", bind=True, max_retries=3)
def take_portfolio_snapshot(self):
    """Takes a portfolio snapshot and logs to Supabase for the dashboard."""
    try:
        import db
        from alpaca.trading.client import TradingClient
        import os

        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper=True,
        )
        account   = client.get_account()
        positions = client.get_all_positions()

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


@app.task(name="tasks.check_portfolio_drawdown", bind=True, max_retries=2)
def check_portfolio_drawdown(self):
    """Alert if portfolio drawdown exceeds MAX_DRAWDOWN_PCT. Does not halt automatically."""
    try:
        import db
        import alerts
        from config import START_VALUE, MAX_DRAWDOWN_PCT

        client  = _get_client()
        pv      = float(client.get_account().portfolio_value)
        drawdown = (START_VALUE - pv) / START_VALUE

        if drawdown >= MAX_DRAWDOWN_PCT:
            log.critical(f"DRAWDOWN LIMIT HIT: {drawdown:.1%} — manual review required")
            alerts.send_halt_alert(pv, START_VALUE)
        else:
            log.debug(f"Drawdown: {drawdown:.1%} (limit: {MAX_DRAWDOWN_PCT:.0%})")
    except Exception as e:
        log.error(f"check_portfolio_drawdown failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.review_trade_outcomes", bind=True, max_retries=2)
def review_trade_outcomes(self):
    """Check and record 30/60/90-day prices for open trade outcomes."""
    try:
        import db
        import yfinance as yf
        from datetime import date

        outcomes = db.get_open_outcomes()
        for o in outcomes:
            if not o.get("entry_date"):
                continue
            days_held = (date.today() - o["entry_date"].date()).days
            if days_held in (30, 60, 90):
                try:
                    price = yf.Ticker(o["ticker"]).fast_info.last_price
                    if price:
                        db.update_outcome_price(o["approval_id"], days_held, price)
                        log.info(f"[{o['ticker']}] {days_held}d review: ${price:.2f}")
                except Exception as e:
                    log.warning(f"[{o['ticker']}] price fetch failed: {e}")
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
