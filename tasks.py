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
}


def _get_pipeline():
    """Lazy-loads the agent pipeline to avoid circular imports."""
    import os
    from alpaca.trading.client import TradingClient
    from risk_manager import RiskManager
    import db
    import alerts

    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    client     = TradingClient(api_key, api_secret, paper=True)
    rm         = RiskManager(client)
    return client, rm


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.task(name="tasks.poll_edgar", bind=True, max_retries=3)
def poll_edgar(self):
    """Polls SEC EDGAR for new filings and processes each through the agent pipeline."""
    try:
        from ingestion.edgar import poll_once
        from agents.crew import process_event
        import db, alerts

        client, rm = _get_pipeline()
        filings = poll_once()
        log.info(f"EDGAR: {len(filings)} new filing(s)")
        for filing in filings:
            process_event(filing, client, rm, db, alerts)
    except Exception as e:
        log.error(f"poll_edgar failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.poll_options_flow", bind=True, max_retries=3)
def poll_options_flow(self):
    """Checks for unusual options activity across all symbols."""
    try:
        from ingestion.options_flow import get_all_flow_summaries
        from agents.crew import process_event
        import db, alerts

        client, rm = _get_pipeline()
        summaries = get_all_flow_summaries()

        for ticker, flow in summaries.items():
            if flow.get("unusual"):
                event = {
                    "source": "options",
                    "ticker": ticker,
                    "title":  f"Unusual options activity: {ticker}",
                    "text":   flow["summary"],
                }
                process_event(event, client, rm, db, alerts)
    except Exception as e:
        log.error(f"poll_options_flow failed: {e}")
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.poll_macro", bind=True, max_retries=3)
def poll_macro(self):
    """Fetches macro snapshot and stores in vector memory for model context."""
    try:
        from ingestion.macro import get_macro_summary_text
        import vector_store
        from datetime import datetime, timezone

        summary = get_macro_summary_text()
        vector_store.store(
            collection="macro",
            doc_id=f"macro_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}",
            text=summary,
            metadata={"type": "macro_snapshot"},
        )
        log.info("Macro snapshot stored.")
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
        import db, alerts

        client, rm = _get_pipeline()
        trades = poll_once()
        log.info(f"Congressional: {len(trades)} new trade(s)")
        for trade in trades:
            # Only send large purchases through the full pipeline
            if trade.get("side", "").upper() == "PURCHASE":
                process_event(trade, client, rm, db, alerts)
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
        from agents.crew import process_event
        import db, alerts, vector_store

        client, rm = _get_pipeline()
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
