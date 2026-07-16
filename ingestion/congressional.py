"""
Congressional trading ingestion via Quiver Quant API.
Members of Congress frequently trade stocks in sectors they regulate
and oversee — often before public announcements. This is legal (STOCK Act
requires disclosure, but doesn't ban trading) and statistically significant.

Requires QUIVER_API_KEY in .env
Sign up at quiverquant.com (~$50/mo)
"""

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

API_KEY  = os.getenv("QUIVER_API_KEY", "")
BASE_URL = "https://api.quiverquant.com/beta"

from config import SYMBOLS

_SOURCE = "congressional"


def _is_seen(trade_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, trade_id)
    except Exception:
        return False


def _mark_seen(trade_id: str) -> None:
    try:
        import db
        db.mark_seen(_SOURCE, trade_id)
    except Exception as e:
        log.debug(f"mark_seen failed for {trade_id}: {e}")


def _get_congressional_trades(ticker: str) -> list[dict]:
    """Fetches recent congressional trades for a ticker."""
    if not API_KEY:
        return []
    try:
        resp = requests.get(
            f"{BASE_URL}/historical/congresstrading/{ticker}",
            headers={"Authorization": f"Token {API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        log.warning(f"Quiver Quant fetch failed for {ticker}: {e}")
        return []


def poll_once() -> list[dict]:
    """Returns new congressional trades across all symbols."""
    if not API_KEY:
        log.debug("QUIVER_API_KEY not set — congressional data skipped.")
        return []

    new_trades = []
    for ticker in SYMBOLS:
        trades = _get_congressional_trades(ticker)
        for trade in trades[:5]:  # most recent 5 per ticker
            trade_id = f"{ticker}_{trade.get('TransactionDate','')}_{trade.get('Representative','')}"
            if _is_seen(trade_id):
                continue

            side = trade.get("Transaction", "").upper()
            rep  = trade.get("Representative", "Unknown")
            amt  = trade.get("Amount", "unknown amount")
            date = trade.get("TransactionDate", "")

            text = (
                f"Congressional trade: {rep} {side} {ticker} "
                f"for {amt} on {date}. "
                f"Committee: {trade.get('Committee', 'Unknown')}."
            )

            new_trades.append({
                "source":      "congressional",
                "ticker":      ticker,
                "side":        side,
                "representative": rep,
                "amount":      amt,
                "date":        date,
                "committee":   trade.get("Committee", ""),
                "title":       text,
                "text":        text,
                # Not marked seen yet — tasks.poll_congressional marks it
                # after successfully routing this trade (dispatch or vector
                # store write), so a Celery retry can't lose it.
                "_seen_key": trade_id,
            })
        time.sleep(0.2)
    return new_trades


def get_gov_contracts(ticker: str, days_back: int = 30) -> list[dict]:
    """
    Fetches recent government contract awards for a ticker via Quiver Quant.
    Returns list of {ticker, agency, amount, date, description}.
    Returns [] on error or missing key.
    """
    if not API_KEY:
        return []
    try:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        resp = requests.get(
            f"{BASE_URL}/historical/govcontracts/{ticker}",
            headers={"Authorization": f"Token {API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        contracts = []
        for c in raw:
            contract_date = c.get("Date", "")
            if contract_date and contract_date < cutoff:
                continue
            amount_raw = c.get("Amount", "0")
            try:
                amount = float(str(amount_raw).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                amount = 0.0
            contracts.append({
                "ticker":      ticker,
                "agency":      c.get("Agency", "Unknown Agency"),
                "amount":      amount,
                "date":        contract_date,
                "description": c.get("Description", ""),
            })
        return contracts
    except Exception as e:
        log.debug(f"Gov contracts fetch failed for {ticker}: {e}")
        return []


def get_recent_contracts_all(days_back: int = 14) -> dict[str, list]:
    """
    Fetches recent government contracts for all watchlist tickers.
    Used by the scanner's _score_catalyst function.
    Returns {ticker: [contracts]} — only tickers with at least one contract.
    """
    if not API_KEY:
        return {}

    from config import SYMBOLS
    results = {}
    for ticker in SYMBOLS:
        contracts = get_gov_contracts(ticker, days_back=days_back)
        if contracts:
            results[ticker] = contracts
        time.sleep(0.2)
    return results


def run_forever(on_new_trade, poll_interval_seconds: int = 3600) -> None:
    """
    Polls every hour. Congressional disclosures have a 30-45 day reporting lag
    but clustering of trades is still statistically meaningful.
    """
    log.info("Congressional trading watcher started.")
    while True:
        try:
            trades = poll_once()
            for trade in trades:
                log.info(f"Congressional trade: {trade['representative']} {trade['side']} {trade['ticker']}")
                try:
                    on_new_trade(trade)
                except Exception as e:
                    log.error(f"on_new_trade callback failed: {e}")
        except Exception as e:
            log.error(f"Congressional poll error: {e}")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not API_KEY:
        print("QUIVER_API_KEY not set in .env — add it to activate congressional data.")
    else:
        trades = poll_once()
        for t in trades[:10]:
            print(f"[{t['side']:4}] {t['ticker']:6} — {t['representative']} — {t['amount']}")
