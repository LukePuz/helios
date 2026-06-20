"""
CrewAI Orchestration — coordinates the full agent pipeline.

Flow for every incoming event:
  1. 7B Screener decides if the event matters
  2. Context Engine assembles the situation brief
  3. 72B Analyst generates the investment thesis
  4. Risk Manager validates the position size
  5. Alpaca executes if conviction meets threshold

This module is the single entry point for event-driven trading decisions.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

MIN_CONVICTION_TO_TRADE = 0.65  # model must be >= 65% confident to execute


def process_event(
    event: dict,
    trading_client,
    risk_manager,
    db_module=None,
    alerts_module=None,
) -> dict:
    """
    Full pipeline for a single event. Returns the thesis dict.

    Args:
        event:          Raw event from any ingestion source
        trading_client: Alpaca TradingClient
        risk_manager:   RiskManager instance
        db_module:      db module for logging (optional)
        alerts_module:  alerts module for Telegram (optional)

    Returns:
        thesis dict from the analyst (or stub if model unavailable)
    """
    from agents.screener import screen
    from agents.analyst import analyze
    from context_engine import assemble_brief

    ticker = event.get("ticker", "UNKNOWN")
    log.info(f"Processing event: [{event.get('source','?')}] {ticker} — {event.get('title','')[:60]}")

    # ── Step 1: Store event in vector memory ─────────────────────────────────
    try:
        import vector_store
        collection = _event_to_collection(event)
        vector_store.store(
            collection=collection,
            doc_id=event.get("url", event.get("title", str(datetime.now()))),
            text=event.get("text", event.get("title", "")),
            metadata={
                "ticker": ticker,
                "source": event.get("source", ""),
                "date":   datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        log.warning(f"Vector store failed: {e}")

    # ── Step 2: 7B Screener ───────────────────────────────────────────────────
    screening = screen(event)

    if not screening.get("relevant", False):
        log.info(f"Event screened out: {screening.get('reason','')}")
        if db_module:
            db_module.log_signal(
                symbol=ticker,
                signal="SCREENED_OUT",
                confidence=0.0,
                reasoning=screening.get("reason", "Not relevant"),
            )
        return {"signal": "SCREENED_OUT", "ticker": ticker, "conviction": 0.0}

    log.info(f"Event relevant — urgency={screening['urgency']} — calling 72B analyst")

    # ── Step 3: Get current positions for context ─────────────────────────────
    positions = {}
    try:
        open_positions = trading_client.get_all_positions()
        for p in open_positions:
            entry   = float(p.avg_entry_price)
            current = float(p.current_price or entry)
            positions[p.symbol] = {
                "qty":           float(p.qty),
                "entry_price":   entry,
                "current_price": current,
                "pl_pct":        (current - entry) / entry,
                "market_value":  float(p.market_value),
            }
    except Exception as e:
        log.warning(f"Could not fetch positions: {e}")

    # ── Step 4: Assemble situation brief ──────────────────────────────────────
    brief = assemble_brief(event, positions, trading_client)

    # ── Step 5: 72B Analyst ───────────────────────────────────────────────────
    thesis = analyze(brief, event, screening)

    # ── Step 6: Log signal ────────────────────────────────────────────────────
    if db_module:
        try:
            db_module.log_signal(
                symbol=thesis.get("ticker", ticker),
                signal=thesis.get("signal", "HOLD"),
                confidence=thesis.get("conviction", 0.0),
                reasoning=thesis.get("thesis", ""),
            )
        except Exception as e:
            log.error(f"Signal logging failed: {e}")

    # ── Step 7: Execute if conviction is high enough ──────────────────────────
    signal     = thesis.get("signal", "HOLD")
    conviction = thesis.get("conviction", 0.0)
    act_ticker = thesis.get("ticker", ticker)

    if signal in ("BUY", "SELL") and conviction >= MIN_CONVICTION_TO_TRADE:
        _execute_thesis(
            thesis=thesis,
            act_ticker=act_ticker,
            signal=signal,
            trading_client=trading_client,
            risk_manager=risk_manager,
            positions=positions,
            db_module=db_module,
            alerts_module=alerts_module,
        )
    else:
        reason = (
            f"conviction {conviction:.0%} below threshold {MIN_CONVICTION_TO_TRADE:.0%}"
            if conviction < MIN_CONVICTION_TO_TRADE
            else "signal is HOLD/WATCH"
        )
        log.info(f"No trade executed: {reason}")

    return thesis


def _execute_thesis(
    thesis, act_ticker, signal, trading_client, risk_manager,
    positions, db_module, alerts_module,
) -> None:
    """Validates with risk manager and submits order to Alpaca."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    try:
        from ingestion.polygon_stream import get_snapshot
        snap = get_snapshot(act_ticker)
        price = snap.get("price") or 0
    except Exception:
        price = 0

    if not price:
        # Fall back to Alpaca last trade price
        try:
            from market_data import get_latest_quotes
            quotes = get_latest_quotes([act_ticker])
            price = quotes.get(act_ticker, {}).get("mid", 0)
        except Exception:
            log.error(f"Could not determine price for {act_ticker} — skipping execution")
            return

    if signal == "BUY":
        qty = risk_manager.shares_affordable(act_ticker, price)
        side = OrderSide.BUY
    else:
        if act_ticker not in positions:
            log.info(f"SELL signal for {act_ticker} but no position — skipping")
            return
        qty = int(positions[act_ticker]["qty"])
        if not risk_manager.check_sell_allowed(act_ticker, qty):
            return
        side = OrderSide.SELL

    if qty <= 0:
        log.info(f"Risk manager returned 0 shares for {act_ticker} — skipping")
        return

    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=act_ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )
        side_str = signal
        reason = f"AI thesis (conviction={thesis.get('conviction'):.0%}): {thesis.get('thesis','')[:100]}"
        log.info(f"ORDER {side_str} {qty} {act_ticker} @ ~${price:.2f} | {reason}")

        if db_module:
            db_module.log_trade(act_ticker, side_str, qty, price, reason)
        if alerts_module:
            alerts_module.send_trade_alert(act_ticker, side_str, qty, price, reason)

    except Exception as e:
        log.error(f"Order submission failed: {e}")


def _event_to_collection(event: dict) -> str:
    """Maps event source to ChromaDB collection name."""
    source_map = {
        "sec_filing":    "filings",
        "sec_form4":     "insider",
        "arxiv":         "arxiv",
        "congressional": "congressional",
        "news":          "news",
        "options":       "options",
        "macro":         "macro",
    }
    return source_map.get(event.get("source", ""), "news")
