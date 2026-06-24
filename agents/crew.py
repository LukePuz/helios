"""
Crew Orchestrator — coordinates the full agent pipeline.

Two entry points:
  1. process_event(event)  — event-driven (SEC filing, options flow, etc.)
     Screen → Brief → Analyze → Risk Size → Queue for Approval
  2. process_candidates(candidates) — scanner-driven (daily scan output)
     Analyze → Risk Size → Queue for Approval

Human approval (Telegram inline buttons) is the only path to order execution.
No autonomous trading.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event-driven path (ingestion events → screen → analyze → queue)
# ---------------------------------------------------------------------------

def process_event(event: dict) -> dict:
    """
    Full pipeline for a single inbound event.

    Args:
        event: Raw event dict with keys: ticker, source, title, text, url

    Returns:
        analyst recommendation dict (or SCREENED_OUT stub)
    """
    from agents.screener import screen
    from agents.analyst import analyze
    from agents.market_regime import get_cached_regime
    from context_engine import assemble_brief

    ticker = event.get("ticker", "UNKNOWN")
    log.info(f"[{ticker}] Processing event: {event.get('source','?')} — {event.get('title','')[:60]}")

    # ── Store in vector memory ────────────────────────────────────────────────
    try:
        import vector_store
        vector_store.store(
            collection=_event_to_collection(event),
            doc_id=event.get("url") or event.get("title") or str(datetime.now()),
            text=event.get("text") or event.get("title", ""),
            metadata={
                "ticker": ticker,
                "source": event.get("source", ""),
                "date":   datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        log.debug(f"[{ticker}] Vector store: {e}")

    # ── Screen for relevance ──────────────────────────────────────────────────
    screening = screen(event)
    if not screening.get("relevant", False):
        reason = screening.get("reason", "Not relevant")
        log.info(f"[{ticker}] Screened out: {reason}")
        try:
            import db
            db.log_signal(ticker, "SCREENED_OUT", 0.0, reason)
        except Exception:
            pass
        return {"action": "SCREENED_OUT", "ticker": ticker, "conviction": 0, "pass_to_risk": False}

    log.info(f"[{ticker}] Relevant (urgency={screening.get('urgency')}) — assembling brief")

    # ── Assemble situation brief ──────────────────────────────────────────────
    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client)
        brief = assemble_brief(event, positions, client)
    except Exception as e:
        log.warning(f"[{ticker}] Brief assembly failed: {e}")
        brief = f"Event: {event.get('title', '')}\n{event.get('text', '')[:1000]}"
        positions = {}
        client = None

    # Build a candidate-like dict from the event (for analyst interface)
    from knowledge_loader import get_watchlist_entry
    wl_entry = get_watchlist_entry(ticker) or {}
    candidate = {
        "ticker":          ticker,
        "role":            wl_entry.get("role", "Accelerator"),
        "tier":            wl_entry.get("tier", "Watch"),
        "sector":          wl_entry.get("sector", ""),
        "composite_score": 60,  # event-triggered, score is nominal
    }

    # ── Analyze ────────────────────────────────────────────────────────────────
    regime = get_cached_regime()
    result = analyze(candidate, situation_brief=brief, regime_packet=regime)

    # ── Route to approval queue if actionable ─────────────────────────────────
    if result.get("pass_to_risk"):
        _route_to_approval(result, regime, client)

    return result


# ---------------------------------------------------------------------------
# Scanner-driven path (batch candidates from daily scan)
# ---------------------------------------------------------------------------

def process_candidates(candidates: list[dict], max_per_run: int = 5) -> list[dict]:
    """
    Analyze a batch of scanner candidates and queue actionable ones for approval.

    Args:
        candidates:  List of scanner candidate dicts (HIGH/MEDIUM priority)
        max_per_run: Max candidates to analyze (limits inference time)

    Returns:
        List of analyst recommendation dicts
    """
    from agents.analyst import analyze_batch
    from agents.market_regime import get_cached_regime

    regime = get_cached_regime()
    log.info(f"Processing {len(candidates)} candidates in {regime.get('regime')} regime")

    # Only analyze HIGH priority and top MEDIUM
    high     = [c for c in candidates if c.get("priority") == "HIGH"]
    medium   = [c for c in candidates if c.get("priority") == "MEDIUM"]
    to_analyze = (high + medium)[:max_per_run]

    if not to_analyze:
        log.info("No candidates to analyze")
        return []

    results = analyze_batch(to_analyze, regime_packet=regime)

    try:
        from ibkr import IBKRClient
        client = IBKRClient()
    except Exception as e:
        log.warning(f"Could not init IBKR client for routing: {e}")
        client = None

    for result in results:
        if result.get("pass_to_risk") and client:
            _route_to_approval(result, regime, client)

    return results


# ---------------------------------------------------------------------------
# Risk sizing + approval queue
# ---------------------------------------------------------------------------

def _route_to_approval(result: dict, regime: dict, trading_client) -> None:
    """
    Run risk sizing on an analyst recommendation and queue it for human approval.
    Sends a Telegram message with Approve / Reject buttons.
    """
    from risk_manager import RiskManager
    import db
    import alerts as alert_module

    ticker     = result["ticker"]
    conviction = result["conviction"]
    bucket     = result["thesis_bucket"]
    regime_str = regime.get("regime", "Neutral")

    try:
        from feed import emit as _feed_emit
    except Exception:
        _feed_emit = None

    # Skip if this ticker already has an open approval waiting for decision
    try:
        if db.has_active_approval(ticker):
            log.info(f"[{ticker}] Active approval already pending — skipping duplicate")
            return
    except Exception:
        pass

    try:
        from catalyst_service.gates import catalyst_gate_check
        catalyst_gate = catalyst_gate_check(ticker, days_ahead=45)
        if not catalyst_gate.get("approved"):
            reason = catalyst_gate.get("reason", "unresolved catalyst calendar risk")
            log.warning(f"[{ticker}] Approval blocked by catalyst gate: {reason}")
            if _feed_emit:
                _feed_emit("risk", ticker, f"Blocked by catalyst gate — {reason}", "alert")
            try:
                db.log_signal(ticker, "CATALYST_BLOCKED", 0.0, reason)
                alert_module._send(
                    f"CATALYST BLOCKED - {ticker}\n"
                    f"{reason}\n"
                    "Resolve the catalyst calendar entry before approval routing."
                )
            except Exception:
                pass
            return
    except Exception as e:
        log.warning(f"[{ticker}] Catalyst gate skipped: {e}")

    try:
        rm = RiskManager(trading_client)
    except Exception as e:
        log.error(f"[{ticker}] RiskManager init failed: {e}")
        return

    # Get current price
    price = _get_price(ticker, trading_client)
    if not price:
        log.warning(f"[{ticker}] Could not determine price — skipping approval queue")
        return

    # Size the position
    sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
    if sizing.get("shares_initial", 0) <= 0:
        flags = sizing.get("flags", [])
        log.info(f"[{ticker}] Risk sizing returned 0 shares: {flags}")
        return

    # Concentration check
    conc = rm.check_concentration(ticker, sizing["position_size_pct"])
    if not conc["passed"]:
        log.info(f"[{ticker}] Concentration check failed: {conc['flags']}")
        # Still queue but with a flag in entry_strategy
        sizing["flags"].extend(conc["flags"])

    # Correlation check
    corr = rm.check_correlation(ticker)
    if corr["reduce_by"] > 0:
        log.info(f"[{ticker}] Correlation adjustment: reducing size 25% ({corr['note']})")
        sizing["shares_initial"] = max(1, int(sizing["shares_initial"] * 0.75))
        sizing["initial_dollar"] = round(sizing["initial_dollar"] * 0.75, 2)

    entry_strategy = (
        f"{sizing['initial_entry_pct']:.0%} initial ({sizing['shares_initial']} shares @ ~${price:.2f}). "
        f"Add on thesis confirmation. Flags: {', '.join(sizing['flags']) or 'none'}."
    )

    # Log to pending_approvals
    approval_id = db.log_pending_approval(
        ticker=ticker,
        action=result["action"],
        conviction=conviction,
        bucket=bucket,
        size_pct=sizing["position_size_pct"],
        stop_price=sizing.get("stop_loss_price"),
        entry_strategy=entry_strategy,
        reasoning=result.get("reasoning", ""),
        analyst_json=result,
        risk_json=sizing,
    )

    if approval_id < 0:
        log.error(f"[{ticker}] Failed to log pending approval")
        return

    # Send Telegram approval request
    msg_id = alert_module.send_approval_request(
        approval_id=approval_id,
        ticker=ticker,
        action=result["action"],
        conviction=conviction,
        bucket=bucket,
        size_pct=sizing["position_size_pct"],
        stop_price=sizing.get("stop_loss_price"),
        entry_strategy=entry_strategy,
        reasoning=result.get("reasoning", ""),
    )

    if msg_id:
        db.set_approval_telegram_id(approval_id, msg_id)
        log.info(f"[{ticker}] Approval #{approval_id} queued — Telegram msg_id={msg_id}")
        if _feed_emit:
            _feed_emit("risk", ticker,
                       f"Sized {sizing['shares_initial']} shares @ ~${price:.2f} "
                       f"({sizing['position_size_pct']:.1%} portfolio) | "
                       f"stop ${sizing.get('stop_loss_price', 0):.2f} | "
                       f"{bucket} bucket — approval request sent to Telegram",
                       "bullish")
    else:
        log.info(f"[{ticker}] Approval #{approval_id} queued (Telegram not configured)")
        if _feed_emit:
            _feed_emit("risk", ticker,
                       f"Sized {sizing['shares_initial']} shares @ ~${price:.2f} — "
                       f"approval #{approval_id} logged (Telegram not configured)",
                       "neutral")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_positions(client) -> dict:
    try:
        positions = {}
        for p in client.get_all_positions():
            entry   = float(p.avg_entry_price)
            current = float(p.current_price or entry)
            positions[p.symbol] = {
                "qty":          float(p.qty),
                "entry_price":  entry,
                "current_price": current,
                "pl_pct":       (current - entry) / entry,
                "market_value": float(p.market_value),
            }
        return positions
    except Exception:
        return {}


def _get_price(ticker: str, client) -> float | None:
    try:
        from ingestion.polygon_stream import get_snapshot
        snap = get_snapshot(ticker)
        if snap.get("price"):
            return float(snap["price"])
    except Exception:
        pass
    try:
        from market_data import get_latest_quotes
        quotes = get_latest_quotes([ticker])
        mid = quotes.get(ticker, {}).get("mid")
        if mid:
            return float(mid)
    except Exception:
        pass
    # Final fallback: yfinance (no API key, always available)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        if price:
            p = float(price)
            # Sanity check: reject anything above $5,000 or below $0.01
            # yfinance occasionally returns split-adjusted historical prices or garbage
            if 0.01 <= p <= 5000.0:
                return p
            log.warning(f"[{ticker}] yfinance returned implausible price ${p:.2f} — discarding")
    except Exception:
        pass
    return None


def _event_to_collection(event: dict) -> str:
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
