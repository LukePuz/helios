"""
Crew Orchestrator — coordinates the full agent pipeline.

Entry points:
  1. screen_event_only(event) + deep_analyze(event, screening) —
     event-driven (SEC filing, options flow, news, etc.), split into two
     stages that run on separate Celery queues/workers (tasks.screen_event
     on the fast CPU-only "screen" queue, tasks.deep_analyze_event on the
     slow GPU-bound llm_p0/llm_p1 queues) so a quick screen-reject never
     waits behind however many multi-minute 72B analyses are already
     queued. process_event(event) is a synchronous wrapper combining both,
     kept for direct/manual callers — the live system always uses the two
     stages separately.
  2. process_candidates(candidates) — scanner-driven (daily scan output),
     already pre-filtered by the scanner's own composite score, so it
     skips the screening stage entirely: Analyze → Risk Size → Queue for
     Approval.

Paper trading only — recommendations that clear Risk Manager auto-execute
immediately (see _route_to_approval) and a Telegram trade alert is sent
after the fact. No human-in-the-loop tap, per explicit user request
2026-07-14. Every other risk control (drawdown halt, concentration caps,
price-drift abort, ATR stops) is unchanged.
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

def screen_event_only(event: dict) -> dict:
    """
    Vector-store write + relevance screening only — the fast (7B, CPU)
    stage. Does not touch the analyst or risk pipeline.

    Split out from the old process_event so screening has its own
    queue/worker (tasks.screen_event) and never waits behind the
    expensive 72B analysis stage (tasks.deep_analyze_event) — on a busy
    news day, a screen-reject used to sit behind however many ~170s
    analyses were already queued just to get its own ~5s verdict.

    Returns the screening result dict.
    """
    from agents.screener import screen

    ticker = event.get("ticker", "UNKNOWN")
    log.info(f"[{ticker}] Screening event: {event.get('source','?')} — {event.get('title','')[:60]}")

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
    else:
        log.info(f"[{ticker}] Relevant (urgency={screening.get('urgency')}) — queued for deep analysis")

    return screening


def deep_analyze(event: dict, screening: dict | None = None) -> dict:
    """
    Full 72B analyst pipeline for an event that already passed screening
    — brief assembly, analysis, and approval routing. Assumes
    screen_event_only already confirmed relevance (does not re-screen).

    Args:
        event:     Raw event dict with keys: ticker, source, title, text, url
        screening: The screening result from screen_event_only, if
                   available (currently informational only — the brief
                   builder pulls its own context — kept as a parameter so
                   future work can use urgency/affected_tickers without
                   another screener call).

    Returns:
        analyst recommendation dict
    """
    from agents.analyst import analyze
    from agents.market_regime import get_cached_regime
    from context_engine import assemble_brief

    ticker = event.get("ticker", "UNKNOWN")
    log.info(f"[{ticker}] Deep analysis: {event.get('source','?')} — {event.get('title','')[:60]}")

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
    try:
        result = analyze(candidate, situation_brief=brief, regime_packet=regime)
    except Exception as exc:
        log.error(f"[{ticker}] analyze() raised an unhandled exception: {exc}", exc_info=True)
        result = {
            "action": "SKIP", "ticker": ticker, "conviction": 0,
            "pass_to_risk": False,
            "reasoning": f"Analysis failed with exception: {exc}",
        }

    # ── Route to approval queue if actionable ─────────────────────────────────
    if result.get("pass_to_risk"):
        _route_to_approval(result, regime, client)

    return result


def process_event(event: dict) -> dict:
    """
    Convenience wrapper combining screen_event_only + deep_analyze in one
    synchronous call — kept for direct/manual callers (e.g. running a
    single event through the pipeline interactively or in tests). The
    live system uses the two stages separately via tasks.screen_event /
    tasks.deep_analyze_event on their own queues, so screening never
    waits behind the expensive analysis stage.

    Args:
        event: Raw event dict with keys: ticker, source, title, text, url

    Returns:
        analyst recommendation dict (or SCREENED_OUT stub)
    """
    screening = screen_event_only(event)
    if not screening.get("relevant", False):
        return {"action": "SCREENED_OUT", "ticker": event.get("ticker", "UNKNOWN"),
                "conviction": 0, "pass_to_risk": False}
    return deep_analyze(event, screening)


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

    # Built BEFORE analyze_batch (not after, as before) so each candidate's
    # situation brief can include live portfolio/account context.
    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client)
    except Exception as e:
        log.warning(f"Could not init IBKR client for brief/routing: {e}")
        client = None
        positions = {}

    results = analyze_batch(to_analyze, regime_packet=regime,
                            positions=positions, trading_client=client)

    for result in results:
        if result.get("pass_to_risk") and client:
            _route_to_approval(result, regime, client)

    return results


# ---------------------------------------------------------------------------
# Risk sizing + approval queue
# ---------------------------------------------------------------------------

def _route_to_approval(result: dict, regime: dict, trading_client) -> None:
    """
    Run risk sizing on an analyst recommendation and auto-execute it (paper
    trading only — no human approval gate, see module docstring). Sends a
    Telegram trade alert after execution instead of an approval request.
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

    action = result["action"]

    if action in ("OPEN", "ADD"):
        try:
            from config import REBUY_COOLDOWN_HOURS
        except ImportError:
            REBUY_COOLDOWN_HOURS = 4
        try:
            prior = db.get_last_executed_approval(ticker, REBUY_COOLDOWN_HOURS)
        except Exception:
            prior = None
        if prior and conviction <= prior["conviction"]:
            log.info(
                f"[{ticker}] Rebuy cooldown — approved {prior['conviction']}/10 at "
                f"{prior['decided_at']}, new signal is {conviction}/10 (not higher) — skipping"
            )
            try:
                db.log_signal(ticker, "COOLDOWN_SKIPPED", conviction / 10.0,
                              f"Within {REBUY_COOLDOWN_HOURS}h of prior approved buy "
                              f"at conviction {prior['conviction']}/10")
            except Exception:
                pass
            return

    if action in ("TRIM", "EXIT"):
        # Risk-reducing sells are sized from the held position, never through
        # size_position() — buy sizing returns 0 shares for at-target held
        # names, which silently dropped every analyst sell (the system could
        # enter risk automatically but never exit it). The catalyst gate and
        # concentration/correlation checks are buy gates; a sell must never
        # be blocked by them.
        try:
            pos   = trading_client.get_open_position(ticker)
            owned = abs(float(pos.qty))
        except Exception:
            owned = 0.0
        if owned <= 0:
            log.info(f"[{ticker}] {action} recommended but no position held — signal logged only")
            try:
                db.log_signal(ticker, "SELL_NO_POSITION", conviction / 10.0,
                              result.get("reasoning", "")[:500])
            except Exception:
                pass
            return

        try:
            from config import TRIM_FRACTION
        except ImportError:
            TRIM_FRACTION = 0.33
        sell_qty = owned if action == "EXIT" else min(owned, max(1, int(owned * TRIM_FRACTION)))

        price = _get_price(ticker, trading_client) or float(getattr(pos, "current_price", 0) or 0)
        sizing = {
            "ticker":            ticker,
            "position_size_pct": 0,
            "dollar_amount":     round(sell_qty * price, 2),
            "initial_entry_pct": 1.0,
            "initial_dollar":    round(sell_qty * price, 2),
            "shares_initial":    sell_qty,
            "stop_loss_price":   None,
            "flags":             [],
            "bucket":            bucket,
            "conviction":        conviction,
            "price_at_sizing":   price,
        }
        entry_strategy = f"{action}: sell {sell_qty:g} of {owned:g} held shares @ ~${price:.2f}."
    else:
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

        # Size the position — stamp the price used so execution can validate freshness
        sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
        sizing["price_at_sizing"] = price
        if sizing.get("shares_initial", 0) <= 0:
            flags = sizing.get("flags", [])
            log.info(f"[{ticker}] Risk sizing returned 0 shares: {flags}")
            return

        # Concentration check
        conc = rm.check_concentration(ticker, sizing["position_size_pct"], bucket=bucket)
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

    # Paper trading only, no real capital at risk — auto-execute instead of
    # waiting on a Telegram approval tap. Per explicit user request 2026-07-14
    # (see CLAUDE.md); every other risk control (drawdown halt, concentration
    # caps, price-drift abort, ATR stop placement) still applies unchanged,
    # only the human-in-the-loop tap is removed.
    from execution import execute_approved

    if not db.claim_approval_for_execution(approval_id):
        log.error(f"[{ticker}] Could not claim approval #{approval_id} for auto-execution")
        return

    approval = db.get_approval_by_id(approval_id)
    success = execute_approved(approval, trigger="auto")
    db.update_approval_status(approval_id, "APPROVED" if success else "FAILED")
    log.info(f"[{ticker}] Approval #{approval_id} auto-{'executed' if success else 'execution FAILED'}")

    if _feed_emit:
        if success:
            # stop_loss_price is None for TRIM/EXIT (sells don't carry an
            # entry stop target) — .get(..., 0) doesn't help when the key
            # is present but None, so this needs an explicit check. This
            # branch previously only ever saw BUYs (TRIM/EXIT was dropped
            # earlier by the buy-sizing bug fixed above), so "bullish" was
            # never wrong until sells could reach it.
            stop_px = sizing.get("stop_loss_price")
            stop_str = f"${stop_px:.2f}" if stop_px else "n/a"
            sentiment = "bullish" if action in ("OPEN", "ADD") else "bearish"
            _feed_emit("execution", ticker,
                       f"Auto-executed {result['action']} {sizing['shares_initial']} shares @ ~${price:.2f} "
                       f"({sizing['position_size_pct']:.1%} portfolio) | "
                       f"stop {stop_str} | {bucket} bucket",
                       sentiment)
        else:
            _feed_emit("execution", ticker,
                       f"Auto-execution FAILED for {result['action']} {ticker} — check logs", "alert")

    # _execute_approved_order already sends a Telegram trade alert on success;
    # only notify separately here on failure, so a broken execution doesn't
    # fail silently.
    if not success:
        try:
            alert_module._send(
                f"⚠️ AUTO-EXECUTION FAILED\n"
                f"<b>{result['action']} {ticker}</b> (approval #{approval_id}) could not be executed.\n"
                f"Check helios.log for details."
            )
        except Exception:
            pass


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
        # Prefer extended-hours price so sizing reflects what we'd actually pay
        full = t.info
        price = (
            full.get("postMarketPrice")
            or full.get("preMarketPrice")
            or full.get("regularMarketPrice")
            or getattr(t.fast_info, "last_price", None)
        )
        if price:
            p = float(price)
            if 0.01 <= p <= 50000.0:
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
