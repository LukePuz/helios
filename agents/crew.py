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
            prov = screening.get("provenance") or {}
            db.log_signal(ticker, "SCREENED_OUT", 0.0, reason,
                          model=prov.get("model"), provenance=prov or None)
        except Exception as e:
            log.warning(f"[{ticker}] SCREENED_OUT signal log failed: {e}")
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
    held_as_of = None
    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client, strict=True)
        held_as_of = datetime.now(timezone.utc).isoformat()
        if positions is None:
            log.warning(f"[{ticker}] Broker positions unavailable — skipping analysis "
                        f"rather than analysing against an empty book")
            return {"action": "SKIP", "ticker": ticker, "conviction": 0, "pass_to_risk": False,
                    "reasoning": "Skipped: broker positions unavailable"}
        # Held tickers get a price via the CURRENT PORTFOLIO section; anything
        # not held previously had NO price anywhere in the brief, so the model
        # would invent one for any dollar-denominated valuation output (found
        # 2026-07-17: a SNDK target of "$300" while it traded near $1,350).
        price = _get_price(ticker, client)
        brief = assemble_brief(event, positions, client, current_price=price)
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
        # None when the brief failed; analyze() only coerces on a known value
        "held":            (ticker in positions) if client is not None else None,
        "held_as_of":      held_as_of if client is not None else None,   # M138
    }

    # ── Analyze ────────────────────────────────────────────────────────────────
    regime = get_cached_regime()
    try:
        from agents.analyst import trigger_record
        trigger = trigger_record("event", event)
        if screening and screening.get("override"):
            trigger["screen_override"] = screening["override"]   # review 4 #4
        result = analyze(candidate, situation_brief=brief, regime_packet=regime,
                         trigger=trigger)
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


# ---------------------------------------------------------------------------
# Scheduled position review — the only other paths into analyze() are new
# scanner candidates (OPEN-only) and ingestion events that happen to fire on
# an already-held ticker, so without this, TRIM/EXIT could only ever occur
# by coincidence. Confirmed 2026-07-25: 37 executed trades ever, zero EXITs,
# one TRIM -- nothing was ever periodically asked "should we still hold
# this?" independent of a new external event. _route_to_approval's TRIM/EXIT
# handling (sized from the held position, bypassing buy gates) already
# worked correctly; this closes the actual gap upstream of it.
# ---------------------------------------------------------------------------

def _log_route_exit(ticker: str, kind: str, result: dict, reason: str) -> None:
    """A passed verdict that dies in routing gets a signals row, like
    EARNINGS_BLOCKED already did (professor M30, 2026-10-28): these exits were
    log.info only, so the final disposition of a pass wasn't observable."""
    try:
        import db
        db.log_signal(ticker, kind, (result.get("conviction") or 0) / 10.0, reason[:500],
                      **_route_row_fields(result, kind))
    except Exception as e:
        log.warning(f"[{ticker}] route-exit signal log failed: {e}")


_VERDICT_ONLY_PROVENANCE = ("plan", "brief_exposure", "gate_flags")


def _route_row_fields(result: dict, kind: str) -> dict:
    """Model and provenance of the verdict a routing row belongs to, so the
    row stands on its own (SEF run 13, 2026-10-28: ROUTE_* rows had NULL model
    and provenance and had to be joined to the preceding verdict). The
    prompt_sha in provenance identifies that verdict's signals row."""
    prov = result.get("provenance") or None
    if prov and any(k in prov for k in _VERDICT_ONLY_PROVENANCE):
        # The plan observation belongs to the verdict row only; copied onto
        # routing rows it was counted 2-3 times (pre-pull review 10-29).
        # brief_exposure and gate_flags likewise.
        prov = {k: v for k, v in prov.items() if k not in _VERDICT_ONLY_PROVENANCE}
    return {
        "model": (prov or {}).get("model"),
        "provenance": prov,
        "decision": {"route_exit": kind, "verdict_action": result.get("action"),
                     "verdict_conviction": result.get("conviction")},
    }


# Swap shadow (Luke 10-29: "sell my lowest rated one and buy the higher rated
# one"). Log-only: records what a ranking swap would do when a buy is blocked
# for room. The live swap (find_swap_candidate) never fired: it skips the
# position cap, Compounders and holdings above c6.
_SHADOW_SWAP_MIN_CONVICTION = 7
_SHADOW_SWAP_MIN_GAP = 2
_SHADOW_SWAP_FRESH_DAYS = 3


def _room_blocked(sizing: dict) -> bool:
    return bool(sizing.get("cash_blocked") or sizing.get("heat_blocked")
                or any("concurrent positions" in f for f in sizing.get("flags", [])))


def _shadow_swap(rm, ticker: str, conviction: int, sizing: dict, result: dict, now=None) -> dict | None:
    """Lowest-rated holding with a fresh verdict, if the blocked buy beats it
    by the gap. Writes a SHADOW_SWAP row either way when it runs. Never raises."""
    try:
        if conviction < _SHADOW_SWAP_MIN_CONVICTION or not _room_blocked(sizing):
            return None
        import db
        import math
        from datetime import datetime, timedelta, timezone
        now = now or datetime.now(timezone.utc)
        held, stale = [], []
        for p in rm._client.get_all_positions():
            if p.symbol == ticker:
                continue
            sig = db.get_latest_signal_for(p.symbol)
            if not sig or not sig.get("timestamp") or now - sig["timestamp"] > timedelta(days=_SHADOW_SWAP_FRESH_DAYS):
                stale.append(p.symbol)
                continue
            mv, pl = float(p.market_value), float(p.unrealized_pl)
            pl_pct = round(100 * pl / (mv - pl), 1) if mv - pl and math.isfinite(mv) and math.isfinite(pl) else None
            held.append({"ticker": p.symbol, "conviction": round(sig["confidence"] * 10),
                         "signal": sig["signal"], "unrealized_pl_pct": pl_pct})
        # Lowest conviction first; among equals, the weakest performer.
        held.sort(key=lambda h: (h["conviction"], h["unrealized_pl_pct"] if h["unrealized_pl_pct"] is not None else 0))
        low = held[0] if held else None
        would = bool(low and conviction - low["conviction"] >= _SHADOW_SWAP_MIN_GAP)
        rec = {"would_swap": would, "buy": ticker, "buy_conviction": conviction,
               "sell": low["ticker"] if would else None, "lowest": low,
               "block": [f for f in sizing.get("flags", [])][:3], "stale_holdings": stale,
               "min_gap": _SHADOW_SWAP_MIN_GAP, "fresh_days": _SHADOW_SWAP_FRESH_DAYS}
        fields = _route_row_fields(result, "SHADOW_SWAP")
        fields["decision"]["shadow_swap"] = rec
        reason = (f"would sell {low['ticker']} (c{low['conviction']}) to buy {ticker} (c{conviction})"
                  if would else f"no swap: lowest fresh holding "
                  f"{(low or {}).get('ticker')} c{(low or {}).get('conviction')}")
        db.log_signal(ticker, "SHADOW_SWAP", conviction / 10.0, reason, **fields)
        log.info(f"[{ticker}] Swap shadow: {reason}")
        return rec
    except Exception as e:
        log.warning(f"[{ticker}] swap shadow failed: {e}")
        return None


def _earnings_review_event(ticker: str, earnings_date, today=None) -> dict:
    """Review event for a held name that reports within a few days, or just
    did (limitations audit 2026-10-27: no earnings playbook; held names went
    into prints with a generic "periodic review"). States the date so the
    model has to decide hold-through vs trim before, and react after."""
    from datetime import date
    days = (earnings_date - (today or date.today())).days
    if days > 0:
        when = f"reports earnings on {earnings_date} (in {days} calendar day{'s' if days != 1 else ''})"
        ask = ("Decide whether to hold through the print, trim, or exit before it. "
               "State the expected move, what result would break the thesis, and "
               "whether the current stop and size are right for an earnings gap.")
    elif days == 0:
        when = f"reports earnings today ({earnings_date})"
        ask = ("If the result is not out yet, decide whether to hold through it. If it "
               "is out, judge whether it confirmed or broke the thesis.")
    else:
        when = f"reported earnings on {earnings_date} ({-days} calendar day{'s' if days != -1 else ''} ago)"
        ask = ("Judge whether the result and the price reaction confirmed or broke the "
               "thesis, and whether the stop and size still fit.")
    return {
        "ticker": ticker, "source": "portfolio_review", "type": "earnings review",
        "title": f"Earnings position review — {ticker} {when}",
        "text": f"{ticker} {when}. {ask}",
    }


def _analyze_held_position(ticker: str, client, positions: dict, regime: dict,
                           event: dict | None = None, path: str | None = None) -> dict:
    """The brief-assembly + analyze() core shared by review_held_position
    and backfill_catalyst_tracking (2026-08-30) -- extracted so the
    backfill can get the same real catalyst/expected-return read every
    scheduled review already produces, without ALSO triggering
    review_held_position's approval-routing side effect (a backfill pass
    exists to capture a catalyst snapshot, not to fire a new TRIM/EXIT)."""
    from agents.analyst import analyze
    from context_engine import assemble_brief
    from knowledge_loader import get_watchlist_entry

    event = event or {
        "ticker": ticker, "source": "portfolio_review", "type": "scheduled review",
        "title": f"Periodic position review — {ticker}", "text": "",
    }
    price = positions.get(ticker, {}).get("current_price") or _get_price(ticker, client)
    brief = assemble_brief(event, positions, client, current_price=price)

    wl_entry = get_watchlist_entry(ticker) or {}
    candidate = {
        "ticker":          ticker,
        "role":            wl_entry.get("role", "Accelerator"),
        "tier":            wl_entry.get("tier", "Watch"),
        "sector":          wl_entry.get("sector", ""),
        "composite_score": 60,
    }

    from agents.analyst import trigger_record
    path = path or ("earnings_review" if event.get("type") == "earnings review" else "scheduled_review")
    try:
        return analyze(candidate, situation_brief=brief, regime_packet=regime,
                       trigger=trigger_record(path, event))
    except Exception as exc:
        log.error(f"[{ticker}] analyze() raised: {exc}", exc_info=True)
        return {
            "action": "SKIP", "ticker": ticker, "conviction": 0, "pass_to_risk": False,
            "reasoning": f"Review failed with exception: {exc}",
        }


def _capture_latest_thesis(ticker: str, result: dict) -> None:
    """Persists the FULL analyst result from a scheduled review, every
    cycle, regardless of action -- fixes the gap found 2026-09-13
    reconstructing Helios's Friday QBTS review for the user: a plain HOLD
    verdict's full reasoning was previously lost, only a 176-char
    activity_feed line survived (analyst.analyze already truncates to
    500 chars for db.log_signal, and only OPEN/ADD/TRIM/EXIT that pass
    the conviction/UDR gate ever reach pending_approvals.analyst_json).
    Failures are logged, never raised -- losing a thesis snapshot must
    never block the review itself.

    Stub results (model unavailable) are skipped: on outage days they wrote
    "conviction 1, WATCH" snapshots that looked like real verdicts."""
    if result.get("model_used") == "stub":
        log.info(f"[{ticker}] Skipping thesis capture for stub result (model unavailable)")
        return
    try:
        import db
        from datetime import datetime, timezone
        weeks_to_catalyst = result.get("weeks_to_catalyst")
        thesis = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "setup_category": None,  # not yet classified live -- setup_gates.py runs in shadow mode
            "action": result.get("action"),
            "conviction": result.get("conviction"),
            "thesis_summary": result.get("thesis_summary"),
            "gate_criteria": None,
            "catalyst": result.get("catalyst"),
            "catalyst_status": "pending",
            "expected_return_pct": result.get("expected_return_pct"),
            "expected_downside_pct": result.get("expected_downside_pct"),
            "weeks_to_catalyst": weeks_to_catalyst,
            "no_near_term_catalyst": db.compute_no_near_term_catalyst(result.get("catalyst"), weeks_to_catalyst),
            "catalyst_stale": result.get("catalyst_stale", False),
            "valuation_tier": None,
            "key_risks": result.get("key_risks", []),
            "reasoning": result.get("reasoning"),
            "provenance": result.get("provenance"),
        }
        db.update_latest_thesis(ticker, thesis)

        # #4 (2026-09-15, user request): a legacy position matching ZERO
        # live setup categories AND carrying no genuine near-term catalyst
        # is exactly the profile the swing-trading redesign says shouldn't
        # sit unexamined -- surfaced as a visible activity_feed alert here
        # (never a silent log line), NOT an auto-trim. See
        # agents.setup_gates.check_legacy_position_health's docstring.
        try:
            from agents.setup_gates import check_legacy_position_health
            check_legacy_position_health(ticker, thesis)
        except Exception as e:
            log.debug(f"[{ticker}] legacy position health check failed: {e}")
    except Exception as e:
        # WARNING, not debug (2026-09-17, found investigating a silent
        # failure): a lost thesis snapshot is non-fatal to the review
        # itself but IS real data loss for the drift-tracking system --
        # a 2026-09-15 off-schedule dispatch via tasks.review_positions.
        # delay() (no explicit queue routing) landed on the busy shared
        # worker queue, likely hit DB pool contention against concurrently
        # running reconcile-stops/check-profit-targets/portfolio-snapshot,
        # and failed here completely invisibly at debug level.
        log.warning(f"[{ticker}] latest thesis capture failed (non-fatal to the review, but the snapshot is lost): {e}")


def review_held_position(ticker: str, client, positions: dict, regime: dict,
                         earnings_date=None) -> dict:
    """Reassess one already-held position via the same brief/analyze/route
    pipeline as the event-driven path, using a synthetic 'scheduled review'
    event instead of a real filing/news item. With earnings_date, the event
    is an earnings review (_earnings_review_event)."""
    log.info(f"[{ticker}] Scheduled position review"
             + (f" (earnings {earnings_date})" if earnings_date else ""))
    event = _earnings_review_event(ticker, earnings_date) if earnings_date else None
    result = _analyze_held_position(ticker, client, positions, regime, event=event)
    _capture_latest_thesis(ticker, result)

    if result.get("pass_to_risk"):
        _route_to_approval(result, regime, client)

    return result


def review_plan_level(ticker: str, kind: str, level, price: float | None) -> dict | None:
    """Review one held position because its trade plan's level was hit
    (tasks.watch_plan_levels, only with PLAN_LEVEL_REVIEW_ENABLED). Same
    brief/analyze/route pipeline as the scheduled review, with an event that
    names the level. None if the name is no longer held."""
    from ibkr import IBKRClient
    from agents.market_regime import get_cached_regime

    client = IBKRClient()
    positions = _get_positions(client, strict=True)
    if not positions or ticker not in positions:
        log.info(f"[{ticker}] Plan-level review skipped: not held (or positions unavailable)")
        return None
    regime = get_cached_regime()
    at = f" at about ${price:.2f}" if price else ""
    event = {
        "ticker": ticker, "source": "plan_watcher", "type": "plan level review",
        "title": f"Plan level hit — {ticker}: {kind} ({level})",
        "text": (f"{ticker} reached its trade plan's {kind} level ({level}){at}. Judge whether the "
                 f"thesis and the plan still hold, and whether to hold, trim or exit."),
    }
    result = _analyze_held_position(ticker, client, positions, regime, event=event,
                                    path="plan_level_review")
    _capture_latest_thesis(ticker, result)
    try:
        # Counts as this name's review for the 10:15 rotation's ordering.
        import json
        import db
        last = json.loads(db.get_state("position_review_last_reviewed") or "{}")
        last[ticker] = datetime.now(timezone.utc).isoformat()
        db.set_state("position_review_last_reviewed", json.dumps(last))
    except Exception as e:
        log.warning(f"[{ticker}] Plan-level review: rotation cursor not updated: {e}")
    if result.get("pass_to_risk"):
        _route_to_approval(result, regime, client)
    return result


def review_all_held_positions(max_per_run: int = 10, priority_tickers: list[str] | None = None) -> list[dict]:
    """
    Batch driver (tasks.review_positions) — capped per run since each review
    is a full 72B call that competes with new-candidate analysis on the same
    GPU queue. Rotates through every held position over successive runs
    (oldest-reviewed-first, tracked in system_state) rather than always
    reviewing the same few if there are more positions than max_per_run.

    priority_tickers (2026-08-29, user request): tickers to place at the
    FRONT of the rotation regardless of when they were last reviewed --
    fed by risk_manager.flag_stale_positions() via tasks.review_positions,
    so a position that's gone stale (old AND flat) gets a fresh LLM look
    promptly instead of possibly waiting weeks behind other holdings in
    the normal oldest-reviewed-first queue. Priority tickers are still
    subject to max_per_run and still get their own last_reviewed
    timestamp updated like any other reviewed position -- this only
    changes ORDER, not whether a position is eligible or how many run per
    cycle.
    """
    import json
    from datetime import datetime, timezone
    import db
    from ibkr import IBKRClient
    from agents.market_regime import get_cached_regime

    client = IBKRClient()
    positions = _get_positions(client, strict=True)
    if positions is None:
        log.warning("Position review skipped: broker positions unavailable")
        return []
    regime = get_cached_regime()

    try:
        last_reviewed = json.loads(db.get_state("position_review_last_reviewed") or "{}")
    except Exception:
        last_reviewed = {}

    # Names reporting within ~T-2..T+1 go first, every day of that window.
    try:
        earnings = db.get_earnings_around(list(positions.keys()))
    except Exception as e:
        log.warning(f"Position review: earnings lookup failed: {e}")
        earnings = {}

    priority_set = set(priority_tickers or [])
    tickers = sorted(
        positions.keys(),
        key=lambda t: (0 if t in earnings else 1 if t in priority_set else 2,
                       last_reviewed.get(t, "")),
    )[:max_per_run]
    log.info(f"Position review: {len(tickers)} of {len(positions)} held tickers this run"
             + (f" ({len(priority_set & set(tickers))} prioritized as stale)" if priority_set else ""))

    results = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for ticker in tickers:
        try:
            results.append(review_held_position(ticker, client, positions, regime,
                                                earnings_date=earnings.get(ticker)))
        except Exception as e:
            log.error(f"[{ticker}] Position review failed: {e}")
        last_reviewed[ticker] = now_iso

    try:
        db.set_state("position_review_last_reviewed", json.dumps(last_reviewed))
    except Exception as e:
        log.warning(f"Position review cursor save failed: {e}")

    return results


def backfill_catalyst_tracking(max_per_run: int = 10) -> list[dict]:
    """One-time gap-closer (2026-08-30, user request): none of the
    positions held before the catalyst/expected-return tracking work
    shipped today have a catalyst_event captured in trade_outcomes.
    Real trades going forward get this automatically at execution time
    (execution.py); this backfills what's already held.

    Uses the exact same brief/analyze() pipeline as a scheduled position
    review (_analyze_held_position) -- NOT review_held_position itself,
    which would also fire _route_to_approval on a pass_to_risk result.
    A backfill's job is to capture today's catalyst snapshot for
    tracking, not to trigger a new TRIM/EXIT as a side effect of
    checking.

    Capped at max_per_run per call (same reasoning as review_all_held_
    positions: each analysis is a full 72B call competing with real
    candidate analysis on the same GPU queue) -- naturally resumable
    across repeated calls, since "still missing catalyst data" IS the
    remaining-work signal, no separate cursor needed the way the
    oldest-reviewed-first rotation needs one.

    Returns [{"ticker", "action": "BACKFILLED" or "SKIPPED", ...}] for
    every ticker attempted this run.
    """
    import db
    from ibkr import IBKRClient
    from agents.market_regime import get_cached_regime

    client = IBKRClient()
    positions = _get_positions(client)
    regime = get_cached_regime()

    missing = db.get_tickers_missing_catalyst_data(list(positions.keys()))[:max_per_run]
    log.info(f"Catalyst backfill: {len(missing)} of {len(positions)} held tickers missing catalyst data this run")

    results = []
    for ticker in missing:
        result = _analyze_held_position(ticker, client, positions, regime, path="backfill")
        if result.get("action") == "SKIP":
            results.append({"ticker": ticker, "action": "SKIPPED", "reason": result.get("reasoning")})
            continue

        stop_rec = db.get_position_stop(ticker)
        pos = positions.get(ticker, {})
        entry_price = (stop_rec or {}).get("entry_price") or pos.get("entry_price")
        entry_date = (stop_rec or {}).get("opened_at") or datetime.now(timezone.utc)
        bucket = (stop_rec or {}).get("thesis_bucket") or result.get("thesis_bucket", "Accelerator")
        conviction = (stop_rec or {}).get("conviction") or result.get("conviction", 5)

        catalyst = result.get("catalyst") or {}
        db.backfill_catalyst_outcome(
            ticker=ticker,
            conviction=conviction,
            bucket=bucket,
            entry_price=entry_price,
            entry_date=entry_date,
            catalyst_event=catalyst.get("event"),
            catalyst_date=catalyst.get("date"),
            catalyst_is_priced_in=catalyst.get("is_priced_in"),
            expected_return_pct=result.get("expected_return_pct"),
            expected_downside_pct=result.get("expected_downside_pct"),
            thesis_summary=result.get("thesis_summary"),
            sources=result.get("sources"),
            source_count=result.get("source_count"),
            weeks_to_catalyst=result.get("weeks_to_catalyst"),
            expected_return_per_week=result.get("expected_return_per_week"),
        )
        log.info(f"[{ticker}] Catalyst backfilled: {catalyst.get('event', 'none identified')} "
                 f"(sources={result.get('source_count', 0)}, expected_return={result.get('expected_return_pct')})")
        results.append({"ticker": ticker, "action": "BACKFILLED",
                         "catalyst_event": catalyst.get("event"),
                         "source_count": result.get("source_count")})

    return results


def research_ticker(ticker: str) -> dict:
    """
    Read-only, on-demand run of the full analyst pipeline for a single
    ticker — real situation brief, real 72B analysis, real conviction
    scoring and evidence, triggered by an explicit chat request (see
    chat_assistant.py's request_deep_analysis tool) rather than an
    ingestion event.

    Deliberately never calls _route_to_approval: no matter what the
    analyst concludes — even a high-conviction OPEN — this can never size
    or execute a trade. A curiosity question in chat must never become a
    real order. Screening is skipped too: screen_event_only exists to
    filter high-volume ambient news, but an explicit "analyze TICKER"
    request is relevant by definition.
    """
    from agents.analyst import analyze
    from agents.market_regime import get_cached_regime
    from context_engine import assemble_brief

    ticker = ticker.upper().strip()
    event = {
        "source": "chat_request",
        "ticker": ticker,
        "title":  f"User-requested research on {ticker}",
        "text":   "",
    }
    log.info(f"[{ticker}] Chat-requested research (read-only, never executes)")

    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client)
        price = _get_price(ticker, client)
        brief = assemble_brief(event, positions, client, current_price=price)
    except Exception as e:
        log.warning(f"[{ticker}] Brief assembly failed: {e}")
        brief = f"User-requested research on {ticker}. No live event context available."

    from knowledge_loader import get_watchlist_entry
    wl_entry = get_watchlist_entry(ticker) or {}
    candidate = {
        "ticker":          ticker,
        "role":            wl_entry.get("role", "Accelerator"),
        "tier":            wl_entry.get("tier", "Watch"),
        "sector":          wl_entry.get("sector", ""),
        "composite_score": 60,  # chat-triggered, score is nominal
    }

    regime = get_cached_regime()
    try:
        from agents.analyst import trigger_record
        result = analyze(candidate, situation_brief=brief, regime_packet=regime,
                         trigger=trigger_record("chat", {"source": "chat", "title": f"research {ticker}"}))
    except Exception as exc:
        log.error(f"[{ticker}] research_ticker analyze() raised: {exc}", exc_info=True)
        result = {
            "action": "SKIP", "ticker": ticker, "conviction": 0,
            "pass_to_risk": False,
            "reasoning": f"Analysis failed with exception: {exc}",
        }

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

def _rule_shadow_average_down(ticker: str, bucket: str, trading_client) -> None:
    """Shadow check of a written rule nothing enforces: "Moonshots: never
    average down" (portfolio_philosophy.md). Logged only (audit 14:53,
    A1453-5); whether it becomes a gate is Luke's decision."""
    if bucket != "Moonshot":
        return
    try:
        import db
        entry = float((db.get_position_stop(ticker) or {}).get("entry_price") or 0)
        price = float(_get_price(ticker, trading_client) or 0)
    except Exception:
        return
    if entry and price and price < entry:
        log.warning(f"[{ticker}] rule shadow (Moonshots: never average down): ADD at "
                    f"${price:.2f} below entry ${entry:.2f}; not enforced")


_UNENCODABLE = object()


def _json_safe(value):
    """A shadow value as plain JSON, or _UNENCODABLE. A value json can't
    encode would make the approval insert fail and drop the buy."""
    import json
    try:
        return json.loads(json.dumps(value, default=float, allow_nan=False))   # jsonb rejects NaN
    except Exception:
        return _UNENCODABLE


def _attach_shadows(sizing: dict, ticker: str, result: dict, price: float) -> None:
    """Log-only fields on a sizing dict, each guaranteed JSON-safe, with
    shadow_status saying why one is missing: ok, n/a (doesn't apply),
    error (the helper raised; the dict holds the message) or unencodable."""
    stop = sizing.get("stop_loss_price")
    values = {"r_shadow": _r_shadow(result, price, stop),
              "plan_compiled": _compiled_plan(ticker, result, price, stop)}
    if "heat_shadow" in sizing:
        values["heat_shadow"] = sizing["heat_shadow"]
    status = {}
    for name, value in values.items():
        safe = _json_safe(value)
        if safe is _UNENCODABLE:
            sizing[name], status[name] = None, "unencodable"
        else:
            sizing[name] = safe
            status[name] = ("n/a" if safe is None
                            else "error" if isinstance(safe, dict) and "error" in safe else "ok")
    sizing["shadow_status"] = status


def _compiled_plan(ticker: str, result: dict, price: float, stop) -> dict | None:
    """Shadow plan from the analyst's view plus the price ladder (audit
    A2114-1). Log-only, never raises."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import db
        from agents.plans import compile_plan
        return compile_plan(result, price, db.get_trade_ladder(ticker), stop,
                            datetime.now(ZoneInfo("America/New_York")).date())
    except Exception as e:
        log.debug(f"[{ticker}] compiled plan failed: {e}")
        return {"error": str(e)[:200]}


def _r_shadow(result: dict, price: float, stop) -> dict | None:
    """Reward-to-risk against the stop that will actually be placed, next to
    the model's own upside_downside_ratio (audit A2114-2: the UDR >= 2 gate
    uses the model's imagined downside, not the real stop). Log-only."""
    try:
        from agents.plans import _dollars
        target = _dollars((result.get("valuation") or {}).get("bull_case_target"))
        source = "bull_case_target"
        if not target and isinstance(result.get("expected_return_pct"), (int, float)):
            target, source = price * (1 + result["expected_return_pct"] / 100), "expected_return_pct"
        stop = float(stop) if stop else None
        r = (round((target - price) / (price - stop), 2)
             if target and stop and target > price > stop else None)
        return {"computed_r": r, "model_udr": result.get("upside_downside_ratio"),
                "target": round(target, 2) if target else None, "target_source": source if target else None,
                "stop": stop, "stop_pct": round((price - stop) / price * 100, 2) if stop else None}
    except Exception as e:
        log.debug(f"R shadow failed: {e}")
        return {"error": str(e)[:200]}


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
            _log_route_exit(ticker, "ROUTE_APPROVAL_PENDING", result, "active approval already pending")
            return
    except Exception:
        pass

    action = result["action"]
    if action == "ADD":
        _rule_shadow_average_down(ticker, bucket, trading_client)

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
        # A failed broker read is not "no position": it used to drop the
        # sell silently as SELL_NO_POSITION (audit A1031-1253-5).
        positions = _get_positions(trading_client, strict=True)
        if positions is None:
            log.warning(f"[{ticker}] {action} recommended but broker positions unreadable — not sold")
            try:
                db.log_signal(ticker, "SELL_POSITION_UNREADABLE", conviction / 10.0,
                              result.get("reasoning", "")[:500])
            except Exception:
                pass
            try:
                import alerts
                alerts._send(f"⚠️ SELL NOT SENT — {ticker}\nThe model recommended {action}, but broker "
                             f"positions couldn't be read, so nothing was sold. Check TWS; the next "
                             f"review of {ticker} decides again.")
            except Exception:
                pass
            return
        pos = positions.get(ticker) or {}
        owned = abs(float(pos.get("qty") or 0))
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

        price = _get_price(ticker, trading_client) or float(pos.get("current_price") or 0)
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
            # Still fails open (making it fail closed is a gate change, Luke's
            # call), but leaves a countable record (audit 16:53, A1653-1).
            log.error(f"[{ticker}] Catalyst gate skipped after an error: {e}")
            try:
                db.log_signal(ticker, "CATALYST_GATE_ERROR", 0.0, str(e)[:500])
            except Exception:
                pass

        # Earnings blackout (2026-10-27, limitations audit). With 1-share buys
        # live, a stop 1.75x ATR away is no protection against an earnings
        # gap, and the default path had no earnings rule at all. Decided by
        # the lead session under Luke's delegation (Jarvis, bus #36).
        from config import EARNINGS_ENTRY_BLACKOUT_DAYS
        try:
            near = (db.is_near_earnings(ticker, EARNINGS_ENTRY_BLACKOUT_DAYS, strict=True)
                    if EARNINGS_ENTRY_BLACKOUT_DAYS else False)
        except Exception:
            near = None  # can't check: fail closed
        if near is not False:
            if near is None:
                reason = f"earnings calendar unreadable — no new {action} until it can be checked"
            else:
                next_er = db.get_next_earnings(ticker)
                reason = (f"earnings {next_er} within {EARNINGS_ENTRY_BLACKOUT_DAYS} days — "
                          f"no new {action} until after the report")
            log.info(f"[{ticker}] Entry blocked: {reason}")
            if _feed_emit:
                _feed_emit("risk", ticker, f"Blocked — {reason}", "info")
            try:
                db.log_signal(ticker, "EARNINGS_BLOCKED", conviction / 10.0, reason,
                              **_route_row_fields(result, "EARNINGS_BLOCKED"))
            except Exception:
                pass
            return

        try:
            rm = RiskManager(trading_client)
        except Exception as e:
            log.error(f"[{ticker}] RiskManager init failed: {e}")
            return

        # Get current price
        price = _get_price(ticker, trading_client)
        if not price:
            log.warning(f"[{ticker}] Could not determine price — skipping approval queue")
            _log_route_exit(ticker, "ROUTE_NO_PRICE", result, "could not determine price")
            return

        # Size the position — stamp the price used so execution can validate freshness
        sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
        sizing["price_at_sizing"] = price
        _attach_shadows(sizing, ticker, result, price)
        if sizing.get("shares_initial", 0) <= 0:
            # 2026-08-13: if this is specifically a cash shortage (not a
            # halt, not "already at target," not conviction too low for the
            # bucket), check whether an existing Accelerator/Moonshot
            # holding has independently fallen weak enough to be worth
            # exiting anyway before giving up on this idea — see
            # RiskManager.find_swap_candidate's docstring for the exact,
            # deliberately conservative eligibility rule (never forced;
            # returns None and changes nothing if no position qualifies).
            #
            # Swing-trading cutover, Phase 4/5 (2026-10-11): also fires on
            # heat_blocked (SETUP_GATES_ENABLED's portfolio-heat cap, see
            # risk_manager._size_position_setup_gates) -- same "out of room,
            # worth checking for a rotation candidate" situation cash_blocked
            # already covers, just a different reason sizing came back at 0
            # shares. heat_blocked is never True unless SETUP_GATES_ENABLED,
            # so this is a no-op when the flag is off.
            if sizing.get("cash_blocked") or sizing.get("heat_blocked"):
                try:
                    swap = rm.find_swap_candidate(ticker, conviction)
                except Exception as e:
                    log.warning(f"[{ticker}] Swap candidate lookup failed: {e}")
                    swap = None
                if swap and rm.would_size_after_swap(ticker, conviction, bucket, regime_str,
                                                     price, swap) <= 0:
                    log.info(f"[{ticker}] Swap out of {swap['ticker']} skipped: the buy would "
                             f"still size to 0 shares after the sale")
                    swap = None
                if swap:
                    # Regular hours only (limitations audit 2026-10-28, finding
                    # 9): analyses run from 05:00, and a pre-market swap sell is
                    # a DAY market order that won't fill, after its stops have
                    # already been cancelled.
                    try:
                        import market_data
                        if not market_data.is_market_hours():
                            log.info(f"[{ticker}] Swap out of {swap['ticker']} skipped: market closed")
                            swap = None
                    except Exception as e:
                        log.warning(f"[{ticker}] Market-hours check failed, skipping swap: {e}")
                        swap = None
                if swap:
                    log.info(
                        f"[{ticker}] Cash-blocked at conviction {conviction}/10 — swapping "
                        f"out of {swap['ticker']} (conviction {swap['conviction']}/10, "
                        f"unrealized P&L ${swap['unrealized_pl']:,.2f}) to fund it"
                    )
                    try:
                        from execution import execute_manual
                        sell_result = execute_manual(
                            swap["ticker"], "SELL", swap["qty"],
                            reason=(f"Swap — freeing capital for {ticker} (conviction "
                                    f"{conviction}/10); {swap['ticker']}'s own conviction had "
                                    f"already fallen to {swap['conviction']}/10, not a new "
                                    f"concern introduced by this trade"),
                            bucket=bucket, conviction=swap["conviction"],
                        )
                    except Exception as e:
                        log.error(f"[{swap['ticker']}] Swap sell failed: {e}")
                        sell_result = {"success": False, "message": str(e)}
                    if sell_result.get("success"):
                        try:
                            import alerts as alert_module
                            pl = swap["unrealized_pl"]
                            pl_word = "gain" if pl >= 0 else "loss"
                            alert_module._send(
                                f"🔄 <b>SWAP</b> — sold {swap['qty']:g} {swap['ticker']} "
                                f"(conviction {swap['conviction']}/10, realized "
                                f"${abs(pl):,.2f} {pl_word}) to fund <b>{ticker}</b> "
                                f"(conviction {conviction}/10). Not a thesis change on "
                                f"{swap['ticker']} — its own conviction had already fallen; "
                                f"this just acted on it instead of leaving it sitting."
                            )
                        except Exception:
                            pass
                        # Re-size now that the sale has freed up cash.
                        sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
                        sizing["price_at_sizing"] = price
                        _attach_shadows(sizing, ticker, result, price)
                    else:
                        log.warning(f"[{swap['ticker']}] Swap sell did not complete: "
                                    f"{sell_result.get('message')}")

            if sizing.get("shares_initial", 0) <= 0:
                flags = sizing.get("flags", [])
                log.info(f"[{ticker}] Risk sizing returned 0 shares: {flags}")
                _log_route_exit(ticker, "ROUTE_ZERO_SHARES", result, "; ".join(flags))
                _shadow_swap(rm, ticker, conviction, sizing, result)
                return

        # Concentration check. position_size_pct is the remaining target; the
        # order itself can be bigger (one whole share above target, e.g. on an
        # ADD), so check whichever is larger (audit 10-28 #5).
        proposed_pct = max(sizing["position_size_pct"], sizing.get("order_pct") or 0)
        conc = rm.check_concentration(ticker, proposed_pct, bucket=bucket)
        if not conc["passed"]:
            log.info(f"[{ticker}] Concentration check failed: {conc['flags']}")
            sizing["flags"].extend(conc["flags"])
            # 2026-08-11: this used to only flag AI-infra breaches in text
            # nobody reviews (auto-execution has no human-in-the-loop tap) —
            # the portfolio sat over its own 60% limit for a week-plus while
            # every new ADD kept sailing through anyway. Generalized
            # 2026-08-18 after the identical gap recurred for RGTI's
            # moonshot-total breach (flagged "13.7% exceeds 10% limit",
            # executed at full size anyway) -- theme_size_cap_pct and
            # moonshot_size_cap_pct now exist alongside ai_infra_size_cap_pct
            # for exactly this reason. Each is None for a ticker whose own
            # size doesn't drive that particular breach (capping it wouldn't
            # address a breach caused by other positions) -- take the
            # tightest of whichever apply, since more than one can breach
            # at once (e.g. a moonshot name in an over-concentrated theme).
            caps = [c for c in (
                conc.get("ai_infra_size_cap_pct"),
                conc.get("theme_size_cap_pct"),
                conc.get("moonshot_size_cap_pct"),
            ) if c is not None]
            cap = min(caps) if caps else None
            if cap is not None and cap < proposed_pct:
                order_pct = sizing.get("order_pct") or 0
                if order_pct > 0:
                    # Cap what the order actually buys, not the target.
                    shares = min(sizing["shares_initial"],
                                 int(sizing["shares_initial"] * cap / order_pct + 1e-9))
                else:
                    shares = int(sizing["shares_initial"] * cap / proposed_pct) if proposed_pct > 0 else 0
                old_shares = sizing["shares_initial"]
                sizing["position_size_pct"] = round(min(sizing["position_size_pct"], cap), 4)
                sizing["shares_initial"] = shares
                sizing["initial_dollar"] = (round(sizing["initial_dollar"] * shares / old_shares, 2)
                                            if old_shares else 0.0)
                sizing["flags"].append(
                    f"Size capped to {cap:.1%} target ({sizing['shares_initial']} shares) "
                    f"to stay within concentration limits"
                )
                if sizing["shares_initial"] <= 0:
                    log.info(f"[{ticker}] Concentration limits leave 0 shares of room — "
                             f"skipping entry (already at/over the limit)")
                    _log_route_exit(ticker, "ROUTE_CONCENTRATION_ZERO", result,
                                    "; ".join(sizing.get("flags", [])))
                    return

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

    from config import AUTO_EXECUTE

    if not AUTO_EXECUTE:
        # Live trading with no explicit LIVE_TRADING_ENABLED opt-in (see
        # config.py) — added 2026-08-13 ahead of the live rollout. Route
        # through the same Telegram Approve/Reject flow paper trading used
        # before 2026-07-14 (alerts.send_approval_request /
        # telegram_bot._handle_callback) instead of firing unattended.
        # Leaves the approval PENDING; a human tap executes it from there.
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
        if msg_id is not None:
            db.set_approval_telegram_id(approval_id, msg_id)
        else:
            log.error(f"[{ticker}] Failed to send approval request for #{approval_id}")
        if _feed_emit:
            _feed_emit("execution", ticker,
                       f"Queued for approval: {result['action']} {sizing['shares_initial']} shares "
                       f"({sizing['position_size_pct']:.1%} portfolio) — awaiting Telegram tap",
                       "neutral")
        return

    # Paper trading, or live with LIVE_TRADING_ENABLED explicitly set — auto-
    # execute instead of waiting on a Telegram approval tap. Per explicit
    # user request 2026-07-14 for paper (see CLAUDE.md); every other risk
    # control (drawdown halt, concentration caps, price-drift abort, ATR
    # stop placement) still applies unchanged, only the human-in-the-loop
    # tap is removed.
    from execution import execute_approved, last_outcome_reason

    if not db.claim_approval_for_execution(approval_id):
        log.error(f"[{ticker}] Could not claim approval #{approval_id} for auto-execution")
        return

    approval = db.get_approval_by_id(approval_id)
    success = execute_approved(approval, trigger="auto")
    db.update_approval_status(approval_id, "APPROVED" if success else "FAILED",
                              reason=None if success else last_outcome_reason())
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

def _get_positions(client, strict: bool = False) -> dict | None:
    """Held positions by symbol. On a broker error returns {} or, with
    strict=True, None, so callers that must not reason against a falsely
    empty book (deep_analyze, the position review) can tell the difference.
    On 10-19 the {} fallback had NVDA analysed as an OPEN while held and the
    review logging "0 of 0 held tickers"."""
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
    except Exception as e:
        log.warning(f"Could not read broker positions: {e}")
        return None if strict else {}


def _get_price(ticker: str, client) -> float | None:
    # Outside regular hours, the last extended-hours trade (Luke D6, 10-29);
    # the regular-hours quote below is the prior close all night.
    try:
        from market_data import get_session_quote
        sq = get_session_quote(ticker)
        if sq and sq.get("price"):
            return float(sq["price"])
    except Exception:
        pass
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
