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
import re
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
            # Record only: a reject of the ticker's own new fact whose reason
            # cites other or watchlist stocks (agents/own_ticker.py). Read by
            # nothing; the screen above is unchanged.
            try:
                from agents.own_ticker import own_ticker_record
                if prov:
                    prov = {**prov, **own_ticker_record(prov.get("title_names_ticker"),
                                                        screening.get("new_fact"),
                                                        screening.get("reason"))}
            except Exception as e:
                log.debug(f"[{ticker}] own_ticker record failed: {e}")
            # The item's identifiers, as on SCREENED_IN, so a rejected item
            # can be found again (audit 2026-11-07 15:07 #1: only a 70-char
            # feed title survived). Never its text.
            db.log_signal(ticker, "SCREENED_OUT", 0.0, reason,
                          model=prov.get("model"), provenance=prov or None,
                          decision={**event_item_fields(event),
                                    "urgency": screening.get("urgency"),
                                    "new_fact": screening.get("new_fact")})
        except Exception as e:
            log.warning(f"[{ticker}] SCREENED_OUT signal log failed: {e}")
    else:
        log.info(f"[{ticker}] Relevant (urgency={screening.get('urgency')}) — queued for deep analysis")
        # Log-only: a pass whose analysis later expires or fails still has a
        # row (with EVENT_LOST from tasks.py).
        try:
            import db
            prov = screening.get("provenance") or {}
            db.log_signal(ticker, "SCREENED_IN", 0.0, str(screening.get("reason") or "")[:500],
                          model=prov.get("model"), provenance=prov or None,
                          decision={**event_item_fields(event),
                                    "urgency": screening.get("urgency"),
                                    "new_fact": screening.get("new_fact")})
        except Exception as e:
            log.warning(f"[{ticker}] SCREENED_IN signal log failed: {e}")

    return screening


def event_item_fields(event) -> dict:
    """The identifiers of an ingestion event for a log row: never its text."""
    event = event if isinstance(event, dict) else {}
    item_id = event.get("id") or event.get("url") or event.get("title")
    return {"source": event.get("source"),
            "title": str(event.get("title") or "")[:200] or None,
            "url": str(event.get("url") or "")[:300] or None,
            "item_id": str(item_id)[:300] if item_id else None,
            "form": event.get("form") or event.get("type")}


# The zone a source's naive time is in, where that source's own code fixes
# it: market_data._session_quote_uncached reads a naive bar label as ET (its
# bars are aware in practice). latest_quote's naive label is TWS's wall
# clock, whose zone the code does not know, so it is not listed: only its
# as_of_utc (from a bar that carried its offset) is used.
_NAIVE_AS_OF_ZONE = {"session_quote": "America/New_York"}


def _as_of_utc(info: dict | None) -> datetime | None:
    """The source time in _get_price's info as an aware UTC datetime:
    info["as_of_utc"] when present, else info["as_of"] when it is aware (an
    ISO string with a zone included), else a naive as_of in its source's
    known zone (_NAIVE_AS_OF_ZONE). None when the zone is unknown: never
    guessed. Raises nothing."""
    try:
        info = info or {}
        for key in ("as_of_utc", "as_of"):
            t = info.get(key)
            if isinstance(t, str):
                try:
                    t = datetime.fromisoformat(t.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if not isinstance(t, datetime):
                continue
            if t.tzinfo is None:
                zone = _NAIVE_AS_OF_ZONE.get(info.get("source")) if key == "as_of" else None
                if zone is None:
                    continue
                from zoneinfo import ZoneInfo
                t = t.replace(tzinfo=ZoneInfo(zone))
            return t.astimezone(timezone.utc)
    except Exception:
        pass
    return None


def _brief_price_record(info: dict | None, built_at: str | None) -> dict:
    """trigger keys for the brief's price (record only): brief_price_source,
    brief_price_as_of (UTC ISO), brief_price_age_s (built_at minus as_of, in
    seconds, 1 dp) and brief_price_is_stale, from _get_price's info. The
    time is _as_of_utc's: None (with the age) when the source's zone is not
    known. None where unknown. Never raises."""
    out = {"brief_price_source": None, "brief_price_as_of": None, "brief_price_age_s": None,
           "brief_price_is_stale": None}
    try:
        info = info or {}
        out["brief_price_source"] = info.get("source")
        st = info.get("is_stale")
        out["brief_price_is_stale"] = bool(st) if st is not None else None
        as_of = _as_of_utc(info)
        if as_of is not None:
            out["brief_price_as_of"] = as_of.isoformat()
            if built_at:
                built = datetime.fromisoformat(str(built_at).replace("Z", "+00:00"))
                out["brief_price_age_s"] = round((built - as_of).total_seconds(), 1)
    except Exception as e:
        log.debug(f"brief price record unavailable: {e}")
    return out


# Read-through record (record only; config.READ_THROUGH_RECORD_ENABLED): which
# held names an analysed event bears on, from the related-symbol mapping the
# brief itself uses. The marker is per (source item, held ticker).
_READ_THROUGH_STAGE = "read_through"
_READ_THROUGH_MARKER = "helios:read_through:{env}:{item}:{held}"
_READ_THROUGH_MARKER_TTL_S = 8 * 86400
_RELATED_SECTION_HEADING = "=== RELATED SYMBOLS IN WATCHLIST ==="


def _read_through_record(event: dict, ticker: str, positions: dict | None, held_as_of,
                         brief: str | None, brief_error, result: dict) -> dict | None:
    """One queued activity row (stage read_through) when the event's ticker
    has related symbols (context_engine._find_related_symbols over
    config.SYMBOLS, called as the brief calls it) that the book holds, per
    the positions snapshot deep_analyze already read; no broker call. Held
    names already recorded for this source item (a Redis NX marker per
    item and held ticker) are listed as deduped and get no new row; with
    Redis unreadable the row is written anyway. No row when the brief failed
    (the book was not read). Returns the row's extra, or None. Never raises
    and never blocks the analysis: the row goes through the activity
    writer's queue."""
    try:
        from celery.exceptions import SoftTimeLimitExceeded
    except Exception:          # celery absent: nothing to re-raise
        SoftTimeLimitExceeded = ()
    try:
        import config
        if not getattr(config, "READ_THROUGH_RECORD_ENABLED", False):
            return None
        if brief_error is not None or positions is None:
            return None
        import context_engine
        related = context_engine._find_related_symbols(ticker, config.SYMBOLS)
        if not related:
            return None

        def held(sym):
            pos = positions.get(sym)
            if not isinstance(pos, dict):
                return False
            try:
                return float(pos.get("qty") or 0) != 0
            except (TypeError, ValueError):
                return True
        rows = [{"ticker": s, "held": held(s)} for s in related]
        held_related = [r["ticker"] for r in rows if r["held"]]
        if not held_related:
            return None
        import db
        env = db.TRADING_ENV
        item = event_item_fields(event)
        fresh, deduped, dedupe_read = list(held_related), [], "ok"
        if item.get("item_id"):
            import hashlib
            key_item = hashlib.sha256(item["item_id"].encode()).hexdigest()[:16]
            try:
                import stop_lock
                r = stop_lock._redis()
                fresh = []
                for h in held_related:
                    if r.set(_READ_THROUGH_MARKER.format(env=env, item=key_item, held=h), "1",
                             nx=True, ex=_READ_THROUGH_MARKER_TTL_S):
                        fresh.append(h)
                    else:
                        deduped.append(h)
            except SoftTimeLimitExceeded:
                raise
            except Exception as e:
                fresh, deduped = list(held_related), []
                dedupe_read = f"error: {type(e).__name__}"
        else:
            dedupe_read = "no_item_id"
        if not fresh:
            return None
        prov = (result or {}).get("provenance") or {}
        try:
            from agents import analyst
            cap = int(getattr(analyst, "_BRIEF_MAX_CHARS"))
        except Exception:
            cap = None
        text = brief or ""
        extra = {
            "env": env, "path": "event", "event_ticker": ticker,
            "signal_id": prov.get("signal_id"),
            "action": (result or {}).get("action"),
            "conviction": (result or {}).get("conviction"),
            "item": item,
            "related": rows,
            "held_related": fresh,
            "held_related_deduped": deduped,
            "mapping": "context_engine._find_related_symbols(ticker, config.SYMBOLS)",
            "held_source": "deep_analyze positions snapshot",
            "held_as_of": held_as_of,
            # The section the brief appends, and whether it is inside the
            # analyst's character cut (the token-budget guard is not judged).
            "section_in_brief": _RELATED_SECTION_HEADING in text,
            "section_within_char_cap": ((_RELATED_SECTION_HEADING in text[:cap])
                                        if cap is not None else None),
            "dedupe_read": dedupe_read,
        }
        db.log_activity_queued(_READ_THROUGH_STAGE, ticker,
                               f"read-through: {ticker} event, held related "
                               f"{', '.join(fresh)}", "neutral", extra)
        return extra
    except SoftTimeLimitExceeded:
        raise
    except Exception as e:
        log.debug(f"[{ticker}] read-through record not written: {e}")
        return None


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
    brief_error = None
    brief_price = None     # the price the brief showed (audit 2026-11-07 17:17 #1)
    price_info: dict = {}  # where it came from and how old (record only)
    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client, strict=True)
        held_as_of = datetime.now(timezone.utc).isoformat()
        if positions is None:
            log.warning(f"[{ticker}] Broker positions unavailable — skipping analysis "
                        f"rather than analysing against an empty book")
            # The SCREENED_IN row this event got would otherwise have no outcome.
            try:
                from tasks import _log_event_lost
                _log_event_lost("tasks.deep_analyze_event", "positions_unavailable", [event])
            except Exception as e:
                log.warning(f"[{ticker}] EVENT_LOST log failed: {e}")
            return {"action": "SKIP", "ticker": ticker, "conviction": 0, "pass_to_risk": False,
                    "reasoning": "Skipped: broker positions unavailable"}
        # Held tickers get a price via the CURRENT PORTFOLIO section; anything
        # not held previously had NO price anywhere in the brief, so the model
        # would invent one for any dollar-denominated valuation output (found
        # 2026-07-17: a SNDK target of "$300" while it traded near $1,350).
        price = _get_price(ticker, client, info=price_info)
        brief_price = price
        brief = assemble_brief(event, positions, client, current_price=price)
    except Exception as e:
        log.warning(f"[{ticker}] Brief assembly failed: {e}")
        brief = f"Event: {event.get('title', '')}\n{event.get('text', '')[:1000]}"
        positions = {}
        client = None
        brief_error = type(e).__name__
    brief_built_at = datetime.now(timezone.utc).isoformat()

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
        # The screener's read of the triggering item (None: not judged).
        "new_fact":        (screening or {}).get("new_fact"),
    }

    # ── Analyze ────────────────────────────────────────────────────────────────
    regime = get_cached_regime()
    try:
        from agents.analyst import trigger_record
        trigger = trigger_record("event", event)
        if screening and screening.get("override"):
            trigger["screen_override"] = screening["override"]   # review 4 #4
            # Whether the title names this ticker (audit 11-06 09:31 #3).
            trigger["title_names_ticker"] = ((screening.get("provenance") or {})
                                             .get("title_names_ticker"))
        trigger["new_fact"] = candidate["new_fact"]
        sprov = (screening or {}).get("provenance") or {}
        trigger["screener_model_digest"] = sprov.get("model_digest")
        trigger["screener_prompt_template_sha"] = sprov.get("prompt_template_sha")
        # Why the keyword fallback judged it (None: the 7B did); M275 alert.
        trigger["screener_fallback_reason"] = sprov.get("fallback_reason")
        # Log-only: the fallback brief (event text only, no book or price).
        trigger["brief_degraded"] = brief_error is not None
        trigger["brief_built_at"] = brief_built_at
        trigger["brief_price"] = brief_price
        # Record only (gap hunt 2026-11-21 item 3): which source gave the
        # brief's price, that source's own time, its age when the brief was
        # built, and the latest quote's stale flag. None where unknown.
        trigger.update(_brief_price_record(price_info, brief_built_at))
        if brief_error:
            trigger["brief_error"] = brief_error
        # Record only: would this story be skipped as a repeat or as stale?
        # Read by nothing in the prompt, brief or routing; never raises.
        try:
            from agents.story_dedup import story_dedup_record
            story_dedup = story_dedup_record(ticker, trigger)
            if story_dedup is not None:
                trigger["story_dedup"] = story_dedup
        except Exception as e:
            log.debug(f"[{ticker}] story dedup shadow not recorded: {e}")
        result = analyze(candidate, situation_brief=brief, regime_packet=regime,
                         trigger=trigger)
    except Exception as exc:
        log.error(f"[{ticker}] analyze() raised an unhandled exception: {exc}", exc_info=True)
        result = {
            "action": "SKIP", "ticker": ticker, "conviction": 0,
            "pass_to_risk": False,
            "reasoning": f"Analysis failed with exception: {exc}",
        }

    # Trim/exit rule, record only here: a row for a TRIM/EXIT verdict, never
    # a routing change. Never raises.
    from agents import trim_exit
    trim_exit.apply(result, path="event", event=event, positions=positions, observe=True)

    # Read-through record (record only): held names this event bears on.
    _read_through_record(event, ticker, positions, held_as_of, brief, brief_error, result)

    # ── Route to approval queue if actionable ─────────────────────────────────
    if result.get("pass_to_risk"):
        _route_guarded(result, regime, client)
    else:
        _c6_bar_unrouted(result, regime, client)   # log-only

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

def _log_route_exit(ticker: str, kind: str, result: dict, reason: str,
                    decision_extra: dict | None = None) -> None:
    """A passed verdict that dies in routing gets a signals row, like
    EARNINGS_BLOCKED already did (professor M30, 2026-10-28): these exits were
    log.info only, so the final disposition of a pass wasn't observable.
    decision_extra adds kind-specific keys to the row's decision."""
    try:
        import db
        fields = _route_row_fields(result, kind)
        if decision_extra:
            fields["decision"].update({k: v for k, v in decision_extra.items()
                                       if k not in fields["decision"]})
        db.log_signal(ticker, kind, (result.get("conviction") or 0) / 10.0, reason[:500],
                      **fields)
    except Exception as e:
        log.warning(f"[{ticker}] route-exit signal log failed: {e}")


_VERDICT_ONLY_PROVENANCE = ("plan", "brief_exposure", "gate_flags", "tighten_sizing_shadow",
                            "sell_confirm", "buy_confirm", "scan_reasons", "regime_entry")


def _route_row_fields(result: dict, kind: str, decision_extra: dict | None = None) -> dict:
    """Model and provenance of the verdict a routing row belongs to, so the
    row stands on its own (SEF run 13, 2026-10-28: ROUTE_* rows had NULL model
    and provenance and had to be joined to the preceding verdict). The
    prompt_sha in provenance identifies that verdict's signals row."""
    prov = result.get("provenance") or None
    if prov and any(k in prov for k in _VERDICT_ONLY_PROVENANCE):
        # The plan observation belongs to the verdict row only; copied onto
        # routing rows it was counted 2-3 times (pre-pull review 10-29).
        # brief_exposure, gate_flags, tighten_sizing_shadow, sell_confirm and
        # buy_confirm likewise.
        prov = {k: v for k, v in prov.items() if k not in _VERDICT_ONLY_PROVENANCE}
    if prov and isinstance(prov.get("track_record"), dict) and "by_position" in prov["track_record"]:
        # The per-position track record is the verdict's alone (m607); the
        # rest of track_record stays on routing rows as before.
        prov = {**prov, "track_record": {k: v for k, v in prov["track_record"].items()
                                         if k != "by_position"}}
    decision = {"route_exit": kind, "verdict_action": result.get("action"),
                "verdict_conviction": result.get("conviction")}
    rule = result.get("trim_exit")
    if isinstance(rule, dict):
        # A copy the trim/exit rule routed (agents/trim_exit.py): the model's
        # action stays verdict_action, the rule's goes beside it with the tag.
        decision.update(verdict_action=rule.get("verdict_action"),
                        rule_action=result.get("action"), trim_exit=rule)
    if decision_extra:
        decision.update({k: v for k, v in decision_extra.items() if k not in decision})
    return {
        "model": (prov or {}).get("model"),
        "provenance": prov,
        "decision": decision,
    }


_ROUTE_ERROR_ALERT_WINDOW_S = 3600


def _alert_route_error(ticker: str, error_class: str, now: datetime | None = None) -> bool:
    """Telegram for a ROUTE_ERROR, at most one per ticker per hour (system_state
    key written after a successful send). True when sent. Never raises."""
    try:
        import alerts
        import db
        from config import TRADING_ENV
        now = now or datetime.now(timezone.utc)
        key = f"route_error_alert:{TRADING_ENV}:{ticker}"
        last = db.get_state(key)
        if last:
            try:
                if (now - datetime.fromisoformat(last)).total_seconds() < _ROUTE_ERROR_ALERT_WINDOW_S:
                    return False
            except (TypeError, ValueError):
                pass   # unreadable stamp: send
        if alerts._send(f"{ticker}: routing raised {error_class} after the verdict; "
                        f"check for a partial swap or order") is None:
            return False
        db.set_state(key, now.isoformat())
        return True
    except Exception as e:
        log.warning(f"[{ticker}] ROUTE_ERROR alert failed: {e}")
        return False


def _route_guarded(result: dict, regime: dict, client) -> None:
    """_route_to_approval for a verdict already written. A raise here used to
    reach the Celery task, whose retry re-ran the analysis: a second verdict
    and a second route. Now ERROR plus a ROUTE_ERROR row, and the caller
    goes on."""
    try:
        _route_to_approval(result, regime, client)
    except Exception as e:
        ticker = result.get("ticker") or "UNKNOWN"
        log.error(f"[{ticker}] Routing raised after the verdict was written; not retried: "
                  f"{type(e).__name__}: {e}", exc_info=True)
        _log_route_exit(ticker, "ROUTE_ERROR", result, f"{type(e).__name__}: {e}",
                        {"error_class": type(e).__name__})
        _alert_route_error(ticker, type(e).__name__)
    _verdict_vote_shadow(result)    # record only, after routing


def _verdict_vote_shadow(result: dict) -> None:
    """Enqueues the shadow re-draws for a routed OPEN/ADD (agents/
    verdict_vote.py, config.VERDICT_VOTE_SHADOW_ENABLED). Never raises."""
    try:
        from agents.verdict_vote import observe_routed
        observe_routed(result)
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] verdict vote shadow not enqueued: {e}")


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


def _fetch_earnings_dates(ticker: str):
    """yfinance's dated earnings table (index: event time, tz-aware). Its own
    function so tests stub the network call."""
    import yfinance as yf
    return yf.Ticker(ticker).get_earnings_dates(limit=8)


def _earnings_timing(ticker: str, earnings_date) -> dict:
    """Log-only (limitations audits 2026-11-08 13:18 #3/#6): when in the day
    the print falls, from yfinance's earnings_dates row on the calendar date:
    AMC at or after 16:00 ET, BMO before 09:30, "during" between, "unknown"
    at midnight (no time given). The calendar marks every date confirmed;
    this source gives no estimate flag either, so is_estimate stays None.
    Read on the calling thread (a daemon-thread version segfaulted in
    pyarrow, roadmap 04d); yfinance bounds the request itself. Held
    earnings reviews are a few a day. Never raises; nothing reads it to
    decide."""
    from zoneinfo import ZoneInfo
    out = {"source": "yfinance.earnings_dates", "calendar_date": str(earnings_date),
           "event_at": None, "time_of_day": None, "is_estimate": None,
           "reported": None, "status": None}
    try:
        df = _fetch_earnings_dates(ticker)
    except Exception as e:
        out["status"] = type(e).__name__
        return out
    try:
        import math
        et = ZoneInfo("America/New_York")
        for ts, row in (df.iterrows() if df is not None else []):
            at = ts.to_pydatetime().astimezone(et)
            if str(at.date()) != str(earnings_date):
                continue
            mins = at.hour * 60 + at.minute
            out["event_at"] = at.isoformat()
            out["time_of_day"] = ("unknown" if mins == 0 else "BMO" if mins < 570
                                  else "AMC" if mins >= 960 else "during")
            rep = row.get("Reported EPS")
            out["reported"] = rep is not None and not (isinstance(rep, float) and math.isnan(rep))
            out["status"] = "ok"
            return out
        out["status"] = "no_match"
    except Exception as e:
        out["status"] = type(e).__name__
    return out


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
        # Not shown in the brief (it reads source/type/title/text/url); lets
        # the analyst's event_plan add-on know the print is still ahead.
        "earnings_phase": "pre" if days > 0 else "day" if days == 0 else "post",
    }


def _analyze_held_position(ticker: str, client, positions: dict, regime: dict,
                           event: dict | None = None, path: str | None = None,
                           held_review: bool = False) -> dict:
    """The brief-assembly + analyze() core shared by review_held_position
    and backfill_catalyst_tracking (2026-08-30) -- extracted so the
    backfill can get the same real catalyst/expected-return read every
    scheduled review already produces, without ALSO triggering
    review_held_position's approval-routing side effect (a backfill pass
    exists to capture a catalyst snapshot, not to fire a new TRIM/EXIT).
    held_review: a routed review of a held position; the analyst alerts
    when such a TRIM/EXIT is lowered to WATCH by a malformed catalyst."""
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
    if "role" not in wl_entry:
        # Log-only: the review still runs with the Accelerator default below.
        log.warning(f"[{ticker}] held review role defaulted to Accelerator "
                    f"(watchlist entry {'has no role' if wl_entry else 'missing'})")
    elif not wl_entry["role"]:
        # .get() returns the empty value as is, so no default applies.
        log.warning(f"[{ticker}] held review role is empty ({wl_entry['role']!r}), passed as is")
    candidate = {
        "ticker":          ticker,
        "role":            wl_entry.get("role", "Accelerator"),
        "tier":            wl_entry.get("tier", "Watch"),
        "sector":          wl_entry.get("sector", ""),
        "composite_score": 60,
    }
    if held_review:
        candidate["held_review"] = True

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
            # Read by review_all_held_positions to record the review as
            # skipped (SEF 148, e6d099f). Not routed.
            "review_error": type(exc).__name__,
        }


def _catalyst_precision(catalyst) -> str | None:
    """How precise a thesis's catalyst date is: day (YYYY-MM-DD), month
    (YYYY-MM), ongoing, other, or None with no date."""
    d = str((catalyst or {}).get("date") or "").strip()
    if not d:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return "day"
    if re.fullmatch(r"\d{4}-\d{2}", d):
        return "month"
    return "ongoing" if d.lower() == "ongoing" else "other"


def _record_thesis_drift(ticker: str, latest: dict, previous: dict | None = None,
                         previous_read: str | None = None) -> None:
    """Log-only: an activity row (stage thesis_drift) diffing this review's
    thesis against the entry thesis with db.compute_thesis_drift, which had
    no caller (audit 11-06 01:32 #2), plus catalyst_date_passed, whether the
    entry thesis's catalyst date is before today (ET). No alert, no routing,
    nothing in the brief. catalyst_status is still always "pending", so
    material_change never comes from it. Calibration fields, so a threshold
    can be set from data before anything triggers on the row (audit 11-06
    03:34 #b): entry and latest conviction, the change since the previous
    review (`previous`, the latest thesis this review replaced), risk counts
    beside the new-risk list, and each thesis's catalyst precision.
    previous_read says whether `previous` was read (ok), absent (none) or
    unreadable (error), and previous_predates_position flags a previous
    thesis from an earlier position, as for the original (professor 62r).
    Never raises."""
    try:
        import db
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from agents.plans import _catalyst_date
        original = (db.get_thesis(ticker) or {}).get("original")
        if not original:
            return              # no entry thesis to diff against
        drift = db.compute_thesis_drift(original, latest)
        cdate = _catalyst_date((original.get("catalyst") or {}).get("date"))
        today = datetime.now(ZoneInfo("America/New_York")).date()
        drift["catalyst_date_passed"] = None if cdate is None else cdate < today
        prev_conv = (previous or {}).get("conviction")
        cur_conv = latest.get("conviction")
        drift.update(
            entry_conviction=original.get("conviction"), latest_conviction=cur_conv,
            previous_conviction=prev_conv, previous_as_of=(previous or {}).get("as_of"),
            previous_read=previous_read, previous_predates_position=None,
            # The fill writes the entry thesis to latest too, so a first
            # review's previous is the entry snapshot (SEF 146d).
            previous_is_entry=(None if not previous else
                               previous.get("as_of") is not None
                               and previous.get("as_of") == original.get("as_of")),
            conviction_delta_vs_previous=(cur_conv - prev_conv
                                          if isinstance(cur_conv, (int, float))
                                          and isinstance(prev_conv, (int, float)) else None),
            original_risk_count=len(original.get("key_risks") or []),
            latest_risk_count=len(latest.get("key_risks") or []),
            new_risk_count=len(drift.get("new_risks") or []),
            original_catalyst_precision=_catalyst_precision(original.get("catalyst")),
            latest_catalyst_precision=_catalyst_precision(latest.get("catalyst")))
        drift.update(original_as_of=original.get("as_of"),
                     original_action=original.get("action"), latest_action=latest.get("action"),
                     env=db.TRADING_ENV)     # activity_feed has no env column
        # Whether the entry thesis belongs to an earlier position: a re-entry
        # over a leftover stop record keeps that record's original_thesis
        # and opened_at, so this diff can be against the old position
        # (professor m587). Reference: the newest open trade_outcomes entry.
        # Record only; None when unknown.
        drift.update(position_entry_at=None, position_entry_basis=None,
                     original_predates_position=None, position_opened_at=None, review_path=None)
        try:
            drift["review_path"] = latest["provenance"]["trigger"]["path"]
        except Exception:
            pass
        try:
            from datetime import timedelta
            legs = [o for o in db.get_open_outcomes()
                    if o.get("ticker") == ticker and o.get("entry_date")]
            # Every BUY approval, ADDs too, opens its own outcome row, so the
            # position's entry is its newest OPEN leg (a re-entry over a
            # leftover record is an OPEN; later ADDs don't move it), else
            # its oldest leg (professor 58h).
            opens = [o["entry_date"] for o in legs if o.get("action") == "OPEN"]
            if opens:
                entry, drift["position_entry_basis"] = max(opens), "open_leg"
            elif legs:
                entry, drift["position_entry_basis"] = min(o["entry_date"] for o in legs), "min_leg"
            else:
                entry = None
            rec = db.get_position_stop(ticker) or {}
            opened = rec.get("opened_at")
            drift["position_opened_at"] = opened.isoformat() if hasattr(opened, "isoformat") else opened
            if entry is not None:
                drift["position_entry_at"] = entry.isoformat()
                as_of = original.get("as_of")
                if as_of:
                    orig = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
                    if orig.tzinfo and entry.tzinfo:
                        drift["original_predates_position"] = orig < entry - timedelta(minutes=5)
                prev_as_of = (previous or {}).get("as_of")
                if prev_as_of:
                    prev = datetime.fromisoformat(str(prev_as_of).replace("Z", "+00:00"))
                    if prev.tzinfo and entry.tzinfo:
                        drift["previous_predates_position"] = prev < entry - timedelta(minutes=5)
        except Exception as e:
            log.debug(f"[{ticker}] thesis drift position check failed: {e}")
        try:
            from agents.analyst import shadow_versions
            drift.update(shadow_versions())     # code_sha, env_sha, lib_versions
        except Exception:
            drift.update(code_sha=None, env_sha=None, lib_versions=None)
        db.log_activity_direct(
            "thesis_drift", ticker,
            f"Thesis drift: conviction {drift['conviction_delta']:+}" if drift["conviction_delta"] is not None
            else "Thesis drift: conviction change unknown",
            "neutral", drift)
    except Exception as e:
        log.warning(f"[{ticker}] thesis drift record failed: {e}")


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
            **db.catalyst_horizon_any(result.get("catalyst")),
            "catalyst_stale": result.get("catalyst_stale", False),
            "valuation_tier": None,
            "key_risks": result.get("key_risks", []),
            "reasoning": result.get("reasoning"),
            "provenance": result.get("provenance"),
        }
        try:
            previous = (db.get_thesis(ticker, strict=True) or {}).get("latest")   # before it's replaced
            previous_read = "ok" if previous else "none"
        except Exception:
            previous, previous_read = None, "error"
        db.update_latest_thesis(ticker, thesis)
        _record_thesis_drift(ticker, thesis, previous, previous_read)

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
    if event is not None:
        # Provenance only (trigger_record); the brief reads title and text.
        event["earnings_timing"] = _earnings_timing(ticker, earnings_date)
        # A print after today's close is still ahead (audit review of the
        # add-ons: AMC-today reviews asked to "decide whether to hold through
        # it" but got no event_plan).
        if (event.get("earnings_phase") == "day"
                and (event["earnings_timing"] or {}).get("time_of_day") == "AMC"):
            event["earnings_phase"] = "pre"
    result = _analyze_held_position(ticker, client, positions, regime, event=event,
                                    held_review=True)
    _capture_latest_thesis(ticker, result)

    # Trim/exit rule (agents/trim_exit.py): off, it returns `result` itself;
    # otherwise it records the review and may route a copy. Never raises.
    from agents import trim_exit
    routed = trim_exit.apply(result, path="earnings_review" if event is not None else "scheduled_review",
                             event=event, earnings_date=earnings_date, positions=positions)
    if routed.get("pass_to_risk"):
        _route_guarded(routed, regime, client)
    else:
        _c6_bar_unrouted(routed, regime, client)   # log-only

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
                                    path="plan_level_review", held_review=True)
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
    # Trim/exit rule, as in review_held_position.
    from agents import trim_exit
    routed = trim_exit.apply(result, path="plan_level_review", event=event, positions=positions)
    if routed.get("pass_to_risk"):
        _route_guarded(routed, regime, client)
    else:
        _c6_bar_unrouted(routed, regime, client)   # log-only
    return result


def log_review_skipped(symbol: str, reasoning: str, decision: dict) -> None:
    """Log-only REVIEW_SKIPPED row, so a held-position review that did not
    happen is on the record rather than indistinguishable from no review.
    symbol is "PORTFOLIO" for a whole-run skip, else the ticker. Never raises."""
    try:
        import db
        from agents.analyst import shadow_versions
        db.log_signal(symbol, "REVIEW_SKIPPED", 0.0, reasoning,
                      decision=decision, provenance=shadow_versions())
    except Exception as e:
        log.warning(f"REVIEW_SKIPPED log failed: {e}")


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
        log_review_skipped("PORTFOLIO",
                           "held review skipped: broker positions unavailable",
                           {"task": "review_all_held_positions",
                            "reason": "positions_unavailable",
                            "priority_tickers": list(priority_tickers or [])})
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
            res = review_held_position(ticker, client, positions, regime,
                                       earnings_date=earnings.get(ticker))
            results.append(res)
            # A stub (no model reply) is not a review: record it like a skip
            # (audit 2026-11-07 05:07 #2). The cursor still advances as before.
            if (res or {}).get("model_used") == "stub":
                log_review_skipped(ticker, "held review skipped: stub verdict (no model reply)",
                                   {"task": "review_all_held_positions", "reason": "stub"})
            elif (res or {}).get("review_error"):
                log_review_skipped(ticker, f"held review skipped: analyze raised {res['review_error']}",
                                   {"task": "review_all_held_positions", "reason": "analyze_raised",
                                    "error_class": res["review_error"]})
        except Exception as e:
            log.error(f"[{ticker}] Position review failed: {e}")
            log_review_skipped(ticker, f"held review failed: {type(e).__name__}",
                               {"task": "review_all_held_positions",
                                "reason": "ticker_review_failed",
                                "error_class": type(e).__name__})
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
    # broker_error: the client or its position read failed, so the book the
    # analysis sees is falsely empty. The verdicts are still logged, but no
    # pass is routed: each gets a ROUTE_NO_BROKER row, plus one alert a day.
    broker_error = None
    try:
        from ibkr import IBKRClient
        client = IBKRClient()
        positions = _get_positions(client, strict=True)
        if positions is None:
            broker_error = "broker positions unreadable"
            positions = {}
    except Exception as e:
        log.warning(f"Could not init IBKR client for brief/routing: {e}")
        broker_error = f"{type(e).__name__}: {e}"
        client = None
        positions = {}

    # Lost analyses get a NO_VERDICT row each, so a dead scan is visible.
    from agents.scan_guard import log_lost_candidates
    try:
        results = analyze_batch(to_analyze, regime_packet=regime,
                                positions=positions, trading_client=client)
    except Exception as e:
        # A logging failure must not replace the batch's own exception.
        try:
            log_lost_candidates(to_analyze, None, batch_error=f"{type(e).__name__}: {e}")
        except Exception as log_error:
            log.warning(f"NO_VERDICT logging failed after analyze_batch raised: {log_error}")
        raise
    try:
        log_lost_candidates(to_analyze, results)
    except Exception as e:
        log.warning(f"NO_VERDICT logging failed: {e}")

    if broker_error is not None:
        from agents.scan_guard import alert_scan_without_broker, log_no_broker_passes
        log.warning(f"Scan ran without a broker ({broker_error}); verdicts logged, none routed")
        alert_scan_without_broker(broker_error, log_no_broker_passes(results, broker_error))
        # The c6 shadow only enqueues (no broker call): verdicts that were
        # not passes are sampled without a broker too.
        for result in results:
            if not result.get("pass_to_risk"):
                _c6_bar_unrouted(result, regime, client)   # log-only
        return results

    from agents import trim_exit
    for result in results:
        # Trim/exit rule, record only here (TRIM/EXIT verdicts). Never raises.
        trim_exit.apply(result, path="daily_scan", positions=positions, observe=True)
        if result.get("pass_to_risk") and client:
            _route_guarded(result, regime, client)
        else:
            _c6_bar_unrouted(result, regime, client)   # log-only

    return results


# ---------------------------------------------------------------------------
# Risk sizing + approval queue
# ---------------------------------------------------------------------------

def _c6_bar_unrouted(result: dict, regime: dict, client) -> None:
    """c6 bar shadow (agents/c6_shadow.py) for a verdict not routed: a
    fire-and-forget enqueue, no price or broker call here. Log-only, never
    raises."""
    try:
        from agents.c6_shadow import observe_unrouted
        observe_unrouted(result, regime)
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow failed: {e}")


def _c6_bar_tier_floor(result: dict, regime_str: str, price, sizing: dict) -> None:
    """c6 bar shadow for a buy the tier floor sized to 0: a fire-and-forget
    enqueue. Log-only, never raises."""
    try:
        from agents.c6_shadow import observe_tier_floor
        observe_tier_floor(result, regime_str, price, sizing)
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow failed: {e}")


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


def _partial_exec_shadow(sizing: dict, bucket: str | None = None) -> dict | None:
    """Which partial sells a position of the initial size could execute in
    whole shares (audit 2026-11-05-0325 #5): the profit-target partial is
    int(qty * pct), so 0 shares at small qty, and a TRIM is
    max(1, int(qty * TRIM_FRACTION)), a full exit at 1-2 shares. Uses the
    shares at sizing, before any later rescale. The profit fields are None
    for a bucket profit targets skip (Compounder). Log-only."""
    qty = int(sizing.get("shares_initial") or 0)
    if qty <= 0:
        return None
    from config import PARTIAL_SELL_PCT_BY_STYLE, PROFIT_TARGET_PCT_BY_STYLE, TRIM_FRACTION
    trim = min(qty, max(1, int(qty * TRIM_FRACTION)))
    applies = any(bucket in t for t in PROFIT_TARGET_PCT_BY_STYLE.values())
    profit = ({style: int(qty * pct) for style, pct in PARTIAL_SELL_PCT_BY_STYLE.items()}
              if applies else None)
    return {"qty": qty, "bucket": bucket, "profit_target_applies": applies,
            "profit_partial_shares": profit,
            "profit_partial_executable": all(n > 0 for n in profit.values()) if applies else None,
            "trim_shares": trim, "trim_is_full_exit": trim >= qty}


def _add_to_loser_shadow(ticker: str, result: dict, price: float, client=None) -> dict | None:
    """Shadow check of a written rule nothing enforces: "Never add to a
    losing position without a new fundamental catalyst" (risk_rules.md,
    every bucket; audit 2026-11-05-0525 #4). The new-fact gate covers news
    buys only by default, so an ADD from a review path is unchecked.

    "Losing" is judged against the broker's average cost when it can be
    read (basis "avg_cost"), else position_stops.entry_price, which every
    bracketed BUY overwrites with its fill, so it is the last buy price
    (basis "last_buy"; professor m538). Both are recorded. new_fact is the
    screener's flag on the triggering news item, None off the news path;
    what counts as a catalyst is Luke's (G58, m539). Log-only."""
    if result.get("action") != "ADD":
        return None
    import db
    last_buy = float((db.get_position_stop(ticker) or {}).get("entry_price") or 0) or None
    avg_cost, avg_read = None, "no_client"
    if client is not None:
        try:
            pos = next((p for p in client.get_all_positions() if p.symbol == ticker), None)
            avg_cost = float(pos.avg_entry_price) if pos and pos.avg_entry_price else None
            avg_read = "ok" if avg_cost else "not_held"
        except Exception as e:
            avg_read = f"error: {type(e).__name__}"
    basis, ref = ("avg_cost", avg_cost) if avg_cost else ("last_buy", last_buy)
    if not ref or not price:
        return {"entry_known": bool(ref), "price_known": bool(price), "below_entry": None,
                "avg_cost_read": avg_read}

    def pct(x):
        return round((price - x) / x * 100, 2) if x else None
    prov = result.get("provenance")
    trig = (prov.get("trigger") if isinstance(prov, dict) else None) or {}
    item = trig.get("new_fact") if trig.get("path") == "event" else None
    rec = {"entry_known": True, "basis": basis, "avg_cost_read": avg_read,
           "pct_vs_avg_cost": pct(avg_cost),
           "pct_vs_last_buy": pct(last_buy), "below_entry": price < ref,
           "path": trig.get("path"), "new_fact": item if isinstance(item, bool) else None}
    rec["breaks_rule"] = rec["below_entry"] and rec["new_fact"] is not True
    if rec["breaks_rule"]:
        log.warning(f"[{ticker}] rule shadow (never add to a loser without a new catalyst): "
                    f"ADD {pct(ref):+.1f}% vs {basis}, path={rec['path']}, "
                    f"new_fact={rec['new_fact']}; not enforced")
    return rec


def _attach_shadows(sizing: dict, ticker: str, result: dict, price: float, client=None,
                    price_info: dict | None = None) -> None:
    """Log-only fields on a sizing dict, each guaranteed JSON-safe, with
    shadow_status saying why one is missing: ok, n/a (doesn't apply),
    error (the helper raised; the dict holds the message) or unencodable.
    price_info: _get_price's info for the sizing price (arrival record)."""
    stop = sizing.get("stop_loss_price")
    values = {"r_shadow": _r_shadow(result, price, stop),
              "plan_compiled": _compiled_plan(ticker, result, price, stop),
              "arrival": _arrival_record(result, price, price_info)}
    try:
        values["partial_exec"] = _partial_exec_shadow(sizing, result.get("thesis_bucket"))
    except Exception as e:
        values["partial_exec"] = {"error": str(e)[:200]}
    try:
        values["add_to_loser"] = _add_to_loser_shadow(ticker, result, price, client)
    except Exception as e:
        values["add_to_loser"] = {"error": str(e)[:200]}
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


def _parse_when(value):
    """An ISO or RFC 2822 time string as an aware UTC datetime, else None."""
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime
    if not value:
        return None
    s = str(value).strip()
    for parse in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            ts = parse(s)
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _arrival_record(result: dict, price: float, price_info: dict | None = None) -> dict:
    """Log-only (audit 2026-11-07 17:17 #1): how far price moved and how much
    time passed between the brief the analyst read and the sizing quote the
    order is checked against, and how old the triggering item was. The drift
    guard and decision_price only see the sizing quote. Event path only has
    a brief price and an event time; elsewhere they are None.

    The sizing price's own provenance, from _get_price's info (price_info):
    sizing_price_source (which source answered), sizing_bar_time (that
    source's bar time in UTC, ISO, by _as_of_utc; None when the source gives
    none or its zone is unknown), sizing_price_age_s (sizing_at minus sizing_bar_time, 1 dp) and
    sizing_price_is_stale (the latest quote's flag, else None). All None when
    no info is passed. Never raises."""
    from datetime import datetime, timezone
    rec = {"brief_price": None, "brief_built_at": None, "event_time": None,
           "event_time_source": None, "event_time_date_only": None,
           "sizing_price": price, "sizing_at": datetime.now(timezone.utc).isoformat(),
           "move_brief_to_sizing_pct": None, "minutes_brief_to_sizing": None,
           "minutes_event_to_sizing": None,
           "sizing_price_source": None, "sizing_bar_time": None, "sizing_price_age_s": None,
           "sizing_price_is_stale": None}
    try:
        info = price_info or {}
        rec["sizing_price_source"] = info.get("source")
        st = info.get("is_stale")
        rec["sizing_price_is_stale"] = bool(st) if st is not None else None
        bt = _as_of_utc(info)   # the brief record's rule: a zone-less time stays None
        if bt is not None:
            rec["sizing_bar_time"] = bt.isoformat()
            rec["sizing_price_age_s"] = round(
                (datetime.fromisoformat(rec["sizing_at"]) - bt).total_seconds(), 1)
    except Exception as e:
        rec["sizing_info_error"] = type(e).__name__
    try:
        prov = result.get("provenance")
        trig = (prov.get("trigger") if isinstance(prov, dict) else None) or {}
        now = datetime.now(timezone.utc)
        bp = trig.get("brief_price")
        rec["brief_price"] = float(bp) if bp else None
        rec["brief_built_at"] = trig.get("brief_built_at")
        rec["event_time"] = trig.get("event_time")
        rec["event_time_source"] = trig.get("event_time_source")
        if rec["event_time"]:
            # A filing_date or "date" key holds a day, not a time: it would
            # parse as 00:00 UTC and inflate the minutes by up to a day
            # (professor 67j).
            rec["event_time_date_only"] = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(rec["event_time"]).strip()))
        if rec["brief_price"] and price:
            rec["move_brief_to_sizing_pct"] = round((price - rec["brief_price"]) / rec["brief_price"] * 100, 3)
        built = _parse_when(rec["brief_built_at"])
        if built:
            rec["minutes_brief_to_sizing"] = round((now - built).total_seconds() / 60, 2)
        ev = None if rec["event_time_date_only"] else _parse_when(rec["event_time"])
        if ev:
            rec["minutes_event_to_sizing"] = round((now - ev).total_seconds() / 60, 1)
    except Exception as e:
        rec["error"] = type(e).__name__
    return rec


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


def _read_implied_move(ticker: str, today, out: dict) -> None:
    """The options poll's stored implied move for ticker (db.get_state of
    options_implied_move_<TRADING_ENV>; no broker or network call), written
    into `out` as it is read: implied_move_last_pct/_last_date/_last_dte from
    the stored move whatever its date (not after `today`, an ET date), and
    implied_move_pct/_dte when the stored move is today's, and the
    implied_move_event_* keys from the entry's event-spanning straddle
    (_read_event_move; record only, the keys above keep the nearest
    expiry). Keys it can't fill are left as they are. Raises when the state
    can't be read or parsed. Shared by _gap_risk_shadow and the trim/exit
    rule's record."""
    import json
    import math
    from datetime import date
    import db
    from config import TRADING_ENV

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    book = json.loads(db.get_state(f"options_implied_move_{TRADING_ENV}") or "{}")
    m = book.get(ticker) if isinstance(book, dict) else None
    last_date = str(m.get("et_date") or "") if isinstance(m, dict) else ""
    if last_date and last_date <= today.isoformat():
        last = num(m.get("implied_move_pct"))
        if last is not None and last >= 0:
            out["implied_move_last_pct"] = last
            out["implied_move_last_date"] = last_date
            try:   # the straddle's days to expiry on its own date
                exp = date.fromisoformat(str(m.get("expiry"))[:10]) if m.get("expiry") else None
                out["implied_move_last_dte"] = (exp - date.fromisoformat(last_date[:10])).days if exp else None
            except ValueError:
                pass
    if isinstance(m, dict) and str(m.get("et_date") or "") == today.isoformat():
        mv = num(m.get("implied_move_pct"))
        out["implied_move_pct"] = mv if mv is not None and mv >= 0 else None
        try:   # days to the straddle's expiry, when the record has one
            exp = date.fromisoformat(str(m.get("expiry"))[:10]) if m.get("expiry") else None
            out["implied_move_dte"] = (exp - today).days if exp else None
        except ValueError:
            pass
    if isinstance(m, dict):
        _read_event_move(m.get("event_spanning"), today, out)


def _read_event_move(ev, today, out: dict) -> None:
    """The book entry's event-spanning straddle (tasks._implied_move_book_extras)
    into `out` when it was priced on a date not after `today`:
    implied_move_event_pct, _expiry, _dte (days to its expiry on the date it
    was priced), _earnings_date (the print it was priced against) and _date
    (the ET date it was priced). An entry without one (written before it was
    kept) fills nothing. Never raises: a malformed one fills nothing."""
    import math
    from datetime import date

    def day(v):
        try:
            return date.fromisoformat(str(v)[:10]).isoformat() if v else None
        except ValueError:
            return None

    try:
        if not isinstance(ev, dict):
            return
        ev_date = day(ev.get("et_date"))
        if not ev_date or ev_date > today.isoformat():
            return
        try:
            pct = float(ev.get("implied_move_pct"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(pct) or pct < 0:
            return
        expiry = day(ev.get("expiry"))
        try:
            dte = float(ev.get("dte"))
            dte = int(dte) if math.isfinite(dte) else None
        except (TypeError, ValueError):
            dte = None
        if dte is None and expiry:
            dte = (date.fromisoformat(expiry) - date.fromisoformat(ev_date)).days
        out.update(implied_move_event_pct=pct, implied_move_event_expiry=expiry,
                   implied_move_event_dte=dte, implied_move_event_earnings_date=day(ev.get("earnings_date")),
                   implied_move_event_date=ev_date)
    except Exception as e:
        log.debug(f"event implied move read failed: {e}")


def _gap_risk_shadow(ticker: str, shares, price, stop, equity, setup_gate_path: bool = False,
                     bucket: str | None = None, regime_multiplier=None, today=None,
                     held_shares=None) -> dict:
    """What a stop that fills through a gap would cost, next to the per-trade
    risk cap, which assumes the stop fills at the stop price. The gap is the
    larger of the stop distance and today's options implied move (the
    nearest-expiry ATM straddle the options poll stores; null when there is
    none for today). All _pct values are percent units (4.0 = 4%), as
    implied_move_pct is; no dollar amount is returned. held_shares (ADDs
    only; None on an OPEN or when unknown) adds
    gap_risk_pct_of_equity_position: an ADD re-places the stop for the whole
    position, so a gap through it hits held plus new shares. Log-only: never
    raises, read by nothing that sizes or routes.
    implied_move_last_pct/_last_date/_last_dte (days to the straddle's expiry
    on that date): the stored move whatever its date, so
    a pre-open or Monday verdict (today's poll not run yet) still has one;
    gap_risk_pct_of_equity_last uses it when today's is missing (m604).
    implied_move_pct and status keep their today-only meaning.
    implied_move_event_pct/_expiry/_dte/_earnings_date/_date: the stored
    event-spanning straddle (_read_event_move), and
    gap_risk_pct_of_equity_event the same cost through max(stop distance,
    that move) when there is one; the other keys never read it."""
    out = {"implied_move_pct": None, "implied_move_dte": None, "stop_distance_pct": None,
           "implied_move_last_pct": None, "implied_move_last_date": None,
           "implied_move_last_dte": None, "gap_risk_pct_of_equity_last": None,
           "gap_risk_pct_of_equity": None, "gap_risk_pct_of_equity_position": None,
           "cap_pct": None, "would_exceed": None, "next_earnings_days": None, "status": "error",
           "implied_move_event_pct": None, "implied_move_event_expiry": None,
           "implied_move_event_dte": None, "implied_move_event_earnings_date": None,
           "implied_move_event_date": None, "gap_risk_pct_of_equity_event": None}
    try:
        import math
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import db
        from config import (RISK_PCT_PER_TRADE, CATEGORY_RISK_PCT_PER_TRADE)
        today = today or datetime.now(ZoneInfo("America/New_York")).date()

        def num(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) else None

        if setup_gate_path and bucket in CATEGORY_RISK_PCT_PER_TRADE:
            cap = CATEGORY_RISK_PCT_PER_TRADE[bucket] * (num(regime_multiplier) or 0.75)
        else:
            cap = RISK_PCT_PER_TRADE
        out["cap_pct"] = round(cap * 100, 4)
        try:
            er = db.get_next_earnings(ticker)
            if er is not None:
                out["next_earnings_days"] = (er - today).days
        except Exception as e:
            log.debug(f"[{ticker}] gap risk shadow earnings read failed: {e}")
        try:
            _read_implied_move(ticker, today, out)
        except Exception as e:
            log.debug(f"[{ticker}] gap risk shadow implied move read failed: {e}")

        shares, price, stop, equity = num(shares), num(price), num(stop), num(equity)
        if not (price and price > 0 and stop and 0 < stop < price):
            out["status"] = "no_stop"
            return out
        dist = (price - stop) / price * 100
        out["stop_distance_pct"] = round(dist, 4)
        if not (equity and equity > 0 and shares and shares > 0):
            out["status"] = "no_equity"
            return out
        gap = max(dist, out["implied_move_pct"] or 0.0)
        # shares is this order (shares_initial), the count the per-trade
        # risk cap bounds in size_position
        risk = shares * price * gap / equity
        out["gap_risk_pct_of_equity"] = round(risk, 4)
        held = num(held_shares)
        if held is not None and held > 0:
            out["gap_risk_pct_of_equity_position"] = round((held + shares) * price * gap / equity, 4)
        if out["implied_move_pct"] is None and out["implied_move_last_pct"] is not None:
            gap_last = max(dist, out["implied_move_last_pct"])
            out["gap_risk_pct_of_equity_last"] = round(shares * price * gap_last / equity, 4)
        try:   # record only: a failure here leaves it None, never the row's status
            ev = num(out["implied_move_event_pct"])
            if ev is not None:
                out["gap_risk_pct_of_equity_event"] = round(shares * price * max(dist, ev) / equity, 4)
        except Exception as e:
            log.debug(f"[{ticker}] gap risk shadow event move failed: {e}")
        out["would_exceed"] = risk > cap * 100
        out["status"] = "ok" if out["implied_move_pct"] is not None else "no_implied_move"
        return out
    except Exception as e:
        log.debug(f"[{ticker}] gap risk shadow failed: {e}")
        out["status"] = "error"
        return out


def _gap_risk_held_shares(ticker: str):
    """Shares held in ticker from the fresh portfolio snapshot
    (tasks._broker_held, no broker call), or None when unknown or not held."""
    try:
        import tasks
        held, _, pos = tasks._broker_held(ticker)
        qty = float((pos or {}).get("qty") or 0) if held else 0.0
        return qty if qty > 0 else None
    except Exception as e:
        log.debug(f"[{ticker}] gap risk shadow held read failed: {e}")
        return None


# Which check_concentration rule each flag comes from (record-only; the flag
# text is unchanged because other readers parse it).
_CONC_RULE_PREFIXES = (
    ("RISK CHECK FAILED", "errored"),
    ("AI infrastructure exposure", "ai_infra"),
    ("Total moonshot exposure", "moonshot_total"),
)


def _conc_failed_rules(flags) -> list:
    rules = []
    for f in flags or []:
        f = str(f)
        rule = next((r for pre, r in _CONC_RULE_PREFIXES if f.startswith(pre)), None)
        if rule is None:
            rule = ("single_name" if " position " in f and "exceeds" in f and " cap " in f
                    else "theme" if "theme exposure" in f else "other")
        if rule not in rules:
            rules.append(rule)
    return rules


_TIGHTEN_SIZING_LOOKAHEAD_DAYS = 3


def _tighten_sizing_shadow(ticker: str) -> dict:
    """Log-only (audit 11-06 11:34 #6): the macro and sector calendars set
    action_flags.tighten_sizing on events such as FOMC, CPI, jobs reports
    and sector earnings clusters, and nothing reads it. Records which such
    events fall from today through the next _TIGHTEN_SIZING_LOOKAHEAD_DAYS
    days, with their dates, so any window can be applied later. Sizing is
    unchanged; whether it should read this is Luke's. Never raises."""
    try:
        from catalyst_service.client import upcoming_events
        rows = [{"date": e.get("event_date"), "scope": e.get("scope"),
                 "event_type": e.get("event_type"), "ticker": e.get("ticker")}
                for e in upcoming_events(ticker, days_ahead=_TIGHTEN_SIZING_LOOKAHEAD_DAYS)
                if (e.get("action_flags") or {}).get("tighten_sizing")]
        return {"lookahead_days": _TIGHTEN_SIZING_LOOKAHEAD_DAYS, "n": len(rows), "events": rows[:10]}
    except Exception as e:
        return {"lookahead_days": _TIGHTEN_SIZING_LOOKAHEAD_DAYS, "n": None, "error": type(e).__name__}


def _udr_at_sizing(result: dict, price) -> dict | None:
    """Log-only: the model's own bull target and downside floor against the
    price the order is sized at, which can be hours after the brief's price.
    gate_at_sizing says whether that ratio still clears the entry UDR
    floor; nothing reads it
    to decide. Never raises."""
    try:
        import math
        from agents.plans import _dollars
        from agents.analyst import _ENTRY_UDR_FLOOR
        val = result.get("valuation") if isinstance(result.get("valuation"), dict) else {}
        target, floor = _dollars(val.get("bull_case_target")), _dollars(val.get("downside_scenario"))
        try:
            model = float(result.get("upside_downside_ratio"))
            model = model if math.isfinite(model) else None
        except (TypeError, ValueError):
            model = None
        p = float(price) if price else None
        raw = ((target - p) / (p - floor)
               if p and target and floor and target > p > floor else None)
        udr = round(raw, 2) if raw is not None else None
        return {"price": p, "target": target, "floor": floor, "udr": udr, "model_udr": model,
                # the unrounded ratio, so 1.995 doesn't pass a 2.0 floor
                "gate_at_sizing": (raw >= _ENTRY_UDR_FLOOR) if raw is not None else None,
                "reason": None if udr is not None else
                ("no_price" if not p else "no_target_or_floor" if not (target and floor)
                 else "price_outside_target_floor")}
    except Exception as e:
        return {"error": str(e)[:120]}


def _rr_vs_stop(result: dict, sizing: dict, price, thresholds=None) -> dict:
    """Record only (config RR_VS_STOP_SHADOW_ENABLED): reward to the model's
    target over risk to the stop sizing placed, for risk_json["rr_vs_stop"]
    on an OPEN/ADD. The entry UDR gate reads the model's own
    upside_downside_ratio, whose downside is its downside_scenario, not this
    stop. Fields:
      r_to_target      (target - price) / (price - stop). target is
                       valuation.bull_case_target, else price x (1 +
                       expected_return_pct / 100); target_source says which
      r_to_plan_t1/_t2 the same with sizing["plan_compiled"]'s t1 (nearest
                       ladder resistance) and t2
      stop_pct/_atr    the stop distance in % of price and in ATR units;
      target_pct/_atr  the same for the target. atr_source: "atr_stop" (the
                       ATR the stop was set from), else "atr_brief" (the
                       nightly ladder's ATR14), else None
      model_udr, model_floor (valuation.downside_scenario), floor_vs_stop_pct
                       ((floor - stop) / price x 100; + means the stop sits
                       under the model's floor), stop_above_model_floor
      would_block      {"r_lt_<t>": r_to_target < t} per threshold (default
                       RR_VS_STOP_LABEL_THRESHOLDS); None per label when
                       r_to_target is None. Unrounded R is compared.
      reason           why r_to_target is None, else None
    A ratio is None unless stop < price < its target. No reads: everything
    comes from result and sizing, so scripts/rr_vs_stop.py replays it on
    stored approvals. On an ADD, execution later sets one stop for the whole
    position (execution._whole_position_stop); this sees the sizing stop.
    Nothing routes or sizes on it. Never raises."""
    import math

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) and not isinstance(v, bool) else None

    out = {"r_to_target": None, "would_block": None, "reason": None}
    try:
        from agents.plans import _dollars
        if thresholds is None:
            try:
                from config import RR_VS_STOP_LABEL_THRESHOLDS as thresholds
            except ImportError:
                thresholds = (1.5, 2.0)
        result = result if isinstance(result, dict) else {}
        sizing = sizing if isinstance(sizing, dict) else {}
        val = result.get("valuation") if isinstance(result.get("valuation"), dict) else {}
        p, stop = num(price), num(sizing.get("stop_loss_price"))
        target, source = num(_dollars(val.get("bull_case_target"))), "bull_case_target"
        if not target and p and num(result.get("expected_return_pct")) is not None:
            target, source = p * (1 + num(result.get("expected_return_pct")) / 100), "expected_return_pct"
        atr, atr_source = num(sizing.get("atr_stop")), "atr_stop"
        if not atr or atr <= 0:
            ab = sizing.get("atr_brief") if isinstance(sizing.get("atr_brief"), dict) else {}
            atr, atr_source = num(ab.get("atr")), "atr_brief"
        if not atr or atr <= 0:
            atr, atr_source = None, None
        floor = num(_dollars(val.get("downside_scenario")))
        plan = sizing.get("plan_compiled") if isinstance(sizing.get("plan_compiled"), dict) else {}

        def r(tgt):
            return (tgt - p) / (p - stop) if p and stop and tgt and tgt > p > stop else None

        raw = r(target)
        rnd = lambda v, n=2: round(v, n) if v is not None else None
        out.update(
            price=p, stop=stop, stop_method=sizing.get("stop_method"),
            target=rnd(target), target_source=source if target else None,
            r_to_target=rnd(raw),
            r_to_plan_t1=rnd(r(num(plan.get("t1")))), r_to_plan_t2=rnd(r(num(plan.get("t2")))),
            stop_pct=rnd((p - stop) / p * 100) if p and stop else None,
            stop_atr=rnd((p - stop) / atr) if p and stop and atr else None,
            target_pct=rnd((target - p) / p * 100) if p and target else None,
            target_atr=rnd((target - p) / atr) if p and target and atr else None,
            atr=rnd(atr, 4), atr_source=atr_source,
            model_udr=num(result.get("upside_downside_ratio")), model_floor=floor,
            floor_vs_stop_pct=rnd((floor - stop) / p * 100) if p and stop and floor else None,
            stop_above_model_floor=(stop > floor) if stop and floor else None,
            thresholds=[float(t) for t in thresholds],
            would_block={f"r_lt_{float(t):g}": (raw < float(t)) if raw is not None else None
                         for t in thresholds},
        )
        if raw is None:
            out["reason"] = ("no_price" if not p else "no_stop" if not stop else "no_target" if not target
                             else "levels_not_stop_lt_price_lt_target")
        return out
    except Exception as e:
        out["error"] = str(e)[:120]
        return out


def _attach_gap_risk_shadow(rm, ticker: str, sizing: dict, price, action: str | None = None) -> None:
    """sizing["gap_risk_shadow"] for the order as it will be sent (after the
    concentration, correlation and micro-name cuts). Equity is read for the
    ratio and not stored. On an ADD the held share count comes from the
    portfolio snapshot. Log-only, never raises."""
    try:
        held_shares = _gap_risk_held_shares(ticker) if action == "ADD" else None
        try:
            equity = rm.portfolio_value()
        except Exception as e:
            log.debug(f"[{ticker}] gap risk shadow equity read failed: {e}")
            equity = None
        shadow = _json_safe(_gap_risk_shadow(
            ticker, sizing.get("shares_initial"), price, sizing.get("stop_loss_price"), equity,
            setup_gate_path="heat_blocked" in sizing, bucket=sizing.get("bucket"),
            regime_multiplier=sizing.get("regime_multiplier"), held_shares=held_shares))
        sizing["gap_risk_shadow"] = None if shadow is _UNENCODABLE else shadow
        status = sizing.get("shadow_status")
        if isinstance(status, dict):
            status["gap_risk_shadow"] = (
                "unencodable" if shadow is _UNENCODABLE
                else "error" if (shadow or {}).get("status") == "error" else "ok")
    except Exception as e:
        log.debug(f"[{ticker}] gap risk shadow attach failed: {e}")


SWAP_CASH_WAIT_S = 16


def _wait_for_swap_cash(rm, cash_before: float | None, sell_result: dict, ticker: str) -> bool:
    """Poll cash (every 2 s, up to SWAP_CASH_WAIT_S) until at least 90% of the
    swap sale's proceeds show up. True when they did; False on timeout or
    when the inputs are unknown (then the re-size runs on whatever cash says)."""
    import time
    try:
        proceeds = float(sell_result.get("filled_qty") or 0) * float(sell_result.get("fill_price") or 0)
    except (TypeError, ValueError):
        proceeds = 0.0
    if cash_before is None or proceeds <= 0:
        return False
    target = cash_before + 0.9 * proceeds
    # ib.sleep runs ib_async's event loop, so the cached account values can
    # update while waiting; time.sleep would block it (review of d1ab4f2).
    try:
        import ibkr
        pause = ibkr.get_ib().sleep
    except Exception:
        pause = time.sleep
    for _ in range(SWAP_CASH_WAIT_S // 2):
        try:
            if rm.cash() >= target:
                return True
        except Exception:
            pass
        pause(2)
    log.warning(f"[{ticker}] swap proceeds not in cash after {SWAP_CASH_WAIT_S}s; re-sizing anyway")
    return False


# RiskManager(trading_client) raising (e.g. TWS unreachable) used to end
# routing with a log line only: no row, no alert (10-19: 10 OPEN c7 verdicts
# lost). Now a ROUTE_RISK_INIT_FAILED row per verdict and at most one alert
# per process per window; hits inside the window go into the next alert.
_RISK_INIT_ALERT_WINDOW_S = 30 * 60
_RISK_INIT_ALERT = {"sent_at": None, "held": []}
_RISK_INIT_ALERT_MAX_LISTED = 20


def _risk_init_would_retry(action) -> bool:
    """Shadow only: whether retry_deferred_entry's rules would take this
    verdict. It retries new entries (OPEN/ADD) under
    tasks.DEFERRED_ENTRY_MAX_AGE_HOURS that no newer approval or non-buy
    verdict supersedes; at routing time the verdict is the newest one, its
    age is ~0, and an active approval already ended routing earlier
    (ROUTE_APPROVAL_PENDING). Nothing is scheduled."""
    return action in ("OPEN", "ADD")


def _arm_risk_init_timer(now: float, last: float) -> None:
    """Start the daemon timer that runs _flush_risk_init_held when the
    current alert window ends. Caller holds the lock. Never raises."""
    try:
        import threading
        t = threading.Timer(max(1.0, _RISK_INIT_ALERT_WINDOW_S - (now - last)),
                            _flush_risk_init_held)
        t.daemon = True
        t.start()
        _RISK_INIT_ALERT["timer"] = t
    except Exception as e:
        log.warning(f"risk-init held-alert timer not started: {e}")


def _flush_risk_init_held(now: float | None = None) -> bool:
    """Timer body (professor m302): send the hits held since the last alert
    once its window is over, so they reach Telegram even when no further
    failure comes. Counts as that window's alert. True when it sent one.
    Never raises."""
    try:
        import threading
        import time
        lock = _RISK_INIT_ALERT.setdefault("lock", threading.Lock())
        now = time.monotonic() if now is None else now
        with lock:
            _RISK_INIT_ALERT["timer"] = None
            held = _RISK_INIT_ALERT["held"]
            if not held:
                return False
            last = _RISK_INIT_ALERT["sent_at"]
            if last is not None and now - last < _RISK_INIT_ALERT_WINDOW_S - 1:
                # A failure's own alert went out since this timer was set and
                # these were held after it: wait out that window too.
                _arm_risk_init_timer(now, last)
                return False
            _RISK_INIT_ALERT["held"] = []
            _RISK_INIT_ALERT["sent_at"] = now
        listed = ", ".join(held[:_RISK_INIT_ALERT_MAX_LISTED])
        more = len(held) - _RISK_INIT_ALERT_MAX_LISTED
        import alerts
        alerts._send(f"Routing blocked: RiskManager init failed when sizing {len(held)} more "
                     f"verdict(s) since the last alert; not executed: {listed}"
                     + (f" and {more} more" if more > 0 else ""))
        return True
    except Exception as e:
        log.warning(f"risk-init held-alert flush failed: {e}")
        return False


def _alert_risk_init_failed(item: str, now: float | None = None) -> bool:
    """One Telegram per process per _RISK_INIT_ALERT_WINDOW_S. A hit inside
    the window is held and listed in the next alert; the first held hit
    starts a timer that sends them when the window ends if no later failure
    has (professor m302). True when it sent one. Never raises."""
    try:
        import threading
        import time
        lock = _RISK_INIT_ALERT.setdefault("lock", threading.Lock())
        now = time.monotonic() if now is None else now
        with lock:
            last = _RISK_INIT_ALERT["sent_at"]
            if last is not None and now - last < _RISK_INIT_ALERT_WINDOW_S:
                _RISK_INIT_ALERT["held"].append(item)
                if _RISK_INIT_ALERT.get("timer") is None:
                    _arm_risk_init_timer(now, last)
                return False
            held, _RISK_INIT_ALERT["held"] = _RISK_INIT_ALERT["held"], []
            _RISK_INIT_ALERT["sent_at"] = now
        text = f"Routing blocked: RiskManager init failed when sizing {item}; verdict not executed"
        if held:
            listed = ", ".join(held[:_RISK_INIT_ALERT_MAX_LISTED])
            more = len(held) - _RISK_INIT_ALERT_MAX_LISTED
            text += (f"\nAlso blocked since the last alert ({len(held)}): {listed}"
                     + (f" and {more} more" if more > 0 else ""))
        import alerts
        alerts._send(text)
        return True
    except Exception as e:
        log.warning(f"risk-init alert failed: {e}")
        return False


def _record_risk_init_failed(ticker: str, result: dict, exc: Exception,
                             extra: dict | None = None) -> None:
    """ROUTE_RISK_INIT_FAILED row plus the throttled alert. The row carries
    the exception class only: the message can include account details.
    extra's keys go into the row's decision (the recheck-contradiction
    record). Never raises."""
    try:
        action = result.get("action")
        conviction = result.get("conviction")
        error_class = type(exc).__name__
        _log_route_exit(ticker, "ROUTE_RISK_INIT_FAILED", result,
                        f"RiskManager init failed ({error_class}); not sized, not executed",
                        decision_extra={"error_class": error_class,
                                        "would_retry": _risk_init_would_retry(action),
                                        **(extra or {})})
        _alert_risk_init_failed(f"{ticker} {action} c{conviction} ({error_class})")
    except Exception as e:
        log.warning(f"[{ticker}] risk-init failure record failed: {e}")


NEW_FACT_SCOPES = ("all_buys", "news_only", "all_but_held_review_add")
NEW_FACT_WINDOWS = {"same_run": 0, "24h": 24, "3d": 72}
# Paths of a review of a held position (analyst.trigger_record's path).
_HELD_REVIEW_PATHS = ("scheduled_review", "earnings_review", "plan_level_review")


# Bad BUY_* new-fact settings already alerted in this process ("NAME=value").
_NEW_FACT_SETTING_ALERTED: set = set()


def _alert_bad_new_fact_setting(name: str, value, used: str) -> None:
    """One low-severity Telegram per process per bad value. Never raises."""
    key = f"{name}={value!r}"
    if key in _NEW_FACT_SETTING_ALERTED:
        return
    _NEW_FACT_SETTING_ALERTED.add(key)
    log.warning(f"{name} {value!r} is not a valid value; using {used}")
    try:
        import alert_policy
        import alerts
        alerts._send(f"Config: {name}={str(value)[:40]!r} is not a valid value; the buy "
                     f"gate uses {used} (fail closed). Fix the setting in .env.",
                     severity=alert_policy.LOW)
    except Exception as e:
        log.warning(f"bad-setting alert failed: {e}")


def _new_fact_settings() -> tuple[bool, str, str]:
    """(on, scope, window) from config, each invalid value replaced by the
    strict choice (gate on, all_buys, same_run) and alerted once."""
    import config
    raw = getattr(config, "BUY_REQUIRES_NEW_FACT_RAW", "true")
    if raw not in ("true", "false"):
        _alert_bad_new_fact_setting("BUY_REQUIRES_NEW_FACT", raw, "true (gate on)")
    on = bool(getattr(config, "BUY_REQUIRES_NEW_FACT", True))
    scope = getattr(config, "BUY_NEW_FACT_SCOPE", "all_buys")
    window = getattr(config, "BUY_NEW_FACT_WINDOW", "same_run")
    if scope not in NEW_FACT_SCOPES:
        _alert_bad_new_fact_setting("BUY_NEW_FACT_SCOPE", scope, "all_buys")
        scope = "all_buys"
    if window not in NEW_FACT_WINDOWS:
        _alert_bad_new_fact_setting("BUY_NEW_FACT_WINDOW", window, "same_run")
        window = "same_run"
    return on, scope, window


def new_fact_gate(ticker: str, action, trigger: dict | None) -> dict | None:
    """Whether an OPEN/ADD may route under BUY_REQUIRES_NEW_FACT: the
    screener's new_fact on the verdict's news item (trigger, the verdict's
    provenance.trigger), per BUY_NEW_FACT_SCOPE and BUY_NEW_FACT_WINDOW.
    None when the gate is off or the action isn't a buy. A missing new_fact
    counts as false. An invalid setting falls back to the strict choice
    (gate on, all_buys, same_run) and is alerted once (_new_fact_settings)."""
    on, scope, window = _new_fact_settings()
    if not on or action not in ("OPEN", "ADD"):
        return None
    trig = trigger if isinstance(trigger, dict) else {}
    path = trig.get("path")
    news = path == "event"
    item = trig.get("new_fact") if news else None
    rec = {"scope": scope, "window": window, "path": path, "news_item": news,
           "item_new_fact": item if isinstance(item, bool) else None}
    if (scope == "news_only" and not news) or (
            scope == "all_but_held_review_add" and action == "ADD" and path in _HELD_REVIEW_PATHS):
        rec.update(applies=False, passed=True, reason=f"not gated ({scope})")
        return rec
    rec["applies"] = True
    if trig.get("source") == "dip_watch":
        # A drop-watch review has no news item, so it has no new_fact by
        # construction. DIP_REVIEW_COUNTS_AS_NEW_FACT (default off; Luke's)
        # lets the drop itself count; otherwise the window rule below decides.
        import config
        rec["dip_review"] = True
        if config.DIP_REVIEW_COUNTS_AS_NEW_FACT:
            rec.update(passed=True, reason="a drop-watch review (DIP_REVIEW_COUNTS_AS_NEW_FACT)")
            return rec
    elif news and not isinstance(item, bool):
        log.warning(f"[{ticker}] screener new_fact missing for the triggering item; "
                    f"counted as false")
        rec["item_new_fact_missing"] = True
    if item is True:
        rec.update(passed=True, reason="the triggering news item is a new fact")
        _note_new_fact_tone(rec, [trig.get("title_tone")])
        return rec
    if NEW_FACT_WINDOWS[window]:
        import db
        items = db.recent_new_fact_items(ticker, NEW_FACT_WINDOWS[window])
        if items:
            rec.update(passed=True, window_items=items,
                       reason=f"a new-fact news item on {ticker} within {window}")
            _note_new_fact_tone(rec, [i.get("title_tone") for i in items])
            return rec
        if items is None:
            rec["window_lookup_error"] = True
    what = ("the triggering item gave no new_fact (counted as false)" if news and item is None
            else "the triggering item is not a new fact (opinion/recap)" if news
            else f"a {path or 'unknown'} verdict has no news item")
    also = ("" if window == "same_run" else
            f"; no new-fact item on {ticker} within {window}"
            + (" (lookup failed)" if rec.get("window_lookup_error") else ""))
    rec.update(passed=False, reason=f"no new fact ({scope}, {window}): {what}{also}")
    return rec


_NEW_FACT_FALLBACK_ALERT_WINDOW_S = 3600


def _note_new_fact_tone(rec: dict, tones: list) -> None:
    """Log-only (audit 2026-11-09 16:53 #3): the headline tones of the
    new-fact items that let a buy pass, and new_fact_tone_mismatch when every
    one with a known tone reads negative (a bad-news fact licensing an
    OPEN/ADD). None when no tone is known. The gate's decision never reads
    it. Never raises."""
    try:
        known = [t for t in tones if t in ("pos", "neg", "neutral", "mixed")]
        rec["licensing_tones"] = list(tones)
        rec["new_fact_tone_mismatch"] = (all(t == "neg" for t in known) if known else None)
    except Exception:
        rec["new_fact_tone_mismatch"] = None


def _alert_new_fact_fallback_block(ticker: str, gate: dict, trigger, now: datetime | None = None) -> bool:
    """Telegram for a buy the new-fact gate blocked because the triggering
    item's new_fact was missing after a keyword-fallback screen (not a real
    false), at most one per hour (system_state key written after a successful
    send). Alert only: the block stands. True when sent. Never raises."""
    try:
        reason = trigger.get("screener_fallback_reason") if isinstance(trigger, dict) else None
        if not (reason and gate.get("item_new_fact_missing")):
            return False
        import alerts
        import db
        from config import TRADING_ENV
        now = now or datetime.now(timezone.utc)
        key = f"new_fact_fallback_block_alert:{TRADING_ENV}"
        last = db.get_state(key)
        if last:
            try:
                if (now - datetime.fromisoformat(last)).total_seconds() < _NEW_FACT_FALLBACK_ALERT_WINDOW_S:
                    return False
            except (TypeError, ValueError):
                pass   # unreadable stamp: send
        if alerts._send(f"{ticker}: buy blocked by the new-fact gate because the screener "
                        f"was on the keyword fallback ({reason}), not a real new_fact=false; "
                        f"check the 7B model server") is None:
            return False
        db.set_state(key, now.isoformat())
        return True
    except Exception as e:
        log.warning(f"[{ticker}] new-fact fallback alert failed: {e}")
        return False


def _c6_bar_no_new_fact(result: dict, regime_str: str) -> None:
    """c6 bar shadow for a buy the new-fact gate stopped before sizing (where
    the tier floor would have been checked): a fire-and-forget enqueue.
    Log-only, never raises."""
    try:
        from agents.c6_shadow import observe_new_fact_block
        observe_new_fact_block(result, regime_str)
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow failed: {e}")


def _add_guard_record(ticker: str, result: dict, sizing: dict | None,
                      today=None) -> dict | None:
    """decision_extra for an ADD that dies in routing after the price read
    (ROUTE_ZERO_SHARES, ROUTE_CONCENTRATION_ZERO, ROUTE_MICRO_NAME_CAP):
    the add-to-loser shadow sizing already holds, and next_earnings_days,
    the raw calendar days from today (ET) to the next earnings date, as
    _gap_risk_shadow counts them, with no window applied. next_earnings_read
    says what that read returned: ok, no_row, or "error: <class>" (the plain
    db.get_next_earnings reads a failed query as no row). None for any other
    action. Record only: read by nothing that routes or sizes. Never raises."""
    try:
        if result.get("action") != "ADD":
            return None
        rec = {"add_to_loser": (sizing or {}).get("add_to_loser"),
               "next_earnings_days": None, "next_earnings_read": None}
        try:
            import db
            from zoneinfo import ZoneInfo
            today = today or datetime.now(ZoneInfo("America/New_York")).date()
            er = db.get_next_earnings(ticker, strict=True)
            if er is None:
                rec["next_earnings_read"] = "no_row"
            else:
                rec["next_earnings_days"] = int((er - today).days)
                rec["next_earnings_read"] = "ok"
        except Exception as e:
            rec["next_earnings_read"] = f"error: {type(e).__name__}"
        return rec
    except Exception as e:
        return {"add_guard_error": type(e).__name__}


def _apply_micro_name_cap(rm, ticker: str, sizing: dict, price: float, result: dict) -> bool:
    """Micro-tier per-name cap on an OPEN/ADD (RiskManager.micro_name_cap):
    cuts sizing in place and records the decision in sizing["micro_name_cap"].
    True when it leaves 0 shares; then a ROUTE_MICRO_NAME_CAP row is written
    and the caller stops. Existing holdings are never sold."""
    rec = rm.micro_name_cap(ticker, sizing["shares_initial"], price)
    if rec is None:            # flag off or not the micro tier
        return False
    sizing["micro_name_cap"] = rec
    for f in list(sizing.get("flags") or []):
        # The micro name cap sits under every tier cap, so a "bucket maximum"
        # flag names a limit that can't bind here (audit 11-06 21:36 #9). The
        # note doesn't claim the name cap binds: another limit may size lower
        # (professor 62n); a name-cap cut gets its own flag below.
        m = re.match(r"Size capped at (\d+(?:\.\d+)?)% bucket maximum$", f)
        if m and float(m.group(1)) > float(rec["cap_pct"]):
            sizing["flags"].append(f"Micro tier: the {rec['cap_pct']:g}% name cap is below the "
                                   f"{m.group(1)}% bucket maximum, so the bucket maximum can't bind")
            break
    old = sizing["shares_initial"]
    if rec["shares_after"] >= old:
        return False
    if rec["shares_after"] <= 0:
        reason = ("micro-tier name cap: account, position or price unusable, buy blocked"
                  if rec["outcome"] == "errored" else
                  f"micro-tier name cap {rec['cap_pct']:g}%: {ticker} held {rec['held_pct']}% of "
                  f"equity; 1 share at ${price:,.2f} would take it over")
        sizing["flags"].append(reason)
        log.info(f"[{ticker}] Entry blocked: {reason}")
        _log_route_exit(ticker, "ROUTE_MICRO_NAME_CAP", result, reason,
                        decision_extra={"micro_name_cap": rec,
                                        "theme_group_shadow": sizing.get("theme_group_shadow"),
                                        "concentration_inflight_shadow":
                                            sizing.get("concentration_inflight_shadow"),
                                        **(_add_guard_record(ticker, result, sizing) or {}),
                                        **({"recheck_contradiction": sizing["recheck_contradiction"]}
                                           if "recheck_contradiction" in sizing else {}),
                                        **({"same_day_consensus": sizing["same_day_consensus"]}
                                           if "same_day_consensus" in sizing else {})})
        return True
    new = rec["shares_after"]
    sizing["shares_initial"] = new
    sizing["initial_dollar"] = round(new * price, 2)
    if sizing.get("order_pct"):
        sizing["order_pct"] = round(sizing["order_pct"] * new / old, 4)
    if rec.get("held_pct") is not None:
        headroom = max(0.0, (rec["cap_pct"] - rec["held_pct"]) / 100)
        sizing["position_size_pct"] = round(min(sizing["position_size_pct"], headroom), 4)
    sizing["flags"].append(f"Cut from {old} to {new} shares by the micro-tier "
                           f"{rec['cap_pct']:g}% name cap")
    log.info(f"[{ticker}] Micro-tier name cap: {old} -> {new} shares")
    return False


def _atr_brief(ticker: str) -> dict | None:
    """The ATR(14) the brief showed (the nightly trade ladder's) and when
    that ladder was computed, beside sizing's atr_stop (audit 2026-11-07
    11:07 #7: the model reasons about one volatility and the stop is set
    from another). ladder_row is False when get_trade_ladder gave nothing,
    which is either no row or a failed read (it swallows errors); None only
    when the call itself raised. Log-only. Never raises."""
    try:
        import db
        row = db.get_trade_ladder(ticker)
        ld = row or {}
        try:
            atr = round(float(ld["atr"]), 4) if ld.get("atr") is not None else None
        except (TypeError, ValueError):
            atr = None
        return {"atr": atr, "ladder_row": bool(row),
                "computed_at": str(ld.get("computed_at")) if ld.get("computed_at") else None}
    except Exception:
        return None

def _add_context(ticker: str, price, sizing: dict, trading_client, rm=None) -> dict:
    """Log-only (audit 2026-11-10 16:53 #2): where an ADD buys relative to
    the position it adds to. kept_stop is max(sizing's new stop, the stop
    record), the whole-position level an add keeps (execution.
    _whole_position_stop); stop_distance_atr is (price - kept_stop) / the
    sizing ATR; below_avg_cost compares the price with the broker's average
    cost (portfolio cache, may lag). NVDA 11-10 15:07 bought 0.35 ATR above
    its kept stop, ~10% below average cost.
    The exit state the ADD's stop-record write resets, read before it
    (record only): trailing_active and record_conviction from the stop
    record, entry_conviction (original_thesis) and profit_target_taken from
    the position context, and whether the gain over the estimated post-ADD
    average (held shares at the broker's average cost plus this order at
    this price) still meets the bucket's trail activation gain
    (trail_gain_qualifies_post_add; None where the gain rule does not apply
    or an input is unknown). state_error names a failed read of these fields
    alone; error keeps its old meaning.
    position_risk_pct_at_kept_stop: held plus new shares times (price -
    kept_stop), over equity (rm.portfolio_value(), read for the ratio and not
    stored), in percent; the per-trade cap counts the new order alone
    against its own stop. over_per_trade_cap compares it with
    per_trade_cap_pct (the gap-risk shadow's cap_pct, else
    RISK_PCT_PER_TRADE). Negative when the kept stop is above the price.
    position_risk_status says why it is missing. Never raises."""
    rec = {"price": None, "new_stop": None, "record_stop": None, "kept_stop": None,
           "atr": None, "stop_distance_atr": None, "avg_cost": None, "below_avg_cost": None,
           "pct_vs_avg_cost": None, "error": None,
           "trailing_active": None, "record_conviction": None, "entry_conviction": None,
           "profit_target_taken": None, "held_qty": None, "post_add_avg_est": None,
           "gain_vs_post_add_avg_pct": None, "trail_activate_pct": None,
           "trail_gain_qualifies_post_add": None, "state_error": None,
           "position_risk_pct_at_kept_stop": None, "per_trade_cap_pct": None,
           "over_per_trade_cap": None, "position_risk_status": None}
    prior, held_qty, avg = None, None, None
    try:
        import db
        px = float(price) if price else None
        rec["price"] = px
        new_stop = sizing.get("stop_loss_price")
        rec["new_stop"] = float(new_stop) if new_stop else None
        try:
            prior = db.get_position_stop(ticker, strict=True)
            rec["record_stop"] = float(prior["stop_price"]) if prior and prior.get("stop_price") else None
        except Exception as e:
            rec["error"] = f"stop record: {type(e).__name__}"
        stops = [x for x in (rec["new_stop"], rec["record_stop"]) if x]
        rec["kept_stop"] = max(stops) if stops else None
        atr = sizing.get("atr_stop")
        rec["atr"] = float(atr) if atr else None
        if px and rec["kept_stop"] and rec["atr"]:
            rec["stop_distance_atr"] = round((px - rec["kept_stop"]) / rec["atr"], 2)
        try:
            pos = trading_client.get_open_position(ticker)
            avg = float(pos.avg_entry_price)
            if avg > 0:
                rec["avg_cost"] = round(avg, 4)
                if px:
                    rec["below_avg_cost"] = px < avg
                    rec["pct_vs_avg_cost"] = round((px / avg - 1) * 100, 2)
            try:
                held_qty = abs(float(getattr(pos, "qty", None)))
            except (TypeError, ValueError):
                held_qty = None
        except Exception as e:
            rec["error"] = (rec["error"] + "; " if rec["error"] else "") + f"position: {type(e).__name__}"
    except Exception as e:
        rec["error"] = type(e).__name__
    rec.update(_add_exit_state(ticker, rec.get("price"), sizing, prior, held_qty, avg))
    rec.update(_add_position_risk(rec.get("price"), rec.get("kept_stop"), sizing, held_qty, rm))
    return rec


def _add_position_risk(px, kept_stop, sizing, held_qty, rm) -> dict:
    """_add_context's whole-position risk at the kept stop (see there).
    Never raises."""
    out = {"position_risk_pct_at_kept_stop": None, "per_trade_cap_pct": None,
           "over_per_trade_cap": None, "position_risk_status": "error"}
    try:
        import math
        from config import RISK_PCT_PER_TRADE
        cap = ((sizing or {}).get("gap_risk_shadow") or {}).get("cap_pct")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap):
            cap = RISK_PCT_PER_TRADE * 100
        out["per_trade_cap_pct"] = round(float(cap), 4)
        shares = float((sizing or {}).get("shares_initial") or 0)
        if not (px and kept_stop):
            out["position_risk_status"] = "no_stop"
            return out
        if not (held_qty and held_qty > 0 and shares > 0):
            out["position_risk_status"] = "no_shares"
            return out
        if rm is None:
            out["position_risk_status"] = "no_equity"
            return out
        try:
            equity = float(rm.portfolio_value())
        except Exception as e:
            out["position_risk_status"] = f"equity: {type(e).__name__}"
            return out
        if not (math.isfinite(equity) and equity > 0):
            out["position_risk_status"] = "no_equity"
            return out
        risk = (held_qty + shares) * (float(px) - float(kept_stop)) / equity * 100
        out["position_risk_pct_at_kept_stop"] = round(risk, 4)
        out["over_per_trade_cap"] = risk > cap
        out["position_risk_status"] = "ok"
    except Exception as e:
        out["position_risk_status"] = f"error: {type(e).__name__}"
    return out


def _add_exit_state(ticker: str, px, sizing, prior, held_qty, avg) -> dict:
    """_add_context's exit-state fields (see there). prior is the stop record
    _add_context read (None when absent or unreadable), held_qty and avg the
    broker position's. Never raises."""
    out, errs = {}, []
    try:
        # The broker share count _add_position_risk also uses, recorded
        # whether or not the post-ADD average below can be estimated.
        out["held_qty"] = held_qty
        if isinstance(prior, dict):
            ta = prior.get("trailing_active")
            out["trailing_active"] = bool(ta) if ta is not None else None
            out["record_conviction"] = prior.get("conviction")
        try:
            import db
            ctx = db.get_position_context(ticker, strict=True) or {}
            orig = ctx.get("original_thesis")
            if isinstance(orig, dict):
                out["entry_conviction"] = orig.get("conviction")
            ptt = ctx.get("profit_target_taken")
            out["profit_target_taken"] = bool(ptt) if ptt is not None else None
        except Exception as e:
            errs.append(f"position context: {type(e).__name__}")
        try:
            import risk_manager as _rm
            bucket = (sizing or {}).get("bucket")
            ladder = bool(_rm.SETUP_GATES_ENABLED) and bucket in _rm.CATEGORY_RISK_PCT_PER_TRADE
            activate = None if ladder else _rm.TRAILING_STOP_ACTIVATE.get(bucket)
            out["trail_activate_pct"] = round(activate * 100, 2) if activate else None
            shares = float((sizing or {}).get("shares_initial") or 0)
            px = float(px) if px else None
            if px and avg and avg > 0 and held_qty and held_qty > 0 and shares > 0:
                post = (held_qty * avg + shares * px) / (held_qty + shares)
                out["post_add_avg_est"] = round(post, 4)
                gain = (px - post) / post
                out["gain_vs_post_add_avg_pct"] = round(gain * 100, 2)
                if activate:
                    out["trail_gain_qualifies_post_add"] = gain >= activate
        except Exception as e:
            errs.append(f"trail rule: {type(e).__name__}")
    except Exception as e:
        errs.append(type(e).__name__)
    out["state_error"] = "; ".join(errs) or None
    return out


_STOPOUT_WINDOWS_DAYS = (1, 3, 5, 10)


def _stopout_lookback_days() -> int:
    """Days of broker exits the stop-out shadow reads: its widest window, or
    the live gate's window (config STOPOUT_REENTRY_GATE_DAYS) if wider."""
    try:
        from config import STOPOUT_REENTRY_GATE_DAYS as gate
    except ImportError:
        gate = 0
    widest = max(_STOPOUT_WINDOWS_DAYS)
    return max(widest, int(gate)) if isinstance(gate, (int, float)) and not isinstance(gate, bool) else widest


def _stopout_reentry_shadow(ticker: str, result: dict, path: str | None = None) -> dict | None:
    """Stop-out re-entry shadow (config STOPOUT_REENTRY_SHADOW_ENABLED). Called
    for an OPEN/ADD once it has passed the rebuy cooldown. When the ticker's
    newest exit the bot did not place (db.get_last_broker_exit: a resting
    stop fill, or a sale made at the broker; not the bot's own backstop or
    manual sells) is under 10 days old, writes one
    STOPOUT_REENTRY_SHADOW row: hours since that exit, its lots' outcomes, and
    would_block per re-entry cooldown window (1, 3, 5, 10 days). No row when
    there is no such exit. A failed read writes a row with read "error:
    <class>" and would_block None. The row itself changes nothing; it returns
    the record it wrote (None when no row) for _entry_gate, which may skip
    the buy and writes its own ENTRY_GATE_BLOCKED row. path, when given
    (tasks._deferred_entry_gate passes "deferred_retry"), is recorded on the
    row's stopout_reentry so a retry's second row for the same verdict can be
    told from the verdict-time one. Never raises."""
    try:
        try:
            from config import STOPOUT_REENTRY_SHADOW_ENABLED
        except ImportError:
            return
        if not STOPOUT_REENTRY_SHADOW_ENABLED or result.get("action") not in ("OPEN", "ADD"):
            return None
        import db
        kind = "STOPOUT_REENTRY_SHADOW"
        try:
            ex = db.get_last_broker_exit(ticker, _stopout_lookback_days(), strict=True)
            read = "ok"
        except Exception as e:
            ex, read = None, f"error: {type(e).__name__}"
        if ex is None and read == "ok":
            return None
        rec = {"read": read, "lookback_days": _stopout_lookback_days(),
               "windows_days": list(_STOPOUT_WINDOWS_DAYS), "would_block": None}
        if path:
            rec["path"] = path
        if ex:
            rec.update(ex)
            hrs = ex.get("hours_since")
            if isinstance(hrs, (int, float)):
                rec["would_block"] = {f"{d}d": hrs < d * 24 for d in _STOPOUT_WINDOWS_DAYS}
            outs = [v for v in ex.get("lot_outcome_pct") or [] if isinstance(v, (int, float))]
            rec["worst_lot_outcome_pct"] = min(outs) if outs else None
            convs = [v for v in ex.get("lot_convictions") or [] if isinstance(v, int)]
            conv = result.get("conviction")
            rec["conviction_above_lots"] = (conv > max(convs)
                                            if convs and isinstance(conv, (int, float)) else None)
            reason = (f"{result.get('action')} {hrs} h after an exit the bot did not place "
                      f"(worst lot {rec['worst_lot_outcome_pct']}%); record only")
        else:
            reason = f"{result.get('action')}: stop-out read failed ({read}); record only"
        # route_exit stays: it marks the row's provenance as a copy of the
        # verdict's (db._VERDICT_COPY_KEY), as on SHADOW_SWAP rows, so readers
        # that count provenance or LLM calls skip it. The buy goes on.
        fields = _route_row_fields(result, kind)
        fields["decision"]["shadow"] = kind
        fields["decision"]["stopout_reentry"] = rec
        try:
            db.log_signal(ticker, kind, 0.0, reason[:500], **fields)
        except Exception as e:
            log.warning(f"[{ticker}] stop-out re-entry shadow row not written: {type(e).__name__}: {e}")
        return rec
    except Exception as e:
        log.warning(f"[{ticker}] stop-out re-entry shadow failed: {type(e).__name__}: {e}")
        return None


# Entry-quality shadow (_entry_quality_shadow). One price_history read of this
# many rows (the brief and the trade ladder read 260); the 20-day high and the
# swing window of agents.price_action.compute_price_action (last 60 bars).
_ENTRY_QUALITY_BAR_ROWS = 260
_ENTRY_QUALITY_HIGH_DAYS = 20
_ENTRY_QUALITY_SWING_BARS = 60
_ENTRY_QUALITY_DRAWDOWN_ATR = 3.0
# Split-discontinuity guard (_split_discontinuity), on the closes the SMA200
# uses. A one-day close ratio r is a split candidate when it is within
# _SPLIT_RATIO_TOL (relative) of a factor in _SPLIT_FACTORS or its inverse
# (rule "factor"), or at or beyond _SPLIT_SCALE_BREAK or its inverse (rule
# "scale_break", for steps off the factor list: the SNDK cache's 2025-11-21
# step, x0.1277, is caught here), and its |ln r| is at least _SPLIT_MOVE_MULT
# times the median |ln r| of the window's other days. A real crash or pop
# passes those tests, so a candidate also needs volume corroboration: an
# unadjusted split steps volume by about 1/r and the step persists. The
# median volume of the _SPLIT_VOLUME_BARS bars after the step (skipping the
# first _SPLIT_VOLUME_SKIP_BARS, where a crash's volume spike sits) over the
# median of the _SPLIT_VOLUME_BARS bars before it, v, must give v x r within
# a factor of 1 + _SPLIT_VOLUME_TOL of 1 on the factor rule (dollar volume
# carries across a split; a crash's usually falls), or v at least
# _SPLIT_VOLUME_BREAK_MIN in the split's direction (up when the price stepped
# down) on the scale_break rule, with v x r also within a factor of
# _SPLIT_BREAK_DOLLAR_MAX of 1. And no step-day spike: an unadjusted split
# carries its new volume from day 0, a crash spikes on days 0-2, so the
# largest of the first _SPLIT_VOLUME_SKIP_BARS volumes must be at most
# _SPLIT_SPIKE_MAX times the after-median (missing there: not corroborated).
# A crash whose dollar volume holds steady with no early spike still looks
# like a split: the tests narrow that case, they cannot remove it. A side with fewer than _SPLIT_VOLUME_MIN_BARS
# bars of positive volume is not corroborated: no flag, the gate as before.
# _entry_gate does not block on a split-suspect read.
_SPLIT_FACTORS = (2, 3, 4, 5, 10)
_SPLIT_RATIO_TOL = 0.02
_SPLIT_SCALE_BREAK = 3.0
_SPLIT_MOVE_MULT = 5.0
_SPLIT_WINDOW_BARS = 200
_SPLIT_VOLUME_BARS = 20
_SPLIT_VOLUME_SKIP_BARS = 3
_SPLIT_VOLUME_MIN_BARS = 5
_SPLIT_VOLUME_TOL = 0.25
_SPLIT_VOLUME_BREAK_MIN = 2.5
_SPLIT_BREAK_DOLLAR_MAX = 2.0
_SPLIT_SPIKE_MAX = 2.0


def _entry_opinion_trigger(result: dict) -> dict:
    """What started the analysis behind this verdict, from the trigger its
    provenance already carries (analyst.trigger_record plus the screener's
    new_fact on the news path). new_fact is the screener's flag on the item
    (None when not a bool: not judged, or not the news path); new_fact_false
    is True only when the screener said False. Never raises."""
    out = {"trigger_present": False, "path": None, "source": None, "type": None,
           "new_fact": None, "new_fact_false": None, "screener_fallback_reason": None}
    try:
        prov = result.get("provenance")
        trig = prov.get("trigger") if isinstance(prov, dict) else None
        if not isinstance(trig, dict):
            return out
        nf = trig.get("new_fact")
        nf = nf if isinstance(nf, bool) else None
        out.update(trigger_present=True, path=trig.get("path"), source=trig.get("source"),
                   type=trig.get("type"), new_fact=nf,
                   new_fact_false=(nf is False) if nf is not None else None,
                   screener_fallback_reason=trig.get("screener_fallback_reason"))
    except Exception as e:
        out["error"] = type(e).__name__
    return out


def _split_discontinuity(dates: list, closes: list, volumes: list | None = None,
                         window: int = _SPLIT_WINDOW_BARS) -> dict:
    """Whether the last `window` closes (the SMA200's bars) look like two
    price scales mixed by an unadjusted split. dates, closes and volumes are
    the full aligned bar lists, oldest first; volume windows may reach before
    the price window. For each one-day ratio r = close / previous close
    inside the window (the step into the window is left out: it doesn't mix
    the average):
      candidate   |r / f - 1| <= _SPLIT_RATIO_TOL for f in _SPLIT_FACTORS or
                  1/f (rule "factor"), else r <= 1/_SPLIT_SCALE_BREAK or
                  r >= _SPLIT_SCALE_BREAK (rule "scale_break")
      move        |ln r| >= _SPLIT_MOVE_MULT x the median |ln r| of the
                  window's other days (a flat window passes any candidate)
      volume      v = median volume of the _SPLIT_VOLUME_BARS bars after the
                  step, skipping its first _SPLIT_VOLUME_SKIP_BARS, over the
                  median of the _SPLIT_VOLUME_BARS bars before it (positive
                  volumes only; under _SPLIT_VOLUME_MIN_BARS on a side, or no
                  volumes, is not corroborated). factor: |ln(v x r)| <=
                  ln(1 + _SPLIT_VOLUME_TOL); scale_break: v >= _SPLIT_VOLUME_BREAK_MIN
                  when r < 1, v <= 1/_SPLIT_VOLUME_BREAK_MIN when r > 1, and
                  |ln(v x r)| <= ln _SPLIT_BREAK_DOLLAR_MAX
      no spike    max volume of the step day and the next
                  _SPLIT_VOLUME_SKIP_BARS - 1 days <= _SPLIT_SPIKE_MAX x the
                  after-median (each of those volumes present and positive)
    A day is flagged when all four hold. A non-positive close gives no ratio.
    A split in the last _SPLIT_VOLUME_SKIP_BARS + _SPLIT_VOLUME_MIN_BARS - 1
    sessions (1-7) can't be corroborated yet and is not flagged: the gate
    still blocks on those bars, the safe direction. The real SNDK 2025-11-21
    step fails the spike rule (day 0 at 2.31x the after-median), and the
    nightly yfinance refresh leaves its split steps a year back, outside this
    window (ingestion/price_history.py rescales those): the guard targets
    unadjusted or fallback feeds.

    Returns split_suspect (True/False; None with under 2 closes or on an
    error, with split_error), split_suspect_count, split_candidates (days
    passing candidate and move), split_uncorroborated (of those, failing a
    volume test) with split_uncorroborated_reasons (no_volume, volume_step,
    dollar_volume, spike: counts), and for the flagged day with the largest move:
    split_suspect_date (the day of the new scale), split_suspect_ratio,
    split_suspect_rule, split_suspect_factor ("1/2" for a halving, "2" for a
    doubling; None on a scale break), split_suspect_volume_ratio,
    split_suspect_spike_ratio (step-window max over the after-median) and
    split_suspect_move_vs_typical (|ln r| / median); split_window_bars is the
    number of closes checked. Record only: the averages are computed as
    before. Never raises."""
    import math
    import statistics
    out = {"split_suspect": None, "split_suspect_count": None, "split_candidates": None,
           "split_uncorroborated": None, "split_suspect_date": None, "split_suspect_ratio": None,
           "split_suspect_rule": None, "split_suspect_factor": None,
           "split_suspect_volume_ratio": None, "split_suspect_spike_ratio": None,
           "split_suspect_move_vs_typical": None, "split_uncorroborated_reasons": None,
           "split_window_bars": None}
    try:
        cl_all, ds_all = list(closes), list(dates)
        vol_all = list(volumes) if volumes is not None else None
        start = max(0, len(cl_all) - window)
        cl = cl_all[start:]
        out["split_window_bars"] = len(cl)
        if len(cl) < 2:
            return out
        ratios = [(cl[i] / cl[i - 1]) if cl[i - 1] > 0 and cl[i] > 0 else None
                  for i in range(1, len(cl))]
        moves = [abs(math.log(r)) if r else None for r in ratios]

        def vmed(lo, hi):
            if vol_all is None:
                return None
            vs = []
            for v in vol_all[max(0, lo):max(0, hi)]:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f) and f > 0:
                    vs.append(f)
            return statistics.median(vs) if len(vs) >= _SPLIT_VOLUME_MIN_BARS else None

        hits, candidates, why = [], 0, {}
        for k, r in enumerate(ratios):
            if not r:
                continue
            factor = next((label for f in _SPLIT_FACTORS
                           for target, label in ((1 / f, f"1/{f}"), (f, str(f)))
                           if abs(r / target - 1) <= _SPLIT_RATIO_TOL), None)
            rule = "factor" if factor else None
            if factor is None and (r <= 1 / _SPLIT_SCALE_BREAK or r >= _SPLIT_SCALE_BREAK):
                rule = "scale_break"
            if rule is None:
                continue
            others = [m for j, m in enumerate(moves) if j != k and m is not None]
            typical = statistics.median(others) if others else 0.0
            vs_typ = moves[k] / typical if typical > 0 else math.inf
            if vs_typ < _SPLIT_MOVE_MULT:
                continue
            candidates += 1
            g = start + k + 1                      # first bar on the new scale, full-list index
            before = vmed(g - _SPLIT_VOLUME_BARS, g)
            after = vmed(g + _SPLIT_VOLUME_SKIP_BARS, g + _SPLIT_VOLUME_SKIP_BARS + _SPLIT_VOLUME_BARS)
            v = after / before if before and after else None
            step_vols = []
            for x in (vol_all or [])[g:g + _SPLIT_VOLUME_SKIP_BARS]:
                try:
                    f = float(x)
                except (TypeError, ValueError):
                    f = 0.0
                step_vols.append(f if math.isfinite(f) else 0.0)
            if (v is None or len(step_vols) < _SPLIT_VOLUME_SKIP_BARS
                    or min(step_vols) <= 0):
                why["no_volume"] = why.get("no_volume", 0) + 1
                continue
            if rule == "factor":
                if abs(math.log(v * r)) > math.log(1 + _SPLIT_VOLUME_TOL):
                    why["volume_step"] = why.get("volume_step", 0) + 1
                    continue
            else:
                if not (v >= _SPLIT_VOLUME_BREAK_MIN if r < 1 else v <= 1 / _SPLIT_VOLUME_BREAK_MIN):
                    why["volume_step"] = why.get("volume_step", 0) + 1
                    continue
                if abs(math.log(v * r)) > math.log(_SPLIT_BREAK_DOLLAR_MAX):
                    why["dollar_volume"] = why.get("dollar_volume", 0) + 1
                    continue
            spike = max(step_vols) / after
            if spike > _SPLIT_SPIKE_MAX:
                why["spike"] = why.get("spike", 0) + 1
                continue
            hits.append((moves[k], str(ds_all[g])[:10], r, rule, factor, v, spike, vs_typ))
        out["split_candidates"] = candidates
        out["split_uncorroborated"] = candidates - len(hits)
        out["split_uncorroborated_reasons"] = why
        out["split_suspect"] = bool(hits)
        out["split_suspect_count"] = len(hits)
        if hits:
            _, day, r, rule, factor, v, spike, vs_typ = max(hits, key=lambda h: h[0])
            fin = lambda x, n: round(x, n) if isinstance(x, (int, float)) and math.isfinite(x) else None
            out.update(split_suspect_date=day, split_suspect_ratio=fin(r, 4),
                       split_suspect_rule=rule, split_suspect_factor=factor,
                       split_suspect_volume_ratio=fin(v, 2), split_suspect_spike_ratio=fin(spike, 2),
                       split_suspect_move_vs_typical=fin(vs_typ, 1))
    except Exception as e:
        out["split_suspect"] = None
        out["split_error"] = type(e).__name__
    return out


def _entry_trend_check(ticker: str, now=None) -> dict:
    """Price structure on completed daily bars for _entry_quality_shadow. One
    read of the nightly price_history cache (db.get_price_history, a database
    read: nothing is fetched from a data vendor or the broker, and an empty
    cache records read "no_cached_bars"). Today's unfinished bar is dropped
    as the regime code drops it (market_regime.completed_bars); non-finite
    bars are carried forward as the trade ladder does (_sanitize_bars), and
    an inf it leaves is dropped (nonfinite_dropped). bar_age_days is the
    calendar age of the last bar.

    Records the last close against the 50- and 200-day simple averages of the
    last 50 and 200 completed closes (None, with a reason, under that many
    bars); the distance below the 20-day high in percent and in ATR14 units
    (trade_ladder._compute_atr14); and, on the last 60 bars, the brief's swing
    structure label (price_action._swing_structure) with whether the last
    swing high is under the one before it, from the same fractal swings
    collapsed the same way. read is "ok" with 200+ bars, "insufficient_bars"
    under that, "no_cached_bars" or "error: <class>". The split_suspect
    keys (_split_discontinuity, on the last 200 closes) say whether those
    bars look like two price scales mixed by an unadjusted split; the
    averages are computed either way and _entry_gate does not block on a
    suspect read. Never raises."""
    rec = {"read": None, "source": "price_history", "rows_read": None, "bars": 0,
           "partial_bar_dropped": None, "nonfinite_bars": None, "last_bar_date": None,
           "last_close": None}
    for n in (50, 200):
        rec.update({f"sma{n}": None, f"above_sma{n}": None, f"below_sma{n}": None,
                    f"pct_vs_sma{n}": None, f"sma{n}_reason": None})
    rec.update({"atr14": None, "high_20d": None, "high_20d_date": None,
                "drawdown_from_20d_high_pct": None, "drawdown_from_20d_high_atr": None,
                "drawdown_reason": None, "swing_bars": _ENTRY_QUALITY_SWING_BARS,
                "swing_structure": None, "swing_highs": None, "last_swing_high_lower": None,
                "swing_reason": None})
    rec.update(_split_discontinuity([], []))   # the split_* keys
    rec["split_window_bars"] = None             # set when bars are read
    try:
        import math
        import db
        try:
            raw = db.get_price_history(ticker, days=_ENTRY_QUALITY_BAR_ROWS, strict=True)
        except Exception as e:
            rec["read"] = f"error: {type(e).__name__}"
            return rec
        raw = list(raw or [])
        rec["rows_read"] = len(raw)
        if not raw:
            rec["read"] = "no_cached_bars"
            return rec
        from agents.market_regime import completed_bars
        from agents.trade_ladder import _compute_atr14, _sanitize_bars
        bars = completed_bars(raw, now=now)
        rec["partial_bar_dropped"] = len(bars) < len(raw)

        def finite(v):
            return isinstance(v, (int, float)) and math.isfinite(v)
        rec["nonfinite_bars"] = sum(1 for b in bars if not all(
            finite(b.get(k)) for k in ("open", "high", "low", "close")))
        bars = _sanitize_bars(bars)
        # _sanitize_bars carries only NaN/None forward; an inf left over is
        # dropped here so no number or label is computed from it.
        kept = [b for b in bars if all(finite(b.get(k)) for k in ("open", "high", "low", "close"))]
        rec["nonfinite_dropped"] = len(bars) - len(kept)
        bars = kept
        rec["bars"] = len(bars)
        if not bars:
            rec["read"] = "no_cached_bars"
            return rec
        rec["last_bar_date"] = str(bars[-1].get("date"))
        # Calendar days from the last bar to today (ET): a name whose nightly
        # refresh failed shows its age here, as read stays "ok".
        try:
            from datetime import date as _date
            from zoneinfo import ZoneInfo
            today = (now or datetime.now(ZoneInfo("America/New_York")))
            today = today.astimezone(ZoneInfo("America/New_York")).date() if today.tzinfo else today.date()
            rec["bar_age_days"] = (today - _date.fromisoformat(str(bars[-1].get("date"))[:10])).days
        except Exception:
            rec["bar_age_days"] = None
        closes = [float(b["close"]) for b in bars]
        last = closes[-1]
        rec["last_close"] = round(last, 4)
        rec.update(_split_discontinuity([b.get("date") for b in bars], closes,
                                        [b.get("volume") for b in bars]))
        for n in (50, 200):
            if len(closes) < n:
                rec[f"sma{n}_reason"] = f"{len(closes)} completed bars, {n} needed"
                continue
            sma = sum(closes[-n:]) / n
            rec[f"sma{n}"] = round(sma, 4)
            rec[f"above_sma{n}"] = last > sma
            rec[f"below_sma{n}"] = last < sma
            rec[f"pct_vs_sma{n}"] = round((last / sma - 1) * 100, 2) if sma else None
        if len(bars) < _ENTRY_QUALITY_HIGH_DAYS:
            rec["drawdown_reason"] = (f"{len(bars)} completed bars, "
                                      f"{_ENTRY_QUALITY_HIGH_DAYS} needed")
        else:
            top = max(bars[-_ENTRY_QUALITY_HIGH_DAYS:], key=lambda b: b["high"])
            hi = float(top["high"])
            rec["high_20d"], rec["high_20d_date"] = round(hi, 4), str(top.get("date"))
            rec["drawdown_from_20d_high_pct"] = round((last / hi - 1) * 100, 2) if hi else None
            atr = _compute_atr14(bars)
            if atr and atr > 0:
                rec["atr14"] = round(atr, 4)
                # Unrounded, so a threshold read off it matches the label.
                rec["drawdown_from_20d_high_atr"] = (hi - last) / atr
            else:
                rec["drawdown_reason"] = "no ATR14"
        if len(bars) < _ENTRY_QUALITY_SWING_BARS:
            rec["swing_reason"] = (f"{len(bars)} completed bars, "
                                   f"{_ENTRY_QUALITY_SWING_BARS} needed")
        else:
            from agents.price_action import _find_swings, _swing_structure
            window = bars[-_ENTRY_QUALITY_SWING_BARS:]
            rec["swing_structure"] = _swing_structure(window)[0]
            swings = _find_swings(window)
            # Collapsed as price_action._swing_structure and
            # trade_ladder.compute_base_contraction collapse them.
            collapsed = []
            for s in swings:
                if collapsed and collapsed[-1][2] == s[2]:
                    if (s[2] == "high" and s[1] > collapsed[-1][1]) or \
                            (s[2] == "low" and s[1] < collapsed[-1][1]):
                        collapsed[-1] = s
                else:
                    collapsed.append(s)
            highs = [s[1] for s in collapsed if s[2] == "high"][-2:]
            if len(swings) < 4 or len(highs) < 2:
                rec["swing_reason"] = "fewer than 4 swings or 2 swing highs"
            else:
                rec["swing_highs"] = [round(float(h), 4) for h in highs]
                rec["last_swing_high_lower"] = highs[1] < highs[0]
        rec["read"] = "ok" if len(bars) >= 200 else "insufficient_bars"
    except Exception as e:
        rec["read"] = f"error: {type(e).__name__}"
    return rec


def _entry_quality_shadow(ticker: str, result: dict, now=None) -> dict | None:
    """Entry-quality shadow (config ENTRY_QUALITY_SHADOW_ENABLED). Called for
    an OPEN/ADD once it has passed the rebuy cooldown, right after
    _stopout_reentry_shadow. Writes one ENTRY_QUALITY_SHADOW row:
    opinion_trigger (_entry_opinion_trigger), trend_check (_entry_trend_check,
    one cached-bar read) and would_block_trend, fixed illustrative labels
    (below_200dma, below_50dma, drawdown_gt_3atr; None when the number is
    missing) that trend_check's raw numbers let anyone rescore. The row
    itself changes nothing and result is not changed; it returns trend_check
    (None when switched off or on a raise) for _entry_gate, which may skip
    the buy and writes its own ENTRY_GATE_BLOCKED row. Never raises."""
    try:
        try:
            from config import ENTRY_QUALITY_SHADOW_ENABLED
        except ImportError:
            return
        if not ENTRY_QUALITY_SHADOW_ENABLED or result.get("action") not in ("OPEN", "ADD"):
            return None
        import db
        kind = "ENTRY_QUALITY_SHADOW"
        trig = _entry_opinion_trigger(result)
        trend = _entry_trend_check(ticker, now=now)

        dd = trend.get("drawdown_from_20d_high_atr")
        would = {"below_200dma": trend.get("below_sma200"), "below_50dma": trend.get("below_sma50"),
                 "drawdown_gt_3atr": (dd > _ENTRY_QUALITY_DRAWDOWN_ATR)
                 if isinstance(dd, (int, float)) else None}
        reason = (f"{result.get('action')}: trend read {trend.get('read')}; "
                  + ", ".join(f"{k}={v}" for k, v in would.items())
                  + f"; new_fact={trig.get('new_fact')}; record only")
        # route_exit stays: it marks the row's provenance as a copy of the
        # verdict's (db._VERDICT_COPY_KEY), as on SHADOW_SWAP rows, so readers
        # that count provenance or LLM calls skip it. The buy goes on.
        fields = _route_row_fields(result, kind)
        fields["decision"]["shadow"] = kind
        fields["decision"]["opinion_trigger"] = trig
        fields["decision"]["trend_check"] = trend
        fields["decision"]["would_block_trend"] = would
        try:
            db.log_signal(ticker, kind, 0.0, reason[:500], **fields)
        except Exception as e:
            log.warning(f"[{ticker}] entry-quality shadow row not written: {type(e).__name__}: {e}")
        return trend
    except Exception as e:
        log.warning(f"[{ticker}] entry-quality shadow failed: {type(e).__name__}: {e}")
        return None


_NON_BUY_VERDICTS = ("HOLD", "WATCH", "PASS", "TRIM", "EXIT", "SELL")
# The earlier verdicts whose non-buy would_block counts: a stale-entry recheck
# of this very decision and a held-position review.
_RECHECK_BLOCK_SOURCES = ("stale_entry_recheck", "held_review")
# A prior verdict stamped up to this far after now still counts (clock skew).
_RECHECK_CLOCK_SKEW_HOURS = 0.05


def _verdict_source_class(path, source) -> str:
    """What started a verdict, from its provenance trigger: stale_entry_recheck
    (source, on the event path), plan_level (path plan_level_review or source
    plan_watcher), held_review (scheduled_review, earnings_review), event,
    daily_scan or chat; unknown without a trigger (rows before 10-29), else
    the path itself. plan_level_review is a held review in
    _HELD_REVIEW_PATHS, but is classed plan_level here and so is not in
    _RECHECK_BLOCK_SOURCES: a definition choice that is Luke's to change.
    The raw trigger_path is recorded beside it."""
    if source == "stale_entry_recheck":
        return "stale_entry_recheck"
    if path == "plan_level_review" or source == "plan_watcher":
        return "plan_level"
    if path in ("scheduled_review", "earnings_review"):
        return "held_review"
    if path in ("event", "daily_scan", "chat"):
        return path
    return str(path) if path else "unknown"


def _recheck_contradiction_record(ticker: str, result: dict, now=None) -> dict | None:
    """Record only (config RECHECK_CONTRADICTION_SHADOW_ENABLED): the most
    recent non-buy verdict (_NON_BUY_VERDICTS) on ticker in this env within
    RECHECK_CONTRADICTION_WINDOW_HOURS before an OPEN/ADD, from one
    db.get_recent_verdicts read: prior {signal_id, kind, model_action (the
    model's own action before any coercion, e.g. an OPEN the analyst demoted to
    WATCH), source (_verdict_source_class), trigger_path, trigger_source,
    age_hours, conviction, is_recheck, at}, or None with none in the window
    (verdicts up to _RECHECK_CLOCK_SKEW_HOURS after now count, at age 0 or a
    small negative age, for clock skew between writers);
    n_non_buy_in_window; the current verdict's trigger path and new_fact (the
    screener's flag on an event verdict, else None); would_block, True when
    prior's source is in _RECHECK_BLOCK_SOURCES and the current new_fact is
    not True. A failed read: read "error: ...", prior and would_block None.
    None for anything but OPEN/ADD or with the record off. Nothing reads it to
    decide. Never raises."""
    try:
        if result.get("action") not in ("OPEN", "ADD"):
            return None
        import config
        if not getattr(config, "RECHECK_CONTRADICTION_SHADOW_ENABLED", False):
            return None
        window = float(config.RECHECK_CONTRADICTION_WINDOW_HOURS)
    except Exception:
        return None
    rec = {"window_hours": window, "read": None, "prior": None, "n_non_buy_in_window": None,
           "current_trigger_path": None, "current_new_fact": None, "would_block": None}
    try:
        import math
        from datetime import datetime, timezone
        prov = result.get("provenance")
        trig = (prov.get("trigger") if isinstance(prov, dict) else None) or {}
        trig = trig if isinstance(trig, dict) else {}
        rec["current_trigger_path"] = trig.get("path")
        nf = trig.get("new_fact") if trig.get("path") == "event" else None
        rec["current_new_fact"] = nf if isinstance(nf, bool) else None
        import db
        rows = db.get_recent_verdicts(ticker, hours=max(1, math.ceil(window)), limit=50,
                                      strict=True, with_trigger=True)
        now = now or datetime.now(timezone.utc)
        found = []
        for r in rows or []:
            if r.get("signal") not in _NON_BUY_VERDICTS:
                continue
            ts = r.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            age = (now - ts).total_seconds() / 3600
            if -_RECHECK_CLOCK_SKEW_HOURS <= age <= window:
                found.append((age, ts, r))
        rec.update(read="ok", n_non_buy_in_window=len(found), would_block=False)
        if found:
            age, ts, r = min(found, key=lambda f: f[0])
            src = _verdict_source_class(r.get("trigger_path"), r.get("trigger_source"))
            rec["prior"] = {"signal_id": r.get("signal_id"), "kind": r.get("signal"),
                            "model_action": r.get("model_action"), "source": src,
                            "trigger_path": r.get("trigger_path"), "trigger_source": r.get("trigger_source"),
                            "age_hours": round(age, 2), "conviction": r.get("conviction"),
                            "is_recheck": src == "stale_entry_recheck", "at": ts.isoformat()}
            rec["would_block"] = src in _RECHECK_BLOCK_SOURCES and rec["current_new_fact"] is not True
    except Exception as e:
        rec.update(read=f"error: {type(e).__name__}", prior=None, n_non_buy_in_window=None,
                   would_block=None)
    return rec


def _same_day_consensus_record(ticker: str, result: dict, now=None) -> dict | None:
    """Record only (config SAME_DAY_CONSENSUS_RECORD_ENABLED): the ticker's
    earlier analyst verdicts in this env before this OPEN/ADD, from one
    bounded db.get_same_day_verdicts read of the last 24 h (the current
    verdict, by provenance.signal_id, excluded; scripts/fire_test.py verdicts
    are counted, as their analyst rows carry no marker), summed over two
    windows that both end at this route:
    since ET midnight: n_earlier; n_buy_grade, the verdicts whose stored
    final_pass_to_risk is True on an OPEN/ADD (the analyst's own pass after
    every gate, as decided when each was written; not recomputed here),
    buy_grade_share (n_buy_grade / n_earlier, None with none) and
    n_buy_grade_unknown (rows without the stored flag, counted as not
    buy-grade); n_model_buy, the model's own OPEN/ADD with its own
    pass_to_risk; median_conviction (n_conviction verdicts with a stored
    confidence) and median_udr (n_udr verdicts with a finite UDR);
    n_story_keys and n_distinct_stories (story_dedup story_key, None when no
    row has one); window {start (ET midnight), basis}; limit_hit (the ET-day
    rows read, the current verdict included, reached
    SAME_DAY_CONSENSUS_LIMIT: an ET-day read of its own would have been cut);
    the last 24 h: n_earlier_24h, n_buy_grade_24h, buy_grade_share_24h (same
    definitions), window_24h {hours, basis} (it starts 24 h before the routing
    row's own time; a stored start would be a wall-clock value) and
    limit_hit_24h (the read itself reached the limit, so its oldest 24 h rows
    may be cut).
    The read starts at the earlier of the two starts (ET midnight on a 25 h
    fall-back day) and returns newest first, so the ET-day rows are its
    newest. No wall-clock value is stored: the record depends only on the
    verdicts read. read "ok", "skipped_budget" (the read hit
    SAME_DAY_CONSENSUS_READ_TIMEOUT_MS) or "error: <class>".
    None for anything but OPEN/ADD or with the record off. Nothing reads it
    to decide. Never raises."""
    try:
        if result.get("action") not in ("OPEN", "ADD"):
            return None
        import config
        if not getattr(config, "SAME_DAY_CONSENSUS_RECORD_ENABLED", False):
            return None
        timeout_ms = int(config.SAME_DAY_CONSENSUS_READ_TIMEOUT_MS)
        limit = int(config.SAME_DAY_CONSENSUS_LIMIT)
    except Exception:
        return None
    from datetime import datetime, time as dtime, timedelta, timezone
    from zoneinfo import ZoneInfo
    rec = {"window": None, "read": None, "limit_hit": None,
           "current_signal_id": None, "n_earlier": None, "n_buy_grade": None,
           "buy_grade_share": None, "n_buy_grade_unknown": None, "n_model_buy": None,
           "median_conviction": None, "n_conviction": None, "median_udr": None,
           "n_udr": None, "n_story_keys": None, "n_distinct_stories": None,
           "window_24h": None, "limit_hit_24h": None, "n_earlier_24h": None,
           "n_buy_grade_24h": None, "buy_grade_share_24h": None,
           "buy_grade_definition": "stored final_pass_to_risk on an OPEN/ADD verdict"}
    try:
        now = now or datetime.now(timezone.utc)
        et = ZoneInfo("America/New_York")
        start = datetime.combine(now.astimezone(et).date(), dtime(0), et)
        start_24h = now - timedelta(hours=24)
        rec["window"] = {"start": start.isoformat(), "basis": "same ET day, before this route"}
        rec["window_24h"] = {"hours": 24, "basis": "the 24 h before this route"}
        prov = result.get("provenance")
        sid = prov.get("signal_id") if isinstance(prov, dict) else None
        sid = sid if isinstance(sid, int) and not isinstance(sid, bool) else None
        rec["current_signal_id"] = sid
        import db
        try:
            rows = db.get_same_day_verdicts(ticker, min(start, start_24h), now, limit=limit,
                                            timeout_ms=timeout_ms)
        except Exception as e:
            canceled = type(e).__name__ == "QueryCanceled" or "statement timeout" in str(e)
            rec["read"] = "skipped_budget" if canceled else f"error: {type(e).__name__}"
            return rec

        def since(r, t0):
            ts = r.get("timestamp")
            if not isinstance(ts, datetime):
                return False                    # undated: in neither window
            return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)) >= t0
        dicts = [r for r in rows or [] if isinstance(r, dict)]
        raw_day = [r for r in dicts if since(r, start)]
        raw_24h = [r for r in dicts if since(r, start_24h)]
        day = [r for r in raw_day if sid is None or r.get("signal_id") != sid]
        last_24h = [r for r in raw_24h if sid is None or r.get("signal_id") != sid]
        import math
        import statistics

        def n_buy(rs):
            return sum(1 for r in rs if r.get("signal") in ("OPEN", "ADD")
                       and r.get("final_pass_to_risk") is True)
        buy, buy_24h = n_buy(day), n_buy(last_24h)
        convs = [r["conviction"] for r in day if isinstance(r.get("conviction"), (int, float))
                 and not isinstance(r.get("conviction"), bool)]
        udrs = [r["udr"] for r in day if isinstance(r.get("udr"), (int, float))
                and not isinstance(r.get("udr"), bool) and math.isfinite(r["udr"])]
        keys = [r["story_key"] for r in day if r.get("story_key")]
        rec.update(
            read="ok", n_earlier=len(day), n_buy_grade=buy,
            buy_grade_share=round(buy / len(day), 4) if day else None,
            n_buy_grade_unknown=sum(1 for r in day if r.get("final_pass_to_risk") is None),
            n_model_buy=sum(1 for r in day if r.get("model_action") in ("OPEN", "ADD")
                            and r.get("model_pass_to_risk") is True),
            median_conviction=statistics.median(convs) if convs else None,
            n_conviction=len(convs),
            median_udr=round(statistics.median(udrs), 3) if udrs else None,
            n_udr=len(udrs), n_story_keys=len(keys),
            n_distinct_stories=len(set(keys)) if keys else None,
            limit_hit=len(raw_day) >= limit,
            n_earlier_24h=len(last_24h), n_buy_grade_24h=buy_24h,
            buy_grade_share_24h=round(buy_24h / len(last_24h), 4) if last_24h else None,
            limit_hit_24h=len(raw_24h) >= limit)
    except Exception as e:
        rec.update(read=f"error: {type(e).__name__}", n_earlier=None, n_buy_grade=None,
                   buy_grade_share=None, n_earlier_24h=None, n_buy_grade_24h=None,
                   buy_grade_share_24h=None)
    return rec


_ADD_LOSER_MODES = ("shadow", "enforce", "off")


def _add_loser_mode() -> str:
    """config.ADD_LOSER_GATE_MODE: "shadow" (record, never block), "enforce"
    or "off" (no read). Any other value reads as "shadow", with a warning, so
    a typo can neither block buys nor drop the record."""
    try:
        import config
        mode = getattr(config, "ADD_LOSER_GATE_MODE", "shadow")
    except Exception:
        return "shadow"
    if mode in _ADD_LOSER_MODES:
        return mode
    log.warning(f"ADD_LOSER_GATE_MODE={mode!r} is not one of {_ADD_LOSER_MODES}; read as 'shadow'")
    return "shadow"


def _add_loser_reason(rec: dict) -> str:
    """The reason text for an add-to-loser block or would-block, from its
    record: where the price sits against the average cost and why the
    trigger is not a new fact."""
    pct = rec.get("pct_vs_avg_cost")
    where = ("at the average cost" if isinstance(pct, (int, float)) and round(pct, 1) == 0
             else f"{pct}% vs the average cost")
    src = rec.get("trigger_new_fact_source")
    if src == "none_on_path":
        why = f"no news item behind the verdict (a {rec.get('trigger_path') or 'untriggered'} verdict)"
    elif src == "no_news_item":
        why = ("a price-only drop-watch review behind the verdict, which has no news item "
               "(DIP_REVIEW_COUNTS_AS_NEW_FACT is off)")
    elif rec.get("trigger_new_fact") is False:
        why = "the triggering item not a new fact"
    else:
        why = "no new fact on the triggering item"
    return f"ADD {where} with {why} (never add to a loser without a new catalyst)"


def _add_loser_check(ticker: str, result: dict, client) -> dict | None:
    """What the add-to-loser gate (config ADD_LOSER_GATE_MODE) reads for an
    ADD: the broker's average cost for the position (client.get_all_positions),
    the current price (_get_price, else the position's own current price), the
    price's percent vs that cost, and the triggering item's new_fact, with
    trigger_new_fact_source saying where it came from:
      "event_provenance"   the screener's judged flag on an event verdict;
      "unjudged"           nothing judged it: a keyword-fallback screen
                           (screener_fallback_reason set), a missing flag, or a
                           stale-entry recheck (source stale_entry_recheck);
                           read like a failed read, so it never blocks;
      "none_on_path"       not an event verdict (a held review, the daily scan):
                           no news item, None;
      "no_news_item"       a drop-watch review (source "dip_watch"), price-only,
                           with DIP_REVIEW_COUNTS_AS_NEW_FACT off: None;
      "dip_review_counts_as_new_fact"   the same with that switch on: True, as
                           new_fact_gate treats it.
    would_block is True only when both reads worked, the price is more than
    ADD_LOSER_GATE_MIN_PCT percent below the cost, the flag was judged (or
    there is no news item) and it is not True. blocked = would_block in mode
    "enforce", else False. Also recorded: mode, screener_fallback_reason, and
    earnings_phase on an earnings_review (pre / day / post; not exempted). A
    failed read, a name not held or no price: would_block False, with the read
    field saying why. None for anything but an ADD, or in mode "off".
    Percent vs cost only: no prices or quantities. Never raises."""
    import math
    try:
        if result.get("action") != "ADD":
            return None
        mode = _add_loser_mode()
        if mode == "off":
            return None
        import config
        min_pct = float(getattr(config, "ADD_LOSER_GATE_MIN_PCT", 0.0))
    except Exception:
        return None
    rec = {"mode": mode, "min_pct": min_pct, "avg_cost_read": None, "price_read": None,
           "pct_vs_avg_cost": None, "trigger_path": None, "trigger_source": None,
           "trigger_new_fact": None, "trigger_new_fact_source": None,
           "screener_fallback_reason": None, "earnings_phase": None,
           "would_block": False, "blocked": False}
    try:
        prov = result.get("provenance")
        trig = (prov.get("trigger") if isinstance(prov, dict) else None) or {}
        trig = trig if isinstance(trig, dict) else {}
        path, source = trig.get("path"), trig.get("source")
        rec["trigger_path"], rec["trigger_source"] = path, source
        rec["screener_fallback_reason"] = trig.get("screener_fallback_reason")
        if path == "earnings_review":
            rec["earnings_phase"] = trig.get("earnings_phase")
        item = trig.get("new_fact")
        if path != "event":
            rec["trigger_new_fact"], rec["trigger_new_fact_source"] = None, "none_on_path"
        elif source == "dip_watch":
            # A drop-watch review (path "event", source "dip_watch") has no
            # news item; Luke's buy-gate switch lets the drop count as the
            # new fact, as in new_fact_gate, so this gate follows it.
            if getattr(config, "DIP_REVIEW_COUNTS_AS_NEW_FACT", False):
                rec["trigger_new_fact"], rec["trigger_new_fact_source"] = True, "dip_review_counts_as_new_fact"
            else:
                rec["trigger_new_fact"], rec["trigger_new_fact_source"] = None, "no_news_item"
        elif source == "stale_entry_recheck" or rec["screener_fallback_reason"] or not isinstance(item, bool):
            rec["trigger_new_fact"] = item if isinstance(item, bool) else None
            rec["trigger_new_fact_source"] = "unjudged"
        else:
            rec["trigger_new_fact"], rec["trigger_new_fact_source"] = item, "event_provenance"
        if client is None:
            rec["avg_cost_read"] = "no_client"
            return rec
        try:
            pos = next((p for p in client.get_all_positions() if p.symbol == ticker), None)
        except Exception as e:
            rec["avg_cost_read"] = f"error: {type(e).__name__}"
            return rec
        try:
            avg = float(pos.avg_entry_price) if pos is not None and pos.avg_entry_price else None
        except (TypeError, ValueError):
            avg = None
        if not avg or avg <= 0 or not math.isfinite(avg):
            rec["avg_cost_read"] = "not_held" if pos is None else "no_avg_cost"
            return rec
        rec["avg_cost_read"] = "ok"
        price, price_source = None, None
        try:
            price, price_source = float(_get_price(ticker, client) or 0) or None, "quote"
        except Exception:
            price = None
        if not price:
            try:
                price, price_source = float(pos.current_price or 0) or None, "position"
            except (TypeError, ValueError, AttributeError):
                price = None
        if not price or price <= 0 or not math.isfinite(price):
            rec["price_read"] = "none"
            return rec
        rec["price_read"] = price_source
        pct = (price - avg) / avg * 100
        rec["pct_vs_avg_cost"] = round(pct, 2)
        rec["would_block"] = bool(pct < -min_pct and rec["trigger_new_fact_source"] != "unjudged"
                                  and rec["trigger_new_fact"] is not True)
        rec["blocked"] = rec["would_block"] and mode == "enforce"
        return rec
    except Exception as e:
        log.warning(f"[{ticker}] add-to-loser check failed ({type(e).__name__}: {e}); not applied")
        rec.update(avg_cost_read=rec.get("avg_cost_read") or f"error: {type(e).__name__}",
                   would_block=False, blocked=False)
        return rec


def _entry_gate(ticker: str, result: dict, stop_rec: dict | None, trend: dict | None,
                path: str | None = None, extra: dict | None = None,
                add_loser: dict | None = None,
                row_extra: dict | None = None) -> bool:
    """Live entry gates (config STOPOUT_REENTRY_GATE_DAYS,
    ENTRY_TREND_GATE_ENABLED) for an OPEN/ADD that has passed the rebuy
    cooldown, on what the two shadows just read (stop_rec from
    _stopout_reentry_shadow, trend from _entry_quality_shadow). Blocks when
    (a) the newest exit the bot did not place is under STOPOUT_REENTRY_GATE_DAYS
    days old, or (b) the last completed close is below both its 50- and
    200-day simple averages on bars at most ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS
    old while ENTRY_TREND_GATE_MARKET_TICKER's last close is below its own
    200-day average (read only for a stock that would be blocked, split-
    suspect or not; a failed, stale or split-suspect market read does not
    block, and a raising one keeps a stop-out block). A failed read, missing
    averages, stale bars or split-suspect bars (_split_discontinuity) do not
    block. gate["trend"]["split_fail_open"] is True when the stock's split
    suspicion withheld a block (it was below both averages and the market
    was below its 200-day on fresh bars, split-suspect or not),
    market["split_fail_open"] when only the market's suspicion did;
    with no other block, either writes one entry_gate_split_fail_open
    activity row (the buy goes on as before). Conviction does not override. On a block writes one
    ENTRY_GATE_BLOCKED row (decision.entry_gate) and returns True; the caller
    returns before sizing. Never raises: an internal error returns False.

    path, when given, is recorded as entry_gate.path with extra's keys beside
    it. path "deferred_retry" (tasks._deferred_entry_gate) applies the
    stop-out half only: the trend half is not evaluated or recorded there,
    since the cached daily bars it reads have not changed between the
    verdict and the next open.

    add_loser, when given (_add_loser_check, an ADD with ADD_LOSER_GATE_MODE
    "enforce"; the route passes none in "shadow"), is recorded as
    entry_gate.add_loser and blocks when its blocked is True: the ADD is more
    than ADD_LOSER_GATE_MIN_PCT below the average cost and the triggering
    item, judged, is not a new fact.

    row_extra's keys go into the ENTRY_GATE_BLOCKED row's decision beside
    entry_gate (the recheck-contradiction record); they never decide."""
    try:
        try:
            from config import STOPOUT_REENTRY_GATE_DAYS as days
        except ImportError:
            days = 0
        try:
            from config import ENTRY_TREND_GATE_ENABLED as trend_on
        except ImportError:
            trend_on = False
        try:
            from config import ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS as max_age
        except ImportError:
            max_age = 5
        try:
            from config import ENTRY_TREND_GATE_MARKET_TICKER as market_ticker
        except ImportError:
            market_ticker = ""
        blocks, gate = [], {}
        if path:
            gate["path"] = path
            gate.update(extra or {})
        if path == "deferred_retry":
            trend_on = False
        if days and days > 0:
            s = stop_rec or {}
            hrs = s.get("hours_since")
            # A negative age (exit time ahead of the clock) does not block.
            hit = (s.get("read") == "ok" and isinstance(hrs, (int, float))
                   and not isinstance(hrs, bool) and 0 <= hrs < days * 24)
            gate["stopout"] = {"days": days, "read": s.get("read"), "hours_since": hrs,
                               "exit_at": s.get("exit_at"), "blocked": hit}
            if hit:
                blocks.append(f"{hrs:.1f} h after an exit the bot did not place (gate {days} d)")
        if trend_on:
            t = trend or {}
            age = t.get("bar_age_days")
            fresh = isinstance(age, int) and not isinstance(age, bool) and 0 <= age <= max_age
            would = (t.get("read") == "ok" and fresh
                     and t.get("below_sma50") is True and t.get("below_sma200") is True)
            # Bars that look like an unadjusted split (_split_discontinuity)
            # give averages on mixed price scales: such a read never blocks.
            suspect = t.get("split_suspect") is True
            market = None
            market_down = True   # no market ticker: the stock condition alone decides
            market_raw = True    # the market condition before the market's own split check
            if would and market_ticker:
                # Read only when the stock itself would be blocked (also when
                # its bars are split-suspect, as a record). Its own try: a
                # failure here must not drop a stop-out block too.
                market_down = market_raw = False
                try:
                    m = _entry_trend_check(market_ticker)
                    m_age = m.get("bar_age_days")
                    m_fresh = (isinstance(m_age, int) and not isinstance(m_age, bool)
                               and 0 <= m_age <= max_age)
                    m_would = m.get("read") == "ok" and m_fresh and m.get("below_sma200") is True
                    m_suspect = m.get("split_suspect") is True
                    market_raw = m_would
                    market_down = m_would and not m_suspect
                    market = {"ticker": market_ticker, "read": m.get("read"), "bar_age_days": m_age,
                              "below_sma200": m.get("below_sma200"),
                              "pct_vs_sma200": m.get("pct_vs_sma200"), "downtrend": market_down,
                              "split_suspect": m.get("split_suspect"),
                              "split_suspect_date": m.get("split_suspect_date"),
                              "split_suspect_ratio": m.get("split_suspect_ratio"),
                              # the block withheld only by the market's own split suspicion
                              "split_fail_open": bool(not suspect and m_would and m_suspect)}
                except Exception as e:
                    market_down = market_raw = False
                    market = {"ticker": market_ticker, "read": f"error: {type(e).__name__}",
                              "bar_age_days": None, "below_sma200": None, "pct_vs_sma200": None,
                              "downtrend": False, "split_suspect": None, "split_suspect_date": None,
                              "split_suspect_ratio": None, "split_fail_open": False}
            # The block the stock's split suspicion withheld: everything else
            # held, on the market's raw condition (a split-suspect SPY below
            # its 200-day still counts, so that case leaves a record).
            split_fail_open = bool(would and suspect and market_raw)
            hit = bool(would and not suspect and market_down)
            gate["trend"] = {"read": t.get("read"), "bar_age_days": age, "max_bar_age_days": max_age,
                             "below_sma50": t.get("below_sma50"), "below_sma200": t.get("below_sma200"),
                             "pct_vs_sma50": t.get("pct_vs_sma50"),
                             "pct_vs_sma200": t.get("pct_vs_sma200"), "market": market,
                             "split_suspect": t.get("split_suspect"),
                             "split_suspect_date": t.get("split_suspect_date"),
                             "split_suspect_ratio": t.get("split_suspect_ratio"),
                             "split_suspect_rule": t.get("split_suspect_rule"),
                             "split_suspect_factor": t.get("split_suspect_factor"),
                             "split_fail_open": split_fail_open, "blocked": hit}
            if hit:
                blocks.append(f"close {t.get('pct_vs_sma50')}% vs the 50-day and "
                              f"{t.get('pct_vs_sma200')}% vs the 200-day average"
                              + (f", with {market_ticker} {market['pct_vs_sma200']}% vs its 200-day"
                                 if market else ""))
        if add_loser is not None:
            gate["add_loser"] = add_loser
            if add_loser.get("blocked") is True:
                blocks.append(_add_loser_reason(add_loser))
        if not blocks:
            tr = gate.get("trend") or {}
            if tr.get("split_fail_open") or (tr.get("market") or {}).get("split_fail_open"):
                # No gate row is written when nothing blocks, so the trend
                # gate's split fail-open leaves an activity row instead.
                try:
                    import db
                    db.log_activity_queued(
                        "entry_gate_split_fail_open", ticker,
                        f"{result.get('action')}: trend gate not applied, split-suspect bars "
                        f"({tr.get('split_suspect_date') or (tr.get('market') or {}).get('split_suspect_date')})",
                        extra={"entry_gate": gate, "action": result.get("action")})
                except Exception as e:
                    log.warning(f"[{ticker}] split fail-open record not written: {type(e).__name__}: {e}")
            return False
        reason = f"{result.get('action')} skipped by the entry gate: " + "; ".join(blocks)
        log.info(f"[{ticker}] {reason}")
        _log_route_exit(ticker, "ENTRY_GATE_BLOCKED", result, reason,
                        {"entry_gate": gate, **(row_extra or {})})
        return True
    except Exception as e:
        log.warning(f"[{ticker}] entry gate failed ({type(e).__name__}: {e}); not applied")
        return False


_BOOK_HEAT_SNAPSHOT_MAX_AGE_MIN = 15   # the same freshness the held checks use (tasks._broker_held)


def _book_heat_inputs():
    """The latest portfolio snapshot of this env (positions_json, total value,
    age in minutes) and the position_stops (ticker, stop_price) rows: two
    SELECTs in one read transaction under a 3 s statement timeout, so a lock
    or a stuck connection can't hold up the buy route for long. No broker
    call. Returns None when there is no snapshot; raises on a failed read
    (_book_heat_record catches it)."""
    import db
    from datetime import datetime, timezone
    from config import TRADING_ENV
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 3000")
            cur.execute("SELECT positions_json, total_value, timestamp FROM portfolio_snapshots "
                        "WHERE trading_env=%s ORDER BY timestamp DESC LIMIT 1", (TRADING_ENV,))
            row = cur.fetchone()
            stops = []
            if row and row[2] is not None:
                cur.execute("SELECT ticker, stop_price FROM position_stops WHERE trading_env=%s",
                            (TRADING_ENV,))
                stops = [{"ticker": r[0], "stop_price": r[1]} for r in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    if not row or row[2] is None:
        return None
    ts = row[2] if row[2].tzinfo else row[2].replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    return {"positions": row[0] or {}, "pv": row[1], "age_min": age, "stops": stops}


def _book_heat_record(ticker: str, sizing: dict) -> dict:
    """Record-only (config BOOK_HEAT_SHADOW_ENABLED): the book's open risk at
    this buy, as fractions of the portfolio. For each held position, qty x
    (snapshot price - its position_stops stop) / portfolio value: what the book
    loses if every recorded stop fills at its price (0 for a stop at or above
    the price; None and counted in n_no_stop without a stop record). The
    order's own risk is shares_initial x (price_at_sizing - stop_loss_price)
    / portfolio value, at the final share count. The order's 1-sigma
    20-session loss comes from the vol record when it has one; held names have
    no vol in hand, so they get none. The broker's resting stops are not read
    (position_stops is the bot's record of them). No pv, dollars or quantities
    are recorded. Never raises."""
    import math
    rec = {"source": "portfolio_snapshot+position_stops", "read": None, "snapshot_age_min": None,
           "n_positions": None, "n_with_stop": None, "n_no_stop": None,
           "n_stop_at_or_above_price": None, "sum_open_risk_pct": None,
           "max_single_risk_pct": None, "max_single_ticker": None, "unprotected_value_pct": None,
           "positions": None, "order_risk_pct": None, "sum_open_risk_after_pct": None,
           "order_risk_sigma20": None, "order_sigma20_loss_pct": None,
           "held_sigma20": None, "held_sigma20_reason": "no per-name vol in hand for held positions",
           "weights_at": "final_order", "reason": None}

    def frac(x, nd=4):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        return round(x, nd) if math.isfinite(x) else None
    try:
        try:
            inp = _book_heat_inputs()
        except Exception as e:
            rec["read"] = f"error: {type(e).__name__}"
            rec["reason"] = "snapshot or stop records unreadable"
            return rec
        if inp is None:
            rec["read"], rec["reason"] = "no_snapshot", "no portfolio snapshot for this env"
            return rec
        rec["snapshot_age_min"] = frac(inp.get("age_min"), 1)
        if rec["snapshot_age_min"] is None or rec["snapshot_age_min"] > _BOOK_HEAT_SNAPSHOT_MAX_AGE_MIN:
            rec["read"] = "stale_snapshot"
            rec["reason"] = f"newest snapshot older than {_BOOK_HEAT_SNAPSHOT_MAX_AGE_MIN} min"
            return rec
        pv = frac(inp.get("pv"), 6)
        if not pv or pv <= 0:
            rec["read"], rec["reason"] = "no_portfolio_value", "snapshot has no portfolio value"
            return rec
        stops = {}
        for s in inp.get("stops") or []:
            t = str(s.get("ticker") or "").upper()
            sp = frac(s.get("stop_price"), 6)
            if t and sp is not None and sp > 0:
                stops[t] = sp
        positions = inp.get("positions") if isinstance(inp.get("positions"), dict) else {}
        rows, total, n_stop, n_none, n_above, unprot = [], 0.0, 0, 0, 0, 0.0
        n_short = n_unpriced = 0
        for t, p in sorted(positions.items()):
            if not isinstance(p, dict):
                continue
            qty, mv = frac(p.get("qty"), 6), frac(p.get("market_value"), 6)
            if qty is not None and qty < 0:
                n_short += 1                  # long-only bot: a short means an oversell; counted
                continue
            if not qty:
                continue                      # flat
            if mv is None or mv <= 0:
                n_unpriced += 1               # held but no usable price: not in the sums, counted
                continue
            px = mv / qty
            stop = stops.get(str(t).upper())
            if stop is None:
                n_none += 1
                unprot += mv / pv
                rows.append({"ticker": t, "risk_pct": None, "stop": "none"})
                continue
            n_stop += 1
            if stop >= px:
                n_above += 1
            risk = max(px - stop, 0.0) * qty / pv
            total += risk
            rows.append({"ticker": t, "risk_pct": frac(risk), "stop": "position_stops"})
        rec.update({"read": "ok", "n_positions": len(rows), "n_with_stop": n_stop, "n_no_stop": n_none,
                    "n_short": n_short, "n_unpriced": n_unpriced,
                    "n_stop_at_or_above_price": n_above, "sum_open_risk_pct": frac(total),
                    "unprotected_value_pct": frac(unprot), "positions": rows})
        risks = [r for r in rows if r["risk_pct"] is not None]
        if risks:
            top = max(risks, key=lambda r: r["risk_pct"])
            rec["max_single_risk_pct"], rec["max_single_ticker"] = top["risk_pct"], top["ticker"]
        n = sizing.get("shares_initial")
        price, stop = frac(sizing.get("price_at_sizing"), 6), frac(sizing.get("stop_loss_price"), 6)
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0 and price and stop is not None:
            order_risk = n * max(price - stop, 0.0) / pv
            rec["order_risk_pct"] = frac(order_risk)
            rec["sum_open_risk_after_pct"] = frac(total + order_risk)
        else:
            rec["reason"] = "order risk unknown (no shares, price or stop)"
        sv = sizing.get("stop_vol") if isinstance(sizing.get("stop_vol"), dict) else {}
        vt = sizing.get("vol_target_shadow") if isinstance(sizing.get("vol_target_shadow"), dict) else {}
        rec["order_risk_sigma20"] = sv.get("stop_sigma20")
        sigma20, ow = sv.get("sigma20_pct"), vt.get("order_weight")
        if isinstance(sigma20, (int, float)) and isinstance(ow, (int, float)):
            rec["order_sigma20_loss_pct"] = frac(ow * sigma20)
    except Exception as e:
        rec["read"] = f"error: {type(e).__name__}"
        rec["reason"] = str(e)[:200]
    return rec


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

    action = result["action"]
    # Skip if this ticker already has an open approval waiting for decision.
    # An unreadable guard skips a buy (both duplicate-buy guards read the same
    # table, so one DB error used to disable both: audit 11-06 21:36 #8); a
    # sell goes on as before.
    try:
        if db.has_active_approval(ticker, strict=True):
            log.info(f"[{ticker}] Active approval already pending — skipping duplicate")
            _log_route_exit(ticker, "ROUTE_APPROVAL_PENDING", result, "active approval already pending")
            return
    except Exception as e:
        if action in ("OPEN", "ADD"):
            log.warning(f"[{ticker}] Pending-approval check unreadable ({e}); {action} skipped")
            _log_route_exit(ticker, "ROUTE_GUARD_UNREADABLE", result,
                            f"pending-approval check unreadable: {type(e).__name__}")
            return

    # Record only: the latest non-buy verdict on the name before this buy,
    # set past the rebuy cooldown; rx carries it onto the rows below.
    recheck, consensus, rx = None, None, {}
    add_loser = None   # the add-to-loser record (_add_loser_check), set at the entry gates
    if action == "ADD":
        _rule_shadow_average_down(ticker, bucket, trading_client)

    if action in ("OPEN", "ADD"):
        try:
            from config import REBUY_COOLDOWN_HOURS
        except ImportError:
            REBUY_COOLDOWN_HOURS = 4
        try:
            prior = db.get_last_executed_approval(ticker, REBUY_COOLDOWN_HOURS, strict=True)
        except Exception as e:
            log.warning(f"[{ticker}] Rebuy-cooldown check unreadable ({e}); {action} skipped")
            _log_route_exit(ticker, "ROUTE_GUARD_UNREADABLE", result,
                            f"rebuy-cooldown check unreadable: {type(e).__name__}")
            return
        if prior and conviction <= prior["conviction"]:
            log.info(
                f"[{ticker}] Rebuy cooldown — approved {prior['conviction']}/10 at "
                f"{prior['decided_at']}, new signal is {conviction}/10 (not higher) — skipping"
            )
            try:
                db.log_signal(ticker, "COOLDOWN_SKIPPED", conviction / 10.0,
                              f"Within {REBUY_COOLDOWN_HOURS}h of prior approved buy "
                              f"at conviction {prior['conviction']}/10",
                              **_route_row_fields(result, "COOLDOWN_SKIPPED"))
            except Exception as e:
                log.warning(f"[{ticker}] COOLDOWN_SKIPPED signal log failed: {e}")
            return
        recheck = _recheck_contradiction_record(ticker, result)
        rx = {"recheck_contradiction": recheck} if recheck is not None else {}
        consensus = _same_day_consensus_record(ticker, result)   # record only
        if consensus is not None:
            rx["same_day_consensus"] = consensus
        stop_rec = _stopout_reentry_shadow(ticker, result)
        trend = _entry_quality_shadow(ticker, result)
        add_loser = _add_loser_check(ticker, result, trading_client) if action == "ADD" else None
        # "enforce": a gate half; "shadow": recorded on the approval only.
        if _entry_gate(ticker, result, stop_rec, trend,
                       add_loser=add_loser if (add_loser or {}).get("mode") == "enforce" else None,
                       row_extra=rx):
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
                              result.get("reasoning", "")[:500],
                              **_route_row_fields(result, "SELL_POSITION_UNREADABLE"))
            except Exception as e:
                log.warning(f"[{ticker}] SELL_POSITION_UNREADABLE signal log failed: {e}")
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
                              result.get("reasoning", "")[:500],
                              **_route_row_fields(result, "SELL_NO_POSITION"))
            except Exception as e:
                log.warning(f"[{ticker}] SELL_NO_POSITION signal log failed: {e}")
            return

        try:
            from config import TRIM_FRACTION
        except ImportError:
            TRIM_FRACTION = 0.33
        rule = result.get("trim_exit")
        fraction = TRIM_FRACTION
        if rule and action == "TRIM":
            # A TRIM the trim/exit rule routed (agents/trim_exit.py) sells
            # TRIM_EXIT_FRACTION and keeps TRIM_EXIT_MIN_REMAINING_SHARES
            # under a resized stop.
            from config import TRIM_EXIT_FRACTION
            from agents import trim_exit
            fraction = float(TRIM_EXIT_FRACTION)
            sell_qty = trim_exit.trim_qty(owned)
            if sell_qty <= 0:
                log.info(f"[{ticker}] trim/exit rule TRIM: no partial sale of {owned:g} shares")
                _log_route_exit(ticker, "ROUTE_ZERO_SHARES", result,
                                f"trim/exit rule TRIM: no partial sale of {owned:g} shares",
                                {"trim_exit": rule, "owned": owned})
                return
        else:
            if rule and action == "EXIT" and rule.get("reason") == "trim_one_share_exit":
                # The rule turned a TRIM into this EXIT because the review's
                # position read held one share. If the fresh read allows a
                # partial sale (shares bought since), the EXIT would sell
                # them all: nothing is routed, and the next review decides.
                from agents import trim_exit
                partial = trim_exit.trim_qty(owned)
                if partial > 0:
                    log.info(f"[{ticker}] trim/exit rule one-share EXIT not routed: "
                             f"{owned:g} shares held now")
                    _log_route_exit(ticker, "ROUTE_ZERO_SHARES", result,
                                    f"trim/exit rule one-share EXIT: {owned:g} shares held now, "
                                    f"a partial sale of {partial} exists; nothing routed",
                                    {"trim_exit": rule, "owned": owned, "partial_qty": partial})
                    return
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
        if action == "TRIM":
            # Record-only: at a few shares the whole-share floor turns a TRIM
            # into a sale of most or all of the position.
            sizing["trim_granularity"] = {"owned": owned, "fraction": fraction,
                                          "intended_qty": round(owned * fraction, 4),
                                          "sell_qty": sell_qty, "full_exit": sell_qty >= owned}
        if rule:
            # execute_approved rechecks the rule's switches on this key.
            sizing["trim_exit"] = {**rule, "owned": owned, "sell_qty": sell_qty}
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
                    db.log_signal(ticker, "CATALYST_BLOCKED", 0.0, reason,
                                  **_route_row_fields(result, "CATALYST_BLOCKED", rx))
                except Exception as e:
                    log.warning(f"[{ticker}] CATALYST_BLOCKED signal log failed: {e}")
                try:
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
                db.log_signal(ticker, "CATALYST_GATE_ERROR", 0.0, str(e)[:500],
                              **_route_row_fields(result, "CATALYST_GATE_ERROR", rx))
            except Exception as log_error:
                log.warning(f"[{ticker}] CATALYST_GATE_ERROR signal log failed: {log_error}")

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
                              **_route_row_fields(result, "EARNINGS_BLOCKED", rx))
            except Exception as e:
                log.warning(f"[{ticker}] EARNINGS_BLOCKED signal log failed: {e}")
            return

        # New-fact gate: before sizing, so a swap never sells for a buy it blocks.
        prov = result.get("provenance")
        new_fact = new_fact_gate(ticker, action,
                                 prov.get("trigger") if isinstance(prov, dict) else None)
        if new_fact is not None and not new_fact["passed"]:
            log.info(f"[{ticker}] Entry blocked: {new_fact['reason']}")
            if _feed_emit:
                _feed_emit("risk", ticker, f"Blocked — {new_fact['reason']}", "info")
            _log_route_exit(ticker, "ROUTE_NO_NEW_FACT", result, new_fact["reason"],
                            decision_extra={"new_fact_gate": new_fact, **rx})
            _c6_bar_no_new_fact(result, regime_str)   # log-only
            _alert_new_fact_fallback_block(            # alert only (M275)
                ticker, new_fact, prov.get("trigger") if isinstance(prov, dict) else None)
            return

        try:
            rm = RiskManager(trading_client)
        except Exception as e:
            log.error(f"[{ticker}] RiskManager init failed: {e}")
            _record_risk_init_failed(ticker, result, e, rx)
            return

        # Get current price
        sizing_price_info: dict = {}   # its source and bar time (record only)
        price = _get_price(ticker, trading_client, info=sizing_price_info)
        if not price:
            log.warning(f"[{ticker}] Could not determine price — skipping approval queue")
            _log_route_exit(ticker, "ROUTE_NO_PRICE", result, "could not determine price",
                            decision_extra=rx or None)
            return

        # Size the position — stamp the price used so execution can validate freshness
        sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
        sizing["price_at_sizing"] = price
        _attach_shadows(sizing, ticker, result, price, trading_client,
                        price_info=sizing_price_info)
        if recheck is not None:
            sizing["recheck_contradiction"] = recheck
        if consensus is not None:
            sizing["same_day_consensus"] = consensus
        if (add_loser or {}).get("mode") == "shadow":
            sizing["add_loser_gate"] = add_loser
        swap_sold = None   # the holding a swap sold for this buy, if any
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
                        cash_before = rm.cash()
                    except Exception:
                        cash_before = None
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
                        # Re-size once the sale shows in cash: IBKR updates
                        # TotalCashValue late, and re-sizing at once could leave
                        # the holding sold and the buy at 0 (A1031-1253-3).
                        swap_sold = swap["ticker"]
                        _wait_for_swap_cash(rm, cash_before, sell_result, ticker)
                        sizing = rm.size_position(ticker, conviction, bucket, regime_str, price)
                        sizing["price_at_sizing"] = price
                        _attach_shadows(sizing, ticker, result, price, trading_client,
                                        price_info=sizing_price_info)
                        if recheck is not None:
                            sizing["recheck_contradiction"] = recheck
                        if consensus is not None:
                            sizing["same_day_consensus"] = consensus
                        if (add_loser or {}).get("mode") == "shadow":
                            sizing["add_loser_gate"] = add_loser
                        if sizing.get("shares_initial", 0) <= 0:
                            try:
                                import alerts as alert_module
                                alert_module._send(
                                    f"⚠️ SWAP INCOMPLETE — sold {swap['ticker']} to fund <b>{ticker}</b>, "
                                    f"but the buy still sizes to 0 shares (flags: "
                                    f"{', '.join(sizing.get('flags') or []) or 'none'}). The cash is "
                                    f"free; the next {ticker} analysis can use it.")
                            except Exception:
                                pass
                    else:
                        log.warning(f"[{swap['ticker']}] Swap sell did not complete: "
                                    f"{sell_result.get('message')}")

            if sizing.get("shares_initial", 0) <= 0:
                flags = sizing.get("flags", [])
                log.info(f"[{ticker}] Risk sizing returned 0 shares: {flags}")
                _log_route_exit(ticker, "ROUTE_ZERO_SHARES", result, "; ".join(flags),
                                decision_extra={**(_add_guard_record(ticker, result, sizing) or {}), **rx})
                _shadow_swap(rm, ticker, conviction, sizing, result)
                _c6_bar_tier_floor(result, regime_str, price, sizing)   # log-only
                return

        # Concentration check. position_size_pct is the remaining target; the
        # order itself can be bigger (one whole share above target, e.g. on an
        # ADD), so check whichever is larger (audit 10-28 #5).
        proposed_pct = max(sizing["position_size_pct"], sizing.get("order_pct") or 0)
        conc = rm.check_concentration(ticker, proposed_pct, bucket=bucket)
        # Log-only theme grouping on the heading prefix (risk_manager.
        # _theme_group_shadow); the caps below use today's grouping.
        sizing["theme_group_shadow"] = conc.get("theme_shadow")
        sizing["atr_brief"] = _atr_brief(ticker)
        # Log-only: the theme total if buys approved or deferred but not yet
        # filled were counted (risk_manager._inflight_theme_shadow).
        from risk_manager import _inflight_theme_shadow
        sizing["concentration_inflight_shadow"] = _inflight_theme_shadow(
            ticker, proposed_pct, conc.get("theme_shadow"))
        if not conc["passed"]:
            # The flags state the checked weight (the larger of target and
            # order), not what the order will buy after the caps below.
            log.info(f"[{ticker}] Concentration check failed: {conc['flags']} "
                     f"(checked {proposed_pct:.1%}: target {sizing['position_size_pct']:.1%}, "
                     f"order {sizing.get('order_pct') or 0:.1%})")
            sizing["flags"].extend(conc["flags"])
            # The flag text states the checked weight (the larger of target
            # and order), before any cap; these keys say which was which and
            # what the cap left.
            conc_rec = {"checked_pct": round(proposed_pct, 4),
                        "target_pct": round(sizing["position_size_pct"], 4),
                        "order_pct": round(sizing.get("order_pct") or 0, 4),
                        "cap_pct": None, "shares_before": sizing["shares_initial"],
                        "shares_after": sizing["shares_initial"], "order_pct_after": None,
                        "failed_rules": _conc_failed_rules(conc.get("flags")),
                        "errored": bool(conc.get("errored")), "cap_binding": False,
                        "binding_cap": None}
            sizing["concentration_check"] = conc_rec
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
            named = [(n, conc.get(k)) for n, k in (("ai_infra", "ai_infra_size_cap_pct"),
                                                    ("theme", "theme_size_cap_pct"),
                                                    ("moonshot_total", "moonshot_size_cap_pct"))
                     if conc.get(k) is not None]
            caps = [c for _, c in named]
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
                if sizing.get("order_pct"):
                    sizing["order_pct"] = (round(sizing["order_pct"] * shares / old_shares, 4)
                                           if old_shares else 0.0)
                # cap_binding: the cap reduced the share count (an errored
                # check's 0.0 caps cut to 0, so it binds there too); a cap
                # between order and target leaves the shares unchanged.
                conc_rec.update(cap_pct=round(cap, 4), shares_after=shares,
                                order_pct_after=sizing.get("order_pct"),
                                cap_binding=shares < old_shares,
                                # An errored check sets every cap to 0.0, so no
                                # single rule binds (professor m534).
                                binding_cap=("errored" if conc.get("errored")
                                             else next(n for n, c in named if c == cap)))
                sizing["flags"].append(
                    f"Size capped to {cap:.1%} target ({sizing['shares_initial']} shares) "
                    f"to stay within concentration limits"
                )
                log.info(f"[{ticker}] {sizing['flags'][-1]}")
                if sizing["shares_initial"] <= 0:
                    log.info(f"[{ticker}] Concentration limits leave 0 shares of room — "
                             f"skipping entry (already at/over the limit)")
                    _log_route_exit(ticker, "ROUTE_CONCENTRATION_ZERO", result,
                                    "; ".join(sizing.get("flags", [])),
                                    decision_extra={"concentration_check": conc_rec,
                                                    "theme_group_shadow": sizing.get("theme_group_shadow"),
                                                    "concentration_inflight_shadow":
                                                        sizing.get("concentration_inflight_shadow"),
                                                    **(_add_guard_record(ticker, result, sizing)
                                                       or {}), **rx})
                    return

        # Correlation check
        corr = rm.check_correlation(ticker)
        if corr["reduce_by"] > 0:
            log.info(f"[{ticker}] Correlation adjustment: reducing size 25% ({corr['note']})")
            shares_before = sizing["shares_initial"]
            sizing["shares_initial"] = max(1, int(sizing["shares_initial"] * 0.75))
            sizing["initial_dollar"] = round(sizing["initial_dollar"] * 0.75, 2)
            # order_pct follows the shares (professor m507/66c): nothing on
            # this route sizes from it after the cut (the micro cap reads
            # shares and price), so only the stored weight changes. The
            # target, position_size_pct, stays. Recorded in correlation_cut.
            op = sizing.get("order_pct") or 0
            if shares_before and op:
                sizing["order_pct"] = round(op * sizing["shares_initial"] / shares_before, 4)
            sizing["correlation_cut"] = {
                "applied": True, "reduce_by": corr.get("reduce_by"),
                # The fail-closed cut (positions unreadable) also reduces 25%.
                "errored": str(corr.get("note") or "").startswith("RISK CHECK FAILED"),
                "shares_before": shares_before, "shares_after": sizing["shares_initial"],
                "order_pct_before": op, "order_pct_after": sizing.get("order_pct")}
        else:
            sizing["correlation_cut"] = {"applied": False, "reduce_by": corr.get("reduce_by")}

        # Last, so it bounds the final order (after the swap's re-size too).
        if _apply_micro_name_cap(rm, ticker, sizing, price, result):
            if swap_sold:
                try:
                    import alerts as alert_module
                    alert_module._send(
                        f"⚠️ SWAP INCOMPLETE — sold {swap_sold} to fund <b>{ticker}</b>, "
                        f"but the micro-tier name cap leaves 0 shares. The cash is "
                        f"free; the next {ticker} analysis can use it.")
                except Exception:
                    pass
            return
        if new_fact is not None:
            sizing["new_fact_gate"] = new_fact
        _attach_gap_risk_shadow(rm, ticker, sizing, price, action)   # log-only, final shares
        sizing["udr_at_sizing"] = _udr_at_sizing(result, price)      # log-only
        try:
            from config import RR_VS_STOP_SHADOW_ENABLED
        except ImportError:
            RR_VS_STOP_SHADOW_ENABLED = False
        if RR_VS_STOP_SHADOW_ENABLED:
            sizing["rr_vs_stop"] = _rr_vs_stop(result, sizing, price)   # record only
        sizing["tighten_sizing_shadow"] = _tighten_sizing_shadow(ticker)   # log-only
        sizing["regime_score"] = regime.get("score")                     # log-only, beside the label
        if result.get("action") == "ADD":
            sizing["add_context"] = _add_context(ticker, price, sizing, trading_client,
                                                 rm=rm)   # log-only
        # Log-only: restate size_position's vol shadow at the final share count,
        # after the concentration, correlation and micro-name cuts above.
        try:
            if isinstance(sizing.get("vol_target_shadow"), dict):
                from risk_manager import _vol_shadow_reweigh
                sizing["vol_target_shadow"] = _vol_shadow_reweigh(sizing["vol_target_shadow"],
                                                                  sizing.get("shares_initial"))
        except Exception as e:
            log.debug(f"[{ticker}] vol shadow restate failed: {e}")
        # Log-only: open risk on the book at this buy (config BOOK_HEAT_SHADOW_ENABLED).
        try:
            from config import BOOK_HEAT_SHADOW_ENABLED as book_heat_on
        except ImportError:
            book_heat_on = False
        if book_heat_on:
            try:
                sizing["book_heat"] = _book_heat_record(ticker, sizing)
            except Exception as e:
                log.debug(f"[{ticker}] book heat record failed: {e}")

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
    from execution import execute_approved, approval_outcome

    if not db.claim_approval_for_execution(approval_id):
        log.error(f"[{ticker}] Could not claim approval #{approval_id} for auto-execution")
        return

    approval = db.get_approval_by_id(approval_id)
    success = execute_approved(approval, trigger="auto")
    # A fill left to reconcile filled: APPROVED with that reason, not FAILED.
    status, reason = approval_outcome(success)
    db.update_approval_status(approval_id, status, reason=reason)
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

    # _execute_approved_order already sends a Telegram trade alert on success;
    # only notify separately here when it didn't execute, so a broken
    # execution doesn't fail silently.
    if not success:
        _report_auto_execution_not_done(_feed_emit, alert_module._send, ticker,
                                        result["action"], approval_id, status, reason)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_auto_execution_not_done(feed_emit, send, ticker: str, action: str,
                                    approval_id: int, status: str, reason: str | None) -> None:
    """Feed row and Telegram alert for an auto-execution that returned False.
    A deferral to a scheduled retry (market closed, stale price, gap check)
    is not a failure: execute_approved already sent its own deferral alert and
    the approval stays FAILED only so retry_deferred_entry can claim it, so
    this writes a deferral feed row and no failure alert. A fill left to
    reconcile (APPROVED) has its own alert. Anything else is a real failure."""
    from execution import outcome_deferred
    deferred = status == "FAILED" and outcome_deferred(reason)
    if feed_emit:
        try:
            if deferred:
                feed_emit("execution", ticker,
                          f"Auto-execution of {action} {ticker} deferred — {reason}", "info")
            else:
                feed_emit("execution", ticker,
                          f"Auto-execution FAILED for {action} {ticker} — check logs", "alert")
        except Exception:
            pass
    if status != "FAILED" or deferred:
        return
    try:
        send(
            f"⚠️ AUTO-EXECUTION FAILED\n"
            f"<b>{action} {ticker}</b> (approval #{approval_id}) could not be executed.\n"
            f"Check helios.log for details."
        )
    except Exception:
        pass


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


def _aware_utc(t):
    """t as an aware UTC datetime when it is an aware datetime, else None
    (a naive time has no known zone). Never raises."""
    import datetime as _dtm
    try:
        if isinstance(t, _dtm.datetime) and t.tzinfo is not None:
            return t.astimezone(_dtm.timezone.utc)
    except Exception:
        pass
    return None


def _get_price(ticker: str, client, info: dict | None = None) -> float | None:
    """The price for a brief, from the first source that answers. info
    (optional dict, record only): filled with which source answered
    ("session_quote", "polygon_snapshot", "latest_quote", "yfinance_info",
    or None when none did), as_of (that source's own bar time when it has
    one, else None), as_of_utc (that time as an aware UTC datetime where the
    source gave its zone, else None) and is_stale (the latest quote's flag;
    None for the others). Only values the sources already return; the price
    is unchanged."""
    def note(source, as_of=None, is_stale=None, as_of_utc=None):
        if info is not None:
            info.update(source=source, as_of=as_of, is_stale=is_stale, as_of_utc=as_of_utc)
    note(None)
    # Outside regular hours, the last extended-hours trade (Luke D6, 10-29);
    # the regular-hours quote below is the prior close all night.
    try:
        from market_data import get_session_quote
        sq = get_session_quote(ticker)
        if sq and sq.get("price"):
            note("session_quote", sq.get("bar_time"), as_of_utc=_aware_utc(sq.get("bar_time")))
            return float(sq["price"])
    except Exception:
        pass
    try:
        from ingestion.polygon_stream import get_snapshot
        snap = get_snapshot(ticker)
        if snap.get("price"):
            note("polygon_snapshot")
            return float(snap["price"])
    except Exception:
        pass
    try:
        from market_data import get_latest_quotes
        quotes = get_latest_quotes([ticker])
        q = quotes.get(ticker, {})
        mid = q.get("mid")
        if mid:
            note("latest_quote", q.get("bar_time"), q.get("is_stale"), q.get("bar_time_utc"))
            return float(mid)
    except Exception:
        pass
    # Final fallback: yfinance (no API key, always available)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Outside regular hours prefer the extended-hours price (what we'd
        # actually pay, Luke D6); during them, postMarketPrice is the previous
        # evening's after-hours print, so the regular price comes first.
        full = t.info
        from config import PRICE_FALLBACK_SESSION_ORDER
        try:
            from market_data import is_market_hours
            rth = PRICE_FALLBACK_SESSION_ORDER and is_market_hours()
        except Exception:
            rth = False
        # Before noon ET the newer extended print is today's pre-market one;
        # after the close, the after-hours one. Off: after-hours first always.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        morning = (PRICE_FALLBACK_SESSION_ORDER
                   and datetime.now(ZoneInfo("America/New_York")).hour < 12)
        ext = ((full.get("preMarketPrice"), full.get("postMarketPrice")) if morning
               else (full.get("postMarketPrice"), full.get("preMarketPrice")))
        order = ((full.get("regularMarketPrice"), *ext) if rth
                 else (*ext, full.get("regularMarketPrice")))
        price = next((v for v in order if v), None) or getattr(t.fast_info, "last_price", None)
        if price:
            p = float(price)
            if 0.01 <= p <= 50000.0:
                note("yfinance_info")
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
