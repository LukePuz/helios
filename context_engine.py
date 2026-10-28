"""
Context Assembly Engine — the brain before the brain.

Before every model call, this assembles a complete "situation brief":
a structured document containing everything the 72B model needs to
reason about an event in relation to our current portfolio.

The model never sees raw data — it always sees this assembled context.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def _safe_import(module_name: str):
    """Imports a module, returns None if unavailable (missing API key etc)."""
    try:
        return __import__(module_name, fromlist=[""])
    except Exception:
        return None


_MACRO_CACHE_MAX_AGE_HOURS = 2


def _get_macro_summary(macro_mod) -> str:
    """Reads the macro summary tasks.poll_macro already refreshes hourly,
    instead of re-fetching live — macro_mod.get_macro_summary_text() issues
    10 FRED HTTP requests per call, and this section used to call it fresh
    on every single brief (a 5-candidate daily scan made 50 FRED calls for
    data that only changes hourly). Falls back to a live fetch if the cache
    is missing or stale."""
    try:
        import db
        cached_at = db.get_state("macro_summary_at")
        if cached_at:
            age_hours = (
                datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)
            ).total_seconds() / 3600
            if age_hours < _MACRO_CACHE_MAX_AGE_HOURS:
                cached_text = db.get_state("macro_summary_text")
                if cached_text:
                    return cached_text
    except Exception as e:
        log.debug(f"Macro cache read failed, falling back to live fetch: {e}")
    return macro_mod.get_macro_summary_text()


def _catalyst_calendar_section(ticker: str) -> str:
    """
    Structured, dated catalyst events (2026-08-30, user request): ticker's
    own (earnings/etc), its sector's (earnings clusters), and market-wide
    (FOMC/CPI/jobs/election) -- so the model can reason about "expected
    return by WHEN" against a real calendar instead of only whatever it
    happens to recall. Distinct from the news-search-based catalyst
    discovery elsewhere in both brief builders: this is the dated,
    confidence-tagged calendar (catalyst_service), not free-text search.
    Shared by assemble_brief and build_scan_brief so both get the same
    view without duplicating the query/formatting logic.
    """
    try:
        from catalyst_service.client import upcoming_events
        cal_events = upcoming_events(ticker=ticker, days_ahead=45)
        if not cal_events:
            return ""
        cal_lines = ["=== UPCOMING CATALYST CALENDAR (next 45 days) ==="]
        for e in cal_events:
            scope_tag = {"ticker": ticker, "sector": e.get("ticker"), "macro": "MARKET"}[e["scope"]]
            cal_lines.append(
                f"  [{e['scope'].upper():<6}] {e['event_date']} — {scope_tag} "
                f"{e['event_type']} ({e['confidence']}): "
                f"{e.get('thesis_note') or e['event_type']}"
            )
        return "\n".join(cal_lines)
    except Exception as e:
        log.debug(f"Catalyst calendar unavailable for {ticker}: {e}")
        return ""


def _position_context_section(ticker: str, price: float | None = None) -> str:
    """For a held ticker: entry, stop, days held, profit-target state, the
    thesis at entry and the latest one, and the last three verdicts. The brief
    showed only qty/entry/price/P&L, so a review (and the earnings review,
    which asks whether the stop and size fit) couldn't see the stop or what
    it decided before (limitations audits 2026-10-27/28).
    Off unless config.BRIEF_POSITION_CONTEXT (Luke's call, Q35)."""
    import config
    if not config.BRIEF_POSITION_CONTEXT:
        return ""
    try:
        import db
        stop = db.get_position_stop(ticker)
        if not stop:
            return ""
        ctx = db.get_position_context(ticker)
    except Exception as e:
        log.debug(f"[{ticker}] position context unavailable: {e}")
        return ""
    from datetime import datetime, timezone
    lines = [f"=== POSITION CONTEXT — {ticker} (held) ==="]
    entry, sp = stop.get("entry_price"), stop.get("stop_price")
    line = f"Entry ${entry:.2f}; protective stop ${sp:.2f}"
    if price and sp:
        line += f" ({(sp - price) / price:+.1%} from current price)"
    if stop.get("stop_method"):
        line += f" [{stop['stop_method']}]"
    lines.append(line)
    if stop.get("opened_at"):
        days = (datetime.now(timezone.utc) - stop["opened_at"]).days
        lines.append(f"Held {days} days; bucket {stop.get('thesis_bucket') or 'n/a'}; "
                     f"conviction at entry {stop.get('conviction') or 'n/a'}; "
                     f"trailing stop {'active' if stop.get('trailing_active') else 'not active'}; "
                     f"profit target {'already taken' if ctx.get('profit_target_taken') else 'not taken'}")

    def _thesis(t):
        if not isinstance(t, dict):
            return None
        return t.get("thesis_summary") or t.get("summary") or t.get("reasoning")

    if _thesis(ctx.get("original_thesis")):
        lines.append(f"Thesis at entry: {_thesis(ctx['original_thesis'])[:300]}")
    if _thesis(ctx.get("latest_thesis")):
        lines.append(f"Latest thesis ({str(ctx.get('thesis_updated_at') or '')[:10]}): "
                     f"{_thesis(ctx['latest_thesis'])[:300]}")
    if ctx.get("recent_verdicts"):
        lines.append("Recent verdicts: " + "; ".join(
            f"{str(v['timestamp'])[:10]} {v['signal']} c{v['conviction']}" for v in ctx["recent_verdicts"]))
    return "\n".join(lines)


def _price_structure_section(ticker: str, price: float | None = None) -> str:
    """Deterministic levels the model never saw (limitations audit
    2026-10-28 09:00, finding 2a): price structure, trend template, distance
    from the Kijun line, ATR, base, fair value and the nearest support and
    resistance rungs from the nightly trade ladder. These were computed only
    after the LLM call (analyst gates) or only with SETUP_GATES_ENABLED.
    Off unless config.BRIEF_PRICE_STRUCTURE (Luke's call, Q35)."""
    import config
    if not config.BRIEF_PRICE_STRUCTURE:
        return ""
    lines = [f"=== PRICE STRUCTURE — {ticker} ==="]
    try:
        from agents.price_action import compute_price_action, compute_trend_template
        pa = compute_price_action(ticker)
        if pa:
            lines.append(f"Structure: {pa.get('structure')} ({pa.get('tag')}); 50-day MA "
                         f"{pa.get('ma50_direction')}, 200-day MA {pa.get('ma200_direction')}. "
                         f"{pa.get('summary', '')}")
        tt = compute_trend_template(ticker)
        if tt:
            lines.append(f"Trend Template: {'PASS' if tt.get('pass') else 'FAIL'} — {tt.get('reasoning', '')}")
    except Exception as e:
        log.debug(f"[{ticker}] price action unavailable: {e}")
    try:
        from agents.chase_risk import compute_chase_risk
        cr = compute_chase_risk(ticker)
        if cr:
            lines.append(f"Distance from Kijun (26-day midline): {cr['pct_from_kijun']:+.1f}% "
                         f"(chase risk {cr['score']}/5, higher = less extended)")
    except Exception as e:
        log.debug(f"[{ticker}] chase risk unavailable: {e}")
    try:
        import db
        ld = db.get_trade_ladder(ticker)
        if ld:
            ref = price or ld.get("current_price")
            if ld.get("atr"):
                lines.append(f"ATR(14): ${ld['atr']:.2f}"
                             + (f" ({ld['atr'] / ref:.1%} of price)" if ref else ""))
            if ld.get("base_floor") and ld.get("base_ceiling"):
                lines.append(f"Recent base: ${ld['base_floor']:.2f} - ${ld['base_ceiling']:.2f}")
            if ld.get("fair_value_blended"):
                lines.append(f"Blended fair value: ${ld['fair_value_blended']:.2f} "
                             f"(confidence {ld.get('valuation_confidence') or 'n/a'}); "
                             f"street consensus target "
                             + (f"${ld['street_consensus_target']:.2f}" if ld.get("street_consensus_target") else "n/a"))
            rungs = sorted(ld.get("rungs") or [], key=lambda r: r.get("price") or 0)
            if ref and rungs:
                below = [r for r in rungs if r["price"] < ref][-2:]
                above = [r for r in rungs if r["price"] >= ref][:2]
                fmt = lambda r: f"${r['price']:.2f} ({r.get('detail') or ','.join(r.get('types', []))})"
                if below:
                    lines.append("Support: " + "; ".join(fmt(r) for r in reversed(below)))
                if above:
                    lines.append("Resistance: " + "; ".join(fmt(r) for r in above))
            if ld.get("computed_at"):
                lines.append(f"(levels computed {str(ld['computed_at'])[:16]})")
    except Exception as e:
        log.debug(f"[{ticker}] trade ladder unavailable: {e}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _fundamentals_section(ticker: str) -> str:
    """
    Real revenue growth, margins, FCF, and valuation multiples (see
    ingestion/fundamentals.py). Before this existed, the analyst LLM had
    nothing to ground upside/downside or valuation estimates in — it was
    self-reporting multiples from nothing. Explicitly says when data isn't
    cached yet rather than silently omitting the section.
    """
    try:
        import db
        from ingestion.fundamentals import get_peer_percentile
        fund = db.get_fundamentals(ticker)
        if not fund or fund.get("market_cap") is None:
            return (f"=== FUNDAMENTALS — {ticker} ===\n"
                    "Not yet cached (nightly refresh pending) — treat valuation "
                    "and upside/downside estimates as lower-confidence and say so.")

        def _pct(v):
            return f"{v:+.1%}" if v is not None else "n/a"
        def _num(v):
            return f"{v:.2f}x" if v is not None else "n/a"

        gm_pctile = get_peer_percentile(ticker, "gross_margin")
        ps_pctile = get_peer_percentile(ticker, "price_to_sales")
        pctile_note = ""
        if gm_pctile is not None or ps_pctile is not None:
            parts = []
            if gm_pctile is not None:
                parts.append(f"gross margin at the {gm_pctile:.0%} percentile of the watchlist")
            if ps_pctile is not None:
                parts.append(f"P/S at the {ps_pctile:.0%} percentile of the watchlist")
            pctile_note = "\nPeer-relative: " + "; ".join(parts)

        return f"""=== FUNDAMENTALS — {ticker} ===
Revenue growth (YoY): {_pct(fund['revenue_growth_yoy'])}
Gross margin: {_pct(fund['gross_margin'])}
FCF margin: {_pct(fund['fcf_margin'])}
Price/Sales: {_num(fund['price_to_sales'])}
EV/EBITDA: {_num(fund['ev_to_ebitda'])}
Forward P/E: {_num(fund['forward_pe'])}
Market cap: ${fund['market_cap']:,.0f}{pctile_note}"""
    except Exception as e:
        log.debug(f"Fundamentals section unavailable for {ticker}: {e}")
        return ""


def assemble_brief(
    event: dict,
    positions: dict[str, dict],
    trading_client=None,
    current_price: float | None = None,
) -> str:
    """
    Builds a complete situation brief for the 72B model.

    Args:
        event:          The triggering event (filing, news, options spike, etc.)
        positions:      Current open positions {symbol: {qty, entry_price, pl_pct}}
        trading_client: Alpaca TradingClient for live account data
        current_price:  Live price for the ticker being evaluated. The CURRENT
                         PORTFOLIO section below only shows a price for tickers
                         already held — for anything not currently held (most
                         chat-requested and event-triggered analyses), the model
                         previously had NO current price anywhere in its prompt
                         and would invent one when computing a dollar-denominated
                         valuation target. Found 2026-07-17: a SNDK research
                         request came back with a "$300 target" while SNDK
                         actually traded around $1,350 — the model had no real
                         price to anchor to and fabricated a plausible-sounding
                         number instead.

    Returns:
        A formatted string ready to be injected into the model prompt.
    """
    now = datetime.now(ET).strftime("%A %B %d, %Y %I:%M %p ET")
    ticker = event.get("ticker", "UNKNOWN")
    sections = []

    price_line = (
        f"Current Price: ${current_price:.2f}" if current_price
        else "Current Price: UNKNOWN — do not state a dollar price, multiple-implied "
             "price, or valuation target anywhere in your answer; describe valuation "
             "only in relative/qualitative terms instead."
    )

    # ── Header ──────────────────────────────────────────────────────────────
    sections.append(f"""=== SITUATION BRIEF ===
Generated: {now}
Triggering Event: {event.get('source', 'unknown').upper()} — {ticker}
Event Type: {event.get('form', event.get('type', 'unknown'))}
Event Title: {event.get('title', 'N/A')}
{price_line}""")

    # ── Event Content ────────────────────────────────────────────────────────
    event_text = event.get("text", "")
    if event_text:
        sections.append(f"""
=== EVENT CONTENT ===
{event_text[:3000]}
{'[Content truncated — see full filing at ' + event.get('url','') + ']' if len(event_text) > 3000 else ''}""")

    # ── Current Portfolio ────────────────────────────────────────────────────
    if positions:
        pos_lines = ["=== CURRENT PORTFOLIO ==="]
        for sym, pos in positions.items():
            pl_str = f"{pos.get('pl_pct', 0):+.1%}"
            pos_lines.append(
                f"  {sym:<6} {pos.get('qty', 0):>6.0f} shares | "
                f"Entry: ${pos.get('entry_price', 0):.2f} | "
                f"Current: ${pos.get('current_price', 0):.2f} | "
                f"P&L: {pl_str} | "
                f"Value: ${pos.get('market_value', 0):,.0f}"
            )
        sections.append("\n".join(pos_lines))
    else:
        sections.append("=== CURRENT PORTFOLIO ===\nNo open positions.")

    # ── Account State ────────────────────────────────────────────────────────
    if trading_client:
        try:
            account = trading_client.get_account()
            sections.append(f"""=== ACCOUNT STATE ===
Portfolio Value: ${float(account.portfolio_value):,.2f}
Cash Available:  ${float(account.cash):,.2f}
Buying Power:    ${float(account.buying_power):,.2f}""")
        except Exception as e:
            log.warning(f"Could not fetch account for brief: {e}")

    # ── Macro Environment ────────────────────────────────────────────────────
    macro_mod = _safe_import("ingestion.macro")
    if macro_mod:
        try:
            sections.append(_get_macro_summary(macro_mod))
        except Exception as e:
            log.warning(f"Macro data unavailable: {e}")

    # ── Catalyst Calendar ────────────────────────────────────────────────────
    cal_section = _catalyst_calendar_section(ticker)
    if cal_section:
        sections.append(cal_section)

    # ── Technical Picture ────────────────────────────────────────────────────
    polygon_mod = _safe_import("ingestion.polygon_stream")
    if polygon_mod:
        try:
            sections.append(polygon_mod.get_technicals_summary(ticker))
        except Exception as e:
            log.warning(f"Technicals unavailable for {ticker}: {e}")

    # ── Options Flow ─────────────────────────────────────────────────────────
    options_mod = _safe_import("ingestion.options_flow")
    if options_mod:
        try:
            flow = options_mod.get_flow_summary(ticker)
            sections.append(f"""=== OPTIONS FLOW — {ticker} ===
{flow['summary']}""")
        except Exception as e:
            log.warning(f"Options flow unavailable: {e}")

    # ── Fundamentals ─────────────────────────────────────────────────────────
    position_section = _position_context_section(ticker, current_price)
    if position_section:
        sections.append(position_section)

    price_section = _price_structure_section(ticker, current_price)
    if price_section:
        sections.append(price_section)

    fund_section = _fundamentals_section(ticker)
    if fund_section:
        sections.append(fund_section)

    # ── Semantic Memory (ChromaDB) ────────────────────────────────────────────
    try:
        import vector_store
        query = f"{ticker} {event.get('title', '')} {event.get('text', '')[:200]}"

        # Search for relevant history across all collections. Skip the
        # event's own doc (same id crew.screen_event stored it under).
        own_id = event.get("url") or event.get("title")
        history = vector_store.search_all_collections(
            query, n_results=4, exclude_ids={own_id} if own_id else None)
        if history:
            mem_lines = ["=== RELEVANT HISTORICAL CONTEXT (from memory) ==="]
            for h in history:
                score = h["score"]
                source = h.get("collection", "unknown")
                meta = h.get("metadata", {})
                date = meta.get("date", meta.get("timestamp", ""))
                mem_lines.append(
                    f"[{source.upper()} | score={score:.2f} | {date[:10]}]\n"
                    f"{h['text'][:400]}\n"
                )
            sections.append("\n".join(mem_lines))
        else:
            sections.append("=== RELEVANT HISTORICAL CONTEXT ===\nNo prior history found for this event.")
    except Exception as e:
        log.warning(f"Vector store search failed: {e}")

    # ── Related Symbols ──────────────────────────────────────────────────────
    from config import SYMBOLS
    related = _find_related_symbols(ticker, SYMBOLS)
    if related:
        sections.append(f"""=== RELATED SYMBOLS IN WATCHLIST ===
{ticker} may have indirect effects on: {', '.join(related)}
Consider second-order impact on these positions.""")

    # ── Final Assembly ───────────────────────────────────────────────────────
    return "\n\n".join(sections)


def build_scan_brief(
    ticker: str,
    positions: dict[str, dict] | None = None,
    trading_client=None,
    current_price: float | None = None,
) -> str:
    """
    Situation brief for the SCANNER-driven path (daily candidate analysis).

    Before this function existed, agents.analyst.analyze_batch() called
    analyze() with no situation_brief at all — so the DAILY scan, which is
    the main decision flow of the whole system, ran with zero vector-memory
    retrieval, zero macro/technicals/options context, and zero fundamentals.
    Only the rare event-triggered path (assemble_brief, above) got a brief.
    This gives scanner candidates the same context quality, built around the
    ticker itself rather than a triggering event.

    current_price: see assemble_brief's docstring — same fix, same reason
    (a ticker not already held had no price anchor anywhere in the prompt).
    """
    positions = positions or {}
    now = datetime.now(ET).strftime("%A %B %d, %Y %I:%M %p ET")
    price_line = (
        f"Current Price: ${current_price:.2f}" if current_price
        else "Current Price: UNKNOWN — do not state a dollar price, multiple-implied "
             "price, or valuation target anywhere in your answer; describe valuation "
             "only in relative/qualitative terms instead."
    )
    sections = [f"""=== SITUATION BRIEF (daily scan) ===
Generated: {now}
Ticker: {ticker}
{price_line}"""]

    # ── Current Portfolio ────────────────────────────────────────────────────
    if positions:
        pos_lines = ["=== CURRENT PORTFOLIO ==="]
        for sym, pos in positions.items():
            pl_str = f"{pos.get('pl_pct', 0):+.1%}"
            pos_lines.append(
                f"  {sym:<6} {pos.get('qty', 0):>6.0f} shares | "
                f"Entry: ${pos.get('entry_price', 0):.2f} | "
                f"Current: ${pos.get('current_price', 0):.2f} | "
                f"P&L: {pl_str} | "
                f"Value: ${pos.get('market_value', 0):,.0f}"
            )
        sections.append("\n".join(pos_lines))
    else:
        sections.append("=== CURRENT PORTFOLIO ===\nNo open positions.")

    # ── Account State ────────────────────────────────────────────────────────
    if trading_client:
        try:
            account = trading_client.get_account()
            sections.append(f"""=== ACCOUNT STATE ===
Portfolio Value: ${float(account.portfolio_value):,.2f}
Cash Available:  ${float(account.cash):,.2f}
Buying Power:    ${float(account.buying_power):,.2f}""")
        except Exception as e:
            log.warning(f"Could not fetch account for scan brief: {e}")

    # ── Macro Environment ────────────────────────────────────────────────────
    macro_mod = _safe_import("ingestion.macro")
    if macro_mod:
        try:
            sections.append(_get_macro_summary(macro_mod))
        except Exception as e:
            log.warning(f"Macro data unavailable: {e}")

    # ── Catalyst Calendar ────────────────────────────────────────────────────
    cal_section = _catalyst_calendar_section(ticker)
    if cal_section:
        sections.append(cal_section)

    # ── Technical Picture ────────────────────────────────────────────────────
    polygon_mod = _safe_import("ingestion.polygon_stream")
    if polygon_mod:
        try:
            sections.append(polygon_mod.get_technicals_summary(ticker))
        except Exception as e:
            log.warning(f"Technicals unavailable for {ticker}: {e}")

    # ── Options Flow ─────────────────────────────────────────────────────────
    options_mod = _safe_import("ingestion.options_flow")
    if options_mod:
        try:
            flow = options_mod.get_flow_summary(ticker)
            sections.append(f"""=== OPTIONS FLOW — {ticker} ===
{flow['summary']}""")
        except Exception as e:
            log.warning(f"Options flow unavailable: {e}")

    # ── Fundamentals ─────────────────────────────────────────────────────────
    position_section = _position_context_section(ticker, current_price)
    if position_section:
        sections.append(position_section)

    price_section = _price_structure_section(ticker, current_price)
    if price_section:
        sections.append(price_section)

    fund_section = _fundamentals_section(ticker)
    if fund_section:
        sections.append(fund_section)

    # ── Semantic Memory (ChromaDB) ────────────────────────────────────────────
    try:
        import vector_store
        query = f"{ticker} recent news developments catalysts"
        history = vector_store.search_all_collections(query, n_results=4)
        if history:
            mem_lines = ["=== RELEVANT HISTORICAL CONTEXT (from memory) ==="]
            for h in history:
                score = h["score"]
                source = h.get("collection", "unknown")
                meta = h.get("metadata", {})
                date = meta.get("date", meta.get("timestamp", ""))
                mem_lines.append(
                    f"[{source.upper()} | score={score:.2f} | {date[:10]}]\n"
                    f"{h['text'][:400]}\n"
                )
            sections.append("\n".join(mem_lines))
    except Exception as e:
        log.warning(f"Vector store search failed for {ticker}: {e}")

    # ── Related Symbols ──────────────────────────────────────────────────────
    from config import SYMBOLS
    related = _find_related_symbols(ticker, SYMBOLS)
    if related:
        sections.append(f"""=== RELATED SYMBOLS IN WATCHLIST ===
{ticker} may have indirect effects on: {', '.join(related)}
Consider second-order impact on these positions.""")

    return "\n\n".join(s for s in sections if s)


def _find_related_symbols(ticker: str, all_symbols: list[str]) -> list[str]:
    """
    Returns symbols that are likely affected by events in ticker.
    Based on supply chain, sector, and competitive relationships.
    """
    relationships: dict[str, list[str]] = {
        # Semiconductor supply chain
        "ASML":  ["NVDA", "AMD", "INTC", "IONQ", "QBTS", "RGTI"],
        "AMAT":  ["NVDA", "AMD", "MU", "IONQ"],
        "LRCX":  ["NVDA", "AMD", "MU"],
        "KLAC":  ["NVDA", "AMD", "MU"],
        "MU":    ["NVDA", "AMD", "QCOM"],
        # AI hyperscaler effects
        "NVDA":  ["AMD", "AVGO", "AMAT", "MU", "IONQ"],
        "MSFT":  ["NVDA", "PLTR", "NOW", "CRM", "NET"],
        # NOTE: watchlist.md lists Alphabet as "GOOG" (not "GOOGL") — this key
        # must match that exactly, or the relationship can never fire since
        # `related` is filtered against config.SYMBOLS/watchlist tickers below.
        "GOOG":  ["NVDA", "META", "NET", "SNOW"],
        "AMZN":  ["NVDA", "NET", "SNOW", "DDOG"],
        # Quantum ecosystem
        "IONQ":  ["RGTI", "QBTS", "QUBT", "ARQQ", "IBM"],
        "RGTI":  ["IONQ", "QBTS", "QUBT"],
        "QBTS":  ["IONQ", "RGTI", "QUBT"],
        # Macro-sensitive
        "META":  ["GOOG", "SNAP", "PINS"],
    }
    related = relationships.get(ticker, [])
    return [s for s in related if s in all_symbols]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with a fake filing event
    test_event = {
        "source":  "sec_filing",
        "ticker":  "IONQ",
        "form":    "8-K",
        "title":   "IonQ Announces Strategic Partnership with US Department of Defense",
        "text":    "IonQ Inc. today announced a contract worth $54.5 million with the "
                   "United States Department of Defense to provide quantum computing "
                   "services across multiple branches of the armed forces...",
        "url":     "https://www.sec.gov/Archives/example",
    }
    test_positions = {
        "IONQ": {"qty": 100, "entry_price": 14.20, "current_price": 15.10, "pl_pct": 0.063, "market_value": 1510},
        "NVDA": {"qty": 5,   "entry_price": 890.0, "current_price": 910.0, "pl_pct": 0.022, "market_value": 4550},
    }
    brief = assemble_brief(test_event, test_positions)
    print(brief)
