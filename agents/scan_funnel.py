"""Record only: the daily scan's feasibility funnel by watchlist tier.

One activity row (stage "scan_tier_funnel") per tasks.run_daily_scan pass,
written after the analyst enqueue. For each watchlist tier (Core, Watch,
Speculative; "unknown" when a name has none) it counts the names that went
in, how many each pre-analyst filter dropped, and how many were sent to the
analyst. Counts only: no tickers, scores or sizes. Nothing reads the row to
decide; the scan, its filters, the selection and what the analyst sees are
unchanged. Off (config.SCAN_TIER_FUNNEL_RECORD_ENABLED): no row, no counting.

Stages, in funnel order, each dropped at one place:
  watchlist_in            the watchlist names the scan loop read
  regime_suppressed       all of them, in a Strong Risk-Off regime (run_scan
                          returns before scoring)
  no_price_data           no completed bars (run_scan)
  earnings_blackout       earnings within EARNINGS_ENTRY_BLACKOUT_DAYS (run_scan)
  below_floor             priority PASS, composite below 25 (run_scan)
  below_regime_threshold  priority LOW: dropped by run_daily_scan's HIGH/MEDIUM
                          filter
  over_run_cap            HIGH/MEDIUM past max_per_run
                          (crew._select_for_analysis, as process_candidates)
  sent_to_analyst         handed to the analyst queue for this pass
  unclassified            a name the scan result doesn't place (expected 0)
Per tier: watchlist_in = every drop + sent_to_analyst + unclassified.
"Sent" means queued for this pass: whether the LLM worker ran each one is
recorded per ticker already (NO_VERDICT, ANALYST_FAILED rows).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

STAGE = "scan_tier_funnel"
DROP_STAGES = ("regime_suppressed", "no_price_data", "earnings_blackout", "below_floor",
               "below_regime_threshold", "over_run_cap")
FUNNEL_KEYS = ("watchlist_in", *DROP_STAGES, "sent_to_analyst", "unclassified")
TIER_ORDER = ("Core", "Watch", "Speculative")
SUPPRESSED_REGIME = "Strong Risk-Off"
_SKIPS = {"no_price_data", "earnings_blackout", "below_floor"}


def _empty() -> dict:
    return {**{k: 0 for k in FUNNEL_KEYS}, "sent_high": 0, "sent_medium": 0}


def _tier_order(tiers) -> list:
    known = [t for t in TIER_ORDER if t in tiers]
    rest = sorted(t for t in tiers if t not in TIER_ORDER and t != "unknown")
    return known + rest + (["unknown"] if "unknown" in tiers else [])


def tier_funnel(watchlist_tiers: dict, scan_result: dict, sent: list[dict],
                suppressed: bool = False) -> dict:
    """{"by_tier": {tier: counts}, "total": counts, "not_in_watchlist": n}.
    watchlist_tiers: {ticker: tier or None}, the names the scan read.
    scan_result: run_scan's packet (candidates, disqualified with "skip").
    sent: the candidates selected for the analyst. suppressed: the scan
    returned before scoring (Strong Risk-Off). Pure."""
    stage: dict = {}
    if not suppressed:
        for d in scan_result.get("disqualified") or []:
            skip = d.get("skip")
            stage[d.get("ticker")] = skip if skip in _SKIPS else "unclassified"
        for c in scan_result.get("candidates") or []:
            pr = c.get("priority")
            stage[c.get("ticker")] = ("over_run_cap" if pr in ("HIGH", "MEDIUM")
                                      else "below_regime_threshold" if pr == "LOW"
                                      else "unclassified")
        for c in sent:
            stage[c.get("ticker")] = "sent_to_analyst"
    priority = {c.get("ticker"): c.get("priority") for c in sent}
    by_tier: dict = {}
    for ticker, tier in watchlist_tiers.items():
        t = by_tier.setdefault(tier or "unknown", _empty())
        t["watchlist_in"] += 1
        s = "regime_suppressed" if suppressed else stage.get(ticker, "unclassified")
        t[s] += 1
        if s == "sent_to_analyst":
            t["sent_high" if priority.get(ticker) == "HIGH" else "sent_medium"] += 1
    ordered = {k: by_tier[k] for k in _tier_order(by_tier)}
    total = _empty()
    for t in ordered.values():
        for k in total:
            total[k] += t[k]
    for counts in (*ordered.values(), total):
        counts["identity_ok"] = counts["watchlist_in"] == sum(
            counts[k] for k in (*DROP_STAGES, "sent_to_analyst", "unclassified"))
    not_in_watchlist = sum(1 for tk in stage if tk not in watchlist_tiers)
    return {"by_tier": ordered, "total": total, "not_in_watchlist": not_in_watchlist}


def _suppressed_tiers() -> tuple[dict, str | None]:
    """The watchlist's tiers when run_scan returned before reading it
    (Strong Risk-Off). Read only."""
    try:
        from knowledge_loader import load_watchlist
        return {t: e.get("tier") for t, e in load_watchlist().items()}, None
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


def build_row(scan_result: dict, high_medium: list[dict], regime: dict | None,
              max_per_run: int, enqueued: bool, task_id=None, attempt: int = 0,
              now: datetime | None = None) -> dict:
    """The row's extra_json. Raises only on a programming error; record()
    catches."""
    import config
    import db
    from agents.crew import _select_for_analysis
    from agents.scanner import _regime_threshold
    now = now or datetime.now(timezone.utc)
    regime = regime or {}
    name = scan_result.get("regime_context") or regime.get("regime") or "Neutral"
    notes = str(scan_result.get("scan_notes") or "")
    watchlist_error = notes if notes.startswith("Watchlist load failed") else None
    suppressed = name == SUPPRESSED_REGIME and "watchlist_tiers" not in scan_result
    if suppressed:
        tiers, err = _suppressed_tiers()
        watchlist_error = watchlist_error or err
    else:
        tiers = scan_result.get("watchlist_tiers") or {}
    sent = _select_for_analysis(high_medium, max_per_run)
    funnel = tier_funnel(tiers, scan_result, sent, suppressed=suppressed)
    from zoneinfo import ZoneInfo
    day = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    return {
        "env": db.TRADING_ENV,
        "day": day,
        "shadow_key": f"{db.TRADING_ENV}:{day}:{task_id}:{attempt}",
        "computed_at": now.isoformat(),
        "regime": name,
        "regime_score": regime.get("score"),
        "threshold": _regime_threshold(name),
        "max_per_run": max_per_run,
        "earnings_blackout_days": getattr(config, "EARNINGS_ENTRY_BLACKOUT_DAYS", None),
        "enqueue": "queued" if enqueued else "none",
        "watchlist_error": watchlist_error,
        "stages": list(FUNNEL_KEYS),
        "by_tier": funnel["by_tier"],
        "total": funnel["total"],
        "not_in_watchlist": funnel["not_in_watchlist"],
    }


def record(scan_result: dict, high_medium: list[dict], regime: dict | None, max_per_run: int,
           enqueued: bool, task_id=None, attempt: int = 0) -> bool:
    """Write the row; True once written. Off: nothing computed. Never raises."""
    try:
        import config
        if not getattr(config, "SCAN_TIER_FUNNEL_RECORD_ENABLED", False):
            return False
        import db
        extra = build_row(scan_result, high_medium, regime, max_per_run, enqueued,
                          task_id=task_id, attempt=attempt)
        tot = extra["total"]
        db.log_activity(STAGE, "",
                        f"scan tier funnel: {tot['watchlist_in']} in, {tot['sent_to_analyst']} "
                        f"sent to the analyst ({extra['regime']})", "neutral", extra)
        return True
    except Exception as e:
        log.warning(f"scan tier funnel row failed: {type(e).__name__}: {e}")
        return False
