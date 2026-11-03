"""Records for daily-scan work that never reaches routing.

ROUTE_NO_BROKER: the broker client or its position read failed when the
scan started, so candidates were analysed against an empty book. Their
verdicts are still logged by analyze(); passes are not routed, and each
gets a ROUTE_NO_BROKER row instead of vanishing.

NO_VERDICT: a scan candidate whose analysis was lost (analyze_batch raised,
or the candidate came back with no result), so "the scan produced nothing"
can be told apart from "the scan died". An analyze() exception inside the
batch is not one: analyze() itself writes an ANALYST_FAILED row for it.

Both kinds are in db._NON_ANALYST_SIGNALS, so neither reads as a verdict.
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_NO_BROKER_ALERT_KEY = "scan_no_broker_alerted_on"


def log_no_broker_passes(results: list[dict], broker_error: str) -> int:
    """A ROUTE_NO_BROKER row per passed verdict that is not routed. Returns
    how many were written (attempted)."""
    from agents.crew import _log_route_exit
    n = 0
    for result in results or []:
        if not result.get("pass_to_risk"):
            continue
        ticker = result.get("ticker") or "UNKNOWN"
        log.warning(f"[{ticker}] Scan pass not routed: no broker ({broker_error})")
        _log_route_exit(ticker, "ROUTE_NO_BROKER", result,
                        f"broker unavailable at scan start, not routed: {broker_error}")
        n += 1
    return n


def alert_scan_without_broker(broker_error: str, dropped_passes: int, now=None) -> bool:
    """One alert per ET day that a scan ran without a broker. Returns True
    when it sent one. Never raises."""
    try:
        import db
        today = (now or datetime.now(timezone.utc)).astimezone(_ET).date().isoformat()
        if db.get_state(_NO_BROKER_ALERT_KEY) == today:
            return False
        import alerts
        sent = alerts._send(
            f"⚠️ <b>Daily scan ran without a broker</b>: {broker_error[:200]}\n"
            f"Candidates were analysed against an empty book and no pass was routed "
            f"({dropped_passes} pass(es) logged as ROUTE_NO_BROKER). "
            f"Further scans today won't alert again."
        )
        if not sent:
            return False
        db.set_state(_NO_BROKER_ALERT_KEY, today)
        return True
    except Exception as e:
        log.warning(f"scan-without-broker alert failed: {e}")
        return False


def _lost_reason(candidate: dict, by_ticker: dict) -> str | None:
    result = by_ticker.get(candidate.get("ticker"))
    if result is None:
        return "no result returned for candidate"
    # A stub (model failure or analyze() exception) already has its
    # ANALYST_FAILED row from analyze(); a NO_VERDICT would be a second row.
    return None


def log_no_verdict(ticker: str, reason: str, candidate: dict | None = None) -> None:
    """One NO_VERDICT signals row. Never raises."""
    try:
        import db
        db.log_signal(ticker, "NO_VERDICT", 0.0, reason[:500],
                      provenance={"trigger": {"path": "daily_scan", "source": "scanner"}},
                      decision={"no_verdict": reason[:500],
                                "priority": (candidate or {}).get("priority"),
                                "composite_score": (candidate or {}).get("composite_score")})
    except Exception as e:
        log.warning(f"[{ticker}] NO_VERDICT signal log failed: {e}")


def log_lost_candidates(candidates: list[dict], results: list[dict] | None,
                        batch_error: str | None = None) -> int:
    """NO_VERDICT rows for scan candidates whose analysis was lost. With
    batch_error every candidate is lost. Returns the number of rows."""
    by_ticker = {r.get("ticker"): r for r in (results or [])}
    n = 0
    for c in candidates or []:
        ticker = c.get("ticker") or "UNKNOWN"
        reason = (f"analyze_batch raised: {batch_error}" if batch_error is not None
                  else _lost_reason(c, by_ticker))
        if reason:
            log_no_verdict(ticker, reason, c)
            n += 1
    return n
