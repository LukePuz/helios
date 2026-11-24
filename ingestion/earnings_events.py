"""Per-event earnings record (record only, for research).

One activity_feed row (stage earnings_event_record) per earnings event and
phase, built from the snapshot the weekday 17:30 ET consensus fetch already
took (yfinance Ticker.calendar, earnings_estimate / revenue_estimate and
earnings_history; no other call, no other vendor):

  kind "pre"   when a ticker's next announcement date is first seen: the
               date, whether the source gives it as an estimated window
               (and the window's dates), time of day, the source's
               consensus EPS and revenue for the event, the quarter the
               estimate is for, and the newest quarter already reported.
  kind "post"  when the actual for that event appears in earnings_history:
               actual EPS, the source's own estimate and difference, and its
               surprisePercent as given (Yahoo's raw value, not converted).

shadow_key {env}:{ticker}:{announce_date}:{pre|post} is the dedupe key: a
key already written is never written again, and no row is ever updated.

A date the source revises before the event is a new pre row with
revision {previous_announce_date, previous_shadow_key, ...}; the earlier row
stays. A different next date is a revision while the latest open event's
date is still after today; once that date is today or past, a new date is a
new event (the source moves on to the next quarter, often the same evening).
A revision back to a date already written finds its key taken and writes
nothing (counted as revision_to_seen_date). An estimated window (the source
gives two dates) is one pre row on its first date, with the window's dates.

The post row closes the event chain (its pre rows, linked by revision) on
the latest pre row's date, once that date is today or past. Its quarter is
the newest earnings_history quarter with an actual EPS that is newer than
the chain's last_reported_quarter, ends before the announce date, and no
earlier post claimed (matched_by "newest_after_last_reported"); without a
last_reported_quarter, the quarter equal to estimate_quarter_end
(matched_by "estimate_quarter_end"); with neither, no post.

A value the source does not give is None, with the reason in missing
{field: reason}; nothing is estimated or filled in. No rule, metric or
horizon is applied: nothing here judges a surprise or an event window.
Nothing the analyst, the brief, the gates or the order path reads uses it.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone

log = logging.getLogger(__name__)

STAGE = "earnings_event_record"
SOURCE = "yfinance"
TIME_OF_DAY_REASON = "the source gives dates only (yfinance calendar earningsDate)"
REVENUE_ACTUAL_REASON = "the source gives no revenue actual (earnings_history is EPS only)"
PRIOR_DAYS = 200                 # pre rows are read back this far for open events


def shadow_key(env: str, ticker: str, announce_date: str, phase: str) -> str:
    return f"{env}:{ticker}:{announce_date}:{phase}"


def _iso(v) -> str | None:
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, str) and v:
        return v[:10]
    return None


def _num(v):
    """A finite int/float as given, else None (bools are not numbers)."""
    import math
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v if math.isfinite(v) else None


def _part_reason(rec: dict, part: str, what: str) -> str:
    status = (rec.get("parts") or {}).get(part)
    if status is None:
        return f"{what}: {part} not read"
    if status != "ok":
        return f"{what}: {part} {status}"
    return f"{what}: not in the source response"


def _value(src: dict | None, key: str, rec: dict, part: str, field: str, missing: dict):
    v = _num((src or {}).get(key))
    if v is None:
        missing[field] = _part_reason(rec, part, f"{key} absent or not a finite number")
    return v


def _newest_reported(history) -> str | None:
    quarters = [_iso(h.get("quarter")) for h in history or ()
                if isinstance(h, dict) and _num(h.get("epsActual")) is not None]
    quarters = [q for q in quarters if q]
    return max(quarters) if quarters else None


def pre_row(env: str, ticker: str, rec: dict, as_of: datetime) -> dict | None:
    """The pre row for rec's next announcement, or None when rec has no date."""
    announce = _iso(rec.get("next_earnings_date"))
    if not announce:
        return None
    cal = rec.get("calendar_event") if isinstance(rec.get("calendar_event"), dict) else {}
    missing: dict = {}
    eps0 = ((rec.get("eps_estimate") or {}).get("0q")) or {}
    rev0 = ((rec.get("revenue_estimate") or {}).get("0q")) or {}
    row = {
        "kind": "pre", "env": env, "ticker": ticker, "source": SOURCE,
        "shadow_key": shadow_key(env, ticker, announce, "pre"),
        "announce_date": announce,
        "announce_date_is_estimate": rec.get("next_earnings_is_estimate"),
        "window_dates": list(cal.get("dates") or []) or None,
        "time_of_day": None,
        "consensus_eps": _value(cal, "eps_avg", rec, "calendar", "consensus_eps", missing),
        "consensus_eps_low": _value(cal, "eps_low", rec, "calendar", "consensus_eps_low", missing),
        "consensus_eps_high": _value(cal, "eps_high", rec, "calendar", "consensus_eps_high",
                                     missing),
        "consensus_revenue": _value(cal, "rev_avg", rec, "calendar", "consensus_revenue", missing),
        "consensus_revenue_low": _value(cal, "rev_low", rec, "calendar", "consensus_revenue_low",
                                        missing),
        "consensus_revenue_high": _value(cal, "rev_high", rec, "calendar",
                                         "consensus_revenue_high", missing),
        "estimate_period": "0q",
        "estimate_quarter_end": _iso(eps0.get("end_date")) or _iso(rev0.get("end_date")),
        "trend_eps_avg": _value(eps0, "avg", rec, "eps_estimate", "trend_eps_avg", missing),
        "trend_eps_analysts": _value(eps0, "numberOfAnalysts", rec, "eps_estimate",
                                     "trend_eps_analysts", missing),
        "trend_revenue_avg": _value(rev0, "avg", rec, "revenue_estimate", "trend_revenue_avg",
                                    missing),
        "trend_revenue_analysts": _value(rev0, "numberOfAnalysts", rec, "revenue_estimate",
                                         "trend_revenue_analysts", missing),
        "last_reported_quarter": _newest_reported(rec.get("eps_history")),
        "revision": None,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
        "parts": {k: v for k, v in (rec.get("parts") or {}).items()},
    }
    missing["time_of_day"] = TIME_OF_DAY_REASON
    if row["window_dates"] is None:
        missing["window_dates"] = _part_reason(rec, "calendar", "earningsDate list")
    if row["estimate_quarter_end"] is None:
        missing["estimate_quarter_end"] = _part_reason(rec, "eps_estimate", "0q endDate")
    if row["last_reported_quarter"] is None:
        missing["last_reported_quarter"] = _part_reason(rec, "eps_history",
                                                        "no quarter with an actual")
    row["missing"] = missing
    return row


def _chains(pres: list[dict]) -> list[list[dict]]:
    """The pre rows of one ticker grouped into events, oldest row first: a
    row with a revision joins the chain whose latest row it revised."""
    chains: list[list[dict]] = []
    for p in sorted(pres, key=lambda r: r.get("as_of") or ""):
        prev = (p.get("revision") or {}).get("previous_shadow_key")
        for c in chains:
            if prev and c[-1].get("shadow_key") == prev:
                c.append(p)
                break
        else:
            chains.append([p])
    return chains


def _match_actual(chain: list[dict], history, claimed: set) -> tuple[dict | None, str | None]:
    """The earnings_history row for the event: with the chain's
    last_reported_quarter known, the newest quarter with an actual that is
    newer than it, ends before the announce date and no earlier post
    claimed; without it, the quarter equal to estimate_quarter_end. None
    when neither is known or nothing qualifies yet."""
    announce = chain[-1].get("announce_date") or ""
    rows = [h for h in history or () if isinstance(h, dict)
            and _num(h.get("epsActual")) is not None and _iso(h.get("quarter"))
            and _iso(h.get("quarter")) not in claimed and _iso(h.get("quarter")) < announce]
    last = next((p.get("last_reported_quarter") for p in reversed(chain)
                 if p.get("last_reported_quarter")), None)
    if last:
        newer = sorted((h for h in rows if _iso(h["quarter"]) > last),
                       key=lambda h: _iso(h["quarter"]), reverse=True)
        return (newer[0], "newest_after_last_reported") if newer else (None, None)
    q_end = next((p.get("estimate_quarter_end") for p in reversed(chain)
                  if p.get("estimate_quarter_end")), None)
    if q_end:
        hit = [h for h in rows if _iso(h["quarter"]) == q_end]
        return (hit[0], "estimate_quarter_end") if hit else (None, None)
    return None, None


def post_row(env: str, ticker: str, chain: list[dict], hist: dict, matched_by: str,
             as_of: datetime) -> dict:
    latest = chain[-1]
    missing = {"revenue_actual": REVENUE_ACTUAL_REASON,
               "revenue_surprise": REVENUE_ACTUAL_REASON}
    row = {
        "kind": "post", "env": env, "ticker": ticker, "source": SOURCE,
        "shadow_key": shadow_key(env, ticker, latest["announce_date"], "post"),
        "announce_date": latest["announce_date"],
        "pre_shadow_key": latest.get("shadow_key"),
        "pre_announce_dates": [p.get("announce_date") for p in chain],
        "quarter": _iso(hist.get("quarter")),
        "matched_by": matched_by,
        "eps_actual": _num(hist.get("epsActual")),
        "eps_estimate_source": _num(hist.get("epsEstimate")),
        "eps_difference_source": _num(hist.get("epsDifference")),
        "surprise_percent_source": _num(hist.get("surprisePercent")),
        "surprise_percent_units": "as given by the source (Yahoo surprisePercent, not converted)",
        "revenue_actual": None, "revenue_surprise": None,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
    }
    for field, key in (("eps_estimate_source", "epsEstimate"),
                       ("eps_difference_source", "epsDifference"),
                       ("surprise_percent_source", "surprisePercent")):
        if row[field] is None:
            missing[field] = f"{key} absent or not a finite number in earnings_history"
    row["missing"] = missing
    return row


def build_rows(env: str, ticker: str, rec: dict, prior: list[dict], written: set[str],
               as_of: datetime, today: str, counts: dict | None = None) -> list[dict]:
    """New rows for one ticker: posts for the open events whose date has
    passed and whose actual has appeared (latest date first), then the pre
    row for rec's next date if its key is new. A different next date while
    the latest open event's date is still ahead (after today) is a revision
    of that event; once its date is today or past, a new date is a new
    event. prior: this ticker's earlier rows (pre and post), written: every
    shadow_key already written. Pure: reads and writes nothing."""
    counts = counts if counts is not None else {}
    out: list[dict] = []
    pres = [p for p in prior if p.get("kind") == "pre"]
    posts = [p for p in prior if p.get("kind") == "post"]
    posted = {p.get("announce_date") for p in posts}
    claimed = {p.get("quarter") for p in posts if p.get("quarter")}
    open_chains = [c for c in _chains(pres)
                   if not any(p.get("announce_date") in posted for p in c)]
    for chain in sorted((c for c in open_chains if (c[-1].get("announce_date") or "") <= today),
                        key=lambda c: c[-1].get("announce_date") or "", reverse=True):
        hist, by = _match_actual(chain, rec.get("eps_history"), claimed)
        if hist is None:
            continue
        row = post_row(env, ticker, chain, hist, by, as_of)
        if row["shadow_key"] not in written:
            out.append(row)
            written.add(row["shadow_key"])
        claimed.add(row["quarter"])
        open_chains.remove(chain)
    pre = pre_row(env, ticker, rec, as_of)
    if pre is None:
        counts["no_date"] = counts.get("no_date", 0) + 1
        return out
    upcoming = [c for c in open_chains if (c[-1].get("announce_date") or "") > today]
    prev = max((c[-1] for c in upcoming), key=lambda p: p.get("as_of") or "", default=None)
    if pre["shadow_key"] in written:
        if prev is not None and prev.get("announce_date") != pre["announce_date"]:
            counts["revision_to_seen_date"] = counts.get("revision_to_seen_date", 0) + 1
        return out
    if prev is not None:
        pre["revision"] = {"previous_announce_date": prev.get("announce_date"),
                           "previous_shadow_key": prev.get("shadow_key"),
                           "previous_is_estimate": prev.get("announce_date_is_estimate"),
                           "previous_as_of": prev.get("as_of")}
    out.append(pre)
    written.add(pre["shadow_key"])
    return out


def record_events(recs: dict, env: str, read_prior=None, read_keys=None, write=None,
                  now: datetime | None = None, budget_s: float = 90.0,
                  clock=time.monotonic) -> dict:
    """Write the new rows for {ticker: consensus snapshot}. read_prior()
    returns earlier rows (both kinds) or None; read_keys() the written
    shadow_keys or None; write(row) writes one row and returns whether it
    landed. Unread prior rows or keys write nothing (a missed day, not a
    guess). Stops at budget_s. Never raises. Returns counts."""
    import db
    from zoneinfo import ZoneInfo
    counts = {"tickers": len(recs), "pre": 0, "post": 0, "revisions": 0, "no_date": 0,
              "revision_to_seen_date": 0, "write_failed": 0, "not_written_budget": 0,
              "read_failed": False}
    if not recs:
        return counts
    try:
        now = now or datetime.now(timezone.utc)
        today = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        if read_prior is None:
            def read_prior():
                rows = []
                for kind in ("pre", "post"):
                    got = db.get_activity_rows(STAGE, kind, days=PRIOR_DAYS)
                    if got is None:
                        return None
                    rows += got
                return rows
        if read_keys is None:
            def read_keys():
                return db.get_activity_shadow_keys(STAGE)
        if write is None:
            def write(row):
                tag = (f"{row['kind']} {row['announce_date']}"
                       + (" (date revised)" if row.get("revision") else ""))
                return db.log_activity_direct(STAGE, row["ticker"],
                                              f"earnings event record {tag}", "neutral", row)
        prior, written = read_prior(), read_keys()
        if prior is None or written is None:
            counts["read_failed"] = True
            log.warning("earnings event record: earlier rows unreadable; nothing written")
            return counts
        written = {k for k in written if k.startswith(f"{env}:")}
        by_ticker: dict = {}
        for r in prior:
            if r.get("env") == env and r.get("ticker"):
                by_ticker.setdefault(r["ticker"], []).append(r)
        start = clock()
        for ticker, rec in recs.items():
            if not isinstance(rec, dict):
                continue
            rows = build_rows(env, ticker, rec, by_ticker.get(ticker, []), written, now, today,
                              counts)
            for row in rows:
                if clock() - start >= budget_s:
                    counts["not_written_budget"] += 1
                    continue
                try:
                    ok = write(row)
                except Exception:
                    ok = False
                if ok:
                    counts[row["kind"]] += 1
                    if row.get("revision"):
                        counts["revisions"] += 1
                else:
                    counts["write_failed"] += 1
                    written.discard(row["shadow_key"])
        log.info(f"earnings event record: {counts}")
        if counts["write_failed"] or counts["not_written_budget"]:
            log.warning(f"earnings event record: {counts['write_failed']} write(s) failed, "
                        f"{counts['not_written_budget']} left for the next run (budget)")
    except Exception as e:
        log.warning(f"earnings event record failed: {type(e).__name__}: {e}")
    return counts
