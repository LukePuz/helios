"""Story de-duplication and staleness shadow (record only).

The limitations audit of 2026-11-21 15:33 (finding 8) found the same story
analysed more than once on one ticker (one SEC-probe article on NVDA, re-titled
by its publisher, reached the 72B twice) and items 21-36 h old triggering
verdicts. The ingestion de-dup (seen_items: per-source id plus a normalised
title hash) misses a re-titled article and anything past the weekly prune.

For each event analysis, crew.deep_analyze calls story_dedup_record just
before the 72B call (the point where a live skip would save the GPU time) and
stores the result on the verdict's provenance.trigger.story_dedup:

- story_key: "<domain>|<normalised title>". The title is lower-cased and
  every run of non-word characters collapsed to one space. Only on a
  news.google.com link, whose titles always end in " - Publisher", is that
  suffix removed and used as the domain ("gn:<publisher>"); elsewhere a
  trailing " - ..." is part of the headline ("X Q3 - revenue beats" and
  "X Q3 - revenue misses" stay different stories). Otherwise the domain is
  the URL host without "www.", or the event source with no URL.
- duplicate_of: the newest earlier event verdict on the same ticker within
  STORY_DEDUP_WINDOW_HOURS whose URL (host and path; scheme, "www.", query
  and fragment dropped) or story_key is the same:
  {signal_id, timestamp, hours_before, match: "url" | "story_key",
  title_changed}. title_changed is true when the earlier verdict's
  normalised title differs (a re-titled or live-updated article at one URL,
  which can carry new facts).
- published_age_hours: decision time minus the item's own time
  (trigger.event_time, the first structured time key the source carries).
  A date-only time (a congressional trade date, an older Form 4
  filing_date) has no hour, so published_age_hours is None with
  published_age_reason "date_only", and published_age_hours_min is the
  lower bound measured from 23:59:59 ET that day. would_skip_stale is then
  True only when even that lower bound is past the cut-off, else None.
- would_skip_duplicate / would_skip_stale: labels at the cut-offs in
  config.py. None means "could not say" (read failed, no title, no time).

A re-analysis the system queues on purpose (source stale_entry_recheck) is
marked exempt and labelled False. Nothing here is read by the prompt, the
brief, the gates or routing. Never raises.
"""

import logging
import re
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

STORY_DEDUP_VERSION = 1

# Event sources that are deliberate re-analyses, never duplicates.
_EXEMPT_SOURCES = ("stale_entry_recheck",)

_PUBLISHER_SUFFIX = re.compile(r"\s+[-–—|]\s+([^-–—|]{2,40})$")


_ET = ZoneInfo("America/New_York")
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _split_publisher(title: str) -> tuple[str, str | None]:
    """(title without a trailing ' - Publisher', that publisher or None)."""
    title = (title or "").strip()
    m = _PUBLISHER_SUFFIX.search(title)
    if not m:
        return title, None
    return title[:m.start()], m.group(1).strip()


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_google_news(url: str | None) -> bool:
    return _host(url) == "news.google.com"


def normalise_title(title: str | None, strip_publisher: bool = False) -> str | None:
    """Lower-cased, punctuation collapsed; the trailing " - Publisher" is
    removed only when strip_publisher (a news.google.com title)."""
    if not title:
        return None
    body = _split_publisher(str(title))[0] if strip_publisher else str(title)
    norm = re.sub(r"[\W_]+", " ", body.lower()).strip()
    return norm or None


def story_domain(url: str | None, title: str | None, source: str | None) -> str | None:
    host = _host(url)
    if host == "news.google.com":
        _, publisher = _split_publisher(str(title or ""))
        if publisher:
            return "gn:" + re.sub(r"[\W_]+", " ", publisher.lower()).strip()
    if host:
        return host
    return str(source).lower() if source else None


def story_key(title: str | None, url: str | None, source: str | None) -> str | None:
    norm = normalise_title(title, strip_publisher=is_google_news(url))
    if norm is None:
        return None
    return f"{story_domain(url, title, source) or 'unknown'}|{norm}"[:240]


def url_key(url: str | None) -> str | None:
    """Host and path of the URL: scheme, "www.", query, fragment and a
    trailing slash dropped (a Google News id is in the path, so it stays)."""
    if not url:
        return None
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    return f"{host}{parts.path.rstrip('/')}"


def parse_time(value) -> datetime | None:
    """A stored or event time as an aware UTC datetime, or None."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            s = str(value).strip()
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def time_precision(value) -> str | None:
    """"date" for a date-only value (a date, or a bare YYYY-MM-DD string),
    "datetime" for anything else parse_time reads, None when unreadable."""
    if parse_time(value) is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return "date"
    if not isinstance(value, datetime) and _DATE_ONLY.fullmatch(str(value).strip()):
        return "date"
    return "datetime"


def classify(trigger: dict, prior: list[dict], decided_at: datetime,
             window_hours: float, stale_cutoff_hours: float) -> dict:
    """The record for one analysis from its trigger and the earlier event
    verdicts on the ticker (each {signal_id, timestamp, title, url, source}).
    Pure; the backfill calls it on stored rows. prior rows outside the window
    or not before decided_at are ignored."""
    t = trigger or {}
    source = t.get("source")
    key = story_key(t.get("title"), t.get("url"), source)
    ukey = url_key(t.get("url"))
    exempt = "recheck" if source in _EXEMPT_SOURCES else None

    matches = []
    for p in prior or []:
        ts = parse_time(p.get("timestamp"))
        if ts is None or ts >= decided_at:
            continue
        hours_before = (decided_at - ts).total_seconds() / 3600.0
        if hours_before > window_hours:
            continue
        if ukey and url_key(p.get("url")) == ukey:
            match = "url"
        elif key and story_key(p.get("title"), p.get("url"), p.get("source")) == key:
            match = "story_key"
        else:
            continue
        matches.append((ts, hours_before, match, p))
    matches.sort(key=lambda m: m[0], reverse=True)
    dup = None
    if matches:
        ts, hours_before, match, p = matches[0]
        dup = {"signal_id": p.get("signal_id"), "timestamp": ts.isoformat(timespec="seconds"),
               "hours_before": round(hours_before, 2), "match": match,
               "title_changed": (normalise_title(p.get("title"), is_google_news(p.get("url")))
                                 != normalise_title(t.get("title"), is_google_news(t.get("url"))))}

    raw_time = t.get("event_time") or t.get("published_at")
    published, precision = parse_time(raw_time), time_precision(raw_time)
    age = age_min = age_reason = None
    if published is None:
        age_reason = "no_item_time"
    elif precision == "date":
        # No hour: the item is at least this old (from the end of that ET day).
        end = datetime.combine(published.date(), time(23, 59, 59), tzinfo=_ET)
        age_min = round((decided_at - end).total_seconds() / 3600.0, 2)
        age_reason = "date_only"
    else:
        age = round((decided_at - published).total_seconds() / 3600.0, 2)

    if exempt:
        would_dup, would_stale = False, False
    else:
        would_dup = (dup is not None) if (key or ukey) else None
        if age is not None:
            would_stale = age > stale_cutoff_hours
        elif age_min is not None and age_min > stale_cutoff_hours:
            would_stale = True
        else:
            would_stale = None
    return {"version": STORY_DEDUP_VERSION, "story_key": key, "url_key": ukey,
            "window_hours": window_hours, "prior_event_verdicts": len(prior or []),
            "duplicate_of": dup, "duplicate_count": len(matches),
            "published_age_hours": age, "published_age_hours_min": age_min,
            "published_age_reason": age_reason, "published_time_precision": precision,
            "published_time_source": t.get("event_time_source") if published else None,
            "stale_cutoff_hours": stale_cutoff_hours,
            "would_skip_duplicate": would_dup, "would_skip_stale": would_stale,
            "exempt": exempt, "decided_at": decided_at.isoformat(timespec="seconds")}


def story_dedup_record(ticker: str, trigger: dict | None) -> dict | None:
    """The live record for an event analysis about to reach the 72B. None
    when the shadow is off. Reads the earlier event verdicts once
    (db.recent_event_verdicts, strict). On a failed read the labels that
    need it are None and read says why. Never raises."""
    try:
        import config
        if not getattr(config, "STORY_DEDUP_SHADOW_ENABLED", False):
            return None
        window = float(config.STORY_DEDUP_WINDOW_HOURS)
        cutoff = float(config.STORY_STALE_CUTOFF_HOURS)
        decided_at = datetime.now(timezone.utc)
        read = "ok"
        try:
            import db
            prior = db.recent_event_verdicts(ticker, int(window * 60), strict=True)
        except Exception as e:
            prior, read = [], f"error: {type(e).__name__}"
        rec = classify(trigger or {}, prior, decided_at, window, cutoff)
        rec["read"] = read
        if read != "ok":
            rec["duplicate_of"], rec["duplicate_count"] = None, None
            rec["prior_event_verdicts"] = None
            if not rec["exempt"]:
                rec["would_skip_duplicate"] = None
        return rec
    except Exception as e:
        log.debug(f"[{ticker}] story dedup shadow failed: {e}")
        try:
            return {"version": STORY_DEDUP_VERSION, "read": f"error: {type(e).__name__}"}
        except Exception:
            return None
