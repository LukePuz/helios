"""
SQLite storage and reconciliation helpers for the catalyst calendar.

This module is intentionally usable without the FastAPI server running so
scheduled jobs can write locally, while other agents can later move to HTTP.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("CATALYST_DB_PATH", Path(__file__).resolve().parent.parent / "catalyst_calendar.db"))

# Ticker-level (original set, unchanged).
TICKER_EVENT_TYPES = {
    "earnings",
    "conference",
    "regulatory",
    "press",
    "indirect_readthrough",
    "insider_filing",
    "other",
}
# Sector-level (2026-08-30, user request): events that affect a whole GICS
# sector rather than one name -- e.g. "3+ constituents report earnings the
# same week" (see ingestion.sector_calendar), a sector-wide regulatory
# ruling, or an index rebalance that moves flows across an entire sector.
SECTOR_EVENT_TYPES = {
    "sector_earnings_cluster",
    "sector_regulatory",
    "index_rebalance",
    "sector_conference",
    "sector_other",
}
# Macro-level (2026-08-30, user request): market-wide, not tied to any one
# ticker or sector -- see ingestion.macro_calendar for the actual ingestion
# of these (FOMC schedule, BLS release calendar, election dates).
MACRO_EVENT_TYPES = {
    "fomc_meeting",
    "cpi_release",
    "jobs_report",
    "gdp_release",
    "election",
    "geopolitical",
    "macro_other",
}
EVENT_TYPES = TICKER_EVENT_TYPES | SECTOR_EVENT_TYPES | MACRO_EVENT_TYPES

# scope groups events by what they affect, independent of event_type (a
# sector can have an "other"-typed event same as a ticker can). Determines
# how `ticker` is interpreted: scope="ticker" -> the actual ticker symbol;
# scope="sector" -> the sector's SPDR ETF symbol (XLK, XLF, ...), same
# symbol agents.market_regime.SECTOR_ETF_MAP/get_trade_style_for_ticker
# already key sector regimes by, so a sector-scope catalyst and a
# sector-scope regime read line up on the same key; scope="macro" -> the
# fixed sentinel "MARKET" (not a real ticker). Kept as the existing
# `ticker` column rather than a new one so the pre-existing dedupe key
# (ticker, related_ticker, event_type, event_date) and indexes keep working
# unchanged across all three scopes.
SCOPES = {"ticker", "sector", "macro"}
MACRO_SENTINEL = "MARKET"

CONFIDENCE_LEVELS = {"confirmed", "estimated", "conflicting", "rumored"}
PRECISION_LEVELS = {"exact", "window", "estimated"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {})


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_add_scope(conn: sqlite3.Connection) -> None:
    """
    Adds the `scope` column to a pre-existing catalyst_events table (2026-
    08-30, user request: sector/macro catalysts alongside the original
    ticker-only ones). SQLite can't ALTER a CHECK constraint in place, and
    the original event_type CHECK only allowed the 7 ticker-level values --
    so this rebuilds the table with the constraint dropped entirely (Python-
    side _validate_event() already enforces EVENT_TYPES, so the DB-level
    CHECK was belt-and-suspenders, not the only guard) rather than needing
    another migration every time a new event_type is added later.

    Idempotent: no-ops if `scope` already exists. Existing rows all become
    scope='ticker' (accurate -- everything before this migration was a
    per-ticker earnings/thesis event).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(catalyst_events)").fetchall()}
    if not cols or "scope" in cols:
        return  # fresh DB (next CREATE TABLE handles it) or already migrated

    conn.executescript(
        """
        ALTER TABLE catalyst_events RENAME TO catalyst_events_pre_scope;

        CREATE TABLE catalyst_events (
            event_id              TEXT PRIMARY KEY,
            scope                 TEXT NOT NULL DEFAULT 'ticker'
                CHECK (scope IN ('ticker','sector','macro')),
            ticker                TEXT NOT NULL,
            related_ticker        TEXT,
            event_date            TEXT NOT NULL,
            event_date_precision  TEXT NOT NULL
                CHECK (event_date_precision IN ('exact','window','estimated')),
            window_start          TEXT,
            window_end            TEXT,
            event_type            TEXT NOT NULL,
            bucket_relevance      TEXT,
            confidence            TEXT NOT NULL
                CHECK (confidence IN ('confirmed','estimated','conflicting','rumored')),
            source_tier           INTEGER NOT NULL,
            sources               TEXT NOT NULL,
            thesis_note           TEXT,
            action_flags          TEXT NOT NULL,
            superseded_by         TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            last_verified_at      TEXT
        );

        INSERT INTO catalyst_events (
            event_id, scope, ticker, related_ticker, event_date, event_date_precision,
            window_start, window_end, event_type, bucket_relevance, confidence,
            source_tier, sources, thesis_note, action_flags, superseded_by,
            created_at, updated_at, last_verified_at
        )
        SELECT
            event_id, 'ticker', ticker, related_ticker, event_date, event_date_precision,
            window_start, window_end, event_type, bucket_relevance, confidence,
            source_tier, sources, thesis_note, action_flags, superseded_by,
            created_at, updated_at, last_verified_at
        FROM catalyst_events_pre_scope;

        DROP TABLE catalyst_events_pre_scope;
        """
    )
    conn.commit()


def init_db() -> None:
    conn = get_conn()
    try:
        _migrate_add_scope(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS catalyst_events (
                event_id              TEXT PRIMARY KEY,
                scope                 TEXT NOT NULL DEFAULT 'ticker'
                    CHECK (scope IN ('ticker','sector','macro')),
                ticker                TEXT NOT NULL,
                related_ticker        TEXT,
                event_date            TEXT NOT NULL,
                event_date_precision  TEXT NOT NULL
                    CHECK (event_date_precision IN ('exact','window','estimated')),
                window_start          TEXT,
                window_end            TEXT,
                event_type            TEXT NOT NULL,
                bucket_relevance      TEXT,
                confidence            TEXT NOT NULL
                    CHECK (confidence IN ('confirmed','estimated','conflicting','rumored')),
                source_tier           INTEGER NOT NULL,
                sources               TEXT NOT NULL,
                thesis_note           TEXT,
                action_flags          TEXT NOT NULL,
                superseded_by         TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL,
                last_verified_at      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_catalyst_ticker_date
                ON catalyst_events(ticker, event_date);
            CREATE INDEX IF NOT EXISTS idx_catalyst_confidence
                ON catalyst_events(confidence);
            CREATE INDEX IF NOT EXISTS idx_catalyst_related
                ON catalyst_events(related_ticker);
            CREATE INDEX IF NOT EXISTS idx_catalyst_scope_date
                ON catalyst_events(scope, event_date);

            CREATE TABLE IF NOT EXISTS thesis_adjacency_map (
                primary_ticker     TEXT NOT NULL,
                adjacent_ticker    TEXT NOT NULL,
                relationship_type  TEXT NOT NULL
                    CHECK (relationship_type IN (
                        'competitor','supplier','customer',
                        'sector_peer','index_proxy','platform_owner'
                    )),
                relevance_note     TEXT,
                PRIMARY KEY (primary_ticker, adjacent_ticker)
            );

            CREATE TABLE IF NOT EXISTS source_reliability (
                source_name             TEXT PRIMARY KEY,
                correct_predictions     INTEGER NOT NULL DEFAULT 0,
                incorrect_predictions   INTEGER NOT NULL DEFAULT 0,
                avg_days_off            REAL,
                last_updated            TEXT
            );

            CREATE TABLE IF NOT EXISTS catalyst_events_history (
                event_id              TEXT PRIMARY KEY,
                ticker                TEXT NOT NULL,
                related_ticker        TEXT,
                event_date            TEXT NOT NULL,
                event_date_precision  TEXT NOT NULL,
                window_start          TEXT,
                window_end            TEXT,
                event_type            TEXT NOT NULL,
                bucket_relevance      TEXT,
                confidence            TEXT NOT NULL,
                source_tier           INTEGER NOT NULL,
                sources               TEXT NOT NULL,
                thesis_note           TEXT,
                action_flags          TEXT NOT NULL,
                superseded_by         TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL,
                last_verified_at      TEXT,
                outcome_note          TEXT,
                archived_at           TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def reconcile(
    tier1_event: dict[str, Any] | None = None,
    filed_event: dict[str, Any] | None = None,
    aggregator_dates: list[str | date | None] | None = None,
) -> dict[str, Any] | None:
    """Turn source observations into one confidence-tagged catalyst event shape."""
    for source_event in (tier1_event, filed_event):
        if source_event:
            event_date = source_event.get("date") or source_event.get("event_date")
            if not event_date:
                continue
            return {
                "event_date": str(event_date),
                "event_date_precision": "exact",
                "window_start": None,
                "window_end": None,
                "confidence": "confirmed",
                "source_tier": 1 if source_event is tier1_event else 5,
            }

    found_dates = sorted({str(d) for d in (aggregator_dates or []) if d})
    if not found_dates:
        return None
    if len(found_dates) == 1:
        return {
            "event_date": found_dates[0],
            "event_date_precision": "estimated",
            "window_start": None,
            "window_end": None,
            "confidence": "estimated",
            "source_tier": 2,
        }
    return {
        "event_date": found_dates[0],
        "event_date_precision": "window",
        "window_start": found_dates[0],
        "window_end": found_dates[-1],
        "confidence": "conflicting",
        "source_tier": 2,
    }


def _validate_event(event: dict[str, Any]) -> None:
    if event.get("event_type") not in EVENT_TYPES:
        raise ValueError(f"invalid event_type: {event.get('event_type')}")
    if event.get("confidence") not in CONFIDENCE_LEVELS:
        raise ValueError(f"invalid confidence: {event.get('confidence')}")
    if event.get("event_date_precision") not in PRECISION_LEVELS:
        raise ValueError(f"invalid event_date_precision: {event.get('event_date_precision')}")
    if event.get("scope") not in SCOPES:
        raise ValueError(f"invalid scope: {event.get('scope')}")


def upsert_event(event: dict[str, Any]) -> str:
    """
    Insert or refresh an active catalyst event.

    Dedupe key is intentionally conservative: ticker, related_ticker,
    event_type, and event_date -- scope is NOT part of the key since
    `ticker` already disambiguates within a scope (a real ticker never
    collides with an ETF symbol or the MACRO_SENTINEL). Conflicting
    windows keep the earliest date. `scope` defaults to "ticker" so every
    pre-existing caller (upsert_earnings_from_sources et al.) keeps working
    unchanged.
    """
    init_db()
    now = _now()
    normalized = {
        **event,
        "scope": event.get("scope") or "ticker",
        "ticker": str(event["ticker"]).upper(),
        "related_ticker": str(event["related_ticker"]).upper() if event.get("related_ticker") else None,
        "sources": event.get("sources") or [],
        "action_flags": event.get("action_flags") or {},
    }
    _validate_event(normalized)

    conn = get_conn()
    try:
        existing = conn.execute(
            """
            SELECT event_id, sources FROM catalyst_events
            WHERE ticker = ?
              AND COALESCE(related_ticker, '') = COALESCE(?, '')
              AND event_type = ?
              AND event_date = ?
            LIMIT 1
            """,
            (
                normalized["ticker"],
                normalized.get("related_ticker"),
                normalized["event_type"],
                normalized["event_date"],
            ),
        ).fetchone()

        if existing:
            event_id = existing["event_id"]
            conn.execute(
                """
                UPDATE catalyst_events SET
                    scope = ?,
                    event_date_precision = ?,
                    window_start = ?,
                    window_end = ?,
                    bucket_relevance = ?,
                    confidence = ?,
                    source_tier = ?,
                    sources = ?,
                    thesis_note = ?,
                    action_flags = ?,
                    updated_at = ?,
                    last_verified_at = ?
                WHERE event_id = ?
                """,
                (
                    normalized["scope"],
                    normalized["event_date_precision"],
                    normalized.get("window_start"),
                    normalized.get("window_end"),
                    normalized.get("bucket_relevance"),
                    normalized["confidence"],
                    int(normalized["source_tier"]),
                    _json_dumps(normalized["sources"]),
                    normalized.get("thesis_note"),
                    _json_dumps(normalized["action_flags"]),
                    now,
                    now,
                    event_id,
                ),
            )
            conn.commit()
            return event_id

        event_id = normalized.get("event_id") or str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO catalyst_events (
                event_id, scope, ticker, related_ticker, event_date, event_date_precision,
                window_start, window_end, event_type, bucket_relevance, confidence,
                source_tier, sources, thesis_note, action_flags, superseded_by,
                created_at, updated_at, last_verified_at
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                normalized["scope"],
                normalized["ticker"],
                normalized.get("related_ticker"),
                normalized["event_date"],
                normalized["event_date_precision"],
                normalized.get("window_start"),
                normalized.get("window_end"),
                normalized["event_type"],
                normalized.get("bucket_relevance"),
                normalized["confidence"],
                int(normalized["source_tier"]),
                _json_dumps(normalized["sources"]),
                normalized.get("thesis_note"),
                _json_dumps(normalized["action_flags"]),
                normalized.get("superseded_by"),
                now,
                now,
                now,
            ),
        )
        conn.commit()
        return event_id
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["sources"] = _json_loads(d.get("sources"), [])
    d["action_flags"] = _json_loads(d.get("action_flags"), {})
    return d


def query_events(
    ticker: str | None = None,
    related_ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    confidence: str | list[str] | None = None,
    event_type: str | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    init_db()
    clauses: list[str] = []
    params: list[Any] = []

    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if related_ticker:
        clauses.append("related_ticker = ?")
        params.append(related_ticker.upper())
    if start:
        clauses.append("event_date >= ?")
        params.append(start)
    if end:
        clauses.append("event_date <= ?")
        params.append(end)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if scope:
        clauses.append("scope = ?")
        params.append(scope)
    if confidence:
        levels = confidence.split(",") if isinstance(confidence, str) else confidence
        clauses.append(f"confidence IN ({','.join('?' for _ in levels)})")
        params.extend(levels)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM catalyst_events {where} ORDER BY event_date ASC",
            params,
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def list_conflicts() -> list[dict[str, Any]]:
    return query_events(confidence="conflicting")


def blocking_events(ticker: str, end: str, start: str | None = None) -> list[dict[str, Any]]:
    """
    start defaults to today — without it, a catalyst event dated in the past
    (e.g. an earnings date that already happened but was never refreshed)
    would block trades indefinitely instead of aging out.
    """
    if start is None:
        start = date.today().isoformat()
    events = query_events(ticker=ticker, start=start, end=end)
    blocked = []
    for event in events:
        flags = event.get("action_flags") or {}
        if flags.get("halt_auto_execution") or event.get("confidence") in {"conflicting", "rumored"}:
            blocked.append(event)
    return blocked


def upsert_earnings_from_sources(
    ticker: str,
    aggregator_dates: list[str | date | None],
    sources: list[dict[str, Any]],
    bucket_relevance: str | None = None,
) -> str | None:
    reconciled = reconcile(aggregator_dates=aggregator_dates)
    if not reconciled:
        return None

    action_flags = {"tighten_sizing": True}
    if reconciled["confidence"] in {"conflicting", "rumored"}:
        action_flags["halt_auto_execution"] = True

    return upsert_event(
        {
            "ticker": ticker,
            "event_type": "earnings",
            "bucket_relevance": bucket_relevance,
            "sources": sources,
            "thesis_note": f"{ticker.upper()} upcoming earnings date from calendar sources.",
            "action_flags": action_flags,
            **reconciled,
        }
    )


def upsert_macro_event(
    event_type: str,
    event_date: str,
    confidence: str,
    source_tier: int,
    sources: list[dict[str, Any]],
    event_date_precision: str = "exact",
    window_start: str | None = None,
    window_end: str | None = None,
    thesis_note: str | None = None,
    action_flags: dict | None = None,
) -> str:
    """Market-wide event (FOMC, CPI/jobs release, election, ...) -- see
    ingestion.macro_calendar for the actual ingestion. Uses MACRO_SENTINEL
    as the `ticker` since these aren't tied to any one name; `related_ticker`
    is left unset (nothing to relate a market-wide event to)."""
    if event_type not in MACRO_EVENT_TYPES:
        raise ValueError(f"invalid macro event_type: {event_type}")
    return upsert_event({
        "scope": "macro",
        "ticker": MACRO_SENTINEL,
        "event_type": event_type,
        "event_date": event_date,
        "event_date_precision": event_date_precision,
        "window_start": window_start,
        "window_end": window_end,
        "confidence": confidence,
        "source_tier": source_tier,
        "sources": sources,
        "thesis_note": thesis_note,
        "action_flags": action_flags,
    })


def upsert_sector_event(
    sector_etf: str,
    event_type: str,
    event_date: str,
    confidence: str,
    source_tier: int,
    sources: list[dict[str, Any]],
    gics_sector: str | None = None,
    event_date_precision: str = "exact",
    window_start: str | None = None,
    window_end: str | None = None,
    thesis_note: str | None = None,
    action_flags: dict | None = None,
) -> str:
    """Sector-wide event -- see ingestion.sector_calendar for the actual
    ingestion (earnings-cluster detection). `ticker` is the sector's SPDR
    ETF symbol (XLK, XLF, ...) -- the same key agents.market_regime's
    SECTOR_ETF_MAP/get_cached_sector_regime already use, so a sector-scope
    catalyst and a sector-scope regime read resolve from the same symbol.
    `bucket_relevance` carries the human-readable GICS sector name since
    the ticker slot is taken by the ETF symbol."""
    if event_type not in SECTOR_EVENT_TYPES:
        raise ValueError(f"invalid sector event_type: {event_type}")
    return upsert_event({
        "scope": "sector",
        "ticker": sector_etf,
        "event_type": event_type,
        "bucket_relevance": gics_sector,
        "event_date": event_date,
        "event_date_precision": event_date_precision,
        "window_start": window_start,
        "window_end": window_end,
        "confidence": confidence,
        "source_tier": source_tier,
        "sources": sources,
        "thesis_note": thesis_note,
        "action_flags": action_flags,
    })
