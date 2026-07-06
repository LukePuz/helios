
# Catalyst Calendar Service — Build Guide

A shared service that collects, scores, and serves thesis-relevant catalyst data
(earnings, conferences, regulatory events, press, and indirect read-throughs) to
every agent in the trading pipeline. This is **not a seventh agent** — it's a
data layer that Scanner writes to and that Analyst, Risk, Monitor, and Reporter
all read from.

> **⚠️ A note on tickers in this guide.** Every ticker symbol that appears
> below is an **illustrative example only**, included to make the schema,
> queries, and pipeline logic concrete. None of it is a recommendation, a
> live position, or production seed data. The catalyst calendar must be
> driven by your **current, live watchlist** at runtime — the collection
> pipeline (§6) should pull tickers from wherever the watchlist actually
> lives (Scanner's active watchlist, a config file, or a database table),
> and `thesis_adjacency_map` (§4.2) should be populated and maintained from
> your real, current theses, not copied in from this document.

---

## 1. Purpose & Scope

The catalyst calendar exists to answer one question, reliably, for any ticker
and any date range: *"What's coming up that could move this name, how sure are
we of the date, and why does it matter to our thesis?"*

It needs to:

- Track **direct catalysts** (earnings, FDA dates, product launches) and
  **indirect/adjacent catalysts** (competitor earnings, supplier/customer
  results, sector conferences) for every name in the watchlist.
- Carry a **confidence tag** per event, because sources disagree with each
  other more often than you'd expect (see §7 — this happened live during
  research, on a name in the current book).
- Be queryable by every agent without each agent re-deriving it independently.
- Persist a history of past events so Reporter can backtest whether the
  calendar's confidence tagging was actually predictive.

## 2. Where This Fits in the Agent Architecture

```
                    ┌─────────────────────────────┐
                    │   Catalyst Calendar Service  │
                    │   (FastAPI + SQLite)         │
                    │                               │
                    │   catalyst_events            │
                    │   thesis_adjacency_map        │
                    │   source_reliability          │
                    │   catalyst_events_history      │
                    └───────────────┬───────────────┘
                                    │
              writes                │  reads
        ┌─────────────┐     ┌───────┴───────┬───────────────┬───────────────┐
        │   Scanner    │     │    Analyst    │      Risk      │    Monitor    │
        │ (collector)  │     │ (expiration/  │ (pre-trade gate,│ (imminent-    │
        │              │     │  strike timing│  kill-switch    │  event watch) │
        └─────────────┘     │  via duration  │  windows)       │               │
                              │  = catalyst+  │                 │               │
                              │  buffer)      │                 │               │
                              └───────────────┘                 └───────────────┘
                                                                         │
                                                                  ┌──────┴──────┐
                                                                  │  Reporter   │
                                                                  │ (scores     │
                                                                  │  source     │
                                                                  │  reliability)│
                                                                  └─────────────┘
```

Scanner is the only agent that *writes* new catalyst entries (plus a daily
collector job that runs independently of any one agent's invocation). Every
other agent *reads* via a thin HTTP client against the FastAPI service — never
by opening the SQLite file directly. This avoids concurrent-write contention
and keeps a single source of truth.

## 3. Source Tiers & Collection Methodology

| Tier | Source type | Examples | Default confidence |
|---|---|---|---|
| 1 | Company IR / SEC EDGAR | 8-K filings, IR "Events" page, official press releases | `confirmed` |
| 2 | Earnings calendar aggregators | TipRanks, Unusual Whales, StockTitan — cross-check at least 2 before trusting | `estimated` until cross-validated, `conflicting` if sources disagree |
| 3 | Conference & speaking calendars | Sector investor days (semis, gaming, biotech, enterprise software), company IR event pages | `estimated` |
| 4 | Indirect / adjacent read-throughs | Competitor earnings, supplier/customer results, hyperscaler capex commentary | `estimated`, tagged `indirect_readthrough` |
| 5 | Regulatory & filing triggers | Form 4 insider filings, FDA calendar entries, material 8-Ks | `confirmed` if filed, `rumored` if reported pre-filing |

**Rule:** an event only gets `confirmed` status from a Tier 1 or filed Tier 5
source. Everything else starts as `estimated` and gets upgraded or flagged
`conflicting` once cross-checked.

## 4. Data Model

SQLite is the right tool here, not TimescaleDB — this is low-frequency,
relational, event-style data (a few hundred rows across the watchlist at any
time), not high-frequency time-series ticks. Keep it as its own database file
(`catalyst_calendar.db`) so it can be backed up, migrated, or queried
independently of the tick-data store.

### 4.1 `catalyst_events` — the core table

```sql
CREATE TABLE catalyst_events (
    event_id              TEXT PRIMARY KEY,            -- UUID
    ticker                 TEXT NOT NULL,                -- ticker this event is about
    related_ticker         TEXT,                          -- for indirect events: the
                                                            -- ticker in YOUR book this informs
    event_date             TEXT NOT NULL,                 -- ISO8601 date or datetime
    event_date_precision    TEXT NOT NULL
                            CHECK (event_date_precision IN ('exact','window','estimated')),
    window_start            TEXT,                          -- populated if precision = 'window'
    window_end               TEXT,
    event_type              TEXT NOT NULL
                            CHECK (event_type IN (
                                'earnings','conference','regulatory',
                                'press','indirect_readthrough',
                                'insider_filing','other'
                            )),
    bucket_relevance        TEXT,                          -- 'core' | 'tail' | '0dte_macro' | combo
    confidence               TEXT NOT NULL
                            CHECK (confidence IN ('confirmed','estimated','conflicting','rumored')),
    source_tier               INTEGER NOT NULL,             -- 1-5, see §3
    sources                    TEXT NOT NULL,                -- JSON array, see below
    thesis_note                TEXT,                          -- why this matters to the position
    action_flags                TEXT,                          -- JSON, see below
    superseded_by                TEXT,                          -- event_id of a correcting entry
    created_at                    TEXT NOT NULL,
    updated_at                     TEXT NOT NULL,
    last_verified_at                TEXT
);

CREATE INDEX idx_catalyst_ticker_date ON catalyst_events(ticker, event_date);
CREATE INDEX idx_catalyst_confidence  ON catalyst_events(confidence);
CREATE INDEX idx_catalyst_related     ON catalyst_events(related_ticker);
```

`sources` is a JSON array, e.g.:
```json
[
  {"name": "ServiceNow IR", "url": "https://investors.servicenow.com", "retrieved_at": "2026-06-21T09:00:00Z"},
  {"name": "TipRanks", "url": "https://www.tipranks.com/stocks/now/earnings", "retrieved_at": "2026-06-21T09:00:00Z", "reported_date": "2026-07-29"}
]
```

`action_flags` is a JSON object the consuming agents read directly, e.g.:
```json
{"tighten_sizing": true, "widen_scan_window": true, "halt_auto_execution": true}
```

### 4.2 `thesis_adjacency_map` — the indirect-catalyst lookup

This is what makes Tier 4 collection systematic instead of ad hoc.

```sql
CREATE TABLE thesis_adjacency_map (
    primary_ticker      TEXT NOT NULL,     -- the position you hold
    adjacent_ticker      TEXT NOT NULL,     -- the name to watch
    relationship_type     TEXT NOT NULL
                          CHECK (relationship_type IN (
                              'competitor','supplier','customer',
                              'sector_peer','index_proxy','platform_owner'
                          )),
    relevance_note         TEXT,
    PRIMARY KEY (primary_ticker, adjacent_ticker)
);
```

Seed it from your real, current theses — not from a static list. The pattern
looks like this (placeholders shown, not real tickers):

```sql
INSERT INTO thesis_adjacency_map VALUES
 ('<PRIMARY_TICKER>', '<ADJACENT_TICKER>', 'competitor',
  'why this name moves on the same narrative as the primary position'),
 ('<PRIMARY_TICKER>', '<ADJACENT_TICKER_2>', 'customer',
  'demand signal for the primary position via this name''s earnings/capex commentary');
```

Relationship types worth distinguishing: `competitor`, `supplier`, `customer`,
`sector_peer`, `index_proxy`, `platform_owner`. Populate this table when you
open a new thesis, as part of Analyst's initial research — not as a one-time
seed.

Scanner's nightly job iterates this table for every primary_ticker in the
live watchlist and pulls Tier 4 events for each adjacent_ticker automatically.

### 4.3 `source_reliability` — feedback loop for Reporter

```sql
CREATE TABLE source_reliability (
    source_name              TEXT PRIMARY KEY,
    correct_predictions        INTEGER NOT NULL DEFAULT 0,
    incorrect_predictions       INTEGER NOT NULL DEFAULT 0,
    avg_days_off                  REAL,                  -- mean |predicted_date - actual_date|
    last_updated                   TEXT
);
```

Reporter updates this weekly by comparing `catalyst_events_history` entries
against what actually happened, and the collection pipeline can use it to
break ties when two Tier 2 sources disagree (prefer the historically more
accurate one, but still surface the conflict — don't silently resolve it).

### 4.4 `catalyst_events_history` — the archive

Same schema as `catalyst_events`, plus an `outcome_note` column. Once an
event's date has passed, the nightly job moves the row here rather than
deleting it, so Reporter has a clean dataset for confidence-tag backtesting.

```sql
CREATE TABLE catalyst_events_history (
    -- identical columns to catalyst_events, plus:
    outcome_note     TEXT,    -- e.g. "date held"; "moved by 5 days, flagged conflicting correctly"
    archived_at       TEXT NOT NULL
);
```

## 5. Storage & Access Layer

Wrap the SQLite file in a small FastAPI service so every agent talks HTTP,
not file paths. This matches the FastAPI service layer already planned for
the rest of the system.

```python
# catalyst_service/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Literal
import sqlite3, uuid, json
from datetime import datetime, timezone

DB_PATH = "catalyst_calendar.db"
app = FastAPI(title="Catalyst Calendar Service")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class CatalystEventIn(BaseModel):
    ticker: str
    related_ticker: Optional[str] = None
    event_date: str
    event_date_precision: Literal["exact", "window", "estimated"]
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    event_type: Literal[
        "earnings", "conference", "regulatory", "press",
        "indirect_readthrough", "insider_filing", "other"
    ]
    bucket_relevance: Optional[str] = None
    confidence: Literal["confirmed", "estimated", "conflicting", "rumored"]
    source_tier: int
    sources: list[dict]
    thesis_note: Optional[str] = None
    action_flags: Optional[dict] = None

@app.post("/catalysts")
def create_event(event: CatalystEventIn):
    now = datetime.now(timezone.utc).isoformat()
    event_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        """INSERT INTO catalyst_events
           (event_id, ticker, related_ticker, event_date, event_date_precision,
            window_start, window_end, event_type, bucket_relevance, confidence,
            source_tier, sources, thesis_note, action_flags,
            created_at, updated_at, last_verified_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (event_id, event.ticker, event.related_ticker, event.event_date,
         event.event_date_precision, event.window_start, event.window_end,
         event.event_type, event.bucket_relevance, event.confidence,
         event.source_tier, json.dumps(event.sources), event.thesis_note,
         json.dumps(event.action_flags or {}), now, now, now)
    )
    conn.commit()
    return {"event_id": event_id}

@app.get("/catalysts")
def query_events(
    ticker: Optional[str] = None,
    related_ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    confidence: Optional[str] = None,   # comma-separated, e.g. "confirmed,estimated"
):
    conn = get_conn()
    clauses, params = [], []
    if ticker:
        clauses.append("ticker = ?"); params.append(ticker)
    if related_ticker:
        clauses.append("related_ticker = ?"); params.append(related_ticker)
    if start:
        clauses.append("event_date >= ?"); params.append(start)
    if end:
        clauses.append("event_date <= ?"); params.append(end)
    if confidence:
        levels = confidence.split(",")
        clauses.append(f"confidence IN ({','.join('?' * len(levels))})")
        params.extend(levels)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM catalyst_events {where} ORDER BY event_date ASC", params
    ).fetchall()
    return [dict(r) for r in rows]

@app.get("/catalysts/conflicts")
def list_conflicts():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM catalyst_events WHERE confidence = 'conflicting' "
        "ORDER BY event_date ASC"
    ).fetchall()
    return [dict(r) for r in rows]

@app.patch("/catalysts/{event_id}")
def update_event(event_id: str, confidence: Optional[str] = None,
                  event_date: Optional[str] = None, sources: Optional[list] = None):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    updates, params = ["updated_at = ?", "last_verified_at = ?"], [now, now]
    if confidence:
        updates.append("confidence = ?"); params.append(confidence)
    if event_date:
        updates.append("event_date = ?"); params.append(event_date)
    if sources:
        updates.append("sources = ?"); params.append(json.dumps(sources))
    params.append(event_id)
    conn.execute(f"UPDATE catalyst_events SET {', '.join(updates)} WHERE event_id = ?", params)
    conn.commit()
    if conn.total_changes == 0:
        raise HTTPException(404, "event not found")
    return {"status": "updated"}
```

This is intentionally minimal — add auth, pagination, and validation as the
rest of the agent stack matures. The point is the contract: every agent talks
to `localhost:PORT/catalysts`, never to the `.db` file directly.

## 6. Collection Pipeline

A nightly batch job (cron, or a scheduled task inside Scanner's loop) does
the actual gathering. Pseudocode for the shape of it:

```python
# collector/nightly_job.py
def run_nightly_collection(watchlist: list[str]):
    for ticker in watchlist:
        # Tier 1: company IR + EDGAR — always check first
        ir_event = check_company_ir(ticker)
        edgar_event = check_edgar_8k(ticker)

        # Tier 2: aggregators, cross-checked against each other
        aggregator_dates = [
            fetch_tipranks_earnings_date(ticker),
            fetch_unusualwhales_earnings_date(ticker),
        ]

        event = reconcile(ir_event, edgar_event, aggregator_dates)
        upsert_catalyst_event(ticker, event)

        # Tier 3: conference calendars relevant to this ticker's sector
        for conf_event in check_sector_conferences(ticker):
            upsert_catalyst_event(ticker, conf_event)

        # Tier 4: walk the adjacency map for indirect read-throughs
        for adj_ticker, rel_type, note in get_adjacent_tickers(ticker):
            adj_event = check_company_ir(adj_ticker) or check_edgar_8k(adj_ticker)
            if adj_event:
                upsert_catalyst_event(
                    ticker=adj_ticker,
                    related_ticker=ticker,
                    event_type="indirect_readthrough",
                    thesis_note=note,
                    **adj_event,
                )

        # Tier 5: insider filings, regulatory calendar
        for filing in check_form4_filings(ticker):
            upsert_catalyst_event(ticker, filing)
```

`reconcile()` is the function that implements the confidence-tagging logic —
see §7.

## 7. Conflict Detection & Resolution

This is the highest-value piece of the whole service, because it's the
failure mode that's easy to build past and expensive to hit live. A real
example from researching the methodology behind this guide: a name's
upcoming earnings date was reported by one aggregator as falling on one date,
and by a second aggregator as falling **seven days later** — on a name that
was actively held. The two sources simply disagreed, and nothing about either
listing made the discrepancy obvious unless you cross-checked them directly.

```python
def reconcile(ir_event, edgar_event, aggregator_dates: list[str | None]) -> dict:
    # Tier 1 sources win outright if present
    if ir_event or edgar_event:
        date = ir_event["date"] if ir_event else edgar_event["date"]
        return {
            "event_date": date,
            "event_date_precision": "exact",
            "confidence": "confirmed",
            "source_tier": 1,
        }

    # No Tier 1 confirmation yet — compare aggregators
    found_dates = {d for d in aggregator_dates if d}
    if len(found_dates) == 0:
        return None  # nothing to record yet
    if len(found_dates) == 1:
        return {
            "event_date": found_dates.pop(),
            "event_date_precision": "estimated",
            "confidence": "estimated",
            "source_tier": 2,
        }
    # Aggregators disagree — this is the conflicting case
    return {
        "event_date": min(found_dates),       # earliest, for conservative gating
        "window_start": min(found_dates),
        "window_end": max(found_dates),
        "event_date_precision": "window",
        "confidence": "conflicting",
        "source_tier": 2,
    }
```

Key design choices baked into this:

- **Tier 1 always wins.** No aggregator outranks a company's own IR page or
  an 8-K, ever.
- **Conflicting dates produce a window, not a guess.** Risk should gate
  against the *full* window (earliest reported date to latest reported
  date), not pick a midpoint — the cost of being early with a gating rule is
  much lower than the cost of being wrong about a binary event's timing.
- **Conflicts get surfaced, not buried.** The `/catalysts/conflicts`
  endpoint exists specifically so something — a daily Reporter digest, or a
  direct Telegram ping — pulls these into a human-in-the-loop review rather
  than letting Risk silently trade against an unresolved date.

```python
# Telegram escalation for unresolved conflicts within N days of "today"
def alert_unresolved_conflicts(bot, chat_id, days_ahead=10):
    conflicts = requests.get(
        "http://localhost:8001/catalysts/conflicts"
    ).json()
    soon = [c for c in conflicts if days_until(c["event_date"]) <= days_ahead]
    for c in soon:
        bot.send_message(
            chat_id,
            f"⚠️ Unresolved catalyst date for {c['ticker']}: "
            f"sources disagree ({c['window_start']}–{c['window_end']}). "
            f"Confirm before any pre-trade gating relies on it."
        )
```

## 8. Per-Agent Integration

### 8.1 Scanner (writer)
Runs the nightly collection job (§6) and also writes ad-hoc entries when it
flags a new setup mid-session (e.g., a press release just hit). Tags
`bucket_relevance` at write time so downstream agents don't have to
re-derive Core/Tail/0DTE classification from the event type.

### 8.2 Analyst (reader — timing logic)
Queries for the nearest confirmed catalyst on the ticker before selecting
strikes/expiration:

```python
events = requests.get(
    "http://localhost:8001/catalysts",
    params={"ticker": "MU", "confidence": "confirmed,estimated"}
).json()
next_catalyst = min(events, key=lambda e: e["event_date"])
expiration = next_catalyst["event_date"] + duration_buffer  # your existing logic
```

### 8.3 Risk (reader — pre-trade gating, kill-switch windows)
Pre-trade gate checks whether a `conflicting` or `rumored` event falls inside
the proposed holding window, and tightens sizing or blocks auto-execution
accordingly — this is exactly what `action_flags.halt_auto_execution` is for.

```python
def catalyst_gate_check(ticker, proposed_expiration):
    events = requests.get(
        "http://localhost:8001/catalysts",
        params={"ticker": ticker, "end": proposed_expiration}
    ).json()
    blocking = [e for e in events if e["action_flags"].get("halt_auto_execution")]
    if blocking:
        return {"approved": False, "reason": f"unresolved catalyst: {blocking[0]['thesis_note']}"}
    return {"approved": True}
```

### 8.4 Monitor (reader — imminent-event watch)
Polls for events within the next 1–3 trading days across open positions and
widens its refresh rate / tightens kill-switch sensitivity ahead of them —
this is also where indirect read-throughs matter: if an adjacent name from
`thesis_adjacency_map` reports and moves sharply, Monitor should flag that
as relevant context for the related open position, even though that
position's own ticker had no news that day.

### 8.5 Reporter (reader/writer — scoring)
Weekly job: pull everything that aged into `catalyst_events_history`,
compare predicted vs. actual dates, update `source_reliability`, and surface
a digest of how often each tier/source was right — this is the data that
should eventually inform whether to keep paying for a given aggregator.

## 9. Example Seed Data (Illustrative Pattern Only — Not Production Data)

The two rows below exist purely to show what a populated row looks like for
each `event_type`. **Do not use this as actual seed data.** In production,
every row in `catalyst_events` should come from the collection pipeline
(§6) running against your live watchlist — never from a hardcoded list.

```sql
INSERT INTO catalyst_events
 (event_id, ticker, related_ticker, event_date, event_date_precision,
  event_type, bucket_relevance, confidence, source_tier, sources,
  thesis_note, action_flags, created_at, updated_at, last_verified_at)
VALUES
 ('evt-001','MU',NULL,'2026-06-24','exact','earnings','tail','confirmed',1,
  '[{"name":"Micron IR","url":"https://investors.micron.com"}]',
  'FQ3''26 earnings; HBM4/Vera Rubin allocation commentary and FY26 capex guidance are the swing factors',
  '{"tighten_sizing": true}', '2026-06-21T00:00:00Z','2026-06-21T00:00:00Z','2026-06-21T00:00:00Z'),

 ('evt-002','TTWO',NULL,'2026-06-28','window','press','core','estimated',3,
  '[{"name":"TechTimes","url":"https://www.techtimes.com/articles/317156"}]',
  'GTA VI Trailer 3 / pre-orders expected to open; relevant to Jan 2027 calls thesis',
  '{"widen_scan_window": true}', '2026-06-21T00:00:00Z','2026-06-21T00:00:00Z','2026-06-21T00:00:00Z');
```

For reference, here's the row shape for an `indirect_readthrough` event
(placeholders, not a real example):

```sql
 ('evt-XXX','<ADJACENT_TICKER>','<PRIMARY_TICKER>','<date>','exact',
  'indirect_readthrough','core','confirmed',4,
  '[{"name":"<source>","url":"<url>"}]',
  'why this adjacent name''s event matters to the primary position',
  '{}', '<created_at>','<updated_at>','<last_verified_at>');
```

## 10. Example Queries

**Everything in the next 30 days, confirmed or estimated only:**
```sql
SELECT ticker, event_date, event_type, confidence, thesis_note
FROM catalyst_events
WHERE event_date BETWEEN date('now') AND date('now', '+30 days')
  AND confidence IN ('confirmed', 'estimated')
ORDER BY event_date;
```

**Unresolved conflicts inside the next 10 days (for the Telegram alert job):**
```sql
SELECT ticker, event_date, window_start, window_end, thesis_note
FROM catalyst_events
WHERE confidence = 'conflicting'
  AND event_date <= date('now', '+10 days');
```

**All indirect read-throughs feeding a given position** (replace `<TICKER>`
with the actual position from your live watchlist):
```sql
SELECT ticker AS adjacent_name, event_date, thesis_note
FROM catalyst_events
WHERE related_ticker = '<TICKER>' AND event_type = 'indirect_readthrough'
ORDER BY event_date;
```

**Binary/tail events with no confirmed date — surfaced for Risk's tail-bucket review:**
```sql
SELECT ticker, event_date_precision, confidence, thesis_note
FROM catalyst_events
WHERE bucket_relevance = 'tail' AND confidence IN ('rumored', 'estimated');
```

## 11. Maintenance & Retention

- **Daily:** nightly collection job (§6) runs for the full watchlist.
- **Daily:** conflict-check job pings Telegram for anything unresolved inside
  a 10-day window (§7).
- **Weekly:** archive job moves anything with `event_date < today` into
  `catalyst_events_history`, tagging an `outcome_note`.
- **Weekly:** Reporter job scores `source_reliability` off the archive.
- **Quarterly:** review `thesis_adjacency_map` — add/remove adjacent tickers
  as positions change; this table will drift out of date faster than the
  schema itself.

## 12. Build Checklist

- [ ] Create `catalyst_calendar.db` with the four tables in §4
- [ ] Stand up the FastAPI service in §5, confirm it runs independently of
      any single agent process
- [ ] Write the Tier 1–5 collector functions in §6 (start with Tier 1 + 2
      only — that alone covers earnings dates, the highest-value case)
- [ ] Implement `reconcile()` (§7) and the `/catalysts/conflicts` endpoint
      before connecting Risk to this service — Risk should never see an
      unflagged single-source date
- [ ] Wire Scanner's nightly job to call the collector and POST results
- [ ] Wire Analyst, Risk, and Monitor to GET from the service (read-only
      clients — no agent except Scanner's collector job should POST)
- [ ] Populate `thesis_adjacency_map` from your live watchlist and actual,
      current theses — §4.2 shows the row pattern with placeholders only;
      don't carry example values into production
- [ ] Add the Telegram conflict-alert job (§7) before going live with any
      auto-execution tied to catalyst gating
- [ ] Build the weekly archive + Reporter scoring job (§11) last — it needs
      a few weeks of history to be useful, but the schema should exist from
      day one so you're not migrating data later
