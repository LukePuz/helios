from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import storage

app = FastAPI(title="Catalyst Calendar Service")


class CatalystEventIn(BaseModel):
    # scope="ticker" (default) -- ticker holds the actual symbol.
    # scope="sector" -- ticker holds the sector's SPDR ETF symbol (XLK, ...).
    # scope="macro" -- ticker holds storage.MACRO_SENTINEL ("MARKET").
    # See catalyst_service/storage.py's SCOPES docstring for the full
    # rationale (2026-08-30, user request).
    scope: Literal["ticker", "sector", "macro"] = "ticker"
    ticker: str
    related_ticker: str | None = None
    event_date: str
    event_date_precision: Literal["exact", "window", "estimated"]
    window_start: str | None = None
    window_end: str | None = None
    event_type: Literal[
        # ticker-level
        "earnings", "conference", "regulatory", "press",
        "indirect_readthrough", "insider_filing", "other",
        # sector-level
        "sector_earnings_cluster", "sector_regulatory", "index_rebalance",
        "sector_conference", "sector_other",
        # macro-level
        "fomc_meeting", "cpi_release", "jobs_report", "gdp_release",
        "election", "geopolitical", "macro_other",
    ]
    bucket_relevance: str | None = None
    confidence: Literal["confirmed", "estimated", "conflicting", "rumored"]
    source_tier: int
    sources: list[dict]
    thesis_note: str | None = None
    action_flags: dict | None = None


@app.on_event("startup")
def startup() -> None:
    storage.init_db()


@app.post("/catalysts")
def create_event(event: CatalystEventIn):
    try:
        event_id = storage.upsert_event(event.dict())
        return {"event_id": event_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/catalysts")
def query_events(
    ticker: str | None = None,
    related_ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    confidence: str | None = None,
    event_type: str | None = None,
    scope: str | None = None,
):
    return storage.query_events(
        ticker=ticker,
        related_ticker=related_ticker,
        start=start,
        end=end,
        confidence=confidence,
        event_type=event_type,
        scope=scope,
    )


@app.get("/catalysts/conflicts")
def list_conflicts():
    return storage.list_conflicts()
