"""
Sector-level catalyst calendar (2026-08-30, user request).

There's no free live feed of "scheduled sector-wide events" the way FOMC/
CPI dates exist for macro or yfinance exists for per-ticker earnings. What
IS derivable from data already ingested: when several constituents of the
same GICS sector report earnings within the same calendar week, that week
is itself a sector-relevant catalyst -- a cluster of Info Tech earnings
sets the tone (and the trading volatility) for the whole sector, not just
the individual names reporting, and a position in a name that ISN'T
reporting that week can still see sector-driven moves from the names that
are. This module detects and records those clusters as scope="sector"
catalyst_service events; it does not (yet) ingest genuinely independent
sector-level sources (regulatory dockets, index-rebalance schedules) --
those are supported by catalyst_service.storage.upsert_sector_event for
whenever such a source is added, just not populated by this module.
"""
import logging
from collections import defaultdict
from datetime import date, timedelta

log = logging.getLogger(__name__)

# A sector week needs at least this many constituents reporting to count as
# a cluster -- one or two names reporting the same week is unremarkable
# (earnings are already ~weekly-distributed across any 50+ name sector);
# this is meant to catch genuine concentration, not every ordinary week.
CLUSTER_MIN_COUNT = 3
# How far ahead to scan for upcoming clusters.
LOOKAHEAD_DAYS = 60


def _week_start(d: date) -> date:
    """Monday of the ISO week containing d."""
    return d - timedelta(days=d.weekday())


def refresh_sector_calendar(
    min_count: int = CLUSTER_MIN_COUNT,
    lookahead_days: int = LOOKAHEAD_DAYS,
) -> int:
    """
    Scans db.earnings_calendar for the next `lookahead_days`, groups by
    GICS sector and by calendar week, and upserts a sector_earnings_cluster
    event for every (sector, week) with >= min_count constituents reporting.

    Idempotent, like refresh_macro_calendar -- re-running just refreshes
    the same (sector ETF, event_type, week-start date) rows via
    upsert_event's existing dedupe key.

    Returns count of cluster events upserted.
    """
    import db
    from catalyst_service.storage import upsert_sector_event
    from ingestion.sp500_universe import SECTOR_ETF_MAP, get_sp500_constituents

    today = date.today()
    end = today + timedelta(days=lookahead_days)
    rows = db.get_earnings_in_range(today, end)
    if not rows:
        log.info("Sector calendar: no upcoming earnings in range, nothing to cluster")
        return 0

    constituents = get_sp500_constituents()

    # {(gics_sector, week_start): {ticker: confirmed}} -- keyed by ticker
    # (not a plain list) so a ticker with multiple earnings_calendar rows
    # in the same week (a stale estimated date alongside a firmer
    # confirmed one -- upsert_earnings_date keeps both since they're
    # different dates under its own (ticker, earnings_date) unique key)
    # counts once toward the cluster, not once per row.
    buckets: dict[tuple[str, date], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        ticker = row["ticker"]
        gics_sector = constituents.get(ticker, {}).get("gics_sector")
        if not gics_sector or gics_sector not in SECTOR_ETF_MAP:
            continue  # not an S&P constituent (our own watchlist has plenty that aren't) -- no sector to cluster into
        earnings_date = row["earnings_date"]
        wk = _week_start(earnings_date)
        confirmed = row.get("confirmed", False)
        # Prefer confirmed=True if this ticker already has an entry for
        # this week from a different row.
        buckets[(gics_sector, wk)][ticker] = buckets[(gics_sector, wk)].get(ticker, False) or confirmed

    count = 0
    for (gics_sector, wk), members in buckets.items():
        if len(members) < min_count:
            continue
        etf = SECTOR_ETF_MAP[gics_sector]
        tickers_sorted = sorted(members)
        confirmed_share = sum(1 for c in members.values() if c) / len(members)
        confidence = "confirmed" if confirmed_share >= 0.5 else "estimated"
        window_end = wk + timedelta(days=6)
        try:
            upsert_sector_event(
                sector_etf=etf,
                event_type="sector_earnings_cluster",
                event_date=wk.isoformat(),
                event_date_precision="window",
                window_start=wk.isoformat(),
                window_end=window_end.isoformat(),
                confidence=confidence,
                source_tier=2,
                sources=[{"name": "db.earnings_calendar (derived cluster)", "as_of": today.isoformat()}],
                gics_sector=gics_sector,
                thesis_note=(
                    f"{len(members)} {gics_sector} constituents report earnings this week: "
                    f"{', '.join(tickers_sorted)}"
                ),
                action_flags={"tighten_sizing": True},
            )
            count += 1
        except Exception as e:
            log.warning(f"Sector cluster upsert failed for {gics_sector}/{wk}: {e}")

    log.info(f"Sector calendar refreshed: {count} earnings-cluster events upserted")
    return count
