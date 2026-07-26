"""
Candidate Scanner Agent — new-ticker/watchlist-candidate discovery.

Every other scanning/screening agent in this codebase (scanner.py,
screener.py) is scoped to the existing ~101-ticker AI/tech watchlist. This
agent looks the other direction: which of the ~400 S&P 500 tickers NOT
already on the watchlist are worth considering for it?

Screens on two independent, quantitative-first dimensions:
  1. Movement — reuses agents.scanner's technical/momentum scoring
     (already ticker-agnostic; no changes needed there).
  2. Thesis match — semantic similarity between the candidate's actual
     business description (GICS sector/sub-industry + yfinance business
     summary) and our real thesis prose (knowledge/*.md), via the existing
     vector_store embeddings. Not a hardcoded keyword list: a candidate
     only scores well here if its business description genuinely reads
     like something from one of our thesis files.

Output is a ranked shortlist logged to watchlist_candidates and (for the
top few) a Telegram digest — never an auto-add to the watchlist. Adding a
name to the actual thesis is a human/analyst judgment call; this agent's
job is only to surface candidates worth that judgment call, the same way
scanner.py surfaces candidates worth the Analyst's judgment call.

Runs weekly (see tasks.run_candidate_scan) — exploratory, not
time-sensitive, and ~400 tickers of scoring plus embedding calls is not
something to repeat daily. No LLM cost anywhere in this module; the 7B/72B
models never see a candidate here.

Known limitation (found while calibrating thresholds against real data):
the thesis-match score can lexically over-match a traditional-finance
name (e.g. a plain bank) against financials.md, because that file's own
text is about AI-enabled fintech specifically but still uses generic
financial-sector vocabulary. The embedding can't yet tell "AI fintech" from
"any bank" apart within the financials.md match itself — thresholds here
are tuned to separate "genuinely unrelated sector" from "plausibly
relevant," not to make that finer distinction. Shortlist is for human
review, which is exactly where that judgment call belongs.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

_THESIS_FILES = [
    "ai_data.md", "ai_infrastructure.md", "ai_software.md", "financials.md",
    "power_infrastructure.md", "quantum.md", "robotics.md", "space_data_centers.md",
]

MOVEMENT_MAX = 50   # technical (25) + momentum (25), from agents.scanner
THESIS_MAX   = 25

# Calibrated 2026-07-26 against real candidates (see conversation/testing,
# not guessed): vector_store.search's "1 - L2 distance" score clusters in a
# negative range here, not the 0..1 a cosine-similarity search would give.
# Real observed spread: genuine matches (IonQ/quantum -0.18, CrowdStrike/
# ai_software -0.31, Intuitive Surgical/robotics -0.36, Constellation
# Energy & Vistra/power -0.40 to -0.43, Lockheed/robotics -0.51 borderline)
# vs genuine non-matches (Coca-Cola -0.79, Walmart -0.81) -- there's a wide,
# clean gap between roughly -0.55 and -0.65 that separates the two groups.
_THESIS_STRONG_THRESHOLD = -0.50
_THESIS_WEAK_THRESHOLD   = -0.65


def seed_thesis_collection() -> int:
    """(Re-)seeds the 'thesis' vector_store collection from knowledge/*.md.
    Idempotent — vector_store.store() upserts by doc_id, so re-running
    after a thesis file edit just re-embeds the updated text. Cheap enough
    to call at the start of every scan rather than needing a separate
    one-time setup step."""
    import vector_store

    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    seeded = 0
    for fname in _THESIS_FILES:
        path = knowledge_dir / fname
        if not path.exists():
            log.warning(f"Thesis file not found: {fname}")
            continue
        vector_store.store("thesis", doc_id=fname, text=path.read_text(),
                            metadata={"thesis_file": fname})
        seeded += 1
    return seeded


def _score_thesis_match(
    ticker: str, company: str, gics_sector: str, gics_sub_industry: str,
) -> tuple[int, str, str | None]:
    """Returns (score 0-25, notes, matched_thesis_file_or_None)."""
    import vector_store

    query_text = f"{company} ({ticker}). Sector: {gics_sector} — {gics_sub_industry}."
    try:
        import yfinance as yf
        summary = yf.Ticker(ticker).info.get("longBusinessSummary", "")
        if summary:
            query_text += f" {summary}"
    except Exception as e:
        log.debug(f"[{ticker}] business summary fetch failed: {e}")

    try:
        results = vector_store.search("thesis", query_text, n_results=1)
    except Exception as e:
        log.debug(f"[{ticker}] thesis search failed: {e}")
        return 0, "Thesis match unavailable", None

    if not results:
        return 0, "No thesis documents indexed", None

    best = results[0]
    similarity   = best["score"]
    matched_file = best["metadata"].get("thesis_file")

    if similarity > _THESIS_STRONG_THRESHOLD:
        return THESIS_MAX, f"Strong thesis match with {matched_file} (similarity={similarity:.2f})", matched_file
    elif similarity > _THESIS_WEAK_THRESHOLD:
        return THESIS_MAX // 2, f"Possible thesis match with {matched_file} (similarity={similarity:.2f})", matched_file
    else:
        return 0, f"No meaningful thesis match (best={matched_file}, similarity={similarity:.2f})", None


def _score_movement(ticker: str, bars: list[dict]) -> tuple[int, str]:
    """Reuses agents.scanner's technical + momentum scoring as-is — both
    are already ticker-agnostic (just need bars), no changes needed there."""
    from agents.scanner import _score_technical, _score_momentum

    tech_score, tech_notes = _score_technical(bars, ticker)
    mom_score,  mom_notes  = _score_momentum(bars, ticker)
    total = tech_score + mom_score
    return total, f"{tech_notes} | {mom_notes}"


def run_candidate_scan(min_composite_score: int = 40, max_results: int = 25) -> dict:
    """
    Screens every S&P 500 ticker not already on the watchlist. Reads bars
    from the price_history cache (populated weekly by
    ingestion.price_history.refresh_sp500_price_history) rather than
    fetching live — this is an exploratory sweep, not something that needs
    same-day price data for ~400 tickers.

    Returns a packet with the ranked shortlist; every candidate is also
    logged to db.watchlist_candidates regardless of whether it clears
    min_composite_score, so the full scan is auditable later.
    """
    import db
    from knowledge_loader import get_all_tickers
    from ingestion.sp500_universe import get_sp500_constituents

    seed_thesis_collection()

    watchlist = set(get_all_tickers())
    constituents = get_sp500_constituents()
    candidate_tickers = [t for t in constituents if t not in watchlist]

    log.info(f"Candidate scan: screening {len(candidate_tickers)} non-watchlist S&P 500 tickers")

    bulk_bars = db.get_price_history_bulk(candidate_tickers, days=220)

    results: list[dict] = []
    for ticker in candidate_tickers:
        bars = bulk_bars.get(ticker, [])
        if len(bars) < 60:
            continue  # same minimum agents.scanner/price_action require

        info = constituents[ticker]
        movement_score, movement_notes = _score_movement(ticker, bars)

        # Thesis-match embedding calls are the expensive part of this loop
        # (yfinance .info + a CPU embedding per candidate) — skip them for
        # tickers with no real movement at all, since a thesis match alone
        # isn't actionable without something happening in the name.
        if movement_score < 15:
            continue

        thesis_score, thesis_notes, matched_thesis = _score_thesis_match(
            ticker, info["company"], info["gics_sector"], info["gics_sub_industry"],
        )
        composite = movement_score + thesis_score
        notes = f"{movement_notes} | {thesis_notes}"

        db.log_watchlist_candidate(
            ticker=ticker, gics_sector=info["gics_sector"],
            movement_score=movement_score, thesis_match_score=thesis_score,
            composite_score=composite, matched_thesis=matched_thesis, notes=notes,
        )

        if composite >= min_composite_score:
            results.append({
                "ticker": ticker, "company": info["company"],
                "gics_sector": info["gics_sector"],
                "movement_score": movement_score, "thesis_match_score": thesis_score,
                "composite_score": composite, "matched_thesis": matched_thesis,
                "notes": notes,
            })

    results.sort(key=lambda c: c["composite_score"], reverse=True)
    results = results[:max_results]

    log.info(f"Candidate scan complete: {len(results)} candidates above threshold")

    return {
        "agent": "candidate_scanner",
        "screened": len(candidate_tickers),
        "candidates": results,
        "scan_notes": f"{len(results)} candidates scored >= {min_composite_score}/{MOVEMENT_MAX + THESIS_MAX} "
                      f"out of {len(candidate_tickers)} non-watchlist S&P 500 tickers screened.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_candidate_scan()
    print(result["scan_notes"])
    for c in result["candidates"][:10]:
        print(f"  {c['ticker']} ({c['company']}): composite={c['composite_score']} — {c['notes'][:150]}")
