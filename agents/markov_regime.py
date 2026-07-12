"""
Markov Regime-Persistence Agent — empirical trend-stickiness signal for SPY.

Classifies each trading day into a 20-day trailing-return "state":
    bull      : 20-trading-day return > +5%
    sideways  : 20-trading-day return in [-5%, +5%]
    bear      : 20-trading-day return < -5%

From the resulting state sequence we build an empirical first-order Markov
transition matrix: P(next state = s2 | current state = s1). The matrix is
estimated from NON-OVERLAPPING 20-day windows only. Sampling every day would
produce a state sequence where consecutive entries share 19 of their 20
underlying closes — the state literally cannot change quickly, which makes
the chain look far "stickier" (higher self-transition probability) than the
market actually is. Striding the sample by the same 20-day window used to
define the state removes that overlap: each sampled transition compares two
independent, back-to-back 20-day return windows.

The directional signal exposed to the regime scorer is:
    signal = P(next=bull | current state) - P(next=bear | current state)
which is in [-1, 1] and captures both the direction and the confidence of
the persistence/mean-reversion tendency from the current state.

This module has no dependency on numpy — transition matrices and
projections are plain nested dicts, which keeps them trivially testable.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATES = ("bull", "sideways", "bear")
WINDOW = 20
_BULL_THRESHOLD = 0.05
_BEAR_THRESHOLD = -0.05

# ---------------------------------------------------------------------------
# Cache — one fetch per calendar day
# ---------------------------------------------------------------------------
_packet_cache: dict | None = None
_packet_cache_date: str | None = None


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def classify_state(pct_change: float) -> str:
    """
    Classifies a fractional return (0.05 == +5%) into a state label.
    Strictly greater/less than the threshold — exactly +/-5% is "sideways".
    """
    if pct_change > _BULL_THRESHOLD:
        return "bull"
    if pct_change < _BEAR_THRESHOLD:
        return "bear"
    return "sideways"


def compute_states(closes: list[float], window: int = WINDOW) -> list[str]:
    """
    Rolling `window`-day return state for each index i >= window, where the
    return at i is closes[i] / closes[i - window] - 1. The returned list is
    aligned so its last element is today's state (i.e. computed from the
    most recent `window`-day return).
    """
    states = []
    for i in range(window, len(closes)):
        prior = closes[i - window]
        if prior == 0:
            states.append("sideways")
            continue
        pct_change = closes[i] / prior - 1
        states.append(classify_state(pct_change))
    return states


def build_transition_matrix(states: list[str], stride: int = WINDOW) -> tuple[dict, dict]:
    """
    Builds an empirical first-order transition matrix by sampling `states`
    at non-overlapping strides (indices 0, stride, 2*stride, ...) and
    counting transitions between consecutive sampled states.

    Returns (matrix, counts):
      matrix[s1][s2] — row-normalized P(next=s2 | current=s1)
      counts[s1]     — number of sampled transitions observed out of s1

    Rows with zero observed samples get a uniform 1/3 distribution.
    """
    sampled = states[0::stride]

    raw_counts = {s1: {s2: 0 for s2 in STATES} for s1 in STATES}
    row_totals = {s1: 0 for s1 in STATES}

    for i in range(len(sampled) - 1):
        s1, s2 = sampled[i], sampled[i + 1]
        raw_counts[s1][s2] += 1
        row_totals[s1] += 1

    matrix = {}
    for s1 in STATES:
        total = row_totals[s1]
        if total == 0:
            matrix[s1] = {s2: 1 / 3 for s2 in STATES}
        else:
            matrix[s1] = {s2: raw_counts[s1][s2] / total for s2 in STATES}

    return matrix, row_totals


def k_step_projection(matrix: dict, state: str, k: int) -> dict:
    """
    Probability distribution over STATES after `k` steps starting from
    `state`, via repeated matrix-vector multiplication. Plain dicts —
    no numpy needed.
    """
    dist = {s: 0.0 for s in STATES}
    dist[state] = 1.0

    for _ in range(k):
        next_dist = {s: 0.0 for s in STATES}
        for s1 in STATES:
            p1 = dist[s1]
            if p1 == 0.0:
                continue
            row = matrix[s1]
            for s2 in STATES:
                next_dist[s2] += p1 * row[s2]
        dist = next_dist

    return dist


# ---------------------------------------------------------------------------
# Data fetch + packet assembly
# ---------------------------------------------------------------------------

def _unavailable(reason: str) -> dict:
    log.warning(f"Markov regime packet unavailable: {reason}")
    return {"available": False, "notes": reason}


def get_markov_packet() -> dict:
    """
    Fetches ~10 years of SPY daily closes, computes the empirical Markov
    transition matrix over 20-day return states, and returns a packet
    describing the current state's persistence/direction signal.

    Cached in-memory per calendar day so repeated calls within the same day
    don't re-fetch from yfinance.
    """
    global _packet_cache, _packet_cache_date

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _packet_cache is not None and _packet_cache_date == today:
        return _packet_cache

    try:
        import yfinance as yf

        hist = yf.Ticker("SPY").history(period="10y")
        if hist is None or hist.empty or "Close" not in hist:
            packet = _unavailable("SPY history fetch returned no data")
        else:
            closes_series = hist["Close"].dropna()
            closes = [float(c) for c in closes_series.tolist()]

            if len(closes) < 500 or any(c <= 0 for c in closes):
                packet = _unavailable(
                    f"Insufficient/invalid SPY close history for Markov analysis "
                    f"({len(closes)} closes)"
                )
            else:
                states = compute_states(closes)
                if not states:
                    packet = _unavailable("Not enough closes to compute 20-day return states")
                else:
                    matrix, counts = build_transition_matrix(states)
                    state = states[-1]
                    stickiness = matrix[state][state]
                    signal = matrix[state]["bull"] - matrix[state]["bear"]
                    row_samples = counts[state]
                    projection_5d = k_step_projection(matrix, state, 5)

                    dates = closes_series.index
                    as_of = dates[-1].strftime("%Y-%m-%d") if len(dates) else today

                    notes = (
                        f"SPY 20d return state={state}, stickiness={stickiness:.2f}, "
                        f"signal={signal:+.2f} (n={row_samples})"
                    )

                    packet = {
                        "available": True,
                        "as_of": as_of,
                        "state": state,
                        "stickiness": stickiness,
                        "signal": signal,
                        "row_samples": row_samples,
                        "matrix": matrix,
                        "projection_5d": projection_5d,
                        "notes": notes,
                    }
    except Exception as e:
        packet = _unavailable(f"Markov regime fetch failed: {e}")

    # Only cache successful packets — a transient fetch failure at the 9:30
    # assessment must not blank the signal for intraday reassessments too.
    if packet.get("available"):
        _packet_cache = packet
        _packet_cache_date = today
    return packet
