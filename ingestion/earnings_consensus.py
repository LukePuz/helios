"""
Earnings consensus and surprise history, via yfinance (free, no API key).

Log-only, for research: what the Street expected for each watchlist ticker
at the time of each fetch, so the analyst's "expectation gap" claims can be
checked later against the real consensus. Nothing in the analyst's brief,
the gates or routing reads this; the earnings_consensus_history table is
written here and read only by the research feature store.

Per ticker and fetch, one append-only row (never updated) with:
  - next earnings date (and whether Yahoo gives it as an estimated window)
  - EPS and revenue estimates for the current (0q) and next (+1q) quarter:
    yfinance's columns as given (avg, low, high, numberOfAnalysts,
    yearAgoEps / yearAgoRevenue, growth, currency), plus end_date: the
    quarter's end from Yahoo's earningsTrend endDate, which yfinance's
    parser drops (null when it isn't available)
  - the last quarters of EPS actual vs estimate: yfinance earnings_history
    columns as given (epsActual, epsEstimate, epsDifference, surprisePercent;
    surprisePercent is Yahoo's raw value, not converted). Yahoo currently
    returns 4 quarters; up to 8 are kept.
  - status: ok / partial / no_data / error, and per-part status in parts,
    with the installed yfinance version as parts["yfinance_version"].

Weekdays 17:30 ET via Celery beat (tasks.refresh_earnings_consensus).
"""
import logging
import math
import time
from datetime import date, datetime

log = logging.getLogger(__name__)

SOURCE = "yfinance"
QUARTER_PERIODS = ("0q", "+1q")
HISTORY_QUARTERS = 8
_PARTS = ("calendar", "eps_estimate", "revenue_estimate", "eps_history")

try:
    # Raised inside a Celery task at its soft time limit: let it through, or
    # the loop keeps going until the hard kill with no heartbeat.
    from celery.exceptions import SoftTimeLimitExceeded as _SoftTimeLimit
except ImportError:  # pragma: no cover - celery is a dependency
    class _SoftTimeLimit(Exception):
        pass

try:
    # yfinance raises this on a 429. Yahoo's limit is per IP, shared with the
    # analyst-input fetches, so the run stops at the first one.
    from yfinance.exceptions import YFRateLimitError as _RateLimited
except ImportError:  # pragma: no cover - older yfinance
    class _RateLimited(Exception):
        pass


def _clean(v):
    """A JSON-safe scalar: numpy/pandas values to Python, NaN/inf/NaT to None,
    timestamps to ISO dates."""
    if v is None:
        return None
    try:
        import pandas as pd
        if v is pd.NaT:  # a datetime subclass, so checked first
            return None
    except Exception:
        pass
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (bool, int, str)):
        return v
    try:
        import pandas as pd
        if pd.isna(v):
            return None
        if isinstance(v, pd.Timestamp):
            return v.date().isoformat()
    except Exception:
        pass
    return str(v)


_YF_VERSION: dict = {}


def _yf_version() -> str | None:
    """yfinance.__version__, read once; None if it can't be read."""
    if "v" not in _YF_VERSION:
        try:
            import yfinance
            _YF_VERSION["v"] = str(yfinance.__version__)
        except Exception:
            _YF_VERSION["v"] = None
    return _YF_VERSION["v"]


def _trend_end_dates(yf_ticker) -> dict:
    """{period: endDate} from the earningsTrend items yfinance already
    fetched (Ticker._analysis._earnings_trend; its _get_periodic_df drops
    endDate). {} when they aren't there. Never raises. yfinance <= 1.5 keeps
    the trend list itself; 1.8 keeps {methodology: module} with the list
    under 'trend', where None is Yahoo's default methodology (the one
    earnings_estimate / revenue_estimate read)."""
    try:
        trend = getattr(getattr(yf_ticker, "_analysis", None), "_earnings_trend", None)
        if isinstance(trend, dict):
            module = trend.get(None)
            trend = module.get("trend") if isinstance(module, dict) else None
        out = {}
        for item in trend if isinstance(trend, list) else []:
            if not isinstance(item, dict) or item.get("period") not in QUARTER_PERIODS:
                continue
            end = item.get("endDate")
            if isinstance(end, dict):   # a formatted response: {"raw": epoch, "fmt": "YYYY-MM-DD"}
                end = end.get("fmt")
            out[item["period"]] = _clean(end) if isinstance(end, str) and end else None
        return out
    except Exception:
        return {}


def _with_end_dates(rows: dict | None, yf_ticker) -> dict | None:
    """Each period row of rows with an end_date key (None when unknown)."""
    if not rows:
        return rows
    ends = _trend_end_dates(yf_ticker)
    for period, row in rows.items():
        row["end_date"] = ends.get(period)
    return rows


def _is_empty(df) -> bool:
    return df is None or getattr(df, "empty", True)


def _period_rows(df) -> dict | None:
    """{period: {column: value}} for the current and next quarter, or None."""
    if _is_empty(df):
        return None
    out = {}
    for period in QUARTER_PERIODS:
        if period in df.index:
            row = df.loc[period]
            out[period] = {str(k): _clean(row[k]) for k in row.index}
    return out or None


def _history_rows(df) -> list | None:
    """Newest quarter first, up to HISTORY_QUARTERS, or None."""
    if _is_empty(df):
        return None
    rows = []
    for quarter, row in df.iterrows():
        rows.append({"quarter": _clean(quarter), **{str(k): _clean(row[k]) for k in row.index}})
    rows.sort(key=lambda r: r["quarter"] or "", reverse=True)
    return rows[:HISTORY_QUARTERS] or None


def _next_date(cal, today: date) -> tuple[date | None, bool]:
    """(next earnings date, is_estimate) from Ticker.calendar, the same
    reading as ingestion.earnings_calendar: two dates are an estimated window
    (its start is used), one is a set date."""
    if not isinstance(cal, dict):
        return None, False
    dates = cal.get("Earnings Date") or []
    days = [d.date() if isinstance(d, datetime) else d for d in dates]
    future = [d for d in days if isinstance(d, date) and d >= today]
    # Counted after the past dates are dropped.
    return (min(future), len(future) > 1) if future else (None, False)


def fetch_consensus(ticker: str, yf_ticker=None, today: date | None = None) -> dict:
    """One ticker's snapshot as a row for db.log_earnings_consensus. A part
    that fails is recorded as error:<exception class>; only a rate limit and
    Celery's soft time limit are raised."""
    if yf_ticker is None:
        import yfinance as yf
        yf_ticker = yf.Ticker(ticker)
    today = today or date.today()
    rec = {"source": SOURCE, "next_earnings_date": None, "next_earnings_is_estimate": None,
           "eps_estimate": None, "revenue_estimate": None, "eps_history": None}
    parts, errors = {}, []

    def part(name, read):
        try:
            value = read()
        except (_SoftTimeLimit, _RateLimited):
            raise
        except Exception as e:
            parts[name] = f"error:{type(e).__name__}"
            errors.append(type(e).__name__)
            return None
        parts[name] = "ok" if value else "no_data"
        return value

    def read_next():
        d, is_estimate = _next_date(yf_ticker.calendar, today)
        return (d, is_estimate) if d else None

    nxt = part("calendar", read_next)
    if nxt:
        rec["next_earnings_date"], rec["next_earnings_is_estimate"] = nxt
    rec["eps_estimate"] = part("eps_estimate", lambda: _with_end_dates(
        _period_rows(yf_ticker.earnings_estimate), yf_ticker))
    rec["revenue_estimate"] = part("revenue_estimate", lambda: _with_end_dates(
        _period_rows(yf_ticker.revenue_estimate), yf_ticker))
    rec["eps_history"] = part("eps_history", lambda: _history_rows(yf_ticker.earnings_history))

    has_data = any(parts[p] == "ok" for p in _PARTS)
    if errors:
        rec["status"] = "partial" if has_data else "error"
        rec["error_class"] = errors[0]
    else:
        rec["status"] = "ok" if has_data else "no_data"
        rec["error_class"] = None
    parts["yfinance_version"] = _yf_version()
    rec["parts"] = parts
    return rec


def refresh_all_earnings_consensus(tickers: list[str] | None = None, fetch=fetch_consensus,
                                   write=None, pause_s: float = 0.3, deadline_s: float | None = None,
                                   clock=time.monotonic) -> dict:
    """Fetch and append one row per watchlist ticker. ETFs (no estimates;
    Yahoo answers 404) are not fetched and get a no_data row.

    Stops fetching at the first rate limit, or once ``deadline_s`` seconds
    have passed: every ticker not yet fetched still gets a row, status error
    with error_class rate_limited / deadline, and Yahoo isn't called for it.
    A failed write is counted and the run goes on. Celery's soft time limit
    is raised. Returns counts by status plus 'written' and 'stopped' (None,
    'rate_limited' or 'deadline'). At most one WARNING per run."""
    from knowledge_loader import get_all_tickers, get_etf_tickers
    write_errors: list[str] = []
    if write is None:
        import db

        def write(t, r):
            return db.log_earnings_consensus(t, r, write_errors)
    tickers = get_all_tickers() if tickers is None else tickers
    try:
        etfs = get_etf_tickers()
    except Exception:
        etfs = set()

    counts = {"ok": 0, "partial": 0, "no_data": 0, "error": 0, "written": 0, "stopped": None}
    fetch_errors, no_data = [], []

    def etf_row():
        return {"source": SOURCE, "status": "no_data", "error_class": None,
                "parts": {"skipped": "etf"}}

    def record(ticker, rec):
        if isinstance(rec.get("parts"), dict):
            rec["parts"].setdefault("yfinance_version", _yf_version())
        counts[rec["status"]] += 1
        if rec["status"] in ("error", "partial") and not counts["stopped"]:
            fetch_errors.append(f"{ticker} ({rec.get('error_class')})")
        elif rec["status"] == "no_data":
            no_data.append(ticker)
        try:
            ok = write(ticker, rec)
        except _SoftTimeLimit:
            raise
        except Exception as e:  # db.log_earnings_consensus doesn't raise; a substitute might
            write_errors.append(f"{type(e).__name__}: {e}")
            ok = False
        if ok:
            counts["written"] += 1

    start = clock()
    stop_at = len(tickers)
    for i, ticker in enumerate(tickers):
        if deadline_s is not None and clock() - start >= deadline_s:
            counts["stopped"], stop_at = "deadline", i
            break
        if ticker in etfs:
            record(ticker, etf_row())
            continue
        try:
            rec = fetch(ticker)
        except _SoftTimeLimit:
            raise
        except _RateLimited:
            counts["stopped"], stop_at = "rate_limited", i
            break
        except Exception as e:  # fetch_consensus raises only the two above; a substitute might
            rec = {"source": SOURCE, "status": "error", "error_class": type(e).__name__,
                   "parts": {}}
        record(ticker, rec)
        time.sleep(pause_s)  # be polite to Yahoo's unofficial endpoint

    stopped = counts["stopped"]
    for ticker in tickers[stop_at:]:
        record(ticker, etf_row() if ticker in etfs else
               {"source": SOURCE, "status": "error", "error_class": stopped,
                "parts": {"skipped": stopped}})

    log.info(f"Earnings consensus: {counts['written']}/{len(tickers)} rows written; "
             f"ok={counts['ok']} partial={counts['partial']} no_data={counts['no_data']} "
             f"error={counts['error']}" + (f"; no data for {', '.join(no_data)}" if no_data else ""))
    problems = []
    if stopped:
        problems.append(f"stopped ({stopped}) with {len(tickers) - stop_at} ticker(s) not fetched")
    if fetch_errors:
        problems.append(f"{len(fetch_errors)} ticker(s) with fetch errors ({', '.join(fetch_errors)})")
    unwritten = len(tickers) - counts["written"]
    if unwritten:
        problems.append(f"{unwritten} row(s) not written"
                        + (f" (first error: {write_errors[0]})" if write_errors else ""))
    if problems:
        log.warning("Earnings consensus run: " + "; ".join(problems))
    return counts
