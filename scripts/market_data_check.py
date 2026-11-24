"""Market-data entitlement readout (engineering diagnostic, not a study analysis).

IBKR Warning 10167 ("Requested market data is not subscribed. Displaying
delayed market data.") is the entitlement signal from our own request: before
every market order, ibkr.place_market_order calls reqMarketDataType(3), which
returns live data where the account is entitled and delayed data otherwise.
So a "delayed" quote on an order, with a 10167 line in the log, means no live
subscription for that market.

Per NYSE session over the last --sessions (default 5; the session in
progress counts when today is one), it reports:
  - order_quote activity rows (ibkr._write_order_quote, one per market order,
    BUY and SELL) by md_type_label and side;
  - entry_ref_age activity rows (execution._record_entry_ref_age, one per
    OPEN/ADD that reached the submit point) by md_type_label;
  - log lines with "Warning 10167" (and the tickers they name);
  - log lines with a Polygon "403 NOT_AUTHORIZED" (the snapshot endpoint the
    plan doesn't include). The ingestion code logs that once per process and
    suppresses repeats, so a count is "processes that hit it", not requests.
Then the newest of each (all time for the rows; within the scan for logs).
Counts, labels, tickers and times only: no prices, quantities or values.

Logs read: the clone's root *.log and logs/*.log, nothing else (never
logs/db_write_spill.jsonl or logs/telegram_outbox.jsonl, nor rotated *.log.N).
Each file is read from its last --max-log-bytes only (default 32 MiB), and only
lines dated from the window's start are counted, so a large log can't make the
readout slow; the output says when a file was cut. Log times are the host's
local time.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/market_data_check.py [--sessions 5] [--json]

Read-only: SUPABASE_PORT must be 6543 (the script exits otherwise). One
transaction, SET TRANSACTION READ ONLY with a statement timeout, rolled back.
The log files are opened read-only.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
DEFAULT_SESSIONS = 5
STATEMENT_TIMEOUT_MS = 15000
MAX_LOG_BYTES = 32 * 1024 * 1024
STAGES = ("order_quote", "entry_ref_age")
NEVER_READ = ("db_write_spill.jsonl", "telegram_outbox.jsonl")
_TS_RE = re.compile(rb"^\[?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_SYMBOL_RE = re.compile(rb"symbol='([A-Z.]{1,8})'")
W10167 = b"Warning 10167"
P403 = b"403 NOT_AUTHORIZED"


# ---------------------------------------------------------------- sessions

def _entry_funnel():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_funnel.py")
    spec = importlib.util.spec_from_file_location("entry_funnel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session_window(now: datetime, n: int, ef=None) -> tuple[list[date], datetime]:
    """The last n NYSE sessions (today when it is one) and the UTC start of
    the first one's rows (entry_funnel.window)."""
    ef = ef or _entry_funnel()
    d = now.astimezone(NY).date()
    while not ef._is_session(d):
        d -= timedelta(days=1)
    sessions = ef.sessions_until(d, n)
    start, _ = ef.window(sessions)
    return sessions, start


# ---------------------------------------------------------------- logs

def log_paths(root: str) -> list[str]:
    """root/*.log and root/logs/*.log, never the NEVER_READ files."""
    paths = sorted(glob.glob(os.path.join(root, "*.log"))
                   + glob.glob(os.path.join(root, "logs", "*.log")))
    return [p for p in paths if p.endswith(".log") and os.path.basename(p) not in NEVER_READ
            and os.path.isfile(p)]


def scan_log(path: str, since: datetime, max_bytes: int = MAX_LOG_BYTES, tz=None) -> dict:
    """The 10167 and Polygon 403 lines of one log from since on, reading only
    its last max_bytes. tz: the zone of the log's naive times (default the
    host's). Returns {"w10167": [(time, ticker|None)], "p403": [time],
    "cut": bool, "error": str|None}."""
    tz = tz or datetime.now().astimezone().tzinfo
    out = {"path": path, "w10167": [], "p403": [], "cut": False, "error": None}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()                  # drop the partial first line
                out["cut"] = True
            for line in f:
                hit10167, hit403 = W10167 in line, (P403 in line and b"olygon" in line)
                if not (hit10167 or hit403):
                    continue
                m = _TS_RE.match(line)
                if not m:
                    continue
                try:
                    t = datetime.strptime(m.group(1).decode(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
                except ValueError:
                    continue
                if t < since:
                    continue
                if hit10167:
                    s = _SYMBOL_RE.search(line)
                    out["w10167"].append((t, s.group(1).decode() if s else None))
                if hit403:
                    out["p403"].append(t)
    except OSError as e:
        out["error"] = type(e).__name__
    return out


# ---------------------------------------------------------------- counting

def _aware(t):
    if t is None:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def build(rows: list[dict], newest: dict, scans: list[dict], sessions: list[date], ef=None) -> dict:
    """rows: [{stage, ts, md_type_label, side}] since the window start;
    newest: {stage: (ts, md_type_label)} all time; scans: scan_log results."""
    ef = ef or _entry_funnel()
    per = {d.isoformat(): {"order_quote": Counter(), "order_quote_side": Counter(),
                           "entry_ref_age": Counter(), "w10167": 0, "w10167_tickers": Counter(),
                           "p403": 0} for d in sessions}

    def bucket(t):
        s = ef.session_of(_aware(t))
        return per.get(s.isoformat()) if s else None

    for r in rows:
        b = bucket(r["ts"])
        if b is None or r["stage"] not in STAGES:
            continue
        label = r.get("md_type_label") or "none"
        b[r["stage"]][label] += 1
        if r["stage"] == "order_quote":
            b["order_quote_side"][r.get("side") or "none"] += 1
    newest_log = {"w10167": None, "p403": None}
    for sc in scans:
        for t, tk in sc["w10167"]:
            b = bucket(t)
            if b is not None:
                b["w10167"] += 1
                b["w10167_tickers"][tk or "?"] += 1
            newest_log["w10167"] = max(filter(None, (newest_log["w10167"], t)))
        for t in sc["p403"]:
            b = bucket(t)
            if b is not None:
                b["p403"] += 1
            newest_log["p403"] = max(filter(None, (newest_log["p403"], t)))
    sess = [{"session": k, "order_quote": dict(v["order_quote"]),
             "order_quote_side": dict(v["order_quote_side"]),
             "entry_ref_age": dict(v["entry_ref_age"]), "w10167": v["w10167"],
             "w10167_tickers": dict(v["w10167_tickers"]), "p403": v["p403"]}
            for k, v in per.items()]
    tot = {"order_quote": Counter(), "entry_ref_age": Counter(), "w10167": 0, "p403": 0}
    for s in sess:
        tot["order_quote"].update(s["order_quote"])
        tot["entry_ref_age"].update(s["entry_ref_age"])
        tot["w10167"] += s["w10167"]
        tot["p403"] += s["p403"]

    def et(t):
        return _aware(t).astimezone(NY).isoformat(timespec="minutes") if t else None
    return {"sessions": sess,
            "totals": {"order_quote": dict(tot["order_quote"]),
                       "entry_ref_age": dict(tot["entry_ref_age"]),
                       "w10167": tot["w10167"], "p403": tot["p403"]},
            "newest": {"order_quote": {"at": et((newest.get("order_quote") or (None, None))[0]),
                                       "md_type_label": (newest.get("order_quote") or (None, None))[1]},
                       "entry_ref_age": {"at": et((newest.get("entry_ref_age") or (None, None))[0]),
                                         "md_type_label": (newest.get("entry_ref_age") or (None, None))[1]},
                       "w10167": et(newest_log["w10167"]), "p403": et(newest_log["p403"])},
            "logs": [{"path": os.path.relpath(sc["path"], ROOT) if sc["path"].startswith(ROOT)
                      else sc["path"], "cut": sc["cut"], "error": sc["error"],
                      "w10167": len(sc["w10167"]), "p403": len(sc["p403"])} for sc in scans]}


def _dist(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


def render_text(res: dict, meta: dict) -> str:
    lines = [f"Market-data check — {meta['env']}, clone {meta['clone']}",
             f"Generated {meta['generated']} ET. Last {len(res['sessions'])} sessions "
             f"(rows and log lines from {meta['window_start']} ET).",
             "IBKR Warning 10167 is the entitlement signal from our own reqMarketDataType(3): "
             "'delayed' means no live subscription.", ""]
    for s in res["sessions"]:
        lines.append(f"## {s['session']}")
        lines.append(f"  order_quote: {_dist(s['order_quote'])} (side: {_dist(s['order_quote_side'])})")
        lines.append(f"  entry_ref_age: {_dist(s['entry_ref_age'])}")
        lines.append(f"  Warning 10167 lines: {s['w10167']}"
                     + (f" ({_dist(s['w10167_tickers'])})" if s["w10167"] else ""))
        lines.append(f"  Polygon 403 lines: {s['p403']}")
        lines.append("")
    t, n = res["totals"], res["newest"]
    lines += ["## Totals",
              f"  order_quote: {_dist(t['order_quote'])}; entry_ref_age: {_dist(t['entry_ref_age'])}; "
              f"10167 lines {t['w10167']}; Polygon 403 lines {t['p403']} (logged once per process).",
              f"  newest order_quote: {n['order_quote']['at'] or 'none'} "
              f"({n['order_quote']['md_type_label'] or '-'}); newest entry_ref_age: "
              f"{n['entry_ref_age']['at'] or 'none'} ({n['entry_ref_age']['md_type_label'] or '-'}); "
              f"newest 10167 line: {n['w10167'] or 'none in the scan'}; newest Polygon 403 line: "
              f"{n['p403'] or 'none in the scan'}.", "",
              "## Logs scanned"]
    for lg in res["logs"]:
        flag = " (read from its last part only)" if lg["cut"] else ""
        err = f" read error {lg['error']}" if lg["error"] else ""
        lines.append(f"  {lg['path']}: 10167 {lg['w10167']}, 403 {lg['p403']}{flag}{err}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def fetch(conn, env: str, since: datetime):
    """(rows since the window start, {stage: (newest ts, its md_type_label)})
    in one READ ONLY transaction, rolled back."""
    rows, newest = [], {}
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT stage, ts, extra_json->>'md_type_label', extra_json->>'side' "
                        "FROM activity_feed WHERE stage = ANY(%s) AND ts >= %s "
                        "AND COALESCE(extra_json->>'env', %s) = %s",
                        (list(STAGES), since, env, env))
            rows = [{"stage": s, "ts": t, "md_type_label": m, "side": sd}
                    for s, t, m, sd in cur.fetchall()]
            for stage in STAGES:
                cur.execute("SELECT ts, extra_json->>'md_type_label' FROM activity_feed "
                            "WHERE stage = %s AND COALESCE(extra_json->>'env', %s) = %s "
                            "ORDER BY ts DESC LIMIT 1", (stage, env, env))
                r = cur.fetchone()
                newest[stage] = (r[0], r[1]) if r else (None, None)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
    return rows, newest


def _git_head(root: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Market-data entitlement readout")
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    ap.add_argument("--max-log-bytes", type=int, default=MAX_LOG_BYTES,
                    help="read at most this many bytes from the end of each log")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    if args.sessions < 1:
        ap.error("--sessions must be 1 or more")
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    import config
    import db
    ef = _entry_funnel()
    now = datetime.now(timezone.utc)
    sessions, start = session_window(now, args.sessions, ef)
    conn = db._get_conn()
    try:
        rows, newest = fetch(conn, config.TRADING_ENV, start)
    finally:
        db._put_conn(conn)
    scans = [scan_log(p, start, args.max_log_bytes) for p in log_paths(ROOT)]
    res = build(rows, newest, scans, sessions, ef)
    meta = {"env": config.TRADING_ENV, "clone": _git_head(ROOT),
            "generated": now.astimezone(NY).isoformat(timespec="minutes"),
            "window_start": start.astimezone(NY).isoformat(timespec="minutes"),
            "max_log_bytes": args.max_log_bytes}
    if args.json:
        print(json.dumps({"meta": meta, **res}, indent=2, default=str))
    else:
        print(render_text(res, meta), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
