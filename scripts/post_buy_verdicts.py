"""Verdicts after each filled buy (engineering readout, not a study analysis).

For every filled OPEN/ADD BUY order in this env (approval is_test excluded),
it lists every later analyst verdict on the same ticker, from the fill to
the end of the Nth session after the fill's session (--sessions, default 5;
0 = the fill's session only), with:

  - minutes after the fill and the session offset (0 = the fill's session);
  - the verdict's action, conviction and UDR;
  - "bar": whether the verdict meets the entry bar as this clone defines it
    (DESCRIPTIVE ONLY: a held review is not an entry, and no verdict here
    was routed as one; the column only says where each verdict sits against
    the numbers below);
  - held_review (the bot's own decision.held_review flag) and the raw
    trigger path / source from provenance.

Then counts per buy and pooled. No metric, threshold or interpretation is
chosen here; the counts are for Luke's held-review / trim_exit decisions.

The entry bar, read from this clone (nothing hard-coded):
  - conviction: the regime rule's number in agents/analyst.py
    analyze_candidate (`regime_rule_fired = (... conviction < N)`, the c>=N
    OPEN rule for the regimes in _ENTRY_RULE_REGIMES). It is a literal in
    that line, not a named constant, so it is read from the source text;
    when the line can't be found the bar is "unknown" and no verdict gets a
    bar value.
  - UDR: config.MIN_UPSIDE_DOWNSIDE_RATIO. agents/analyst.py's
    _ENTRY_UDR_FLOOR (the gate's own copy) is read too and shown; the header
    says when the two differ.
  A verdict meets the bar when conviction >= the conviction bar and UDR >=
  the UDR bar; a missing UDR or conviction is "unknown", never below.

Fields as the bot reads them (db.get_recent_verdicts): conviction is
round(signals.confidence * 10), the final conviction; UDR is the model's
stated decision.model_output.upside_downside_ratio. Verdicts are signals
rows not in db._NON_ANALYST_SIGNALS. Sessions are NYSE sessions by ET date
(entry_funnel: a row on a weekend or holiday counts toward the next one).
Fill time is orders.filled_at; a buy without one is listed as skipped.
A filled_at later than the order row's own last write (updated_at, plus
SUSPECT_FILL_SLACK_S) can't be a real fill time: before 11-06 a TWS
execution time could be stored 5 h late (tests/test_fill_time_tz.py, the
MSFT BUY of 2026-11-03). Such a buy is flagged (fill_time_flag) and, by
default, still uses filled_at as recorded; --suspect-fill submitted uses
the order's submitted_at for the flagged buys only. Which one a readout
uses is not decided here.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/post_buy_verdicts.py [--sessions 5] [--since 2026-11-01] \
        [--suspect-fill recorded|submitted] [--json]

Read-only: SUPABASE_PORT must be 6543 (the script exits otherwise). One
transaction, SET TRANSACTION READ ONLY with a statement timeout, rolled back.
No prices, quantities or account values: conviction and UDR only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
DEFAULT_SESSIONS = 5
STATEMENT_TIMEOUT_MS = 20000
BUY_ACTIONS = ("OPEN", "ADD")
SUSPECT_FILL_SLACK_S = 60
SUSPECT_FILL_MODES = ("recorded", "submitted")
_REGIME_RULE_RE = re.compile(
    r"regime_rule_fired\s*=\s*\(action\s*==\s*\"OPEN\"\s+and\s+regime_str\s+in\s+"
    r"_ENTRY_RULE_REGIMES\s+and\s+conviction\s*<\s*(\d+)\)")
_UDR_FLOOR_RE = re.compile(r"^_ENTRY_UDR_FLOOR\s*=\s*([0-9.]+)", re.M)


# ---------------------------------------------------------------- the bar

def entry_bar(cfg, analyst_source: str | None) -> dict:
    """{"conviction": int | None, "udr": float | None, "analyst_udr_floor":
    float | None, "sources": {...}} from config and agents/analyst.py's text."""
    conv = floor = None
    if analyst_source:
        m = _REGIME_RULE_RE.search(analyst_source)
        conv = int(m.group(1)) if m else None
        f = _UDR_FLOOR_RE.search(analyst_source)
        floor = float(f.group(1)) if f else None
    udr = getattr(cfg, "MIN_UPSIDE_DOWNSIDE_RATIO", None)
    udr = float(udr) if isinstance(udr, (int, float)) and not isinstance(udr, bool) else None
    return {"conviction": conv, "udr": udr, "analyst_udr_floor": floor,
            "sources": {"conviction": "agents/analyst.py regime_rule_fired (c >= N OPEN rule)",
                        "udr": "config.MIN_UPSIDE_DOWNSIDE_RATIO",
                        "analyst_udr_floor": "agents/analyst.py _ENTRY_UDR_FLOOR"}}


def meets_bar(conviction, udr, bar: dict) -> bool | None:
    """True / False against the bar; None when the bar or a value is unknown."""
    if bar.get("conviction") is None or bar.get("udr") is None:
        return None
    if conviction is None or udr is None:
        return None
    return conviction >= bar["conviction"] and udr >= bar["udr"]


def _analyst_source() -> str | None:
    try:
        with open(os.path.join(ROOT, "agents", "analyst.py")) as f:
            return f.read()
    except OSError:
        return None


# ---------------------------------------------------------------- sessions

def _entry_funnel():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_funnel.py")
    spec = importlib.util.spec_from_file_location("entry_funnel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _aware(t):
    if t is None:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def sessions_after(ef, first: date, n: int) -> list[date]:
    """first and the n sessions after it."""
    out, d = [first], first
    while len(out) < n + 1:
        d += timedelta(days=1)
        if ef._is_session(d):
            out.append(d)
    return out


def buy_window(ef, filled_at: datetime, n: int) -> tuple[list[date], datetime]:
    """(the fill's session and the n after it, the UTC end of the last one)."""
    first = ef.session_of(_aware(filled_at))
    sessions = sessions_after(ef, first, n)
    end = datetime.combine(sessions[-1] + timedelta(days=1), time(0), tzinfo=NY)
    return sessions, end.astimezone(timezone.utc)


def fill_time(b: dict, suspect_fill: str = "recorded") -> tuple[datetime | None, str | None, str | None]:
    """(the fill time used, its source column, the flag) for one buy row.
    The flag is "filled_at_after_last_write" when filled_at is later than
    updated_at + SUSPECT_FILL_SLACK_S."""
    filled, updated, submitted = (_aware(b.get("filled_at")), _aware(b.get("updated_at")),
                                  _aware(b.get("submitted_at")))
    if filled is None:
        return None, None, None
    flag = None
    if updated is not None and filled > updated + timedelta(seconds=SUSPECT_FILL_SLACK_S):
        flag = "filled_at_after_last_write"
    if flag and suspect_fill == "submitted" and submitted is not None:
        return submitted, "submitted_at", flag
    return filled, "filled_at", flag


# ---------------------------------------------------------------- counting

def _conviction(confidence):
    try:
        return round(float(confidence) * 10) if confidence is not None else None
    except (TypeError, ValueError):
        return None


def _udr(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def build(buys: list[dict], verdicts: list[dict], bar: dict, n_sessions: int, ef=None,
          suspect_fill: str = "recorded") -> dict:
    """buys: [{approval_id, ticker, action, filled_at, submitted_at,
    updated_at}] (the last two optional), verdicts: [{id,
    ticker, ts, kind, confidence, udr, held_review, path, source}].
    Returns {"buys": [...], "pooled": {...}}."""
    ef = ef or _entry_funnel()
    out, pooled = [], {"buys": 0, "buys_skipped_no_fill_time": 0, "buys_fill_time_flagged": 0,
                       "verdicts": 0,
                       "verdicts_same_session": 0, "meets_bar": 0, "below_bar": 0,
                       "bar_unknown": 0, "same_session_below_bar": 0,
                       "held_review_verdicts": 0, "buys_with_same_session_below_bar": 0}
    seen, seen_below = set(), set()     # distinct verdicts: buy windows can overlap
    by_ticker: dict = {}
    for v in verdicts:
        by_ticker.setdefault(v["ticker"], []).append(v)
    for b in sorted(buys, key=lambda b: (_aware(b.get("filled_at")) or datetime.max.replace(
            tzinfo=timezone.utc), b.get("approval_id") or 0)):
        rec = {"approval_id": b.get("approval_id"), "ticker": b["ticker"], "action": b["action"],
               "filled_at": None, "fill_time_source": None, "fill_time_flag": None,
               "sessions": [], "verdicts": [], "counts": None, "skipped": None}
        fill, rec["fill_time_source"], rec["fill_time_flag"] = fill_time(b, suspect_fill)
        if fill is None:
            rec["skipped"] = "no filled_at on the order"
            pooled["buys_skipped_no_fill_time"] += 1
            out.append(rec)
            continue
        sessions, end = buy_window(ef, fill, n_sessions)
        rec["filled_at"] = fill.astimezone(NY).isoformat(timespec="minutes")
        rec["sessions"] = [d.isoformat() for d in sessions]
        offsets = {d: i for i, d in enumerate(sessions)}
        rows = []
        for v in sorted(by_ticker.get(b["ticker"], []), key=lambda v: (_aware(v["ts"]), v.get("id") or 0)):
            ts = _aware(v["ts"])
            if ts is None or not (fill < ts < end):
                continue
            conv, udr = _conviction(v.get("confidence")), _udr(v.get("udr"))
            rows.append({"signal_id": v.get("id"),
                         "minutes_after_fill": round((ts - fill).total_seconds() / 60, 1),
                         "session_offset": offsets.get(ef.session_of(ts)),
                         "at": ts.astimezone(NY).isoformat(timespec="minutes"),
                         "action": v.get("kind"), "conviction": conv, "udr": udr,
                         "meets_bar": meets_bar(conv, udr, bar),
                         "held_review": v.get("held_review"),
                         "trigger_path": v.get("path"), "trigger_source": v.get("source")})
        same = [r for r in rows if r["session_offset"] == 0]
        below_same = [r for r in same if r["meets_bar"] is False]
        counts = {"verdicts": len(rows), "verdicts_same_session": len(same),
                  "meets_bar": sum(1 for r in rows if r["meets_bar"] is True),
                  "below_bar": sum(1 for r in rows if r["meets_bar"] is False),
                  "bar_unknown": sum(1 for r in rows if r["meets_bar"] is None),
                  "same_session_below_bar": len(below_same),
                  "held_review_verdicts": sum(1 for r in rows if r["held_review"] is True),
                  "first_below_bar_minutes": next((r["minutes_after_fill"] for r in rows
                                                   if r["meets_bar"] is False), None),
                  "by_session": [{"session": s, "verdicts": sum(1 for r in rows if r["session_offset"] == i),
                                  "below_bar": sum(1 for r in rows if r["session_offset"] == i
                                                   and r["meets_bar"] is False)}
                                 for i, s in enumerate(rec["sessions"])]}
        rec["verdicts"], rec["counts"] = rows, counts
        pooled["buys"] += 1
        pooled["buys_fill_time_flagged"] += 1 if rec["fill_time_flag"] else 0
        for k in ("verdicts", "verdicts_same_session", "meets_bar", "below_bar", "bar_unknown",
                  "same_session_below_bar", "held_review_verdicts"):
            pooled[k] += counts[k]
        pooled["buys_with_same_session_below_bar"] += 1 if below_same else 0
        for r in rows:
            key = r["signal_id"] if r["signal_id"] is not None else (b["ticker"], r["at"])
            seen.add(key)
            if r["meets_bar"] is False:
                seen_below.add(key)
        out.append(rec)
    # The per-buy sums count a verdict once per buy whose window holds it
    # (two buys of one ticker a session apart share their later verdicts).
    pooled["distinct_verdicts"], pooled["distinct_below_bar"] = len(seen), len(seen_below)
    return {"buys": out, "pooled": pooled}


# ---------------------------------------------------------------- output

def _fmt(v, nd=None):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if nd is not None and isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_text(result: dict, meta: dict) -> str:
    bar = meta["bar"]
    bar_txt = (f"conviction >= {_fmt(bar['conviction'])} ({bar['sources']['conviction']}) and "
               f"UDR >= {_fmt(bar['udr'])} ({bar['sources']['udr']})")
    lines = [f"Verdicts after filled buys — {meta['env']}, clone {meta['clone']}",
             f"Generated {meta['generated']} ET. Buys filled since {meta['since']}; each followed "
             f"through its fill session and the next {meta['n_sessions']}. Flagged fill times: "
             f"{meta.get('suspect_fill', 'recorded')} (see the module docstring).",
             f"Entry bar (DESCRIPTIVE ONLY, not applied to these verdicts by the bot): {bar_txt}."]
    if bar["analyst_udr_floor"] is not None and bar["udr"] is not None \
            and abs(bar["analyst_udr_floor"] - bar["udr"]) > 1e-9:
        lines.append(f"Note: analyst._ENTRY_UDR_FLOOR is {bar['analyst_udr_floor']}, not the config "
                     f"value used above.")
    if bar["conviction"] is None:
        lines.append("Note: the regime rule's conviction number was not found in agents/analyst.py; "
                     "bar is unknown for every verdict.")
    lines.append("")
    head = ("+min", "sess", "at (ET)", "action", "c", "UDR", "bar", "held_review", "path / source")
    for b in result["buys"]:
        title = f"{b['ticker']} {b['action']} (approval {b['approval_id']})"
        if b["skipped"]:
            lines += [f"## {title}: skipped, {b['skipped']}", ""]
            continue
        lines.append(f"## {title}, filled {b['filled_at']} ET"
                     + (f" ({b['fill_time_source']}; FLAG {b['fill_time_flag']})"
                        if b["fill_time_flag"] else ""))
        rows = [(_fmt(r["minutes_after_fill"]), _fmt(r["session_offset"]), r["at"][5:16].replace("T", " "),
                 _fmt(r["action"]), _fmt(r["conviction"]), _fmt(r["udr"], 2),
                 {True: "meets", False: "below", None: "unknown"}[r["meets_bar"]],
                 _fmt(r["held_review"]), f"{_fmt(r['trigger_path'])} / {_fmt(r['trigger_source'])}")
                for r in b["verdicts"]]
        if rows:
            widths = [max(len(x[i]) for x in rows + [head]) for i in range(len(head))]
            for r in [head] + rows:
                lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        else:
            lines.append("  no later verdicts in the window")
        c = b["counts"]
        lines.append(f"  counts: {c['verdicts']} verdicts ({c['verdicts_same_session']} same session); "
                     f"bar meets {c['meets_bar']}, below {c['below_bar']}, unknown {c['bar_unknown']}; "
                     f"same-session below {c['same_session_below_bar']}; held_review "
                     f"{c['held_review_verdicts']}; first below at "
                     f"{_fmt(c['first_below_bar_minutes'])} min.")
        lines.append("  by session: " + ", ".join(f"{s['session']} {s['verdicts']}/{s['below_bar']} below"
                                                for s in c["by_session"]))
        lines.append("")
    p = result["pooled"]
    lines += ["## Pooled",
              f"  buys {p['buys']} (skipped without a fill time: {p['buys_skipped_no_fill_time']}; "
              f"fill time flagged: {p['buys_fill_time_flagged']}); "
              f"verdicts {p['verdicts']} ({p['verdicts_same_session']} same session), summed over "
              f"buys; distinct verdicts {p['distinct_verdicts']} (below the bar "
              f"{p['distinct_below_bar']}), as buy windows can overlap.",
              f"  bar: meets {p['meets_bar']}, below {p['below_bar']}, unknown {p['bar_unknown']}; "
              f"same-session below {p['same_session_below_bar']}; held_review verdicts "
              f"{p['held_review_verdicts']}.",
              f"  buys with at least one same-session verdict below the bar: "
              f"{p['buys_with_same_session_below_bar']} of {p['buys']}."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def fetch(conn, env: str, since: datetime, n_sessions: int, non_analyst: tuple, ef=None):
    """(buys, verdicts) in one READ ONLY transaction, rolled back. Verdicts
    are read per ticker over every candidate fill time (filled_at and
    submitted_at), so either --suspect-fill choice finds its rows."""
    ef = ef or _entry_funnel()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT pa.id, o.ticker, pa.action, o.filled_at, o.submitted_at, "
                        "o.updated_at FROM orders o "
                        "JOIN pending_approvals pa ON pa.id = o.approval_id "
                        "WHERE o.trading_env = %s AND o.side = 'BUY' AND o.status = 'filled' "
                        "AND pa.action = ANY(%s) AND COALESCE(pa.is_test, false) = false "
                        "AND COALESCE(o.filled_at, o.created_at) >= %s "
                        "ORDER BY COALESCE(o.filled_at, o.created_at), o.id",
                        (env, list(BUY_ACTIONS), since))
            buys = [{"approval_id": i, "ticker": t, "action": a, "filled_at": f,
                     "submitted_at": sub_at, "updated_at": upd}
                    for i, t, a, f, sub_at, upd in cur.fetchall()]
            verdicts = []
            for tk in sorted({b["ticker"] for b in buys if b["filled_at"] is not None}):
                fills = [_aware(t) for b in buys if b["ticker"] == tk and b["filled_at"] is not None
                         for t in (b["filled_at"], b.get("submitted_at")) if t is not None]
                end = max(buy_window(ef, f, n_sessions)[1] for f in fills)
                cur.execute("SELECT id, symbol, timestamp, signal, confidence, "
                            "decision->'model_output'->>'upside_downside_ratio', "
                            "decision->'held_review', provenance->'trigger'->>'path', "
                            "provenance->'trigger'->>'source' FROM signals "
                            "WHERE trading_env = %s AND symbol = %s AND timestamp > %s "
                            "AND timestamp < %s AND NOT (signal = ANY(%s)) ORDER BY timestamp, id",
                            (env, tk, min(fills), end, list(non_analyst)))
                verdicts += [{"id": i, "ticker": s, "ts": t, "kind": k, "confidence": c, "udr": u,
                              "held_review": h if isinstance(h, bool) else None,
                              "path": p, "source": src}
                             for i, s, t, k, c, u, h, p, src in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
    return buys, verdicts


def _git_head(root: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def parse_since(text: str | None) -> datetime:
    """ET 'YYYY-MM-DD'; default: no lower bound (every filled buy)."""
    if not text:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return datetime.combine(date.fromisoformat(text), time(0), tzinfo=NY)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verdicts after each filled OPEN/ADD buy")
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                    help="sessions after the fill's own to include (0 = the fill's session only)")
    ap.add_argument("--since", help="ET 'YYYY-MM-DD': buys filled on or after; default all")
    ap.add_argument("--suspect-fill", choices=SUSPECT_FILL_MODES, default="recorded",
                    help="flagged fill times (filled_at after the row's last write): use filled_at "
                         "as recorded (default) or the order's submitted_at")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    if args.sessions < 0:
        ap.error("--sessions must be 0 or more")
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    import config
    import db
    ef = _entry_funnel()
    since = parse_since(args.since)
    conn = db._get_conn()
    try:
        buys, verdicts = fetch(conn, config.TRADING_ENV, since, args.sessions,
                               tuple(db._NON_ANALYST_SIGNALS), ef)
    finally:
        db._put_conn(conn)
    bar = entry_bar(config, _analyst_source())
    result = build(buys, verdicts, bar, args.sessions, ef, suspect_fill=args.suspect_fill)
    meta = {"env": config.TRADING_ENV, "clone": _git_head(ROOT),
            "generated": datetime.now(timezone.utc).astimezone(NY).isoformat(timespec="minutes"),
            "since": args.since or "the first buy", "n_sessions": args.sessions,
            "suspect_fill": args.suspect_fill, "bar": bar,
            "bar_note": "descriptive only"}
    if args.json:
        print(json.dumps({"meta": meta, **result}, indent=2, default=str))
    else:
        print(render_text(result, meta), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
