"""Shadow outcomes readout: counts, and the forward returns after the rows,
for the record-only shadows that had no readout (shadow registry 11-21).
Engineering diagnostic, not a study analysis. Sections:

  held_exit     TRIM_EXIT_RULE rows' decision.held_exit_candidates: fires
                per candidate rule (True / False / None), and the forward
                return after a fire vs no fire, per rule.
  story_dedup   event verdicts' provenance.trigger.story_dedup, per trigger
                source: would_skip_duplicate, would_skip_stale, either, and
                analysed (neither), plus a missing record and read errors.
  book_heat     risk_json.book_heat on OPEN/ADD approvals: the read states
                and the spread (min / median / max) of the open risk before
                and after the order, the largest single risk and the order's
                own risk, as % of portfolio value, and n_positions.
  buy_blocks    risk_json.recheck_contradiction (OPEN/ADD approvals and the
                routing rows that ended before sizing) and
                risk_json.add_loser_gate (ADD approvals, and activity rows
                add_loser_gate_shadow on deferred retries): would_block
                counts, and the forward return of the approved buys by
                would_block.
  stop_guard    activity rows stop_above_market_guard: outcome by path, and
                the mode.
  setup_gates   scanner_candidates.setup_gates_detail: scanned rows and
                qualifiers per gate, and the forward return of qualifying
                vs non-qualifying rows.
  dip_watch     DIP_WATCH_SHADOW rows split by provenance.would_review, with
                forward returns.

The forward returns are forward_readout's own (collect, readout): one row
per name per session 0, 5 / 10 / 20 sessions on the name's cached bars, raw
and minus SPY over the same dates. Each such table says "definition:
forward_readout default; Luke to confirm". No new metric is defined here,
and which number decides anything is not this script's.

activity_feed and scanner_candidates have no env column: their rows are
kept for this env only where the row names one. Counts and percentages
only: no prices, share counts or account values.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        scripts/shadow_outcomes.py [--since 2026-11-24] [--out FILE]

Read-only: one transaction opened READ ONLY with a statement timeout, then
rolled back.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NY = ZoneInfo("America/New_York")
STATEMENT_TIMEOUT_MS = 15000
BUY_ACTIONS = ("OPEN", "ADD")
GATES = ("Breakout", "Catalyst", "Continuation")
DEFINITION_NOTE = ("definition: forward_readout default (5 / 10 / 20 sessions after session 0, "
                   "x = minus SPY over the same dates); Luke to confirm.")
SECTION_KEYS = ("held_exit", "story_dedup", "book_heat", "buy_blocks", "stop_guard",
                "setup_gates", "dip_watch")

_FR = None


def _fr():
    """scripts/forward_readout.py, loaded once (scripts/ is not a package)."""
    global _FR
    if _FR is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_readout.py")
        spec = importlib.util.spec_from_file_location("helios_forward_readout", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FR = mod
    return _FR


# ---------------------------------------------------------------- helpers

def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _tri(v) -> str:
    return "true" if v is True else "false" if v is False else "unknown"


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.0f}%" if d else "-"


def _is_error(v) -> bool:
    return isinstance(v, str) and v.strip().lower().startswith("error")


def _forward(rows, bars, since: date, env: str, head: str) -> str:
    """rows: (symbol, label, ts). forward_readout's table, its header dropped,
    and the definition line."""
    fr = _fr()
    groups = fr.collect([(s, label, ts, None) for s, label, ts in rows], bars,
                        label_fn=lambda kind, _d: kind)
    table = fr.readout(groups, since, env, head).splitlines()
    start = next((i for i, line in enumerate(table) if line.startswith("| group")), 0)
    return "\n".join(table[start:]).rstrip() + "\n\n" + DEFINITION_NOTE + "\n"


# ---------------------------------------------------------------- sections

def section_held_exit(trim_rows, bars, since, env, head) -> str:
    """trim_rows: (symbol, ts, held_exit_candidates)."""
    from agents.held_exit import CANDIDATES
    counts = {c: Counter() for c in CANDIDATES}
    missing, fwd = 0, []
    for symbol, ts, rec in trim_rows:
        cands = _as_dict(_as_dict(rec).get("candidates"))
        if not cands:
            missing += 1
            continue
        for c in CANDIDATES:
            v = _as_dict(cands.get(c)).get("would_exit")
            counts[c][_tri(v)] += 1
            if v is True:
                fwd.append((symbol, f"{c}: fire", ts))
            elif v is False:
                fwd.append((symbol, f"{c}: no fire", ts))
    lines = [f"TRIM_EXIT_RULE rows: {len(trim_rows)}; without the record: {missing}.", "",
             "| candidate rule | fire | no fire | unknown | fire share of known |",
             "|---|---|---|---|---|"]
    for c in CANDIDATES:
        k = counts[c]
        lines.append(f"| {c} | {k['true']} | {k['false']} | {k['unknown']} | "
                     f"{_pct(k['true'], k['true'] + k['false'])} |")
    return "\n".join(lines) + "\n\n" + _forward(fwd, bars, since, env, head)


def section_story_dedup(event_rows) -> str:
    """event_rows: (symbol, ts, trigger)."""
    by = {}
    for _s, _ts, trig in event_rows:
        trig = _as_dict(trig)
        src = trig.get("source") or "unknown"
        c = by.setdefault(src, Counter())
        c["rows"] += 1
        rec = trig.get("story_dedup")
        if not isinstance(rec, dict):
            c["missing"] += 1
            continue
        if _is_error(rec.get("read")):
            c["read_error"] += 1
            continue
        dup, stale = rec.get("would_skip_duplicate") is True, rec.get("would_skip_stale") is True
        c["duplicate"] += dup
        c["stale"] += stale
        c["either"] += dup or stale
        c["analysed"] += not (dup or stale)
    lines = ["| trigger source | event verdicts | record missing | read error | would skip duplicate "
             "| would skip stale | either | analysed (neither) |",
             "|---|---|---|---|---|---|---|---|"]
    tot = Counter()
    for src in sorted(by):
        c = by[src]
        tot.update(c)
        lines.append(f"| {src} | {c['rows']} | {c['missing']} | {c['read_error']} | {c['duplicate']} | "
                     f"{c['stale']} | {c['either']} ({_pct(c['either'], c['rows'])}) | {c['analysed']} |")
    if not by:
        lines.append("| (no rows) | | | | | | | |")
    else:
        lines.append(f"| all | {tot['rows']} | {tot['missing']} | {tot['read_error']} | {tot['duplicate']} | "
                     f"{tot['stale']} | {tot['either']} ({_pct(tot['either'], tot['rows'])}) | "
                     f"{tot['analysed']} |")
    return "\n".join(lines) + "\n"


def _spread(vals, scale=100.0, fmt="{:.2f}") -> str:
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return "- | - | -"
    return " | ".join(fmt.format(x * scale) for x in (min(vals), statistics.median(vals), max(vals)))


def _buy_approvals(approvals):
    return [a for a in approvals if a.get("action") in BUY_ACTIONS and not a.get("is_test")]


def section_book_heat(approvals) -> str:
    """approvals: dicts {id, ticker, action, ts, is_test, risk_json}."""
    recs, reads = [], Counter()
    for a in _buy_approvals(approvals):
        rec = _as_dict(a.get("risk_json")).get("book_heat")
        if not isinstance(rec, dict):
            reads["missing"] += 1
            continue
        reads[str(rec.get("read"))] += 1
        recs.append(rec)
    lines = ["Read states: " + (", ".join(f"{k} {v}" for k, v in reads.most_common()) or "none") + ".", "",
             "| at each buy | buys | min | median | max |", "|---|---|---|---|---|"]
    for label, key, scale, fmt in (
            ("open risk before the order, % of portfolio", "sum_open_risk_pct", 100.0, "{:.2f}"),
            ("open risk after the order, % of portfolio", "sum_open_risk_after_pct", 100.0, "{:.2f}"),
            ("largest single position risk, % of portfolio", "max_single_risk_pct", 100.0, "{:.2f}"),
            ("the order's own risk, % of portfolio", "order_risk_pct", 100.0, "{:.2f}"),
            ("positions held", "n_positions", 1.0, "{:.0f}"),
            ("positions with no stop", "n_no_stop", 1.0, "{:.0f}")):
        vals = [r.get(key) for r in recs]
        n = sum(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        lines.append(f"| {label} | {n} | {_spread(vals, scale, fmt)} |")
    return "\n".join(lines) + "\n"


def section_buy_blocks(approvals, routing_rows, loser_activity, bars, since, env, head) -> str:
    """routing_rows: (symbol, ts, recheck_contradiction). loser_activity:
    (ticker, ts, extra) of add_loser_gate_shadow rows."""
    buys = _buy_approvals(approvals)
    rc, rc_route, al, al_retry = Counter(), Counter(), Counter(), Counter()
    fwd = []
    for a in buys:
        risk = _as_dict(a.get("risk_json"))
        rec = risk.get("recheck_contradiction")
        if isinstance(rec, dict):
            wb = _tri(rec.get("would_block"))
            rc[wb] += 1
            fwd.append((a["ticker"], f"recheck would_block {wb}", a["ts"]))
        else:
            rc["no record"] += 1
        if a.get("action") == "ADD":
            rec = risk.get("add_loser_gate")
            if isinstance(rec, dict):
                wb = _tri(rec.get("would_block"))
                al[wb] += 1
                fwd.append((a["ticker"], f"add_loser would_block {wb}", a["ts"]))
            else:
                al["no record"] += 1
    for _s, _ts, rec in routing_rows:
        rc_route[_tri(_as_dict(rec).get("would_block"))] += 1
    for _t, _ts, extra in loser_activity:
        al_retry[_tri(_as_dict(_as_dict(extra).get("add_loser_gate")).get("would_block"))] += 1

    def row(name, c):
        return (f"| {name} | {sum(c.values())} | {c['true']} | {c['false']} | {c['unknown']} | "
                f"{c['no record']} |")
    lines = ["| record | rows | would_block true | false | unknown | no record |",
             "|---|---|---|---|---|---|",
             row("recheck_contradiction, OPEN/ADD approvals", rc),
             row("recheck_contradiction, routing rows that ended before sizing", rc_route),
             row("add_loser_gate, ADD approvals (shadow)", al),
             row("add_loser_gate_shadow, deferred retries", al_retry), "",
             "Forward return of the approved buys by would_block (a would-blocked buy went "
             "through: the shadow never blocks):", ""]
    return "\n".join(lines) + "\n" + _forward(fwd, bars, since, env, head)


def section_stop_guard(guard_rows) -> str:
    """guard_rows: (ticker, ts, extra) of stop_above_market_guard rows."""
    by, modes = {}, Counter()
    for _t, _ts, extra in guard_rows:
        extra = _as_dict(extra)
        modes[str(extra.get("mode"))] += 1
        by.setdefault(str(extra.get("path")), Counter())[str(extra.get("outcome"))] += 1
    outcomes = sorted({o for c in by.values() for o in c})
    lines = [f"Rows: {len(guard_rows)}; mode: "
             + (", ".join(f"{k} {v}" for k, v in modes.most_common()) or "none") + ".", ""]
    if not by:
        return "\n".join(lines + ["No rows."]) + "\n"
    lines += ["| path | rows | " + " | ".join(outcomes) + " |",
              "|---|---|" + "---|" * len(outcomes)]
    for path in sorted(by):
        c = by[path]
        lines.append(f"| {path} | {sum(c.values())} | " + " | ".join(str(c[o]) for o in outcomes) + " |")
    return "\n".join(lines) + "\n"


def section_setup_gates(scan_rows, bars, since, env, head) -> str:
    """scan_rows: (ticker, ts, setup_gates_detail)."""
    per_gate, n, fwd = Counter(), 0, []
    for ticker, ts, detail in scan_rows:
        detail = _as_dict(detail)
        if not detail:
            continue
        n += 1
        q = [g for g in GATES if _as_dict(detail.get(g)).get("qualifies") is True]
        per_gate.update(q)
        per_gate["any"] += bool(q)
        fwd.append((ticker, "qualifies (any gate)" if q else "no gate qualifies", ts))
    lines = [f"Scanned rows with the record: {n}.", "",
             "| gate | qualifying rows | share of scanned |", "|---|---|---|"]
    for g in (*GATES, "any"):
        lines.append(f"| {g} | {per_gate[g]} | {_pct(per_gate[g], n)} |")
    return "\n".join(lines) + "\n\n" + _forward(fwd, bars, since, env, head)


def section_dip_watch(dip_rows, bars, since, env, head) -> str:
    """dip_rows: (symbol, ts, would_review)."""
    c = Counter(_tri(w) for _s, _ts, w in dip_rows)
    fwd = [(s, f"DIP_WATCH_SHADOW would_review {_tri(w)}", ts) for s, ts, w in dip_rows]
    return (f"Rows: {len(dip_rows)}; would_review true {c['true']}, false {c['false']}, "
            f"unknown {c['unknown']}.\n\n" + _forward(fwd, bars, since, env, head))


# ---------------------------------------------------------------- build

def build(data: dict, since: date, env: str, head: str, now: datetime | None = None) -> str:
    """The markdown from _fetch's data. Each section has its own try."""
    now = now or datetime.now(timezone.utc)
    bars = data.get("bars") or {}
    makers = {
        "held_exit": ("Held-exit candidates per rule",
                      lambda: section_held_exit(data["trim"], bars, since, env, head)),
        "story_dedup": ("Story de-duplication per trigger source",
                        lambda: section_story_dedup(data["events"])),
        "book_heat": ("Book heat at each buy", lambda: section_book_heat(data["approvals"])),
        "buy_blocks": ("Recheck contradiction and add-to-loser: would_block",
                       lambda: section_buy_blocks(data["approvals"], data["routing"], data["loser"],
                                                  bars, since, env, head)),
        "stop_guard": ("Above-market stop guard by path", lambda: section_stop_guard(data["guard"])),
        "setup_gates": ("Setup-gate qualifiers",
                        lambda: section_setup_gates(data["scans"], bars, since, env, head)),
        "dip_watch": ("Drop watch by would_review",
                      lambda: section_dip_watch(data["dips"], bars, since, env, head)),
    }
    parts = [f"# Shadow outcomes ({env}, rows since {since})", "",
             f"Generated {now.astimezone(NY):%Y-%m-%d %H:%M} ET at code {head}. Engineering "
             "diagnostic, not a study result. Counts and percentages only.", ""]
    for key in SECTION_KEYS:
        title, fn = makers[key]
        try:
            body = fn()
        except Exception as e:
            body = f"section failed: {type(e).__name__}\n"
        parts += [f"## {title}", "", body.rstrip(), ""]
    return "\n".join(parts).rstrip() + "\n"


def _row_env_ok(extra, env: str) -> bool:
    return _as_dict(extra).get("env") in (None, env)


def _fetch(since: date, env: str) -> dict:
    import db
    non_analyst = list(getattr(db, "_NON_ANALYST_SIGNALS", ()))
    start = datetime.combine(since, time(0), tzinfo=NY)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT symbol, timestamp, decision->'held_exit_candidates' FROM signals "
                        "WHERE signal = 'TRIM_EXIT_RULE' AND timestamp >= %s AND trading_env = %s",
                        (start, env))
            trim = cur.fetchall()
            cur.execute("SELECT symbol, timestamp, provenance->'trigger' FROM signals "
                        "WHERE timestamp >= %s AND trading_env = %s "
                        "AND provenance->'trigger'->>'path' = 'event' AND NOT (signal = ANY(%s))",
                        (start, env, non_analyst))
            events = cur.fetchall()
            cur.execute("SELECT id, ticker, action, timestamp, is_test, risk_json FROM pending_approvals "
                        "WHERE trading_env = %s AND timestamp >= %s", (env, start))
            approvals = [{"id": i, "ticker": t, "action": a, "ts": ts, "is_test": x, "risk_json": rj}
                         for i, t, a, ts, x, rj in cur.fetchall()]
            cur.execute("SELECT symbol, timestamp, decision->'recheck_contradiction' FROM signals "
                        "WHERE timestamp >= %s AND trading_env = %s "
                        "AND decision ? 'route_exit' AND decision ? 'recheck_contradiction'",
                        (start, env))
            routing = cur.fetchall()
            cur.execute("SELECT stage, ticker, ts, extra_json FROM activity_feed WHERE ts >= %s "
                        "AND stage = ANY(%s)", (start, ["add_loser_gate_shadow", "stop_above_market_guard"]))
            activity = [r for r in cur.fetchall() if _row_env_ok(r[3], env)]
            cur.execute("SELECT ticker, timestamp, setup_gates_detail FROM scanner_candidates "
                        "WHERE timestamp >= %s AND setup_gates_detail IS NOT NULL", (start,))
            scans = cur.fetchall()
            cur.execute("SELECT symbol, timestamp, provenance->'would_review' FROM signals "
                        "WHERE signal = 'DIP_WATCH_SHADOW' AND timestamp >= %s AND trading_env = %s",
                        (start, env))
            dips = cur.fetchall()
            fwd_tickers = ({r[0] for r in trim} | {a["ticker"] for a in approvals}
                           | {r[0] for r in scans} | {r[0] for r in dips} | {_fr().BENCHMARK})
            cur.execute("SELECT ticker, date, close FROM price_history WHERE ticker = ANY(%s) "
                        "AND date >= %s", (sorted(t for t in fwd_tickers if t), since - timedelta(days=7)))
            bars: dict = {}
            for t, d, c in cur.fetchall():
                bars.setdefault(t, []).append((d, c))
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return {"trim": trim, "events": events, "approvals": approvals, "routing": routing,
            "loser": [(t, ts, e) for s, t, ts, e in activity if s == "add_loser_gate_shadow"],
            "guard": [(t, ts, e) for s, t, ts, e in activity if s == "stop_above_market_guard"],
            "scans": scans, "dips": dips, "bars": bars}


def run(since: date, env: str, head: str) -> str:
    return build(_fetch(since, env), since, env, head)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", help="YYYY-MM-DD, New York date (default 8 weeks ago)")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    a = ap.parse_args(argv)
    since = (date.fromisoformat(a.since) if a.since
             else datetime.now(NY).date() - timedelta(weeks=8))
    from config import TRADING_ENV
    text = run(since, TRADING_ENV, _git_head())
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
