"""Within-day verdict spread: how far the analyst's own conviction and UDR
move for one ticker on one day (audit 2026-11-05-0525 #2).

One 72B sample per event decides each verdict, so a ticker reviewed several
times a day on much the same facts can land on both sides of the UDR entry
floor. This lists, per ticker and ET day with at least --min-verdicts
verdicts, the count, the conviction and UDR ranges (the model's values,
decision.model_conviction and decision.udr_check.model) and the final
actions, and flags a day whose UDR range straddles the live entry floor.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/verdict_spread.py [--days 21] [--min-verdicts 3] [--out FILE]

After the table it reports three aggregates over the same verdict rows:

- the analyst calls that would be left if each ticker's queued events were
  coalesced into one call per 15, 30 or 60 minute window, next to the actual
  count. A window opens at a ticker's first queued event
  (provenance.trigger.llm_enqueued_at, the time it passed the screener) and
  every event of that ticker enqueued before it closes shares one call.
  Verdicts from scans, reviews and events with no enqueue time are counted
  unchanged;
- the screener's new_fact on the triggering item of the event-triggered
  verdicts (true, false, not judged) and the share that was false;
- the buys (final action OPEN or ADD) by the ticker's previous verdict
  (decision.prior_verdicts_72h): a buy, not a buy, none in 72 h, or not
  recorded.

The rows are verdicts, so a call that returned no verdict is not counted.

Read-only and descriptive: no prices, no account data, no verdict texts.
Whether a spread matters, which window or rule to use, and what to do about
it (G27/G30), is a study choice, not this script's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")
ENTRY_ACTIONS = ("OPEN", "ADD")
COALESCE_WINDOWS_MIN = (15, 30, 60)


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def spread(rows, floor: float, min_verdicts: int) -> list[dict]:
    """rows: (symbol, timestamp, decision[, trigger]). One dict per ticker-day
    with at least min_verdicts verdicts carrying udr_check, ordered by day,
    ticker."""
    days: dict[tuple, list] = defaultdict(list)
    for symbol, ts, decision, *_ in rows:
        d = decision or {}
        if not isinstance(d.get("udr_check"), dict):
            continue
        days[(ts.astimezone(ET).date(), symbol)].append(d)
    out = []
    for (day, symbol), ds in sorted(days.items()):
        if len(ds) < min_verdicts:
            continue
        conv = [c for c in (_num(d.get("model_conviction")) for d in ds) if c is not None]
        udr = [u for u in (_num(d["udr_check"].get("model")) for d in ds) if u is not None]
        out.append({
            "day": str(day), "ticker": symbol, "n": len(ds),
            "conviction": (min(conv), max(conv)) if conv else None,
            "udr": (min(udr), max(udr)) if udr else None,
            "crosses_floor": bool(udr) and min(udr) < floor <= max(udr),
            "actions": sorted({str(d.get("final_action")) for d in ds}),
        })
        acts = out[-1]["actions"]
        out[-1]["actions_differ"] = len(acts) > 1
        out[-1]["entry_action"] = any(a in ENTRY_ACTIONS for a in acts)
    return out


def _as_time(v) -> datetime | None:
    """An ISO timestamp string as an aware datetime (a naive one is read as
    UTC); None when it is missing or unreadable."""
    if not isinstance(v, str) or not v:
        return None
    try:
        t = datetime.fromisoformat(v)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _windows_used(times, minutes: int) -> int:
    """How many windows of `minutes` it takes to hold the times, when a
    window opens at the first time that is not inside the one before it."""
    n, end = 0, None
    for t in sorted(times):
        if end is None or t >= end:
            n += 1
            end = t + timedelta(minutes=minutes)
    return n


def calls_left(rows, windows=COALESCE_WINDOWS_MIN) -> dict:
    """rows: (symbol, timestamp, decision, trigger), one per verdict. Replays
    per-ticker coalescing on the verdicts triggered by a queued event
    (trigger.path "event" with a readable trigger.llm_enqueued_at): a window
    opens at a ticker's first such event and every event of that ticker
    enqueued before it closes shares one call. Returns the verdict count, how
    many of them are replayed, and for each window length in minutes the
    calls left among the replayed ones. The other verdicts are not changed
    by the replay."""
    queued: dict[str, list] = defaultdict(list)
    total = 0
    for symbol, _ts, _decision, trigger in rows:
        total += 1
        t = trigger if isinstance(trigger, dict) else {}
        at = _as_time(t.get("llm_enqueued_at")) if t.get("path") == "event" else None
        if at is not None:
            queued[symbol].append(at)
    return {"verdicts": total, "replayed": sum(len(v) for v in queued.values()),
            "left": {w: sum(_windows_used(v, w) for v in queued.values()) for w in windows}}


def new_fact_counts(rows) -> dict:
    """The screener's new_fact on the triggering item of each event-triggered
    verdict (trigger.path "event"): how many were true, false or not judged
    (anything but a boolean: the keyword fallback, a reply with no readable
    value, or an analysis queued with no screen and no news item, such as a
    drop-watch review or a stale-entry re-check), and false as a share of
    them all (None with no such verdict)."""
    n = {True: 0, False: 0, None: 0}
    for _symbol, _ts, _decision, trigger in rows:
        t = trigger if isinstance(trigger, dict) else {}
        if t.get("path") != "event":
            continue
        v = t.get("new_fact")
        n[v if isinstance(v, bool) else None] += 1
    event = sum(n.values())
    return {"event": event, "true": n[True], "false": n[False], "not_judged": n[None],
            "false_share": n[False] / event if event else None}


def buys_after_non_buy(rows) -> dict:
    """Buy verdicts (final_action OPEN or ADD) by the ticker's previous
    verdict, the first entry of decision.prior_verdicts_72h (the ticker's
    verdicts in the 72 h before the buy, newest first): it was a buy, it was
    not a buy, the list is empty (none in 72 h, or the read failed), or the
    list was not recorded. prior_not_buy_share is over all buys (None with no
    buy)."""
    out = {"buys": 0, "prior_buy": 0, "prior_not_buy": 0, "no_prior": 0, "not_recorded": 0}
    for _symbol, _ts, decision, _trigger in rows:
        d = decision if isinstance(decision, dict) else {}
        if d.get("final_action") not in ENTRY_ACTIONS:
            continue
        out["buys"] += 1
        prior = d.get("prior_verdicts_72h")
        if not isinstance(prior, list):
            out["not_recorded"] += 1
        elif not prior:
            out["no_prior"] += 1
        elif isinstance(prior[0], dict) and prior[0].get("signal") in ENTRY_ACTIONS:
            out["prior_buy"] += 1
        else:
            out["prior_not_buy"] += 1
    out["prior_not_buy_share"] = out["prior_not_buy"] / out["buys"] if out["buys"] else None
    return out


def _share(n: int, of: int) -> str:
    return f"{n / of:.0%}" if of else "n/a"


def render_replay(calls: dict, facts: dict, buys: dict) -> str:
    """The three aggregates as markdown sections: counts and shares only."""
    replayed, rest = calls["replayed"], calls["verdicts"] - calls["replayed"]
    lines = ["", "## Analyst calls left under per-ticker coalescing", "",
             f"Verdicts: {calls['verdicts']}; triggered by a queued event and replayed: {replayed}; "
             f"counted unchanged (scans, reviews, events with no enqueue time): {rest}.", "",
             "A window opens at a ticker's first queued event; every event of that ticker enqueued",
             "before it closes shares one call. A call that returned no verdict is not counted.", "",
             "| window | queued-event calls | share of actual | all calls |",
             "|---|---|---|---|",
             f"| actual | {replayed} | {_share(replayed, replayed)} | {calls['verdicts']} |"]
    lines += [f"| {w} min | {n} | {_share(n, replayed)} | {n + rest} |" for w, n in calls["left"].items()]
    lines += ["", "## Triggers with no new fact", "",
              f"The screener's new_fact on the triggering item of the {facts['event']} "
              f"event-triggered verdicts. Shares are of all of them.", "",
              "Not judged: the screener gave no true or false (the keyword fallback, or a reply with",
              "no readable value), or the analysis was queued with no screen and no news item (a",
              "drop-watch review, a re-check of a stale deferred buy). The two are not told apart here.", "",
              "| new_fact | verdicts | share |", "|---|---|---|"]
    lines += [f"| {label} | {facts[key]} | {_share(facts[key], facts['event'])} |"
              for label, key in (("true", "true"), ("false", "false"), ("not judged", "not_judged"))]
    lines += ["", "## Buys by the ticker's previous verdict", "",
              f"Verdicts with final action OPEN or ADD: {buys['buys']}. The previous verdict is the "
              f"newest one in the 72 h before the buy.", "",
              "| previous verdict | buys | share |", "|---|---|---|"]
    lines += [f"| {label} | {buys[key]} | {_share(buys[key], buys['buys'])} |"
              for label, key in (("a buy", "prior_buy"), ("not a buy", "prior_not_buy"),
                                 ("none in 72 h, or the read failed", "no_prior"),
                                 ("not recorded", "not_recorded"))]
    return "\n".join(lines) + "\n"


def render(days: list[dict], floor: float, window: int, min_verdicts: int, env: str, head: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    crossing = sum(1 for d in days if d["crosses_floor"])
    crossing_mixed = sum(1 for d in days if d["crosses_floor"] and d["actions_differ"])

    def rng(r, fmt):
        return f"{fmt.format(r[0])}-{fmt.format(r[1])}" if r else "n/a"
    lines = [f"# Within-day verdict spread ({env}, last {window} days)", "",
             f"Generated {now} at code {head}. Descriptive only.", "",
             f"Ticker-days with at least {min_verdicts} verdicts: {len(days)}; "
             f"UDR range straddles the {floor:g} entry floor on {crossing}; "
             f"of those, the final action differs within the day on {crossing_mixed}.", "",
             "Days with 3 or more verdicts are mostly held names and news-heavy days. The script does",
             "not check whether the facts were the same across a day's verdicts.", "",
             "| day | ticker | verdicts | conviction | UDR | straddles floor | actions differ | OPEN/ADD occurred | final actions |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {d['day']} | {d['ticker']} | {d['n']} | {rng(d['conviction'], '{:g}')} | "
              f"{rng(d['udr'], '{:.2f}')} | {'yes' if d['crosses_floor'] else ''} | "
              f"{'yes' if d['actions_differ'] else ''} | {'yes' if d['entry_action'] else ''} | "
              f"{', '.join(d['actions'])} |" for d in days]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--min-verdicts", type=int, default=3)
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()

    import db
    from agents.analyst import _ENTRY_UDR_FLOOR
    from config import TRADING_ENV
    from datetime import time
    # Whole ET days: from midnight ET, --days days back, so the first day is complete.
    start = datetime.combine(datetime.now(ET).date() - timedelta(days=args.days), time(0), tzinfo=ET)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute("SELECT symbol, timestamp, decision, provenance->'trigger' FROM signals "
                        "WHERE trading_env=%s AND timestamp >= %s "
                        "AND decision ? 'udr_check'", (TRADING_ENV, start))
            rows = cur.fetchall()
        conn.rollback()   # end the read-only transaction before the conn goes back to the pool
    finally:
        db._put_conn(conn)
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = render(spread(rows, _ENTRY_UDR_FLOOR, args.min_verdicts), _ENTRY_UDR_FLOOR,
                  args.days, args.min_verdicts, TRADING_ENV, head or "unknown")
    text += render_replay(calls_left(rows), new_fact_counts(rows), buys_after_non_buy(rows))
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
