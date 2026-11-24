"""Reward-to-risk against the placed stop, and realised R per closed lot
(limitations audit 2026-11-21 15:33, finding 3).

Entries: every OPEN/ADD approval in pending_approvals (both trading envs)
replayed through agents/crew.py:_rr_vs_stop, the function that writes the
record-only risk_json["rr_vs_stop"] field, on the stored analyst_json and
risk_json at the sizing price (risk_json.price_at_sizing). A filled buy (one
with a trade_outcomes row) is also replayed at its fill price against the
stop placed for those shares (execution._anchored_stop). The model's own
upside_downside_ratio, which the entry UDR gate reads, is shown beside it.

Closed lots: realised R for each trade_outcomes row with an exit price,
(exit - entry) / risk per share. r_fill divides by fill_risk_per_share (the
fill minus the stop placed for the lot), r_planned by planned_risk_per_share
(the sizing price minus the sizing stop). Rows written before those columns
existed get both from their approval through execution._risk_per_share, the
rule that fills the columns (risk_source "derived"). On an ADD the stop is
later merged into one stop for the whole position; R here is per lot, against
the lot's own initial stop.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python scripts/rr_vs_stop.py [--out FILE]

Read-only (SET TRANSACTION READ ONLY, then rollback). Prints percentages and
R only: no prices, share counts or account figures. Descriptive: the
thresholds are the field's illustrative labels, and whether to gate on any
of this is a study choice, not this script's.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENTRY_ACTIONS = ("OPEN", "ADD")

APPROVALS_SQL = ("SELECT id, trading_env, ticker, action, status, timestamp, stop_loss_price, "
                 "analyst_json, risk_json FROM pending_approvals "
                 "WHERE action IN ('OPEN', 'ADD') ORDER BY id")
OUTCOMES_SQL = ("SELECT approval_id, trading_env, ticker, action, entry_price, entry_date, exit_price, "
                "exit_date, exit_source, outcome_pct, planned_risk_per_share, fill_risk_per_share, "
                "orphan_reason FROM trade_outcomes ORDER BY entry_date")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def entry_rows(approvals: list[dict], fills: dict) -> list[dict]:
    """One row per OPEN/ADD approval: rr_vs_stop at the sizing price and, for
    a filled buy (fills: approval id -> fill price), at the fill against the
    stop placed for the lot."""
    from agents.crew import _rr_vs_stop
    from execution import _anchored_stop
    rows = []
    for a in approvals:
        if a.get("action") not in ENTRY_ACTIONS:
            continue
        risk = a.get("risk_json") if isinstance(a.get("risk_json"), dict) else {}
        res = a.get("analyst_json") if isinstance(a.get("analyst_json"), dict) else {}
        risk = {**risk, "stop_loss_price": a.get("stop_loss_price") or risk.get("stop_loss_price")}
        sized_at = _num(risk.get("price_at_sizing"))
        at_sizing = _rr_vs_stop(res, risk, sized_at)
        fill = _num(fills.get(a["id"]))
        at_fill = None
        stop = _num(risk.get("stop_loss_price"))
        if fill and stop:
            at_fill = _rr_vs_stop(res, {**risk, "stop_loss_price": _anchored_stop(stop, sized_at, fill)},
                                  fill)
        rows.append({"id": a["id"], "env": a.get("trading_env"), "ticker": a.get("ticker"),
                     "action": a.get("action"), "status": a.get("status"),
                     "day": str(a.get("timestamp"))[:10], "filled": fill is not None,
                     "at_sizing": at_sizing, "at_fill": at_fill,
                     "stored": isinstance(risk.get("rr_vs_stop"), dict)})
    return rows


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def _share(flags):
    flags = [f for f in flags if f is not None]
    return (sum(1 for f in flags if f), len(flags))


def summarize(rows: list[dict], thresholds) -> list[dict]:
    """Per (env, filled) group: counts, medians and would-block shares of the
    at-sizing replay."""
    groups = {}
    for r in rows:
        groups.setdefault((r["env"], r["filled"]), []).append(r["at_sizing"])
    out = []
    for (env, filled), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), not kv[0][1])):
        rr = [x.get("r_to_target") for x in rs]
        udr = [x.get("model_udr") for x in rs]
        both = [(m, r) for m, r in zip(udr, rr) if m is not None and r is not None]
        out.append({
            "env": env, "filled": filled, "n": len(rs), "n_r": sum(1 for v in rr if v is not None),
            "median_r": _median(rr), "median_model_udr": _median(udr),
            "median_stop_pct": _median([x.get("stop_pct") for x in rs]),
            "median_stop_atr": _median([x.get("stop_atr") for x in rs]),
            "median_target_pct": _median([x.get("target_pct") for x in rs]),
            "median_target_atr": _median([x.get("target_atr") for x in rs]),
            "would_block": {f"r_lt_{float(t):g}": _share([(x.get("would_block") or {}).get(f"r_lt_{float(t):g}")
                                                          for x in rs]) for t in thresholds},
            "stop_above_model_floor": _share([x.get("stop_above_model_floor") for x in rs]),
            "udr_passes_r_fails_2": sum(1 for m, r in both if m >= 2 and r < 2),
            "udr_fails_r_passes_2": sum(1 for m, r in both if m < 2 and r >= 2),
            "n_both": len(both),
            "reasons": _counts(x.get("reason") for x in rs if x.get("r_to_target") is None),
        })
    return out


def _counts(items):
    c = {}
    for i in items:
        c[i] = c.get(i, 0) + 1
    return c


def closed_lots(outcomes: list[dict], approvals_by_id: dict) -> list[dict]:
    """Realised R per trade_outcomes row with an exit price."""
    from execution import _risk_per_share
    rows = []
    for o in outcomes:
        entry, exit_ = _num(o.get("entry_price")), _num(o.get("exit_price"))
        if not entry or exit_ is None:
            continue
        planned, fill = _num(o.get("planned_risk_per_share")), _num(o.get("fill_risk_per_share"))
        source = "stored" if (planned or fill) else None
        if not (planned or fill):
            a = approvals_by_id.get(o.get("approval_id"))
            if a:
                try:
                    planned, fill = _risk_per_share(
                        {"stop_loss_price": a.get("stop_loss_price"),
                         "risk_json": a.get("risk_json") if isinstance(a.get("risk_json"), dict) else {}},
                        entry)
                except Exception:
                    planned, fill = None, None
                source = "derived" if (planned or fill) else None
        move = exit_ - entry
        pct = _num(o.get("outcome_pct"))
        rows.append({"approval_id": o.get("approval_id"), "env": o.get("trading_env"),
                     "ticker": o.get("ticker"), "action": o.get("action"),
                     "entry_day": str(o.get("entry_date"))[:10], "exit_day": str(o.get("exit_date"))[:10],
                     "exit_source": o.get("exit_source"), "orphan_reason": o.get("orphan_reason"),
                     "outcome_pct": round(pct * 100, 2) if pct is not None else None,
                     "r_fill": round(move / fill, 2) if fill and fill > 0 else None,
                     "r_planned": round(move / planned, 2) if planned and planned > 0 else None,
                     "risk_source": source})
    return rows


def _f(v, fmt="{:.2f}"):
    return "" if v is None else fmt.format(v)


def render(summary, live_rows, lots, thresholds, head: str) -> str:
    keys = [f"r_lt_{float(t):g}" for t in thresholds]
    L = ["# Reward-to-risk vs the placed stop, and realised R", "",
         f"Code at {head}. R = (target - price) / (price - stop), target = bull_case_target "
         "(else expected_return_pct); stop = the sizing stop (at sizing) or the stop placed for "
         "the lot (at fill). model UDR = the ratio the entry gate reads. Thresholds are "
         f"illustrative labels: {', '.join(f'{float(t):g}' for t in thresholds)}.", "",
         "## Entries: summary at the sizing price", "",
         "| env | filled | n | R computed | median R | median model UDR | median stop % | median stop ATR "
         "| median target % | median target ATR | " + " | ".join(f"R<{k[5:]}" for k in keys)
         + " | stop above model floor | UDR>=2 but R<2 | UDR<2 but R>=2 |",
         "|" + "---|" * (14 + len(keys))]
    for s in summary:
        wb = " | ".join(f"{s['would_block'][k][0]}/{s['would_block'][k][1]}" for k in keys)
        sf = s["stop_above_model_floor"]
        L.append(f"| {s['env']} | {'yes' if s['filled'] else 'no'} | {s['n']} | {s['n_r']} | "
                 f"{_f(s['median_r'])} | {_f(s['median_model_udr'])} | {_f(s['median_stop_pct'])} | "
                 f"{_f(s['median_stop_atr'])} | {_f(s['median_target_pct'])} | {_f(s['median_target_atr'])} | "
                 f"{wb} | {sf[0]}/{sf[1]} | {s['udr_passes_r_fails_2']}/{s['n_both']} | "
                 f"{s['udr_fails_r_passes_2']}/{s['n_both']} |")
    reasons = {}
    for s in summary:
        for k, v in s["reasons"].items():
            reasons[f"{s['env']}:{k}"] = reasons.get(f"{s['env']}:{k}", 0) + v
    if reasons:
        L += ["", "R not computed: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + "."]
    L += ["", "## Live entries, one row per approval", "",
          "| id | day | ticker | action | status | model UDR | R at sizing | R at fill | R to plan t1 "
          "| stop % | stop ATR | target % | target ATR | stop vs model floor % | "
          + " | ".join(f"R<{k[5:]}" for k in keys) + " |",
          "|" + "---|" * (14 + len(keys))]
    for r in live_rows:
        a, b = r["at_sizing"], r["at_fill"] or {}
        wb = " | ".join({True: "yes", False: "no"}.get((a.get("would_block") or {}).get(k), "")
                        for k in keys)
        L.append(f"| {r['id']} | {r['day']} | {r['ticker']} | {r['action']} | {r['status']} | "
                 f"{_f(a.get('model_udr'))} | {_f(a.get('r_to_target'))} | {_f(b.get('r_to_target'))} | "
                 f"{_f(a.get('r_to_plan_t1'))} | {_f(a.get('stop_pct'))} | {_f(a.get('stop_atr'))} | "
                 f"{_f(a.get('target_pct'))} | {_f(a.get('target_atr'))} | {_f(a.get('floor_vs_stop_pct'))} | {wb} |")
    L += ["", "## Closed lots: realised R", "",
          "| env | approval | ticker | action | entry | exit | exit source | outcome % | R (fill risk) "
          "| R (planned risk) | risk source |", "|" + "---|" * 11]
    for x in lots:
        L.append(f"| {x['env']} | {x['approval_id']} | {x['ticker']} | {x['action']} | {x['entry_day']} | "
                 f"{x['exit_day']} | {x['exit_source'] or ''}{' (orphan: ' + x['orphan_reason'] + ')' if x['orphan_reason'] else ''} | "
                 f"{_f(x['outcome_pct'])} | {_f(x['r_fill'])} | {_f(x['r_planned'])} | {x['risk_source'] or ''} |")
    for env in sorted({x["env"] for x in lots}, key=str):
        rf = [x["r_fill"] for x in lots if x["env"] == env and x["r_fill"] is not None]
        L.append("")
        L.append(f"{env}: {len([x for x in lots if x['env'] == env])} closed lots, R (fill risk) on {len(rf)}"
                 + (f", mean {statistics.mean(rf):.2f}, median {statistics.median(rf):.2f}" if rf else "") + ".")
    return "\n".join(L) + "\n"


def _fetch(sql):
    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.rollback()   # end the read-only transaction before the conn goes back to the pool
    finally:
        db._put_conn(conn)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    from config import RR_VS_STOP_LABEL_THRESHOLDS as thresholds
    approvals, outcomes = _fetch(APPROVALS_SQL), _fetch(OUTCOMES_SQL)
    fills = {o["approval_id"]: o["entry_price"] for o in outcomes if o.get("approval_id") is not None}
    rows = entry_rows(approvals, fills)
    lots = closed_lots(outcomes, {a["id"]: a for a in approvals})
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    text = render(summarize(rows, thresholds), [r for r in rows if r["env"] == "live"], lots,
                  thresholds, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
