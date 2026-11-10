"""Summary of a prompt_order_ab.py JSONL: the arms (A = pre-31d52f8 answer
order, B = post-31d52f8, C = post order without the fill-order instruction
when run with --arms ABC), paired by signal. It applies no threshold and
states no verdict.

Run consistency: every run line records seed, timeout, live timeout, arms,
num_predict per arm, the live options (num_ctx, temperature,
repeat_penalty), model tag and digest, Ollama version, schema hashes and
tool_sha (sha256 of prompt_order_ab.py and the two schema-block files).
When run lines that have pairs differ in any of these, the report stops;
--split reports each configuration separately instead. The repo head is on
the run line for information only and is not compared, so a resume after
another deploy pools with the earlier nights.

--primary (required, no default) names the metric printed first, with its
test and intervals; the other three follow under "secondary":
  c7_share         conviction 7+ share; exact McNemar
  mean_conviction  paired mean conviction difference; sign-flip permutation
  action_change    share of pairs whose action differs; Bowker symmetry
                   statistic with a permutation test swapping arms in pairs
  logprob_ev       paired difference in the logprob expected conviction;
                   sign-flip permutation. Said and excluded when the stored
                   replies carry no conviction_logprobs.
Intervals are seeded percentile bootstraps over pairs (90/95/99%);
permutation tests are exact up to 16 pairs, else seeded Monte Carlo.

Printed first: the sha256 of this report script and the runner tool_sha
values found on the run lines.

Printed per configuration:
  - denominators: signals in the sample, skip lines by reason, signals
    attempted, last attempt complete or with a failed arm, not attempted;
    attempts per signal (--retry-errors makes more than one);
  - per arm, on each signal's last attempt: replies with a conviction, and
    failures by type (timeout, other call error, parse failed, no
    conviction), plus salvaged and num_predict-cut (done_reason "length")
    replies; prompt_eval_count per arm;
  - per comparison (B vs A; with C also C vs A and B vs C), for pairs whose
    two arms both gave a conviction, and again leaving out pairs where
    either arm was salvaged or num_predict-cut:
      the primary metric, the secondary metrics, and the conviction
      distributions and c7+ by run position;
  - the two bounds over every attempted signal, a failed or cut-off reply
    (no conviction, or done_reason "length" even with a conviction) either
    dropped with its pair ("dropped") or counted as not c7+ ("not-c7");
  - with arm A2 (--repeat-arm), the A-vs-A2 c7+ and action discordance as
    the run-to-run noise floor, next to the same figures for the
    comparison on the same signals.

    venv/bin/python scripts/prompt_order_ab_report.py FILE --primary METRIC [--split] [--boot 10000] [--boot-seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict

LEVELS = (0.90, 0.95, 0.99)
NONE = "(none)"
CONFIG_KEYS = ("seed", "timeout_s", "arms", "arm_a_num_predict", "model_digest", "ollama_version",
               "schema_sha_A", "schema_sha_B", "schema_sha_C", "tool_sha", "model", "live_timeout_s",
               "repeat_arm", "repeat_subsample")
REPEAT_ARM = "A2"
OPTION_KEYS = ("num_ctx", "temperature", "repeat_penalty")
PRIMARY = ("c7_share", "mean_conviction", "action_change", "logprob_ev")


# -- loading -------------------------------------------------------------------

def load(path: str) -> dict:
    """{runs: [run lines], attempts: {run index: {signal id: [pair lines]}},
    skips: {signal id: reason}}. A pair belongs to the run line with its
    run_id, else to the last run line before it."""
    runs, skips = [], {}
    attempts = defaultdict(lambda: defaultdict(list))
    by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            kind = rec.get("kind")
            if kind == "run":
                runs.append(rec)
                if rec.get("run_id"):
                    by_id[rec["run_id"]] = len(runs) - 1
            elif kind == "skip":
                skips[rec["signal_id"]] = rec.get("reason")
            elif kind == "pair":
                ri = by_id.get(rec.get("run_id"), len(runs) - 1)
                attempts[ri][rec["signal_id"]].append(rec)
    return {"runs": runs, "attempts": attempts, "skips": skips}


def run_config(run: dict) -> dict:
    """What has to match across run lines: CONFIG_KEYS, the live options
    (num_ctx, temperature, repeat_penalty) and num_predict per arm."""
    cfg = {k: run.get(k) for k in CONFIG_KEYS}
    opts = run.get("options") or {}
    cfg.update({k: opts.get(k) for k in OPTION_KEYS})
    arms = run.get("arms") or "AB"
    cfg["num_predict_by_arm"] = {a: (run.get("arm_a_num_predict") if a == "A" and run.get("arm_a_num_predict")
                                     else opts.get("num_predict")) for a in arms}
    return cfg


def config_of(run: dict) -> tuple:
    return tuple((k, json.dumps(v, sort_keys=True)) for k, v in run_config(run).items())


def groups(data: dict) -> list[dict]:
    """One group per distinct run configuration among runs that have pairs
    (runs without pairs join a group only through their sample)."""
    out: dict = {}
    for ri, run in enumerate(data["runs"]):
        g = out.setdefault(config_of(run), {"config": run_config(run),
                                             "runs": [], "attempts": defaultdict(list), "sample": []})
        g["runs"].append(ri)
        for sid in run.get("signal_ids") or []:
            if sid not in g["sample"]:
                g["sample"].append(sid)
        for sid, recs in data["attempts"].get(ri, {}).items():
            g["attempts"][sid].extend(recs)
    orphan = data["attempts"].get(-1)
    if orphan:   # pair lines before any run line
        g = out.setdefault((("no run line", "1"),), {"config": {"no_run_line": True}, "runs": [],
                                                      "attempts": defaultdict(list), "sample": []})
        for sid, recs in orphan.items():
            g["attempts"][sid].extend(recs)
    return [g for g in out.values() if g["attempts"]] or list(out.values())[:1]


# -- per-arm tables ----------------------------------------------------------------

def arm_failure(a: dict | None) -> str:
    if a is None:
        return "not_called"
    err = a.get("error")
    if err:
        return "timeout" if str(err).startswith("timeout") else "call_error"
    if not a.get("parse_ok", a.get("conviction") is not None):
        return "parse_failed"
    if a.get("conviction") is None:
        return "no_conviction"
    return "ok"


def arm_tables(last: list[dict], arms: tuple) -> dict:
    out = {}
    for arm in arms:
        c = Counter()
        pe = []
        for r in last:
            a = (r.get("arms") or {}).get(arm)
            c[arm_failure(a)] += 1
            if a and a.get("salvaged"):
                c["salvaged"] += 1
            if a and a.get("done_reason") == "length":
                c["num_predict_cut"] += 1
            if a and a.get("prompt_eval_count") is not None:
                pe.append(a["prompt_eval_count"])
        out[arm] = {"counts": c, "prompt_eval": (len(pe), sum(pe) / len(pe), min(pe), max(pe)) if pe else None}
    return out


# -- comparisons -------------------------------------------------------------------

def _conv(r, arm):
    a = (r.get("arms") or {}).get(arm) or {}
    return a.get("conviction")


def usable(rows: list[dict], x: str, y: str, clean_only: bool) -> list[dict]:
    out = []
    for r in rows:
        ax, ay = (r.get("arms") or {}).get(x), (r.get("arms") or {}).get(y)
        if not ax or not ay or ax.get("error") or ay.get("error"):
            continue
        if ax.get("conviction") is None or ay.get("conviction") is None:
            continue
        if clean_only and any(a.get("salvaged") or a.get("done_reason") == "length" for a in (ax, ay)):
            continue
        out.append(r)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs under Binomial(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_diff(yx: list[int], yy: list[int], reps: int, seed: int) -> list[float]:
    """Sorted bootstrap means of (yy - yx) over pairs resampled with replacement."""
    d = [b - a for a, b in zip(yx, yy)]
    n = len(d)
    if n == 0:
        return []
    rng = random.Random(seed)
    return sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))


def percentile_interval(sorted_vals: list[float], level: float) -> tuple[float, float] | None:
    if not sorted_vals:
        return None
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    n = len(sorted_vals)

    def q(p):
        xx = p * (n - 1)
        i = int(math.floor(xx))
        j = min(i + 1, n - 1)
        return sorted_vals[i] + (sorted_vals[j] - sorted_vals[i]) * (xx - i)
    return q(lo_q), q(hi_q)


def summarize(pairs: list[dict], reps: int, seed: int, x: str = "A", y: str = "B") -> dict:
    yx = [int(_conv(r, x) >= 7) for r in pairs]
    yy = [int(_conv(r, y) >= 7) for r in pairs]
    n = len(pairs)
    b = sum(1 for p, q in zip(yx, yy) if p == 1 and q == 0)   # x yes, y no
    c = sum(1 for p, q in zip(yx, yy) if p == 0 and q == 1)   # x no, y yes
    boots = bootstrap_diff(yx, yy, reps, seed)
    cx = [_conv(r, x) for r in pairs]
    cy = [_conv(r, y) for r in pairs]
    xtab = Counter((r["arms"][x].get("action") or NONE, r["arms"][y].get("action") or NONE) for r in pairs)

    def expected(arm):
        return [(r["arms"][arm].get("conviction_logprobs") or {}).get("expected") for r in pairs]
    e_pairs = [(p, q) for p, q in zip(expected(x), expected(y)) if p is not None and q is not None]
    by_pos = {}
    for arm in (x, y):
        positions = sorted({r["arms"][arm].get("position") for r in pairs} - {None})
        for pos in positions:
            ys = [int(_conv(r, arm) >= 7) for r in pairs if r["arms"][arm].get("position") == pos]
            by_pos[f"{arm}@{pos}"] = (sum(ys), len(ys))
    return {
        "x": x, "y": y, "n": n,
        "c7_x": sum(yx), "c7_y": sum(yy),
        "share_x": sum(yx) / n if n else None, "share_y": sum(yy) / n if n else None,
        "diff": (sum(yy) - sum(yx)) / n if n else None,
        "boot": {lv: percentile_interval(boots, lv) for lv in LEVELS},
        "mcnemar": {"x_yes_y_no": b, "x_no_y_yes": c, "p_two_sided": mcnemar_exact(b, c)},
        "xtab": xtab,
        "dist_x": Counter(cx), "dist_y": Counter(cy),
        "mean_x": sum(cx) / n if n else None, "mean_y": sum(cy) / n if n else None,
        "expected_pairs": len(e_pairs),
        "expected_mean_x": sum(p for p, _ in e_pairs) / len(e_pairs) if e_pairs else None,
        "expected_mean_y": sum(q for _, q in e_pairs) / len(e_pairs) if e_pairs else None,
        "by_position": by_pos,
    }


def _failed_or_cut(r, arm) -> bool:
    """No conviction (call error, timeout, parse failure, no conviction), or
    cut off by the output limit (done_reason "length") even with a conviction."""
    a = (r.get("arms") or {}).get(arm) or {}
    return a.get("conviction") is None or a.get("done_reason") == "length"


def bounds(last: list[dict], x: str, y: str) -> dict:
    """The two bounds of design item 2 over every attempted signal, with
    failed or cut-off replies (_failed_or_cut):
      dropped  pairs where either arm failed or was cut off are left out;
               c7+ shares and y - x over the remaining pairs;
      not_c7   every attempted signal kept, a failed or cut-off reply
               counted as not c7+; c7+ shares and y - x over all of them."""
    n = len(last)
    if not n:
        return {"n": 0}

    def c7(r, a):
        return not _failed_or_cut(r, a) and _conv(r, a) >= 7
    fx = sum(1 for r in last if _failed_or_cut(r, x))
    fy = sum(1 for r in last if _failed_or_cut(r, y))
    kept = [r for r in last if not _failed_or_cut(r, x) and not _failed_or_cut(r, y)]
    k = len(kept)
    dx, dy = sum(c7(r, x) for r in kept), sum(c7(r, y) for r in kept)
    nx, ny = sum(c7(r, x) for r in last), sum(c7(r, y) for r in last)
    return {"n": n, "failed_or_cut": {x: fx, y: fy},
            "dropped": {"n": k, "x": dx / k if k else None, "y": dy / k if k else None,
                        "diff": (dy - dx) / k if k else None},
            "not_c7": {"n": n, "x": nx / n, "y": ny / n, "diff": (ny - nx) / n}}


# -- rendering -------------------------------------------------------------------

def _f(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


ARM_LABEL = {"A": "A (pre-31d52f8 order)", "B": "B (post-31d52f8 order)",
             "C": "C (post order, no fill-order instruction)"}


def sign_flip_p(stat, n: int, reps: int, seed: int) -> float:
    """Two-sided permutation p for paired data under exchangeable arms:
    stat(flips) with flips in {+1, -1}^n (swap the arms of pair i when -1),
    compared with the observed stat (all +1). Exact for n <= 16, else
    seeded Monte Carlo with (hits + 1) / (reps + 1)."""
    if n == 0:
        return 1.0
    obs = stat([1] * n)
    eps = 1e-12
    if n <= 16:
        hits = 0
        for m in range(2 ** n):
            flips = [-1 if (m >> k) & 1 else 1 for k in range(n)]
            hits += stat(flips) >= obs - eps
        return hits / 2 ** n
    rng = random.Random(seed)
    hits = sum(stat([rng.choice((1, -1)) for _ in range(n)]) >= obs - eps for _ in range(reps))
    return (hits + 1) / (reps + 1)


def mean_diff_test(xs: list[float], ys: list[float], reps: int, seed: int) -> dict:
    d = [b - a for a, b in zip(xs, ys)]
    n = len(d)
    boots = bootstrap_diff(xs, ys, reps, seed)
    return {"n": n, "mean_x": sum(xs) / n if n else None, "mean_y": sum(ys) / n if n else None,
            "diff": sum(d) / n if n else None,
            "boot": {lv: percentile_interval(boots, lv) for lv in LEVELS},
            "p": sign_flip_p(lambda f: abs(sum(fi * di for fi, di in zip(f, d))), n, reps, seed)}


def bowker_stat(pairs_xy: list[tuple], flips=None) -> float:
    """Bowker's symmetry statistic on the cross-tab of (x action, y action),
    with pair i's arms swapped where flips[i] == -1."""
    cnt = Counter()
    for k, (a, b) in enumerate(pairs_xy):
        if flips is not None and flips[k] == -1:
            a, b = b, a
        cnt[(a, b)] += 1
    total = 0.0
    seen = set()
    for (a, b) in list(cnt):
        if a == b or (b, a) in seen:
            continue
        seen.add((a, b))
        nab, nba = cnt.get((a, b), 0), cnt.get((b, a), 0)
        if nab + nba:
            total += (nab - nba) ** 2 / (nab + nba)
    return total


def action_change_test(pairs: list[dict], x: str, y: str, reps: int, seed: int) -> dict:
    acts = [(r["arms"][x].get("action") or NONE, r["arms"][y].get("action") or NONE) for r in pairs]
    n = len(acts)
    changed = [int(a != b) for a, b in acts]
    boots = bootstrap_diff([0] * n, changed, reps, seed)
    # only discordant pairs change under a swap
    disc = [ab for ab in acts if ab[0] != ab[1]]
    return {"n": n, "changed": sum(changed), "share": sum(changed) / n if n else None,
            "boot": {lv: percentile_interval(boots, lv) for lv in LEVELS},
            "bowker": bowker_stat(acts),
            "p": sign_flip_p(lambda f: bowker_stat(disc, f), len(disc), reps, seed),
            "xtab": Counter(acts)}


def _ev(r, arm):
    return (r["arms"][arm].get("conviction_logprobs") or {}).get("expected")


def _ci_lines(boot: dict, reps: int, seed: int, digits=3) -> list[str]:
    return [f"  bootstrap {int(lv * 100)}% percentile interval ({reps} reps, seed {seed}): "
            f"[{_f(ci[0], digits)}, {_f(ci[1], digits)}]" if ci else "  bootstrap: n/a"
            for lv, ci in boot.items()]


def _perm_note(n: int) -> str:
    return "exact" if n <= 16 else "Monte Carlo, bootstrap reps and seed"


def metric_lines(metric: str, pairs: list[dict], x: str, y: str, reps: int, seed: int) -> list[str]:
    """One metric's estimate, intervals and test, or why it can't be computed."""
    if metric == "c7_share":
        s = summarize(pairs, reps, seed, x, y)
        if not s["n"]:
            return ["c7_share: no pairs"]
        m = s["mcnemar"]
        return ([f"c7_share: {ARM_LABEL.get(x, x)} {s['c7_x']}/{s['n']} = {_f(s['share_x'])}; "
                 f"{ARM_LABEL.get(y, y)} {s['c7_y']}/{s['n']} = {_f(s['share_y'])}",
                 f"  paired difference {y} - {x}: {_f(s['diff'])}"]
                + _ci_lines(s["boot"], reps, seed)
                + [f"  McNemar exact (c7+): {x} yes/{y} no = {m['x_yes_y_no']}, {x} no/{y} yes = "
                   f"{m['x_no_y_yes']}, two-sided p = {m['p_two_sided']:.4g}"])
    if metric == "mean_conviction":
        t = mean_diff_test([_conv(r, x) for r in pairs], [_conv(r, y) for r in pairs], reps, seed)
        if not t["n"]:
            return ["mean_conviction: no pairs"]
        return ([f"mean_conviction: {x} {_f(t['mean_x'], 2)}, {y} {_f(t['mean_y'], 2)} (n {t['n']})",
                 f"  paired mean difference {y} - {x}: {_f(t['diff'], 3)}"]
                + _ci_lines(t["boot"], reps, seed)
                + [f"  sign-flip permutation test ({_perm_note(t['n'])}), two-sided p = {t['p']:.4g}"])
    if metric == "action_change":
        t = action_change_test(pairs, x, y, reps, seed)
        if not t["n"]:
            return ["action_change: no pairs"]
        acts = sorted({a for a, _ in t["xtab"]} | {b for _, b in t["xtab"]})
        w = max(8, *(len(a) for a in acts))
        out = ([f"action_change: {t['changed']}/{t['n']} pairs with a different action = {_f(t['share'])}"]
               + _ci_lines(t["boot"], reps, seed)
               + [f"  direction of changes (symmetry), p = {t['p']:.4g} (Bowker statistic "
                  f"{t['bowker']:.3f}, permutation test swapping arms within pairs, "
                  f"{_perm_note(t['changed'])}; not a test of the change share)",
                  f"  action cross-tab (rows {x}, columns {y}):",
                  "  " + " " * (w + 2) + "".join(f"{a:>{w}}" for a in acts)])
        out += [f"    {a:<{w}}" + "".join(f"{t['xtab'].get((a, b), 0):>{w}}" for b in acts) for a in acts]
        return out
    if metric == "logprob_ev":
        both = [r for r in pairs if _ev(r, x) is not None and _ev(r, y) is not None]
        if not both:
            return [f"logprob_ev: cannot be computed from the stored output (no conviction_logprobs "
                    f"expected value in both arms of any of the {len(pairs)} pairs); excluded"]
        t = mean_diff_test([_ev(r, x) for r in both], [_ev(r, y) for r in both], reps, seed)
        ten = sum(1 for r in both if any((r["arms"][a].get("conviction_logprobs") or {}).get("first_digit_only")
                                         for a in (x, y)))
        return ([f"logprob_ev: expected conviction {x} {_f(t['mean_x'], 2)}, {y} {_f(t['mean_y'], 2)} "
                 f"(n {t['n']} of {len(pairs)} pairs; {ten} with a 10 read as its first digit)",
                 f"  paired mean difference {y} - {x}: {_f(t['diff'], 3)}"]
                + _ci_lines(t["boot"], reps, seed)
                + [f"  sign-flip permutation test ({_perm_note(t['n'])}), two-sided p = {t['p']:.4g}"])
    raise ValueError(metric)


def render(pairs: list[dict], x: str, y: str, primary: str, reps: int, seed: int) -> str:
    """The primary metric first, the others under "secondary", then the
    descriptive tables."""
    lines = [f"pairs: {len(pairs)}"]
    if not pairs:
        return "\n".join(lines)
    lines.append(f"primary ({primary}):")
    lines += ["  " + ln for ln in metric_lines(primary, pairs, x, y, reps, seed)]
    lines.append("secondary:")
    lines.append("  (only the primary metric is confirmatory; the secondary p-values are descriptive, "
                 "with no multiplicity correction)")
    for m in PRIMARY:
        if m != primary:
            lines += ["  " + ln for ln in metric_lines(m, pairs, x, y, reps, seed)]
    s = summarize(pairs, reps, seed, x, y)
    lines.append("descriptive:")
    lines.append("  conviction distribution:")
    lines.append("         " + "".join(f"{c:>5}" for c in range(1, 11)) + "   mean")
    for arm, key in ((x, "x"), (y, "y")):
        d = s[f"dist_{key}"]
        lines.append(f"    {arm}    " + "".join(f"{d.get(c, 0):>5}" for c in range(1, 11))
                     + f"   {_f(s[f'mean_{key}'], 2)}")
    lines.append("  c7+ by run position: " + ", ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in s["by_position"].items()))
    return "\n".join(lines)


def noise_floor_lines(last: list[dict], x: str, y: str, clean: bool) -> list[str]:
    """A vs A2 (two independent draws of arm A) next to x vs y, on the
    signals where all four calls gave a conviction: c7+ discordance and
    action discordance. No lines when no record has arm A2."""
    have = [r for r in last if (r.get("arms") or {}).get(REPEAT_ARM)]
    if not have:
        return []
    xy_ids = {r["signal_id"] for r in usable(have, x, y, clean)}
    both = [r for r in usable(have, "A", REPEAT_ARM, clean) if r["signal_id"] in xy_ids]
    n = len(both)

    def disc(a, b, rows=None):
        rows = both if rows is None else rows
        m = len(rows)
        c7 = sum(1 for r in rows if (_conv(r, a) >= 7) != (_conv(r, b) >= 7))
        act = sum(1 for r in rows if (r["arms"][a].get("action") or NONE) != (r["arms"][b].get("action") or NONE))
        return (f"c7+ discordant {c7}/{m} = {_f(c7 / m if m else None)}, "
                f"action differs {act}/{m} = {_f(act / m if m else None)}")

    def right_after_a(r):
        pa, p2 = r["arms"]["A"].get("position"), r["arms"][REPEAT_ARM].get("position")
        return pa is not None and p2 is not None and p2 == pa + 1
    after = [r for r in both if right_after_a(r)]
    other = [r for r in both if not right_after_a(r)]
    return [f"noise floor on {n} signals with arm {REPEAT_ARM} (of {len(have)} with it called):",
            f"  A vs {REPEAT_ARM} (same prompt, two draws): {disc('A', REPEAT_ARM)}",
            f"  {x} vs {y} on the same signals:          {disc(x, y)}",
            f"  A vs {REPEAT_ARM} by position (right after A shares A's prompt cache): "
            f"right after A: {disc('A', REPEAT_ARM, after)}; not: {disc('A', REPEAT_ARM, other)}"]


def render_bounds(b: dict, x: str, y: str) -> str:
    if not b["n"]:
        return "bounds: no attempted signals"
    d, z = b["dropped"], b["not_c7"]
    return (f"bounds over {b['n']} attempted signals; failed or cut-off replies (no conviction, "
            f"or done_reason length): {x} {b['failed_or_cut'][x]}, {y} {b['failed_or_cut'][y]}\n"
            f"  bound 'dropped' (pairs with a failed or cut-off arm left out, {d['n']} pairs): "
            f"c7+ {x} {_f(d['x'])}, {y} {_f(d['y'])}, {y} - {x} {_f(d['diff'])}\n"
            f"  bound 'not-c7' (failed or cut-off replies counted as not c7+, {z['n']} signals): "
            f"c7+ {x} {_f(z['x'])}, {y} {_f(z['y'])}, {y} - {x} {_f(z['diff'])}")


FAIL_COLS = ("ok", "timeout", "call_error", "parse_failed", "no_conviction", "not_called",
             "salvaged", "num_predict_cut")


def report_group(g: dict, skips: dict, reps: int, seed: int, primary: str) -> str:
    arms_s = g["config"].get("arms") or "AB"
    arms = tuple(arms_s)
    last = [recs[-1] for recs in g["attempts"].values()]
    sample = g["sample"] or list(g["attempts"])
    skipped = {s: skips[s] for s in sample if s in skips}
    attempted = set(g["attempts"])
    not_attempted = [s for s in sample if s not in attempted and s not in skipped]
    out = [f"config: {json.dumps(g['config'], sort_keys=True)}"]
    out.append(f"denominators: sample {len(sample)}, skipped {len(skipped)} "
               f"{dict(Counter(str(r).split(':')[0] for r in skipped.values()))}, "
               f"attempted {len(attempted)} (last attempt complete "
               f"{sum(1 for r in last if r.get('status') == 'ok')}, with a failed arm "
               f"{sum(1 for r in last if r.get('status') != 'ok')}), not attempted {len(not_attempted)}")
    out.append("attempts per signal: " + ", ".join(
        f"{k} x{v}" for k, v in sorted(Counter(len(r) for r in g["attempts"].values()).items())))
    tables = arm_tables(last, arms)
    out.append(f"{'arm (last attempt)':<22}" + " ".join(f"{c:>15}" for c in FAIL_COLS))
    for arm in arms:
        c = tables[arm]["counts"]
        out.append(f"  {arm:<20}" + " ".join(f"{c.get(k, 0):>15}" for k in FAIL_COLS))
    with_rep = [r for r in last if (r.get("arms") or {}).get(REPEAT_ARM)]
    if with_rep or g["config"].get("repeat_arm"):
        c = arm_tables(with_rep, (REPEAT_ARM,))[REPEAT_ARM]["counts"]
        out.append(f"  {REPEAT_ARM + f' ({len(with_rep)} signals)':<20}"
                   + " ".join(f"{c.get(k, 0):>15}" for k in FAIL_COLS))
    for arm in arms:
        pe = tables[arm]["prompt_eval"]
        out.append(f"prompt_eval_count {arm}: " + (f"n {pe[0]}, mean {pe[1]:.0f}, min {pe[2]}, max {pe[3]}"
                                                  if pe else "n/a"))
    comps = [("A", "B")] + ([("A", "C"), ("C", "B")] if "C" in arms else [])
    for x, y in comps:
        out.append(f"\n--- {y} vs {x} ---")
        out.append(render_bounds(bounds(last, x, y), x, y))
        for title, clean in (("pairs with a conviction in both arms", False),
                             ("excluding pairs with a salvaged or num_predict-cut arm", True)):
            out.append(f"\n== {title} ==")
            out.append(render(usable(last, x, y, clean), x, y, primary, reps, seed))
            out.extend(noise_floor_lines(last, x, y, clean))
    return "\n".join(out)


def report_sha() -> str:
    """sha256 of this report script, so a report ties to the code that produced it."""
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"


def provenance_line(data: dict) -> str:
    """The report script's sha256 and the runner tool_sha values on the run lines."""
    shas = sorted({str(r.get("tool_sha")) for r in data["runs"] if r.get("tool_sha")})
    return (f"report_sha256: {report_sha()}; runner tool_sha on run lines: "
            + (", ".join(shas) if shas else "none recorded"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarise a prompt_order_ab.py JSONL (no threshold applied).")
    ap.add_argument("path")
    ap.add_argument("--split", action="store_true",
                    help="report each run configuration separately instead of stopping")
    ap.add_argument("--primary", required=True, choices=PRIMARY,
                    help="the metric printed first, with its test and intervals; the rest follow as secondary")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args(argv)
    data = load(args.path)
    print(provenance_line(data))
    gs = groups(data)
    if not gs:
        print("no run or pair lines")
        return
    if len(gs) > 1 and not args.split:
        print("run lines with pairs differ in configuration; use --split to report them separately:",
              file=sys.stderr)
        for g in gs:
            print(f"  {json.dumps(g['config'], sort_keys=True)}", file=sys.stderr)
        sys.exit(2)
    for i, g in enumerate(gs):
        if len(gs) > 1:
            print(f"\n######## configuration {i + 1} of {len(gs)} ########")
        print(report_group(g, data["skips"], args.boot, args.boot_seed, args.primary))


if __name__ == "__main__":
    main()
