"""Record-only valuation consistency check on a verdict (audit 2026-11-21
23:32 #5). It compares the analyst's own valuation.current_multiple and
valuation.peer_range with the FUNDAMENTALS block of the prompt it was sent
(forward P/E, P/S, EV/EBITDA), and its "below/above/within the peer range"
wording with its own numbers.

It runs after the verdict, from the decision dict in analyst.analyze. Nothing
in the prompt, the brief, the gates or routing reads it. It is pure (no I/O)
and never raises.

Four fields go on signals.decision:
  valuation_parse     what was parsed from current_multiple and peer_range
  valuation_vs_brief  the prompt's multiples, each one's % gap, and the closest
  valuation_mismatch  True when no prompt multiple is within
                      VALUATION_MISMATCH_TOL_PCT of the stated multiple. This is
                      a descriptive label for reading the rows, chosen for
                      illustration. It is not a gate or a threshold anyone
                      decided on.
  range_contradiction True when the text puts the stated multiple below or
                      above the stated range but the number is inside it, or
                      the reverse (claimed position != actual position)
plus range_check: the phrases found, where the number actually falls, and
whether the stated multiple and the range name different bases.
Each is None with a reason when it can't be computed (the reason is in
valuation_parse, valuation_vs_brief or range_check).
"""
from __future__ import annotations

import math
import re

# Illustrative only (see the module docstring): the tolerance for the
# valuation_mismatch label, in percent of the prompt's multiple.
VALUATION_MISMATCH_TOL_PCT = 10.0

BRIEF_MULTIPLES = (("forward_pe", "Forward P/E"), ("price_to_sales", "Price/Sales"),
                   ("ev_to_ebitda", "EV/EBITDA"))

_NUM = r"(\d+(?:\.\d+)?)"
_MULT_RE = re.compile(_NUM + r"\s*[x×](?![a-z])", re.I)
# "6x to 12x", "6-12x", "6.5x - 12.0x", "10–15x": the high end needs its "x",
# so percent ranges ("8% to 60%") and percentile text don't parse.
_RANGE_RE = re.compile(_NUM + r"\s*[x×]?\s*(?:-|–|—|to)\s*" + _NUM + r"\s*[x×](?![a-z])", re.I)
_CLAIM_RE = re.compile(
    r"\b(below|under|beneath|less than|cheaper than|a discount to|above|over|more than|"
    r"a premium to|within|inside|in line with|in the middle of|"
    r"at the (?:low|lower|high|upper|top|bottom) end of)\b([^.;\n]{0,30}?)\brange\b", re.I)
_NOT_PEER_RANGE = re.compile(r"week|day|month|year|trading|price|support|resistance|target", re.I)
_CLAIM_POS = {"below": "below", "under": "below", "beneath": "below", "less than": "below",
              "cheaper than": "below", "a discount to": "below", "above": "above",
              "over": "above", "more than": "above", "a premium to": "above"}


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _basis(text: str) -> str | None:
    """The multiple a phrase names; "mixed" when it names more than one."""
    t = (text or "").lower()
    found = [k for k, pat in (("ev_to_ebitda", r"ebitda"),
                              ("price_to_sales", r"revenue|sales|p/s\b"),
                              ("forward_pe", r"earnings|p/e\b|\bpe\b|\beps\b"))
             if re.search(pat, t)]
    return found[0] if len(found) == 1 else "mixed" if found else None


def _segment_basis(text: str, start: int, end: int) -> str | None:
    """The basis named right after a number (up to the next number or comma),
    else right before it (back to the previous comma or number)."""
    after = re.split(r"[,;(]|\d", text[end:], maxsplit=1)[0]
    b = _basis(after)
    if b:
        return b
    before = re.split(r"[,;)]|\d", text[:start][::-1], maxsplit=1)[0][::-1]
    return _basis(before)


def parse_valuation(current_multiple, peer_range) -> dict:
    """What can be read from the two valuation strings. Pure."""
    out = {"stated": None, "stated_basis": None, "n_stated": 0, "range_lo": None,
           "range_hi": None, "range_basis": None, "n_ranges": 0, "reason": None}
    reasons = []
    cm = current_multiple if isinstance(current_multiple, str) else ""
    pr = peer_range if isinstance(peer_range, str) else ""
    ms = list(_MULT_RE.finditer(cm))
    out["n_stated"] = len(ms)
    r0 = _RANGE_RE.search(cm)
    if ms and r0 and r0.start() <= ms[0].start():
        reasons.append("current_multiple is a range, not one multiple")
    elif ms:
        out["stated"] = _finite(ms[0].group(1))
        out["stated_basis"] = _segment_basis(cm, ms[0].start(), ms[0].end())
    else:
        reasons.append("no '<n>x' in current_multiple" if cm else "current_multiple missing")
    rs = list(_RANGE_RE.finditer(pr))
    out["n_ranges"] = len(rs)
    if rs:
        a, b = _finite(rs[0].group(1)), _finite(rs[0].group(2))
        if a is not None and b is not None:
            out["range_lo"], out["range_hi"] = min(a, b), max(a, b)
            out["range_basis"] = _segment_basis(pr, rs[0].start(), rs[0].end())
    else:
        reasons.append("no '<lo> to <hi>x' in peer_range" if pr else "peer_range missing")
    out["reason"] = "; ".join(reasons) or None
    return out


def brief_multiples(prompt: str, ticker: str) -> dict | None:
    """{forward_pe, price_to_sales, ev_to_ebitda} from the prompt's FUNDAMENTALS
    block for ticker: a value, or None for n/a or <= 0. None when the block
    isn't in the prompt or has no multiples (not yet cached)."""
    if not isinstance(prompt, str) or not ticker:
        return None
    head = f"=== FUNDAMENTALS — {ticker} ==="
    i = prompt.find(head)
    if i < 0:
        return None
    body = prompt[i + len(head):]
    j = body.find("===")
    block = body[:j] if j >= 0 else body
    out, seen = {}, False
    for key, label in BRIEF_MULTIPLES:
        m = re.search(r"^" + re.escape(label) + r":\s*(-?\d+(?:\.\d+)?)x", block, re.M)
        if re.search(r"^" + re.escape(label) + r":", block, re.M):
            seen = True
        v = _finite(m.group(1)) if m else None
        out[key] = v if v is not None and v > 0 else None
    return out if seen else None


def _position(v: float, lo: float, hi: float) -> str:
    return "below" if v < lo else "above" if v > hi else "inside"


def _claims(texts: dict) -> list[dict]:
    out = []
    for where, text in texts.items():
        if not isinstance(text, str):
            continue
        for m in _CLAIM_RE.finditer(text):
            if _NOT_PEER_RANGE.search(m.group(2)):
                continue   # a price or trading range, not the peer multiple range
            word = m.group(1).lower()
            out.append({"in": where, "phrase": m.group(0)[:80],
                        "claimed": _CLAIM_POS.get(word, "inside")})
    return out


def valuation_check(parsed: dict, prompt: str, ticker: str) -> dict:
    """The four decision fields from the model's parsed reply and the prompt
    it was sent. Never raises."""
    out = {"valuation_parse": None, "valuation_vs_brief": None,
           "valuation_mismatch": None, "range_contradiction": None}
    try:
        val = (parsed or {}).get("valuation")
        if not isinstance(val, dict):
            out["valuation_parse"] = {"reason": "no valuation object in the reply"}
            out["valuation_vs_brief"] = {"reason": "no valuation object in the reply"}
            out["range_check"] = {"claims": [], "actual": None,
                                  "reason": "no valuation object in the reply"}
            return out
        p = parse_valuation(val.get("current_multiple"), val.get("peer_range"))
        out["valuation_parse"] = p
        stated = p["stated"]
        brief = brief_multiples(prompt, ticker)
        vb = {"brief": brief, "gaps_pct": None, "closest": None, "closest_gap_pct": None,
              "basis_gap_pct": None, "tol_pct": VALUATION_MISMATCH_TOL_PCT, "reason": None}
        if brief is None:
            vb["reason"] = "no FUNDAMENTALS multiples in the prompt"
        elif not any(v is not None for v in brief.values()):
            vb["reason"] = "the prompt's multiples are all n/a or <= 0"
        elif stated is None or stated <= 0:
            vb["reason"] = "no stated multiple"
        else:
            gaps = {k: round((stated / v - 1) * 100, 2) for k, v in brief.items() if v}
            vb["gaps_pct"] = gaps
            k = min(gaps, key=lambda x: abs(gaps[x]))
            vb["closest"], vb["closest_gap_pct"] = k, gaps[k]
            vb["basis_gap_pct"] = gaps.get(p["stated_basis"]) if p["stated_basis"] else None
            out["valuation_mismatch"] = abs(gaps[k]) > VALUATION_MISMATCH_TOL_PCT
        out["valuation_vs_brief"] = vb
        claims = _claims({"current_multiple": val.get("current_multiple"),
                          "peer_range": val.get("peer_range"),
                          "reasoning": (parsed or {}).get("reasoning")})
        rc = {"claims": claims, "actual": None, "reason": None}
        if stated is None or p["range_lo"] is None:
            rc["reason"] = "no stated multiple" if stated is None else "no parsed range"
        else:
            rc["actual"] = _position(stated, p["range_lo"], p["range_hi"])
            rc["basis_differs"] = (None if not (p["stated_basis"] and p["range_basis"])
                                   else p["stated_basis"] != p["range_basis"])
            if not claims:
                rc["reason"] = "no below/above/within-range phrase"
            else:
                for c in claims:
                    c["contradicts"] = c["claimed"] != rc["actual"]
                out["range_contradiction"] = any(c["contradicts"] for c in claims)
        out["range_check"] = rc
    except Exception as e:
        out["valuation_check_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out
