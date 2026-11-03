"""Output schemas of the panel and their parsers.

Every schema lists its reasoning fields before its verdict fields, the same
order the live analyst schema uses (31d52f8), so the model writes its
reasons before it commits to a number. Parsers are strict on the verdict
fields (missing or out of range is a parse failure) and lenient on text
(trimmed to a cap). A failed parse keeps the reason; nothing is guessed.

Scales: score and conviction 1-10 (the live conviction scale, so a panel
score sits beside the live verdict), confidence 0-1. These are a shadow
pilot's placeholders, listed as open questions in the Phase 2 report.
"""
from __future__ import annotations

import math

SCORE_MIN, SCORE_MAX = 1, 10
LIVE_ACTIONS = ("OPEN", "ADD", "HOLD", "TRIM", "EXIT", "WATCH", "PASS")
BUY_ACTIONS = ("OPEN", "ADD")
AGREEMENT_VALUES = ("consensus", "split")
OBJECTION_VALUES = ("fatal", "material", "minor")
RATIONALE_CAP = 600
SHORT_CAP = 300
THESIS_CAP = 1200

# Field order is the order the model is asked to write them.
ANALYST_FIELDS = ("rationale", "what_would_change_my_mind", "score", "confidence")
PM_FIELDS = ("thesis", "agreement_rationale", "agreement", "plan", "verdict", "conviction")
RED_TEAM_FIELDS = ("counter_thesis", "strongest_risks", "what_would_change_my_mind",
                   "objection", "case_strength", "confidence")
PLAN_FIELDS = ("entry_zone", "add_trigger", "targets", "invalidation", "time_stop_days",
               "scenarios", "earnings_playbook")

ANALYST_SCHEMA_TEXT = """{
  "rationale": "<2-3 short sentences: the evidence in your data that drives your view>",
  "what_would_change_my_mind": "<one sentence: the observation that would flip your view>",
  "score": <integer 1-10: 1 = strongly bearish, 5 = no edge, 10 = strongly bullish>,
  "confidence": <number 0-1: how sure you are of the score given your data>
}"""

PM_SCHEMA_TEXT = """{
  "thesis": "<3-5 sentences: the investment case from the panel's evidence>",
  "agreement_rationale": "<1-2 sentences: where the analysts agree and where they split>",
  "agreement": "<consensus or split>",
  "plan": {
    "entry_zone": {"low": <price>, "high": <price>},
    "add_trigger": "<the condition to add to the position>",
    "targets": [{"name": "T1", "price": <price>, "trim_pct": <percent of position to trim>},
                {"name": "T2", "price": <price>, "trim_pct": <percent>}],
    "invalidation": {"price": <price>, "reason": "<what being wrong looks like>"},
    "time_stop_days": <integer trading days before an idle position is exited>,
    "scenarios": [{"name": "bull", "probability": <0-1>, "description": "<one sentence>"},
                  {"name": "base", "probability": <0-1>, "description": "<one sentence>"},
                  {"name": "bear", "probability": <0-1>, "description": "<one sentence>"}],
    "earnings_playbook": "<what to do into and after the next earnings report>"
  },
  "verdict": "<OPEN or ADD or HOLD or TRIM or EXIT or WATCH or PASS>",
  "conviction": <integer 1-10>
}"""

RED_TEAM_SCHEMA_TEXT = """{
  "counter_thesis": "<3-4 sentences: the strongest case against this buy>",
  "strongest_risks": ["<risk>", "<risk>", "<risk>"],
  "what_would_change_my_mind": "<one sentence: what would make you drop the objection>",
  "objection": "<fatal or material or minor>",
  "case_strength": <integer 1-10: how strong the case against is>,
  "confidence": <number 0-1>
}"""


class SchemaError(ValueError):
    pass


def _text(d: dict, key: str, cap: int, required: bool = True) -> str | None:
    v = d.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise SchemaError(f"{key} missing")
        return None
    if not isinstance(v, str):
        v = str(v)
    return v.strip()[:cap]


def _number(v, key: str):
    if isinstance(v, bool) or v is None:
        raise SchemaError(f"{key} missing or not a number")
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            raise SchemaError(f"{key} not a number: {v[:20]!r}") from None
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SchemaError(f"{key} not a finite number")
    return v


def _score(d: dict, key: str) -> int:
    v = _number(d.get(key), key)
    if v != int(v) or not (SCORE_MIN <= v <= SCORE_MAX):
        raise SchemaError(f"{key} must be an integer {SCORE_MIN}-{SCORE_MAX}, got {v}")
    return int(v)


def _confidence(d: dict) -> float:
    v = float(_number(d.get("confidence"), "confidence"))
    if not (0.0 <= v <= 1.0):
        raise SchemaError(f"confidence must be 0-1, got {v}")
    return v


def _choice(d: dict, key: str, allowed, upper: bool) -> str:
    v = d.get(key)
    if not isinstance(v, str):
        raise SchemaError(f"{key} missing")
    v = v.strip().upper() if upper else v.strip().lower()
    if v not in allowed:
        raise SchemaError(f"{key} must be one of {allowed}, got {v[:20]!r}")
    return v


def field_order_ok(d: dict, fields) -> bool:
    """Whether the keys that are present came in schema order (the reply
    wrote reasoning before verdict). Recorded, not enforced."""
    pos = [list(d).index(f) for f in fields if f in d]
    return pos == sorted(pos)


def parse_analyst(d: dict) -> dict:
    return {"rationale": _text(d, "rationale", RATIONALE_CAP),
            "what_would_change_my_mind": _text(d, "what_would_change_my_mind", SHORT_CAP),
            "score": _score(d, "score"),
            "confidence": _confidence(d),
            "field_order_ok": field_order_ok(d, ANALYST_FIELDS)}


def _price(v):
    if v is None:
        return None
    try:
        v = float(_number(v, "price"))
    except SchemaError:
        return None
    return v if v > 0 else None


def parse_plan(p) -> tuple[dict | None, list[str]]:
    """The plan block, normalised, and a list of issues (kept, not fatal:
    a plan is the PM's proposal, and a hole in it is data)."""
    issues: list[str] = []
    if not isinstance(p, dict):
        return None, ["plan missing"]
    out: dict = {}
    ez = p.get("entry_zone") if isinstance(p.get("entry_zone"), dict) else {}
    lo, hi = _price(ez.get("low")), _price(ez.get("high"))
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
        issues.append("entry_zone low/high swapped")
    out["entry_zone"] = {"low": lo, "high": hi}
    out["add_trigger"] = _text(p, "add_trigger", SHORT_CAP, required=False)
    targets = []
    for t in (p.get("targets") or [])[:2] if isinstance(p.get("targets"), list) else []:
        if not isinstance(t, dict):
            continue
        trim = t.get("trim_pct")
        try:
            trim = float(_number(trim, "trim_pct"))
            if not (0 <= trim <= 100):
                issues.append(f"trim_pct out of range: {trim}")
                trim = None
        except SchemaError:
            trim = None
        targets.append({"name": str(t.get("name") or f"T{len(targets) + 1}")[:8],
                        "price": _price(t.get("price")), "trim_pct": trim})
    if len(targets) < 2:
        issues.append("fewer than two targets")
    out["targets"] = targets
    inv = p.get("invalidation") if isinstance(p.get("invalidation"), dict) else {}
    out["invalidation"] = {"price": _price(inv.get("price")),
                           "reason": _text(inv, "reason", SHORT_CAP, required=False)}
    if out["invalidation"]["price"] is None:
        issues.append("no invalidation price")
    ts = p.get("time_stop_days")
    try:
        ts = int(_number(ts, "time_stop_days"))
        ts = ts if ts > 0 else None
    except (SchemaError, ValueError):
        ts = None
    out["time_stop_days"] = ts
    scen = []
    for s in (p.get("scenarios") or [])[:3] if isinstance(p.get("scenarios"), list) else []:
        if not isinstance(s, dict):
            continue
        try:
            prob = float(_number(s.get("probability"), "probability"))
            prob = prob if 0 <= prob <= 1 else None
        except SchemaError:
            prob = None
        scen.append({"name": str(s.get("name") or "")[:16], "probability": prob,
                     "description": _text(s, "description", SHORT_CAP, required=False)})
    if len(scen) != 3:
        issues.append(f"{len(scen)} scenarios, not 3")
    probs = [s["probability"] for s in scen]
    if scen and all(x is not None for x in probs) and abs(sum(probs) - 1.0) > 0.05:
        issues.append(f"scenario probabilities sum to {sum(probs):.2f}")
    out["scenarios"] = scen
    out["earnings_playbook"] = _text(p, "earnings_playbook", SHORT_CAP, required=False)
    return out, issues


def parse_pm(d: dict) -> dict:
    plan, issues = parse_plan(d.get("plan"))
    return {"thesis": _text(d, "thesis", THESIS_CAP),
            "agreement_rationale": _text(d, "agreement_rationale", SHORT_CAP, required=False),
            "agreement": _choice(d, "agreement", AGREEMENT_VALUES, upper=False),
            "plan": plan, "plan_issues": issues,
            "verdict": _choice(d, "verdict", LIVE_ACTIONS, upper=True),
            "conviction": _score(d, "conviction"),
            "field_order_ok": field_order_ok(d, PM_FIELDS)}


def parse_red_team(d: dict) -> dict:
    risks = d.get("strongest_risks")
    risks = [str(r).strip()[:SHORT_CAP] for r in risks if str(r).strip()][:3] \
        if isinstance(risks, list) else []
    return {"counter_thesis": _text(d, "counter_thesis", THESIS_CAP),
            "strongest_risks": risks,
            "what_would_change_my_mind": _text(d, "what_would_change_my_mind", SHORT_CAP),
            "objection": _choice(d, "objection", OBJECTION_VALUES, upper=False),
            "case_strength": _score(d, "case_strength"),
            "confidence": _confidence(d),
            "field_order_ok": field_order_ok(d, RED_TEAM_FIELDS)}


def is_would_be_buy(pm: dict | None) -> bool:
    """The red-team trigger: the PM's parsed verdict is OPEN or ADD."""
    return bool(pm) and pm.get("verdict") in BUY_ACTIONS
